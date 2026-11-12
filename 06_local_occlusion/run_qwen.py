"""Local Qwen-Image-2.1 two-reference foreground fusion experiment.

Independent of the deterministic RGBA path: both original images go straight
to Qwen. No colour-key cutout, pixel paste-back or fabricated occlusion mask.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image, ImageOps


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
HELPER_PATH = PROJECT / '02_qwen_image_edit/run.py'
SPEC = importlib.util.spec_from_file_location('qwen_weather_helpers', HELPER_PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reference(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        return image.convert('RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB')


def prepare_inputs(config):
    source = (PROJECT / config['input_image']).resolve()
    reference = (PROJECT / config['reference_image']).resolve()
    model_dir = (PROJECT / config['model_dir']).resolve()
    scene_path = (PROJECT / config['scene_config']).resolve()
    for path in (source, reference, scene_path, model_dir / 'model_index.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    width, height = config['width'], config['height']
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 32 for v in (width, height)):
        raise ValueError('Canvas dimensions must be positive multiples of 32')
    if not isinstance(config['num_inference_steps'], int) or config['num_inference_steps'] <= 0:
        raise ValueError('num_inference_steps must be positive')
    if not isinstance(config.get('prompt'), str) or not config['prompt'].strip():
        raise ValueError('A nonempty fusion prompt is required')
    original = read_reference(source).convert('RGB')
    material = read_reference(reference)
    canvas, content, mapping = helpers.prepare_image(original, width, height)
    scenes = json.loads(scene_path.read_text(encoding='utf-8'))
    if source.name not in scenes:
        raise ValueError('Add a scene record for the target image to enable part comparison')
    scene = scenes[source.name]
    if not scene.get('regions') or not scene.get('subject_polygon'):
        raise ValueError('Qwen part review requires regions and subject_polygon in the scene record')
    metadata_path = model_dir / '.cache/huggingface/download/model_index.json.metadata'
    revision = metadata_path.read_text(encoding='utf-8').splitlines()[0]
    if revision != config['model_revision']:
        raise ValueError(f'Unexpected downloaded revision: {revision}')
    return source, reference, model_dir, scene_path, scene, canvas, content, material, mapping, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=METHOD / 'configs/qwen_cyclist_fusion.json')
    parser.add_argument('--output-dir', type=Path, default=METHOD / 'outputs/qwen_cyclist_fusion_v1')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    environment = helpers.configure_environment()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source, reference, model_dir, scene_path, scene, canvas, content, material, mapping, revision = prepare_inputs(config)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if (output.is_relative_to(source.parent) or output.is_relative_to(reference.parent)
            or output.is_relative_to(model_dir)
            or any(p.is_relative_to(output) for p in (source, reference, config_path, scene_path))):
        raise ValueError('Output must not contain or overwrite source, reference, configuration or weights')
    width, height = canvas.size
    output_resolution = round(math.sqrt(width * height))
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    reference_sizes = [list(calculate_dimensions(output_resolution ** 2, im.width / im.height)[:2])
                       for im in (canvas, material)]
    if reference_sizes[0] != [width, height]:
        raise ValueError(f'Unexpected primary reference resize: {reference_sizes[0]}')
    mapping['output_resolution_parameter'] = output_resolution
    mapping['pipeline_reference_sizes'] = reference_sizes
    if args.check_config:
        print(json.dumps({'geometry': mapping, 'reference': str(reference),
                          'reference_mode': material.mode, 'revision': revision}, indent=2))
        return

    output.mkdir(parents=True)
    started = time.perf_counter()
    inputs = {str(p): sha256(p) for p in (source, reference, config_path, scene_path)}
    manifest = {
        'method': 'qwen_two_reference_foreground_fusion', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'input_sha256': inputs, 'mapping': mapping, 'reference_order': ['target_vehicle_scene', 'foreground_material'],
        'model_dir': str(model_dir), 'verified_download_revision': revision,
        'model_index_sha256': sha256(model_dir / 'model_index.json'),
        'script_sha256': sha256(__file__), 'helper_sha256': sha256(HELPER_PATH),
        'environment_overrides': environment, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'constraints': 'Prompt-only preservation. No mask_image, strength or region lock in this pipeline. Unoccluded pixels may change.',
        'postprocessing': 'Remove recorded source padding only; no original-pixel paste-back, material compositing, enhancement or upscaling.',
        'occlusion_mask': None,
    }
    helpers.save_json(output / 'manifest.json', manifest)
    helpers.save_json(output / 'config.json', config)
    helpers.save_json(output / 'scene_reference.json', scene)
    shutil.copy2(__file__, output / 'run_snapshot.py')
    shutil.copy2(HELPER_PATH, output / 'weather_helpers_snapshot.py')
    canvas.save(output / 'source_canvas.png')
    content.save(output / 'source_working.png')
    material.save(output / 'material_reference.png')
    (output / 'prompt.txt').write_text(config['prompt'] + '\n', encoding='utf-8')
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
    (output / 'environment_freeze.txt').write_text(''.join(f'{name}=={version}\n' for name, version in packages), encoding='utf-8')

    try:
        import torch
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        manifest.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                        cuda_runtime=torch.version.cuda, pipeline_source_sha256=sha256(inspect.getfile(QwenImage21Pipeline)))
        manifest['gpu_before'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv,noheader'], text=True)
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, local_files_only=True,
            use_safetensors=True, low_cpu_mem_usage=True).to('cuda')
        if config['vae_tiling']:
            pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        manifest['load_seconds'] = time.perf_counter() - load_start
        manifest['scheduler_config'] = dict(pipe.scheduler.config)
        manifest['attention_processors'] = sorted({type(p).__name__ for p in pipe.transformer.attn_processors.values()})
        manifest['sample_sigmas'] = pipe.config.sample_sigmas
        helpers.save_json(output / 'manifest.json', manifest)
        print(f'Loaded Qwen in {manifest["load_seconds"]:.1f}s; fusing 2 references at {width}x{height}', flush=True)
        inference_start = time.perf_counter()
        step_times = []

        def on_step(pipeline, step, timestep, values):
            if not torch.isfinite(values['latents']).all().item():
                raise FloatingPointError(f'Non-finite latents at step {step + 1}')
            elapsed = time.perf_counter() - inference_start
            step_times.append(elapsed)
            progress = {'completed_steps': step + 1, 'total_steps': pipeline.num_timesteps,
                        'elapsed_seconds': round(elapsed, 2),
                        'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)}
            helpers.save_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            return values

        with torch.inference_mode():
            result = pipe(
                image=[canvas, material], prompt=config['prompt'], width=width, height=height,
                output_resolution=output_resolution, num_inference_steps=config['num_inference_steps'],
                true_cfg_scale=config['true_cfg_scale'], use_kv_cache=config['use_kv_cache'],
                generator=torch.Generator(device='cuda').manual_seed(config['seed']),
                callback_on_step_end=on_step,
            ).images[0]
        torch.cuda.synchronize()
        manifest.update(inference_seconds=time.perf_counter() - inference_start,
                        completed_steps=len(step_times), step_end_elapsed_seconds=step_times,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        if result.size != canvas.size:
            raise ValueError(f'Unexpected generated dimensions: {result.size}')
        result.save(output / 'edited_canvas.png')
        edited = result.crop(mapping['content_box_xyxy'])
        edited.save(output / 'edited.png')
        manifest['output_mode'] = result.mode
        if result.mode == 'RGBA':
            alpha = np.asarray(result.getchannel('A'))
            manifest['alpha_range'] = [int(alpha.min()), int(alpha.max())]
            manifest['nonopaque_pixel_fraction'] = float(np.mean(alpha < 255))
        manifest['status'] = 'generated_pending_previews'
        helpers.save_json(output / 'manifest.json', manifest)
        helpers.create_previews(output, content, edited, scene, config)
        # Lossless review sheets in addition to the shared helper's JPEG previews.
        helpers.pair_panel(content, edited, 'Audi A6L | Qwen two-reference cyclist fusion', width=800).save(output / 'previews/overview.png')
        if not all(sha256(Path(path)) == digest for path, digest in inputs.items()):
            raise RuntimeError('An input changed during inference')
        pngs = sorted(output.rglob('*.png'))
        for path in pngs:
            with Image.open(path) as opened:
                opened.verify()
        manifest.update(input_files_unchanged=True,
                        output_sha256={str(p.relative_to(output)): sha256(p) for p in pngs},
                        png_files_checked=len(pngs), status='complete_pending_visual_review',
                        total_seconds=time.perf_counter() - started)
        helpers.save_json(output / 'manifest.json', manifest)
        print('GENERATION_OK', output, flush=True)
        print(json.dumps({k: manifest[k] for k in ('load_seconds', 'inference_seconds', 'peak_allocated_gib', 'peak_reserved_gib')}, indent=2), flush=True)
    except Exception as error:
        manifest.update(status='failed', error=str(error), traceback=traceback.format_exc(), total_seconds=time.perf_counter() - started)
        helpers.save_json(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
