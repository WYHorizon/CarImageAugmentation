"""One-image, offline Qwen-Image-2.1 weather-edit experiment."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
CACHE = Path('/nas/wyh/.cache')


def configure_environment():
    # These per-process overrides take precedence over the shared root shell settings.
    values = {
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HOME': str(CACHE / 'huggingface'),
        'HF_HUB_CACHE': str(CACHE / 'huggingface/hub'),
        'HUGGINGFACE_HUB_CACHE': str(CACHE / 'huggingface/hub'),
        'HF_XET_CACHE': str(CACHE / 'huggingface/xet'),
        'TORCH_HOME': str(CACHE / 'torch'),
        'XDG_CACHE_HOME': str(CACHE / 'xdg'),
        'PIP_CACHE_DIR': str(CACHE / 'pip'),
        'TMPDIR': str(CACHE / 'tmp'), 'TMP': str(CACHE / 'tmp'),
        'TEMP': str(CACHE / 'tmp'),
    }
    os.environ.update(values)
    return values


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, data):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_image(image, width, height):
    """Exact rational aspect ratio, then edge padding to the model's canvas."""
    divisor = math.gcd(*image.size)
    aspect_w, aspect_h = image.width // divisor, image.height // divisor
    units = min(width // aspect_w, height // aspect_h)
    if units < 1:
        raise ValueError('Canvas is too small to retain the exact input aspect ratio')
    content_size = (aspect_w * units, aspect_h * units)
    content = image.resize(content_size, Image.Resampling.LANCZOS)
    left = (width - content.width) // 2
    top = (height - content.height) // 2
    right, bottom = width - content.width - left, height - content.height - top
    pixels = np.pad(np.asarray(content), ((top, bottom), (left, right), (0, 0)), mode='edge')
    canvas = Image.fromarray(pixels)
    mapping = {
        'original_size': list(image.size), 'canvas_size': [width, height],
        'content_size': list(content_size), 'scale': units / divisor,
        'padding_ltrb': [left, top, right, bottom],
        'content_box_xyxy': [left, top, left + content.width, top + content.height],
        'resampling': 'Pillow LANCZOS', 'padding_mode': 'edge',
        'inverse_mapping': 'x_original=(x_canvas-left)/scale; y_original=(y_canvas-top)/scale',
        'output_upscaled_to_original': False,
    }
    return canvas, content, mapping


def display_rgb(image):
    # Preview only: preserve the unmodified RGBA model output separately.
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert('RGB')
    return image.convert('RGB')


def pair_panel(before, after, title, width=960):
    height = round(before.height * width / before.width)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    panel = Image.new('RGB', (width * 2 + 24, height + 72), '#eeeeee')
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), title, font=font, fill='#222222')
    for column, (label, image) in enumerate([('Original (working resolution)', before), ('Qwen edited', after)]):
        x = 8 + column * (width + 8)
        draw.text((x, 38), label, font=font, fill='#222222')
        panel.paste(display_rgb(image).resize((width, height), Image.Resampling.LANCZOS), (x, 68))
    return panel


def create_previews(output, before, after, scene, config):
    previews = output / 'previews'
    crops = output / 'crops'
    previews.mkdir()
    crops.mkdir()
    weather_label = config.get('weather_label', 'overcast, wet road, no rain streaks')
    title = f'{scene["label"]} | seed {config["seed"]} | {weather_label}'
    pair_panel(before, after, title).save(
        previews / 'overview.jpg', quality=95)
    regions = dict(scene['regions'])
    vertices = scene['subject_polygon']
    regions['vehicle'] = [max(0, min(p[0] for p in vertices) - .02),
                          max(0, min(p[1] for p in vertices) - .02),
                          min(1, max(p[0] for p in vertices) + .02),
                          min(1, max(p[1] for p in vertices) + .02)]
    regions['tire_contact'] = [.16, .70, .86, .85]
    detail_panels = []
    boxes = {}
    for name, box in regions.items():
        pixels = [round(box[0] * before.width), round(box[1] * before.height),
                  round(box[2] * before.width), round(box[3] * before.height)]
        boxes[name] = {'normalized': box, 'content_pixels_xyxy': pixels}
        a, b = before.crop(pixels), after.crop(pixels)
        a.save(crops / f'{name}_original.png')
        b.save(crops / f'{name}_edited.png')
        panel = pair_panel(a, b, name.replace('_', ' '), width=600 if name in scene['regions'] else 960)
        panel.save(previews / f'{name}.jpg', quality=95)
        if name in scene['regions']:
            detail_panels.append(panel)
    sheet = Image.new('RGB', (max(p.width for p in detail_panels), sum(p.height for p in detail_panels)), 'white')
    y = 0
    for panel in detail_panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(previews / 'details.jpg', quality=95)
    save_json(output / 'crop_regions.json', boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=METHOD / 'configs/audi_overcast_wet_seed42.json')
    parser.add_argument('--output-dir', type=Path, default=METHOD / 'outputs/audi_overcast_wet_seed42_2k')
    parser.add_argument('--check-config', action='store_true', help='Validate input geometry without loading the model')
    args = parser.parse_args()
    environment = configure_environment()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source = (PROJECT / config['input_image']).resolve()
    model_dir = (PROJECT / config['model_dir']).resolve()
    scene_path = (PROJECT / config['scene_config']).resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(source.parent) or output.is_relative_to(model_dir):
        raise ValueError('Output must not be inside the original-image or model directory')
    if output.exists():
        raise FileExistsError(f'Output already exists; select a fresh run directory: {output}')
    width, height = config['width'], config['height']
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError('Canvas dimensions must be positive multiples of 32')
    if config['num_inference_steps'] <= 0:
        raise ValueError('num_inference_steps must be positive')
    with Image.open(source) as opened:
        original = ImageOps.exif_transpose(opened).convert('RGB')
    canvas, content, mapping = prepare_image(original, width, height)
    output_resolution = round(math.sqrt(width * height))
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    actual_size = calculate_dimensions(output_resolution ** 2, width / height)[:2]
    if actual_size != (width, height):
        raise ValueError(f'Pipeline would resize the reference unexpectedly: {actual_size}')
    mapping['output_resolution_parameter'] = output_resolution
    mapping['pipeline_reference_size'] = list(actual_size)
    scene = json.loads(scene_path.read_text(encoding='utf-8'))[source.name]
    metadata_path = model_dir / '.cache/huggingface/download/model_index.json.metadata'
    downloaded_revision = metadata_path.read_text(encoding='utf-8').splitlines()[0]
    if downloaded_revision != config['model_revision']:
        raise ValueError(f'Unexpected downloaded revision: {downloaded_revision}')
    if args.check_config:
        print(json.dumps({'config': str(args.config), 'geometry': mapping, 'revision': downloaded_revision}, indent=2))
        return

    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'source_sha256': sha256(source),
        'config': config, 'mapping': mapping, 'environment_overrides': environment,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_dir': str(model_dir), 'verified_download_revision': downloaded_revision,
        'model_index_sha256': sha256(model_dir / 'model_index.json'),
        'script_sha256': sha256(__file__), 'scene_config_sha256': sha256(scene_path),
        'postprocessing': 'Remove recorded edge padding only; no original-pixel paste-back, enhancement, or upscaling',
    }
    save_json(output / 'manifest.json', manifest)
    shutil.copyfile(__file__, output / 'run_snapshot.py')
    save_json(output / 'config.json', config)
    save_json(output / 'scene_reference.json', scene)
    canvas.save(output / 'source_canvas.png')
    content.save(output / 'source_working.png')
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
    (output / 'environment_freeze.txt').write_text(''.join(f'{name}=={version}\n' for name, version in packages), encoding='utf-8')

    try:
        import torch
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        manifest['gpu'] = torch.cuda.get_device_name(0)
        manifest['torch_version'] = torch.__version__
        manifest['cuda_runtime'] = torch.version.cuda
        manifest['pipeline_source_sha256'] = sha256(inspect.getfile(QwenImage21Pipeline))
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
        manifest['tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
        save_json(output / 'manifest.json', manifest)
        print(f'Loaded model in {manifest["load_seconds"]:.1f}s; generating {width}x{height}', flush=True)
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
            save_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            return values

        with torch.inference_mode():
            result = pipe(
                image=canvas, prompt=config['prompt'], width=width, height=height,
                output_resolution=output_resolution,
                num_inference_steps=config['num_inference_steps'],
                true_cfg_scale=config['true_cfg_scale'],
                use_kv_cache=config['use_kv_cache'],
                generator=torch.Generator(device='cuda').manual_seed(config['seed']),
                callback_on_step_end=on_step,
            ).images[0]
        torch.cuda.synchronize()
        manifest['inference_seconds'] = time.perf_counter() - inference_start
        manifest['completed_steps'] = len(step_times)
        manifest['step_end_elapsed_seconds'] = step_times
        manifest['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
        manifest['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024 ** 3
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
        save_json(output / 'manifest.json', manifest)
        create_previews(output, content, edited, scene, config)
        assert sha256(source) == manifest['source_sha256'], 'Source image changed during the run'
        manifest['source_unchanged'] = True
        manifest['output_sha256'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*.png'))}
        manifest['status'] = 'complete_pending_visual_review'
        manifest['total_seconds'] = time.perf_counter() - started
        save_json(output / 'manifest.json', manifest)
        print('GENERATION_OK', output, flush=True)
        print(json.dumps({k: manifest[k] for k in ['load_seconds', 'inference_seconds', 'peak_allocated_gib', 'peak_reserved_gib']}, indent=2), flush=True)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        manifest['traceback'] = traceback.format_exc()
        manifest['total_seconds'] = time.perf_counter() - started
        save_json(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
