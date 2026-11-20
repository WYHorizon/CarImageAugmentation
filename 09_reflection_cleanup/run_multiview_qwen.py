"""Generate seven paint-cleanup candidates and reuse the accepted side image.

Only 2D images are processed. All inputs are the original v5 prepared RGBs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from PIL import Image
from run_qwen import helpers, PROJECT, METHOD


PROMPT = "清理车身漆面上反射出的车辆、建筑和杂乱条纹，让这些环境倒影消失，显露连续、干净的原有深黑蓝色车漆。保留车身曲面的自然明暗和柔和光泽，车身造型、机位及其他区域不变。"
STRONG_NEGATIVE = "漆面上清晰的车辆倒影、建筑倒影、杂乱条纹、碳纤维纹理、布纹、网格纹、划痕、重影、车身变形、车色变浅。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepared-root', type=Path, default=PROJECT/'08_3d_probe/photo_texture/outputs/bmw5_inputs_v2')
    parser.add_argument('--reuse-side', type=Path, default=METHOD/'outputs/bmw5_side_qwen_v2')
    parser.add_argument('--variant', choices=['mild', 'strong'], default='mild')
    parser.add_argument('--views', nargs='+', help='Generate only these view IDs into a fresh output directory')
    args = parser.parse_args()
    cfg = 2.5 if args.variant == 'strong' else 1.
    negative = STRONG_NEGATIVE if args.variant == 'strong' else None
    mix = .85 if args.variant == 'strong' else .45
    env = helpers.configure_environment()
    out = args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(METHOD/'outputs'):
        raise ValueError('Use a fresh output directory inside 09_reflection_cleanup/outputs')
    prepared = args.prepared_root.resolve()
    refs = json.loads((prepared/'manifest.json').read_text())['views']
    if args.views:
        unknown=set(args.views)-{r['id'] for r in refs}
        if unknown:raise ValueError(f'Unknown views: {sorted(unknown)}')
        refs=[r for r in refs if r['id'] in args.views]
    model = PROJECT/'models/Qwen-Image-2.1'
    revision = (model/'.cache/huggingface/download/model_index.json.metadata').read_text().splitlines()[0]
    if revision != 'd26bb61231c349cf6b7896fa83353113880e1ba3':
        raise ValueError('Unexpected model revision')
    for ref in refs:
        p = prepared/ref['files']['source_rgb.png']['path']
        if helpers.sha256(p) != ref['files']['source_rgb.png']['sha256']:
            raise ValueError(f'Prepared input changed: {p}')
        if Image.open(p).size != (1536, 1024):
            raise ValueError('Use the original prepared image coordinates')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'run_snapshot.py')
    (out/'prompt.txt').write_text(PROMPT+'\n')
    packages=sorted((p.metadata['Name'],p.version) for p in importlib.metadata.distributions() if p.metadata['Name'])
    (out/'environment_freeze.txt').write_text(''.join(f'{n}=={v}\n' for n,v in packages))
    record = dict(status='loading', started_utc=datetime.now(timezone.utc).isoformat(),
        method=f'single_image_Qwen_{args.variant}_candidates_for_frozen_v5_texture',
        variant=args.variant,
        selected_views=[r['id'] for r in refs],
        model=str(model), model_revision=revision, prepared_root=str(prepared),
        prepared_manifest_sha256=helpers.sha256(prepared/'manifest.json'),
        prompt=PROMPT, seed=42, steps=40, true_cfg_scale=cfg, negative_prompt=negative,
        width=1536, height=1024, use_kv_cache=True, vae_tiling=False,
        environment_overrides=env, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        intended_compositing_mix=mix,
        limitations='Independent per-photo editing; no guaranteed cross-view consistency. Masks and compositing are separate.',
        views=[])
    save=lambda: helpers.save_json(out/'manifest.json',record)
    save()
    try:
        import torch
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA BF16 GPU is required')
        record.update(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),pipeline_source_sha256=helpers.sha256(inspect.getfile(QwenImage21Pipeline)))
        start=time.perf_counter()
        pipe=QwenImage21Pipeline.from_pretrained(str(model),dtype=torch.bfloat16,
            local_files_only=True,use_safetensors=True,low_cpu_mem_usage=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        record['load_seconds']=time.perf_counter()-start
        record['scheduler_config']=dict(pipe.scheduler.config)
        record['sample_sigmas']=pipe.config.sample_sigmas
        print(f"Loaded Qwen in {record['load_seconds']:.1f}s",flush=True)
        for ref in refs:
            view_id=ref['id']; dest=out/view_id;dest.mkdir()
            source=prepared/ref['files']['source_rgb.png']['path']
            shutil.copy2(source,dest/'source.png')
            row=dict(id=view_id,label=ref['label'],source_path=str(source),source_sha256=helpers.sha256(source))
            if view_id=='side_004':
                if helpers.sha256(args.reuse_side/'source.png')!=row['source_sha256']:
                    raise ValueError('Previously accepted side input differs')
                previous = json.loads((args.reuse_side/'manifest.json').read_text())['variants'][args.variant]
                if previous['true_cfg_scale'] != cfg or previous['negative_prompt'] != negative:
                    raise ValueError('Previously reviewed side candidate uses different model settings')
                shutil.copy2(args.reuse_side/args.variant/'raw.png',dest/'raw.png')
                shutil.copy2(args.reuse_side/args.variant/'clean.png',dest/'accepted_clean.png')
                shutil.copy2(args.reuse_side/args.variant/'prompt.txt',dest/'prompt.txt')
                row.update(mode='reused_reviewed_side_candidate',reuse_directory=str(args.reuse_side.resolve()),reuse_variant=args.variant)
            else:
                (dest/'prompt.txt').write_text(PROMPT+'\n')
                print('Generating '+view_id,flush=True)
                record['status']='generating_'+view_id;save()
                torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                def on_step(pipeline, step, timestep, values):
                    if not torch.isfinite(values['latents']).all().item():
                        raise FloatingPointError(f'Invalid latents {view_id} step {step}')
                    progress=dict(view=view_id,step=step+1,total=pipeline.num_timesteps,seconds=round(time.perf_counter()-started,2))
                    helpers.save_json(out/'progress.json',progress)
                    if (step+1)%10==0: print(json.dumps(progress),flush=True)
                    return values
                with torch.inference_mode():
                    image=pipe(image=Image.open(source).convert('RGB'),prompt=PROMPT,
                        width=1536,height=1024,output_resolution=1254,num_inference_steps=40,
                        true_cfg_scale=cfg,negative_prompt=negative,use_kv_cache=True,
                        generator=torch.Generator(device='cuda').manual_seed(42),
                        callback_on_step_end=on_step).images[0]
                torch.cuda.synchronize()
                if image.size!=(1536,1024): raise ValueError('Qwen changed image dimensions')
                image.save(dest/'raw.png')
                row.update(mode='Qwen_generated',seconds=time.perf_counter()-started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            row['raw_sha256']=helpers.sha256(dest/'raw.png')
            record['views'].append(row);save()
        record['status']='raw_candidates_complete_pending_protection_and_review'
        record['sources_unchanged']=all(helpers.sha256(Path(r['source_path']))==r['source_sha256'] for r in record['views'])
        if not record['sources_unchanged']: raise ValueError('Source files changed')
        save();print('MULTIVIEW_QWEN_COMPLETE',flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    main()
