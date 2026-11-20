"""Edit the front photograph's windshield with Qwen; save the raw prediction.

This script does not read or write 3D data. Masked source protection is separate.
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


PROMPT = "只修复前挡风玻璃：清除玻璃中透过车厢看到的卷帘门横纹和成排黑色方孔，让这些区域自然融入低对比的车厢暗部。保留座椅、头枕、方向盘、内后视镜、黄色贴纸及中下部浅灰色包覆物的位置和形状。玻璃保留柔和自然明暗，不要把整块玻璃涂黑，窗框、车身和机位不变。"
NEGATIVE = "卷帘门横纹，成排矩形孔洞，栅栏图案，全黑不透明玻璃，新增人物，重复座椅，重复后视镜，车身变形。"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prompt-file',type=Path)
    p.add_argument('--cfg',type=float,default=2.5)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    prompt=a.prompt_file.read_text().strip() if a.prompt_file else PROMPT
    if not prompt:raise ValueError('Empty prompt')
    env=helpers.configure_environment()
    source=PROJECT/'08_3d_probe/photo_texture/outputs/bmw5_inputs_v2/front_002/source_rgb.png'
    prepared=source.parent.parent/'manifest.json'
    ref=next(x for x in json.loads(prepared.read_text())['views'] if x['id']=='front_002')
    if helpers.sha256(source)!=ref['files']['source_rgb.png']['sha256']:
        raise ValueError('Prepared source changed')
    im=Image.open(source).convert('RGB')
    if im.size!=(1536,1024):raise ValueError('Unexpected working image dimensions')
    model=PROJECT/'models/Qwen-Image-2.1'
    revision=(model/'.cache/huggingface/download/model_index.json.metadata').read_text().splitlines()[0]
    if revision!='d26bb61231c349cf6b7896fa83353113880e1ba3':raise ValueError('Unexpected local Qwen revision')
    out=a.output_dir.resolve()
    if out.exists() or not out.is_relative_to(METHOD/'outputs'):
        raise ValueError('Use a fresh output directory inside 09_reflection_cleanup/outputs')
    out.mkdir(parents=True)
    shutil.copy2(source,out/'source.png');shutil.copy2(__file__,out/'run_snapshot.py')
    (out/'prompt.txt').write_text(prompt+'\n')
    (out/'negative_prompt.txt').write_text(NEGATIVE+'\n')
    packages=sorted((x.metadata['Name'],x.version) for x in importlib.metadata.distributions() if x.metadata['Name'])
    (out/'environment_freeze.txt').write_text(''.join(f'{n}=={v}\n' for n,v in packages))
    r=dict(status='loading',started_utc=datetime.now(timezone.utc).isoformat(),
        method='Qwen_single_photo_windshield_background_cleanup',source=str(source),
        source_sha256=helpers.sha256(source),original_photo=ref['source_path'],
        original_photo_sha256=ref['source_sha256'],model=str(model),model_revision=revision,
        prompt=prompt,negative_prompt=NEGATIVE,true_cfg_scale=a.cfg,seed=a.seed,steps=40,
        size_wh=[1536,1024],input_crop=None,warp=None,use_kv_cache=True,vae_tiling=False,
        environment_overrides=env,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        script_sha256=helpers.sha256(__file__),
        scope='Only a 2D candidate; Qwen has no region lock. Original-pixel protection is a separate operation.',
        limitation='Generated removal of photographed background does not recover true hidden interior geometry or transparency.')
    save=lambda:helpers.save_json(out/'manifest.json',r)
    save()
    try:
        import torch
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        r.update(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),pipeline_source_sha256=helpers.sha256(inspect.getfile(QwenImage21Pipeline)))
        start=time.perf_counter()
        pipe=QwenImage21Pipeline.from_pretrained(str(model),dtype=torch.bfloat16,
            local_files_only=True,use_safetensors=True,low_cpu_mem_usage=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize();r['load_seconds']=time.perf_counter()-start
        r['scheduler_config']=dict(pipe.scheduler.config);r['sample_sigmas']=pipe.config.sample_sigmas
        r['status']='generating';save();print('Generating windshield edit',flush=True)
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        def step_callback(pipeline,step,timestep,values):
            if not torch.isfinite(values['latents']).all().item():raise FloatingPointError('Non-finite latents')
            progress=dict(step=step+1,total=pipeline.num_timesteps,seconds=round(time.perf_counter()-started,2))
            helpers.save_json(out/'progress.json',progress)
            if (step+1)%10==0:print(json.dumps(progress),flush=True)
            return values
        with torch.inference_mode():
            edited=pipe(image=im,prompt=prompt,width=1536,height=1024,output_resolution=1254,
                num_inference_steps=40,true_cfg_scale=a.cfg,negative_prompt=NEGATIVE,use_kv_cache=True,
                generator=torch.Generator(device='cuda').manual_seed(a.seed),callback_on_step_end=step_callback).images[0]
        torch.cuda.synchronize()
        if edited.size!=im.size:raise ValueError('Qwen output dimensions changed')
        edited.save(out/'raw.png')
        r.update(status='raw_complete_pending_mask_and_visual_review',raw_sha256=helpers.sha256(out/'raw.png'),
            raw_mode=edited.mode,inference_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            source_unchanged=helpers.sha256(source)==r['source_sha256'])
        if not r['source_unchanged']:raise ValueError('Source changed')
        save();print('WINDSHIELD_RAW_COMPLETE',flush=True)
    except Exception as error:
        r.update(status='failed',error=str(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
