"""Two single-image Qwen reflection-removal candidates; no 3D inputs or writes.

The model is loaded once, and each prompt edits the SAME original reference with
the SAME seed. Raw predictions are preserved. Region-protected compositing is a
separate, explicitly recorded postprocess, not a native Qwen mask/strength API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
HELPER = PROJECT / "02_qwen_image_edit/run.py"
spec = importlib.util.spec_from_file_location("qwen_reflection_helpers", HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROJECT / "test_images/宝马5系/side/5_2026_255841_151515_004.jpg")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/Qwen-Image-2.1")
    parser.add_argument("--model-revision", default="d26bb61231c349cf6b7896fa83353113880e1ba3")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-dir", type=Path, default=METHOD / "prompts")
    parser.add_argument("--true-cfg-scale", type=float, default=1.)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--strong-cfg-scale", type=float, default=None)
    parser.add_argument("--strong-negative-prompt", default=None)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    env = helpers.configure_environment()
    if any(x <= 0 or x % 32 for x in (args.width, args.height)) or args.steps <= 0:
        raise ValueError("Require positive multiples of 32 for dimensions and positive steps")
    source, model_dir, out = args.source.resolve(), args.model_dir.resolve(), args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Select a fresh output directory: {out}")
    if not out.is_relative_to(METHOD / "outputs"):
        raise ValueError("This experiment only writes inside 09_reflection_cleanup/outputs")
    revision = (model_dir / ".cache/huggingface/download/model_index.json.metadata").read_text().splitlines()[0]
    if revision != args.model_revision:
        raise ValueError("Downloaded model revision does not match the pinned revision")
    with Image.open(source) as original_file:
        original = ImageOps.exif_transpose(original_file).convert("RGB")
    canvas, working, mapping = helpers.prepare_image(original, args.width, args.height)
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    resolution = round(math.sqrt(args.width * args.height))
    actual = calculate_dimensions(resolution ** 2, canvas.width / canvas.height)[:2]
    if actual != canvas.size:
        raise ValueError(f"Unexpected pipeline reference resize: {actual}")
    prompt_files = {name: args.prompts_dir / f"{name}.txt" for name in ("mild", "strong")}
    prompts = {name: file.read_text().strip() for name, file in prompt_files.items()}
    if not all(prompts.values()):
        raise ValueError("Both prompts must be nonempty")
    if args.check_config:
        print(json.dumps({"source": str(source), "mapping": mapping, "model_revision": revision,
                          "prompt_characters": {k:len(v) for k,v in prompts.items()}}, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True)
    manifest = {
        "method": "Qwen_Image_2_1_single_image_reflection_edit_pair",
        "status": "loading", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": helpers.sha256(source),
        "model_dir": str(model_dir), "model_revision": revision,
        "model_index_sha256": helpers.sha256(model_dir / "model_index.json"),
        "mapping": mapping, "model_canvas": [args.width, args.height],
        "seed": args.seed, "num_inference_steps": args.steps, "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "strong_cfg_override": args.strong_cfg_scale,
        "strong_negative_prompt_override": args.strong_negative_prompt,
        "use_kv_cache": True, "vae_tiling": False,
        "environment_overrides": env, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": helpers.sha256(__file__), "helper_sha256": helpers.sha256(HELPER),
        "prompt_sha256": {name: helpers.sha256(path) for name, path in prompt_files.items()},
        "constraints": "Qwen prompt-only generation has no native mask/strength/region lock. Raw images may drift; protection is a separate postprocess.",
        "postprocessing": "Raw output only has recorded input padding removed; no warp, paste-back, color normalization, sharpening or upscaling.",
        "three_d_policy": "No 3D files are loaded or modified by this script.",
        "variants": {},
    }
    save = lambda: helpers.save_json(out / "manifest.json", manifest)
    working.save(out / "source.png")
    canvas.save(out / "source_canvas.png")
    shutil.copy2(__file__, out / "run_snapshot.py")
    shutil.copy2(HELPER, out / "helper_snapshot.py")
    for name, prompt_file in prompt_files.items():
        (out / name).mkdir()
        shutil.copy2(prompt_file, out / name / "prompt.txt")
    packages = sorted((p.metadata["Name"], p.version) for p in importlib.metadata.distributions() if p.metadata["Name"])
    (out / "environment_freeze.txt").write_text("".join(f"{name}=={version}\n" for name, version in packages))
    start = time.perf_counter()
    save()
    try:
        import torch
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Need a CUDA GPU supporting BF16")
        manifest.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0), pipeline_source_sha256=helpers.sha256(inspect.getfile(QwenImage21Pipeline)))
        manifest["gpu_before"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv,noheader"], text=True)
        print("Loading local Qwen-Image-2.1 on selected GPU", flush=True)
        load_start = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16,
            local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        manifest["load_seconds"] = time.perf_counter() - load_start
        manifest["scheduler_config"] = dict(pipe.scheduler.config)
        manifest["sample_sigmas"] = pipe.config.sample_sigmas
        print(f"Loaded in {manifest['load_seconds']:.1f}s", flush=True)
        for name, prompt in prompts.items():
            cfg = args.strong_cfg_scale if name == "strong" and args.strong_cfg_scale is not None else args.true_cfg_scale
            negative = args.strong_negative_prompt if name == "strong" and args.strong_negative_prompt is not None else args.negative_prompt
            print(f"Generating {name} from the unmodified original", flush=True)
            manifest["status"] = f"generating_{name}"
            torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            step_times = []
            def on_step(pipeline, step, timestep, values):
                if not torch.isfinite(values["latents"]).all().item():
                    raise FloatingPointError(f"Non-finite latents in {name} step {step+1}")
                elapsed = time.perf_counter() - inference_start
                step_times.append(elapsed)
                progress = {"variant": name, "completed_steps": step + 1,
                            "total_steps": pipeline.num_timesteps, "elapsed_seconds": round(elapsed, 2),
                            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
                helpers.save_json(out / "progress.json", progress)
                if (step+1) % 5 == 0 or step == 0:
                    print(json.dumps(progress), flush=True)
                return values
            save()
            with torch.inference_mode():
                result = pipe(image=canvas, prompt=prompt, width=args.width, height=args.height,
                    output_resolution=resolution, num_inference_steps=args.steps,
                    true_cfg_scale=cfg, negative_prompt=negative,
                    use_kv_cache=True, generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    callback_on_step_end=on_step).images[0]
            torch.cuda.synchronize()
            if result.size != canvas.size:
                raise ValueError(f"Unexpected generated size: {result.size}")
            result.save(out / name / "raw_canvas.png")
            edited = result.crop(mapping["content_box_xyxy"])
            edited.save(out / name / "raw.png")
            manifest["variants"][name] = {
                "prompt": prompt, "true_cfg_scale": cfg, "negative_prompt": negative,
                "inference_seconds": time.perf_counter() - inference_start,
                "completed_steps": len(step_times), "step_end_elapsed_seconds": step_times,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "output_mode": edited.mode, "raw_sha256": helpers.sha256(out / name / "raw.png")}
            save()
        manifest["source_unchanged"] = helpers.sha256(source) == manifest["source_sha256"]
        if not manifest["source_unchanged"]:
            raise RuntimeError("Original source file changed")
        manifest["status"] = "raw_candidates_complete_pending_protection_and_review"
        manifest["total_seconds"] = time.perf_counter() - start
        save()
        print("QWEN_PAIR_COMPLETE", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                        total_seconds=time.perf_counter() - start)
        save()
        raise


if __name__ == "__main__":
    main()
