"""Generate vehicle geometry from four real cardinal views using Hunyuan3D-2mv.

Run from the project root with the existing Hunyuan environment::

    /nas/wyh/.conda/envs/car-3d-hunyuan/bin/python \
      08_3d_probe/multiview_probe/run_multiview.py \
      --output-dir 08_3d_probe/multiview_probe/outputs/bmw5_mv_seed42_v1

The original full-resolution photographs and the prepared RGBA foregrounds are
hashed separately. All four slots are genuine photographs, not synthesized views.
The default run is restricted to physical GPU 2 before torch is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = Path("/nas/wyh/3d_tools/Hunyuan3D-2")
MODEL_REVISION = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
INPUTS = {
    "front": ("front_002", "front/5_2026_255841_151515_002.jpg"),
    "left": ("side_004", "side/5_2026_255841_151515_004.jpg"),
    "back": ("rear_006", "rear/5_2026_255841_151515_006.jpg"),
    "right": ("side_008", "side/5_2026_255841_151515_008.jpg"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/Hunyuan3D-2mv")
    parser.add_argument("--prepared-root", type=Path, default=PROJECT_ROOT / "08_3d_probe/photo_texture/outputs/bmw5_inputs_v2")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "test_images/宝马5系")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", default="2", choices=["2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-chunks", type=int, default=8000)
    args = parser.parse_args()

    # Set visibility before importing torch or any third-party model package.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires physical CUDA GPU 2.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {args.output_dir}")
    weights = args.model_dir / "hunyuan3d-dit-v2-mv/model.fp16.ckpt"
    config = args.model_dir / "hunyuan3d-dit-v2-mv/config.yaml"
    required = [weights, config]
    for _, (name, relative_source) in INPUTS.items():
        required.extend([args.prepared_root / name / "foreground_rgba.png", args.source_root / relative_source])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "inputs"
    input_dir.mkdir()
    record = {
        "experiment": "hunyuan3d_2mv_four_real_views",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "physical_gpu": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "model": {
            "repo_id": "tencent/Hunyuan3D-2mv", "revision": MODEL_REVISION,
            "subfolder": "hunyuan3d-dit-v2-mv", "dtype": "float16",
            "weights_path": str(weights.resolve()), "weights_sha256": sha256_file(weights),
            "config_sha256": sha256_file(config),
        },
        "implementation": {
            "upstream": "https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
            "code_path": str(args.repo.resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "source_files": {name: sha256_file(args.repo / name) for name in [
                "hy3dgen/shapegen/pipelines.py", "hy3dgen/shapegen/preprocessors.py",
                "hy3dgen/shapegen/models/conditioner.py",
            ]},
        },
        "params": {"seed": args.seed, "steps": args.steps,
                   "octree_resolution": args.octree_resolution,
                   "guidance_scale": args.guidance_scale, "num_chunks": args.num_chunks},
        "slot_semantics": {
            "order": ["front", "left", "back", "right"],
            "left": "Side photograph with vehicle nose pointing to image left (official example convention).",
            "right": "Side photograph with vehicle nose pointing to image right.",
            "note": "Cardinal view labels are not calibrated camera poses; no images are mirrored.",
        },
        "inputs": {},
        "limitations": ["Shape-only geometry; no generated texture or identity guarantee.",
                        "Photographs are uncalibrated and independently framed; the model independently recenters each view."],
    }
    images = {}
    for slot, (name, relative_source) in INPUTS.items():
        prepared = args.prepared_root / name / "foreground_rgba.png"
        source = args.source_root / relative_source
        image = Image.open(prepared)
        if image.mode != "RGBA":
            raise ValueError(f"Prepared foreground must contain genuine alpha: {prepared}")
        alpha = np.asarray(image.getchannel("A"))
        if not (np.any(alpha == 0) and np.any(alpha > 127)):
            raise ValueError(f"Prepared foreground has missing or empty alpha: {prepared}")
        copy = input_dir / f"{slot}_rgba.png"
        shutil.copyfile(prepared, copy)
        images[slot] = image.copy()
        record["inputs"][slot] = {
            "source_path": str(source.resolve()), "source_sha256": sha256_file(source),
            "prepared_path": str(prepared.resolve()), "prepared_sha256": sha256_file(prepared),
            "copied_path": str(copy.relative_to(args.output_dir)), "image_size": list(image.size),
            "foreground_fraction_alpha_gt_127": float((alpha > 127).mean()),
        }
    write_json(args.output_dir / "manifest.json", record)

    try:
        print("Loading official Hunyuan3D-2mv standard fp16 weights on physical GPU 2", flush=True)
        t0 = time.perf_counter()
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(args.model_dir.resolve()), subfolder="hunyuan3d-dit-v2-mv",
            variant="fp16", use_safetensors=False, device="cuda", dtype=torch.float16,
        )
        torch.cuda.synchronize()
        record["load_seconds"] = round(time.perf_counter() - t0, 3)
        # Persist what the model sees after the official per-view preprocessing.
        normalized = pipe.image_processor(images)
        for i, slot in enumerate(record["slot_semantics"]["order"]):
            tensor = normalized["image"][0, i]
            rgb = ((tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(input_dir / f"{slot}_model_input.png")
        record["model_input_shape"] = list(normalized["image"].shape)
        record["model_view_indices"] = list(normalized["view_idxs"])

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        print(f"Generating mesh: seed={args.seed}, steps={args.steps}, octree={args.octree_resolution}", flush=True)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        mesh = pipe(image=images, num_inference_steps=args.steps,
                    octree_resolution=args.octree_resolution, guidance_scale=args.guidance_scale,
                    num_chunks=args.num_chunks, generator=generator, output_type="trimesh")[0]
        torch.cuda.synchronize()
        record["generation_seconds"] = round(time.perf_counter() - t0, 3)
        record["peak_allocated_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if mesh is None or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Pipeline did not return a finite, nonempty mesh")
        mesh_path = args.output_dir / "mesh.glb"
        mesh.export(mesh_path, include_normals=True)
        record["mesh"] = {
            "path": "mesh.glb", "sha256": sha256_file(mesh_path),
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "bounds": mesh.bounds.tolist(), "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
        }
        record["status"] = "complete"
        print(json.dumps({key: record[key] for key in ["generation_seconds", "peak_allocated_gpu_gib", "mesh"]}, indent=2), flush=True)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output_dir / "manifest.json", record)


if __name__ == "__main__":
    main()
