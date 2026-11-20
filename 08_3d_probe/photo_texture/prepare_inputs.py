"""Prepare the eight BMW reference photographs for geometry fitting and texturing.

Uses the preinstalled rembg u2net model on CPU. Source pixels are only resized;
the original photographs are never changed. Preserve raw model alpha separately
from any explicitly requested small, enclosed-hole repair.

Example (from the project root):
  OMP_NUM_THREADS=4 /nas/wyh/.conda/envs/car-3d-hunyuan/bin/python \
    08_3d_probe/photo_texture/prepare_inputs.py \
    --output-dir 08_3d_probe/photo_texture/outputs/bmw5_inputs_v2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "5_2026_255841_151515"
# Cardinal views first, so the shape-generation experiment can start immediately.
# Left/right are the vehicle's own sides; degrees below are approximate category
# centers, not estimated or calibrated camera poses.
VIEWS = [
    ("front", "002", "front", 0, "正前"),
    ("rear", "006", "back", 180, "正后"),
    ("side", "004", "left", 90, "车辆左侧；车头朝画面左"),
    ("side", "008", "right", 270, "车辆右侧；车头朝画面右"),
    ("front", "001", "front_left", 45, "左前斜视；车头朝画面左"),
    ("front", "003", "front_right", 315, "右前斜视；车头朝画面右"),
    ("rear", "005", "back_left", 135, "左后斜视；车头朝画面左"),
    ("rear", "007", "back_right", 225, "右后斜视；车头朝画面右"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def repair_small_holes(alpha: np.ndarray, maximum_area: int) -> tuple[np.ndarray, int]:
    if maximum_area == 0:
        return alpha.copy(), 0
    binary = alpha >= 128
    enclosed = ndimage.binary_fill_holes(binary) & ~binary
    labels, count = ndimage.label(enclosed)
    areas = np.bincount(labels.ravel())
    accepted = (areas <= maximum_area) & (areas > 0)
    accepted[0] = False
    fill = accepted[labels]
    repaired = alpha.copy()
    repaired[fill] = 255
    return repaired, int(fill.sum())


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def contact_sheet(output: Path, records: list[dict]) -> None:
    width, row_height = 1200, 300
    sheet = Image.new("RGB", (width, row_height * len(records)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        folder = output / record["id"]
        for col, filename in enumerate(("source_rgb.png", "foreground_rgba.png", "mask.png")):
            im = Image.open(folder / filename)
            if im.mode == "RGBA":
                backdrop = Image.new("RGBA", im.size, (150, 170, 180, 255))
                im = Image.alpha_composite(backdrop, im).convert("RGB")
            else:
                im = im.convert("RGB")
            im = ImageOps.contain(im, (390, 270), Image.Resampling.LANCZOS)
            sheet.paste(im, (col * 400 + (400 - im.width) // 2, row * row_height + 25))
            draw.text((col * 400 + 8, row * row_height + 5), f'{record["id"]} {filename}', fill=(0, 0, 0))
    sheet.save(output / "contact_sheet.jpg", quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "test_images/宝马5系")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=1536)
    parser.add_argument("--fill-small-holes", type=int, default=512,
                        help="Maximum enclosed hole area in working pixels; set 0 to preserve raw alpha")
    args = parser.parse_args()
    if args.long_side <= 0 or args.fill_small_holes < 0:
        parser.error("long-side must be positive and fill-small-holes must be nonnegative")
    output = args.output_dir.resolve()
    inputs = args.input_root.resolve()
    if output == inputs or inputs.is_relative_to(output):
        parser.error("Output cannot contain the original input directory")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output}")
    model_path = Path(os.environ.get("U2NET_HOME", str(Path.home() / ".u2net"))) / "u2net.onnx"
    if not model_path.is_file():
        raise SystemExit(f"Preinstall u2net locally before running: {model_path}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from rembg import new_session, remove

    session = new_session("u2net", providers=["CPUExecutionProvider"])
    providers = session.inner_session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected ONNX providers: {providers}")
    records: list[dict] = []
    manifest = {
        "method": "rembg_u2net_cpu_reference_preparation",
        "status": "running",
        "input_root": str(inputs),
        "long_side": args.long_side,
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "script_sha256": sha256_file(Path(__file__)),
        "providers": providers,
        "versions": {name: importlib.metadata.version(name) for name in ("rembg", "onnxruntime", "pillow", "numpy", "scipy")},
        "mask_threshold": 128,
        "bbox_convention": "[left, top, right_exclusive, bottom_exclusive] in working image pixels",
        "angle_convention": "Nominal view categories only: front=0, vehicle left=90, back=180, vehicle right=270; NOT calibrated camera poses",
        "postprocessing": {"alpha_matting": False, "rembg_post_process_mask": False,
                           "fill_enclosed_holes_maximum_area": args.fill_small_holes,
                           "geometry_mask_largest_connected_component": True,
                           "crop": False, "source_rgb_edited": False},
        "views": records,
    }
    started = time.perf_counter()
    save_json(output / "manifest.json", manifest)
    for category, number, semantic, nominal_angle, label in VIEWS:
        path = inputs / category / f"{PREFIX}_{number}.jpg"
        before_hash = sha256_file(path)
        with Image.open(path) as opened:
            original_size = opened.size
            source = ImageOps.exif_transpose(opened).convert("RGB")
        oriented_size = source.size
        scale = min(1.0, args.long_side / max(source.size))
        working_size = tuple(max(1, round(d * scale)) for d in source.size)
        source = source.resize(working_size, Image.Resampling.LANCZOS)
        identifier = f"{category}_{number}"
        folder = output / identifier
        folder.mkdir()
        source.save(folder / "source_rgb.png")
        tick = time.perf_counter()
        raw_alpha = np.asarray(remove(source, session=session, only_mask=True,
                                      alpha_matting=False, post_process_mask=False).convert("L"))
        alpha, filled = repair_small_holes(raw_alpha, args.fill_small_holes)
        for name, array in (("alpha_raw.png", raw_alpha), ("alpha.png", alpha),
                            ("mask_raw.png", (raw_alpha >= 128).astype(np.uint8) * 255),
                            ("mask.png", (alpha >= 128).astype(np.uint8) * 255)):
            Image.fromarray(array).save(folder / name)
        rgba = source.copy().convert("RGBA")
        rgba.putalpha(Image.fromarray(raw_alpha))
        rgba.save(folder / "foreground_raw_rgba.png")
        rgba.putalpha(Image.fromarray(alpha))
        rgba.save(folder / "foreground_rgba.png")
        solid = alpha >= 128
        if not solid.any():
            raise RuntimeError(f"rembg returned an empty foreground for {path}")
        component_labels, _ = ndimage.label(solid)
        component_sizes = np.bincount(component_labels.ravel())
        component_sizes[0] = 0
        geometry_mask = component_labels == int(component_sizes.argmax())
        Image.fromarray(geometry_mask.astype(np.uint8) * 255).save(folder / "mask_geometry.png")
        holes = ndimage.binary_fill_holes(solid) & ~solid
        hole_labels, count = ndimage.label(holes)
        hole_areas = np.bincount(hole_labels.ravel())[1:]
        record = {
            "id": identifier, "view": semantic, "label": label,
            "nominal_azimuth_deg": nominal_angle, "pose_calibrated": False,
            "source_path": str(path), "source_sha256": before_hash,
            "original_size": list(original_size), "exif_oriented_size": list(oriented_size),
            "working_size": list(working_size),
            "scale_xy": [working_size[0] / oriented_size[0], working_size[1] / oriented_size[1]],
            "bbox_xyxy": bbox_xyxy(solid), "raw_bbox_xyxy": bbox_xyxy(raw_alpha >= 128),
            "geometry_bbox_xyxy": bbox_xyxy(geometry_mask),
            "geometry_removed_component_pixels": int(solid.sum() - geometry_mask.sum()),
            "foreground_fraction": float(solid.mean()),
            "filled_hole_pixels": filled,
            "remaining_enclosed_holes": {"count": int(count), "total_pixels": int(holes.sum()),
                                         "largest_area": int(hole_areas.max()) if len(hole_areas) else 0},
            "seconds": round(time.perf_counter() - tick, 3),
            "files": {p.name: {"path": str(p.relative_to(output)), "sha256": sha256_file(p)}
                      for p in sorted(folder.glob("*.png"))},
        }
        if sha256_file(path) != before_hash:
            raise RuntimeError(f"Original file changed: {path}")
        save_json(folder / "metadata.json", record)
        records.append(record)
        save_json(output / "manifest.json", manifest)
        print(json.dumps({k: record[k] for k in ("id", "view", "bbox_xyxy", "seconds", "remaining_enclosed_holes")}, ensure_ascii=False), flush=True)
    contact_sheet(output, records)
    manifest.update(status="complete_pending_visual_review", total_seconds=round(time.perf_counter() - started, 3),
                    original_files_unchanged=True)
    save_json(output / "manifest.json", manifest)
    print(f"Prepared {len(records)} views at {output}", flush=True)


if __name__ == "__main__":
    main()
