"""Build a two-column local gallery for the completed 3-background/4-car batch.

This script reads generated images only. Preview resizing and RGBA compositing
are for display; source images and Qwen outputs are never modified.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
FONT_PATHS = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int):
    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("A Chinese font is required for the report labels")


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode == "RGBA" or "transparency" in image.info:
            image = image.convert("RGBA")
            white = Image.new("RGBA", image.size, "white")
            return Image.alpha_composite(white, image).convert("RGB")
        return image.convert("RGB")


def fitted_text(draw, xy, text, size, max_width, fill):
    selected = font(size)
    while draw.textlength(text, font=selected) > max_width and size > 16:
        size -= 1
        selected = font(size)
    draw.text(xy, text, font=selected, fill=fill)


def grid_image(title: str, panels: list[tuple[str, Path]], destination: Path):
    """Four 4:3 display areas in a 2x2 grid, preserving each image's ratio."""
    if len(panels) != 4:
        raise ValueError("Each grouped preview must contain four panels")
    image_width, image_height = 768, 576
    margin, gap, title_height, label_height = 20, 16, 100, 52
    cell_height = label_height + image_height
    board = Image.new(
        "RGB",
        (2 * image_width + 2 * margin + gap,
         title_height + 2 * cell_height + gap + margin),
        "#eef2f5",
    )
    draw = ImageDraw.Draw(board)
    fitted_text(draw, (margin, 14), title, 35, board.width - 2 * margin, "#16334e")
    draw.text((margin, 61), "每行两张 · 按原比例显示 · 点击本地图库中的图片可查看原尺寸", font=font(23), fill="#526478")
    for index, (label, path) in enumerate(panels):
        left = margin + (index % 2) * (image_width + gap)
        top = title_height + (index // 2) * (cell_height + gap)
        draw.rectangle((left, top, left + image_width - 1, top + cell_height - 1), fill="white")
        fitted_text(draw, (left + 12, top + 9), label, 28, image_width - 24, "#17334d")
        display = ImageOps.contain(read_rgb(path), (image_width, image_height), Image.Resampling.LANCZOS)
        board.paste(display, (left + (image_width - display.width) // 2,
                              top + label_height + (image_height - display.height) // 2))
    board.save(destination)


def url_from(report: Path, path: Path) -> str:
    return quote(Path(os.path.relpath(path, report)).as_posix(), safe="/")


def validate_batch(config_path: Path, batch_dir: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = config.get("jobs", [])
    if len(jobs) != 12:
        raise ValueError("Expected exactly 12 jobs (3 backgrounds x 4 vehicles)")
    ids, combinations = set(), set()
    backgrounds, vehicles = OrderedDict(), OrderedDict()
    source_hashes = {}
    vehicle_source_hashes = {}
    for job in jobs:
        for key in ("id", "vehicle_id", "background_id"):
            if not isinstance(job.get(key), str) or not SAFE_ID.fullmatch(job[key]):
                raise ValueError(f"Invalid {key}: {job.get(key)!r}")
        if job["id"] in ids:
            raise ValueError(f"Duplicate job id: {job['id']}")
        ids.add(job["id"])
        pair = (job["background_id"], job["vehicle_id"])
        if pair in combinations:
            raise ValueError(f"Duplicate background/vehicle pair: {pair}")
        combinations.add(pair)
        for key in ("vehicle_label", "vehicle_name", "background_label", "config"):
            if not isinstance(job.get(key), str) or not job[key]:
                raise ValueError(f"Missing string {key} in {job['id']}")
        backgrounds.setdefault(job["background_id"], job["background_label"])
        vehicles.setdefault(job["vehicle_id"], job["vehicle_name"])
        if backgrounds[job["background_id"]] != job["background_label"]:
            raise ValueError("Background labels must agree across jobs")
        if vehicles[job["vehicle_id"]] != job["vehicle_name"]:
            raise ValueError("Vehicle names must agree across jobs")
        directory = batch_dir / "jobs" / job["id"]
        manifest_path = directory / "manifest.json"
        job_config_path = (PROJECT / job["config"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        job_config = json.loads(job_config_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in ("complete_pending_visual_review", "complete"):
            raise ValueError(f"Job has not completed: {job['id']}")
        if manifest.get("config") != job_config:
            raise ValueError(f"Current config differs from run snapshot: {job['id']}")
        if manifest.get("completed_steps") != job_config["num_inference_steps"]:
            raise ValueError(f"Inference steps incomplete: {job['id']}")
        if not manifest.get("input_files_unchanged"):
            raise ValueError(f"Input preservation unverified: {job['id']}")
        sizes = []
        for name in ("edited.png", "source_working.png", "previews/overview.png"):
            path = directory / name
            digest = sha256(path)
            if manifest.get("output_sha256", {}).get(name) != digest:
                raise ValueError(f"Output hash mismatch or absent: {path}")
            with Image.open(path) as opened:
                if name != "previews/overview.png":
                    sizes.append(opened.size)
                opened.verify()
            source_hashes[str(path.relative_to(batch_dir))] = digest
        if sizes[0] != sizes[1]:
            raise ValueError(f"Source/output canvas mismatch: {job['id']}")
        source_key = str((directory / "source_working.png").relative_to(batch_dir))
        digest = source_hashes[source_key]
        previous = vehicle_source_hashes.setdefault(job["vehicle_id"], digest)
        if previous != digest:
            raise ValueError(f"Vehicle references differ across backgrounds: {job['vehicle_id']}")
        source_hashes[str(manifest_path.relative_to(batch_dir))] = sha256(manifest_path)
        job["_directory"] = directory
    if len(backgrounds) != 3 or len(vehicles) != 4:
        raise ValueError("Batch must have exactly 3 backgrounds and 4 vehicles")
    expected = {(background, vehicle) for background in backgrounds for vehicle in vehicles}
    if combinations != expected:
        raise ValueError("Batch is not a complete 3x4 Cartesian product")
    return config, jobs, backgrounds, vehicles, source_hashes


def build_html(report, title, jobs, backgrounds, vehicles, background_previews, vehicle_previews):
    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{escape(title)}</title>",
            """<style>
body{margin:0;background:#eef2f5;color:#16334e;font:16px/1.65 system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:24px}h1{font-size:30px;line-height:1.3}
h2{margin-top:40px;border-bottom:1px solid #bacbd8;padding-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
article,details{background:white;border-radius:10px;padding:14px}figure{margin:0}
img{display:block;width:100%;height:auto;background:white}a{color:#09599d}
.note{max-width:1000px}.links{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}
summary{cursor:pointer;font-weight:600}summary+div{margin-top:16px}
@media(max-width:760px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}
</style><main>""",
            f"<h1>{escape(title)}</h1>",
            "<p class='note'>12 张 Qwen 背景替换结果：3 个背景 × 4 辆车。每行最多两张，点击成图可打开原尺寸 PNG。允许车辆整体调整大小、位置；重点核对车系判别特征与自然融合。本轮未进行车系分类器评估。</p>",
            "<p class='note'>以下分组图仅等比缩小用于展示，生成原图未作美化或贴回。原固定坐标部件 ROI 只作位置参考，车辆移动后不一定对应同一部件。</p>",
            "<nav class='links'>"]
    for background_id, label in backgrounds.items():
        html.append(f"<a href='#background-{background_id}'>{escape(label)}</a>")
    html.append("<a href='#by-vehicle'>按车辆对比</a></nav>")
    for background_id, label in backgrounds.items():
        preview_url = url_from(report, background_previews[background_id])
        html.append(f"<section id='background-{background_id}'><h2>{escape(label)}</h2>")
        html.append(f"<p><a href='{preview_url}'>打开该背景的四车 2×2 总览</a></p><div class='grid'>")
        for job in jobs:
            if job["background_id"] != background_id:
                continue
            directory = job["_directory"]
            result_url = url_from(report, directory / "edited.png")
            original_url = url_from(report, directory / "source_working.png")
            pair_url = url_from(report, directory / "previews/overview.png")
            name = escape(job["vehicle_name"])
            html.append(f"<article><h3>{name}</h3><a href='{result_url}'><img loading='lazy' src='{result_url}' alt='{name}：{escape(label)}'></a>")
            html.append(f"<div class='links'><a href='{result_url}'>原尺寸成图</a><a href='{original_url}'>原图（同尺寸）</a><a href='{pair_url}'>原图／生成整图对照</a></div></article>")
        html.append("</div></section>")
    html.append("<section id='by-vehicle'><h2>按车辆：原图与三个背景</h2><p>每张对比图为 2×2：左上为缩至生成尺寸的原图，其余按背景顺序排列。</p><div class='grid'>")
    for vehicle_id, name in vehicles.items():
        preview_url = url_from(report, vehicle_previews[vehicle_id])
        html.append(f"<details><summary>{escape(name)}</summary><div><a href='{preview_url}'><img loading='lazy' src='{preview_url}' alt='{escape(name)}：原图与三个背景'></a></div></details>")
    html.append("</div></section><p><a href='report_manifest.json'>文件来源与展示处理记录</a></p></main></html>")
    (report / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path, batch_dir = args.batch_config.resolve(), args.batch_dir.resolve()
    config, jobs, backgrounds, vehicles, hashes = validate_batch(config_path, batch_dir)
    report = batch_dir / "report"
    if report.exists():
        raise FileExistsError(f"Report already exists; use a fresh batch directory: {report}")
    font(28)  # Validate the display dependency before creating the report directory.
    report.mkdir()
    background_previews, vehicle_previews = {}, {}
    for background_id, label in backgrounds.items():
        path = report / f"background_{background_id}.png"
        panels = [(job["vehicle_name"], job["_directory"] / "edited.png")
                  for job in jobs if job["background_id"] == background_id]
        grid_image(f"{label} · 四车背景替换", panels, path)
        background_previews[background_id] = path
    for vehicle_id, name in vehicles.items():
        matches = [job for job in jobs if job["vehicle_id"] == vehicle_id]
        path = report / f"vehicle_{vehicle_id}.png"
        panels = [("原图（同尺寸）", matches[0]["_directory"] / "source_working.png")]
        panels += [(job["background_label"], job["_directory"] / "edited.png") for job in matches]
        grid_image(f"{name} · 原图与三个背景", panels, path)
        vehicle_previews[vehicle_id] = path
    build_html(report, config.get("label", "汽车背景替换：三个背景 × 四辆车"),
               jobs, backgrounds, vehicles, background_previews, vehicle_previews)
    report_files = sorted(report.glob("*.png")) + [report / "index.html"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_config": str(config_path.relative_to(PROJECT)) if config_path.is_relative_to(PROJECT) else str(config_path),
        "batch_config_sha256": sha256(config_path),
        "builder_sha256": sha256(Path(__file__)),
        "job_count": len(jobs),
        "background_order": list(backgrounds),
        "vehicle_order": list(vehicles),
        "display_processing": "RGB white-background compositing if needed, aspect-preserving LANCZOS resizing and labels only; no source or generated image modified.",
        "columns": 2,
        "cell_image_size": [768, 576],
        "geometry_acceptance": "Vehicle translation and uniform scaling are allowed.",
        "classification_evaluation_performed": False,
        "fixed_roi_note": "Fixed-coordinate crops are positional references, not corresponding-part evidence after vehicle movement.",
        "sources_sha256_relative_to_batch": hashes,
        "report_sha256": {path.name: sha256(path) for path in report_files},
    }
    (report / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT_OK {report / 'index.html'}")
    print(f"Grouped previews: {len(background_previews)} by background, {len(vehicle_previews)} by vehicle")


if __name__ == "__main__":
    main()
