"""Build offline image/part comparisons without modifying generated images.

The short-edge-224 panels are an additional resolution diagnostic, not an
evaluation using a trained classifier. All figure labels are ASCII so a missing
Chinese font cannot hide experiment names.
"""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


BACKGROUND = "#eef1f5"


def report_groups(presets: dict) -> dict:
    """Validate optional group metadata without assuming the number of members.

    Explicit groups keep preset order and may contain any number of members.
    IDs become filenames, so validate them before creating report directories.
    """
    groups = {}
    for name, preset in presets.items():
        group_id = preset.get("group")
        if group_id is None:
            if "group_label" in preset:
                raise ValueError(f"group_label requires group: {name}")
            continue
        if not isinstance(group_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", group_id):
            raise ValueError(f"Invalid report group identifier: {group_id!r}")
        title = preset.get("group_label", group_id.replace("_", " ").title())
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Invalid report group label: {group_id}")
        if group_id not in groups:
            groups[group_id] = (title, [])
        elif title != groups[group_id][0]:
            raise ValueError(f"Conflicting report group labels: {group_id}")
        groups[group_id][1].append(name)
    return groups


def _font(size: int):
    for path in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str,
           width: int, size: int = 17) -> None:
    font = _font(size)
    while size > 10 and draw.textbbox((0, 0), label, font=font)[2] > width:
        size -= 1
        font = _font(size)
    draw.text(xy, label, fill="#1d2939", font=font)


def _tile(image: Image.Image, width: int, height: int, label: str,
          resampling=Image.Resampling.LANCZOS) -> Image.Image:
    """Fit inside a fixed image area, retaining all pixels' geometric extent."""
    tile = Image.new("RGB", (width, height + 32), BACKGROUND)
    thumb = ImageOps.contain(image, (width - 8, height - 8), resampling)
    tile.paste(thumb, ((width - thumb.width) // 2,
                      32 + (height - thumb.height) // 2))
    _label(ImageDraw.Draw(tile), (8, 6), label, width - 16)
    return tile


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _box(region: list, size: tuple[int, int]) -> tuple[int, int, int, int]:
    if (len(region) != 4 or not all(math.isfinite(v) for v in region)
            or not 0 <= region[0] < region[2] <= 1
            or not 0 <= region[1] < region[3] <= 1):
        raise ValueError(f"Invalid normalized crop region: {region}")
    width, height = size
    return (math.floor(region[0] * width), math.floor(region[1] * height),
            math.ceil(region[2] * width), math.ceil(region[3] * height))


def _save_png(path: Path, image: Image.Image, artifacts: list[str], root: Path):
    image.save(path, format="PNG", compress_level=6)
    artifacts.append(path.relative_to(root).as_posix())


def _href(path: Path, report: Path) -> str:
    return quote(os.path.relpath(path, report).replace(os.sep, "/"), safe="/")


def _link(path: Path, report: Path, label: str) -> str:
    return f'<a href="{_href(path, report)}">{html.escape(label)}</a>'


def _figure(path: Path, report: Path, alt: str) -> str:
    href = _href(path, report)
    return (f'<a href="{href}"><img loading="lazy" src="{href}" '
            f'alt="{html.escape(alt, quote=True)}"></a>')


def _write_html(output: Path, rows: list, presets: dict, groups: dict) -> Path:
    report = output / "report"
    report.mkdir(exist_ok=True)
    preset_ids = list(presets)
    title = f"成像质量增强：{len(rows)} 张原图 × {len(presets)} 种预设"
    parameter_rows = []
    for name, preset in presets.items():
        parameters = {key: value for key, value in preset.items() if key != "label"}
        parameter_rows.append(
            f'<tr><td>{html.escape(name)}</td>'
            f'<td>{html.escape(preset["label"])}</td>'
            f'<td><code>{html.escape(json.dumps(parameters, ensure_ascii=False))}</code></td></tr>')
    sections = [
        "<h2>参数与处理方式</h2>",
        "<p>各预设独立作用于同一原图，不连续编辑上一张输出。失焦与运动模糊在近似线性 RGB "
        "中处理；传感器噪声为信号相关散粒噪声与读出噪声的近似。配置 grain_short_edge 时，"
        "仅在较小网格生成随机残差，再将残差放大加回原图，模拟较粗的空间相关颗粒；原图内容不随噪声网格缩放。"
        "该参数不是实际设备或 ISO 标定。低分辨率预设先缩小再恢复原尺寸。"
        "JPEG 预设直接对原尺寸图像编码并解码，不混入额外降采样。输入本身为 JPEG 时属于再次压缩。"
        "原尺寸 JPEG 码流与低分辨率原生图保存在运行目录，具体路径和有效参数可在运行清单中核对。</p>",
        '<div class="scroll"><table><thead><tr><th>预设</th><th>图中标题</th><th>参数</th>'
        '</tr></thead><tbody>' + "".join(parameter_rows) + "</tbody></table></div>",
        "<h2>整图总览</h2>",
        f"<p>每行一张源图，各列为原图与 {len(presets)} 种增强。点击图片可打开完整预览尺寸。"
        "全部预览以无损 PNG 保存，图中画面为适应展示而缩放；以下链接还提供原尺寸 PNG 和未经缩放的部件裁剪。</p>",
        _figure(output / "previews/overview.png", report, "Original images and all imaging quality presets"),
    ]
    group_labels = {"blur": "模糊：失焦与运动", "noise": "传感器噪声", "resolution": "低分辨率", "compression": "JPEG 压缩"}
    for group_id, (group_title, _) in groups.items():
        group_label = group_labels.get(group_id, group_title)
        sections.extend([
            f"<h2>{html.escape(group_label)}</h2>",
            _figure(output / f"previews/{group_id}_overview.png", report, group_label),
        ])
    for row in rows:
        vehicle_id, label = row["vehicle_id"], row["label"]
        sections.extend([
            f'<section id="{vehicle_id}"><h2>{html.escape(label)}</h2>',
            _figure(output / f"previews/{vehicle_id}_comparison.png", report,
                    label + " full-frame comparisons"),
            "<p>" + _link(Path(row["source"]), report, "源图") + " · "
            + " · ".join(_link(Path(row["outputs"][name]), report, presets[name]["label"] + " PNG")
                         for name in preset_ids) + "</p>",
            "<h3>关键部件：原生分辨率裁剪对照</h3>",
            "<p>每行对应相同位置的部件区域；预览中按比例缩放至固定格子。"
            "下表链接的是未经缩放的同坐标 PNG 裁剪。</p>",
            _figure(output / f"previews/{vehicle_id}_details.png", report,
                    label + " part crops from native-resolution images"),
        ])
        headers = "<th>部件</th><th>原图</th>" + "".join(
            f'<th>{html.escape(presets[name]["label"])}</th>' for name in preset_ids)
        crop_rows = []
        for region in row["scene"]["regions"]:
            cells = []
            for name in ["original"] + preset_ids:
                path = output / "crops" / vehicle_id / region / f"{name}.png"
                cells.append("<td>" + _link(path, report, "PNG") + "</td>")
            crop_rows.append(f"<tr><th>{html.escape(region)}</th>{''.join(cells)}</tr>")
        sections.extend([
            '<div class="scroll"><table><thead><tr>' + headers
            + "</tr></thead><tbody>" + "".join(crop_rows) + "</tbody></table></div>",
            "<details><summary>补充诊断：全图缩至短边 224 后的部件对比</summary>",
            "<p>先将完整图像等比例缩至短边 224，再按同一归一化坐标裁剪部件，"
            "使用最近邻放大便于观察像素。这里的 224 只是统一分辨率诊断，"
            "不是已建立分类器的实际输入配置，也不是分类准确率评估。缩小会削弱噪声、压缩与轻度模糊的可见差异，"
            "应同时检查原生部件裁剪；这类衰减是处理结果的一部分。</p>",
            _figure(output / f"previews/{vehicle_id}_details_224.png", report,
                    label + " short-edge-224 diagnostic, enlarged for display"),
            "</details></section>",
        ])
    body = "\n".join(sections)
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{max-width:1450px;margin:auto;padding:24px;font:16px/1.7 system-ui,sans-serif;color:#1d2939;background:#fafbfc}
h1,h2,h3{line-height:1.35}h2{margin-top:36px}img{display:block;max-width:100%;height:auto;margin:16px 0;border:1px solid #d0d5dd}
a{color:#175cd3}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d0d5dd;padding:9px;text-align:left}
code{font-size:14px;overflow-wrap:anywhere}.scroll{overflow-x:auto}summary{cursor:pointer;font-weight:bold}details{margin:22px 0}
.note{padding:16px;background:#edf4ff;border-left:4px solid #528bff}nav{display:flex;flex-wrap:wrap;gap:14px}
</style></head><body>
<h1>__TITLE__</h1>
<p class="note">本页展示程序化成像质量增强，不使用生成模型。输出画布尺寸和部件坐标保持固定，模糊、噪声、降采样和有损压缩可能损失车型判别信息。低分辨率图放大回原尺寸不会恢复已丢失的细节。
生成完成与文件校验不代表细粒度保真或训练收益通过；请结合运行记录、原尺寸部件和人工检查判断。</p>
""" + "<nav>" + " · ".join([
        _link(output / "manifest.json", report, "运行清单与质量指标"),
        _link(output / "crop_regions.json", report, "裁剪坐标"),
    ] + [f'<a href="#{row["vehicle_id"]}">{html.escape(row["label"])}</a>' for row in rows]) \
        + "</nav>\n" + body + "\n</body></html>\n"
    page = page.replace("__TITLE__", html.escape(title))
    path = report / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_report(output: Path, rows: list, presets: dict) -> dict:
    """Create full-frame, native-part, and short-edge-224 comparisons.

    ``rows`` hold source paths, scene regions, and generated paths keyed by
    preset ID. This function does not read or alter the run manifest. Native
    part PNGs are exact crops of EXIF-corrected RGB images, with identical boxes
    across all presets. Return paths are relative to ``output``.
    """
    output = Path(output).resolve()
    if not rows or not presets:
        raise ValueError("A report needs at least one vehicle and one preset")
    vehicle_ids = [row["vehicle_id"] for row in rows]
    if len(set(vehicle_ids)) != len(vehicle_ids):
        raise ValueError("Vehicle IDs must be unique")
    for name in vehicle_ids + list(presets):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "original":
            raise ValueError(f"Invalid or reserved report identifier: {name}")
    for row in rows:
        if not row["scene"].get("regions"):
            raise ValueError(f"No part regions for {row['vehicle_id']}")
        if set(row["outputs"]) != set(presets):
            raise ValueError(f"Output presets do not match for {row['vehicle_id']}")
        for region, coords in row["scene"]["regions"].items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", region):
                raise ValueError(f"Invalid crop region name: {region}")
            _box(coords, (1, 1))

    groups = report_groups(presets)
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    variants = ["original"] + list(presets)
    labels = {"original": "Original", **{name: p["label"] for name, p in presets.items()}}
    frame_width, frame_height, vehicle_header = 400, 300, 30
    row_height = frame_height + 32 + vehicle_header
    overview = Image.new("RGB", (frame_width * len(variants), row_height * len(rows)), BACKGROUND)
    group_width, group_height = 520, 390
    group_row_height = group_height + 32 + vehicle_header
    group_overviews = {
        name: Image.new("RGB", (group_width * (1 + len(members)), group_row_height * len(rows)), BACKGROUND)
        for name, (_, members) in groups.items()
    }
    artifacts, crop_records = [], []
    crop_count = 0
    for row_index, row in enumerate(rows):
        vehicle_id, label = row["vehicle_id"], row["label"]
        frame = Image.new("RGB", (overview.width, row_height), BACKGROUND)
        _label(ImageDraw.Draw(frame), (8, 5), label, frame.width - 16, 20)
        group_frames = {}
        for group_id, (title, members) in groups.items():
            group_frame = Image.new("RGB", (group_width * (1 + len(members)), group_row_height), BACKGROUND)
            _label(ImageDraw.Draw(group_frame), (8, 5), f"{label} / {title}", group_frame.width - 16, 20)
            group_frames[group_id] = group_frame

        regions = row["scene"]["regions"]
        detail_width, detail_height, region_header = 300, 220, 28
        region_height = detail_height + 32 + region_header
        sheet_height = vehicle_header + region_height * len(regions)
        details = Image.new("RGB", (detail_width * len(variants), sheet_height), BACKGROUND)
        details_224 = Image.new("RGB", details.size, BACKGROUND)
        _label(ImageDraw.Draw(details), (8, 5), label + " / Native-resolution part crops (fit to display)", details.width - 16, 20)
        _label(ImageDraw.Draw(details_224), (8, 5), label + " / Full-frame short edge 224, parts enlarged for diagnosis", details.width - 16, 20)
        for region_index, region in enumerate(regions):
            y = vehicle_header + region_index * region_height
            for sheet in (details, details_224):
                _label(ImageDraw.Draw(sheet), (8, y + 3), region.replace("_", " "), sheet.width - 16)

        original_size = None
        for column, variant in enumerate(variants):
            path = Path(row["source"] if variant == "original" else row["outputs"][variant])
            image = _read_rgb(path)
            if original_size is None:
                original_size = image.size
            elif image.size != original_size:
                raise ValueError(f"Source/output size mismatch for {path}: {image.size} != {original_size}")
            frame.paste(_tile(image, frame_width, frame_height, labels[variant]),
                        (column * frame_width, vehicle_header))
            for group_id, (_, ids) in groups.items():
                group_variants = ["original", *ids]
                if variant in group_variants:
                    group_frames[group_id].paste(_tile(image, group_width, group_height, labels[variant]),
                                                  (group_variants.index(variant) * group_width, vehicle_header))
            scale = 224 / min(image.size)
            reduced_size = (round(image.width * scale), round(image.height * scale))
            reduced = image.resize(reduced_size, Image.Resampling.LANCZOS)
            for region_index, (region, normalized_box) in enumerate(regions.items()):
                box = _box(normalized_box, image.size)
                reduced_box = _box(normalized_box, reduced.size)
                crop = image.crop(box)
                crop_dir = output / "crops" / vehicle_id / region
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{variant}.png"
                crop.save(crop_path, format="PNG", compress_level=3)
                artifacts.append(crop_path.relative_to(output).as_posix())
                crop_count += 1
                y = vehicle_header + region_index * region_height + region_header
                details.paste(_tile(crop, detail_width, detail_height, labels[variant]),
                              (column * detail_width, y))
                details_224.paste(_tile(reduced.crop(reduced_box), detail_width, detail_height,
                                        labels[variant], Image.Resampling.NEAREST),
                                  (column * detail_width, y))
                if variant == "original":
                    crop_records.append({
                        "vehicle_id": vehicle_id, "region": region,
                        "normalized_box_xyxy": list(normalized_box),
                        "native_image_size": list(image.size), "native_box_xyxy": list(box),
                        "native_crop_size": list(crop.size),
                        "short_edge_224_size": list(reduced.size),
                        "short_edge_224_box_xyxy": list(reduced_box),
                    })
            image.close()
            reduced.close()
        _save_png(previews / f"{vehicle_id}_comparison.png", frame, artifacts, output)
        _save_png(previews / f"{vehicle_id}_details.png", details, artifacts, output)
        _save_png(previews / f"{vehicle_id}_details_224.png", details_224, artifacts, output)
        overview.paste(frame, (0, row_index * row_height))
        for group_id, frame_group in group_frames.items():
            group_overviews[group_id].paste(frame_group, (0, row_index * group_row_height))

    _save_png(previews / "overview.png", overview, artifacts, output)
    for name, sheet in group_overviews.items():
        _save_png(previews / f"{name}_overview.png", sheet, artifacts, output)
    crop_metadata = {
        "coordinate_frame": "EXIF-corrected full RGB image; x1/y1 exclusive",
        "rounding": "floor left/top, ceil right/bottom",
        "native_crop_processing": "exact RGB pixel crop, no resize or enhancement",
        "preview_encoding": "lossless PNG; displayed cells resized to fit, native crops remain unscaled",
        "short_edge_224_processing": "full image resized preserving aspect ratio using LANCZOS, then normalized-region crop; nearest-neighbor enlarged in preview only",
        "short_edge_224_scope": "resolution diagnostic only; not a classifier input specification or accuracy evaluation",
        "regions": crop_records,
    }
    crop_metadata_path = output / "crop_regions.json"
    crop_metadata_path.write_text(json.dumps(crop_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts.append(crop_metadata_path.relative_to(output).as_posix())
    html_path = _write_html(output, rows, presets, groups)
    artifacts.append(html_path.relative_to(output).as_posix())
    return {
        "artifacts": artifacts,
        "native_crop_count": crop_count,
        "preview_png_count": sum(name.startswith("previews/") and name.endswith(".png") for name in artifacts),
        "report": html_path.relative_to(output).as_posix(),
        "visual_review": "not_performed_by_report_builder",
        "short_edge_diagnostic": 224,
    }
