"""Lossless composition reports with transformed, clipped part coordinates.

Main outputs remain untouched. Native part crops retain their actual rotation
and visible fragment; resizing happens only in preview sheets. No missing part
is reconstructed, and no inverse alignment cancels the augmentation.
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

from .transforms import describe_polygon, transform_points


BACKGROUND = "#eef1f5"
WARNING = "#b42318"
COLORS = ("#00e5ff", "#ffe500", "#ff65e5", "#7dff00", "#ff9c38")
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def report_groups(presets: dict) -> dict:
    groups = {}
    for name, preset in presets.items():
        group_id = preset.get("group")
        if group_id is None:
            if "group_label" in preset:
                raise ValueError(f"group_label requires group: {name}")
            continue
        if not isinstance(group_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", group_id):
            raise ValueError(f"Invalid group ID: {group_id!r}")
        title = preset.get("group_label", group_id.replace("_", " ").title())
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Invalid group label: {group_id}")
        if group_id not in groups:
            groups[group_id] = (title, [])
        elif groups[group_id][0] != title:
            raise ValueError(f"Conflicting group labels: {group_id}")
        groups[group_id][1].append(name)
    return groups


def _font(size):
    for path in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(draw, xy, text, width, size=17, color="#1d2939"):
    font = _font(size)
    while size > 10 and draw.textbbox((0, 0), text, font=font)[2] > width:
        size -= 1
        font = _font(size)
    draw.text(xy, text, fill=color, font=font)


def _tile(image, width, height, label, note=None, warning=False,
          resampling=Image.Resampling.LANCZOS):
    header = 52 if note is not None else 32
    tile = Image.new("RGB", (width, height + header), BACKGROUND)
    draw = ImageDraw.Draw(tile)
    _label(draw, (8, 6), label, width - 16)
    if note is not None:
        _label(draw, (8, 29), note, width - 16, 14,
               WARNING if warning else "#475467")
    if image is None:
        _label(draw, (12, header + height // 2), "OUT OF FRAME", width - 24, 21, WARNING)
    else:
        thumb = ImageOps.contain(image, (width - 8, height - 8), resampling)
        tile.paste(thumb, ((width - thumb.width) // 2,
                          header + (height - thumb.height) // 2))
    return tile


def _read_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _source_polygon(box, size):
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"Invalid normalized part box: {box}")
    w, h = size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _original_geometry(row, size):
    w, h = size
    subject = row["scene"].get("subject_polygon")
    return {
        "transform": {"forward_matrix": IDENTITY, "inverse_matrix": IDENTITY,
                      "source_view_polygon_xy": [[0, 0], [w, 0], [w, h], [0, h]],
                      "scale": 1.0, "angle_degrees": 0.0,
                      "retention_policy": "complete"},
        "subject": describe_polygon([[x * w, y * h] for x, y in subject], IDENTITY, size) if subject else None,
        "regions": {name: describe_polygon(_source_polygon(box, size), IDENTITY, size)
                    for name, box in row["scene"]["regions"].items()},
    }


def _reduced_box(visible_polygon, native_size, reduced_size):
    """Scale the clipped polygon with the actual rounded image dimensions."""
    if not visible_polygon:
        return None
    sx, sy = reduced_size[0] / native_size[0], reduced_size[1] / native_size[1]
    xs = [point[0] * sx for point in visible_polygon]
    ys = [point[1] * sy for point in visible_polygon]
    box = [max(0, math.floor(min(xs))), max(0, math.floor(min(ys))),
           min(reduced_size[0], math.ceil(max(xs))), min(reduced_size[1], math.ceil(max(ys)))]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _overlay(image, geometry, framing=False):
    """Draw only on a display copy; no annotations enter native output/crops."""
    display = ImageOps.contain(image, (800, 600), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(display)
    sx, sy = display.width / image.width, display.height / image.height

    def line(points, color, title=None):
        if not points:
            return
        xy = [(float(x) * sx, float(y) * sy) for x, y in points]
        draw.line(xy + [xy[0]], fill=color, width=3)
        if title:
            x = max(2, min(display.width - 110, min(v[0] for v in xy)))
            y = max(1, min(display.height - 18, min(v[1] for v in xy) - 17))
            draw.text((x, y), title, font=_font(14), fill=color, stroke_width=1, stroke_fill="#101828")

    if framing:
        line(geometry["transform"]["source_view_polygon_xy"], "#ff9c38", "Output viewport")
    else:
        if geometry.get("subject"):
            line(geometry["subject"]["visible_polygon_xy"], "#ffffff", "vehicle")
        for index, (name, region) in enumerate(geometry["regions"].items()):
            line(region["visible_polygon_xy"], COLORS[index % len(COLORS)], name)
    return display


def _save_png(path, image, artifacts, root):
    image.save(path, format="PNG", compress_level=6)
    artifacts.append(path.relative_to(root).as_posix())


def _href(path, report):
    return quote(os.path.relpath(path, report).replace(os.sep, "/"), safe="/")


def _link(path, report, label):
    return f'<a href="{_href(path, report)}">{html.escape(label)}</a>'


def _figure(path, report, alt):
    href = _href(path, report)
    return f'<a href="{href}"><img loading="lazy" src="{href}" alt="{html.escape(alt, quote=True)}"></a>'


def _subject_row(name, record, original, size, presets):
    transform = record["transform"]
    subject = record.get("subject")
    label = "Original" if name == "original" else presets[name]["label"]
    policy = transform.get("retention_policy", "partial_allowed")
    policy_label = "完整主体目标" if policy == "complete" else "允许截断"
    fields = [html.escape(label), f'{transform["scale"]:.3f}',
              f'{transform["angle_degrees"]:+.1f}°', policy_label]
    if subject:
        box = subject["transformed_bbox_xyxy"]
        width = (box[2] - box[0]) / size[0]
        source_box = original["subject"]["transformed_bbox_xyxy"]
        source_center_px = [(source_box[0] + source_box[2]) / 2,
                            (source_box[1] + source_box[3]) / 2]
        mapped_center = transform_points([source_center_px], transform["forward_matrix"])[0]
        center = [mapped_center[0] / size[0], mapped_center[1] / size[1]]
        source_center = [source_center_px[0] / size[0], source_center_px[1] / size[1]]
        fields += [f'{center[0]:.1%}, {center[1]:.1%}',
                   f'{(center[0] - source_center[0]) * 100:+.1f}%, {(center[1] - source_center[1]) * 100:+.1f}%',
                   f'{width:.1%}', f'{subject["retained_fraction"]:.2%}']
    else:
        fields += ["未配置车辆轮廓"] * 4
    return "<tr>" + "".join(f"<td>{v}</td>" for v in fields) + "</tr>"


def _write_html(output, rows, presets, groups, original_records):
    report = output / "report"
    report.mkdir(exist_ok=True)
    title = f"构图与尺度增强：{len(rows)} 张原图 × {len(presets)} 种预设"
    pieces = [
        "<h2>整图总览</h2><p>各预设独立从原图生成。只进行二维裁剪、等比缩放与画面旋转，不生成新背景或真实新视角。"
        "主图不绘制任何诊断线。预览为无损 PNG，但显示时会缩放。</p>",
        _figure(output / "previews/overview.png", report, "Full composition comparisons"),
    ]
    for group_id, (title_group, _) in groups.items():
        pieces.extend([f"<h2>{html.escape(title_group)}</h2>",
                       _figure(output / f"previews/{group_id}_overview.png", report, title_group)])
    for row in rows:
        vid, label = row["vehicle_id"], row["label"]
        original = original_records[vid]
        size = row["size"] if "size" in row else _read_rgb(Path(row["source"])).size
        variants = {"original": original, **row["geometry"]}
        metrics_rows = [_subject_row(name, record, original, size, presets) for name, record in variants.items()]
        pieces.extend([
            f'<section id="{vid}"><h2>{html.escape(label)}</h2>',
            _figure(output / f"previews/{vid}_comparison.png", report, label + " composition comparisons"),
            "<p>" + _link(Path(row["source"]), report, "原图") + " · "
            + " · ".join(_link(Path(row["outputs"][name]), report, presets[name]["label"] + " PNG") for name in presets) + "</p>",
            "<h3>实际几何变化</h3><p>车辆中心为原车辆包围框中心经正向矩阵映射的位置；中心及位移按画宽／画高归一化。"
            "宽度为裁切前变换后车辆包围框宽度，可能超过 100%；"
            "保留率为车辆多边形与画面相交面积／变换后面积，不是车型判别信息或真实遮挡比例。"
            "“完整主体目标”与“允许截断”分别标注，实际保留率以下表为准。几何差异不使用 MAE 作为主要判据。</p>",
            '<div class="scroll"><table><tr><th>预设</th><th>倍率</th><th>角度</th><th>模式</th>'
            '<th>中心 x,y</th><th>中心位移</th><th>车宽占比</th><th>主体保留率</th></tr>' + "".join(metrics_rows) + "</table></div>",
            "<details><summary>查看原图取景框与输出区域坐标</summary><p>橙线是输出画布逆映射到原图的四边形；"
            "旋转时它不再是轴对齐矩形。输出诊断图中的部件轮廓按同一变换矩阵移动，只显示画面内可见部分。</p>",
            _figure(output / f"previews/{vid}_framing.png", report, label + " source viewport polygons"),
            _figure(output / f"previews/{vid}_annotations.png", report, label + " transformed subject and part polygons"),
            "</details><h3>对应部件：变换后的原生像素裁剪</h3><p>部件框随图像一起变换，再裁可见多边形的轴对齐包围框。"
            "旋转不会被反向校正，框内可能含周围背景。预览按固定格子缩放，实际原生 PNG 不缩放。"
            "红色标记表示部件被裁切，完全出界显示 OUT OF FRAME，不生成虚构部件 PNG。保留率衡量标注区域面积。</p>",
            _figure(output / f"previews/{vid}_details.png", report, label + " transformed part crops"),
        ])
        header = "<th>部件</th>" + "".join(f'<th>{html.escape("Original" if name == "original" else presets[name]["label"])}</th>' for name in variants)
        crop_rows = []
        for region in row["scene"]["regions"]:
            cells = []
            for name, record in variants.items():
                geometry = record["regions"][region]
                if geometry["visible"]:
                    path = output / "crops" / vid / region / f"{name}.png"
                    label_crop = f'PNG · {geometry["retained_fraction"]:.1%}'
                    cell = _link(path, report, label_crop)
                    if geometry["retained_fraction"] < 1 - 1e-8:
                        cell += ' <span class="warn">裁切</span>'
                else:
                    cell = '<span class="warn">OUT OF FRAME</span>'
                cells.append("<td>" + cell + "</td>")
            crop_rows.append(f"<tr><th>{html.escape(region)}</th>{''.join(cells)}</tr>")
        pieces.extend([
            '<div class="scroll"><table><tr>' + header + "</tr>" + "".join(crop_rows) + "</table></div>",
            "<details><summary>全图短边 224 后的对应部件</summary><p>先缩小完整原图／增强图，再按变换后的可见区域位置裁剪，"
            "以最近邻放大显示像素。没有使用旧部件坐标，也没有先裁部件再缩小。224 只是统一诊断尺寸，不是分类准确率评估。</p>",
            _figure(output / f"previews/{vid}_details_224.png", report, label + " transformed part crops after full-frame reduction"),
            "</details></section>",
        ])
    nav = " · ".join([_link(output / "manifest.json", report, "运行清单与矩阵"),
                      _link(output / "crop_regions.json", report, "各部件坐标与保留率")]
                     + [f'<a href="#{row["vehicle_id"]}">{html.escape(row["label"])}</a>' for row in rows])
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += f"<title>{html.escape(title)}</title>" + """<style>
body{max-width:1450px;margin:auto;padding:24px;font:16px/1.7 system-ui,sans-serif;color:#1d2939;background:#fafbfc}
h1,h2,h3{line-height:1.35}h2{margin-top:36px}img{display:block;max-width:100%;height:auto;margin:16px 0;border:1px solid #d0d5dd}
a{color:#175cd3}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d0d5dd;padding:9px;text-align:left}
.scroll{overflow-x:auto}summary{cursor:pointer;font-weight:bold}details{margin:22px 0}.warn{color:#b42318;font-weight:bold}
.note{padding:16px;background:#edf4ff;border-left:4px solid #528bff}nav{display:flex;flex-wrap:wrap;gap:14px}
</style></head><body>"""
    page += f'<h1>{html.escape(title)}</h1><p class="note">构图变化使用已有原图像素，允许截断的预设会损失部分车辆或部件。输出画布尺寸保持不变，但物体位置和部件坐标已改变。文件校验通过不等于训练收益或细粒度识别保真通过。</p><nav>{nav}</nav>'
    page += "\n".join(pieces) + "</body></html>\n"
    path = report / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_report(output: Path, rows: list, presets: dict) -> dict:
    """Use runner-provided transformed region geometry without recomputing it."""
    output = Path(output).resolve()
    if not rows or not presets:
        raise ValueError("At least one source and one preset are required")
    groups = report_groups(presets)
    ids = [row["vehicle_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Vehicle IDs must be unique")
    for name in ids + list(presets):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "original":
            raise ValueError(f"Invalid report identifier: {name}")
    for row in rows:
        if not row["scene"].get("regions"):
            raise ValueError("No comparison regions")
        if set(row["outputs"]) != set(presets) or set(row["geometry"]) != set(presets):
            raise ValueError("Preset, image, and geometry IDs must match")
        for name, box in row["scene"]["regions"].items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid region ID: {name}")
            _source_polygon(box, (1, 1))
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    variants = ["original", *presets]
    labels = {"original": "Original", **{name: value["label"] for name, value in presets.items()}}
    fw, fh, vehicle_header = 400, 300, 30
    frame_height = vehicle_header + fh + 32
    overview = Image.new("RGB", (fw * len(variants), frame_height * len(rows)), BACKGROUND)
    gw, gh = 520, 390
    group_height = vehicle_header + gh + 32
    group_sheets = {name: Image.new("RGB", (gw * (1 + len(members)), group_height * len(rows)), BACKGROUND)
                    for name, (_, members) in groups.items()}
    artifacts, crop_records, originals = [], [], {}
    crop_count, invisible_count = 0, 0
    for row_index, row in enumerate(rows):
        vid, label, regions = row["vehicle_id"], row["label"], row["scene"]["regions"]
        source = _read_rgb(Path(row["source"]))
        original = _original_geometry(row, source.size)
        originals[vid] = original
        records = {"original": original, **row["geometry"]}
        frame = Image.new("RGB", (overview.width, frame_height), BACKGROUND)
        framing = Image.new("RGB", frame.size, BACKGROUND)
        annotations = Image.new("RGB", frame.size, BACKGROUND)
        for sheet, suffix in [(frame, ""), (framing, " / Source viewport"), (annotations, " / Transformed regions")]:
            _label(ImageDraw.Draw(sheet), (8, 5), label + suffix, sheet.width - 16, 20)
        group_frames = {name: Image.new("RGB", (gw * (1 + len(members)), group_height), BACKGROUND)
                        for name, (_, members) in groups.items()}
        for name, (group_label, _) in groups.items():
            _label(ImageDraw.Draw(group_frames[name]), (8, 5), label + " / " + group_label, group_frames[name].width - 16, 20)
        dw, dh, rh = 300, 220, 28
        part_row_height = dh + 52 + rh
        details = Image.new("RGB", (dw * len(variants), vehicle_header + len(regions) * part_row_height), BACKGROUND)
        details_224 = Image.new("RGB", details.size, BACKGROUND)
        _label(ImageDraw.Draw(details), (8, 5), label + " / Transformed native crops; visible region only", details.width - 16, 20)
        _label(ImageDraw.Draw(details_224), (8, 5), label + " / Full frame short edge 224; transformed part positions", details.width - 16, 20)
        for index, name in enumerate(regions):
            for sheet in (details, details_224):
                _label(ImageDraw.Draw(sheet), (8, vehicle_header + index * part_row_height + 3), name.replace("_", " "), sheet.width - 16)
        for column, variant in enumerate(variants):
            image = source if variant == "original" else _read_rgb(Path(row["outputs"][variant]))
            if image.size != source.size:
                raise ValueError(f"Source/output size mismatch: {vid}/{variant}")
            record = records[variant]
            frame.paste(_tile(image, fw, fh, labels[variant]), (column * fw, vehicle_header))
            framing.paste(_tile(_overlay(source, record, framing=True), fw, fh, labels[variant]), (column * fw, vehicle_header))
            annotations.paste(_tile(_overlay(image, record), fw, fh, labels[variant]), (column * fw, vehicle_header))
            for group_id, (_, members) in groups.items():
                subset = ["original", *members]
                if variant in subset:
                    group_frames[group_id].paste(_tile(image, gw, gh, labels[variant]), (subset.index(variant) * gw, vehicle_header))
            scale = 224 / min(image.size)
            reduced = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
            for index, (name, normalized) in enumerate(regions.items()):
                geometry = record["regions"][name]
                box = geometry["crop_box_xyxy"]
                fraction = float(geometry["retained_fraction"])
                if not 0 <= fraction <= 1:
                    raise ValueError("Invalid retained fraction")
                visible = bool(geometry["visible"])
                if visible and (box is None or not 0 <= box[0] < box[2] <= image.width or not 0 <= box[1] < box[3] <= image.height):
                    raise ValueError(f"Invalid visible crop: {vid}/{variant}/{name}")
                reduced_box = _reduced_box(geometry["visible_polygon_xy"], image.size, reduced.size) if visible else None
                crop, small_crop, crop_path = None, None, None
                if visible:
                    crop = image.crop(box)
                    small_crop = reduced.crop(reduced_box)
                    directory = output / "crops" / vid / name
                    directory.mkdir(parents=True, exist_ok=True)
                    crop_path = directory / f"{variant}.png"
                    crop.save(crop_path, format="PNG", compress_level=3)
                    artifacts.append(crop_path.relative_to(output).as_posix())
                    crop_count += 1
                else:
                    invisible_count += 1
                note = f"retained {fraction:.1%}" + (" / CLIPPED" if visible and fraction < 1 - 1e-8 else "")
                warning = fraction < 1 - 1e-8
                xy = (column * dw, vehicle_header + index * part_row_height + rh)
                details.paste(_tile(crop, dw, dh, labels[variant], note, warning), xy)
                details_224.paste(_tile(small_crop, dw, dh, labels[variant], note, warning, Image.Resampling.NEAREST), xy)
                crop_records.append({
                    "vehicle_id": vid, "preset_id": variant, "region": name,
                    "source_box_normalized": list(normalized), "native_image_size": list(image.size),
                    "transformed_polygon_xy": geometry["transformed_polygon_xy"],
                    "visible_polygon_xy": geometry["visible_polygon_xy"],
                    "native_box_xyxy": box if visible else None,
                    "native_crop_size": list(crop.size) if visible else None,
                    "crop_path": crop_path.relative_to(output).as_posix() if crop_path else None,
                    "retained_fraction": fraction, "visible": visible,
                    "short_edge_224_size": list(reduced.size), "short_edge_224_box_xyxy": reduced_box,
                })
            if image is not source:
                image.close()
            reduced.close()
        source.close()
        for suffix, sheet in [("comparison", frame), ("details", details), ("details_224", details_224), ("framing", framing), ("annotations", annotations)]:
            _save_png(previews / f"{vid}_{suffix}.png", sheet, artifacts, output)
        overview.paste(frame, (0, row_index * frame_height))
        for group_id, sheet in group_frames.items():
            group_sheets[group_id].paste(sheet, (0, row_index * group_height))
    _save_png(previews / "overview.png", overview, artifacts, output)
    for name, sheet in group_sheets.items():
        _save_png(previews / f"{name}_overview.png", sheet, artifacts, output)
    metadata = {
        "coordinate_frame": "EXIF-corrected image pixel boundaries, x1/y1 exclusive; each variant has transformed coordinates",
        "native_crop_processing": "exact crop of visible transformed polygon bounding box; no resize, no rotation alignment, no fabricated missing pixels",
        "preview_encoding": "lossless PNG; cells resized for display, native crops unscaled",
        "short_edge_224_processing": "resize full image LANCZOS; scale clipped polygon by actual rounded image width/height ratios; floor/ceil crop; nearest enlargement for display",
        "retained_fraction_definition": "visible transformed annotation polygon area / total transformed annotation area",
        "regions": crop_records,
    }
    path = output / "crop_regions.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts.append(path.relative_to(output).as_posix())
    page = _write_html(output, rows, presets, groups, originals)
    artifacts.append(page.relative_to(output).as_posix())
    return {"artifacts": artifacts, "native_crop_count": crop_count,
            "invisible_part_count": invisible_count,
            "preview_png_count": sum(name.startswith("previews/") and name.endswith(".png") for name in artifacts),
            "report": page.relative_to(output).as_posix(),
            "visual_review": "not_performed_by_report_builder", "short_edge_diagnostic": 224}
