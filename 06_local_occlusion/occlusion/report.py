"""Same-coordinate image and part comparisons for local occlusion experiments."""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


BACKGROUND = "#eef2f6"
INK = "#18304d"
WARNING = "#a83225"


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(draw, position, text, width, size=20, color=INK):
    while size > 10 and draw.textbbox((0, 0), text, font=_font(size))[2] > width:
        size -= 1
    draw.text(position, text, font=_font(size), fill=color)


def _tile(image, width, height, title, note="", *, nearest=False, warning=False):
    header = 66
    panel = Image.new("RGB", (width, height + header), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    _label(draw, (12, 9), title, width - 24, 23)
    _label(draw, (12, 39), note, width - 24, 15, WARNING if warning else "#526276")
    thumb = ImageOps.contain(image, (width - 12, height - 12),
                            Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    panel.paste(thumb, ((width - thumb.width) // 2, header + (height - thumb.height) // 2))
    return panel


def _sheet(rows):
    width = max(sum(image.width for image in row) for row in rows)
    height = sum(max(image.height for image in row) for row in rows)
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    y = 0
    for row in rows:
        x = 0
        for image in row:
            sheet.paste(image, (x, y))
            x += image.width
        y += max(image.height for image in row)
    return sheet


def _read_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def region_box(box, size):
    """Normalized pixel-boundary box, rounded outward at the actual image size."""
    if (len(box) != 4 or not all(math.isfinite(value) for value in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"Invalid normalized region: {box}")
    w, h = size
    return (math.floor(box[0] * w), math.floor(box[1] * h),
            math.ceil(box[2] * w), math.ceil(box[3] * h))


def diagnostic_image(image, short_edge=224):
    scale = short_edge / min(image.size)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _percent(value):
    return f"{100 * value:.1f}%"


def _note(stats):
    return f"Occluded: {_percent(stats['opaque_fraction'])} | alpha-weighted: {_percent(stats['weighted_fraction'])}"


def _diagnostic_overlay(original, alpha, scene):
    display = ImageOps.contain(original, (900, 675), Image.Resampling.LANCZOS)
    reduced_alpha = alpha.resize(display.size, Image.Resampling.BILINEAR)
    tint = reduced_alpha.point(lambda a: round(a * 0.55))
    display = Image.composite(Image.new("RGB", display.size, "#ff492f"), display, tint)
    draw = ImageDraw.Draw(display)
    if scene.get("subject_polygon"):
        points = [(x * display.width, y * display.height) for x, y in scene["subject_polygon"]]
        draw.line(points + points[:1], fill="#00eaff", width=3)
    for name, box in scene["regions"].items():
        left, top, right, bottom = region_box(box, display.size)
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), outline="#fff24c", width=2)
        _label(draw, (left + 2, top + 2), name, max(80, right - left - 4), 12, "#fff24c")
    return display


def build_report(output_dir: Path, source: Path, augmented: Path, alpha: Path,
                 scene: dict, statistics: dict) -> dict:
    """Write previews, lossless native/reduced crops and a local HTML report.

    Source and augmented images keep exactly the same coordinates. Region boxes
    contain background as well as the named part; their rates are diagnostic.
    """
    output_dir, source, augmented, alpha = map(Path, (output_dir, source, augmented, alpha))
    original, edited = _read_rgb(source), _read_rgb(augmented)
    with Image.open(alpha) as opened:
        mask = opened.convert("L")
    if original.size != edited.size or mask.size != original.size:
        raise ValueError("Original, output and alpha must have identical dimensions")
    regions = scene.get("regions")
    if not isinstance(regions, dict) or not regions:
        raise ValueError("At least one comparison region is required")
    for name, box in regions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Unsafe comparison region name: {name!r}")
        expected_box = region_box(box, original.size)
        stats = statistics["regions"][name]
        if tuple(stats["crop_box_xyxy"]) != expected_box:
            raise ValueError(f"Statistics/crop coordinates disagree: {name}")
        for key in ("opaque_fraction", "weighted_fraction"):
            if not math.isfinite(stats[key]) or not 0 <= stats[key] <= 1:
                raise ValueError(f"Invalid coverage fraction: {name}/{key}")

    for folder in ("previews", "report", "crops/original", "crops/occluded",
                   "crops_224/original", "crops_224/occluded"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    vehicle = statistics.get("vehicle")
    vehicle_note = _note(vehicle) if vehicle else "Vehicle outline not provided; vehicle coverage unavailable"
    display_height = max(240, min(900, round(900 * original.height / original.width)))
    _sheet([[_tile(original, 900, display_height, "Original", "Unmodified source"),
             _tile(edited, 900, display_height, "Foreground occlusion", vehicle_note)]]).save(
                 output_dir / "previews/comparison.png")

    reduced = {"original": diagnostic_image(original), "occluded": diagnostic_image(edited)}
    images = {"original": original, "occluded": edited}
    crop_metadata, native_rows, reduced_rows = {}, [], []
    for name, box in regions.items():
        stats = statistics["regions"][name]
        native_box, small_box = region_box(box, original.size), region_box(box, reduced["original"].size)
        native_panels, small_panels = [], []
        entry = {"normalized_box": box, "crop_box_xyxy": list(native_box),
                 "diagnostic_box_xyxy": list(small_box), "statistics": stats, "files": {}}
        for variant, image in images.items():
            native = image.crop(native_box)
            small = reduced[variant].crop(small_box)
            native_path, small_path = f"crops/{variant}/{name}.png", f"crops_224/{variant}/{name}.png"
            native.save(output_dir / native_path)
            small.save(output_dir / small_path)
            entry["files"][variant] = {"native": native_path, "short_edge_224": small_path}
            title = f"{name.replace('_', ' ').title()} / {variant.title()}"
            note = _note(stats) if variant == "occluded" else f"Native crop {native.width} x {native.height}"
            native_panels.append(_tile(native, 600, 250, title, note,
                                       warning=variant == "occluded" and stats["opaque_fraction"] > 0))
            reduced_note = f"224-short-edge image: crop {small.width} x {small.height}"
            small_panels.append(_tile(small, 600, 220, title, reduced_note, nearest=True))
        crop_metadata[name] = entry
        native_rows.append(native_panels)
        reduced_rows.append(small_panels)
    _sheet(native_rows).save(output_dir / "previews/parts.png")
    _sheet(reduced_rows).save(output_dir / "previews/parts_224.png")
    _sheet([[_tile(mask.convert("RGB"), 900, display_height, "Foreground alpha",
                  "White: opaque | black: no foreground | gray: transition"),
             _tile(_diagnostic_overlay(original, mask, scene), 900, display_height,
                   "Occlusion diagnostic overlay", "Red: foreground | cyan: vehicle | yellow: part boxes")]]).save(
                       output_dir / "previews/masks.png")
    metadata = {"native_size": list(original.size), "diagnostic_size": list(reduced["original"].size),
                "diagnostic_short_edge": 224, "regions": crop_metadata}
    (output_dir / "crop_regions.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    report_dir = output_dir / "report"
    def link(path):
        return html.escape(quote(os.path.relpath(Path(path).resolve(), report_dir.resolve()), safe="/"), quote=True)

    def anchor(path, text):
        return f'<a href="{link(path)}">{html.escape(text)}</a>'

    def preview(name, description):
        path = output_dir / f"previews/{name}.png"
        return f'<p>{html.escape(description)}</p><a href="{link(path)}"><img src="{link(path)}" alt="{html.escape(description, quote=True)}"></a>'

    rows = []
    for name, entry in crop_metadata.items():
        stats = entry["statistics"]
        links = []
        for variant, files in entry["files"].items():
            links.extend(anchor(output_dir / files[key], f"{variant} / {label}")
                         for key, label in (("native", "native"), ("short_edge_224", "224 crop")))
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{_percent(stats["opaque_fraction"])}</td>'
                    f'<td>{_percent(stats["weighted_fraction"])}</td><td>{" · ".join(links)}</td></tr>')
    threshold = int(statistics.get("opaque_alpha_threshold", 128))
    title = html.escape(f"Local occlusion — {scene.get('label', source.stem)}")
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#18304d;background:#f8fafc}}
img{{max-width:100%;height:auto;border:1px solid #d4deea;border-radius:8px}}a{{color:#1755a1}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d4deea;padding:9px;text-align:left}}
</style><h1>{title}</h1>
<p>{anchor(source, 'Original PNG')} · {anchor(augmented, 'Full-resolution occluded PNG')} · {anchor(alpha, 'Foreground alpha mask')} · {anchor(output_dir / 'crop_regions.json', 'Crop coordinates')}</p>
<p>{html.escape(vehicle_note)}. Source and output use the same pixel coordinates; annotations are drawn only in diagnostic previews.</p>
{preview('comparison', 'Original and foreground occlusion, displayed at the same scale.')}
<h2>Corresponding parts</h2>
<p>Each native crop uses the same original-coordinate rectangle. Display tiles resize crops; use the native PNG links to inspect source pixels. Rectangles may include background and are not exact semantic part masks.</p>
{preview('parts', 'Native part comparisons; occluded percentages refer to the annotated rectangles.')}
<table><thead><tr><th>Region</th><th>Opaque occlusion</th><th>Alpha-weighted coverage</th><th>Lossless crops</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Short-edge 224 diagnostic</h2>
<p>The entire original/output image is first resized to {reduced['original'].width} × {reduced['original'].height}, then part rectangles are mapped to those actual dimensions and cropped. Display uses nearest-neighbor enlargement. This is not a classifier accuracy measurement.</p>
{preview('parts_224', 'Parts after resizing the complete image to short edge 224.')}
<h2>Masks and interpretation</h2>
{preview('masks', 'Alpha and diagnostic overlay. Overlay colors are not part of the augmented output.')}
<p>Opaque occlusion counts alpha ≥ {threshold} on the 0–255 scale inside the region. Alpha-weighted coverage is the mean alpha/255 inside the region and includes partial edge/blur coverage. Vehicle rates use the supplied vehicle outline; part rates use their rectangles. Coverage is not a measure of retained identity information or training benefit.</p>
</html>'''
    (report_dir / "index.html").write_text(document + "\n", encoding="utf-8")
    return {"html": "report/index.html", "previews": {name: f"previews/{name}.png"
            for name in ("comparison", "parts", "parts_224", "masks")},
            "crop_metadata": "crop_regions.json", "native_crop_count": len(regions) * 2,
            "diagnostic_crop_count": len(regions) * 2, "diagnostic_size": metadata["diagnostic_size"]}
