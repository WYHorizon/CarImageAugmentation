"""Reusable RGBA foreground compositing and image-space visibility diagnostics.

Only the foreground is transformed. Coordinates and part boxes on the target
image therefore remain fixed. All resizing, rotation and blur use premultiplied
alpha to keep invisible RGB pixels from producing halos at material edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class OcclusionResult:
    image: Image.Image
    alpha: Image.Image
    transformed_asset_size: tuple[int, int]
    metadata: dict


@dataclass
class VisibilityResult:
    statistics: dict
    vehicle_mask: Image.Image | None
    vehicle_visible_mask: Image.Image | None


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _size_guard(size: tuple[int, int]) -> None:
    if min(size) < 1 or max(size) > 32768 or size[0] * size[1] > 50_000_000:
        raise ValueError("Transformed foreground exceeds the 50 MP / 32768 pixel safety limit")


def compose_occlusion(
    base: Image.Image,
    asset: Image.Image,
    *,
    width_fraction: float,
    position_xy: tuple[float, float],
    angle_degrees: float = 0,
    flip_horizontal: bool = False,
    blur_radius: float = 0,
    brightness: float = 1,
    saturation: float = 1,
) -> OcclusionResult:
    """Place an RGBA asset over an RGB target without altering exposed pixels.

    ``width_fraction`` sets the asset width before rotation, relative to target
    width. Positive angles rotate counterclockwise. ``position_xy`` gives the
    normalized top-left of the expanded rotation canvas, rounded to the nearest
    output pixel. Transparent padding for blur is added without moving that
    placement anchor. Blur radius is measured in output pixels. An opaque RGBA
    rectangle is valid; an RGB file must first acquire an explicit alpha mask.
    """
    if not isinstance(base, Image.Image) or base.mode != "RGB":
        raise ValueError("base must be a PIL RGB image")
    if not isinstance(asset, Image.Image) or asset.mode != "RGBA":
        raise ValueError("asset must be PIL RGBA with an explicit foreground alpha mask")
    _size_guard(base.size)
    _size_guard(asset.size)
    if asset.getchannel("A").getextrema()[1] == 0:
        raise ValueError("The foreground alpha mask is empty")
    width_fraction = _number(width_fraction, "width_fraction")
    angle_degrees = _number(angle_degrees, "angle_degrees")
    blur_radius = _number(blur_radius, "blur_radius")
    brightness = _number(brightness, "brightness")
    saturation = _number(saturation, "saturation")
    if not 0 < width_fraction <= 4:
        raise ValueError("width_fraction must be in (0, 4]")
    if not -180 <= angle_degrees <= 180:
        raise ValueError("angle_degrees must be in [-180, 180]")
    if not 0 <= blur_radius <= 128:
        raise ValueError("blur_radius must be in [0, 128] output pixels")
    if not 0 < brightness <= 4 or not 0 <= saturation <= 4:
        raise ValueError("brightness must be in (0, 4] and saturation in [0, 4]")
    if not isinstance(flip_horizontal, bool):
        raise ValueError("flip_horizontal must be a boolean")
    if not isinstance(position_xy, (tuple, list)) or len(position_xy) != 2:
        raise ValueError("position_xy must be a normalized (x, y) pair")
    position = tuple(_number(v, "position_xy") for v in position_xy)
    if any(abs(v) > 8 for v in position):
        raise ValueError("position_xy must be within [-8, 8]")

    target_width = max(1, round(base.width * width_fraction))
    target_height = max(1, round(asset.height * target_width / asset.width))
    target_size = target_width, target_height
    _size_guard(target_size)
    radians = math.radians(angle_degrees)
    expanded_bound = (
        math.ceil(abs(math.cos(radians)) * target_width + abs(math.sin(radians)) * target_height) + 2,
        math.ceil(abs(math.sin(radians)) * target_width + abs(math.cos(radians)) * target_height) + 2,
    )
    pad = math.ceil(4 * blur_radius) if blur_radius else 0
    _size_guard((expanded_bound[0] + 2 * pad, expanded_bound[1] + 2 * pad))

    foreground_rgb = asset.convert("RGB")
    if saturation != 1:
        foreground_rgb = ImageEnhance.Color(foreground_rgb).enhance(saturation)
    if brightness != 1:
        foreground_rgb = ImageEnhance.Brightness(foreground_rgb).enhance(brightness)
    foreground = foreground_rgb.convert("RGBA")
    foreground.putalpha(asset.getchannel("A"))
    if flip_horizontal:
        foreground = ImageOps.mirror(foreground)
    # RGBa stores RGB already multiplied by alpha. Filtering all four channels
    # in this mode prevents arbitrary RGB under transparent pixels from leaking.
    foreground = foreground.convert("RGBa").resize(target_size, Image.Resampling.LANCZOS)
    if angle_degrees:
        foreground = foreground.rotate(
            angle_degrees, resample=Image.Resampling.BICUBIC, expand=True,
            fillcolor=(0, 0, 0, 0),
        )
    rotation_size = foreground.size
    if pad:
        foreground = ImageOps.expand(foreground, border=pad, fill=(0, 0, 0, 0))
        foreground = foreground.filter(ImageFilter.GaussianBlur(blur_radius))

    anchor_x, anchor_y = (round(position[0] * base.width), round(position[1] * base.height))
    left, top = anchor_x - pad, anchor_y - pad
    dest_box = (max(0, left), max(0, top), min(base.width, left + foreground.width),
                min(base.height, top + foreground.height))
    if dest_box[0] >= dest_box[2] or dest_box[1] >= dest_box[3]:
        raise ValueError("The foreground falls completely outside the output")
    source_box = (dest_box[0] - left, dest_box[1] - top,
                  dest_box[2] - left, dest_box[3] - top)
    premultiplied = np.asarray(foreground.crop(source_box), dtype=np.uint16)
    alpha = premultiplied[..., 3]
    if not alpha.any():
        raise ValueError("No nontransparent foreground pixels intersect the output")
    # Lanczos can overshoot near edges; valid premultiplied color never exceeds
    # its alpha. Clamp before compositing instead of unpremultiplying tiny alpha.
    rgb_premultiplied = np.minimum(premultiplied[..., :3], alpha[..., None])
    original_patch = np.asarray(base.crop(dest_box), dtype=np.uint16)
    composite = rgb_premultiplied + (original_patch * (255 - alpha[..., None]) + 127) // 255
    result = base.copy()
    result.paste(Image.fromarray(np.minimum(composite, 255).astype(np.uint8)), dest_box[:2])
    alpha_image = Image.new("L", base.size, 0)
    alpha_image.paste(Image.fromarray(alpha.astype(np.uint8)), dest_box[:2])
    nonzero_box = alpha_image.getbbox()
    return OcclusionResult(
        image=result,
        alpha=alpha_image,
        transformed_asset_size=foreground.size,
        metadata={
            "input_size": list(base.size), "asset_size": list(asset.size),
            "width_fraction": width_fraction, "position_xy": list(position),
            "angle_degrees": angle_degrees, "positive_angle": "counterclockwise",
            "flip_horizontal": flip_horizontal, "blur_radius_output_pixels": blur_radius,
            "brightness": brightness, "saturation": saturation,
            "resized_asset_size": list(target_size), "rotation_canvas_size": list(rotation_size),
            "transformed_asset_size": list(foreground.size), "blur_padding_pixels": pad,
            "placement_anchor_xy": [anchor_x, anchor_y], "padded_top_left_xy": [left, top],
            "nonzero_alpha_bbox_xyxy": list(nonzero_box),
            "compositing": "8-bit premultiplied-alpha foreground over unmodified RGB source",
            "resampling": "premultiplied LANCZOS resize; BICUBIC rotation; Gaussian blur",
            "outside_alpha_unchanged": True,
        },
    )


def _vehicle_mask(size: tuple[int, int], polygon) -> np.ndarray:
    """Rasterize normalized polygon with a half-open pixel-center scanline rule."""
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("subject_polygon must be finite normalized N x 2 points") from None
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1)):
        raise ValueError("subject_polygon must be finite normalized N x 2 points, N >= 3")
    width, height = size
    points = points * [width, height]
    result = np.zeros((height, width), dtype=bool)
    edges = list(zip(points, np.roll(points, -1, axis=0)))
    for y in range(height):
        center_y = y + 0.5
        intersections = sorted(
            a[0] + (center_y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            for a, b in edges if (a[1] <= center_y < b[1]) or (b[1] <= center_y < a[1])
        )
        for left, right in zip(intersections[::2], intersections[1::2]):
            start = max(0, min(width, math.ceil(left - 0.5)))
            stop = max(0, min(width, math.ceil(right - 0.5)))
            result[y, start:stop] = True
    if not result.any():
        raise ValueError("subject_polygon covers no output pixel centers")
    return result


def compute_visibility(
    alpha: Image.Image, scene: dict, *, opaque_alpha_threshold: int = 128
) -> VisibilityResult:
    """Measure opaque and alpha-weighted overlap without changing annotation boxes.

    Region ratios describe annotated rectangles, which may contain background.
    They do not measure semantic part pixels or retained classification evidence.
    The visible vehicle mask is binary, using alpha < opaque_alpha_threshold.
    """
    if not isinstance(alpha, Image.Image) or alpha.mode != "L":
        raise ValueError("alpha must be a PIL L image")
    _size_guard(alpha.size)
    if (isinstance(opaque_alpha_threshold, bool) or not isinstance(opaque_alpha_threshold, int)
            or not 1 <= opaque_alpha_threshold <= 255):
        raise ValueError("opaque_alpha_threshold must be an integer in [1, 255]")
    if not isinstance(scene, dict) or not isinstance(scene.get("regions", {}), dict):
        raise ValueError("scene must be a dictionary with a regions mapping")
    values = np.asarray(alpha)
    opaque = values >= opaque_alpha_threshold

    def metrics(selected: np.ndarray) -> dict:
        pixel_count = int(selected.size)
        opaque_count = int(np.count_nonzero(selected >= opaque_alpha_threshold))
        opaque_fraction = opaque_count / pixel_count
        weighted_fraction = float(np.mean(selected, dtype=np.float64) / 255)
        return {"pixel_count": pixel_count, "opaque_pixel_count": opaque_count,
                "opaque_fraction": opaque_fraction, "weighted_fraction": weighted_fraction,
                "visible_fraction": 1 - opaque_fraction,
                "weighted_visible_fraction": 1 - weighted_fraction}

    vehicle = None
    visible = None
    vehicle_stats = None
    if scene.get("subject_polygon") is not None:
        vehicle = _vehicle_mask(alpha.size, scene["subject_polygon"])
        vehicle_stats = metrics(values[vehicle])
        visible = vehicle & ~opaque
    regions = {}
    for name, box in scene.get("regions", {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Region names must be nonempty strings")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError(f"Region {name} must be normalized [left, top, right, bottom]")
        x0, y0, x1, y1 = (_number(v, f"region {name}") for v in box)
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError(f"Region {name} must be a nonempty box within [0, 1]")
        crop = [math.floor(x0 * alpha.width), math.floor(y0 * alpha.height),
                math.ceil(x1 * alpha.width), math.ceil(y1 * alpha.height)]
        regions[name] = metrics(values[crop[1]:crop[3], crop[0]:crop[2]])
        regions[name]["crop_box_xyxy"] = crop
    return VisibilityResult(
        statistics={
            "opaque_alpha_threshold": opaque_alpha_threshold,
            "opaque_definition": "foreground alpha >= threshold",
            "weighted_definition": "mean foreground alpha / 255 within the annotated region",
            "vehicle": vehicle_stats, "regions": regions,
            "region_semantics": "rectangular annotations, not semantic part segmentation masks",
        },
        vehicle_mask=Image.fromarray(vehicle.astype(np.uint8) * 255) if vehicle is not None else None,
        vehicle_visible_mask=Image.fromarray(visible.astype(np.uint8) * 255) if visible is not None else None,
    )
