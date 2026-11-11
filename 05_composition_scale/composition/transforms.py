"""Single-resampling image-space composition transforms and annotation geometry.

Coordinates refer to pixel boundaries: the image rectangle is [0, width] by
[0, height], and pixel centers lie at (x + 0.5, y + 0.5). Matrices map source
coordinates to output coordinates. Pillow AFFINE receives the inverse matrix
and uses this same pixel-center convention. Positive roll is clockwise because
image y increases downward. These transforms change framing, not viewpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import re

import numpy as np
from PIL import Image


@dataclass
class CompositionResult:
    image: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _size(size) -> tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size must be (width, height)")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
           or not 1 <= v <= 65536 for v in size):
        raise ValueError("Image dimensions must be integers in [1, 65536]")
    return int(size[0]), int(size[1])


def _points(points, minimum: int = 0) -> np.ndarray:
    try:
        values = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Points must be finite N x 2 coordinates") from None
    if values.size == 0 and minimum == 0:
        return np.empty((0, 2), dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < minimum
            or not np.isfinite(values).all()):
        raise ValueError(f"Points must be finite N x 2 coordinates with N >= {minimum}")
    return values


def _matrix(matrix) -> np.ndarray:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Matrix must be a finite 3 x 3 affine transform") from None
    if (values.shape != (3, 3) or not np.isfinite(values).all()
            or not np.allclose(values[2], [0, 0, 1], atol=1e-12, rtol=0)
            or abs(np.linalg.det(values[:2, :2])) < 1e-12):
        raise ValueError("Matrix must be a finite invertible 3 x 3 affine transform")
    return values


def validate_preset(preset: dict) -> None:
    if not isinstance(preset, dict):
        raise ValueError("Preset must be a dictionary")
    if not isinstance(preset.get("label"), str) or not preset["label"].strip():
        raise ValueError("Preset must have a nonempty label")
    kind = preset.get("kind")
    if not isinstance(kind, str) or kind not in {"zoom", "position", "roll"}:
        raise ValueError("kind must be zoom, position or roll")
    if kind in {"zoom", "position"}:
        scale = _number(preset.get("scale"), "scale")
        if not 1 <= scale <= 8:
            raise ValueError("scale must be in [1, 8]; zoom-out requires a separate border policy")
    if kind == "position" and preset.get("direction") not in ("left", "right", "up", "down"):
        raise ValueError("direction must be left, right, up or down")
    if kind == "roll":
        if not -45 <= _number(preset.get("angle"), "angle") <= 45:
            raise ValueError("roll angle must be in [-45, 45] degrees")
    if preset.get("retention_policy", "partial_allowed") not in ("complete", "partial_allowed"):
        raise ValueError("retention_policy must be complete or partial_allowed")
    group = preset.get("group")
    if group is not None and (not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", group)):
        raise ValueError("group must be a safe identifier")
    if "group_label" in preset:
        if group is None or not isinstance(preset["group_label"], str) or not preset["group_label"].strip():
            raise ValueError("group_label must be nonempty and requires group")


def transform_points(points, matrix) -> np.ndarray:
    """Apply a source-to-output affine matrix to N x 2 pixel-boundary points."""
    values, affine = _points(points), _matrix(matrix)
    return values @ affine[:2, :2].T + affine[:2, 2]


def polygon_area(points) -> float:
    values = _points(points)
    if len(values) < 3:
        return 0.0
    # Translate before the shoelace sum to reduce cancellation for far-off ROIs.
    values = values - values[0]
    return float(abs(np.sum(values[:, 0] * np.roll(values[:, 1], -1)
                            - values[:, 1] * np.roll(values[:, 0], -1))) / 2)


def clip_polygon(points, width: int, height: int) -> np.ndarray:
    """Clip an ordered polygon against the output rectangle, preserving winding."""
    width, height = _size((width, height))
    polygon = _points(points).copy()
    for axis, boundary, keep_greater in ((0, 0.0, True), (0, float(width), False),
                                         (1, 0.0, True), (1, float(height), False)):
        if len(polygon) == 0:
            break
        result = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in polygon:
            inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if inside != previous_inside:
                fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                intersection = previous + fraction * (current - previous)
                intersection[axis] = boundary
                result.append(intersection)
            if inside:
                result.append(current.copy())
            previous, previous_inside = current, inside
        polygon = np.asarray(result, dtype=np.float64).reshape(-1, 2)
    # Remove consecutive duplicate vertices introduced by boundary intersections.
    cleaned = []
    for point in polygon:
        if not cleaned or not np.allclose(point, cleaned[-1], atol=1e-10, rtol=0):
            cleaned.append(point)
    if len(cleaned) > 1 and np.allclose(cleaned[0], cleaned[-1], atol=1e-10, rtol=0):
        cleaned.pop()
    return np.asarray(cleaned, dtype=np.float64).reshape(-1, 2)


def describe_polygon(points_pixel_xy, matrix, output_size) -> dict:
    """Describe a transformed ROI without hiding clipping with a fixed old box.

    Visibility is geometric polygon area within the output divided by its full
    transformed area. It is not the fraction of a vehicle's semantic identity.
    Integer crop boxes bound the visible polygon and may include background.
    """
    points = _points(points_pixel_xy, minimum=3)
    width, height = _size(output_size)
    original_area = polygon_area(points)
    if original_area <= 1e-12:
        raise ValueError("Source polygon must have positive area")
    transformed = transform_points(points, matrix)
    clipped = clip_polygon(transformed, width, height)
    transformed_area, visible_area = polygon_area(transformed), polygon_area(clipped)
    visible = visible_area > 1e-10
    full_box = [*transformed.min(axis=0), *transformed.max(axis=0)]
    visible_box, crop_box = None, None
    if visible:
        low, high = clipped.min(axis=0), clipped.max(axis=0)
        visible_box = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
        crop_box = [max(0, math.floor(low[0])), max(0, math.floor(low[1])),
                    min(width, math.ceil(high[0])), min(height, math.ceil(high[1]))]
    return {
        "transformed_polygon_xy": transformed.tolist(),
        "visible_polygon_xy": clipped.tolist() if visible else [],
        "transformed_bbox_xyxy": [float(v) for v in full_box],
        "visible_bbox_xyxy": visible_box,
        "crop_box_xyxy": crop_box,
        "original_area": original_area,
        "transformed_area": transformed_area,
        "visible_area": visible_area if visible else 0.0,
        "retained_fraction": min(1.0, max(0.0, visible_area / transformed_area)),
        "visible": visible,
    }


def plan_transform(size, preset: dict, subject_polygon=None) -> dict:
    """Plan a content-independent transform, with an optional subject zoom anchor.

    subject_polygon is normalized to the EXIF-corrected image rectangle. It only
    sets the zoom anchor; position and roll presets have explicit camera-frame
    behavior. No source filename or known vehicle class participates.
    """
    validate_preset(preset)
    width, height = _size(size)
    center = np.asarray([width / 2, height / 2], dtype=np.float64)
    subject_center = None
    if subject_polygon is not None:
        normalized = _points(subject_polygon, minimum=3)
        if np.any(normalized < 0) or np.any(normalized > 1) or polygon_area(normalized) <= 1e-12:
            raise ValueError("subject_polygon must have positive area and normalized coordinates in [0, 1]")
        points = normalized * np.asarray([width, height])
        subject_center = (points.min(axis=0) + points.max(axis=0)) / 2

    kind = preset["kind"]
    matrix = np.eye(3, dtype=np.float64)
    angle, margin, minimum_cover = 0.0, 0.0, None
    crop_window = None
    anchor = center.copy()
    if kind == "zoom":
        scale = float(preset["scale"])
        anchor = subject_center if subject_center is not None else center
        matrix[:2, :2] *= scale
        matrix[:2, 2] = (1 - scale) * anchor
        low = anchor * (1 - 1 / scale)
        high = low + np.asarray([width, height]) / scale
        crop_window = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
    elif kind == "position":
        scale = float(preset["scale"])
        window = np.asarray([width, height], dtype=np.float64) / scale
        low = (np.asarray([width, height]) - window) / 2
        direction = preset["direction"]
        if direction == "left":
            low[0] = width - window[0]
        elif direction == "right":
            low[0] = 0
        elif direction == "up":
            low[1] = height - window[1]
        else:
            low[1] = 0
        matrix[:2, :2] *= scale
        matrix[:2, 2] = -scale * low
        high = low + window
        crop_window = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
        anchor = low + window / 2
    else:
        angle = float(preset["angle"])
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        span_x = abs(cosine) * width + abs(sine) * height
        span_y = abs(sine) * width + abs(cosine) * height
        minimum_cover = max(span_x / width, span_y / height)
        # Small source-pixel margin prevents floating-point/cubic sampling from
        # revealing empty corners. It is recorded, not hidden strength tuning.
        margin = min(2.0, min(width, height) / 8) if angle else 0.0
        scale = max(span_x / (width - 2 * margin), span_y / (height - 2 * margin))
        matrix[:2, :2] = scale * rotation
        matrix[:2, 2] = center - matrix[:2, :2] @ center

    inverse = np.linalg.inv(matrix)
    corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    source_view = transform_points(corners, inverse)
    tolerance = 1e-8 * max(width, height)
    if np.any(source_view < -tolerance) or np.any(source_view > np.asarray([width, height]) + tolerance):
        raise ValueError("Transform would expose pixels outside the source; crop-only policy rejected it")
    return {
        "kind": kind,
        "input_size": [width, height],
        "output_size": [width, height],
        "forward_matrix": matrix.tolist(),
        "inverse_matrix": inverse.tolist(),
        "source_view_polygon_xy": source_view.tolist(),
        "scale": float(scale),
        "angle_degrees": angle,
        "angle_convention": "positive is clockwise in image coordinates (x right, y down)",
        "anchor_xy": anchor.tolist(),
        "anchor_kind": ("subject_bbox_center" if subject_center is not None else "image_center") if kind == "zoom" else ("source_crop_center" if kind == "position" else "image_center"),
        "crop_window_xyxy": crop_window,
        "rotation_minimum_cover_scale": minimum_cover,
        "rotation_source_margin_px": margin,
        "retention_policy": preset.get("retention_policy", "partial_allowed"),
        "coordinate_frame": "pixel boundaries [0,width] x [0,height]; pixel centers at half-integers",
        "border_policy": "crop_only_no_padding",
        "resampling": "single Pillow inverse AFFINE BICUBIC on sRGB RGB pixels",
        "viewpoint_scope": "2D framing, scale and camera roll; no new 3D viewpoint or hidden scene content",
    }


def apply_composition(rgb: np.ndarray, preset: dict, subject_polygon=None) -> CompositionResult:
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError("Input must be a nonempty uint8 H x W x 3 RGB array")
    height, width = rgb.shape[:2]
    metadata = plan_transform((width, height), preset, subject_polygon)
    inverse = np.asarray(metadata["inverse_matrix"])
    coefficients = tuple(float(v) for v in inverse[:2].reshape(-1))
    source = Image.fromarray(rgb)
    output = source.transform((width, height), Image.Transform.AFFINE, coefficients,
                              resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    mask = Image.new("L", (width, height), 255).transform(
        (width, height), Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.NEAREST, fillcolor=0)
    valid = np.array(mask, dtype=np.uint8)
    metadata["valid_pixel_fraction"] = float(np.mean(valid == 255))
    metadata["valid_mask_definition"] = "inverse-mapped pixel centers are within source extent; not a semantic subject mask"
    if not np.all(valid == 255):
        raise RuntimeError("Crop-only transform exposed padding; output rejected")
    return CompositionResult(np.array(output, dtype=np.uint8), valid, metadata)
