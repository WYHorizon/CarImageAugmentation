"""Independent imaging-quality degradations for uint8 sRGB photographs.

Blur and noise operate in approximate linear RGB. An already processed JPEG is
not RAW sensor data: the noise model does not reproduce a Bayer array, camera
demosaicing or an ISP. Optional coarse noise resamples only a random residual
field, retaining the original image signal. Defocus uses three separable,
normalized box filters per axis as an efficient Gaussian approximation;
its realized sigma is reported.
Motion uses a normalized, centered subpixel line kernel, with reflected edges.
Neither operation moves the vehicle or synthesizes new geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from numbers import Integral, Real
import re

import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    """Decoded pixels plus exact JPEG bytes / native low-resolution pixels."""

    image: np.ndarray
    metadata: dict
    encoded_jpeg: bytes | None = None
    native_image: np.ndarray | None = None


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return int(value)


def validate_preset(preset: dict) -> None:
    """Check all parameters before processing a photo; do not alter the preset.

    Numerical bounds prevent accidental huge kernels and invalid RNG/codec
    arguments. These bounds are not thresholds for training-data acceptance.
    """
    if not isinstance(preset, dict):
        raise ValueError("Preset must be a dictionary")
    if not isinstance(preset.get("label"), str) or not preset["label"].strip():
        raise ValueError("Preset must have a nonempty label")
    group = preset.get("group")
    if group is not None:
        if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", group):
            raise ValueError("group must be a safe alphanumeric identifier")
    if "group_label" in preset:
        if group is None:
            raise ValueError("group_label requires group")
        if not isinstance(preset["group_label"], str) or not preset["group_label"].strip():
            raise ValueError("group_label must be a nonempty string")

    kind = preset.get("kind")
    if not isinstance(kind, str):
        raise ValueError("kind must name a supported imaging-quality operation")
    if kind in {"defocus", "motion"}:
        _integer(preset.get("reference_short_edge"), "reference_short_edge", 1, 65536)
    if kind == "defocus":
        sigma = _number(preset.get("sigma"), "sigma")
        if not 0 <= sigma <= 32:
            raise ValueError("sigma must be in [0, 32] at the reference resolution")
    elif kind == "motion":
        length = _number(preset.get("length"), "length")
        angle = _number(preset.get("angle"), "angle")
        if not 1 <= length <= 129:
            raise ValueError("length must be in [1, 129] at the reference resolution")
        if not -180 <= angle <= 180:
            raise ValueError("angle must be in [-180, 180] degrees")
    elif kind == "noise":
        peak = _number(preset.get("peak"), "peak")
        read_sigma = _number(preset.get("read_sigma"), "read_sigma")
        if not 1 <= peak <= 1e9:
            raise ValueError("peak must be in [1, 1e9]")
        if not 0 <= read_sigma <= 0.25:
            raise ValueError("read_sigma must be in [0, 0.25] in normalized linear RGB")
        if "grain_short_edge" in preset:
            _integer(preset["grain_short_edge"], "grain_short_edge", 1, 65536)
    elif kind == "resolution":
        _integer(preset.get("short_edge"), "short_edge", 1, 65536)
    elif kind == "jpeg":
        _integer(preset.get("quality"), "quality", 1, 95)
        # Keeping one chroma mode makes the initial compression experiment easy
        # to interpret; 2 is Pillow's numeric name for 4:2:0 subsampling.
        if preset.get("subsampling", 2) not in (2, "4:2:0"):
            raise ValueError("JPEG subsampling must be 2 or '4:2:0'")
    else:
        raise ValueError(f"Unknown imaging-quality kind: {kind!r}")


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise ValueError("Expected a NumPy uint8 RGB image")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if rgb.dtype != np.uint8:
        raise ValueError("RGB input must have dtype uint8")
    return rgb


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode normalized sRGB [0, 1] into float32 linear RGB."""
    values = np.asarray(rgb)
    if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Normalized sRGB must contain finite real values in [0, 1]")
    values = values.astype(np.float32, copy=False)
    return np.where(values <= 0.04045, values / 12.92,
                    ((values + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Clip finite linear values to [0, 1] and encode normalized float32 sRGB."""
    values = np.asarray(linear)
    if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise ValueError("Linear RGB must contain finite real values")
    clipped = np.clip(values, 0, 1).astype(np.float32, copy=False)
    return np.where(clipped <= 0.0031308, clipped * 12.92,
                    1.055 * clipped ** (1 / 2.4) - 0.055).astype(np.float32)


_SRGB_TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / 255)


def _encode_rgb(linear: np.ndarray) -> np.ndarray:
    # Work one channel at a time, limiting temporary arrays for 4K photos.
    output = np.empty(linear.shape, dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.rint(
            linear_to_srgb(linear[..., channel]) * 255
        ).astype(np.uint8)
    return output


def _box_widths(sigma: float) -> list[int]:
    """Three odd widths with variance closest to the requested Gaussian."""
    ideal = math.sqrt(4 * sigma * sigma + 1)
    lower = int(math.floor(ideal))
    if lower % 2 == 0:
        lower -= 1
    lower = max(1, lower)
    upper = lower + 2
    count_lower = round((12 * sigma * sigma - 3 * lower * lower - 12 * lower - 9)
                        / (-4 * lower - 4))
    count_lower = min(3, max(0, count_lower))
    return [lower] * count_lower + [upper] * (3 - count_lower)


def _box_axis(plane: np.ndarray, width: int, axis: int) -> np.ndarray:
    if width == 1:
        return plane.copy()
    pad = [(0, 0), (0, 0)]
    pad[axis] = (width // 2, width // 2)
    # NumPy reflect also handles a singleton axis by repeating its only value.
    padded = np.pad(plane, pad, mode="reflect")
    integral_shape = list(padded.shape)
    integral_shape[axis] += 1
    integral = np.zeros(integral_shape, dtype=np.float64)
    start = [slice(None), slice(None)]
    start[axis] = slice(1, None)
    np.cumsum(padded, axis=axis, dtype=np.float64, out=integral[tuple(start)])
    high, low = [slice(None), slice(None)], [slice(None), slice(None)]
    high[axis], low[axis] = slice(width, None), slice(None, -width)
    result = (integral[tuple(high)] - integral[tuple(low)]) / width
    return result.astype(np.float32)


def _defocus(linear: np.ndarray, widths: list[int]) -> np.ndarray:
    result = np.empty_like(linear)
    for channel in range(3):
        plane = linear[..., channel]
        for width in widths:
            plane = _box_axis(plane, width, 1)
            plane = _box_axis(plane, width, 0)
        result[..., channel] = plane
    return result


def _motion_kernel(length: float, angle: float) -> dict[tuple[int, int], float]:
    """Rasterize a centered uniform line to a sparse bilinear pixel kernel.

    A length of L pixels spans L-1 pixel-center units. Subpixel samples are at
    most 0.5 pixel apart. Pairing opposite weights makes the discrete kernel
    exactly center-symmetric, including non-axis-aligned directions.
    """
    half_span = max(0.0, (length - 1) / 2)
    if half_span == 0:
        return {(0, 0): 1.0}
    samples = max(3, 2 * math.ceil(half_span * 2) + 1)
    radians = math.radians(angle)
    kernel: dict[tuple[int, int], float] = {}
    for distance in np.linspace(-half_span, half_span, samples):
        dx, dy = distance * math.cos(radians), distance * math.sin(radians)
        x0, y0 = math.floor(dx), math.floor(dy)
        fx, fy = dx - x0, dy - y0
        for x, wx in ((x0, 1 - fx), (x0 + 1, fx)):
            for y, wy in ((y0, 1 - fy), (y0 + 1, fy)):
                weight = float(wx * wy)
                if weight > 1e-12:
                    kernel[(y, x)] = kernel.get((y, x), 0.0) + weight
    offsets = set(kernel) | {(-y, -x) for y, x in kernel}
    symmetric = {key: (kernel.get(key, 0) + kernel.get((-key[0], -key[1]), 0)) / 2
                 for key in offsets}
    total = math.fsum(symmetric.values())
    return {key: weight / total for key, weight in sorted(symmetric.items())}


def _motion(linear: np.ndarray, kernel: dict[tuple[int, int], float]) -> np.ndarray:
    pad_y = max(abs(y) for y, _ in kernel)
    pad_x = max(abs(x) for _, x in kernel)
    height, width = linear.shape[:2]
    result = np.zeros_like(linear)
    for channel in range(3):
        padded = np.pad(linear[..., channel], ((pad_y, pad_y), (pad_x, pad_x)),
                        mode="reflect")
        # Small row blocks keep convolution scratch space bounded and reusable.
        for top in range(0, height, 128):
            bottom = min(height, top + 128)
            plane = result[top:bottom, :, channel]
            scratch = np.empty((bottom - top, width), dtype=np.float32)
            for (dy, dx), weight in kernel.items():
                source = padded[top + pad_y + dy:bottom + pad_y + dy,
                                pad_x + dx:pad_x + dx + width]
                np.multiply(source, np.float32(weight), out=scratch)
                np.add(plane, scratch, out=plane)
    return result


def apply_quality(rgb: np.ndarray, preset: dict, seed: int = 0) -> QualityResult:
    """Apply one preset to the original RGB array, without mutating any input.

    Output dimensions always equal input dimensions. JPEG bytes are returned
    separately so callers can retain the actual compressed stream; PNG exports
    of ``image`` are the decoded JPEG pixels. Resolution presets also expose
    the real native low-resolution image before bicubic restoration.
    """
    validate_preset(preset)
    rgb = _rgb_uint8(rgb)
    seed = _integer(seed, "seed", 0, (1 << 128) - 1)
    height, width = rgb.shape[:2]
    short_edge = min(height, width)
    kind = preset["kind"]
    metadata = {
        "kind": kind,
        "input_size": [width, height],
        "output_size": [width, height],
        "input_short_edge": short_edge,
        "seed": seed,
        "seed_used": kind == "noise",
        "color_space": "sRGB",
    }
    if "group" in preset:
        metadata["group"] = preset["group"]
    if "group_label" in preset:
        metadata["group_label"] = preset["group_label"]

    if kind in {"defocus", "motion"}:
        reference = int(preset["reference_short_edge"])
        scale = short_edge / reference
        metadata.update({
            "working_color_space": "approximate linear sRGB",
            "reference_short_edge": reference,
            "reference_scale": scale,
            "boundary": "reflect",
        })
        linear = _SRGB_TO_LINEAR[rgb]
        if kind == "defocus":
            sigma = float(preset["sigma"]) * scale
            if sigma > 512:
                raise ValueError("Effective sigma exceeds the 512 pixel resource limit")
            widths = _box_widths(sigma)
            metadata.update({
                "operator": "three separable box filters approximating a Gaussian",
                "sigma_at_reference": float(preset["sigma"]),
                "effective_sigma_target_px": sigma,
                "effective_sigma_kernel_px": math.sqrt(sum((n * n - 1) / 12 for n in widths)),
                "box_widths_px": widths,
            })
            image = _encode_rgb(_defocus(linear, widths))
        else:
            length = max(1.0, float(preset["length"]) * scale)
            if length > 513:
                raise ValueError("Effective motion length exceeds the 513 pixel resource limit")
            angle = float(preset["angle"])
            kernel = _motion_kernel(length, angle)
            metadata.update({
                "operator": "centered uniform line with bilinear subpixel weights",
                "length_at_reference": float(preset["length"]),
                "effective_length_px": length,
                "effective_center_span_px": length - 1,
                "angle_degrees": angle,
                "angle_convention": "positive x to the right, positive y downward",
                "kernel_size": [2 * max(abs(x) for _, x in kernel) + 1,
                                2 * max(abs(y) for y, _ in kernel) + 1],
                "kernel_nonzero_taps": len(kernel),
            })
            image = _encode_rgb(_motion(linear, kernel))
        return QualityResult(image, metadata)

    if kind == "noise":
        peak = float(preset["peak"])
        read_sigma = float(preset["read_sigma"])
        requested_grain = int(preset["grain_short_edge"]) if "grain_short_edge" in preset else None
        grain_edge = min(requested_grain, short_edge) if requested_grain is not None else short_edge
        grain_scale = grain_edge / short_edge
        grain_size = (max(1, round(width * grain_scale)), max(1, round(height * grain_scale)))
        grain_resampled = grain_size != (width, height)
        rng = np.random.Generator(np.random.PCG64(seed))
        # Process one plane at a time. The documented traversal is part of
        # reproducibility: channel-major, shot samples then read samples.
        image = np.empty_like(rgb)
        below, above = 0, 0
        minimum, maximum = math.inf, -math.inf
        for channel in range(3):
            linear = _SRGB_TO_LINEAR[rgb[..., channel]]
            if grain_resampled:
                # Resample a noise residual, not the image itself. A coarser
                # field keeps visible spatially correlated grain after display
                # reduction without discarding original vehicle detail first.
                # BILINEAR keeps the downsampled signal inside [0, 1].
                signal = np.asarray(Image.fromarray(linear).resize(
                    grain_size, Image.Resampling.BILINEAR), dtype=np.float32)
                residual = rng.poisson(signal.astype(np.float64) * peak) / peak
                residual += rng.normal(0, read_sigma, size=signal.shape)
                residual -= signal
                enlarged = np.asarray(Image.fromarray(residual.astype(np.float32)).resize(
                    (width, height), Image.Resampling.BILINEAR), dtype=np.float32)
                noisy = linear + enlarged
            else:
                # Preserve the original numeric operations and RNG traversal
                # exactly for legacy presets and full-resolution grain grids.
                noisy = rng.poisson(linear.astype(np.float64) * peak) / peak
                noisy += rng.normal(0, read_sigma, size=linear.shape)
            below += int(np.count_nonzero(noisy < 0))
            above += int(np.count_nonzero(noisy > 1))
            minimum = min(minimum, float(np.min(noisy)))
            maximum = max(maximum, float(np.max(noisy)))
            image[..., channel] = np.rint(linear_to_srgb(noisy) * 255).astype(np.uint8)
        metadata.update({
            "working_color_space": "approximate linear sRGB",
            "operator": ("spatially correlated RGB Poisson-plus-read residual field"
                         if grain_resampled else "independent RGB Poisson shot noise plus Gaussian read noise"),
            "peak": peak,
            "read_sigma_linear": read_sigma,
            "rng": "NumPy PCG64",
            "rng_traversal": "channel-major; shot then read for each channel",
            "grain_requested_short_edge": requested_grain,
            "grain_grid_size": list(grain_size),
            "grain_resampled": grain_resampled,
            "grain_signal_downsample_filter": "BILINEAR" if grain_resampled else None,
            "grain_residual_restore_filter": "BILINEAR" if grain_resampled else None,
            "grain_scope": "image-space noise approximation; no calibrated sensor or ISO model; original signal is not resized",
            "unclipped_linear_min": minimum,
            "unclipped_linear_max": maximum,
            "clipped_below_zero_fraction_channels": below / rgb.size,
            "clipped_above_one_fraction_channels": above / rgb.size,
        })
        return QualityResult(image, metadata)

    if kind == "resolution":
        requested = int(preset["short_edge"])
        target = min(requested, short_edge)
        scale = target / short_edge
        native_size = (max(1, int(round(width * scale))),
                       max(1, int(round(height * scale))))
        if native_size == (width, height):
            native = rgb.copy()
            image = rgb.copy()
        else:
            native_pil = Image.fromarray(rgb).resize(native_size, Image.Resampling.LANCZOS)
            native = np.array(native_pil, dtype=np.uint8)
            image = np.array(native_pil.resize((width, height), Image.Resampling.BICUBIC),
                             dtype=np.uint8)
        metadata.update({
            "operator": "downsample then restore original size",
            "requested_short_edge": requested,
            "effective_short_edge": min(native_size),
            "native_size": list(native_size),
            "native_pixel_fraction": math.prod(native_size) / (width * height),
            "downsample_scale": scale,
            "downsample_filter": "LANCZOS",
            "restore_filter": "BICUBIC",
            "resized": native_size != (width, height),
        })
        return QualityResult(image, metadata, native_image=native)

    # JPEG is always encoded at native input resolution, with no resize stage.
    quality = int(preset["quality"])
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=quality, subsampling=2,
                              optimize=False, progressive=False)
    encoded = stream.getvalue()
    with Image.open(BytesIO(encoded)) as decoded:
        image = np.array(decoded.convert("RGB"), dtype=np.uint8)
    metadata.update({
        "operator": "native-resolution JPEG encode and decode",
        "quality": quality,
        "subsampling": "4:2:0",
        "encoded_bytes": len(encoded),
        "resized": False,
        "progressive": False,
        "optimize": False,
    })
    return QualityResult(image, metadata, encoded_jpeg=encoded)
