"""Prepare RGBA foregrounds; light-checker removal is an explicit special case.

This is a colour-key matte for dark/coloured objects on a light neutral
checkerboard. It is not a general photographic segmentation algorithm.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _profile(rgb: np.ndarray, horizontal: bool) -> tuple[np.ndarray, int]:
    """Find an unobstructed border carrying a regular, bright checker pattern."""
    candidates = []
    extent = rgb.shape[0 if horizontal else 1]
    for offset in (0, 1, 2, extent - 3, extent - 2, extent - 1):
        line = rgb[offset] if horizontal else rgb[:, offset]
        neutral = (line.max(1) - line.min(1) <= 8) & (line.min(1) >= 218)
        if neutral.mean() < 0.97:
            continue
        gray = line.mean(1)
        low, high = np.percentile(gray, (20, 80))
        if not (218 <= low <= 247 and high >= 248 and high - low >= 7):
            continue
        transitions = np.flatnonzero(np.diff(gray > (low + high) / 2))
        lengths = np.diff(transitions)
        if len(lengths) < 5 or np.median(lengths) < 4:
            continue
        if np.mean(np.abs(lengths - np.median(lengths)) <= 2) < 0.9:
            continue
        candidates.append((float(neutral.mean()), gray, offset))
    if not candidates:
        raise ValueError(
            "light-checker requires a regular pale neutral checkerboard and "
            "a clear horizontal and vertical border; use a prepared RGBA asset otherwise"
        )
    _, gray, offset = max(candidates, key=lambda value: value[0])
    return gray, offset


def _checker_background(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    horizontal, row = _profile(rgb, True)
    vertical, col = _profile(rgb, False)
    low, high = np.percentile(np.concatenate((horizontal, vertical)), (20, 80))
    midpoint = (low + high) / 2
    # Multiplication of the two square waves reconstructs the checker phase;
    # using the measured border profiles also retains antialiased tile edges.
    xwave = np.clip((horizontal - midpoint) / ((high - low) / 2), -1, 1)
    ywave = np.clip((vertical - midpoint) / ((high - low) / 2), -1, 1)
    corner_sign = 1 if rgb[row, col].mean() > midpoint else -1
    background = midpoint + corner_sign * ywave[:, None] * xwave[None, :] * ((high - low) / 2)
    neutral = (rgb.max(2) - rgb.min(2) <= 8) & (rgb.min(2) >= 218)
    agreement = float(np.mean(np.abs(rgb.mean(2)[neutral] - background[neutral]) <= 7))
    if neutral.mean() < 0.25 or agreement < 0.85:
        raise ValueError("image does not match a consistent light checkerboard background")
    return background.astype(np.float32), {
        "checker_low": float(low), "checker_high": float(high),
        "checker_agreement": agreement, "clear_border_row": int(row),
        "clear_border_column": int(col),
    }


def prepare_material(image: Image.Image, mode: str = "alpha") -> tuple[Image.Image, dict]:
    """Return RGBA plus diagnostics, preserving an existing useful alpha exactly.

    ``alpha`` requires actual transparency. ``light-checker`` explicitly enables
    the narrow colour-key operation for material with a baked-in checkerboard.
    White or pale neutral foreground details cannot be reliably distinguished
    from this background and should use a separately supplied mask instead.
    """
    if mode not in {"alpha", "light-checker"}:
        raise ValueError("mode must be alpha or light-checker")
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    supplied_alpha = np.asarray(rgba.getchannel("A"))
    if np.any(supplied_alpha < 255):
        if not np.any(supplied_alpha > 0):
            raise ValueError("asset is completely transparent")
        return rgba.copy(), {"method": "existing_alpha", "size": list(image.size),
                             "alpha_preserved": True}
    if mode == "alpha":
        raise ValueError("asset has no transparency; provide RGBA or explicitly use light-checker")

    rgb = np.asarray(rgba, dtype=np.float32)[..., :3]
    background, metadata = _checker_background(rgb)
    minimum, spread = rgb.min(2), np.ptp(rgb, axis=2)
    # Keep coloured leaves and dark twigs solid. Only pale boundary mixtures
    # receive a partial alpha; no overall opacity reduction is applied.
    foreground_core = (minimum < 190) | (spread > 40)
    # A colour-key edge may already contain a substantial amount of the pale
    # background. Use interior colours to unmix it, while keeping genuinely
    # dark thin twigs as anchors even when they have no eroded interior.
    eroded = np.asarray(Image.fromarray(foreground_core.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 0
    core = eroded | (minimum < 75)
    candidate = (minimum < 224) | (spread > 12)
    if core.sum() < 16:
        raise ValueError("no suitable dark or coloured foreground was found")
    known = core.copy()
    neighbour_color = rgb.copy()
    h, w = known.shape
    for _ in range(8):
        new_known = known.copy()
        new_color = neighbour_color.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            sy = slice(max(0, -dy), min(h, h - dy))
            sx = slice(max(0, -dx), min(w, w - dx))
            ty = slice(max(0, dy), min(h, h + dy))
            tx = slice(max(0, dx), min(w, w + dx))
            fill = candidate[ty, tx] & ~new_known[ty, tx] & known[sy, sx]
            new_color[ty, tx][fill] = neighbour_color[sy, sx][fill]
            new_known[ty, tx][fill] = True
        if np.array_equal(new_known, known):
            break
        known, neighbour_color = new_known, new_color

    bg = background[..., None]
    difference = neighbour_color - bg
    denominator = np.sum(difference * difference, axis=2)
    alpha = np.clip(np.sum((rgb - bg) * difference, axis=2) / np.maximum(denominator, 1), 0, 1)
    alpha[core] = 1
    alpha[~candidate | ~known] = 0
    # Very weak chromatic JPEG speckle outside the material is not useful matte.
    alpha[alpha < 0.025] = 0
    foreground = (rgb - (1 - alpha[..., None]) * bg) / np.maximum(alpha[..., None], 1e-6)
    foreground[core] = rgb[core]
    foreground[alpha == 0] = 0
    result = np.dstack((np.clip(np.rint(foreground), 0, 255).astype(np.uint8),
                        np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)))
    metadata.update({
        "method": "light_checker_colour_key", "size": list(image.size),
        "alpha_preserved": False, "foreground_fraction_alpha_ge_128": float(np.mean(result[..., 3] >= 128)),
        "alpha_weighted_fraction": float(np.mean(result[..., 3] / 255)),
        "parameters": {"opaque_min_channel_below": 190, "opaque_chroma_above": 40,
                       "core_erosion_pixels": 1, "thin_dark_anchor_below": 75,
                       "candidate_min_channel_below": 224, "candidate_chroma_above": 12,
                       "neighbour_radius_pixels": 8, "minimum_alpha": 0.025},
        "limitations": "Designed for coloured/dark foregrounds on light checkerboard; pale neutral foreground may be removed. Inspect the matte before reuse.",
    })
    return Image.fromarray(result, "RGBA"), metadata
