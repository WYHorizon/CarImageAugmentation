"""Deterministic image-space optical artifacts, not geometric scene relighting.

All additions operate on approximate linear RGB. `strength` multiplies EV and
additive optical fields; zero gives the exact identity. Source coordinates are
normalized image x/y, radii are fractions of the shorter image edge. The optical
field is evaluated at at most 960 pixels on its longest edge, then bilinearly
upsampled in floating point. No quantization or clipping occurs in this module.

Bloom extracts max(exposed linear luminance - threshold, 0), blurs it at radius
multiples 0.45/1/2.5 and adds a 0.45/0.35/0.20 weighted sum. Blur uses three box
passes approximating a Gaussian, with reflected boundaries and float64 running
sums; input and output remain float32. Glare is a broad Gaussian veil plus two
narrower source lobes. Flare adds weak axis-aligned ghosts and source star rays.
Backlight combines exposure reduction and a directional source/glare field;
it does not estimate surface normals, cast shadows, occlusion, or genuine rims.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MAX_FIELD_EDGE = 960
DEFAULTS = {
    'strength': 1.0, 'ev': 0.0, 'source': [0.8, 0.15],
    'color': [1.0, 0.85, 0.65], 'radius': 0.5,
    'veil': 0.0, 'core': 0.0, 'rays': 0.0, 'ghosts': 0.0,
    'bloom_threshold': 0.6, 'bloom_radius': 0.025, 'bloom_strength': 0.0,
}
RANGES = {
    'strength': (0, 2), 'ev': (-4, 4), 'radius': (0.005, 2),
    'veil': (0, 4), 'core': (0, 12), 'rays': (0, 4), 'ghosts': (0, 2),
    'bloom_threshold': (0, 16), 'bloom_radius': (0.001, 0.3), 'bloom_strength': (0, 8),
}


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


def validate_effect(preset: dict) -> None:
    if not isinstance(preset, dict) or preset.get('kind') != 'optical':
        raise ValueError('Optical preset requires kind=optical')
    if not isinstance(preset.get('label'), str) or not preset['label'].strip():
        raise ValueError('Optical preset requires a nonempty label')
    if preset.get('effect') not in {'bloom', 'glare', 'flare', 'backlight'}:
        raise ValueError('Unknown optical effect')
    permitted = set(DEFAULTS) | {'kind', 'effect', 'label', 'group', 'group_label'}
    if set(preset) - permitted:
        raise ValueError(f'Unknown optical parameter(s): {sorted(set(preset) - permitted)}')
    for name, (minimum, maximum) in RANGES.items():
        number = _finite_number(preset.get(name, DEFAULTS[name]), name)
        if not minimum <= number <= maximum:
            raise ValueError(f'{name} must be in [{minimum}, {maximum}]')
    for name, count, minimum, maximum in [('source', 2, -0.5, 1.5), ('color', 3, 0, 2)]:
        values = preset.get(name, DEFAULTS[name])
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != count:
            raise ValueError(f'{name} must contain {count} numbers')
        if any(not minimum <= _finite_number(value, name) <= maximum for value in values):
            raise ValueError(f'{name} components must be in [{minimum}, {maximum}]')
    if not any(preset.get('color', DEFAULTS['color'])):
        raise ValueError('Light color cannot be black')


def _resize(field, size):
    """Pillow F mode preserves floats, including values above one."""
    if field.shape == (size[1], size[0]):
        return field.astype(np.float32, copy=True)
    return np.asarray(Image.fromarray(field.astype(np.float32)).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _box_blur(field, radius, axis):
    if radius == 0:
        return field.copy()
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    # Edge replication handles one-pixel fields as well as very large kernels.
    padded = np.pad(field, padding, mode='reflect' if field.shape[axis] > 1 else 'edge')
    sums = np.cumsum(padded, axis=axis, dtype=np.float64)
    leading = list(sums.shape)
    leading[axis] = 1
    sums = np.concatenate([np.zeros(leading, dtype=np.float64), sums], axis=axis)
    size = 2 * radius + 1
    first, second = [slice(None)] * 2, [slice(None)] * 2
    first[axis], second[axis] = slice(size, None), slice(None, -size)
    return ((sums[tuple(first)] - sums[tuple(second)]) / size).astype(np.float32)


def _gaussian(field, sigma):
    # Three equal box filters have variance r(r+1); choose r for the target sigma.
    radius = max(0, int(round((math.sqrt(1 + 4 * sigma * sigma) - 1) / 2)))
    result = field.copy()
    for _ in range(3):
        result = _box_blur(_box_blur(result, radius, 1), radius, 0)
    return result


def _bloom(linear, params, size):
    exposed_luma = np.einsum('...c,c->...', linear, LUMA)
    excess = np.maximum(exposed_luma - params['bloom_threshold'], 0)
    # Threshold before reducing so bright emitters contribute to the reduced
    # field. Bilinear reduction is an approximation, not exact energy transport.
    reduced = _resize(excess, size)
    sigma = params['bloom_radius'] * min(size)
    return sum(weight * _gaussian(reduced, max(0.4, sigma * multiplier))
               for weight, multiplier in [(0.45, 0.45), (0.35, 1), (0.20, 2.5)])


def apply_effect(linear: np.ndarray, preset: dict) -> np.ndarray:
    validate_effect(preset)
    image = np.asarray(linear)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1 or not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError('Expected a nonempty real H x W x 3 image')
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError('Linear input must be finite and nonnegative')
    if np.any(image > np.finfo(np.float32).max / 256):
        raise ValueError('Linear input exceeds safe floating-point range')
    params = {**DEFAULTS, **preset}
    strength = float(params['strength'])
    if strength == 0:
        return image.astype(np.float32, copy=True)
    result = image.astype(np.float32, copy=True)
    result *= np.float32(2 ** (strength * params['ev']))
    height, width = result.shape[:2]
    scale = min(1.0, MAX_FIELD_EDGE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Pixel centers in short-edge units give symmetric source mirroring and
    # invariant field proportions across landscape and portrait resolutions.
    short = min(size)
    x = ((np.arange(size[0], dtype=np.float32) + 0.5) / size[0] - params['source'][0]) * size[0] / short
    y = ((np.arange(size[1], dtype=np.float32) + 0.5) / size[1] - params['source'][1]) * size[1] / short
    distance_squared = y[:, None] ** 2 + x[None, :] ** 2
    radius = params['radius']
    field = params['veil'] * np.exp(-distance_squared / (2 * radius * radius))
    field += params['core'] * (
        0.85 * np.exp(-distance_squared / (2 * (radius * 0.075) ** 2))
        + 0.15 * np.exp(-distance_squared / (2 * (radius * 0.30) ** 2)))
    if params['rays']:
        # Several smooth anisotropic lobes, restricted around the emitter.
        rays = np.zeros_like(field)
        for angle in [0.0, math.pi / 3, 2 * math.pi / 3]:
            along = x[None, :] * math.cos(angle) + y[:, None] * math.sin(angle)
            across = -x[None, :] * math.sin(angle) + y[:, None] * math.cos(angle)
            rays += np.exp(-0.5 * ((along / (radius * 0.8)) ** 2 + (across / (radius * 0.009)) ** 2))
        field += params['rays'] * rays
    if params['bloom_strength']:
        field += params['bloom_strength'] * _bloom(result, params, size)
    colors = np.asarray(params['color'], dtype=np.float32)
    for channel in range(3):
        result[..., channel] += np.float32(strength * colors[channel]) * _resize(field, (width, height))

    if params['ghosts']:
        # Reflections follow the source -> optical center axis. Broad soft edges
        # and small additive energy avoid opaque circles covering body details.
        sx, sy = params['source']
        for travel, ghost_radius, tint, energy in [
            (0.52, 0.045, [1.0, 0.72, 0.48], 0.45),
            (1.12, 0.080, [0.48, 0.78, 0.67], 0.28),
            (1.56, 0.125, [0.77, 0.57, 0.42], 0.22),
            (1.96, 0.060, [0.58, 0.67, 0.85], 0.30),
        ]:
            gx, gy = sx + travel * (0.5 - sx), sy + travel * (0.5 - sy)
            dx = x[None, :] + (sx - gx) * size[0] / short
            dy = y[:, None] + (sy - gy) * size[1] / short
            r2 = (dx * dx + dy * dy) / (ghost_radius * ghost_radius)
            ghost = np.exp(-r2 * 0.5) + 0.22 * np.exp(-0.5 * ((np.sqrt(r2) - 1.6) / 0.22) ** 2)
            for channel in range(3):
                result[..., channel] += np.float32(strength * params['ghosts'] * energy * tint[channel]) * _resize(ghost, (width, height))
    if not np.isfinite(result).all():
        raise ValueError('Optical effect produced non-finite RGB')
    return result
