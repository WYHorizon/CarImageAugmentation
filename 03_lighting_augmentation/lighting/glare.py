"""Off-frame camera glare without inserting a scene light or relighting the car.

This is a phenomenological optical model, not a calibrated lens simulation or
measured flare texture. A source outside the frame drives an anisotropic
long-tailed scattering field. Lens ghosts are weak, softly defocused aperture
images on the source--optical-center axis, with small chromatic displacement.
There is no visible sun disk, star overlay, segmentation mask, or depth effect.

Coordinates are normalized x/y; radii use the shorter image dimension. Fields
are computed at no more than 960 pixels on the long edge and resized in float.
Strength multiplies optical energy and EV; strength zero is exact identity.
The result remains unclipped so the pipeline can measure lost display range.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from PIL import Image


MAX_FIELD_EDGE = 960
DEFAULTS = {
    'strength': 1.0, 'ev': 0.0, 'source': [-0.10, 0.15],
    'color': [1.0, 0.97, 0.91], 'scatter_radius': 0.45,
    'scatter_aspect': 1.5, 'scatter_falloff': 2.2,
    'veil': 0.30, 'near_spill': 0.35,
    'ghost_strength': 0.0, 'ghost_scale': 1.0,
}
RANGES = {
    'strength': (0, 2), 'ev': (-2, 2),
    'scatter_radius': (0.03, 1.2), 'scatter_aspect': (0.5, 3),
    'scatter_falloff': (1, 5), 'veil': (0, 2), 'near_spill': (0, 4),
    'ghost_strength': (0, 1), 'ghost_scale': (0.3, 2),
}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


def validate_glare(preset: dict) -> None:
    """Reject invalid or unused settings before generating a batch."""
    if not isinstance(preset, dict) or preset.get('kind') != 'camera_glare':
        raise ValueError('Camera glare preset requires kind=camera_glare')
    if not isinstance(preset.get('label'), str) or not preset['label'].strip():
        raise ValueError('Camera glare preset requires a nonempty label')
    if preset.get('effect') not in {'sun', 'lens'}:
        raise ValueError('Camera glare effect must be sun or lens')
    allowed = set(DEFAULTS) | {'kind', 'effect', 'label', 'group', 'group_label'}
    if set(preset) - allowed:
        raise ValueError(f'Unknown camera-glare parameters: {sorted(set(preset) - allowed)}')
    for name, (minimum, maximum) in RANGES.items():
        if not minimum <= _number(preset.get(name, DEFAULTS[name]), name) <= maximum:
            raise ValueError(f'{name} must be in [{minimum}, {maximum}]')
    for name, length, minimum, maximum in [('source', 2, -0.75, 1.75), ('color', 3, 0, 2)]:
        values = preset.get(name, DEFAULTS[name])
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != length:
            raise ValueError(f'{name} must contain {length} numbers')
        if any(not minimum <= _number(value, name) <= maximum for value in values):
            raise ValueError(f'{name} components must be in [{minimum}, {maximum}]')
    sx, sy = preset.get('source', DEFAULTS['source'])
    if 0 <= sx <= 1 and 0 <= sy <= 1:
        raise ValueError('The camera-glare source must be outside the image frame')
    if not any(preset.get('color', DEFAULTS['color'])):
        raise ValueError('Glare color must not be black')
    if preset['effect'] == 'sun' and preset.get('ghost_strength', 0) != 0:
        raise ValueError('Use effect=lens to enable aperture ghosts')


def _resize(field, size):
    if field.shape == (size[1], size[0]):
        return field.astype(np.float32, copy=True)
    return np.asarray(Image.fromarray(field.astype(np.float32)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32)


def _soft_aperture(along, across, radius, rotation):
    """A defocused seven-blade pupil with soft edges and mild radial elongation.

    The blade profile only gently perturbs a circular pupil; defocus blurs its
    boundary. This keeps the ghost translucent rather than a solid polygon.
    """
    u, v = along / (radius * 1.18), across / radius
    theta = np.arctan2(v, u) - rotation
    sector = np.mod(theta + math.pi / 7, 2 * math.pi / 7) - math.pi / 7
    polygon_radius = math.cos(math.pi / 7) / np.cos(sector)
    boundary = 0.55 + 0.45 * polygon_radius
    q = np.sqrt(u * u + v * v) / boundary
    # A broad transition imitates a defocused aperture image. No bright rim.
    transition = np.clip((q - 0.30) / 1.35, 0, 1)
    disk = 1 - transition * transition * (3 - 2 * transition)
    # Weak asymmetric spill follows the optical axis rather than a ring.
    tail = np.exp(-0.5 * (((u - 0.45) / 1.05) ** 2 + (v / 0.75) ** 2))
    return (0.83 * disk + 0.17 * tail).astype(np.float32)


def _field(params, size):
    width, height = size
    short = min(size)
    sx, sy = params['source']
    x = ((np.arange(width, dtype=np.float32) + 0.5) / width - sx) * width / short
    y = ((np.arange(height, dtype=np.float32) + 0.5) / height - sy) * height / short
    axis = np.array([(0.5 - sx) * width / short, (0.5 - sy) * height / short], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    along = x[None, :] * axis[0] + y[:, None] * axis[1]
    across = -x[None, :] * axis[1] + y[:, None] * axis[0]
    radius = params['scatter_radius']
    r2 = (along / (radius * params['scatter_aspect'])) ** 2 + (across / radius) ** 2
    scattering = params['veil'] / (1 + r2) ** params['scatter_falloff']
    # An extended edge spill, not a luminous scene object. Its center is always
    # off-frame and its radial extent is broader along the optical axis.
    scattering += params['near_spill'] * np.exp(-0.5 * (
        (along / (radius * 0.68)) ** 2 + (across / (radius * 0.42)) ** 2))
    color = np.asarray(params['color'], dtype=np.float32)
    field = scattering[..., None] * color
    if params['effect'] == 'lens' and params['ghost_strength'] > 0:
        # Positions are tied to one source and optical center. These three soft
        # pupils represent generic lens interfaces, not a claimed lens design.
        source_to_center = math.hypot((0.5 - sx) * width / short,
                                     (0.5 - sy) * height / short)
        for travel, pupil_radius, energy, tint, rotation in [
            (0.72, 0.065, 0.26, [1.00, 0.96, 0.88], 0.10),
            (1.30, 0.105, 0.17, [0.90, 1.00, 0.94], 0.32),
            (1.85, 0.160, 0.11, [1.00, 0.94, 0.86], 0.18),
        ]:
            ghost_radius = pupil_radius * params['ghost_scale']
            centered = along - travel * source_to_center
            for channel, shift in enumerate([-0.018, 0.0, 0.018]):
                pupil = _soft_aperture(centered - shift * ghost_radius,
                                       across, ghost_radius, rotation)
                field[..., channel] += params['ghost_strength'] * energy * tint[channel] * pupil
    return np.asarray(field, dtype=np.float32)


def apply_glare(linear: np.ndarray, preset: dict) -> np.ndarray:
    """Apply deterministic camera glare, returning unquantized, unclipped RGB."""
    validate_glare(preset)
    image = np.asarray(linear)
    if (image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1
            or not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image)):
        raise ValueError('Expected a nonempty real H x W x 3 image')
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError('Linear image must be finite and nonnegative')
    if np.any(image > np.finfo(np.float32).max / 64):
        raise ValueError('Linear image exceeds safe floating-point range')
    params = {**DEFAULTS, **preset}
    strength = float(params['strength'])
    result = image.astype(np.float32, copy=True)
    if strength == 0:
        return result
    result *= np.float32(2 ** (strength * params['ev']))
    height, width = result.shape[:2]
    scale = min(1, MAX_FIELD_EDGE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    field = _field(params, size)
    for channel in range(3):
        result[..., channel] += np.float32(strength) * _resize(field[..., channel], (width, height))
    if not np.isfinite(result).all():
        raise ValueError('Camera glare produced non-finite values')
    return result
