"""Photometric and optical-effect transforms in approximate linear sRGB.

JPEG inputs already contain camera processing, so these operations approximate
exposure, tone-curve and white-balance changes; they do not reconstruct lighting,
geometry or material reflectance. Results remain unclipped until output encoding
so callers can measure newly out-of-range channels before information is lost.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _rgb_array(value, *, nonnegative: bool = True, unit_range: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError('Expected a nonempty H x W x 3 RGB image')
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError('RGB values must be real numbers')
    if not np.isfinite(array).all():
        raise ValueError('RGB values must be finite')
    if nonnegative and np.any(array < 0):
        raise ValueError('Linear RGB values must be nonnegative')
    if unit_range and np.any(array > 1):
        raise ValueError('sRGB input must be normalized to [0, 1]')
    if np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError('RGB values exceed the float32 range')
    return array.astype(np.float32, copy=False)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


def validate_preset(preset: dict) -> None:
    """Validate a JSON preset before an experiment creates any output files.

    Exposure is limited to +/-8 EV and contrast exponents to [0.25, 4] to reject
    extreme/accidental values. The shipped presets are much milder than these
    numerical limits. They are not acceptance thresholds for training images.
    """
    if not isinstance(preset, dict):
        raise ValueError('Preset must be a dictionary')
    if not isinstance(preset.get('label'), str) or not preset['label'].strip():
        raise ValueError('Preset must have a nonempty label')
    kind = preset.get('kind')
    if kind == 'exposure':
        ev = _number(preset.get('ev'), 'ev')
        if not -8 <= ev <= 8:
            raise ValueError('ev must be in [-8, 8]')
    elif kind == 'contrast':
        factor = _number(preset.get('factor'), 'factor')
        pivot = _number(preset.get('pivot', 0.18), 'pivot')
        if not 0.25 <= factor <= 4:
            raise ValueError('contrast factor must be in [0.25, 4]')
        if not 0 < pivot <= 1:
            raise ValueError('contrast pivot must be in (0, 1]')
    elif kind == 'white_balance':
        gains = preset.get('gains')
        if not isinstance(gains, (list, tuple, np.ndarray)) or len(gains) != 3:
            raise ValueError('white-balance gains must contain three numbers')
        values = [_number(value, 'gain') for value in gains]
        if any(value <= 0 or value > np.finfo(np.float32).max / 4 for value in values):
            raise ValueError('white-balance gains must be positive and safely representable')
    elif kind == 'optical':
        from .effects import validate_effect
        validate_effect(preset)
    elif kind == 'camera_glare':
        from .glare import validate_glare
        validate_glare(preset)
    else:
        raise ValueError(f'Unknown lighting kind: {kind!r}')


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode a normalized sRGB H x W x 3 image to float32 linear RGB."""
    rgb = _rgb_array(rgb, unit_range=True)
    return np.where(rgb <= 0.04045, rgb / 12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Clip once to the display range and encode to normalized float32 sRGB."""
    linear = _rgb_array(linear, nonnegative=False)
    clipped = np.clip(linear, 0, 1)
    return np.where(clipped <= 0.0031308, clipped * 12.92,
                    1.055 * clipped ** (1 / 2.4) - 0.055).astype(np.float32)


def apply_lighting(linear: np.ndarray, preset: dict) -> np.ndarray:
    """Return unclipped float32 RGB without mutating or warping the input.

    Contrast uses Y' = pivot * (Y / pivot)**factor and scales all three channels
    by the same per-pixel ratio. This preserves chromaticity before clipping.
    White-balance gains are normalized to preserve the luminance of neutral gray,
    not the average luminance of every possible colored scene.
    """
    validate_preset(preset)
    linear = _rgb_array(linear)
    kind = preset['kind']
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            if kind == 'exposure':
                result = linear * np.float32(2 ** float(preset['ev']))
            elif kind == 'contrast':
                factor = float(preset['factor'])
                if factor == 1:
                    return linear.copy()
                pivot = float(preset.get('pivot', 0.18))
                luminance = np.einsum('...c,c->...', linear, LUMA_WEIGHTS)
                # Compute only positive pixels: zero must not enter a negative
                # power when the contrast exponent is below one.
                positive = luminance > 0
                gain = np.zeros_like(luminance)
                gain[positive] = (luminance[positive] / pivot) ** (factor - 1)
                result = linear * gain[..., None]
            elif kind == 'white_balance':
                # Normalize in float64 to avoid underflow for valid small gains.
                gains = np.asarray(preset['gains'], dtype=np.float64)
                gains /= np.dot(gains, LUMA_WEIGHTS.astype(np.float64))
                result = linear * gains.astype(np.float32)
            elif kind == 'optical':
                from .effects import apply_effect
                result = apply_effect(linear, preset)
            else:
                from .glare import apply_glare
                result = apply_glare(linear, preset)
        except FloatingPointError as error:
            raise ValueError('Lighting transform exceeded finite numerical range') from error
    if not np.isfinite(result).all():
        raise ValueError('Lighting transform produced non-finite RGB')
    return result.astype(np.float32, copy=False)
