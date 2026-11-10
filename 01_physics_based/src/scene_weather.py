"""Non-generative, approximate relighting and thin-water-film road rendering.

This is a 2.5-D engineering model, not a calibrated inverse renderer. Vehicle
pixels are transformed only by a bounded positive RGB gain. Resampling is used
only for reflections on the road; it never changes the source vehicle.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from .physics_weather import polygon_mask, srgb_to_linear


def _blur(field: np.ndarray, radius: float) -> np.ndarray:
    # Pillow's GaussianBlur does not support float images. Quantization affects
    # only the low-frequency control field, never the original texture.
    encoded = Image.fromarray(np.rint(np.clip(field, 0, 1) * 255).astype(np.uint8))
    return np.asarray(encoded.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def _polygons(size: tuple[int, int], polygons: list, feather: float = 0) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), dtype=np.float32)
    for polygon in polygons:
        np.maximum(mask, polygon_mask(size, polygon, feather), out=mask)
    return mask


def water_fresnel(cos_theta: np.ndarray, ior: float = 1.333) -> np.ndarray:
    """Unpolarized air-to-water Fresnel reflectance, no Schlick fitting constant."""
    if not math.isfinite(ior) or ior <= 1:
        raise ValueError('Water refractive index must be finite and greater than 1')
    cosine = np.clip(cos_theta, 0, 1)
    transmitted = np.sqrt(1 - (1 - cosine * cosine) / (ior * ior))
    parallel = (ior * cosine - transmitted) / (ior * cosine + transmitted)
    perpendicular = (cosine - ior * transmitted) / (cosine + ior * transmitted)
    return (parallel * parallel + perpendicular * perpendicular) / 2


def _sample_rows(image: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Bilinear sampling at fixed columns. Only used for ROAD reflections."""
    height, width = image.shape[:2]
    y0 = np.floor(np.clip(rows, 0, height - 1)).astype(np.int32)
    y1 = np.minimum(y0 + 1, height - 1)
    fraction = np.clip(rows, 0, height - 1) - y0
    xx = np.arange(width)[None, :]
    if image.ndim == 3:
        fraction = fraction[..., None]
    return image[y0, xx] * (1 - fraction) + image[y1, xx] * fraction


def _noise(x: np.ndarray, z: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Continuous value noise in approximate ground-plane world coordinates."""
    x, z = x * scale, z * scale
    ix, iz = np.floor(x).astype(np.int32), np.floor(z).astype(np.int32)
    tx, tz = x - ix, z - iz
    tx, tz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)

    def lattice(dx, dz):
        value = (ix + dx).astype(np.uint32) * np.uint32(374761393)
        value ^= (iz + dz).astype(np.uint32) * np.uint32(668265263)
        value ^= np.uint32(seed % (2 ** 32))
        value = (value ^ (value >> 13)) * np.uint32(1274126177)
        value ^= value >> 16
        return value.astype(np.float32) / np.float32(4294967295)

    return (lattice(0, 0) * (1 - tx) + lattice(1, 0) * tx) * (1 - tz) + (
        lattice(0, 1) * (1 - tx) + lattice(1, 1) * tx) * tz


def scene_masks(size: tuple[int, int], scene: dict) -> dict:
    geometry = scene['weather_geometry']
    width, _ = size
    subject = polygon_mask(size, geometry.get('vehicle_outline', scene['subject_polygon']))
    # A small explicit guard includes possible manual contour errors. Outside
    # that guard, the road transition is feathered without leaking onto the car.
    guard_radius = max(1, round(width * 0.003))
    protected = np.asarray(Image.fromarray((subject * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(guard_radius * 2 + 1)), dtype=np.float32) / 255
    exclusion = _polygons(size, geometry.get('road_exclusions', []), 0)
    road = polygon_mask(size, geometry['road_polygon'], max(0.5, width * 0.002))
    road *= (1 - protected) * (1 - exclusion)
    # The hard core is protected exactly, while an exterior-only transition
    # prevents the approximate contour creating a polygonal dry-road rim.
    road *= _smoothstep(0.10, 0.90, 1 - _blur(protected, max(1, width * 0.008)))
    shadow = _polygons(size, geometry['sun_shadow_polygons'], max(0.5, width * 0.001))
    contact = _polygons(size, geometry['contact_shadow_polygons'], max(1, width * 0.012))
    sky = _polygons(size, geometry.get('sky_polygons', []), max(1, width * 0.006))
    sky *= 1 - protected
    return {'subject': subject, 'protected': protected, 'road': road,
            'shadow': shadow, 'contact': contact, 'sky': sky}


def relight_overcast(linear: np.ndarray, scene: dict, config: dict, masks: dict):
    """Estimate smooth illumination ratios; never smooth the output car RGB."""
    height, width = linear.shape[:2]
    geometry = scene['weather_geometry']
    exposure = float(config['exposure'])
    minimum, maximum = config['vehicle_gain_bounds']
    white_balance = np.asarray(config['white_balance'], dtype=np.float32)
    if not all(math.isfinite(float(v)) for v in [exposure, minimum, maximum,
            config['remaining_direct_fraction'], config['sky_strength']]):
        raise ValueError('Overcast parameters must be finite')
    if not 0 < minimum <= maximum <= 2 or not 0 < exposure <= 2:
        raise ValueError('Invalid exposure or vehicle gain bounds')
    if not 0 <= config['remaining_direct_fraction'] <= 1 or not 0 <= config['sky_strength'] <= 1:
        raise ValueError('Direct-light and sky blend fractions must be in [0, 1]')
    if white_balance.shape != (3,) or not np.isfinite(white_balance).all() or np.any(white_balance <= 0):
        raise ValueError('White balance must contain three finite positive gains')
    luminance = np.einsum('ijk,k->ij', linear, np.array([0.2126, 0.7152, 0.0722], np.float32))
    low_frequency = _blur(luminance, max(2, width * 0.025))
    # Positive, smooth illumination ratio reduces the prominence of brightly
    # sunlit surfaces while leaving high-frequency source texture in place.
    gain = exposure * (0.35 / (0.22 + low_frequency)) ** 0.14
    gain = np.clip(gain, 0.54, 1.10)

    road = masks['road']
    shadow = masks['shadow'] * road
    contact = masks['contact']
    x0, y0, x1, y1 = geometry['lit_road_reference']
    reference = luminance[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)]
    if reference.size == 0:
        raise ValueError('Empty lit road reference rectangle')
    lit_level = max(0.015, float(np.median(reference)))
    # The source ambient fraction is manually estimated. Luminance helps adapt
    # the coarse footprint; contact occlusion is retained to anchor the wheels.
    ambient = float(geometry['source_ambient_fraction'])
    if not 0.10 <= ambient <= 1:
        raise ValueError('Source ambient fraction must be in [0.10, 1]')
    local = _blur(luminance, max(0.5, width * 0.0007))
    attenuation = np.clip(local / lit_level, ambient, 1)
    target_direct = float(config['remaining_direct_fraction'])
    source_visibility = (attenuation - ambient) / (1 - ambient + 1e-6)
    target_illumination = 1 - target_direct * (1 - source_visibility)
    recovery = target_illumination / attenuation
    recovery = 1 + (recovery - 1) * shadow * (1 - 0.95 * contact)
    gain *= recovery

    # All pixels in the guarded vehicle area get only bounded multiplicative
    # RGB changes. No sky/road layer is allowed to contribute there.
    subject_gain = np.clip(gain[..., None] * white_balance, minimum, maximum)
    global_gain = gain[..., None] * white_balance
    protected = masks['protected'] >= 0.5
    effective_gain = np.where(protected[..., None], subject_gain, global_gain)
    result = linear * effective_gain

    sky_color = srgb_to_linear(np.asarray(config['sky_srgb'], dtype=np.float32))
    if sky_color.shape != (3,) or not np.isfinite(sky_color).all() or np.any((sky_color < 0) | (sky_color > 1)):
        raise ValueError('Sky color must contain three finite sRGB values in [0, 1]')
    sky_gradient = (0.88 + 0.20 * np.arange(height, dtype=np.float32)[:, None] / height)
    sky_rgb = sky_color[None, None, :] * sky_gradient[..., None]
    # Dark branches inside a coarse sky polygon keep their source appearance.
    sky_alpha = masks['sky'] * _smoothstep(0.12, 0.34, luminance) * float(config['sky_strength'])
    sky_alpha[protected] = 0
    result = result * (1 - sky_alpha[..., None]) + sky_rgb * sky_alpha[..., None]
    selected = masks['subject'] > 0.99
    car_gain = effective_gain[selected]
    stats = {
        'vehicle_rgb_gain_min': float(car_gain.min()),
        'vehicle_rgb_gain_max': float(car_gain.max()),
        'vehicle_stage_equation_max_error': float(np.max(np.abs(result[selected] - linear[selected] * car_gain))),
        'shadow_gain_max': float(gain[road > 0.5].max()) if np.any(road > 0.5) else 1.0,
        'source_lit_road_linear_luminance': lit_level,
    }
    return result.astype(np.float32), gain, sky_color, stats


def wet_road(linear: np.ndarray, scene: dict, config: dict, masks: dict, sky_color: np.ndarray, seed: int):
    """Thin film: absorbed diffuse return plus Fresnel-weighted rough reflection."""
    height, width = linear.shape[:2]
    road = masks['road']
    strength = float(config['wetness'])
    diffuse_retention = float(config['diffuse_retention'])
    reflection_strength = float(config['reflection_strength'])
    blur_ratio = float(config['reflection_blur_ratio'])
    if not all(math.isfinite(v) for v in [strength, diffuse_retention, reflection_strength, blur_ratio]) or blur_ratio <= 0:
        raise ValueError('Wet-road parameters must be finite, with positive blur ratio')
    if not 0 <= strength <= 1 or not 0 <= diffuse_retention <= 1 or not 0 <= reflection_strength <= 1:
        raise ValueError('Wetness, diffuse retention and reflection strength must be in [0, 1]')
    camera = scene['camera']
    focal = width / (2 * math.tan(math.radians(camera['horizontal_fov_degrees']) / 2))
    cx = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / focal
    geometry = scene['weather_geometry']
    (x0, y0), (x1, y1) = geometry['reflection_contact_line']
    if x0 == x1:
        raise ValueError('Reflection contacts must span different columns')
    normalized_x = np.arange(width, dtype=np.float32) / max(1, width - 1)
    contact_line = (y0 + (normalized_x - x0) * (y1 - y0) / (x1 - x0)) * height

    # Premultiplied, blurred source texture produces a rough 2.5-D reflection.
    # This uses visible car surfaces as a proxy; unseen mirror-view surfaces are
    # not reconstructed, hence reflection is deliberately rough and restrained.
    reflection_mask = masks['subject']
    blur_radius = max(1, width * blur_ratio)
    smooth_alpha = _blur(reflection_mask, blur_radius)
    reflected_channels, background_channels = [], []
    for channel in range(3):
        reflected_channels.append(_blur(np.clip(linear[..., channel], 0, 1) * reflection_mask, blur_radius))
        background_channels.append(_blur(np.clip(linear[..., channel], 0, 1), blur_radius * 1.8))
    premultiplied = np.stack(reflected_channels, axis=2)
    background = np.stack(background_channels, axis=2)
    result = linear.copy()
    fresnel_map = np.zeros((height, width), dtype=np.float32)
    wet_map = np.zeros((height, width), dtype=np.float32)
    reflection_map = np.zeros((height, width), dtype=np.float32)
    for first in range(0, height, 192):
        last = min(height, first + 192)
        if not np.any(road[first:last]):
            continue
        rows = np.arange(first, last, dtype=np.float32)[:, None]
        ry = (rows + 0.5 - height * camera['horizon_y']) / focal
        distance_z = camera['height_m'] / np.maximum(ry, 0.015)
        world_x = distance_z * cx[None, :]
        noise = 0.65 * _noise(world_x, distance_z, 0.85, seed) + 0.35 * _noise(world_x, distance_z, 3.5, seed + 91)
        film = np.clip(strength * (0.80 + 0.28 * noise), 0, 1)
        pooled = _smoothstep(0.38, 0.68, noise)
        cosine = np.maximum(ry, 0) / np.sqrt(1 + cx[None, :] ** 2 + ry ** 2)
        fresnel = water_fresnel(cosine, float(config['water_ior']))
        # Rough aggregate and incomplete water coverage reduce coherent specular
        # reflection; all coefficients remain energy bounded.
        specular = fresnel * reflection_strength
        source_rows = 2 * contact_line[None, :] - rows
        alpha = _sample_rows(smooth_alpha, source_rows)
        rgb_premultiplied = _sample_rows(premultiplied, source_rows)
        valid = (source_rows >= 0) & (source_rows <= height - 1) & (rows >= contact_line[None, :])
        alpha *= valid
        rgb_premultiplied *= valid[..., None]
        # Project a coarse vertical background card through a mirror ground
        # plane. This uses the same camera and assumed background distance.
        background_ground_row = height * camera['horizon_y'] + focal * camera['height_m'] / scene['background_depth_m']
        background_rows = np.broadcast_to(2 * background_ground_row - rows, source_rows.shape)
        bg_rgb = _sample_rows(background, background_rows)
        bg_valid = (background_rows >= 0) & (background_rows <= height - 1)
        bg_rgb = np.where(bg_valid[..., None], bg_rgb, sky_color)
        environment_coherent = bg_rgb * (1 - alpha[..., None]) + rgb_premultiplied
        coherence = 0.30 + 0.70 * pooled
        environment = sky_color * (1 - coherence[..., None]) + environment_coherent * coherence[..., None]
        environment *= (1 - 0.70 * masks['contact'][first:last, :, None])
        diffuse = linear[first:last] * diffuse_retention * (1 - fresnel[..., None]) ** 2
        water = diffuse + specular[..., None] * environment
        mix = road[first:last] * film
        result[first:last] = linear[first:last] * (1 - mix[..., None]) + water * mix[..., None]
        wet_map[first:last] = mix
        fresnel_map[first:last] = fresnel * road[first:last]
        reflection_map[first:last] = alpha * specular * mix
    protected = masks['protected'] >= 0.5
    stats = {
        'wet_layer_max_change_on_vehicle_guard': float(np.max(np.abs(result[protected] - linear[protected]))),
        'wet_road_image_fraction': float(np.mean(wet_map > 0.1)),
        'wet_road_mean_fresnel': float(np.mean(fresnel_map[road > 0.5])),
        'road_reflection_model': 'rough visible-surface 2.5D proxy, no unseen geometry recovery',
    }
    return result, wet_map, fresnel_map, reflection_map, stats


def apply_scene_weather(linear: np.ndarray, scene: dict, config: dict, seed: int, include_maps: bool = False):
    size = (linear.shape[1], linear.shape[0])
    masks = scene_masks(size, scene)
    relit, gain, sky_color, lighting_stats = relight_overcast(linear, scene, config['overcast'], masks)
    result, wetness, fresnel, reflection, surface_stats = wet_road(
        relit, scene, config['wet_road'], masks, sky_color, seed)
    stats = {'scene_lighting': lighting_stats, 'wet_surface': surface_stats}
    if include_maps:
        maps = {'illumination_gain_div_3': np.clip(gain / 3, 0, 1),
                'road_mask': masks['road'], 'water_coverage': wetness,
                'fresnel_reflectance': fresnel, 'car_reflection_weight': reflection}
        previews = {}
        for name, field in maps.items():
            preview = Image.fromarray(np.rint(np.clip(field, 0, 1) * 255).astype(np.uint8))
            preview.thumbnail((640, 480))
            previews[name] = preview.convert('RGB')
        stats['_scene_maps'] = previews
    return result, stats
