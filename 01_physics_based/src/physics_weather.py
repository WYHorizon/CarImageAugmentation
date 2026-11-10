"""Deterministic, non-generative weather formation in linear RGB.

Scene distances are hand-authored approximations, not measured depth. Original
vehicle coordinates and texture remain fixed; road-only reflections reuse source
samples in the optional scene-surface module.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def to_image(linear: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(linear_to_srgb(linear) * 255).astype(np.uint8))


def polygon_mask(size: tuple[int, int], polygon: list, feather: float = 0) -> np.ndarray:
    width, height = size
    if len(polygon) < 3 or any(len(p) != 2 or not all(0 <= v <= 1 for v in p) for p in polygon):
        raise ValueError('Polygons need at least three normalized [x, y] coordinates')
    mask = Image.new('L', size)
    ImageDraw.Draw(mask).polygon(
        [(round(x * (width - 1)), round(y * (height - 1))) for x, y in polygon], fill=255
    )
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(mask, dtype=np.float32) / 255


def build_depth(size: tuple[int, int], scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """Approximate camera-space Z using background layers, a plane, and a car silhouette."""
    width, height = size
    camera = scene['camera']
    fov = camera['horizontal_fov_degrees']
    if not 10 < fov < 150 or camera['height_m'] <= 0:
        raise ValueError('Invalid camera FOV or height')
    if not 0 <= camera['horizon_y'] < scene['ground_start_y'] <= 1:
        raise ValueError('Ground must start below the horizon')
    far = float(scene['background_depth_m'])
    if not math.isfinite(far) or far <= 0:
        raise ValueError('Background depth must be finite and positive')
    depth = np.full((height, width), far, dtype=np.float32)
    feather = max(0.5, width / 2500)
    for layer in scene.get('layers', []):
        z = float(layer['depth_m'])
        if not math.isfinite(z) or z <= 0:
            raise ValueError('Layer depth must be finite and positive')
        # Coarse background polygons have no reliable fine boundaries. Broad
        # transitions prevent their artificial edges becoming training cues.
        alpha = polygon_mask(size, layer['polygon'], max(feather, width * 0.018))
        depth *= 1 - alpha
        depth += alpha * z

    focal = width / (2 * math.tan(math.radians(fov) / 2))
    rows = np.arange(height, dtype=np.float32) + 0.5
    plane = camera['height_m'] * focal / np.maximum(rows - camera['horizon_y'] * height, 1)
    plane = np.clip(plane, 0.5, far)
    # A small transition avoids a full-width seam in the coarse scene geometry.
    mix = np.clip((rows / height - scene['ground_start_y']) / 0.015, 0, 1)
    depth *= 1 - mix[:, None]
    depth += plane[:, None] * mix[:, None]

    subject = polygon_mask(size, scene['subject_polygon'], feather)
    x0 = min(p[0] for p in scene['subject_polygon'])
    x1 = max(p[0] for p in scene['subject_polygon'])
    z0, z1 = scene['subject_depth_m']
    if not all(math.isfinite(z) and z > 0 for z in [z0, z1]) or x0 == x1:
        raise ValueError('Invalid subject depth or silhouette')
    fraction = np.clip((np.arange(width, dtype=np.float32) / max(width - 1, 1) - x0) / (x1 - x0), 0, 1)
    subject_z = z0 + fraction * (z1 - z0)
    depth *= 1 - subject
    depth += subject * subject_z[None, :]
    return depth, subject


def fog(linear: np.ndarray, depth: np.ndarray, beta: float, airlight: np.ndarray):
    if not math.isfinite(beta) or beta < 0 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('Fog requires finite positive depth and non-negative beta')
    # Use path length along each ray; callers supply ray distances, not inverse depth.
    transmittance = np.exp(-beta * depth).astype(np.float32)
    result = linear * transmittance[..., None]
    result += np.asarray(airlight, dtype=np.float32) * (1 - transmittance[..., None])
    return result, transmittance


def ray_distance(depth_z: np.ndarray, scene: dict) -> np.ndarray:
    height, width = depth_z.shape
    camera = scene['camera']
    focal = width / (2 * math.tan(math.radians(camera['horizontal_fov_degrees']) / 2))
    x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / focal
    y = (np.arange(height, dtype=np.float32) + 0.5 - height * camera['horizon_y']) / focal
    return depth_z * np.sqrt(1 + y[:, None] ** 2 + x[None, :] ** 2)


def rain_transmittance(depth_z: np.ndarray, scene: dict, preset: dict, rng: np.random.Generator):
    """Sample particles in a pinhole-camera frustum, occlude, then integrate streaks.

    Stroke length follows shutter time and projected velocity. Opacity approximates
    exposure occupancy; Gaussian cross sections approximate optics. This is not a
    calibrated raindrop refraction renderer or a measured rainfall-rate simulation.
    """
    height, width = depth_z.shape
    camera = scene['camera']
    focal = width / (2 * math.tan(math.radians(camera['horizontal_fov_degrees']) / 2))
    density = float(preset['particle_density_per_m3'])
    shutter = float(preset['shutter_seconds'])
    speed = float(preset['fall_speed_mps'])
    wind = float(preset['wind_mps'])
    wind_jitter = float(preset.get('wind_jitter_mps', 0))
    max_depth = float(preset.get('max_rain_depth_m', 40))
    d0, d1 = preset['drop_diameter_mm']
    opacity = float(preset['opacity_scale'])
    if not all(math.isfinite(v) for v in [density, shutter, speed, wind, wind_jitter, max_depth, d0, d1, opacity]):
        raise ValueError('Rain parameters must be finite')
    if density < 0 or shutter <= 0 or speed <= 0 or not 0 < d0 <= d1 or opacity < 0:
        raise ValueError('Invalid rain parameters')
    if wind_jitter < 0 or not 1.2 < max_depth <= 40:
        raise ValueError('Rain wind jitter must be non-negative; max depth must be in (1.2, 40]')
    radiance = np.asarray(preset['radiance_linear'], dtype=np.float32)
    if radiance.shape != (3,) or not np.isfinite(radiance).all() or np.any(radiance < 0):
        raise ValueError('Rain radiance must be a finite non-negative RGB vector (HDR allowed)')
    transmittance = np.ones((height, width), dtype=np.float32)
    sampled, visible = 0, 0
    # Distant unresolved rain is approximated by the scattering term. Restricting
    # explicit particles avoids spending the particle budget on subpixel drops.
    edges = [z for z in [1.2, 2.5, 5, 9, 16, 26, 40] if z < max_depth] + [max_depth]
    for near, far in zip(edges[:-1], edges[1:]):
        volume = (width / focal) * (height / focal) * (far ** 3 - near ** 3) / 3
        pad = focal * math.hypot(speed, abs(wind) + 4 * wind_jitter) * shutter / near
        volume *= (width + 2 * pad) * (height + 2 * pad) / (width * height)
        count = int(rng.poisson(density * volume))
        if sampled + count > 200_000:
            raise ValueError('Too many rain particles; lower particle density')
        sampled += count
        # z**2-weighted sampling is uniform in 3-D frustum volume.
        zs = (near ** 3 + rng.random(count) * (far ** 3 - near ** 3)) ** (1 / 3)
        # Uniform positions refer to the midpoint of exposure; padding includes
        # particles whose streaks enter the image during exposure.
        us = rng.uniform(-pad, width + pad, count)
        vs = rng.uniform(-pad, height + pad, count)
        diameters = rng.uniform(d0, d1, count) / 1000
        winds = rng.normal(wind, wind_jitter, count) if wind_jitter else np.full(count, wind)
        for u, v, z, diameter, particle_wind in zip(us, vs, zs, diameters, winds):
            du, dv = focal * particle_wind * shutter / z, focal * speed * shutter / z
            projected_diameter = focal * diameter / z
            # Near drops have a broader approximate defocus PSF.
            defocus = width / 3840 * max(0, 2.0 / z - 0.15)
            sigma = math.sqrt((projected_diameter / 2.355) ** 2 + defocus ** 2 + 0.45 ** 2)
            ax, ay = u - du / 2, v - dv / 2
            bx, by = u + du / 2, v + dv / 2
            radius = 3 * sigma
            x0 = max(0, math.floor(min(ax, bx) - radius))
            x1 = min(width, math.ceil(max(ax, bx) + radius) + 1)
            y0 = max(0, math.floor(min(ay, by) - radius))
            y1 = min(height, math.ceil(max(ay, by) + radius) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            scene_depth = depth_z[y0:y1, x0:x1]
            if not np.any(scene_depth > z):
                continue
            xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
            yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
            along = np.clip(((xx - ax) * du + (yy - ay) * dv) / (du * du + dv * dv), 0, 1)
            distance2 = (xx - ax - along * du) ** 2 + (yy - ay - along * dv) ** 2
            # Occupancy and pixel footprint limit thin subpixel streak opacity.
            occupancy = diameter / (math.hypot(speed, particle_wind) * shutter + diameter)
            peak = min(0.25, opacity * occupancy * min(1, projected_diameter / (2.355 * sigma)))
            alpha = peak * np.exp(-distance2 / (2 * sigma * sigma))
            alpha *= scene_depth > z
            transmittance[y0:y1, x0:x1] *= 1 - alpha
            visible += 1
    return transmittance, {
        'particles_sampled': sampled, 'particles_visible': visible,
        'explicit_rain_max_depth_m': max_depth,
    }


def render(linear: np.ndarray, depth_z: np.ndarray, scene: dict, preset: dict, seed: int,
           include_scene_maps: bool = False):
    kind = preset['kind']
    if kind not in {'fog', 'rain'}:
        raise ValueError(f'Unsupported weather: {kind}')
    stats = {}
    atmospheric_source = linear
    if 'surface_weather' in preset:
        from .scene_weather import apply_scene_weather
        atmospheric_source, stats = apply_scene_weather(
            linear, scene, preset['surface_weather'], seed, include_scene_maps)
    air_srgb = np.asarray(preset.get('airlight_srgb', scene['airlight_srgb']), dtype=np.float32)
    if air_srgb.shape != (3,) or not np.isfinite(air_srgb).all() or np.any((air_srgb < 0) | (air_srgb > 1)):
        raise ValueError('Airlight must be an RGB vector in [0, 1]')
    result, fog_t = fog(atmospheric_source, ray_distance(depth_z, scene), float(preset['beta_per_m']), srgb_to_linear(air_srgb))
    rain_t = np.ones(depth_z.shape, dtype=np.float32)
    if kind == 'rain':
        rain_t, particle_stats = rain_transmittance(depth_z, scene, preset, np.random.default_rng(seed))
        stats.update(particle_stats)
        color = np.asarray(preset['radiance_linear'], dtype=np.float32)
        result *= rain_t[..., None]
        result += color * (1 - rain_t[..., None])
    return result, fog_t, rain_t, stats


def region_diagnostics(fog_t: np.ndarray, rain_t: np.ndarray, scene: dict, subject: np.ndarray) -> dict:
    height, width = fog_t.shape
    report = {}
    for name, box in scene['regions'].items():
        if len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise ValueError(f'Invalid region: {name}')
        x0, y0, x1, y1 = box
        crop = (slice(int(y0 * height), max(int(y0 * height) + 1, int(y1 * height))),
                slice(int(x0 * width), max(int(x0 * width) + 1, int(x1 * width))))
        selected = subject[crop] >= 0.99
        if not np.any(selected):
            raise ValueError(f'Region does not intersect the vehicle silhouette: {name}')
        t = fog_t[crop][selected]
        alpha = 1 - rain_t[crop][selected]
        report[name] = {
            'measurement_scope': 'intersection_with_vehicle_silhouette',
            'subject_pixel_fraction_in_box': float(selected.mean()),
            'fog_transmittance_min': float(t.min()),
            'fog_transmittance_mean': float(t.mean()),
            'rain_alpha_mean': float(alpha.mean()),
            'rain_alpha_max': float(alpha.max()),
            'rain_alpha_gt_0_10_fraction': float(np.mean(alpha > 0.10)),
        }
    return report
