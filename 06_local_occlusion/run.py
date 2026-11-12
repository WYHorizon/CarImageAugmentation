#!/usr/bin/env python3
"""Composite an RGBA foreground onto a vehicle image and report its visibility."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from occlusion.core import compose_occlusion, compute_visibility
from occlusion.report import build_report


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
TRANSFORM_KEYS = {
    'width_fraction', 'position_xy', 'angle_degrees', 'flip_horizontal',
    'blur_short_edge_fraction', 'brightness', 'saturation',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_input(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert('RGB')


def load_asset(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        if 'A' not in oriented.getbands() and 'transparency' not in oriented.info:
            raise ValueError('Material has no alpha channel. Prepare the cutout first; an RGB checkerboard is not transparency.')
        return oriented.convert('RGBA')


def validate_scene(scene: dict) -> None:
    regions = scene.get('regions', {})
    if not isinstance(regions, dict):
        raise ValueError('Scene regions must be a mapping.')
    import re
    for name in regions:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Unsafe comparison region name: {name!r}')


def load_configuration(preset_path: Path, scene_path: Path | None, source: Path) -> tuple[dict, dict]:
    preset = json.loads(preset_path.read_text(encoding='utf-8'))
    if not isinstance(preset, dict) or not isinstance(preset.get('transform'), dict):
        raise ValueError('Preset requires a transform mapping.')
    unknown = preset['transform'].keys() - TRANSFORM_KEYS
    if unknown:
        raise ValueError(f'Unknown transform settings: {sorted(unknown)}')
    transform = preset['transform']
    if not {'width_fraction', 'position_xy'} <= transform.keys():
        raise ValueError('transform requires width_fraction and position_xy.')
    blur = transform.get('blur_short_edge_fraction', 0)
    if isinstance(blur, bool) or not isinstance(blur, (int, float)) or not math.isfinite(blur) or not 0 <= blur <= 0.03:
        raise ValueError('blur_short_edge_fraction must be in [0, 0.03].')
    scenes = json.loads(scene_path.read_text(encoding='utf-8')) if scene_path else {}
    if not isinstance(scenes, dict):
        raise ValueError('Scene file must map image filenames to scene records.')
    scene = scenes.get(source.name, {'label': source.stem, 'regions': {'full_image': [0, 0, 1, 1]}})
    if not isinstance(scene, dict):
        raise ValueError('Scene record must be a mapping.')
    scene = dict(scene)
    if 'regions' not in scene or scene['regions'] == {}:
        scene['regions'] = {'full_image': [0, 0, 1, 1]}
    validate_scene(scene)
    targets = preset.get('review_targets', {})
    if not isinstance(targets, dict):
        raise ValueError('review_targets must be a mapping.')
    band = targets.get('vehicle_opaque_fraction', [0, 1])
    if (not isinstance(band, list) or len(band) != 2
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in band)
            or not 0 <= band[0] <= band[1] <= 1):
        raise ValueError('vehicle_opaque_fraction requires an ordered [0, 1] range.')
    count = targets.get('minimum_visible_regions', 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('minimum_visible_regions must be a nonnegative integer.')
    visible = targets.get('minimum_region_visible_fraction', 0.7)
    if not isinstance(visible, (int, float)) or isinstance(visible, bool) or not math.isfinite(visible) or not 0 <= visible <= 1:
        raise ValueError('minimum_region_visible_fraction must be in [0, 1].')
    return preset, scene


def review_targets(statistics: dict, preset: dict) -> dict:
    """Descriptive checks, never a claim of training quality or automatic placement."""
    target = preset.get('review_targets', {})
    vehicle = statistics['vehicle']
    band = target.get('vehicle_opaque_fraction', [0, 1])
    vehicle_ok = band[0] <= vehicle['opaque_fraction'] <= band[1] if vehicle else None
    minimum = target.get('minimum_region_visible_fraction', 0.7)
    retained = [name for name, region in statistics['regions'].items()
                if name != 'full_image' and region['visible_fraction'] >= minimum]
    expected = target.get('minimum_visible_regions', 0)
    part_ok = len(retained) >= expected if any(name != 'full_image' for name in statistics['regions']) else None
    return {
        'vehicle_coverage_in_target_range': vehicle_ok,
        'vehicle_target_range': band,
        'region_visibility_target': minimum,
        'regions_meeting_visibility_target': retained,
        'minimum_region_count_met': part_ok,
        'note': 'Review flags only; missing annotations are not a pass. Presets do not automatically search or guarantee these targets on new photos.',
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', type=Path, default=PROJECT / 'test_images/奥迪A6L.jpg')
    result.add_argument('--asset', type=Path, default=METHOD / 'assets/branches_user/rgba.png')
    result.add_argument('--preset', type=Path, default=METHOD / 'configs/branches_demo.json')
    result.add_argument('--scenes', type=Path, default=PROJECT / '01_physics_based/configs/scenes.json')
    result.add_argument('--output-dir', type=Path, default=METHOD / 'outputs/branches_demo_v1')
    result.add_argument('--check-config', action='store_true')
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    source, asset_path, output = args.input.resolve(), args.asset.resolve(), args.output_dir.resolve()
    preset_path, scene_path = args.preset.resolve(), args.scenes.resolve() if args.scenes else None
    inputs = [source, asset_path, preset_path] + ([scene_path] if scene_path else [])
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.is_relative_to((PROJECT / 'test_images').resolve()) or output.is_relative_to((PROJECT / 'models').resolve()):
        raise ValueError('Output must be outside test_images and models.')
    if any(path.is_relative_to(output) for path in inputs):
        raise ValueError('Output may not contain or replace an input file.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output must be a new or empty directory: {output}')
    hashes = {str(path): sha256(path) for path in inputs}
    preset, scene = load_configuration(preset_path, scene_path, source)
    base, asset = load_input(source), load_asset(asset_path)
    # Geometry is checked before creating the run directory. No unannotated image
    # is assigned invented vehicle or part coverage.
    compute_visibility(Image.new('L', base.size, 0), scene)
    transform = dict(preset['transform'])
    transform['blur_radius'] = transform.pop('blur_short_edge_fraction', 0) * min(base.size)
    result = compose_occlusion(base, asset, **transform)
    if not result.alpha.getbbox():
        raise ValueError('The material is entirely outside the output; adjust position_xy.')
    visibility = compute_visibility(result.alpha, scene)
    if args.check_config:
        checked = {'status': 'configuration_valid', 'size': base.size, 'statistics': visibility.statistics,
                   'review': review_targets(visibility.statistics, preset), 'writes': 0}
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return checked

    for relative in ['inputs', 'images', 'masks', 'config_snapshot', 'code_snapshot/occlusion']:
        (output / relative).mkdir(parents=True, exist_ok=True)
    base.save(output / 'inputs/original.png')
    asset.save(output / 'inputs/material.png')
    result.image.save(output / 'images/occluded.png')
    result.alpha.save(output / 'masks/alpha.png')
    opaque = result.alpha.point(lambda a: 255 if a >= 128 else 0)
    opaque.save(output / 'masks/opaque.png')
    if visibility.vehicle_mask is not None:
        visibility.vehicle_mask.save(output / 'masks/vehicle.png')
        visibility.vehicle_visible_mask.save(output / 'masks/vehicle_visible.png')
    save_json(output / 'config_snapshot/preset.json', preset)
    save_json(output / 'config_snapshot/scene.json', scene)
    code_files = [METHOD / 'run.py'] + sorted((METHOD / 'occlusion').glob('*.py'))
    code_hashes = {str(path.relative_to(METHOD)): sha256(path) for path in code_files}
    for path in code_files:
        shutil.copy2(path, output / 'code_snapshot' / path.relative_to(METHOD))

    artifacts = build_report(output, output / 'inputs/original.png', output / 'images/occluded.png',
                             output / 'masks/alpha.png', scene, visibility.statistics)
    alpha_values = np.asarray(result.alpha)
    outside_equal = bool(np.array_equal(np.asarray(base)[alpha_values == 0], np.asarray(result.image)[alpha_values == 0]))
    if not outside_equal:
        raise RuntimeError('Pixels outside foreground alpha support changed.')
    unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    if not unchanged:
        raise RuntimeError('Input file was modified during execution.')
    png_files = sorted(output.rglob('*.png'))
    for path in png_files:
        with Image.open(path) as opened:
            opened.verify()
    verification = {
        'input_files_unchanged': unchanged,
        'outside_alpha_support_pixel_identical': outside_equal,
        'output_size_matches_source': result.image.size == base.size,
        'png_files_checked': len(png_files),
        'source_size': list(base.size),
        'affected_pixels': int(np.count_nonzero(alpha_values)),
        'opaque_pixels': int(np.count_nonzero(alpha_values >= 128)),
    }
    save_json(output / 'verification.json', verification)
    manifest = {
        'method': 'local_occlusion', 'status': 'complete_pending_visual_review',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version,
        'source': str(source), 'material': str(asset_path), 'input_sha256': hashes,
        'implementation_sha256': code_hashes, 'preset': preset, 'transform': result.metadata,
        'statistics': visibility.statistics, 'review': review_targets(visibility.statistics, preset),
        'verification': verification, 'report': artifacts,
        'output_sha256': {str(p.relative_to(output)): sha256(p) for p in png_files},
        'total_seconds': round(time.perf_counter() - started, 3),
        'scope': 'Single foreground layer; fixed configurable placement; no new vehicle geometry, model inference, physical relighting, or added shadows.',
    }
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'statistics': visibility.statistics,
                      'verification': verification, 'seconds': manifest['total_seconds']}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    run(parser().parse_args())
