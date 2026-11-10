#!/usr/bin/env python3
"""Generate full-resolution basic lighting or strong optical-effect experiments."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from lighting.transforms import apply_lighting, linear_to_srgb, srgb_to_linear, validate_preset
from lighting.report import build_report, report_groups


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
VEHICLE_IDS = {'Audi A6L': 'audi', 'BMW 4 Series': 'bmw', 'Xiaomi YU7': 'xiaomi', 'Honda CR-V': 'honda'}
PROFILE_FILES = {'basic': 'presets.json', 'strong': 'strong_presets.json', 'refined': 'refined_presets.json'}


def resolve_preset(preset: dict, source_name: str) -> dict:
    """Resolve explicitly configured per-image parameters; never infer sunlight."""
    resolved = copy.deepcopy(preset)
    overrides = resolved.pop('source_overrides', {})
    resolved.update(overrides.get(source_name, {}))
    return resolved


def validate_preset_template(preset: dict) -> None:
    if not isinstance(preset, dict):
        raise ValueError('Preset must be a dictionary')
    base = {key: value for key, value in preset.items() if key != 'source_overrides'}
    validate_preset(base)
    if 'source_overrides' not in preset:
        return
    overrides = preset['source_overrides']
    if preset.get('kind') != 'camera_glare' or not isinstance(overrides, dict):
        raise ValueError('source_overrides must be a mapping for camera_glare presets')
    for filename, parameters in overrides.items():
        if not isinstance(filename, str) or not filename or filename in {'.', '..'} or '/' in filename or '\\' in filename:
            raise ValueError('Override keys must be source filenames without directories')
        if not isinstance(parameters, dict) or not parameters or set(parameters) - {'source', 'strength', 'ghost_strength'}:
            raise ValueError('Per-source overrides may set source, strength and ghost_strength only')
        validate_preset({**base, **parameters})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert('RGB')


def crop_box(box: list, size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(box) != 4 or not all(np.isfinite(box)) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'Invalid normalized region: {box}')
    width, height = size
    result = (math.floor(box[0] * width), math.floor(box[1] * height), math.ceil(box[2] * width), math.ceil(box[3] * height))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f'Region has no pixels at {size}: {box}')
    return result


def metrics(before: np.ndarray, after: np.ndarray, raw: np.ndarray) -> dict:
    """Diagnostic fractions, not automatic semantic or identity acceptance scores."""
    source_luma = np.sum(before.astype(np.float32) * LUMA, axis=-1) / 255
    output_luma = np.sum(after.astype(np.float32) * LUMA, axis=-1) / 255
    new_clipped = (raw > 1) & (before < 255)
    code_delta = np.abs(after.astype(np.float32) - before)
    source_std, output_std = float(source_luma.std()), float(output_luma.std())
    report = {
        'pixel_count': int(source_luma.size),
        'preclip_above_one_pixel_fraction': float(np.mean(np.any(raw > 1, axis=-1))),
        'preclip_below_zero_pixel_fraction': float(np.mean(np.any(raw < 0, axis=-1))),
        'new_clipped_pixel_fraction': float(np.mean(np.any(new_clipped, axis=-1))),
        'new_clipped_channel_fraction': float(np.mean(new_clipped)),
        'source_saturated_pixel_fraction': float(np.mean(np.any(before == 255, axis=-1))),
        'output_saturated_pixel_fraction': float(np.mean(np.any(after == 255, axis=-1))),
        'new_saturated_pixel_fraction': float(np.mean(np.any((after == 255) & (before < 255), axis=-1))),
        'source_near_black_fraction': float(np.mean(source_luma <= 5 / 255)),
        'output_near_black_fraction': float(np.mean(output_luma <= 5 / 255)),
        'new_near_black_fraction': float(np.mean((output_luma <= 5 / 255) & (source_luma > 5 / 255))),
        'source_display_luma_mean': float(source_luma.mean()),
        'output_display_luma_mean': float(output_luma.mean()),
        'source_display_luma_std': source_std,
        'output_display_luma_std': output_std,
        'display_luma_std_ratio': output_std / source_std if source_std > 1e-8 else None,
        'mean_absolute_code_value_change': float(np.mean(code_delta)),
        'changed_pixel_fraction_gt20': float(np.mean(np.mean(code_delta, axis=-1) > 20)),
        'new_near_white_fraction': float(np.mean(np.all(after >= 250, axis=-1) & ~np.all(before >= 250, axis=-1))),
    }
    if source_luma.ndim == 2 and min(source_luma.shape) > 1:
        def edge_energy(image):
            return float((np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2)
        original_edges, edited_edges = edge_energy(source_luma), edge_energy(output_luma)
        report['display_luma_gradient_ratio'] = edited_edges / original_edges if original_edges > 1e-8 else None
    return report


def inspect_input(source: Path, scenes: dict) -> dict:
    if source.name not in scenes:
        raise ValueError(f'Add a scene entry with label and regions for comparison reports: {source.name}')
    scene = scenes[source.name]
    if not scene.get('regions'):
        raise ValueError(f'No comparison regions configured: {source.name}')
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        size = image.size
    for name, box in scene['regions'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Unsafe region ID: {name}')
        crop_box(box, size)
    label = scene.get('label', source.stem)
    vehicle_id = VEHICLE_IDS.get(label, 'image_' + hashlib.sha256(source.name.encode('utf-8')).hexdigest()[:12])
    if 'subject_polygon' in scene:
        polygon = np.asarray(scene['subject_polygon'], dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(polygon).all() or np.any((polygon < 0) | (polygon > 1)):
            raise ValueError(f'Invalid diagnostic vehicle polygon: {source.name}')
    return {'source': source, 'scene': scene, 'label': label, 'vehicle_id': vehicle_id, 'size': size, 'outputs': {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(PROFILE_FILES), default='basic',
                        help='basic: mild; strong: historical six effects; refined: retained exposure/bloom and off-frame glare')
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'test_images')
    parser.add_argument('--output-dir', type=Path, help='Default: outputs/<profile>_v1')
    parser.add_argument('--presets', type=Path, help='Custom JSON, overrides the selected profile presets')
    parser.add_argument('--scenes', type=Path, default=PROJECT / '01_physics_based/configs/scenes.json')
    parser.add_argument('--check-config', action='store_true', help='Validate inputs and report regions without writing outputs')
    args = parser.parse_args()
    custom_presets = args.presets is not None
    args.output_dir = args.output_dir or METHOD / 'outputs' / f'{args.profile}_v1'
    args.presets = args.presets or METHOD / 'configs' / PROFILE_FILES[args.profile]
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {input_dir}')
    if output.is_relative_to(input_dir) or output.is_relative_to((PROJECT / 'models').resolve()):
        raise ValueError('Output must be outside the original-image and model directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output is not an empty directory; select a fresh run: {output}')
    presets = json.loads(args.presets.read_text(encoding='utf-8'))
    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    if not isinstance(presets, dict) or not presets:
        raise ValueError('Presets must be a nonempty ID-to-configuration mapping')
    for name, preset in presets.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'original':
            raise ValueError(f'Unsafe preset ID: {name}')
        validate_preset_template(preset)
    report_groups(presets)
    sources = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not sources:
        raise ValueError('No input images found')
    rows = [inspect_input(p, scenes) for p in sources]
    if len({r['vehicle_id'] for r in rows}) != len(rows):
        raise ValueError('Vehicle IDs must be unique within a run')
    source_hashes = {p.name: sha256(p) for p in sources}
    if args.check_config:
        print(json.dumps({'profile': args.profile, 'presets_path': str(args.presets.resolve()),
                          'inputs': len(rows), 'presets': list(presets),
                          'expected_images': len(rows) * len(presets), 'output': str(output)},
                         ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'config_snapshot', 'code_snapshot'):
        (output / name).mkdir()
    save_json(output / 'config_snapshot/presets.json', presets)
    save_json(output / 'config_snapshot/scenes.json', {p.name: scenes[p.name] for p in sources})
    code_paths = [Path(__file__).resolve(), *sorted((METHOD / 'lighting').glob('*.py'))]
    code_hashes = {}
    for path in code_paths:
        relative = path.relative_to(METHOD)
        destination = output / 'code_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code_hashes[relative.as_posix()] = sha256(path)
    started = time.perf_counter()
    manifest = {
        'method': '03_lighting_augmentation', 'status': 'running',
        'profile': args.profile, 'custom_presets': custom_presets, 'presets_path': str(args.presets.resolve()),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'pillow_version': Image.__version__,
        'input_dir': str(input_dir), 'presets': presets, 'source_sha256': source_hashes,
        'implementation_sha256': code_hashes, 'scene_config_sha256': sha256(args.scenes),
        'expected_images': len(rows) * len(presets),
        'randomness': 'none; deterministic presets applied independently to each original',
        'geometry': 'EXIF orientation correction only; original resolution and coordinates retained',
        'processing': 'sRGB to approximate linear RGB; preset photometric/optical effects; measure preclip values; clip once; sRGB; round to uint8; lossless PNG',
        'optical_scope': 'Image-space approximation; bloom spreads existing highlights, camera glare models off-frame optical scatter/ghosts; no estimated geometry, cast shadows or material relighting. Legacy optical presets remain for historical reproduction.',
        'source_placement': 'source_overrides are manual illustrative off-frame source assumptions, not recovered/calibrated sun positions; effective parameters recorded per output',
        'metric_definitions': {
            'new_clipped_pixel_fraction': 'Fraction of pixels with any raw linear channel > 1 whose corresponding source uint8 channel was < 255',
            'new_saturated_pixel_fraction': 'Fraction with any output uint8 channel == 255 whose corresponding source channel was < 255',
            'new_near_black_fraction': 'Display luma <= 5/255 after but > 5/255 before; display luma is Rec709 weighted sRGB code values, not physical luminance',
            'roi_scope': 'Full manually configured rectangles; may include background; polygons are used only for separate vehicle diagnostics',
            'contrast_metrics': 'Display luma standard deviation and gradient ratios are descriptive; expected to change with lighting; not identity certification',
            'changed_pixel_fraction_gt20': 'Fraction whose mean absolute RGB difference exceeds 20 on the 0-255 scale (20/255 normalized); appearance-change diagnostic, not semantic usefulness',
            'new_near_white_fraction': 'Fraction whose three output channels are all >=250, but whose source channels were not all >=250; approximate washed-out highlights diagnostic',
        },
        'records': [],
    }
    save_json(output / 'manifest.json', manifest)
    try:
        for row in rows:
            source = row['source']
            before_image = read_rgb(source)
            before = np.asarray(before_image)
            linear = srgb_to_linear(before.astype(np.float32) / 255)
            vehicle_mask = None
            if 'subject_polygon' in row['scene']:
                width, height = before_image.size
                mask_image = Image.new('L', (width, height), 0)
                ImageDraw.Draw(mask_image).polygon([(round(x * (width - 1)), round(y * (height - 1))) for x, y in row['scene']['subject_polygon']], fill=255)
                vehicle_mask = np.asarray(mask_image) > 0
                if not vehicle_mask.any():
                    raise ValueError(f'Empty vehicle diagnostic mask: {source.name}')
            for preset_id, preset in presets.items():
                image_started = time.perf_counter()
                effective_preset = resolve_preset(preset, source.name)
                raw = apply_lighting(linear, effective_preset)
                if raw.shape != linear.shape or not np.isfinite(raw).all():
                    raise ValueError(f'Invalid transform output: {source.name}/{preset_id}')
                after = np.rint(linear_to_srgb(raw) * 255).astype(np.uint8)
                image_path = output / 'images' / f'{row["vehicle_id"]}__{preset_id}.png'
                Image.fromarray(after).save(image_path, compress_level=3)
                record = {
                    'source': source.name, 'source_sha256': source_hashes[source.name],
                    'vehicle_id': row['vehicle_id'], 'vehicle_label': row['label'], 'size': list(before_image.size),
                    'preset_id': preset_id, 'preset': effective_preset,
                    'source_override_applied': source.name in preset.get('source_overrides', {}),
                    'output': image_path.relative_to(output).as_posix(), 'output_sha256': sha256(image_path),
                    'global': metrics(before, after, raw), 'regions': {},
                }
                if vehicle_mask is not None:
                    record['vehicle'] = metrics(before[vehicle_mask], after[vehicle_mask], raw[vehicle_mask])
                for name, box in row['scene']['regions'].items():
                    x0, y0, x1, y1 = crop_box(box, before_image.size)
                    selection = (slice(y0, y1), slice(x0, x1))
                    record['regions'][name] = {'box_xyxy': [x0, y0, x1, y1], **metrics(before[selection], after[selection], raw[selection])}
                record['seconds'] = round(time.perf_counter() - image_started, 3)
                manifest['records'].append(record)
                row['outputs'][preset_id] = image_path
                save_json(output / 'manifest.json', manifest)
                print(f'{len(manifest["records"]):02d}/{manifest["expected_images"]} {row["vehicle_id"]} {preset_id} ({record["seconds"]:.1f}s)', flush=True)
                del raw, after
            if sha256(source) != source_hashes[source.name]:
                raise RuntimeError(f'Source changed during generation: {source}')
            del linear, before, before_image
        manifest['status'] = 'generated_pending_previews'
        save_json(output / 'manifest.json', manifest)
        print('Building comparisons, native crops, and HTML report...', flush=True)
        manifest['report'] = build_report(output, rows, presets)
        verification = {'source_unchanged': all(sha256(p) == source_hashes[p.name] for p in sources), 'png_files_checked': 0, 'generated_images': len(manifest['records'])}
        if not verification['source_unchanged']:
            raise RuntimeError('Source image changed during report generation')
        for path in sorted(output.rglob('*.png')):
            with Image.open(path) as image:
                image.load()
                if image.format != 'PNG':
                    raise ValueError(f'Unexpected image format: {path}')
            verification['png_files_checked'] += 1
        if verification['png_files_checked'] != len(manifest['records']) + manifest['report']['native_crop_count']:
            raise RuntimeError('PNG count does not match full-frame outputs and native crops')
        for record in manifest['records']:
            path = output / record['output']
            if sha256(path) != record['output_sha256']:
                raise RuntimeError(f'Output changed during report generation: {path}')
            with Image.open(path) as image:
                if list(image.size) != record['size']:
                    raise ValueError(f'Output size changed: {path}')
        manifest['artifacts_sha256'] = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.html'}}
        manifest['verification'] = verification
        manifest['status'] = 'complete_pending_visual_review'
        manifest['total_seconds'] = round(time.perf_counter() - started, 3)
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(output / 'manifest.json', manifest)
        save_json(output / 'verification.json', verification)
        print(json.dumps({'status': manifest['status'], 'output': str(output), **verification, 'total_seconds': manifest['total_seconds']}, ensure_ascii=False), flush=True)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        manifest['traceback'] = traceback.format_exc()
        manifest['total_seconds'] = round(time.perf_counter() - started, 3)
        save_json(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
