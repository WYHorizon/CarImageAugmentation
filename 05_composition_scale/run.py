#!/usr/bin/env python3
"""Generate visible composition changes with transformed subject and part geometry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import time
import traceback

import numpy as np
from PIL import Image, ImageOps

from composition.transforms import (
    apply_composition, describe_polygon, plan_transform, transform_points, validate_preset,
)
from composition.report import build_report, report_groups


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}
VEHICLE_IDS = {'Audi A6L': 'audi', 'BMW 4 Series': 'bmw', 'Xiaomi YU7': 'xiaomi', 'Honda CR-V': 'honda'}


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
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')


def region_polygon(box, size) -> np.ndarray:
    """A normalized rectangle becomes four pixel-boundary corners before mapping."""
    try:
        values = np.asarray(box, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(f'Invalid normalized region: {box}') from None
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f'Invalid normalized region: {box}')
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f'Invalid normalized region: {box}')
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * np.asarray(size)


def annotation_record(scene: dict, size, transform: dict) -> dict:
    """Map geometry and report visibility without comparing misaligned pixels."""
    matrix = transform['forward_matrix']
    regions = {name: describe_polygon(region_polygon(box, size), matrix, size)
               for name, box in scene['regions'].items()}
    subject, metrics = None, {
        'scale': transform['scale'], 'roll_degrees': transform['angle_degrees'],
        'subject_available': 'subject_polygon' in scene,
    }
    if 'subject_polygon' in scene:
        original = np.asarray(scene['subject_polygon'], dtype=np.float64) * np.asarray(size)
        subject = describe_polygon(original, matrix, size)
        low, high = original.min(axis=0), original.max(axis=0)
        center = (low + high) / 2
        moved_center = transform_points([center], matrix)[0]
        transformed_box = subject['transformed_bbox_xyxy']
        visible_box = subject['visible_bbox_xyxy']
        metrics.update({
            'source_subject_center_normalized': (center / np.asarray(size)).tolist(),
            'transformed_subject_center_normalized': (moved_center / np.asarray(size)).tolist(),
            'subject_center_shift_normalized': ((moved_center - center) / np.asarray(size)).tolist(),
            'subject_center_shift_pixels': (moved_center - center).tolist(),
            'source_subject_bbox_width_fraction': float((high[0] - low[0]) / size[0]),
            'transformed_subject_bbox_width_fraction': float((transformed_box[2] - transformed_box[0]) / size[0]),
            'visible_subject_bbox_width_fraction': float((visible_box[2] - visible_box[0]) / size[0]) if visible_box else 0.0,
            'subject_retained_fraction': subject['retained_fraction'],
            'subject_source_area_fraction': subject['original_area'] / (size[0] * size[1]),
            'subject_visible_area_fraction': subject['visible_area'] / (size[0] * size[1]),
        })
    policy = transform['retention_policy']
    retention_check = ('not_available_without_subject_polygon' if subject is None else
                       'passed' if policy != 'complete' or subject['retained_fraction'] >= 1 - 1e-8 else 'failed')
    return {'transform': transform, 'subject': subject, 'regions': regions,
            'geometry_metrics': metrics, 'retention_check': retention_check}


def inspect_input(source: Path, scenes: dict, presets: dict) -> dict:
    if source.name not in scenes or not isinstance(scenes[source.name], dict):
        raise ValueError(f'Add comparison regions in --scenes for: {source.name}')
    scene = scenes[source.name]
    if not isinstance(scene.get('regions'), dict) or not scene['regions']:
        raise ValueError(f'No comparison regions configured: {source.name}')
    with Image.open(source) as image:
        size = ImageOps.exif_transpose(image).size
    for name, box in scene['regions'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Unsafe region ID: {name}')
        region_polygon(box, size)
    label = scene.get('label', source.stem)
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f'Invalid image label: {source.name}')
    vehicle_id = VEHICLE_IDS.get(label, 'image_' + hashlib.sha256(source.name.encode('utf-8')).hexdigest()[:12])
    planned = {}
    for preset_id, preset in presets.items():
        transform = plan_transform(size, preset, scene.get('subject_polygon'))
        record = annotation_record(scene, size, transform)
        if record['retention_check'] == 'failed':
            raise ValueError(f'{source.name}/{preset_id}: complete-subject preset would retain only '
                             f'{record["subject"]["retained_fraction"]:.3%}; choose a different preset/anchor or explicitly allow partial retention')
        if record['subject'] is not None and not record['subject']['visible']:
            raise ValueError(f'{source.name}/{preset_id}: entire subject would leave the output')
        planned[preset_id] = record
    return {'source': source, 'scene': scene, 'label': label, 'vehicle_id': vehicle_id,
            'size': size, 'outputs': {}, 'geometry': {}, 'planned': planned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'test_images')
    parser.add_argument('--output-dir', type=Path, default=METHOD / 'outputs/composition_v1')
    parser.add_argument('--presets', type=Path, default=METHOD / 'configs/presets.json')
    parser.add_argument('--scenes', type=Path, default=PROJECT / '01_physics_based/configs/scenes.json')
    parser.add_argument('--effects', nargs='+', help='Run only the listed preset IDs')
    parser.add_argument('--check-config', action='store_true', help='Plan transforms and validate visibility without writing images')
    args = parser.parse_args()
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {input_dir}')
    if output.is_relative_to(input_dir) or output.is_relative_to((PROJECT / 'models').resolve()):
        raise ValueError('Output must be outside original-image and model directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output is not empty; choose a fresh run: {output}')
    presets = json.loads(args.presets.read_text(encoding='utf-8'))
    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    if not isinstance(presets, dict) or not presets:
        raise ValueError('Presets must be a nonempty ID-to-configuration mapping')
    if not isinstance(scenes, dict):
        raise ValueError('Scenes must be a filename-to-scene mapping')
    for name, preset in presets.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'original':
            raise ValueError(f'Unsafe preset ID: {name}')
        validate_preset(preset)
    if args.effects:
        if len(set(args.effects)) != len(args.effects) or set(args.effects) - presets.keys():
            raise ValueError('effects must contain distinct configured preset IDs')
        presets = {name: presets[name] for name in args.effects}
    report_groups(presets)
    sources = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not sources:
        raise ValueError('No input images found')
    rows = [inspect_input(source, scenes, presets) for source in sources]
    duplicate_ids = {row['vehicle_id'] for row in rows if sum(other['vehicle_id'] == row['vehicle_id'] for other in rows) > 1}
    for row in rows:
        if row['vehicle_id'] in duplicate_ids:
            row['vehicle_id'] += '_' + hashlib.sha256(row['source'].name.encode('utf-8')).hexdigest()[:12]
    source_hashes = {source.name: sha256(source) for source in sources}
    if args.check_config:
        print(json.dumps({'inputs': len(rows), 'presets': list(presets),
                          'expected_images': len(rows) * len(presets), 'output': str(output),
                          'subject_retention_checks': {row['source'].name: {name: record['retention_check'] for name, record in row['planned'].items()} for row in rows}}, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'masks', 'config_snapshot', 'code_snapshot'):
        (output / name).mkdir()
    save_json(output / 'config_snapshot/presets.json', presets)
    save_json(output / 'config_snapshot/scenes.json', {p.name: scenes[p.name] for p in sources})
    code_hashes = {}
    for path in [Path(__file__).resolve(), *sorted((METHOD / 'composition').glob('*.py'))]:
        relative = path.relative_to(METHOD)
        destination = output / 'code_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code_hashes[relative.as_posix()] = sha256(path)
    started = time.perf_counter()
    manifest = {
        'method': '05_composition_scale', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'pillow_version': Image.__version__,
        'input_dir': str(input_dir), 'presets': presets, 'source_sha256': source_hashes,
        'scene_config_sha256': sha256(args.scenes), 'implementation_sha256': code_hashes,
        'expected_images': len(rows) * len(presets), 'randomness': 'none; each deterministic preset starts from the original',
        'processing': 'EXIF-corrected RGB; one inverse AFFINE BICUBIC resampling; lossless PNG at original canvas size',
        'coordinate_frame': 'Source and destination pixel boundaries [0,width] x [0,height], pixel centers at half-integers; matrices map source to output',
        'scope': '2D framing, uniform scale and camera roll. No new viewpoint, synthetic background, reflection or generated pixels. Crops can remove vehicle parts.',
        'geometry_definitions': {
            'retained_fraction': 'Visible clipped polygon area / full transformed polygon area, not a semantic information or classification score',
            'subject_center': 'Forward-mapped original subject bounding-box center; not the center of the clipped visible bounding box',
            'subject_bbox_width_fraction': 'Axis-aligned bounding-box width divided by canvas width; transformed width can exceed 1',
            'region_scope': 'Manually configured rectangle corners transformed with the same matrix; native crop is visible polygon bounding rectangle and may include background',
            'mask': 'Output pixels sampling inside the original image extent, not a vehicle segmentation mask',
            'complete_policy': 'When a subject polygon is available, preflight requires retention >= 1 - 1e-8; otherwise no completeness claim',
            'diagnostic_224': 'Resize complete output to short edge 224 first, then transform ROI coordinates by actual horizontal/vertical resize ratios',
        },
        'records': [],
    }
    save_json(output / 'manifest.json', manifest)
    try:
        for row in rows:
            source = row['source']
            before_image = read_rgb(source)
            before = np.asarray(before_image)
            for preset_id, preset in presets.items():
                image_started = time.perf_counter()
                result = apply_composition(before, preset, row['scene'].get('subject_polygon'))
                if result.image.shape != before.shape or result.image.dtype != np.uint8:
                    raise RuntimeError(f'Invalid output pixels: {source.name}/{preset_id}')
                if result.valid_mask.shape != before.shape[:2] or result.valid_mask.dtype != np.uint8 or not np.all(result.valid_mask == 255):
                    raise RuntimeError(f'Output contains padding: {source.name}/{preset_id}')
                np.testing.assert_allclose(result.metadata['forward_matrix'], row['planned'][preset_id]['transform']['forward_matrix'], atol=1e-12, rtol=0)
                record = annotation_record(row['scene'], before_image.size, result.metadata)
                stem = f'{row["vehicle_id"]}__{preset_id}.png'
                image_path, mask_path = output / 'images' / stem, output / 'masks' / stem
                Image.fromarray(result.image).save(image_path, compress_level=3)
                Image.fromarray(result.valid_mask).save(mask_path, compress_level=3)
                record.update({
                    'source': source.name, 'source_sha256': source_hashes[source.name],
                    'vehicle_id': row['vehicle_id'], 'vehicle_label': row['label'], 'size': list(before_image.size),
                    'preset_id': preset_id, 'preset': preset,
                    'output': image_path.relative_to(output).as_posix(), 'output_sha256': sha256(image_path),
                    'valid_mask': mask_path.relative_to(output).as_posix(), 'valid_mask_sha256': sha256(mask_path),
                    'seconds': round(time.perf_counter() - image_started, 3),
                })
                manifest['records'].append(record)
                row['outputs'][preset_id] = image_path
                row['geometry'][preset_id] = record
                save_json(output / 'manifest.json', manifest)
                print(f'{len(manifest["records"]):02d}/{manifest["expected_images"]} {row["vehicle_id"]} {preset_id} ({record["seconds"]:.1f}s)', flush=True)
                del result
            if sha256(source) != source_hashes[source.name]:
                raise RuntimeError(f'Source changed during generation: {source}')
            del before, before_image
        manifest['status'] = 'generated_pending_previews'
        save_json(output / 'manifest.json', manifest)
        print('Building transformed part crops, framing diagrams, and HTML report...', flush=True)
        manifest['report'] = build_report(output, rows, presets)
        verification = {'source_unchanged': all(sha256(p) == source_hashes[p.name] for p in sources),
                        'generated_images': len(manifest['records']), 'png_files_checked': 0,
                        'valid_masks_checked': 0, 'all_pixels_from_source': True}
        if not verification['source_unchanged']:
            raise RuntimeError('Source changed during report generation')
        for path in sorted(output.rglob('*.png')):
            with Image.open(path) as image:
                image.load()
                if image.format != 'PNG':
                    raise RuntimeError(f'Unexpected image format: {path}')
            verification['png_files_checked'] += 1
        for record in manifest['records']:
            for key, digest_key in [('output', 'output_sha256'), ('valid_mask', 'valid_mask_sha256')]:
                path = output / record[key]
                if sha256(path) != record[digest_key]:
                    raise RuntimeError(f'Artifact changed during report generation: {path}')
                with Image.open(path) as image:
                    if list(image.size) != record['size']:
                        raise RuntimeError(f'Artifact dimensions changed: {path}')
                    if key == 'valid_mask':
                        if image.mode != 'L' or not np.all(np.asarray(image) == 255):
                            raise RuntimeError(f'Invalid source-coverage mask: {path}')
                        verification['valid_masks_checked'] += 1
        expected_png = len(manifest['records']) + verification['valid_masks_checked'] + manifest['report']['native_crop_count'] + manifest['report']['preview_png_count']
        if verification['png_files_checked'] != expected_png:
            raise RuntimeError(f'Unexpected PNG count: {verification["png_files_checked"]} vs {expected_png}')
        manifest['artifacts_sha256'] = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.suffix.lower() in {'.png', '.html'}}
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
