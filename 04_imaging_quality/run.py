#!/usr/bin/env python3
"""Generate independent imaging-quality degradations and lossless comparisons."""

from __future__ import annotations

import argparse
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

from quality.transforms import apply_quality, validate_preset
from quality.report import build_report, report_groups


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
VEHICLE_IDS = {'Audi A6L': 'audi', 'BMW 4 Series': 'bmw', 'Xiaomi YU7': 'xiaomi', 'Honda CR-V': 'honda'}
PROFILE_FILES = {'basic': 'presets.json', 'strong': 'strong_presets.json'}


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
    return (math.floor(box[0] * width), math.floor(box[1] * height), math.ceil(box[2] * width), math.ceil(box[3] * height))


def derive_seed(run_seed: int, source_sha256: str, preset_id: str) -> int:
    """Stable under input reordering and effect selection; independent of Python hash."""
    payload = json.dumps([run_seed, source_sha256, preset_id], separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def diagnostic_size(size: tuple[int, int], short_edge: int = 224) -> tuple[int, int]:
    scale = short_edge / min(size)
    return tuple(max(1, round(side * scale)) for side in size)


def metrics(before: np.ndarray, after: np.ndarray) -> dict:
    """Descriptive image differences, not image quality or classification scores."""
    delta = after.astype(np.float32) - before
    mse = float(np.mean(delta * delta, dtype=np.float64))
    source_luma = np.sum(before.astype(np.float32) * LUMA, axis=-1) / 255
    output_luma = np.sum(after.astype(np.float32) * LUMA, axis=-1) / 255
    source_std, output_std = float(source_luma.std()), float(output_luma.std())
    result = {
        'pixel_count': int(source_luma.size),
        'mae_code_values': float(np.mean(np.abs(delta), dtype=np.float64)),
        'mse_code_values': mse,
        'psnr_db': 10 * math.log10(255 ** 2 / mse) if mse else None,
        'identical_pixels': bool(mse == 0),
        'changed_pixel_fraction_gt20': float(np.mean(np.mean(np.abs(delta), axis=-1) > 20)),
        'source_display_luma_std': source_std,
        'output_display_luma_std': output_std,
        'display_luma_std_ratio': output_std / source_std if source_std > 1e-8 else None,
    }
    if source_luma.ndim == 2 and min(source_luma.shape) > 1:
        def gradient(image):
            return float((np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2)
        original, modified = gradient(source_luma), gradient(output_luma)
        result.update(source_display_luma_gradient=original, output_display_luma_gradient=modified,
                      display_luma_gradient_ratio=modified / original if original > 1e-8 else None)
    return result


def inspect_input(source: Path, scenes: dict) -> dict:
    if source.name not in scenes:
        raise ValueError(f'Add scene label and comparison regions: {source.name}')
    scene = scenes[source.name]
    if not scene.get('regions'):
        raise ValueError(f'No comparison regions: {source.name}')
    with Image.open(source) as opened:
        size = ImageOps.exif_transpose(opened).size
    for name, box in scene['regions'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Unsafe region ID: {name}')
        crop_box(box, size)
    label = scene.get('label', source.stem)
    vehicle_id = VEHICLE_IDS.get(label, 'image_' + hashlib.sha256(source.name.encode('utf-8')).hexdigest()[:12])
    if 'subject_polygon' in scene:
        polygon = np.asarray(scene['subject_polygon'], dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(polygon).all() or np.any((polygon < 0) | (polygon > 1)):
            raise ValueError(f'Invalid vehicle polygon: {source.name}')
    return {'source': source, 'scene': scene, 'label': label, 'vehicle_id': vehicle_id, 'size': size, 'outputs': {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(PROFILE_FILES), default='basic',
                        help='basic: initial eight presets; strong: retained defocus with more visible other degradations')
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'test_images')
    parser.add_argument('--output-dir', type=Path, help='Default: outputs/quality_v1 or outputs/quality_strong_v1 by profile')
    parser.add_argument('--presets', type=Path, help='Custom JSON overrides profile presets')
    parser.add_argument('--scenes', type=Path, default=PROJECT / '01_physics_based/configs/scenes.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--effects', nargs='+', help='Run only these preset IDs, in the specified order')
    parser.add_argument('--check-config', action='store_true', help='Validate inputs without writing outputs')
    args = parser.parse_args()
    custom_presets = args.presets is not None
    args.presets = args.presets or METHOD / 'configs' / PROFILE_FILES[args.profile]
    args.output_dir = args.output_dir or METHOD / 'outputs' / ('quality_v1' if args.profile == 'basic' else 'quality_strong_v1')
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {input_dir}')
    if output.is_relative_to(input_dir) or output.is_relative_to((PROJECT / 'models').resolve()):
        raise ValueError('Output must be outside original-image and model directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output is not empty; select a fresh run: {output}')
    presets = json.loads(args.presets.read_text(encoding='utf-8'))
    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    if not isinstance(presets, dict) or not presets:
        raise ValueError('Presets must be a nonempty ID-to-configuration mapping')
    for name, preset in presets.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'original':
            raise ValueError(f'Unsafe preset ID: {name}')
        validate_preset(preset)
    if args.effects:
        if len(set(args.effects)) != len(args.effects) or set(args.effects) - presets.keys():
            raise ValueError('effects must contain distinct, configured preset IDs')
        presets = {name: presets[name] for name in args.effects}
    report_groups(presets)
    sources = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not sources:
        raise ValueError('No input images found')
    rows = [inspect_input(p, scenes) for p in sources]
    if len({row['vehicle_id'] for row in rows}) != len(rows):
        raise ValueError('Vehicle IDs must be unique within a run')
    source_hashes = {p.name: sha256(p) for p in sources}
    if args.check_config:
        print(json.dumps({'profile': args.profile, 'presets_path': str(args.presets.resolve()),
                          'inputs': len(rows), 'presets': list(presets), 'seed': args.seed,
                          'expected_images': len(rows) * len(presets), 'output': str(output)}, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'encoded', 'native', 'config_snapshot', 'code_snapshot'):
        (output / name).mkdir()
    save_json(output / 'config_snapshot/presets.json', presets)
    save_json(output / 'config_snapshot/scenes.json', {p.name: scenes[p.name] for p in sources})
    code_hashes = {}
    for path in [Path(__file__).resolve(), *sorted((METHOD / 'quality').glob('*.py'))]:
        relative = path.relative_to(METHOD)
        destination = output / 'code_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        code_hashes[relative.as_posix()] = sha256(path)
    started = time.perf_counter()
    manifest = {
        'method': '04_imaging_quality', 'status': 'running',
        'profile': args.profile, 'custom_presets': custom_presets,
        'presets_path': str(args.presets.resolve()),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'pillow_version': Image.__version__,
        'input_dir': str(input_dir), 'presets': presets, 'source_sha256': source_hashes,
        'implementation_sha256': code_hashes, 'scene_config_sha256': sha256(args.scenes),
        'expected_images': len(rows) * len(presets), 'seed': args.seed,
        'seed_derivation': 'SHA256(UTF-8 compact JSON [run_seed, source_sha256, preset_id]) first 8 bytes, unsigned big endian; PCG64',
        'processing': 'Each preset starts from the EXIF-corrected original RGB independently. Blur and noise use approximate linear RGB. Resolution and JPEG operate on sRGB code values. Main outputs and comparison previews are lossless PNG.',
        'geometry': 'Original dimensions and coordinates retained. Low-resolution samples are restored to the original size; native downsampled images also saved.',
        'scope': 'Image-space approximations, not calibrated camera RAW/ISP simulations. Optional coarse noise resamples only the random residual field, not original image content. Original JPEG sources are already compressed; JPEG presets perform another encoding at the original dimensions.',
        'metric_definitions': {
            'mae_code_values': 'Mean absolute RGB difference on the 0-255 scale',
            'psnr_db': '10 log10(255 squared / RGB MSE); null for identical pixels (infinite PSNR)',
            'display_luma_gradient_ratio': 'Mean absolute horizontal/vertical Rec709-weighted sRGB gradient ratio; noise can increase it, so this is NOT a sharpness or semantic-quality score',
            'diagnostic_224': 'Resize full original and output to short edge 224 using LANCZOS, then measure differences; no crop-first resizing',
            'roi_scope': 'Manual rectangular regions may include background; optional vehicle polygon measured separately',
        },
        'records': [],
    }
    save_json(output / 'manifest.json', manifest)
    try:
        for row in rows:
            source = row['source']
            before_image = read_rgb(source)
            before = np.asarray(before_image)
            small_size = diagnostic_size(before_image.size)
            before_small = np.asarray(before_image.resize(small_size, Image.Resampling.LANCZOS))
            vehicle_mask = None
            if 'subject_polygon' in row['scene']:
                width, height = before_image.size
                mask = Image.new('L', before_image.size, 0)
                ImageDraw.Draw(mask).polygon([(round(x * (width - 1)), round(y * (height - 1))) for x, y in row['scene']['subject_polygon']], fill=255)
                vehicle_mask = np.asarray(mask) > 0
                if not vehicle_mask.any():
                    raise ValueError(f'Empty vehicle mask: {source.name}')
            for preset_id, preset in presets.items():
                image_started = time.perf_counter()
                seed = derive_seed(args.seed, source_hashes[source.name], preset_id)
                result = apply_quality(before, preset, seed=seed)
                after = result.image
                if after.shape != before.shape or after.dtype != np.uint8:
                    raise ValueError(f'Invalid transform output: {source.name}/{preset_id}')
                stem = f'{row["vehicle_id"]}__{preset_id}'
                image_path = output / 'images' / (stem + '.png')
                after_image = Image.fromarray(after)
                after_image.save(image_path, compress_level=3)
                record = {
                    'source': source.name, 'source_sha256': source_hashes[source.name],
                    'vehicle_id': row['vehicle_id'], 'vehicle_label': row['label'], 'size': list(before_image.size),
                    'preset_id': preset_id, 'preset': preset, 'seed': seed, 'effective_parameters': result.metadata,
                    'output': image_path.relative_to(output).as_posix(), 'output_sha256': sha256(image_path),
                    'global': metrics(before, after), 'regions': {},
                    'diagnostic_224': {'size': list(small_size), **metrics(before_small, np.asarray(after_image.resize(small_size, Image.Resampling.LANCZOS)))},
                }
                if result.encoded_jpeg is not None:
                    encoded_path = output / 'encoded' / (stem + '.jpg')
                    encoded_path.write_bytes(result.encoded_jpeg)
                    if not np.array_equal(np.asarray(read_rgb(encoded_path)), after):
                        raise RuntimeError('Saved JPEG decoding differs from PNG output')
                    record['encoded_jpeg'] = {'path': encoded_path.relative_to(output).as_posix(), 'sha256': sha256(encoded_path), 'bytes': encoded_path.stat().st_size}
                if result.native_image is not None:
                    native_path = output / 'native' / (stem + '.png')
                    native_image = Image.fromarray(result.native_image)
                    native_image.save(native_path, compress_level=3)
                    record['native_low_resolution'] = {'path': native_path.relative_to(output).as_posix(), 'sha256': sha256(native_path), 'size': list(native_image.size)}
                if vehicle_mask is not None:
                    record['vehicle'] = metrics(before[vehicle_mask], after[vehicle_mask])
                for name, box in row['scene']['regions'].items():
                    x0, y0, x1, y1 = crop_box(box, before_image.size)
                    selection = (slice(y0, y1), slice(x0, x1))
                    record['regions'][name] = {'box_xyxy': [x0, y0, x1, y1], **metrics(before[selection], after[selection])}
                record['seconds'] = round(time.perf_counter() - image_started, 3)
                manifest['records'].append(record)
                row['outputs'][preset_id] = image_path
                save_json(output / 'manifest.json', manifest)
                print(f'{len(manifest["records"]):02d}/{manifest["expected_images"]} {row["vehicle_id"]} {preset_id} ({record["seconds"]:.1f}s)', flush=True)
                del result, after, after_image
            if sha256(source) != source_hashes[source.name]:
                raise RuntimeError(f'Source changed during generation: {source}')
            del before, before_image
        manifest['status'] = 'generated_pending_previews'
        save_json(output / 'manifest.json', manifest)
        print('Building lossless comparisons, native crops, and HTML report...', flush=True)
        manifest['report'] = build_report(output, rows, presets)
        verification = {'source_unchanged': all(sha256(p) == source_hashes[p.name] for p in sources),
                        'png_files_checked': 0, 'generated_images': len(manifest['records']),
                        'encoded_jpeg_checked': 0, 'native_low_resolution_checked': 0}
        if not verification['source_unchanged']:
            raise RuntimeError('Source changed during report generation')
        for path in sorted(output.rglob('*.png')):
            with Image.open(path) as image:
                image.load()
                if image.format != 'PNG':
                    raise ValueError(f'Unexpected format: {path}')
            verification['png_files_checked'] += 1
        for record in manifest['records']:
            path = output / record['output']
            if sha256(path) != record['output_sha256']:
                raise RuntimeError(f'Output changed: {path}')
            with Image.open(path) as image:
                if list(image.size) != record['size']:
                    raise ValueError(f'Output dimensions changed: {path}')
                for field, counter in [('encoded_jpeg', 'encoded_jpeg_checked'), ('native_low_resolution', 'native_low_resolution_checked')]:
                    if field not in record:
                        continue
                    artifact = record[field]
                    artifact_path = output / artifact['path']
                    if sha256(artifact_path) != artifact['sha256']:
                        raise RuntimeError(f'Auxiliary output changed: {artifact_path}')
                    with Image.open(artifact_path) as auxiliary:
                        auxiliary.load()
                        if field == 'encoded_jpeg' and not np.array_equal(np.asarray(auxiliary.convert('RGB')), np.asarray(image)):
                            raise RuntimeError('JPEG decoding differs from saved PNG')
                        if field == 'native_low_resolution' and list(auxiliary.size) != artifact['size']:
                            raise RuntimeError('Native low-resolution dimensions changed')
                    verification[counter] += 1
        expected_png = len(manifest['records']) + manifest['report']['native_crop_count'] + manifest['report']['preview_png_count'] + verification['native_low_resolution_checked']
        if verification['png_files_checked'] != expected_png:
            raise RuntimeError(f'Unexpected PNG count: {verification["png_files_checked"]} vs {expected_png}')
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
