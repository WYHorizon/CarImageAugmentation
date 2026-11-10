#!/usr/bin/env python3
"""Run the physics baseline from any working directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.physics_weather import build_depth, region_diagnostics, render, srgb_to_linear, to_image


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, relative_name: str, preset: str, variant: int) -> int:
    payload = json.dumps([base_seed, relative_name, preset, variant], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def font(size: int):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def fit_tile(image: Image.Image, width: int, height: int, title: str) -> Image.Image:
    tile = Image.new('RGB', (width, height + 32), '#eeeeee')
    thumb = ImageOps.contain(image, (width, height))
    tile.paste(thumb, ((width - thumb.width) // 2, 32 + (height - thumb.height) // 2))
    ImageDraw.Draw(tile).text((9, 6), title, fill='black', font=font(17))
    return tile


def save_geometry(image: Image.Image, depth: np.ndarray, scene: dict, destination: Path):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size
    points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in scene['subject_polygon']]
    draw.line(points + points[:1], fill='#00ff8c', width=max(2, width // 650))
    for name, box in scene['regions'].items():
        x0, y0, x1, y1 = box
        draw.rectangle((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)),
                       outline='#ff8b24', width=max(2, width // 700))
        draw.text((int(x0 * width), int(y0 * height)), name, fill='#ff8b24', font=font(max(15, width // 100)))
    scaled = np.clip(np.log1p(depth) / np.log1p(scene['background_depth_m']), 0, 1)
    depth_image = Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).convert('RGB')
    sheet = Image.new('RGB', (1600, 632), 'white')
    sheet.paste(fit_tile(annotated, 800, 600, scene['label'] + ' / manual geometry'), (0, 0))
    sheet.paste(fit_tile(depth_image, 800, 600, 'Approximate depth Z: dark = near'), (800, 0))
    sheet.save(destination, quality=94)


def save_previews(rows: list[dict], destination: Path, names: list[str]):
    cols = ['original'] + names
    overview = Image.new('RGB', (480 * len(cols), 392 * len(rows)), '#dddddd')
    for index, row in enumerate(rows):
        images = [row['source']] + [row['outputs'][name] for name in names]
        comparison = Image.new('RGB', (480 * len(cols), 392), '#dddddd')
        for col, (name, path) in enumerate(zip(cols, images)):
            tile = fit_tile(read_rgb(path), 480, 360, row['scene']['label'] + ' / ' + name)
            overview.paste(tile, (col * 480, index * 392))
            comparison.paste(tile, (col * 480, 0))
        comparison.save(destination / (row['source'].stem + '_comparison.jpg'), quality=95)
        regions = row['scene']['regions']
        detail = Image.new('RGB', (320 * len(cols), 272 * len(regions)), '#dddddd')
        for col, (name, path) in enumerate(zip(cols, images)):
            image = read_rgb(path)
            width, height = image.size
            for region_index, (region_name, box) in enumerate(regions.items()):
                x0, y0, x1, y1 = box
                crop = image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
                tile = fit_tile(crop, 320, 240, region_name + ' / ' + name)
                detail.paste(tile, (col * 320, region_index * 272))
        detail.save(destination / (row['source'].stem + '_details.jpg'), quality=95)
    overview.save(destination / 'overview.jpg', quality=94)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT.parent / 'test_images')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'demo')
    parser.add_argument('--scenes', type=Path, default=ROOT / 'configs' / 'scenes.json')
    parser.add_argument('--presets', type=Path, default=ROOT / 'configs' / 'presets.json')
    parser.add_argument('--weather', nargs='+', default=['fog_light', 'fog_medium', 'rain_light'])
    parser.add_argument('--count', type=int, default=1, help='Random variants per rain preset; fog is deterministic')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit('--count must be positive')
    if len(set(args.weather)) != len(args.weather):
        raise SystemExit('--weather contains duplicate names')
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f'Input directory does not exist: {input_dir}')
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise SystemExit('Output must be outside the original-image directory')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f'Output directory is not empty; use a new --output-dir: {output_dir}')
    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    presets = json.loads(args.presets.read_text(encoding='utf-8'))
    for name in args.weather:
        if name not in presets or not name.replace('_', '').isalnum():
            raise SystemExit(f'Unknown or invalid weather preset: {name}')
        if presets[name]['kind'] not in {'fog', 'rain'}:
            raise SystemExit(f'Unsupported weather kind: {presets[name]["kind"]}')
    sources = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not sources:
        raise SystemExit(f'No supported images in {input_dir}')
    unknown = [p.name for p in sources if p.name not in scenes]
    if unknown:
        raise SystemExit('Add manual scene geometry in --scenes for these images: ' + ', '.join(unknown))
    if len({p.stem for p in sources}) != len(sources):
        raise SystemExit('Source filename stems must be unique to avoid output collisions')

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ['images', 'diagnostics', 'previews', 'config_snapshot']:
        (output_dir / directory).mkdir()
    # Snapshot the exact configuration, not merely a path to a mutable file.
    for filename, value in [('scenes.json', scenes), ('presets.json', presets)]:
        (output_dir / 'config_snapshot' / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    implementation = {str(path.relative_to(ROOT)): sha256(path)
                      for path in [Path(__file__).resolve(), ROOT / 'src' / 'physics_weather.py',
                                   ROOT / 'src' / 'scene_weather.py']}
    manifest = {
        'method': '01_physics_based', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_dir': str(input_dir), 'base_seed': args.seed, 'implementation_sha256': implementation,
        'numpy_version': np.__version__, 'pillow_version': Image.__version__,
        'geometry': 'hand-authored approximate depths and camera parameters; uncalibrated',
        'review_status': 'preview required; visibility diagnostics are not identity certification',
        'records': [],
    }
    preview_rows = []
    for source in sources:
        image = read_rgb(source)
        scene = scenes[source.name]
        linear = srgb_to_linear(np.asarray(image, dtype=np.float32) / 255)
        depth, subject = build_depth(image.size, scene)
        save_geometry(image, depth, scene, output_dir / 'diagnostics' / (source.stem + '_geometry.jpg'))
        source_hash = sha256(source)
        outputs = {}
        for name in args.weather:
            preset = presets[name]
            variant_count = args.count if preset['kind'] == 'rain' else 1
            for variant in range(variant_count):
                seed = stable_seed(args.seed, source.name, name, variant)
                result, fog_t, rain_t, stats = render(linear, depth, scene, preset, seed, include_scene_maps=True)
                maps = stats.pop('_scene_maps', None)
                if maps:
                    panel = Image.new('RGB', (640 * len(maps), 512), 'white')
                    for index, (title, map_image) in enumerate(maps.items()):
                        panel.paste(fit_tile(map_image, 640, 480, title), (index * 640, 0))
                    panel.save(output_dir / 'diagnostics' / (source.stem + '__' + name + f'__{variant:03d}_surface.jpg'), quality=94)
                if result.shape != linear.shape or not np.isfinite(result).all():
                    raise RuntimeError(f'Invalid render: {source.name} / {name}')
                relative_output = Path('images') / (source.stem + '__' + name + f'__{variant:03d}.png')
                path = output_dir / relative_output
                to_image(result).save(path, compress_level=3)
                record = {
                    'source': source.name, 'source_sha256': source_hash, 'size': list(image.size),
                    'weather': name, 'variant': variant, 'seed': seed, 'preset': preset,
                    'output': relative_output.as_posix(), 'output_sha256': sha256(path),
                    'linear_pixels_above_display_range_fraction': float(np.mean(np.any(result > 1, axis=2))),
                    'regions': region_diagnostics(fog_t, rain_t, scene, subject), **stats,
                }
                manifest['records'].append(record)
                if variant == 0:
                    outputs[name] = path
                print(f'{source.name} -> {relative_output.name}', flush=True)
                del result, fog_t, rain_t
        preview_rows.append({'source': source, 'scene': scene, 'outputs': outputs})
        del linear, depth, subject
    save_previews(preview_rows, output_dir / 'previews', args.weather)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Completed {len(manifest["records"])} full-resolution images: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
