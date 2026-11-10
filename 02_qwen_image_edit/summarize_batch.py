"""Verify local Qwen outputs and build an offline, refreshable batch gallery."""

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps

from run import PROJECT, display_rgb, save_json, sha256


SUCCESS_STATUS = 'complete_pending_visual_review'
INFERENCE_KEYS = (
    'input_image', 'model_dir', 'model_revision', 'width', 'height', 'seed',
    'num_inference_steps', 'true_cfg_scale', 'use_kv_cache', 'vae_tiling', 'prompt',
)
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(path):
    return (PROJECT / path).resolve()


def validate_job(job):
    """Checks technical integrity only, never visual fidelity or training suitability."""
    output = resolve(job['output_dir'])
    record = {
        **job, 'output_dir': str(output), 'generation_status': 'pending',
        'validation_status': 'not_ready', 'valid': None, 'errors': [],
        'source_unchanged': None, 'png_files_checked': 0,
        'visual_review': 'not_performed_by_this_tool',
    }
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        return record
    try:
        manifest = read_json(manifest_path)
        record['generation_status'] = manifest.get('status', 'unknown')
        for key in ('load_seconds', 'inference_seconds', 'total_seconds',
                    'peak_allocated_gib', 'peak_reserved_gib', 'completed_steps'):
            record[key] = manifest.get(key)
        if record['generation_status'] != SUCCESS_STATUS:
            if manifest.get('error'):
                record['generation_error'] = manifest['error']
            return record

        expected = read_json(resolve(job['config']))
        actual = manifest['config']
        errors = record['errors']
        for key in INFERENCE_KEYS:
            if actual.get(key) != expected.get(key):
                errors.append(f'Configuration differs for {key}')
        if manifest['completed_steps'] != expected['num_inference_steps']:
            errors.append('Completed step count differs from requested inference steps')
        source = resolve(expected['input_image'])
        if Path(manifest['source']).resolve() != source:
            errors.append('Manifest source path differs from batch configuration')
        record['source_unchanged'] = sha256(source) == manifest['source_sha256']
        if not record['source_unchanged']:
            errors.append('Source SHA256 changed')

        mapping = manifest['mapping']
        dimensions = {
            'source_canvas.png': mapping['canvas_size'],
            'edited_canvas.png': mapping['canvas_size'],
            'source_working.png': mapping['content_size'],
            'edited.png': mapping['content_size'],
        }
        if mapping['canvas_size'] != [expected['width'], expected['height']]:
            errors.append('Canvas mapping differs from requested size')
        left, top, right, bottom = mapping['content_box_xyxy']
        if [right - left, bottom - top] != mapping['content_size']:
            errors.append('Crop box differs from content dimensions')
        with Image.open(source) as original:
            if list(ImageOps.exif_transpose(original).size) != mapping['original_size']:
                errors.append('Original image dimensions differ from mapping')

        hashes = manifest['output_sha256']
        actual_pngs = {str(p.relative_to(output)) for p in output.rglob('*.png')}
        if actual_pngs != set(hashes):
            errors.append('PNG inventory differs from manifest SHA256 inventory')
        for name in dimensions:
            if name not in hashes:
                errors.append(f'Required PNG absent from manifest: {name}')
        for name, digest in hashes.items():
            image_path = (output / name).resolve()
            if not image_path.is_relative_to(output) or image_path.suffix.lower() != '.png':
                errors.append(f'Unexpected PNG path in manifest: {name}')
                continue
            try:
                if sha256(image_path) != digest:
                    errors.append(f'PNG SHA256 mismatch: {name}')
                with Image.open(image_path) as opened:
                    if name in dimensions and list(opened.size) != dimensions[name]:
                        errors.append(f'PNG dimensions differ from mapping: {name}')
                    if opened.format != 'PNG':
                        errors.append(f'File is not PNG: {name}')
                    opened.verify()
                with Image.open(image_path) as decoded:
                    decoded.load()
                record['png_files_checked'] += 1
            except (OSError, ValueError, SyntaxError) as error:
                errors.append(f'PNG validation failed ({name}): {error}')
        record['valid'] = not errors
        record['validation_status'] = 'passed' if record['valid'] else 'failed'
        record['mapping'] = mapping
        record['source'] = str(source)
    except (OSError, ValueError, KeyError, TypeError) as error:
        record['validation_status'] = 'failed'
        record['valid'] = False
        record['errors'].append(f'Cannot validate result: {error}')
    return record


def load_thumbnail(path, size):
    with Image.open(path) as opened:
        rgb = display_rgb(ImageOps.exif_transpose(opened))
        return ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)


def write_jpeg(path, image):
    temporary = path.with_name('.' + path.name + '.tmp')
    image.save(temporary, format='JPEG', quality=94)
    temporary.replace(path)


def make_row(vehicle_id, records, weather_ids, labels, cell_width=580):
    vehicle = [r for r in records if r['vehicle_id'] == vehicle_id]
    by_weather = {r['weather_id']: r for r in vehicle}
    cell_height = round(cell_width * 3 / 4)
    gap, title_height, column_height = 12, 34, 32
    width = (cell_width + gap) * (len(weather_ids) + 1) + gap
    height = title_height + column_height + cell_height + gap
    sheet = Image.new('RGB', (width, height), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT_PATH, 20)
    draw.text((gap, 6), vehicle[0]['vehicle_label'], font=font, fill='#222222')

    original = next((Path(r['output_dir']) / 'source_working.png'
                     for r in vehicle if r['valid']), None)
    if original is None:
        original = resolve(read_json(resolve(vehicle[0]['config']))['input_image'])
    columns = [('Original', original, 'original')]
    for weather_id in weather_ids:
        record = by_weather.get(weather_id)
        image_path = (Path(record['output_dir']) / 'edited.png'
                      if record and record['valid'] else None)
        state = (record['validation_status'] if record and record['valid'] is False
                 else record['generation_status'] if record else 'not requested')
        columns.append((labels[weather_id], image_path, state))

    for index, (label, path, state) in enumerate(columns):
        x = gap + index * (cell_width + gap)
        y = title_height + column_height
        draw.text((x, title_height + 4), label, font=font, fill='#222222')
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill='#dddddd')
        if path is not None:
            thumb = load_thumbnail(path, (cell_width, cell_height))
            sheet.paste(thumb, (x + (cell_width - thumb.width) // 2,
                               y + (cell_height - thumb.height) // 2))
        else:
            draw.text((x + 12, y + 20), state.replace('_', ' '), font=font, fill='#555555')
    return sheet


def href(path, output):
    return quote(os.path.relpath(path, output).replace(os.sep, '/'), safe='/')


def write_html(output, batch, records, vehicle_ids, counts):
    esc = html.escape
    blocks = []
    for vehicle_id in vehicle_ids:
        vehicle = [r for r in records if r['vehicle_id'] == vehicle_id]
        cards = []
        for record in vehicle:
            result = Path(record['output_dir'])
            links = []
            for label, name in [('生成 PNG', 'edited.png'), ('零件对比', 'previews/details.jpg'),
                                ('运行记录', 'manifest.json')]:
                target = result / name
                if target.exists():
                    links.append(f'<a href="{href(target, output)}">{label}</a>')
            metrics = ''
            if record.get('inference_seconds') is not None:
                metrics += f"生成 {record['inference_seconds']:.2f} 秒；"
            if record.get('peak_allocated_gib') is not None:
                metrics += f"峰值已分配 {record['peak_allocated_gib']:.2f} GiB"
            validation = ('文件检查通过' if record['valid'] else '文件检查失败'
                          if record['valid'] is False else '尚未完成')
            reuse = '；复用既有结果' if record.get('reuse') else ''
            errors = ''.join(f'<li>{esc(e)}</li>' for e in record['errors'])
            cards.append(f'<article><h3>{esc(record["weather_label"])}</h3>'
                         f'<p>{esc(record["generation_status"])}；{validation}{reuse}</p>'
                         f'<p>{esc(metrics)}</p><p>{" · ".join(links)}</p>'
                         f'<ul>{errors}</ul></article>')
        blocks.append(f'<section><h2>{esc(vehicle[0]["vehicle_label"])}</h2>'
                      f'<a href="{vehicle_id}_comparison.jpg"><img loading="lazy" '
                      f'src="{vehicle_id}_comparison.jpg" alt="{esc(vehicle[0]["vehicle_label"])} 天气对比"></a>'
                      f'<div class="cards">{"".join(cards)}</div></section>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(batch['batch_id'])} — Qwen 天气编辑</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1600px;color:#222;background:#fafafa}}
img{{display:block;width:100%;height:auto}}a{{color:#175e9b}}section{{margin:36px 0}}h1{{font-size:1.8rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
article{{padding:0 16px;border:1px solid #ddd;border-radius:8px;background:white}}li{{overflow-wrap:anywhere}}</style></head>
<body><h1>{esc(batch['batch_id'])} — Qwen 天气编辑</h1>
<p>技术检查通过 {counts['valid']} / {counts['total']}；生成完成 {counts['completed']}；等待或运行中 {counts['pending_or_running']}；生成失败 {counts['generation_failed']}。</p>
<p>这里只核验文件、哈希、尺寸与运行记录，不自动评定车型保真或训练适用性。RGBA 仅在预览中合成到白底，原始 PNG 保留。</p>
<p><a href="overview.jpg">完整对比总览</a> · <a href="summary.json">检查与运行数据</a></p>
{''.join(blocks)}</body></html>
'''
    temporary = output / '.index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(output / 'index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    batch = read_json(args.batch_config.resolve())
    jobs = batch['jobs']
    if not jobs:
        raise ValueError('Batch must contain at least one job')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Job IDs must be unique')
    if len({(j['vehicle_id'], j['weather_id']) for j in jobs}) != len(jobs):
        raise ValueError('Each vehicle/weather combination must be unique')
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', job['vehicle_id']):
            raise ValueError('Vehicle IDs must be safe filename components')
    output = args.output_dir.resolve()
    protected = [PROJECT / 'test_images', PROJECT / 'models'] + [resolve(j['output_dir']) for j in jobs]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('Summary directory must be separate from inputs, weights, and individual results')
    output.mkdir(parents=True, exist_ok=True)
    records = [validate_job(job) for job in jobs]
    vehicle_ids = list(dict.fromkeys(j['vehicle_id'] for j in jobs))
    weather_ids = list(dict.fromkeys(j['weather_id'] for j in jobs))
    labels = {j['weather_id']: j['weather_label'] for j in jobs}
    rows = []
    for vehicle_id in vehicle_ids:
        row = make_row(vehicle_id, records, weather_ids, labels)
        write_jpeg(output / f'{vehicle_id}_comparison.jpg', row)
        rows.append(row)
    overview = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)), '#eeeeee')
    y = 0
    for row in rows:
        overview.paste(row, (0, y))
        y += row.height
    write_jpeg(output / 'overview.jpg', overview)
    counts = {
        'total': len(records),
        'completed': sum(r['generation_status'] == SUCCESS_STATUS for r in records),
        'valid': sum(r['valid'] is True for r in records),
        'validation_failed': sum(r['valid'] is False for r in records),
        'generation_failed': sum(r['generation_status'] == 'failed' for r in records),
        'pending_or_running': sum(r['generation_status'] not in (SUCCESS_STATUS, 'failed') for r in records),
        'reused': sum(bool(r.get('reuse')) for r in records),
        'png_files_checked': sum(r['png_files_checked'] for r in records),
    }
    summary = {
        'batch_id': batch['batch_id'], 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'batch_config': str(args.batch_config.resolve()),
        'batch_config_sha256': sha256(args.batch_config),
        'summary_script_sha256': sha256(__file__),
        'counts': counts, 'jobs': records,
        'visual_review': 'not_performed_by_this_tool',
        'preview_processing': 'RGBA composited on white for display only; originals untouched',
    }
    save_json(output / 'summary.json', summary)
    shutil.copyfile(__file__, output / 'summarize_batch_snapshot.py')
    write_html(output, batch, records, vehicle_ids, counts)
    print(json.dumps({'output_dir': str(output), 'counts': counts}, ensure_ascii=False))
    if counts['validation_failed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
