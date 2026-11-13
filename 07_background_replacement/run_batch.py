"""Run a configured background/vehicle matrix sequentially on the selected GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-config', type=Path, default=METHOD / 'configs/background_3x4_v1.json')
    parser.add_argument('--output-dir', type=Path, default=METHOD / 'outputs/background_3x4_v1')
    parser.add_argument('--check-config', action='store_true', help='Validate the batch inventory without inference or creating outputs')
    args = parser.parse_args()
    batch_path = args.batch_config.resolve()
    config = json.loads(batch_path.read_text(encoding='utf-8'))
    jobs = config['jobs']
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('A nonempty jobs list is required')
    ids = [job['id'] for job in jobs]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-z0-9_]+', value) for value in ids):
        raise ValueError('Job IDs must be unique lowercase identifiers')
    paths = []
    protected_files = [batch_path, Path(__file__).resolve(), METHOD / 'run.py']
    protected_dirs = [batch_path.parent]
    model_dirs = []
    for job in jobs:
        path = (PROJECT / job['config']).resolve()
        single = json.loads(path.read_text(encoding='utf-8'))
        prompt_path = PROJECT / single['prompt_provenance']['readable_copy']
        if single['prompt'] != prompt_path.read_text(encoding='utf-8').rstrip('\n'):
            raise ValueError(f'Prompt copy differs from config: {job["id"]}')
        sources = [(PROJECT / single[key]).resolve() for key in ('input_image', 'reference_image', 'scene_config')]
        if not all(source.is_file() for source in sources):
            raise FileNotFoundError(f'Missing input for {job["id"]}')
        protected_files.extend([path, prompt_path.resolve(), *sources])
        protected_dirs.extend([path.parent, prompt_path.resolve().parent, *(source.parent for source in sources)])
        model_dirs.append((PROJECT / single['model_dir']).resolve())
        paths.append(path)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Batch output already exists; choose a new directory: {output}')
    if any(path.is_relative_to(output) for path in protected_files) or any(output.is_relative_to(directory) for directory in protected_dirs):
        raise ValueError('Output must not overlap source/configuration directories')
    if any(output.is_relative_to(model) or model.is_relative_to(output) for model in model_dirs):
        raise ValueError('Output must not overlap model weights')
    if args.check_config:
        print(json.dumps({'jobs': len(jobs), 'job_ids': ids, 'output': str(output)}, indent=2))
        return

    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    (output / 'logs').mkdir()
    shutil.copy2(__file__, output / 'run_batch_snapshot.py')
    save_json(output / 'batch_config.json', config)
    started = time.perf_counter()
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'batch_config_sha256': sha256(batch_path), 'driver_sha256': sha256(__file__),
        'single_runner_sha256': sha256(METHOD / 'run.py'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'job_count': len(jobs), 'completed_jobs': 0, 'failed_jobs': 0, 'jobs': [],
        'visual_acceptance': 'pending', 'classification_evaluation_performed': False,
    }
    manifest_path = output / 'batch_manifest.json'
    save_json(manifest_path, manifest)
    for number, (job, path) in enumerate(zip(jobs, paths), 1):
        target = output / 'jobs' / job['id']
        log = output / 'logs' / f'{job["id"]}.log'
        command = [sys.executable, str(METHOD / 'run.py'), '--config', str(path), '--output-dir', str(target)]
        item = {**job, 'status': 'running', 'config_sha256': sha256(path), 'output': str(target.relative_to(output)), 'log': str(log.relative_to(output))}
        manifest['jobs'].append(item)
        save_json(manifest_path, manifest)
        print(f'[{number}/{len(jobs)}] START {job["id"]}', flush=True)
        begin = time.perf_counter()
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        item['elapsed_seconds'] = time.perf_counter() - begin
        item['returncode'] = result.returncode
        if result.returncode == 0:
            single_manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
            if single_manifest['status'] != 'complete_pending_visual_review':
                raise RuntimeError(f'Unexpected final state for {job["id"]}')
            item['status'] = 'complete_pending_visual_review'
            item['edited_sha256'] = sha256(target / 'edited.png')
            manifest['completed_jobs'] += 1
        else:
            item['status'] = 'failed'
            manifest['failed_jobs'] += 1
        manifest['total_seconds'] = time.perf_counter() - started
        save_json(manifest_path, manifest)
        print(f'[{number}/{len(jobs)}] {item["status"]} {job["id"]} ({item["elapsed_seconds"]:.1f}s)', flush=True)
    manifest['status'] = 'partial_failure' if manifest['failed_jobs'] else 'complete_pending_visual_review'
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['total_seconds'] = time.perf_counter() - started
    save_json(manifest_path, manifest)
    print(json.dumps({'status': manifest['status'], 'completed': manifest['completed_jobs'], 'failed': manifest['failed_jobs']}), flush=True)
    if manifest['failed_jobs']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
