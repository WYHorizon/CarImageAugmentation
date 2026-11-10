"""Run a fixed local Qwen experiment queue sequentially, preserving every result."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run import PROJECT, METHOD, configure_environment, save_json, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def completed(output, config):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete_pending_visual_review':
        raise RuntimeError(f'Existing result is incomplete; preserve and inspect it: {output}')
    if manifest['config'] != config or manifest['completed_steps'] != config['num_inference_steps']:
        raise RuntimeError(f'Existing result has different settings: {output}')
    if manifest['source_sha256'] != sha256(PROJECT / config['input_image']):
        raise RuntimeError(f'Source has changed: {output}')
    for name, expected in manifest['output_sha256'].items():
        if sha256(output / name) != expected:
            raise RuntimeError(f'Output hash mismatch: {output / name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    configure_environment()
    batch = json.loads(args.batch_config.read_text(encoding='utf-8'))
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(METHOD / 'outputs'):
        raise ValueError('Batch records must stay inside the method outputs directory')
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.batch.lock').open('a', encoding='utf-8')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = destination / 'batch_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8')) != batch:
        raise ValueError('Batch directory already belongs to a different configuration')
    save_json(snapshot, batch)
    shutil.copyfile(__file__, destination / 'batch_runner_snapshot.py')
    logs = destination / 'logs'
    logs.mkdir(exist_ok=True)
    state = {'batch_id': batch['batch_id'], 'started_utc': now(), 'pid': os.getpid(),
             'gpu': args.gpu, 'status': 'running', 'total': len(batch['jobs']),
             'finished': 0, 'jobs': [], 'run_script_sha256': sha256(METHOD / 'run.py')}
    state_path = destination / 'batch_status.json'
    save_json(state_path, state)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    try:
        for job in batch['jobs']:
            config_path = PROJECT / job['config']
            config = json.loads(config_path.read_text(encoding='utf-8'))
            output = PROJECT / job['output_dir']
            state['current_job'] = job['id']
            state['updated_utc'] = now()
            save_json(state_path, state)
            if output.exists():
                manifest = completed(output, config)
                execution = 'reused' if job.get('reuse') else 'already_complete'
            else:
                if job.get('reuse'):
                    raise FileNotFoundError(f'Required previous result is missing: {output}')
                # A single run previously reserved 76.27 GiB; wait for an idle 80 GB card.
                waiting_started = time.monotonic()
                while True:
                    free = int(subprocess.check_output([
                        'nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'], text=True).strip())
                    if free >= 80000:
                        break
                    state.update(status='waiting_for_gpu', free_memory_mib=free, updated_utc=now())
                    save_json(state_path, state)
                    print(f'Waiting for GPU {args.gpu}: {free} MiB free', flush=True)
                    if time.monotonic() - waiting_started > 1800:
                        raise RuntimeError('GPU did not become idle within 30 minutes')
                    time.sleep(30)
                state.update(status='running', updated_utc=now())
                command = [sys.executable, '-u', str(METHOD / 'run.py'),
                           '--config', str(config_path), '--output-dir', str(output)]
                log_path = logs / (job['id'] + '.log')
                if log_path.exists():
                    raise FileExistsError(f'Log exists without a reusable result: {log_path}')
                print(f'START {job["id"]}', flush=True)
                with log_path.open('x', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid, current_log=str(log_path), updated_utc=now())
                    save_json(state_path, state)
                    result = child.wait()
                state.pop('child_pid', None)
                if result:
                    raise RuntimeError(f'Job {job["id"]} failed with exit {result}; see {log_path}')
                manifest = completed(output, config)
                execution = 'generated'
            state['jobs'].append({'id': job['id'], 'execution': execution,
                                  'output_dir': str(output),
                                  'inference_seconds': manifest['inference_seconds'],
                                  'peak_allocated_gib': manifest['peak_allocated_gib']})
            state['finished'] += 1
            state['updated_utc'] = now()
            save_json(state_path, state)
            print(f'DONE {state["finished"]}/{state["total"]} {job["id"]} ({execution})', flush=True)
        state.update(status='complete_pending_visual_review', completed_utc=now())
        state.pop('current_job', None)
        state.pop('current_log', None)
        save_json(state_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), updated_utc=now())
        save_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
