"""Resumable, SHA256-verified public range transfer for the pinned E0 checkpoint.

One process with bounded I/O workers; no torch/model imports or inference.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import threading
import time
import requests
import urllib3.util.connection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
GIB = 2**30


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--chunk-MiB', type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.workers <= 16 and args.chunk_MiB == 4 and args.max_seconds > 0
    urllib3.util.connection.HAS_IPV6 = False  # Only this transfer process.
    inventory = json.loads((OUT/'resume_20261006/checkpoint_download_inventory.json').read_text())
    revision = json.loads((OUT/'model_revision.json').read_text())['resolved_commit']
    assert inventory['resolved_commit'] == revision and len(inventory['weight_files']) == 2
    assert inventory['model_id'] == 'Salesforce/blip2-opt-2.7b-coco'
    base = ROOT/'assets/model_cache'
    target = base/'models--Salesforce--blip2-opt-2.7b-coco'/'snapshots'/revision
    staging = base/'range_downloads'/revision
    staging.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if __import__('shutil').disk_usage(base).free < 40*GIB:
        raise RuntimeError('Less than40GiB free; no transfer started.')
    chunk_size = args.chunk_MiB*2**20
    started = time.monotonic()
    deadline = started+args.max_seconds
    mutex = threading.Lock()
    stop = threading.Event()
    state_path = staging/'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'resolved_commit': revision, 'chunk_bytes': chunk_size, 'files': {}}
    assert state['resolved_commit'] == revision and state['chunk_bytes'] == chunk_size
    for spec in inventory['weight_files']:
        name = spec['filename']
        assert Path(name).name == name and name.endswith('.safetensors')
        assert len(spec['sha256']) == 64
        item = state['files'].setdefault(name, {'size_bytes': spec['size_bytes'], 'sha256': spec['sha256'], 'chunks': {}})
        assert item['sha256'] == spec['sha256'] and item['size_bytes'] == spec['size_bytes']
        final = target/name
        if final.exists():
            assert final.stat().st_size == spec['size_bytes'] and digest(final) == spec['sha256']
            item['complete'] = True
            continue
        part = staging/(name+'.part')
        part.touch(exist_ok=True)
        # Resume only chunk content actually verified on disk, not a file-size guess.
        with part.open('rb') as stream:
            for number, record in list(item['chunks'].items()):
                stream.seek(int(number)*chunk_size)
                block = stream.read(record['bytes'])
                if len(block) != record['bytes'] or hashlib.sha256(block).hexdigest() != record['sha256']:
                    del item['chunks'][number]
    tasks = []
    for spec in inventory['weight_files']:
        item = state['files'][spec['filename']]
        if item.get('complete'):
            continue
        for number in range((spec['size_bytes']+chunk_size-1)//chunk_size):
            if str(number) not in item['chunks']:
                tasks.append((spec, number))
    initial_bytes = sum(sum(r['bytes'] for r in f['chunks'].values()) if not f.get('complete') else f['size_bytes']
                        for f in state['files'].values())
    progress = {'status': 'downloading_pinned_weights_only', 'model_id': inventory['model_id'],
                'resolved_commit': revision, 'workers_in_one_process': args.workers,
                'model_loads': 0, 'captions_generated': 0, 'total_bytes': inventory['total_bytes'],
                'completed_bytes': initial_bytes, 'failed_chunks': 0, 'process_id': os.getpid(),
                'max_seconds_this_run': args.max_seconds, 'weights_SHA256_verified': False}
    def persist():
        progress['elapsed_seconds'] = round(time.monotonic()-started, 2)
        transferred = progress['completed_bytes']-initial_bytes
        progress['this_run_MiB_per_second'] = round(transferred/max(0.001, progress['elapsed_seconds'])/2**20, 3)
        write_json(state_path, state)
        write_json(OUT/'weight_download_progress.json', progress)
    persist()
    heartbeat_stop = threading.Event()
    def heartbeat():
        while not heartbeat_stop.wait(15):
            with mutex:
                persist()
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    print(f"One transfer process PID{os.getpid()}, {len(tasks)} missing4MiB chunks, {args.workers} I/O workers.", flush=True)

    def transfer(spec, number):
        start = number*chunk_size
        end = min(spec['size_bytes'], start+chunk_size)-1
        expected = end-start+1
        part = staging/(spec['filename']+'.part')
        last_error = None
        for attempt in range(3):
            if stop.is_set() or time.monotonic() >= deadline:
                return False
            url = (f"https://huggingface.co/{inventory['model_id']}/resolve/{revision}/{spec['filename']}"
                   f"?download=true&part={number}&attempt={attempt}")
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                                  stream=True, timeout=(10, 30)) as response:
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{spec["size_bytes"]}':
                        raise ValueError('Unexpected status/range; refused content')
                    h = hashlib.sha256()
                    received = 0
                    with part.open('r+b') as stream:
                        stream.seek(start)
                        for block in response.iter_content(64*1024):
                            if stop.is_set() or time.monotonic() >= deadline:
                                return False
                            received += len(block)
                            if received > expected:
                                raise ValueError('Oversized range')
                            stream.write(block)
                            h.update(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if received != expected:
                        raise ValueError('Incomplete range')
                with mutex:
                    state['files'][spec['filename']]['chunks'][str(number)] = {'bytes': received, 'sha256': h.hexdigest()}
                    progress['completed_bytes'] += received
                    persist()
                return True
            except Exception as error:
                last_error = type(error).__name__  # Never print signed URLs/auth headers.
                time.sleep(min(2*(attempt+1), max(0, deadline-time.monotonic())))
        with mutex:
            progress['failed_chunks'] += 1
            progress['last_error_type'] = last_error
            persist()
        return False

    next_report = started
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(transfer, spec, number) for spec, number in tasks]
            for future in as_completed(futures):
                future.result()
                if time.monotonic() >= deadline:
                    stop.set()
                if time.monotonic() >= next_report:
                    with mutex:
                        print(f"{progress['completed_bytes']/GIB:.3f}/{inventory['total_bytes']/GIB:.3f}GiB, {progress['this_run_MiB_per_second']}MiB/s; model not loaded.", flush=True)
                    next_report = time.monotonic()+30
        complete = True
        for spec in inventory['weight_files']:
            item = state['files'][spec['filename']]
            if item.get('complete'):
                continue
            expected_chunks = (spec['size_bytes']+chunk_size-1)//chunk_size
            if len(item['chunks']) != expected_chunks:
                complete = False
                continue
            part = staging/(spec['filename']+'.part')
            assert part.stat().st_size == spec['size_bytes'] and digest(part) == spec['sha256'], 'Full weight SHA256 mismatch'
            part.replace(target/spec['filename'])  # One local copy, same filesystem.
            item['complete'] = True
        progress.update(status='complete_weights_verified_not_model_load' if complete else 'paused_resumable_not_model_load',
                        weights_SHA256_verified=complete,
                        completed_shards=sum(bool(f.get('complete')) for f in state['files'].values()))
        persist()
        print(json.dumps(progress, indent=2), flush=True)
        return 0 if complete else 2
    finally:
        stop.set()
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)


if __name__ == '__main__':
    raise SystemExit(main())
