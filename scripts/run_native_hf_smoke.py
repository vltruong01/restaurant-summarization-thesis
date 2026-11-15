"""Native Hugging Face download/resume, verify full weights, then standard from_pretrained smoke."""
from pathlib import Path
import os
os.environ['HF_HUB_DISABLE_XET']='1'  # Native HTTP resume preserves the seeded byte prefix.
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'  # This legacy accelerator discards incomplete files.
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='30'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import hashlib
import json
import threading
import time
import traceback
import requests
import yaml
from filelock import FileLock
from huggingface_hub import snapshot_download
from huggingface_hub.utils._http import reset_sessions
import blip2_caption_pilot as backend

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
WRITE_LOCK=threading.RLock()


def save(path,value):
    with WRITE_LOCK:
        temp=path.with_suffix('.native.tmp')
        temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        for attempt in range(5):
            try:temp.replace(path);return
            except PermissionError:
                if attempt==4:raise
                time.sleep(0.1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remaining-hours',type=float,required=True)
    args=parser.parse_args()
    assert 0<args.remaining_hours<=24
    started=time.monotonic()
    cfg=yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text())
    inventory=json.loads((OUT/'resume_20261006/checkpoint_download_inventory.json').read_text())
    revision=inventory['resolved_commit']
    cache=ROOT/cfg['model_cache']
    repo=cache/'models--Salesforce--blip2-opt-2.7b-coco'
    phase={'status':'starting_native_Hub_download_before_from_pretrained','controller_PID':os.getpid(),
           'transfer_mode':'HuggingFace_snapshot_download_native_HTTP_resume_then_from_pretrained',
           'model_id':cfg['model_id'],'resolved_commit':revision,'maximum_new_smoke_photos':1,
           'fixed_pilot_photos':24,'model_load_started':False,'caption_generation_started':False,
           'captions_generated':0,'research_parameters_changed':False,'remaining_deadline_hours':args.remaining_hours,
           'manual_caption_review':'pending','teacher_training_rental_started':False}
    progress={'status':'native_HuggingFace_download_running','model_id':cfg['model_id'],'resolved_commit':revision,
              'total_bytes':inventory['total_bytes'],'completed_bytes':0,'model_loads':0,'captions_generated':0,
              'process_id':os.getpid(),'weights_SHA256_verified':False,
              'partial_byte_counts_are_received_not_full_SHA_verified':True}
    stop=threading.Event()
    download_active=threading.Event();download_active.set()
    def refresh():
        received=0
        for spec in inventory['weight_files']:
            candidates=[repo/'snapshots'/revision/spec['filename'],repo/'blobs'/spec['sha256'],repo/'blobs'/(spec['sha256']+'.incomplete')]
            sizes=[]
            for path in candidates:
                try:sizes.append(path.stat().st_size)
                except FileNotFoundError:pass # Native Hub moves blobs atomically on completion.
            received+=min(spec['size_bytes'],max(sizes or [0]))
        progress.update(completed_bytes=received,elapsed_seconds=round(time.monotonic()-started,2))
        save(OUT/'weight_download_progress.json',progress)
    def monitor():
        while not stop.wait(15):
            if download_active.is_set():refresh()
    def deadline():
        if not download_active.is_set():return
        progress['status']='native_download_deadline_paused_resumable'
        phase['status']='download_deadline_paused_no_model_load'
        refresh();save(OUT/'local_smoke_controller_report.json',phase)
        os._exit(2)
    timer=threading.Timer(args.remaining_hours*3600,deadline);timer.daemon=True
    thread=threading.Thread(target=monitor,daemon=True)
    try:
        with FileLock(OUT/'local_smoke_controller.lock',timeout=0):
            backend.ORIGIN='agent'
            backend.check_sources(backend.read(OUT/'pilot_identity.json'))
            save(OUT/'local_smoke_controller_report.json',phase)
            refresh();timer.start();thread.start()
            print('Native Hugging Face downloader; HTTP resume enabled, no custom chunk requests.',flush=True)
            attempt=0
            while True:
                attempt+=1
                try:
                    phase.update(status='downloading_native_Hub_before_from_pretrained',network_attempt=attempt)
                    save(OUT/'local_smoke_controller_report.json',phase)
                    snapshot=snapshot_download(cfg['model_id'],revision=revision,cache_dir=cache,
                               allow_patterns=['*.json','*.txt','*.safetensors'],max_workers=2)
                    break
                except (requests.exceptions.ConnectionError,requests.exceptions.Timeout,requests.exceptions.ChunkedEncodingError) as error:
                    reset_sessions() # Fresh connection pools after a network change/broken response.
                    backoff=min(300,30*attempt)
                    phase.update(status='retrying_native_network_error_resume_preserved',network_attempt=attempt,
                                 last_network_error_type=type(error).__name__,retry_backoff_seconds=backoff)
                    progress.update(status='native_network_retry_pending_resume_preserved',network_retry_count=attempt)
                    refresh();save(OUT/'local_smoke_controller_report.json',phase)
                    print(f'Network interruption {type(error).__name__}; resume cached bytes after {backoff}s (original deadline unchanged).',flush=True)
                    stop.wait(backoff)
                    progress['status']='native_HuggingFace_download_running'
            download_active.clear();timer.cancel();stop.set();thread.join(timeout=2)
            phase['status']='native_weights_received_verifying_full_LFS_SHA256'
            save(OUT/'local_smoke_controller_report.json',phase)
            for spec in inventory['weight_files']:
                path=Path(snapshot)/spec['filename']
                assert path.stat().st_size==spec['size_bytes'] and backend.sha(path)==spec['sha256'],'Native full weight checksum mismatch'
            progress.update(status='complete_weights_verified_not_model_load',weights_SHA256_verified=True)
            refresh()
            env=backend.preflight(cfg,OUT)  # Fresh after transfer, not a stale pre-download resource budget.
            phase.update(status='loading_standard_from_pretrained_for_one_photo',model_load_started=True,
                         available_RAM_GiB_at_fresh_preflight=env['available_RAM_GiB'])
            save(OUT/'local_smoke_controller_report.json',phase)
            result=backend.generate(cfg,OUT,allow_download=False,max_new_photos=1)
            assert result['completed']==1
            plan=backend.read(OUT/'pilot_photo_plan.json')
            record=backend.read(OUT/'caption_checkpoints'/(plan[0]['photo_id']+'.json'))
            phase.update(status='one_photo_generated_pending_manual_review',caption_generation_started=True,
                         captions_generated=1,raw_caption=record['raw_caption'],first_photo_id=record['photo_id'],
                         peak_GPU_allocated_GiB=max(a['peak_GPU_allocated_GiB'] for a in record['attempts']),
                         model_released_after_smoke=True)
            save(OUT/'local_smoke_controller_report.json',phase)
            print(json.dumps(phase,indent=2),flush=True)
            return 0
    except Exception as error:
        phase.update(status='blocked_native_download_or_smoke_no_automatic_method_change',
                     error_type=type(error).__name__,error=str(error).split('?')[0][:800],
                     captions_generated=len(list((OUT/'caption_checkpoints').glob('*.json'))))
        save(OUT/'local_smoke_controller_report.json',phase)
        if not progress['weights_SHA256_verified']:
            progress.update(status='native_download_stopped_error_resume_preserved',worker_active=False,error_type=type(error).__name__)
            refresh()
        print(json.dumps(phase,indent=2),flush=True)
        return 1
    finally:
        timer.cancel();stop.set()


if __name__=='__main__':raise SystemExit(main())
