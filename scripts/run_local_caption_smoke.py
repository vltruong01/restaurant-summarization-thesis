"""Managed local-first E0: resumable verified weights, fresh preflight, one photo, exit."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'


def save(value):
    path=OUT/'local_smoke_controller_report.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--download-deadline-hours',type=float,default=24)
    args=parser.parse_args()
    assert 1<=args.workers<=16 and 0<args.download_deadline_hours<=24
    started=time.monotonic()
    cfg=yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))
    import blip2_caption_pilot as backend
    import download_caption_checkpoint as transfer
    backend.ORIGIN='agent'
    report={'status':'starting_local_one_photo_smoke','execution_origin':'agent',
            'controller_PID':os.getpid(),'fixed_pilot_photos':24,'maximum_new_smoke_photos':1,
            'workers_in_single_download_process':args.workers,
            'model_load_started':False,'caption_generation_started':False,
            'captions_generated':0,'full_24_photo_pilot_complete':False,
            'manual_caption_review':'pending','teacher_training_rental_started':False,
            'download_deadline_hours':args.download_deadline_hours,'research_parameters_changed':False}
    report['transfer_mode']='continuous_until_complete_or_remaining_deadline_per_chunk_checkpointing'
    save(report)
    try:
        with FileLock(OUT/'local_smoke_controller.lock',timeout=0):
            backend.check_sources(backend.read(OUT/'pilot_identity.json'))
            previous=backend.read(OUT/'generation_report.json') if (OUT/'generation_report.json').exists() else {}
            if previous.get('completed',0)>=1:
                report.update(status='existing_caption_outputs_pending_agent_review',
                              captions_generated=previous['completed'],
                              caption_generation_started=True)
                save(report)
                print(json.dumps(report,indent=2),flush=True)
                return 0
            report['status']='downloading_verified_weights_before_one_photo'
            save(report)
            deadline=started+args.download_deadline_hours*3600
            while time.monotonic()<deadline:
                remaining=int(deadline-time.monotonic())
                if remaining<1:
                    break
                original_argv=sys.argv
                try:
                    sys.argv=['download_caption_checkpoint.py','--workers',str(args.workers),
                              '--max-seconds',str(remaining)]
                    code=transfer.main()
                finally:
                    sys.argv=original_argv
                if code==0:
                    break
                report.update(status='downloading_resumable_weights_before_one_photo',
                              elapsed_seconds=round(time.monotonic()-started,2))
                save(report)
            else:
                code=2
            progress=backend.read(OUT/'weight_download_progress.json')
            if not progress.get('weights_SHA256_verified'):
                report.update(status='download_deadline_paused_resumable_no_model_load',
                              elapsed_seconds=round(time.monotonic()-started,2))
                save(report)
                return 2
            report.update(status='weights_verified_checking_fresh_RAM_CUDA_preflight')
            save(report)
            env=backend.preflight(cfg,OUT)
            report.update(status='loading_model_for_one_photo',
                          RAM_GiB_at_fresh_preflight=env['available_RAM_GiB'],
                          model_load_started=True)
            save(report)
            result=backend.generate(cfg,OUT,allow_download=False,max_new_photos=1)
            assert result['completed']==1 and result['planned']==24
            plan=backend.read(OUT/'pilot_photo_plan.json')
            first=backend.read(OUT/'caption_checkpoints'/(plan[0]['photo_id']+'.json'))
            report.update(status='one_photo_generated_pending_agent_picture_caption_review',
                          caption_generation_started=True,captions_generated=1,
                          first_photo_id=first['photo_id'],first_business=first['name'],
                          raw_caption=first['raw_caption'],surface_eligible=first['surface_eligible'],
                          attempts=len(first['attempts']),
                          caption_seconds=sum(a['seconds'] for a in first['attempts']),
                          peak_GPU_allocated_GiB=max(a['peak_GPU_allocated_GiB'] for a in first['attempts']),
                          elapsed_seconds=round(time.monotonic()-started,2),
                          model_released_after_smoke=True)
            save(report)
            print(json.dumps(report,indent=2),flush=True)
            return 0
    except Exception as error:
        report.update(status='blocked_no_automatic_model_or_research_change',
                      error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc(),elapsed_seconds=round(time.monotonic()-started,2))
        save(report)
        print(json.dumps(report,indent=2),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
