"""One managed pipeline: calibration captions/queue first, then rest; no automatic thresholds/K."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time
from filelock import FileLock
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/validation_calibration_v1'


def save(name,value):
    p=OUT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();deadline=started+12*3600
    with FileLock(OUT/'managed_worker.lock',timeout=0):
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',RAYON_NUM_THREADS='4')
        calibration=sum(p['role']=='calibration' for p in json.loads((OUT/'validation_photo_plan.json').read_text(encoding='utf-8')))
        steps=[('caption_preflight',['scripts/generate_stage_f_validation_captions.py','preflight']),
               ('calibration_captions',['scripts/generate_stage_f_validation_captions.py','generate','--max-new-photos',str(calibration)]),
               ('calibration_scoring',['scripts/stage_f_score_validation.py','--role','calibration']),
               ('human_check_queue',['scripts/build_stage_f_human_check.py']),
               ('remaining_preflight',['scripts/generate_stage_f_validation_captions.py','preflight']),
               ('remaining_validation_captions',['scripts/generate_stage_f_validation_captions.py','generate']),
               ('K_study_scoring_not_K_metrics',['scripts/stage_f_score_validation.py','--role','K_study'])]
        for phase,args in steps:
            if phase=='calibration_captions':
                completed=len(list((OUT/'caption_checkpoints').glob('*.json')))
                if completed>=calibration:continue
                args[-1]=str(calibration-completed)
            if phase=='calibration_scoring' and (OUT/'scores_calibration/scoring_report.json').exists():continue
            if phase=='human_check_queue' and (OUT/'human_check/queue_report.json').exists():continue
            if phase=='K_study_scoring_not_K_metrics' and (OUT/'scores_K_study/scoring_report.json').exists():continue
            log=OUT/(phase+'.log')
            with log.open('a',encoding='utf-8') as stream:
                process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                while process.poll() is None:
                    save('worker_progress.json',{'status':'running','phase':phase,'worker_pid':os.getpid(),'child_pid':process.pid,'completed_caption_photos':len(list((OUT/'caption_checkpoints').glob('*.json'))),'planned_caption_photos':2504,'human_queue_ready':(OUT/'human_check/queue_report.json').exists(),'elapsed_seconds':round(time.monotonic()-started,3),'thresholds_or_K_selected':False,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                    if time.monotonic()>deadline:
                        descendants=psutil.Process(process.pid).children(recursive=True)
                        for child in reversed(descendants):
                            try:child.terminate()
                            except psutil.NoSuchProcess:pass
                        process.terminate();process.wait(timeout=30)
                        save('worker_progress.json',{'status':'stopped_at_12h_deadline_checkpoints_retained','phase':phase,'thresholds_K_selected':False});return
                    time.sleep(15)
            if process.returncode:
                save('worker_progress.json',{'status':'stopped_on_actual_error_checkpoints_retained','phase':phase,'exit_code':process.returncode,'log':log.relative_to(ROOT).as_posix(),'no_automatic_method_change':True});return
        save('worker_progress.json',{'status':'FULL_VALIDATION_CAPTIONS_AND_SCORES_COMPLETE_WAITING_HUMAN_LABELS','completed_caption_photos':2504,'planned_caption_photos':2504,'human_queue_ready':True,'elapsed_seconds':round(time.monotonic()-started,3),'thresholds_K_selected':False,'teacher_training_rental_started':False})


if __name__=='__main__':main()
