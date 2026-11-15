"""One resumable managed production job; stop at validated Stage F, never teacher/training."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
import psutil
from filelock import FileLock
from stage_f_verification_workflow import read,write,sha
from stage_f_d033_protocol import freeze_method

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_v1'
GEN=OUT/'generation_train_test'


def status(phase,state='running',**extra):
    completed=len(list((GEN/'caption_checkpoints').glob('*.json')))
    write(OUT/'worker_progress.json',{'status':state,'phase':phase,'worker_pid':os.getpid(),
          'new_photo_checkpoints':completed,'new_photo_plan':27511,'compatible_reused_captions':2516,
          'total_generated_captions_including_reused':completed+2516,'total_photo_pool':30027,
          'remaining_new_photos':27511-completed,'K':5,'teacher_calls':False,'LED_training':False,
          'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})


def execute(phase,args,env):
    with (OUT/(phase+'.stdout.log')).open('a',encoding='utf-8') as stdout,(OUT/(phase+'.stderr.log')).open('a',encoding='utf-8') as stderr:
        process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while process.poll() is None:
            status(phase,child_pid=process.pid);time.sleep(15)
        if process.returncode:raise RuntimeError(f'{phase} failed exit{process.returncode}; inspect {phase}.stderr.log; no automatic method change')


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-generator-pid',type=int);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(OUT/'managed_worker.lock',timeout=0):
        freeze_method()
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',RAYON_NUM_THREADS='4')
        try:
            seal=read(OUT/'production_adapter_seal.json')
            for relative,digest in seal.items():assert sha(ROOT/relative)==digest,'Production adapter/config changed; stop for review'
            if (OUT/'completion_report.json').exists():
                complete=read(OUT/'completion_report.json')
                for relative,digest in complete['artifact_sha256'].items():assert sha(OUT/relative)==digest
                status('complete','STAGE_F_PRODUCTION_COMPLETE_VALIDATED_STOP_BEFORE_STAGE_G');return
            if a.wait_generator_pid:
                process=psutil.Process(a.wait_generator_pid)
                assert 'generate_stage_f_production_d033.py' in ' '.join(process.cmdline()),'Refuse to attach an unrelated process'
                created=process.create_time()
                while True:
                    try:
                        if not process.is_running() or process.create_time()!=created:break
                    except psutil.NoSuchProcess:break
                    status('generation_train_test',attached_generator_pid=process.pid)
                    time.sleep(15)
                assert read(GEN/'generation_report.json')['completed']==27511,'Generator exited incomplete; no automatic method change'
            else:
                execute('preflight',['scripts/generate_stage_f_production_d033.py','preflight'],env)
                execute('generation_train_test',['scripts/generate_stage_f_production_d033.py','generate'],env)
            execute('production_scoring',['scripts/construct_stage_f_production_d033.py','score'],env)
            execute('fixed_review_inputs',['scripts/prepare_stage_f_production_reviews_d033.py','--workers','2'],env)
            execute('production_construction',['scripts/construct_stage_f_production_d033.py','construct'],env)
            complete=read(OUT/'completion_report.json')
            assert complete['status']=='STAGE_F_PRODUCTION_COMPLETE_VALIDATED_STOP_BEFORE_STAGE_G'
            status('complete',complete['status'],completion_report='completion_report.json')
        except Exception as error:
            status('blocked','STOP_ON_PRODUCTION_ERROR_CHECKPOINTS_RETAINED_NO_METHOD_CHANGE',error=str(error))
            raise


if __name__=='__main__':main()
