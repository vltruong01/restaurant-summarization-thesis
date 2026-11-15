"""Run the approved immutable D060 train/evaluate runner once; never retry."""
import platform,subprocess,sys,time,traceback
from stage_i_d060 import *

AUDIT=ROOT/'outputs/stage_i/d060_execution_v1'
BINDINGS={'config_sha256':'4513dc202e72f8d2589408e573ecf21dba0e8a409950f4c69f95b1a689a22c01',
    'preflight_manifest_sha256':'c27796dfb001d38190a56f3888a97c6fabf9dbd39ee689bed4b73400f0a32080',
    'runner_sha256':'c00f74a61b92fcafd106e2b5cb656865eead232a05a63a70aeb739bbe1fd83c1'}

def now():return datetime.now(timezone.utc).isoformat()
def verify():
    actual={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER)}
    require(actual==BINDINGS,'Approved binding mismatch');return actual

def main():
    require(sys.argv[1:]==['--execute'],'Explicit --execute required')
    approval_path=AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json'
    require(approval_path.exists(),'Record researcher approval first')
    require(not (AUDIT/'ORCHESTRATION_STARTED.json').exists(),'Already invoked; no retry')
    write(AUDIT/'ORCHESTRATION_STARTED.json',{'UTC':now(),'orchestrator_sha256':sha(Path(__file__)),'no_retry':True})
    started=time.perf_counter()
    try:
        verify();approval=gate(approval_path,'train');cfg,m=preflight()
        require(len(m['files'])==426,'Bound file count differs')
        write(AUDIT/'ACTUAL_ENVIRONMENT_PREFLIGHT.json',{'status':'PASS','UTC':now(),'three_bindings':verify(),
            'bound_files_checked':len(m['files']),'runtime_pins':cfg['runtime_pins'],'python_executable':sys.executable,
            'platform':platform.platform(),'execution_authorized_at_utc':approval['execution_authorized_at_utc'],
            'viability_deadline_utc':approval['viability_deadline_utc'],'GPU_execution_so_far':False})
        print('D060 explicit approval recorded; three SHA256 and426 file bindings PASS.',flush=True)
        for mode in ('train','evaluate'):
            verify();gate(approval_path,mode);preflight(allow_existing_run=mode=='evaluate')
            if mode=='evaluate':
                p=ROOT/cfg['run_root']/'train'
                require(read(p/'COMPLETION.json')['steps']==750,'Training incomplete')
                require(read(p/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Backbone changed')
                require(not (p/'FAILURE_STOP_NO_RETRY.json').exists(),'Training failure present')
            cmd=[sys.executable,str(RUNNER),mode,'--execute','--approval',str(approval_path)]
            write(AUDIT/(mode.upper()+'_INVOCATION_INTENT.json'),{'command':cmd,'UTC':now(),'invocation':1,'bindings':verify(),'no_retry':True})
            print('Invoking approved D060 '+mode+' exactly once.',flush=True);t=time.perf_counter()
            with (AUDIT/(mode.upper()+'_STDOUT.log')).open('x',encoding='utf-8') as out,(AUDIT/(mode.upper()+'_STDERR.log')).open('x',encoding='utf-8') as err:
                result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err)
            write(AUDIT/(mode.upper()+'_PROCESS_EXIT.json'),{'returncode':result.returncode,'UTC':now(),'seconds':time.perf_counter()-t,'invocations':1,'retry_invocations':0,'bindings_after':verify()})
            if result.returncode:
                write(AUDIT/'STOP_NO_RETRY.json',{'failed_mode':mode,'returncode':result.returncode,'UTC':now(),'automatic_retry':False,'further_GPU_execution':False})
                print('STOP: '+mode+' failed. No retry/fallback/next mode.',flush=True);return result.returncode
        p=ROOT/cfg['run_root']/'evaluate'
        require(read(p/'COMPLETION.json')['generation_calls']==48,'Unexpected generation count')
        require(read(p/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Evaluation changed backbone')
        write(AUDIT/'ORCHESTRATION_COMPLETION.json',{'status':'TRAIN_AND48_GENERATIONS_COMPLETE_STOP_FOR_REVIEW','UTC':now(),'total_seconds':time.perf_counter()-started,'train_invocations':1,'evaluation_invocations':1,'retry_invocations':0,'RO_U_started':False})
        print('D060 approved GPU scope complete; stop for independent quality review.',flush=True);return 0
    except Exception as exc:
        write(AUDIT/'ORCHESTRATION_FAILURE_STOP.json',{'error':str(exc),'traceback':traceback.format_exc(),'UTC':now(),'no_retry':True})
        raise

if __name__=='__main__':sys.exit(main())
