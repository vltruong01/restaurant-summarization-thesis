"""Exactly-once execution of the researcher-approved D059 gate -> F -> 48 outputs."""
import os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from stage_i_d059 import *

AUDIT=ROOT/'outputs/stage_i/d059_execution_v1'
BINDINGS={'config_sha256':'48180270358624d2301207c9ca424772a5969a1c59232d7d9da8ae65755eb908',
    'preflight_manifest_sha256':'8bb54fdd407bb6ea5f2a6184052763c5cfb58fb90f2f8fc27db4f78866e6856d',
    'runner_sha256':'c19f754fcc97e37d06b8d29049d1d4acc4753cd3cb631abc62457f9473b1375c'}

def now():return datetime.now(timezone.utc).isoformat()

def verify_three():
    actual={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER)}
    require(actual==BINDINGS,'Approved three SHA256 bindings differ')
    return actual

def main():
    require(sys.argv[1:]==['--execute'],'Explicit --execute required')
    ap=AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json'
    require(ap.exists(),'Authorization must be recorded before orchestration')
    require(not (AUDIT/'ORCHESTRATION_STARTED.json').exists(),'Exactly-once orchestration already started; no retry')
    write(AUDIT/'ORCHESTRATION_STARTED.json',{'UTC':now(),'orchestrator_sha256':sha(Path(__file__)),'no_retry':True})
    started=time.perf_counter()
    try:
        verify_three();approval=gate(ap,'feasibility');cfg,m=preflight()
        write(AUDIT/'ACTUAL_ENVIRONMENT_PREFLIGHT.json',{'status':'PASS','UTC':now(),'three_bindings':verify_three(),
            'bound_files_checked':len(m['files']),'runtime_pins':cfg['runtime_pins'],'python_executable':sys.executable,
            'python':sys.version,'platform':platform.platform(),'execution_authorized_at_utc':approval['execution_authorized_at_utc'],
            'improvement_deadline_utc':approval['improvement_deadline_utc'],'GPU_execution_so_far':False})
        print('D059 explicit authorization recorded. All3 hashes and76 integrity bindings PASS.',flush=True)
        for mode in ('feasibility','train','evaluate'):
            verify_three();approval=gate(ap,mode);preflight(allow_existing_run=mode!='feasibility')
            if mode!='feasibility':require_feasibility(cfg,approval)
            cmd=[sys.executable,str(RUNNER),mode,'--execute','--approval',str(ap)]
            write(AUDIT/(mode.upper()+'_INVOCATION_INTENT.json'),{'command':cmd,'UTC':now(),'invocation':1,'no_retry':True,'bindings':verify_three()})
            print('Invoking frozen D059 '+mode+' exactly once.',flush=True);t=time.perf_counter()
            with (AUDIT/(mode.upper()+'_STDOUT.log')).open('x',encoding='utf-8') as out,(AUDIT/(mode.upper()+'_STDERR.log')).open('x',encoding='utf-8') as err:
                result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err)
            write(AUDIT/(mode.upper()+'_PROCESS_EXIT.json'),{'returncode':result.returncode,'UTC':now(),'seconds':time.perf_counter()-t,
                'invocations':1,'retry_invocations':0,'bindings_after':verify_three()})
            if result.returncode:
                write(AUDIT/'STOP_NO_RETRY.json',{'failed_mode':mode,'returncode':result.returncode,'UTC':now(),
                    'further_GPU_execution':False,'automatic_retry':False,'total_seconds':time.perf_counter()-started})
                print('STOP: '+mode+' failed; no retry, fallback or further mode.',flush=True)
                return result.returncode
            if mode=='feasibility':
                r=require_feasibility(cfg,approval)
                require(r['optimizer_updates']==3 and r['microbatches']==24 and r['generation_calls']==r['replay_updates']==0,'Gate scope mismatch')
                print('Feasibility PASS; headroom '+str(r['headroom_fraction'])+'. Conditional fresh F training is already authorized.',flush=True)
            if mode=='train':
                train=ROOT/cfg['run_root']/'train'
                require(read(train/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Training backbone integrity missing')
                r=read(train/'COMPLETION.json');require(r['steps']==1250 and r['epochs']==5,'Training scope mismatch')
        r=read(ROOT/cfg['run_root']/'evaluate/COMPLETION.json')
        require(r['generation_calls']==48,'Evaluation count mismatch')
        require(read(ROOT/cfg['run_root']/'evaluate/FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Evaluation backbone integrity missing')
        write(AUDIT/'ORCHESTRATION_COMPLETION.json',{'status':'GATE_TRAIN_AND48_GENERATIONS_COMPLETE_STOP_FOR_REVIEW',
            'feasibility_invocations':1,'train_invocations':1,'evaluation_invocations':1,'retry_invocations':0,
            'total_seconds':time.perf_counter()-started,'UTC':now(),'RO_U_started':False})
        print('D059 authorized GPU scope complete. No subsequent experiment.',flush=True)
        return 0
    except Exception as exc:
        write(AUDIT/'ORCHESTRATION_FAILURE_STOP.json',{'error':str(exc),'traceback':traceback.format_exc(),'UTC':now(),'no_retry':True})
        raise

if __name__=='__main__':sys.exit(main())
