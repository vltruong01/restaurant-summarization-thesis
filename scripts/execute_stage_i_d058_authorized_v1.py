"""Exactly-once D058 orchestration. Runs frozen train then frozen evaluation, no retry."""
import datetime,json,os,platform,subprocess,sys,time,traceback
from stage_i_d058 import *

AUDIT=ROOT/'outputs/stage_i/d058_execution_v1'
BINDINGS={'config_sha256':'9cf921ad327d8bd337ce60aea4a15bd512524b6cf4cea8c6db260245d97c1d5b',
    'preflight_manifest_sha256':'fdf53852f2d9d702228b883c08ff92d66e0d065dd2e53b2422a34b8bdbbff687',
    'runner_sha256':'19b4a8adea20524192ebd821c2268b05d9d160aaf7f4d091c48a3ebd40b70bf3'}
AUTHORIZATION_TEXT='''I have reviewed both D058_PLAN_AND_AUTHORIZATION.md and D058_READINESS_REPORT.md.

**Researcher decision: D-058 GPU TRAINING AND EVALUATION OFFICIALLY APPROVED.**

Proceed with the approved D-058 experiment under the following exact SHA-256 bindings:

- Config: `9cf921ad327d8bd337ce60aea4a15bd512524b6cf4cea8c6db260245d97c1d5b`
- Preflight: `fdf53852f2d9d702228b883c08ff92d66e0d065dd2e53b2422a34b8bdbbff687`
- Runner: `19b4a8adea20524192ebd821c2268b05d9d160aaf7f4d091c48a3ebd40b70bf3`

**1. Training**

- Train exactly one fresh Filtered-only LED-LoRA architecture B model.
- Use 2,000 training businesses and 200 validation businesses.
- Maximum 5 epochs / 1,250 optimizer updates, warmup 63 steps.
- Disable loss-based early stopping, retain all 5 epoch checkpoints, and disable automatic best-loss checkpoint reloading.
- Preserve all previously frozen data, targets, tokenizer, optimization settings and scientific artifacts.

**2. Evaluation**

- Generate exactly 24 summaries from epoch 3 and 24 from epoch 5.
- Preserve the original beam4 generation configuration.
- Compare with the saved historical full-F outputs using the approved D-053 references.
- Evaluate repetition, factual grounding, Core information coverage, opinion balance, visual evidence, structure and completeness.
- Separate 20 random validation cases from 4 targeted cases.
- Use blinded model comparisons where practical. AI assessments must not be presented as human-confirmed labels.

**3. Strict stop-loss policy**

This is our final planned LED-LoRA architecture-B improvement attempt.

- **GO:** Substantial, researcher-confirmed improvement in usable summaries. Propose fair RO/U/F experiments using the same architecture and training protocol.
- **NO-GO:** Mixed, inconclusive or poor quality. Stop further incremental LED-LoRA optimization and propose the most efficient alternative-model strategy.
- Do not automatically launch additional experiments, diagnostics, architecture changes or decoding searches.

**4. Execution requirements**

Verify all three hashes and run the integrity preflight before GPU execution. Stop on unexpected failure without silent retries or fallback.

After training and evaluation, produce the D-058 report, full evidence and portable ChatGPT review bundle. Update DECISIONS.md and THESIS_PROGRESS.md.

Preserve D-051 strict replay FAIL. No held-out test access, new Teacher API calls, data modifications or additional training runs.

**Execute the approved D-058 scope now, then stop for independent scientific review.**

Prioritize completing the thesis over further optimization cycles.'''

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify_three():
    actual={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
        'runner_sha256':sha(ROOT/'scripts/run_stage_i_d058_v1.py')}
    require(actual==BINDINGS,'Researcher-bound three hashes mismatch')
    return actual

def main():
    require(sys.argv[1:]==['--execute'],'Explicit --execute required')
    AUDIT.mkdir(exist_ok=False);started=time.perf_counter()
    try:
        verify_three()
        cfg=read(CONFIG)
        approval={**read(PREP/'GPU_AUTHORIZATION_TEMPLATE.json'),'GPU_execution_approved':True,'training_authorized':True,
            'approval_source':'researcher_message','researcher_explicit_authorization':AUTHORIZATION_TEXT,
            'recorded_UTC':now(),'D051_strict_replay':'FAIL_UNCHANGED','policy_adaptations_explicitly_approved':True,
            'max_epochs':5,'warmup_steps':63,'evaluation_epochs':[3,5],'run_invocations_max':1,'retry_authorized':False}
        ap=AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json';write(ap,approval)
        config,m=preflight()
        write(AUDIT/'ACTUAL_ENVIRONMENT_PREFLIGHT.json',{'status':'PASS','UTC':now(),'three_bindings':verify_three(),
            'bound_files_checked':len(m['files']),'runtime_pins':m['runtime_pins'],'python_executable':sys.executable,
            'python':sys.version,'platform':platform.platform(),'orchestrator_sha256':sha(Path(__file__).resolve()),
            'model_loads_so_far':0,'GPU_execution_so_far':False})
        print('D058 authorization recorded; all3 hashes and50 preflight bindings PASS.',flush=True)
        for mode in ['train','evaluate']:
            verify_three();gate(ap,mode)
            cmd=[sys.executable,str(ROOT/'scripts/run_stage_i_d058_v1.py'),mode,'--execute','--approval',str(ap)]
            write(AUDIT/(mode.upper()+'_INVOCATION_INTENT.json'),{'command':cmd,'UTC':now(),'invocation':1,'no_retry':True,'bindings':verify_three()})
            print('Invoking frozen D058 '+mode+' exactly once.',flush=True);t=time.perf_counter()
            with (AUDIT/(mode.upper()+'_STDOUT.log')).open('x',encoding='utf-8') as out,(AUDIT/(mode.upper()+'_STDERR.log')).open('x',encoding='utf-8') as err:
                result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err)
            write(AUDIT/(mode.upper()+'_PROCESS_EXIT.json'),{'returncode':result.returncode,'UTC':now(),'seconds':time.perf_counter()-t,
                'invocations':1,'retry_invocations':0,'bindings_after':verify_three()})
            if result.returncode:
                write(AUDIT/'STOP_NO_RETRY.json',{'failed_mode':mode,'returncode':result.returncode,'UTC':now(),
                    'automatic_evaluation_started':False,'next_experiment_started':False,'total_seconds':time.perf_counter()-started})
                print('STOP: '+mode+' exited '+str(result.returncode)+'; no retry or fallback.',flush=True)
                sys.exit(result.returncode)
            if mode=='train':
                trainout=ROOT/cfg['run_root']/'train'
                require(read(trainout/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Training backbone integrity missing')
                require(read(trainout/'COMPLETION.json')['steps']==1250,'Unexpected completed training scope')
                print('Training complete; exact epoch3/5 evaluation is within existing authorization.',flush=True)
        write(AUDIT/'ORCHESTRATION_COMPLETION.json',{'status':'TRAIN_AND48_GENERATIONS_COMPLETE_STOP_FOR_REPORT_REVIEW',
            'train_invocations':1,'evaluation_invocations':1,'retry_invocations':0,'total_seconds':time.perf_counter()-started,'UTC':now()})
        print('D058 train/evaluation complete. No subsequent experiment.',flush=True)
    except Exception as exc:
        write(AUDIT/'ORCHESTRATION_FAILURE_STOP.json',{'error':str(exc),'traceback':traceback.format_exc(),'UTC':now(),'no_retry':True})
        raise

if __name__=='__main__':main()
