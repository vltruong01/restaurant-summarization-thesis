"""Record current explicit authorization; preflight; invoke frozen D056-v2 once.

No settings are edited, no retry is performed, and no model code is imported here.
"""
import datetime,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d057_bounded_gpu_diagnostic_v1'
BINDINGS={
    'configs/stage_i_d056_minimal_diagnostic_v2.json':'6f41647861b62c82186d7911a5166f916b2cd081abf3bef9287cf6c336bd538e',
    'outputs/stage_i/d056_diagnostic_preparation_v2/PREFLIGHT_MANIFEST.json':'9983ad481c61d7886e2b4683fce6f177f9a445c02b34906d3f628f7e3b53d1f0',
    'scripts/run_stage_i_d056_gpu_diagnostic_v2.py':'efe4b279da099aa0c3ebc4b51551df2e05bb4b6dc9ef41bcc9827edff1f2dc3d',
}
AUTHORIZATION_TEXT='''I officially authorize the execution of the D-056-v2 bounded GPU diagnostic.

**GPU_execution_approved = true**, strictly for the previously reviewed and hash-bound D-056-v2 protocol.

Approved scope:

- Exactly 3 frozen validation cases and 4 read-only model states.
- Maximum 24 teacher-forced forwards.
- Maximum 144 cache-control forwards, using p=3 and p=11.
- Maximum 36 diagnostic generation traces, with at most 64 newly generated tokens per trace.
- Preserve the three previously approved SHA-256 bindings without modification.

**Execution requirements:**

1. Record this explicit researcher authorization and its exact scope before execution.
2. Rerun offline preflight and integrity checks in the actual GPU environment.
3. Execute the approved diagnostic once, without changing any scientific settings.
4. Preserve complete measurements, raw evidence, errors, numerical comparisons and runtime logs.
5. Stop immediately on unexpected integrity, numerical, configuration or execution failures. Do not silently retry or modify tolerances.
6. Analyze the results against D-055 hypotheses, distinguishing confirmed observations from unresolved explanations.
7. Produce the D-057 diagnostic report and a portable ChatGPT review bundle.
8. Update DECISIONS.md and THESIS_PROGRESS.md.

**Not authorized:** retraining, additional decoding experiments, architecture changes, Teacher API calls, dataset/target/checkpoint modifications, held-out test access or automatic follow-up experiments.

Preserve D-051 strict replay FAIL and all previously frozen artifacts.

After completing D-057, stop for independent scientific review before proposing or executing further experiments.'''

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def verify_bindings():
    actual={rel:sha(ROOT/rel) for rel in BINDINGS}
    if actual!=BINDINGS:raise ValueError('Reviewed three-way hash mismatch: STOP')
    return actual

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    if sys.argv[1:]!=['--execute']:raise ValueError('Explicit --execute required')
    OUT.mkdir(exist_ok=False)
    started=time.perf_counter()
    try:
        verify_bindings()
        cfg=json.loads((ROOT/'configs/stage_i_d056_minimal_diagnostic_v2.json').read_text(encoding='utf-8'))
        if (ROOT/cfg['future_output']).exists():raise ValueError('Diagnostic namespace already exists; no retry')
        approval={'decision':'D-056','amendment':'D-056-v2','scope':'bounded_read_only_GPU_diagnosis',
            'GPU_execution_approved':True,'training_authorized':False,'approval_source':'researcher_message',
            'researcher_explicit_authorization':AUTHORIZATION_TEXT,'authorization_text_sha256':hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest(),
            'recorded_at_UTC':now(),'config_sha256':BINDINGS['configs/stage_i_d056_minimal_diagnostic_v2.json'],
            'preflight_manifest_sha256':BINDINGS['outputs/stage_i/d056_diagnostic_preparation_v2/PREFLIGHT_MANIFEST.json'],
            'GPU_runner_sha256':BINDINGS['scripts/run_stage_i_d056_gpu_diagnostic_v2.py'],
            'case_ids':cfg['cases'],'model_states':cfg['states'],'ceilings':cfg['ceilings'],
            'cache_prefix_lengths':[3,11],'cache_calls_per_comparison':3,'trace_new_token_cap':64,
            'D051_strict_replay':'FAIL_UNCHANGED','execution_attempts_authorized':1}
        ap=OUT/'RESEARCHER_GPU_AUTHORIZATION.json';write(ap,approval)
        write(OUT/'EXECUTION_ENVIRONMENT_BEFORE.json',{'python_executable':sys.executable,'python_version':sys.version,
            'platform':platform.platform(),'cwd':str(ROOT),'pid':os.getpid(),'UTC':now(),
            'verified_bindings':verify_bindings(),'orchestrator_sha256':sha(Path(__file__).resolve())})
        print('Explicit authorization recorded. Running original offline preflight.',flush=True)
        pre=subprocess.run([sys.executable,str(ROOT/'scripts/check_stage_i_d056_offline_v2.py')],cwd=ROOT,capture_output=True,text=True)
        with (OUT/'OFFLINE_PREFLIGHT_STDOUT.log').open('x',encoding='utf-8') as f:f.write(pre.stdout)
        with (OUT/'OFFLINE_PREFLIGHT_STDERR.log').open('x',encoding='utf-8') as f:f.write(pre.stderr)
        if pre.returncode:raise ValueError('Actual-environment offline preflight failed; no GPU invocation')
        write(OUT/'OFFLINE_PREFLIGHT_RESULT.json',json.loads(pre.stdout))
        verify_bindings()
        command=[sys.executable,str(ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py'),'--execute','--approval',str(ap)]
        write(OUT/'GPU_EXECUTION_INTENT.json',{'command':command,'UTC':now(),'approved_runner_sha256':sha(ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py'),
            'execution_invocation':1,'no_retry':True,'scientific_configuration_changes':False})
        print('Offline preflight PASS. Invoking frozen GPU runner exactly once.',flush=True)
        with (OUT/'GPU_RUNNER_STDOUT.log').open('x',encoding='utf-8') as stdout,(OUT/'GPU_RUNNER_STDERR.log').open('x',encoding='utf-8') as stderr:
            run=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        write(OUT/'GPU_PROCESS_EXIT.json',{'returncode':run.returncode,'UTC':now(),'total_wall_seconds':time.perf_counter()-started,
            'execution_invocations':1,'retry_invocations':0,'bindings_after':verify_bindings(),
            'status':'RUNNER_COMPLETE_STOP_FOR_REVIEW' if run.returncode==0 else 'RUNNER_FAILED_STOP_NO_RETRY'})
        print('GPU runner exited with code '+str(run.returncode)+'; no retry.',flush=True)
        sys.exit(run.returncode)
    except Exception as e:
        write(OUT/'ORCHESTRATION_STOP.json',{'error':str(e),'traceback':traceback.format_exc(),'UTC':now(),
            'total_wall_seconds':time.perf_counter()-started,'no_retry':True})
        raise

if __name__=='__main__':main()
