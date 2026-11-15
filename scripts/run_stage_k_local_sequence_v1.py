"""Sequential existing-local-GPU execution, no rental, no retries, no reference access."""
from pathlib import Path
import sys,time,subprocess,json,datetime,psutil
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_k/s1000_test100_phase1_v1';P=R/'outputs/final_evaluation/s1000_preparation_v1'

def main():
    write(O/'EVIDENCE/LOCAL_SEQUENCE_STARTED.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),order=['F_already_running','RO','U','S'],retries=0,rental=False,reference_access=False))
    f=O/'EXECUTION/F'
    while not (f/'COMPLETION.json').exists():
        require(not (f/'FAILURE_STOP_NO_RETRY.json').exists(),'F failed; no automatic retry')
        time.sleep(5)
    # Wait for the already-started F interpreter to exit and release its CUDA context.
    while True:
        found=False
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmd=proc.info['cmdline'] or []
                if any('run_stage_k_s1000_phase1_v1.py' in s for s in cmd) and '--condition' in cmd and cmd[cmd.index('--condition')+1]=='F':found=True
            except (psutil.NoSuchProcess,psutil.AccessDenied,IndexError):pass
        if not found:break
        time.sleep(2)
    for a in ['RO','U','S']:
        require(not (O/'EXECUTION'/a).exists(),'Refuse existing condition/retry')
        cmd=[sys.executable,'-u','-X','utf8','-B',str(R/'scripts/run_stage_k_s1000_phase1_v1.py'),'--workspace',str(R),'--preparation',str(P),'--condition',a,'--execute','--approval-message',str(O/'EVIDENCE/RESEARCHER_STAGE_K_AUTHORIZATION.txt'),'--output',str(O/'EXECUTION'/a)]
        if a=='S':cmd+=['--F-results',str(f)]
        write(O/'EVIDENCE'/f'{a}_EXACT_EXECUTION_COMMAND.json',dict(argv=cmd,runner_sha256=file_hash(R/'scripts/run_stage_k_s1000_phase1_v1.py'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),retry=False))
        print(json.dumps(dict(stage='STARTED',condition=a)),flush=True)
        with (O/'EXECUTION'/f'{a}_native.stdout.txt').open('x',encoding='utf8') as log:
            result=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        require(result.returncode==0 and (O/'EXECUTION'/a/'COMPLETION.json').exists(),a+' technical failure: preserve evidence, no automatic retry')
        print(json.dumps(dict(stage='COMPLETE',condition=a)),flush=True)
    write(O/'EVIDENCE/LOCAL_INFERENCE_SEQUENCE_COMPLETED.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),conditions=4,retries=0,reference_access=False))
    print('All4 condition executions complete; reference-free analysis may now run.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(O/'EVIDENCE/LOCAL_SEQUENCE_STOPPED.json',dict(error=str(e),automatic_retry=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
