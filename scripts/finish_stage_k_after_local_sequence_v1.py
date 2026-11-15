"""Finish the currently authorized run after its finite local sequence; no retries."""
from pathlib import Path
import sys,time,subprocess,datetime
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_k/s1000_test100_phase1_v1';P=R/'outputs/final_evaluation/s1000_preparation_v1'
def main():
    while not (O/'EVIDENCE/LOCAL_INFERENCE_SEQUENCE_COMPLETED.json').exists():
        require(not (O/'EVIDENCE/LOCAL_SEQUENCE_STOPPED.json').exists(),'Local inference stopped; no automatic recovery')
        time.sleep(5)
    cmds=[
        [sys.executable,'-X','utf8','-B',str(R/'scripts/stage_k_reference_free_analysis_v1.py'),'--preparation',str(P),'--execution',str(O/'EXECUTION'),'--output',str(O/'ANALYSIS')],
        [sys.executable,'-X','utf8','-B',str(R/'scripts/finalize_stage_k_phase1_v1.py')]]
    for i,cmd in enumerate(cmds,1):
        print('Starting post-inference phase '+str(i),flush=True)
        with (O/'EVIDENCE'/f'POST_INFERENCE_PHASE_{i}.stdout.txt').open('x',encoding='utf8') as f:result=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
        require(result.returncode==0,'Post-inference phase '+str(i)+' failed; preserve evidence')
    print('Actual400-record Phase1 package complete and verified.',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(O/'EVIDENCE/POST_INFERENCE_STOPPED.json',dict(error=str(e),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False));raise
