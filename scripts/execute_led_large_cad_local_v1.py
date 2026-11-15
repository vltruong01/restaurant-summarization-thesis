"""Launch exactly one foreground local CAD process with durable stdout/exit."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess
R=Path(__file__).resolve().parents[1]
E=R/'outputs/stage_i/led_large_cad_v1_local_execution_v1'
cmd=[str(R/'.venv-caption-pilot/Scripts/python.exe'),'-u',str(R/'scripts/run_led_large_cad_v1.py'),'pilot',
    '--workspace',str(R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'),
    '--checkpoint',str(R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'),
    '--output',str(E/'run'), '--approval',str(E/'LOCAL_INFERENCE_AUTHORIZATION.json'),'--execute']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
started=datetime.now(timezone.utc).isoformat()
with (E/'RUN.log').open('x',encoding='utf-8') as f:
    p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
receipt=dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),return_code=p.returncode,
    launch_count=1,automatic_retry=False)
with (E/'PROCESS_EXIT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
