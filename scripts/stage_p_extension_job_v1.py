"""Finite no-restart subprocess wrapper; never changes a model configuration."""
from pathlib import Path
import datetime as dt,json,os,subprocess,sys,time
R=Path('/workspace/STAGE_P_THREE_SYSTEM_20261106_V1');Q=Path('/workspace/STAGE_P_QWEN35_9B_20261106_V2')
def main():
    phase=sys.argv[1];assert phase in ['led','image'];assert os.environ.get('CONTAINER_ID')=='59313861'
    if phase=='led':py=R/'LED_VENV/bin/python';script=R/'benchmark_stage_p_led_3090_v1.py';extra=[];receipt=R/'LED_JOB_EXIT.json'
    else:py=Q/'VENV/bin/python';script=R/'run_stage_p_qwen_image_v1.py';extra=['run'];receipt=R/'IMAGE_JOB_EXIT.json'
    assert not receipt.exists();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false');t=time.perf_counter()
    p=subprocess.run([str(py),'-u',str(script),*extra,'--root',str(R)],env=env)
    receipt.write_text(json.dumps(dict(exit_code=p.returncode,seconds=time.perf_counter()-t,automatic_restart=False,at_utc=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2)+chr(10))
    raise SystemExit(p.returncode)
if __name__=='__main__':main()
