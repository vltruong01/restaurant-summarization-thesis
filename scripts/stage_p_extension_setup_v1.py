"""CPU-only asset/software setup on the current already-authorized RTX3090 host."""
import datetime as dt,hashlib,json,os,platform,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
R=Path('/workspace/STAGE_P_THREE_SYSTEM_20261106_V1');Q=Path('/workspace/STAGE_P_QWEN35_9B_20261106_V2')
def put(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(args):subprocess.run(list(map(str,args)),check=True)
def main():
    assert os.environ.get('CONTAINER_ID')=='59313861' and platform.node()=='239abaa00671'
    assert not (R/'ASSET_SETUP_COMPLETE.json').exists()
    started=time.perf_counter();phase='vision_packages'
    try:
        qpy=Q/'VENV/bin/python'
        run(['uv','pip','install','--no-cache','--no-deps','--python',qpy,'torchvision==0.24.1','--index-url','https://download.pytorch.org/whl/cu128'])
        run(['uv','pip','install','--no-cache','--python',qpy,'Pillow==12.2.0'])
        phase='LED_environment'
        run(['uv','venv','--python','/usr/bin/python3',R/'LED_VENV'])
        site=R/'LED_VENV/lib/python3.12/site-packages';site.mkdir(parents=True,exist_ok=True)
        (site/'shared_torch_runtime.pth').write_text(str(Q/'VENV/lib/python3.12/site-packages')+'\n')
        lp=R/'LED_VENV/bin/python'
        run(['uv','pip','install','--no-cache','--no-deps','--python',lp,'transformers==4.51.3','tokenizers==0.21.1','huggingface-hub==0.36.2','accelerate==1.6.0','peft==0.15.2','numpy==1.26.4'])
        run(['uv','pip','install','--no-cache','--python',lp,'requests==2.32.5'])
        phase='LED_backbone_download'
        target=R/'ASSETS/BACKBONE/pytorch_model.bin';target.parent.mkdir(exist_ok=True);assert not target.exists()
        url='https://huggingface.co/allenai/led-large-16384/resolve/cd59d11c3528415b7dda4dfc95cc8f138aceda2e/pytorch_model.bin'
        with urllib.request.urlopen(url,timeout=120) as response,target.open('wb') as output:
            while data:=response.read(8*1024*1024):output.write(data)
        assert target.stat().st_size==1839633783 and sha(target)=='b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081'
        put('LED_BACKBONE_DOWNLOAD.json',dict(status='PASS',url=url,bytes=target.stat().st_size,sha256=sha(target),original_weights_unchanged=True))
        phase='actual_import_checks'
        run([qpy,'-c',"import torch,torchvision;from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration;from PIL import Image;print('Qwen multimodal imports',torch.__version__,torchvision.__version__)"])
        run([lp,'-c',"import torch,transformers,peft;from transformers import LEDForConditionalGeneration,LEDTokenizerFast;print('LED imports',torch.__version__,transformers.__version__,peft.__version__)"])
        for name,py in [('QWEN_ENVIRONMENT_FREEZE.txt',qpy),('LED_ENVIRONMENT_FREEZE.txt',lp)]:
            x=subprocess.run(['uv','pip','freeze','--python',str(py)],capture_output=True,text=True,check=True);(R/name).write_text(x.stdout)
        put('ASSET_SETUP_COMPLETE.json',dict(status='PASS_CPU_SETUP_ONLY',seconds=time.perf_counter()-started,GPU_generation=False,torch_shared_runtime='2.9.1+cu128',original_LED_torch='2.1.0+cu118',original_LED_transformers='4.51.3',benchmark_LED_transformers='4.51.3',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),note='Framework difference must accompany frozen-output equivalence results; model weights,precision and decoder settings unchanged'))
    except Exception as e:
        put('ASSET_SETUP_FAILURE.json',dict(phase=phase,error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
