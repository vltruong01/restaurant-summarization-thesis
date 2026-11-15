"""Transfer finite frozen payload and manage only the authorized extension jobs."""
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];L=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1';P=L/'PAYLOAD';R='/workspace/STAGE_P_THREE_SYSTEM_20261106_V1';HOST='root@77.48.24.250';PORT='48324'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def remote(code,name):
    x=subprocess.run(['ssh','-p',PORT,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],input=code,capture_output=True,text=True,encoding='utf8');(L/(name+'.stdout.log')).write_text(x.stdout,encoding='utf8');(L/(name+'.stderr.log')).write_text(x.stderr,encoding='utf8');assert x.returncode==0,x.stdout+x.stderr;print(x.stdout);return x.stdout
def transfer():
    for n in ['stage_p_resource_common_v1.py','benchmark_stage_p_led_3090_v1.py']:
        shutil.copy2(ROOT/'scripts'/n,P/n)
    original=Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/c189f9b6-b5a7-4fec-a487-b03387f50ba3/Pasted text.txt');shutil.copy2(original,P/'PROVENANCE/RESEARCHER_EXTENSION_REQUEST.md')
    members=[dict(file=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='TRANSFER_MANIFEST.json'];(P/'TRANSFER_MANIFEST.json').write_text(json.dumps(members,indent=2)+'\n')
    archive=L/'EXTENSION_INPUTS.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(P).as_posix())
    x=subprocess.run(['scp','-P',PORT,'-o','BatchMode=yes',str(archive),HOST+':'+R+'/EXTENSION_INPUTS.zip'],capture_output=True,text=True);assert x.returncode==0,x.stderr
    remote(f'''from pathlib import Path
import hashlib,json,os,zipfile
r=Path({R!r});assert os.environ.get('CONTAINER_ID')=='59313861'
assert (r/'ASSET_SETUP_COMPLETE.json').exists()
f=r/'ASSETS/F/adapter_model.safetensors';assert f.stat().st_size==19726200 and hashlib.sha256(f.read_bytes()).hexdigest()=='6dcb08275e96f3972b15058f6abf370d83fa8ce05553214af94b5a0bd9c1c699'
z=r/'EXTENSION_INPUTS.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()=={sha(archive)!r}
with zipfile.ZipFile(z) as a:
 for n in a.namelist():assert (r/n).resolve().is_relative_to(r)
 a.extractall(r)
for row in json.loads((r/'TRANSFER_MANIFEST.json').read_text()):
 f=r/row['file'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(status='PASS_TRANSFER_AND_ASSET_BINDINGS',files={len(members)},ZIP_sha256={sha(archive)!r})))
''','TRANSFER_VERIFICATION')
def launch_led():
    remote(f'''from pathlib import Path
import subprocess,os
r=Path({R!r});assert os.environ.get('CONTAINER_ID')=='59313861' and not (r/'LED_BENCHMARK').exists()
wrapper="""from pathlib import Path
import subprocess,json,os,time,datetime
r=Path(__file__).resolve().parent;t=time.perf_counter()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
p=subprocess.run([str(r/'LED_VENV/bin/python'),'-u',str(r/'benchmark_stage_p_led_3090_v1.py'),'--root',str(r)],env=env)
(r/'LED_JOB_EXIT.json').write_text(json.dumps(dict(exit_code=p.returncode,seconds=time.perf_counter()-t,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\\n')
raise SystemExit(p.returncode)
"""
(r/'RUN_LED_JOB.py').write_text(wrapper)
name='stage_p_extension_led_benchmark_v1';c=Path('/etc/supervisor/conf.d/'+name+'.conf');assert not c.exists()
c.write_text('[program:'+name+']\\ncommand=/usr/bin/python3 -u '+str(r/'RUN_LED_JOB.py')+'\\ndirectory='+str(r)+'\\nautostart=false\\nautorestart=false\\nstartsecs=0\\nstdout_logfile='+str(r/'LED.stdout.log')+'\\nstderr_logfile='+str(r/'LED.stderr.log')+'\\nstdout_logfile_maxbytes=0\\nstderr_logfile_maxbytes=0\\n')
for cmd in [['supervisorctl','reread'],['supervisorctl','update'],['supervisorctl','start',name]]:
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr;print(p.stdout.strip())
''','LED_LAUNCH')
def status():
    remote(f'''from pathlib import Path
import json,subprocess,os
r=Path({R!r});assert os.environ.get('CONTAINER_ID')=='59313861';out=dict(receipts={{}},logs={{}})
out['services']=subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout.splitlines()
out['services']=[s for s in out['services'] if 'stage_p_extension' in s]
for n in ['ASSET_SETUP_FAILURE.json','LED_JOB_EXIT.json','LED_BENCHMARK/FAILURE.json','LED_BENCHMARK/COMPLETION.json','IMAGE_PREFLIGHT.json','IMAGE_JOB_EXIT.json','QWEN_IMAGE/FAILURE.json','QWEN_IMAGE/COMPATIBILITY_PASS.json','QWEN_IMAGE/COMPLETION.json']:
 p=r/n
 if p.exists():
  d=json.loads(p.read_text());out['receipts'][n]={{k:v for k,v in d.items() if k not in ['VRAM_phase_peaks','calls','input_checks']}}
for n in ['LED.stdout.log','LED.stderr.log','IMAGE.stdout.log','IMAGE.stderr.log']:
 p=r/n
 if p.exists():out['logs'][n]=p.read_text(errors='replace')[-1700:]
out['LED_saved']={{x:len(list((r/'LED_BENCHMARK'/x).glob('*.json'))) for x in ['WARMUP','REPEAT_1','REPEAT_2']}}
out['image_saved']=len(list((r/'QWEN_IMAGE/CALLS').glob('*.json')))
print(json.dumps(out,indent=2))
''','LATEST_STATUS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['transfer','launch-led','status']);a=p.parse_args();{'transfer':transfer,'launch-led':launch_led,'status':status}[a.command]()
