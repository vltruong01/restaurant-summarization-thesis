"""Current researcher-supplied endpoint only. Bootstrap/run finite no-restart job."""
import argparse,hashlib,json,shutil,subprocess,sys,zipfile,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LOCAL=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2'
HOST='root@77.48.24.250';PORT='48324';REMOTE='/workspace/STAGE_P_QWEN35_9B_20261106_V2';UUID='GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759';SERVICE='stage_p_qwen35_bf16_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def remote(code,label):
    r=subprocess.run(['ssh','-p',PORT,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],input=code,capture_output=True,text=True,encoding='utf8')
    (LOCAL/(label+'_stdout.log')).write_text(r.stdout,encoding='utf8');(LOCAL/(label+'_stderr.log')).write_text(r.stderr,encoding='utf8')
    if r.returncode:raise RuntimeError(label+' failed: '+r.stderr+'\n'+r.stdout)
    print(r.stdout);return r.stdout
SETUP='''import json,os,subprocess,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
def put(name,x): (root/name).write_text(json.dumps(x,indent=2)+'\\n',encoding='utf8')
started=time.perf_counter()
try:
 assert os.environ.get('CONTAINER_ID')=='59313861'
 py=root/'VENV/bin/python'
 assert not py.exists(),'Do not reuse/retry setup'
 subprocess.run(['uv','venv','--python','/usr/bin/python3',str(root/'VENV')],check=True)
 subprocess.run(['uv','pip','install','--no-cache','--python',str(py),'torch==2.9.1','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
 subprocess.run(['uv','pip','install','--no-cache','--python',str(py),'transformers==5.22.1','accelerate==1.15.0','safetensors==0.8.0','Jinja2==3.1.6'],check=True)
 put('SOFTWARE_SETUP_RECEIPT.json',dict(status='PASS_INSTALL',seconds=time.perf_counter()-started))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HOME=str(root/'HF_CACHE'),HF_HUB_DISABLE_XET='1',HF_XET_CHUNK_CACHE_SIZE_BYTES='0',PYTHONUNBUFFERED='1')
 code=subprocess.run([str(py),'-u',str(root/'RUN_QWEN.py'),'execute','--root',str(root),'--destination',str(root/'EXECUTION'),'--gpu-uuid','GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759'],env=env).returncode
 put('JOB_EXIT_RECEIPT.json',dict(exit_code=code,seconds=time.perf_counter()-started,automatic_restart=False))
 sys.exit(code)
except Exception as e:
 put('SETUP_FAILURE_RECEIPT.json',dict(error=str(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
 raise
'''
def launch():
    # All transfer files are finite frozen text/tokenizer data, no image/rating/app payloads.
    shutil.copy2(ROOT/'scripts/stage_p_qwen_bf16_3090_v2.py',LOCAL/'RUN_QWEN.py')
    (LOCAL/'SETUP_AND_RUN.py').write_text(SETUP,encoding='utf8')
    put(LOCAL/'CURRENT_AUTHORIZATION_REFERENCE.json',dict(authority='Actual current researcher messages in this conversation',SSH='ssh -p 48324 root@77.48.24.250',scope='Official Qwen3.5-9B BF16 batch1 thinking disabled. One-restaurant compatibility test then conditional15 restaurants. Stop on failure. No model fallback, rental, training or human evaluation.',instance_id='59313861',container='239abaa00671',GPU_uuid=UUID,recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),paid_GPU_already_supplied_by_researcher=True,destroy_authorization_for_this_instance=False))
    allowed=['FROZEN_SPEC.json','PROMPTS.json','INFERENCE_INPUTS.jsonl','RENDERED_CALLS_30.json','EXECUTION_PLAN.json','INPUT_ALIGNMENT.json','OFFICIAL_MODEL_REVISION_CHECK.json','CURRENT_AUTHORIZATION_REFERENCE.json','RUN_QWEN.py','SETUP_AND_RUN.py']
    files=[LOCAL/f for f in allowed]+[f for f in (LOCAL/'MODEL_PROVENANCE').rglob('*') if f.is_file()]
    put(LOCAL/'TRANSFER_MANIFEST.json',[dict(file=f.relative_to(LOCAL).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files]);files.append(LOCAL/'TRANSFER_MANIFEST.json')
    archive=LOCAL/'REMOTE_INPUTS.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,f.relative_to(LOCAL).as_posix())
    code=f'''import json,os,platform,subprocess,datetime
from pathlib import Path
root=Path({REMOTE!r})
assert not root.exists(),'Remote task already exists; no overwrite/retry'
assert os.environ.get('CONTAINER_ID')=='59313861' and platform.node()=='239abaa00671'
r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
assert {UUID!r} in r and 'RTX 3090' in r
root.mkdir()
identity=dict(instance_id=os.environ['CONTAINER_ID'],hostname=platform.node(),gpu_uuid={UUID!r},nvidia_smi=r,SSH='77.48.24.250:48324',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'REMOTE_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\\n')
print(json.dumps(identity,indent=2))
'''
    identity=json.loads(remote(code,'REMOTE_INITIAL_IDENTITY'));put(LOCAL/'REMOTE_IDENTITY.json',identity)
    transfer=subprocess.run(['scp','-P',PORT,'-o','BatchMode=yes',str(archive),HOST+':'+REMOTE+'/REMOTE_INPUTS.zip'],capture_output=True,text=True)
    (LOCAL/'SCP_TRANSFER.log').write_text(transfer.stdout+transfer.stderr,encoding='utf8');assert transfer.returncode==0,transfer.stderr
    code=f'''import json,hashlib,subprocess,zipfile
from pathlib import Path
root=Path({REMOTE!r});archive=root/'REMOTE_INPUTS.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with zipfile.ZipFile(archive) as z:
 for n in z.namelist(): assert (root/n).resolve().is_relative_to(root)
 z.extractall(root)
for r in json.loads((root/'TRANSFER_MANIFEST.json').read_text()):
 p=root/r['file'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
conf=Path('/etc/supervisor/conf.d/{SERVICE}.conf');assert not conf.exists()
conf.write_text('[program:{SERVICE}]\\ncommand=/usr/bin/python3 -u '+str(root/'SETUP_AND_RUN.py')+'\\ndirectory='+str(root)+'\\nautostart=false\\nautorestart=false\\nstartsecs=0\\nstdout_logfile='+str(root/'JOB.stdout.log')+'\\nstderr_logfile='+str(root/'JOB.stderr.log')+'\\nstdout_logfile_maxbytes=0\\nstderr_logfile_maxbytes=0\\nenvironment=PYTHONUNBUFFERED="1"\\n')
for cmd in [['supervisorctl','reread'],['supervisorctl','update'],['supervisorctl','start','{SERVICE}']]:
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr;print(p.stdout.strip())
print(json.dumps(dict(status='STARTED_FINITE_SETUP_AND_INFERENCE',root=str(root),autorestart=False,transfer_SHA256={sha(archive)!r})))
'''
    remote(code,'REMOTE_LAUNCH')
def status():
    remote(f'''import json,subprocess,os
from pathlib import Path
r=Path({REMOTE!r});assert os.environ.get('CONTAINER_ID')=='59313861'
p=subprocess.run(['supervisorctl','status','stage_p_qwen35_bf16_exec_v3'],capture_output=True,text=True)
out=dict(supervisor=p.stdout.strip(),receipts={{}},logs={{}})
for name in ['SOFTWARE_SETUP_RECEIPT.json','SETUP_FAILURE_RECEIPT.json','JOB_EXIT_RECEIPT.json','INFERENCE_JOB_EXIT_RECEIPT.json','EXECUTION/HARDWARE_AND_EXECUTION_RECEIPT.json','EXECUTION/COMPATIBILITY_PASS.json','EXECUTION/COMPLETION_RECEIPT.json','EXECUTION/FAILURE_RECEIPT.json']:
 f=r/name
 if f.exists():out['receipts'][name]=json.loads(f.read_text())
for name in ['INFERENCE.stdout.log','INFERENCE.stderr.log']:
 f=r/name
 if f.exists():out['logs'][name]=f.read_text(errors='replace')[-4500:]
out['saved_calls']=len(list((r/'EXECUTION/CALLS').glob('*.json')))
print(json.dumps(out,indent=2))
''','LATEST_STATUS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','status']);a=p.parse_args()
    launch() if a.command=='launch' else status()
