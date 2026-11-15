"""Single researcher-authorized new RTX3090 host deployment. No automatic retries."""
from pathlib import Path
import base64,json,subprocess
R=Path(__file__).resolve().parents[1]
E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-p','44054','root@171.5.182.162']
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P','44054']

HOST=r'''from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,os,platform,subprocess,sys,tarfile,time,traceback
ROOT=Path('/workspace/CAD_V1_VAST_FRESH');OPS=Path('/workspace/cad_v1_vast_ops')
RUN=ROOT/'runs/F_CE3_CAD_TRAIN_V1_VAST';CP=ROOT/'checkpoint-750';WS=ROOT/'parent'
ARCHIVE=Path('/workspace/CAD_V1_FRESH_HOST_PAYLOAD.tar')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,obj):
 with (OPS/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
status=dict(status='PREFLIGHT_STARTED',started_utc=datetime.now(timezone.utc).isoformat(),pilot_invocations=0,
            automatic_retry=False,local_resource_stop_preserved=True,training=False)
try:
 assert os.environ['CONTAINER_ID']=='57481997','Wrong instance'
 assert platform.python_version()=='3.11.7','Pinned Python differs'
 assert not ROOT.exists() and not RUN.exists(),'Fresh namespace required'
 assert sha(ARCHIVE)=='4f86febdcae153a8774f8860cbbf3eb135a6ff84a176f4a1ead92d34ab1488d0','Transfer archive differs'
 with tarfile.open(ARCHIVE) as t:
  members=t.getmembers()
  for m in members:
   p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in m.name,'Unsafe transfer member'
  ROOT.mkdir();t.extractall(ROOT)
 manifest=read(ROOT/'TRANSFER_MANIFEST.json')
 for rel,v in manifest['files'].items():assert sha(ROOT/rel)==v['sha256'] and (ROOT/rel).stat().st_size==v['bytes'],rel
 assert {x.relative_to(ROOT).as_posix() for x in ROOT.rglob('*') if x.is_file()}==set(manifest['files'])|{'TRANSFER_MANIFEST.json'}
 assert manifest['validation_or_test_transferred'] is False and manifest['original_Teacher_targets_transferred'] is False
 sys.path.insert(0,str(ROOT/'scripts'))
 import run_led_large_cad_v1 as runner
 cfg=runner.preflight(WS,CP);approval,deadline=runner.authorize(OPS/'RESEARCHER_AUTHORIZATION.json',runner.binding())
 actual={k:importlib.metadata.version(k) for k in cfg['runtime_pins']};assert actual==cfg['runtime_pins'],actual
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
          PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 cpu=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_led_large_cad_v1.py','-v'],
                    cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 (OPS/'HOST_CPU_TESTS.txt').write_text(cpu.stdout+'\n'+cpu.stderr)
 assert cpu.returncode==0 and 'Ran 11 tests' in cpu.stderr and 'skipped=1' in cpu.stderr,cpu.stderr
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv,noheader'],
                    capture_output=True,text=True,check=True,timeout=15).stdout.strip()
 assert gpu.count('NVIDIA GeForce RTX 3090')==1 and 'GPU-386606c5-3717-8b4f-c0fe-70907f18b717' in gpu,gpu
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 env['CUDA_VISIBLE_DEVICES']='0';env['CAD_RUN_CUDA_TESTS']='1'
 code="import sys,unittest;sys.path.insert(0,'tests');from test_led_large_cad_v1 import CADTests;r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([CADTests('test_cuda_fp16_alpha0_and_nonzero')]));sys.exit(not r.wasSuccessful())"
 cuda=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 (OPS/'HOST_CUDA_TEST.txt').write_text(cuda.stdout+'\n'+cuda.stderr)
 assert cuda.returncode==0 and 'Ran 1 test' in cuda.stderr,cuda.stderr
 env.pop('CAD_RUN_CUDA_TESTS')
 write('HOST_PREFLIGHT_PASS.json',dict(status='PASS',instance_id=57481997,GPU=gpu,Python=platform.python_version(),
       runtime=actual,transfer_files_verified=len(manifest['files']),parent_code_bindings=len(cfg['parent_code_bindings']),
       CPU_tests_passed=10,CUDA_FP16_reduced_LED_test_passed=1,bindings=runner.binding(),source_cases=12,
       zero_caption_cases=3,validation_or_test_access=False,optimizer_updates=0))
 write('INSTALLED_PACKAGES.json',{d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
 command=[sys.executable,'-u',str(ROOT/'scripts/run_led_large_cad_v1.py'),'pilot','--workspace',str(WS),'--checkpoint',str(CP),
          '--output',str(RUN),'--approval',str(OPS/'RESEARCHER_AUTHORIZATION.json'),'--execute']
 write('PILOT_INVOCATION.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=command,
       maximum_invocations=1,maximum_generate_calls=37,timeout_seconds=3700,approval_sha256=sha(OPS/'RESEARCHER_AUTHORIZATION.json')))
 status['pilot_invocations']=1
 with (OPS/'PILOT.log').open('x') as f:job=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=3700)
 status.update(exit_code=job.returncode,status='COMPLETE_STOP_FOR_REVIEW' if job.returncode==0 else 'FAIL_STOP_NO_RETRY')
except BaseException as e:
 status.update(status='FAIL_STOP_NO_RETRY',error=repr(e),traceback=traceback.format_exc())
 print(traceback.format_exc(),flush=True)
finally:
 status['finished_utc']=datetime.now(timezone.utc).isoformat()
 if (RUN/'COMPLETION.json').is_file():status['completion']=read(RUN/'COMPLETION.json')
 if (RUN/'FAILURE_STOP_NO_RETRY.json').is_file():status['failure']=read(RUN/'FAILURE_STOP_NO_RETRY.json')
 write('STATUS.json',status)
 try:
  if ROOT.is_dir() and 'runner' in globals():
   runner.preflight(WS,CP)
   write('POSTFLIGHT_PRESERVATION.json',dict(status='PASS',parent_and_checkpoint_bytes_unchanged=True,bindings=runner.binding()))
 except BaseException as e:write('POSTFLIGHT_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()))
 try:
  files={}
  if RUN.is_dir():
   for p in RUN.rglob('*'):
    if p.is_file():files['run/'+p.relative_to(RUN).as_posix()]=p
  for p in OPS.iterdir():
   if p.is_file() and p.name not in ['HOST_WRAPPER.log','ARTIFACT_MANIFEST.json']:
    files['operations/'+p.name]=p
  for rel in ['scripts/run_led_large_cad_v1.py','scripts/led_large_cad_v1.py','tests/test_led_large_cad_v1.py',
             'outputs/stage_i/led_large_cad_v1/CONFIG.json','outputs/stage_i/led_large_cad_v1/BINDINGS.json',
             'outputs/stage_i/led_large_cad_v1/PREFLIGHT_MANIFEST.json','TRANSFER_MANIFEST.json']:
   if (ROOT/rel).is_file():files['locked_payload/'+rel]=ROOT/rel
  hashes={rel:dict(sha256=sha(p),bytes=p.stat().st_size) for rel,p in files.items()}
  write('ARTIFACT_MANIFEST.json',dict(files=hashes))
  output=Path('/workspace/LED_LARGE_CAD_V1_VAST_EXECUTION_EVIDENCE.tar.gz')
  with tarfile.open(output,'x:gz') as t:
   for rel,p in files.items():t.add(p,arcname=rel,recursive=False)
   t.add(OPS/'ARTIFACT_MANIFEST.json',arcname='ARTIFACT_MANIFEST.json')
  write('EXPORT_RECEIPT.json',dict(path=str(output),bytes=output.stat().st_size,sha256=sha(output),files=len(files),status=status))
  print('EXPORT_READY',sha(output),flush=True)
 except BaseException as e:write('EXPORT_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()))
 print(json.dumps(status),flush=True)
if status.get('exit_code')!=0:raise SystemExit(1)
'''

def remote(code):
    payload=base64.b64encode(code.encode()).decode()
    command="python3 -c \"import base64;exec(base64.b64decode('"+payload+"'))\""
    r=subprocess.run(SSH+[command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    print(r.stdout);print(r.stderr)
    if r.returncode:raise RuntimeError('SSH operational command failed '+str(r.returncode))
    return r.stdout

def main():
    (E/'HOST_WRAPPER.py').write_text(HOST,encoding='utf-8')
    remote("from pathlib import Path\np=Path('/workspace/cad_v1_vast_ops');p.mkdir(exist_ok=False)\nprint('OPS_CREATED')")
    for local,name in [(E/'HOST_WRAPPER.py','HOST_WRAPPER.py'),(E/'VAST_CAD_RESEARCHER_AUTHORIZATION.json','RESEARCHER_AUTHORIZATION.json')]:
        subprocess.run(SCP+[str(local),'root@171.5.182.162:/workspace/cad_v1_vast_ops/'+name],check=True)
    config='''[program:cad_v1_vast]
command=/workspace/cad_runtime/bin/python -u /workspace/cad_v1_vast_ops/HOST_WRAPPER.py
directory=/workspace/cad_v1_vast_ops
autostart=false
autorestart=false
startretries=0
startsecs=0
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/cad_v1_vast_ops/HOST_WRAPPER.log
redirect_stderr=true
stdout_logfile_maxbytes=0
environment=OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false"
'''
    (E/'cad_v1_vast.conf').write_text(config,encoding='utf-8')
    subprocess.run(SCP+[str(E/'cad_v1_vast.conf'),'root@171.5.182.162:/etc/supervisor/conf.d/cad_v1_vast.conf'],check=True)
    remote("import subprocess\nfor command in [['supervisorctl','reread'],['supervisorctl','update'],['supervisorctl','start','cad_v1_vast']]:\n r=subprocess.run(command,capture_output=True,text=True);print(r.stdout,r.stderr);assert r.returncode==0\n")
if __name__=='__main__':main()
