"""Single authorized provider DELETE from the exact container's valid auth route.

Local-origin GET was rejected before any DELETE. Reuse verified local preservation
and recheck terminal/GPU/provider identity inside the allocated instance. No keys
leave the container, and SSH disconnect alone is never confirmation.
"""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,datetime

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-p','55308','root@1.208.108.242']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')

REMOTE=r'''
from pathlib import Path
import os,socket,json,subprocess,hashlib,sys,urllib.request,urllib.error,time,datetime
T=Path('/workspace/STAGE_O_TRIPADVISOR_20261104_V1');O=T/'project/outputs/stage_o/tripadvisor_external_v1';E=O/'RTX3080_EXECUTION_V1'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert socket.gethostname()=='77eb3c4ad31e' and os.environ.get('CONTAINER_ID')=='59069440'
s=subprocess.run(['supervisorctl','status','stage_o_inference'],capture_output=True,text=True)
assert s.stdout.split()[:2]==['stage_o_inference','EXITED']
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-953a5550-f5a3-b664-9017-0acd8461f37c'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
state=json.loads((E/'WORKER_STATUS.json').read_text())
assert state['state']=='NATIVE_GENERATION_COMPLETE_PENDING_LOCAL_DOWNLOAD_VERIFICATION' and state['outputs']==45 and state['components']==75
assert not (E/'FAILURE_PRESERVE_NO_AUTORETRY.json').exists()
receipt=json.loads((T/'NATIVE_EXPORT_RECEIPT.json').read_text())
assert sha(T/receipt['archive'])==receipt['sha256']=='EXPECTED_ARCHIVE_SHA'
assert sha(E/'WORKER_STATUS.json')=='EXPECTED_TERMINAL_SHA'
key=os.environ['CONTAINER_API_KEY'];url='https://console.vast.ai/api/v0/instances/59069440/'
def call(method):
 req=urllib.request.Request(url,data=b'{}' if method=='DELETE' else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method=method)
 try:
  with urllib.request.urlopen(req,timeout=25) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,{}
def instance(d):
 obj=d.get('instances',d.get('instance',d)) if isinstance(d,dict) else d
 if isinstance(obj,list):obj=next((x for x in obj if str(x.get('id'))=='59069440'),None)
 return obj if isinstance(obj,dict) and str(obj.get('id'))=='59069440' else None
code,data=call('GET');obj=instance(data)
assert code==200 and obj and obj['machine_id']==139250
assert obj['gpu_name']=='RTX 3080' and obj['num_gpus']==1 and obj['public_ipaddr']=='1.208.108.242'
proof=dict(UTC=now(),instance_id=59069440,container=socket.gethostname(),service=s.stdout.strip(),GPU_compute_empty=True,archive_sha256=receipt['sha256'],terminal_sha256=sha(E/'WORKER_STATUS.json'),provider_identity={k:obj.get(k) for k in ['id','machine_id','gpu_name','num_gpus','public_ipaddr','actual_status']})
if sys.argv[-1]=='verify':print(json.dumps(proof),flush=True);raise SystemExit
try:
 code,data=call('DELETE') # ONE DELETE; never retried.
 accepted=code==200 and isinstance(data,dict) and data.get('success') is True
 print(json.dumps(dict(event='provider_DELETE_response',UTC=now(),instance_id=59069440,http_status=code,success=data.get('success') if isinstance(data,dict) else None,status='DESTROY_CONFIRMED_BY_PROVIDER_RESPONSE' if accepted else 'DESTROY_BLOCKED_PROVIDER_REJECTED')),flush=True)
 if accepted:
  for _ in range(4):
   time.sleep(0.5);code,data=call('GET');obj=instance(data)
   absent=code==404 or (code==200 and isinstance(data,dict) and 'instances' in data and data['instances'] in [None,[],{}]) or (code==200 and data==[])
   destroyed=bool(obj and obj.get('actual_status') in ['destroyed','deleted'])
   print(json.dumps(dict(event='provider_postquery',UTC=now(),http_status=code,instance_present=obj is not None,actual_status=obj.get('actual_status') if obj else None,intended_status=obj.get('intended_status') if obj else None,confirmed_absent_or_destroyed=absent or destroyed)),flush=True)
   if absent or destroyed or code in [401,403]:break
except Exception as e:print(json.dumps(dict(event='provider_error',exception_type=type(e).__name__,UTC=now())),flush=True)
'''

def main():
 assert not (E/'INSTANCE_DESTRUCTION_REQUEST.json').exists(),'Previous DELETE attempt exists: never repeat'
 spec=importlib.util.spec_from_file_location('guard',ROOT/'scripts/destroy_stage_o_after_verified_closeout_v1.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 backup=guard.local_proof()
 preserved=read(E/'FINAL_REMOTE_PRESERVATION_VERIFICATION.json')
 assert preserved['status']=='PASS' and preserved['instance_id']==59069440
 assert preserved['archive_sha256']==backup['archive_sha256'] and preserved['terminal_sha256']==backup['terminal_sha256']
 for r in preserved['supplement']:
  p=E/'FINAL_RUNTIME_SUPPLEMENT'/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 for r in preserved['remaining_project_files']:
  p=ROOT/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 remote=REMOTE.replace('EXPECTED_ARCHIVE_SHA',backup['archive_sha256']).replace('EXPECTED_TERMINAL_SHA',backup['terminal_sha256'])
 check=subprocess.run(SSH+['python3 -u -B - verify'],input=remote,capture_output=True,text=True,timeout=65)
 assert check.returncode==0,'Host-route identity verification failed; no DELETE sent'
 proof=json.loads(check.stdout);assert proof['instance_id']==59069440
 put(E/'FINAL_HOST_ROUTE_PREDELETE_VERIFICATION.json',proof)
 assert not (E/'KEEP_INSTANCE.json').exists()
 request=dict(UTC=now(),instance_id=59069440,machine_id=139250,method='DELETE',endpoint='https://console.vast.ai/api/v0/instances/59069440/',route='Authenticated in-container provider API on exact authorized host',automatic_retry=False,credentials_stored=False,backup=backup,authorization_sha256=sha(E/'INSTANCE_DESTROY_AUTHORIZATION.json'),preservation_sha256=sha(E/'FINAL_REMOTE_PRESERVATION_VERIFICATION.json'),pre_destroy=proof,prior_provider_DELETE_requests=0)
 with (E/'INSTANCE_DESTRUCTION_REQUEST.json').open('x',encoding='utf8') as f:json.dump(request,f,indent=2)
 result=dict(UTC=now(),instance_id=59069440,status='DESTROY_RESULT_UNCERTAIN_NO_RETRY',request_sha256=sha(E/'INSTANCE_DESTRUCTION_REQUEST.json'),credentials_stored=False,automatic_retry=False)
 put(E/'INSTANCE_DESTRUCTION_RESULT.json',result)
 try:
  p=subprocess.run(SSH+['python3 -u -B - destroy'],input=remote,capture_output=True,text=True,timeout=155);output=p.stdout;result['SSH_return_code']=p.returncode
 except subprocess.TimeoutExpired as e:
  output=e.stdout or '';output=output.decode('utf8') if isinstance(output,bytes) else output;result['SSH_timeout']=True
 events=[]
 for line in output.splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 result['events']=events
 response=next((e for e in events if e.get('event')=='provider_DELETE_response'),None)
 if response:result['status']=response['status']
 if response and response['http_status']==200 and response['success'] is True:
  if any(e.get('event')=='provider_postquery' and e.get('confirmed_absent_or_destroyed') for e in events):result['status']='DESTROY_CONFIRMED_PROVIDER_AND_POSTQUERY'
  result['limitation']='Provider HTTP 200 success=true is the direct destroy receipt. Postquery availability is recorded separately; SSH disconnection is not used as proof.'
 result['finished_UTC']=now();put(E/'INSTANCE_DESTRUCTION_RESULT.json',result)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
