"""One explicitly authorized provider DELETE for59313861; no GPU experiments.
Secrets remain inside the exact instance. A disconnect is never destruction proof.
"""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'outputs/stage_p/pre_destroy_backup_audit_20261106_v1';D=ROOT/'outputs/stage_p/instance_59313861_destroy_20261106_v1';ID=59313861
SSH=['ssh','-p','48324','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','root@77.48.24.250']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
REMOTE=r'''
import os,socket,json,hashlib,subprocess,sys,urllib.request,urllib.error,time,datetime
from pathlib import Path
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert os.environ.get('CONTAINER_ID')=='59313861' and socket.gethostname()=='239abaa00671'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
s=subprocess.run(['supervisorctl','status'],capture_output=True,text=True);assert s.returncode in [0,3];services=s.stdout.splitlines();research=[s for s in services if s.startswith('stage_p_')];assert research and all(s.split()[1]=='EXITED' for s in research)
r=Path('/workspace/STAGE_P_THREE_SYSTEM_20261106_V1')
for f in ['LED_JOB_EXIT.json','IMAGE_JOB_EXIT.json']:assert json.loads((r/f).read_text())['exit_code']==0
assert json.loads((r/'LED_BENCHMARK/COMPLETION.json').read_text())['measured_calls']==60
assert json.loads((r/'QWEN_IMAGE/COMPLETION.json').read_text())['new_visual_calls']==15
expected=EXPECTED_FILES
for row in expected:
 p=Path(row['remote_file']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
key=os.environ['CONTAINER_API_KEY'];url='https://console.vast.ai/api/v0/instances/59313861/'
def call(method):
 request=urllib.request.Request(url,data=b'{}' if method=='DELETE' else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method=method)
 try:
  with urllib.request.urlopen(request,timeout=25) as response:return response.status,json.load(response)
 except urllib.error.HTTPError as e:return e.code,{}
def instance(d):
 obj=d.get('instances',d.get('instance',d)) if isinstance(d,dict) else d
 if isinstance(obj,list):obj=next((x for x in obj if str(x.get('id'))=='59313861'),None)
 return obj if isinstance(obj,dict) and str(obj.get('id'))=='59313861' else None
code,data=call('GET');obj=instance(data)
assert code==200 and obj and obj['num_gpus']==1 and obj['gpu_name']=='RTX 3090' and obj['public_ipaddr']=='77.48.24.250'
if EXPECTED_MACHINE is not None:assert obj['machine_id']==EXPECTED_MACHINE
proof=dict(event='verified_predelete_state',UTC=now(),instance_id=59313861,container=socket.gethostname(),GPU_UUID='GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759',provider_HTTP=code,provider_identity={k:obj.get(k) for k in ['id','machine_id','gpu_name','num_gpus','public_ipaddr','actual_status','intended_status']},research_services=research,remote_research_hashes_rechecked=len(expected),GPU_compute_empty=True)
print(json.dumps(proof),flush=True)
if sys.argv[-1]=='verify':raise SystemExit
try:
 code,data=call('DELETE') # Exactly one authenticated DELETE. No retries.
 accepted=code==200 and isinstance(data,dict) and data.get('success') is True
 print(json.dumps(dict(event='provider_DELETE_response',UTC=now(),instance_id=59313861,http_status=code,success=data.get('success') if isinstance(data,dict) else None,response={k:data[k] for k in ['success','msg','message'] if isinstance(data,dict) and k in data},status='DESTROY_CONFIRMED_BY_PROVIDER_RESPONSE' if accepted else 'DESTROY_NOT_CONFIRMED_NO_RETRY')),flush=True)
 if accepted:
  for _ in range(4):
   time.sleep(.5);code,data=call('GET');obj=instance(data)
   absent=code==404 or (code==200 and isinstance(data,dict) and 'instances' in data and data['instances'] in [None,[],{}]) or (code==200 and data==[])
   destroyed=bool(obj and obj.get('actual_status') in ['destroyed','deleted'])
   print(json.dumps(dict(event='provider_postquery',UTC=now(),http_status=code,instance_present=obj is not None,actual_status=obj.get('actual_status') if obj else None,confirmed_absent_or_destroyed=absent or destroyed)),flush=True)
   if absent or destroyed or code in [401,403]:break
except Exception as e:print(json.dumps(dict(event='provider_error',UTC=now(),exception_type=type(e).__name__,automatic_retry=False)),flush=True)
'''
def events(output):
    result=[]
    for line in output.splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result
def main():
    D.mkdir(exist_ok=True);assert not (D/'INSTANCE_DESTRUCTION_REQUEST.json').exists(),'An earlier DELETE attempt exists; do not repeat'
    verdict=read(A/'FINAL_BACKUP_VERDICT.json');assert verdict['instance_id']=='59313861' and verdict['remaining_missing_irreplaceable_backups']==0
    foundation=read(A/'LOCAL_FOUNDATION_AUDIT.json');assert foundation['status']=='PASS' and not foundation['issues']
    backup=read(A/'REMOTE_TO_LOCAL_FINAL_VERIFICATION.json');assert backup['status']=='PASS' and backup['files']==342
    for row in backup['checks']:
        p=ROOT/row['verified_local_file'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    # Refresh critical full-state/model/source archive/rating hashes without repeating all20,990 checks.
    important=[x for x in read(A/'LOCAL_FOUNDATION_FILE_HASHES.json') if x['label'] in ['RO final LoRA S1000','U final LoRA S1000','F final LoRA S1000','RO full training state S1000','U full training state S1000','F full training state S1000','Shared LED backbone','Original publisher Gijon dataset archive','Final90 human ratings','Private SQLite human backup','Final Stage P review ZIP']]
    for row in important:assert sha(ROOT/row['file'])==row['sha256']
    auth=dict(instance_id=ID,SSH='77.48.24.250:48324',authorization='Latest explicit researcher request: nếu xong hãy destroy (destroy if complete), following verified backup audit.',UTC=now(),backup_verdict_sha256=sha(A/'FINAL_BACKUP_VERDICT.json'),foundation_audit_sha256=sha(A/'LOCAL_FOUNDATION_AUDIT.json'),local_remote_file_hashes_rechecked=342,critical_foundation_files_rechecked=len(important),GPU_work_authorized=False)
    put(D/'DESTROY_AUTHORIZATION_AND_LOCAL_GUARDS.json',auth)
    code=REMOTE.replace('EXPECTED_FILES',repr(backup['checks'])).replace('EXPECTED_MACHINE','None')
    p=subprocess.run(SSH+['python3 -u -B - verify'],input=code,capture_output=True,text=True,encoding='utf8',timeout=90)
    assert p.returncode==0,'Pre-delete verification failed: no DELETE sent. '+p.stderr[-300:]
    es=events(p.stdout);assert len(es)==1 and es[0]['event']=='verified_predelete_state';proof=es[0];put(D/'FINAL_PREDELETE_VERIFICATION.json',proof)
    machine=proof['provider_identity']['machine_id'];code=REMOTE.replace('EXPECTED_FILES',repr(backup['checks'])).replace('EXPECTED_MACHINE',str(machine))
    request=dict(instance_id=ID,machine_id=machine,method='DELETE',endpoint='https://console.vast.ai/api/v0/instances/59313861/',UTC=now(),automatic_retry=False,credentials_stored=False,preflight_sha256=sha(D/'FINAL_PREDELETE_VERIFICATION.json'),authorization_sha256=sha(D/'DESTROY_AUTHORIZATION_AND_LOCAL_GUARDS.json'))
    with (D/'INSTANCE_DESTRUCTION_REQUEST.json').open('x',encoding='utf8') as f:json.dump(request,f,indent=2)
    result=dict(instance_id=ID,status='DESTROY_RESULT_UNCERTAIN_NO_RETRY',request_sha256=sha(D/'INSTANCE_DESTRUCTION_REQUEST.json'),UTC=now(),credentials_stored=False)
    put(D/'INSTANCE_DESTRUCTION_RESULT.json',result)
    try:
        p=subprocess.run(SSH+['python3 -u -B - destroy'],input=code,capture_output=True,text=True,encoding='utf8',timeout=155);output=p.stdout;result['SSH_exit_code']=p.returncode
    except subprocess.TimeoutExpired as e:output=e.stdout or '';output=output.decode('utf8') if isinstance(output,bytes) else output;result['SSH_timeout']=True
    es=events(output);result['events']=es;response=next((e for e in es if e.get('event')=='provider_DELETE_response'),None)
    if response:result['status']=response['status']
    if response and response['http_status']==200 and response['success'] is True:
        result['provider_accepted_destroy']=True
        result['independent_absence_check']=any(e.get('confirmed_absent_or_destroyed') for e in es if e.get('event')=='provider_postquery')
        result['limitation']='Provider HTTP200 success=true is the direct receipt. SSH disconnect is not used as evidence. Postquery confirmation is reported separately.'
    result['finished_UTC']=now();put(D/'INSTANCE_DESTRUCTION_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
