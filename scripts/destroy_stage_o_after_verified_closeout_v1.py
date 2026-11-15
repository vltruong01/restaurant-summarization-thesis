"""One authorized DELETE of instance 59069440 after verified successful closeout.

No inference changes, lifecycle retries, credential files or credential logging.
The local watcher must stay running; this is not a provider-side scheduled job.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, time
import urllib.request, urllib.error, zipfile

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1'
TASK = '/workspace/STAGE_O_TRIPADVISOR_20261104_V1'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=20', '-p', '55308', 'root@1.208.108.242']
ID = 59069440
URL = 'https://console.vast.ai/api/v0/instances/59069440/'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(p, value): p.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def local_proof():
    auth = read(E/'INSTANCE_DESTROY_AUTHORIZATION.json')
    assert auth['instance_id'] == ID and auth['destroy_after_successful_verified_closeout'] is True
    assert not (E/'KEEP_INSTANCE.json').exists(), 'Destruction revoked locally'
    assert not (E/'INSTANCE_DESTRUCTION_REQUEST.json').exists(), 'DELETE previously attempted; no automatic retry'
    backup = read(E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json')
    receipt = read(E/'NATIVE_EXPORT_RECEIPT.json')
    assert backup['status'] == 'PASS' and receipt['status'] == 'EXPORTED_NATIVE_COMPLETE'
    archive = E/receipt['archive']
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256'] == backup['archive_sha256']
    D = E/'DOWNLOADED_NATIVE'
    members = read(D/'NATIVE_EXPORT_MEMBER_SHA256.json')
    assert len(members) == backup['every_member_verified']
    for r in members:
        p = D/r['file']
        assert p.resolve().is_relative_to(D.resolve()) and p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], r['file']
    terminal = read(D/'STAGE_O/RTX3080_EXECUTION_V1/WORKER_STATUS.json')
    assert terminal['state'] == 'NATIVE_GENERATION_COMPLETE_PENDING_LOCAL_DOWNLOAD_VERIFICATION'
    result = read(E/'COMPLETED_NATIVE_VALIDATION.json')
    assert result['status'] == 'PASS_NATIVE_75_COMPONENTS_45_OUTPUTS_VERIFIED_LOCALLY_STOP_BEFORE_HUMAN'
    assert result['native_LED_components'] == 75 and result['assembled_outputs'] == 45
    package = read(E/'FINAL_COMPLETED_PACKAGE_RECEIPT.json')
    assert package['status'] == 'PASS' and package['outputs'] == 45 and package['components'] == 75
    zpath = ROOT/package['ZIP']
    assert zpath.resolve().is_relative_to(E.resolve()) and sha(zpath) == package['sha256']
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        for r in json.loads(z.read('PACKAGE_MEMBER_SHA256.json')):
            data = z.read(r['file'])
            assert len(data) == r['bytes'] and hashlib.sha256(data).hexdigest() == r['sha256']
    return dict(archive_sha256=receipt['sha256'], review_zip_sha256=package['sha256'],
                verified_native_members=len(members), terminal_sha256=sha(D/'STAGE_O/RTX3080_EXECUTION_V1/WORKER_STATUS.json'))

REMOTE_PROOF = r'''
from pathlib import Path
import os,socket,json,hashlib,subprocess
T=Path('/workspace/STAGE_O_TRIPADVISOR_20261104_V1');P=T/'project'
E=P/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert socket.gethostname()=='77eb3c4ad31e' and os.environ.get('CONTAINER_ID')=='59069440'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-953a5550-f5a3-b664-9017-0acd8461f37c'
service=subprocess.run(['supervisorctl','status','stage_o_inference'],capture_output=True,text=True)
s=service.stdout.strip()
# supervisorctl uses a nonzero exit status for an EXITED program; inspect its explicit state.
assert s.split()[:2]==['stage_o_inference','EXITED'] and 'RUNNING' not in s
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert read(E/'WORKER_STATUS.json')['state']=='NATIVE_GENERATION_COMPLETE_PENDING_LOCAL_DOWNLOAD_VERIFICATION'
assert not (E/'FAILURE_PRESERVE_NO_AUTORETRY.json').exists()
r=read(T/'NATIVE_EXPORT_RECEIPT.json');assert r['status']=='EXPORTED_NATIVE_COMPLETE'
assert sha(T/r['archive'])==r['sha256']
members=read(T/'NATIVE_EXPORT_MEMBER_SHA256.json')
for m in members:
 n=m['file']
 p=(P/'outputs/stage_o/tripadvisor_external_v1'/n[8:]) if n.startswith('STAGE_O/') else ((T/n[8:]) if n.startswith('RUNTIME/') else P/'scripts'/n[5:])
 assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],n
allowed={'AGENTS.md','CLAUDE.md','.hf_home','.venv-backups',T.name}
assert {p.name for p in Path('/workspace').iterdir()}<=allowed,'Unexpected workspace requires preservation review'
task_files={'bootstrap.sh','BOOTSTRAP_SUPERVISOR.log','BOOTSTRAP.log','ENVIRONMENT_FREEZE.txt','STAGE_O_INPUTS_AND_LED_ASSETS.tar','restore_pinned_assets.py','ASSET_DOWNLOAD.log','STAGE_O_INPUTS_SMALL.tar.gz','restore_led_backbone.py','BACKBONE_DOWNLOAD.log','READ_EXECUTION_STATUS.py','run_once.sh','PINNED_ASSET_DOWNLOAD_RECEIPTS.json','EXECUTION.log','EXPORT_NATIVE_EVIDENCE.py','NATIVE_EXPORT_MEMBER_SHA256.json','NATIVE_EXPORT_RECEIPT.json',r['archive']}
assert {p.name for p in T.iterdir()} <= task_files|{'project','venv'},'Unexpected task artifacts require preservation review'
supplement=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in T.iterdir() if p.is_file() and p.suffix not in ['.tar','.gz']]
# Every non-software project file must have an exact local counterpart or be in the verified export.
exported={m['file'][8:] for m in members if m['file'].startswith('STAGE_O/')}
remaining=[]
for p in P.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(P).as_posix()
 if '__pycache__' in p.parts or '.cache' in p.parts or p.suffix=='.pyc':continue
 if rel.startswith('assets/model_cache/'):continue # Pinned, publicly recoverable model cache; receipts are exported.
 prefix='outputs/stage_o/tripadvisor_external_v1/'
 if rel.startswith(prefix) and rel[len(prefix):] in exported:continue
 remaining.append(dict(file=rel,bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(key=os.environ['CONTAINER_API_KEY'],instance_id=59069440,container=socket.gethostname(),service=s,no_GPU_compute_processes=True,archive_sha256=r['sha256'],terminal_sha256=sha(E/'WORKER_STATUS.json'),supplement=supplement,remaining_project_files=remaining,workspace_inventory=sorted(allowed))))
'''

def remote_proof(proof):
    p = subprocess.run(SSH+['python3 -B -'], input=REMOTE_PROOF, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, 'Remote terminal/identity/preservation check failed; instance left intact'
    data = json.loads(p.stdout); p = None
    key = data.pop('key')
    assert data['archive_sha256'] == proof['archive_sha256'] and data['terminal_sha256'] == proof['terminal_sha256']
    for r in data['remaining_project_files']:
        p = ROOT/r['file']
        assert p.resolve().is_relative_to(ROOT.resolve()) and p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], 'Unbacked project artifact: '+r['file']
    S = E/'FINAL_RUNTIME_SUPPLEMENT'; S.mkdir(exist_ok=True)
    for r in data['supplement']:
        assert Path(r['file']).name == r['file']
        p = S/r['file']
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P','55308',
                        'root@1.208.108.242:'+TASK+'/'+r['file'], str(p)], check=True, capture_output=True, timeout=120)
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
    save(E/'FINAL_REMOTE_PRESERVATION_VERIFICATION.json', dict(status='PASS',UTC=now(),**data))
    return key, data

def provider(key, method):
    req = urllib.request.Request(URL, data=b'{}' if method == 'DELETE' else None,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=35) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, {}

def instance(data):
    obj = data.get('instances',data.get('instance',data)) if isinstance(data,dict) else data
    if isinstance(obj,list): obj = next((x for x in obj if str(x.get('id')) == str(ID)), None)
    return obj if isinstance(obj,dict) and str(obj.get('id')) == str(ID) else None

def destroy():
    proof = local_proof()
    key, remote = remote_proof(proof)
    status, data = provider(key, 'GET'); obj = instance(data)
    assert status == 200 and obj and obj['machine_id'] == 139250
    assert obj['gpu_name'] == 'RTX 3080' and obj['num_gpus'] == 1 and obj['public_ipaddr'] == '1.208.108.242'
    assert not (E/'KEEP_INSTANCE.json').exists(), 'Destruction revoked locally'
    request = dict(UTC=now(),instance_id=ID,machine_id=139250,endpoint=URL,method='DELETE',automatic_retry=False,
                   authorization_sha256=sha(E/'INSTANCE_DESTROY_AUTHORIZATION.json'),backup=proof,
                   remote_verification_sha256=sha(E/'FINAL_REMOTE_PRESERVATION_VERIFICATION.json'),
                   provider_identity={k:obj.get(k) for k in ['id','machine_id','gpu_name','num_gpus','public_ipaddr','actual_status']},credentials_stored=False)
    with (E/'INSTANCE_DESTRUCTION_REQUEST.json').open('x',encoding='utf8') as f: json.dump(request,f,indent=2)
    result = dict(instance_id=ID,UTC=now(),status='DELETE_RESULT_UNCERTAIN_NO_RETRY',credentials_stored=False)
    save(E/'INSTANCE_DESTRUCTION_RESULT.json',result)
    try:
        status, data = provider(key, 'DELETE') # Exactly one DELETE, never retry.
        result.update(http_status=status,provider_success=data.get('success') if isinstance(data,dict) else None)
        result['status'] = 'DELETE_ACCEPTED_BY_PROVIDER' if status == 200 and data.get('success') is True else 'DELETE_NOT_CONFIRMED_NO_RETRY'
        save(E/'INSTANCE_DESTRUCTION_RESULT.json',result)
        if result['status'] == 'DELETE_ACCEPTED_BY_PROVIDER':
            result['postqueries'] = []
            for _ in range(10):
                time.sleep(3)
                st, d = provider(key,'GET'); current = instance(d)
                absent = st == 404 or (st == 200 and isinstance(d,dict) and 'instances' in d and d['instances'] in [None,[]]) or (st == 200 and d == [])
                result['postqueries'].append(dict(UTC=now(),http_status=st,instance_present=current is not None,actual_status=current.get('actual_status') if current else None))
                if absent or (current and current.get('actual_status') in ['destroyed','deleted']):
                    result['status']='DESTROY_CONFIRMED_BY_PROVIDER_AND_POSTQUERY'; break
                if st in [401,403]: break
    except Exception as exc: result['exception_type'] = type(exc).__name__
    finally:
        key = None
        result['finished_UTC'] = now();save(E/'INSTANCE_DESTRUCTION_RESULT.json',result)
    print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--check-local',action='store_true');args=ap.parse_args()
    if args.check_local:
        try: local_proof(); print('READY_LOCAL_PROOF_VALID_NO_DELETE')
        except Exception as e: print('NOT_READY_NO_DELETE: '+str(e))
        return
    assert args.watch, 'Only deferred watch or read-only check is supported'
    while True:
        if (E/'KEEP_INSTANCE.json').exists(): print('REVOKED_INSTANCE_LEFT_INTACT',flush=True); return
        if (E/'INSTANCE_DESTRUCTION_REQUEST.json').exists(): print('PREVIOUS_DELETE_ATTEMPT_NO_RETRY',flush=True); return
        if (E/'FINAL_COMPLETED_PACKAGE_RECEIPT.json').exists():
            try: destroy()
            except Exception as e:
                save(E/'INSTANCE_DESTROY_BLOCKER.json',dict(UTC=now(),status='BLOCKED_NO_DELETE_RETRY',error_type=type(e).__name__,error=str(e) if isinstance(e,AssertionError) else 'See error type; raw network/credential output not recorded',instance_left_intact_unless_request_already_sent=(not (E/'INSTANCE_DESTRUCTION_REQUEST.json').exists())))
                print('DESTROY_BLOCKED_OR_UNCERTAIN; inspect local receipts',flush=True)
            return
        if (E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json').exists() and read(E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json').get('state')=='EXPORTED_FAILURE':
            save(E/'INSTANCE_DESTROY_BLOCKER.json',dict(UTC=now(),status='FAILED_RUN_LEFT_INTACT',reason='Authorization binds destruction to successful complete job; failed evidence preserved for researcher review'))
            return
        time.sleep(55)

if __name__ == '__main__': main()
