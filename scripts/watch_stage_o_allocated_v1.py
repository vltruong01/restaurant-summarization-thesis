"""Local noninteractive read-only monitoring and verified native backup. No auto-retry/lifecycle."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,tarfile,time,traceback
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/stage_o/tripadvisor_external_v1';E=O/'RTX3080_EXECUTION_V1'
TARGET='root@1.208.108.242';REMOTE='/workspace/STAGE_O_TRIPADVISOR_20261104_V1'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p','55308',TARGET]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def ssh(command,timeout=60):
    r=subprocess.run(SSH+[command],capture_output=True,text=True,encoding='utf8',timeout=timeout)
    if r.returncode:raise RuntimeError('SSH command failed: '+r.stderr[-800:])
    return r.stdout
def native_backup():
    receipt=json.loads(ssh(REMOTE+'/venv/bin/python '+REMOTE+'/EXPORT_NATIVE_EVIDENCE.py',timeout=300))
    write(E/'NATIVE_EXPORT_RECEIPT.json',receipt)
    destination=E/receipt['archive']
    subprocess.run(['scp','-q','-P','55308',TARGET+':'+REMOTE+'/'+receipt['archive'],str(destination)],check=True,timeout=1800)
    assert destination.stat().st_size==receipt['bytes'] and sha(destination)==receipt['sha256']
    extracted=E/'DOWNLOADED_NATIVE';extracted.mkdir(exist_ok=False)
    with tarfile.open(destination,'r:gz') as t:
        for member in t.getmembers():
            assert member.isfile() and not member.issym() and (extracted/member.name).resolve().is_relative_to(extracted.resolve())
        t.extractall(extracted,filter='data')
    members=json.loads((extracted/'NATIVE_EXPORT_MEMBER_SHA256.json').read_text(encoding='utf8'))
    for r in members:
        p=extracted/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['file']
    write(E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json',dict(status='PASS',archive_sha256=sha(destination),every_member_verified=len(members),state=receipt['status'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if receipt['status']=='EXPORTED_NATIVE_COMPLETE':
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/verify_stage_o_completed_native_v1.py')],check=True,timeout=600)
    return receipt

def main():
    if (E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json').exists():print('Terminal evidence already backed up; no duplicate export.');return
    errors=0
    while True:
        try:
            d=json.loads(ssh(REMOTE+'/venv/bin/python '+REMOTE+'/READ_EXECUTION_STATUS.py'))
            write(E/'LATEST_STATUS.json',d);errors=0
            with (E/'POLL_HISTORY.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(UTC=d['UTC'],caption_checkpoints=d['caption_checkpoints'],worker=d['worker'],failure=bool(d['failure']),final_outputs=d['final_outputs']))+'\n')
            print(d['UTC'],d['caption_checkpoints'],d['worker'],flush=True)
            if d.get('failure') or d.get('final_outputs'):
                print(json.dumps(native_backup(),indent=2),flush=True);return
        except Exception as exc:
            errors+=1;write(E/'MONITOR_LAST_ERROR.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=str(exc),consecutive_errors=errors,traceback=traceback.format_exc(),instance_left_intact=True))
            print('Monitor/backup error:',exc,flush=True)
            if errors>=3:raise
        time.sleep(55)
if __name__=='__main__':main()
