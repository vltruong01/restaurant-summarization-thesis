"""Download and verify the one completed/failed V3 execution, no GPU operations."""
from pathlib import Path
import base64,hashlib,json,shutil,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    code="from pathlib import Path; p=Path('/workspace/cf_native_ce_v3_ops/EXPORT_RECEIPT.json'); print(p.read_text() if p.is_file() else '{}')"
    payload=base64.b64encode(code.encode()).decode()
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-p','30556','root@163.142.222.112',
        "python3 -c \"import base64;exec(base64.b64decode('"+payload+"'))\""],capture_output=True,text=True,timeout=50)
    if r.returncode!=0:raise RuntimeError(r.stderr)
    receipt=json.loads(r.stdout)
    if not receipt:print('EXPORT_NOT_READY');return
    assert receipt['path']=='/workspace/LED_LARGE_CF_NATIVE_CE_V3_EXECUTION_EVIDENCE.tar.gz'
    archive=OUT/Path(receipt['path']).name
    if not archive.exists():
        r=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-P','30556',
            'root@163.142.222.112:'+receipt['path'],str(archive)],capture_output=True,text=True,timeout=600)
        if r.returncode!=0:raise RuntimeError(r.stderr)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    dest=OUT/'downloaded';dest.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            rel=Path(member.name);target=(dest/rel).resolve()
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and target.is_relative_to(dest.resolve())
            assert all(':' not in part for part in rel.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    manifest=json.loads((dest/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))['files']
    for rel,proof in manifest.items():
        p=dest/rel;assert p.stat().st_size==proof['bytes'] and sha(p)==proof['sha256'],rel
    actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    assert actual==set(manifest)|{'ARTIFACT_MANIFEST.json'}
    (OUT/'REMOTE_EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ALL_EXPORTED_BYTES_VERIFIED',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        files=len(manifest),manifest_PASS=True,remote_status=receipt['status'])
    with (OUT/'DOWNLOAD_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
