"""Collect terminal native evidence with archive/member SHA256, no GPU work."""
import subprocess,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];L=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1';R='/workspace/STAGE_P_THREE_SYSTEM_20261106_V1';HOST='root@77.48.24.250';PORT='48324'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    code='''from pathlib import Path
import json,os,hashlib,subprocess,shutil,zipfile,datetime
r=Path(REMOTE_ROOT);assert os.environ.get('CONTAINER_ID')=='59313861'
for n in ['LED_JOB_EXIT.json','IMAGE_JOB_EXIT.json']:assert json.loads((r/n).read_text())['exit_code']==0
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
q=Path('/workspace/STAGE_P_QWEN35_9B_20261106_V2')
p=subprocess.run([str(q/'VENV/bin/python'),str(r/'verify_stage_p_extension_cpu_v1.py'),'image','--root',str(r)],env=env,capture_output=True,text=True)
(r/'CPU_IMAGE_REPLAY.stdout.log').write_text(p.stdout);(r/'CPU_IMAGE_REPLAY.stderr.log').write_text(p.stderr);assert p.returncode==0,p.stdout+p.stderr
for label,python in [('QWEN',q/'VENV/bin/python'),('LED',r/'LED_VENV/bin/python')]:
 p=subprocess.run(['uv','pip','freeze','--python',str(python)],capture_output=True,text=True);assert p.returncode==0;(r/(label+'_ENVIRONMENT_FINAL.txt')).write_text(p.stdout)
snapshot=q/'HF_CACHE/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a';dest=r/'PROCESSOR_FILES';dest.mkdir(exist_ok=True)
for f in snapshot.iterdir():
 if f.is_file() and f.suffix in ['.json','.jinja','.txt'] and 'safetensors' not in f.name:shutil.copy2(f,dest/f.name)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=[]
for p in sorted(r.rglob('*')):
 rel=p.relative_to(r)
 if not p.is_file() or any(x in rel.parts for x in ['LED_VENV','__pycache__']):continue
 if p.suffix in ['.bin','.safetensors','.zip','.sha256'] or p.name in ['NATIVE_EXPORT_MANIFEST.json','NATIVE_EXPORT_RECEIPT.json']:continue
 files.append(dict(file=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(r/'NATIVE_EXPORT_MANIFEST.json').write_text(json.dumps(files,indent=2))
archive=r/'NATIVE_EXTENSION_EVIDENCE.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for row in files:z.write(r/row['file'],row['file'])
 z.write(r/'NATIVE_EXPORT_MANIFEST.json','NATIVE_EXPORT_MANIFEST.json')
receipt=dict(status='PASS_TERMINAL_ARCHIVE',bytes=archive.stat().st_size,sha256=sha(archive),members=len(files)+1,scientific_files=len(files),GPU_jobs_restarted=False,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(r/'NATIVE_EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('REMOTE_ROOT',repr(R))
    p=subprocess.run(['ssh','-p',PORT,'-o','BatchMode=yes',HOST,'python3','-'],input=code,capture_output=True,text=True,encoding='utf8');(L/'NATIVE_COLLECTION.stdout.log').write_text(p.stdout,encoding='utf8');(L/'NATIVE_COLLECTION.stderr.log').write_text(p.stderr,encoding='utf8');assert p.returncode==0,p.stdout+p.stderr
    receipt=json.loads(p.stdout.strip().splitlines()[-1]);print(receipt)
    archive=L/'NATIVE_EXTENSION_EVIDENCE.zip';p=subprocess.run(['scp','-P',PORT,'-o','BatchMode=yes',HOST+':'+R+'/NATIVE_EXTENSION_EVIDENCE.zip',str(archive)],capture_output=True,text=True);assert p.returncode==0,p.stderr
    assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    out=L/'NATIVE';assert not out.exists();out.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==receipt['members']
        for n in z.namelist():assert (out/n).resolve().is_relative_to(out.resolve())
        z.extractall(out)
    manifest=json.loads((out/'NATIVE_EXPORT_MANIFEST.json').read_text())
    for row in manifest:
        f=out/row['file'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
    (L/'NATIVE_LOCAL_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',archive=receipt,every_scientific_member_verified=len(manifest),ZIP_reopened_extracted_and_parsed=True),indent=2),encoding='utf8');print('PASS local every-member download verification')
if __name__=='__main__':main()
