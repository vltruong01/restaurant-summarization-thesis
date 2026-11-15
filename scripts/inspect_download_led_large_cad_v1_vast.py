"""Read-only monitoring and verified export for the explicitly approved CAD run."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,tarfile
R=Path(__file__).resolve().parents[1];E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-p','44054','root@171.5.182.162']
def remote(code):
 p=base64.b64encode(code.encode()).decode();cmd="python3 -c \"import base64;exec(base64.b64decode('"+p+"'))\""
 r=subprocess.run(SSH+[cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 return json.loads(r.stdout)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['status','download','outputs']);args=ap.parse_args()
 code='''from pathlib import Path
import json,subprocess
from datetime import datetime,timezone
p=Path('/workspace/cad_v1_vast_ops');r=Path('/workspace/CAD_V1_VAST_FRESH/runs/F_CE3_CAD_TRAIN_V1_VAST')
o=dict(UTC=datetime.now(timezone.utc).isoformat(),ops_exists=p.exists(),run_exists=r.exists())
for n in ['HOST_PREFLIGHT_PASS.json','STATUS.json','EXPORT_RECEIPT.json','EXPORT_FAILURE.json']:
 if (p/n).is_file():o[n]=json.loads((p/n).read_text())
o['complete_pairs']=len([x for x in (r/'cases').glob('*.json') if not x.name.endswith('IDENTITY.json')])
o['generation_calls_started']=len(list((r/'calls').glob('*STARTED.json')))
for n in ['COMPLETION.json','FAILURE_STOP_NO_RETRY.json']:
 if (r/n).is_file():o[n]=json.loads((r/n).read_text())
o['supervisor']=subprocess.run(['supervisorctl','status','cad_v1_vast'],capture_output=True,text=True).stdout.strip()
for n in ['PILOT.log','HOST_WRAPPER.log']:
 if (p/n).is_file():o[n]=(p/n).read_text(errors='replace').splitlines()[-8:]
o['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout.strip()
print(json.dumps(o))'''
 if args.mode=='outputs':
  code='''from pathlib import Path
import json
p=Path('/workspace/CAD_V1_VAST_FRESH/runs/F_CE3_CAD_TRAIN_V1_VAST/cases')
rows=[]
for x in sorted(p.glob('*.json')):
 if x.name.endswith('IDENTITY.json'):continue
 r=json.loads(x.read_text());rows.append(dict(business_id=r['business_id'],
 vanilla=r['vanilla']['text'],cad=r['cad']['text']))
print(json.dumps(rows))'''
  print(json.dumps(remote(code),ensure_ascii=False,indent=2));return
 o=remote(code)
 with (E/'STATUS_SNAPSHOTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(o)+'\n')
 if args.mode=='status':print(json.dumps(o,indent=2));return
 assert 'EXPORT_RECEIPT.json' in o and 'STATUS.json' in o,'Export not ready'
 rec=o['EXPORT_RECEIPT.json'];archive=E/'LED_LARGE_CAD_V1_VAST_EXECUTION_EVIDENCE.tar.gz'
 assert not archive.exists() and not (E/'downloaded').exists(),'Do not overwrite previous export'
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P','44054',
     'root@171.5.182.162:'+rec['path'],str(archive)],check=True)
 assert sha(archive)==rec['sha256'] and archive.stat().st_size==rec['bytes'],'Downloaded archive differs'
 dest=E/'downloaded';dest.mkdir()
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in m.name
  t.extractall(dest)
 files=json.loads((dest/'ARTIFACT_MANIFEST.json').read_text())['files']
 assert {x.relative_to(dest).as_posix() for x in dest.rglob('*') if x.is_file()}==set(files)|{'ARTIFACT_MANIFEST.json'}
 for rel,v in files.items():assert sha(dest/rel)==v['sha256'] and (dest/rel).stat().st_size==v['bytes'],rel
 result=dict(status='ALL_EXPORTED_BYTES_VERIFIED',archive_sha256=sha(archive),bytes=archive.stat().st_size,
     files=len(files),remote_status=o['STATUS.json'],instance_id=57481997)
 with (E/'DOWNLOAD_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
