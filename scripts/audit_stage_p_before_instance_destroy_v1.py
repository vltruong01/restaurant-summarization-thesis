"""Read-only remote research inventory and local hash matching; no GPU computation or lifecycle action."""
import json,hashlib,subprocess,os,collections,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage_p/pre_destroy_backup_audit_20261106_v1';HOST='root@77.48.24.250';PORT='48324'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def inventory():
    code='''from pathlib import Path
import os,json,hashlib,subprocess,collections,datetime
assert os.environ.get('CONTAINER_ID')=='59313861' and os.uname().nodename=='239abaa00671'
files=[];excluded=[];errors=[];links=[]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for current,dirs,names in os.walk('/workspace',followlinks=False):
 p=Path(current)
 for name in list(dirs):
  d=p/name
  reason=None
  if d.is_symlink():links.append(dict(path=str(d),target=os.readlink(d)));reason='Directory symlink'
  elif name in ['VENV','LED_VENV','.venv-backups','__pycache__','.hf_home']:reason='Reinstallable environment/cache; not research outputs'
  elif name=='HF_CACHE':reason='Pinned public pretrained Qwen cache; reproducible from revision and SHA256 receipts'
  if reason:excluded.append(dict(path=str(d),reason=reason));dirs.remove(name)
 for name in names:
  f=p/name
  try:
   if f.is_symlink():links.append(dict(path=str(f),target=os.readlink(f)));continue
   s=f.stat();files.append(dict(path=str(f),bytes=s.st_size,sha256=sha(f),mtime_ns=s.st_mtime_ns))
  except Exception as e:errors.append(dict(path=str(f),error=type(e).__name__))
for f in sorted(Path('/etc/supervisor/conf.d').glob('stage_p*.conf')):
 s=f.stat();files.append(dict(path=str(f),bytes=s.st_size,sha256=sha(f),mtime_ns=s.st_mtime_ns))
outside=[]
for base in ['/root','/home','/tmp','/opt/workspace-internal']:
 for current,dirs,names in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ['node_modules','site-packages','pkgs','envs','__pycache__'] and not (Path(current)/d).is_symlink()]
  for name in names:
   f=Path(current)/name
   if not f.is_symlink() and f.suffix.lower() in ['.ipynb','.safetensors','.pt','.pth','.parquet','.csv','.jsonl','.jpg','.jpeg','.png','.py']:
    outside.append(dict(path=str(f),bytes=f.stat().st_size))
print(json.dumps(dict(instance_id='59313861',hostname=os.uname().nodename,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,excluded=excluded,symlinks=links,errors=errors,outside_workspace_candidate_files=outside,GPU=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),compute_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),services=subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout.splitlines(),GPU_experiments_run=False,instance_lifecycle_action=False)))
'''
    p=subprocess.run(['ssh','-p',PORT,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],input=code,capture_output=True,text=True,encoding='utf8');(OUT/'REMOTE_INVENTORY.stderr.log').write_text(p.stderr,encoding='utf8');assert p.returncode==0,p.stderr;data=json.loads(p.stdout);put(OUT/'REMOTE_RESEARCH_INVENTORY.json',data);print('remote scientific files',len(data['files']),'outside candidates',len(data['outside_workspace_candidate_files']));return data
def match(data):
    sizes={r['bytes'] for r in data['files']};by_size=collections.defaultdict(list)
    # Find content-equivalent backups under Stage P and canonical model source directories.
    for current,dirs,names in os.walk(ROOT/'outputs/stage_p'):
        dirs[:]=[d for d in dirs if d!='pre_destroy_backup_audit_20261106_v1']
        for name in names:
            f=Path(current)/name;s=f.stat().st_size
            if s in sizes:by_size[s].append(f)
    model=read(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    for m in model['models'].values():
        for f in (ROOT/m['path']).iterdir():
            if f.is_file():by_size[f.stat().st_size].append(f)
    bb=Path(model['backbone']['local_file']);by_size[bb.stat().st_size].append(bb)
    cache={};rows=[];missing=[]
    for r in data['files']:
        hits=[]
        for f in by_size[r['bytes']]:
            # Prioritize native same-name files, but only accept actual SHA256 equality.
            if f not in cache:cache[f]=sha(f)
            if cache[f]==r['sha256']:hits.append(str(f.relative_to(ROOT)))
        row=dict(**r,local_backups=hits,status='VERIFIED_LOCAL_COPY' if hits else 'MISSING_LOCAL_COPY');rows.append(row)
        if not hits:missing.append(row)
    put(OUT/'REMOTE_TO_LOCAL_HASH_MATCH_INITIAL.json',rows);put(OUT/'MISSING_BACKUPS_INITIAL.json',missing);print('verified',len(rows)-len(missing),'missing',len(missing));print(json.dumps([dict(path=x['path'],bytes=x['bytes']) for x in missing],indent=2));return rows,missing
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);data=inventory();match(data)
