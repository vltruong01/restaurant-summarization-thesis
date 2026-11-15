"""Read-only storage inventory and private backup availability; no source deletion."""
from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,subprocess,urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_closeout_20261112_v1'
def save(name,data):
 OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def inventory():
 totals=collections.defaultdict(lambda:dict(bytes=0,files=0));rows=[];errors=[];links=[]
 for base,dirs,files in os.walk(ROOT,followlinks=False):
  for name in list(dirs):
   p=Path(base)/name
   if p.is_symlink() or getattr(p.lstat(),'st_file_attributes',0)&0x400:links.append(p.relative_to(ROOT).as_posix());dirs.remove(name)
  for name in files:
   p=Path(base)/name
   try:
    stat=p.stat();relative=p.relative_to(ROOT).as_posix();parts=relative.split('/')
    if p.is_symlink() or getattr(p.lstat(),'st_file_attributes',0)&0x400:links.append(relative);continue
    row=dict(path=relative,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns);rows.append(row)
    for level in range(1,min(4,len(parts))):
     key='/'.join(parts[:level]);totals[key]['bytes']+=stat.st_size;totals[key]['files']+=1
   except OSError as error:errors.append(dict(path=str(p),error=type(error).__name__))
 result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(ROOT),logical_bytes=sum(r['bytes'] for r in rows),file_count=len(rows),free_bytes=shutil.disk_usage(ROOT).free,errors=errors,reparse_points=links,groups=[dict(path=k,**v) for k,v in sorted(totals.items(),key=lambda p:-p[1]['bytes'])],largest_files=sorted(rows,key=lambda r:-r['bytes'])[:100])
 save('INVENTORY.json',result);save('ALL_FILE_METADATA.json',rows)
 return dict(logical_GiB=round(result['logical_bytes']/2**30,3),files=len(rows),free_GiB=round(result['free_bytes']/2**30,3),errors=len(errors),top_groups=[dict(path=r['path'],GiB=round(r['bytes']/2**30,3),files=r['files']) for r in result['groups'][:45]],largest_files=[dict(path=r['path'],GiB=round(r['bytes']/2**30,3)) for r in result['largest_files'][:25]])
def remote():
 result={}
 try:
  from huggingface_hub import HfApi
  api=HfApi();user=api.whoami();info=api.repo_info('vltruong01/multimodal_summarization_thesis',repo_type='dataset',files_metadata=True)
  result['HF']=dict(status='AVAILABLE',user=user['name'],repo=info.id,private=info.private,revision=info.sha,files=len(info.siblings))
  save('HF_REMOTE_FILES.json',[dict(path=r.rfilename,bytes=r.size,lfs_sha256=r.lfs.sha256 if r.lfs else None) for r in info.siblings])
 except Exception as error:result['HF']=dict(status='UNAVAILABLE',error_type=type(error).__name__)
 try:
  env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
  proc=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=25)
  auth=dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line)
  assert proc.returncode==0 and auth.get('password')
  req=urllib.request.Request('https://api.github.com/repos/vltruong01/multimodal_summarization_thesis',headers={'Authorization':'Bearer '+auth['password'],'Accept':'application/vnd.github+json','User-Agent':'thesis-storage-closeout'})
  with urllib.request.urlopen(req,timeout=30) as response:data=json.load(response)
  result['GitHub']=dict(status='AVAILABLE',repo=data['full_name'],private=data['private'],default_branch=data['default_branch'],size_KiB=data['size'])
 except Exception as error:result['GitHub']=dict(status='UNAVAILABLE',error_type=type(error).__name__)
 save('REMOTE_AVAILABILITY.json',result);return result
if __name__=='__main__':
 print(json.dumps(dict(inventory=inventory(),remote=remote()),indent=2))
