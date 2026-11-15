"""Private, content-addressed archival; remove only unchanged, remotely verified cold files.

Explicit researcher authorization: archive completed research to existing private
HF/GitHub and remove unnecessary local copies. Every removal has a per-file hash,
pinned HF revision and independently streamed, verified blob backing it.
"""
from pathlib import Path,PurePosixPath
import argparse,collections,datetime,gzip,hashlib,io,json,os,re,shutil,sys,tarfile,time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_closeout_20261112_v1'
REPO='vltruong01/multimodal_summarization_thesis'
NAMESPACE='research_closeout/20261112_v1'
COLD=['outputs/stage_c','outputs/stage_i','outputs/cloud_handoff','outputs/stage_f','outputs/stage_g','outputs/stage_k','assets/experimental_scorers']
TRANSPORT='outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/STAGE_O_INPUTS_AND_LED_ASSETS.tar'
TOKEN=re.compile(rb'(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_-]{20,}')
PASSWORD=re.compile(rb'''(?im)^\s*["']?(?:password|passwd|api_key|access_token|secret_key|hf_token|github_token)["']?\s*[:=]\s*["']([^"'\r\n]{5,})["']''')
TEXT={'.py','.json','.jsonl','.yaml','.yml','.toml','.txt','.md','.sh','.ps1','.cmd','.bat','.html','.ipynb','.csv','.tsv','.log','.ini','.cfg'}
SAFE_DIRECTORIES={}
FILE_DIGESTS={}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.writing')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(path)
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def event(kind,**values):
 with (OUT/'ARCHIVE_PROGRESS.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(dict(time=now(),event=kind,**values),ensure_ascii=False)+'\n')
 print(json.dumps(dict(event=kind,**values),ensure_ascii=False),flush=True)
def safe(relative):
 relative_path=Path(relative);assert not relative_path.is_absolute() and '..' not in relative_path.parts
 p=ROOT/relative_path;parent=p.parent;parent_stat=parent.stat();identity=(parent_stat.st_dev,parent_stat.st_ino)
 if SAFE_DIRECTORIES.get(parent)!=identity:
  resolved=parent.resolve();assert resolved.is_relative_to(ROOT.resolve()),'Outside workspace'
  for q in [parent,*parent.parents]:
   if q==ROOT.parent:break
   stat=q.lstat();assert not getattr(stat,'st_file_attributes',0)&0x400,'Reparse directory retained'
  SAFE_DIRECTORIES[parent]=identity
 if p.exists():assert not getattr(p.lstat(),'st_file_attributes',0)&0x400,'Reparse file retained'
 return p
def credential_filename(relative):
 parts=PurePosixPath(relative).parts;name=parts[-1].lower()
 return any(x.lower() in {'.ssh','.aws','.azure','.git','state','private_backups','node_modules','__pycache__','.ipynb_checkpoints'} for x in parts) or name.startswith('.env') or name in {'id_rsa','id_ed25519','credentials','credentials.json','secrets.json','token','token.txt','accounts.json','accounts.sqlite3','auth.json','passwords.txt'} or name.endswith(('.sqlite','.sqlite3','.sqlite-wal','.sqlite-shm','.sqlite3-wal','.sqlite3-shm','.db','.pem','.key','.pyc','.pyo'))
def keep_prefixes():
 lock=read(ROOT/'outputs/final_research_analysis_20261106_v1/PACKAGE/SOURCES/M/PROTOCOL/MODEL_AND_DECODER_LOCK.json')
 return [m['path'] for m in lock['models'].values()]
def is_cold(path,keepers):
 if any(path==p or path.startswith(p+'/') for p in keepers):return False
 return path==TRANSPORT or any(path.startswith(p+'/') for p in COLD)
def include(path):
 if path.startswith('outputs/storage') or path.startswith('outputs/hf_dataset_backup_'):return False
 if path.startswith('outputs/') or path.startswith('data/') or path.startswith('assets/'):return True
 return path.startswith('notebooks/')
def plan():
 assert read(OUT/'GITHUB_VERIFIED_RECEIPT.json')['status']=='PASS_PRIVATE_PUSH_FRESH_CLONE_AND_ALL_FILE_HASHES'
 rows=read(OUT/'ALL_FILE_METADATA.json');keepers=keep_prefixes();blocked=[];chosen=[]
 for r in rows:
  if not include(r['path']):continue
  if credential_filename(r['path']):blocked.append(dict(path=r['path'],reason='Credential/account/environment/compiled file;retain local'));continue
  chosen.append({**r,'delete_after_verified':is_cold(r['path'],keepers)})
 def priority(row):
  path=row['path']
  if any(path.startswith(p+'/') for p in keepers):return 0
  if path.startswith(('outputs/stage_m/','outputs/stage_n/','outputs/stage_o/','outputs/stage_p/','outputs/stage_l/','outputs/final_research_analysis_','outputs/chapter4_','outputs/professor_notebook/')) and row['bytes']<100*2**20:return 1
  if path.startswith(('outputs/stage_f/','outputs/stage_g/','data/')):return 2
  return 3
 chosen.sort(key=lambda r:(priority(r),r['path']))
 groups=[];current=[];size=0
 for r in chosen:
  if current and (size+r['bytes']>768*2**20 or len(current)>=8000):groups.append(current);current=[];size=0
  current.append(r);size+=r['bytes']
 if current:groups.append(current)
 result=dict(format='FINAL_RESEARCH_OFFLOAD_PLAN_V1',created_at_utc=now(),root=str(ROOT),repo=REPO,namespace=NAMESPACE,retained_final_checkpoint_prefixes=keepers,cold_prefixes=COLD,cold_transport=TRANSPORT,excluded_and_retained=blocked,batches=[dict(id=f'part-{i+1:04d}',files=g,logical_bytes=sum(r['bytes'] for r in g)) for i,g in enumerate(groups)],logical_bytes=sum(r['bytes'] for r in chosen),cold_bytes=sum(r['bytes'] for r in chosen if r['delete_after_verified']),all_files=len(chosen),credentials_scan='Names and streaming recognizable literal/token scan;binary archive contents not exhaustively audited',no_changes_to_models_or_ratings=True)
 save(OUT/'OFFLOAD_PLAN.json',result);save(OUT/'OFFLOAD_PLAN_BINDING.json',dict(sha256=sha(OUT/'OFFLOAD_PLAN.json')))
 event('PLAN_READY',batches=len(groups),files=len(chosen),logical_GiB=round(result['logical_bytes']/2**30,3),cold_GiB=round(result['cold_bytes']/2**30,3),excluded=len(blocked))
 return result
class HashSource:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
class Remote:
 def __init__(self,response):self.chunks=iter(response.iter_content(4*2**20));self.buf=b'';self.h=hashlib.sha256();self.n=0
 def read(self,n):
  assert n>=0;out=bytearray()
  while len(out)<n:
   if not self.buf:
    try:self.buf=next(self.chunks)
    except StopIteration:break
    self.h.update(self.buf);self.n+=len(self.buf)
   take=min(n-len(out),len(self.buf));out.extend(self.buf[:take]);self.buf=self.buf[take:]
  return bytes(out)
def response(path,revision):
 import requests
 from huggingface_hub import hf_hub_url
 from huggingface_hub.utils import build_hf_headers
 r=requests.get(hf_hub_url(REPO,path,repo_type='dataset',revision=revision),headers=build_hf_headers(token=True),stream=True,timeout=(30,180));r.raise_for_status();return r
def filehash_scan(p):
 stat=p.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
 if identity in FILE_DIGESTS:return FILE_DIGESTS[identity]
 h=hashlib.sha256();tail=b'';sensitive=False;text=p.suffix.lower() in TEXT
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):
   h.update(b)
   if text:
    window=tail+b
    if TOKEN.search(window) or PASSWORD.search(window) or b'-----BEGIN OPENSSH PRIVATE KEY-----' in window or b'-----BEGIN RSA PRIVATE KEY-----' in window:sensitive=True
    tail=window[-2048:]
 result=(h.hexdigest(),sensitive);FILE_DIGESTS[identity]=result;return result
def pack(batch,locations,plan_hash):
 folder=OUT/'BATCHES'/batch['id'];folder.mkdir(parents=True,exist_ok=True)
 manifest_path=folder/'manifest.json';archive=folder/'blobs.tar.gz'
 if manifest_path.exists():
  manifest=read(manifest_path);assert manifest['plan_sha256']==plan_hash
  assert archive.exists() and sha(archive)==manifest['archive_sha256'];return manifest,archive,manifest_path
 if archive.exists():
  # An interrupted local pack has no verification receipt or deletion authority.
  orphan=folder/('incomplete-'+str(time.time_ns())+'.tar.gz');archive.rename(orphan)
 records=[];blobs={};excluded=[]
 with archive.open('xb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as tar:
  for row_index,row in enumerate(batch['files']):
   if row_index and row_index%2000==0:event('PACK_PROGRESS',batch=batch['id'],files_scanned=row_index,total_files=len(batch['files']))
   p=safe(row['path'])
   if not p.exists():excluded.append(dict(path=row['path'],reason='Missing since inventory;no deletion'));continue
   before=p.stat()
   if (before.st_size,before.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):excluded.append(dict(path=row['path'],reason='Changed since inventory;retain local'));continue
   digest,sensitive=filehash_scan(p)
   if sensitive:excluded.append(dict(path=row['path'],reason='Possible credential literal;retain local without upload'));continue
   record={**row,'sha256':digest,'source_mode':before.st_mode,'source_mtime_ns':before.st_mtime_ns}
   if digest in locations:record['blob']=locations[digest]
   elif digest in blobs:record['blob']=dict(kind='this_batch',member='blobs/'+digest)
   else:
    info=tar.gettarinfo(p,arcname='blobs/'+digest)
    with p.open('rb') as f:
     content=HashSource(f);tar.addfile(info,content)
    after=p.stat();assert content.n==row['bytes'] and content.h.hexdigest()==digest and (after.st_size,after.st_mtime_ns)==(before.st_size,before.st_mtime_ns),'Source changed while packing'
    blobs[digest]=dict(bytes=row['bytes'],member='blobs/'+digest);record['blob']=dict(kind='this_batch',member='blobs/'+digest)
   records.append(record)
 manifest=dict(format='FINAL_RESEARCH_CONTENT_BLOBS_V1',batch=batch['id'],created_at_utc=now(),plan_sha256=plan_hash,files=records,unique_blobs=blobs,excluded_and_retained=excluded,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,source_files=len(records))
 save(manifest_path,manifest);return manifest,archive,manifest_path
def publish_verify(api,batch,manifest,archive,mfile):
 from huggingface_hub import CommitOperationAdd
 prefix=NAMESPACE+'/'+batch['id']+'/'+manifest['archive_sha256'];arcpath=prefix+'/blobs.tar.gz';mpath=prefix+'/manifest.json'
 event('UPLOAD_START',batch=batch['id'],archive_MiB=round(archive.stat().st_size/2**20,2),source_files=len(manifest['files']),new_unique_blobs=len(manifest['unique_blobs']))
 latest=api.repo_info(REPO,repo_type='dataset',files_metadata=True);assert latest.private is True
 present={f.rfilename:f for f in latest.siblings};entry=present.get(arcpath)
 if entry is not None and entry.lfs and entry.lfs.sha256==manifest['archive_sha256'] and entry.size==manifest['archive_bytes'] and mpath in present:
  revision=latest.sha;event('VERIFY_EXISTING_REMOTE',batch=batch['id'],revision=revision)
 else:
  commit=api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=arcpath,path_or_fileobj=str(archive)),CommitOperationAdd(path_in_repo=mpath,path_or_fileobj=str(mfile))],commit_message='Preserve completed thesis research '+batch['id'])
  revision=commit.oid
 expected=manifest['unique_blobs'];seen=set()
 with response(arcpath,revision) as r:
  remote=Remote(r)
  with tarfile.open(fileobj=remote,mode='r|gz') as tar:
   for member in tar:
    assert member.isfile() and member.name.startswith('blobs/');digest=member.name[6:]
    assert digest in expected and digest not in seen and member.size==expected[digest]['bytes']
    h=hashlib.sha256();n=0
    with tar.extractfile(member) as f:
     for b in iter(lambda:f.read(4*2**20),b''):h.update(b);n+=len(b)
    assert h.hexdigest()==digest and n==member.size;seen.add(digest)
  while remote.read(4*2**20):pass
  assert remote.h.hexdigest()==manifest['archive_sha256'] and remote.n==manifest['archive_bytes'] and seen==set(expected)
 with response(mpath,revision) as r:
  h=hashlib.sha256()
  for b in r.iter_content(4*2**20):h.update(b)
 assert h.hexdigest()==sha(mfile)
 info=api.repo_info(REPO,repo_type='dataset',revision=revision,files_metadata=True);assert info.private is True
 remote_file=next(f for f in info.siblings if f.rfilename==arcpath)
 assert remote_file.size==manifest['archive_bytes'] and remote_file.lfs.sha256==manifest['archive_sha256']
 receipt=dict(status='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256',batch=batch['id'],revision=revision,repo=REPO,archive_remote_path=arcpath,manifest_remote_path=mpath,archive_sha256=manifest['archive_sha256'],archive_bytes=manifest['archive_bytes'],manifest_sha256=sha(mfile),verified_blobs=len(seen),source_files=len(manifest['files']),checked_at_utc=now())
 save(mfile.parent/'REMOTE_RECEIPT.json',receipt);event('REMOTE_VERIFIED',batch=batch['id'],files=len(manifest['files']),blobs=len(seen),revision=revision)
 return receipt
def local_delete(manifest,receipt,locations,keepers):
 assert receipt['status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
 journal=OUT/'DELETED_FILES.jsonl';deleted=0;size=0;retained=0
 for row in manifest['files']:
  if not row['delete_after_verified'] or not is_cold(row['path'],keepers):continue
  p=safe(row['path'])
  if not p.exists():continue
  s=p.stat()
  if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):retained+=1;continue
  if sha(p)!=row['sha256']:retained+=1;continue
  location=locations[row['sha256']]
  # All referenced locations come exclusively from completed fresh-remote verification receipts.
  assert location['verification_status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
  with journal.open('a',encoding='utf8') as stream:
   stream.write(json.dumps(dict(event='DELETE_INTENT',path=row['path'],bytes=row['bytes'],sha256=row['sha256'],restore=location,time=now()))+'\n');stream.flush();os.fsync(stream.fileno())
  p.unlink();deleted+=1;size+=row['bytes']
  with journal.open('a',encoding='utf8') as stream:stream.write(json.dumps(dict(event='DELETED',path=row['path'],bytes=row['bytes'],sha256=row['sha256'],time=now()))+'\n')
 return dict(deleted_files=deleted,deleted_logical_bytes=size,changed_retained=retained)
def run(delete=False):
 from huggingface_hub import HfApi
 from huggingface_hub.utils import disable_progress_bars
 disable_progress_bars();api=HfApi();assert api.whoami()['name']=='vltruong01' and api.repo_info(REPO,repo_type='dataset').private is True
 plan_path=OUT/'OFFLOAD_PLAN.json';d=read(plan_path) if plan_path.exists() else plan();plan_hash=sha(plan_path);assert plan_hash==read(OUT/'OFFLOAD_PLAN_BINDING.json')['sha256']
 locations={};startfree=shutil.disk_usage(ROOT).free;totals=collections.Counter()
 for batch in d['batches']:
  folder=OUT/'BATCHES'/batch['id'];receipt_path=folder/'REMOTE_RECEIPT.json'
  if receipt_path.exists():
   receipt=read(receipt_path);manifest=read(folder/'manifest.json');assert sha(folder/'manifest.json')==receipt['manifest_sha256']
   assert manifest['plan_sha256']==plan_hash
   # Pinned LFS identity rechecked on every resumed batch before enabling any deletion.
   info=api.repo_info(REPO,repo_type='dataset',revision=receipt['revision'],files_metadata=True);assert info.private is True
   entry=next(f for f in info.siblings if f.rfilename==receipt['archive_remote_path']);assert entry.lfs.sha256==receipt['archive_sha256'] and entry.size==receipt['archive_bytes']
  else:
   manifest,archive,mfile=pack(batch,locations,plan_hash);receipt=publish_verify(api,batch,manifest,archive,mfile)
  for digest,blob in manifest['unique_blobs'].items():locations[digest]=dict(repo=REPO,revision=receipt['revision'],archive_path=receipt['archive_remote_path'],member=blob['member'],verification_status=receipt['status'])
  for row in manifest['files']:assert row['sha256'] in locations,'Every duplicate reference must resolve to verified bytes'
  totals['backed_up_files']+=len(manifest['files']);totals['backed_up_logical_bytes']+=sum(r['bytes'] for r in manifest['files']);totals['remote_archive_bytes']+=manifest['archive_bytes'];totals['excluded_sensitive_or_changed']+=len(manifest['excluded_and_retained'])
  if delete:
   outcome=local_delete(manifest,receipt,locations,d['retained_final_checkpoint_prefixes']);totals.update(outcome)
  save(OUT/'BLOB_RESTORE_INDEX.json',locations)
  arc=folder/'blobs.tar.gz'
  if arc.exists():assert sha(arc)==receipt['archive_sha256'];arc.unlink()
  save(OUT/'STATUS.json',dict(status='RUNNING',updated_at_utc=now(),batch=batch['id'],total_batches=len(d['batches']),totals=dict(totals),free_GiB=round(shutil.disk_usage(ROOT).free/2**30,3),delete_authorized=delete))
  event('BATCH_COMPLETE',batch=batch['id'],deleted_files=totals['deleted_files'],deleted_logical_GiB=round(totals['deleted_logical_bytes']/2**30,3),free_GiB=round(shutil.disk_usage(ROOT).free/2**30,3))
 result=dict(status='PASS_BACKUP_AND_VERIFIED_COLD_OFFLOAD_COMPLETE',completed_at_utc=now(),repo=REPO,namespace=NAMESPACE,plan_sha256=plan_hash,batches=len(d['batches']),unique_content_blobs=len(locations),totals=dict(totals),free_before_bytes=startfree,free_after_bytes=shutil.disk_usage(ROOT).free,retained_final_checkpoint_prefixes=d['retained_final_checkpoint_prefixes'],retained_live_apps=True,delete_authorized=delete)
 save(OUT/'FINAL_HF_OFFLOAD_RECEIPT.json',result);event('COMPLETE',**{k:v for k,v in result.items() if k!='retained_final_checkpoint_prefixes'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan-only',action='store_true');p.add_argument('--delete-verified-cold',action='store_true');a=p.parse_args()
 if a.plan_only:plan()
 else:run(delete=a.delete_verified_cold)
if __name__=='__main__':
 try:main()
 except Exception as error:
  event('STOPPED',error_type=type(error).__name__,source_policy='Unverified and changed local files retained;only already verified cold files may have been removed')
  raise SystemExit(1)
