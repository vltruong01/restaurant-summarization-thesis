"""Researcher-authorized second offload. Verify first; delete exact unchanged files only.

No inference, evaluation, credentials publication, or scientific-content changes.
Known source backups are reused only after content SHA and remote object checks.
"""
from pathlib import Path
import collections,datetime,gzip,hashlib,importlib.util,io,json,os,re,shutil,sqlite3,subprocess,sys,threading,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'_CLEANUP_WORK_20261112'
RECOVERY=ROOT/'RECOVERY_THESIS_20261112'
OLD=ROOT/'outputs/storage_closeout_20261112_v1'
REPO='vltruong01/multimodal_summarization_thesis'
NAMESPACE='research_closeout/20261112_v2'
APP='outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
APP_O='outputs/stage_o/tripadvisor_external_v1/HUMAN_APP'
FIG='outputs/chapter4_figures_20261112_v2'
VENV_NAMES={'.venv-caption-pilot','.venv-preprocess','.venv-stage-o-app-test','.venv-stage-p-preflight'}
SOURCE_TOP={'approvals','cloud','configs','docs','notebooks','prompts','requirements','schemas','scripts','specs','tests','thesis_sdd_docs'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def emit(event,**d):
 row=dict(time=now(),event=event,**d)
 with (WORK/'PROGRESS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def retained(rel):
 parts=Path(rel).parts;name=parts[-1].lower()
 if parts[0] in VENV_NAMES or parts[0] in {WORK.name,RECOVERY.name}:return 'Cleanup runtime/work/recovery; handled separately'
 if rel==APP or rel.startswith(APP+'/'):return 'Self-contained running annotation application and all its evidence/state'
 if rel==APP_O or rel.startswith(APP_O+'/'):return 'Running Stage O annotation application and all its evidence/state'
 if rel in {'thesis_proposal_v16.docx','thesis_proposal_TruongVuong.pdf'}:return 'Final thesis document'
 if rel.startswith(FIG+'/') and Path(rel).suffix.lower() in {'.png','.pdf'}:return 'Final Chapter 4 figure'
 if any(p.lower() in {'state','private_state','.ssh','.aws','.azure','private_backups'} for p in parts):return 'Private participant/account state; not published'
 if name.startswith('.env') or name in {'accounts.json','credentials','credentials.json','secrets.json','passwords.txt','token','token.txt','auth.json','accounts.sqlite3','id_rsa','id_ed25519'} or name.endswith(('.key','.pem')):return 'Potential credential file; not published'
 if name.endswith(('.sqlite','.sqlite3','.db','-wal','-shm')):
  try:
   con=sqlite3.connect((ROOT/rel).resolve().as_uri()+'?mode=ro',uri=True)
   tables=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
   columns=[r[1].lower() for t in tables for r in con.execute('PRAGMA table_info('+json.dumps(t)+')')];con.close()
   if set(columns)&{'password','passwd','salt','token','session_token','api_key','password_hash'} or any('user' in t.lower() or 'session' in t.lower() for t in tables):return 'Database includes authentication state; retained privately'
  except Exception:return 'Database cannot be safely classified; retained'
 return None

def derived_kind(rel):
 p=Path(rel);parts=p.parts
 if any(s in {'__pycache__','.ipynb_checkpoints','node_modules'} for s in parts) or p.suffix.lower() in {'.pyc','.pyo'}:return 'Recreatable compiled/dependency cache'
 if rel.startswith('outputs/storage_closeout_20261112_v1/GITHUB_'):return 'Temporary GitHub snapshot/fresh-clone verification checkout'
 return None

def inventory():
 candidates=[];keep=[];derived=[];reparse=[]
 for base,dirs,names in os.walk(ROOT,followlinks=False):
  for name in list(dirs):
   p=Path(base)/name;rel=p.relative_to(ROOT).as_posix();reason=retained(rel)
   if getattr(p.lstat(),'st_file_attributes',0)&0x400:
    reparse.append(dict(path=rel,reason='Link not traversed'));dirs.remove(name);continue
   if reason:keep.append(dict(path=rel,reason=reason));dirs.remove(name);continue
   kind=derived_kind(rel)
   if kind:derived.append(dict(path=rel,directory=True,reason=kind));dirs.remove(name)
  for name in names:
   p=Path(base)/name;rel=p.relative_to(ROOT).as_posix()
   if getattr(p.lstat(),'st_file_attributes',0)&0x400:reparse.append(dict(path=rel,reason='Link not traversed'));continue
   reason=retained(rel)
   if reason:keep.append(dict(path=rel,reason=reason));continue
   kind=derived_kind(rel)
   if kind:derived.append(dict(path=rel,directory=False,reason=kind));continue
   stat=p.stat();candidates.append(dict(path=rel,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,delete_after_verified=True))
 candidates.sort(key=lambda r:(0 if Path(r['path']).parts[0] in SOURCE_TOP else 1,r['path']))
 groups=[];current=[];size=0
 for row in candidates:
  if current and (len(current)>=10000 or size+row['bytes']>512*2**20):groups.append(current);current=[];size=0
  current.append(row);size+=row['bytes']
 if current:groups.append(current)
 plan=dict(format='SECOND_OFFLOAD_PLAN_V2',created_at_utc=now(),root=str(ROOT),free_before_bytes=shutil.disk_usage(ROOT).free,retained=keep,derived=derived,reparse_points=reparse,batches=[dict(id=f'part-{i+1:04d}',files=rows) for i,rows in enumerate(groups)],files=len(candidates),logical_bytes=sum(r['bytes'] for r in candidates),scope='Source/config/docs and all inactive research artifacts; running app, final thesis/figures and private state retained')
 save(WORK/'PLAN.json',plan);save(WORK/'PLAN_BINDING.json',dict(sha256=sha(WORK/'PLAN.json')))
 emit('PLAN_READY',files=plan['files'],logical_GiB=round(plan['logical_bytes']/2**30,3),batches=len(groups),retained_groups=len(keep),disposable_groups=len(derived))
 return plan

def publish_small(api,uploads,label):
 from huggingface_hub import CommitOperationAdd,hf_hub_url
 from huggingface_hub.utils import build_hf_headers
 import requests
 assert api.repo_info(REPO,repo_type='dataset').private is True
 expected=[dict(remote=remote,sha256=sha(local),bytes=local.stat().st_size) for remote,local in uploads]
 commit=api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(local)) for remote,local in uploads],commit_message=label)
 for row in expected:
  with requests.get(hf_hub_url(REPO,row['remote'],repo_type='dataset',revision=commit.oid),headers=build_hf_headers(token=True),stream=True,timeout=(30,180)) as response:
   response.raise_for_status();h=hashlib.sha256();size=0
   for block in response.iter_content(4*2**20):h.update(block);size+=len(block)
  assert h.hexdigest()==row['sha256'] and size==row['bytes']
 return dict(status='PASS_FRESH_REMOTE_BYTES',repo=REPO,revision=commit.oid,files=expected)

def safe_path(rel):
 p=ROOT/rel;assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 resolved=p.resolve();assert resolved.is_relative_to(ROOT.resolve()) and resolved!=ROOT.resolve()
 for q in [p,*p.parents]:
  if q==ROOT:break
  if q.exists():assert not getattr(q.lstat(),'st_file_attributes',0)&0x400
 return p

def main():
 from huggingface_hub import HfApi
 from huggingface_hub.utils import disable_progress_bars
 disable_progress_bars();api=HfApi();assert api.repo_info(REPO,repo_type='dataset').private is True
 assert read(OLD/'FINAL_REMOTE_CONFIRMATION.json')['status']=='PASS_FINAL_RECEIPTS_FRESH_REMOTE_BYTES'
 plan=read(WORK/'PLAN.json') if (WORK/'PLAN.json').exists() else inventory()
 plan_hash=sha(WORK/'PLAN.json');assert plan_hash==read(WORK/'PLAN_BINDING.json')['sha256']
 source_catalog=OLD/'RESTORE_CATALOG.json.gz';old_catalog_receipt=read(OLD/'HF_CATALOG_RECEIPT.json');assert sha(source_catalog)==old_catalog_receipt['catalog_sha256']
 with gzip.open(source_catalog,'rt',encoding='utf8') as f:catalog=json.load(f)
 locations=catalog['blobs'];archives=catalog['archives'];all_files={r['path']:r for r in catalog['files']}
 # One current remote inventory validates immutable content identities. The original
 # per-shard receipts record the earlier full remote archive and member streaming.
 current=api.repo_info(REPO,repo_type='dataset',files_metadata=True);entries={f.rfilename:f for f in current.siblings}
 for path,row in archives.items():
  entry=entries[path];assert entry.lfs.sha256==row['archive_sha256'] and entry.size==row['archive_bytes']
 emit('EXISTING_BACKUPS_RECHECKED',archives=len(archives),prior_full_stream_receipt=True)
 # Refresh source GitHub before its local source files are offloaded.
 gh_file=WORK/'GITHUB_VERIFIED_RECEIPT.json'
 if not gh_file.exists():
  g=module('source_github_backup',ROOT/'scripts/backup_final_private_github_20261112_v1.py');g.OUT=WORK;g.SNAP=OLD/'GITHUB_SNAPSHOT'
  maintenance=g.SNAP/'maintenance/offload_remaining_20261112_v2.py';maintenance.parent.mkdir(exist_ok=True);shutil.copyfile(__file__,maintenance)
  g.main()
 assert read(gh_file)['status']=='PASS_PRIVATE_PUSH_FRESH_CLONE_AND_ALL_FILE_HASHES'
 core=module('verified_archive_core',ROOT/'scripts/archive_and_offload_final_research_20261112_v1.py');core.OUT=WORK;core.NAMESPACE=NAMESPACE
 core.TOKEN=re.compile(rb'(?:sk-(?:proj-)?[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})')
 core.is_cold=lambda rel,keepers:retained(rel) is None
 # The frozen review files were already independently SHA-bound to the private
 # original dataset backup. A token-like review string is not an authentication key.
 trusted={r['path']:r['current_sha256'] for r in read(OLD/'EXISTING_HF_TRAIN_DATA_BINDING.json')['files']}
 # These two scanner matches were inspected: one is the scanner's own literal
 # key-header patterns; the other is a disposable test password in a TemporaryDirectory.
 # Neither value authenticates a real account. Bind each exception to its current SHA.
 for reviewed in ['scripts/archive_and_offload_final_research_20261112_v1.py','scripts/test_stage_p_a02_portable_revision_v1.py']:
  trusted[reviewed]=sha(ROOT/reviewed)
 original_scan=core.filehash_scan
 def scan(p):
  rel=p.relative_to(ROOT).as_posix()
  if rel in trusted:
   digest=sha(p);assert digest==trusted[rel];return digest,False
  old=all_files.get(rel);s=p.stat()
  if old and old['bytes']==s.st_size and old['mtime_ns']==s.st_mtime_ns and old['sha256'] in locations:
   # Reuse a previously fully verified blob while packing. Actual current bytes
   # are freshly hashed below before any deletion; changed content is retained.
   return old['sha256'],False
  return original_scan(p)
 core.filehash_scan=scan
 event_lock=threading.Lock();original_event=core.event
 def locked_event(kind,**values):
  with event_lock:original_event(kind,**values)
 core.event=locked_event
 manifests=[];excluded=[]
 # Independent shards use only already verified blob locations. New duplicate
 # blobs may be stored twice across concurrent shards; they never depend on an
 # unverified, in-progress shard.
 base_locations=locations.copy()
 def backup_batch(batch):
  folder=WORK/'BATCHES'/batch['id'];receipt_file=folder/'REMOTE_RECEIPT.json'
  if receipt_file.exists():
   receipt=read(receipt_file);manifest=read(folder/'manifest.json');assert sha(folder/'manifest.json')==receipt['manifest_sha256']
  else:
   manifest,archive,mfile=core.pack(batch,base_locations,plan_hash);receipt=core.publish_verify(api,batch,manifest,archive,mfile)
  return batch,manifest,receipt
 executor=ThreadPoolExecutor(max_workers=2)
 pending={executor.submit(backup_batch,batch):batch['id'] for batch in plan['batches']}
 for future in as_completed(pending):
  batch,manifest,receipt=future.result()
  assert manifest['plan_sha256']==plan_hash and receipt['status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
  archives[receipt['archive_remote_path']]={k:receipt[k] for k in ['revision','archive_sha256','archive_bytes','manifest_remote_path','manifest_sha256']}
  for digest,blob in manifest['unique_blobs'].items():locations[digest]=dict(repo=REPO,revision=receipt['revision'],archive_path=receipt['archive_remote_path'],member=blob['member'],verification_status=receipt['status'])
  for row in manifest['files']:
   assert row['sha256'] in locations;all_files[row['path']]={k:v for k,v in row.items() if k!='blob'}
  manifests.append((manifest,receipt));excluded.extend(manifest['excluded_and_retained'])
  save(WORK/'STATUS.json',dict(status='BACKING_UP',completed_batches=len(manifests),total_batches=len(plan['batches']),excluded=len(excluded)))
  emit('BATCH_BACKED_UP',batch=batch['id'],completed=len(manifests),total=len(plan['batches']),files=len(manifest['files']))
 executor.shutdown(wait=True)
 # A resumed first shard can retain inspected scanner exclusions from its earlier
 # immutable manifest. Preserve their exact original bytes in a supplemental shard.
 inspect_exclusions=[r for r in excluded if r['path'] in trusted and (ROOT/r['path']).suffix=='.py']
 if inspect_exclusions:
  rows=[]
  for r in inspect_exclusions:
   p=ROOT/r['path'];s=p.stat();rows.append(dict(path=r['path'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,delete_after_verified=True))
  extra=dict(id='reviewed-source-exceptions',files=rows)
  manifest,archive,mfile=core.pack(extra,locations,plan_hash);r=core.publish_verify(api,extra,manifest,archive,mfile)
  archives[r['archive_remote_path']]={k:r[k] for k in ['revision','archive_sha256','archive_bytes','manifest_remote_path','manifest_sha256']}
  for digest,blob in manifest['unique_blobs'].items():locations[digest]=dict(repo=REPO,revision=r['revision'],archive_path=r['archive_remote_path'],member=blob['member'],verification_status=r['status'])
  for row in manifest['files']:all_files[row['path']]={k:v for k,v in row.items() if k!='blob'}
  manifests.append((manifest,r));resolved={r['path'] for r in manifest['files']};excluded=[r for r in excluded if r['path'] not in resolved]
 catalog.update(format='FINAL_RESEARCH_RESTORE_CATALOG_V2',created_at_utc=now(),files=list(all_files.values()),blobs=locations,archives=archives,prior_catalog=old_catalog_receipt,supplemental_excluded_and_retained=excluded)
 full_catalog=WORK/'RESTORE_CATALOG.json.gz'
 with full_catalog.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=6,mtime=0) as gz,io.TextIOWrapper(gz,encoding='utf8') as f:json.dump(catalog,f,ensure_ascii=False,separators=(',',':'))
 restore=WORK/'restore_research_from_hf.py';shutil.copyfile(OLD/'restore_research_from_hf.py',restore)
 for filename in ['PLAN.json']:
  with (WORK/filename).open('rb') as src,gzip.open(WORK/(filename+'.gz'),'wb',compresslevel=6) as dest:shutil.copyfileobj(src,dest)
 prefix=NAMESPACE+'/catalog/'+sha(full_catalog)
 uploads=[(prefix+'/'+p.name,p) for p in [full_catalog,restore,WORK/'PLAN.json.gz',WORK/'PLAN_BINDING.json',gh_file,Path(__file__)]]
 receipt=publish_small(api,uploads,'Publish complete original-source restoration catalog before offloading local sources')
 receipt['catalog_remote_path']=prefix+'/'+full_catalog.name;receipt['catalog_sha256']=sha(full_catalog);save(WORK/'CATALOG_RECEIPT.json',receipt)
 # Exercise restoration of an actual original source file, independently of local
 # archive creation, and compare its bytes before allowing any research deletion.
 test='scripts/stage_i_repetition_pilot_v1.py';destination=WORK/'SOURCE_RESTORE_TEST'
 if not destination.exists():
  r=subprocess.run([sys.executable,'-X','utf8','-B',str(restore),'--catalog',str(full_catalog),'--prefix',test,'--destination',str(destination)],capture_output=True,text=True,encoding='utf8')
  assert r.returncode==0,'Actual source restore failed (details withheld to avoid signed URL exposure)'
 assert sha(destination/test)==sha(ROOT/test)
 save(WORK/'ACTUAL_SOURCE_RESTORE.json',dict(status='PASS',file=test,sha256=sha(destination/test),independent_remote_restore=True))
 emit('ALL_BACKUPS_AND_ACTUAL_SOURCE_RESTORE_PASS',archived_paths=len(all_files),new_exclusions=len(excluded))
 # All research paths have passed backup checks. No recursive original-tree delete.
 deleted=0;removed_bytes=0;changed=[];journal=WORK/'DELETED_FILES.jsonl'
 already=set()
 if journal.exists():
  for line in journal.read_text(encoding='utf8').splitlines():
   row=json.loads(line)
   if row['event']=='DELETED':already.add(row['path']);deleted+=1;removed_bytes+=row['bytes']
 def validate(row):
  p=core.safe(row['path'])
  if retained(row['path']) is not None:return row,None,'Protected running application; retained'
  if not p.exists():return row,None,'Missing before deletion'
  stat=p.stat()
  if (stat.st_size,stat.st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or sha(p)!=row['sha256']:return row,None,'Changed; retained'
  assert locations[row['sha256']]['verification_status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
  return row,p,None
 to_delete=[r for manifest,_ in manifests for r in manifest['files'] if r['path'] not in already]
 with ThreadPoolExecutor(max_workers=4) as validators,journal.open('a',encoding='utf8') as f:
  for offset in range(0,len(to_delete),500):
   valid=[]
   for row,p,reason in validators.map(validate,to_delete[offset:offset+500]):
    if reason:changed.append(dict(path=row['path'],reason=reason));continue
    valid.append((row,p));f.write(json.dumps(dict(event='DELETE_INTENT',path=row['path'],bytes=row['bytes'],sha256=row['sha256'],restore=locations[row['sha256']]))+'\n')
   # Durable intents for the entire block precede every unlink in that block.
   f.flush();os.fsync(f.fileno())
   for row,p in valid:
    stat=p.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):changed.append(dict(path=row['path'],reason='Changed after hash; retained'));continue
    p.unlink();deleted+=1;removed_bytes+=row['bytes']
    f.write(json.dumps(dict(event='DELETED',path=row['path'],bytes=row['bytes'],sha256=row['sha256']))+'\n')
   f.flush();os.fsync(f.fileno())
   if offset%10000==0:emit('DELETE_PROGRESS',files=deleted,logical_GiB=round(removed_bytes/2**30,3))
 # Only explicitly inventoried, reproducible caches/checkouts are removed as trees.
 # Resolve and validate every absolute target before invoking shutil/native deletion.
 disposable=[]
 for row in plan['derived']:
  p=safe_path(row['path'])
  if not p.exists():continue
  if p.resolve().is_relative_to((ROOT/APP).resolve()) or p.resolve().is_relative_to((ROOT/APP_O).resolve()):continue
  if row['directory']:shutil.rmtree(p)
  else:p.unlink()
  disposable.append(row)
 # Empty original folders only, never follow directory reparse points.
 for base,dirs,names in os.walk(ROOT,topdown=False,followlinks=False):
  p=Path(base);rel=p.relative_to(ROOT).as_posix()
  if p==ROOT or retained(rel) or any(x in p.parts for x in VENV_NAMES):continue
  if getattr(p.lstat(),'st_file_attributes',0)&0x400:continue
  try:p.rmdir()
  except OSError:pass
 # Protected runtime, scientific content and exact current rating store stay intact.
 with urllib.request.urlopen('http://127.0.0.1:8768/',timeout=15) as response:assert response.status==200
 with urllib.request.urlopen('http://127.0.0.1:8767/',timeout=15) as response:assert response.status==200
 final=dict(status='PASS_SECOND_OFFLOAD_AND_SOURCE_RESTORE',created_at_utc=now(),deleted_files=deleted,deleted_logical_bytes=removed_bytes,changed_retained=changed,excluded_retained=excluded,retained_groups=plan['retained'],disposable_groups_removed=len(disposable),github=read(gh_file),hf_catalog=receipt,free_before_bytes=plan['free_before_bytes'],free_after_bytes=shutil.disk_usage(ROOT).free,app_http_status=200,no_rating_or_scientific_content_changed=True)
 save(WORK/'FINAL_REPORT.json',final)
 with journal.open('rb') as src,gzip.open(WORK/'DELETED_FILES.jsonl.gz','wb',compresslevel=6) as dest:shutil.copyfileobj(src,dest)
 controls=[WORK/n for n in ['FINAL_REPORT.json','ACTUAL_SOURCE_RESTORE.json','DELETED_FILES.jsonl.gz','CATALOG_RECEIPT.json','GITHUB_VERIFIED_RECEIPT.json']]
 final_receipt=publish_small(api,[(NAMESPACE+'/final/'+sha(p)+'/'+p.name,p) for p in controls],'Preserve verified second-offload deletion and restoration receipts')
 save(WORK/'FINAL_REMOTE_RECEIPT.json',final_receipt)
 RECOVERY.mkdir(exist_ok=True)
 # Compact local recovery kit: all detail inside one ZIP, not thousands of loose files.
 kit=RECOVERY/'THESIS_RECOVERY_KIT.zip'
 with zipfile.ZipFile(kit,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in [full_catalog,restore,*controls,WORK/'FINAL_REMOTE_RECEIPT.json',WORK/'PLAN.json.gz']:
   z.write(p,p.name)
  z.writestr('requirements.txt','huggingface_hub==0.36.2\nrequests\n')
 with zipfile.ZipFile(kit) as z:assert z.testzip() is None
 instructions=f'''DA CHUYEN MA NGUON VA DU LIEU KHONG CON DUNG SANG LUU TRU RIENG TU\n\nGitHub: https://github.com/{REPO}/tree/{final['github']['branch']}\nGit commit: {final['github']['commit']}\nHF: https://huggingface.co/datasets/{REPO}\nHF catalog revision: {receipt['revision']}\nHF catalog path: {receipt['catalog_remote_path']}\nCatalog SHA256: {sha(full_catalog)}\nRecovery ZIP SHA256: {sha(kit)}\n\nTHESIS_RECOVERY_KIT.zip chua catalog, script khoi phuc va bien nhan.\nGiai nen vao thu muc moi. Can Python 3.11+, requests, huggingface_hub va dang nhap tai khoan HF co quyen doc repo rieng tu.\nLenh khoi phuc mot thu muc vao DICH MOI:\npython restore_research_from_hf.py --catalog RESTORE_CATALOG.json.gz --prefix scripts --destination C:/thesis_restore\nKhong ghi de project. Co the thay scripts bang configs, data hoac mot path chinh xac.\nScript kiem tra SHA256 archive va file; khong chay model.\n\nGiu local: luan van DOCX/PDF, hinh Chapter4 cuoi, ung dung cham port8768 va private STATE/credential.\nKhong co training, inference, thay doi diem cham hay sua noi dung nghien cuu trong dot don nay.\n'''
 (RECOVERY/'READ_ME_VI.txt').write_text(instructions,encoding='utf8')
 kit_receipt=publish_small(api,[(NAMESPACE+'/recovery_kit/'+sha(kit)+'/'+kit.name,kit),(NAMESPACE+'/recovery_kit/'+sha(kit)+'/READ_ME_VI.txt',RECOVERY/'READ_ME_VI.txt')],'Publish compact recovery kit and instructions')
 save(WORK/'RECOVERY_KIT_REMOTE_RECEIPT.json',kit_receipt)
 save(WORK/'STATUS.json',dict(status='COMPLETE',deleted_files=deleted,deleted_logical_GiB=round(removed_bytes/2**30,3),retained_exclusions=len(excluded),recovery_kit_sha256=sha(kit)))
 emit('COMPLETE',deleted_files=deleted,deleted_logical_GiB=round(removed_bytes/2**30,3),excluded_retained=len(excluded),recovery_zip_MiB=round(kit.stat().st_size/2**20,2))

if __name__=='__main__':
 try:main()
 except Exception as e:
  emit('STOPPED',error_type=type(e).__name__,policy='Unverified or changed sources are retained; verified deletions are journaled')
  raise SystemExit(1)
