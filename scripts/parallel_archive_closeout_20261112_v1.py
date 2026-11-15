"""Run independent archival shards concurrently, with one sequential deletion coordinator."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import collections,datetime,importlib.util,json,shutil,sys,threading

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_core',ROOT/'scripts/archive_and_offload_final_research_20261112_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
LOCK=threading.Lock();original_event=m.event
def event(kind,**values):
 with LOCK:original_event(kind,**values)
m.event=event
def work(batch,locations,plan_hash):
 from huggingface_hub import HfApi
 manifest,archive,mfile=m.pack(batch,locations,plan_hash)
 receipt=m.publish_verify(HfApi(),batch,manifest,archive,mfile)
 return batch,manifest,receipt
def main():
 from huggingface_hub import HfApi
 from huggingface_hub.utils import disable_progress_bars
 disable_progress_bars();api=HfApi();assert api.whoami()['name']=='vltruong01' and api.repo_info(m.REPO,repo_type='dataset').private is True
 plan=m.read(m.OUT/'OFFLOAD_PLAN.json');plan_hash=m.sha(m.OUT/'OFFLOAD_PLAN.json');assert plan_hash==m.read(m.OUT/'OFFLOAD_PLAN_BINDING.json')['sha256']
 locations={};totals=collections.Counter();done=0;free_start=shutil.disk_usage(ROOT).free
 already_complete=set()
 for line in (m.OUT/'ARCHIVE_PROGRESS.jsonl').read_text(encoding='utf8').splitlines():
  old=json.loads(line)
  if old['event']=='BATCH_COMPLETE':already_complete.add(old['batch'])
 old_deletes=m.OUT/'DELETED_FILES.jsonl'
 if old_deletes.exists():
  for line in old_deletes.read_text(encoding='utf8').splitlines():
   row=json.loads(line)
   if row['event']=='DELETED':totals['deleted_files']+=1;totals['deleted_logical_bytes']+=row['bytes']
 def integrate(batch,manifest,receipt):
  nonlocal done
  assert manifest['plan_sha256']==plan_hash and m.sha(m.OUT/'BATCHES'/batch['id']/'manifest.json')==receipt['manifest_sha256']
  assert receipt['status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
  for digest,blob in manifest['unique_blobs'].items():locations[digest]=dict(repo=m.REPO,revision=receipt['revision'],archive_path=receipt['archive_remote_path'],member=blob['member'],verification_status=receipt['status'])
  for row in manifest['files']:assert row['sha256'] in locations
  totals['backed_up_files']+=len(manifest['files']);totals['backed_up_logical_bytes']+=sum(r['bytes'] for r in manifest['files']);totals['remote_archive_bytes']+=manifest['archive_bytes'];totals['excluded_sensitive_or_changed']+=len(manifest['excluded_and_retained'])
  if batch['id'] not in already_complete:totals.update(m.local_delete(manifest,receipt,locations,plan['retained_final_checkpoint_prefixes']))
  m.save(m.OUT/'BLOB_RESTORE_INDEX.json',locations)
  archive=m.OUT/'BATCHES'/batch['id']/'blobs.tar.gz'
  if archive.exists():assert m.sha(archive)==receipt['archive_sha256'];archive.unlink()
  done+=1
  m.save(m.OUT/'STATUS.json',dict(status='RUNNING',updated_at_utc=m.now(),batch=batch['id'],completed_batches=done,total_batches=len(plan['batches']),totals=dict(totals),free_GiB=round(shutil.disk_usage(ROOT).free/2**30,3),workers=2,delete_authorized=True))
  event('BATCH_COMPLETE',batch=batch['id'],completed_batches=done,deleted_files=totals['deleted_files'],deleted_logical_GiB=round(totals['deleted_logical_bytes']/2**30,3),free_GiB=round(shutil.disk_usage(ROOT).free/2**30,3))
 pending=[]
 for batch in plan['batches']:
  folder=m.OUT/'BATCHES'/batch['id']
  if (folder/'REMOTE_RECEIPT.json').exists():
   receipt=m.read(folder/'REMOTE_RECEIPT.json');manifest=m.read(folder/'manifest.json')
   info=api.repo_info(m.REPO,repo_type='dataset',revision=receipt['revision'],files_metadata=True);assert info.private is True
   entry=next(f for f in info.siblings if f.rfilename==receipt['archive_remote_path']);assert entry.lfs.sha256==receipt['archive_sha256'] and entry.size==receipt['archive_bytes']
   integrate(batch,manifest,receipt)
  else:pending.append(batch)
 executor=ThreadPoolExecutor(max_workers=2);running={};iterator=iter(pending)
 def submit_next():
  batch=next(iterator,None)
  if batch is not None:running[executor.submit(work,batch,locations.copy(),plan_hash)]=batch['id']
 for _ in range(2):submit_next()
 try:
  while running:
   future=next(as_completed(running));running.pop(future);integrate(*future.result());submit_next()
 finally:executor.shutdown(wait=True,cancel_futures=True)
 result=dict(status='PASS_BACKUP_AND_VERIFIED_COLD_OFFLOAD_COMPLETE',completed_at_utc=m.now(),repo=m.REPO,namespace=m.NAMESPACE,plan_sha256=plan_hash,batches=len(plan['batches']),unique_content_blobs=len(locations),totals=dict(totals),free_before_bytes=free_start,free_after_bytes=shutil.disk_usage(ROOT).free,retained_final_checkpoint_prefixes=plan['retained_final_checkpoint_prefixes'],retained_live_apps=True,delete_authorized=True,workers=2)
 m.save(m.OUT/'FINAL_HF_OFFLOAD_RECEIPT.json',result);event('COMPLETE',**{k:v for k,v in result.items() if k!='retained_final_checkpoint_prefixes'})
if __name__=='__main__':
 try:main()
 except Exception as error:
  event('STOPPED',error_type=type(error).__name__,source_policy='Unverified/changed files retained;only already verified cold copies may have been removed')
  raise SystemExit(1)
