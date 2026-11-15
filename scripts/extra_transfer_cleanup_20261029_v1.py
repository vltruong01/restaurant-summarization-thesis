"""CMD entry: delete only fixed old transfer files after full live keeper verification.

No recursive removal, no wildcard, no upload, no model/run/source-tree deletion.
"""
from pathlib import Path,PurePosixPath
import argparse,json,hashlib,os,datetime
from audit_extra_transfer_archives_20261029_v1 import TARGETS
R=Path(__file__).resolve().parents[1]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class LiveVerifier:
 def __init__(self,root):self.root=root.resolve();self.parents={};self.hashes={}
 def path(self,rel):
  need(not Path(rel).is_absolute() and '..' not in PurePosixPath(rel).parts,'Invalid relative path')
  p=self.root/rel
  if p.parent not in self.parents:
   resolved=p.parent.resolve();need(resolved.is_relative_to(self.root),'Path escapes workspace')
   for q in [p.parent,*p.parent.parents]:
    if q==self.root.parent:break
    if q.exists():need(not q.is_symlink() and not getattr(q.lstat(),'st_file_attributes',0)&0x400,'Reparse ancestor')
   self.parents[p.parent]=True
  if p.exists():need(not p.is_symlink() and not getattr(p.lstat(),'st_file_attributes',0)&0x400,'Reparse file')
  return p
 def file(self,rel,size,digest):
  p=self.path(rel);need(p.is_file() and p.stat().st_size==size,'Missing/changed file '+rel)
  before=p.stat();key=(str(p),before.st_size,before.st_mtime_ns)
  if key not in self.hashes:
   self.hashes[key]=sha(p);after=p.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed during hash')
  need(self.hashes[key]==digest,'File SHA256 mismatch '+rel);return p
def check_archive(row,verifier):
 need(row['path'] in TARGETS and row['eligible'] and not row['errors'],'Unapproved or incomplete archive proof')
 if not (verifier.root/row['path']).exists():return None
 target=verifier.file(row['path'],row['bytes'],row['sha256'])
 if row.get('verification_kind')=='EXACT_DUPLICATE_CONTAINER':
  need(row['keep_archive'] not in TARGETS,'Keeper archive is also proposed for deletion')
  verifier.file(row['keep_archive'],row['bytes'],row['sha256']);return target
 need(len(row['members'])==row['files']>0,'Incomplete member ledger')
 for m in row['members']:
  need(m['keep'] not in TARGETS,'Keeper is also a proposed deletion')
  verifier.file(m['keep'],m['bytes'],m['sha256'])
 return target
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--delete',action='store_true');a=p.parse_args()
 need(sha(a.plan)==a.plan_sha,'Plan changed');plan=json.loads(a.plan.read_text(encoding='utf8'));need(Path(plan['root']).resolve()==R,'Wrong workspace')
 verifier=LiveVerifier(R);approved=[]
 rows=[]
 for summary in plan['archives']:
  if summary.get('proof_file'):
   proof=a.plan.parent/summary['proof_file'];need(proof.resolve().is_relative_to(a.plan.parent.resolve()),'Proof path escape')
   need(sha(proof)==summary['proof_sha256'],'Proof ledger changed');row=json.loads(proof.read_text(encoding='utf8'))
   need(row['path']==summary['path'] and row['sha256']==summary['sha256'],'Proof/plan binding mismatch')
  else:row=summary
  rows.append(row)
 for row in rows:
  target=check_archive(row,verifier)
  if target:approved.append(row);print('Live content/keeper SHA PASS: '+row['path'],flush=True)
 print('Verified '+str(len(approved))+' archives; mode '+('DELETE' if a.delete else 'READ_ONLY_CHECK'),flush=True)
 if not a.delete:return
 log=a.plan.parent/('EXECUTED_EXTRA_ARCHIVE_DELETE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
 for row in approved:
  target=check_archive(row,verifier)
  if target:
   # Exact regular file inside verified workspace; no recursive operation or shell interpolation.
   need(target.resolve().is_relative_to(R) and target.is_file(),'Unsafe final target')
   need(sha(target)==row['sha256'],'Archive changed before deletion');target.unlink()
   with log.open('a',encoding='utf8') as f:f.write(json.dumps(dict(action='DELETED_REDUNDANT_OLD_TRANSFER_ARCHIVE',path=row['path'],original_archive_sha256=row['sha256'],member_files_individually_verified=row['files'],whole_container_copy_verified=row.get('verification_kind')=='EXACT_DUPLICATE_CONTAINER',verification_plan_sha256=a.plan_sha,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
if __name__=='__main__':main()
