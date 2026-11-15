"""Read-only archive-content redundancy audit after the first cleanup. No removal."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,tarfile,collections,os,datetime,ctypes
R=Path(__file__).resolve().parents[1];O=R/'outputs/storage_cleanup/20261029_extra_v1'
TARGETS=[
 'outputs/cloud_handoff/LED_LARGE_VAST_FRESH_V1.zip',
 'outputs/cloud_handoff/LED_LARGE_VAST_DYNAMIC_65536_V2.zip',
 'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1.zip',
 'outputs/cloud_handoff/LED_LARGE_RO_U_CE3_PARENT_TRANSPORT_V1.tar.gz',
 'outputs/stage_i/led_large_cad_v1_vast_execution_v1/CAD_V1_FRESH_HOST_PAYLOAD.tar',
 'outputs/cloud_handoff/LED_LARGE_RO_U_DATA_AND_GATE_CODE_TRANSPORT_V1.tar.gz',
 'outputs/cloud_handoff/LED_LARGE_V3_ALL_IN_ONE.zip',
 'outputs/cloud_handoff/LED-Large_multimodal_UPLOAD_V1.zip',
 'outputs/cloud_handoff/LED_LARGE_F_COLAB_V1.zip',
]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'EXTRA_ARCHIVE_AUDIT.json').exists(),'Audit already complete'
 old=R/'outputs/storage_cleanup/20261029_review_v1';scan=json.loads((old/'CURRENT_STORAGE_INVENTORY.json').read_text(encoding='utf8'));plan=json.loads((old/'CLEANUP_PLAN.json').read_text(encoding='utf8'))
 aliases={d['path']:d['keep'] for d in plan['duplicate_deletions']};index=collections.defaultdict(list)
 for row in scan['files']:index[(PurePosixPath(row['path']).name,row['bytes'])].append(row['path'])
 for row in index.values():row.sort(key=lambda p:(0 if p.startswith(('data/','outputs/stage_f/','outputs/stage_g/','scripts/','configs/')) else 1,len(p),p))
 cache={};identity={}
 def local_digest(p):
  p=Path(p);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
  if key not in cache:
   phys=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
   if s.st_ino and phys in identity:h=identity[phys]
   else:
    h=sha(p);after=p.stat();assert (after.st_size,after.st_mtime_ns)==(s.st_size,s.st_mtime_ns),'Keeper changed while hashing'
    if s.st_ino:identity[phys]=h
   cache[key]=h
  return cache[key]
 def candidate_paths(name,size,archive_rel):
  rels=[];name=name.removeprefix('./');parts=PurePosixPath(name).parts
  if not parts or '..' in parts or PurePosixPath(name).is_absolute():return []
  rels+=[name,'outputs/cloud_handoff/'+name]
  if archive_rel.endswith('.zip') and len(parts)>1:
   stem=PurePosixPath(archive_rel).name.removesuffix('.zip')
   if parts[0]==stem:rels.append('outputs/cloud_handoff/'+name)
   rels+=['outputs/cloud_handoff/led_large_full_train_colab_verification_v2/extracted/'+ '/'.join(parts[1:]),'outputs/cloud_handoff/led_large_f_colab_v1/'+ '/'.join(parts[1:])]
  for p in list(rels):
   if p in aliases:rels.insert(0,aliases[p])
  if name.endswith('assets/models/led-large-16384/pytorch_model.bin'):rels.insert(0,'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin')
  rels+=index.get((PurePosixPath(name).name,size),[])
  seen=set()
  for rel in rels:
   if rel in seen or rel in TARGETS:continue
   seen.add(rel);p=R/rel
   if Path(rel).is_absolute() or '..' in PurePosixPath(rel).parts:continue
   if not p.is_file() or p.is_symlink():continue
   if p.stat().st_size==size and p.resolve().is_relative_to(R):yield rel
 results=[]
 for rel in TARGETS:
  path=R/rel
  if not path.exists():continue
  print('Auditing members:',rel,flush=True)
  s=path.stat();rows=[];errors=[];seen=set();dirs=[]
  def member(name,size,stream,metadata):
   if name in seen:errors.append('duplicate member '+name)
   seen.add(name);h=hashlib.sha256();n=0
   for b in iter(lambda:stream.read(4*2**20),b''):h.update(b);n+=len(b)
   digest=h.hexdigest();keep=None
   for candidate in candidate_paths(name,size,rel):
    if local_digest(R/candidate)==digest:keep=candidate;break
   if n!=size:errors.append('member length mismatch '+name)
   if keep is None:errors.append('no independently retained identical file '+name)
   rows.append(dict(name=name,bytes=size,sha256=digest,keep=keep,metadata=metadata))
   if len(rows)%2000==0:print(' members checked',len(rows),flush=True)
  try:
   if rel.endswith('.zip'):
    with zipfile.ZipFile(path) as arc:
     for info in arc.infolist():
      if info.is_dir():dirs.append(info.filename);continue
      with arc.open(info) as f:member(info.filename,info.file_size,f,dict(date_time=info.date_time,CRC32=info.CRC,compression=info.compress_type,external_attr=info.external_attr))
   else:
    with tarfile.open(path,'r:*') as arc:
     for info in arc:
      if info.isdir():dirs.append(info.name);continue
      if not info.isfile():errors.append('nonregular tar entry '+info.name);continue
      with arc.extractfile(info) as f:member(info.name,info.size,f,dict(mtime=info.mtime,mode=info.mode,pax_headers=info.pax_headers))
  except Exception as e:errors.append(type(e).__name__+': '+str(e)[:250])
  after=path.stat();assert (s.st_size,s.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive changed'
  result=dict(path=rel,bytes=s.st_size,nlink=s.st_nlink,sha256=sha(path),mtime_ns=s.st_mtime_ns,files=len(rows),directories=dirs,members=rows,errors=errors,
   eligible=bool(rows) and not errors,reason='Every member file has a fresh SHA256-identical independent retained local file; archive is old transport packaging, not final result evidence',
   estimated_reclaimable_bytes=s.st_size if s.st_nlink==1 else 0)
  results.append(result)
  (O/(PurePosixPath(rel).name+'.verification.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
  print('PASS' if result['eligible'] else 'KEEP/BLOCKED','members',len(rows),'unmatched/errors',len(errors),'GiB',round(s.st_size/2**30,3),flush=True)
 summary=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(R),read_only=True,original_files_deleted=0,HF_uploads=0,archives=results,
  eligible_archives=sum(r['eligible'] for r in results),estimated_additional_reclaimable_bytes=sum(r['estimated_reclaimable_bytes'] for r in results if r['eligible']),
  note='Content redundancy verified, not necessarily bit-identical archive-container reconstruction. All archive/member hashes, entry metadata and retained paths preserved.')
 (O/'EXTRA_ARCHIVE_AUDIT.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps({k:summary[k] for k in ['eligible_archives','estimated_additional_reclaimable_bytes','original_files_deleted','HF_uploads']},indent=2))
if __name__=='__main__':main()
