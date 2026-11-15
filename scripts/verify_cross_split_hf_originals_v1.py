"""Read only pinned private HF backup. Fetch matched original JPEG members by HTTP Range.
Stream one existing gzip shard to recover original photos.json; never materialize full datasets.
No upload, auth mutation, model, filtering or inference.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,csv,hashlib,tarfile,io,threading,datetime,time
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
import requests
from audit_cross_split_images_fresh_v1 import pixel_info
R=Path(__file__).resolve().parents[1];O=R/'outputs/data_integrity/cross_split_images_s1000_v1';E=O/'EVIDENCE';RAW=E/'HF_ORIGINAL_JPEGS'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def connection(url,headers):
 try:r=requests.get(url,headers=headers,stream=True,timeout=(30,60))
 except Exception as e:raise RuntimeError('HF read transport failed: '+type(e).__name__) from None
 return r
def range_bytes(archive,start,size):
 url=hf_hub_url(repo,filename=archive,repo_type='dataset',revision=revision)
 headers=build_hf_headers();headers['Range']=f'bytes={start}-{start+size-1}'
 # Query separates range-specific CDN cache keys; response range is always validated.
 url+='?audit_range_start='+str(start)
 response=connection(url,headers)
 try:
  assert response.status_code==206, 'Range request not honored; full archive download refused, HTTP '+str(response.status_code)
  expected=f'bytes {start}-{start+size-1}/'
  assert response.headers.get('Content-Range','').startswith(expected),'Unexpected HTTP Content-Range'
  data=response.raw.read(size+1);assert len(data)==size,'Unexpected ranged response size'
  return data
 finally:response.close()
meta=json.loads((E/'HF_PINNED_REPOSITORY_METADATA.json').read_text());repo=meta['repo_id'];revision=meta['revision'];assert meta['private']
mp=R/'outputs/hf_dataset_backup_20261008/upload/backup_manifest.json';m=json.loads(mp.read_text(encoding='utf8'))
info=next(x for x in meta['files'] if x['path']=='backup_manifest.json');assert sha(mp)==info['lfs_sha256'] and mp.stat().st_size==info['size']
assert m['repository']==repo and m['format']=='private-yelp-data-backup-v1'
files=m['files'];rawroot=[f for f in files if len(Path(f['path']).parts)==2];photos=[f for f in files if f['path'].startswith('data/yelp_photos/photos/')]
rest=[f for f in files if f not in rawroot and not f['path'].startswith('data/yelp_photos/photos/')]
groups=[[r] for r in rawroot]+[photos[i:i+10000] for i in range(0,len(photos),10000)]
g=[];size=0
for f in rest:
 if g and size+f['bytes']>1024**3:groups.append(g);g=[];size=0
 g.append(f);size+=f['bytes']
if g:groups.append(g)
assert len(groups)==len(m['archives'])
for a in m['archives']:
 pin=next(f for f in meta['files'] if f['path']==a['name']);assert pin['lfs_sha256']==a['sha256'] and pin['size']==a['bytes']
pairs=list(csv.DictReader((E/'FRESH_CROSS_SPLIT_PAIRS.csv').open(encoding='utf8')));wanted={p[k] for p in pairs for k in ['test_photo_id','other_photo_id']}
index={};metadata=None
for group,archive in zip(groups,m['archives']):
 offset=0
 for f in group:
  if f['path'].startswith('data/yelp_photos/photos/') and Path(f['path']).stem in wanted:
   assert archive['name'].endswith('.tar') and len(f['path'].encode())<=100
   index[Path(f['path']).stem]=dict(archive=archive['name'],header_offset=offset,**f)
  if f['path']=='data/yelp_photos/photos.json':metadata=dict(archive=archive,member=f)
  offset+=512+((f['bytes']+511)//512)*512
assert set(index)==wanted and metadata
RAW.mkdir(parents=True,exist_ok=True)
def fetch(pid):
 spec=index[pid];p=RAW/(pid+'.jpg');header=None;origin='FRESH_HTTP_RANGE'
 if p.exists():
  assert sha(p)==spec['sha256'];data=p.read_bytes();origin='PREVIOUSLY_VERIFIED_SAME_PINNED_AUDIT_MEMBER'
 else:
  block=range_bytes(spec['archive'],spec['header_offset'],512+spec['bytes'])
  ti=tarfile.TarInfo.frombuf(block[:512],encoding='utf8',errors='strict')
  assert ti.name==spec['path'] and ti.size==spec['bytes'] and ti.isfile()
  data=block[512:];assert hashlib.sha256(data).hexdigest()==spec['sha256']
  with p.open('xb') as f:f.write(data)
  header=dict(name=ti.name,size=ti.size,mtime=ti.mtime,checksum=ti.chksum)
 pix,_=pixel_info(data)
 local=R/'data/processed/yelp_2300_v1/photos'/p.name;localdata=local.read_bytes();localpix,_=pixel_info(localdata)
 assert data==localdata and pix['pixel_sha256']==localpix['pixel_sha256']
 return dict(photo_id=pid,repo_id=repo,revision=revision,archive=spec['archive'],header_offset=spec['header_offset'],source_member=spec['path'],
  archive_declared_sha256=next(a['sha256'] for a in m['archives'] if a['name']==spec['archive']),member_sha256=spec['sha256'],bytes=len(data),fresh_header=header,fetch_origin=origin,
  normalized_pixel_sha256=pix['pixel_sha256'],normalized_width=pix['normalized_width'],normalized_height=pix['normalized_height'],
  processed_bytes_equal_original=True,processed_normalized_pixels_equal_original=True,local_evidence_path=str(p),
  original_archive_full_bytes_rehashed_in_this_audit=False)
start=time.perf_counter();result=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for n,x in enumerate(pool.map(fetch,sorted(wanted)),1):
  result.append(x)
  if n%50==0:print(json.dumps(dict(event='HF_ORIGINAL_JPEGS_VERIFIED',members=n,total=len(wanted),seconds=round(time.perf_counter()-start,1))),flush=True)
save(E/'HF_ORIGINAL_JPEG_VERIFICATION.json',dict(status='PASS_ALL_MATCHED_ORIGINAL_JPEG_MEMBERS',repo_id=repo,revision=revision,backup_manifest_sha256=sha(mp),raw_original_JPEG_members=len(result),records=result,network_fetch_scope='Only matched member header+bytes; no full uncompressed JPEG shards downloaded',
 original_source_snapshot_created_UTC=m['created_at_utc'],private_repository_unchanged=True))

class DigestStream:
 def __init__(self,raw):self.raw=raw;self.h=hashlib.sha256();self.bytes=0;self.report=0
 def read(self,n=-1):
  assert n>=0,'Unbounded remote reads not allowed'
  b=self.raw.read(n);self.h.update(b);self.bytes+=len(b)
  if self.bytes-self.report>=64*1024**2:
   self.report=self.bytes;print(json.dumps(dict(event='ORIGINAL_METADATA_SHARD_STREAM',bytes=self.bytes,total=metadata['archive']['bytes'])),flush=True)
  return b
target=E/'HF_ORIGINAL_METADATA/photos.json';target.parent.mkdir(exist_ok=True)
archive=metadata['archive'];member=metadata['member']
if not target.exists():
 url=hf_hub_url(repo,filename=archive['name'],repo_type='dataset',revision=revision)
 response=connection(url,build_hf_headers())
 try:
  assert response.status_code==200,'Metadata stream HTTP failure '+str(response.status_code)
  stream=DigestStream(response.raw);found=False
  with tarfile.open(fileobj=stream,mode='r|gz') as tar:
   for ti in tar:
    if ti.name==member['path']:
     assert not found and ti.isfile() and ti.size==member['bytes'];found=True
     inp=tar.extractfile(ti)
     with target.open('xb') as out:
      for b in iter(lambda:inp.read(1024*1024),b''):out.write(b)
  while stream.read(1024*1024):pass
  assert found and stream.bytes==archive['bytes'] and stream.h.hexdigest()==archive['sha256']
  assert target.stat().st_size==member['bytes'] and sha(target)==member['sha256']
  save(E/'HF_ORIGINAL_METADATA_SHARD_RECEIPT.json',dict(status='PASS_FRESH_REMOTE_ARCHIVE_SHA_AND_ORIGINAL_PHOTO_JSON',repo_id=repo,revision=revision,archive=archive['name'],bytes_streamed=stream.bytes,archive_sha256=stream.h.hexdigest(),member=member['path'],member_sha256=sha(target),full_archive_saved=False,source_metadata_saved=True))
 finally:response.close()
else:
 assert target.stat().st_size==member['bytes'] and sha(target)==member['sha256']
 assert json.loads((E/'HF_ORIGINAL_METADATA_SHARD_RECEIPT.json').read_text())['status']=='PASS_FRESH_REMOTE_ARCHIVE_SHA_AND_ORIGINAL_PHOTO_JSON'
original_records=collections_placeholder if False else {}
with target.open(encoding='utf8') as f:
 for line in f:
  r=json.loads(line)
  if r['photo_id'] in wanted:
   assert r['photo_id'] not in original_records;original_records[r['photo_id']]=dict(photo_id=r['photo_id'],business_id=r['business_id'],label=r.get('label'))
assert set(original_records)==wanted
for p in pairs:
 assert original_records[p['test_photo_id']]['business_id']==p['test_business_id']
 assert original_records[p['other_photo_id']]['business_id']==p['other_business_id']
save(E/'ORIGINAL_YELP_MATCHED_PHOTO_METADATA.json',dict(status='PASS_SOURCE_ASSIGNMENTS_MATCH_CURRENT_COHORT',repo_id=repo,revision=revision,original_photos_json_sha256=member['sha256'],records=list(original_records.values()),
 captions_or_references_emitted=False,source_authenticity_scope='Verified original-source paths/bytes in the preserved pre-training HF snapshot; not a new independent download from Yelp servers'))
print(json.dumps(dict(status='PASS_ORIGINAL_SNAPSHOT_JPEGS_AND_METADATA',members=len(result),matched_pairs=len(pairs),source_assignment_records=len(original_records),seconds=time.perf_counter()-start)),flush=True)
