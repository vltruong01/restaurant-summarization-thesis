"""Read-only HF original business metadata for matched IDs; no review/reference/output access."""
from pathlib import Path
import json,csv,hashlib,tarfile,requests,datetime
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
R=Path(__file__).resolve().parents[1];E=R/'outputs/data_integrity/cross_split_images_s1000_v1/EVIDENCE'
def save(n,x):(E/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
meta=json.loads((E/'HF_PINNED_REPOSITORY_METADATA.json').read_text());m=json.loads((R/'outputs/hf_dataset_backup_20261008/upload/backup_manifest.json').read_text())
raw=[r for r in m['files'] if len(Path(r['path']).parts)==2];i=next(i for i,r in enumerate(raw) if r['path']=='data/yelp_academic_dataset_business.json');member=raw[i];archive=m['archives'][i]
pairs=list(csv.DictReader((E/'FRESH_CROSS_SPLIT_PAIRS.csv').open(encoding='utf8')));ids={p[k] for p in pairs for k in ['test_business_id','other_business_id']}
url=hf_hub_url(meta['repo_id'],filename=archive['name'],repo_type='dataset',revision=meta['revision'])
try:response=requests.get(url,headers=build_hf_headers(),stream=True,timeout=(30,60))
except Exception as e:raise RuntimeError('HF original business read failed: '+type(e).__name__) from None
class Stream:
 def __init__(self,raw):self.raw=raw;self.h=hashlib.sha256();self.size=0
 def read(self,n):
  assert n>=0;b=self.raw.read(n);self.h.update(b);self.size+=len(b);return b
found={};mhash=hashlib.sha256();msize=0
try:
 assert response.status_code==200
 stream=Stream(response.raw)
 with tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for ti in tar:
   assert ti.name==member['path'] and ti.isfile() and ti.size==member['bytes']
   f=tar.extractfile(ti)
   for line in f:
    mhash.update(line);msize+=len(line);r=json.loads(line)
    if r['business_id'] in ids:
     assert r['business_id'] not in found
     found[r['business_id']]={k:r.get(k) for k in ['business_id','name','address','city','state','postal_code','latitude','longitude','categories']}
 while stream.read(1024*1024):pass
 assert stream.size==archive['bytes'] and stream.h.hexdigest()==archive['sha256'] and msize==member['bytes'] and mhash.hexdigest()==member['sha256'] and set(found)==ids
 current={r['business_id']:r for r in map(json.loads,(R/'data/processed/yelp_2300_v1/businesses.jsonl').read_text().splitlines())}
 assert all(all(current[bid].get(k)==v for k,v in r.items()) for bid,r in found.items())
 save('ORIGINAL_BUSINESS_METADATA_VERIFICATION.json',dict(status='PASS_HF_ORIGINAL_BUSINESS_METADATA_AND_CURRENT_ASSIGNMENTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo_id=meta['repo_id'],revision=meta['revision'],archive=archive['name'],archive_bytes_streamed=stream.size,archive_sha256=stream.h.hexdigest(),original_business_member_sha256=mhash.hexdigest(),matched_business_records=len(found),records=list(found.values()),all_current_identity_fields_equal_original=True,full_raw_business_file_saved=False))
 print(json.dumps(dict(status='PASS_ORIGINAL_BUSINESS_METADATA',businesses=len(found),bytes_streamed=stream.size)),flush=True)
finally:response.close()
