"""Fresh pixel/byte identity computation from actual cohort files, not cached StageD hashes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,csv,hashlib,io,os,stat,datetime,time,collections,platform
from PIL import Image,ImageOps,ImageFile
R=Path(__file__).resolve().parents[1];O=R/'outputs/data_integrity/cross_split_images_s1000_v1';E=O/'EVIDENCE';P=R/'data/processed/yelp_2300_v1'
ImageFile.LOAD_TRUNCATED_IMAGES=False
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def csvwrite(p,rows):
 with p.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pixel_info(data):
 with Image.open(io.BytesIO(data)) as im:
  im.load();original=[im.width,im.height];mode=im.mode;orientation=im.getexif().get(274,1)
  rgb=ImageOps.exif_transpose(im).convert('RGB');rgb.load();pixels=rgb.tobytes()
  digest=hashlib.sha256(b'EXIF_RGB_dimensions_v1:'+rgb.width.to_bytes(8,'big')+rgb.height.to_bytes(8,'big')+pixels).hexdigest()
  return dict(width=original[0],height=original[1],mode=mode,EXIF_orientation=orientation,normalized_width=rgb.width,normalized_height=rgb.height,pixel_sha256=digest),pixels
def inspect(row):
 p=P/row['relative_path'];before=p.stat();ls=p.lstat();resolved=p.resolve(strict=True)
 assert resolved.is_relative_to(P.resolve()) and p.stem==row['photo_id'] and before.st_size==int(row['bytes'])
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest();assert h==row['sha256']
 fields,_=pixel_info(data);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 raw=R/'data/yelp_photos/photos'/p.name
 return dict(photo_id=row['photo_id'],business_id=row['business_id'],split=row['split'],processed_path=str(p),resolved_processed_path=str(resolved),
  original_source_path_declared=str(raw),original_local_exists=raw.is_file(),bytes=len(data),file_sha256=h,**fields,
  st_dev=before.st_dev,st_ino=str(before.st_ino),st_nlink=before.st_nlink,is_symlink=p.is_symlink(),reparse_point=bool(getattr(ls,'st_file_attributes',0)&0x400),file_attributes=getattr(ls,'st_file_attributes',None))
def main(output_directory=None):
 global O,E
 if output_directory:
  O=Path(output_directory)
  if not O.is_absolute():O=R/O
  O=O.resolve();assert O.is_relative_to((R/'outputs').resolve()),'Audit evidence must stay in outputs/'
  E=O/'EVIDENCE'
 assert not (E/'FRESH_IMAGE_INVENTORY.csv').exists(),'Do not overwrite a completed fresh image pass'
 E.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 rows=list(csv.DictReader((P/'photo_manifest.csv').open(encoding='utf8')))
 splits=list(csv.DictReader((P/'business_split.csv').open(encoding='utf8')));businesses={r['business_id']:r['split'] for r in splits}
 assert len(rows)==30027 and len({r['photo_id'] for r in rows})==30027 and all(businesses[r['business_id']]==r['split'] for r in rows)
 original=[];fingerprints=collections.defaultdict(list);devino=collections.defaultdict(list)
 with ThreadPoolExecutor(max_workers=8) as pool:
  for n,r in enumerate(pool.map(inspect,rows),1):
   original.append(r);fingerprints[r['pixel_sha256']].append(r);devino[(r['st_dev'],r['st_ino'])].append(r['photo_id'])
   if n%3000==0:print(json.dumps(dict(event='FRESH_DECODE_PROGRESS',photos=n,total=len(rows),seconds=round(time.perf_counter()-start,1))),flush=True)
 pairs=[]
 for digest,g in sorted(fingerprints.items()):
  tests=sorted([r for r in g if r['split']=='test'],key=lambda r:r['photo_id']);other=sorted([r for r in g if r['split'] in ['train','validation']],key=lambda r:r['photo_id'])
  for t in tests:
   for v in other:
    assert t['business_id']!=v['business_id'] and t['photo_id']!=v['photo_id']
    ta=Path(t['processed_path']);va=Path(v['processed_path']);td=ta.read_bytes();vd=va.read_bytes();tp= pixel_info(td);vp=pixel_info(vd)
    assert tp[0]['pixel_sha256']==vp[0]['pixel_sha256']==digest and tp[1]==vp[1]
    samefile=os.path.samefile(ta,va)
    pairs.append(dict(pair_id=f'PAIR_{len(pairs)+1:04d}',test_business_id=t['business_id'],test_photo_id=t['photo_id'],other_business_id=v['business_id'],other_photo_id=v['photo_id'],
      source_split=v['split'],target_split='test',test_processed_path=t['processed_path'],other_processed_path=v['processed_path'],test_original_source_path=t['original_source_path_declared'],other_original_source_path=v['original_source_path_declared'],
      test_file_sha256=t['file_sha256'],other_file_sha256=v['file_sha256'],test_pixel_sha256=digest,other_pixel_sha256=digest,
      test_width=t['width'],test_height=t['height'],other_width=v['width'],other_height=v['height'],normalized_width=t['normalized_width'],normalized_height=t['normalized_height'],
      byte_identical=td==vd,pixel_identical=True,normalized_pixels_direct_equal=True,only_visually_similar=False,identity_class='BYTE_AND_PIXEL_IDENTICAL' if td==vd else 'PIXEL_IDENTICAL_DIFFERENT_BYTES',
      resolved_paths_equal=ta.resolve()==va.resolve(),physical_samefile=samefile,test_st_dev=t['st_dev'],other_st_dev=v['st_dev'],test_file_id=t['st_ino'],other_file_id=v['st_ino'],test_nlink=t['st_nlink'],other_nlink=v['st_nlink'],test_reparse=t['reparse_point'],other_reparse=v['reparse_point']))
 csvwrite(E/'FRESH_IMAGE_INVENTORY.csv',original);csvwrite(E/'FRESH_CROSS_SPLIT_PAIRS.csv',pairs)
 ancestors=[]
 for p in [R,*R.parents,R/'data',R/'data/processed',P,P/'photos',R/'data/yelp_photos']:
  s=p.lstat();ancestors.append(dict(path=str(p),resolved=str(p.resolve()),is_symlink=p.is_symlink(),reparse_point=bool(getattr(s,'st_file_attributes',0)&0x400),st_dev=s.st_dev,st_ino=str(s.st_ino),file_attributes=getattr(s,'st_file_attributes',None)))
 save(E/'FILESYSTEM_OBSERVATIONS.json',dict(ancestors=ancestors,processed_same_physical_file_groups=[ids for ids in devino.values() if len(ids)>1],files_with_multiple_hardlinks=[r['photo_id'] for r in original if r['st_nlink']>1],reparse_files=[r['photo_id'] for r in original if r['reparse_point']],THESIS_DATA_ROOT_env_path=os.environ.get('THESIS_DATA_ROOT'),scope='Current local filesystem observations, not proof of historical source mounting'))
 # Only after fresh computation, compare with the old diagnostic without using its hashes to select work.
 old=list(csv.DictReader((R/'outputs/stage_d/photo_integrity_duplicate_audit_v1/test_image_disjoint_diagnostic_businesses.csv').open(encoding='utf8')))
 oldset={r['business_id'] for r in old if r['image_disjoint_from_train_validation']=='False'};freshset={r['test_business_id'] for r in pairs}
 save(E/'FRESH_RECOMPUTATION_RECEIPT.json',dict(status='COMPLETE_INDEPENDENT_CURRENT_FILE_RECOMPUTATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),photos_fresh_decoded=len(original),split_photo_counts=dict(collections.Counter(r['split'] for r in original)),
  total_cross_split_pairs=len(pairs),affected_TEST_businesses=len(freshset),unique_TEST_duplicate_photo_ids=len({r['test_photo_id'] for r in pairs}),unique_other_duplicate_photo_ids=len({r['other_photo_id'] for r in pairs}),
  pairs_by_other_split=dict(collections.Counter(r['source_split'] for r in pairs)),byte_identical_pairs=sum(r['byte_identical'] for r in pairs),pixel_only_pairs=sum(not r['byte_identical'] for r in pairs),
  prior_businesses=len(oldset),prior_affected_businesses_reproduced=freshset==oldset,fresh_additions=sorted(freshset-oldset),fresh_absences=sorted(oldset-freshset),physical_samefile_pairs=sum(r['physical_samefile'] for r in pairs),
  normalization='Full decode, EXIF transpose, convert RGB, no resize; SHA256(bEXIF_RGB_dimensions_v1:+8-byte-big-endian width+8-byte height+RGBbytes)',
  Python=platform.python_version(),Pillow=Image.__version__,seconds=time.perf_counter()-start,
  canonical_photo_manifest_sha256=sha(P/'photo_manifest.csv'),business_split_sha256=sha(P/'business_split.csv'),cached_pixel_results_used_to_construct_matches=False,
  near_duplicate_search='Not performed: exact byte/normalized-pixel investigation; no claim that perceptually similar pairs are absent'))
 print(json.dumps(read_receipt:=json.loads((E/'FRESH_RECOMPUTATION_RECEIPT.json').read_text()),indent=2),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output-directory');args=parser.parse_args();main(args.output_directory)
