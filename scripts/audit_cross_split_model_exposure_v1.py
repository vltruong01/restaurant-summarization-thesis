"""Trace fresh image pairs to frozen candidate, input and supervised TRAIN/VALIDATION evidence."""
from pathlib import Path
import csv,json,hashlib,collections,sys,math,unicodedata
R=Path(__file__).resolve().parents[1];O=R/'outputs/data_integrity/cross_split_images_s1000_v1';E=O/'EVIDENCE'
S=R/'outputs/stage_f/production_caption_construction_d033_colab_v1';M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1';N=M/'RECOVERY_R1';F=R/'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1'
sys.path.insert(0,str(M));from source_separated_contract import isolated_sources
def rows(p):
 with p.open(encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def writecsv(p,rs):
 with p.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
pairs=list(csv.DictReader((E/'FRESH_CROSS_SPLIT_PAIRS.csv').open(encoding='utf8')));testids={p['test_photo_id'] for p in pairs};otherids={p['other_photo_id'] for p in pairs};otherbids={p['other_business_id'] for p in pairs}
pool={r['photo_id']:r for r in rows(S/'manifests/complete_caption_pool_inventory.jsonl')};assert testids|otherids<=set(pool)
selected={};caption={};inputbinding={}
for split in ['train','validation','test']:
 for c in ['unfiltered','filtered']:
  p=S/f'manifests/{split}/{c}_captions.jsonl';rr=list(rows(p));selected[split,c]={r['business_id']:r['selected_photo_ids'] for r in rr}
  for r in rr:
   for ca in r['captions']:caption[split,c,ca['photo_id']]=ca['normalized_caption']
  inputbinding[p.relative_to(R).as_posix()]=sha(p)
source_data={};supported=collections.defaultdict(set);claimproof=[]
T=R/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset';bindings=read(M/'FROZEN_DATA_BINDINGS.json')['dataset_files']
for split in ['train','validation']:
 for c in ['unfiltered','filtered']:
  p=T/f'{split}_{c}.jsonl';assert sha(p)==bindings[str(p)];inputbinding[p.relative_to(R).as_posix()]=sha(p)
  for r in rows(p):
   if r['business_id'] not in otherbids:continue
   assert r['selected_photo_ids']==selected[split,c][r['business_id']]
   source_data[split,c,r['business_id']]=r
   for cl in r['visual_claims_with_evidence']:
    for support in cl['support']:
     pid=support['photo_id']
     if pid in otherids:
      assert pid in r['selected_photo_ids'] and support['caption_quote'].casefold() in caption[split,c,pid].casefold()
      supported[split,c].add(pid);claimproof.append(dict(split=split,condition=c,business_id=r['business_id'],photo_id=pid,claim_text=cl['text'],support=support,target_artifact=r['target_artifact'],target_artifact_sha256=r['target_artifact_sha256']))
# Actual retained new-phase Visual windows, without assuming selected datasets were executed.
actual=collections.Counter();sourcehash={}
for c,run in [('unfiltered',N/'DOWNLOADED/U/ARMS/U/RUN'),('filtered',F/'downloaded/RUN')]:
 for i in range(1,1001):
  w=read(run/f'window_{i:03d}.json');assert w['step']==i
  for p in w['task_presentations']:
   if p['task']=='visual' and p['business_id'] in otherbids:
    row=source_data.get(('train',c,p['business_id']))
    if row is None:continue
    vs=isolated_sources(row['source'])[1];assert hashlib.sha256(vs.encode()).hexdigest()==p['source_sha256']
    actual[c,p['business_id']]+=1;sourcehash[c,p['business_id']]=p['source_sha256']
 for bid in {r['other_business_id'] for r in pairs if r['source_split']=='train'}:assert actual[c,bid]==2
 for split in ['train','validation']:
  for (sp,co,bid),row in source_data.items():
   if sp==split and co==c and split=='train':assert actual[c,bid]==2
ledger=[];allbiz={r['business_id']:r for r in rows(R/'data/processed/yelp_2300_v1/businesses.jsonl')}
for pid in sorted(testids):
 pp=[p for p in pairs if p['test_photo_id']==pid];bid=pp[0]['test_business_id']
 r=dict(test_business_id=bid,test_photo_id=pid,basic_U_candidate=bool(pool[pid]['basic_eligible']),raw_pool_present=True,
        selected_TEST_U=pid in selected['test','unfiltered'][bid],selected_TEST_F=pid in selected['test','filtered'][bid],raw_pool_only_not_basic_candidate=not pool[pid]['basic_eligible'],
        raw_or_candidate_only_not_selected_U_or_F=pid not in selected['test','unfiltered'][bid] and pid not in selected['test','filtered'][bid],matching_pair_count=len(pp),matching_TRAIN_photo_ids='|'.join(sorted({p['other_photo_id'] for p in pp if p['source_split']=='train'})),matching_VALIDATION_photo_ids='|'.join(sorted({p['other_photo_id'] for p in pp if p['source_split']=='validation'})),RO_image_or_caption_input=False)
 for short,c in [('U','unfiltered'),('F','filtered')]:
  for split in ['train','validation']:
   q=[p for p in pp if p['source_split']==split and p['other_photo_id'] in selected[split,c][p['other_business_id']]]
   r[f'{short}_{split}_counterpart_selected_input']=bool(q)
   r[f'{short}_{split}_counterpart_used_in_teacher_source']=bool(q)
   r[f'{short}_{split}_counterpart_supports_retained_visual_target']=any(p['other_photo_id'] in supported[split,c] for p in q)
   r[f'{short}_{split}_selected_counterpart_photo_ids']='|'.join(sorted({p['other_photo_id'] for p in q}))
   r[f'{short}_{split}_same_caption_text_as_TEST']=any(caption[split,c,p['other_photo_id']]==caption.get(('test',c,pid)) for p in q) if r[f'selected_TEST_{short}'] else False
  r[f'{short}_own_condition_TEST_input_and_TRAIN_input_overlap']=r[f'selected_TEST_{short}'] and r[f'{short}_train_counterpart_selected_input']
  r[f'{short}_own_condition_TEST_input_and_TRAIN_target_support_overlap']=r[f'selected_TEST_{short}'] and r[f'{short}_train_counterpart_supports_retained_visual_target']
  r[f'{short}_own_condition_TEST_input_and_VALIDATION_only_overlap']=r[f'selected_TEST_{short}'] and not r[f'{short}_train_counterpart_selected_input'] and r[f'{short}_validation_counterpart_selected_input']
 ledger.append(r)
writecsv(E/'PER_TEST_DUPLICATE_PHOTO_EXPOSURE.csv',ledger)
businessrows=[];splits=list(csv.DictReader((R/'data/processed/yelp_2300_v1/business_split.csv').open(encoding='utf8')));testbiz=sorted(r['business_id'] for r in splits if r['split']=='test')
for bid in testbiz:
 rr=[r for r in ledger if r['test_business_id']==bid];row=dict(test_business_id=bid,business_name=allbiz[bid]['name'],duplicate_photo_ids=len(rr),duplicate_pair_count=sum(r['matching_pair_count'] for r in rr),U_total_selected_captions=len(selected['test','unfiltered'][bid]),F_total_selected_captions=len(selected['test','filtered'][bid]),
    duplicate_U_candidate_count=sum(r['basic_U_candidate'] for r in rr),duplicate_U_selected_count=sum(r['selected_TEST_U'] for r in rr),duplicate_F_selected_count=sum(r['selected_TEST_F'] for r in rr),
    duplicated_photos_not_selected_U_or_F=sum(r['raw_or_candidate_only_not_selected_U_or_F'] for r in rr))
 for a in ['U','F']:
  for tag in ['own_condition_TEST_input_and_TRAIN_input_overlap','own_condition_TEST_input_and_TRAIN_target_support_overlap','own_condition_TEST_input_and_VALIDATION_only_overlap']:
   row[f'{a}_{tag}_photos']=sum(r[f'{a}_{tag}'] for r in rr)
 businessrows.append(row)
writecsv(O/'TEST100_DUPLICATE_IMAGE_EXPOSURE.csv',businessrows)
pairflags=[]
for p in pairs:
 r=dict(p);tbid=p['test_business_id'];tpid=p['test_photo_id'];sp=p['source_split'];bid=p['other_business_id'];pid=p['other_photo_id']
 for a,c in [('U','unfiltered'),('F','filtered')]:
  r[f'TEST_{a}_candidate']=bool(pool[tpid]['basic_eligible'])
  r[f'TEST_{a}_selected']=tpid in selected['test',c][tbid]
  r[f'other_{a}_selected']=pid in selected[sp,c][bid]
  r[f'other_{a}_Teacher_source']=r[f'other_{a}_selected']
  r[f'other_{a}_retained_label_support']=pid in supported[sp,c]
  r[f'other_{a}_new_phase_Visual_presentations']=actual[c,bid] if sp=='train' and r[f'other_{a}_selected'] else 0
  r[f'own_{a}_TEST_and_TRAIN_source_overlap']=sp=='train' and r[f'TEST_{a}_selected'] and r[f'other_{a}_selected']
 pairflags.append(r)
writecsv(E/'PAIR_INPUT_AND_SUPERVISION_FLAGS.csv',pairflags)
norm=lambda s:unicodedata.normalize('NFKC',s or '').casefold().strip()
ident=[]
for a,b in sorted({(p['test_business_id'],p['other_business_id']) for p in pairs}):
 x=allbiz[a];y=allbiz[b];sameaddr=norm(x['address'])==norm(y['address']) and bool(x['address'])
 def dist(x,y):
  rad=math.pi/180;lat1=x['latitude']*rad;lat2=y['latitude']*rad;dlat=lat2-lat1;dlon=(y['longitude']-x['longitude'])*rad
  return 6371000*2*math.asin(min(1,math.sqrt(math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)))
 ident.append(dict(test_business_id=a,other_business_id=b,test_name=x['name'],other_name=y['name'],test_address=x['address'],other_address=y['address'],test_city=x['city'],other_city=y['city'],test_state=x['state'],other_state=y['state'],normalized_names_equal=norm(x['name'])==norm(y['name']),street_address_equal=sameaddr,full_address_equal=sameaddr and norm(x['city'])==norm(y['city']) and norm(x['state'])==norm(y['state']),coordinate_distance_m=dist(x,y),
  same_business_id=False,source_metadata_independent_reverification='Pending raw HF business metadata verification'))
writecsv(E/'BUSINESS_IDENTITY_COMPARISON.csv',ident)
summary=dict(status='COMPLETE_FROZEN_INPUT_AND_SUPERVISION_TRACE',affected_TEST_businesses=sum(r['duplicate_photo_ids']>0 for r in businessrows),duplicated_TEST_photo_ids=len(ledger),duplicate_pairs=len(pairs),business_pairs=len(ident),
 U_candidate_duplicate_photos=sum(r['basic_U_candidate'] for r in ledger),unused_duplicate_photos_not_selected_U_or_F=sum(r['raw_or_candidate_only_not_selected_U_or_F'] for r in ledger),
 conditions={a:dict(duplicated_TEST_selected_photos=sum(r[f'selected_TEST_{a}'] for r in ledger),TEST_businesses_with_selected_duplicates=len({r['test_business_id'] for r in ledger if r[f'selected_TEST_{a}']}),
   own_TRAIN_source_overlap_photos=sum(r[f'{a}_own_condition_TEST_input_and_TRAIN_input_overlap'] for r in ledger),own_TRAIN_source_overlap_businesses=len({r['test_business_id'] for r in ledger if r[f'{a}_own_condition_TEST_input_and_TRAIN_input_overlap']}),
   own_TRAIN_target_supported_overlap_photos=sum(r[f'{a}_own_condition_TEST_input_and_TRAIN_target_support_overlap'] for r in ledger),own_VALIDATION_only_overlap_photos=sum(r[f'{a}_own_condition_TEST_input_and_VALIDATION_only_overlap'] for r in ledger),
   same_caption_TRAIN_overlap_photos=sum(r[f'selected_TEST_{a}'] and r[f'{a}_train_same_caption_text_as_TEST'] for r in ledger)) for a in ['U','F']},
 current_full_address_equal_business_pairs=sum(r['full_address_equal'] for r in ident),same_name_business_pairs=sum(r['normalized_names_equal'] for r in ident),minimum_business_coordinate_distance_m=min(r['coordinate_distance_m'] for r in ident),
 model_input='LED receives captions, not raw pixels. Own-condition TRAIN caption/source overlap is distinguished from merely any raw image overlap. VALIDATION is development exposure, not optimizer training.',
 retained_new_phase_evidence='Actual1000successful window logs verify2Visualpresentations per involved TRAINbusiness per U/Fcondition; Original750ancestral exposure follows frozen own-condition inputs and source/sampler-derived accounting, not direct historical photo-ID journals.',
 TEST_references_read=0,TEST_predictions_generated=0,TEST_quality_metrics=0)
save(E/'EXPOSURE_SUMMARY.json',summary);save(E/'TRAIN_VALIDATION_TARGET_SUPPORT_RECORDS.json',claimproof);save(E/'EXPOSURE_INPUT_BINDINGS.json',inputbinding)
print(json.dumps(summary,indent=2))
