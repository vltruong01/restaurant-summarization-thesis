"""Read-only CPU model freeze and TEST input readiness. Never reads TEST references or calls a model."""
from pathlib import Path
import json,csv,hashlib,collections,re,sys,datetime,zipfile,xml.etree.ElementTree as ET
from tokenizers import Tokenizer
R=Path(__file__).resolve().parents[1];O=R/'outputs/pre_evaluation/s1000_readiness_v1'
M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1';N=M/'RECOVERY_R1'
F=R/'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1';V=R/'outputs/stage_i/led_large_filtered_validation200_selection_v1'
S=R/'outputs/stage_f/production_caption_construction_d033_colab_v1';E=O/'EVIDENCE'
O.mkdir(parents=True,exist_ok=True);E.mkdir(exist_ok=True)
sys.path.insert(0,str(M));from source_separated_contract import isolated_sources,R_TASK
sys.path.insert(0,str(R/'scripts'));from source_budget_audit_template import review_block
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):
 with p.open(encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def hs(t):return hashlib.sha256(t.encode()).hexdigest()
def obj(x):return hs(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')))
def save(name,x):
 p=E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
bindings={}
def bind(p):bindings[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return bindings[p.relative_to(R).as_posix()]['sha256']

# Frozen models, actual backups and tokenization dependencies.
close=read(M/'VERIFIED_CLOSEOUT.json')
for x in close['training_archives']:assert sha(Path(x['absolute_local_path']))==x['sha256'];bind(Path(x['absolute_local_path']))
assert sha(M/'RO_U_F_S1000_VALIDATION200_REVIEW.zip')==close['review_archive_sha256'];bind(M/'RO_U_F_S1000_VALIDATION200_REVIEW.zip')
freeze=read(M/'SELECTED_F_CHECKPOINT_FREEZE.json');assert all(sha(Path(p))==h for p,h in freeze['files'].items())
for p in freeze['files']:bind(Path(p))
bb=read(M/'BACKBONE_TRANSFER_BINDING.json');assert sha(Path(bb['local_file']))==bb['sha256'];bind(Path(bb['local_file']))
tokdir=M/'COMMON/assets/tokenizers/led-base-16384';tok=Tokenizer.from_file(str(tokdir/'tokenizer.json'))
stageF=read(S/'completion_report.json')
for f,h in stageF['pinned_tokenizer']['sha256'].items():assert sha(tokdir/f)==h;bind(tokdir/f)
assert tok.token_to_id('<s>')==0 and tok.token_to_id('<pad>')==1 and tok.token_to_id('</s>')==2
assert tok.encode('Compatibility probe.').ids[0]==0 and tok.encode('Compatibility probe.').ids[-1]==2
gen=read(M/'VALIDATION_PHASE/EFFECTIVE_GENERATION_CONFIG.json');bind(M/'VALIDATION_PHASE/EFFECTIVE_GENERATION_CONFIG.json')
assert gen['num_beams']==4 and gen['length_penalty']==1 and gen['max_length']==768 and not gen['do_sample'] and gen['forced_eos_token_id']==2
models={}
for a in ['RO','U','F']:
 root=N/f'DOWNLOADED/{a}/ARMS/{a}/RUN/checkpoint-1000' if a!='F' else F/'downloaded/RUN/checkpoint-1000'
 cfg=read(root/'adapter_config.json');assert cfg['r']==8 and cfg['lora_alpha']==16 and cfg['lora_dropout']==0 and cfg['bias']=='none'
 ah=bind(root/'adapter_model.safetensors');bind(root/'adapter_config.json');state=bind(root/'state.pt')
 if a!='F':
  proof=read(N/f'DOWNLOADED/{a}_TRAINING_CPU_VERIFICATION.json');assert proof['successful_updates']==1000 and proof['all456_gradients_finite_every_successful_window']
  assert ah==proof['checkpoints']['1000']['adapter_sha256'] and state==proof['checkpoints']['1000']['full_state_sha256']
  config=read(N/f'ARMS/{a}/TRAINING_CONFIG.json');init=config['initialization_adapter_sha256'];newR=proof['review_presentations'];newV=proof['visual_presentations']
  proofpath=N/f'DOWNLOADED/{a}_TRAINING_CPU_VERIFICATION.json'
 else:
  proof=read(F/'downloaded/RUN/COMPLETION.json');assert proof['successful_updates']==1000
  init='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60';newR=4000;newV=4000;proofpath=F/'downloaded/RUN/COMPLETION.json'
 bind(proofpath)
 predpath=N/f'DOWNLOADED/VALIDATION_RESULTS/{a}/predictions.json' if a!='F' else V/'MERGED/V1_1000/predictions.json'
 predictions=read(predpath);assert len(predictions)==200 and len({p['business_id'] for p in predictions})==200
 for p in predictions:
  for g in p['passes'].values():
   assert g['checkpoint_sha256']==ah and hs(g['text'])==g['text_sha256'] and tok.decode(g['token_ids'],skip_special_tokens=True)==g['text']
 bind(predpath)
 models[a]=dict(adapter_sha256=ah,full_state_sha256=state,path=root.relative_to(R).as_posix(),adapter_configuration=cfg,initialization_adapter_sha256=init,
  successful_new_phase_updates=1000,original_parent_updates=750,review_presentations_new_phase=newR,visual_presentations_new_phase=newV,
  proof=proofpath.relative_to(R).as_posix(),validation_predictions=200,predictions_sha256=sha(predpath),restore_requirements='Load original local LED-large backbone explicitly; historical base_model_name_or_path is an obsolete container path, not a portable resolver')
result=dict(status='PASS_FROZEN_ARTIFACT_HASH_AND_STORED_OUTPUT_BINDINGS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,backbone=bb,tokenizer=stageF['pinned_tokenizer'],special_token_ids=dict(BOS=0,PAD=1,EOS=2,decoder_start=2),generation=gen,
 backups='Local off-instance full training/validation archives and extracted500/1000 full states; SHA reverified, prior per-member/timestamp receipts retained',
 restore_scope='Hash, full-state structure/numerical receipts, tokenizer and stored-token restoration checked; no new GPU forward or independent future GPU restore trial performed',
 instance_close=read(M/'INSTANCE_CLOSE_RESULT.json'))
assert result['instance_close']['status']=='PROVIDER_CONFIRMED_DELETE_SUCCESS';save('MODEL_FREEZE_AUDIT.json',result)

# Input-only TEST100 audit: source manifests have no TEST labels/references. Never print TEST source text.
split=list(csv.DictReader((R/'data/processed/yelp_2300_v1/business_split.csv').open(encoding='utf8')))
sets={s:{r['business_id'] for r in split if r['split']==s} for s in ['train','validation','test']}
assert list(map(len,[sets['train'],sets['validation'],sets['test']]))==[2000,200,100]
assert not any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
bind(R/'data/processed/yelp_2300_v1/business_split.csv')
shared={r['business_id']:r for r in rows(S/'manifests/shared_review_inputs.jsonl') if r['split']=='test'}
src={c:{r['business_id']:r for r in rows(S/f'manifests/test/{c}_sources.jsonl')} for c in ['reviews_only','unfiltered','filtered']}
cap={c:{r['business_id']:r for r in rows(S/f'manifests/test/{c}_captions.jsonl')} for c in ['unfiltered','filtered']}
pools={r['business_id']:r for r in rows(S/'manifests/frozen_filter_pools.jsonl') if r['split']=='test'}
for group in [shared,pools,*src.values(),*cap.values()]:assert set(group)==sets['test']
for c in src:
 for row in src[c].values():assert row.get('teacher_target') is None or row.get('teacher_target') is False,'Unexpected nonempty target field; no reference content inspected or emitted'
selected={rid for x in shared.values() for rid in x['selected_review_ids']};raw={}
for r in rows(R/'data/processed/yelp_2300_v1/reviews/test.jsonl'):
 if r['review_id'] in selected:assert r['review_id'] not in raw;raw[r['review_id']]=r
assert set(raw)==selected
photos={r['photo_id']:r for r in csv.DictReader((R/'data/processed/yelp_2300_v1/photo_manifest.csv').open(encoding='utf8'))}
testphotos={p:r for p,r in photos.items() if r['split']=='test'}
photo_metadata={r['photo_id']:r for r in rows(R/'data/processed/yelp_2300_v1/photo_metadata/test.jsonl')}
assert set(photo_metadata)==set(testphotos)
for pid,ph in testphotos.items():
 pp=R/'data/processed/yelp_2300_v1'/ph['relative_path']
 assert ph['business_id'] in sets['test'] and photo_metadata[pid]['business_id']==ph['business_id']
 assert pp.is_file() and pp.stat().st_size==int(ph['bytes']) and sha(pp)==ph['sha256']
ledger=[];photo_checked={};caption_records={};errors=[]
for bid in sorted(sets['test']):
 sh=shared[bid];reviewrows=[raw[rid] for rid in sh['selected_review_ids']]
 assert all(x['business_id']==bid for x in reviewrows)
 assert review_block(reviewrows)==sh['review_block'] and hs(sh['review_block'])==sh['review_block_sha256']
 assert sh['review_text_sha256_in_order']==[hs(r['text']) for r in reviewrows]
 assert [(r['date'],r['review_id']) for r in reviewrows]==sorted((r['date'],r['review_id']) for r in reviewrows)
 assert 0<len(reviewrows)<=30
 base=src['reviews_only'][bid]['source'];assert hs(base)==sh['base_sha256'] and base.endswith(sh['review_block'])
 reviewids=tok.encode(R_TASK+base).ids;assert len(reviewids)<=5000 and len(tok.encode(sh['review_block']).ids)<=4000
 item=dict(business_id=bid,split='test',selected_review_ids=sh['selected_review_ids'],review_text_sha256_in_order=sh['review_text_sha256_in_order'],review_block_sha256=sh['review_block_sha256'],review_tokens=len(tok.encode(sh['review_block']).ids),review_call_source_sha256=hs(R_TASK+base),review_input_ids_sha256=obj(reviewids),review_call_tokens=len(reviewids),metadata_categories=sh['metadata']['categories'],conditions={})
 for c in src:
  x=src[c][bid];assert x['selected_review_ids']==sh['selected_review_ids'] and hs(x['source'])==x['source_sha256']
  assert x['source'].split('\n\n[VISUAL_CONTEXT]\n')[0]==base
  full=tok.encode(x['source']).ids;increment=len(full)-len(tok.encode(base).ids)
  assert len(full)<=5000 and increment<=750
  selectedcaps=[];vs=None;vids=[]
  if c!='reviews_only':
   cc=cap[c][bid];assert cc['K']==5 and cc['selected_photo_ids']==x['selected_photo_ids'] and len(cc['captions'])==len(x['selected_photo_ids'])<=5
   expected=pools[bid]['unfiltered_fixed_seed_ranking' if c=='unfiltered' else 'filtered_diversity_ranking'][:5];assert cc['selected_photo_ids']==expected
   rs,vs=isolated_sources(x['source']);assert rs==R_TASK+base and '[REV_' not in vs and '[BUSINESS_INFORMATION]' not in vs
   body=x['source'].split('\n\n[VISUAL_CONTEXT]\n')[1];serialized='\n\n'.join(f'[IMG_CAP_{i+1}]\n{r["normalized_caption"]}' for i,r in enumerate(cc['captions']));assert body==serialized
   vids=tok.encode(vs).ids;assert len(vids)<=5000
   for ca in cc['captions']:
    pid=ca['photo_id'];ph=photos[pid];assert ph['business_id']==bid and ph['split']=='test' and ca['basic_eligible'] is True
    pp=R/'data/processed/yelp_2300_v1'/ph['relative_path']
    if pid not in photo_checked:
     assert pp.exists() and pp.stat().st_size==int(ph['bytes']) and sha(pp)==ph['sha256'];photo_checked[pid]=dict(photo_id=pid,sha256=ph['sha256'],path=pp.relative_to(R).as_posix())
    rec=R/ca['caption_record_path'];assert sha(rec)==ca['caption_record_sha256'];record=read(rec)
    assert record['photo_id']==pid and record['business_id']==bid and record['normalized_caption']==ca['normalized_caption']
    caption_records[pid]=dict(path=rec.relative_to(R).as_posix(),sha256=sha(rec))
    selectedcaps.append(dict(photo_id=pid,caption_text_sha256=hs(ca['normalized_caption']),caption_record_sha256=ca['caption_record_sha256']))
   assert len(selectedcaps)>0 or not body
   if c=='filtered':assert not pools[bid]['CLIP_hard_rejection'] and not pools[bid]['pHash_hard_rejection']
  else:assert not x['selected_photo_ids'] and '[VISUAL_CONTEXT]' not in x['source']
  item['conditions'][c]=dict(original_source_sha256=x['source_sha256'],original_source_tokens=len(full),caption_increment_tokens=increment,caption_count=len(selectedcaps),selected_captions=selectedcaps,visual_call_source_sha256=hs(vs) if vs is not None else None,visual_input_ids_sha256=obj(vids) if vs is not None else None,visual_call_tokens=len(vids) if vs is not None else None)
 ledger.append(item)
for f in [S/'manifests/shared_review_inputs.jsonl',S/'manifests/frozen_filter_pools.jsonl',S/'completion_report.json',R/'data/processed/yelp_2300_v1/reviews/test.jsonl',R/'data/processed/yelp_2300_v1/photo_manifest.csv']:
 bind(f)
for c in src:bind(S/f'manifests/test/{c}_sources.jsonl')
for c in cap:bind(S/f'manifests/test/{c}_captions.jsonl')
for rel,h in stageF['artifact_sha256'].items():
 p=S/rel
 if p.is_file():assert sha(p)==h,rel
bind(R/'data/processed/yelp_2300_v1/photo_metadata/test.jsonl')
overlapfile=R/'outputs/stage_d/photo_integrity_duplicate_audit_v1/test_image_disjoint_diagnostic_businesses.csv'
overlap=list(csv.DictReader(overlapfile.open(encoding='utf8')));assert len(overlap)==100;bind(overlapfile)
save('TEST100_INPUT_LEDGER.json',ledger);save('TEST100_PHOTO_AND_CAPTION_BINDINGS.json',dict(original_photos=list(photo_checked.values()),caption_records=caption_records))
save('TEST100_READINESS.json',dict(status='PASS_SOURCE_ONLY_DATA_INTEGRITY_PENDING_PROTOCOL_AND_SHUFFLED_POLICY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),split_counts={k:len(v) for k,v in sets.items()},business_overlap=0,selected_review_count=len(selected),review_full_text_order_and_IDs_match_raw=True,
 caption_counts={c:sum(r['conditions'][c]['caption_count'] for r in ledger) for c in ['unfiltered','filtered']},zero_captions={c:[r['business_id'] for r in ledger if r['conditions'][c]['caption_count']==0] for c in ['unfiltered','filtered']},
 original_selected_photo_files_SHA_checked=len(photo_checked),all_TEST_photo_files_SHA_checked=len(testphotos),caption_record_bindings_checked=len(caption_records),all_sources_within_frozen_budgets=True,all_task_tokenization_compatible=True,
 selected_F_U_rank_prefixes_verified=True,photo_business_split_bindings_verified=True,TEST_reference_payloads_opened=0,TEST_inference_calls=0,TEST_metrics_computed=0,
 image_overlap_policy=read_yaml_placeholder if False else dict(decision='D-022',main100=True,secondary83_preserved=True,manifest_path=overlapfile.relative_to(R).as_posix(),note='Business disjointness does not imply exact-pixel image disjointness. Preserve prior main100 plus83secondary policy; no split edits.'),
 zero_caption_policy='Keep empty Visual input and frozen original task marker; do not invent captions or drop businesses. Future raw output preserved and judged separately.',
 TEST_tuning=False))

proposal=R/'thesis_proposal_v16.docx';bind(proposal);ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
xml=ET.fromstring(zipfile.ZipFile(proposal).read('word/document.xml'));paras=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in xml.findall('.//w:p',ns)]
sel=[dict(paragraph_index=i,text=t) for i,t in enumerate(paras) if i in [621,655,724,730,732,734,736,870,898,900,902,929,936,938,940,942,944,946,979]]
save('PROPOSAL_V16_METHOD_EXCERPTS.json',dict(proposal_sha256=sha(proposal),paragraph_numbering='Zero-based DOCX w:p index including table cells',excerpts=sel))
save('INPUT_BINDINGS.json',bindings)
print(json.dumps(read(E/'TEST100_READINESS.json'),indent=2))
