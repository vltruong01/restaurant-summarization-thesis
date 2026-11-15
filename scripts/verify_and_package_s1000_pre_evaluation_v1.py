"""Final independent CPU content/binding/package checks. No model, TEST reference or API access."""
from pathlib import Path
import json,hashlib,shutil,zipfile,tempfile,csv,re,datetime,sys
R=Path(__file__).resolve().parents[1];O=R/'outputs/pre_evaluation/s1000_readiness_v1';E=O/'EVIDENCE';M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def hs(s):return hashlib.sha256(s.encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
reports=['FINAL_MODEL_FREEZE_AUDIT.md','TEST100_READINESS_REPORT.md','SHUFFLED_NEGATIVE_CONTROL_SPEC.md','TEACHER_MIMICRY_PRE_EVALUATION_AUDIT.md','EXTERNAL_GENERALIZATION_FEASIBILITY.md','PRE_EVALUATION_DECISIONS_AND_BLOCKERS.md','THESIS_SDD_DOCS_UPDATE_SUMMARY.md']
assert all((O/n).is_file() and (O/n).stat().st_size>300 for n in reports)
inputs=read(E/'INPUT_BINDINGS.json')
for rel,b in inputs.items():assert sha(R/rel)==b['sha256'] and (R/rel).stat().st_size==b['bytes'],rel
stage=R/'outputs/stage_f/production_caption_construction_d033_colab_v1'
expected=read(stage/'completion_report.json')['artifact_sha256']
for rel,h in expected.items():assert (stage/rel).is_file() and sha(stage/rel)==h,rel
changes=read(E/'DOCUMENTATION_CHANGE_MANIFEST.json')
for c in changes:assert sha(R/c['path'])==c['after_sha256'] and sha(O/c['original_snapshot'])==c['before_sha256']
for c in changes:
 if c['path'].startswith('thesis_sdd_docs/'):
  text=(R/c['path']).read_text(encoding='utf8')
  assert '_V 1.md' not in text and 's 1000' not in text and '_v 1' not in text
assert (R/'DECISIONS.md').read_bytes().endswith((O/'DOCUMENTATION_BEFORE/DECISIONS.md').read_bytes())
freeze=read(E/'MODEL_FREEZE_AUDIT.json');cfg=[x['adapter_configuration'] for x in freeze['models'].values()]
prose=(O/'FINAL_MODEL_FREEZE_AUDIT.md').read_text(encoding='utf8')
for s in [freeze['backbone']['sha256'],freeze['tokenizer']['revision'],*[r['adapter_sha256'] for r in freeze['models'].values()]]:assert s in prose,'Literal identifier corrupted in report prose'
assert '2 e-5' not in prose and 'S 1000' not in prose
for k in ['r','lora_alpha','lora_dropout','bias','task_type','peft_type']:assert len({json.dumps(c[k],sort_keys=True) for c in cfg})==1
assert len({tuple(sorted(c['target_modules'])) for c in cfg})==1
ready=read(E/'TEST100_READINESS.json');assert ready['TEST_inference_calls']==ready['TEST_metrics_computed']==ready['TEST_reference_payloads_opened']==0
ledger=read(E/'TEST100_INPUT_LEDGER.json');assert len(ledger)==len({x['business_id'] for x in ledger})==100
assert all(not any(k in r for k in ['reference','target','source','prediction']) for r in ledger)
m=read(E/'MIMICRY_SUMMARY.json');rr=read(E/'MIMICRY_ALL_VALIDATION_SCREEN.json');assert len(rr)==1000
for name,s in m['summary'].items():
 a,t=name.split('_',1);part=[r for r in rr if r['condition']==a and r['task']==t];assert len(part)==200
 assert sum(r['normalized_exact_own_teacher_sentences'] for r in part)==s['normalized_exact_own_teacher_sentences']
 assert sum(r['near_own_teacher_sentences'] for r in part)==s['near_own_teacher_sentences']
 assert all(r['business_id'] not in {l['business_id'] for l in ledger} for r in part)
src=read(M/'COMPARISON/EXACT_VALIDATION200_SOURCES.json');src={r['business_id']:r for r in src}
examples=read(E/'MIMICRY_SELECTED_EXAMPLES.json');verified=0
for x in examples:
 s=src[x['business_id']]
 assert x['teacher_review']==s['common_original_review_reference']
 for issue in x['confirmed_prior_source_grounded_findings']:
  assert issue['exact_output_span'] in x['raw_output']
  for ev in issue['source_evidence']:
   records=s['exact_review_blocks']+s['conditions'][x['condition']]['captions'] if x['condition']!='RO' else s['exact_review_blocks']
   rec=next(r for r in records if r.get('review_id',r.get('photo_id'))==ev['id'])
   assert ev['exact_excerpt'] in rec.get('exact_text',rec.get('exact_caption'));verified+=1

# Explicitly checked positive/omission examples supplement previously bound issue records.
checks=[('am5qD6LFjkV8f_4MLp8Img','REV_8','Drinks were forgotten. Waters weren\'t refiled.'),
 ('8PydwFenDDgKgV768QwgEg','REV_16',"I especially like the salmon chowder -- it was the best I've had."),
 ('8PydwFenDDgKgV768QwgEg','REV_26','the steak and frites was absolutely the best I have ever had.')]
positive=[]
for bid,label,quote in checks:
 r=next(x for x in src[bid]['exact_review_blocks'] if x['label']==label);assert quote in r['exact_text'];positive.append(dict(business_id=bid,label=label,id=r['review_id'],exact_excerpt=quote))
u=read(M/'RECOVERY_R1/DOWNLOADED/VALIDATION_RESULTS/U/predictions.json');uu={r['business_id']:r for r in u}
bid='NRl1WVY_HJNilsMWaVMQPg';s=src[bid];cap=next(c for c in s['conditions']['U']['captions'] if c['label']=='IMG_CAP_4')
span='The tanker truck is next to a tractor trailer.';assert span in uu[bid]['passes']['visual']['text'] and 'next to a tractor trailer' in cap['exact_caption']
positive.append(dict(business_id=bid,condition='U',output_span=span,caption=cap,Teacher_visual_omits_relation='next to a tractor trailer' not in s['conditions']['U']['original_visual_reference']))
bid='-IQsXtexaUmIfSBkhBZrKQ';s=src[bid];span='The photo shows a close-up of a plate of food.'
assert span in uu[bid]['passes']['visual']['text'] and span in s['conditions']['U']['original_visual_reference']
positive.append(dict(business_id=bid,condition='U',output_span=span,caption=s['conditions']['U']['captions'][0],Teacher_same_sentence=True))
save(E/'MIMICRY_POSITIVE_AND_OMISSION_BINDINGS.json',positive)
sys.path.insert(0,str(R/'scripts'))
from test_s1000_shuffled_preparation_v1 import SourceSeparatedShuffled
from prepare_s1000_shuffled_control_v1 import prepare_mapping
SourceSeparatedShuffled.setUpClass();f=SourceSeparatedShuffled();fixtures=f.sample();mapped=prepare_mapping(fixtures,f.tok,f.policy)
save(E/'SHUFFLED_SYNTHETIC_FIXTURE_MAPPING.json',dict(input_rows=fixtures,result=mapped))
assert mapped['TEST_mapping_created'] is False and read(E/'SHUFFLED_POLICY_PROPOSAL.json')['actual_TEST_mapping_created'] is False
source_paths=[M/'COMPARISON/EXACT_VALIDATION200_SOURCES.json',M/'COMPARISON/CLAIM_LEVEL_ISSUES.json',M/'COMPARISON/QUALITATIVE_SOURCE_CHECKLISTS.json',M/'COMPARISON/HISTORICAL_F_PANEL_REUSE.json']
save(E/'VALIDATION_QUALITATIVE_PROVENANCE.json',{p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in source_paths})
checks_receipt=dict(status='PASS_PRE_EVALUATION_CONTENT_AND_BINDINGS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_models=3,VALIDATION_components_screened=1000,TEST_source_businesses=100,
 INPUT_bindings_rehashed=len(inputs),StageF_original_manifest_hashes_reverified=len(expected),documentation_files_changed=len(changes),source_evidence_excerpts_rechecked=verified,
 positive_and_omission_records_checked=len(positive),shuffled_CPU_tests=8,actual_TEST_mapping_created=False,TEST_reference_payloads_opened=0,TEST_inference_calls=0,
 checkpoint_source_target_splits_unchanged=True,methodological_decisions_pending=True,no_new_GPU_or_paid_resources=True)
save(E/'FINAL_CONTENT_VERIFICATION.json',checks_receipt)

# Small portable package; no checkpoint binaries, bulk sources, TEST text/references or old exploratory narratives.
P=O/'REVIEW_PACKAGE';P.mkdir(exist_ok=False)
for n in reports:cp(O/n,P/n)
for p in E.iterdir():
 if p.is_file() and p.name not in ['MIMICRY_RUN.stdout.txt']:cp(p,P/'EVIDENCE'/p.name)
for n in ['INSTANCE_CLOSE_REQUEST.json','INSTANCE_CLOSE_RESULT.json','VERIFIED_CLOSEOUT.json','BACKBONE_TRANSFER_BINDING.json','SELECTED_F_CHECKPOINT_FREEZE.json']:
 cp(M/n,P/'PRESERVATION'/n)
for a in ['RO','U']:
 for n in [f'{a}_TRAINING_CPU_VERIFICATION.json',f'{a}_TERMINAL_TRAINING_LOCAL_VERIFICATION.json',f'{a}_TERMINAL_TRAINING_MEMBERS.json',f'{a}_TERMINAL_VALIDATION_LOCAL_VERIFICATION.json']:
  cp(M/'RECOVERY_R1/DOWNLOADED'/n,P/'PRESERVATION'/n)
for name in ['COMPARISON_VALIDITY_AND_RUN_PLAN.md']:
 cp(M/name,P/'METHOD_PROVENANCE'/name)
for name in ['COHORT_MANIFEST.json','QUALITATIVE_SUBSET_MANIFEST.json']:
 cp(R/'outputs/stage_i/led_large_filtered_validation200_selection_v1'/name,P/'METHOD_PROVENANCE'/name)
cp(R/'outputs/stage_i/test_reference_availability_v1/TEST_REFERENCE_AVAILABILITY_RECEIPT.json',P/'METHOD_PROVENANCE/PRIOR_REFERENCE_METADATA_ONLY.json')
for name in ['image_overlap_evaluation_policy.yaml','led_source_template_d032.json']:
 cp(R/'configs'/name,P/'METHOD_PROVENANCE'/name)
for name in ['audit_s1000_pre_evaluation_v1.py','audit_s1000_teacher_mimicry_v1.py','prepare_s1000_shuffled_control_v1.py','test_s1000_shuffled_preparation_v1.py','led_large_shuffled_production_v2.py','source_budget_audit_template.py','complete_s1000_pre_evaluation_v1.py','polish_s1000_pre_evaluation_docs_v1.py','finalize_current_s1000_thesis_docs_v1.py','verify_and_package_s1000_pre_evaluation_v1.py']:
 cp(R/'scripts'/name,P/'SCRIPTS'/name)
cp(M/'source_separated_contract.py',P/'SCRIPTS/source_separated_contract.py')
for c in changes:
 if c['path'].startswith('thesis_sdd_docs/') or c['path']=='THESIS_PROGRESS.md':cp(R/c['path'],P/'UPDATED_DOCUMENTATION'/c['path'])
decision=(R/'DECISIONS.md').read_text(encoding='utf8').split('## ',2)[1] # exact current first entry, historical entries excluded
(P/'UPDATED_DOCUMENTATION/CURRENT_DECISION_LOG_ENTRY.md').write_text('## '+decision,encoding='utf8')
(P/'README.md').write_text('''# S1000 Pre-Evaluation Readiness review

Start with PRE_EVALUATION_DECISIONS_AND_BLOCKERS.md. The six requested reports, structured evidence, CPU scripts/tests and four updated existing thesis_sdd_docs files are included. Final RO/U/F checkpoints remain selected/frozen; S is an evaluation intervention using F, not training. No TEST inference/reference/quality evaluation or external experiment was performed. Readiness is not execution authorization.

TEST evidence contains source hashes/IDs/budget counts and photo/caption provenance, not TEST reference text or model output. Mimicry examples use TRAIN/VALIDATION only. Lexical similarity flags are not semantic error judgments. Confirmed examples cite previous bounded model-assisted source reading; no independent human assessment is claimed.

The actual Shuffled TEST mapping is intentionally uncommitted pending essential policy approval. Synthetic mapping/test records demonstrate separate Review/Visual intervention. With-replacement donor proposal is not a permutation. Source-Separated F/S Review invariance is structural and must be reflected in the evaluation protocol.

No checkpoint binaries, bulk original Yelp datasets or historical exploratory-technique diary are included. Full original checkpoints/logs/off-instance backups remain under their hash-bound local paths in PRESERVATION and model freeze evidence. Exact previous documentation and unified diffs remain locally in DOCUMENTATION_BEFORE / DOCUMENTATION_DIFFS beside this package; their before/after hashes are included, without importing their old narratives into the current thesis progress.

Scripts are audit implementations bound to the existing repository paths, not standalone inference tools or authorization to execute. complete_s1000_pre_evaluation_v1.py intentionally refuses to overwrite original doc snapshots. Use existing frozen inputs; do not provision compute. Public external pages were inspected only for feasibility/license evidence, without downloading dataset payloads.

PACKAGE_MANIFEST.csv lists all payload filenames/bytes/SHA-256 except itself. External PACKAGE_INTEGRITY_REPORT.json provides ZIP checksum and extraction/hash verification. STOP for researcher/ChatGPT review.
''',encoding='utf8')
files=sorted(p for p in P.rglob('*') if p.is_file());manifest=[dict(filename=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
assert not any(p.suffix in ['.pt','.npz','.safetensors','.bin','.jpg','.jpeg'] for p in files)
with (P/'PACKAGE_MANIFEST.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['filename','bytes','sha256']);w.writeheader();w.writerows(manifest)
Z=O/'S1000_PRE_EVALUATION_READINESS_REVIEW.zip';assert not Z.exists()
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(P.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(P).as_posix())
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
 with tempfile.TemporaryDirectory(prefix='s1000_pre_eval_check_',dir=O) as tmp:
  t=Path(tmp);z.extractall(t)
  for f in manifest:assert (t/f['filename']).stat().st_size==f['bytes'] and sha(t/f['filename'])==f['sha256']
  assert (t/'PACKAGE_MANIFEST.csv').read_bytes()==(P/'PACKAGE_MANIFEST.csv').read_bytes()
  for f in t.rglob('*.json'):json.loads(f.read_text(encoding='utf8'))
  for n in reports:assert (t/n).read_bytes()==(O/n).read_bytes()
receipt=dict(status='PASS_REOPEN_EXTRACT_CRC_PARSE_EVERY_PAYLOAD_SHA256',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=Z.name,bytes=Z.stat().st_size,sha256=sha(Z),payload_files=len(manifest),ZIP_members=len(manifest)+1,manifest_sha256=sha(P/'PACKAGE_MANIFEST.csv'),reports=7,requested_six_reports_present=True,existing_thesis_sdd_docs_updated=4,checkpoint_binaries_included=0,TEST_predictions_or_references_included=0,TEST_inference_calls=0,scope='Complete preparatory audit, pending decisions; stop before final evaluation')
save(O/'PACKAGE_INTEGRITY_REPORT.json',receipt)
(O/'S1000_PRE_EVALUATION_READINESS_REVIEW.zip.sha256').write_text(receipt['sha256']+'  '+Z.name+'\n',encoding='utf8')
print(json.dumps(receipt,indent=2))
