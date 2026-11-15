"""D054 strict v2 reference loader and bounded 24-case input preflight."""
import hashlib,json
from pathlib import Path
from check_stage_i_d053_final_references_v2 import check as check_references,load_context
from stage_i_paired_pilot import ROOT,checked_preparation,require,sha,read,ARMS

OUT=ROOT/'outputs/stage_i/d054_paired_generation_v1'
CONFIG=ROOT/'configs/stage_i_d053_evaluation_v1.json'
def text_sha(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def semantic_generation(g):return {k:v for k,v in g.items() if k not in ['transformers_version','_from_model_config','_commit_hash']}
def preflight():
 ref=check_references(ROOT);require(ref['valid'] and ref['reference_freeze_manifest_checked'],'D053 final freeze invalid')
 ctx=load_context(ROOT);cfg,prep,manifest=checked_preparation(ROOT/'configs/stage_i_paired_lora_pilot_v1.json');ec=read(CONFIG)
 require(sha(CONFIG)==ctx['bindings']['evaluation_config_sha256'],'Frozen evaluation configuration changed')
 inputs=[json.loads(s) for s in (prep/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
 cases=ctx['cases'];by={c['business_id']:c for c in cases}
 require([c['business_id'] for c in inputs]==[c['business_id'] for c in cases]==ec['random_business_ids']+ec['targeted_business_ids'],'24 business IDs/order mismatch')
 require(len(inputs)==24 and sum(c['subset']=='random' for c in inputs)==20 and sum(c['subset']=='targeted' for c in inputs)==4,'Scope differs')
 for c in inputs:
  s=by[c['business_id']];require(c['split']=='validation' and c['condition']=='filtered','Foreign split/condition')
  require(c['source']==s['source'] and text_sha(c['source'])==c['source_sha256']==s['source_sha256'],'Source hash/text changed')
  require(c['selected_review_ids']==s['selected_review_ids'] and c['selected_photo_ids']==s['selected_photo_ids'],'Review/photo IDs/order mismatch')
  require(0<len(c['input_ids'])<=5000 and len(c['label_ids'])<=768,'Token budget changed')
 expected=read(prep/'expected_generation_config.json');require(semantic_generation(expected)==semantic_generation(ec['effective_generation_config']),'Frozen generation configurations differ')
 training=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json');descriptors={a['arm']:a for a in training['architectures']};protected={}
 for arm in ARMS:
  a=descriptors[arm];adapter=ROOT/ec['arms'][arm]['adapter_path'];require(sha(adapter)==a['best_adapter']['sha256']==ec['arms'][arm]['adapter_sha256'],'Adapter hash mismatch')
  selected=next(x for x in a['checkpoints'] if x['step']==128)
  for name,record in selected['files'].items():
   p=ROOT/selected['path']/name;require(sha(p)==record['sha256'],'Checkpoint file changed: '+str(p));protected[p.relative_to(ROOT).as_posix()]=record['sha256']
  require(sha(adapter.parent/'adapter_config.json')==selected['files']['adapter_config.json']['sha256'],'Best adapter config changed')
  protected[adapter.relative_to(ROOT).as_posix()]=sha(adapter);protected[(adapter.parent/'adapter_config.json').relative_to(ROOT).as_posix()]=sha(adapter.parent/'adapter_config.json')
  ac=read(adapter.parent/'adapter_config.json');require(ac['r']==8 and ac['lora_alpha']==16 and ac['lora_dropout']==0.0 and ac['base_model_name_or_path']==cfg['model_id'],'Adapter settings differ')
 for p in [ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json',ctx['final']/'FINAL_REFERENCE_ANNOTATIONS.json',ctx['final']/'REFERENCE_FREEZE_MANIFEST.json',CONFIG,prep/'evaluation_inputs.jsonl',prep/'expected_generation_config.json',prep/'blinding_key_keep_separate.json']:
  protected[p.relative_to(ROOT).as_posix()]=sha(p)
 return ctx,cfg,prep,ec,inputs,training,protected,ref

def load_compatible_comparison(predictions):
 """Dispatch only v2 reference validation; reuse unchanged D053 metric functions.

 No patching of the v1 loader and no fabricated output human annotations.
 """
 from stage_i_d053_evaluation import validate_predictions,repetition_diagnostics,structural_diagnostics,token_termination,human_metrics,human_output_annotation_template,paired_comparison
 ctx,cfg,prep,ec,inputs,training,protected,ref=preflight();cases=ctx['cases'];validate_predictions(predictions,cases,ec)
 refpath=ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json';frozen=read(refpath)
 require(frozen['reference_format_version']==2,'Unsupported reference version')
 require(all(p['reference_freeze_sha256']==sha(refpath) for p in predictions),'Prediction reference binding mismatch')
 rows=[];shells=[]
 for p in predictions:
  case=next(c for c in cases if c['business_id']==p['business_id']);units=[u for u in frozen['units'] if u['business_id']==p['business_id']]
  rows.append({'arm':p['arm'],'business_id':p['business_id'],'subset':case['subset'],'prediction_text_sha256':text_sha(p['text']),
   'automatic':{'repetition':repetition_diagnostics(p['text'],p['token_ids'],ec['effective_generation_config']),'structure':structural_diagnostics(p['text'],len(case['captions'])),'termination':token_termination(p['token_ids'],ec['effective_generation_config'])},
   'human':human_metrics(None,p,case,units)})
  shells.append(human_output_annotation_template(p,units))
 return {'status':'ACTUAL48_OUTPUTS_EVALUATED_WITH_EXPLICIT_D053_V2_LOADER','per_output':rows,'paired':paired_comparison(rows,ec),'pending_researcher_output_annotations':shells,'source_reference_validation':ref,'output_human_annotations_created':0,'automatic_factual_judgment':False}
