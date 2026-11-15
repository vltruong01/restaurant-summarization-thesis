"""Offline-only pilot preparation: fixed data, meta attachment, masks and initialization hashes."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import random
import faulthandler

faulthandler.dump_traceback_later(45, repeat=False)
print('Offline preparation: importing pinned CPU/meta dependencies', flush=True)

import torch
from transformers import LEDTokenizerFast

from stage_i_paired_pilot import ROOT,ARMS,read,sha,write_new as strict_write_new,require,meta_probe,expected_modules,initialized_A,epoch_orders,verify_mask_batch
from stage_hi_led_data import verify_bundle,PackedLEDDataset,LEDCompleteCollator
from stage_i_decoding_diagnostics import read_jsonl,captions

OUT=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
CONFIG=ROOT/'configs/stage_i_paired_lora_pilot_v1.json'
CLOUD=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
OLD=ROOT/'outputs/stage_i/training_failure_audit_v1/proposed_pilot_manifest.json'


def write_new(path,value):
    if Path(path).exists():
        require(read(path)==value,'Interrupted offline preparation contains different metadata: '+str(path))
    else:strict_write_new(path,value)


def main():
    print('Offline preparation: checking frozen bundle hashes', flush=True)
    require(OUT.is_dir() and not (OUT/'PREPARATION_MANIFEST.json').exists(),'Preserve completed preparation')
    old=read(OLD);original=read(CLOUD/'configs/stage_i_lora_led_training_approved_v2.json')
    manifest,tm,mc=verify_bundle(original,root=CLOUD)
    print('Offline preparation: frozen data hashes PASS', flush=True)
    pins=read(ROOT/'outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json')['required_runtime_versions']
    require(all(importlib.metadata.version(k).split('+')[0]==v for k,v in pins.items()),'Runtime drift')
    config={'status':'OFFLINE_PREPARED_NO_GPU_EXECUTION_APPROVED','default_execution_enabled':False,
        'authorization_policy':'Separate hash-bound approval JSON per scope; historical approvals are invalid',
        'condition':'filtered','model_id':original['model_id'],'revision':original['revision'],'seed':42,
        'lora':{k:v for k,v in original['lora'].items() if k!='target_modules'},'optimization':{**original['optimization'],'num_train_epochs':2},
        'training':{'businesses':512,'epochs':2,'steps':128,'steps_per_epoch':64,'warmup_steps':7},
        'generation':original['generation'],'budgets':{'source':5000,'reviews':4000,'captions':750,'target':768,'K':5},
        'preparation_directory':OUT.relative_to(ROOT).as_posix(),'frozen_training_root':CLOUD.relative_to(ROOT).as_posix(),
        'original_training_config':'configs/stage_i_lora_led_training_approved_v2.json',
        'run_root':'outputs/stage_i/paired_lora_pilot_runs_v1','runtime_pins':pins,
        'feasibility':{'optimizer_updates':3,'replay_updates':1,'total_microbatches_per_arm':32,'source_padding':5120,'label_padding':768,
                       'minimum_headroom_fraction':.15,'generation_calls':0},
        'initialization':'Independent Kaiming A/zero B with SHA256(common seed42 | canonical module name); shared decoder tensor equality',
        'historical_full_F_role':'Historical reference only, not a matched architecture baseline',
        'quality_decision':'Predefined multidomain researcher rubric; no rigid numerical quality-win thresholds',
        'source_unit_freeze_required_before_generation':True,'held_out_access':False,'automatic_fallback':False}
    write_new(CONFIG,config)
    sample={k:old[k] for k in ['training_business_ids','training_count','sample_seed','sampling','source_quartile_boundaries','strata',
                             'selected_zero_filtered_train_ids','validation_loss_business_ids','generation_evaluation_random20','generation_evaluation_targeted4']}
    sample.update(status='PREPARED_NOT_EXECUTED',original_manifest_sha256=sha(OLD),historical_full_F_role=config['historical_full_F_role'])
    require(len(sample['training_business_ids'])==512 and len(set(sample['training_business_ids']))==512,'Train count differs')
    require(not set(sample['training_business_ids'])&set(sample['validation_loss_business_ids']),'Train/validation overlap')
    write_new(OUT/'pilot_manifest.json',sample)
    orders=epoch_orders(512,42,2)
    write_new(OUT/'epoch_orders.json',{'indices':orders,'business_ids':[[sample['training_business_ids'][i] for i in o] for o in orders],
        'definition':'torch.randperm(512, CPU Generator(seed42+epoch)); same two fixed permutations in both arms'})
    probes={a:meta_probe(config,mc,a) for a in ARMS};write_new(OUT/'module_attachment_verification.json',probes)
    print('Offline preparation: actual meta module scopes PASS; calculating CPU adapter initialization hashes', flush=True)
    init={}
    for name in sorted(expected_modules(ARMS[1])):
        a=initialized_A((8,768),42,name);b=torch.zeros((768,8),dtype=torch.float32)
        init[name]=hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()).hexdigest()
    write_new(OUT/'initialization_hashes.json',{'common_seed':42,'per_module_sha256':init,
        'decoder_hashes_equal_between_independent_arms':True,'real_pretrained_weights_or_forward_used':False})
    print('Offline preparation: initialization hashes complete; checking24 source/label/mask cases', flush=True)
    all_cases=read_jsonl(ROOT/'outputs/stage_i/decoding_diagnostic_v1/cases.jsonl')
    cases=[c for c in all_cases if c['condition']=='filtered']
    require(len(cases)==24 and [c['business_id'] for c in cases]==sample['generation_evaluation_random20']+sample['generation_evaluation_targeted4'],'Validation case order differs')
    tokenizer=LEDTokenizerFast.from_pretrained(CLOUD/tm['relative_directory'],local_files_only=True)
    checks=[]
    for c in cases:
        require(tokenizer.encode(c['source'],truncation=False)==c['input_ids'] and tokenizer.encode(c['target'],truncation=False)==c['label_ids'],'Tokenizer drift or truncation')
        require(hashlib.sha256(c['source'].encode()).hexdigest()==c['source_sha256'],'Source text hash changed')
        f={'input_ids':c['input_ids'],'labels':c['label_ids']};batch=LEDCompleteCollator()( [f] );verify_mask_batch(batch,[f])
        checks.append({'business_id':c['business_id'],'source_shape':list(batch['input_ids'].shape),'labels_shape':list(batch['labels'].shape),'PASS':True})
    with (OUT/'evaluation_inputs.jsonl').open('x',encoding='utf-8') as f:
        for c in cases:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    candidates=[];reviews={}
    for c in cases:
        bid=c['business_id'];reviews[bid]={'human_verified':False,'positive_negative_and_specificity_checked':False,'confirmed_by':None,'notes':''}
        for i,claim in enumerate(c['shared_review_evidence']['claims']):
            candidates.append({'unit_id':bid+f'_REVIEW_CANDIDATE_{i+1}','business_id':bid,'domain':'review','text':claim['text'],
                'teacher_suggested_polarity':claim.get('polarity'),'original_teacher_evidence':claim.get('support',[]),
                'evidence':[{'marker':e['review_marker'],'review_id':e['review_id'],'quote':e['quote']} for e in claim.get('support',[])],
                'support_status':None,'importance':None,'polarity':None,'aspect':None,'restaurant_specific':None,
                'atomicity_review':'Pending split/confirm; do not assume teacher claim is one fact','human_verified':False,'confirmed_by':None})
        for cap in captions(c):
            candidates.append({'unit_id':bid+'_'+cap['caption_id']+'_CANDIDATE','business_id':bid,'domain':'visual','text':cap['text'],
                'evidence':[{'marker':cap['caption_id'],'photo_id':cap['photo_id'],'quote':cap['text']}],
                'support_status':None,'importance':None,'aspect':None,'restaurant_specific':None,
                'atomicity_review':'Pending split/confirm; source-caption support is not independent photo truth','human_verified':False,'confirmed_by':None})
    with (OUT/'reference_unit_candidates.jsonl').open('x',encoding='utf-8') as f:
        for r in candidates:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_new(OUT/'reference_freeze_template.json',{'case_inputs_sha256':sha(OUT/'evaluation_inputs.jsonl'),
        'protocol_sha256':sha(OUT/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),'researcher_confirmed_complete':False,
        'business_reviews':reviews,'units':[],'exclusion_log':[],'status':'TEMPLATE_UNVERIFIED_NOT_GROUND_TRUTH'})
    write_new(OUT/'business_annotation_template.json',{'status':'NO_NEW_OUTPUTS_OR_HUMAN_LABELS','businesses':[
        {'business_id':c['business_id'],'subset':c['subset'],'arms':{a:{'claims':[],'unit_coverage':[],'semantic_groups':[],
         'human_verified':False,'confirmed_by':None} for a in ARMS},'domain_decisions':{},'paired_rationale':None} for c in cases]})
    rng=random.Random(20261016);blinding={}
    for c in cases:
        arms=list(ARMS);rng.shuffle(arms);blinding[c['business_id']]={'left':arms[0],'right':arms[1]}
    write_new(OUT/'blinding_key_keep_separate.json',{'seed':20261016,'mapping':blinding,'annotation_use':'Keep this key hidden during future anonymized review; no new output exists now.'})
    train=PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768,root=CLOUD)
    positions={b:i for i,b in enumerate(train.business_ids)};indices=[positions[b] for b in sample['training_business_ids']]
    longest_source=max(indices,key=lambda i:len(train[i]['input_ids']));longest_label=max(indices,key=lambda i:len(train[i]['labels']))
    controls=list(dict.fromkeys([train.business_ids[longest_source],train.business_ids[longest_label],sample['selected_zero_filtered_train_ids'][0]]))
    write_new(OUT/'feasibility_cases.json',{'business_ids':controls,'selection':'Same frozen train512: longest input, longest target, first sampled zero-F; no model-output selection',
        'actual_lengths':{b:{k:len(v) for k,v in train[positions[b]].items()} for b in controls}})
    generation=read(ROOT/'outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json')['experiments']['baseline']['expected_effective_generation_config_reconstruction']
    write_new(OUT/'expected_generation_config.json',generation)
    write_new(OUT/'OFFLINE_CHECKS.json',{'status':'OFFLINE_SCOPE_INPUT_MASK_INITIALIZATION_PASS','module_probes':{a:{k:v for k,v in x.items() if k not in ('modules','trainable_tensor_shapes')} for a,x in probes.items()},
        'validation_cases':checks,'train_businesses':512,'validation_loss_businesses':200,'random':20,'targeted':4,
        'reference_candidates':len(candidates),'human_verified_reference_units':0,'new_forward_backward_generation_or_GPU_calls':0,
        'initialization_hashes_are_CPU_adapter_math_only':True,'two_epoch_orders_verified':all(sorted(o)==list(range(512)) for o in orders)})
    code=['scripts/stage_i_paired_pilot.py','scripts/run_stage_i_paired_lora_pilot_v1.py','scripts/stage_hi_led_data.py',
          'scripts/stage_i_decoding_diagnostics.py','scripts/prepare_stage_i_paired_lora_pilot_v1.py','scripts/test_stage_i_paired_lora_pilot_v1.py']
    upstream={OLD.relative_to(ROOT).as_posix():sha(OLD),
        (CLOUD/'configs/stage_i_lora_led_training_approved_v2.json').relative_to(ROOT).as_posix():sha(CLOUD/'configs/stage_i_lora_led_training_approved_v2.json'),
        (CLOUD/original['stage_h_manifest']).relative_to(ROOT).as_posix():sha(CLOUD/original['stage_h_manifest'])}
    for info in manifest['datasets'].values():
        for path in [CLOUD/info['path'],ROOT/info['stage_g_jsonl']]:upstream[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [CLOUD/'assets/models/led-base-16384.config_manifest.json',CLOUD/read(CLOUD/original['model_config_manifest'])['config_file']]+[CLOUD/tm['relative_directory']/n for n in tm['sha256']]:
        upstream[path.relative_to(ROOT).as_posix()]=sha(path)
    files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix!='.log'}
    write_new(OUT/'PREPARATION_MANIFEST.json',{'config_sha256':sha(CONFIG),'files':files,'code':{n:sha(ROOT/n) for n in code},'upstream':upstream})
    write_new(OUT/'approval_example_NOT_APPROVED.json',{'approved':False,'scopes':[],'arms':list(ARMS),'config_sha256':sha(CONFIG),
        'preparation_manifest_sha256':sha(OUT/'PREPARATION_MANIFEST.json'),'protocol_sha256':sha(OUT/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),
        'allow_backbone_download':False,'reviewed_feasibility':{},'verified_reference_units_path':None,'verified_reference_units_sha256':None,
        'note':'Template only. No GPU permission. Training needs both reviewed feasibility reports; generation additionally needs source-unit freeze.'})
    print(json.dumps({'status':'OFFLINE_PREPARATION_COMPLETE_GPU_AND_HUMAN_REFERENCE_PENDING','candidates':len(candidates),
        'modules':{a:len(v['modules']) for a,v in probes.items()},'GPU_actions':0}))
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
