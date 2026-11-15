"""Prepare D058 using existing frozen data; CPU randperm only, no model loading."""
import collections,copy,importlib.metadata,os,shutil,subprocess,sys
import numpy as np
from tokenizers import Tokenizer
from stage_i_d058 import *

def main():
    PREP.mkdir(exist_ok=False)
    old=read(ROOT/'configs/stage_i_paired_lora_pilot_v1.json');previous=read(ROOT/'configs/stage_i_d056_minimal_diagnostic_v2.json')
    cfg={k:copy.deepcopy(old[k]) for k in ['condition','model_id','revision','seed','lora','optimization','generation','budgets']}
    cfg.update(decision='D-058',status='OFFLINE_PREPARED_AWAITING_RESEARCHER_APPROVAL',arm='B_encoder_decoder',
        fresh_start=True,resume_short_pilot=False,GPU_execution_approved=False,training_authorized=False,
        training={'businesses':2000,'epochs':5,'steps':1250,'steps_per_epoch':250,'warmup_steps':63},
        evaluation_epochs=[3,5],checkpoint_selection='researcher_quality_review_of_epochs3_and5_only',load_best_model_at_end=False,
        run_root='outputs/stage_i/d058_full_filtered_B_v1',stage_h_manifest='outputs/stage_h/led_dataset_v1/manifest.json',
        final_references='outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json',
        runtime_pins=previous['runtime_pins'],D051_strict_replay_status='FAIL_UNCHANGED',
        held_out_access=False,Teacher_calls_authorized=False,additional_experiments_authorized=False,automatic_fallback=False,
        initialization=old['initialization'],historical_full_F_role='Practical usability benchmark; not matched causal architecture control',
        stop_loss='One bounded attempt; no convincing substantial quality improvement means stop LED-LoRA optimization, no extra numerical diagnostics.',
        schedule_adaptation='Fresh linear1250/warmup63, same ratio0.05. No completed short-schedule resume. Disable loss early-stop; keep5 checkpoints; no best-loss reload.')
    cfg['optimization'].update(num_train_epochs=5,early_stopping_patience=None,save_total_limit=5)
    check_config(cfg);write(CONFIG,cfg)
    h=read(ROOT/cfg['stage_h_manifest']);packs={};files={cfg['stage_h_manifest']:sha(ROOT/cfg['stage_h_manifest'])};stats={}
    for split in ['train','validation']:
        info=h['datasets'][split+'_filtered'];require(sha(ROOT/info['path'])==info['sha256'],'Frozen NPZ mismatch')
        require(sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256'],'Original Teacher JSONL mismatch')
        files[info['path']]=info['sha256'];files[info['stage_g_jsonl']]=info['stage_g_jsonl_sha256']
        with np.load(ROOT/info['path'],allow_pickle=False) as data:packs[split]={k:data[k].copy() for k in data.files}
        p=packs[split];n=2000 if split=='train' else 200;require(len(p['business_ids'])==len(set(p['business_ids']))==n,'Dataset IDs/count invalid')
        il=np.diff(p['input_offsets']);ll=np.diff(p['label_offsets'])
        require(np.all((il>0)&(il<=5000)) and np.all((ll>0)&(ll<=768)),'Content overflow; truncation forbidden')
        require(np.all(p['input_ids'][p['input_offsets'][:-1]]==0) and np.all(p['input_ids'][p['input_offsets'][1:]-1]==2),'Incomplete sources')
        require(np.all(p['labels'][p['label_offsets'][:-1]]==0) and np.all(p['labels'][p['label_offsets'][1:]-1]==2),'Incomplete targets')
        stats[split]={'businesses':n,'source_min':int(il.min()),'source_mean':float(il.mean()),'source_max':int(il.max()),
            'label_min':int(ll.min()),'label_mean':float(ll.mean()),'label_max':int(ll.max())}
    require(not set(packs['train']['business_ids']) & set(packs['validation']['business_ids']),'Split overlap')
    cases=[json.loads(x) for x in (BASE_PREP/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    ec=read(ROOT/'configs/stage_i_d053_evaluation_v1.json');require([x['business_id'] for x in cases]==ec['random_business_ids']+ec['targeted_business_ids'],'Fixed24 changed')
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    for c in cases:
        require(c['split']=='validation' and c['condition']=='filtered','Foreign evaluation case')
        require(tok.encode(c['source']).ids==c['input_ids'] and tok.encode(c['target']).ids==c['label_ids'],'Exact tokenization mismatch')
        p=packs['validation'];i=p['business_ids'].tolist().index(c['business_id']);io=p['input_offsets'];lo=p['label_offsets']
        require(p['input_ids'][io[i]:io[i+1]].tolist()==c['input_ids'] and p['labels'][lo[i]:lo[i+1]].tolist()==c['label_ids'],'Packed evaluation alignment differs')
    hist=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_i_chatgpt_review_bundle/validation/filtered/validation_predictions.jsonl'
    baseline={x['business_id']:x for x in [json.loads(s) for s in hist.read_text(encoding='utf-8').splitlines()]}
    for c in cases:require(baseline[c['business_id']]['text']==c['baseline_prediction']['text'] and baseline[c['business_id']]['token_ids']==c['baseline_prediction']['token_ids'],'Saved baseline mismatch')
    write(PREP/'EVALUATION_MANIFEST.json',{'business_ids':[c['business_id'] for c in cases],'random20':ec['random_business_ids'],'targeted4':ec['targeted_business_ids'],
        'zero_caption_business_ids':[c['business_id'] for c in cases if not c['selected_photo_ids']],
        'evaluation_input_sha256':sha(BASE_PREP/'evaluation_inputs.jsonl'),'reference_sha256':sha(ROOT/cfg['final_references']),
        'historical_predictions_path':hist.relative_to(ROOT).as_posix(),'historical_predictions_sha256':sha(hist),
        'new_generation_outputs':48,'prior_predictions_reused_without_regeneration':24})
    # Same accepted torch.randperm CPU algorithm, not a model/meta/forward check.
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    code="import sys,json,torch;sys.path.insert(0,'scripts');from stage_i_paired_pilot import epoch_orders;print(json.dumps({'indices':epoch_orders(2000,42,5),'cuda_initialized':torch.cuda.is_initialized()}))"
    order=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
    write(PREP/'CPU_ORDER_GENERATION.log',order.stdout+order.stderr);require(order.returncode==0,'CPU order generation failed')
    generated=json.loads(order.stdout);require(generated['cuda_initialized'] is False,'Unexpected CUDA initialization')
    require(len(generated['indices'])==5 and all(sorted(x)==list(range(2000)) for x in generated['indices']),'Invalid full-data permutations')
    write(PREP/'EPOCH_ORDERS.json',{'algorithm':'Existing stage_i_paired_pilot.epoch_orders CPU torch.randperm(seed42+epoch)',
        'training_business_ids':packs['train']['business_ids'].tolist(),'indices':generated['indices'],'GPU_execution':False})
    # Reuse the accepted final-reference validator; no reference or annotation changes.
    from check_stage_i_d053_final_references_v2 import check
    refcheck=check(ROOT);require(refcheck['valid'],'Approved references fail existing validator')
    write(PREP/'EXISTING_REFERENCE_VALIDATOR_RESULT.json',refcheck)
    versions={k:importlib.metadata.version(k) for k in cfg['runtime_pins']};require(versions==cfg['runtime_pins'],'Runtime version mismatch')
    needed=['configs/stage_i_paired_lora_pilot_v1.json','configs/stage_i_d053_evaluation_v1.json',cfg['final_references'],
        'outputs/stage_i/d053_final_references_v1/REFERENCE_FREEZE_MANIFEST.json',
        'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json',
        'outputs/stage_i/d052_paired_pilot_v1/B_encoder_decoder/observation_completion.json',
        'outputs/stage_i/d057_bounded_gpu_diagnostic_v1/D057_SUMMARY.json',
        'outputs/stage_i/d057_bounded_gpu_diagnostic_v1/VERIFIED_OBSERVATIONS_AND_HYPOTHESES.json',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/initialization_hashes.json',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/STAGE_I_PILOT_EVALUATION_PROTOCOL.md',
        'assets/models/led-base-16384/config.json','assets/tokenizers/led-base-16384.manifest.json',
        'scripts/stage_i_d058.py','scripts/run_stage_i_d058_v1.py','scripts/prepare_stage_i_d058_v1.py',
        'scripts/stage_i_paired_pilot.py','scripts/run_stage_i_paired_lora_pilot_v1.py','scripts/run_stage_i_d050_feasibility_v1.py',
        'scripts/stage_hi_led_data.py','scripts/stage_i_d053_evaluation.py','scripts/stage_i_decoding_diagnostics.py',
        'tests/test_stage_i_d058.py','specs/009l_d058_full_filtered_b_viability.md']
    for rel in needed:files[rel]=sha(ROOT/rel)
    files[hist.relative_to(ROOT).as_posix()]=sha(hist)
    tm=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    for name,digest in tm['sha256'].items():
        p=ROOT/tm['relative_directory']/name;require(sha(p)==digest,'Tokenizer asset mismatch');files[p.relative_to(ROOT).as_posix()]=digest
    # Preserve existing checkpoint bindings and pinned pretrained bytes without loading tensors.
    for row in previous['adapter_states'].values():require(sha(ROOT/row['path'])==row['sha256'],'Historical adapter changed');files[row['path']]=row['sha256']
    cached=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/PRE_EXECUTION_INTEGRITY.json')['cached_backbone_files']
    for name,row in cached.items():
        path=Path(previous['local_backbone_snapshot'])/name;require(sha(path)==row['sha256'],'Cached backbone bytes differ');files[str(path)]=row['sha256']
    shutil.copyfile(ROOT/'specs/009l_d058_full_filtered_b_viability.md',PREP/'D058_PLAN_AND_AUTHORIZATION.md')
    for p in [CONFIG,PREP/'EPOCH_ORDERS.json',PREP/'EVALUATION_MANIFEST.json',PREP/'D058_PLAN_AND_AUTHORIZATION.md']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'decision':'D-058','status':'OFFLINE_DATA_AND_BINDINGS_PASS_GPU_UNTESTED','files':files,
        'config_sha256':sha(CONFIG),'runner_sha256':sha(ROOT/'scripts/run_stage_i_d058_v1.py'),'runtime_pins':versions,
        'model_loads':0,'GPU_execution':False,'training_updates':0,'new_generation_outputs':0,'Teacher_calls':0,'held_out_access':False})
    write(PREP/'EXECUTION_BUDGET.json',budget(cfg))
    write(PREP/'DATASET_AND_REFERENCE_READINESS.json',{'status':'PASS_OFFLINE','dataset_statistics':stats,'split_overlap':0,
        'fixed_eval_random':20,'fixed_eval_targeted':4,'zero_caption_evaluation':[c['business_id'] for c in cases if not c['selected_photo_ids']],
        'exact_eval_source_target_alignment_checked':24,'new_targets_or_predictions':0,'reference_validation':refcheck,
        'no_new_source_annotation':True,'CPU_randperm_only_no_model_or_GPU':True,'D051_strict_replay':'FAIL_UNCHANGED'})
    write(PREP/'GPU_AUTHORIZATION_TEMPLATE.json',{'decision':'D-058','GPU_execution_approved':False,'training_authorized':False,
        'approval_source':None,'researcher_explicit_authorization':None,'config_sha256':sha(CONFIG),
        'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(ROOT/'scripts/run_stage_i_d058_v1.py'),
        'max_optimizer_steps':1250,'generation_outputs_max':48,'held_out_access':False,'extra_experiments_authorized':False})
    preflight()
    print(json.dumps({'status':'D058_OFFLINE_PREPARED_NO_GPU_APPROVAL','bound_files':len(files),'budget':budget(cfg),
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(ROOT/'scripts/run_stage_i_d058_v1.py')},indent=2))

if __name__=='__main__':main()
