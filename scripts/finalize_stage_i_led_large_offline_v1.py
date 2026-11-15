"""Continue offline tokenizer/config audit after preserved raw-tokenizer failure.
No model construction/forward/weights/GPU. Initial meta evidence is reused exactly.
"""
from prepare_stage_i_led_large_v1 import *

def main():
    import numpy as np
    import torch
    from transformers import LEDTokenizerFast,LEDConfig,GenerationConfig
    require(not CONFIG.exists(),'Preserve already finalized config')
    inventory=read(PREP/'META_LORA_INVENTORY.json')
    require(inventory['module_count']==228 and inventory['trainable_parameters']==4915200 and inventory['adapter_serialization_exact'],'Meta verification not accepted')
    unique_parameters=inventory['unique_backbone_parameters'];buffer_elements=inventory['registered_buffer_elements']
    metadata=read(PREP/'REMOTE_PROVENANCE.json');primary=read(PREP/'remote_metadata/official/config.json')
    tok=LEDTokenizerFast.from_pretrained(PREP/'remote_metadata/official',local_files_only=True)
    amazon=LEDTokenizerFast.from_pretrained(PREP/'remote_metadata/amazon',local_files_only=True)
    original_vocab=read(ROOT/'assets/tokenizers/led-base-16384/vocab.json')
    raw_policies={role:{'truncation':t.backend_tokenizer.truncation,'padding':t.backend_tokenizer.padding} for role,t in [('official',tok),('amazon',amazon)]}
    # Explicit IN-MEMORY comparison policy, never alter downloaded or frozen files.
    for t in (tok,amazon):
        t.backend_tokenizer.no_truncation();t.backend_tokenizer.no_padding()
    write(PREP/'EXPLICIT_NO_TRUNCATION_POLICY.json',{'raw_candidate_policies':raw_policies,'comparison_policy':'explicit no_truncation/no_padding for both candidates; full originals unchanged','raw_Amazon_default_was_not_drop_in_compatible':True,'frozen_training_tokenizer_preserved':True})
    require(tok.get_vocab()==amazon.get_vocab()==original_vocab,'Vocabulary/ID mappings differ')
    require(tok.bos_token_id==0 and tok.eos_token_id==2 and tok.pad_token_id==1,'Special token IDs differ')
    old=read(ROOT/'configs/stage_i_d059_expanded_lora_v2.json');hm=read(ROOT/old['stage_h_manifest'])
    stats={};encodes=0;all_ids={}
    for split,count in [('train',2000),('validation',200)]:
        for cond in ('reviews_only','unfiltered','filtered'):
            key=split+'_'+cond;info=hm['datasets'][key]
            require(sha(ROOT/info['path'])==info['sha256'] and sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256'],'Frozen StageH/G changed')
            rows=[json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
            require(len(rows)==count and all(r['split']==split and r['condition']==cond for r in rows),'Split/condition mismatch')
            with np.load(ROOT/info['path'],allow_pickle=False) as z:pack={k:z[k] for k in z.files}
            require(pack['business_ids'].tolist()==[r['business_id'] for r in rows],'Input-target ID mapping differs')
            all_ids[key]=pack['business_ids'].tolist()
            for offset in range(0,len(rows),32):
                chunk=rows[offset:offset+32];texts=[s for r in chunk for s in (r['source'],r['target'])]
                a=tok.backend_tokenizer.encode_batch(texts);b=amazon.backend_tokenizer.encode_batch(texts)
                for j,r in enumerate(chunk):
                    i=offset+j
                    for ix,ids,offs,limit in [(2*j,'input_ids','input_offsets',5000),(2*j+1,'labels','label_offsets',768)]:
                        gold=pack[ids][pack[offs][i]:pack[offs][i+1]].tolist()
                        require(a[ix].ids==b[ix].ids==gold,'Candidate tokenizer IDs differ; no dataset rewrite allowed')
                        require(len(gold)<=limit and gold[0]==0 and gold[-1]==2,'Budget/BOS/EOS mismatch')
                        encodes+=2
            stats[key]={'businesses':count,'max_source':int(np.diff(pack['input_offsets']).max()),
                'max_target':int(np.diff(pack['label_offsets']).max()),'zero_captions':int((pack['caption_counts']==0).sum())}
        require(all_ids[split+'_reviews_only']==all_ids[split+'_unfiltered']==all_ids[split+'_filtered'],'Cross-condition alignment differs')
        print('Tokenizer/packed labels exact for'+split+' RO/U/F.',flush=True)
    require(not set(all_ids['train_filtered'])&set(all_ids['validation_filtered']),'Split leakage')
    require(stats['train_filtered']['zero_captions']==9 and stats['validation_filtered']['zero_captions']==1,'Zero-caption cohort changed')
    write(PREP/'TOKENIZER_DATA_COMPATIBILITY.json',{'status':'PASS_ALL6600TRAIN_VALIDATION_PAIRS','business_targets':6600,
        'candidate_source_or_label_encodes':encodes,'tokenizer_candidates':2,'exact_against_frozen_packs':True,
        'stats':stats,'selected_training_tokenizer':'existing frozen assets/tokenizers/led-base-16384; same IDs proven',
        'new_tokenization_artifacts_or_truncation':False,'held_out_content_access':False,'test_teacher_targets_created':0})
    cfg={k:old[k] for k in ('seed','lora','optimization','generation','budgets','runtime_pins','stage_h_manifest','final_references','training','evaluation_epochs')}
    cfg.update(experiment='LED_large_filtered_viability_proposal_v1',model_id=metadata['official']['repo'],
        revision=metadata['official']['revision'],recommended_initialization='official_LED_large_fresh_LoRA_only',
        secondary_candidate_assessed_not_selected=metadata['amazon'],
        target_modules='all-linear',expected_modules=228,expected_trainable_parameters=4915200,
        source_tokenizer='assets/tokenizers/led-base-16384',source_limit=5000,label_limit=768,
        global_attention='first_nonpadding_BOS_only; identical RO/U/F/Shuffled policy',
        loss='standard_positive_CE_only; no_UL_or_separate_scaled_component_probes',
        initialization='fresh named seeded KaimingA/zeroB; no LED-base adapter or Amazon state import',
        checkpoint_policy='retain_all5_epochs; no_early_stop_or_best_loss_reload; quality_inspect_epochs3_and5_only',
        checkpoint_selection='researcher_select_common_epoch3_or5_from_F_then_freeze_same_epoch_for_matched_RO_U_F',
        F_evaluation_outputs=48,technical_gate_generation_outputs_max=2,
        feasibility={'optimizer_updates_max':3,'microbatches_max':24,'source_padded_shape':5120,'label_padded_shape':768,
            'beam4_full_limit_train_input_traces_max':2,'generation_max_length':768,'minimum_memory_headroom':0.15,
            'checkpoint_save_restore':True,'replay_updates':0,'discard_gate_states_before_fresh_training':True},
        matched_family={'conditions':['reviews_only','unfiltered','filtered'],'same_initial_backbone':True,
            'same_optimization_and_epoch_selection':True,'Shuffled_reuses_exact_F_adapter':True,
            'RO_U_training_needs_separate_approval':True,'historical_base_family_primary_until_complete':True},
        timeline={'researcher_defense_estimate':'more than three weeks from2026-10-19',
            'planning_earliest_defense_date':'2026-11-09','confirmed_defense_date':None,
            'minimum_protected_final_phase_days':14,'proposed_absolute_model_family_freeze_utc':'2026-10-26T00:00:00+00:00',
            'F_training_quality_decision_budget_hours_max':48,'entire_large_family_must_finish_before_freeze':True,
            'GPU_approval_timestamp':None,'execution_deadline_approved':False},
        GPU_execution_approved=False,training_authorized=False,RO_U_authorized=False,
        held_out_access=False,Teacher_calls=False,scientific_amendment_approved=False,
        unlikelihood_status='CLOSED_NUMERICAL_FEASIBILITY_FAILURE_QUALITY_NOT_EVALUATED_NO_PATCH_NO_RETRY',
        D051_strict_replay='FAIL_UNCHANGED',
        run_root='outputs/stage_i/led_large_filtered_viability_v1')
    write(CONFIG,cfg)
    gen=GenerationConfig.from_model_config(LEDConfig(**primary))
    for k,v in cfg['generation'].items():setattr(gen,k,v)
    gen.use_cache=True
    write(PREP/'EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',gen.to_dict())
    require({k:importlib.metadata.version(k) for k in cfg['runtime_pins']}==cfg['runtime_pins'],'Runtime pins differ')
    previous=read(ROOT/'outputs/stage_i/unlikelihood_preparation_v2/PREFLIGHT_MANIFEST.json')
    for rel,h in previous['files'].items():require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Historical frozen binding changed')
    write(PREP/'OFFLINE_PREFLIGHT_REPORT.json',{'status':'PASS_OFFLINE_STRUCTURE_TOKENIZATION_ONLY',
        'prior_frozen_bindings_unchanged':len(previous['files']),'runtime_pins':cfg['runtime_pins'],
        'meta_models_only':True,'backbone_tensor_values_loaded':False,'candidate_checkpoint_binary_hash_verified':False,
        'checkpoint_weight_LFS_hashes_pinned_from_Hub':True,'untrained_CPU_adapter_serialization_exact':True,
        'model_forward_backward_generation':0,'optimizer_steps':0,'GPU_initialized':torch.cuda.is_initialized(),
        'held_out_content_access':False,'new_teacher_calls':0,'readiness':'RESEARCHER_PROTOCOL_REVIEW; GPU_FEASIBILITY_AND_BINARY_VERIFICATION_PENDING'})
    print(json.dumps({'module_count':228,'trainable_parameters':4915200,'unique_backbone_parameters':unique_parameters,
        'registered_buffer_elements':buffer_elements,'candidate_encodes':encodes,'historical_bindings_verified':len(previous['files']),
        'GPU_operations':0,'model_forwards':0}))


if __name__=='__main__':main()
