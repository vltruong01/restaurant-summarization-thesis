"""Offline LED-large architecture/tokenizer verification. NO model weight loading.

Public pinned metadata is already downloaded. Only meta modules and small CPU
adapter tensors are constructed. No forward/backward/generation/optimizer/GPU.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_preparation_v1'
CONFIG=ROOT/'configs/stage_i_led_large_proposal_v1.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def expected_modules():
    enc={f'led.encoder.layers.{i}.self_attn.longformer_self_attn.{p}' for i in range(12)
         for p in ('query','key','value','query_global','key_global','value_global')}
    enc|={f'led.encoder.layers.{i}.self_attn.output' for i in range(12)}
    dec={f'led.decoder.layers.{i}.{a}.{p}' for i in range(12) for a in ('self_attn','encoder_attn')
         for p in ('q_proj','k_proj','v_proj','out_proj')}
    ffn={f'led.{s}.layers.{i}.{p}' for s in ('encoder','decoder') for i in range(12) for p in ('fc1','fc2')}
    return enc|dec|ffn


def validate_architecture(c):
    expected={'model_type':'led','d_model':1024,'encoder_layers':12,'decoder_layers':12,
        'encoder_attention_heads':16,'decoder_attention_heads':16,'encoder_ffn_dim':4096,'decoder_ffn_dim':4096,
        'vocab_size':50265,'max_encoder_position_embeddings':16384,'max_decoder_position_embeddings':1024,
        'bos_token_id':0,'pad_token_id':1,'eos_token_id':2,'decoder_start_token_id':2,'attention_window':[1024]*12}
    for k,v in expected.items():require(c[k]==v,'Architecture differs: '+k)
    return expected


def main():
    require(not (PREP/'META_LORA_INVENTORY.json').exists(),'Existing offline evidence must not be overwritten')
    metadata=read(PREP/'REMOTE_PROVENANCE.json')
    require(metadata['official']['revision']=='cd59d11c3528415b7dda4dfc95cc8f138aceda2e','Official revision drift')
    require(metadata['amazon']['revision']=='4fdc41ed9bd4fbebe4fd6de131af1f27181f51ea','Amazon revision drift')
    for role,m in metadata.items():
        for name,entry in m['files'].items():require(sha(PREP/'remote_metadata'/role/name)==entry['sha256'],'Remote metadata changed')
    primary=read(PREP/'remote_metadata/official/config.json')
    secondary=read(PREP/'remote_metadata/amazon/config.json')
    architecture=validate_architecture(primary);require(validate_architecture(secondary)==architecture,'Different candidate architecture')
    import torch
    import numpy as np
    from transformers import LEDConfig,LEDForConditionalGeneration,LEDTokenizerFast,GenerationConfig
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from safetensors.torch import load_file
    from unittest.mock import patch
    from stage_i_paired_pilot import initialized_A,canonical
    require(not torch.cuda.is_initialized(),'Offline CUDA must remain uninitialized')
    deny=RuntimeError('LED-large preparation forbids model execution or GPU initialization')
    with patch.object(torch.cuda,'_lazy_init',side_effect=deny),patch.object(LEDForConditionalGeneration,'forward',side_effect=deny),patch.object(LEDForConditionalGeneration,'generate',side_effect=deny):
        with torch.device('meta'):
            base=LEDForConditionalGeneration(LEDConfig(**primary))
            unique_parameters=sum(p.numel() for p in base.parameters())
            buffer_elements=sum(b.numel() for b in base.buffers())
            linear={n:m for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)}
            require(set(linear)==expected_modules()|{'lm_head'},'Installed LED linear inventory differs')
            require(base.get_output_embeddings() is linear['lm_head'] and base.get_input_embeddings().weight is linear['lm_head'].weight,'Tied head/embedding assumption differs')
            lc=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='SEQ_2_SEQ_LM',target_modules='all-linear')
            with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
                model=get_peft_model(base,lc)
        require(set(model.targeted_module_names)==expected_modules(),'Actual PEFT attachment differs')
        require(all(p.device.type=='meta' for p in model.parameters()),'Full model allocation forbidden')
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        require(len(trainable)==456 and sum(p.numel() for p in trainable.values())==4915200,'Actual LoRA tensor/parameter count differs')
        require(all('.lora_A.' in n or '.lora_B.' in n for n in trainable),'Unexpected unfrozen pretrained parameter')
        modules=[];state={}
        for name,m in model.named_modules():
            if canonical(name) not in expected_modules():continue
            modules.append({'name':canonical(name),'in_features':m.in_features,'out_features':m.out_features,
                'A_shape':list(m.lora_A['default'].weight.shape),'B_shape':list(m.lora_B['default'].weight.shape),
                'trainable_parameters':m.lora_A['default'].weight.numel()+m.lora_B['default'].weight.numel()})
            state[name+'.lora_A.default.weight']=initialized_A(m.lora_A['default'].weight.shape,42,name)
            state[name+'.lora_B.default.weight']=torch.zeros(m.lora_B['default'].weight.shape,dtype=torch.float32,device='cpu')
        folder=PREP/'untrained_CPU_serialization_probe'
        model.save_pretrained(folder,state_dict=state,safe_serialization=True,save_embedding_layers=False)
        saved=load_file(folder/'adapter_model.safetensors',device='cpu')
        expected_state={k.replace('.default.weight','.weight'):v for k,v in state.items()}
        require(set(saved)==set(expected_state) and all(torch.equal(saved[k],v) for k,v in expected_state.items()),'CPU adapter serialization mismatch')
        reloaded=LoraConfig.from_pretrained(folder)
        with torch.device('meta'):
            other=LEDForConditionalGeneration(LEDConfig(**primary))
            with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
                other=get_peft_model(other,reloaded)
        require(set(other.targeted_module_names)==expected_modules(),'Serialized target scope does not reload')
        require(set(get_peft_model_state_dict(other))==set(saved),'Restored adapter key set differs')
    write(PREP/'META_LORA_INVENTORY.json',{'status':'PASS_META_ATTACHMENT_AND_CPU_ADAPTER_SERIALIZATION_ONLY',
        'modules':sorted(modules,key=lambda x:x['name']),'module_count':228,'trainable_parameters':4915200,
        'trainable_tensors':456,'unique_backbone_parameters':unique_parameters,'registered_buffer_elements':buffer_elements,
        'encoder_attention_projections':84,'decoder_attention_projections':96,'feed_forward_projections':48,
        'encoder_local_QKV':36,'encoder_global_QKV':36,'encoder_attention_output':12,
        'excluded':['lm_head','shared/tied/token/position embeddings'],'untrained_adapter_bytes':(folder/'adapter_model.safetensors').stat().st_size,
        'adapter_serialization_exact':True,'reloaded_meta_scope_exact':True,'full_checkpoint_weights_loaded':False,
        'model_forward_backward_generation':0,'GPU_operations':0,'actual_gradient_flow':'NOT_TESTED',
        'CPU_probe_is_not_an_accepted_checkpoint_or_training_initialization':True})
    print('Meta inventory228modules/4915200LoRAparameters verified. No model execution.',flush=True)
    tok=LEDTokenizerFast.from_pretrained(PREP/'remote_metadata/official',local_files_only=True)
    amazon=LEDTokenizerFast.from_pretrained(PREP/'remote_metadata/amazon',local_files_only=True)
    original_vocab=read(ROOT/'assets/tokenizers/led-base-16384/vocab.json')
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
