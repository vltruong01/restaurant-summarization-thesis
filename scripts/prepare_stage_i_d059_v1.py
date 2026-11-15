"""Offline D059 preparation: meta inventory, small CPU adapter serialization, existing data only."""
import copy,json,os,shutil,sys,hashlib,importlib.metadata
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from stage_i_d059 import *


def main():
    import numpy as np
    import torch
    from unittest.mock import patch
    from transformers import LEDConfig,LEDForConditionalGeneration
    from peft import get_peft_model_state_dict,LoraConfig
    from safetensors.torch import load_file
    from stage_i_paired_pilot import initialized_A,canonical
    from stage_hi_led_data import PackedLEDDataset,LEDCompleteCollator
    from tokenizers import Tokenizer
    require(not torch.cuda.is_initialized(),'CUDA must remain uninitialized')
    PREP.mkdir(exist_ok=False)
    cfg=copy.deepcopy(read(ROOT/'configs/stage_i_d058_full_filtered_b_v1.json'))
    cfg.update(decision='D-059',arm='Expanded_all_linear',target_modules='all-linear',quantization=False,
        expected_modules=114,expected_trainable_parameters=1843200,stop_loss_days=7,extra_experiments_authorized=False,
        status='OFFLINE_PREPARED_GPU_EXECUTION_NOT_AUTHORIZED',GPU_execution_approved=False,training_authorized=False,
        run_root='outputs/stage_i/d059_expanded_filtered_v1',
        feasibility={'optimizer_updates':3,'microbatches':24,'source_shape':5120,'label_shape':768,
            'minimum_headroom_fraction':.15,'generation_calls':0,'replay_updates':0},
        stop_loss='Seven elapsed days from explicit execution authorization, including any subsequently approved RO/U training. No extra searches.',
        schedule_adaptation='None versus D058; fresh1250/warmup63, all5 epoch checkpoints, no best-loss reload.',
        historical_full_F_role='Practical benchmark alongside D058 epoch5; not isolated causal module comparison')
    check_config(cfg)
    if CONFIG.exists():require(read(CONFIG)==cfg,'Existing unapproved config differs; preserve and stop')
    else:write(CONFIG,cfg)
    mc=read(ROOT/'assets/models/led-base-16384/config.json')
    # A forward or CUDA lazy initialization here is a preparation failure, never a fallback.
    with patch.object(torch.cuda,'_lazy_init',side_effect=RuntimeError('Offline CUDA initialization forbidden')), \
         patch.object(LEDForConditionalGeneration,'forward',side_effect=RuntimeError('Offline model forward forbidden')):
        with torch.device('meta'):
            base=LEDForConditionalGeneration(LEDConfig(**mc));model=attach(base,cfg)
            record=inspect(model)
        require(all(p.device.type=='meta' for p in model.parameters()),'Real backbone allocation forbidden')
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        cpu_state={};init={}
        for name,m in model.named_modules():
            if hasattr(m,'lora_A') and 'default' in m.lora_A:
                av=initialized_A(m.lora_A['default'].weight.shape,cfg['seed'],name)
                bv=torch.zeros(m.lora_B['default'].weight.shape,dtype=torch.float32,device='cpu')
                cpu_state[name+'.lora_A.default.weight']=av;cpu_state[name+'.lora_B.default.weight']=bv
                init[canonical(name)]=hashlib.sha256(av.numpy().tobytes()+bv.numpy().tobytes()).hexdigest()
        require(set(cpu_state)==set(trainable),'CPU serialization scope differs')
        folder=PREP/'adapter_serialization_cpu_only'
        model.save_pretrained(folder,state_dict=cpu_state,safe_serialization=True,save_embedding_layers=False)
        loaded=load_file(folder/'adapter_model.safetensors',device='cpu')
        expected={k.replace('.default.weight','.weight'):v for k,v in cpu_state.items()}
        require(set(loaded)==set(expected) and all(torch.equal(loaded[k],v) for k,v in expected.items()),'Adapter serialization differs')
        saved=LoraConfig.from_pretrained(folder)
        # PEFT compresses full names to suffixes on save; resolve the saved config
        # against the exact LED tree instead of equating strings with full paths.
        require(saved.modules_to_save is None,'Unexpected saved full modules')
        from peft import get_peft_model
        with torch.device('meta'):
            restored_base=LEDForConditionalGeneration(LEDConfig(**mc))
            with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
                restored_model=get_peft_model(restored_base,saved)
        require(set(restored_model.targeted_module_names)==expected_modules(),'Serialized config resolves to different LED scope')
        require(set(get_peft_model_state_dict(restored_model))==set(loaded),'Restored adapter key inventory differs')
        require(saved.r==8 and saved.lora_alpha==16 and saved.lora_dropout==0. and saved.bias=='none','Saved adapter settings differ')
        meta_state=get_peft_model_state_dict(model)
        require(set(meta_state)==set(loaded) and all(meta_state[k].shape==loaded[k].shape for k in loaded),'PEFT serialized key geometry differs')
    record.update(meta_only=True,backbone_weights_loaded=False,forward_backward_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        CPU_adapter_serialization_exact=True,serialization_tensors=len(loaded),quantized_dispatchers_disabled=True,
        reason='Optional bitsandbytes backend is unused and fails on this CPU-only environment; explicit torch.nn.Linear dispatch only, in prep and runtime.')
    write(PREP/'MODULE_INVENTORY.json',record)
    write(PREP/'INITIALIZATION_HASHES.json',{'seed':42,'algorithm':cfg['initialization'],'per_module_sha256':init,
        'small_CPU_adapter_tensors_only':True,'backbone_weights_loaded':False})
    write(PREP/'OFFLINE_ENVIRONMENT_NOTE.json',{'initial_probe':'Optional bitsandbytes/Triton import failed with GPU hidden; no forward/backward occurred.',
        'failure_observed':'bitsandbytes CUDA-spec IndexError then triton CUDA-properties KeyError:0',
        'resolution':'Explicitly disable only PEFT bnb4/8 dispatchers for the unquantized torch.nn.Linear experiment. No package install or Codex config change.',
        'CUDA_initialized_after_successful_guarded_probe':torch.cuda.is_initialized()})
    previous=read(ROOT/'outputs/stage_i/d058_preparation_v1/PREFLIGHT_MANIFEST.json')
    files=dict(previous['files'])
    for rel,h in files.items():require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Historical binding changed: '+rel)
    for name in ('EPOCH_ORDERS.json','EVALUATION_MANIFEST.json'):
        shutil.copyfile(ROOT/'outputs/stage_i/d058_preparation_v1'/name,PREP/name)
    h=read(ROOT/cfg['stage_h_manifest']);datasets={};stats={};zero={};rows={};tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    for split,n in [('train',2000),('validation',200)]:
        info=h['datasets'][split+'_filtered'];d=PackedLEDDataset(info,5000,768);datasets[split]=d
        rows[split]=[json.loads(line) for line in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
        require(len(d)==n and [r['business_id'] for r in rows[split]]==d.business_ids,'Frozen ID/order mismatch')
        require(all(r['split']==split and r['condition']=='filtered' for r in rows[split]),'Foreign split/condition')
        for i,r in enumerate(rows[split]):
            f=d[i]
            require(tok.encode(r['source']).ids==f['input_ids'] and tok.encode(r['target']).ids==f['labels'],'Source/target token alignment differs')
            require(hashlib.sha256(r['source'].encode()).hexdigest()==r['source_sha256'],'Source text hash differs')
        il=np.diff(d.pack['input_offsets']);ll=np.diff(d.pack['label_offsets'])
        zero[split]=[r['business_id'] for r in rows[split] if not r['selected_photo_ids']]
        stats[split]={'businesses':n,'source_mean':float(il.mean()),'source_max':int(il.max()),
            'label_mean':float(ll.mean()),'label_max':int(ll.max()),'zero_filtered_caption_businesses':len(zero[split])}
    require(not set(datasets['train'].business_ids)&set(datasets['validation'].business_ids),'Split overlap')
    require(len(zero['train'])==9 and len(zero['validation'])==1,'Rare zero-caption cohort changed')
    orders=read(PREP/'EPOCH_ORDERS.json');require(orders['training_business_ids']==datasets['train'].business_ids,'Training order changed')
    cases=[json.loads(x) for x in (BASE_PREP/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation=read(PREP/'EVALUATION_MANIFEST.json')
    require([c['business_id'] for c in cases]==evaluation['business_ids'],'Fixed24 changed')
    for c in cases:
        i=datasets['validation'].business_ids.index(c['business_id']);f=datasets['validation'][i]
        require(f['input_ids']==c['input_ids'] and f['labels']==c['label_ids'],'Fixed evaluation alignment differs')
    trainrows=rows['train'];case_ids=[];case_roles={}
    choices=[('longest_source',max(trainrows,key=lambda r:(r['source_tokens'],r['business_id']))['business_id']),
        ('longest_target',max(trainrows,key=lambda r:(r['target_tokens'],r['business_id']))['business_id']),
        ('zero_filtered_captions',sorted(zero['train'])[0])]
    for role,b in choices:
        if b not in case_ids:case_ids.append(b)
        case_roles.setdefault(b,[]).append(role)
    collate=LEDCompleteCollator(5000,768,1024,pad_source_to=5120,pad_label_to=768)
    masks=[]
    from stage_i_paired_pilot import verify_mask_batch
    for b in case_ids:
        f=datasets['train'][datasets['train'].business_ids.index(b)];batch=collate([f]);verify_mask_batch(batch,[f])
        masks.append({'business_id':b,'roles':case_roles[b],'actual_source':len(f['input_ids']),'actual_target':len(f['labels']),
            'source_shape':list(batch['input_ids'].shape),'label_shape':list(batch['labels'].shape),'global_tokens':int(batch['global_attention_mask'].sum())})
    write(PREP/'FEASIBILITY_CASES.json',{'business_ids':case_ids,'cases':masks,'selection':'Deterministic train-only length maxima + first sorted zero-caption ID. No synthetic or altered tokens.',
        'stress_note':'5120/768 padding tests tensor dimensions; no claim that masked padding simulates 768 supervised tokens.'})
    comparisons=read(ROOT/'outputs/stage_i/d058_result_report_v1/COMPARISON_RECORDS.json')
    comparisons=[r for r in comparisons if r['arm'] in ('historical_full_F','B_full_epoch5')]
    require(len(comparisons)==48,'Missing existing benchmark predictions')
    require({r['business_id'] for r in comparisons}==set(evaluation['business_ids']),'Benchmark ID mismatch')
    write(PREP/'BENCHMARKS_UNCHANGED.json',comparisons)
    write(PREP/'DATA_AND_MASK_READINESS.json',{'status':'PASS_OFFLINE','stats':stats,'exact_sources_targets_checked':2200,
        'fixed_validation_cases':24,'random20':len(evaluation['random20']),'targeted4':len(evaluation['targeted4']),
        'zero_business_ids':zero,'source_label_truncation':False,'masks':masks,'held_out_contents_accessed':False,
        'pretrained_weights_loaded':False,'model_forward_backward_calls':0,'GPU_calls':0,'Teacher_calls':0})
    write(PREP/'EXECUTION_BUDGET.json',{**budget(cfg),'feasibility_optimizer_updates':3,'feasibility_microbatches':24,
        'feasibility_generation_calls':0,'feasibility_replay_updates':0,'initial_training_runs':1,
        'RO_U_runs_authorized':0,'seven_day_window_start':'Future explicit execution authorization UTC',
        'seven_day_window_includes':'Feasibility + F train/eval + researcher review + any separately approved RO/U training',
        'conditional_all_three_training_updates_max':3750,'initial_reference_or_teacher_regeneration':0})
    needed=['scripts/stage_i_d059.py','scripts/run_stage_i_d059_v1.py','scripts/prepare_stage_i_d059_v1.py','tests/test_stage_i_d059.py',
        'specs/009m_d059_expanded_lora_viability.md','configs/stage_i_d059_expanded_lora_v1.json',
        'outputs/stage_i/d058_result_report_v1/D058_REPORT.md','outputs/stage_i/d058_result_report_v1/D058_COMPLETION_AND_REVIEW_STATUS.json',
        'outputs/stage_i/d058_result_report_v1/COMPARISON_RECORDS.json',
        'outputs/stage_i/d058_full_filtered_B_v1/train/checkpoint-1250/adapter_model.safetensors',
        'outputs/stage_i/d058_full_filtered_B_v1/train/checkpoint-1250/adapter_config.json']
    # Bind installed implementation used to resolve inventory; relocation needs explicit equivalent-runtime rebind.
    import peft.tuners.tuners_utils,peft.tuners.lora.model,transformers.models.led.modeling_led
    for mod in (peft.tuners.tuners_utils,peft.tuners.lora.model,transformers.models.led.modeling_led):
        p=Path(mod.__file__);files[str(p)]=sha(p)
    for rel in needed:files[rel]=sha(ROOT/rel)
    for p in PREP.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'decision':'D-059','files':files,'config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER),
        'offline_only':True,'meta_only_backbone':True,'GPU_execution_approved':False})
    write(PREP/'GPU_AUTHORIZATION_TEMPLATE.json',{'decision':'D-059','GPU_execution_approved':False,'approval_source':None,
        'researcher_explicit_authorization':None,'scopes':['feasibility','train','evaluate'],'conditional_train_after_feasibility_pass':True,'config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER),
        'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'feasibility_optimizer_updates_max':3,
        'max_optimizer_steps':1250,'generation_outputs_max':48,'execution_authorized_at_utc':None,'improvement_deadline_utc':None,
        'approved_feasibility_report_sha256':None,'RO_U_training_authorized':False,'held_out_access':False,'extra_experiments_authorized':False})
    preflight();require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    print(json.dumps({'status':'OFFLINE_PASS_GPU_NOT_AUTHORIZED','modules':record['module_count'],'trainable_parameters':record['trainable_parameters'],
        'bound_files':len(files),'config_sha256':sha(CONFIG),'preflight_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER)},indent=2))


if __name__=='__main__':main()
