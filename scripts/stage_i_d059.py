"""D059 Expanded-LoRA contracts. Importing this module performs no model/GPU work."""
import copy
import importlib.metadata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from stage_i_d058 import ROOT, BASE_PREP, read, write, sha, require, training_arguments, budget

CONFIG = ROOT/'configs/stage_i_d059_expanded_lora_v1.json'
PREP = ROOT/'outputs/stage_i/d059_preparation_v1'
RUNNER = ROOT/'scripts/run_stage_i_d059_v1.py'


def expected_modules():
    enc = {f'led.encoder.layers.{i}.self_attn.longformer_self_attn.{p}'
           for i in range(6) for p in ('query','key','value','query_global','key_global','value_global')}
    enc |= {f'led.encoder.layers.{i}.self_attn.output' for i in range(6)}
    dec = {f'led.decoder.layers.{i}.{a}.{p}' for i in range(6)
           for a in ('self_attn','encoder_attn') for p in ('q_proj','k_proj','v_proj','out_proj')}
    ffn = {f'led.{s}.layers.{i}.{p}' for s in ('encoder','decoder') for i in range(6) for p in ('fc1','fc2')}
    return enc | dec | ffn


def check_config(c):
    old=read(ROOT/'configs/stage_i_d058_full_filtered_b_v1.json')
    for key in ('model_id','revision','seed','lora','optimization','generation','budgets','training','evaluation_epochs',
                'initialization','load_best_model_at_end','runtime_pins','stage_h_manifest','final_references'):
        require(c[key]==old[key], 'D058 frozen setting changed: '+key)
    require(c['decision']=='D-059' and c['arm']=='Expanded_all_linear' and c['condition']=='filtered','Wrong scope')
    require(c['fresh_start'] and not c['resume_short_pilot'],'Fresh initialization required')
    require(c['target_modules']=='all-linear' and c['expected_modules']==114 and c['expected_trainable_parameters']==1843200,'Architecture differs')
    require(c['quantization'] is False and c['extra_experiments_authorized'] is False,'Unapproved method')
    require(c['stop_loss_days']==7 and c['D051_strict_replay_status']=='FAIL_UNCHANGED','History/deadline differs')
    require(c['feasibility']=={'optimizer_updates':3,'microbatches':24,'source_shape':5120,'label_shape':768,
        'minimum_headroom_fraction':0.15,'generation_calls':0,'replay_updates':0},'Feasibility scope differs')
    require(c['checkpoint_selection']==old['checkpoint_selection'],'Checkpoint policy differs')
    require(budget(c)['max_optimizer_steps']==1250 and budget(c)['generation_outputs']==48,'Execution budget differs')
    return True


def preflight(allow_existing_run=False):
    cfg=read(CONFIG);check_config(cfg);m=read(PREP/'PREFLIGHT_MANIFEST.json')
    require({k:importlib.metadata.version(k) for k in cfg['runtime_pins']}==cfg['runtime_pins'],'Runtime version drift')
    for rel,h in m['files'].items():
        p=Path(rel) if Path(rel).is_absolute() else ROOT/rel
        require(sha(p)==h,'Integrity mismatch: '+rel)
    if not allow_existing_run: require(not (ROOT/cfg['run_root']).exists(),'Existing run; no automatic retry/resume')
    return cfg,m


def check_deadline(a, now=None):
    now=now or datetime.now(timezone.utc)
    start=datetime.fromisoformat(a['execution_authorized_at_utc'])
    end=datetime.fromisoformat(a['improvement_deadline_utc'])
    require(start.tzinfo is not None and end.tzinfo is not None,'Timezone required')
    require(end==start+timedelta(days=7),'Seven-day window changed')
    require(start<=now<end,'Improvement authorization not current or expired; STOP')
    return end


def gate(path,mode):
    require(mode in ('feasibility','train','evaluate'),'Wrong mode')
    a=read(path)
    require(a.get('decision')=='D-059' and a.get('GPU_execution_approved') is True,'Explicit D059 GPU approval required')
    require(a.get('approval_source')=='researcher_message' and a.get('researcher_explicit_authorization'),'Researcher source required')
    require(a.get('config_sha256')==sha(CONFIG) and a.get('runner_sha256')==sha(RUNNER)
            and a.get('preflight_manifest_sha256')==sha(PREP/'PREFLIGHT_MANIFEST.json'),'Approval hash mismatch')
    require(a.get('scopes') and mode in a['scopes'] and set(a['scopes'])<=set(('feasibility','train','evaluate')),'Unapproved scope')
    require(a.get('max_optimizer_steps')==1250 and a.get('generation_outputs_max')==48
            and a.get('feasibility_optimizer_updates_max')==3,'Wrong execution ceilings')
    require(a.get('held_out_access') is False and a.get('extra_experiments_authorized') is False,'Restrictions differ')
    require(a.get('RO_U_training_authorized') is False,'RO/U need separate GO and approval within same deadline')
    check_deadline(a)
    a['_approval_sha256']=sha(path)
    return a


def attach(base,cfg):
    """Use actual PEFT all-linear; unused quantized dispatchers are explicitly disabled."""
    from unittest.mock import patch
    from peft import LoraConfig,get_peft_model
    import torch
    require(not getattr(base,'is_loaded_in_4bit',False) and not getattr(base,'is_loaded_in_8bit',False),'Quantization forbidden')
    raw={n:m for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)}
    require(set(raw)==expected_modules()|{'lm_head'},'Unexpected LED linear inventory')
    require(base.get_output_embeddings() is raw['lm_head'],'LM head exclusion binding differs')
    require(base.get_input_embeddings().weight is raw['lm_head'].weight,'Expected tied embeddings differ')
    with patch('peft.tuners.lora.model.is_bnb_available',return_value=False), \
         patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
        model=get_peft_model(base,LoraConfig(**cfg['lora'],target_modules='all-linear'))
    return model


def inspect(model):
    import torch
    actual=set(model.targeted_module_names)
    require(actual==expected_modules(),'Resolved PEFT scope differs from approved exact inventory')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    require(len(trainable)==228 and sum(p.numel() for p in trainable.values())==1843200,'Adapter count differs')
    require(all('.lora_A.' in n or '.lora_B.' in n for n in trainable),'Backbone/LM head unexpectedly trainable')
    require(all(p.dtype==torch.float32 for p in model.parameters()),'FP32 parameter storage changed')
    modules=[]
    for n,m in model.named_modules():
        name=n.removeprefix('base_model.model.')
        if name in actual:
            require(m.scaling['default']==2.0 and not m.disable_adapters,'Adapter inactive/scaling differs')
            modules.append({'name':name,'in_features':m.in_features,'out_features':m.out_features,
                'A_shape':list(m.lora_A['default'].weight.shape),'B_shape':list(m.lora_B['default'].weight.shape),
                'trainable_parameters':m.lora_A['default'].weight.numel()+m.lora_B['default'].weight.numel()})
    return {'modules':sorted(modules,key=lambda x:x['name']),'module_count':len(actual),
        'trainable_parameters':1843200,'trainable_tensors':228,'excluded':['lm_head','tied/shared/token/position embeddings'],
        'frozen_backbone_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def load_gpu(cfg,unused_prep,arm,approval,mc,unused_root):
    """Called only after explicit execution gate. No downloading or backend fallback."""
    import torch
    from transformers import LEDForConditionalGeneration,set_seed
    from stage_i_paired_pilot import initialize_named_adapters
    require(arm=='Expanded_all_linear' and torch.cuda.is_available(),'Wrong architecture or CUDA unavailable')
    check_deadline(approval)
    set_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    base=LEDForConditionalGeneration.from_pretrained(cfg['model_id'],revision=cfg['revision'],torch_dtype=torch.float32,local_files_only=True)
    for k in ('attention_window','max_encoder_position_embeddings','max_decoder_position_embeddings','d_model',
              'encoder_layers','decoder_layers','bos_token_id','pad_token_id','eos_token_id','decoder_start_token_id'):
        require(getattr(base.config,k)==mc[k],'LED config mismatch: '+k)
    model=attach(base,cfg);scope=inspect(model)
    initialization=initialize_named_adapters(model,cfg['seed'])
    require(initialization==read(PREP/'INITIALIZATION_HASHES.json')['per_module_sha256'],'Fresh initialization differs')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    model.to('cuda');set_seed(cfg['seed'])
    return model,scope,initialization


def require_feasibility(cfg,a):
    p=ROOT/cfg['run_root']/'feasibility/REPORT.json'
    r=read(p)
    require(r['status']=='PASS' and r['config_sha256']==sha(CONFIG),'Successful matching feasibility required')
    require(r['runner_sha256']==sha(RUNNER) and r['preflight_manifest_sha256']==sha(PREP/'PREFLIGHT_MANIFEST.json'),'Feasibility implementation differs')
    require(r['deadline_utc']==a['improvement_deadline_utc'],'Deadline cannot reset between modes')
    same_conditional_approval=(a.get('conditional_train_after_feasibility_pass') is True
        and 'feasibility' in a['scopes'] and r['approval_sha256']==a['_approval_sha256'])
    require(same_conditional_approval or a.get('approved_feasibility_report_sha256')==sha(p),
        'Require the same explicitly conditional approval, or separately approved gate report SHA')
    require(not (p.parent/'FAILURE_STOP_NO_RETRY.json').exists(),'Failed feasibility; stop')
    return r


def feasibility(cfg,a):
    """Future minimal gate: 3 accumulated updates, roundtrip, no generation/replay."""
    import time,traceback
    import torch
    from stage_hi_led_data import PackedLEDDataset,LEDCompleteCollator
    from stage_i_paired_pilot import gradient_record,verify_mask_batch
    from run_stage_i_paired_lora_pilot_v1 import save_feas_checkpoint,restore_feas_checkpoint
    from run_stage_i_d050_feasibility_v1 import parameters,states
    from transformers import get_linear_schedule_with_warmup
    from peft import get_peft_model_state_dict
    folder=ROOT/cfg['run_root']/'feasibility';folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        manifest=read(ROOT/cfg['stage_h_manifest'])
        train=PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768)
        case_ids=read(PREP/'FEASIBILITY_CASES.json')['business_ids']
        features=[train[train.business_ids.index(b)] for b in case_ids]
        mc=read(ROOT/'assets/models/led-base-16384/config.json')
        model,scope,initialization=load_gpu(cfg,PREP,cfg['arm'],a,mc,ROOT)
        before=parameters(model)
        historical=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')['architectures'][0]['canonical_initial_backbone_hashes']
        canonical={n.removeprefix('base_model.model.').replace('.base_layer.','.'):v['sha256'] for n,v in before.items() if not v['requires_grad']}
        require(canonical==historical,'Pretrained tensors changed')
        collator=LEDCompleteCollator(5000,768,1024,pad_source_to=5120,pad_label_to=768)
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'weight_decay':.01},
            {'params':[],'weight_decay':0.}],lr=.0002,betas=(.9,.999),eps=1e-8)
        scheduler=get_linear_schedule_with_warmup(optimizer,63,1250)
        scaler=torch.cuda.amp.GradScaler(enabled=True)
        records=[];memory=[];torch.cuda.reset_peak_memory_stats();model.train()
        for step in range(1,4):
            optimizer.zero_grad(set_to_none=True);losses=[]
            for j in range(8):
                check_deadline(a);f=features[((step-1)*8+j)%len(features)]
                batch=collator([f]);verify_mask_batch(batch,[f]);batch={k:v.cuda() for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.float16):loss=model(**batch,use_cache=False).loss
                require(torch.isfinite(loss).item(),'Nonfinite loss');losses.append(float(loss.detach()))
                scaler.scale(loss/8).backward()
            scaler.unscale_(optimizer);grads=gradient_record(model,expect_later_A=step>=3)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            lr=optimizer.param_groups[0]['lr'];scaler.step(optimizer);scaler.update();scheduler.step();torch.cuda.synchronize()
            require({int(s['step']) for s in optimizer.state.values()}=={step},'AMP skipped update')
            records.append({'step':step,'losses':losses,'used_lr':lr,'gradients':grads})
            free,total=torch.cuda.mem_get_info();memory.append({'used':total-free,'reserved':torch.cuda.memory_reserved(),'total':total})
            write(folder/f'update_{step}.json',records[-1])
        bind={'decision':'D-059','config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER)}
        target={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        state_before=states(optimizer,scheduler,scaler)
        save_feas_checkpoint(model,optimizer,scheduler,scaler,folder/'checkpoint_step3',bind,3)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:p.zero_()
        restore_feas_checkpoint(model,optimizer,scheduler,scaler,folder/'checkpoint_step3',bind)
        require(all(torch.equal(v.detach().cpu(),target[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter roundtrip mismatch')
        require(states(optimizer,scheduler,scaler)==state_before,'Optimizer/scheduler/scaler/RNG roundtrip mismatch')
        after=parameters(model)
        require(all(after[n]['sha256']==v['sha256'] for n,v in before.items() if not v['requires_grad']),'Frozen backbone changed')
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        non_torch=max(max(0,m['used']-m['reserved']) for m in memory)
        headroom=1-(reserved+non_torch)/memory[-1]['total']
        write(folder/'RAW_MEASUREMENTS.json',{'scope':scope,'initialization':initialization,'updates':records,
            'peak_allocated':peak,'peak_reserved':reserved,'headroom_fraction':headroom,'memory':memory,
            'states_before_save':state_before,'states_after_restore':states(optimizer,scheduler,scaler)})
        require(headroom>=.15,'Insufficient 15% headroom; stop without fallback')
        check_deadline(a)
        write(folder/'REPORT.json',{'status':'PASS','config_sha256':sha(CONFIG),'optimizer_updates':3,'microbatches':24,
            'runner_sha256':sha(RUNNER),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'approval_sha256':a['_approval_sha256'],
            'generation_calls':0,'replay_updates':0,'adapter_roundtrip_exact':True,'state_roundtrip_exact':True,
            'frozen_backbone_unchanged':True,'GPU':torch.cuda.get_device_name(),'GPU_total_bytes':memory[-1]['total'],
            'peak_allocated':peak,'peak_reserved':reserved,'headroom_fraction':headroom,'seconds':time.perf_counter()-started,
            'deadline_utc':a['improvement_deadline_utc'],'D051_strict_replay':'FAIL_UNCHANGED',
            'strict_update_replay_tested':False,'feasibility_weights_discarded':True})
    except Exception as exc:
        write(folder/'FAILURE_STOP_NO_RETRY.json',{'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
