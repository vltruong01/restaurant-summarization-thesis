"""CPU-safe contract/integrity primitives. Import never touches a GPU or dataset."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, os, re, math

EXPERIMENT='LED_LARGE_BOUNDARY_DUAL_V2'
ARMS=('bos_only','boundary')
STEPS=750
CHECKPOINTS=(250,500,750)
AMP=dict(max_overflow_skips=8,minimum_scale=256.0,accumulation=8,retry_same_window=True,
         restore_window_RNG=True,gate_microbatches_max=88,train_microbatches_max=6064)
SCALER=dict(init_scale=65536.0,growth_factor=2.0,backoff_factor=.5,growth_interval=2000)
LIMITS=dict(updates_per_arm=750,epochs=3,scheduler_steps=1250,warmup_steps=63,
    generation_per_arm=30,gate_generation_per_arm=2,sanity_per_arm=4,final_per_arm=24,
    memory_headroom=.15,pair_seconds=28800,gate_seconds=7200,train_seconds=21600,
    authorization_hours=48,heartbeat_seconds=60)

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation, no silent overwrite. Receipt is invisible until complete.
    temp=path.with_name(path.name+'.writing')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,path)
    finally:temp.unlink()
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
def uuid(value):
    require(isinstance(value,str) and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value), 'Full canonical GPU UUID required')
    return value
def lr_multiplier(step):
    require(type(step) is int and 0<=step<=1250,'Scheduler step outside original horizon')
    return step/63 if step<63 else max(0.,(1250-step)/1187)
def scheduler_accounting(scheduler,step):
    native=getattr(scheduler,'scheduler',scheduler)
    rates=scheduler.get_last_lr();expected=2e-4*lr_multiplier(step)
    require(native.last_epoch==step and rates and all(math.isclose(x,expected,abs_tol=1e-12,rel_tol=1e-10) for x in rates),'Original LR schedule/accounting differs')
    return dict(horizon=1250,warmup_steps=63,last_epoch=step,next_learning_rates=rates,expected_next_lr=expected)
def private_path(suite,arm):
    require(arm in ARMS,'Unknown arm')
    suite=Path(suite).resolve();p=suite/arm
    require(not p.is_symlink() and p.resolve().parent==suite,'Unsafe arm output path')
    return p
def validate_runtime(old,new,arm):
    expected=dict(old,experiment=EXPERIMENT,run_root='runs/'+EXPERIMENT+'/'+arm,
        optimization=dict(old['optimization'],num_train_epochs=3),
        training=dict(old['training'],epochs=3,steps=750),dynamic_overflow_policy=AMP,evaluation_epochs=[3],
        checkpoint_policy='retain_250_500_750_no_best_loss_reload',
        checkpoint_selection='fixed750_scientific_and_adoption_decisions_separate',
        global_attention=old['global_attention'] if arm=='bos_only' else 'BOS_PLUS_FIRST_BPE_OF_EACH_EXISTING_REVIEW_CAPTION_MARKER_V1',
        vast_fresh=dict(old['vast_fresh'],experiment=EXPERIMENT,output='runs/'+EXPERIMENT+'/'+arm))
    require(new==expected,'Unapproved scientific configuration difference')

def clean_worker_env(env,device,arm):
    device=uuid(device);require(arm in ARMS,'Unknown arm')
    blocked=('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT',
        'ACCELERATE_USE_DEEPSPEED','ACCELERATE_USE_FSDP','ACCELERATE_USE_TP','FORK_LAUNCHED')
    distributed_prefixes=('OMPI_','PMI_','PMIX_','MV2_','TORCHELASTIC_','ACCELERATE_','FSDP_','DEEPSPEED_')
    result={k:v for k,v in env.items() if k not in blocked and not k.startswith(distributed_prefixes)}
    result.update(CUDA_VISIBLE_DEVICES=device,LED_BOUNDARY_GPU_UUID=device,LED_BOUNDARY_ARM=arm,
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return result
def single_runtime(torch,device,accelerator=None,env=None):
    env=os.environ if env is None else env
    require(env.get('CUDA_VISIBLE_DEVICES')==uuid(device),'CUDA visibility differs from assigned UUID')
    require(torch.cuda.device_count()==1 and torch.cuda.current_device()==0,'Worker must expose only logical GPU0')
    require(not torch.distributed.is_initialized(),'Distributed process group forbidden')
    require(torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090' and tuple(torch.cuda.get_device_capability(0))==(8,6),'Wrong GPU')
    if accelerator is not None:
        require(accelerator.num_processes==1 and str(accelerator.distributed_type).split('.')[-1]=='NO','DDP/parallel execution forbidden')
def verify_manifest(root,manifest):
    root=Path(root).resolve()
    for rel,digest in manifest.items():
        p=(root/rel).resolve()
        require(p.is_relative_to(root) and not (root/rel).is_symlink() and sha(p)==digest,'Integrity mismatch: '+rel)
def package_binding(package):
    package=Path(package)
    m=package/'outputs/stage_i/led_large_boundary_dual_v2/PREFLIGHT_MANIFEST.json'
    return dict(config_sha256=sha(package/'configs/stage_i_led_large_boundary_dual_v2.json'),preflight_sha256=sha(m))
def authorize(path,phase,package,now=None):
    require(phase in ('gates','train','decode'),'Unknown execution phase')
    a=read(path);now=now or datetime.now(timezone.utc)
    require(a.get('experiment')==EXPERIMENT and a.get('GPU_execution_approved') is True,'No fresh GPU authorization')
    require(a.get('approval_source')=='researcher_message' and bool(a.get('researcher_explicit_authorization')),'Explicit researcher decision required')
    require(a.get('bindings')==package_binding(package),'Code/configuration authorization binding differs')
    require(a.get('limits')==LIMITS and a.get('amp_policy')==AMP and a.get('amp_scaler')==SCALER,'Locked budgets/numerical policy differ')
    require(all(type(a['limits'][k]) is type(v) for k,v in LIMITS.items()) and all(type(a['amp_policy'][k]) is type(v) for k,v in AMP.items()),'Boolean/integer contract substitution forbidden')
    require(type(a.get('instance_id')) is int and a['instance_id']>0,'Actual instance ID required')
    require(set(a.get('training_devices',{}))==set(ARMS),'Two explicit arm devices required')
    devices=[uuid(a['training_devices'][x]) for x in ARMS];require(devices[0]!=devices[1],'Workers cannot share GPU')
    require(uuid(a.get('common_decoding_device'))==devices[0],'Common decoding GPU must be predeclared BOS-only GPU')
    require(a.get('automatic_retry') is False and a.get('held_out_access') is False and a.get('model_selection_authorized') is False,'Forbidden extra scope')
    require(a.get('Teacher_calls') is False and a.get('RO_U_Shuffled_authorized') is False,'Other scientific stages forbidden')
    require(a.get('validation_integrity_access') is True,'Pinned-parent preflight requires explicit validation-integrity permission')
    require(isinstance(a.get('scopes'),list) and len(set(a['scopes']))==len(a['scopes']) and set(a['scopes'])<=set(('gates','train','decode')) and phase in a['scopes'],'Phase not authorized')
    if phase in ('train','decode'):
        require(a.get('validation_payload_access') is True,'Explicit validation payload permission required')
    if phase=='train':require(a.get('training_authorized') is True and a.get('early_sanity_generation_approved') is True,'Training/sanity not approved')
    if phase=='decode':require(a.get('decoding_authorized') is True and a.get('reference_binding_access') is True,'Decoding/reference integrity not approved')
    clocks=[datetime.fromisoformat(a[k]) for k in ('execution_authorized_at_utc','execution_deadline_utc','researcher_absolute_stop_utc')]
    start,end,absolute=clocks
    require(all(x.tzinfo is not None and x.utcoffset().total_seconds()==0 for x in clocks),'Explicit UTC timestamps required')
    require(start<=now<end<=min(start+timedelta(hours=48),absolute),'Expired/invalid authorization')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True,'Mandatory researcher reserve field missing')
    return dict(a,_approval_sha256=sha(path))

def verify_checkpoint(folder,binding,step):
    require(step in CHECKPOINTS,'Unexpected scientific checkpoint')
    folder=Path(folder);p=folder/f'checkpoint-{step}';r=read(p/'checkpoint_integrity.json')
    require(r['binding']==binding and r['global_step']==step,'Checkpoint binding/step differs')
    needed={'adapter_model.safetensors','adapter_config.json','training_args.bin','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json','AMP_DYNAMIC_STATE.json','SCHEDULER_ACCOUNTING.json'}
    require(needed<=set(r['files']),'Incomplete checkpoint state')
    for name,digest in r['files'].items():
        require(Path(name).name==name and not (p/name).is_symlink() and sha(p/name)==digest,'Changed/unsafe checkpoint component')
    state=read(p/'trainer_state.json');amp=read(p/'AMP_DYNAMIC_STATE.json')
    require(state['global_step']==step and state['max_steps']==750 and state['epoch']==step/250,'Trainer checkpoint accounting differs')
    require(amp['successful_updates']==step and amp['policy']==AMP and amp['overflow_skips']<=8 and amp['actual_forward_backward_microbatches']<=6064,'AMP checkpoint accounting differs')
    schedule=read(p/'SCHEDULER_ACCOUNTING.json')
    require(schedule['horizon']==1250 and schedule['warmup_steps']==63 and schedule['last_epoch']==step and all(math.isclose(v,2e-4*lr_multiplier(step),rel_tol=1e-10,abs_tol=1e-12) for v in schedule['next_learning_rates']),'Saved scheduler differs')
    return sha(p/'checkpoint_integrity.json')
def joint_training(suite,bindings):
    suite=Path(suite);rows={}
    require(not (suite/'PAIR_FAILURE.json').exists(),'Failed pair cannot complete')
    for arm in ARMS:
        p=private_path(suite,arm)/'train';r=read(p/'COMPLETION.json')
        require(not (p/'FAILURE_STOP_NO_RETRY.json').exists(),'Failed arm')
        require(r['binding']==bindings[arm] and r['steps']==750 and r['epochs']==3 and r['checkpoint_steps']==list(CHECKPOINTS),'Missing/incomplete 750-step arm')
        require(r['generation_calls']==4 and read(p/'early_sanity/REPORT.json')['status']=='PASS_TECHNICAL_ONLY','Missing early sanity')
        done=read(private_path(suite,arm)/'train_WORKER_COMPLETE.json')
        require(done['status']=='PASS' and done['phase']=='train' and done['arm']==arm and done['binding']==bindings[arm],'Missing/mismatched train postflight')
        require(read(p/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'] is True,'Backbone preservation not established')
        args=read(p/'ACTUAL_TRAINING_ARGUMENTS.json')
        required=dict(num_train_epochs=3,max_steps=750,warmup_steps=63,gradient_accumulation_steps=8,per_device_train_batch_size=1,fp16=True,bf16=False,load_best_model_at_end=False)
        require(all(args.get(k)==v for k,v in required.items()),'Actual training settings differ')
        progress=[readline for readline in (p/'progress.jsonl').read_text().splitlines() if readline]
        require([json.loads(v)['step'] for v in progress]==list(range(1,751)),'Noncontiguous successful update ledger')
        proof={str(k):verify_checkpoint(p,bindings[arm],k) for k in CHECKPOINTS}
        rows[arm]=dict(completion_sha256=sha(p/'COMPLETION.json'),checkpoint_receipts=proof,
            initialization=read(p/'INITIAL_MODEL_BINDINGS.json')['initialization'])
    require(rows['bos_only']['initialization']==rows['boundary']['initialization'],'Unmatched fresh initialization')
    return dict(status='PAIR_TRAINING_COMPLETE_STOP_NO_AUTOMATIC_DECODING',arms=rows,automatic_evaluation=False,scientific_GO='NOT_ASSESSED',adoption_GO='NOT_ASSESSED')
