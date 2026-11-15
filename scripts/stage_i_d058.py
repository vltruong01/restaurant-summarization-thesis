"""Small D058 configuration, budget, integrity and authorization helpers (CPU only)."""
import hashlib,json,math,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/d058_preparation_v1'
CONFIG=ROOT/'configs/stage_i_d058_full_filtered_b_v1.json'
BASE_PREP=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write((v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))+'\n')

def budget(c):
    o=c['optimization'];n=c['training']['businesses'];epochs=o['num_train_epochs']
    updates=math.ceil(n/(o['physical_batch_size']*o['gradient_accumulation_steps']))
    return {'steps_per_epoch':updates,'max_optimizer_steps':updates*epochs,'warmup_steps':math.ceil(o['warmup_ratio']*updates*epochs),
        'training_microbatches':n*epochs//o['physical_batch_size'],'validation_teacher_forced_batches':200*epochs,
        'generation_outputs':24*len(c['evaluation_epochs']),'max_new_token_slots':24*len(c['evaluation_epochs'])*(c['generation']['max_length']-1),
        'quality_checkpoint_steps':[e*updates for e in c['evaluation_epochs']]}

def check_config(c):
    old=read(ROOT/'configs/stage_i_paired_lora_pilot_v1.json')
    require(c['decision']=='D-058' and c['condition']=='filtered' and c['arm']=='B_encoder_decoder','Wrong experiment')
    require(c['model_id']==old['model_id'] and c['revision']==old['revision'] and c['seed']==old['seed'],'Backbone/seed changed')
    require(c['lora']==old['lora'] and c['generation']==old['generation'] and c['budgets']==old['budgets'],'Frozen LoRA/generation/budgets differ')
    changes={'num_train_epochs':5,'early_stopping_patience':None,'save_total_limit':5}
    expected={**old['optimization'],**changes}
    require(c['optimization']==expected,'Unreported optimization change')
    require(c['training']=={'businesses':2000,'epochs':5,'steps':1250,'steps_per_epoch':250,'warmup_steps':63},'Training scope differs')
    require(c['evaluation_epochs']==[3,5] and c['fresh_start'] and not c['resume_short_pilot'],'Checkpoint scope differs')
    require(c['checkpoint_selection']=='researcher_quality_review_of_epochs3_and5_only' and not c['load_best_model_at_end'],'Loss-based checkpoint substitution forbidden')
    require(c['D051_strict_replay_status']=='FAIL_UNCHANGED','Replay history changed')
    require(budget(c)=={'steps_per_epoch':250,'max_optimizer_steps':1250,'warmup_steps':63,'training_microbatches':10000,
        'validation_teacher_forced_batches':1000,'generation_outputs':48,'max_new_token_slots':36816,'quality_checkpoint_steps':[750,1250]},'Execution ceiling arithmetic differs')
    return True

def preflight(allow_existing_run=False):
    cfg=read(CONFIG);check_config(cfg);m=read(PREP/'PREFLIGHT_MANIFEST.json')
    require({k:importlib.metadata.version(k) for k in cfg['runtime_pins']}==cfg['runtime_pins'],'Pinned execution environment differs')
    for rel,h in m['files'].items():require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Integrity mismatch: '+rel)
    if not allow_existing_run:require(not (ROOT/cfg['run_root']).exists(),'D058 run already exists; no automatic retry/resume')
    return cfg,m

def gate(approval,mode):
    require(mode in ['train','evaluate'],'Invalid execution mode')
    a=read(approval)
    require(a.get('decision')=='D-058' and a.get('GPU_execution_approved') is True,'D058 GPU execution not approved')
    require(a.get('approval_source')=='researcher_message' and a.get('researcher_explicit_authorization'),'Explicit researcher authorization required')
    require(a.get('config_sha256')==sha(CONFIG) and a.get('preflight_manifest_sha256')==sha(PREP/'PREFLIGHT_MANIFEST.json')
        and a.get('runner_sha256')==sha(ROOT/'scripts/run_stage_i_d058_v1.py'),'Approval binding mismatch')
    require(a.get('training_authorized') is True and a.get('max_optimizer_steps')==1250 and a.get('generation_outputs_max')==48,'Execution scope differs')
    require(a.get('held_out_access') is False and a.get('extra_experiments_authorized') is False,'Wrong restrictions')
    return a

def training_arguments(c,out):
    o=c['optimization']
    return dict(output_dir=str(out),num_train_epochs=5,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,learning_rate=o['learning_rate'],weight_decay=o['weight_decay'],optim='adamw_torch',
        adam_beta1=o['adam_beta1'],adam_beta2=o['adam_beta2'],adam_epsilon=o['adam_epsilon'],max_grad_norm=1.,
        lr_scheduler_type='linear',warmup_ratio=.05,fp16=True,bf16=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='epoch',save_strategy='epoch',save_total_limit=5,
        load_best_model_at_end=False,metric_for_best_model='eval_loss',greater_is_better=False,
        seed=42,data_seed=42,full_determinism=False,remove_unused_columns=False,label_names=['labels'],
        predict_with_generate=False,label_smoothing_factor=0.,report_to='none',logging_steps=8,dataloader_num_workers=0)
