"""D-060 RO-initialized continuation. Import has no model/GPU side effects."""
import hashlib
import importlib.metadata
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/stage_i_d060_ro_to_filtered_v1.json'
PREP = ROOT/'outputs/stage_i/d060_preparation_v1'
BASE_PREP = ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
RUNNER = ROOT/'scripts/run_stage_i_d060_v1.py'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
    if not v: raise ValueError(m)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write((v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))+'\n')

def check_config(c):
    old=read(ROOT/'configs/stage_i_d058_full_filtered_b_v1.json')
    require(c['decision']=='D-060' and c['condition']=='filtered' and c['arm']=='A_decoder_only','Wrong scope')
    for k in ('model_id','revision','seed','lora','generation','budgets','runtime_pins','stage_h_manifest','final_references'):
        require(c[k]==old[k],'Unapproved scientific change: '+k)
    require(c['optimization']=={**old['optimization'],'num_train_epochs':3,'save_total_limit':3},'Optimization differs')
    require(c['training']=={'businesses':2000,'epochs':3,'steps':750,'steps_per_epoch':250,'warmup_steps':38},'Wrong schedule')
    require(c['evaluation_epochs']==[1,3] and c['fresh_optimizer'] and not c['resume_from_checkpoint'],'Wrong continuation policy')
    require(c['initialization']['kind']=='exact_historical_RO_adapter' and c['expected_modules']==24 and c['expected_trainable_parameters']==294912,'Wrong initialization/scope')
    require(c['D051_strict_replay_status']=='FAIL_UNCHANGED' and c['viability_hours']==72,'History/deadline differs')
    require(c['GPU_execution_approved'] is False and c['training_authorized'] is False,'Preparation is not authorization')
    return True

def preflight(allow_existing_run=False):
    c=read(CONFIG);check_config(c);m=read(PREP/'PREFLIGHT_MANIFEST.json')
    for rel,h in m['files'].items():
        p=Path(rel);p=p if p.is_absolute() else ROOT/p
        require(sha(p)==h,'Integrity mismatch: '+rel)
    require({k:importlib.metadata.version(k) for k in c['runtime_pins']}==c['runtime_pins'],'Runtime drift')
    if not allow_existing_run: require(not (ROOT/c['run_root']).exists(),'No overwrite/retry/resume')
    return c,m

def check_deadline(a,now=None):
    start=datetime.fromisoformat(a['execution_authorized_at_utc'])
    deadline=datetime.fromisoformat(a['viability_deadline_utc'])
    historical=datetime.fromisoformat('2026-10-25T15:10:48.941817+00:00')
    require(start.utcoffset()==timedelta(0) and deadline==min(start+timedelta(hours=72),historical),'Clock extended/reset')
    require((now or datetime.now(timezone.utc)) < deadline,'STOP: viability decision deadline reached')

def gate(path,mode):
    a=read(path)
    require(mode in ('train','evaluate') and a.get('decision')=='D-060','Wrong authorization')
    require(a.get('GPU_execution_approved') is True and a.get('approval_source')=='researcher_message' and a.get('researcher_explicit_authorization'),'Explicit GPU approval absent')
    for key,p in [('config_sha256',CONFIG),('preflight_manifest_sha256',PREP/'PREFLIGHT_MANIFEST.json'),('runner_sha256',RUNNER)]:
        require(a.get(key)==sha(p),'Approval hash mismatch: '+key)
    require(a.get('max_optimizer_steps')==750 and a.get('generation_outputs_max')==48 and a.get('training_authorized') is True,'Execution ceiling differs')
    require(a.get('RO_U_training_authorized') is False and a.get('held_out_access') is False and a.get('Teacher_calls_authorized') is False,'Forbidden scope')
    check_deadline(a);return a

def training_arguments(c,out):
    from stage_i_d058 import training_arguments as old_args
    a=old_args(c,out)
    a.update(num_train_epochs=3,save_total_limit=3,warmup_ratio=0.0,warmup_steps=38)
    return a

def load_ro_cpu(c):
    """Actual PEFT load, exact RO tensors, CPU only, no forward/gradient update."""
    import torch
    from transformers import LEDForConditionalGeneration
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from unittest.mock import patch
    from stage_i_paired_pilot import inspect_attached
    adapter=ROOT/c['initialization']['adapter_path']
    require(sha(adapter/'adapter_model.safetensors')==c['initialization']['adapter_sha256'],'RO adapter changed')
    require(sha(adapter/'adapter_config.json')==c['initialization']['adapter_config_sha256'],'RO adapter configuration changed')
    base=LEDForConditionalGeneration.from_pretrained(c['model_id'],revision=c['revision'],torch_dtype=torch.float32,local_files_only=True)
    # Same accepted no-quantization dispatch isolation as D059. No weight quantization.
    with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
        model=PeftModel.from_pretrained(base,adapter,is_trainable=True,torch_device='cpu')
    scope=inspect_attached(model,'A_decoder_only')
    require(all(p.device.type=='cpu' for p in model.parameters()),'Unexpected device')
    weights=load_file(adapter/'adapter_model.safetensors',device='cpu')
    actual=get_peft_model_state_dict(model)
    require(set(actual)==set(weights) and all(torch.equal(v.detach().cpu(),weights[k]) for k,v in actual.items()),'RO load not exact')
    require(all(torch.isfinite(v).all().item() for v in actual.values()),'Nonfinite adapter')
    return model,scope,{'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'loaded_tensor_count':len(actual),'exact_RO_tensor_equality':True,'optimizer_state_loaded':False}
