"""One matched exploratory pilot despite unresolved historical V2.1 FAIL.

No diagnostic PASS is inferred. Native joint CE path, strict live safety,
no retries/fallback/resume, two independent CE3 initializations, TRAIN probes.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,collections,gc,hashlib,importlib.metadata,json,os,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_exploratory_native_ce_v3'
CONFIG=PREP/'CONFIG.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def append(p,v):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush()


def preflight():
    import run_led_large_exploratory_contrastive_v2 as prior
    old=prior.preflight();c=read(CONFIG)
    for k in c['preserved_scientific_keys']:require(c[k]==old[k],'Science drift: '+k)
    for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'File changed: '+rel)
    require(c['objective']==dict(pair_weight=.1,margin=.1,temperature=1.),'Objective drift')
    require(c['historical_equivalence_status']=='FAIL_UNRESOLVED','Historical failure must remain valid')
    require(c['max_successful_updates_per_arm']==16 and c['accumulation']==8,'Budget drift')
    return c


def authorize(path,c):
    a=read(path)
    require(a['experiment']==c['experiment'] and a['GPU_execution_approved'] is True,'Approval absent')
    require(a['researcher_authorized_post_failure_deviation'] is True and a['historical_gate_passed'] is False,'Deviation disclosure absent')
    require(a['approval_source']=='current_explicit_researcher_message' and a['researcher_explicit_authorization'],'Explicit approval required')
    for k,p in [('config_sha256',CONFIG),('runner_sha256',Path(__file__)),('objective_sha256',ROOT/c['objective_module']),('preflight_sha256',PREP/'PREFLIGHT_MANIFEST.json')]:
        require(a[k]==sha(p),'Authorization hash differs: '+k)
    require(a['successful_updates_per_arm_max']==16 and a['generation_calls_max']==12 and a['automatic_retry'] is False,'Scope differs')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and timedelta(0)<end-start<=timedelta(hours=48) and start<=datetime.now(timezone.utc)<end,'Expired approval')
    return a


def trainable_signature(model):
    from peft import get_peft_model_state_dict
    from run_stage_i_d050_feasibility_v1 import fingerprint
    return fingerprint({n:v.detach().cpu().clone() for n,v in get_peft_model_state_dict(model).items()})


def fresh_optimizer(model,c):
    import torch
    from transformers import get_linear_schedule_with_warmup
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],
        betas=tuple(c['betas']),eps=c['epsilon'],weight_decay=c['weight_decay'])
    schedule=get_linear_schedule_with_warmup(opt,num_warmup_steps=c['warmup_steps'],num_training_steps=16)
    scaler=torch.cuda.amp.GradScaler(init_scale=c['AMP']['init_scale'],growth_factor=c['AMP']['growth_factor'],
        backoff_factor=c['AMP']['backoff_factor'],growth_interval=c['AMP']['growth_interval'])
    require(not opt.state,'Fresh optimizer state required')
    return opt,schedule,scaler


def gradient_stats(model):
    import torch
    result=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        require(p.grad is not None,'Missing gradient: '+name)
        finite=bool(torch.isfinite(p.grad).all().item())
        require(finite,'Nonfinite gradient: '+name)
        result.append(dict(name=name,finite=True,norm=float(p.grad.float().norm()),max_abs=float(p.grad.abs().max())))
    require(len(result)==456,'456 intended LoRA gradients required')
    return result


def batch_to_cuda(item):return {key:{n:t.cuda() for n,t in b.items()} for key,b in item.items()}


class PrefixGuard:
    def __init__(self,model):self.model=model;self.current=None;self.receipts=[]
    def observe(self,kind,batch):
        from transformers.models.led.modeling_led import shift_tokens_right
        import hashlib
        labels=batch['labels'];valid=labels[labels!=-100]
        require(valid[0].item() not in (0,1,2) and valid[-1].item()==2 and not (valid==0).any().item(),'BOS-free labels differ')
        self.current=(kind,shift_tokens_right(labels,1,2).detach().cpu())
    def hook(self,module,args,kwargs):
        import torch
        require(self.current is not None,'Unexpected decoder forward')
        kind,expected=self.current;actual=kwargs['input_ids'].detach().cpu()
        require(torch.equal(actual,expected),'Own-prefix decoder alignment failed: '+kind)
        self.receipts.append(dict(kind=kind,shape=list(actual.shape),sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),own_prefix=True))
        # Recomputations call individual layers, not the decoder module.
        self.current=None
    def __enter__(self):
        self.handle=self.model.base_model.model.led.decoder.register_forward_pre_hook(self.hook,with_kwargs=True);return self
    def __exit__(self,*exc):self.handle.remove()


def effective_batch(model,c,items,arm,scaler,memory,check,folder,index):
    import torch
    from led_large_contrastive_native_joint_v3 import objective
    values=[]
    with PrefixGuard(model) as prefixes:
        for micro,item in enumerate(items):
            check();batch=batch_to_cuda(item)
            # Same autocast policy in both arms. No no-grad scout or custom replay.
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                loss,metrics=objective(model,**batch,arm=arm,observe=prefixes.observe)
            value=float(loss.detach());require(__import__('math').isfinite(value),'Nonfinite loss')
            scaler.scale(loss/8).backward()
            values.append(dict(microbatch=micro,total=value,**metrics))
            append(folder/'microbatches.jsonl',dict(window=index,arm=arm,**values[-1]))
            del loss,batch
            memory.check()
    require(len(values)==8 and len(prefixes.receipts)==(8 if arm=='ce' else 16),'Effective batch/prefix count differs')
    append(folder/'decoder_prefixes.jsonl',dict(window=index,arm=arm,receipts=prefixes.receipts))
    return values


def live_safety(model,c,folder,memory,check):
    import torch
    import led_large_contrastive_sequential_v1 as rng
    import run_led_large_exploratory_contrastive_v2 as prior
    from stage_i_led_large_decoder_policy_v2 import verify_backbone
    before=trainable_signature(model);initial=rng.capture_rng(torch.device('cuda'))
    data=prior.collated(c,'cpu');results=[]
    for arm in ('ce','pair'):
        dest=folder/arm;dest.mkdir()
        rng.restore_rng(initial,torch.device('cuda'));model.zero_grad(set_to_none=True)
        opt,sched,scaler=fresh_optimizer(model,c)
        steps=[];hook=opt.register_step_post_hook(lambda *args:steps.append(True))
        values=effective_batch(model,c,data[:8],arm,scaler,memory,check,dest,0)
        scaled=gradient_stats(model);scaler.unscale_(opt);unscaled=gradient_stats(model)
        before_scale=scaler.get_scale();scaler.update()
        require(scaler.get_scale()==before_scale==65536 and scaler.state_dict()['_growth_tracker']==1,'Unexpected live AMP behavior')
        require(not steps and not opt.state and trainable_signature(model)==before,'Unintended live-check optimizer update')
        result=dict(arm=arm,status='LIVE_TRAINING_PATH_SAFE',losses=values,scaled_gradients=scaled,
            unscaled_gradients=unscaled,scaler_after_finite_unscale_update=scaler.state_dict(),
            successful_optimizer_updates=0,optimizer_state_empty=True,adapter_unchanged=True,
            native_CE_only=True,manual_logit_CE=False,negative_own_prefix=(arm=='pair'),memory=memory.summary())
        write(dest/'SAFETY.json',result);results.append(result)
        hook.remove();model.zero_grad(set_to_none=True);del opt,sched,scaler
        memory.check()
    rng.restore_rng(initial,torch.device('cuda'))
    write(folder/'SAFETY_CHECKS_PASS.json',dict(status='PASS_LIVE_SAFETY_ONLY',historical_lambda0='FAIL_UNRESOLVED_NOT_OVERRIDDEN',
        successful_optimizer_updates=0,native_joint_training_path=True,arms=[x['arm'] for x in results],memory=memory.summary()))
    return before


def save_checkpoint(model,opt,sched,scaler,c,folder,step,arm,exposures):
    import torch
    import led_large_contrastive_sequential_v1 as rng
    cp=folder/f'checkpoint-{step}';cp.mkdir()
    model.save_pretrained(cp,safe_serialization=True)
    torch.save(opt.state_dict(),cp/'optimizer.pt');torch.save(sched.state_dict(),cp/'scheduler.pt')
    torch.save(scaler.state_dict(),cp/'scaler.pt');torch.save(rng.capture_rng(torch.device('cuda')),cp/'rng_state.pth')
    write(cp/'trainer_state.json',dict(global_step=step,arm=arm,max_steps=16,positive_exposures=exposures,
        config_sha256=sha(CONFIG),historical_equivalence='FAIL_UNRESOLVED',no_resume_authorized=True))
    entries={p.name:sha(p) for p in cp.iterdir() if p.is_file()}
    require({'adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'}<=entries.keys(),'Incomplete checkpoint')
    write(cp/'checkpoint_integrity.json',dict(global_step=step,arm=arm,files=entries,config_sha256=sha(CONFIG),runner_sha256=sha(__file__)))
    return str(cp.relative_to(ROOT))


def train_arm(model,c,arm,folder,memory,check,initial_signature):
    import torch
    import run_led_large_exploratory_contrastive_v2 as prior
    require(trainable_signature(model)==initial_signature,'Arm initial adapter differs')
    model.train();opt,sched,scaler=fresh_optimizer(model,c);calls=[]
    def record_update(*args):
        calls.append(True)
        append(folder/'optimizer_updates.jsonl',dict(successful_optimizer_updates=len(calls),
            utc=datetime.now(timezone.utc).isoformat(),lr=[g['lr'] for g in opt.param_groups]))
    hook=opt.register_step_post_hook(record_update)
    data=prior.collated(c,'cpu');checkpoints=[]
    write(folder/'INITIALIZATION.json',dict(adapter_fingerprint=initial_signature,optimizer_state_empty=not opt.state,
        seed=42,scaler=scaler.state_dict(),source_adapter_sha256=c['adapter_sha256']))
    for i in range(16):
        check();opt.zero_grad(set_to_none=True);lr_before=[g['lr'] for g in opt.param_groups]
        values=effective_batch(model,c,data[i*8:(i+1)*8],arm,scaler,memory,check,folder,i+1)
        scaled=gradient_stats(model);scaler.unscale_(opt);gradients=gradient_stats(model)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['max_grad_norm'],error_if_nonfinite=True)
        require(torch.isfinite(norm).item(),'Nonfinite gradient norm')
        before_scale=scaler.get_scale();before_calls=len(calls)
        scaler.step(opt);scaler.update()
        require(len(calls)==before_calls+1==i+1,'Skipped/unexplained optimizer update')
        require(scaler.get_scale()==before_scale==65536,'Unexpected scaler backoff/growth; no retry')
        steps={int(v['step']) for v in opt.state.values()}
        require(steps=={i+1} and len(opt.state)==456,'Per-tensor Adam step count mismatch')
        sched.step();memory.check()
        record=dict(step=i+1,arm=arm,positive_exposures=(i+1)*8,optimizer_successful_steps=len(calls),
            optimizer_state_tensors=len(opt.state),optimizer_state_steps=sorted(steps),lr_before=lr_before,
            lr_after=[g['lr'] for g in opt.param_groups],mean_total_loss=sum(x['total'] for x in values)/8,
            mean_CE=sum(x['CE'] for x in values)/8,mean_contrastive=(sum(x['contrastive'] for x in values)/8 if arm=='pair' else None),
            global_unclipped_gradient_norm=float(norm),scaled_gradient_max=max(x['max_abs'] for x in scaled),
            scaler=scaler.state_dict(),memory=memory.summary())
        append(folder/'progress.jsonl',record);append(folder/'gradient_checks.jsonl',dict(step=i+1,gradients=gradients))
        print('TRAIN_UPDATE',arm,i+1,json.dumps({k:record[k] for k in ('mean_total_loss','mean_CE','global_unclipped_gradient_norm','scaler')}),flush=True)
        if i+1 in (8,16):checkpoints.append(save_checkpoint(model,opt,sched,scaler,c,folder,i+1,arm,(i+1)*8))
    hook.remove();opt.zero_grad(set_to_none=True)
    write(folder/'COMPLETION.json',dict(status='TRAINING_COMPLETE_EXPLORATORY_NOT_QUALITY_GO',arm=arm,
        successful_optimizer_updates=16,positive_exposures=128,checkpoints=checkpoints,
        manual_CE=False,native_joint_CE=True,historical_equivalence='FAIL_UNRESOLVED',memory=memory.summary()))
    return checkpoints[-1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','suite']);ap.add_argument('--execute',action='store_true');ap.add_argument('--approval',type=Path)
    args=ap.parse_args();c=preflight()
    if args.mode=='preflight':print('V3_CPU_PREFLIGHT_PASS_NO_GPU');return
    require(args.execute and args.approval,'Explicit V3 approval required');a=authorize(args.approval,c)
    suite=(ROOT/c['execution_root']).resolve()
    require(suite.parent==ROOT/'runs' and not suite.exists(),'Exact fresh V3 namespace required')
    require(sha(ROOT/'VAST_SHA256_MANIFEST.json')==c['host_parent_manifest_sha256'],'Parent manifest differs')
    # Full parent is hashed for preservation only; no validation/held-out examples
    # are materialized, scored, generated or used for selection/tuning.
    parent=read(ROOT/'VAST_SHA256_MANIFEST.json')['files']
    for rel,digest in parent.items():require(sha(ROOT/rel)==digest,'Parent changed: '+rel)
    require(sha(ROOT/c['host_backbone_path'])==c['host_backbone_sha256'],'Original backbone differs')
    for rel,digest in c['historical_failure_files'].items():require(sha(ROOT/rel)==digest,'Historical FAIL changed')
    suite.mkdir();write(suite/'AUTHORIZATION.json',a)
    end=min(datetime.fromisoformat(a['deadline_utc']),datetime.now(timezone.utc)+timedelta(hours=1))
    write(suite/'BUDGET.json',dict(deadline_utc=end.isoformat(),updates_per_arm_max=16,generation_calls_max=12,attempts=1,automatic_retry=False))
    def check():require(datetime.now(timezone.utc)<end,'Execution time limit reached')
    def timeout():
        try:write(suite/'TIMEOUT_STOP_NO_RETRY.json',dict(deadline_utc=end.isoformat()))
        finally:os._exit(124)
    timer=threading.Timer((end-datetime.now(timezone.utc)).total_seconds(),timeout);timer.daemon=True;timer.start()
    stage='LOAD_LIVE_SAFETY';memory=None
    try:
        import torch
        import run_led_large_exploratory_contrastive_v2 as prior
        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter,hardware_profile
        from stage_i_led_large_decoder_policy_v2 import verify_backbone
        require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3090','Exact RTX3090 required')
        write(suite/'HARDWARE.json',dict(hardware_profile(torch),name=torch.cuda.get_device_name(),parent_hash_files=len(parent)))
        model,base,scope=prior.load(c)
        with MemoryMeter(torch) as memory:
            safety=suite/'live_safety';safety.mkdir();initial_signature=live_safety(model,c,safety,memory,check)
            verify_backbone(model,base);memory.check()
            del model,base;gc.collect();torch.cuda.empty_cache()
            checkpoints={}
            for arm in ('ce','pair'):
                stage='TRAIN_'+arm.upper();check()
                model,base,scope=prior.load(c)
                dest=suite/('train_'+arm);dest.mkdir()
                checkpoints[arm]=train_arm(model,c,arm,dest,memory,check,initial_signature)
                verify_backbone(model,base);memory.check()
                del model,base;gc.collect();torch.cuda.empty_cache()
            # No inference until BOTH matched arms have completed successfully.
            stage='GENERATE_12_TRAIN_PROBES';check()
            model,base,scope=prior.load(c)
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict,get_peft_model_state_dict
            total=0;call_counter=[0]
            original_generate=model.generate
            def counted_generate(*args,**kwargs):
                require(call_counter[0]<12,'Generation call ceiling reached')
                call_counter[0]+=1
                append(suite/'generation_calls.jsonl',dict(call=call_counter[0],utc=datetime.now(timezone.utc).isoformat()))
                return original_generate(*args,**kwargs)
            model.generate=counted_generate
            variants=[('original_ce3',c['adapter']),('ce16',checkpoints['ce']+'/adapter_model.safetensors'),('pair16',checkpoints['pair']+'/adapter_model.safetensors')]
            for name,adapter in variants:
                state=load_file(str(ROOT/adapter),device='cpu');set_peft_model_state_dict(model,state)
                actual=get_peft_model_state_dict(model)
                require(set(state)==set(actual) and all(torch.equal(v,actual[k].detach().cpu()) for k,v in state.items()),'Evaluation adapter restore differs')
                dest=suite/('evaluate_'+name);dest.mkdir()
                write(dest/'MODEL_BINDING.json',dict(adapter=adapter,sha256=sha(ROOT/adapter)))
                results=prior.generate_probes(model,c,dest,memory,check)
                require(len(results)==4,'Four TRAIN probes required');total+=len(results)
                write(dest/'COMPLETION.json',dict(status='FOUR_TRAIN_PROBES_COMPLETE',generation_calls=4,quality_GO=False))
            require(total==call_counter[0]==12,'Exactly12 generation calls required')
            model.generate=original_generate
            verify_backbone(model,base);memory.check();check()
        require(memory.failure is None,'Memory measurement failed')
        write(suite/'MEMORY_EVIDENCE.json',dict(summary=memory.summary(),samples=memory.samples,failure=memory.failure))
        for rel,digest in c['historical_failure_files'].items():require(sha(ROOT/rel)==digest,'Historical FAIL changed after execution')
        for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'Bound artifact changed after execution: '+rel)
        write(suite/'COMPLETION.json',dict(status='EXPLORATORY_MATCHED_PILOT_COMPLETE_STOP_FOR_REVIEW',
            historical_equivalence='FAIL_UNRESOLVED',protocol_deviation=True,successful_updates={'ce':16,'pair':16},
            generation_calls=12,checkpoints=checkpoints,quality_GO=False,automatic_continuation=False,
            config_sha256=sha(CONFIG),runner_sha256=sha(__file__),objective_sha256=sha(ROOT/c['objective_module'])))
        print('V3_COMPLETE_STOP_FOR_INDEPENDENT_REVIEW',flush=True)
    except BaseException as e:
        write(suite/'FAILURE_STOP_NO_RETRY.json',dict(stage=stage,error=repr(e),traceback=traceback.format_exc(),
            historical_equivalence='FAIL_UNRESOLVED',automatic_retry=False,automatic_continuation=False))
        raise
    finally:
        timer.cancel()
        if memory is not None and not (suite/'MEMORY_EVIDENCE.json').exists():
            try:write(suite/'MEMORY_EVIDENCE.json',dict(summary=memory.summary(),samples=memory.samples,failure=memory.failure))
            except Exception as e:write(suite/'MEMORY_EVIDENCE_UNAVAILABLE.json',dict(error=repr(e)))


if __name__=='__main__':main()
