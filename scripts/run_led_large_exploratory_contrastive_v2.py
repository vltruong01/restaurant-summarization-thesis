"""Fail-closed candidate runner: CPU preflight, separately approved diagnosis/gates/pilot.

Exploratory Contrastive Pilot V2 — Post-Adjudication Amendment. Conditional execution authorized separately; never preregistered.
Existing Trainer, bounded recovery, memory meter and state-continuity code reused.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse
import hashlib
import importlib.metadata
import json
import os
import time

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
CONFIG=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2/CONFIG.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def json_safe(value):
    import math
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(json_safe(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def append(p,value):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(json_safe(value),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def require(ok,message):
    if not ok:raise ValueError(message)

def preflight():
    c=read(CONFIG)
    for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'Bound file changed: '+rel)
    for dist,expected in c['runtime_pins'].items():require(importlib.metadata.version(dist)==expected,'Runtime differs: '+dist)
    require(c['objective']==dict(pair_weight=.1,margin=.1,temperature=1.),'Objective drift')
    require(c['pilot_updates']==16 and c['accumulation']==8,'Budget drift')
    require(c['pair_ids']==[f'P{i:02d}' for i in range(1,17) if i!=3],'Locked15 pair drift')
    require(c['excluded_pairs']==['P03','P17','P18'],'Excluded pairs drift')
    require(len(c['fit'])==8 and len(c['probe'])==4 and not set(c['fit'])&set(c['probe']),'TRAIN roster drift')
    cp=ROOT/c['adapter'];proof=read(cp.parent/'checkpoint_integrity.json')
    for name,h in proof['files'].items():require(Path(name).name==name and sha(cp.parent/name)==h,'CE3 component changed')
    require(sha(cp)==c['adapter_sha256'],'Foreign adapter')
    counts={}
    for bid in c['fit']:
        p=read(PREP/'positives'/f'{bid}.json')
        require(p['split']=='train' and p['condition']=='filtered','Foreign data')
        require(p['corrected_labels']==p['original_labels'][1:] and p['original_labels'][0]==0 and p['corrected_labels'][-1]==2,'Decoder labels differ')
    for pid in c['pair_ids']:
        p=read(PREP/'pairs'/f'{pid}.json');counts[p['category']]=counts.get(p['category'],0)+1
    require(counts==dict(core_omission=5,same_aspect_disagreement_loss=6,grounding_error=4),'Quotas drift')
    return c

def authorization(args,c):
    require(args.execute and args.approval,'Explicit V2 authorization required before CUDA')
    a=read(args.approval)
    require(a['experiment']=='EXPLORATORY_CONTRASTIVE_PILOT_V2_POST_ADJUDICATION','Foreign approval')
    require(a['GPU_execution_approved'] is True and a['approval_source']=='researcher_message','Approval absent')
    require(a['post_adjudication_amendment'] is True and a['not_preregistered'] is True,'Amendment disclosure absent')
    for key,path in [('config_sha256',CONFIG),('runner_sha256',Path(__file__)),('objective_sha256',ROOT/'scripts/led_large_contrastive_sequential_v1.py'),('preflight_sha256',CONFIG.parent/'PREFLIGHT_MANIFEST.json')]:
        require(a[key]==sha(path),'Approval hash differs: '+key)
    require(args.mode in a['scopes'] and args.arm in ('ce','pair'),'Unapproved mode')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and timedelta(0)<end-start<=timedelta(hours=48) and start<=datetime.now(timezone.utc)<end,'Expired authorization')
    require(a['updates_per_arm_max']==16 and a['generation_calls_max']==12 and a['diagnostic_attempts_max']==1,'Scope drift')
    require(args.decisions and sha(args.decisions)==a['researcher_decisions_sha256'],'Adjudication binding differs')
    d=read(args.decisions)
    require(d['provenance']=='Explicit current researcher message; historical PENDING files preserved','Foreign adjudication provenance')
    require(set(d['positives'])==set(c['fit']) and all(x['decision']=='ACCEPT' for x in d['positives'].values()),'Eight accepted positives required')
    require(set(d['pairs'])==set(c['pair_ids']) and all(x['decision']=='ACCEPT' for x in d['pairs'].values()),'Exactly15 accepted locked pairs required')
    require(d['excluded_pairs']==['P03','P17','P18'] and d['P03_decision']=='REJECT','Exclusions differ')
    for bid,v in d['positives'].items():require(v['sha256']==sha(PREP/'positives'/f'{bid}.json'),'Positive decision hash differs')
    for pid,v in d['pairs'].items():require(v['sha256']==sha(PREP/'pairs'/f'{pid}.json'),'Pair decision hash differs')
    if args.mode!='diagnose':
        require(args.numerical_receipt is not None,'Diagnostic receipt required')
        expected=ROOT/c['execution_root']/'diagnose_pair/COMPLETION.json'
        require(args.numerical_receipt.resolve()==expected.resolve() and expected.is_file(),'Foreign diagnostic path')
        nr=read(expected)
        require(nr['status']=='PASS' and nr['full_pretrained_CUDA_FP16'] is True,'Mandatory diagnostic did not pass')
        require(not (expected.parent/'FAILURE_STOP_NO_RETRY.json').exists(),'Diagnostic failed; no training')
    return a

def collated(c,device):
    import torch
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    col=CorrectedLabelCollator();positives={bid:read(PREP/'positives'/f'{bid}.json') for bid in c['fit']}
    by_bid={bid:[] for bid in c['fit']}
    for pid in c['pair_ids']:
        pair=read(PREP/'pairs'/f'{pid}.json');bid=pair['business_id'];positive=positives[bid]
        require(pair['positive_text']==positive['positive_text'],'Positive rewritten')
        def pack(labels):return col([dict(input_ids=positive['input_ids'],labels=labels)])
        by_bid[bid].append(dict(positive=pack(pair['positive_labels']),negative=pack(pair['negative_labels'])))
    order=[by_bid[bid][i%len(by_bid[bid])] for i in range(16) for bid in c['fit']]
    if device=='cuda':return [{k:{n:v.cuda() for n,v in b.items()} for k,b in item.items()} for item in order]
    return order

def load(c):
    import torch
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_i_led_large_decoder_policy_v2 import load_model
    original=read(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json')
    model,before,scope=load_model(original)
    state=load_file(str(ROOT/c['adapter']),device='cpu')
    set_peft_model_state_dict(model,state)
    actual=get_peft_model_state_dict(model)
    require(set(actual)==set(state) and all(torch.equal(actual[n].detach().cpu(),v) for n,v in state.items()),'CE3 weight restore differs')
    model.train();require(sum(p.requires_grad for p in model.parameters())==456,'LoRA scope differs')
    return model,before,scope

def hardware():
    import subprocess
    import torch
    driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version,uuid','--format=csv,noheader'],
        check=True,capture_output=True,text=True,timeout=15).stdout.strip()
    return dict(name=torch.cuda.get_device_name(),total_bytes=torch.cuda.get_device_properties(0).total_memory,
        driver_and_uuid=driver,torch_cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability()))

def diagnose(model,c,folder,memory,check):
    import torch
    import led_large_contrastive_sequential_v1 as seq
    from run_stage_i_d050_feasibility_v1 import fingerprint
    params=tuple(p for p in model.parameters() if p.requires_grad)
    batches=collated(c,'cuda');sample=batches[1] # Frozen Tempo pair P02, source4021 -> padded4096; both own prefixes.
    positive,negative=sample['positive'],sample['negative'];device=params[0].device
    def event(stage,**fields):
        append(folder/'NUMERICAL_STAGES.jsonl',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**fields))
        print('DIAGNOSTIC_STAGE',stage,flush=True)
    def gradstats(grads):
        return [dict(index=i,shape=list(g.shape),finite=bool(torch.isfinite(g).all()),max_abs=float(torch.nan_to_num(g,nan=0.,posinf=0.,neginf=0.).abs().max())) for i,g in enumerate(grads)]
    modes=seq._mode_signature(model)
    weights={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    event('START',intended_accumulation=8,physical_batch=1,AMP_init_scale=65536,comparison_atol=1e-5,comparison_rtol=1e-4)
    initial=seq.capture_rng(device)
    from transformers.models.led.modeling_led import shift_tokens_right
    shifted={}
    def observe_prefix(module,args,kwargs):shifted['ids']=kwargs['input_ids'].detach().cpu().tolist()
    hook=model.base_model.model.led.decoder.register_forward_pre_hook(observe_prefix,with_kwargs=True)
    try:
        with torch.autocast('cuda',dtype=torch.float16):
            output=model(**positive,use_cache=False);native=output.loss
            manual_ce=torch.nn.functional.cross_entropy(output.logits.reshape(-1,output.logits.shape[-1]),positive['labels'].reshape(-1),ignore_index=-100)
    finally:hook.remove()
    expected_prefix=shift_tokens_right(positive['labels'],1,2).detach().cpu().tolist()
    own_prefix=shifted['ids']==expected_prefix
    event('DECODER_PREFIX_AND_REDUCTION',own_positive_prefix=own_prefix,decoder_ids=shifted['ids'],native_loss=float(native.detach()),manual_token_CE=float(manual_ce.detach()),headers_EOS_included=True)
    require(own_prefix and torch.equal(native,manual_ce),'Decoder shift/token CE reduction mismatch')
    del manual_ce,output
    ga=torch.autograd.grad(native,params);seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):delegated=seq.loss(model,positive,negative,weight=0.)
    gb=torch.autograd.grad(delegated,params)
    equal=torch.equal(native,delegated) and all(torch.equal(x,y) for x,y in zip(ga,gb))
    event('LAMBDA0_NATIVE_CE',loss=float(native),delegated_loss=float(delegated),bit_exact=equal,gradients=gradstats(gb))
    require(equal,'Lambda0 native CE loss/gradient mismatch')
    require(all(torch.isfinite(g).all().item() for g in ga+gb),'Lambda0 nonfinite gradients')
    del native,delegated,ga,gb
    check();memory.check();initial=seq.capture_rng(device)
    with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
        sp=-model(**positive,use_cache=False).loss.float()
        hook=model.base_model.model.led.decoder.register_forward_pre_hook(observe_prefix,with_kwargs=True)
        try:sn=-model(**negative,use_cache=False).loss.float()
        finally:hook.remove()
        require(shifted['ids']==shift_tokens_right(negative['labels'],1,2).detach().cpu().tolist(),'Negative scored under wrong decoder prefix')
        event('NEGATIVE_OWN_PREFIX',decoder_ids=shifted['ids'])
        joint=-sp+.1*torch.nn.functional.softplus(.1-sp+sn)
    joint_value=float(joint.detach());reference=torch.autograd.grad(joint,params)
    expected_rng=seq.capture_rng(device);del sp,sn,joint
    seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):streamed=seq.loss(model,positive,negative,recorder=lambda x:append(folder/'rng.jsonl',x))
    streamed_value=float(streamed.detach());got=torch.autograd.grad(streamed,params);del streamed
    finite=all(torch.isfinite(x).all().item() for x in reference+got)
    comparison=[dict(index=i,max_abs_error=float((x-y).abs().max()),allclose=bool(torch.allclose(x,y,atol=1e-5,rtol=1e-4))) for i,(x,y) in enumerate(zip(reference,got))]
    rng_ok=seq.same_rng(seq.capture_rng(device),expected_rng)
    event('JOINT_VS_SEQUENTIAL',joint_loss=joint_value,sequential_loss=streamed_value,finite=finite,RNG_equal=rng_ok,per_tensor=comparison)
    require(finite and all(x['allclose'] for x in comparison),'Full LED joint/sequential gradient equivalence failed')
    require(abs(joint_value-streamed_value)<=1e-5+1e-4*abs(joint_value) and rng_ok,'Loss/RNG equivalence failed')
    del reference,got
    optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    opt_before=fingerprint(optimizer.state_dict())
    accumulation_initial=seq.capture_rng(device)
    expected=[torch.zeros_like(p) for p in params]
    # Explicit sum of eight per-example gradients is the accumulation reference.
    for i,item in enumerate(batches[:8]):
        check()
        with torch.autocast('cuda',dtype=torch.float16):loss=seq.loss(model,**item,recorder=lambda x:append(folder/'rng.jsonl',x))
        grads=torch.autograd.grad(65536*(loss/8),params)
        for x,g in zip(expected,grads):x.add_(g)
        event('MANUAL_ACCUMULATION_MICROBATCH',index=i,loss=float(loss.detach()),all_finite=all(torch.isfinite(g).all().item() for g in grads))
        del grads,loss;memory.check()
    reference_rng=seq.capture_rng(device)
    event('MANUAL_ACCUMULATION_REFERENCE',gradients=gradstats(expected))
    require(all(torch.isfinite(g).all().item() for g in expected),'FP16 scaled accumulation nonfinite; stop without retry or changing scale')
    seq.restore_rng(accumulation_initial,device)
    scaler=torch.cuda.amp.GradScaler(init_scale=65536,growth_factor=2.,backoff_factor=.5,growth_interval=2000)
    for i,item in enumerate(batches[:8]):
        check()
        with torch.autocast('cuda',dtype=torch.float16):loss=seq.loss(model,**item,recorder=lambda x:append(folder/'rng.jsonl',x))
        scaler.scale(loss/8).backward();del loss;memory.check()
    require(all(p.grad is not None for p in params),'Missing native accumulated gradients')
    accumulated_equal=all(torch.equal(p.grad,g) for p,g in zip(params,expected))
    require(accumulated_equal and seq.same_rng(seq.capture_rng(device),reference_rng),'Native8 accumulation/RNG mismatch')
    scaler.unscale_(optimizer)
    event('NATIVE_FP16_ACCUMULATION',exact_scaled_sum=accumulated_equal,gradients=gradstats([p.grad for p in params]),memory=memory.summary())
    require(all(torch.isfinite(p.grad).all().item() for p in params),'Nonfinite unscaled gradients')
    scaler.update();optimizer.zero_grad(set_to_none=True);del expected
    # Native nonfinite skip must leave AdamW state and all adapter weights unchanged.
    skip=torch.cuda.amp.GradScaler(init_scale=65536,growth_interval=2000);skip.scale(torch.ones((),device=device))
    for p in params:p.grad=torch.zeros_like(p)
    params[0].grad.fill_(float('inf'));skip.step(optimizer);skip.update();optimizer.zero_grad(set_to_none=True)
    unchanged=all(torch.equal(p.detach().cpu(),weights[n]) for n,p in model.named_parameters() if p.requires_grad)
    require(skip.get_scale()==32768 and fingerprint(optimizer.state_dict())==opt_before and unchanged,'Overflow skip/optimizer integrity failed')
    require(seq._mode_signature(model)==modes,'Dropout/cache/adapter mode changed')
    check();memory.check()
    event('ALL_MANDATORY_DIAGNOSTICS_PASS',optimizer_unchanged=True,adapter_unchanged=unchanged,memory=memory.summary())
    return dict(status='PASS',full_pretrained_CUDA_FP16=True,lambda0_exact=True,joint_vs_sequential=comparison,
        native_eight_example_accumulation_exact=True,injected_overflow_skip_verified=True,successful_optimizer_updates=0,
        generation_calls=0,all_modes_unchanged=True,memory=memory.summary())

def train(model,c,a,args,folder,memory,check):
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainerCallback,get_linear_schedule_with_warmup
    import led_large_contrastive_sequential_v1 as seq
    from led_large_dynamic_amp_v2 import Recovery
    from stage_i_led_large_decoder_policy_v2 import configure_amp_scaler
    from stage_i_paired_pilot import gradient_record
    maximum=16
    cap=192
    data=collated(c,'cpu')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=1,num_training_steps=16)
    class Bounded(Recovery):
        def __init__(self):
            super().__init__(folder,3 if maximum==3 else 1250,check,write,append)
            self.maximum=maximum
        def microbatch(self):
            self.microbatches+=1
            require(self.microbatches<=cap,'Example microbatch ceiling exceeded')
        def state(self,scaler):
            return dict(super().state(scaler),pilot_maximum=maximum,example_microbatch_cap=cap)
    class Trainer(Seq2SeqTrainer):
        def create_accelerator_and_postprocess(self):
            super().create_accelerator_and_postprocess()
            configure_amp_scaler(self.accelerator,read(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json'))
        def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset)
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            require(not return_outputs,'No prediction/evaluation path in training')
            check()
            return seq.loss(model,**inputs,weight=0. if args.arm=='ce' else .1,recorder=lambda e:append(folder/'rng.jsonl',e))
        def training_step(self,model,inputs,num_items_in_batch=None):
            r=self.amp_recovery
            if not r.window:r.begin()
            r.window.append({k:{n:v.detach().cpu().clone() for n,v in b.items()} for k,b in inputs.items()})
            require(len(r.window)<=8,'Accumulation overflow');r.microbatch()
            return super().training_step(model,inputs,num_items_in_batch)
        def _save_checkpoint(self,model,trial):
            super()._save_checkpoint(model,trial)
            cp=folder/f'checkpoint-{self.state.global_step}'
            files={p.name:sha(p) for p in cp.iterdir() if p.is_file()}
            require({'adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'}<=set(files),'Incomplete checkpoint')
            write(cp/'checkpoint_integrity.json',dict(global_step=self.state.global_step,files=files,config_sha256=sha(CONFIG),runner_sha256=sha(__file__)))
    class Record(TrainerCallback):
        def on_pre_optimizer_step(self,aa,state,control,**kw):
            gradients=gradient_record(model,expect_later_A=True);require(len(gradients)==456,'Missing gradients')
            append(folder/'gradient_checks.jsonl',dict(step=state.global_step+1,gradients=gradients))
        def on_optimizer_step(self,aa,state,control,**kw):
            require(not trainer.accelerator.optimizer_step_was_skipped,'Unexpected outer skipped step')
            trainer.amp_recovery.commit(trainer.accelerator.scaler)
            require({int(v['step']) for v in trainer.optimizer.state.values()}=={state.global_step+1},'Optimizer step integrity failed')
        def on_step_end(self,aa,state,control,**kw):
            check();memory.check()
            require(state.global_step==trainer.amp_recovery.successful_updates,'Step accounting drift')
            append(folder/'progress.jsonl',dict(step=state.global_step,scale=trainer.accelerator.scaler.get_scale(),memory=memory.summary()))
    training_args=Seq2SeqTrainingArguments(output_dir=str(folder),per_device_train_batch_size=1,gradient_accumulation_steps=8,
        max_steps=maximum,learning_rate=2e-5,weight_decay=.01,fp16=True,bf16=False,max_grad_norm=1.,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},seed=42,data_seed=42,
        eval_strategy='no',save_strategy='steps',save_steps=3 if maximum==3 else 8,save_total_limit=None,
        remove_unused_columns=False,report_to=[],dataloader_num_workers=0,logging_steps=1)
    trainer=Trainer(model=model,args=training_args,train_dataset=data,data_collator=lambda x:x[0],optimizers=(opt,sched),callbacks=[Record()])
    trainer.model_accepts_loss_kwargs=False
    require(trainer.accelerator.gradient_accumulation_steps==1,'Unexpected double gradient-accumulation division')
    trainer.amp_recovery=Bounded()
    def clip(parameters,max_norm,norm_type=2):
        r=trainer.amp_recovery;require(len(r.window)==8,'Incomplete effective batch')
        def repeat():
            for inputs in r.window:r.microbatch();Seq2SeqTrainer.training_step(trainer,trainer.model,inputs)
        def skip():
            trainer.optimizer.step()
            require(trainer.accelerator.optimizer_step_was_skipped,'Native overflow did not skip')
        return r.finish(model,trainer.optimizer,trainer.accelerator.scaler,
            lambda:trainer.accelerator.unscale_gradients(trainer.optimizer),skip,repeat,max_norm,norm_type)
    trainer.accelerator.clip_grad_norm_=clip
    trainer.train(resume_from_checkpoint=None)
    require(trainer.state.global_step==maximum,'Incomplete pilot/gate')
    results=generate_probes(model,c,folder,memory,check,trainer)
    return dict(status='PASS',arm=args.arm,steps=maximum,generation_calls=len(results),memory=memory.summary(),
        amp=trainer.amp_recovery.state(trainer.accelerator.scaler),config_sha256=sha(CONFIG),runner_sha256=sha(__file__),
        quality_GO=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','diagnose','baseline','pilot'])
    ap.add_argument('--arm',choices=['ce','pair'],default='pair');ap.add_argument('--execute',action='store_true')
    for x in ('approval','decisions','numerical-receipt','gate-receipt','output'):ap.add_argument('--'+x,type=Path)
    args=ap.parse_args();c=preflight()
    if args.mode=='preflight':
        print(json.dumps(dict(status='CPU_FILES_PASS_GPU_NOT_AUTHORIZED',accepted_pairs=15,full_pretrained_CUDA_FP16='UNVERIFIED_UNTIL_CURRENT_DIAGNOSTIC')));return
    a=authorization(args,c)
    require(args.mode!='diagnose' or args.arm=='pair','One diagnostic attempt only, containing both objective checks')
    require(args.mode!='baseline' or args.arm=='ce','Exactly one untouched CE3 baseline, not two baseline arms')
    require(args.output is not None,'Fresh output namespace required')
    folder=args.output.resolve();suite=(ROOT/c['execution_root']).resolve()
    require(folder==suite/f'{args.mode}_{args.arm}' and not folder.exists(),'Exact fresh single-attempt namespace required')
    folder.mkdir(parents=True,exist_ok=False);write(folder/'AUTHORIZATION.json',a)
    budget=suite/'SUITE_BUDGET.json'
    if not budget.exists():write(budget,dict(first_start_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
    end=min(datetime.fromisoformat(a['deadline_utc']),datetime.fromisoformat(read(budget)['deadline_utc']))
    def check():require(datetime.now(timezone.utc)<end,'Hard authorization/suite time limit reached')
    check()
    import threading,traceback
    def timeout():
        try:write(folder/'TIMEOUT_STOP.json',dict(deadline_utc=end.isoformat()))
        finally:os._exit(124)
    timer=threading.Timer((end-datetime.now(timezone.utc)).total_seconds(),timeout);timer.daemon=True;timer.start()
    try:
        # No CUDA import/query occurs before every authorization and adjudication guard.
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3090','Exact RTX3090 required')
        profile=hardware()
        if args.mode!='diagnose':
            nr=read(args.numerical_receipt)
            require(nr.get('hardware')==profile and nr.get('config_sha256')==sha(CONFIG)
                and nr.get('runner_sha256')==sha(__file__),'Foreign/stale numerical evidence or GPU host')
        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
        from stage_i_led_large_decoder_policy_v2 import verify_backbone
        model,before,scope=load(c);check()
        with MemoryMeter(torch) as memory:
            report=diagnose(model,c,folder,memory,check) if args.mode=='diagnose' else (dict(status='PASS',arm='untouched_ce3',generation_calls=len(generate_probes(model,c,folder,memory,check)),steps=0) if args.mode=='baseline' else train(model,c,a,args,folder,memory,check))
            verify_backbone(model,before);check();memory.check()
            report.update(hardware=profile,config_sha256=sha(CONFIG),runner_sha256=sha(__file__),
                objective_sha256=sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),
                original_adapter_sha256=c['adapter_sha256'],researcher_decisions_sha256=sha(args.decisions))
            write(folder/'COMPLETION.json',report)
    except BaseException as error:
        write(folder/'FAILURE_STOP_NO_RETRY.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:timer.cancel()


def generate_probes(model,c,folder,memory,check,trainer=None):
    import torch
    from contextlib import contextmanager
    from peft import get_peft_model_state_dict
    from led_large_decoder_early_sanity_v2 import preserve_training_state,sanity_result
    from led_large_dynamic_amp_v2 import capture_rng,restore_rng
    from led_large_nbest_rng_v2 import rng_report
    from run_stage_i_d050_feasibility_v1 import fingerprint
    from transformers import AutoTokenizer,GenerationConfig
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    @contextmanager
    def baseline_guard():
        rng=capture_rng();modes={n:m.training for n,m in model.named_modules()};cache=model.config.use_cache
        state=fingerprint(get_peft_model_state_dict(model))
        try:yield
        finally:
            for n,m in model.named_modules():m.training=modes[n]
            model.config.use_cache=cache;restore_rng(rng)
            require(fingerprint(get_peft_model_state_dict(model))==state,'Untouched CE3 adapter changed')
            require(fingerprint(capture_rng())==fingerprint(rng),'Baseline RNG restore failed')
    source={x['business_id']:x for x in read(ROOT/c['TRAIN_source_packets'])}
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
    g=GenerationConfig(**read(ROOT/c['effective_generation_config']))
    results=[];continuity={}
    try:
        with (preserve_training_state(model,trainer,continuity) if trainer is not None else baseline_guard()):
            model.eval();require(all(not m.training for m in model.modules()),'Eval mode differs')
            for bid in c['probe']:
                check();case=source[bid];require(case['split']=='train','Foreign probe split')
                batch=CorrectedLabelCollator()([dict(input_ids=case['input_ids'],labels=[5,2])])
                batch={k:v.cuda() for k,v in batch.items() if k!='labels'}
                before=capture_rng();start=time.perf_counter()
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,generation_config=g)[0].cpu().tolist()
                    after=capture_rng();text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    record=dict(business_id=bid,token_ids=ids,text=text,seconds=time.perf_counter()-start,technical=sanity_result(ids,text,g.max_length),source_sha256=case['source_sha256'])
                    write(folder/'generations'/f'{bid}.json',record)
                finally:restore_rng(before)
                receipt=rng_report(before,after,capture_rng(),torch,fingerprint)
                write(folder/'generations'/f'{bid}_RNG.json',receipt)
                require(receipt['status']=='PASS' and record['technical']['status']=='PASS_TECHNICAL_ONLY','Generation/RNG failure')
                results.append(record);memory.check()
    finally:write(folder/'STATE_CONTINUITY.json',continuity if trainer is not None else dict(baseline_adapter_and_RNG_guard=True))
    return results

if __name__=='__main__':main()
