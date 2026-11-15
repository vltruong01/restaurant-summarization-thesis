"""Fail-closed candidate runner: CPU preflight, separately approved diagnosis/gates/pilot.

Not GPU-qualified. No GPU mode is authorized by the offline preparation receipt.
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
CONFIG=PREP/'PRODUCTION_CONFIG_NOT_APPROVED.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def append(p,value):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()
def require(ok,message):
    if not ok:raise ValueError(message)

def preflight():
    c=read(CONFIG)
    for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'Bound file changed: '+rel)
    for dist,expected in c['runtime_pins'].items():require(importlib.metadata.version(dist)==expected,'Runtime differs: '+dist)
    require(c['objective']==dict(pair_weight=.1,margin=.1,temperature=1.),'Objective drift')
    require(c['pilot_updates']==16 and c['gate_updates']==3 and c['accumulation']==8,'Budget drift')
    require(len(c['fit'])==8 and len(c['probe'])==4 and not set(c['fit'])&set(c['probe']),'TRAIN split drift')
    require(set(c['pair_ids'])=={f'P{i:02d}' for i in range(1,17)},'Pair membership drift')
    for bid in c['fit']:
        p=read(PREP/'positives'/f'{bid}.json')
        require(p['split']=='train' and p['condition']=='filtered','Forbidden source split')
    return c

def authorization(args,c):
    # This function is called before importing Torch or querying a CUDA device.
    require(args.execute and args.approval,'New explicit GPU authorization required')
    a=read(args.approval)
    require(a.get('experiment')=='LED_LARGE_CONTRASTIVE_PAIR_QUALIFICATION_V1','Foreign authorization')
    require(a.get('GPU_execution_approved') is True and a.get('approval_source')=='researcher_message','Researcher GPU approval absent')
    require(a.get('config_sha256')==sha(CONFIG) and a.get('runner_sha256')==sha(__file__),'Authorization binding differs')
    require(a.get('objective_sha256')==sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),'Objective binding differs')
    require(a.get('preflight_sha256')==sha(PREP/'PREFLIGHT_MANIFEST.json'),'Preflight binding differs')
    require(args.mode in a.get('scopes',[]) and args.arm in ('ce','pair'),'Mode/arm not approved')
    require(a.get('protected_thesis_days_confirmed') is True,'Fresh researcher reserve confirmation required')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and timedelta(0)<end-start<=timedelta(hours=48),'Invalid48h authorization')
    require(start<=datetime.now(timezone.utc)<end,'Authorization not active')
    require(args.decisions and sha(args.decisions)==a.get('researcher_decisions_sha256'),'Exact adjudication receipt required')
    d=read(args.decisions)
    require(d.get('decider_role')=='researcher' and d.get('independently_reviewed') is True,'Independent adjudication absent')
    require(set(d.get('positives',{}))==set(c['fit']) and all(x.get('decision')=='ACCEPT' for x in d['positives'].values()),'All eight positives must be accepted')
    require(set(d.get('pairs',{}))==set(c['pair_ids']) and all(x.get('decision')=='ACCEPT' for x in d['pairs'].values()),'Exactly16 accepted pairs required')
    for bid,v in d['positives'].items():require(v['sha256']==sha(PREP/'positives'/f'{bid}.json'),'Foreign positive decision')
    for pid,v in d['pairs'].items():require(v['sha256']==sha(PREP/'pairs'/f'{pid}.json'),'Foreign pair decision')
    if args.mode!='diagnose':
        require(args.numerical_receipt and sha(args.numerical_receipt)==a.get('numerical_receipt_sha256'),'Full-model numerical evidence absent')
        nr=read(args.numerical_receipt)
        require(nr.get('status')=='PASS' and nr.get('full_pretrained_CUDA_FP16') is True
                and a.get('numerical_evidence_researcher_accepted') is True,'Numerical evidence not independently accepted')
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
    """Full-weight GPU checks only after separate diagnostic approval; no successful optimizer step."""
    import torch
    import led_large_contrastive_sequential_v1 as seq
    from run_stage_i_d050_feasibility_v1 import fingerprint
    params=tuple(p for p in model.parameters() if p.requires_grad)
    batches=collated(c,'cuda');sample=next(x for i,x in enumerate(batches[:8]) if c['fit'][i]=='EzJI5x25U5GgbYBWhu5ZQQ')
    p,n=sample['positive'],sample['negative'];device=params[0].device
    initial=seq.capture_rng(device)
    with torch.autocast('cuda',dtype=torch.float16):a=model(**p,use_cache=False).loss
    ga=torch.autograd.grad(a,params);seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):b=seq.loss(model,p,n,weight=0.)
    gb=torch.autograd.grad(b,params)
    require(torch.equal(a,b) and all(torch.equal(x,y) for x,y in zip(ga,gb)),'Lambda0 native CE mismatch')
    del a,b,ga,gb
    check();memory.check();initial=seq.capture_rng(device)
    with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
        sp=-model(**p,use_cache=False).loss.float();sn=-model(**n,use_cache=False).loss.float()
        joint=-sp+.1*torch.nn.functional.softplus(.1-sp+sn)
    reference=torch.autograd.grad(joint,params);del sp,sn,joint
    expected_rng=seq.capture_rng(device);seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):v=seq.loss(model,p,n,recorder=lambda e:append(folder/'rng.jsonl',e))
    got=torch.autograd.grad(v,params);del v
    require(all(torch.isfinite(x).all().item() for x in reference+got),'Nonfinite joint/replay diagnostic')
    max_error=max(float((x-y).abs().max()) for x,y in zip(reference,got))
    require(all(torch.allclose(x,y,atol=1e-5,rtol=1e-4) for x,y in zip(reference,got)),'Full LED gradient equivalence failed')
    require(seq.same_rng(seq.capture_rng(device),expected_rng),'Full LED RNG continuity failed')
    del reference,got
    optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=65536,growth_factor=2.,backoff_factor=.5,growth_interval=2000)
    state_before=fingerprint(optimizer.state_dict())
    for item in batches[:8]:
        check()
        with torch.autocast('cuda',dtype=torch.float16):loss=seq.loss(model,**item,recorder=lambda e:append(folder/'rng.jsonl',e))
        scaler.scale(loss/8).backward();memory.check()
    scaler.unscale_(optimizer)
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in params),'Native FP16 accumulation nonfinite; diagnostic stops without retry')
    scaler_before=scaler.state_dict();scaler.update() # Discard-only diagnostic, no successful optimizer step.
    optimizer.zero_grad(set_to_none=True)
    require(fingerprint(optimizer.state_dict())==state_before,'Diagnostic optimizer changed')
    # Explicitly injected overflow verifies native skip/backoff without any successful model update.
    skip_scaler=torch.cuda.amp.GradScaler(init_scale=65536,growth_interval=2000)
    skip_scaler.scale(torch.ones((),device='cuda'))
    weights=[p.detach().cpu().clone() for p in params]
    for p in params:p.grad=torch.zeros_like(p)
    params[0].grad.fill_(float('inf'))
    skip_scaler.step(optimizer);skip_scaler.update()
    require(skip_scaler.get_scale()==32768 and fingerprint(optimizer.state_dict())==state_before,'Native overflow handling differs')
    require(all(torch.equal(p.detach().cpu(),v) for p,v in zip(params,weights)),'Injected overflow changed weights')
    optimizer.zero_grad(set_to_none=True);check();memory.check()
    return dict(status='PASS',full_pretrained_CUDA_FP16=True,lambda0_exact=True,gradient_max_abs_error=max_error,
        native_eight_example_accumulation=True,injected_overflow_skip_verified=True,successful_optimizer_updates=0,
        original_scaler_state_at_finite_window=scaler_before,generation_calls=0,memory=memory.summary())

def train(model,c,a,args,folder,memory,check):
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainerCallback,get_linear_schedule_with_warmup
    import led_large_contrastive_sequential_v1 as seq
    from led_large_dynamic_amp_v2 import Recovery
    from stage_i_led_large_decoder_policy_v2 import configure_amp_scaler
    from stage_i_paired_pilot import gradient_record
    maximum=3 if args.mode=='gate' else 16
    cap=88 if maximum==3 else 192
    if args.mode=='pilot':
        require(args.gate_receipt and sha(args.gate_receipt)==a.get('gate_receipts',{}).get(args.arm),'Accepted same-arm gate absent')
        gate=read(args.gate_receipt)
        require(gate['status']=='PASS' and gate['steps']==3 and gate['arm']==args.arm,'Foreign gate')
        require(gate['config_sha256']==sha(CONFIG) and gate['runner_sha256']==sha(__file__),'Stale gate binding')
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
    # TRAIN-only generation, using unchanged generation settings and accepted continuity machinery.
    from led_large_decoder_early_sanity_v2 import preserve_training_state,sanity_result
    from transformers import AutoTokenizer,GenerationConfig
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    from led_large_nbest_rng_v2 import rng_report
    from led_large_dynamic_amp_v2 import capture_rng,restore_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    source={x['business_id']:x for x in read(ROOT/c['TRAIN_source_packets'])}
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
    g=GenerationConfig(**read(ROOT/c['effective_generation_config']))
    results=[];continuity={}
    try:
        with preserve_training_state(model,trainer,continuity):
            model.eval()
            for bid in c['probe']:
                check();case=source[bid]
                batch=CorrectedLabelCollator()([dict(input_ids=case['input_ids'],labels=[5,2])])
                batch={k:v.cuda() for k,v in batch.items() if k!='labels'}
                before=capture_rng()
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,generation_config=g)[0].cpu().tolist()
                    after=capture_rng();text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    record=dict(business_id=bid,token_ids=ids,text=text,technical=sanity_result(ids,text,g.max_length))
                    write(folder/'generations'/f'{bid}.json',record)
                finally:restore_rng(before)
                receipt=rng_report(before,after,capture_rng(),torch,fingerprint)
                write(folder/'generations'/f'{bid}_RNG.json',receipt)
                require(receipt.get('status')=='PASS','Eval RNG guard failed')
                require(record['technical']['status']=='PASS_TECHNICAL_ONLY','Technical generation collapse')
                results.append(record);memory.check()
    finally:write(folder/'STATE_CONTINUITY.json',continuity)
    return dict(status='PASS',arm=args.arm,steps=maximum,generation_calls=len(results),memory=memory.summary(),
        amp=trainer.amp_recovery.state(trainer.accelerator.scaler),config_sha256=sha(CONFIG),runner_sha256=sha(__file__),
        quality_GO=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','diagnose','gate','pilot'])
    ap.add_argument('--arm',choices=['ce','pair'],default='pair');ap.add_argument('--execute',action='store_true')
    for x in ('approval','decisions','numerical-receipt','gate-receipt','output'):ap.add_argument('--'+x,type=Path)
    args=ap.parse_args();c=preflight()
    if args.mode=='preflight':
        print(json.dumps(dict(status='CPU_FILES_PASS_GPU_NOT_AUTHORIZED',accepted_pairs=0,full_pretrained_CUDA_FP16='UNTESTED')));return
    a=authorization(args,c)
    require(args.mode!='diagnose' or args.arm=='pair','One diagnostic attempt only, containing both objective checks')
    require(args.output is not None,'Fresh output namespace required')
    folder=args.output.resolve();suite=(ROOT/'runs/CF_QUALIFICATION_V1').resolve()
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
        if args.mode=='pilot':
            require(args.gate_receipt and read(args.gate_receipt).get('hardware')==profile,'Gate GPU differs')
        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
        from stage_i_led_large_decoder_policy_v2 import verify_backbone
        model,before,scope=load(c);check()
        with MemoryMeter(torch) as memory:
            report=diagnose(model,c,folder,memory,check) if args.mode=='diagnose' else train(model,c,a,args,folder,memory,check)
            verify_backbone(model,before);check();memory.check()
            report.update(hardware=profile,config_sha256=sha(CONFIG),runner_sha256=sha(__file__),
                objective_sha256=sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),
                original_adapter_sha256=c['adapter_sha256'],researcher_decisions_sha256=sha(args.decisions))
            write(folder/'COMPLETION.json',report)
    except BaseException as error:
        write(folder/'FAILURE_STOP_NO_RETRY.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:timer.cancel()

if __name__=='__main__':main()
