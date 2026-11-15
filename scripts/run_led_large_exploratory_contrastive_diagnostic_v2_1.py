"""V2.1 diagnostic-only instrumentation. OFFLINE preparation; fresh GPU approval required."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.metadata,json,os,time
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
CONFIG=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1/CONFIG.json'


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


def preflight():
    c=read(CONFIG)
    for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'Bound file differs: '+rel)
    import run_led_large_exploratory_contrastive_v2 as preserved
    old=preserved.preflight()
    for key in c['unchanged_scientific_fields']:
        require(c['science'][key]==old[key],'Unrequested scientific change: '+key)
    require(c['lambda0_thresholds']==dict(atol=1e-5,rtol=1e-4,reference='native',equal_nan=False),'Threshold drift')
    from led_large_contrastive_diagnostic_evidence_v2_1 import ATOL,RTOL
    require(ATOL==1e-5 and RTOL==1e-4,'Helper threshold drift')
    require(c['scopes']==['diagnose'] and c['training_authorized'] is False
        and c['generation_calls_max']==0 and c['successful_optimizer_updates_max']==0,'Diagnostic-only scope drift')
    return dict(c['science'],**{k:v for k,v in c.items() if k!='science'})


def authorization(args,c):
    require(args.execute and args.approval,'Fresh explicit V2.1 diagnostic-only authorization required before CUDA')
    a=read(args.approval)
    require(a.get('experiment')==c['experiment'],'Foreign/failed V2 approval cannot authorize V2.1')
    require(a.get('GPU_execution_approved') is True and a.get('approval_source')=='researcher_message'
        and bool(a.get('researcher_explicit_authorization')),'Diagnostic GPU approval absent')
    require(args.mode=='diagnose' and a.get('scopes')==['diagnose'],'Diagnostic only; no training or generation modes')
    require(a.get('training_authorized') is False and a.get('baseline_inference_authorized') is False
        and a.get('automatic_continuation') is False,'No chained continuation allowed')
    require(a.get('successful_optimizer_updates_max')==0 and a.get('generation_calls_max')==0
        and a.get('diagnostic_attempts_max')==1,'Diagnostic budget differs')
    for key,path in [('config_sha256',CONFIG),('runner_sha256',Path(__file__)),
        ('helper_sha256',ROOT/c['helper']),('objective_sha256',ROOT/c['objective_module']),
        ('preflight_sha256',CONFIG.parent/'PREFLIGHT_MANIFEST.json')]:
        require(a.get(key)==sha(path),'Approval binding differs: '+key)
    require(a.get('researcher_decisions_sha256')==sha(ROOT/c['adjudication_receipt']),'Locked adjudication binding differs')
    require(a.get('output_namespace')==c['output_namespace'],'Fresh V2.1 namespace required')
    require(a.get('authorized_utc') is not None and a.get('deadline_utc') is not None,'Fresh UTC authorization/deadline absent')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and timedelta(0)<end-start<=timedelta(hours=48)
        and start<=datetime.now(timezone.utc)<end,'Authorization expired or invalid')
    require(a.get('stop_for_independent_diagnostic_review') is True,'Independent review stop must be explicit')
    return a

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
    from led_large_contrastive_diagnostic_evidence_v2_1 import (ATOL,RTOL,save_path_snapshot,
        lambda0_comparison,batch_signature,model_context,diagnostic_receipt)
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    require(len(names)==456,'All456 trainable tensor names required')
    from transformers.models.led.modeling_led import shift_tokens_right
    shifted={}
    def observe_prefix(module,args,kwargs):shifted['ids']=kwargs['input_ids'].detach().cpu().tolist()
    initial=seq.capture_rng(device)
    seq.restore_rng(initial,device)
    native_start=seq.capture_rng(device);native_mode=model_context(model);native_batch=batch_signature(positive)
    hook=model.base_model.model.led.decoder.register_forward_pre_hook(observe_prefix,with_kwargs=True)
    try:
        with torch.autocast('cuda',dtype=torch.float16):
            output=model(**positive,use_cache=False);native=output.loss
            manual_ce=torch.nn.functional.cross_entropy(output.logits.reshape(-1,output.logits.shape[-1]),positive['labels'].reshape(-1),ignore_index=-100)
    finally:hook.remove()
    expected_prefix=shift_tokens_right(positive['labels'],1,2).detach().cpu().tolist()
    native_prefix=shifted['ids']==expected_prefix;manual_equal=torch.equal(native,manual_ce)
    event('DECODER_PREFIX_AND_REDUCTION',own_positive_prefix=native_prefix,decoder_ids=shifted['ids'],native_loss=float(native.detach()),manual_token_CE=float(manual_ce.detach()),headers_EOS_included=True)
    del manual_ce,output
    ga=torch.autograd.grad(native,params)
    native_end=seq.capture_rng(device);native_mode_end=model_context(model);native_batch_end=batch_signature(positive)
    native_payload,native_receipt=save_path_snapshot(folder/'lambda0','native',names,ga,native,
        dict(RNG_start=seq.fingerprints(native_start),RNG_end=seq.fingerprints(native_end),
             mode_start=native_mode,mode_end=native_mode_end,batch_start=native_batch,batch_end=native_batch_end,
             own_prefix=native_prefix,decoder_ids=shifted['ids'],manual_CE_reduction=manual_equal))
    seq.restore_rng(initial,device)
    delegated_start=seq.capture_rng(device);delegated_mode=model_context(model);delegated_batch=batch_signature(positive)
    hook=model.base_model.model.led.decoder.register_forward_pre_hook(observe_prefix,with_kwargs=True)
    try:
        with torch.autocast('cuda',dtype=torch.float16):delegated=seq.loss(model,positive,negative,weight=0.)
    finally:hook.remove()
    delegated_prefix=shifted['ids']==expected_prefix
    gb=torch.autograd.grad(delegated,params)
    delegated_end=seq.capture_rng(device);delegated_mode_end=model_context(model);delegated_batch_end=batch_signature(positive)
    delegated_payload,delegated_receipt=save_path_snapshot(folder/'lambda0','delegated',names,gb,delegated,
        dict(RNG_start=seq.fingerprints(delegated_start),RNG_end=seq.fingerprints(delegated_end),
             mode_start=delegated_mode,mode_end=delegated_mode_end,batch_start=delegated_batch,batch_end=delegated_batch_end,
             own_prefix=delegated_prefix,decoder_ids=shifted['ids']))
    preconditions=dict(rng_start_equal=seq.same_rng(native_start,delegated_start),
        rng_end_equal=seq.same_rng(native_end,delegated_end),mode_start_equal=native_mode==delegated_mode,
        mode_native_unchanged=native_mode==native_mode_end,mode_delegated_unchanged=delegated_mode==delegated_mode_end,
        batch_equal=native_batch==delegated_batch,batch_native_unchanged=native_batch==native_batch_end,
        batch_delegated_unchanged=delegated_batch==delegated_batch_end,native_own_prefix=native_prefix,
        delegated_own_prefix=delegated_prefix,manual_CE_reduction=manual_equal)
    lambda0=lambda0_comparison(folder/'lambda0',native_payload,delegated_payload,native_receipt,delegated_receipt,preconditions)
    event('LAMBDA0_NATIVE_CE',lambda0_within_tolerance=lambda0['lambda0_within_tolerance'],
        bit_exact=lambda0['lambda0_bit_exact'],preconditions=preconditions,atol=ATOL,rtol=RTOL,
        evidence='lambda0/LAMBDA0_COMPARISON.json')
    require(lambda0['status']=='PASS_WITHIN_TOLERANCE','Lambda0 nonfinite/out-of-tolerance/context failure; raw evidence preserved')
    del native,delegated,ga,gb,native_payload,delegated_payload
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
    summary=memory.summary()
    gates=dict(lambda0_within_tolerance=lambda0['lambda0_within_tolerance'],
        lambda0_preconditions=lambda0['preconditions_pass'],joint_vs_sequential=True,
        negative_own_prefix=True,eight_example_FP16_accumulation=True,
        AMP_skip_optimizer_adapter_integrity=True,RNG_continuity=True,
        model_modes_unchanged=True,memory_headroom=summary['headroom_fraction']>=.15)
    report=diagnostic_receipt(lambda0,gates,summary)
    report.update(joint_vs_sequential=comparison,native_eight_example_accumulation_exact=True,
                  injected_overflow_skip_verified=True,all_modes_unchanged=True)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','diagnose'])
    ap.add_argument('--execute',action='store_true');ap.add_argument('--approval',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args();c=preflight()
    if args.mode=='preflight':
        print(json.dumps(dict(status='OFFLINE_FILES_PASS_NO_GPU_AUTHORIZATION',full_LED_CUDA_FP16='UNVERIFIED')));return
    a=authorization(args,c)
    expected=(ROOT/c['output_namespace']).resolve()
    require(args.output is not None and args.output.resolve()==expected and not expected.parent.exists(),'Exact fresh V2.1 single-attempt namespace required')
    # CPU verification before importing/querying CUDA. These are not replaced by earlier host receipts.
    require(sha(ROOT/'VAST_SHA256_MANIFEST.json')==c['host_parent_manifest_sha256'],'Pinned parent manifest changed')
    for rel,digest in read(ROOT/'VAST_SHA256_MANIFEST.json')['files'].items():require(sha(ROOT/rel)==digest,'Parent workspace changed: '+rel)
    require(sha(ROOT/c['host_backbone_path'])==c['host_backbone_sha256'],'Backbone changed')
    import stage_i_led_large_decoder_policy_v2 as parent
    parent.preflight()
    require(importlib.metadata.version('huggingface-hub')=='0.36.2','Pinned hub runtime differs')
    folder=expected;folder.mkdir(parents=True,exist_ok=False);write(folder/'AUTHORIZATION.json',a)
    end=min(datetime.fromisoformat(a['deadline_utc']),datetime.now(timezone.utc)+timedelta(hours=1))
    def check():require(datetime.now(timezone.utc)<end,'Hard diagnostic authorization/time limit')
    import threading,traceback
    def timeout():
        try:write(folder/'TIMEOUT_STOP_NO_RETRY.json',dict(deadline_utc=end.isoformat(),automatic_retry=False))
        finally:os._exit(124)
    timer=threading.Timer(max(0,(end-datetime.now(timezone.utc)).total_seconds()),timeout);timer.daemon=True;timer.start()
    memory=None
    try:
        check()
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1
            and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3090','Exact approved single RTX3090 required')
        profile=hardware()
        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
        model,before,scope=load(c);check()
        with MemoryMeter(torch) as memory:
            report=diagnose(model,c,folder,memory,check)
            parent.verify_backbone(model,before);memory.check();check()
            report.update(hardware=profile,config_sha256=sha(CONFIG),runner_sha256=sha(__file__),
                helper_sha256=sha(ROOT/c['helper']),objective_sha256=sha(ROOT/c['objective_module']),
                original_adapter_sha256=c['adapter_sha256'],researcher_decisions_sha256=sha(ROOT/c['adjudication_receipt']))
        # No PASS receipt until MemoryMeter exit and final evidence persistence succeed.
        check()
        require(memory.failure is None,'Memory sampler failure; no completion')
        write(folder/'MEMORY_EVIDENCE.json',dict(summary=memory.summary(),samples=memory.samples,measurement_failure=memory.failure))
        write(folder/'COMPLETION.json',report)
        print('DIAGNOSTIC COMPLETE. STOP for independent review. No baseline, CE or contrastive continuation.',flush=True)
    except BaseException as error:
        write(folder/'FAILURE_STOP_NO_RETRY.json',dict(error=repr(error),traceback=traceback.format_exc(),
            automatic_retry=False,successful_optimizer_updates=0,generation_calls=0,diagnostic_complete=False))
        raise
    finally:
        timer.cancel()
        if memory is not None and not (folder/'MEMORY_EVIDENCE.json').exists():
            try:write(folder/'MEMORY_EVIDENCE.json',dict(summary=memory.summary(),samples=memory.samples,measurement_failure=memory.failure))
            except Exception as error:
                write(folder/'MEMORY_EVIDENCE_UNAVAILABLE.json',dict(error=repr(error),headroom_verified=False))

if __name__=='__main__':main()
