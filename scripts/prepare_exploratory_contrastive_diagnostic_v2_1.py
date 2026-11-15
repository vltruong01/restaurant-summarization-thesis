"""OFFLINE-only construction. Never connects to a host or queries a GPU."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2'
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1'
RUNNER=ROOT/'scripts/run_led_large_exploratory_contrastive_diagnostic_v2_1.py'
HELPER=ROOT/'scripts/led_large_contrastive_diagnostic_evidence_v2_1.py'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

LAMBDA0_BLOCK='''    from led_large_contrastive_diagnostic_evidence_v2_1 import (ATOL,RTOL,save_path_snapshot,
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
'''

RETURN_BLOCK='''    summary=memory.summary()
    gates=dict(lambda0_within_tolerance=lambda0['lambda0_within_tolerance'],
        lambda0_preconditions=lambda0['preconditions_pass'],joint_vs_sequential=True,
        negative_own_prefix=True,eight_example_FP16_accumulation=True,
        AMP_skip_optimizer_adapter_integrity=True,RNG_continuity=True,
        model_modes_unchanged=True,memory_headroom=summary['headroom_fraction']>=.15)
    report=diagnostic_receipt(lambda0,gates,summary)
    report.update(joint_vs_sequential=comparison,native_eight_example_accumulation_exact=True,
                  injected_overflow_skip_verified=True,all_modes_unchanged=True)
    return report
'''

AUTH='''
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
'''

PREFLIGHT='''
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
'''

MAIN='''
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
'''

def main():
    assert not OUT.exists() and not RUNNER.exists(),'Versioned V2.1 output exists; preserve it.'
    source=(ROOT/'scripts/run_led_large_exploratory_contrastive_v2.py').read_text(encoding='utf-8')
    tree=ast.parse(source);functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    diagnostic=functions['diagnose']
    start=diagnostic.index('    initial=seq.capture_rng(device)')
    end=diagnostic.index('    check();memory.check();initial=seq.capture_rng(device)',start)
    diagnostic=diagnostic[:start]+LAMBDA0_BLOCK+diagnostic[end:]
    start=diagnostic.index("    return dict(status='PASS'")
    diagnostic=diagnostic[:start]+RETURN_BLOCK.rstrip()
    header='''"""V2.1 diagnostic-only instrumentation. OFFLINE preparation; fresh GPU approval required."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.metadata,json,os,time
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
CONFIG=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1/CONFIG.json'
'''
    selected=['read','sha','json_safe','write','append','require','collated','load','hardware']
    new=header+'\n\n'+'\n\n'.join(functions[n] for n in selected)+'\n\n'+PREFLIGHT+'\n'+AUTH+'\n'+diagnostic+'\n'+MAIN+"\nif __name__=='__main__':main()\n"
    ast.parse(new);OUT.mkdir(parents=True);RUNNER.write_text(new,encoding='utf-8')
    (OUT/'V2_TO_V2_1_DIAGNOSTIC.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='V2 executed preserved',tofile='V2.1 diagnostic-only offline')),encoding='utf-8')
    old=read(OLD/'CONFIG.json')
    science_keys=['objective','runtime_pins','fit','probe','pair_ids','excluded_pairs','accepted_pairs','pilot_updates','accumulation',
        'learning_rate','warmup_steps','optimizer','betas','epsilon','weight_decay','max_grad_norm','sampler','initialization',
        'adapter','adapter_sha256','TRAIN_source_packets','effective_generation_config','AMP','headroom_min','checkpoint_policy']
    files=dict(old['files'])
    for p in [OLD/'CONFIG.json',OLD/'PREFLIGHT_MANIFEST.json',OLD/'DELIVERY_RECEIPT.json',RUNNER,HELPER,
              ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_DIAGNOSTIC_REVIEW.zip']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    workspace=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
    c=dict(experiment='LED_LARGE_EXPLORATORY_CONTRASTIVE_DIAGNOSTIC_V2_1',scopes=['diagnose'],
        output_namespace='runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY/diagnose_pair',
        lambda0_thresholds=dict(atol=1e-5,rtol=1e-4,reference='native',equal_nan=False),
        unchanged_scientific_fields=science_keys,science={k:old[k] for k in science_keys},files=files,
        helper=HELPER.relative_to(ROOT).as_posix(),objective_module='scripts/led_large_contrastive_sequential_v1.py',
        adjudication_receipt='outputs/stage_i/led_large_exploratory_contrastive_v2/RESEARCHER_ADJUDICATION_AMENDMENT.json',
        GPU_execution_approved=False,training_authorized=False,baseline_inference_authorized=False,automatic_continuation=False,
        generation_calls_max=0,successful_optimizer_updates_max=0,diagnostic_attempts_max=1,max_wall_clock_hours=1,
        host_parent_manifest_sha256=sha(workspace/'VAST_SHA256_MANIFEST.json'),
        host_backbone_path='assets/models/led-large-16384/pytorch_model.bin',
        host_backbone_sha256='b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081',
        stop_for_independent_diagnostic_review=True,prior_failed_approval_reusable=False,prior_failed_namespace_reusable=False)
    write(OUT/'CONFIG.json',c)
    pre=dict(files);pre[(OUT/'CONFIG.json').relative_to(ROOT).as_posix()]=sha(OUT/'CONFIG.json')
    write(OUT/'PREFLIGHT_MANIFEST.json',dict(files=pre))
    a=dict(experiment=c['experiment'],GPU_execution_approved=False,approval_source=None,researcher_explicit_authorization=None,
        scopes=['diagnose'],training_authorized=False,baseline_inference_authorized=False,automatic_continuation=False,
        successful_optimizer_updates_max=0,generation_calls_max=0,diagnostic_attempts_max=1,
        config_sha256=sha(OUT/'CONFIG.json'),runner_sha256=sha(RUNNER),helper_sha256=sha(HELPER),
        objective_sha256=sha(ROOT/c['objective_module']),preflight_sha256=sha(OUT/'PREFLIGHT_MANIFEST.json'),
        researcher_decisions_sha256=sha(ROOT/c['adjudication_receipt']),authorized_utc=None,deadline_utc=None,
        output_namespace=c['output_namespace'],stop_for_independent_diagnostic_review=True,
        notice='PENDING fresh explicit diagnostic-only approval. Do not copy timestamps or approval from failed V2. No baseline/training follows PASS.')
    write(OUT/'DIAGNOSTIC_AUTHORIZATION_PENDING.json',a)
    write(OUT/'OFFLINE_PREPARATION_RECEIPT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization_scope='OFFLINE instrumentation correction only',GPU_execution_approved=False,remote_connections=0,
        GPU_operations=0,optimizer_updates=0,generation_calls=0,V2_evidence_preserved=True))
    print(json.dumps(dict(runner=str(RUNNER),config_sha256=sha(OUT/'CONFIG.json'),runner_sha256=sha(RUNNER),helper_sha256=sha(HELPER))))

if __name__=='__main__':main()
