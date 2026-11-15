"""One CAD pilot: 12 locked TRAIN cases, no optimizers, no retries.

Vanilla, alpha0 identity, alpha0.3 per case, plus first-case CAD exact repeat:
37 generate invocations maximum, 24 scientific paired outputs. A resource or
technical failure stops the sole run without a fallback or new run namespace.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib
import json
import os
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_cad_v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()


def require(ok,message):
    if not ok:raise ValueError(message)


def preflight(workspace,checkpoint):
    c=read(PREP/'CONFIG.json')
    for rel,h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==h,'CAD file changed: '+rel)
    for rel,h in c['parent_code_bindings'].items():
        require(sha(workspace/rel)==h,'Original parent binding changed: '+rel)
    for rel,h in c['model_files'].items():require(sha(workspace/rel)==h,'Model/tokenizer changed: '+rel)
    for name,h in c['runtime_source_hashes'].items():
        require(sha(importlib.import_module(name).__file__)==h,'Installed source drift: '+name)
    proof=read(checkpoint/'checkpoint_integrity.json')
    require(sha(checkpoint/'checkpoint_integrity.json')==c['checkpoint_receipt_sha256'] and proof['global_step']==750,'Wrong CE3 checkpoint')
    for name,h in proof['files'].items():
        require(Path(name).name==name and sha(checkpoint/name)==h,'Checkpoint component changed: '+name)
    require(sha(checkpoint/'adapter_model.safetensors')==c['adapter_sha256'],'Wrong adapter')
    return c


def binding():
    return dict(config_sha256=sha(PREP/'CONFIG.json'),runner_sha256=sha(Path(__file__)),
        cad_module_sha256=sha(ROOT/'scripts/led_large_cad_v1.py'),
        preflight_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))


def authorize(path,bindings):
    a=read(path);now=datetime.now(timezone.utc)
    require(a['experiment']=='LED_LARGE_CAD_V1' and a['inference_approved'] is True,'Inference not approved')
    require(a['bindings']==bindings,'Authorization binding differs')
    require(a['training_authorized'] is False and a['validation_access'] is False and a['held_out_access'] is False,
            'Forbidden scope')
    require(a['generation_calls_max']==37 and a['automatic_retry'] is False,'Budget differs')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.utcoffset()==end.utcoffset()==timedelta(0) and start<=now<end<=start+timedelta(hours=48),
            'Expired/invalid actual UTC authorization')
    return a,end


def run(args):
    ws=args.workspace.resolve();checkpoint=args.checkpoint.resolve();out=args.output.resolve()
    c=preflight(ws,checkpoint);bindings=binding();a,deadline=authorize(args.approval,bindings)
    require(not out.exists() and out!=checkpoint and checkpoint not in out.parents,'Fresh isolated output required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    runtime={k:importlib.metadata.version(k) for k in c['runtime_pins']}
    require(runtime==c['runtime_pins'],'Runtime dependency drift')
    sys.path.insert(0,str(ws/'scripts'))
    import torch
    from transformers import LEDTokenizerFast,GenerationConfig
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    import stage_i_led_large_decoder_policy_v2 as parent
    from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
    from led_large_dynamic_amp_v2 import capture_rng,restore_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    from led_large_nbest_rng_v2 import component_fingerprints
    from led_large_decoder_early_sanity_v2 import sanity_result
    from led_large_cad_v1 import CADSession,contextless_source
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required')
    require(torch.cuda.get_device_name(0)==a['GPU_name'],'Actual GPU differs from authorization')
    out.mkdir(parents=True,exist_ok=False);save(out/'AUTHORIZATION.json',a);save(out/'BINDINGS.json',bindings)
    calls=0;records=[];memory=None;started=time.perf_counter()
    def timeout_stop():
        save(out/'DEADLINE_STOP_NO_RETRY.json',dict(generation_calls=calls,automatic_retry=False))
        os._exit(124)
    timer=threading.Timer(min(3600,max(.001,(deadline-datetime.now(timezone.utc)).total_seconds())),timeout_stop)
    timer.daemon=True;timer.start()
    def check():
        require(datetime.now(timezone.utc)<deadline and time.perf_counter()-started<3600,'Execution cap expired')
        if memory is not None:memory.check()
    try:
        require(parent.ROOT.resolve()==ws,'Wrong parent import')
        pc=read(ws/'configs/stage_i_led_large_decoder_policy_v2.json')
        # Reuse the EXACT verified FP32-storage/FP16-autocast parent loader.
        model,backbone,_=parent.load_model(pc)
        weights=load_file(checkpoint/'adapter_model.safetensors',device='cpu')
        require(set(weights)==set(get_peft_model_state_dict(model)),'LoRA inventory differs')
        set_peft_model_state_dict(model,weights);model.eval()
        require(all(torch.equal(v.detach().cpu(),weights[n]) for n,v in get_peft_model_state_dict(model).items()),'Adapter not loaded exactly')
        require(all(not p.training for p in model.modules()),'Eval state drift')
        require(len(model.get_base_model().led.encoder.layers)==12,'Expected12 pinned encoder RNG draws')
        require(all(p.dtype==torch.float32 for p in model.parameters()),'Parameter storage precision drift')
        tokenizer=LEDTokenizerFast.from_pretrained(ws/pc['source_tokenizer'],local_files_only=True)
        g=read(PREP/'ORIGINAL_GENERATION_CONFIG.json')
        require(parent.generation(pc).to_dict()==g,'Effective generation config drift')
        cases=read(PREP/'TRAIN_SOURCE_ONLY_CASES.json')
        require(len(cases)==12 and sum(not x['captions'] for x in cases)==3,'TRAIN sample drift')
        for case in cases:require(case['split']=='train' and case['condition']=='filtered','Non-TRAIN source prohibited')
        save(out/'ENVIRONMENT.json',dict(runtime=runtime,GPU=torch.cuda.get_device_name(0),
            GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda,
            tf32=torch.backends.cuda.matmul.allow_tf32,precision='original_FP32_storage_FP16_autocast',
            model_config_use_cache=model.config.use_cache,generation_use_cache=g['use_cache'],
            no_optimizer_constructed=True,no_backward=True,validation_access=False,held_out_access=False))

        def batch(ids):
            length=((len(ids)+1023)//1024)*1024
            x=torch.full((1,length),1,dtype=torch.long,device='cuda');x[0,:len(ids)]=torch.tensor(ids,device='cuda')
            mask=torch.zeros_like(x);mask[0,:len(ids)]=1;glob=torch.zeros_like(x);glob[0,0]=1
            return dict(input_ids=x,attention_mask=mask,global_attention_mask=glob)

        with MemoryMeter(torch) as memory:
            check()
            for case_index,case in enumerate(cases):
                full=batch(case['input_ids'])
                ctx_text=contextless_source(case['source'])
                ctx_ids=tokenizer.encode(ctx_text,add_special_tokens=True)
                ctx=batch(ctx_ids)
                result=dict(business_id=case['business_id'],source_sha256=case['source_sha256'],
                    source_input_ids_sha256=hashlib.sha256(json.dumps(case['input_ids']).encode()).hexdigest(),
                    captions=len(case['captions']),contextless_text=ctx_text,contextless_ids=ctx_ids,
                    full_shape=list(full['input_ids'].shape),contextless_shape=list(ctx['input_ids'].shape))

                def generate(label,alpha):
                    nonlocal calls
                    check();require(calls<37,'Generation budget exceeded')
                    calls+=1;before=capture_rng();raw=None;errors=[];session=None;elapsed=None;rows=[]
                    save(out/'calls'/f'{case_index:02d}_{label}_STARTED.json',dict(call=calls,business_id=case['business_id'],alpha=alpha))
                    try:
                        torch.cuda.synchronize();t=time.perf_counter()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                            cm=contextlib.nullcontext() if alpha is None else CADSession(model,ctx,alpha,
                                on_step=lambda row:(append(out/'calls'/f'{case_index:02d}_{label}_steps.jsonl',row),check()))
                            with cm as session:
                                tokens=model.generate(**full,generation_config=GenerationConfig(**g))
                        torch.cuda.synchronize();elapsed=time.perf_counter()-t
                        raw=tokens.detach().cpu().tolist()[0]
                    except BaseException as e:
                        errors.append(type(e).__name__+': '+str(e));generation_trace=traceback.format_exc()
                    after=capture_rng();restore_rng(before);restored=capture_rng()
                    expected=dict(before);generator=torch.Generator(device='cpu');generator.set_state(before['cpu'])
                    for _ in range(12 if alpha is None else 24):torch.rand([],generator=generator)
                    expected['cpu']=generator.get_state()
                    b=component_fingerprints(before,fingerprint);actual=component_fingerprints(after,fingerprint)
                    e=component_fingerprints(expected,fingerprint);r=component_fingerprints(restored,fingerprint)
                    rng=dict(before=b,expected_after=e,actual_after=actual,restored=r,
                        expected_CPU_draws=12 if alpha is None else 24,transition_exact=actual==e,restoration_exact=b==r)
                    if actual!=e or b!=r:errors.append('Unexpected RNG transition/restoration')
                    text=tokenizer.decode(raw,skip_special_tokens=True,clean_up_tokenization_spaces=False) if raw else None
                    screen=sanity_result(raw,text,768) if raw else None
                    record=dict(raw_token_ids=raw,text=text,seconds=elapsed,RNG=rng,technical_screen=screen,
                        errors=errors,generation_traceback=generation_trace if errors and 'generation_trace' in locals() else None,
                        memory=memory.summary(),CAD_steps=session.steps if session is not None else 0)
                    save(out/'calls'/f'{case_index:02d}_{label}.json',record)
                    require(not errors,'Generation/RNG stop: '+'; '.join(errors))
                    require(screen['status']=='PASS_TECHNICAL_ONLY','Generation collapse')
                    require(raw[0]==2 and raw[-1]==2 and len(raw)<=768,'EOS/start/length violation')
                    require(all(not m.training for m in model.modules()),'Eval state changed')
                    require(all(p.grad is None for p in model.parameters()),'Unexpected gradients')
                    check();return record

                result['vanilla']=generate('vanilla',None)
                result['alpha0']=generate('alpha0',0.)
                save(out/'cases'/f'{case_index:02d}_IDENTITY.json',dict(
                    raw_ids_exact=result['vanilla']['raw_token_ids']==result['alpha0']['raw_token_ids'],
                    text_exact=result['vanilla']['text']==result['alpha0']['text']))
                require(result['vanilla']['raw_token_ids']==result['alpha0']['raw_token_ids'] and
                    result['vanilla']['text']==result['alpha0']['text'],'Alpha0 does not reproduce vanilla: STOP')
                result['cad']=generate('cad',.3)
                if case_index==0:
                    result['cad_repeat']=generate('cad_repeat',.3)
                    require(result['cad']['raw_token_ids']==result['cad_repeat']['raw_token_ids'], 'CAD deterministic repeat differs')
                records.append(result);save(out/'cases'/f'{case_index:02d}.json',result)
                print(f'CAD: {len(records)}/12 paired TRAIN cases saved',flush=True)
                del full,ctx
            require(calls==37 and len(records)==12,'Incomplete pilot')
            require(all(torch.equal(v.detach().cpu(),weights[n]) for n,v in get_peft_model_state_dict(model).items()),'Adapter changed')
            parent.verify_backbone(model,backbone);check()
        preflight(ws,checkpoint)  # Original file bytes unchanged after inference.
        save(out/'PAIRED_OUTPUTS.json',records)
        save(out/'COMPLETION.json',dict(status='TECHNICAL_COMPLETE_QUALITY_REVIEW_REQUIRED',generation_calls=calls,
            businesses=12,scientific_outputs=24,alpha0_outputs=12,determinism_repeat_outputs=1,
            optimizer_updates=0,backward_calls=0,weights_unchanged=True,seconds=time.perf_counter()-started,
            **memory.summary()))
    except BaseException as e:
        save(out/'FAILURE_STOP_NO_RETRY.json',dict(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),
            generation_calls=calls,complete_pairs=len(records),optimizer_updates=0,automatic_retry=False,
            CAD_quality='UNESTABLISHED',seconds=time.perf_counter()-started));raise
    finally:
        timer.cancel()
        if memory is not None:save(out/'MEMORY_SAMPLES.json',dict(samples=memory.samples,failure=memory.failure))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['preflight','pilot']);ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--approval',type=Path);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if args.phase=='preflight':preflight(args.workspace.resolve(),args.checkpoint.resolve());print(binding());return
    require(args.execute and args.approval and args.output,'Explicit inference-only execution required');run(args)


if __name__=='__main__':main()
