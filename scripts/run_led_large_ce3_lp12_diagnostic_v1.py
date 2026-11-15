"""One authorized TRAIN12 inference diagnostic. Only length_penalty changes.

No optimizer, backward pass, training, test/validation data reader or retry entry point.
The existing strict GPU ownership function is reused unchanged, including its PID check.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def require(ok,message):
    if not ok:raise ValueError(message)


def generation_b(original):
    require(original['num_beams']==4 and original['length_penalty']==1.0 and
            original['max_length']==768 and original['do_sample'] is False and
            original['num_return_sequences']==1,'Baseline decoding mismatch')
    new=dict(original,length_penalty=1.2)
    require([k for k in new if original[k]!=new[k]]==['length_penalty'],'More than one decoding change')
    return new


def verify_cases(cases,baseline,sample):
    require(len(cases)==len(baseline)==len(sample['case_ids'])==12,'Exactly locked TRAIN12 required')
    require([c['business_id'] for c in cases]==sample['case_ids'],'Locked order mismatch')
    require(len(set(sample['case_ids']))==12,'Duplicate case')
    indexed={b['business_id']:b for b in baseline}
    require(set(indexed)==set(sample['case_ids']),'Baseline identities mismatch')
    for c in cases:
        require(c['split']=='train' and c['condition']=='filtered','Non-TRAIN/Filtered input')
        require(hashlib.sha256(c['source'].encode()).hexdigest()==c['source_sha256'],'Source text hash differs')
        b=indexed[c['business_id']]
        require(b['split']=='train' and b['condition']=='filtered' and
                b['source_sha256']==c['source_sha256'],'Saved baseline source binding differs')
        require(b['attention_tokens']==len(c['input_ids']) and b['global_attention_tokens']==1 and
                b['source_padded_shape']==[1,((len(c['input_ids'])+1023)//1024)*1024],
                'Saved baseline mask/input geometry differs')
    require(sum(not c['captions'] for c in cases)==3,'Expected 9 captioned / 3 zero-caption')
    return indexed


def inventory():
    queries=[['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],
             ['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']]
    records=[]
    for command in queries:
        p=subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
        records.append(dict(command=command,stdout=p.stdout,stderr=p.stderr))
    return dict(utc=datetime.now(timezone.utc).isoformat(),python_PID=os.getpid(),
                namespace_fields=[s for s in Path('/proc/self/status').read_text().splitlines()
                                  if s.startswith(('Name:','Pid:','PPid:','NSpid:'))],queries=records)


def run():
    root=Path(sys.argv[1]).resolve();ops=root/'ops';data=root/'payload';out=root/'result'
    out.mkdir(exist_ok=False)
    calls=0;records=[];started=time.monotonic();model=None;memory=None;timer=None;stage='PAYLOAD'
    try:
        c=read(data/'CONFIG.json');a=read(ops/'AUTHORIZATION.json')
        for rel,digest in read(root/'TRANSFER_MANIFEST.json')['files'].items():
            p=(root/rel).resolve();require(p.is_relative_to(root) and sha(p)==digest,'Payload hash differs: '+rel)
        now=datetime.now(timezone.utc);start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
        require(start<=now<end and (end-start).total_seconds()<=48*3600,'Invalid/expired fresh approval')
        require(a['config_sha256']==sha(data/'CONFIG.json') and a['runner_sha256']==sha(Path(__file__)),
                'Exact new authorization binding mismatch')
        require(a['inference_approved'] and a['generation_calls_max']==12 and
                not a['training_authorized'] and not a['automatic_retry'],'Wrong authorization scope')
        require(os.environ.get('CONTAINER_ID')=='57481997','Unexpected instance')
        cases=read(data/'TRAIN_SOURCE_ONLY_CASES.json');baseline=read(data/'BASELINE_CANDIDATES.json')
        indexed=verify_cases(cases,baseline,read(data/'TRAIN_SAMPLE_MANIFEST.json'))
        g=read(data/'ORIGINAL_GENERATION_CONFIG.json');new=generation_b(g)
        require(read(data/'HISTORICAL_EFFECTIVE_GENERATION_PAIR.json')['single']==g,'Historical actual decoding mismatch')
        for x in baseline:
            bid=x['business_id'];raw=read(data/'N1_CALLS'/f'{bid}_single.json')
            require(read(data/'N1_CALLS'/f'{bid}_single_PASS.json')['status']=='PASS' and
                    raw['status']=='EVIDENCE_FLUSHED' and not raw['errors'] and
                    raw['raw_token_ids'][0]==x['single']['token_ids'],'Incomplete historical N1 receipt')
            require(x['checkpoint_sha256']==x['single']['checkpoint_sha256']==c['adapter_sha256'],
                    'Historical baseline adapter differs')
        ws=Path(c['parent_workspace']);cp=Path(c['checkpoint'])
        stage='PARENT_MODEL_CHECKPOINT_RUNTIME_PREFLIGHT'
        sys.path[:0]=[c['existing_cad_scripts'],str(ws/'scripts'),c['ownership_guard_scripts']]
        # This function only verifies immutable code/model/tokenizer/checkpoint/runtime source files.
        import run_led_large_cad_v1 as verification
        require(sha(Path(verification.__file__))==c['existing_preflight_runner_sha256'],'Parent verifier drift')
        parent_contract=verification.preflight(ws,cp)
        for path,digest in c['inherited_helper_bindings'].items():
            require(sha(path)==digest,'Inherited helper changed: '+path)
        require(sha(cp/'adapter_model.safetensors')==c['adapter_sha256'],'Adapter hash mismatch')
        runtime={k:importlib.metadata.version(k) for k in parent_contract['runtime_pins']}
        require(runtime==parent_contract['runtime_pins'],'Runtime pins differ')
        import run_led_large_boundary_global_feasibility_v1 as guard
        require(sha(Path(guard.__file__))==c['ownership_guard_sha256'],'Existing ownership guard changed')
        save(out/'OWNERSHIP_BEFORE_MODEL.json',inventory());stage='GPU_OWNERSHIP_BEFORE_MODEL'
        identity=guard.gpu_identity()
        require(identity['uuid']==c['GPU_uuid'],'Wrong dedicated GPU')
        save(out/'GENERATION_CONFIGS.json',dict(original=g,LP12=new,only_change={'length_penalty':[1.0,1.2]}))
        save(out/'PREFLIGHT_PASS.json',dict(runtime=runtime,source_cases=12,zero_caption_cases=3,
             existing_model_checkpoint_verifier='PASS',baseline_actual_generation_config_exact=True,
             historical_N1_raw_tokens_exact=12,identity=identity,weights_modified=False))
        def expire():
            save(out/'RUNTIME_STOP_NO_RETRY.json',dict(calls=calls,stage=stage));os._exit(124)
        timer=threading.Timer(min(3600,(end-datetime.now(timezone.utc)).total_seconds()),expire)
        timer.daemon=True;timer.start()
        import torch
        from transformers import LEDTokenizerFast,GenerationConfig
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        import stage_i_led_large_decoder_policy_v2 as parent
        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
        from led_large_dynamic_amp_v2 import capture_rng,restore_rng
        from run_stage_i_d050_feasibility_v1 import fingerprint
        from led_large_nbest_rng_v2 import recorded_generation
        from led_large_decoder_early_sanity_v2 import sanity_result
        require(parent.ROOT.resolve()==ws,'Wrong original model loader')
        pc=read(ws/'configs/stage_i_led_large_decoder_policy_v2.json')
        stage='ORIGINAL_CE3_MODEL_LOAD'
        model,backbone,_=parent.load_model(pc)
        weights=load_file(cp/'adapter_model.safetensors',device='cpu')
        require(set(weights)==set(get_peft_model_state_dict(model)),'Adapter keys mismatch')
        set_peft_model_state_dict(model,weights);model.eval()
        require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter activation mismatch')
        require(parent.generation(pc).to_dict()==g,'Actual generation default mismatch')
        tokenizer=LEDTokenizerFast.from_pretrained(ws/pc['source_tokenizer'],local_files_only=True)
        require(all(not m.training for m in model.modules()),'Eval mode mismatch')
        require(all(p.dtype==torch.float32 for p in model.parameters()),'Parameter precision mismatch')
        for case in cases:
            require(tokenizer.encode(case['source'],add_special_tokens=True,truncation=True,max_length=5000)==case['input_ids'],
                    'Frozen input tokenization mismatch')
        with MemoryMeter(torch) as memory:
            stage='GPU_OWNERSHIP_AFTER_MODEL_BEFORE_INFERENCE'
            save(out/'OWNERSHIP_AFTER_MODEL.json',inventory())
            require(guard.gpu_identity()==identity,'GPU identity/ownership changed after model load')
            memory.check()
            for case in cases:
                stage='LP12_INFERENCE_'+case['business_id']
                require(datetime.now(timezone.utc)<end and calls<12,'Deadline/call budget reached')
                save(out/f'OWNERSHIP_BEFORE_CALL_{calls:02d}.json',inventory())
                require(guard.gpu_identity()==identity,'GPU ownership failed before generation')
                ids=case['input_ids'];length=((len(ids)+1023)//1024)*1024
                x=torch.full((1,length),1,dtype=torch.long,device='cuda');x[0,:len(ids)]=torch.tensor(ids,device='cuda')
                mask=torch.zeros_like(x);mask[0,:len(ids)]=1;glob=torch.zeros_like(x);glob[0,0]=1
                def generate():
                    nonlocal calls
                    calls+=1
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                        return model.generate(input_ids=x,attention_mask=mask,global_attention_mask=glob,
                                              generation_config=GenerationConfig(**new))
                results,seconds=recorded_generation(folder=out,key=case['business_id']+'_LP12',
                    generate=generate,decode=lambda seq:tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    screen=sanity_result,torch=torch,memory=memory,capture=capture_rng,restore=restore_rng,
                    fingerprint=fingerprint,synchronize=torch.cuda.synchronize,
                    metadata=dict(business_id=case['business_id'],source_sha256=case['source_sha256'],checkpoint_sha256=c['adapter_sha256']),
                    expected_sequences=1,on_returned=lambda count:require(count==1,'Returned sequence budget differs'),
                    eval_check=lambda:all(not m.training for m in model.modules()))
                records.append(dict(business_id=case['business_id'],original=indexed[case['business_id']]['single'],
                                    LP12=results[0],seconds=seconds))
                save(out/'cases'/(case['business_id']+'.json'),records[-1])
                save(out/f'OWNERSHIP_AFTER_CALL_{calls:02d}.json',inventory())
                require(guard.gpu_identity()==identity,'GPU ownership failed after generation')
            require(calls==12,'Incomplete pilot')
            require(all(p.grad is None for p in model.parameters()),'Unexpected gradients')
            require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter changed')
            parent.verify_backbone(model,backbone);memory.check()
        save(out/'ALL_PAIRED_OUTPUTS.json',records)
        save(out/'COMPLETION.json',dict(status='COMPLETE_STOP_FOR_REVIEW',calls=calls,returned_sequences=len(records),
             seconds=time.monotonic()-started,optimizer_updates=0,no_optimizer_constructed=True,
             model_weights_unchanged=True,**memory.summary()))
    except BaseException as error:
        save(out/'FAILURE_STOP_NO_RETRY.json',dict(status='FAIL_STOP_NO_RETRY',stage=stage,error=str(error),
             traceback=traceback.format_exc(),generation_calls=calls,completed_cases=len(records),
             optimizer_updates=0,automatic_retry=False,quality_result='UNASSESSABLE_IF_NO_LP12_OUTPUTS',
             seconds=time.monotonic()-started))
        print(traceback.format_exc(),flush=True)
    finally:
        if timer is not None:timer.cancel()
        if memory is not None:save(out/'MEMORY_SAMPLES.json',dict(samples=memory.samples,failure=memory.failure))
        try:
            if 'cp' in locals():
                proof=read(cp/'checkpoint_integrity.json')
                save(out/'CHECKPOINT_PRESERVATION.json',dict(all_components_unchanged=all(sha(cp/n)==h for n,h in proof['files'].items()),
                     adapter_sha256=sha(cp/'adapter_model.safetensors'),optimizer_steps=0))
        except Exception as error:save(out/'POSTFLIGHT_FAILURE.json',dict(error=str(error)))
        print(json.dumps(dict(status='STOP',stage=stage,calls=calls,completed_cases=len(records))),flush=True)


if __name__=='__main__':run()
