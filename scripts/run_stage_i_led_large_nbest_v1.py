"""Approval-gated inference only; never trains, edits candidates, or calls a judge."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import traceback
import threading
from stage_i_led_large_nbest_v1 import (EXPERIMENT,read,write,sha,require,validate_case,
    generation_pair,compare_top1,lexical_diversity,selection_packet,diversity_gate)


def payload(config_path):
    root=config_path.resolve().parents[1];c=read(config_path)
    p=root/c['preparation'];manifest=read(p/'PREFLIGHT_MANIFEST.json')
    for rel,h in manifest['files'].items():
        require(sha(root/rel)==h,'N-best payload changed: '+rel)
    return root,p,c,dict(config_sha256=sha(config_path),runner_sha256=sha(Path(__file__)),
                        preflight_sha256=sha(p/'PREFLIGHT_MANIFEST.json'))


def authorization(path,phase,binding,execute):
    require(execute and path,'OFFLINE ONLY: new explicit researcher inference authorization required')
    a=read(path);now=datetime.now(timezone.utc)
    require(a['experiment']==EXPERIMENT and a['GPU_inference_approved'] is True
            and a['researcher_explicit_authorization'] is True and phase in a['phases'],'Inference not approved')
    require(a['bindings']==binding,'Approval does not bind exact new payload')
    require(a['training_authorized'] is False and a['held_out_access'] is False
            and a['Teacher_calls'] is False and a['automatic_retry'] is False,'Forbidden authorization scope')
    start=datetime.fromisoformat(a['authorized_at_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and start.utcoffset().total_seconds()==end.utcoffset().total_seconds()==0,'UTC clocks required')
    require(start<=now<end<=start+timedelta(hours=48),'New approved window invalid/expired')
    require(a['generation_calls_max']=={'training_pilot':24,'validation_pair':48}[phase]
            and a['returned_sequences_max']=={'training_pilot':60,'validation_pair':120}[phase],'Budget differs')
    return a,end


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('preflight','training_pilot','validation_pair'))
    ap.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/stage_i_led_large_nbest_v1.json')
    ap.add_argument('--workspace',type=Path);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--approval',type=Path)
    ap.add_argument('--policy-lock',type=Path);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();root,p,c,binding=payload(args.config)
    if args.phase=='preflight':
        print('PAYLOAD_HASH_PASS_NO_MODEL_RUNTIME_OR_GPU_CHECK',binding);return
    a,deadline=authorization(args.approval,args.phase,binding,args.execute)
    require(args.workspace and args.checkpoint and args.output,'Workspace/checkpoint/fresh output required')
    ws=args.workspace.resolve();checkpoint=args.checkpoint.resolve();out=args.output.resolve()
    require(not out.exists() and checkpoint.is_dir(),'Fresh namespace and existing checkpoint required')
    require(out==ws/'runs'/c['output_names'][args.phase],'Use the exact new run namespace')
    require(checkpoint not in out.parents and out!=checkpoint,'Do not write into checkpoint')
    lock=None
    if args.phase=='validation_pair':
        require(args.policy_lock is not None,'Validation requires completed TRAIN-only policy lock')
        lock=read(args.policy_lock)
        require(lock['experiment']==EXPERIMENT and lock['bindings']==binding
                and lock['policy_sha256']==sha(p/'RERANKING_POLICY.json')
                and lock['training_only'] is True and lock['researcher_approved_for_validation'] is True,
                'Policy not locked using TRAIN only')
        # External gate/audit files must be preserved alongside the lock, not asserted by a flag alone.
        for key in ('pilot_completion','pilot_candidates','training_assessment','verifier_audit'):
            proof=args.policy_lock.parent/lock[key]['file']
            require(proof.resolve().parent==args.policy_lock.parent.resolve() and sha(proof)==lock[key]['sha256'],'Missing/foreign lock evidence: '+key)
        assessment=read(args.policy_lock.parent/lock['training_assessment']['file'])
        train_ids=[x['business_id'] for x in read(p/'TRAIN_SOURCE_ONLY_CASES.json')]
        require(diversity_gate(assessment,train_ids)['status'].startswith('PASS'),'Training candidate headroom failed')
        audit=read(args.policy_lock.parent/lock['verifier_audit']['file'])
        from stage_i_led_large_nbest_v1 import verification_audit
        require(verification_audit(audit,train_ids)['status']=='PASS_TRAIN_ONLY','Verifier audit insufficient/failed')
        completion=read(args.policy_lock.parent/lock['pilot_completion']['file'])
        require(completion['phase']=='training_pilot' and completion['bindings']==binding
                and completion['status']=='COMPLETE_STOP_FOR_TRAIN_SOURCE_REVIEW','Training pilot receipt differs')
        candidate_path=args.policy_lock.parent/lock['pilot_candidates']['file']
        require(sha(candidate_path)==completion['candidates_sha256'],'Pilot candidates not bound to completion')
        actual={r['business_id']:r for r in read(candidate_path)}
        require(set(actual)==set(train_ids),'Pilot candidates missing/foreign')
        require(all(r['materially_distinct_profiles']<=actual[r['business_id']]['diversity']['exact_unique_texts']
                    for r in assessment['cases']),'Semantic profiles exceed distinct actual texts')
        backend=lock['verifier_backend_lock']
        require(backend['kind'] in ('manual','model_assisted') and backend['identity_version']
                and backend['prompt_sha256']==sha(p/'SOURCE_JUDGE_PROTOCOL.md')
                and backend['training_only_verified'] is True,'Source verifier backend/prompt not frozen')
    for rel,h in c['parent_bindings'].items():
        require(sha(ws/rel)==h,'Original parent changed: '+rel)
    proof=read(checkpoint/'checkpoint_integrity.json');expected=read(p/'CHECKPOINT_BINDING.json')
    require(proof==expected and proof['global_step']==750,'Only successful Original CE V2 epoch3 allowed')
    require(sha(checkpoint/'checkpoint_integrity.json')==read(p/'OFFLINE_INTEGRITY.json')['checkpoint_receipt_sha256'],'Original checkpoint receipt bytes differ')
    for name,h in proof['files'].items():
        require(Path(name).name==name and sha(checkpoint/name)==h,'Checkpoint component differs: '+name)
    cases=read(p/('TRAIN_SOURCE_ONLY_CASES.json' if args.phase=='training_pilot' else 'VALIDATION_SOURCE_ONLY_CASES.json'))
    for case in cases:
        validate_case(case);require(case['split']==('train' if args.phase=='training_pilot' else 'validation'),'Wrong phase split')
    historical={x['business_id']:x for x in read(p/'identity_only/HISTORICAL_CE_EPOCH3.json')} if args.phase=='validation_pair' else {}
    # No torch import / CUDA query occurs above this boundary.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(ws/'scripts'))
    import stage_i_led_large_decoder_policy_v2 as parent
    require(parent.ROOT.resolve()==ws,'Loaded wrong parent workspace module')
    require({k:importlib.metadata.version(k) for k in c['runtime_pins']}==c['runtime_pins'],'Pinned dependencies differ; no global-env patch')
    parent_config=parent.preflight()
    import torch
    from transformers import GenerationConfig,LEDTokenizerFast
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from led_large_decoder_early_sanity_v2 import sanity_result
    from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter
    from led_large_dynamic_amp_v2 import capture_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090','Approved single RTX3090 required')
    out.mkdir(parents=True,exist_ok=False);write(out/'AUTHORIZATION.json',a)
    write(out/'BINDINGS.json',binding)
    if lock is not None:write(out/'TRAIN_POLICY_LOCK.json',lock)
    start=time.perf_counter();calls=sequences=0;records=[]
    def timeout_stop():
        try:
            write(out/'DEADLINE_STOP_NO_RETRY.json',dict(deadline=deadline.isoformat(),calls=calls,
                  returned_sequences=sequences,automatic_retry=False))
        finally:
            os._exit(124)
    watchdog=threading.Timer(max(.001,(deadline-datetime.now(timezone.utc)).total_seconds()),timeout_stop)
    watchdog.daemon=True;watchdog.start()
    try:
        model,before,_=parent.load_model(parent_config)
        weights=load_file(checkpoint/'adapter_model.safetensors',device='cpu')
        require(set(weights)==set(get_peft_model_state_dict(model)),'Adapter keys differ')
        set_peft_model_state_dict(model,weights)
        require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Saved adapter not active/exact')
        tok=LEDTokenizerFast.from_pretrained(ws/parent_config['source_tokenizer'],local_files_only=True)
        original=parent.generation(parent_config).to_dict();require(original==read(p/'ORIGINAL_GENERATION_CONFIG.json'),'Effective original generation config differs')
        single,n4=generation_pair(original);model.eval()
        write(out/'EFFECTIVE_GENERATION_PAIR.json',{'single':single,'nbest':n4})
        write(out/'ENVIRONMENT.json',dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
              GPU=torch.cuda.get_device_name(0),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
              runtime={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate','huggingface-hub','tokenizers')},
              attention='encoder_BOS_only',precision='historical_autocast_FP16_FP32_parameter_storage'))
        with MemoryMeter(torch) as memory:
            for case in cases:
                require(datetime.now(timezone.utc)<deadline,'Execution deadline reached')
                ids=case['input_ids'];length=((len(ids)+1023)//1024)*1024
                x=torch.full((1,length),1,dtype=torch.long,device='cuda');x[0,:len(ids)]=torch.tensor(ids,device='cuda')
                mask=torch.zeros_like(x);mask[0,:len(ids)]=1;global_mask=torch.zeros_like(x);global_mask[0,0]=1
                batch=dict(input_ids=x,attention_mask=mask,global_attention_mask=global_mask)
                result=dict(business_id=case['business_id'],split=case['split'],condition=case['condition'],
                            source_sha256=case['source_sha256'],checkpoint_sha256=sha(checkpoint/'adapter_model.safetensors'),
                            source_padded_shape=list(x.shape),attention_tokens=int(mask.sum().item()),
                            global_attention_tokens=int(global_mask.sum().item()),model_eval=not model.training)
                rng_before=fingerprint(capture_rng())
                result['rng_before_sha256']=rng_before
                for label,g in [('single',single),('candidates',n4)]:
                    require(datetime.now(timezone.utc)<deadline,'Execution deadline reached')
                    t=time.perf_counter();torch.cuda.synchronize()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                        generated=model.generate(**batch,generation_config=GenerationConfig(**g))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-t
                    raw=generated.cpu().tolist();calls+=1;sequences+=len(raw)
                    require(len(raw)==(1 if label=='single' else 4),'Unexpected return count')
                    rows=[]
                    for i,seq in enumerate(raw):
                        text=tok.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        from stage_i_led_large_nbest_v1 import unpad
                        screen=sanity_result(unpad(seq),text,768)
                        rows.append(dict(candidate_id=f'c{i}',generation_rank=i,token_ids=seq,text=text,
                                         source_sha256=result['source_sha256'],checkpoint_sha256=result['checkpoint_sha256'],technical_screen=screen))
                    result[label]=rows[0] if label=='single' else rows
                    result[label+'_seconds']=elapsed
                    write(out/'raw'/f'{case["business_id"]}_{label}.json',rows)
                    require(all(r['technical_screen']['status']=='PASS_TECHNICAL_ONLY' for r in rows),'Technical generation collapse; no retry')
                    require(fingerprint(capture_rng())==rng_before,'Unexpected stochastic eval behavior')
                    memory.check()
                result['identity']=compare_top1(result['single'],result['candidates'][0],historical.get(case['business_id']))
                result['diversity']=lexical_diversity(result['candidates'])
                records.append(result);write(out/'cases'/f'{case["business_id"]}.json',result)
                # This packet is the ONLY input to a future scorer. No history, targets, references, photos.
                write(out/'selection_packets'/f'{case["business_id"]}.json',selection_packet(case,result['candidates']))
                print(f'{args.phase}: {len(records)}/{len(cases)} source-only businesses saved',flush=True)
            require(calls==a['generation_calls_max'] and sequences==a['returned_sequences_max'],'Exact planned budget incomplete')
            require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Inference modified adapter')
            parent.verify_backbone(model,before);memory.check()
        write(out/'CANDIDATES.json',records);write(out/'MEMORY_SAMPLES.json',memory.samples)
        write(out/'COMPLETION.json',dict(status='COMPLETE_STOP_FOR_TRAIN_SOURCE_REVIEW' if args.phase=='training_pilot' else 'COMPLETE_STOP_FOR_PAIRED_RESEARCHER_REVIEW',
              phase=args.phase,bindings=binding,source_only_case_file_sha256=sha(p/('TRAIN_SOURCE_ONLY_CASES.json' if args.phase=='training_pilot' else 'VALIDATION_SOURCE_ONLY_CASES.json')),
              cases=len(cases),candidates_sha256=sha(out/'CANDIDATES.json'),generation_calls=calls,returned_sequences=sequences,seconds=time.perf_counter()-start,
              no_optimizer=True,no_training=True,no_judge_calls=True,weights_unchanged=True,**memory.summary()))
    except BaseException as e:
        write(out/'FAILURE_STOP_NO_RETRY.json',dict(error=str(e),traceback=traceback.format_exc(),calls=calls,
              returned_sequences=sequences,automatic_retry=False,quality_comparison_allowed=False));raise
    finally:
        watchdog.cancel()


if __name__=='__main__':
    main()
