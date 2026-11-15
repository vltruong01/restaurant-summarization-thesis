"""Phase A V2 only: exact expected CPU RNG transition, restoration and durable receipts."""
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
    generation_pair,compare_top1,lexical_diversity,selection_packet)
from led_large_nbest_rng_v2 import recorded_generation, durable_write, LED_SOURCE_SHA256
EXPERIMENT = 'LED_LARGE_NBEST_RERANKING_V2'


def payload(config_path):
    root=config_path.resolve().parents[1];c=read(config_path)
    p=root/c['preparation'];manifest=read(p/'PREFLIGHT_MANIFEST.json')
    for rel,h in c['v1_execution_bindings'].items():
        require(sha(root/rel)==h, 'V1 chain changed: '+rel)
    for rel,h in read(root/c['v1_preflight_manifest'])['files'].items():
        require(sha(root/rel)==h, 'V1 preserved payload differs: '+rel)
    for rel,h in manifest['files'].items():
        require(sha(root/rel)==h,'N-best payload changed: '+rel)
    return root,p,c,dict(config_sha256=sha(config_path),runner_sha256=sha(Path(__file__)),
                        preflight_sha256=sha(p/'PREFLIGHT_MANIFEST.json'))


def authorization(path,phase,binding,execute):
    require(execute and path,'OFFLINE ONLY: new explicit researcher inference authorization required')
    require(phase == 'training_pilot', 'V2 revision is TRAIN Phase A only')
    a=read(path);now=datetime.now(timezone.utc)
    require(a.get('phases') == ['training_pilot'], 'Only Phase A may be authorized')
    require(a['experiment']==EXPERIMENT and a['GPU_inference_approved'] is True
            and a['researcher_explicit_authorization'] is True and phase in a['phases'],'Inference not approved')
    require(a['bindings']==binding,'Approval does not bind exact new payload')
    require(a['training_authorized'] is False and a['held_out_access'] is False
            and a['Teacher_calls'] is False and a['automatic_retry'] is False,'Forbidden authorization scope')
    start=datetime.fromisoformat(a['authorized_at_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and start.utcoffset().total_seconds()==end.utcoffset().total_seconds()==0,'UTC clocks required')
    require(start<=now<end<=start+timedelta(hours=48),'New approved window invalid/expired')
    require(a['generation_calls_max']==24
            and a['returned_sequences_max']==60,'Budget differs')
    return a,end


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('preflight','training_pilot'))
    ap.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/stage_i_led_large_nbest_v2.json')
    ap.add_argument('--workspace',type=Path);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--approval',type=Path)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();root,p,c,binding=payload(args.config)
    if args.phase=='preflight':
        print('PAYLOAD_HASH_PASS_NO_MODEL_RUNTIME_OR_GPU_CHECK',binding);return
    a,deadline=authorization(args.approval,args.phase,binding,args.execute)
    require(args.workspace and args.checkpoint and args.output,'Workspace/checkpoint/fresh output required')
    ws=args.workspace.resolve();checkpoint=args.checkpoint.resolve();out=args.output.resolve()
    require(not out.exists() and checkpoint.is_dir(),'Fresh namespace and existing checkpoint required')
    require(out==ws/'runs'/c['output_names'][args.phase],'Use the exact new run namespace')
    require(checkpoint not in out.parents and out!=checkpoint,'Do not write into checkpoint')
    for rel,h in c['parent_bindings'].items():
        require(sha(ws/rel)==h,'Original parent changed: '+rel)
    proof=read(checkpoint/'checkpoint_integrity.json');expected=read(p/'CHECKPOINT_BINDING.json')
    require(proof==expected and proof['global_step']==750,'Only successful Original CE V2 epoch3 allowed')
    require(sha(checkpoint/'checkpoint_integrity.json')==read(p/'OFFLINE_INTEGRITY.json')['checkpoint_receipt_sha256'],'Original checkpoint receipt bytes differ')
    for name,h in proof['files'].items():
        require(Path(name).name==name and sha(checkpoint/name)==h,'Checkpoint component differs: '+name)
    cases=read(p/'TRAIN_SOURCE_ONLY_CASES.json')
    for case in cases:
        validate_case(case);require(case['split']==('train' if args.phase=='training_pilot' else 'validation'),'Wrong phase split')
    require(len(cases)==12 and sum(not x['captions'] for x in cases)==3, 'Wrong TRAIN pilot composition')
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
    from led_large_dynamic_amp_v2 import capture_rng, restore_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090','Approved single RTX3090 required')
    out.mkdir(parents=True,exist_ok=False);write(out/'AUTHORIZATION.json',a)
    write(out/'BINDINGS.json',binding)
    start=time.perf_counter();calls=sequences=0;records=[];memory=None
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
        from transformers.models.led import modeling_led
        require(sha(Path(modeling_led.__file__))==LED_SOURCE_SHA256==c['rng_guard']['encoder_source_sha256'], 'Pinned LED encoder source differs')
        require(len(model.get_base_model().led.encoder.layers)==12, 'Expected twelve encoder LayerDrop draws')
        require(all(not module.training for module in model.modules()), 'Every module must be in eval mode')
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
                for label,g in [('single',single),('candidates',n4)]:
                    require(datetime.now(timezone.utc)<deadline,'Execution deadline reached')
                    require(all(not module.training for module in model.modules()), 'Eval mode drift before generation')
                    def generate():
                        nonlocal calls
                        calls+=1
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                            return model.generate(**batch,generation_config=GenerationConfig(**g))
                    def returned(count):
                        nonlocal sequences
                        sequences+=count
                    rows,elapsed=recorded_generation(folder=out,key=f'{case["business_id"]}_{label}',
                        generate=generate,decode=lambda seq:tok.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                        screen=sanity_result,torch=torch,memory=memory,capture=capture_rng,restore=restore_rng,
                        fingerprint=fingerprint,synchronize=torch.cuda.synchronize,
                        metadata=dict(source_sha256=result['source_sha256'],checkpoint_sha256=result['checkpoint_sha256']),
                        expected_sequences=1 if label=='single' else 4,on_returned=returned,
                        eval_check=lambda:all(not module.training for module in model.modules()))
                    result[label]=rows[0] if label=='single' else rows
                    result[label+'_seconds']=elapsed
                result['identity']=compare_top1(result['single'],result['candidates'][0])
                result['diversity']=lexical_diversity(result['candidates'])
                records.append(result);write(out/'cases'/f'{case["business_id"]}.json',result)
                # This packet is the ONLY input to a future scorer. No history, targets, references, photos.
                write(out/'selection_packets'/f'{case["business_id"]}.json',selection_packet(case,result['candidates']))
                print(f'{args.phase}: {len(records)}/{len(cases)} source-only businesses saved',flush=True)
            require(calls==a['generation_calls_max'] and sequences==a['returned_sequences_max'],'Exact planned budget incomplete')
            require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Inference modified adapter')
            parent.verify_backbone(model,before);memory.check()
        write(out/'CANDIDATES.json',records);write(out/'MEMORY_SAMPLES.json',memory.samples)
        write(out/'COMPLETION.json',dict(status='COMPLETE_STOP_FOR_TRAIN_SOURCE_REVIEW',
              phase=args.phase,bindings=binding,source_only_case_file_sha256=sha(p/'TRAIN_SOURCE_ONLY_CASES.json'),
              cases=len(cases),candidates_sha256=sha(out/'CANDIDATES.json'),generation_calls=calls,returned_sequences=sequences,seconds=time.perf_counter()-start,
              no_optimizer=True,no_training=True,no_judge_calls=True,weights_unchanged=True,**memory.summary()))
    except BaseException as e:
        write(out/'FAILURE_STOP_NO_RETRY.json',dict(error=str(e),traceback=traceback.format_exc(),calls=calls,
              returned_sequences=sequences,automatic_retry=False,quality_comparison_allowed=False));raise
    finally:
        watchdog.cancel()
        if memory is not None and not (out/'MEMORY_SAMPLES.json').exists():
            durable_write(out/'MEMORY_SAMPLES.json',memory.samples)


if __name__=='__main__':
    main()
