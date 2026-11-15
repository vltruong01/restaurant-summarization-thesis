"""Prepared finite inference entry point. Default is CPU manifest check, never generation.

Stage K Phase1 permits frozen inference before independent references. Local/allocated GPU only.
No optimizer, backwards pass, retry, selective replacement or automatic rental.
"""
import argparse,sys,os,importlib.metadata,time,datetime,traceback
from pathlib import Path
from s1000_final_eval_core_v1 import *

def preflight(root,prep):
    from tokenizers import Tokenizer
    lock=read(prep/'MODEL_AND_DECODER_LOCK.json');plan=read(prep/'TEST100_INFERENCE_PLAN.json')
    tokdir=root/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1/COMMON/assets/tokenizers/led-base-16384'
    for f,h in lock['tokenizer']['sha256'].items():require(file_hash(tokdir/f)==h,'Tokenizer mismatch')
    tok=Tokenizer.from_file(str(tokdir/'tokenizer.json'))
    for c in plan['calls']:
        ids=tok.encode(c['source']).ids
        require(text_hash(c['source'])==c['source_sha256'] and ids==c['input_ids'] and digest(ids)==c['input_ids_sha256'],'Input drift')
        require(len(ids)<=5000 and ids[0]==0 and ids[-1]==2,'Source token budget')
    for a,m in lock['models'].items():require(file_hash(root/m['path']/'adapter_model.safetensors')==m['adapter_sha256'],'Adapter drift')
    bb=lock['backbone'];bp=root/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    require(file_hash(bp)==bb['sha256'],'Backbone drift')
    return lock,plan,tokdir,bp

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--condition',choices=['RO','U','F','S']);p.add_argument('--execute',action='store_true');p.add_argument('--approval-message',type=Path)
    p.add_argument('--approved-references',type=Path);p.add_argument('--F-results',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();lock,plan,tokdir,bp=preflight(a.workspace,a.preparation)
    if not a.execute:print('PASS CPU file/source/token preflight only. No model constructed; TEST generation remains unexecuted.');return
    require(a.condition and a.output and a.approval_message and a.approval_message.is_file(),'Separate researcher release and explicit output required')
    require(file_hash(a.approval_message)=='56dd33dfbd2821048bbc04568c1058a76d0ef9fff4f531307d057fc5202a2c44','Wrong Stage K researcher release')
    require(file_hash(a.preparation/'TEST100_INFERENCE_PLAN.json')=='701a2b13f8743229d2dd74f435fdb9b69943a73c743e515b465d1e85791b2ad6','Frozen inference plan changed')
    require(file_hash(a.preparation/'MODEL_AND_DECODER_LOCK.json')=='342979621ad532c6029bc331cf596f8706ef21f2e7f6a33d718395f66edd1cee','Frozen model/decoder lock changed')
    require(a.approved_references is None,'Phase1 does not read independent reference payloads')
    # Human authentication/scope adjudication is external; no code invents an approval.
    pins=read(a.preparation/'INFERENCE_RUNTIME_LOCK.json')
    require(all(importlib.metadata.version(k)==v for k,v in pins['packages'].items()),'Runtime pins differ')
    selected=[c for c in plan['calls'] if c['condition']==a.condition];require(len(selected)==(100 if a.condition=='RO' else 200),'Call frame')
    prior={}
    if a.condition=='S':
        require(a.F_results and (a.F_results/'COMPLETION.json').exists(),'Complete frozen F results required for reuse')
        prior={(r['business_id'],r['task']):r for r in read(a.F_results/'raw_predictions.json')}
        require(len(prior)==200,'Incomplete F raw components')
    a.output.mkdir(parents=True,exist_ok=False);current={};results=[];calls=0;start=time.perf_counter()
    shuffled={r['recipient_id']:r for r in read(a.preparation/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings']}
    write(a.output/'EXECUTION_SCOPE.json',dict(stage='K_PHASE1',approval_sha256=file_hash(a.approval_message),inference_only=True,references_read=False,checkpoint_or_input_changes=False))
    try:
        import torch
        from transformers import LEDConfig,LEDForConditionalGeneration,LEDTokenizerFast,GenerationConfig,set_seed
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import load_file
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Select one researcher-allocated GPU using CUDA_VISIBLE_DEVICES')
        set_seed(42);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        arm='F' if a.condition=='S' else a.condition;cp=a.workspace/lock['models'][arm]['path']
        tok=LEDTokenizerFast.from_pretrained(tokdir,local_files_only=True)
        base=LEDForConditionalGeneration(LEDConfig(**read(bp.parent/'config.json')))
        state=torch.load(bp,map_location='cpu',weights_only=True);base.load_state_dict(state,strict=True);del state;base.tie_weights()
        require(sum(p.numel() for p in base.parameters())==459801600,'Backbone parameter inventory')
        base.name_or_path=str(bp.parent);config=LoraConfig.from_pretrained(cp,local_files_only=True)
        model=get_peft_model(base,config);weights=load_file(str(cp/'adapter_model.safetensors'),device='cpu')
        require(len(weights)==456 and set(weights)==set(get_peft_model_state_dict(model)),'Expanded-LoRA key set')
        set_peft_model_state_dict(model,weights)
        require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter restore')
        require(all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters()),'Invalid parameter precision/numerics')
        model.eval();model.to('cuda');g=GenerationConfig(**lock['generation'])
        write(a.output/'RESTORATION.json',dict(status='PASS',checkpoint_sha256=lock['models'][arm]['adapter_sha256'],adapter_tensors=len(weights),FP32_parameters=True,precision='FP16 autocast',GPU=torch.cuda.get_device_name(),generation=g.to_dict(),model_eval=True,optimizer_created=False))
        require(g.num_beams==4 and g.length_penalty==1.0 and g.max_length==768 and not g.do_sample and g.forced_eos_token_id==2,'Decoder changed')
        def finite(module,args,out):require(bool(torch.isfinite(out.logits).all()),'Nonfinite logits; no retry')
        hook=model.base_model.model.register_forward_hook(finite);torch.cuda.reset_peak_memory_stats()
        for c in selected:
            current={k:c[k] for k in ['condition','business_id','task']};t=time.perf_counter()
            if c['reuse_condition']:
                r=dict(prior[c['business_id'],c['task']]);require(r['checkpoint_sha256']==c['checkpoint_sha256'] and r['source_sha256']==c['source_sha256'] and r['input_ids_sha256']==c['input_ids_sha256'] and r['generation_config_sha256']==digest(lock['generation']),'F reuse identity mismatch')
                require(text_hash(r['text'])==r['text_sha256'] and tok.decode(r['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['text'],'F raw output corrupted')
                r.update(condition='S',reuse_of=dict(condition='F',business_id=c['business_id'],task=c['task']),seconds=0.,independent_generation=False)
            else:
                ids=c['input_ids'];N=((len(ids)+1023)//1024)*1024
                batch=dict(input_ids=torch.tensor([ids+[1]*(N-len(ids))],device='cuda'),attention_mask=torch.tensor([[1]*len(ids)+[0]*(N-len(ids))],device='cuda'),global_attention_mask=torch.tensor([[1]+[0]*(N-1)],device='cuda'))
                calls+=1
                write(a.output/'STARTED_CALLS'/f'{calls:03d}.json',dict(**current,source_sha256=c['source_sha256'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):tokens=model.generate(**batch,generation_config=g)[0].cpu().tolist()
                text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                r=dict(**current,token_ids=tokens,text=text,text_sha256=text_hash(text),checkpoint_sha256=c['checkpoint_sha256'],source_sha256=c['source_sha256'],input_ids_sha256=c['input_ids_sha256'],generation_config_sha256=digest(lock['generation']),seconds=time.perf_counter()-t,independent_generation=True,
                    termination=dict(length=len(tokens),EOS_seen=2 in tokens[1:],at_length_limit=len(tokens)>=768),FP32_parameters_FP16_autocast=True)
                require(0<len(tokens)<=768 and tokens[0]==2,'Invalid generation tokens')
            r.update(selected_review_ids=c['selected_review_ids'],selected_photo_ids=c['selected_photo_ids'],caption_ids=[f'[IMG_CAP_{i+1}]' for i in range(len(c['selected_photo_ids']))],input_ids=c['input_ids'],task_configuration='SOURCE_SEPARATED_V1',generation_settings=lock['generation'],checkpoint_condition=arm,shuffled_donor_id=shuffled[c['business_id']]['donor_id'] if a.condition=='S' else None,no_visual_intervention=a.condition in ['F','S'] and shuffled[c['business_id']]['caption_count']==0,generation_status='SUCCESS',source_sha256=c['source_sha256']);write(a.output/'RAW'/f"{c['business_id']}_{c['task']}.json",r);results.append(r)
            if len(results)%10==0:print(__import__('json').dumps(dict(condition=a.condition,logical_components=len(results),new_calls=calls,seconds=time.perf_counter()-start)),flush=True)
        require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter mutated')
        require(all(p.grad is None for p in model.parameters()),'Unexpected gradients')
        hook.remove();write(a.output/'raw_predictions.json',results)
        assembled=[]
        for b in sorted({r['business_id'] for r in results}):
            rr={r['task']:r for r in results if r['business_id']==b};texts=[rr[t]['text'] for t in ['review','visual'] if t in rr]
            assembled.append(dict(business_id=b,condition=a.condition,combined_text='\n\n'.join(texts),component_texts={t:r['text'] for t,r in rr.items()},component_hashes={t:r['text_sha256'] for t,r in rr.items()},
                format_flags={t:dict(expected_header=r['text'].startswith('[REVIEW_CONSENSUS]\n' if t=='review' else '[VISUAL_EVIDENCE]\n'),foreign_header=('[VISUAL_EVIDENCE]' if t=='review' else '[REVIEW_CONSENSUS]') in r['text']) for t,r in rr.items()},rewriting=False))
        write(a.output/'predictions.json',assembled)
        write(a.output/'COMPLETION.json',dict(condition=a.condition,businesses=100,raw_components=len(results),new_generation_calls=calls,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU_name=torch.cuda.get_device_name(),runtime_packages=pins['packages'],approval_message_sha256=file_hash(a.approval_message),reference_freeze_sha256=None,reference_payloads_read=0,stage='K_PHASE1',
            plan_sha256=file_hash(a.preparation/'TEST100_INFERENCE_PLAN.json'),raw_predictions_sha256=file_hash(a.output/'raw_predictions.json'),adapter_unchanged=True,backbone_file_unchanged=file_hash(bp)==lock['backbone']['sha256'],no_retry=True))
    except BaseException as e:
        write(a.output/'FAILURE_STOP_NO_RETRY.json',dict(error=str(e),traceback=traceback.format_exc(),current=current,started_calls=calls));raise

if __name__=='__main__':main()
