"""Frozen LED F benchmark only:2warmup calls and2 complete30-call repetitions."""
import argparse,importlib.metadata,os,sys,time,traceback
from pathlib import Path
from stage_p_resource_common_v1 import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();r=args.root.resolve();out=r/'LED_BENCHMARK';assert not out.exists();out.mkdir()
    import torch
    from transformers import LEDConfig,LEDForConditionalGeneration,LEDTokenizerFast,GenerationConfig,set_seed
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    prop=guard(torch);cfg=read(r/'EXPERIMENT_CONFIG.json');lock=read(r/'INPUTS/LED_MODEL_AND_DECODER_LOCK.json');calls=read(r/'INPUTS/LED_F_CALL_PLAN_30.json');orig={(c['business_id'],c['task']):c for c in read(r/'INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')}
    assert len(calls)==30 and len(orig)==30 and lock['models']['F']['adapter_sha256']==cfg['LED']['adapter_sha256']
    expected={'torch':'2.9.1+cu128','transformers':'4.51.3','tokenizers':'0.21.1','peft':'0.15.2','accelerate':'1.6.0'}
    assert all(importlib.metadata.version(k)==v for k,v in expected.items())
    bb=r/'ASSETS/BACKBONE';cp=r/'ASSETS/F';assert sha(bb/'pytorch_model.bin')==lock['backbone']['sha256'];assert sha(cp/'adapter_model.safetensors')==lock['models']['F']['adapter_sha256']
    tok=LEDTokenizerFast.from_pretrained(r/'ASSETS/LED_TOKENIZER',local_files_only=True)
    for c in calls:assert tok.encode(c['source'])==c['input_ids'] and digest(c['input_ids'])==c['input_ids_sha256'] and tsha(c['source'])==c['source_sha256']
    torch.set_num_threads(1);set_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sampler=Sampler();sampler.start();records=[];peaks=[];phase='model_load';started=time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();base=LEDForConditionalGeneration(LEDConfig(**read(bb/'config.json')))
        state=torch.load(bb/'pytorch_model.bin',map_location='cpu',weights_only=True);base.load_state_dict(state,strict=True);del state;base.tie_weights();assert sum(p.numel() for p in base.parameters())==459801600
        base.name_or_path=str(bb);model=get_peft_model(base,LoraConfig.from_pretrained(cp,local_files_only=True));weights=load_file(str(cp/'adapter_model.safetensors'),device='cpu')
        assert len(weights)==456 and set(weights)==set(get_peft_model_state_dict(model));set_peft_model_state_dict(model,weights)
        assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items())
        assert all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters())
        model.eval().to('cuda');torch.cuda.synchronize();loading=time.perf_counter()-t;peaks.append(dict(phase=phase,**mem(torch)))
        gen=GenerationConfig(**lock['generation']);assert gen.to_dict()==lock['generation'] and gen.num_beams==4
        def finite(module,args,result):assert bool(torch.isfinite(result.logits).all()),'Nonfinite logits: STOP, preserve all outputs'
        hook=model.base_model.model.register_forward_hook(finite)
        put(out/'RESTORATION.json',dict(status='PASS',model_loading_seconds=loading,adapter_sha256=sha(cp/'adapter_model.safetensors'),backbone_sha256=sha(bb/'pytorch_model.bin'),adapter_tensors=456,parameters_dtype='FP32',autocast='FP16',GPU=prop.name,GPU_uuid=GPU,batch_size=1,TF32=False,torch_threads=1,versions=expected,original_torch='2.1.0+cu118',original_transformers='4.51.3',generation=lock['generation'],optimizer_created=False))
        warm=cfg['LED']['warmup_restaurant'];phases=[('WARMUP',[c for c in calls if c['business_id']==warm]),('REPEAT_1',calls),('REPEAT_2',calls)]
        for phase,selected in phases:
            set_seed(42)
            for c in selected:
                ids=c['input_ids'];N=((len(ids)+1023)//1024)*1024
                batch=dict(input_ids=torch.tensor([ids+[1]*(N-len(ids))],device='cuda'),attention_mask=torch.tensor([[1]*len(ids)+[0]*(N-len(ids))],device='cuda'),global_attention_mask=torch.tensor([[1]+[0]*(N-1)],device='cuda'))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):generated=model.generate(**batch,generation_config=gen)
                torch.cuda.synchronize();seconds=time.perf_counter()-t;m=mem(torch);tokens=generated[0].cpu().tolist();t=time.perf_counter();text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False);decoding=time.perf_counter()-t
                prior=orig[c['business_id'],c['task']]
                x=dict(phase=phase,business_id=c['business_id'],task=c['task'],condition='F',input_ids=ids,padded_input_tokens=N,unpadded_input_tokens=len(ids),input_ids_sha256=c['input_ids_sha256'],source_sha256=c['source_sha256'],selected_review_ids=c['selected_review_ids'],selected_photo_ids=c['selected_photo_ids'],token_ids=tokens,text=text,text_sha256=tsha(text),decoder_sequence_token_count=len(tokens),generated_token_count=len(tokens)-1,decoder_start_token_excluded_from_throughput=True,generation_seconds=seconds,CPU_decode_seconds=decoding,generated_tokens_per_second=(len(tokens)-1)/seconds,VRAM=m,original_tokens_exact=tokens==prior['token_ids'],original_text_exact=text==prior['text'],original_text_sha256=prior['text_sha256'],generation_config_sha256=digest(lock['generation']),flags=dict(empty=not text.strip(),hit_length_cap=len(tokens)>=768,EOS_seen=2 in tokens[1:],heading_missing=not text.startswith('[REVIEW_CONSENSUS]' if c['task']=='review' else '[VISUAL_EVIDENCE]')))
                put(out/phase/(c['business_id']+'__'+c['task']+'.json'),x);records.append(x);peaks.append(dict(phase=phase,call_id=c['business_id']+'__'+c['task'],**m));del generated,batch
                print(__import__('json').dumps(dict(phase=phase,call=c['business_id']+'__'+c['task'],seconds=seconds,decoder_tokens=len(tokens),original_exact=x['original_tokens_exact'])),flush=True)
        hook.remove();assert all(p.grad is None for p in model.parameters());assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items());assert sha(bb/'pytorch_model.bin')==lock['backbone']['sha256'] and sha(cp/'adapter_model.safetensors')==lock['models']['F']['adapter_sha256']
        put(out/'BENCHMARK_RECORDS_62.json',records);put(out/'COMPLETION.json',dict(status='COMPLETE_TWO_MEASURED_REPETITIONS',restaurants=15,measured_repetitions=2,measured_calls=60,warmup_calls=2,model_loading_seconds=loading,elapsed_seconds=time.perf_counter()-started,VRAM_phase_peaks=peaks,checkpoint_unchanged=True,original_predictions_replaced=False,human_ratings_modified=False,completed_at_utc=now()))
    except Exception as e:
        put(out/'FAILURE.json',dict(phase=phase,completed_calls=len(records),error=str(e),traceback=traceback.format_exc(),memory=mem(torch),at_utc=now()));raise
    finally:sampler.close(out/'DEVICE_MEMORY_SAMPLES.json')
if __name__=='__main__':main()
