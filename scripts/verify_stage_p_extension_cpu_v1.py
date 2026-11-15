"""CPU-only replay of saved inputs/tokens, never generate model predictions."""
import argparse, json, hashlib, os
from pathlib import Path
from stage_p_resource_common_v1 import read,put,sha,tsha,digest,now

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['led','image']);a.add_argument('--root',type=Path,required=True);x=a.parse_args();r=x.root
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if x.mode=='led':
        from transformers import LEDTokenizerFast
        tok=LEDTokenizerFast.from_pretrained(r/'ASSETS/LED_TOKENIZER',local_files_only=True)
        rows=read(r/'LED_BENCHMARK/BENCHMARK_RECORDS_62.json');plan={(c['business_id'],c['task']):c for c in read(r/'INPUTS/LED_F_CALL_PLAN_30.json')};orig={(c['business_id'],c['task']):c for c in read(r/'INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')};checks=[]
        for c in rows:
            p=plan[c['business_id'],c['task']];o=orig[c['business_id'],c['task']]
            assert tok.encode(p['source'])==c['input_ids']==p['input_ids']
            assert digest(c['input_ids'])==c['input_ids_sha256'] and tsha(p['source'])==c['source_sha256']
            assert tok.decode(c['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==c['text'] and tsha(c['text'])==c['text_sha256']
            assert c['original_tokens_exact']==(c['token_ids']==o['token_ids']) and c['original_text_exact']==(c['text']==o['text'])
            checks.append(dict(phase=c['phase'],business_id=c['business_id'],task=c['task'],input_and_decode='PASS',original_text_exact=c['original_text_exact'],original_tokens_exact=c['original_tokens_exact']))
        assert len(checks)==62
        put(r/'CPU_LED_REPLAY.json',dict(status='PASS',calls=checks,GPU_model_calls=0,at_utc=now()))
    else:
        import torch
        from run_stage_p_qwen_image_v1 import processor_and_inputs,batch_for,tensor_hash
        torch.set_num_threads(1);proc,items,instruction=processor_and_inputs(r);pre=read(r/'IMAGE_PREFLIGHT.json');frozen={c['business_id']:c for c in pre['calls']};checks=[]
        assert read(r/'IMAGE_JOB_EXIT.json')['exit_code']==0
        assembled={c['business_id']:c for c in read(r/'QWEN_IMAGE/QWEN_IMAGE_OUTPUTS_15.json')};old={c['business_id']:c for c in read(r/'INPUTS/QWEN_CAPTION_OUTPUTS_15.json')}
        for item in items:
            b,info=batch_for(r,proc,item,instruction,torch);c=read(r/'QWEN_IMAGE/CALLS'/f"{item['business_id']}.json")
            assert info==frozen[item['business_id']]==c['input_provenance']
            cast={k:dict(shape=list(v.shape),dtype=str(v.to(dtype=torch.bfloat16) if v.is_floating_point() else v.dtype),sha256=tensor_hash(v.to(dtype=torch.bfloat16) if v.is_floating_point() else v,torch)) for k,v in []}
            cast={}
            for k,v in b.items():
                if torch.is_tensor(v):
                    t=v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
                    cast[k]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=tensor_hash(t,torch))
            assert cast==c['GPU_input_tensors']
            assert proc.tokenizer.decode(c['generation_token_ids'],skip_special_tokens=True)==c['text'] and tsha(c['text'])==c['text_sha256']
            assert proc.tokenizer.decode(c['generation_token_ids'],skip_special_tokens=False)==c['raw_decoded_with_special_tokens']
            z=assembled[item['business_id']];o=old[item['business_id']]
            assert z['review_text']==o['review_text'] and z['review_text_sha256']==o['review_text_sha256']==tsha(z['review_text'])
            assert z['visual_text']==c['text'] and z['combined_text']==o['review_text']+'\n\n'+c['text']
            checks.append(dict(business_id=item['business_id'],photo_count=len(item['photos']),photos_CPU_pixels_BF16_GPU_tensors_prompt_and_tokens='PASS',review_verbatim_reuse='PASS'))
        assert len(checks)==15
        put(r/'CPU_IMAGE_REPLAY.json',dict(status='PASS',calls=checks,new_model_calls=0,original_JPEG_count=sum(c['photo_count'] for c in checks),at_utc=now()))
    print(x.mode+' CPU replay PASS')
if __name__=='__main__':main()
