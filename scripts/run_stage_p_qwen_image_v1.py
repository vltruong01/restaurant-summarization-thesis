"""Qwen direct-image15 Visual passes; reuse existing Review text only after decoding."""
import argparse,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
from stage_p_resource_common_v1 import *
QMODEL=Path('/workspace/STAGE_P_QWEN35_9B_20261106_V2/HF_CACHE/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a')
def tensor_hash(t,torch):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def processor_and_inputs(root):
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(QMODEL,local_files_only=True,trust_remote_code=False)
    return processor,read(root/'IMAGE_PLAN_15.json'),read(root/'IMAGE_PROMPT.json')['visual']
def batch_for(root,processor,item,instruction,torch):
    from PIL import Image
    content=[dict(type='text',text='[TASK_VISUAL_EVIDENCE]\n[VISUAL_CONTEXT]\n')];images=[]
    for i,p in enumerate(item['photos'],1):
        f=root/p['file'];assert sha(f)==p['sha256']
        with Image.open(f) as im:
            assert list(im.size)==[p['original_width'],p['original_height']];images.append(im.convert('RGB'))
        content.extend([dict(type='text',text=f'[IMAGE_{i}]\n'),dict(type='image',image=str(f))])
    messages=[dict(role='system',content=instruction),dict(role='user',content=content)]
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert rendered.endswith('<think>\n\n</think>\n\n') and '[REVIEWS]' not in rendered and '[IMG_CAP_' not in rendered
    batch=processor(text=[rendered],images=images,return_tensors='pt')
    grids=batch['image_grid_thw'].tolist();assert len(grids)==len(item['photos'])
    patch=processor.image_processor.patch_size;merge=processor.image_processor.merge_size
    per_image=[dict(photo_id=p['photo_id'],original_size=[p['original_width'],p['original_height']],image_grid_thw=g,processed_height=g[1]*patch,processed_width=g[2]*patch,expanded_image_tokens=g[0]*g[1]*g[2]//(merge**2)) for p,g in zip(item['photos'],grids)]
    tokens=int((batch['input_ids']==processor.image_token_id).sum());assert tokens==sum(x['expanded_image_tokens'] for x in per_image)
    info=dict(business_id=item['business_id'],selected_photo_ids=item['selected_photo_ids'],photo_count=len(images),input_ids=batch['input_ids'][0].tolist(),input_tokens=int(batch['input_ids'].shape[-1]),image_token_count=tokens,image_grid_thw=grids,per_image=per_image,rendered_prompt_sha256=tsha(rendered),CPU_input_tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_hash(v,torch)) for k,v in batch.items() if torch.is_tensor(v)},source_images_only=True,no_review_caption_LED_or_reference_input=True)
    return batch,info
def preflight(root):
    import torch
    torch.set_num_threads(1);p,items,instruction=processor_and_inputs(root);checks=[]
    for item in items:
        batch,info=batch_for(root,p,item,instruction,torch);assert info['input_tokens']<=8192;checks.append(info)
    eligible=[x for x in checks if x['photo_count']==5];assert eligible
    pilot=sorted(eligible,key=lambda x:(-x['input_tokens'],x['business_id']))[0]['business_id']
    assert not (root/'IMAGE_PREFLIGHT.json').exists(),'Do not overwrite frozen processor mapping'
    result=dict(status='PASS_CPU_PROCESSOR_MAPPING_ONLY',model_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',image_processor_class=type(p.image_processor).__name__,processor_class=type(p).__name__,processor_configuration=p.image_processor.to_dict(),official_preprocessor_config=read(QMODEL/'preprocessor_config.json'),official_preprocessor_config_sha256=sha(QMODEL/'preprocessor_config.json'),chat_template_sha256=tsha(p.chat_template),image_prompt_sha256=sha(root/'IMAGE_PROMPT.json'),photo_order_and_count_preserved=True,RGB_conversion='PIL convert RGB; no manual resize,alternate file,EXIF transpose or min/max-pixel override; official processor defaults applied',pilot_restaurant=pilot,pilot_selection='Largest expanded prompt among5-photo restaurants,lexicographic ID tie-break',largest_input_tokens=max(x['input_tokens'] for x in checks),calls=checks,at_utc=now(),GPU_inference=False)
    put(root/'IMAGE_PREFLIGHT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['calls','processor_configuration','official_preprocessor_config']},indent=2))
def run(root):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    prop=guard(torch);assert torch.cuda.is_bf16_supported();assert read(root/'LED_JOB_EXIT.json')['exit_code']==0
    out=root/'QWEN_IMAGE';assert not out.exists();out.mkdir()
    pre=read(root/'IMAGE_PREFLIGHT.json');processor,items,instruction=processor_and_inputs(root);spec=read(root/'INPUTS/QWEN_CAPTION_FROZEN_SPEC.json');mi=spec['models']['Qwen3.5-9B'];checks={x['business_id']:x for x in pre['calls']}
    # JSON freeze converts processor tuple mean/std to lists; compare serialized values.
    assert sha(root/'IMAGE_PROMPT.json')==pre['image_prompt_sha256'] and json.loads(json.dumps(processor.image_processor.to_dict()))==pre['processor_configuration']
    for f,h in mi['tokenizer_files'].items():assert sha(QMODEL/f)==h
    for w in mi['weights']:assert (QMODEL/w['file']).stat().st_size==w['bytes'] and sha(QMODEL/w['file'])==w['sha256']
    assert torch.cuda.mem_get_info()[0]>=22*1024**3
    sampler=Sampler();sampler.start();records=[];peaks=[];phase='model_load';started=time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(QMODEL,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation=spec['attn_implementation']).eval();torch.cuda.synchronize();load_seconds=time.perf_counter()-t;peaks.append(dict(phase='model_load',**mem(torch)))
        assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters());torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
        put(out/'RESTORATION.json',dict(status='PASS',model_id=mi['model_id'],revision=mi['revision'],weights=mi['weights'],model_loading_seconds=load_seconds,GPU=prop.name,GPU_uuid=GPU,precision='BF16',batch_size=1,enable_thinking=False,torch_CPU_threads=torch.get_num_threads(),versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','Pillow','safetensors','accelerate']},processor_freeze_sha256=sha(root/'IMAGE_PREFLIGHT.json'),model_parameters_verified_unchanged_from_caption_checkpoint=True))
        pilot=pre['pilot_restaurant'];ordered=sorted(items,key=lambda x:(x['business_id']!=pilot,x['business_id']))
        for i,item in enumerate(ordered):
            phase='pilot_visual' if i==0 else 'full_visual';assert i==0 or (out/'COMPATIBILITY_PASS.json').exists()
            t=time.perf_counter();batch,info=batch_for(root,processor,item,instruction,torch);preprocessing=time.perf_counter()-t
            assert info==checks[item['business_id']],'CPU photo preprocessing or tokens changed after freeze'
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();gpu=batch.to('cuda',dtype=torch.bfloat16);torch.cuda.synchronize();transfer=time.perf_counter()-t
            actual={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_hash(v,torch)) for k,v in gpu.items() if torch.is_tensor(v)}
            settings={k:v for k,v in spec['generation'].items() if k!='max_new_tokens_by_task'};tok=processor.tokenizer;settings.update(max_new_tokens=spec['generation']['max_new_tokens_by_task']['visual'],pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=tok.eos_token_id)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():generated=model.generate(**gpu,**settings)
            torch.cuda.synchronize();seconds=time.perf_counter()-t;m=mem(torch);t=time.perf_counter();tokens=generated[0,info['input_tokens']:].cpu().tolist();text=tok.decode(tokens,skip_special_tokens=True);raw=tok.decode(tokens,skip_special_tokens=False);postprocessing=time.perf_counter()-t
            eos=tok.eos_token_id if isinstance(tok.eos_token_id,list) else [tok.eos_token_id]
            row=dict(business_id=item['business_id'],condition='QWEN_IMAGE',phase=phase,task='visual',model_id=mi['model_id'],model_revision=mi['revision'],selected_photo_ids=item['selected_photo_ids'],photo_sha256s=[p['sha256'] for p in item['photos']],input_provenance=info,GPU_input_tensors=actual,generation_token_ids=tokens,output_tokens=len(tokens),text=text,raw_decoded_with_special_tokens=raw,text_sha256=tsha(text),CPU_image_preprocessing_seconds=preprocessing,GPU_transfer_seconds=transfer,CPU_postprocessing_seconds=postprocessing,generation_seconds=seconds,tokens_per_second_including_prefill=len(tokens)/seconds,VRAM=m,generation_settings=settings,flags=dict(empty=not text.strip(),heading_missing=not text.startswith('[VISUAL_EVIDENCE]'),unexpected_thinking=('<think>' in text or '</think>' in text),hit_token_cap=len(tokens)>=settings['max_new_tokens'],stop_token_seen=bool(tokens and tokens[-1] in eos)),original_photographs_not_captions=True,review_or_LED_output_or_reference_supplied=False,completed_at_utc=now())
            put(out/'CALLS'/f"{item['business_id']}.json",row);records.append(row);peaks.append(dict(phase=phase,business_id=item['business_id'],**m));del generated,gpu,batch
            print(json.dumps(dict(phase=phase,business_id=item['business_id'],photos=len(item['photos']),input_tokens=info['input_tokens'],image_tokens=info['image_token_count'],output_tokens=len(tokens),seconds=seconds,peak_allocated_GiB=m['peak_allocated_bytes']/1024**3)),flush=True)
            assert not row['flags']['empty'] and not row['flags']['unexpected_thinking'],'Compatibility/non-thinking failure: stop, no retries'
            if i==0:put(out/'COMPATIBILITY_PASS.json',dict(status='PASS_ACTUAL_FIVE_PHOTO_MULTIMODAL_INFERENCE',business_id=pilot,photo_count=5,selected_photo_ids=item['selected_photo_ids'],memory=m,input_tokens=info['input_tokens'],image_tokens=info['image_token_count'],output_tokens=len(tokens),pilot_output_reused=True,quality_not_used_to_select_model_or_prompt=True,at_utc=now()))
        assert len(records)==15
        # Existing Review text is accessed solely for final verbatim assembly, after all Visual passes.
        reviews={x['business_id']:x for x in read(root/'INPUTS/QWEN_CAPTION_OUTPUTS_15.json')};assembled=[]
        for v in sorted(records,key=lambda x:x['business_id']):
            q=reviews[v['business_id']];assert tsha(q['review_text'])==q['review_text_sha256'];assembled.append(dict(business_id=v['business_id'],condition='QWEN_IMAGE',review_text=q['review_text'],review_text_sha256=q['review_text_sha256'],review_reused_from_Qwen_Caption=True,visual_text=v['text'],visual_text_sha256=v['text_sha256'],combined_text=q['review_text']+'\n\n'+v['text'],selected_photo_ids=v['selected_photo_ids']))
        put(out/'QWEN_IMAGE_OUTPUTS_15.json',assembled);put(out/'COMPLETION.json',dict(status='COMPLETE_15_NEW_VISUAL_PASSES_REUSED_REVIEW',restaurants=15,new_visual_calls=15,new_review_calls=0,model_loading_seconds=load_seconds,visual_generation_seconds=sum(x['generation_seconds'] for x in records),visual_output_tokens=sum(x['output_tokens'] for x in records),VRAM_phase_peaks=peaks,outputs_sha256=sha(out/'QWEN_IMAGE_OUTPUTS_15.json'),processor_preflight_sha256=sha(root/'IMAGE_PREFLIGHT.json'),elapsed_seconds=time.perf_counter()-started,completed_at_utc=now(),human_ratings_created=0))
    except Exception as e:
        put(out/'FAILURE.json',dict(phase=phase,completed_calls=len(records),error=str(e),traceback=traceback.format_exc(),memory=mem(torch),previous_phase_peaks=peaks,STOP_no_quantization_or_photo_reduction_or_model_switch_or_rental=True,A100_estimate='If true24GB OOM,40GB A100 BF16 batch1 likely provides substantial headroom over~18GiB weights plus these measured image grids. Unbenchmarked estimate; no A100 execution authorized here.',at_utc=now()));raise
    finally:sampler.close(out/'DEVICE_MEMORY_SAMPLES.json')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['preflight','run']);a.add_argument('--root',type=Path,required=True);x=a.parse_args();preflight(x.root.resolve()) if x.command=='preflight' else run(x.root.resolve())
