"""Freeze extension inputs, original photos, controls and preservation bindings."""
import datetime as dt,hashlib,json,shutil
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1';P=OUT/'PAYLOAD'
Q=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2/REVIEW_PACKAGE'
O=ROOT/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE/STAGE_O';W=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP/WORKSPACE'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    assert not (P/'EXPERIMENT_CONFIG.json').exists(),'Do not overwrite a completed freeze';P.mkdir(parents=True,exist_ok=True);protected=[]
    def cp(src,dest):
        d=P/dest;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d)
        protected.append(dict(original=src.relative_to(ROOT).as_posix(),payload=d.relative_to(P).as_posix(),bytes=src.stat().st_size,sha256=sha(src)))
    lock=read(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    cp(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json','INPUTS/LED_MODEL_AND_DECODER_LOCK.json')
    backbone=Path(lock['backbone']['local_file']);assert sha(backbone)==lock['backbone']['sha256'];cp(backbone.parent/'config.json','ASSETS/BACKBONE/config.json')
    adapter=ROOT/lock['models']['F']['path'];assert sha(adapter/'adapter_model.safetensors')==lock['models']['F']['adapter_sha256']
    cp(adapter/'adapter_config.json','ASSETS/F/adapter_config.json')
    protected.append(dict(original=(adapter/'adapter_model.safetensors').relative_to(ROOT).as_posix(),payload='SEPARATELY_UPLOADED_ASSETS/F/adapter_model.safetensors',bytes=(adapter/'adapter_model.safetensors').stat().st_size,sha256=sha(adapter/'adapter_model.safetensors')))
    for f in (ROOT/'assets/tokenizers/led-base-16384').iterdir():
        if f.is_file():cp(f,Path('ASSETS/LED_TOKENIZER')/f.name)
    full=read(O/'LED_INFERENCE_CALL_PLAN_75.json');calls=[r for r in full['calls'] if r['condition']=='F'];assert len(calls)==30
    put(P/'INPUTS/LED_F_CALL_PLAN_30.json',calls);cp(O/'LED_INFERENCE_CALL_PLAN_75.json','PROVENANCE/ORIGINAL_LED_PLAN_75.json');cp(O/'LED_OUTPUTS/F/raw_predictions.json','INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')
    cp(O/'LED_OUTPUTS/F/RESTORATION.json','PROVENANCE/ORIGINAL_LED_RESTORATION.json');cp(O/'LED_OUTPUTS/F/COMPLETION.json','PROVENANCE/ORIGINAL_LED_COMPLETION.json')
    cp(Q/'NATIVE/FROZEN_SPEC.json','INPUTS/QWEN_CAPTION_FROZEN_SPEC.json');cp(Q/'NATIVE/PROMPTS.json','INPUTS/QWEN_CAPTION_PROMPTS.json')
    cp(Q/'NATIVE/EXECUTION/QWEN_OUTPUTS_15.json','INPUTS/QWEN_CAPTION_OUTPUTS_15.json');cp(Q/'NATIVE/EXECUTION/COMPLETION_RECEIPT.json','PROVENANCE/QWEN_CAPTION_COMPLETION_RECEIPT.json');cp(Q/'RESOURCE_MEASUREMENTS.json','INPUTS/QWEN_CAPTION_RESOURCES.json')
    for f in (Q/'NATIVE/EXECUTION/CALLS').glob('*.json'):cp(f,Path('INPUTS/QWEN_CAPTION_CALLS')/f.name)
    for src,dest in [('PREPARED_CAPTION_JOB/generation_report.json','HISTORICAL_SHARED_PREPARATION/BLIP_GENERATION_REPORT.json'),('PREPARED_CAPTION_JOB/model_load_report.json','HISTORICAL_SHARED_PREPARATION/BLIP_LOAD_REPORT.json'),('PREPARED_CAPTION_JOB/environment_report.json','HISTORICAL_SHARED_PREPARATION/BLIP_ENVIRONMENT.json'),('CAPTION_SCORES/scoring_report.json','HISTORICAL_SHARED_PREPARATION/CAPTION_SCORING_REPORT.json')]:cp(O/src,dest)
    original_prompts=read(Q/'NATIVE/PROMPTS.json');visual=original_prompts['visual']
    adapted=visual.replace('Use filtered captions only.','Use the provided original photographs only.').replace('Each numbered caption is a separate observation;','Each numbered photograph is a separate observation;').replace('sparse captions.','sparse visual evidence.').replace(' Do not claim to have inspected photographs.','')
    assert adapted!=visual and 'captions' not in adapted
    put(P/'IMAGE_PROMPT.json',dict(version='stage_p_direct_photos_source_adaptation_v1',visual=adapted,original_visual_prompt_sha256=hashlib.sha256(visual.encode()).hexdigest(),adaptation='Only source-modality wording changes: captions to original photographs; remove the caption-only prohibition on claiming photo inspection. Scope/length/other instructions unchanged.'))
    images=[];photo_bindings=[]
    for c in calls:
        if c['task']!='visual':continue
        b=c['business_id'];source=read(W/'SOURCES'/f'{b}.json');by={r['photo_id']:r for r in source['photos']};photos=[]
        assert 1<=len(c['selected_photo_ids'])<=5 and not c['selected_review_ids']
        for order,pid in enumerate(c['selected_photo_ids'],1):
            f=W/'PHOTOS'/f'{pid}.jpg';h=sha(f);assert h==by[pid]['sha256']==by[pid]['original_source_sha256']
            cp(f,Path('PHOTOS')/f.name)
            with Image.open(f) as im:width,height=im.size;mode=im.mode;fmt=im.format
            row=dict(order=order,photo_id=pid,file='PHOTOS/'+f.name,bytes=f.stat().st_size,sha256=h,original_width=width,original_height=height,original_mode=mode,format=fmt)
            photos.append(row);photo_bindings.append(dict(business_id=b,**row))
        images.append(dict(business_id=b,photos=photos,selected_photo_ids=c['selected_photo_ids'],filtered_caption_source_sha256_for_alignment_only=c['source_sha256'],review_reuse_from_Qwen_Caption=True,review_or_caption_text_sent_to_visual_model=False))
    assert len(images)==15 and len(photo_bindings)==73
    put(P/'IMAGE_PLAN_15.json',images);put(P/'PHOTO_PROVENANCE_73.json',photo_bindings)
    warm=max((c for c in calls if c['task']=='review'),key=lambda r:len(r['input_ids']))['business_id']
    cfg=dict(format='stage_p_three_system_extension_v1',frozen_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),cohort_sha256=full['cohort_freeze_sha256'],restaurants=15,
        LED=dict(model='LED-large F-S1000',adapter_sha256=lock['models']['F']['adapter_sha256'],backbone_sha256=lock['backbone']['sha256'],parameters_dtype='FP32',autocast='FP16',batch_size=1,num_beams=4,max_length=768,warmup_restaurant=warm,warmup_calls=2,measured_full_repetitions=2,original_predictions_never_replaced=True,original_torch='2.1.0+cu118',benchmark_torch='2.9.1+cu128',transformers='4.51.3',TF32=False),
        Qwen_Image=dict(model='Qwen/Qwen3.5-9B',revision=read(Q/'NATIVE/FROZEN_SPEC.json')['models']['Qwen3.5-9B']['revision'],parameters_dtype='BF16',batch_size=1,enable_thinking=False,decoding='Exact Qwen-Caption Visual policy',review_reused_verbatim=True,processor_policy='Pinned official processor defaults; no size/min/max-pixel override. Record actual image grids and tensor shapes/hashes before generation.',pilot_selection='Among5-photo restaurants choose largest processor-expanded input token count, ID tie-break, before output inspection',pilot_reused=True,no_quantization_or_evidence_reduction_or_model_fallback=True),
        resource_boundaries='Synchronize CUDA before/after model.generate; reset allocator peaks per call; exclude CPU preprocessing/transfers/decoding/export from generation timers and report separately. Loading and download/hash times separate.',
        selection_cost_caveat='All three systems inherit the same Filtered selection in this design. Qwen-Image pixels replace caption text only after selection; it is not a caption-free from-scratch image-selection pipeline.',human_collection=False,new_references=False,training=False,automatic_rental=False,automatic_destroy=False)
    put(P/'EXPERIMENT_CONFIG.json',cfg)
    for src in [ROOT/'outputs/stage_o/human_collection_complete_20261106_v1/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json',W/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json',O/'FINAL_U_F_CAPTION_POOLS.json']:
        protected.append(dict(original=src.relative_to(ROOT).as_posix(),payload=None,bytes=src.stat().st_size,sha256=sha(src)))
    put(OUT/'PROTECTED_ORIGINALS.json',protected);put(P/'PROVENANCE/PROTECTED_ORIGINAL_HASHES.json',protected)
    put(P/'SOURCE_ALIGNMENT.json',dict(status='PASS_FROZEN_SOURCE_BINDINGS',restaurants=15,photo_presentations=73,photo_order_identical_to_LED_Filtered_inputs=True,original_JPEG_bytes_verified=True,alternate_or_higher_resolution_images_used=False,review_reuse_verbatim_required=True,original_LED_plan_sha256=sha(O/'LED_INFERENCE_CALL_PLAN_75.json')))
    print(json.dumps(dict(payload=str(P),photos=73,LED_measured_calls=60,LED_warmup_calls=2,Qwen_new_visual_calls=15,original_model_hashes='PASS'),indent=2))
if __name__=='__main__':main()
