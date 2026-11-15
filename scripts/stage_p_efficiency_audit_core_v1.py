"""Pure standard-library arithmetic over frozen receipts. Never imports ML code."""
import csv,hashlib,json,math,statistics,zipfile
from collections import Counter
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8),(a,b)

def compute(root):
    root=Path(root);e=root/'EVIDENCE'
    led=read(e/'LED_RECORDS_62.json');qcap=[read(p) for p in sorted((e/'QCAP_CALLS').glob('*.json'))];image=[read(p) for p in sorted((e/'QIMAGE_CALLS').glob('*.json'))]
    lc=read(e/'LED_COMPLETION.json');qc=read(e/'QCAP_COMPLETION.json');ic=read(e/'QIMAGE_COMPLETION.json');summary=read(e/'PRIOR_RESOURCE_SUMMARY.json')
    bg=read(e/'BLIP_GENERATION_REPORT.json');bl=read(e/'BLIP_LOAD_REPORT.json');score=read(e/'CAPTION_SCORING_REPORT.json')
    native=[];native_bytes={}
    with zipfile.ZipFile(e/'BLIP_CHECKPOINTS.zip') as z:
        assert z.testzip() is None
        for name in sorted(z.namelist()):
            raw=z.read(name);native.append(json.loads(raw));native_bytes[name]=hashlib.sha256(raw).hexdigest()
    assert len(native)==len({r['photo_id'] for r in native})==bg['completed']==bg['planned']==5143
    checkpoint_binding=read(e/'BLIP_ARCHIVE_MEMBER_HASHES.json')
    assert {r['zip_member']:r['sha256'] for r in checkpoint_binding['members']}==native_bytes
    assert len(checkpoint_binding['members'])==5143
    assert next(r['sha256'] for r in read(root/'SOURCE_MANIFEST.json')['files'] if r['package_file']=='EVIDENCE/BLIP_CHECKPOINTS.zip')==checkpoint_binding['source_archive_sha256']
    assert len(qcap)==30 and len(image)==15 and len(led)==62
    attempts=[a for r in native for a in r['attempts']]
    assert Counter(len(r['attempts']) for r in native)=={1:5136,2:7}
    assert sum(r['basic_eligible'] for r in native)==bg['basic_eligible']==score['scored']==5138
    for r in native:
        assert r['runtime_sha256']==bg['runtime_sha256']==bl['runtime_sha256']
        assert r['device_map_sha256']==bg['device_map_sha256']==bl['device_map_sha256']
    repeats=[[r for r in led if r['phase']==phase] for phase in ['REPEAT_1','REPEAT_2']]
    assert all(len(r)==30 for r in repeats) and len([r for r in led if r['phase']=='WARMUP'])==2
    led_seconds=[sum(r['generation_seconds'] for r in rows) for rows in repeats]
    q_review=sum(r['generation_seconds'] for r in qcap if r['task']=='review');q_visual=sum(r['generation_seconds'] for r in qcap if r['task']=='visual')
    cap_total=sum(r['generation_seconds'] for r in qcap);image_visual=sum(r['generation_seconds'] for r in image)
    close(cap_total,qc['model_generation_seconds']);close(image_visual,ic['visual_generation_seconds'])
    for i,total in enumerate(led_seconds):close(total,summary['comparison'][i]['total_seconds'])
    close(statistics.mean(led_seconds),summary['LED_variability']['total_generation_seconds_mean'])
    close(q_review+image_visual,summary['comparison'][4]['total_seconds'])
    assert round(statistics.mean(led_seconds),3)==97.931 and round(cap_total,3)==186.844 and round(q_review+image_visual,3)==379.697
    # Exact model-visible photo order is inherited from Filtered selection.
    led_visual={r['business_id']:r for r in repeats[0] if r['task']=='visual'}
    cap_visual={r['business_id']:r for r in qcap if r['task']=='visual'}
    image_by={r['business_id']:r for r in image}
    original=read(e/'ORIGINAL_F_CONDITION_INPUTS.json')
    assert len(original)==15
    for row in original:
        bid=row['business_id'];ids=row['selected_photo_ids']
        assert ids==led_visual[bid]['selected_photo_ids']==cap_visual[bid]['selected_photo_ids']==image_by[bid]['selected_photo_ids']
    selected=[p for r in image for p in r['selected_photo_ids']];assert len(selected)==len(set(selected))==73
    allphotos={r['photo_id'] for r in native};assert set(selected)<=allphotos
    led_token_match=sum(r['token_ids']==next(x['token_ids'] for x in repeats[1] if x['business_id']==r['business_id'] and x['task']==r['task']) for r in repeats[0]);assert led_token_match==30
    origmatch=sum(r['original_tokens_exact'] for r in repeats[0]);assert origmatch==22
    total_outputs=[sum(r['generated_token_count'] for r in rows) for rows in repeats];assert total_outputs==[3140,3140]
    assert sum(r['output_tokens'] for r in qcap)==3651 and sum(r['output_tokens'] for r in image)==1342
    def peak(rows,key='peak_allocated_bytes'):return max(r['VRAM'][key] for r in rows)/2**30
    bpeak=max(a['peak_GPU_allocated_GiB'] for a in attempts);bres=max(a['peak_GPU_reserved_GiB'] for a in attempts)
    lp=peak(repeats[0]);qp=peak(qcap);ip=peak(image);review_peak=peak([r for r in qcap if r['task']=='review'])
    prep=bg['elapsed_seconds']+score['elapsed_seconds'];close(prep,summary['historical_shared_preparation_subtotal_seconds'])
    facts=dict(led_generation_seconds=statistics.mean(led_seconds),led_repeat_seconds=led_seconds,led_sample_sd_seconds=statistics.stdev(led_seconds),
        led_warmup_seconds=sum(r['generation_seconds'] for r in led if r['phase']=='WARMUP'),
        qcap_generation_seconds=cap_total,qcap_review_seconds=q_review,qcap_visual_seconds=q_visual,
        qimage_visual_seconds=image_visual,qimage_constructed_generation_seconds=q_review+image_visual,
        blip_elapsed_seconds=bg['elapsed_seconds'],blip_load_seconds=bl['model_load_seconds'],blip_attempt_seconds_sum=math.fsum(a['seconds'] for a in attempts),
        blip_photos=len(native),blip_attempts=len(attempts),blip_retry_photos=sum(len(r['attempts'])>1 for r in native),blip_basic_eligible=bg['basic_eligible'],
        blip_peak_allocated_GiB=bpeak,blip_peak_reserved_GiB=bres,scoring_elapsed_seconds=score['elapsed_seconds'],
        historical_measured_preparation_subtotal_seconds=prep,filtered_selected_photos=73,source_photos=5143,
        led_peak_allocated_GiB=lp,led_peak_reserved_GiB=peak(repeats[0],'peak_reserved_bytes'),
        qcap_peak_allocated_GiB=qp,qcap_peak_reserved_GiB=peak(qcap,'peak_reserved_bytes'),
        qimage_visual_peak_allocated_GiB=ip,qimage_visual_peak_reserved_GiB=peak(image,'peak_reserved_bytes'),
        qimage_constructed_peak_allocated_GiB=max(ip,review_peak),
        recorded_sequential_LED_stage_max_GiB=max(bpeak,bl['GPU_allocated_GiB'],score['peak_GPU_allocated_GiB'],lp),
        recorded_sequential_QCAP_stage_max_GiB=max(bpeak,score['peak_GPU_allocated_GiB'],qp),
        recorded_sequential_QIMAGE_stage_max_GiB=max(bpeak,score['peak_GPU_allocated_GiB'],ip,review_peak),
        qimage_CPU_preprocessing_seconds=sum(r['CPU_image_preprocessing_seconds'] for r in image),
        qimage_GPU_transfer_seconds=sum(r['GPU_transfer_seconds'] for r in image),qimage_CPU_postprocessing_seconds=sum(r['CPU_postprocessing_seconds'] for r in image),
        led_tokens=3140,qcap_tokens=3651,qimage_visual_tokens=1342,qimage_constructed_tokens=sum(r['output_tokens'] for r in qcap if r['task']=='review')+1342,
        led_accepted_output_exact_matches=origmatch,led_between_repeat_exact_matches=led_token_match,
        qcap_over_led_generation_ratio=cap_total/statistics.mean(led_seconds),qimage_constructed_over_led_generation_ratio=(q_review+image_visual)/statistics.mean(led_seconds),
        all_three_inherit_Filtered_photo_selection=True,complete_pipeline_measured=False,original_checkpoint_bundle_members=len(native_bytes))
    for model,g in [('LED',facts['led_generation_seconds']),('QCAP',cap_total),('QIMAGE',q_review+image_visual)]:facts[model+'_historical_measured_component_subtotal_seconds']=prep+g
    cached=[]
    for n in [1,10,100,1000]:cached.append(dict(hypothetical_full_cohort_requests=n,measured_prep_components_per_request=prep/n,
        LED_known_components_per_request=prep/n+facts['led_generation_seconds'],QCAP_known_components_per_request=prep/n+cap_total,
        QIMAGE_inherited_selection_known_components_per_request=prep/n+q_review+image_visual))
    facts['hypothetical_amortization']=cached
    facts.update(led_input_tokens=sum(r['unpadded_input_tokens'] for r in repeats[0]),led_padded_input_tokens=sum(r['padded_input_tokens'] for r in repeats[0]),
        qcap_input_tokens=sum(r['input_tokens'] for r in qcap),qimage_visual_expanded_input_tokens=sum(r['input_provenance']['input_tokens'] for r in image),
        led_generated_tokens_per_second=3140/facts['led_generation_seconds'],qcap_generated_tokens_per_second=3651/cap_total,
        qimage_visual_generated_tokens_per_second=1342/image_visual,qimage_constructed_generated_tokens_per_second=facts['qimage_constructed_tokens']/(q_review+image_visual))
    # Long-form CSV: source values kept separate from missing values and arithmetic.
    measurements=[]
    def row(id,system,stage,metric,value,unit,status,hardware,precision,definition,source,locator='',formula='',photos='',calls='',repetitions=''):
        measurements.append(dict(measurement_id=id,system=system,stage=stage,metric=metric,value=value,unit=unit,status=status,hardware=hardware,precision=precision,photo_count=photos,call_count=calls,repetitions=repetitions,definition=definition,source_file=source,source_locator=locator,source_sha256=sha(root/source) if source else '',derivation=formula))
    def measured(id,system,stage,metric,value,unit,hardware,precision,definition,source,locator='',**kwargs):row(id,system,stage,metric,value,unit,'MEASURED',hardware,precision,definition,source,locator,**kwargs)
    def derived(id,system,stage,metric,key,unit,hardware,precision,definition,source,formula,**kwargs):row(id,system,stage,metric,facts[key],unit,'DERIVED_FROM_MEASUREMENTS',hardware,precision,definition,source,'facts.'+key,formula,**kwargs)
    gpu='RTX3090, same GPU UUID in receipts';blip_hw='RTX3080 + CPU offload';led_dtype='FP32 parameters + FP16 autocast';bf='BF16, batch1, greedy, thinking disabled'
    measured('blip_elapsed','Shared selected-photo preparation','BLIP2','elapsed_seconds',bg['elapsed_seconds'],'s',blip_hw,'FP16 vision/language; FP32 Q-former; CPU offload','Timer starts before model/processor load; includes all photos, retries, per-photo preprocessing/decoding/hash/checkpoint I/O; excludes earlier asset download and final source recheck. Load already included.','EVIDENCE/BLIP_GENERATION_REPORT.json','/elapsed_seconds',photos=5143)
    measured('blip_load','Shared selected-photo preparation','BLIP2','model_load_seconds',bl['model_load_seconds'],'s',blip_hw,'mixed as above','Subset of BLIP elapsed; do not add again.','EVIDENCE/BLIP_LOAD_REPORT.json','/model_load_seconds')
    derived('blip_attempt_sum','Shared selected-photo preparation','BLIP2','sum_all_attempt_seconds','blip_attempt_seconds_sum','s',blip_hw,'mixed as above','5150 recorded attempt timers, including7 retries; not pure GPU-only timing.','EVIDENCE/BLIP_CHECKPOINTS.zip','fsum(attempt.seconds)',photos=5143,calls=5150)
    derived('blip_peak','Shared selected-photo preparation','BLIP2','peak_allocated','blip_peak_allocated_GiB','GiB',blip_hw,'mixed as above','Maximum torch.cuda.max_memory_allocated over every saved attempt; not whole-device VRAM or minimum capacity.','EVIDENCE/BLIP_CHECKPOINTS.zip','max(attempt.peak_GPU_allocated_GiB)',photos=5143,calls=5150)
    derived('blip_reserved','Shared selected-photo preparation','BLIP2','peak_reserved','blip_peak_reserved_GiB','GiB',blip_hw,'mixed as above','Maximum PyTorch allocator reserved memory over all attempts.','EVIDENCE/BLIP_CHECKPOINTS.zip','max(attempt.peak_GPU_reserved_GiB)',photos=5143,calls=5150)
    measured('score_elapsed','Shared selected-photo preparation','CLIP/SBERT scoring','elapsed_seconds',score['elapsed_seconds'],'s','RTX3080','FP32, TF32 false','Includes scorer loading/weight hashes, source image reads, text/image scoring, embeddings export; excludes final threshold/redundancy/diversity/K selection.','EVIDENCE/CAPTION_SCORING_REPORT.json','/elapsed_seconds',photos=5138)
    for key in ['peak_GPU_allocated_GiB','peak_GPU_reserved_GiB']:measured('score_'+key,'Shared selected-photo preparation','CLIP/SBERT scoring',key,score[key],'GiB','RTX3080','FP32','Allocator peak for scoring stage, not total device memory.','EVIDENCE/CAPTION_SCORING_REPORT.json','/'+key)
    derived('shared_known_prep','All three actual selection lineages','Historical preparation','known_components_subtotal','historical_measured_preparation_subtotal_seconds','s','RTX3080 + CPU offload','Mixed','BLIP elapsed + scorer elapsed; selection/acquisition/review preparation unmeasured. Not a complete preprocessing time.','EVIDENCE/BLIP_GENERATION_REPORT.json','48642.959 + 178.529')
    for i,rows in enumerate(repeats,1):
        measured('led_repeat_'+str(i),'LED F-S1000','prepared-input generation','generation_seconds',led_seconds[i-1],'s',gpu,led_dtype,'Sum of30 CUDA-synchronized model.generate timers after2 warmups; CPU transfers/decoding/files excluded.','EVIDENCE/LED_RECORDS_62.json','phase=REPEAT_'+str(i)+';sum(generation_seconds)',calls=30,repetitions=1,photos=73)
    derived('led_generation_mean','LED F-S1000','prepared-input generation','generation_mean_seconds','led_generation_seconds','s',gpu,led_dtype,'Mean of two full-cohort repetitions, not the60-call total.','EVIDENCE/LED_RECORDS_62.json','mean(sum(REPEAT_1.generation_seconds),sum(REPEAT_2.generation_seconds))',photos=73,calls=30,repetitions=2)
    derived('led_generation_sd','LED F-S1000','prepared-input generation','sample_sd_seconds','led_sample_sd_seconds','s',gpu,led_dtype,'Descriptive sample SD,n=2; not a confidence interval or inferential speed claim.','EVIDENCE/LED_RECORDS_62.json','stdev(two repetition totals)',calls=30,repetitions=2)
    for system,prefix,source,precision in [('LED F-S1000','led','EVIDENCE/LED_RECORDS_62.json',led_dtype),('Qwen-Caption','qcap','EVIDENCE/QCAP_COMPLETION.json',bf),('Qwen-Image Visual','qimage_visual','EVIDENCE/QIMAGE_COMPLETION.json',bf)]:
        for kind in ['allocated','reserved']:derived(prefix+'_peak_'+kind,system,'prepared-input generation','peak_'+kind,prefix+'_peak_'+kind+'_GiB','GiB',gpu,precision,'Per-generation PyTorch allocator peak; includes resident model and tensors, not device total or capacity.','EVIDENCE/LED_RECORDS_62.json' if prefix=='led' else source,'max per-call peak_'+kind+'_bytes / 2**30',photos=73)
    measured('qcap_generation','Qwen-Caption','prepared-input generation','generation_seconds',cap_total,'s',gpu,bf,'30 timed calls including2 cold compatibility calls, not repeated/warmed like LED; no BLIP/filter/load/CPU decode.','EVIDENCE/QCAP_COMPLETION.json','/model_generation_seconds',photos=73,calls=30,repetitions=1)
    for task,key in [('Review','qcap_review_seconds'),('Visual','qcap_visual_seconds')]:derived('qcap_'+task,'Qwen-Caption','prepared-input generation',task+'_generation_seconds',key,'s',gpu,bf,'Sum of15 native component calls.','EVIDENCE/QCAP_COMPLETION.json','sum(QCAP_CALLS where task='+task.lower()+')',calls=15)
    measured('qimage_new_visual','Qwen-Image','prepared-image generation','visual_generation_seconds',image_visual,'s',gpu,bf,'Only15 new image Visual calls. Review was reused; no new Review call. CPU image processing/transfer/postprocessing excluded.','EVIDENCE/QIMAGE_COMPLETION.json','/visual_generation_seconds',photos=73,calls=15,repetitions=1)
    derived('qimage_constructed','Qwen-Image','constructed Review+Visual','constructed_generation_seconds','qimage_constructed_generation_seconds','s',gpu,bf,'Earlier15 Review calls +15 new image Visual calls from separate sessions; not a fresh combined run and not actual incremental GPU time.','EVIDENCE/QIMAGE_COMPLETION.json','qcap_review_seconds + qimage_visual_seconds',photos=73,calls=30)
    derived('qimage_constructed_peak','Qwen-Image','constructed Review+Visual','peak_allocated','qimage_constructed_peak_allocated_GiB','GiB',gpu,bf,'Maximum of earlier Review peak and new Visual peak, not Visual-only17.757GiB.','EVIDENCE/QCAP_COMPLETION.json','max(QCAP Review peaks,QIMAGE Visual peaks)')
    for key,definition in [('qimage_CPU_preprocessing_seconds','Image read/hash/RGB conversion, processor/prompt construction outside generate; not cohort selection or complete source acquisition.'),('qimage_GPU_transfer_seconds','Measured GPU transfer and synchronization outside generate.'),('qimage_CPU_postprocessing_seconds','Token CPU copy and decode outside generate.')]:derived(key,'Qwen-Image','outside generation',key,key,'s',gpu,bf,definition,'EVIDENCE/QIMAGE_COMPLETION.json','sum(QIMAGE_CALLS.'+key.replace('qimage_','').replace('CPU_preprocessing','CPU_image_preprocessing')+')',photos=73,calls=15)
    for system,source,field,precision in [('LED F-S1000','LED_COMPLETION.json','model_loading_seconds',led_dtype),('Qwen-Caption','QCAP_COMPLETION.json','load_seconds',bf),('Qwen-Image Visual','QIMAGE_COMPLETION.json','model_loading_seconds',bf)]:measured(system+'_load',system,'model loading','model_load_seconds',read(e/source)[field],'s',gpu,precision,'Separate load timer; excluded from generation. Image and Caption are separate model loads in actual experiment.','EVIDENCE/'+source,'/'+field)
    for field in ['download_seconds','weight_verification_seconds','wall_elapsed_seconds']:measured('qcap_'+field,'Qwen-Caption','job lifecycle',field,qc[field],'s',gpu,bf,'Job lifecycle observation; wall includes download/hash/load/pilot/generation/file work. Not complete source pipeline.','EVIDENCE/QCAP_COMPLETION.json','/'+field)
    for system,source in [('LED F-S1000','LED_COMPLETION.json'),('Qwen-Image Visual','QIMAGE_COMPLETION.json')]:measured(system+'_jobwall',system,'job lifecycle','elapsed_seconds',read(e/source)['elapsed_seconds'],'s',gpu,led_dtype if system.startswith('LED') else bf,'Instrumented job elapsed. LED includes two warmups +60 measured calls; Image includes15 new Visual calls. Not comparable one-request E2E.','EVIDENCE/'+source,'/elapsed_seconds')
    measured('extension_setup','Stage P extension shared','CPU asset/runtime setup','seconds',read(e/'ASSET_SETUP_COMPLETE.json')['seconds'],'s','CPU, RTX3090 host','not GPU inference','Existing setup-only receipt, excludes prior Qwen downloads and Stage O setup; not total GPU setup.','EVIDENCE/ASSET_SETUP_COMPLETE.json','/seconds')
    measured('led_original_stageO','Original accepted LED F-S1000','historical generation loop','elapsed_seconds',read(e/'ORIGINAL_LED_F_COMPLETION.json')['seconds'],'s','RTX3080','FP32 parameters + FP16 autocast','Original loop wall timer includes per-call tensor prep, decode/export and validation; different GPU/runtime; not the synchronized Stage P97.931s benchmark.','EVIDENCE/ORIGINAL_LED_F_COMPLETION.json','/seconds',photos=73,calls=30)
    for system,prefix,key in [('LED F-S1000','LED','led_tokens'),('Qwen-Caption','QCAP','qcap_tokens'),('Qwen-Image Visual','QIMAGE','qimage_visual_tokens'),('Qwen-Image Review+Visual','QIMAGEtotal','qimage_constructed_tokens')]:derived(prefix+'_tokens',system,'output accounting','generated_tokens',key,'tokens',gpu,led_dtype if prefix=='LED' else bf,'Model-native generation tokens, stop/special included; LED decoder-start excluded; cross-tokenizer throughput is descriptive only.','EVIDENCE/LED_RECORDS_62.json' if prefix=='LED' else 'EVIDENCE/QCAP_COMPLETION.json','sum(native output counts)',calls=30 if prefix!='QIMAGE' else 15)
    for key,system,unit in [('led_input_tokens','LED F-S1000','tokens'),('led_padded_input_tokens','LED F-S1000','tokens'),('qcap_input_tokens','Qwen-Caption','tokens'),('qimage_visual_expanded_input_tokens','Qwen-Image Visual','tokens'),('led_generated_tokens_per_second','LED F-S1000','tokens/s'),('qcap_generated_tokens_per_second','Qwen-Caption','tokens/s'),('qimage_visual_generated_tokens_per_second','Qwen-Image Visual','tokens/s'),('qimage_constructed_generated_tokens_per_second','Qwen-Image constructed','tokens/s')]:
        derived(key,system,'model-native token accounting',key,key,unit,gpu,led_dtype if key.startswith('led') else bf,'Exact sum/native throughput includes prefill; image input includes processor-expanded visual tokens. Not a common-tokenizer or equal-output-length comparison.','EVIDENCE/LED_RECORDS_62.json' if key.startswith('led') else 'EVIDENCE/QCAP_COMPLETION.json' if key.startswith('qcap') else 'EVIDENCE/QIMAGE_COMPLETION.json','native input sum or native output tokens / matching generation seconds')
    for prefix,key in [('LED','recorded_sequential_LED_stage_max_GiB'),('QCAP','recorded_sequential_QCAP_stage_max_GiB'),('QIMAGE','recorded_sequential_QIMAGE_stage_max_GiB')]:derived(prefix+'_stage_max',prefix,'sequential historical pipeline stages','max_recorded_stage_allocated',key,'GiB','RTX3080 preprocessing + RTX3090 summary','mixed policies','Maximum,NOT sum,of recorded sequential stage allocated peaks. BLIP CPU offload/RAM and driver/workspace/transients mean this is not minimum GPU capacity.','EVIDENCE/BLIP_CHECKPOINTS.zip','max(BLIP,scorer,summary stage allocated peaks)')
    for label,file in [('LED','LED_DEVICE_MEMORY_SAMPLES.json'),('QCAP','QCAP_DEVICE_MEMORY_SAMPLES.json'),('QIMAGE Visual','QIMAGE_DEVICE_MEMORY_SAMPLES.json')]:
        d=read(e/file);measured(label+'_whole_device',label,'sampled job memory','sampled_device_used_peak',d['peak_device_used_MiB'],'MiB',gpu,'nvidia-smi memory.used','Whole-device samples include driver/context,may miss peaks and span load/job phases. Do not compare as if PyTorch allocated.','EVIDENCE/'+file,'/peak_device_used_MiB')
    missing=[('photo_acquisition','All','source collection','Source photo/review acquisition/download/decode total; source cohort construction and overlap checks'),('review_selection','All','source preprocessing','Representative review selection/token caching/source assembly duration'),('filter_selection','All actual selected-photo lineages','caption filtering/selection','Final relevance/redundancy/grouping/diversity/K selection + source construction;178.529s scoring excludes this'),('BLIP_download','Shared prep','cold setup','BLIP/CLIP/SBERT download/hashing/setup durations as complete stage'),('BLIP_cpu_ram','Shared prep','resource usage','Measured CPU RSS/peak RAM during CPU-offloaded BLIP execution; cgroup/budget is not observed usage'),('caption_request_cpu','LED and Qwen-Caption','outside generation','Complete request tokenization,CPU prep,transfers,decoding and serialization timers'),('direct_image_selection','Hypothetical caption-independent Qwen-Image','source selection','Selection of comparable73 photos without BLIP/filter selection has not been run/timed'),('whole_pipeline','Each system','complete pipeline','A same-host,same-source-boundary uninterrupted fresh pipeline wall timer'),('energy_cost','All','resource usage','Power/energy/monetary efficiency; no comparable metering or validated provider rates')]
    for id,system,stage,definition in missing:row(id,system,stage,'duration_or_resource_unavailable','','','NOT MEASURED','NOT MEASURED','NOT MEASURED',definition,'')
    return facts,measurements,native_bytes

def verify(root,repo=None):
    root=Path(root);manifest=read(root/'PACKAGE_MANIFEST.json')
    expected={r['file'] for r in manifest['files']}|{'PACKAGE_MANIFEST.json'}
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected
    for r in manifest['files']:assert (root/r['file']).stat().st_size==r['bytes'] and sha(root/r['file'])==r['sha256'],r['file']
    facts,rows,native=compute(root)
    actual=read(root/'VERIFIED_FACTS.json')
    for key,value in facts.items():
        if isinstance(value,float):close(value,actual[key])
        else:assert value==actual[key],key
    with (root/'EFFICIENCY_MEASUREMENTS.csv').open(encoding='utf8',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(rows)==len(csvrows)
    for expected,actual in zip(rows,csvrows):assert {k:str(v) for k,v in expected.items()}==actual,expected['measurement_id']
    source_checks=0
    if repo:
        repo=Path(repo)
        for r in read(root/'SOURCE_MANIFEST.json')['files']:
            assert sha(repo/r['original_file'])==r['sha256'],r['original_file'];source_checks+=1
    return dict(status='PASS',metric_rows=len(rows),NOT_MEASURED_rows=sum(r['status']=='NOT MEASURED' for r in rows),
        original_native_BLIP_photo_records=len(native),original_native_BLIP_attempts=facts['blip_attempts'],LED_measured_calls=60,Qwen_Caption_calls=30,new_Qwen_Image_Visual_calls=15,
        shared_photo_order_matches=15,selected_photos=73,all_three_selection_lineages_verified=True,
        recomputed_headline_seconds={k:facts[k] for k in ['led_generation_seconds','qcap_generation_seconds','qimage_constructed_generation_seconds','blip_elapsed_seconds','scoring_elapsed_seconds']},
        full_pipeline_performance='NOT MEASURED',repository_source_hashes_checked=source_checks,GPU_calls=0,human_rating_analysis=False)
