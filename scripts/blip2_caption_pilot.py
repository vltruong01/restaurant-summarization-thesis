"""E0 backend called by user Run All in a separate environment, never at import."""
from pathlib import Path
import argparse,csv,gc,hashlib,json,os,platform,re,subprocess,sys,time,traceback
from importlib.metadata import version
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/blip2_caption_feasibility_pilot.yaml'
ORIGIN='user'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def canon_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def check_sources(identity):
    for name,digest in identity['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Changed source: '+name)


def prepare(cfg,out):
    policy=yaml.safe_load((ROOT/'configs/image_overlap_evaluation_policy.yaml').read_text(encoding='utf-8'))
    assert policy['preserve_readable_photo_pool'] and not policy['cross_split_image_exclusion_applied']
    photo_root=ROOT/'outputs/stage_d/photo_integrity_duplicate_audit_v1'
    assert read(photo_root/'validation_report.json')['validation_passed']
    photos=csv_rows(photo_root/'unfiltered_readable_photo_pool.csv');info=csv_rows(ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/businesses.csv')
    pools={}
    for photo in photos:
        if photo['split'] in cfg['pilot']['allowed_splits']:pools.setdefault(photo['business_id'],[]).append(photo)
    selected=[]
    for band in cfg['pilot']['review_volume_bands']:
        for split in cfg['pilot']['allowed_splits']:
            choices=[r for r in info if r['volume_band']==band and r['split']==split and len(pools.get(r['business_id'],[]))>=cfg['pilot']['images_per_business']]
            choices.sort(key=lambda r:hashlib.sha256(f"42|E0_business|{band}|{split}|{r['business_id']}".encode()).hexdigest())
            n=cfg['pilot']['businesses_per_split_per_band']
            if len(choices)<n:raise RuntimeError('Insufficient predeclared pilot stratum: '+band+'/'+split)
            selected.extend(choices[:n])
    plan=[]
    for business in selected:
        candidates=sorted(pools[business['business_id']],key=lambda r:hashlib.sha256(('42|E0_photo|'+r['photo_id']).encode()).hexdigest())
        for photo in candidates[:cfg['pilot']['images_per_business']]:
            path=ROOT/'data/processed/yelp_2300_v1'/photo['relative_path']
            assert sha(path)==photo['raw_sha256']
            plan.append({**photo,'name':business['name'],'volume_band':business['volume_band'],'image_path':path.relative_to(ROOT).as_posix(),'caption_id':'E0_'+photo['photo_id'],'final_caption_set':False})
    assert len(plan)==24 and len({r['business_id'] for r in plan})==8 and not any(r['split']=='test' for r in plan)
    source_names=['configs/image_overlap_evaluation_policy.yaml','configs/blip2_caption_feasibility_pilot.yaml','configs/research_pending.yaml','requirements/caption_pilot.in','scripts/blip2_caption_pilot.py','scripts/caption_pilot_notebook.py','outputs/stage_d/photo_integrity_duplicate_audit_v1/unfiltered_readable_photo_pool.csv','outputs/stage_d/photo_integrity_duplicate_audit_v1/test_image_disjoint_diagnostic_businesses.csv','outputs/stage_c/review_selection_source_length_expanded_v1/businesses.csv','outputs/stage_c/review_selection_source_length_expanded_v1/selection_runs.jsonl','outputs/stage_c/led_source_token_allocation_audit_v1/source_token_ledger.csv','data/processed/yelp_2300_v1/businesses.jsonl','data/processed/yelp_2300_english_v1/reviews/train.jsonl','data/processed/yelp_2300_english_v1/reviews/validation.jsonl','assets/tokenizers/led-base-16384/tokenizer.json']
    identity={'config_sha256':sha(CONFIG),'source_sha256':{name:sha(ROOT/name) for name in source_names},'plan_sha256':canon_hash(plan),'image_overlap_policy':policy,'execution':'managed_or_user_E0_run'}
    identity['identity_sha256']=canon_hash(identity)
    if (out/'pilot_identity.json').is_file():
        old=read(out/'pilot_identity.json')
        if old['identity_sha256']!=identity['identity_sha256']:raise RuntimeError('Incompatible existing E0 pilot identity; use a new output_directory. Existing captions preserved.')
        assert read(out/'pilot_photo_plan.json')==plan
    else:save(out/'pilot_identity.json',identity);save(out/'pilot_photo_plan.json',plan)
    subset=csv_rows(ROOT/policy['secondary_membership_file']);approved=[r['business_id'] for r in subset if r[policy['secondary_membership_column']].lower()=='true']
    assert len(approved)==83 and len(subset)==100
    save(out/'approved_evaluation_protocol.json',{'decision':'D-022','main_test_businesses':100,'secondary_business_ids':approved,'secondary_test_businesses':83,'evaluated':False,'preserve_pool_cohort_splits':True,'exact_only_not_near_or_pretraining_independence':True})
    report={'status':'pilot_prepared','businesses':8,'photos':24,'split_counts':{s:sum(r['split']==s for r in plan) for s in ['train','validation']},'identity_sha256':identity['identity_sha256'],'source_changed':False,'caption_generation_started':False,'final_k':None,'execution_origin':ORIGIN,'agent_executed':ORIGIN=='agent'}
    save(out/'pilot_preparation_report.json',report);return report


def preflight(cfg,out):
    import torch,psutil
    import shutil
    report={'status':'checking','python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'versions':{p:version(p) for p in ['torch','torchvision','transformers','tokenizers','huggingface-hub','accelerate','numpy','Pillow','psutil','safetensors']},'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'available_RAM_GiB':psutil.virtual_memory().available/2**30,'free_disk_GiB':shutil.disk_usage(ROOT).free/2**30,'quantization':'none','weight_dtype':'float16','execution_origin':ORIGIN,'agent_executed':ORIGIN=='agent'}
    (out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,encoding='utf-8'),encoding='utf-8')
    reasons=[]
    if not report['cuda_available']:reasons.append('CUDA unavailable in isolated environment; no automatic CPU/quantization fallback.')
    else:
        free,total=torch.cuda.mem_get_info(0);budget=min(free,total)-int(cfg['device']['gpu_safety_reserve_GiB']*2**30)
        report.update(gpu_name=torch.cuda.get_device_name(0),GPU_total_GiB=total/2**30,GPU_free_GiB=free/2**30,GPU_weight_budget_bytes=budget)
        if budget<2*2**30:reasons.append('Less than2GiB GPU budget after safety reserve; close other GPU jobs or review feasibility.')
    if report['available_RAM_GiB']<cfg['device']['minimum_available_RAM_GiB']:reasons.append('Insufficient available RAM for declared GPU/CPU offload profile.')
    if report['free_disk_GiB']<cfg['device']['minimum_free_disk_GiB']:reasons.append('Insufficient free disk for model cache/install/load profile.')
    report['CPU_weight_budget_bytes']=int(max(0,report['available_RAM_GiB']-cfg['device']['cpu_safety_reserve_GiB'])*2**30)
    report['status']='blocked' if reasons else 'preflight_passed_not_model_load_passed';report['reasons']=reasons
    save(out/'environment_report.json',report)
    if reasons:raise RuntimeError('; '.join(reasons))
    return report


def surface(text,cfg):
    words=re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*",text)
    normal=re.sub(r'\s+',' ',text.lower()).strip(' .!?\n\t')
    flags=[]
    if len(words)<cfg['surface_diagnostics']['minimum_words']:flags.append('under5_words')
    if not text.rstrip().endswith(tuple(cfg['surface_diagnostics']['terminal_punctuation'])):flags.append('no_terminal_punctuation')
    if normal in cfg['surface_diagnostics']['exact_generic_phrases']:flags.append('exact_generic_phrase')
    return {'word_count':len(words),'surface_flags':flags,'surface_eligible':not flags,'grammar_information_quality':'manual_review_pending'}


def generate(cfg,out,allow_download,max_new_photos=None):
    from filelock import FileLock
    with FileLock(out/'caption_model_execution.lock',timeout=0):
        return _generate(cfg,out,allow_download,max_new_photos)


def _generate(cfg,out,allow_download,max_new_photos=None):
    if max_new_photos is not None and max_new_photos<1:raise ValueError('Execution limit must be positive; it does not change the fixed24-photo plan.')
    identity=read(out/'pilot_identity.json');check_sources(identity)
    env=read(out/'environment_report.json')
    if env['status']!='preflight_passed_not_model_load_passed':raise RuntimeError('Preflight blocked: '+'; '.join(env.get('reasons',[]))+'. No model download/load or caption generation started.')
    import torch
    from PIL import Image,ImageOps
    from transformers import Blip2Processor,Blip2ForConditionalGeneration
    from huggingface_hub import HfApi
    started=time.perf_counter()
    cache=ROOT/cfg['model_cache'];cache.mkdir(parents=True,exist_ok=True)
    if (out/'model_revision.json').is_file():revision=read(out/'model_revision.json')['resolved_commit']
    else:
        if not allow_download:raise RuntimeError('No pinned model revision yet; enable authorized model download in notebook.')
        revision=HfApi().model_info(cfg['model_id'],revision=cfg['requested_revision']).sha
        if not revision:raise RuntimeError('Cannot resolve model commit; do not generate from unpinned main.')
        save(out/'model_revision.json',{'model_id':cfg['model_id'],'requested_revision':cfg['requested_revision'],'resolved_commit':revision})
    runtime={'model_id':cfg['model_id'],'resolved_commit':revision,'plan_identity':identity['identity_sha256'],'generation':cfg['generation'],'surface':cfg['surface_diagnostics'],'weight_dtype':'float16','quantization':'none','versions':env['versions'],'gpu_name':env['gpu_name'],'torch_cuda':env['torch_cuda']};runtime_sha=canon_hash(runtime)
    checkpoints=out/'caption_checkpoints';checkpoints.mkdir(exist_ok=True);plan=read(out/'pilot_photo_plan.json');records={}
    for photo in plan:
        path=checkpoints/(photo['photo_id']+'.json')
        if path.is_file():
            record=read(path)
            if record['runtime_sha256']!=runtime_sha or record['raw_sha256']!=photo['raw_sha256']:raise RuntimeError('Incompatible caption checkpoint: '+photo['photo_id'])
            records[photo['photo_id']]=record
    previous=read(out/'generation_report.json') if (out/'generation_report.json').is_file() else None
    if len(records)==len(plan) and previous and previous.get('status')=='raw_pilot_complete_pending_review':
        assert previous['runtime_sha256']==runtime_sha
        print('Compatible24-photo raw pilot exists; reusing captions without loading model.',flush=True);return previous
    memory={0:int(env['GPU_weight_budget_bytes']),'cpu':int(env['CPU_weight_budget_bytes'])}
    model=None
    try:
        print(f'Loading {cfg["model_id"]} commit{revision}, FP16 GPU/CPU offload, no quantization.',flush=True)
        processor=Blip2Processor.from_pretrained(cfg['model_id'],revision=revision,cache_dir=cache,local_files_only=not allow_download,trust_remote_code=False)
        model=Blip2ForConditionalGeneration.from_pretrained(cfg['model_id'],revision=revision,cache_dir=cache,local_files_only=not allow_download,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float16,device_map='auto',max_memory=memory,low_cpu_mem_usage=True,offload_state_dict=True)
        model.eval();device_map={k:str(v) for k,v in model.hf_device_map.items()}
        if any(value=='disk' for value in device_map.values()):raise RuntimeError('Unexpected disk offload; declared pilot permits GPU/CPU only.')
        map_sha=canon_hash(device_map)
        if previous and records and previous.get('device_map_sha256')!=map_sha:raise RuntimeError('Partial run device map differs; do not mix runtimes. Use original profile or a new output directory.')
        save(out/'model_load_report.json',{'status':'loaded_not_pilot_verified','runtime':runtime,'runtime_sha256':runtime_sha,'device_map':device_map,'device_map_sha256':map_sha,'processor_query_tokens':getattr(processor,'num_query_tokens',None),'model_image_token_index':getattr(model.config,'image_token_index',None),'model_load_seconds':round(time.perf_counter()-started,3),'GPU_allocated_GiB':torch.cuda.memory_allocated()/2**30,'module_weight_dtypes':{name:str(next(module.parameters()).dtype) for name,module in [('vision',model.vision_model),('qformer',model.qformer),('language',model.language_model)]}})
        report={'status':'partial_raw_pilot','planned':len(plan),'completed':len(records),'runtime_sha256':runtime_sha,'device_map':device_map,'device_map_sha256':map_sha,'caption_generation_started':False,'elapsed_seconds':round(time.perf_counter()-started,3),'execution_origin':ORIGIN}
        def one_attempt(rgb,prompt):
            torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            inputs=processor(images=rgb,return_tensors='pt',**({'text':prompt} if prompt else {}))
            inputs={key:value.to('cuda:0',dtype=torch.float16) if value.is_floating_point() else value.to('cuda:0') for key,value in inputs.items()}
            prefix=inputs['input_ids'].shape[1] if 'input_ids' in inputs else 1+(model.config.num_query_tokens if getattr(model.config,'image_token_index',None) is not None else 0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                generated=model.generate(**inputs,num_beams=cfg['generation']['num_beams'],do_sample=False,max_new_tokens=cfg['generation']['max_new_tokens'])
            torch.cuda.synchronize();suffix=generated[0,prefix:];text=processor.tokenizer.decode(suffix,skip_special_tokens=True).strip()
            return {'prompt':prompt,'raw_caption':text,**surface(text,cfg),'generated_suffix_tokens':int(suffix.numel()),'seconds':round(time.perf_counter()-begin,3),'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_GPU_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
        new_photos=0
        for number,photo in enumerate(plan,1):
            if photo['photo_id'] in records:continue
            path=ROOT/photo['image_path'];assert sha(path)==photo['raw_sha256']
            with Image.open(path) as original:
                original.load();rgb=ImageOps.exif_transpose(original).convert('RGB')
                attempts=[one_attempt(rgb,cfg['generation']['initial_prompt'])]
                if not attempts[0]['surface_eligible'] and cfg['generation']['retry_once_if_surface_failure']:attempts.append(one_attempt(rgb,cfg['generation']['retry_prompt']))
            chosen=next((i for i,a in enumerate(attempts) if a['surface_eligible']),0)
            record={**photo,'runtime_sha256':runtime_sha,'device_map_sha256':map_sha,'model_id':cfg['model_id'],'resolved_commit':revision,'attempts':attempts,'chosen_attempt':chosen+1,'raw_caption':attempts[chosen]['raw_caption'],'surface_eligible':attempts[chosen]['surface_eligible'],'generation_settings':cfg['generation'],'final_caption_set':False,'teacher_target':False}
            save(checkpoints/(photo['photo_id']+'.json'),record);records[photo['photo_id']]=record
            report={'status':'partial_raw_pilot','planned':len(plan),'completed':len(records),'runtime_sha256':runtime_sha,'device_map':device_map,'device_map_sha256':map_sha,'caption_generation_started':True,'elapsed_seconds':round(time.perf_counter()-started,3),'execution_origin':ORIGIN}
            save(out/'generation_report.json',report);print(f'{number}/{len(plan)} {photo["name"]}: {record["raw_caption"]} | surface_ok={record["surface_eligible"]}',flush=True)
            new_photos+=1
            if max_new_photos is not None and new_photos>=max_new_photos and len(records)<len(plan):
                report.update(status='partial_raw_pilot_smoke_pending_review',invocation_limit_new_photos=max_new_photos,new_photos_this_invocation=new_photos,full_pilot_complete=False)
                save(out/'generation_report.json',report);check_sources(identity);return report
        report.update(status='raw_pilot_complete_pending_review',surface_eligible=sum(r['surface_eligible'] for r in records.values()),manual_picture_caption_review='pending',production_settings_frozen=False)
        save(out/'generation_report.json',report);check_sources(identity);return report
    finally:
        if model is not None:del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def report_pilot(cfg,out):
    from tokenizers import Tokenizer
    generation=read(out/'generation_report.json');assert generation['status']=='raw_pilot_complete_pending_review'
    identity=read(out/'pilot_identity.json');check_sources(identity);plan=read(out/'pilot_photo_plan.json');business_ids={r['business_id'] for r in plan}
    records=[read(out/'caption_checkpoints'/(r['photo_id']+'.json')) for r in plan]
    selections={}
    with (ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/selection_runs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['business_id'] in business_ids and r['method']=='v4_2_length_balanced' and r['review_budget']==4000:selections[r['business_id']]=r
    needed={rid for r in selections.values() for rid in r['selected_review_ids']};reviews={}
    for split in ['train','validation']:
        with (ROOT/'data/processed/yelp_2300_english_v1/reviews'/f'{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)
                if r['review_id'] in needed:reviews[r['review_id']]=r
    assert set(reviews)==needed and set(selections)==business_ids
    businesses={r['business_id']:r for r in map(json.loads,(ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open(encoding='utf-8')) if r['business_id'] in business_ids}
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_truncation();tok.no_padding()
    ledger=[];sources=[]
    for bid in sorted(business_ids):
        choice=selections[bid];selected=[reviews[rid] for rid in choice['selected_review_ids']];assert selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
        review_block='\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i,r in enumerate(selected,1))
        assert len(tok.encode(review_block,add_special_tokens=True).ids)==choice['tokens_after_budget_enforcement']
        metadata='[BUSINESS]\n'+'\n'.join(f"{field}={businesses[bid].get(field) or ''}" for field in ['name','categories','city','state'])
        base=metadata+'\n\n[REVIEWS]\n'+review_block;base_tokens=len(tok.encode(base,add_special_tokens=True).ids)
        current=[r for r in records if r['business_id']==bid]
        for condition,caption_records in [('all_raw_pilot_captions_capacity_only',current),('surface_eligible_pilot_captions_NOT_final_filtered_set',[r for r in current if r['surface_eligible']])]:
            block='\n'.join(cfg['source_diagnostics']['caption_template'].format(index=i,text=r['raw_caption']) for i,r in enumerate(caption_records,1))
            full=base+('\n\n[CAPTIONS]\n'+block if caption_records else '')
            encoding=tok.encode(full,add_special_tokens=True);assert sum(encoding.special_tokens_mask)==2
            total=len(encoding.ids);delta=total-base_tokens
            ledger.append({'business_id':bid,'name':businesses[bid]['name'],'split':choice['split'],'condition':condition,'review_budget_provisional':4000,'retained_reviews':len(selected),'captions':len(caption_records),'text_only_base_tokens':base_tokens,'actual_full_source_tokens':total,'actual_incremental_caption_block_tokens':delta,'within_provisional_caption750':delta<=750,'within_diagnostic_total5000':total<=5000,'review_IDs_unchanged':True,'caption_set_final':False})
            sources.append({**ledger[-1],'selected_review_ids':choice['selected_review_ids'],'caption_photo_ids':[r['photo_id'] for r in caption_records],'complete_provisional_source':full,'teacher_or_training_input':False})
    flat=[]
    for r in records:
        chosen=r['attempts'][r['chosen_attempt']-1]
        flat.append({'business_id':r['business_id'],'name':r['name'],'split':r['split'],'photo_id':r['photo_id'],'caption_id':r['caption_id'],'image_path':r['image_path'],'raw_caption':r['raw_caption'],'attempts':len(r['attempts']),'chosen_attempt':r['chosen_attempt'],'surface_eligible':r['surface_eligible'],'word_count':chosen['word_count'],'surface_flags':'|'.join(chosen['surface_flags']),'LED_body_tokens':len(tok.encode(r['raw_caption'],add_special_tokens=False).ids),'seconds_all_attempts':sum(a['seconds'] for a in r['attempts']),'peak_GPU_allocated_GiB':max(a['peak_GPU_allocated_GiB'] for a in r['attempts'])})
    for filename,rows in [('raw_caption_statistics.csv',flat),('pilot_source_token_ledger.csv',ledger)]:
        with (out/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(out/'complete_provisional_source_examples.json',sources)
    result={'status':'generated_outputs_pending_review','pilot_businesses':8,'pilot_photos':24,'caption_attempts':sum(len(r['attempts']) for r in records),'surface_eligible_photos':sum(r['surface_eligible'] for r in records),'source_rows_measured':len(ledger),'max_raw_caption_LED_body_tokens':max(r['LED_body_tokens'] for r in flat),'max_provisional_full_source_tokens':max(r['actual_full_source_tokens'] for r in ledger),'caption750_overflows':sum(not r['within_provisional_caption750'] for r in ledger),'source5000_overflows':sum(not r['within_diagnostic_total5000'] for r in ledger),'generation_report':generation,'manual_picture_caption_review':'pending','final_K':None,'final_caption_or_source_allocation':None,'final_manifest':False,'teacher_or_LED_training_started':False,'execution_origin':ORIGIN,'agent_executed':ORIGIN=='agent'}
    check_sources(identity);save(out/'pilot_report.json',result);return result


def main():
    global ORIGIN
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','preflight','generate','report']);parser.add_argument('--allow-model-download',action='store_true');parser.add_argument('--max-new-photos',type=int,default=None);parser.add_argument('--execution-origin',choices=['agent','user'],default='user');args=parser.parse_args();ORIGIN=args.execution_origin
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];out.mkdir(parents=True,exist_ok=True)
    try:
        result=prepare(cfg,out) if args.step=='prepare' else preflight(cfg,out) if args.step=='preflight' else generate(cfg,out,args.allow_model_download,args.max_new_photos) if args.step=='generate' else report_pilot(cfg,out)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        checkpoints=out/'caption_checkpoints';has_outputs=checkpoints.is_dir() and any(checkpoints.glob('*.json'))
        failure={'status':'blocked','step':args.step,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'caption_generation_may_be_partial':has_outputs,'caption_checkpoints_exist':has_outputs,'automatic_model_quantization_source_budget_changes':False,'execution_origin':ORIGIN,'agent_executed':ORIGIN=='agent'}
        save(out/(args.step+'_failure.json'),failure);print(json.dumps(failure,ensure_ascii=False,indent=2),flush=True);raise


if __name__=='__main__':main()
