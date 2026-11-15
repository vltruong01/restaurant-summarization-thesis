"""Stage O adapter around accepted C–F functions. No automatic downloads or rentals.

Default CPU preparation is safe to run after COHORT15_FREEZE.json exists.
GPU caption/scoring require a later explicit execution approval file and the original assets.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os
import yaml
from tokenizers import Tokenizer
from stage_o_tripadvisor_feasibility_v1 import ROOT,OUT,save,digest,utc
from prepare_stage_o_tripadvisor_cohort_v1 import canonical,h


def jl(path):return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
def writejl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf8')


def verify_freeze():
    f=json.loads((OUT/'COHORT15_FREEZE.json').read_text(encoding='utf8'))
    assert len(f['restaurants'])==15
    for x in f['artifact_bindings']:assert digest(OUT/x['file'])==x['sha256']
    return f


def reviews_cpu():
    from budget_aware_review_selection_v4_2 import search_population
    from token_length_cache import TokenLengthCache
    from source_budget_audit_template import review_block,base_source,pinned_tokenizer,encoding
    frozen=verify_freeze();dest=OUT/'PREPROCESSING';dest.mkdir(exist_ok=True)
    final=dest/'REVIEW_SELECTION_FREEZE.json'
    if final.exists():
        data=json.loads(final.read_text(encoding='utf8'))
        for x in data['files']:assert digest(OUT/x['file'])==x['sha256']
        print('Verified completed review selection; no re-selection.');return
    cfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_expanded.yaml').read_text(encoding='utf8'))
    tok,tokenizer_manifest=pinned_tokenizer();memo=TokenLengthCache(tok)
    shared=[];ledger=[];photos=[]
    for item in frozen['restaurants']:
        source=json.loads((OUT/item['source_file']).read_text(encoding='utf8'));bid=source['business_id']
        pop=source['English_reviews'];assert len(pop)>=10
        choice=search_population(pop,cfg,memo,bid)['choices'][4000]['v4_2_length_balanced']
        assert choice is not None
        byid={r['review_id']:r for r in pop};selected=[byid[rid] for rid in choice['ids']]
        block=review_block(selected);base=base_source(source['business_metadata'],block);task='[TASK_REVIEW_CONSENSUS]\n'+base
        assert len(selected)<=30 and len(encoding(tok,block).ids)<=4000 and len(encoding(tok,task).ids)<=5000
        assert all(hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256'] for r in pop)
        row=dict(business_id=bid,metadata=source['business_metadata'],selected_review_ids=[r['review_id'] for r in selected],
                 reviews=selected,review_block=block,review_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
                 base_source=base,base_sha256=hashlib.sha256(base.encode()).hexdigest(),review_task_source=task,
                 review_task_source_sha256=hashlib.sha256(task.encode()).hexdigest(),review_input_ids=encoding(tok,task).ids,
                 review_input_ids_sha256=h(encoding(tok,task).ids))
        shared.append(row)
        ledger.append(dict(business_id=bid,available_English_reviews=len(pop),selected_reviews=len(selected),review_block_tokens=len(encoding(tok,block).ids),
                           base_tokens=len(encoding(tok,base).ids),review_task_tokens=len(row['review_input_ids']),rating_counts=dict(__import__('collections').Counter(r['stars'] for r in selected))))
        for p in source['photos']:
            assert digest(OUT/p['file'])==p['sha256']
            photos.append(dict(photo_id=p['photo_id'],business_id=bid,name=source['business_metadata']['name'],split='external_tripadvisor',
                               image_path=(OUT/p['file']).relative_to(ROOT).as_posix(),raw_sha256=p['sha256'],pixel_sha256=p['decoded_RGB_sha256'],
                               phash64_hex='NOT_COMPUTED_DIAGNOSTIC_ONLY',source_url=p['download_url']))
        print(bid,'review selection',len(selected),'/',len(pop),'tokens',ledger[-1]['review_block_tokens'],flush=True)
    writejl(dest/'shared_review_inputs.jsonl',shared);save(dest/'REVIEW_BUDGET_LEDGER.json',ledger)
    job=OUT/'PREPARED_CAPTION_JOB';job.mkdir(exist_ok=True)
    save(job/'pilot_photo_plan.json',photos)
    config=yaml.safe_load((ROOT/'configs/stage_f_validation_caption_pool.yaml').read_text(encoding='utf8'))
    (job/'CONFIG.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf8')
    sources={p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'COHORT15_FREEZE.json',dest/'shared_review_inputs.jsonl',ROOT/'configs/basic_caption_quality.yaml',ROOT/'configs/stage_f_validation_protocol.yaml']}
    for photo in photos:sources[photo['image_path']]=photo['raw_sha256']
    identity=dict(source_sha256=sources,basic_policy_sha256=digest(ROOT/'configs/basic_caption_quality.yaml'),plan_sha256=h(photos),cohort_freeze_sha256=digest(OUT/'COHORT15_FREEZE.json'))
    identity['identity_sha256']=h(identity);save(job/'pilot_identity.json',identity)
    save(job/'model_revision.json',{'resolved_commit':config['requested_revision']})
    qwen=[]
    for row in shared:
        source=json.loads((OUT/next(r['source_file'] for r in frozen['restaurants'] if r['business_id']==row['business_id'])).read_text(encoding='utf8'))
        qwen.append(dict(business_id=row['business_id'],selected_review_ids=row['selected_review_ids'],review_block_sha256=row['review_block_sha256'],
                         exact_selected_reviews_file='PREPROCESSING/shared_review_inputs.jsonl',source_photos=[{k:p[k] for k in ['photo_id','file','sha256','source_links']} for p in source['photos']],
                         unfiltered_captions=None,filtered_captions=None,caption_status='PENDING_MISSING_ASSETS_AND_EXECUTION_REVIEW',reference_summary=None,Qwen_config=None,Qwen_inference_authorized=False))
    save(OUT/'QWEN_SHARED_INPUT_MANIFEST.json',dict(status='FROZEN_REVIEWS_AND_PHOTOS_CAPTIONS_PENDING',cohort_freeze_sha256=digest(OUT/'COHORT15_FREEZE.json'),restaurants=qwen))
    files=[dest/'shared_review_inputs.jsonl',dest/'REVIEW_BUDGET_LEDGER.json',job/'pilot_photo_plan.json',job/'pilot_identity.json',job/'CONFIG.yaml',job/'model_revision.json',OUT/'QWEN_SHARED_INPUT_MANIFEST.json']
    save(final,dict(status='PASS_FROZEN_REVIEW_SELECTION_AND_ALL_PHOTO_PLAN',created_at_utc=utc(),restaurants=15,source_photos=len(photos),
                    actual_selected_reviews=sum(x['selected_reviews'] for x in ledger),
                    files=[{'file':p.relative_to(OUT).as_posix(),'sha256':digest(p)} for p in files],
                    original_selector_sha256=digest(ROOT/'scripts/budget_aware_review_selection_v4_2.py'),
                    selector_config_sha256=digest(ROOT/'configs/review_selection_source_length_expanded.yaml'),
                    tokenizer_manifest_sha256=digest(ROOT/'assets/tokenizers/led-base-16384.manifest.json'),
                    original_method_reused=True,new_incumbents_or_new_search_settings=False,
                    schema_conversion='rating /10 exact; day dates padded with 00:00:00; title retained as metadata but model-visible Review uses original text only; absent state empty.',
                    no_GPU_model_calls=True))


def permit(path):
    assert path and path.is_file(),'Later explicit execution approval is required; no automatic paid resources.'
    a=json.loads(path.read_text(encoding='utf8'))
    assert a['scope']=='STAGE_O_FROZEN_CAPTION_AND_LED_INFERENCE_ONLY' and a['execution_approved'] is True
    assert a['cohort_freeze_sha256']==digest(OUT/'COHORT15_FREEZE.json')
    assert a['training_allowed'] is False and a['Qwen_allowed'] is False and a['Teacher_allowed'] is False
    import datetime
    issued=datetime.datetime.fromisoformat(a['issued_at_utc']);end=datetime.datetime.fromisoformat(a['deadline_utc'])
    now=datetime.datetime.now(datetime.timezone.utc)
    assert issued<=now<end and end-issued<=datetime.timedelta(hours=48)
    return a


def caption(approval):
    permit(approval);verify_freeze()
    import blip2_caption_pilot as legacy
    from generate_stage_f_validation_captions import generate_locked
    from filelock import FileLock
    job=OUT/'PREPARED_CAPTION_JOB';cfg=yaml.safe_load((job/'CONFIG.yaml').read_text(encoding='utf8'))
    legacy.preflight(cfg,job)  # Frozen FP16 + GPU/CPU offload profile, no quantization or CPU fallback.
    with FileLock(job/'caption_model_execution.lock',timeout=0):generate_locked(cfg,job)


def score_construct(approval):
    permit(approval);frozen=verify_freeze()
    from stage_f_score_validation import score
    from apply_stage_f_frozen_filters import filter_business
    from safetensors.numpy import load_file
    from source_budget_audit_template import counts,assemble,encoding,pinned_tokenizer
    from s1000_final_eval_core_v1 import isolated
    job=OUT/'PREPARED_CAPTION_JOB';plan=json.loads((job/'pilot_photo_plan.json').read_text(encoding='utf8'))
    records=[]
    for photo in plan:
        p=job/'caption_checkpoints'/(photo['photo_id']+'.json');r=json.loads(p.read_text(encoding='utf8'))
        assert r['raw_sha256']==photo['raw_sha256'] and r['resolved_commit']=='cda95b9319b722f79c9451a3d8ff92eea02048dc'
        r['_record_sha256']=digest(p);r['role']='external_tripadvisor';records.append(r)
    score_dir=OUT/'CAPTION_SCORES';score(records,score_dir)
    rows=json.loads((score_dir/'caption_scores.json').read_text(encoding='utf8'));vectors=load_file(str(score_dir/'embeddings.safetensors'))['SBERT']
    photoby={r['photo_id']:dict(r) for r in plan};recordby={r['photo_id']:r for r in records}
    diagnostics=json.loads((OUT/'SOURCE_IMAGE_DIAGNOSTICS.json').read_text(encoding='utf8'))
    assert diagnostics['Yelp_normalized_pixel_matches']==[]
    for r in diagnostics['photos']:
        photoby[r['photo_id']]['pixel_sha256']=r['pixel_sha256']
        photoby[r['photo_id']]['phash64_hex']=r['phash64_hex']
    reviews=jl(OUT/'PREPROCESSING/shared_review_inputs.jsonl');tok,_=pinned_tokenizer()
    lock=json.loads((ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json').read_text(encoding='utf8'))
    calls=[];pools=[];manifests=[]
    for review in reviews:
        bid=review['business_id'];rr=[r for r in rows if r['business_id']==bid]
        pool=filter_business(bid,rr,vectors,photoby,{'relevance':.08,'redundancy':.59,'grouping':.46});pools.append(pool)
        assert pool==filter_business(bid,list(reversed(rr)),vectors,photoby,{'relevance':.08,'redundancy':.59,'grouping':.46})
        for condition,key in [('RO',None),('U','unfiltered_fixed_seed_ranking'),('F','filtered_diversity_ranking')]:
            pids=pool[key][:5] if key else []
            captions=[recordby[p]['normalized_caption'] for p in pids]
            if condition=='RO':source=review['base_source'];rs=review['review_task_source'];vs=None
            else:
                accounting,source=counts(tok,review['metadata'],review['review_block'],captions)
                assert not any(accounting[x] for x in ['review_budget_overflow','caption_budget_overflow','source_budget_overflow'])
                rs,vs=isolated(source)
            assert rs==review['review_task_source']
            manifests.append(dict(business_id=bid,condition=condition,selected_photo_ids=pids,captions=captions,source=source))
            for task,text in [('review',rs)]+([] if vs is None else [('visual',vs)]):
                ids=encoding(tok,text).ids;assert len(ids)<=5000
                calls.append(dict(condition=condition,business_id=bid,task=task,source=text,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  input_ids=ids,input_ids_sha256=h(ids),checkpoint_sha256=lock['models'][condition]['adapter_sha256'],
                                  selected_review_ids=review['selected_review_ids'] if task=='review' else [],selected_photo_ids=pids if task=='visual' else [],generation=lock['generation']))
    assert len(calls)==75 and len(manifests)==45
    save(OUT/'FINAL_U_F_CAPTION_POOLS.json',pools);save(OUT/'FINAL_CONDITION_INPUTS_45.json',manifests)
    save(OUT/'LED_INFERENCE_CALL_PLAN_75.json',dict(calls=calls,cohort_freeze_sha256=digest(OUT/'COHORT15_FREEZE.json'),no_references=True,no_training=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['reviews-cpu','caption','score-construct']);p.add_argument('--approval',type=Path);a=p.parse_args()
    if a.step=='reviews-cpu':reviews_cpu()
    elif a.step=='caption':caption(a.approval)
    else:score_construct(a.approval)
