"""Frozen complete-pool scoring, condition construction and fail-closed source validation."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,json,os
for option in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(option,'1')
import numpy as np
import orjson
from safetensors.numpy import load_file
from stage_f_verification_workflow import read,write,sha
from stage_f_d033_protocol import freeze_method,STATE
from source_budget_audit_template import review_block,base_source,assemble,counts,pinned_tokenizer,encoding,text_sha
from apply_stage_f_frozen_filters import filter_business

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_v1'
VAL=ROOT/'outputs/stage_f/validation_calibration_v1'
GEN=OUT/'generation_train_test'
THRESHOLDS={'relevance':.08,'redundancy':.59,'grouping':.46}


def immutable(path,value):
    if path.exists():assert read(path)==value,f'Existing output incompatible: {path}'
    else:write(path,value)


def caption_pool():
    method=freeze_method();plan=read(OUT/'complete_photo_plan.json');reused=read(OUT/'reused_caption_sources.json')
    import yaml
    from caption_basic_quality import evaluate
    policy=yaml.safe_load((ROOT/'configs/basic_caption_quality.yaml').read_text(encoding='utf-8'))
    byid={r['photo_id']:r for r in plan};records={}
    for pid,photo in byid.items():
        item=reused.get(pid)
        path=ROOT/item['checkpoint_path'] if item else GEN/'caption_checkpoints'/f'{pid}.json'
        assert path.exists(),f'Required caption missing {pid}; do not preselect five photos'
        if item:assert sha(path)==item['checkpoint_sha256']
        r=read(path)
        assert r['photo_id']==pid and r['business_id']==photo['business_id'] and r['split']==photo['split']
        assert r['raw_sha256']==photo['raw_sha256'] and r['resolved_commit']==method['model_revision']
        assert r['model_id']==method['model_id'] and r['generation_settings']==method['generation']
        assert r['basic_quality_policy_sha256']==method['basic_policy_sha256']
        attempts=r['attempts'];assert len(attempts) in [1,2]
        assert attempts[0]['prompt']==method['generation']['initial_prompt']
        assert (len(attempts)==2)==attempts[0]['basic_quality_fail']
        if len(attempts)==2:assert attempts[1]['prompt']==method['generation']['retry_prompt']
        for attempt in attempts:
            qc=evaluate(attempt,policy)
            for key,value in qc.items():assert attempt[key]==value,f'Basic Quality mismatch: {pid}/{key}'
        chosen=next((i for i,a in enumerate(attempts) if not a['basic_quality_fail']),0)
        assert r['chosen_attempt']==chosen+1
        assert r['raw_caption']==attempts[chosen]['raw_caption'] and r['normalized_caption']==attempts[chosen]['normalized_caption']
        assert r['basic_eligible']==(not attempts[chosen]['basic_quality_fail'])
        assert sha(ROOT/r['image_path'])==r['raw_sha256'],'Source image changed'
        # Attach current input role only to in-memory copies; original checkpoints are untouched.
        r['role']='production_'+r['split'];r['_record_sha256']=sha(path)
        r['_record_path']=path.relative_to(ROOT).as_posix();records[pid]=r
    assert len(records)==30027
    return records,byid


def score_production():
    from stage_f_score_validation import score
    records,plan=caption_pool();bybusiness=defaultdict(list)
    for r in records.values():
        if r['split']!='validation':bybusiness[r['business_id']].append(r)
    # Business-preserving deterministic operational chunks; no photo/ranking preselection.
    chunks=[];current=[]
    for bid,rows in sorted(bybusiness.items()):
        rows=sorted(rows,key=lambda r:r['photo_id'])
        if current and len(current)+len(rows)>2000:chunks.append(current);current=[]
        current+=rows
    if current:chunks.append(current)
    registry=[]
    for i,chunk in enumerate(chunks):
        folder=OUT/'scores'/f'chunk_{i:03d}'
        identity={'protocol_sha256':sha(ROOT/'configs/stage_f_validation_protocol.yaml'),
                  'scoring_code_sha256':sha(ROOT/'scripts/stage_f_score_validation.py'),
                  'records':{r['photo_id']:r['_record_sha256'] for r in chunk}}
        if (folder/'chunk_receipt.json').exists():
            receipt=read(folder/'chunk_receipt.json');assert receipt['identity']==identity
            for filename,digest in receipt['outputs'].items():assert sha(folder/filename)==digest
        else:
            # Partial chunks are not a finished scientific output; never replace a completed chunk.
            report=score(chunk,folder)
            assert report['overlength']==0,'Unexpected scoring overlength: protocol blocker, no silent truncation'
            scored=read(folder/'caption_scores.json')
            assert all(r['score_status'] in ['scored','basic_failure_not_scored'] for r in scored)
            assets={name:sha(folder/name) for name in ['caption_scores.json','scoring_report.json']}
            if (folder/'embeddings.safetensors').exists():assets['embeddings.safetensors']=sha(folder/'embeddings.safetensors')
            receipt={'identity':identity,'outputs':assets,'status':'PASS'};write(folder/'chunk_receipt.json',receipt)
        registry.append({'folder':folder.relative_to(ROOT).as_posix(),'receipt_sha256':sha(folder/'chunk_receipt.json')})
        print(f'Pinned production score chunk {i+1}/{len(chunks)} complete',flush=True)
    immutable(OUT/'scores/complete_scoring_registry.json',registry)


def scoring_namespaces():
    entries=[]
    for role in ['calibration','K_study']:
        folder=VAL/('scores_'+role)
        from stage_f_d033_protocol import assert_scoring_binding
        assert_scoring_binding(role,read(folder/'scoring_report.json'));entries.append(folder)
    registry=read(OUT/'scores/complete_scoring_registry.json')
    for item in registry:
        folder=ROOT/item['folder'];assert sha(folder/'chunk_receipt.json')==item['receipt_sha256']
        receipt=read(folder/'chunk_receipt.json')
        for name,digest in receipt['outputs'].items():assert sha(folder/name)==digest
        entries.append(folder)
    return entries


def stats(values):
    a=np.asarray(values,dtype=np.int64)
    assert len(a)
    return {'n':len(a),'min':int(a.min()),'mean':float(a.mean()),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'max':int(a.max()),'total':int(a.sum())}


def condition_manifest(bid,split,pool,records,condition):
    key='filtered_diversity_ranking' if condition=='filtered' else 'unfiltered_fixed_seed_ranking'
    ids=pool[key][:5]
    selected=[]
    for pid in ids:
        r=records[pid];assert r['business_id']==bid and r['split']==split and r['basic_eligible']
        selected.append({k:r[k] for k in ['business_id','photo_id','split','raw_caption','normalized_caption',
              'basic_eligible','surface_ok','chosen_attempt','attempts','model_id','resolved_commit',
              'generation_settings','runtime_sha256','device_map_sha256','basic_quality_policy_sha256']})
        selected[-1].update(caption_record_path=r['_record_path'],caption_record_sha256=r['_record_sha256'])
    assert len(ids)==min(5,len(pool[key])) and len(ids)==len(set(ids))
    return {'business_id':bid,'split':split,'condition':condition,'K':5,'final_caption_set':True,
            'selected_photo_ids':ids,'captions':selected,'available_caption_count':len(pool[key]),
            'retained_caption_count':len(ids),'empty_caption_business':not ids,
            'frozen_method_sha256':sha(STATE/'frozen_production_method.json')}


def construct():
    method=freeze_method();records,plan=caption_pool()
    cohort={r['business_id']:r['split'] for r in csv.DictReader((ROOT/'data/processed/yelp_2300_english_v1/business_split.csv').open(encoding='utf-8'))}
    pools={};seen=set()
    historical={r['business_id']:r for r in read(VAL/'fixed_filter_pools.json')['businesses']}
    for folder in scoring_namespaces():
        scored=read(folder/'caption_scores.json');bybusiness=defaultdict(list)
        vectors=load_file(str(folder/'embeddings.safetensors'))['SBERT'] if (folder/'embeddings.safetensors').exists() else np.empty((0,384),dtype=np.float32)
        assert vectors.dtype==np.float32 and np.isfinite(vectors).all()
        for r in scored:
            pid=r['photo_id'];source=records[pid]
            assert pid not in seen and r['normalized_caption']==source['normalized_caption'] and r['raw_caption']==source['raw_caption']
            assert r['caption_record_sha256']==source['_record_sha256'] and r['raw_image_sha256']==source['raw_sha256']
            assert r['basic_eligible']==source['basic_eligible']
            seen.add(pid);bybusiness[r['business_id']].append(r)
        for bid,rows in sorted(bybusiness.items()):
            assert bid not in pools,'Businesses must not be split across score namespaces'
            p=filter_business(bid,rows,vectors,plan,THRESHOLDS)
            # Recompute independently with reversed row input to check deterministic invariance.
            repeat=filter_business(bid,list(reversed(rows)),vectors,plan,THRESHOLDS)
            assert p==repeat,'Frozen selection is not deterministic'
            if bid in historical:assert p==historical[bid],'Previously fixed validation pools changed'
            p.update(role='production_'+cohort[bid],split=cohort[bid],K=5,final_caption_set=True)
            pools[bid]=p
    assert seen==set(records) and set(pools)==set(cohort)
    refs={r['business_id']:r for r in read(OUT/'review_inputs/shared_review_references.json')}
    assert set(refs)==set(cohort)
    wanted={rid for r in refs.values() for rid in r['selected_review_ids']};reviews={}
    for split in ['train','validation','test']:
        for line in (ROOT/f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl').open('rb'):
            r=orjson.loads(line)
            if r['review_id'] in wanted:reviews[r['review_id']]=r
    assert set(reviews)==wanted
    businesses={orjson.loads(line)['business_id']:orjson.loads(line) for line in (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open('rb')}
    oldval={r['business_id']:r for r in read(ROOT/'outputs/stage_f/methodology_amendment_d032/frozen_shared_review_reference.json')}
    previousK5={(r['business_id'],r['condition']):r for r in read(ROOT/'outputs/stage_f/k_sensitivity_d032_v1/K_business_ledger.json') if r['K']==5}
    tok,tokenizer_manifest=pinned_tokenizer()
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(ROOT/tokenizer_manifest['relative_directory'],local_files_only=True)
    slow.model_max_length=100000000
    manifests=defaultdict(list);sources=defaultdict(list);shared=[];ledger=[];overflows=[];slow_checks=0
    for bid in sorted(cohort):
        split=cohort[bid];ref=refs[bid];rr=[reviews[rid] for rid in ref['selected_review_ids']]
        assert all(r['business_id']==bid for r in rr)
        block=review_block(rr);assert text_sha(block)==ref['review_block_sha256']
        base=base_source(businesses[bid],block)
        identity={'business_id':bid,'split':split,'selected_review_ids':ref['selected_review_ids'],
                  'review_text_sha256_in_order':[text_sha(r['text']) for r in rr],
                  'review_block_sha256':text_sha(block),'base_sha256':text_sha(base),'review_block':block,
                  'metadata':{k:businesses[bid].get(k) for k in ['name','categories','city','state']}}
        if bid in oldval:
            old=oldval[bid]
            for key in ['selected_review_ids','review_text_sha256_in_order','review_block_sha256','base_sha256']:
                assert identity[key]==old[key],'Existing shared review IDs/text/order/base changed'
        shared.append(identity)
        for condition in ['reviews_only','unfiltered','filtered']:
            if condition=='reviews_only':
                source=base;native=encoding(tok,source).ids
                cost={'review_block_tokens_with_special':len(encoding(tok,block).ids),'reviews_only_base_tokens':len(native),
                      'caption_block_increment_tokens':0,'full_source_tokens':len(native),
                      'review_budget_overflow':len(encoding(tok,block).ids)>4000,
                      'caption_budget_overflow':False,'source_budget_overflow':len(native)>5000,'source_sha256':text_sha(source)}
                ids=[]
            else:
                manifest=condition_manifest(bid,split,pools[bid],records,condition)
                manifests[split,condition].append(manifest);ids=manifest['selected_photo_ids']
                texts=[c['normalized_caption'] for c in manifest['captions']]
                cost,source=counts(tok,businesses[bid],block,texts);native=encoding(tok,source).ids
                if (bid,condition) in previousK5:
                    old=previousK5[bid,condition]
                    assert ids==old['selected_caption_photo_ids'] and text_sha(source)==old['source_sha256']
            # Full source IDs, never approximate component addition or tokenizer truncation.
            assert native==slow.encode(source,add_special_tokens=True,truncation=False)
            slow_checks+=1
            row={'business_id':bid,'split':split,'condition':condition,'K':5,'selected_photo_ids':ids,
                 'selected_review_ids':ref['selected_review_ids'],'review_block_sha256':text_sha(block),
                 'base_sha256':text_sha(base),**cost}
            ledger.append(row);sources[split,condition].append({'business_id':bid,'split':split,'condition':condition,
                       'source':source,'source_sha256':text_sha(source),'tokens':len(native),
                       'selected_review_ids':ref['selected_review_ids'],'review_block_sha256':text_sha(block),
                       'selected_photo_ids':ids,'teacher_target':False})
            if any(row[k] for k in ['review_budget_overflow','caption_budget_overflow','source_budget_overflow']):overflows.append(row)
    if overflows:
        write(OUT/'budget_blocker.json',{'status':'STOP_PRODUCTION_BUDGET_OVERFLOW_NO_SILENT_CHANGES','cases':overflows})
        raise RuntimeError(f'Production budget overflow: {len(overflows)}; no final manifests published')
    artifacts={}
    def jsonl(relative,rows):
        path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        payload=b''.join(orjson.dumps(row,option=orjson.OPT_SORT_KEYS)+b'\n' for row in rows)
        if path.exists():assert path.read_bytes()==payload
        else:
            tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
        artifacts[relative]=sha(path)
    jsonl('manifests/shared_review_inputs.jsonl',shared)
    jsonl('manifests/complete_caption_pool_inventory.jsonl',[
        {'photo_id':pid,'business_id':r['business_id'],'split':r['split'],
         'caption_record_path':r['_record_path'],'caption_record_sha256':r['_record_sha256'],
         'raw_caption':r['raw_caption'],'normalized_caption':r['normalized_caption'],
         'basic_eligible':r['basic_eligible'],'chosen_attempt':r['chosen_attempt'],
         'retry_attempts':len(r['attempts'])-1,'model_id':r['model_id'],'resolved_commit':r['resolved_commit'],
         'generation_settings':r['generation_settings'],'basic_quality_policy_sha256':r['basic_quality_policy_sha256']}
        for pid,r in sorted(records.items())])
    jsonl('manifests/token_ledger.jsonl',ledger)
    jsonl('manifests/frozen_filter_pools.jsonl',[pools[bid] for bid in sorted(pools)])
    for (split,condition),rows in sorted(manifests.items()):jsonl(f'manifests/{split}/{condition}_captions.jsonl',rows)
    for (split,condition),rows in sorted(sources.items()):jsonl(f'manifests/{split}/{condition}_sources.jsonl',rows)
    for relative,digest in read(STATE/'protected_before_sha256.json').items():
        assert sha(ROOT/relative)==digest,f'Historical scientific evidence changed: {relative}'
    report={'status':'STAGE_F_PRODUCTION_COMPLETE_VALIDATED_STOP_BEFORE_STAGE_G','decision':'D-033','K':5,
            'method_sha256':sha(STATE/'frozen_production_method.json'),'cohort_businesses':2300,
            'pinned_tokenizer':tokenizer_manifest,'budgets':{'full_source':5000,'review':4000,'caption':750},
            'split_reports':{},'source_ID_verification_checks':slow_checks,'budget_overflows':0,
            'selected_review_IDs_text_order_identical_across_conditions':True,'selected_review2200accepted_IDs_unchanged':True,
            'prior_validation160_K5_sources_and_caption_IDs_identical':True,'all_businesses_row_order_reproducibility_checked':True,
            'historical_protected_artifacts_unchanged':True,'test_tuning':False,'test_teacher_targets':False,
            'human_test_references_used':False,'teacher_calls':False,'LED_training':False,'artifact_sha256':artifacts,
            'shuffled_assignment_not_yet_constructed_or_empirically_audited':True}
    for split in ['train','validation','test']:
        bids=[bid for bid,s in cohort.items() if s==split];caps=[r for r in records.values() if r['split']==split]
        pp=[pools[bid] for bid in bids]
        item={'businesses':len(bids),'caption_pool_generated':len(caps),'basic_pass':sum(r['basic_eligible'] for r in caps),
              'basic_failures':sum(not r['basic_eligible'] for r in caps),'retried_photos':sum(len(r['attempts'])>1 for r in caps),
              'exact_duplicate_collapses_Filtered':sum(len(p['image_duplicate_removed_to_representative']) for p in pp),
              'relevance_removals_Filtered':sum(p['after_exact_duplicate_count']-p['after_relevance_count'] for p in pp),
              'redundancy_removals_Filtered':sum(len(p['semantic_removed_to_representative']) for p in pp),'conditions':{}}
        for condition in ['unfiltered','filtered']:
            mr=manifests[split,condition];lr=[r for r in ledger if r['split']==split and r['condition']==condition]
            item['conditions'][condition]={'retained_captions':stats([m['retained_caption_count'] for m in mr]),
                'retained_count_histogram':dict(sorted(Counter(m['retained_caption_count'] for m in mr).items())),
                'zero_caption_businesses':[m['business_id'] for m in mr if m['empty_caption_business']],
                'caption_block_tokens':stats([r['caption_block_increment_tokens'] for r in lr]),
                'full_source_tokens':stats([r['full_source_tokens'] for r in lr]),'caption_over750':0,'source_over5000':0}
        report['split_reports'][split]=item
    immutable(OUT/'completion_report.json',report)
    lines=['# Stage F production completion (D-033)','',f'Final K=5; all2300businesses validated, {slow_checks}full-source slow/native token-ID checks. No teacher or training.','',
           '| Split | Photos | Basic pass | Fail | Unfiltered retained | Filtered retained | Unfiltered zero | Filtered zero |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,s in report['split_reports'].items():
        u=s['conditions']['unfiltered'];f=s['conditions']['filtered']
        lines.append(f"| {split} | {s['caption_pool_generated']} | {s['basic_pass']} | {s['basic_failures']} | {u['retained_captions']['total']} | {f['retained_captions']['total']} | {len(u['zero_caption_businesses'])} | {len(f['zero_caption_businesses'])} |")
    lines+=['','All counts/distributions/removals/token statistics and manifestSHA256values: completion_report.json. Full per-business token accounting: manifests/token_ledger.jsonl. Whole reviews/captions retained; no content dropped to fit. Accepted2200review references and all previous validation scientific evidence unchanged. Test inputs processed only with frozen method; no tuning or human-reference use. Shuffled construction remains later.','']
    (OUT/'completion_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(report['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','construct']);a=p.parse_args()
    if a.action=='score':score_production()
    else:construct()
