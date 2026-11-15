"""Independent CPU-only verification of imported production captions/manifests/sources; no tuning."""
from pathlib import Path
from collections import Counter
import csv,json,time
import numpy as np,orjson,yaml
from safetensors.numpy import load_file
from stage_f_verification_workflow import read,write,sha
from stage_f_d033_protocol import freeze_method,STATE
from source_budget_audit_template import review_block,base_source,assemble,pinned_tokenizer,encoding,text_sha
from run_stage_f_k_sensitivity_d032 import stats
from stage_f_d030_protocol import unfiltered_fixed_seed_order
import construct_stage_f_production_d033 as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_colab_v1'
REVIEW=ROOT/'outputs/stage_f/colab_result_review_20261010'


def rows(path):return [orjson.loads(line) for line in path.open('rb')]


def run():
    start=time.perf_counter();method=freeze_method();report=read(OUT/'completion_report.json')
    assert report['status']=='STAGE_F_PRODUCTION_COMPLETE_VALIDATED_STOP_BEFORE_STAGE_G'
    assert report['K']==5 and report['method_sha256']==sha(STATE/'frozen_production_method.json')
    assert report['budgets']=={'full_source':5000,'review':4000,'caption':750}
    for name,h in report['artifact_sha256'].items():assert sha(OUT/name)==h
    for name,h in read(STATE/'protected_before_sha256.json').items():assert sha(ROOT/name)==h,name
    core.OUT=OUT;core.GEN=OUT/'generation_remaining';records,photo_plan=core.caption_pool()
    assert len(records)==30027
    cohort={r['business_id']:r['split'] for r in csv.DictReader((ROOT/'data/processed/yelp_2300_english_v1/business_split.csv').open(encoding='utf-8'))}
    assert Counter(cohort.values())=={'train':2000,'validation':200,'test':100}
    inventory=rows(OUT/'manifests/complete_caption_pool_inventory.jsonl')
    assert len(inventory)==len({r['photo_id'] for r in inventory})==30027
    for r in inventory:
        current=records[r['photo_id']]
        for key in ['photo_id','business_id','split','raw_caption','normalized_caption','basic_eligible','model_id','resolved_commit','generation_settings','basic_quality_policy_sha256']:
            assert r[key]==current[key]
        assert r['caption_record_sha256']==current['_record_sha256'] and r['caption_record_path']==current['_record_path']
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
    scored={};scoring_checks=0
    for folder in core.scoring_namespaces():
        s=read(folder/'scoring_report.json')
        assert s['overlength']==0 and s['dtype']=='float32' and s['TF32'] is False and s['models']==cfg['models']
        vectors=load_file(str(folder/'embeddings.safetensors'))['SBERT'];assert vectors.dtype==np.float32 and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        for r in read(folder/'caption_scores.json'):
            pid=r['photo_id'];assert pid not in scored;scored[pid]=r;actual=records[pid]
            assert r['caption_record_sha256']==actual['_record_sha256'] and r['raw_image_sha256']==actual['raw_sha256']
            for key in ['business_id','split','raw_caption','normalized_caption','basic_eligible']:assert r[key]==actual[key]
            assert r['score_status']==('scored' if actual['basic_eligible'] else 'basic_failure_not_scored')
            scoring_checks+=1
    assert set(scored)==set(records)
    pools={p['business_id']:p for p in rows(OUT/'manifests/frozen_filter_pools.jsonl')}
    shared={r['business_id']:r for r in rows(OUT/'manifests/shared_review_inputs.jsonl')}
    assert set(pools)==set(shared)==set(cohort)
    identity=read(OUT/'review_inputs/review_preparation_identity.json')
    assert identity['cohort_sha256']==sha(ROOT/'data/processed/yelp_2300_english_v1/business_split.csv')
    assert identity['test_input_review_sha256']==sha(ROOT/'data/processed/yelp_2300_english_v1/reviews/test.jsonl')
    assert identity['accepted_train_validation_sha256']==sha(ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1/selection_reference.jsonl')
    for name,h in identity['config_sha256'].items():assert sha(ROOT/'configs'/name)==h
    for name,h in identity['code_sha256'].items():assert sha(ROOT/'scripts'/name)==h
    stamp=text_sha(json.dumps(identity,sort_keys=True))
    refs=read(OUT/'review_inputs/shared_review_references.json')
    test_refs=[r for r in refs if r['split']=='test'];assert len(test_refs)==100
    for r in test_refs:assert r['production_identity']==stamp and r['method']=='v4_2_length_balanced' and r['review_budget']==4000
    old_pools=read(ROOT/'outputs/stage_f/validation_calibration_v1/fixed_filter_pools.json')['businesses']
    for p in old_pools:
        for key,value in p.items():
            if key not in ['role','K','final_caption_set']:assert pools[p['business_id']][key]==value
    wanted={rid for r in shared.values() for rid in r['selected_review_ids']};original_reviews={}
    for split in ['train','validation','test']:
        for line in (ROOT/f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl').open('rb'):
            r=orjson.loads(line)
            if r['review_id'] in wanted:original_reviews[r['review_id']]=r
    assert set(original_reviews)==wanted
    accepted={r['business_id']:r for r in rows(ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1/selection_reference.jsonl') if r['review_budget']==4000}
    for bid,reference in accepted.items():
        assert shared[bid]['selected_review_ids']==reference['selected_review_ids'] and shared[bid]['review_block_sha256']==reference['review_block_sha256']
    assert len(accepted)==2200
    for bid,r in shared.items():
        rr=[original_reviews[rid] for rid in r['selected_review_ids']]
        assert all(review['business_id']==bid for review in rr)
        block=review_block(rr);assert block==r['review_block'] and text_sha(block)==r['review_block_sha256']
        assert [text_sha(review['text']) for review in rr]==r['review_text_sha256_in_order']
    manifests={};counts_by_split={}
    for split in ['train','validation','test']:
        actual=[r for r in records.values() if r['split']==split];s=report['split_reports'][split]
        assert len(actual)==s['caption_pool_generated'] and sum(r['basic_eligible'] for r in actual)==s['basic_pass']
        assert sum(not r['basic_eligible'] for r in actual)==s['basic_failures'] and sum(len(r['attempts'])>1 for r in actual)==s['retried_photos']
        for condition in ['unfiltered','filtered']:
            m=rows(OUT/f'manifests/{split}/{condition}_captions.jsonl');assert len(m)==s['businesses']
            assert {r['business_id'] for r in m}=={bid for bid,role in cohort.items() if role==split}
            for item in m:
                bid=item['business_id'];pool=pools[bid]
                expected=core.condition_manifest(bid,split,pool,records,condition);assert item==expected
                manifests[bid,condition]=item
            measured=stats([r['retained_caption_count'] for r in m]);assert measured==s['conditions'][condition]['retained_captions']
            assert {str(k):v for k,v in sorted(Counter(r['retained_caption_count'] for r in m).items())}==s['conditions'][condition]['retained_count_histogram']
            assert [r['business_id'] for r in m if r['empty_caption_business']]==s['conditions'][condition]['zero_caption_businesses']
        counts_by_split[split]=s
    by_business={}
    for r in scored.values():by_business.setdefault(r['business_id'],[]).append(r)
    for bid,pool in pools.items():
        basic=[r for r in by_business[bid] if r['basic_eligible']]
        assert pool['unfiltered_fixed_seed_ranking']==[r['photo_id'] for r in unfiltered_fixed_seed_order(basic)]
        assert pool['CLIP_mode']=='soft_only' and pool['pHash_mode']=='diagnostic_only' and not pool['CLIP_hard_rejection'] and not pool['pHash_hard_rejection']
        assert len(pool['image_duplicate_removed_to_representative'])==pool['basic_pool_count']-pool['after_exact_duplicate_count']
        assert len(pool['semantic_removed_to_representative'])==pool['after_relevance_count']-pool['post_redundancy_count']
        assert all(round(scored[pid]['relevance_margin'],6)>=.08 for pid in pool['relevance_eligible_photo_ids'])
    for split,s in counts_by_split.items():
        pp=[pools[bid] for bid,role in cohort.items() if role==split]
        assert sum(len(p['image_duplicate_removed_to_representative']) for p in pp)==s['exact_duplicate_collapses_Filtered']
        assert sum(p['after_exact_duplicate_count']-p['after_relevance_count'] for p in pp)==s['relevance_removals_Filtered']
        assert sum(len(p['semantic_removed_to_representative']) for p in pp)==s['redundancy_removals_Filtered']
    ledger={(r['business_id'],r['condition']):r for r in rows(OUT/'manifests/token_ledger.jsonl')};assert len(ledger)==6900
    previous=[r for r in read(ROOT/'outputs/stage_f/k_sensitivity_d032_v1/K_business_ledger.json') if r['K']==5]
    assert len(previous)==320
    for old in previous:
        current=ledger[old['business_id'],old['condition']]
        assert current['source_sha256']==old['source_sha256'] and current['selected_photo_ids']==old['selected_caption_photo_ids']
    tok,tm=pinned_tokenizer()
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(ROOT/tm['relative_directory'],local_files_only=True);slow.model_max_length=100000000
    checks=0;overflow=0
    for split in ['train','validation','test']:
        for condition in ['reviews_only','unfiltered','filtered']:
            source_rows=rows(OUT/f'manifests/{split}/{condition}_sources.jsonl')
            assert len(source_rows)==report['split_reports'][split]['businesses']
            for r in source_rows:
                bid=r['business_id'];ref=shared[bid];block=ref['review_block'];base=base_source(ref['metadata'],block)
                assert r['selected_review_ids']==ref['selected_review_ids'] and r['review_block_sha256']==ref['review_block_sha256']
                captions=None if condition=='reviews_only' else [c['normalized_caption'] for c in manifests[bid,condition]['captions']]
                expected=assemble(base,captions);assert expected==r['source'] and text_sha(expected)==r['source_sha256']
                native=encoding(tok,expected).ids;assert native==slow.encode(expected,add_special_tokens=True,truncation=False)
                entry=ledger[bid,condition];n=len(native);b=len(encoding(tok,base).ids);review_n=len(encoding(tok,block).ids)
                assert r['tokens']==n==entry['full_source_tokens'] and b==entry['reviews_only_base_tokens'] and review_n==entry['review_block_tokens_with_special']
                assert entry['caption_block_increment_tokens']==(0 if condition=='reviews_only' else n-b)
                assert entry['selected_review_ids']==ref['selected_review_ids'] and entry['source_sha256']==r['source_sha256']
                overflow+=int(n>5000 or review_n>4000 or entry['caption_block_increment_tokens']>750)
                checks+=1
                if checks%500==0:print(f'Independent full-source/native-slowID check {checks}/6900',flush=True)
    assert checks==6900 and overflow==0
    result={'status':'PASS_INDEPENDENT_IMPORTED_STAGE_F_COMPLETE_STOP_BEFORE_STAGE_G','decision':'D-033','K':5,
            'caption_records_verified':len(records),'score_caption_bindings_verified':scoring_checks,'businesses':2300,
            'source_token_ID_checks':checks,'overflows':overflow,'accepted2200_review_IDs_text_order_unchanged':True,
            'whole_captions_and_reviews_preserved':True,'unfiltered_Basic_only_seed_order_verified':True,
            'all_final_manifests_match_frozen_rank_prefixes':True,'protected_history_files_unchanged':3025,
            'completion_report_sha256':sha(OUT/'completion_report.json'),'elapsed_seconds':round(time.perf_counter()-start,3),
            'test_tuning':False,'teacher_or_LED_training':False,'model_inference_run_locally':False,'production_namespace':str(OUT)}
    write(REVIEW/'independent_validation_report.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':run()
