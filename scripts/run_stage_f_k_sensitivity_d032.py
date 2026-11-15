"""Actual160-business K sensitivity under D-032; rule-implied candidate only, never freeze K."""
from pathlib import Path
import json
import orjson
import numpy as np
from stage_f_verification_workflow import read,write,sha
from stage_f_d032_protocol import ROOT,BASE,STATE,CONFIG,require_validated_amendment,freeze_budget_manifest
from source_budget_audit_template import review_block,base_source,assemble,pinned_tokenizer,encoding,text_sha

OUT=ROOT/'outputs/stage_f/k_sensitivity_d032_v1'
KS=[3,5,8,10]


def stats(values):
    a=np.array(values,dtype=np.int64)
    return {'n':len(a),'min':int(a.min()),'mean':float(a.mean()),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'max':int(a.max()),'total':int(a.sum())}


def group_coverage(pool,ids):
    available=set(pool['available_reference_group_ids'])
    assert len(available)==pool['available_reference_groups']
    if not available:
        assert not ids
        return None,[]
    represented={pool['reference_group_by_photo_id'][pid] for pid in ids}
    assert represented.issubset(available)
    return len(represented)/len(available),sorted(represented)


def rule_candidate(rows):
    best=max(row['macro_coarser_group_coverage'] for row in rows)
    eligible=[row['K'] for row in rows if row['budget_feasible_both_conditions'] and row['macro_coarser_group_coverage']>=.95*best]
    return min(eligible) if eligible else None,best


def run():
    assert not (OUT/'K_report.json').exists(),'Use a new namespace for a new study state'
    require_validated_amendment();budget=freeze_budget_manifest();assert budget['budgets_finalized'] is True
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[CONFIG,STATE/'source_budget_manifest.json',STATE/'frozen_shared_review_reference.json',
            ROOT/'configs/led_source_template_d032.json',BASE/'fixed_filter_pools.json',BASE/'frozen_thresholds.json',
            BASE/'business_roles.json',BASE/'scores_K_study/caption_scores.json',
            ROOT/'data/processed/yelp_2300_english_v1/reviews/validation.jsonl',ROOT/'data/processed/yelp_2300_v1/businesses.jsonl',
            ROOT/'outputs/stage_f/actual_source_budget_freeze_audit_v1/token_ledger.json']
    before={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs};write(OUT/'protected_before_sha256.json',before)
    write(OUT/'study_settings.json',{'K_candidates':KS,'primary_condition':'Filtered',
          'coverage':'unique_frozen_available_coarser_groups_represented_divided_by_same_business_available_groups',
          'macro':'equal_weight_positive_group_businesses_zero_group_coverage_NA_empty_rate_separate',
          'relative_coverage':'macro_coverage_divided_by_best_macro_among_same_four_candidates',
          'selection_rule':'smallest_both_condition_budget_feasible_K_at_least95pct_of_best_macro',
          'selected_set_change_definition':'photo/caption_record_ID_sets; unique_normalized_text_set_changes_reported_separately',
          'budgets':budget['budgets'],'no_per_K_regroup_or_redundancy_refit':True,'no_automatic_K_freeze':True})
    pools=read(BASE/'fixed_filter_pools.json')['businesses'];assert len(pools)==160
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')}
    assert {p['business_id'] for p in pools}=={bid for bid,role in roles.items() if role=='K_study'}
    frozen_refs={r['business_id']:r for r in read(STATE/'frozen_shared_review_reference.json')}
    refs={p['business_id']:frozen_refs[p['business_id']] for p in pools}
    wanted={rid for r in refs.values() for rid in r['selected_review_ids']};reviews={}
    for line in (ROOT/'data/processed/yelp_2300_english_v1/reviews/validation.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['review_id'] in wanted:reviews[r['review_id']]=r
    assert set(reviews)==wanted
    businesses={}
    for line in (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['business_id'] in refs:businesses[r['business_id']]=r
    captions={r['photo_id']:r for r in read(BASE/'scores_K_study/caption_scores.json')}
    assert all(r['split']=='validation' and r['role']=='K_study' for r in captions.values())
    prior_audit={(r['business_id'],r['condition']):r for r in read(ROOT/'outputs/stage_f/actual_source_budget_freeze_audit_v1/token_ledger.json') if r['internal_role']=='K_study'}
    tok,tokenizer_manifest=pinned_tokenizer();ledger=[];cache={};previous={};previous_text={}
    for pool in sorted(pools,key=lambda p:p['business_id']):
        bid=pool['business_id'];ref=refs[bid]
        rr=[reviews[rid] for rid in ref['selected_review_ids']];assert all(r['business_id']==bid for r in rr)
        block=review_block(rr);assert text_sha(block)==ref['review_block_sha256']
        base=base_source(businesses[bid],block);assert text_sha(base)==ref['base_sha256']
        review_tokens=len(encoding(tok,block).ids);base_tokens=len(encoding(tok,base).ids)
        assert review_tokens<=4000
        for k in KS:
            for condition,key in [('filtered','filtered_diversity_ranking'),('unfiltered','unfiltered_fixed_seed_ranking')]:
                ranking=pool[key];ids=ranking[:k];cap=[captions[pid]['normalized_caption'] for pid in ids]
                assert len(ids)==min(k,len(ranking)) and all(captions[pid]['business_id']==bid for pid in ids)
                cache_key=(bid,condition,tuple(ids))
                if cache_key not in cache:
                    source=assemble(base,cap);full=len(encoding(tok,source).ids)
                    cache[cache_key]={'full_source_tokens':full,'caption_block_tokens':full-base_tokens,'source_sha256':text_sha(source)}
                cost=cache[cache_key]
                # Independent equality to every relevant full-prefix accounting from the prior actual audit.
                old=prior_audit[bid,condition]
                expected=old['reviews_only_base_tokens']+old['visual_header_separator_increment_tokens']+sum(old['complete_caption_prefix_increments'][:len(ids)])
                assert cost['full_source_tokens']==expected
                assert ids==old['caption_photo_ids'][:len(ids)]
                coverage,represented=group_coverage(pool,ids) if condition=='filtered' else (None,[])
                pkey=(bid,condition);ids_set=set(ids);text_set=set(cap)
                changed=None if k==KS[0] else ids_set!=previous[pkey]
                text_changed=None if k==KS[0] else text_set!=previous_text[pkey]
                ledger.append({'business_id':bid,'K':k,'condition':condition,'available_captions':len(ranking),
                               'captions_retained':len(ids),'fewer_than_K_available':len(ranking)<k,'empty_business':not ranking,
                               'selected_caption_photo_ids':ids,'selected_normalized_captions':cap,
                               'changed_from_previous_K':changed,'unique_caption_text_set_changed':text_changed,
                               'previous_K':None if k==KS[0] else KS[KS.index(k)-1],
                               'available_coarser_groups':pool['available_reference_groups'],'represented_coarser_group_ids':represented,
                               'coarser_group_coverage':coverage,'review_tokens':review_tokens,'base_source_tokens':base_tokens,
                               'review_block_sha256':text_sha(block),'review_ids':ref['selected_review_ids'],
                               'base_sha256':text_sha(base),**cost,
                               'caption_overflow':cost['caption_block_tokens']>750,'source_overflow':cost['full_source_tokens']>5000})
                previous[pkey]=ids_set;previous_text[pkey]=text_set
    table=[]
    for k in KS:
        primary=[r for r in ledger if r['K']==k and r['condition']=='filtered'];valid=[r['coarser_group_coverage'] for r in primary if r['coarser_group_coverage'] is not None]
        assert len(primary)==160 and len(valid)==159
        item={'K':k,'macro_coarser_group_coverage':float(np.mean(valid)),'coverage_business_denominator':len(valid),
              'coverage_NA_businesses':160-len(valid),'conditions':{}}
        for condition in ['filtered','unfiltered']:
            rows=[r for r in ledger if r['K']==k and r['condition']==condition];assert len(rows)==160
            item['conditions'][condition]={'businesses':160,'actual_captions_retained':stats([r['captions_retained'] for r in rows]),
                'businesses_fewer_than_K_available':sum(r['fewer_than_K_available'] for r in rows),'empty_businesses':sum(r['empty_business'] for r in rows),
                'caption_block_tokens':stats([r['caption_block_tokens'] for r in rows]),'full_source_tokens':stats([r['full_source_tokens'] for r in rows]),
                'caption_blocks_over750':sum(r['caption_overflow'] for r in rows),'full_sources_over5000':sum(r['source_overflow'] for r in rows),
                'changed_caption_record_sets_from_previous_K':None if k==KS[0] else sum(r['changed_from_previous_K'] for r in rows),
                'changed_unique_caption_text_sets_from_previous_K':None if k==KS[0] else sum(r['unique_caption_text_set_changed'] for r in rows)}
        item['budget_feasible_both_conditions']=all(c['caption_blocks_over750']==c['full_sources_over5000']==0 for c in item['conditions'].values())
        table.append(item)
    candidate,best=rule_candidate(table)
    for row in table:
        row['relative_coverage_vs_best']=row['macro_coarser_group_coverage']/best if best else None
        row['meets95pct_best']=row['macro_coarser_group_coverage']>=.95*best
    report={'status':'ACTUAL_K_STUDY_COMPLETE_RULE_IMPLIED_CANDIDATE_REQUIRES_RESEARCHER_REVIEW',
            'decision':'D-032','K_candidates':KS,'K_table':table,'best_macro_coverage':best,
            'coverage_target_95pct_of_best':.95*best,'rule_implied_K':candidate,'K_selected':None,'K_frozen':False,
            'primary_coverage_condition':'Filtered','coverage_definition':read(OUT/'study_settings.json')['coverage'],
            'coverage_macro_denominator':159,'empty_filtered_businesses_retained':1,'zero_group_coverage':'NA_not_zero',
            'Unfiltered_coverage_not_used_for_selection':True,'businesses':160,'construction_rows':len(ledger),
            'same_review_ids_text_order_across_K_and_conditions':True,'pinned_tokenizer':tokenizer_manifest,
            'budgets':budget['budgets'],'no_source_truncation_or_removal':True,'no_perK_group_or_rank_refit':True,
            'caption_filters_or_thresholds_changed':False,'heldout_test_used':False,'teacher_calls':False,
            'production_train_test_caption_generation':False,'LED_training':False,
            'source_budget_manifest_sha256':sha(STATE/'source_budget_manifest.json'),
            'fixed_groups_rankings_sha256':sha(BASE/'fixed_filter_pools.json'),'code_sha256':sha(Path(__file__))}
    write(OUT/'K_report.json',report);write(OUT/'K_business_ledger.json',ledger)
    assert before=={name:sha(ROOT/name) for name in before}
    render(report)
    print(json.dumps({'rule_implied_K':candidate,'best_coverage':best,'K_table':table,'K_frozen':False}))
    return report


def render(report):
    lines=['# Actual validation K sensitivity under D-032','',
           f"**Rule-implied candidate: K={report['rule_implied_K']}. Not selected/frozen; return for researcher review.**",'',
           'Exactly160K-study businesses, frozen D-031groups/rankings and D-0325000/4000/750source contract. Primary macro coverage is Filtered only, averaging159positive-group businesses equally; the one empty business remains in all other metrics and hasNAcoverage. Groups are not reconstructed perK.','',
           '| K | Macro group coverage | Relative to best | Meets95%best | Both-condition budget feasible | Filtered captions mean | Unfiltered captions mean |',
           '|---:|---:|---:|---|---|---:|---:|']
    for r in report['K_table']:
        lines.append(f"| {r['K']} | {r['macro_coarser_group_coverage']:.6%} | {r['relative_coverage_vs_best']:.6%} | {r['meets95pct_best']} | {r['budget_feasible_both_conditions']} | {r['conditions']['filtered']['actual_captions_retained']['mean']:.4f} | {r['conditions']['unfiltered']['actual_captions_retained']['mean']:.4f} |")
    lines+=['','## Retention/availability and actual selected-set changes','',
            '| K | Condition | Retained min / median / p95 / max / total | Available<K | Empty | Changed record sets from previousK | Changed unique-text sets |',
            '|---:|---|---|---:|---:|---:|---:|']
    for r in report['K_table']:
        for condition,s in r['conditions'].items():
            a=s['actual_captions_retained']
            lines.append(f"| {r['K']} | {condition} | {a['min']} / {a['median']:.2f} / {a['p95']:.2f} / {a['max']} / {a['total']} | {s['businesses_fewer_than_K_available']} | {s['empty_businesses']} | {s['changed_caption_record_sets_from_previous_K']} | {s['changed_unique_caption_text_sets_from_previous_K']} |")
    lines+=['','## Exact token budgets','',
            '| K | Condition | Caption min / mean / median / p95 / max | Source min / mean / median / p95 / max | Caption>750 | Source>5000 |',
            '|---:|---|---|---|---:|---:|']
    for r in report['K_table']:
        for condition,s in r['conditions'].items():
            c=s['caption_block_tokens'];f=s['full_source_tokens']
            fmt=lambda x:f"{x['min']} / {x['mean']:.3f} / {x['median']:.2f} / {x['p95']:.2f} / {x['max']}"
            lines.append(f"| {r['K']} | {condition} | {fmt(c)} | {fmt(f)} | {s['caption_blocks_over750']} | {s['full_sources_over5000']} |")
    lines+=['',f"Best macro coverage={report['best_macro_coverage']:.12f}; target=.95*best={report['coverage_target_95pct_of_best']:.12f}. Select the smallest feasible K reaching that unrounded target; displayed percentages are not used to round the decision.",'',
            'All caption sets are untouched nested prefixes of frozen rankings, with whole captions and identical selected review blocks. No captions/reviews removed to fit. Caption tokens are exact full-minus-reviews-only-base increments, including all visual headers/markers/separators and joint BPE effects; sources are fully encoded with once-added BOS/EOS. Empty Filtered retains its visual header. Actual token lengths additionally match the previous audited exact-prefix increments.','',
            'Selected-set changes use photo/caption-record ID sets; unique normalized-text set changes are additionally reported because Unfiltered may repeat identical caption text from different photos. The first candidate has no previousK, so its changes are not applicable. Per-business records/selected IDs/source hashes/coverage are in K_business_ledger.json.','',
            'Relative-to-best is among the same four candidates. This is validation sensitivity, not held-out summary quality or evidence that a larger unused token capacity needs filling. No test data, shuffled assignment, GPTteacher, production train/test captions or LED training. K is not frozen until explicit researcher approval.','']
    (OUT/'K_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':run()
