"""Actual Kmax10 caption source audit only; no coverage, K selection or production freeze."""
from pathlib import Path
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from importlib.metadata import version
import json
import orjson
import numpy as np
from safetensors.numpy import load_file
from stage_f_verification_workflow import read,write,sha
from stage_f_d031_protocol import require_validated_amendment,assert_scoring_binding
from apply_stage_f_frozen_filters import filter_business
from source_budget_audit_template import ROOT,TEMPLATE,pinned_tokenizer,review_block,base_source,assemble,counts,text_sha

OUT=ROOT/'outputs/stage_f/actual_source_budget_freeze_audit_v1'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
C_AUDIT=ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1'


def stats(values):
    a=np.array(values,dtype=np.int64)
    return {'n':len(a),'min':int(a.min()),'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'max':int(a.max())}


def run():
    assert not (OUT/'audit_report.json').exists(),'Do not overwrite prior audit evidence'
    OUT.mkdir(parents=True,exist_ok=True);require_validated_amendment()
    tok,token_manifest=pinned_tokenizer()
    paths=['configs/stage_f_validation_protocol.yaml','configs/research_pending.yaml',
           'configs/review_selection_source_length_expanded.yaml','scripts/review_selection_pilot.py',
           'data/processed/yelp_2300_english_v1/reviews/validation.jsonl','data/processed/yelp_2300_v1/businesses.jsonl',
           'outputs/stage_c/led_source_token_allocation_audit_v1/selection_reference.jsonl',
           'outputs/stage_c/led_source_token_allocation_audit_v1/completion_report.json',
           'outputs/stage_c/review_selection_source_length_expanded_v1/acceptance_report.json',
           'outputs/stage_f/validation_calibration_v1/business_roles.json','outputs/stage_f/validation_calibration_v1/validation_photo_plan.json',
           'outputs/stage_f/validation_calibration_v1/frozen_thresholds.json','outputs/stage_f/validation_calibration_v1/fixed_filter_pools.json']
    for role in ['calibration','K_study']:
        paths.extend(f'outputs/stage_f/validation_calibration_v1/scores_{role}/{name}' for name in ['caption_scores.json','embeddings.safetensors','scoring_report.json'])
    protected={name:sha(ROOT/name) for name in paths}
    original_identity=read(C_AUDIT/'scientific_identity.json')
    for name in ['data/processed/yelp_2300_english_v1/reviews/validation.jsonl','data/processed/yelp_2300_v1/businesses.jsonl']:
        assert protected[name]==original_identity['protected_sha256'][name]
    old_completion=read(C_AUDIT/'completion_report.json')
    assert sha(C_AUDIT/'selection_reference.jsonl')==old_completion['artifact_sha256']['selection_reference.jsonl']
    assert read(ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/acceptance_report.json')['status']=='PASS'
    write(OUT/'audit_template.json',TEMPLATE);write(OUT/'protected_before_sha256.json',protected)
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')};assert len(roles)==200
    selected={}
    for line in (C_AUDIT/'selection_reference.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['business_id'] in roles and r['review_budget']==4000:
            assert r['method']=='v4_2_length_balanced' and r['business_id'] not in selected
            selected[r['business_id']]=r
    assert set(selected)==set(roles)
    wanted={rid for r in selected.values() for rid in r['selected_review_ids']}
    reviews={}
    for line in (ROOT/'data/processed/yelp_2300_english_v1/reviews/validation.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['review_id'] in wanted:
            assert r['business_id'] in roles and r['review_id'] not in reviews
            reviews[r['review_id']]=r
    assert set(reviews)==wanted
    business={}
    # Only validation-ID business metadata is retained; no test review/caption file is opened.
    for line in (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['business_id'] in roles:business[r['business_id']]=r
    assert set(business)==set(roles)
    frozen=read(BASE/'frozen_thresholds.json');fixed={r['business_id']:r for r in read(BASE/'fixed_filter_pools.json')['businesses']}
    assert len(fixed)==160 and {bid for bid,role in roles.items() if role=='K_study'}==set(fixed)
    plan={r['photo_id']:r for r in read(BASE/'validation_photo_plan.json')}
    caption_records={};rankings=dict(fixed)
    for role in ['K_study','calibration']:
        folder=BASE/('scores_'+role);scoring=read(folder/'scoring_report.json');assert_scoring_binding(role,scoring)
        records=read(folder/'caption_scores.json');by_business={}
        for r in records:
            assert r['split']=='validation' and roles[r['business_id']]==role
            caption_records[r['photo_id']]=r
            by_business.setdefault(r['business_id'],[]).append(r)
        if role=='calibration':
            vectors=load_file(str(folder/'embeddings.safetensors'))['SBERT']
            for bid,rr in by_business.items():
                result=filter_business(bid,rr,vectors,plan,frozen['thresholds'])
                result['role']='calibration';result['purpose']='budget_construction_only_no_coverage_or_K_comparison'
                rankings[bid]=result
        else:
            for bid,rr in by_business.items():
                assert set(fixed[bid]['unfiltered_fixed_seed_ranking'])=={r['photo_id'] for r in rr if r['basic_eligible']}
    assert len(caption_records)==2504 and set(rankings)==set(roles)
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(str(ROOT/'assets/tokenizers/led-base-16384'),local_files_only=True)
    ledger=[];review_refs=[];slow_checks=0;source_cache={}
    for number,bid in enumerate(sorted(roles),1):
        ref=selected[bid];rr=[reviews[rid] for rid in ref['selected_review_ids']]
        assert all(r['business_id']==bid for r in rr) and rr==sorted(rr,key=lambda r:(r['date'],r['review_id']))
        block=review_block(rr)
        assert text_sha(block)==ref['review_block_sha256']
        assert __import__('hashlib').sha256(orjson.dumps(ref['selected_review_ids'])).hexdigest()==ref['selected_ids_sha256']
        assert len(tok.encode(block,add_special_tokens=True).ids)<=4000
        base=base_source(business[bid],block)
        review_refs.append({'business_id':bid,'internal_role':roles[bid],'review_budget':4000,
                            'selected_review_ids':ref['selected_review_ids'],'selected_ids_sha256':ref['selected_ids_sha256'],
                            'review_block_sha256':text_sha(block),'review_text_sha256_in_order':[text_sha(r['text']) for r in rr],
                            'base_sha256':text_sha(base),'all_controlled_conditions_share_identical_review_block':True,
                            'shuffled_assignment_not_instantiated':True})
        # Independent tokenizer-ID equality for reviews-only base and both real full sources.
        assert tok.encode(base,add_special_tokens=True).ids==slow.encode(base,add_special_tokens=True,truncation=False)
        slow_checks+=1
        for condition,key in [('unfiltered','unfiltered_fixed_seed_ranking'),('filtered','filtered_diversity_ranking')]:
            photo_ids=rankings[bid][key][:10]
            assert len(photo_ids)==min(10,len(rankings[bid][key]))
            cap=[caption_records[pid]['normalized_caption'] for pid in photo_ids]
            assert all(caption_records[pid]['basic_eligible'] and caption_records[pid]['business_id']==bid for pid in photo_ids)
            m,source=counts(tok,business[bid],block,cap)
            assert source.startswith(base) and all(r['text'] in source for r in rr)
            assert tok.encode(source,add_special_tokens=True).ids==slow.encode(source,add_special_tokens=True,truncation=False)
            slow_checks+=1
            ledger.append({'business_id':bid,'business_name':business[bid]['name'],'internal_role':roles[bid],
                           'condition':condition,'maximum_candidate_for_budget_stress_only':10,
                           'selected_review_count':len(rr),'selected_review_ids':ref['selected_review_ids'],
                           'selected_ids_sha256':ref['selected_ids_sha256'],'caption_photo_ids':photo_ids,
                           'normalized_captions':cap,**m})
            source_cache[bid,condition]={'source':source,'base':base,'metadata':{field:business[bid].get(field) for field in TEMPLATE['fields']},
                                         'review_block':block,'caption_photo_ids':photo_ids,'normalized_captions':cap,'accounting':m}
        if number%20==0:print(f'ACTUAL_BUDGET_AUDIT {number}/200; no coverage or K selection',flush=True)
    summary={}
    for scope in ['all_validation_200','K_study_160','calibration_40']:
        summary[scope]={}
        for condition in ['unfiltered','filtered']:
            subset=[r for r in ledger if r['condition']==condition and (scope=='all_validation_200' or r['internal_role']==scope.rsplit('_',1)[0])]
            assert len(subset)=={'all_validation_200':200,'K_study_160':160,'calibration_40':40}[scope]
            summary[scope][condition]={
                'businesses':len(subset),'caption_block_tokens':stats([r['caption_block_increment_tokens'] for r in subset]),
                'complete_source_tokens':stats([r['full_source_tokens'] for r in subset]),
                'reviews_only_base_tokens':stats([r['reviews_only_base_tokens'] for r in subset]),
                'review_block_tokens':stats([r['review_block_tokens_with_special'] for r in subset]),
                'marginal_metadata_marker_overhead':stats([r['marginal_metadata_review_header_overhead'] for r in subset]),
                'actual_caption_count':stats([r['caption_count'] for r in subset]),
                'caption_blocks_over750':sum(r['caption_budget_overflow'] for r in subset),
                'full_sources_over5000':sum(r['source_budget_overflow'] for r in subset),
                'review_blocks_over4000':sum(r['review_budget_overflow'] for r in subset),
                'declared_budget_envelopes_over5000':sum(r['declared_envelope_overflow'] for r in subset)}
    overflows=[r for r in ledger if r['caption_budget_overflow'] or r['source_budget_overflow'] or r['review_budget_overflow']]
    examples=[];worst={}
    for condition in ['unfiltered','filtered']:
        rr=[r for r in ledger if r['condition']==condition];worst[condition]={}
        for label,key in [('largest_caption_block','caption_block_increment_tokens'),('largest_complete_source','full_source_tokens'),('largest_metadata_marker_overhead','marginal_metadata_review_header_overhead')]:
            r=max(rr,key=lambda r:(r[key],r['business_id']));worst[condition][label]=r
            if not any(e['business_id']==r['business_id'] and e['condition']==condition for e in examples):
                examples.append({'business_id':r['business_id'],'condition':condition,'reason':label,**source_cache[r['business_id'],condition]})
    status='PASS_ZERO_OVERFLOW_D032_PROPOSAL_REQUIRES_RESEARCHER_APPROVAL' if not overflows else 'OVERFLOW_STOP_NO_BUDGET_OR_TEXT_CHANGE'
    report={'status':status,'date_local':'2026-10-09','timezone':'Asia/Saigon','validation_businesses':200,
            'scope':{'calibration':40,'K_study':160,'heldout_test':0},'provisional_allocation':{'full_source_tokens':5000,'review_tokens':4000,'caption_tokens':750,'remaining_declared_overhead_capacity':250},
            'template_sha256':sha(OUT/'audit_template.json'),'template_is_production_frozen':False,'tokenizer':token_manifest,
            'tokenizer_runtime_versions':{'tokenizers':version('tokenizers'),'transformers':version('transformers')},
            'independent_slow_LEDTokenizer_full_ID_equality_checks':slow_checks,'summary':summary,'worst_cases':worst,
            'overflows':overflows,'review_ids_text_stars_dates_order_unchanged':True,'selected_reviews_unique':len(reviews),
            'review_selection_reference_sha256':sha(C_AUDIT/'selection_reference.jsonl'),'same_shared_review_base_for_all_condition_contracts':True,
            'conditions_actually_token_audited':['reviews_only','unfiltered','filtered'],'shuffled_assignment_generated':False,
            'shuffled_later_source_checks_required':True,'no_caption_removed_for_fit':True,'no_review_or_caption_truncation':True,
            'K_coverage_computed':False,'K_performance_compared':False,'K_selected':None,'production_budget_frozen':False,
            'active_config_budget_values_modified':False,'heldout_test_used':False,'teacher_calls':False,'LED_training':False,
            'audit_code_sha256':sha(Path(__file__)),'serializer_code_sha256':sha(ROOT/'scripts/source_budget_audit_template.py')}
    write(OUT/'token_ledger.json',ledger);write(OUT/'shared_review_reference.json',review_refs)
    write(OUT/'worst_complete_source_examples.json',examples);write(OUT/'audit_report.json',report)
    assert protected=={name:sha(ROOT/name) for name in paths}
    write(OUT/'integrity_report.json',{'status':'PASS','protected_inputs_byte_unchanged':True,'input_sha256':protected,
                                      'review_ids_text_order_unchanged':True,'caption_text_order_and_up_to10_prefixes_unchanged':True,
                                      'independent_tokenizer_ID_checks':slow_checks,'coverage_K_selection_or_freeze_executed':False})
    render(report)
    print(json.dumps({'status':status,'all_validation_summary':summary['all_validation_200'],'worst_businesses':{c:{k:r['business_id'] for k,r in rows.items()} for c,rows in worst.items()}}))
    return report


def render(report):
    lines=['# Actual-caption source-budget audit —2026-10-09','',f"**{report['status']}**",'',
           'One exact candidate template, all200validation businesses, accepted4000-token review selections unchanged. Kmax10is used solely for token stress; no coverage/performance/K selection. Active production budgets remainTBD.','',
           '| Scope | Condition | Caption min | Mean | Median | p95 | Max | Source min | Mean | p95 | Max | Caption >750 | Source >5000 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope,conditions in report['summary'].items():
        for condition,s in conditions.items():
            c=s['caption_block_tokens'];f=s['complete_source_tokens']
            lines.append(f"| {scope} | {condition} | {c['min']} | {c['mean']:.2f} | {c['median']:.2f} | {c['p95']:.2f} | {c['max']} | {f['min']} | {f['mean']:.2f} | {f['p95']:.2f} | {f['max']} | {s['caption_blocks_over750']} | {s['full_sources_over5000']} |")
    lines+=['','Caption allowance is the exact incremental FULL source length over the same reviews-only base, including visual header, image-caption markers, separators and BPE boundary effects. Complete source is jointly encoded with BOS/EOS once. Components are exact telescoping increments, not approximately added standalone body counts.','',
            '## Worst cases and exact components','',
            '| Condition | Extreme | Business ID | Review block | Marginal metadata/outer-review overhead | Entire caption increment | Full source | Captions |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for condition,cases in report['worst_cases'].items():
        for name,r in cases.items():
            lines.append(f"| {condition} | {name} | {r['business_id']} | {r['review_block_tokens_with_special']} | {r['marginal_metadata_review_header_overhead']} | {r['caption_block_increment_tokens']} | {r['full_source_tokens']} | {r['caption_count']} |")
    lines+=['','Detailed header/marker/prefix increments, standalone metadata/caption body counts, BPE boundary deltas and limited complete-source examples are in token_ledger.json and worst_complete_source_examples.json. Every actual source satisfies full = marked_review + marginal_base_overhead + marginal_caption_increment.','',
            f"Pinned tokenizer: {report['tokenizer']['model_id']} @ {report['tokenizer']['revision']}; all assets SHA-verified. Independent slow tokenizer checks exact full-source token IDs in{report['independent_slow_LEDTokenizer_full_ID_equality_checks']}encodings. Review block hashes/IDs/order match the immutable Stage C references; reviews/captions are complete and unchanged.",'',
            'Reviews-only has no visual section. Caption-conditioned empty pools retain the visual header. The generic serializer shares the identical review base for Reviews-only/Unfiltered/Filtered/future Shuffled. No actual shuffled donor assignments are constructed, so this audit does not claim token validation of an instantiated shuffled control; that later control must obey the same approved budgets and pass explicit checks.','',
            '## Stop for D-032 approval','',
            'If zero overflow holds, propose5000full-source/4000review-block/750entire-caption-increment and this exact template, with zero silent truncation/removal. This validates required validation constructions, not arbitrary unseen metadata/captions or final2300-business manifests. Any later overflow must be recorded and stopped rather than silently adjusted. Budget/template production freeze is pending researcher approval; no K sensitivity can run yet.','']
    (OUT/'audit_report.md').write_text('\n'.join(lines),encoding='utf-8')
    proposal=['# D-032 PROPOSAL — not approved/frozen','',f"Audit status: {report['status']}",'',
              'Requested candidate: full_source_tokens=5000; review_tokens=4000; caption_tokens=750 as the exact complete-source marginal caption block. Metadata/outer markers fit in the remaining declared250-token capacity; retain exact template in audit_template.json. All controlled conditions share the same selected review allocation; overflow policy is zero silent truncation/removal.','',
              'Production adoption requires explicit researcher approval. Do not change active config or frozen manifest now. Preserve D-030/D-031 and K={3,5,8,10}; after approval only then run actual validation-only K study and return the rule-implied candidate without automatic K freeze.','',
              'Shared review evidence: accepted v4.2 length-balanced4000scenario; this audit pins its200validation review references and texts. It does not create/freeze a final2300-business manifest. Shuffled control donor assignments are not generated here and must later satisfy the same budget/count rules.','',
              f"Evidence: audit_report.json SHA {sha(OUT/'audit_report.json')}; audit_template.json SHA {sha(OUT/'audit_template.json')}; shared_review_reference.json SHA {sha(OUT/'shared_review_reference.json')}."]
    if report['overflows']:proposal+=['','BLOCKED: overflow evidence exists. No budget/template adoption is proposed until the researcher reviews it.']
    (OUT/'D032_PROPOSAL.md').write_text('\n'.join(proposal)+'\n',encoding='utf-8')


if __name__=='__main__':run()
