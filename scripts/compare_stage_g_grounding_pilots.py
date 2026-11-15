"""Read-only before/after inventory and literal-verdict audit; never adjudicates or rewrites targets."""
from collections import Counter
from pathlib import Path
import json
import re

from prepare_stage_g_teacher_pilot import ROOT, read, file_sha

BASE = ROOT / 'outputs/stage_g/teacher_pilot_v3_canonical_review_ids_v1'
NEW = ROOT / 'outputs/stage_g/teacher_pilot_v4_1_grounding_d038_v1'


def normalized(text):
    return ' '.join(text.lower().split())


def collect(folder, partial=False):
    jobs = read(folder / 'pilot_jobs.json')
    businesses = read(folder / 'pilot_businesses.json')
    execution = folder / ('execution_v2_quote_recovery' if folder == NEW else 'execution_v1')
    rows, reviews, visual, missing = [], [], [], []
    for business in businesses:
        bid = business['business_id']
        target_path = execution / 'provisional_targets' / (bid + '.json')
        if not target_path.exists():
            missing.append(bid)
            continue
        target = read(target_path)
        for job in [j for j in jobs if j['business_id'] == bid]:
            cp = read(execution / 'checkpoints' / (job['job_id'] + '.json'))
            if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                raise ValueError('Accepted target depends on a failed task')
            if job['task'] == 'review_consensus':
                for i, claim in enumerate(cp['resolved_output']['claims']):
                    reviews.append({'business_id':bid, 'split':business['split'], 'claim_index':i,
                        'claim':claim['text'], 'polarity':claim['polarity'], 'support':claim['support'],
                        'distinct_support_reviews':len({s['review_id'] for s in claim['support']}),
                        'strong_prevalence_language_candidate':bool(re.search(
                            r'\b(most|consistently|universally|everyone|overwhelmingly|always|all customers)\b',claim['text'],re.I)),
                        'candidate_only_not_an_error_label':True})
            else:
                for caption, image, assessment in zip(job['evidence']['captions'],job['images'],cp['output']['assessments']):
                    for i, claim in enumerate(assessment['claims']):
                        visual.append({'business_id':bid,'split':business['split'],
                            'condition':job['condition'],'photo_id':caption['photo_id'],
                            'photo_sha256':image['raw_sha256'], 'photo_marker':caption['photo_marker'],
                            'caption_id':caption['caption_id'],'caption':caption['caption'],
                            'claim_index':i, 'claim':claim['text'], 'caption_quote':claim['caption_quote'],
                            'verdict':claim['verdict'],'explanation':claim['explanation']})
        rows.append({'business_id':bid,'split':business['split'],'targets':target['targets'],
                     'zero_filtered_edge_case':business['zero_filtered_edge_case']})
    attempted_path=execution/'completion_20_attempted_businesses.json'
    attempted=read(attempted_path) if attempted_path.exists() else None
    if missing and not partial:
        if not attempted or set(missing)!={f['business_id'] for f in attempted['failed_tasks']}:
            raise ValueError(f'Rerun incomplete: {len(rows)}/20 businesses')
    return {'businesses':rows,'review_claims':reviews,'visual_claims':visual,'missing':missing,'attempted_inventory':attempted}


def consistency(rows, normalize_claim=False):
    groups = {}
    for row in rows:
        key = (row['business_id'], row['photo_id'], row['photo_sha256'], row['caption'],
               normalized(row['claim']) if normalize_claim else row['claim'])
        groups.setdefault(key,[]).append(row)
    paired, conflicts = [], []
    for key, items in groups.items():
        if {r['condition'] for r in items} != {'unfiltered','filtered'}:
            continue
        pair = {'business_id':key[0],'photo_id':key[1],'caption':key[3],
                'matched_claim_key':key[4], 'rows':items,
                'conflict':len({r['verdict'] for r in items})>1,
                'resolution':'FLAG_ONLY_NO_RESULT_CHANGED' if len({r['verdict'] for r in items})>1 else 'same_verdict'}
        paired.append(pair)
        if pair['conflict']:
            conflicts.append(pair)
    return {'matching_rule':'same_photo_hash_exact_caption_and_'+('case_whitespace_normalized_literal_claim' if normalize_claim else 'exact_claim_text'),
            'shared_claim_groups':len(paired),'conflict_groups':len(conflicts),
            'conflicts':conflicts,'all_shared_groups':paired,
            'unmatched_paraphrases_not_classified_as_agreement':True,'verdicts_modified':False}


def run(partial=False):
    before, after = collect(BASE), collect(NEW,partial)
    preflight = read(NEW / 'rerun_preflight_and_preservation.json')
    for relative, expected in preflight['original_artifact_sha256'].items():
        if file_sha(ROOT/relative) != expected:
            raise ValueError('Preserved original artifact changed: '+relative)
    original_jobs, new_jobs = read(BASE/'pilot_jobs.json'),read(NEW/'pilot_jobs.json')
    strip = lambda j:{k:v for k,v in j.items() if k!='prompt_sha256'}
    if [strip(j) for j in original_jobs] != [strip(j) for j in new_jobs]:
        raise ValueError('Input cohort/evidence changed in rerun')
    versions = {}
    for label, data in [('original',before),('revised',after)]:
        versions[label] = {'businesses_complete':len(data['businesses']),
            'review_claims':len(data['review_claims']), 'visual_claims':len(data['visual_claims']),
            'visual_verdict_counts':dict(Counter(r['verdict'] for r in data['visual_claims'])),
            'strong_review_prevalence_candidates':sum(r['strong_prevalence_language_candidate'] for r in data['review_claims']),
            'exact_cross_condition':consistency(data['visual_claims']),
            'normalized_literal_cross_condition':consistency(data['visual_claims'],True)}
    baseline_by_bid={b['business_id']:b for b in before['businesses']}
    target_deltas=[{'business_id':b['business_id'],'split':b['split'],
                   'changed_conditions':[c for c in b['targets'] if b['targets'][c] != baseline_by_bid[b['business_id']]['targets'][c]]}
                  for b in after['businesses']]
    inventory_finished=bool(after['attempted_inventory'])
    accepted_ids={b['business_id'] for b in after['businesses']}
    matched_before_visual=[r for r in before['visual_claims'] if r['business_id'] in accepted_ids]
    matched_before_reviews=[r for r in before['review_claims'] if r['business_id'] in accepted_ids]
    report={'status':('COMPLETE_ATTEMPTED_PILOT_COMPARISON_WITH_REJECTED_CASES' if inventory_finished else 'PARTIAL_COMPARISON_NOT_COMPLETION'),
        'existing_baseline_artifacts_unchanged':True, 'same_frozen20_inputs':True,
        'generation_settings_and_schema_unchanged':True, 'versions':versions,
        'missing_revised_businesses':after['missing'],'all20_businesses_attempted':inventory_finished,
        'attempted_task_inventory':after['attempted_inventory'],'changed_target_texts':target_deltas,
        'matched_accepted_business_comparison':{
            'businesses':len(accepted_ids),'business_ids':sorted(accepted_ids),
            'original_visual_claims':len(matched_before_visual),'revised_visual_claims':len(after['visual_claims']),
            'original_visual_verdict_counts':dict(Counter(r['verdict'] for r in matched_before_visual)),
            'revised_visual_verdict_counts':dict(Counter(r['verdict'] for r in after['visual_claims'])),
            'original_review_claims':len(matched_before_reviews),'revised_review_claims':len(after['review_claims']),
            'denominator_warning':'Different claim decomposition; these are not human correctness rates.'},
        'claim_texts_and_verdicts':{'original':before['visual_claims'],'revised':after['visual_claims']},
        'review_claims_and_source_quotes':{'original':before['review_claims'],'revised':after['review_claims']},
        'factual_error_count_not_inferred_from_structural_pass':True,
        'review_frequency_candidates_are_not_error_counts':True,
        'human_verified_businesses':0,'production_protocol_frozen':False,
        'limitations':'Literal matches cover a subset: different decompositions/paraphrases need explicit semantic review. One stochastic generation per prompt cannot isolate prompt effect from sampling variation.'}
    path=NEW/('machine_comparison.json' if inventory_finished else 'partial_machine_comparison.json')
    if path.exists() and inventory_finished and read(path)!=report:
        raise ValueError('Complete comparison exists with different contents')
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'], 'summary':{k:{a:b for a,b in v.items() if a not in ['exact_cross_condition','normalized_literal_cross_condition']} for k,v in versions.items()},
        'conflicts':{k:{'exact':v['exact_cross_condition']['conflict_groups'],
                        'normalized_literal':v['normalized_literal_cross_condition']['conflict_groups']} for k,v in versions.items()}},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    run(parser.parse_args().partial)
