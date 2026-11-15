"""Offline integrity audit of current accepted components; no teacher calls or edits."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import continue_stage_g_guarded_production6_v1 as runtime
from audit_stage_g_d039_focused_pilot import consistency

f = runtime.frozen
ROOT = f.ROOT
REPORT = runtime.RUN / 'spot_audit_v1'
SAMPLE = ['16tjKOvMw_nOgNViq2LwOQ', '1fGaY2WlGv2l4as6UkMFrg',
          '1PYqegX1SruVbkCsRF-4lA', '1Uj_Ob1C2z5Ykx-WHorjWQ', '1ZDFYvYjBqmz4z8u_CKDNg']


def main():
    cfg, jobs, businesses, imports = runtime.checked_inputs()
    state = f.read(runtime.OUT / 'execution_progress.json')
    assert not state['current_jobs'] and state['status'] == 'STOPPED_REQUIRES_REVIEW'
    components = defaultdict(dict); accepted = []; failed = []; response_ids = []; hashes = {}
    all_visual = []; new_visual = []; review_spans = 0; photo_pairs = 0; warnings = []
    newest = {p.stem for p in (runtime.OUT / 'targets').glob('*.json')}
    for j in jobs:
        directory = runtime.location(j, imports)
        path = directory / 'checkpoints' / (j['job_id'] + '.json')
        if not path.exists(): continue
        cp = f.read(path)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failed.append({'job_id': j['job_id'], 'local_validation_error': cp.get('local_validation_error')})
            continue
        cp = runtime.verify_checkpoint(j, cfg, directory, runtime.CONFIG)
        accepted.append(j['job_id']); hashes[path.relative_to(ROOT).as_posix()] = f.file_sha(path)
        components[j['business_id']][j['condition']] = cp.get('resolved_output', cp['output'])
        if cp.get('api_called'):
            raw = f.read(directory / 'raw_responses' / path.name)
            response_ids.append(raw['id'])
        if j['task'] == 'review_consensus':
            review_spans += len(cp['evidence_resolution'])
            warnings += [dict(d, business_id=j['business_id']) for d in cp['citation_metadata_diagnostics']]
        else:
            for cap, photo, assessment in zip(j['evidence']['captions'], j['images'], cp['output']['assessments']):
                photo_pairs += 1
                for ci, claim in enumerate(assessment['claims']):
                    row = dict(claim, business_id=j['business_id'], condition=j['condition'],
                        caption_id=cap['caption_id'], caption=cap['caption'], photo_id=cap['photo_id'],
                        photo_marker=cap['photo_marker'], photo_sha256=photo['raw_sha256'], claim_index=ci)
                    all_visual.append(row)
                    if j['business_id'] in newest: new_visual.append(row)
    assert len(accepted) == state['accepted_tasks'] and len(response_ids) == len(set(response_ids))
    pointer = f.read(ROOT / cfg['source_pointer']); srcroot = ROOT / pointer['production_directory']
    sources = {(split, cond): {r['business_id']: r for r in f.rows(srcroot / f'manifests/{split}/{cond}_sources.jsonl')}
        for split in ('train', 'validation') for cond in ('reviews_only', 'unfiltered', 'filtered')}
    target_index = f.read(runtime.RUN / 'reuse_targets.json')
    complete = []; serialized_verified = 0; zero_filtered = []; source_checks = 0; review_claims = 0
    for b in businesses:
        content = components[b['business_id']]
        if set(content) != {'shared', 'unfiltered', 'filtered'}: continue
        expected = f.build_target(b, content)
        p = ROOT / target_index[b['business_id']] if b['business_id'] in target_index else runtime.OUT / 'targets' / (b['business_id'] + '.json')
        saved = f.read(p); assert saved == expected
        hashes[p.relative_to(ROOT).as_posix()] = f.file_sha(p); complete.append(b['business_id'])
        review_claims += len(saved['evidence_records']['shared']['claims'])
        for condition, text in saved['targets'].items():
            source = sources[b['split'], condition][b['business_id']]
            assert source['selected_review_ids'] == b['selected_review_ids']
            assert source['review_block_sha256'] == b['review_block_sha256']
            source_checks += 1
            if condition != 'reviews_only':
                assert text.startswith(saved['targets']['reviews_only'] + '\n\n[VISUAL_EVIDENCE]\n')
                for c in saved['visual_target_claims_with_all_evidence'][condition]:
                    assert all(s['teacher_verdict'] == 'verified' for s in c['support'])
                    serialized_verified += 1
        if b['zero_filtered_edge_case']:
            assert saved['targets']['filtered'] == saved['targets']['reviews_only'] + '\n\n[VISUAL_EVIDENCE]\n'
            zero_filtered.append(b['business_id'])
    assert len(complete) == state['businesses_completed']
    cross_all = consistency(all_visual); cross_new = consistency(new_visual)
    failed_job = next(j for j in jobs if j['job_id'] == failed[0]['job_id'])
    raw = f.read(runtime.OUT / 'raw_responses' / (failed_job['job_id'] + '.json'))
    output = json.loads(runtime.legacy.response_text(raw))
    allowed = {r['review_marker']: {s['evidence_span_id'] for s in runtime.evidence_spans(r)}
        for r in failed_job['evidence']['reviews']}
    invalid = []
    for ci, claim in enumerate(output['claims']):
        for si, support in enumerate(claim['support']):
            ids = support['evidence_span_ids']; valid = allowed.get(support['review_marker'], set())
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= valid:
                invalid.append({'claim_index': ci, 'support_index': si, 'review_marker': support['review_marker'],
                    'reported_span_ids': ids, 'invalid_span_ids': sorted(set(ids) - valid)})
    report = {'status': 'ACCEPTED_MAPPING_SOURCE_SERIALIZATION_PASS_NOT_FACTUAL_CERTIFICATE',
        'execution_status': state['status'], 'complete_businesses': len(complete), 'condition_aligned_targets': len(complete)*3,
        'accepted_components_verified': len(accepted), 'accepted_API_response_ids_unique': len(response_ids),
        'review_claims': review_claims, 'exact_resolved_review_spans': review_spans,
        'caption_photo_pairs_verified': photo_pairs, 'source_condition_checks': source_checks,
        'shared_review_text_identical': True, 'target_text_identical_to_frozen_API_serializer': True,
        'serialized_verified_visual_statements': serialized_verified,
        'uncertain_rejected_assessments_excluded_by_serializer': True,
        'zero_filtered_businesses_verified': len(zero_filtered), 'zero_filtered_ids': zero_filtered,
        'auxiliary_review_ID_warnings': len(warnings), 'new_complete_businesses_in_this_namespace': len(newest),
        'exact_cross_condition_matched_groups_all': cross_all['matched_groups'],
        'exact_cross_condition_conflict_groups_all': cross_all['conflict_groups'],
        'exact_cross_condition_matched_groups_new': cross_new['matched_groups'],
        'exact_cross_condition_conflict_groups_new': cross_new['conflict_groups'],
        'teacher_verdicts_new': dict(Counter(r['verdict'] for r in new_visual)),
        'failed_attempts_not_accepted': failed, 'failed_review_invalid_support_entries': invalid,
        'qualitative_sample_business_ids': SAMPLE, 'sample_selection': 'sorted_new_complete_targets_indices_0_3_6_9_13',
        'qualitative_photo_rule': 'first_two_filtered_photo-caption_pairs_of_each_sample_business',
        'manual_human_final_labels_created': 0, 'teacher_API_calls': 0, 'teacher_outputs_edited': False,
        'held_out_test_inspected': False, 'thresholds_K_prompts_changed': False,
        'artifact_sha256': hashes, 'audited_utc': f.now()}
    f.immutable(REPORT / 'technical_report.json', report)
    f.immutable(REPORT / 'cross_condition_conflicts.json', {'all_accepted': cross_all['conflicts'], 'new_complete': cross_new['conflicts']})
    f.immutable(REPORT / 'auxiliary_review_ID_warnings.json', warnings)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifact_sha256'}, indent=2), flush=True)


if __name__ == '__main__': main()
