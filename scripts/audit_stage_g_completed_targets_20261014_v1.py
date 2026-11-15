"""Offline completed-target screening; never calls API or changes scientific outputs."""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / 'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1'
OUT = PRODUCTION / 'post_completion_target_audit_v1'
SEED = 20261014


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    completion = read(PRODUCTION / 'completion_report.json')
    assert completion['status'] == 'STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS'
    before = {rel: sha(ROOT / rel) for rel in completion['dataset_sha256']}
    assert before == completion['dataset_sha256']
    records = defaultdict(dict)
    counts = defaultdict(Counter)
    empty_visual = []
    screen_flags = []
    review_support_entries = 0
    quote_checks = 0
    duplicate_visual = 0
    body_counts = defaultdict(list)
    for rel in sorted(before):
        with (ROOT / rel).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                jid = row['business_id']
                condition = row['condition']
                assert condition not in records[jid], 'Duplicate business-condition'
                records[jid][condition] = row
                text = row['target']
                key = row['split'] + '/' + condition
                counts[key]['targets'] += 1
                counts[key]['whole_target_empty'] += not text.strip()
                assert text.startswith('[REVIEW_CONSENSUS]\n')
                assert text.count('[REVIEW_CONSENSUS]') == 1
                review_body = text.split('\n\n[VISUAL_EVIDENCE]\n', 1)[0][len('[REVIEW_CONSENSUS]\n'):]
                counts[key]['review_body_empty'] += not review_body.strip()
                body_counts[condition].append(row['target_tokens'])
                assert not re.search(r'\b(?:REV_\d+(?:_SEG_\d+)?|IMG_CAP_\d+|PHOTO_\d+)\b', text)
                assert '\ufffd' not in text
                assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text)
                if re.search(r'\b(?:as an AI|cannot provide|unable to summarize)\b', text, re.I):
                    screen_flags.append({'business_id': jid, 'condition': condition, 'flag': 'possible_refusal_language', 'automated_screen_only': True})
                if condition == 'reviews_only':
                    assert '[VISUAL_EVIDENCE]' not in text
                    reviews = row['shared_review_evidence']['claims']
                    assert reviews and all(claim['text'].strip() for claim in reviews)
                    counts[key]['review_claims'] += len(reviews)
                    assert review_body == ' '.join(claim['text'] for claim in reviews)
                    for ci, claim in enumerate(reviews):
                        assert len({s['review_id'] for s in claim['support']}) >= 2
                        assert all(s['quote'].strip() for s in claim['support'])
                        review_support_entries += len(claim['support'])
                        if re.search(r'\b(?:everyone|all customers|always|never)\b', claim['text'], re.I):
                            screen_flags.append({'business_id': jid, 'condition': condition, 'claim_index': ci,
                                'flag': 'strong_generalization_language', 'text': claim['text'], 'automated_screen_only': True})
                    continue
                assert text.count('[VISUAL_EVIDENCE]') == 1
                visual_body = text.split('\n\n[VISUAL_EVIDENCE]\n', 1)[1]
                claims = row['visual_claims_with_evidence']
                assert visual_body == ' '.join(c['text'] for c in claims)
                assessments = row['evidence_records']['assessments']
                caption_count = len(row['selected_photo_ids'])
                assert caption_count == len(assessments)
                assert row['selected_photo_ids'] == [a['photo_id'] for a in assessments]
                counts[key]['captions'] += caption_count
                counts[key]['serialized_visual_statements'] += len(claims)
                counts[key]['empty_visual_body'] += not visual_body.strip()
                seen = set()
                for c in claims:
                    normalized = ' '.join(c['text'].lower().split())
                    duplicate_visual += normalized in seen
                    seen.add(normalized)
                    assert c['support']
                    for support in c['support']:
                        a = next(a for a in assessments if (a['caption_id'], a['photo_id']) == (support['caption_id'], support['photo_id']))
                        raw = a['claims'][support['raw_claim_index']]
                        assert raw['verdict'] == support['teacher_verdict'] == 'verified'
                        assert raw['text'] == c['text'] and raw['caption_quote'] == support['caption_quote']
                        quote_checks += 1
                for a in assessments:
                    counts[key].update({'teacher_' + c['verdict']: 1 for c in []})
                    for c in a['claims']:
                        counts[key]['teacher_' + c['verdict']] += 1
                if not visual_body.strip():
                    reason = 'no_input_captions' if caption_count == 0 else 'captions_present_but_no_teacher_verified_claims'
                    counts[key]['empty_visual_' + reason] += 1
                    empty_visual.append({'business_id': jid, 'split': row['split'], 'condition': condition,
                        'selected_caption_count': caption_count, 'reason': reason,
                        'target_review_body_present': bool(review_body.strip())})
    assert len(records) == 2200
    assert duplicate_visual == 0
    unique_reviews = defaultdict(list)
    for jid, conditions in records.items():
        assert set(conditions) == {'reviews_only', 'unfiltered', 'filtered'}
        review = conditions['reviews_only']
        unique_reviews[review['target']].append(jid)
        for condition in ('unfiltered', 'filtered'):
            row = conditions[condition]
            assert row['target'].startswith(review['target'] + '\n\n[VISUAL_EVIDENCE]\n')
            assert row['selected_review_ids'] == review['selected_review_ids']
            assert row['shared_review_evidence'] == review['shared_review_evidence']
            assert row['split'] == review['split']
    conflicts = read(PRODUCTION / 'final_validation/cross_condition_conflicts.json')
    tech = read(PRODUCTION / 'final_validation/technical_flags.json')
    rng = random.Random(SEED)
    sample = []
    for split, n in [('train', 15), ('validation', 5)]:
        pool = sorted(jid for jid, c in records.items() if c['reviews_only']['split'] == split
                      and not c['reviews_only']['imported_D041_without_regeneration'])
        sample += [{'business_id': jid, 'split': split} for jid in sorted(rng.sample(pool, n))]
    sample_cases = []
    for item in sample:
        c = records[item['business_id']]
        sample_cases.append(dict(item, targets={k: v['target'] for k, v in c.items()},
            review_claims_with_exact_evidence=c['reviews_only']['shared_review_evidence']['claims'],
            visual_claims_with_evidence={k: c[k]['visual_claims_with_evidence'] for k in ('unfiltered', 'filtered')},
            visual_verification={k: c[k]['evidence_records'] for k in ('unfiltered', 'filtered')},
            target_artifact=c['reviews_only']['target_artifact']))
    after = {rel: sha(ROOT / rel) for rel in before}
    assert before == after
    report = {
        'status': 'AUTOMATED_TARGET_SCREENING_PASS_NOT_FACTUAL_CERTIFICATE',
        'audited_utc': datetime.now(timezone.utc).isoformat(), 'businesses': len(records), 'targets': 6600,
        'by_split_condition': {k: dict(v) for k, v in sorted(counts.items())},
        'whole_empty_targets': sum(v['whole_target_empty'] for v in counts.values()),
        'empty_review_consensus_targets': sum(v['review_body_empty'] for v in counts.values()),
        'malformed_section_or_alignment_failures': 0, 'replacement_or_control_character_failures': 0,
        'unverified_visual_claims_serialized': 0, 'exact_duplicate_visual_statements_within_target': duplicate_visual,
        'serialized_visual_support_bindings_checked': quote_checks,
        'resolved_review_support_entries_checked': review_support_entries,
        'identical_review_consensus_across_businesses': [ids for ids in unique_reviews.values() if len(ids) > 1],
        'potential_language_screen_flags': len(screen_flags),
        'exact_cross_condition_conflict_groups': len(conflicts),
        'businesses_with_cross_condition_conflicts': len({r['business_id'] for r in conflicts}),
        'auxiliary_review_ID_diagnostic_entries': len(tech),
        'businesses_with_auxiliary_ID_diagnostics': len({r['business_id'] for r in tech}),
        'diagnostics_not_reclassified_as_invalid_canonical_mapping': True,
        'empty_visual_cases_are_not_whole_empty_targets': True,
        'human_factual_labels_created': 0, 'teacher_API_calls': 0, 'target_content_changed': False,
        'held_out_test_inspected': False, 'methodology_changed': False,
        'dataset_sha256_before_equals_after': True, 'dataset_sha256': before,
        'sample_seed': SEED, 'sample_businesses': 20, 'sample_split': {'train': 15, 'validation': 5},
        'sample_rule': 'fixed-seed random sampling excluding the imported D041 cohort; read-only qualitative review',
        'factual_quality_human_certified': False,
    }
    save('report.json', report)
    save('empty_visual_cases.json', empty_visual)
    save('language_screen_flags.json', screen_flags)
    save('sample_manifest.json', {'seed': SEED, 'businesses': sample})
    save('sample_cases.json', sample_cases)
    print(json.dumps({k: v for k, v in report.items() if k != 'dataset_sha256'}, indent=2))


if __name__ == '__main__':
    main()
