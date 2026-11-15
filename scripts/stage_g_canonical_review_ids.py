"""Canonical source IDs from strict marker/span mappings; raw reported IDs remain diagnostic."""
import copy
from stage_g_evidence_id_protocol import resolve_review as strict_resolve
from prepare_stage_g_review_evidence_ids import evidence_spans


def projection(job, output):
    if output['business_id'] != job['business_id']:
        raise ValueError('Teacher returned another business ID')
    original = {r['review_marker']: r for r in job['evidence']['reviews']}
    derived = copy.deepcopy(output)
    diagnostics = []
    for ci, claim in enumerate(derived['claims']):
        for si, support in enumerate(claim['support']):
            source = original.get(support['review_marker'])
            if source is None:
                raise ValueError('Nonexistent source review marker')
            allowed = {s['evidence_span_id'] for s in evidence_spans(source)}
            ids = support['evidence_span_ids']
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= allowed:
                raise ValueError('Invalid, duplicate, empty or wrong-review source spans')
            if support['review_id'] != source['review_id']:
                diagnostics.append({'claim_index': ci, 'support_index': si,
                    'review_marker': support['review_marker'], 'evidence_span_ids': ids,
                    'teacher_reported_review_id': support['review_id'],
                    'canonical_source_review_id': source['review_id'],
                    'resolution': 'code_derived_from_exact_agreeing_marker_and_span_IDs',
                    'human_verified': False})
            # This is a separate code-derived projection, never an edit of the raw model object.
            support['review_id'] = source['review_id']
    return derived, diagnostics


def resolve_review(job, output):
    derived, diagnostics = projection(job, output)
    resolved, provenance = strict_resolve(job, derived)
    by_support = {(d['claim_index'], d['support_index']): d for d in diagnostics}
    for item in provenance:
        raw = output['claims'][item['claim_index']]['support'][item['support_index']]
        item['teacher_reported_review_id'] = raw['review_id']
        item['canonical_review_id_from_source'] = item['review_id']
        item['redundant_reported_ID_mismatch'] = (item['claim_index'], item['support_index']) in by_support
    if [(c['text'], c['polarity']) for c in resolved['claims']] != [(c['text'], c['polarity']) for c in output['claims']]:
        raise ValueError('Claim text/polarity changed during metadata resolution')
    return resolved, provenance


def citation_diagnostics(job, output):
    return projection(job, output)[1]
