"""Resolve teacher-selected source spans; never rewrite model claims or source text."""
import copy
from prepare_stage_g_teacher_pilot import digest
from prepare_stage_g_review_evidence_ids import evidence_spans


def review_input(job):
    result = {'business_id': job['business_id'], 'reviews': []}
    for review in job['evidence']['reviews']:
        spans = evidence_spans(review)
        if ''.join(s['exact_source_text'] for s in spans) != review['text']:
            raise ValueError('Complete review reconstruction changed')
        result['reviews'].append({
            'review_marker': review['review_marker'], 'review_id': review['review_id'],
            'stars': review['stars'], 'date': review['date'],
            'evidence_spans': [{'evidence_span_id': s['evidence_span_id'],
                                'text': s['exact_source_text']} for s in spans]})
    return result


def resolve_review(job, output):
    if output['business_id'] != job['business_id']:
        raise ValueError('Teacher returned another business ID')
    originals = {r['review_marker']: r for r in job['evidence']['reviews']}
    indexes = {marker: {s['evidence_span_id']: s for s in evidence_spans(review)}
               for marker, review in originals.items()}
    resolved = {'business_id': output['business_id'], 'claims': []}
    provenance = []
    for claim_i, claim in enumerate(output['claims']):
        resolved_claim = {'text': claim['text'], 'polarity': claim['polarity'], 'support': []}
        for support_i, support in enumerate(claim['support']):
            marker = support['review_marker']
            review = originals.get(marker)
            if review is None or support['review_id'] != review['review_id']:
                raise ValueError('Span support refers to another review')
            ids = support['evidence_span_ids']
            if not ids or len(ids) != len(set(ids)):
                raise ValueError('Missing or duplicate evidence span IDs')
            for span_id in ids:
                span = indexes[marker].get(span_id)
                if span is None:
                    raise ValueError('Evidence span absent from the referenced review')
                quote = review['text'][span['start_character']:span['end_character']]
                if quote != span['exact_source_text'] or digest(quote.encode('utf-8')) != span['text_sha256']:
                    raise ValueError('Source span text/offset/hash changed')
                if not quote.strip():
                    raise ValueError('Whitespace-only evidence cannot support a claim')
                resolved_claim['support'].append({'review_marker': marker,
                    'review_id': review['review_id'], 'quote': quote})
                provenance.append({'claim_index': claim_i, 'support_index': support_i,
                    'review_marker': marker, 'review_id': review['review_id'],
                    'evidence_span_id': span_id, 'start_character': span['start_character'],
                    'end_character': span['end_character'], 'quote_sha256': span['text_sha256'],
                    'quote_origin': 'code_copied_from_immutable_original_review'})
        resolved['claims'].append(resolved_claim)
    # A source resolver must never fix or expand a teacher-generated claim.
    if [(c['text'], c['polarity']) for c in resolved['claims']] != [(c['text'], c['polarity']) for c in output['claims']]:
        raise ValueError('Resolver changed claim text or polarity')
    return resolved, provenance
