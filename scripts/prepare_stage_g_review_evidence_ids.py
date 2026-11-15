"""Offline proposal: stable source-span IDs avoid model retyping evidence quotes."""
from pathlib import Path
import json
import re
from prepare_stage_g_teacher_pilot import ROOT, read, canonical, digest, immutable

SOURCE = ROOT / 'outputs/stage_g/teacher_pilot_v2/pilot_jobs.json'
OUT = ROOT / 'outputs/stage_g/review_evidence_id_proposal_v1'


def evidence_spans(review):
    text = review['text']
    # Sentence-like boundaries only; not a linguistic sentence classifier.
    # Include all intervening whitespace in the preceding segment so nothing is dropped.
    ends = [match.end() for match in re.finditer(r'(?<=[.!?])\s+', text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    spans = []
    start = 0
    for end in ends:
        if end > start:
            quote = text[start:end]
            spans.append({'evidence_span_id': f"{review['review_marker']}_SEG_{len(spans)+1}",
                          'start_character': start, 'end_character': end,
                          'exact_source_text': quote, 'text_sha256': digest(quote.encode('utf-8'))})
        start = end
    if ''.join(span['exact_source_text'] for span in spans) != text:
        raise ValueError('Evidence segmentation lost/modified original review text')
    return spans


def prepare():
    jobs = read(SOURCE)
    records = []
    for job in jobs:
        if job['task'] != 'review_consensus':
            continue
        if job['split'] not in {'train', 'validation'}:
            raise ValueError('Held-out evidence proposal forbidden')
        reviews = []
        for review in job['evidence']['reviews']:
            reviews.append({'review_marker': review['review_marker'], 'review_id': review['review_id'],
                            'stars': review['stars'], 'date': review['date'],
                            'complete_text_sha256': digest(review['text'].encode('utf-8')),
                            'evidence_spans': evidence_spans(review)})
        records.append({'business_id': job['business_id'], 'split': job['split'],
                        'review_block_sha256': job['review_block_sha256'], 'reviews': reviews})
    immutable(OUT / 'review_evidence_index.json', records)
    summary = {'status': 'OFFLINE_EVIDENCE_ID_PROPOSAL_NOT_ADOPTED_NOT_EXECUTED',
               'businesses': len(records), 'reviews': sum(len(r['reviews']) for r in records),
               'spans': sum(len(v['evidence_spans']) for r in records for v in r['reviews']),
               'complete_review_text_reconstructed_exactly': True,
               'original_review_IDs_order_unchanged': True,
               'selected_review_block_unchanged': True,
               'units': 'Python Unicode character offsets; SHA256 of UTF8 source strings',
               'boundary_rule': r'(?<=[.!?])\s+; preserve whitespace in preceding span',
               'index_sha256': digest(canonical(records)),
               'source_jobs_sha256': digest(canonical(jobs)),
               'API_calls': 0, 'existing_targets_or_labels_modified': False,
               'remaining_limitation': 'Valid span IDs establish source binding, not semantic support or factual correctness.'}
    immutable(OUT / 'proposal_integrity_report.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
