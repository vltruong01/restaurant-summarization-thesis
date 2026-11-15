"""Metadata-only wire alias projection. Never change generated claims or evidence."""
from copy import deepcopy


def project(job, original):
    projected = deepcopy(original)
    changes = []
    if job['task'] != 'review_consensus':
        return projected, changes
    for ci, claim in enumerate(projected.get('claims', [])):
        for si, support in enumerate(claim.get('support', [])):
            if 'original_review_id' not in support:
                continue
            if 'review_id' in support:
                raise ValueError('Ambiguous review ID fields; do not choose or discard an API value')
            support['review_id'] = support.pop('original_review_id')
            changes.append({'claim_index': ci, 'support_index': si,
                            'from_field': 'original_review_id', 'to_field': 'review_id',
                            'value_changed': False})
    return projected, changes


def self_check():
    job = {'task': 'review_consensus'}
    fixture = {'claims': [{'text': 'format fixture', 'polarity': 'mixed',
                         'support': [{'original_review_id': 'fixture',
                                      'review_marker': 'REV_1', 'evidence_span_ids': ['REV_1_SEG_1']}]}]}
    result, changes = project(job, fixture)
    assert fixture['claims'][0]['support'][0].get('original_review_id') == 'fixture'
    assert result['claims'][0]['text'] == fixture['claims'][0]['text']
    assert len(changes) == 1 and result['claims'][0]['support'][0]['review_id'] == 'fixture'
    assert project(job, result) == (result, [])
    fixture['claims'][0]['support'][0]['review_id'] = 'conflicting'
    try:
        project(job, fixture)
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting ID fields must reject')
    assert project({'task': 'visual_evidence'}, fixture) == (fixture, [])
    return {'alias_unchanged_values': True, 'raw_unchanged': True,
            'ambiguous_fields_rejected': True, 'claim_text_not_edited': True}
