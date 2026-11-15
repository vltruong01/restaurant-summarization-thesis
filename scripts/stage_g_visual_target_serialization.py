"""Serialize repeated verified statements once, preserving all caption/photo evidence."""


def visual_claims(assessments):
    groups = {}
    for assessment in assessments:
        seen_in_caption = set()
        for i, claim in enumerate(assessment['claims']):
            if claim['verdict'] != 'verified':
                continue
            key = ' '.join(claim['text'].lower().split())
            if not key or key in seen_in_caption:
                raise ValueError('Repeated/empty verified statement within one caption')
            seen_in_caption.add(key)
            if key not in groups:
                groups[key] = {'text': claim['text'], 'support': []}
            groups[key]['support'].append({
                'caption_id': assessment['caption_id'], 'photo_id': assessment['photo_id'],
                'raw_claim_index': i, 'caption_quote': claim['caption_quote'],
                'teacher_verdict': claim['verdict']})
    return list(groups.values())
