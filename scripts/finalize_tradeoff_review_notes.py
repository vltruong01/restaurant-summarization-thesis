"""Record the exact full originals read by the assistant in the tradeoff audit."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_c/review_selection_tradeoff_audit_v1'
scopes=json.loads((OUT/'qualitative_read_scope.json').read_text(encoding='utf-8'))
observations={
    'ctHjyadbDQAtUFfkcAFEHw': 'V3 examples include useful concise hummus/salatim opinions and a generic hype/decor comment. V4 examples add portion/price, drinks/service, and a long restaurant-week menu narrative. That narrative spends many tokens on one visit; added detail does not justify a worse rating distribution or prove source-aspect coverage.',
    'UcJbErL_KZbzKzxddiaIwQ': 'Both versions contain substantial detail. V3-only examples include reservations/lounge advice and a specific dining-room noise complaint. V4-only examples include a long early-opening Bar Gibbon visit, brightness/steak-value complaints and a later critical food/price review. Temporal bins matter because the venue changed from interim bar to larger dining room; these six reviews alone do not establish a whole-selection temporal narrative or missing-aspect frequency.',
    'RoUP3lcY4wEtdASLNIcmSw': 'V3-only examples already describe riverside ambience, food/value and poor service. V4-only examples include an atmospheric long review explicitly saying the author had not eaten dinner, an outdoor-reservation/service complaint, and a detailed positive meal. Longer reviews are not automatically more relevant meal evidence. V3 retains30 within2955 tokens with better rating JSD and identical time JSD, so the10-review v4 result cannot be accepted merely because its reviews are longer.',
    'eOWBvwP1MgJx6PrrWBoUuA': 'Both versions describe vegan buffet and dietary options. V3 examples already cover high price, spice variability, family service and disappointing vegan desserts. V4 examples add individual dishes and vegan-night visit detail, including a drink-service lapse. All three sampled v4-only examples are five-star reviews whereas two sampled v3-only examples are four-star; this purposeful six-review sample illustrates the tradeoff but is not a source-wide sentiment estimate.'
}
for case in scopes:
    case['status']='assistant_read_complete'
    case['read_review_ids']=case['v3_only_ids_to_read']+case['v4_only_ids_to_read']
    case['observations']=observations[case['business_id']]
report={'date':'2026-10-06','read_reviews':sum(len(c['read_review_ids']) for c in scopes),'businesses':len(scopes),
        'sampling':'First3 chronological v3-only and first3 chronological v4-only reviews at the specified budget in four purposive tradeoff cases; all24 full originals read.',
        'limitations':'Assistant illustrative inspection only. Not formal human evaluation, exhaustive source/selection reading, semantic diversity scoring, aspect recall or proof of better summaries. No new quality filter applied.',
        'cases':scopes}
assert report['read_reviews']==24
(OUT/'qualitative_review_notes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Recorded24 full-original reads in4 tradeoff cases.')
