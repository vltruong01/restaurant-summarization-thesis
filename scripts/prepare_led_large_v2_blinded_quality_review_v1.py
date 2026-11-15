"""One offline export of saved outputs. Standard library only; never loads a model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path
import random
import secrets
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2_execution_v1'
SOURCE = EXEC / 'CHATGPT_REVIEW/actual_generated_quality_review'
DEST = ROOT / 'outputs/stage_i/led_large_v2_blinded_quality_review_v1'
MODELS = {
    'D059_base_epoch3': 'benchmarks/D059_Expanded_epoch3.json',
    'large_v2_epoch3': 'predictions/epoch3.json',
    'large_v2_epoch5': 'predictions/epoch5.json',
}
DIMENSIONS = ['core_coverage', 'opinion_balance', 'factual_grounding',
              'caption_supported_visuals', 'semantic_nonredundancy',
              'conciseness', 'completeness', 'overall_usability']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def esc(value):
    return html.escape(str(value), quote=True)


# Qualitative assistant observations, written outside the blinded package.
# Not unit-level scoring, researcher labels, or an independently blinded assessment.
NOTES = [
    ('JOEY Mayfield',
     'Large3 names steak/ravioli/calamari and atmosphere; base3 is more generic. Large5 loses the dish specificity.',
     'Both large epochs attribute food preparation to the women from a different caption. All omit important severe-service/food specifics.',
     ['REVIEW_CANDIDATE_2_U04', 'REVIEW_CANDIDATE_2_U05', 'REVIEW_CANDIDATE_4_U03'],
     {'large_v2_epoch3': ['The women are preparing food.'], 'large_v2_epoch5': ['The women are preparing food.']}),
    ("Walk-On's Sports Bistreaux - New Orleans",
     'Both large epochs restore the sports-bar identity. Base3 retains cold/soggy food criticism that the large summaries omit.',
     'Large5 drops the overpriced side of the value contrast. Both large outputs still say delicious, delicious.',
     ['REVIEW_CANDIDATE_2_U01', 'REVIEW_CANDIDATE_2_U06', 'REVIEW_CANDIDATE_3_U05'], {}),
    ("Andy's Chicken",
     'Both large summaries add seating and sauce detail; base3 repeats the fried-chicken praise across sentences.',
     'Large3 retains sauce criticism better; large5 names sauces but loses the sweetness/spice criticism. Takeout identity remains weak.',
     ['REVIEW_CANDIDATE_2_U03', 'REVIEW_CANDIDATE_2_U04', 'REVIEW_CANDIDATE_4_U01'], {}),
    ('Peel Wood Fired Pizza',
     'Large summaries avoid the base maple-bourbon-pork-pizza loop. Large5 adds noise and value contrasts and a fuller visual account.',
     'Both large summaries omit specific pizza faults and the approved happy-hour/manager dispute Core units.',
     ['REVIEW_CANDIDATE_2_U03', 'OMISSION_2_U01', 'OMISSION_2_U02'], {}),
    ('Starbucks',
     'Both large summaries state the long-line problem directly and retain more caption details than base3.',
     'Both omit incorrect orders, misdirected paid mobile orders and payment recovery. Large5 adds an unclear sentence about space.',
     ['REVIEW_CANDIDATE_4_U01', 'REVIEW_CANDIDATE_5_U02', 'REVIEW_CANDIDATE_5_U03'], {}),
    ('Juicy Seafood Indy',
     'Both large summaries recover the bag of seafood/corn/potatoes. Base3 retains negative food experiences more explicitly.',
     'Large3/5 omit oily sauce, overcooked crab and shrimp-quality concerns. Large5 drops the negative value contrast.',
     ['REVIEW_CANDIDATE_2_U02', 'REVIEW_CANDIDATE_2_U04', 'REVIEW_CANDIDATE_2_U05'], {}),
    ("Thai O'Cha",
     'Large summaries name dishes and portions; large5 states reasonable prices more clearly than large3.',
     'Both miss distinctive Thai-hot spice feedback and food negatives. Both add plates of food where captions describe white plates/napkins.',
     ['REVIEW_CANDIDATE_4_U01', 'REVIEW_CANDIDATE_4_U03', 'REVIEW_CANDIDATE_5_U04'], {}),
    ('State & Main Kitchen & Bar',
     'Both large summaries name actual dishes and cover food/service/value contrasts. Large5 preserves more caption objects and relationships.',
     'Large3 repeats welcoming and paraphrases drinks twice. Generic adjectives still displace some evidence-specific details.',
     ['REVIEW_CANDIDATE_4_U01'], {}),
    ('Aquarium Restaurant',
     'Large3 explicitly restores the aquarium-centered identity and the visual fish tank. Base3 also retains a fish tank visually.',
     'Large5 drops the fish tank from VISUAL_EVIDENCE despite its approved visual Core status, and weakens the value contrast.',
     [], {'large_v2_epoch5': ['The photo shows a restaurant interior.']}),
    ('BOX - Bongo & Bakery on 10th',
     'Large summaries name Waverly Biscuit/avocado toast unlike base3. Large5 restores slow-service complaints; large3 retains coffee praise.',
     'The bowl topped with bowls of food is an unsupported relationship. Both lose negative atmosphere and specific food complaints.',
     ['REVIEW_CANDIDATE_1_U05', 'REVIEW_CANDIDATE_1_U06', 'REVIEW_CANDIDATE_5_U04'], {}),
    ("Prairie Fish 'N' Chips",
     'Large3 retains chowder and value; large5 has a cleaner visual account of wine and fish/chips.',
     'Large5 loses chowder. Both large summaries omit distant/unwelcoming service; base3 preserves it. Large3 changes beer bottle to wine.',
     ['REVIEW_CANDIDATE_3_U01', 'REVIEW_CANDIDATE_3_U04', 'REVIEW_CANDIDATE_4_U03'], {}),
    ('Novella Wine Bar',
     'Large3 retains the wine knowledge, tasting-service problem and value contrast. Both large visual accounts cover more captions than base3.',
     'Large5 is shorter but drops the tasting wait/poor service and pricey side. Large3 repeats stuffed dates; base3 repeats entire visual sentences.',
     ['REVIEW_CANDIDATE_3_U03', 'REVIEW_CANDIDATE_3_U04', 'REVIEW_CANDIDATE_5_U02'], {}),
    ('The Mainstay',
     'Large3 includes chicken and waffles and sports/entertainment identity; large5 retains the sports setting but becomes more generic.',
     'Both omit cold food, limited wine choices and long waits. Large3 assigns a wooden-table relation to people; large5 also joins caption scenes.',
     ['REVIEW_CANDIDATE_1_U04', 'REVIEW_CANDIDATE_6_U05', 'REVIEW_CANDIDATE_7_U06'], {}),
    ('Fiorella',
     'Both large summaries retain service complaints unlike the positive/repetitive base3. Visual object assignment improves over base3.',
     'Both omit disruptive music and specific gnocchi praise. Large5 loses overpriced wording and depicts music positively.',
     ['REVIEW_CANDIDATE_2_U01', 'REVIEW_CANDIDATE_4_U04', 'REVIEW_CANDIDATE_5_U02'], {}),
    ("Paco's Taqueria",
     'Large5 avoids large3 repeated-beans sentence and base3 mural-on-foil relation; it restores tough-meat criticism.',
     'Large5 loses the negative price side and location/cleanliness concerns. Large3 invents a taco relationship from a general meal caption.',
     ['REVIEW_CANDIDATE_3_U03', 'REVIEW_CANDIDATE_5_U04', 'REVIEW_CANDIDATE_5_U05'], {}),
    ('Bonefish Grill',
     'Large3 includes Bang Bang shrimp and a fuller visual account. Both large summaries add value contrast missing from base3.',
     'Large5 drops Bang Bang shrimp and links food to a wooden bar instead of the bottles in the caption. Dish specificity favors large3.',
     ['REVIEW_CANDIDATE_2_U01', 'REVIEW_CANDIDATE_2_U02'], {}),
    ('Honky Tonk Central',
     'Large3 retains dance/music genres/dishes; large5 is more concise. Both large visual accounts better preserve the performance than base3.',
     'All omit disputed billing/security incidents; both large summaries omit music disappointment and crowding. No epoch resolves these Core gaps.',
     ['REVIEW_CANDIDATE_1_U04', 'REVIEW_CANDIDATE_8_U01', 'REVIEW_CANDIDATE_8_U02', 'REVIEW_CANDIDATE_9_U02'], {}),
    ('Hikari Sushi',
     'Large3 retains order/wait problems supported by approved omission units; large5 retains chef attention and value contrast. Both visual accounts are mostly literal.',
     'All omit BYOB. Large5 drops pickup/order problems; cozy, cozy remains in both large summaries.',
     ['REVIEW_CANDIDATE_4_U01', 'OMISSION_3_U01', 'OMISSION_3_U02'], {}),
    ('Mission BBQ',
     'Both large summaries restore patriotic identity and more visual content; large3 retains salty/dry meat criticism.',
     'Large5 adds side dishes but removes meat-quality negatives. Repeated reasonable and generic service language remain.',
     ['REVIEW_CANDIDATE_1_U02', 'REVIEW_CANDIDATE_2_U04', 'REVIEW_CANDIDATE_4_U02'], {}),
    ('Scarlett Begonia',
     'Large summaries retain food-quality contrast and patio/setting information; base3 is more repetitive in the review section.',
     'Both large visual sections add egg/lemon-syrup details absent from all five captions. Base3 avoids these additions and keeps caption-based objects.',
     ['REVIEW_CANDIDATE_2_U05'],
     {'large_v2_epoch3': ['The bowl is topped with lemon syrup.'], 'large_v2_epoch5': ['The bowl is topped with lemon syrup.']}),
    ('Brooklyn Water Bagel',
     'All three preserve some bagel/service/value contrasts. No model solves the zero-caption requirement.',
     'Both large outputs invent a visual sandwich despite zero captions. All omit coffee contrasts and detailed order failures. Coffee ice cubes is Supporting, not Core.',
     ['REVIEW_CANDIDATE_2_U01', 'REVIEW_CANDIDATE_2_U03', 'REVIEW_CANDIDATE_3_U04', 'REVIEW_CANDIDATE_3_U06'],
     {'large_v2_epoch3': ['The photo shows a bagel sandwich.'], 'large_v2_epoch5': ['The photo shows a bagel sandwich.']}),
    ('Pool Bar & Grill - Seminole Hard Rock Tampa',
     'Large5 restores fish-taco criticism; large3/5 avoid the severe base3 cocktail phrase loop.',
     'All omit the approved visual DJ Core unit. Large visual sections remain verbose and combine spatial relations across captions.',
     ['REVIEW_CANDIDATE_6_U01', 'REVIEW_CANDIDATE_6_U02'], {}),
    ('KFC',
     'Both large summaries improve service balance over the overly positive base3 service wording. Large5 gives a coherent caption summary.',
     'Both omit repeated missing/wrong-order Core units, tiny/old chicken and value concern. Both retain the word uneditable instead of inedible.',
     ['REVIEW_CANDIDATE_2_U03', 'REVIEW_CANDIDATE_2_U04', 'REVIEW_CANDIDATE_3_U01', 'REVIEW_CANDIDATE_3_U02'], {}),
    ("Dunkin'",
     'Large3 retains the coffee complaint; large5 cleanly separates coffee/donut, iced-tea/lemon and breakfast-caption objects.',
     'Large5 loses coffee criticism and repeats average/bread-like bagels. All omit approved hygiene and missing-order Core units.',
     ['REVIEW_CANDIDATE_2_U01', 'REVIEW_CANDIDATE_2_U02', 'REVIEW_CANDIDATE_5_U01', 'REVIEW_CANDIDATE_5_U02', 'REVIEW_CANDIDATE_5_U03'], {}),
]


def main():
    assert not DEST.exists(), 'Versioned export already exists; never overwrite a review.'
    cases = read(SOURCE / 'SOURCES_AND_TEACHER_TARGETS.json')
    refs = read(SOURCE / 'APPROVED_REFERENCES.json')
    rows = {m: read(SOURCE / path) for m, path in MODELS.items()}
    ids = [c['business_id'] for c in cases]
    assert len(ids) == len(set(ids)) == 24
    assert Counter(c['subset'] for c in cases) == {'random': 20, 'targeted': 4}
    assert refs == read(ROOT / 'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json')
    assert refs['researcher_confirmed_complete'] is True
    assert all([r['business_id'] for r in records] == ids for records in rows.values())
    eligible = [u for u in refs['units'] if u.get('scoring_eligible') and
                u['approved_reference_role'] in ('core', 'supporting') and
                u['support_status'] == 'supported' and u['human_verified']]
    assert len(eligible) == len(refs['units']) == 576
    assert Counter(u['approved_reference_role'] for u in eligible) == {'core': 261, 'supporting': 315}
    assert len(NOTES) == len(cases)
    input_paths = [SOURCE / p for p in MODELS.values()] + [
        SOURCE / 'SOURCES_AND_TEACHER_TARGETS.json', SOURCE / 'APPROVED_REFERENCES.json',
        EXEC / 'EXECUTION_AUDIT.json', EXEC / 'INSTANCE_RETIREMENT_RESULT.json']
    originals = {p.relative_to(ROOT).as_posix(): digest(p) for p in input_paths}
    seed = secrets.randbits(128)
    rng = random.Random(seed)
    permutations = list(itertools.permutations(MODELS)) * 4
    rng.shuffle(permutations)
    public = DEST / 'BLINDED_REVIEW'
    admin = DEST / 'ADMIN_KEEP_SEPARATE'
    public.mkdir(parents=True)
    admin.mkdir()
    blind_cases, key, templates, observations, links = [], [], [], [], []
    for i, case in enumerate(cases):
        bid, cid = case['business_id'], f'C{i+1:02d}'
        assert text_hash(case['source']) == case['source_sha256']
        assert NOTES[i][0] == case['source'].split('name=')[1].splitlines()[0]
        units = [u for u in eligible if u['business_id'] == bid]
        assert units
        candidates = []
        for slot, model in zip('ABC', permutations[i]):
            row = rows[model][i]
            assert row['source_sha256'] == case['source_sha256'] and row['subset'] == case['subset']
            candidate = {'candidate': slot, 'text': row['text']}
            candidates.append(candidate)
            key.append({'case_id': cid, 'business_id': bid, 'candidate': slot, 'model': model,
                        'original_file': MODELS[model], 'original_row': i,
                        'exact_text_sha256': text_hash(row['text']),
                        'raw_token_ids_sha256': text_hash(json.dumps(row['token_ids'], separators=(',', ':')))})
            templates.append({'case_id': cid, 'business_id': bid, 'subset': case['subset'],
                'candidate': slot, 'reviewer_id': None, 'reviewer_role': None,
                'researcher_confirmed': False, 'status': 'UNREVIEWED',
                'ratings_0_to_4_higher_is_better': dict.fromkeys(DIMENSIONS),
                'zero_caption_visual_assertion': None, 'unsupported_claims': [],
                'reference_coverage': [{'unit_id': u['unit_id'], 'role': u['approved_reference_role'],
                     'coverage': None, 'output_quote': None, 'notes': None} for u in units],
                'important_omissions': [], 'semantic_repetition_examples': [],
                'strengths': None, 'limitations': None})
        blind_cases.append({'case_id': cid, 'business_id': bid, 'subset': case['subset'],
            'business_name': NOTES[i][0], 'source': case['source'],
            'caption_count': len(case['selected_photo_ids']),
            'photo_ids': case['selected_photo_ids'], 'approved_units': units, 'candidates': candidates})
        # Resolve every evidence-linked observation to the unchanged approved D053 unit.
        selected = []
        for suffix in NOTES[i][3]:
            matches = [u for u in units if u['unit_id'].endswith(suffix)]
            assert len(matches) == 1, (bid, suffix)
            selected.append(matches[0])
        if i in (8, 21):
            selected += [u for u in units if u['domain'] == 'visual' and u['approved_reference_role'] == 'core']
        for model, quotes in NOTES[i][4].items():
            assert all(q in rows[model][i]['text'] for q in quotes)
        observations.append({'case_id': cid, 'business_id': bid, 'subset': case['subset'],
            'business_name': NOTES[i][0], 'assessment_origin': 'MODEL_ASSISTED_IDENTITY_AWARE',
            'researcher_confirmed': False, 'exhaustive_unit_scoring_performed': False,
            'comparative_strengths': NOTES[i][1], 'limitations_and_tradeoffs': NOTES[i][2],
            'priority_reference_units_to_check': selected, 'literal_output_examples': NOTES[i][4]})
        links.append({'case_id': cid, 'business_id': bid, 'source_sha256': case['source_sha256'],
                      'references': len(units), 'core': sum(u['approved_reference_role'] == 'core' for u in units)})
        for pid in case['selected_photo_ids']:
            src, dst = SOURCE / 'images' / f'{pid}.jpg', public / 'images' / f'{pid}.jpg'
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            assert digest(src) == digest(dst)
    write(public / 'CASES.json', blind_cases)
    write(public / 'BLANK_RESEARCHER_ANNOTATIONS.json', {'status': 'UNREVIEWED',
          'ratings': templates, 'paired_preferences': [dict(case_id=c['case_id'],
            subset=c['subset'], comparisons=[dict(pair=list(pair), preference=None,
              rationale=None, evidence_unit_ids=[]) for pair in itertools.combinations('ABC', 2)],
            overall_choice=None, rationale=None) for c in blind_cases]})
    # Full source-approved reference object is preserved, including its actual approval history.
    shutil.copyfile(SOURCE / 'APPROVED_REFERENCES.json', public / 'D053_APPROVED_REFERENCES.json')
    write(admin / 'BLINDING_KEY_DO_NOT_SEND_TO_REVIEWER.json', {'private_seed': str(seed),
         'scope': 'Per-case A/B/C permutation; identities change between businesses.', 'assignments': key})
    write(admin / 'PROVISIONAL_MODEL_ASSISTED_OBSERVATIONS.json', observations)
    write(admin / 'INPUT_AND_EVIDENCE_BINDINGS.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'original_files_sha256': originals, 'cases': links, 'model_operations_performed': 0,
          'original_artifacts_overwritten': 0})
    protocol = '''# Blinded paired review — 24 development businesses

Open REVIEW.html. Review C01–C20 (random) and C21–C24 (targeted) separately.
Compare A/B, A/C and B/C within each business. A/B/C identities change between
businesses; do not pool by letter. Return completed BLANK_RESEARCHER_ANNOTATIONS.json.
Keep all output wording unchanged. Do not infer identity from writing style.

Evidence: complete frozen selected reviews/captions, original selected photos and
576 researcher-approved D053 units (261 Core, 315 Supporting). D053 references
are approved evidence anchors, not exhaustive truth. Review the complete source
for supported information beyond them. Excluded units never enter a denominator.
Teacher text is deliberately omitted as a rating target: agreement is not truth.

For each candidate assess Core coverage, positive/negative opinion synthesis,
factual grounding, caption-supported visual usefulness, semantic nonredundancy,
conciseness, completeness and usability. Rating anchors: 0 unusable/severe failure;
1 major shortcomings; 2 mixed/usable only with substantial correction;
3 mostly useful with limited corrections; 4 strong, accurate, useful synthesis.
These are ordinal guidance, not a validated scale or a weighted automatic GO.

Mark each reference full/partial/missing/contradicted/uncertain, with the exact
output phrase and source unit/evidence marker. Generic food praise is not full
coverage of a named dish or a specific failure. For unsupported claims quote the
claim, cite relevant REV/IMG_CAP/photo IDs, and explain the unsupported relation.
Do not count repeated adjectives, paraphrases or sentence splitting as extra
information. Legitimate names (e.g. Bang Bang shrimp) are not harmful repetition.

When caption_count=0, VISUAL_EVIDENCE must contain no visual assertions. Any such
assertion is unsupported even if plausible from restaurant type or review text.
When captions exist, do not merge subjects/objects from different photos into a
new relationship. Photo-supported but caption-absent details should be recorded
separately; do not credit them as caption-conditioned information recovery.

Pairwise preference: A/B/C as applicable, tie, or inconclusive. Explain tradeoffs
and cite evidence; do not force a winner. Review quality before any unblinding.
All annotation fields begin null/blank and researcher_confirmed remains false.
Model-assisted labels must be identified and remain provisional until explicit
researcher confirmation. No researcher-confirmed labels have been invented.

This is a repeatedly exposed development set, not fresh held-out evaluation.
The candidates come from differing backbone/protocol histories: practical
comparison, not an isolated causal architecture or label-policy experiment.
Previous reviewers may recognize outputs: identity masking cannot undo exposure.
Use a reviewer who has not read the identity-aware recommendation where feasible.
No significance/generalization claim should be made from these 24 reused cases.

GO requires substantive information retention, balanced synthesis, grounded
visuals, zero-caption compliance and usable summaries with meaningful reduction
in harmful repetition. Loss, lengths, headers or exact-repeat counts alone are
insufficient. Do not authorize training from this worksheet. Stop for researcher
decision; any later matched-family preparation/execution requires separate scope.
'''
    (public / 'README_FIRST.md').write_text(protocol, encoding='utf-8')
    (public / 'REVIEW_REQUEST.md').write_text('Independently review the three anonymized candidates per case using README_FIRST.md. '
        'Return evidence-linked findings and pairwise preferences separately for random20 and targeted4. '
        'Fill the blank annotations provisionally; do not claim researcher confirmation. Do not guess identities, '
        'select from loss, or run training/inference. All original wording and source references are preserved.\n', encoding='utf-8')
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Blinded paired quality review</title><style>body{font:16px system-ui;line-height:1.5;max-width:1600px;margin:24px auto;padding:0 16px}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}article{border-top:3px solid #777;padding:20px 0}'
             '.candidates{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.candidate{background:#f3f5f7;padding:16px}'
             'img{max-width:210px;max-height:170px}figure{display:inline-block;margin:8px}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}'
             'table{border-collapse:collapse;width:100%}details{margin:12px 0}@media(max-width:900px){.candidates{grid-template-columns:1fr}}</style>'
             '<h1>Blinded paired quality review</h1><p>24 reused development cases; three candidates per business. '
             '<b>A/B/C assignments change across businesses.</b> No researcher output labels assigned. '
             '<a href="README_FIRST.md">Rubric and limitations</a> · <a href="BLANK_RESEARCHER_ANNOTATIONS.json">Blank annotation worksheet</a></p>']
    for c in blind_cases:
        cid = c['case_id']
        parts.append(f'<article id="{cid}"><h2>{cid} · {esc(c["subset"])} · {esc(c["business_name"])}</h2>'
                     f'<p>Business ID: {esc(c["business_id"])}. Filtered captions: <b>{c["caption_count"]}</b>.</p>')
        parts.append('<details><summary>Complete source reviews and captions</summary><pre>' + esc(c['source']) + '</pre></details>')
        for pid in c['photo_ids']:
            parts.append(f'<figure id="{cid}-{esc(pid)}"><a href="images/{esc(pid)}.jpg"><img loading="lazy" src="images/{esc(pid)}.jpg" alt="Selected photo {esc(pid)}"></a><figcaption>{esc(pid)}</figcaption></figure>')
        parts.append('<div class="candidates">')
        for row in c['candidates']:
            parts.append(f'<section class="candidate" id="{cid}-{row["candidate"]}"><h3>Candidate {row["candidate"]}</h3><pre>{esc(row["text"])}</pre></section>')
        parts.append('</div><details><summary>Approved Core / Supporting evidence units</summary><table><tr><th>Role / ID</th><th>Unit</th><th>Evidence</th></tr>')
        for u in sorted(c['approved_units'], key=lambda u: (u['approved_reference_role'] != 'core', u['unit_id'])):
            evidence = '<br>'.join('<b>' + esc(e['marker']) + '</b> ' + esc(e['quote']) for e in u['evidence'])
            parts.append(f'<tr id="{esc(u["unit_id"])}"><td>{esc(u["approved_reference_role"])}<br>{esc(u["unit_id"])}</td>'
                         f'<td>{esc(u["text"])}</td><td>{evidence}</td></tr>')
        parts.append('</table></details></article>')
    parts.append('</html>')
    (public / 'REVIEW.html').write_text('\n'.join(parts), encoding='utf-8')
    metrics = []
    for model, records in rows.items():
        for subset in ('random', 'targeted'):
            group = [r for r in records if r['subset'] == subset]
            zero = [(r, c) for r, c in zip(records, cases) if c['subset'] == subset and not c['selected_photo_ids']]
            metrics.append({'model': model, 'subset': subset, 'n': len(group),
                'exact_sentence_flags': sum(r['automatic']['repetition']['exact_repeated_sentence'] for r in group),
                'legacy_word_flags_not_semantic_labels': sum(r['automatic']['repetition']['consecutive_repeated_word'] for r in group),
                'zero_caption_cases': len(zero),
                'zero_caption_nonempty_visual_sections': sum(bool(r['text'].split('[VISUAL_EVIDENCE]', 1)[-1].strip()) for r, _ in zero)})
    write(admin / 'SOFTWARE_METRICS_NOT_QUALITY_LABELS.json', metrics)
    recommendation = '''# Researcher-facing recommendation — provisional, offline

**Recommendation: select neither LED-large checkpoint for substantive GO now.**
Technical success is established. This recommendation is identity-aware model
assistance; researcher quality GO/NO-GO remains PENDING. This assessment uses text,
source captions and approved evidence references; the assistant has not freshly
visually reassessed all 90 photos. No new human output
annotations, coverage percentages, quantitative quality scores or win rates
have been fabricated. Complete the separate blinded review before deciding.

Both epoch3 and epoch5 still assert a photographed bagel sandwich for Brooklyn
Water Bagel despite zero Filtered captions. That directly violates the stated
grounding requirement. D059 base3 also violates it; its primary status derives
from the established matched experiment, not a claim of perfect quality.

JOEY Mayfield combines women sitting with drinks and a separate person preparing
food into women preparing food. Scarlett Begonia adds egg/lemon-syrup imagery
absent from all five captions. These are unsupported visual claims/relations,
not harmless wording variation. Exact phrases and source/caption evidence are
in PROVISIONAL_MODEL_ASSISTED_OBSERVATIONS.json and the blinded case pages.

Core gains exist, but are mixed. Large3 names dishes and restores identity in
several cases. Large5 improves selected visual accounts and removes the one
exact-sentence flag, but loses important content: Novella tasting-service wait
and price criticism; Mission BBQ meat-quality criticism; Bonefish Bang Bang
shrimp; Aquarium's visual fish-tank Core. Large5 helps other cases (e.g. Peel
noise/value and Dunkin caption separation). Therefore neither lower loss nor
cleaner exact-repetition counts justify a consistent large5 preference.

Random20: base3/large3/large5 exact-sentence flags = 3/1/0; legacy word flags =
12/4/2. Targeted4: exact flags = 0/0/0; legacy word flags = 1/0/1. Legacy flags
include legitimate names and miss repetitions such as inattentive, or
inattentive. These are software diagnostics, not harmful-semantic-repeat labels.
All targeted candidates still miss important information: bagel coffee/order
contrasts; Pool DJ visual Core; KFC order errors; Dunkin hygiene and missing
orders. See per-case D053 evidence anchors; no exhaustive recall is asserted.

The annotation package contains all 72 original summaries, the exact 24 frozen
sources, 90 photos, and unchanged D053 references (261 Core/315 Supporting).
Three within-case pairs are provided for each business. A/B/C is permuted per
case, balanced across 24 cases; do not aggregate by letter. Blinding key and
these model-assisted observations are deliberately excluded from that ZIP.
The assistant has previously seen model identities and is NOT a blinded judge.
Text style and previous exposure may still reveal identities to a reviewer.

These are 20 random +4 targeted REUSED validation businesses, not fresh held-out
evidence. D059 differs in architecture, initialization and training/label history:
this is a practical, non-causal comparison. Corrected labels and technical BOS
recovery are supported in this run, not an isolated causal quality experiment.
Human references are the original explicit D053 approvals; new output labels
remain blank. Supporting is not Core: bagel coffee ice cubes is Supporting.

If independent researcher-confirmed review establishes GO on substantive quality,
choose one checkpoint based on evidence and only then seek separate authorization
to PREPARE matched corrected-label large RO/U. No automatic training follows.
Otherwise retain V2 as technically successful but quality-limited supplementary
evidence and proceed with established matched LED-base RO/U/F/Shuffled evaluation.
No held-out access, target changes, Teacher calls, GPU work or decoding searches
occurred in this task. All execution backups/checkpoints remain preserved;
instance57028166 was already destroyed after verified backup.
'''
    (admin / 'CHECKPOINT_RECOMMENDATION_VI.md').write_text(
        '# Khuyến nghị sơ bộ cho nhà nghiên cứu\n\n'
        '**Chưa chọn checkpoint large nào cho quality GO. Quyết định nghiên cứu vẫn PENDING.**\n\n'
        'Hai epoch đều vi phạm zero-caption và còn lỗi liên kết thông tin hình ảnh. Epoch3/5 '
        'có đánh đổi về Core và độ gọn; chưa có căn cứ chọn epoch5 từ loss hoặc exact repeats. '
        'Các nhận xét hỗ trợ bằng mô hình là tạm thời, người đánh giá đã biết danh tính model.\n\n'
        'Gửi riêng ZIP mù cho reviewer; giữ blinding key và nhận xét này ở ngoài. '
        'Sau review, nếu không có xác nhận GO về chất lượng nội dung, giữ large là thí nghiệm '
        'bổ sung và tiếp tục đánh giá matched LED-base RO/U/F/Shuffled. Không tự train tiếp.\n\n'
        'Xem RECOMMENDATION.md để có lập luận, giới hạn và dẫn chứng chi tiết.\n', encoding='utf-8')
    (admin / 'RECOMMENDATION.md').write_text(recommendation, encoding='utf-8')
    # Focused export checks; these are integrity assertions, not scientific tests.
    assert len(templates) == len(key) == 72
    assert all(all(v is None for v in t['ratings_0_to_4_higher_is_better'].values()) and
               not t['researcher_confirmed'] for t in templates)
    assert all(Counter(k['model'] for k in key if k['candidate'] == slot) ==
               Counter({m: 8 for m in MODELS}) for slot in 'ABC')
    for c in blind_cases:
        assert len({r['candidate'] for r in c['candidates']}) == 3
        for r in c['candidates']:
            k = next(k for k in key if k['case_id'] == c['case_id'] and k['candidate'] == r['candidate'])
            assert text_hash(r['text']) == k['exact_text_sha256']
    assert all(digest(ROOT / rel) == sha for rel, sha in originals.items())
    # Check that no per-model metadata / original output files / private recommendations leaked.
    for p in public.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.html', '.md'):
            raw = p.read_text(encoding='utf-8')
            assert not any(forbidden in raw for forbidden in [
                'LED_large_Expanded_epoch3', 'LED_large_Expanded_epoch5', 'Expanded_full_epoch3',
                'large_v2_epoch3', 'large_v2_epoch5', 'D059_base_epoch3',
                'baseline_prediction', 'BLINDING_KEY_DO_NOT_SEND_TO_REVIEWER', 'private_seed'])
    manifest = {p.relative_to(public).as_posix(): digest(p) for p in public.rglob('*') if p.is_file()}
    write(public / 'SHA256_MANIFEST.json', {'files': manifest})
    archive = ROOT / 'outputs/cloud_handoff/LED_LARGE_V2_BLINDED_QUALITY_REVIEW_V1.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(public.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(public).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest) | {'SHA256_MANIFEST.json'}
        for name, sha in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == sha
    receipt = {'status': 'OFFLINE_REVIEW_READY_RESEARCHER_DECISION_PENDING',
        'cases': 24, 'random': 20, 'targeted': 4, 'unaltered_output_texts': 72,
        'within_case_pairs': 72, 'core_units': 261, 'supporting_units': 315,
        'photo_files': len(list((public / 'images').glob('*.jpg'))),
        'blank_output_annotations': 72, 'researcher_output_labels_assigned': 0,
        'model_assisted_notes_blinded': False, 'key_in_public_archive': False,
        'original_input_hashes_unchanged': True, 'archive_crc_and_all_file_hashes_pass': True,
        'archive': str(archive.relative_to(ROOT)), 'sha256': digest(archive),
        'bytes': archive.stat().st_size, 'manifest_files': len(manifest),
        'model_operations': 0, 'recommendation': 'NO_CHECKPOINT_SELECTED_FOR_QUALITY_GO_PENDING_RESEARCHER_REVIEW'}
    write(DEST / 'READINESS_AND_INTEGRITY.json', receipt)
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256'] + '  ' + archive.name + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
