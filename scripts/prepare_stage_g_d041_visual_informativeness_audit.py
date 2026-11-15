"""Offline, annotation-ready D041 visual informativeness comparisons; no AI labels."""
import base64
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import statistics
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from prepare_stage_g_teacher_pilot import ROOT, read, canonical, digest, file_sha, immutable
from stage_g_visual_target_serialization import visual_claims

SOURCE = ROOT / 'outputs/stage_g/teacher_extended_pilot_d041_v1/recovery_v4'
OUT = ROOT / 'outputs/stage_g/d041_visual_informativeness_audit_v1'
PACKAGE_ID = 'D041_visual_informativeness_random25_v1'
CONDITIONS = ['unfiltered', 'filtered']
STOPWORDS = set('a an the and or of on in at to for with by from is are was were be been being '
                'it its this that these those there some as into next up has have holding sitting '
                'shown showing shows photo image picture close'.split())
GENERIC_PATTERNS = {
    'plain_food_container_on_table': r'^(?:the photo shows )?(?:a |the )?(?:white )?(?:plate|bowl|container)(?: of food)? (?:is |sits |sitting |rests |is sitting |is resting )?on (?:a |the )?(?:table|tabletop)(?: surface)?$',
    'plain_food_in_container': r'^(?:a |the )?(?:plate|bowl|container) (?:contains|holds|has) food$',
    'container_color_only': r'^(?:a |the )?(?:plate|bowl|cup|container) is (?:white|black|red|blue|green)$',
    'plain_photo_of_food': r'^(?:the photo shows |there is |a photo of )?(?:a |the )?(?:plate|bowl) of food$',
    'plain_tables_chairs': r'^(?:a |the )?(?:large |dining )?room (?:with|contains|has|is filled with) (?:lots of )?tables and chairs(?: in it)?$',
    'plain_restaurant': r'^(?:the photo shows |there is )?(?:a |the )?restaurant$',
}
MOTIFS = {'tables_and_chairs': r'\btables and chairs\b', 'food_on_table': r'\bfood\b.*\btable\b',
          'plate_or_bowl_of_food': r'\b(?:plate|bowl) of food\b'}
SPECIFICITY_CUES = set('chicken strawberries strawberry pecans pecan waterfront ocean river lake patio outdoor pizza hamburger '
                      'burger sandwich lobster shrimp sushi salad taco tacos pancake pancakes waffles doughnuts doughnut '
                      'beef pork steak salmon dining bar counter dumplings asparagus beans'.split())
RUBRIC = '''# Visual informativeness rubric — human assessment, not an automatic filter

Use the exact frozen25 D-041 businesses and both conditions. Inspect the original photo, exact BLIP-2 caption, raw teacher assessment, retained visual target and selected reviews. No rating below is prefilled. Teacher verdicts are model-produced labels, not human ground truth.

## Factual support — assess separately first

For each exact teacher fact: supported / unsupported / uncertain. For a whole caption, mixed is also available when only some claims are supported. Check the original photo AND caption support. Do not invent photo-only details absent from the caption or rewrite a fact to make it pass. A teacher rejection is not independently certified correct by this export; inspect the explanation and evidence. Ordinary reasonable visual interpretations and correct short/simple descriptions are acceptable.

## Informativeness levels

**Level0 — Generic:** Correct or plausible, but conveys little meaningful restaurant-specific information. Example: “A plate of food is on a table.”

**Level1 — Descriptive:** Provides concrete observable details about food, ingredients, presentation, interior or environment. Example: “The salad contains chicken, strawberries and pecans.”

**Level2 — Informative:** Provides specific, potentially useful restaurant-related information, such as a distinctive dish, meaningful presentation detail, dining environment, scenic view or clearly observable condition. Example: “The dining area overlooks the waterfront.”

Keep correctness and informativeness separate. Do not reward an unsupported detailed claim. Leave useful-information ratings empty for unsupported/uncertain facts; record the factual issue separately. Do not penalize short text merely for being short. Lexical flags do not assign Level0 or any other quality score.

## Novelty relative to the SELECTED original reviews

- Already substantially covered by selected reviews.
- Partially overlapping with reviews.
- Potentially new information beyond reviews.
- Uncertain/not assessed.

Record relevant REV IDs and a note. Compare with the complete selected reviews, not just the teacher consensus. Overlap does not make a fact worthless: visual evidence may concretely illustrate or corroborate reviews. Keyword matches are search hints only, not entailment or novelty judgments.

## Comparing captions and teacher outputs

Score the caption and retained teacher facts separately. Check whether the teacher preserved supported concrete details; whether omitted details were unsupported; whether simple repeated sentences crowd out useful material; and whether Filtered has more useful information per caption/fact or per business. Consider caption-count differences and zero-Filtered businesses. Fewer facts are not automatically better or worse. Exact wording repetition is not proof of semantic redundancy or low usefulness.

## Diagnose after your own review

A — Caption bottleneck: source captions already generic, leaving little useful material.
B — Teacher selection bottleneck: supported meaningful caption details were lost or generic material overemphasized.
C — Adequate visual information: useful concrete details survive, even with some simple descriptions.
Mixed / uncertain is permitted. Do not assume A,B orC from lexical counts alone. Record rationale and caption/fact/PHOTO/REV evidence IDs.

## Annotation provenance

Researcher is the primary human annotator. ChatGPT comparison is optional. Store ChatGPT second opinion separately from human final decision, and mark a row human_verified only after explicit user confirmation. This diagnostic review is separate from final30-business thesis human evaluation. No source labels, teacher verdicts, targets, methods or filtering rules are changed by exporting annotation files.
'''


def normalize(text):
    return ' '.join(text.lower().split()).rstrip('.!?')


def tokens(text):
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())) - STOPWORDS


def hits(text, patterns):
    return [name for name, pattern in patterns.items() if re.search(pattern, normalize(text))]


def summary(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def repeats(texts):
    counts = Counter(normalize(t) for t in texts)
    return {'instances': len(texts), 'distinct_normalized_wordings': len(counts),
            'wordings_occurring_more_than_once': sum(n > 1 for n in counts.values()),
            'instances_in_repeated_wordings': sum(n for n in counts.values() if n > 1),
            'extra_occurrences_beyond_first': sum(n-1 for n in counts.values() if n > 1),
            'repeated_wordings': [{'wording': t, 'count': n} for t, n in sorted(counts.items(), key=lambda p: (-p[1], p[0])) if n > 1]}


def keyword_hints(text, reviews):
    terms = tokens(text)
    candidates = []
    for review in reviews:
        common = terms & tokens(review['text'])
        if common:
            candidates.append({'review_marker': review['review_marker'], 'review_id': review['review_id'],
                               'shared_lexical_tokens': sorted(common)})
    candidates.sort(key=lambda x: (-len(x['shared_lexical_tokens']), int(x['review_marker'].split('_')[1])))
    return candidates[:5]


def blank_assessment(kind):
    row = {'factual_support': None, 'informativeness': None, 'novelty_vs_reviews': None,
           'review_evidence_ids': [], 'notes': '', 'human_verified': False,
           'chatgpt_second_opinion': None, 'human_final_provenance': None}
    if kind == 'business':
        row['bottleneck'] = None
    if kind == 'caption':
        row['exclusion_appropriate'] = None
    return row


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding='utf-8-sig') == text
    else:
        path.write_text(text, encoding='utf-8')


def csv_export(path, rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
    for row in rows:
        writer.writerow({k: '' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    # Plain UTF8 is portable; original source strings remain exact in authoritative JSON.
    write_text(path, stream.getvalue())


def preserved_snapshot():
    roots = [SOURCE.parent, ROOT/'outputs/stage_g/d041_final_pilot_audit_v1']
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for folder in roots for p in folder.rglob('*') if p.is_file()}


def extract():
    bundle_report = read(SOURCE/'human_review_bundle_report.json')
    bundle = ROOT/bundle_report['zip_path']
    assert file_sha(bundle) == bundle_report['zip_sha256']
    with zipfile.ZipFile(bundle) as archive:
        sample = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('/cases.json'))))['businesses']
    manual = read(SOURCE/'manual_review_25_manifest.json')
    assert [b['business_id'] for b in sample] == [b['business_id'] for b in manual['businesses']]
    assert len(sample) == 25 and file_sha(SOURCE/'manual_review_25_manifest.json') == file_sha(SOURCE.parent/'manual_review_25_manifest.json')
    jobs = read(SOURCE/'pilot_jobs.json'); by_job = {j['job_id']: j for j in jobs}
    cases, all_caption_rows, all_fact_rows, all_business_rows, annotation_rows = [], [], [], [], []
    images, source_checks = {}, {}
    for old in sample:
        bid = old['business_id']; target_path = SOURCE/'execution_v1/provisional_targets'/f'{bid}.json'
        target = read(target_path); source_checks[target_path.relative_to(ROOT).as_posix()] = file_sha(target_path)
        assert target['targets'] == old['final_targets']
        shared_job = by_job[old['components']['shared']['job_id']]
        reviews = shared_job['evidence']['reviews']; assert old['reviews'] == reviews
        case = {'case_id': old['case_id'], 'business_id': bid, 'split': old['split'],
                'selected_reviews': reviews, 'review_consensus': target['targets']['reviews_only'],
                'shared_review_claims_with_exact_support': target['evidence_records']['shared'],
                'conditions': {}, 'human_verified': False}
        for condition in CONDITIONS:
            component = old['components'][condition]; job = by_job[component['job_id']]
            cp_path = SOURCE/'execution_v1/checkpoints'/f"{job['job_id']}.json"
            cp = read(cp_path); source_checks[cp_path.relative_to(ROOT).as_posix()] = file_sha(cp_path)
            assert cp == component['checkpoint']
            raw = SOURCE/'execution_v1/raw_responses'/f"{job['job_id']}.json"
            if raw.exists():
                assert file_sha(raw) == cp['raw_response_sha256']; source_checks[raw.relative_to(ROOT).as_posix()] = file_sha(raw)
            assessments = cp['output']['assessments']
            facts = target['visual_target_claims_with_all_evidence'][condition]
            assert facts == visual_claims(assessments)
            exact_visual_text = target['targets'][condition].split('\n\n[VISUAL_EVIDENCE]\n',1)[1]
            assert exact_visual_text == ' '.join(f['text'] for f in facts)
            assert target['targets'][condition] == case['review_consensus']+'\n\n[VISUAL_EVIDENCE]\n'+exact_visual_text
            cond = {'original_condition_target': target['targets'][condition], 'visual_target_text': exact_visual_text,
                    'captions': [], 'retained_visual_facts': [], 'zero_caption': not job['evidence']['captions']}
            for index, fact in enumerate(facts):
                row_id = f'{bid}|{condition}|AUDIT_FACT_{index+1}'
                field = {'audit_row_id': row_id, 'audit_fact_index': index, **fact,
                         'lexical_generic_template_hits': hits(fact['text'],GENERIC_PATTERNS),
                         'lexical_motif_hits': hits(fact['text'],MOTIFS),
                         'review_keyword_hints_not_entailment': keyword_hints(fact['text'], reviews),
                         'human_assessment': blank_assessment('fact')}
                cond['retained_visual_facts'].append(field)
                annotation_rows.append({'row_id': row_id, 'scope': 'fact','business_id':bid,'condition':condition,
                                        **blank_assessment('fact')})
                all_fact_rows.append({'row_id':row_id,'business_id':bid,'condition':condition,'original_caption':
                                      ' || '.join(c['caption'] for c in job['evidence']['captions'] if c['caption_id'] in {s['caption_id'] for s in fact['support']}),
                                      'teacher_visual_evidence':fact['text'],'evidence_ids':fact['support'],
                                      'factual_support':None,'informativeness':None,'novelty_vs_reviews':None,
                                      'review_evidence_ids':None,'notes':None,'human_verified':False})
            for cap, photo, assessment, exported_cap in zip(job['evidence']['captions'],job['images'],assessments,component['captions']):
                assert all(exported_cap[k] == cap[k] for k in cap)
                assert (assessment['photo_id'],assessment['caption_id']) == (cap['photo_id'],cap['caption_id'])
                original_image = ROOT/photo['relative_path']; assert file_sha(original_image) == photo['raw_sha256']
                image_file = 'images/'+cap['photo_id']+'.jpg'; images[image_file] = original_image
                source_checks[photo['relative_path']] = photo['raw_sha256']
                referenced = [f for f in cond['retained_visual_facts'] if any(s['caption_id']==cap['caption_id'] for s in f['support'])]
                target_tokens = tokens(' '.join(f['text'] for f in referenced))
                omitted = sorted(tokens(cap['caption'])-target_tokens)
                exclusions = [dict(raw_claim_index=i,**c) for i,c in enumerate(assessment['claims']) if c['verdict']!='verified']
                row_id = f"{bid}|{condition}|{cap['caption_id']}"
                field = {**cap, 'image_file':image_file,'original_image_sha256':photo['raw_sha256'],
                         'raw_teacher_assessment':assessment,'retained_fact_ids':[f['audit_row_id'] for f in referenced],
                         'retained_teacher_fact_texts':[f['text'] for f in referenced], 'audit_row_id':row_id,
                         'lexical_generic_template_hits':hits(cap['caption'],GENERIC_PATTERNS),
                         'lexical_motif_hits':hits(cap['caption'],MOTIFS),
                         'lexical_specificity_cue_words':sorted(tokens(cap['caption'])&SPECIFICITY_CUES),
                         'caption_content_tokens_not_in_its_retained_teacher_facts':omitted,
                         'lexical_omission_candidate':len(omitted)>=2,
                         'teacher_rejected_or_uncertain_claims':exclusions,
                         'appropriateness_of_teacher_exclusion':None, 'human_assessment':blank_assessment('caption')}
                cond['captions'].append(field)
                annotation_rows.append({'row_id':row_id,'scope':'caption','business_id':bid,'condition':condition,**blank_assessment('caption')})
                all_caption_rows.append({'row_id':row_id,'business_id':bid,'condition':condition,'caption_id':cap['caption_id'],
                                         'photo_id':cap['photo_id'],'photo_marker':cap['photo_marker'],'image_file':image_file,
                                         'original_caption':cap['caption'],'teacher_visual_evidence':' || '.join(f['text'] for f in referenced),
                                         'teacher_verdicts':[c['verdict'] for c in assessment['claims']],
                                         'factual_support':None,'informativeness':None,'novelty_vs_reviews':None,
                                         'exclusion_appropriate':None,'review_evidence_ids':None,'notes':None,'human_verified':False})
            # Every original caption and target fact has exactly one worksheet row.
            assert len(cond['captions']) == len(job['evidence']['captions']) == len(assessments) == len(component['captions'])
            assert len(cond['retained_visual_facts']) == len(facts)
            raw_claims = [c for a in assessments for c in a['claims']]
            raw_verified = sum(c['verdict']=='verified' for c in raw_claims)
            business_row = {'business_id':bid,'split':old['split'],'condition':condition,'captions':len(cond['captions']),
                            'raw_candidate_claims':len(raw_claims),'raw_verified_claims':raw_verified,
                            'retained_target_facts':len(facts),'exact_repeated_verified_occurrences_collapsed':raw_verified-len(facts),
                            'teacher_rejected':sum(c['verdict']=='rejected' for c in raw_claims),
                            'teacher_uncertain':sum(c['verdict']=='uncertain' for c in raw_claims),
                            'retained_facts_per_caption':len(facts)/len(cond['captions']) if cond['captions'] else None,
                            'captions_matching_generic_templates_lexical_only':sum(bool(c['lexical_generic_template_hits']) for c in cond['captions']),
                            'target_facts_matching_generic_templates_lexical_only':sum(bool(c['lexical_generic_template_hits']) for c in cond['retained_visual_facts']),
                            'lexical_omission_candidate_captions':sum(c['lexical_omission_candidate'] for c in cond['captions']),
                            'zero_caption':cond['zero_caption']}
            cond['descriptive_counts'] = business_row; case['conditions'][condition] = cond
            all_business_rows.append(business_row)
            annotation_rows.append({'row_id':f'{bid}|{condition}|BUSINESS','scope':'business','business_id':bid,'condition':condition,
                                    **blank_assessment('business')})
        cases.append(case)
    assert len(all_caption_rows) == 206 and len(images) == 168 and sum(len(c['selected_reviews']) for c in cases) == 719
    assert len({r['row_id'] for r in annotation_rows}) == len(annotation_rows)
    return cases,images,all_caption_rows,all_fact_rows,all_business_rows,annotation_rows,source_checks


def stats_for(cases, condition):
    rows = [c['conditions'][condition]['descriptive_counts'] for c in cases]
    captions = [r for c in cases for r in c['conditions'][condition]['captions']]
    facts = [r for c in cases for r in c['conditions'][condition]['retained_visual_facts']]
    nonempty = sum(r['captions']>0 for r in rows)
    return {'business_denominator':25,'nonempty_businesses':nonempty,
            'caption_count':len(captions),'retained_target_fact_count':len(facts),
            'caption_count_per_business_including_zero':summary([r['captions'] for r in rows]),
            'target_facts_per_business_including_zero':summary([r['retained_target_facts'] for r in rows]),
            'target_facts_per_nonempty_business':summary([r['retained_target_facts'] for r in rows if r['captions']]),
            'aggregate_retained_facts_per_input_caption':len(facts)/len(captions),
            'raw_candidate_claims':sum(r['raw_candidate_claims'] for r in rows),
            'teacher_verdicts':{kind:sum(r['raw_verified_claims' if kind=='verified' else 'teacher_'+kind] for r in rows) for kind in ['verified','rejected','uncertain']},
            'exact_verified_repetition_removed_by_existing_serializer':sum(r['exact_repeated_verified_occurrences_collapsed'] for r in rows),
            'caption_generic_lexical_matches':sum(bool(r['lexical_generic_template_hits']) for r in captions),
            'caption_generic_lexical_match_denominator':len(captions),
            'fact_generic_lexical_matches':sum(bool(r['lexical_generic_template_hits']) for r in facts),
            'fact_generic_lexical_match_denominator':len(facts),
            'caption_lexical_motif_counts':{name:sum(name in r['lexical_motif_hits'] for r in captions) for name in MOTIFS},
            'fact_lexical_motif_counts':{name:sum(name in r['lexical_motif_hits'] for r in facts) for name in MOTIFS},
            'caption_repetition_across_businesses':repeats([r['caption'] for r in captions]),
            'fact_wording_repetition_across_businesses':repeats([r['text'] for r in facts]),
            'caption_repetition_within_business_extra_occurrences':sum(repeats([r['caption'] for r in c['conditions'][condition]['captions']])['extra_occurrences_beyond_first'] for c in cases),
            'lexical_omission_candidate_caption_count':sum(r['lexical_omission_candidate'] for r in captions),
            'caption_has_no_retained_facts':sum(not r['retained_fact_ids'] for r in captions),
            'captions_with_teacher_exclusions':sum(bool(r['teacher_rejected_or_uncertain_claims']) for r in captions),
            'zero_caption_businesses':[c['business_id'] for c in cases if c['conditions'][condition]['zero_caption']],
            'subjective_quality_ratings_created':0, 'appropriately_excluded_claims_human_confirmed':None}


def examples(cases):
    generic, generic_facts, motifs, cues, omissions, exclusions = [],[],[],[],[],[]
    for case in cases:
        for condition in CONDITIONS:
            for cap in case['conditions'][condition]['captions']:
                row = {'business_id':case['business_id'],'condition':condition,'caption_id':cap['caption_id'],
                       'photo_id':cap['photo_id'],'photo_marker':cap['photo_marker'],'image_file':cap['image_file'],
                       'exact_caption':cap['caption'],'retained_teacher_fact_texts':cap['retained_teacher_fact_texts']}
                if cap['lexical_generic_template_hits']:
                    generic.append({**row,'lexical_template_hits':cap['lexical_generic_template_hits']})
                if cap['lexical_motif_hits']:
                    motifs.append({**row,'lexical_motif_hits':cap['lexical_motif_hits']})
                if cap['lexical_specificity_cue_words']:
                    cues.append({**row,'lexical_cue_words':cap['lexical_specificity_cue_words']})
                if cap['lexical_omission_candidate']:
                    omissions.append({**row,'missing_lexical_tokens':cap['caption_content_tokens_not_in_its_retained_teacher_facts']})
                if cap['teacher_rejected_or_uncertain_claims']:
                    exclusions.append({**row,'raw_excluded_claims':cap['teacher_rejected_or_uncertain_claims'],
                                       'exclusion_appropriate_human_label':None})
            for fact in case['conditions'][condition]['retained_visual_facts']:
                if fact['lexical_generic_template_hits']:
                    generic_facts.append({'business_id':case['business_id'],'condition':condition,
                                          'audit_row_id':fact['audit_row_id'],'exact_teacher_fact':fact['text'],
                                          'support':fact['support'],'lexical_template_hits':fact['lexical_generic_template_hits'],
                                          'human_informativeness':None})
    return {'selection_rule':'All omission/exclusion candidates retained in frozen sample order; six template-match and eight cue-word examples for illustration, not quality-ranked.',
            'generic_lexical_examples_not_human_Level0':generic[:6],
            'generic_teacher_fact_examples_not_human_Level0':generic_facts[:8],
            'caption_motif_examples_not_whole_caption_generic_labels':motifs[:6],
            'specific_cue_word_examples_not_human_usefulness_scores':cues[:8],
            'all_lexical_omission_candidates_not_confirmed_information_loss':omissions,
            'all_teacher_exclusion_cases_appropriateness_not_independently_confirmed':exclusions}


def report_text(stats, example_rows):
    u,f = (stats['conditions'][c] for c in CONDITIONS)
    lines = ['# D-041 visual informativeness diagnostic audit', '',
             '**Annotation-ready offline package; no subjective scoring or methodology changes.**', '',
             'Exact preselected25 businesses (19train/6validation),719 selected reviews and168 original photos. '
             'All reviews, captions, targets, teacher verdicts and evidence mappings are preserved. '
             'Researcher provisionally accepted structural/factual protocol; production remains pending this separate informativeness review.', '',
             '| Descriptive measure | Unfiltered | Filtered |','|---|---:|---:|']
    pairs = [('Businesses (all means include zero)','business_denominator'),('BLIP-2 captions','caption_count'),
             ('Raw teacher candidate claims','raw_candidate_claims'),('Distinct retained target facts (sum of within-business wording groups)','retained_target_fact_count'),
             ('Exact repeated verified occurrences collapsed by frozen serializer','exact_verified_repetition_removed_by_existing_serializer'),
             ('Caption lexical-template matches; not human generic scores','caption_generic_lexical_matches'),
             ('Fact lexical-template matches; not human generic scores','fact_generic_lexical_matches'),
             ('Caption lexical-omission candidates; not confirmed loss','lexical_omission_candidate_caption_count'),
             ('Captions with no retained teacher fact','caption_has_no_retained_facts')]
    lines += [f'| {label} | {u[key]} | {f[key]} |' for label,key in pairs]
    lines += [f"| Mean retained facts/business | {u['target_facts_per_business_including_zero']['mean']:.2f} | {f['target_facts_per_business_including_zero']['mean']:.2f} |",
              f"| Median retained facts/business | {u['target_facts_per_business_including_zero']['median']} | {f['target_facts_per_business_including_zero']['median']} |",
              f"| Aggregate target facts/input caption (post-dedup) | {u['aggregate_retained_facts_per_input_caption']:.3f} | {f['aggregate_retained_facts_per_input_caption']:.3f} |",
              f"| Zero-caption businesses | {len(u['zero_caption_businesses'])} | {len(f['zero_caption_businesses'])} |", '',
              '## Denominators and interpretation', '',
              'Caption totals count condition-specific caption/photo evidence entries, so a shared photo may contribute once to each condition. '
              '168 is the number of unique original photos across both conditions. Retained facts are the frozen target sentence groups '
              '(lowercase/whitespace-normalized exact duplicate key within a business/condition), not human-defined semantic facts. '
              'Raw verified occurrences may support the same single retained sentence through multiple photos. '
              'Across-business repeated wording is reported separately and is not removed or declared useless.', '',
              f"Unfiltered raw verdicts: {u['teacher_verdicts']}; Filtered: {f['teacher_verdicts']}.",
              'Teacher rejected/uncertain status is not independent confirmation that an exclusion was appropriate. '
              'The worksheet supplies original photos, exact quotes and explanations to check that question.', '',
              '## Transparent lexical rules', '',
              'lexical_rules.json declares every anchored generic template, motif, normalization rule, stopword and cue word. '
              'Anchored matches are narrow whole-text templates; motif hits merely indicate a phrase occurs in possibly rich text. '
              'No motif or template assigns a human informativeness level. Exact repetition uses lowercase, whitespace collapse and terminal .!? removal '
              'for descriptive reporting only; this does not change the stricter frozen teacher serializer.', '',
              'A lexical-omission candidate has at least two caption content tokens absent from that caption’s retained teacher facts. '
              'Tokens use the declared regex/stopwords without embeddings, lemmatization or synonym matching. '
              'These may reflect unsupported specifics, valid paraphrases or formatting rather than information loss. '
              'Keyword matches to selected reviews are search hints only; they never prefill novelty or factual support.', '',
              'No original caption matches the narrow whole-text generic templates in this sample; this is NOT evidence that none is generic. '
              f"Broader phrase motifs in captions: Unfiltered {u['caption_lexical_motif_counts']}; Filtered {f['caption_lexical_motif_counts']}. "
              'A motif can occur inside a concrete caption. Human grading must decide whether its complete content is useful.', '',
              '## Side-by-side review', '',
              'index.html shows all25 cases, complete shared consensus and selected original reviews, paired conditions, '
              'each original caption/photo, raw candidate claims/verdicts, exact retained target sentences and supporting evidence IDs. '
              'Caption worksheet and fact worksheet have separate blank factual/informativeness/novelty fields; '
              'business worksheet also leaves bottleneck diagnosis blank. annotations_template.json contains the same blank rows. '
              'Interactive entries are separate drafts and exported files, never changes to saved targets or source annotations.', '',
              '## Examples (lexical selection only)', '']
    for group,label in [('generic_lexical_examples_not_human_Level0','Exact generic-template candidates'),
                        ('caption_motif_examples_not_whole_caption_generic_labels','Captions containing broad food/table or tables/chairs motifs; whole caption remains unscored'),
                        ('specific_cue_word_examples_not_human_usefulness_scores','Captions with explicit cue words; usefulness/factual support still unscored')]:
        lines += ['### '+label,'']
        for row in example_rows[group]:
            lines += [f"- `{row['business_id']}` / {row['condition']} / {row['caption_id']} / PHOTO `{row['photo_id']}`: {row['exact_caption']}",
                      '  Retained exact teacher sentences: '+' '.join(row['retained_teacher_fact_texts'])]
        lines += ['']
    lines += ['### Retained teacher facts matching narrow generic templates (not human Level0 labels)', '']
    for row in example_rows['generic_teacher_fact_examples_not_human_Level0']:
        ids='; '.join(s['caption_id']+' / PHOTO '+s['photo_id'] for s in row['support'])
        lines += [f"- `{row['business_id']}` / {row['condition']} / {ids}: {row['exact_teacher_fact']}"]
    lines += ['']
    lines += ['## Where could information loss originate?', '',
              'No A/B/C conclusion is assigned before human assessment. Generic-template examples allow testing A (caption bottleneck); '
              'lexical omissions plus teacher explanations allow testing B (loss of supported details); cue-word examples allow testing C '
              '(adequate concrete information). None establishes that a caption is useful, that a teacher exclusion is excessive, '
              'or that a detail is factually true. Review original photos and all selected reviews before assigning ratings.', '',
              'Filtered starts with fewer captions (81 versus125); compare paired-business totals AND per-caption/per-fact ratings. '
              'Fewer target sentences alone establish neither improvement nor degradation. The two zero-Filtered businesses retain '
              'the original empty visual header, unchanged review consensus and empty visual text; no evidence is fabricated. '
              'Visual novelty cannot be inferred from absence of an exact word in reviews. Overlapping visual facts can still corroborate useful review information. '
              'A measured contribution to model performance ultimately requires the planned controlled student experiment; this package assesses available evidence, not downstream improvement.', '',
              '## Stop status', '',
              'No human or LLM subjective labels created. No APIcalls, BLIP-2 rerun, new filter, teacher prompt/settings/targets/verdict changes, '
              'Stage F/K/budget changes, production freeze, remaining2100generation or LEDtraining. '
              'The25 sample remains unchanged, and final30-business human evaluation remains separate. '
              'Stop after package preparation for researcher/optional ChatGPT review.', '']
    return '\n'.join(lines)


PAGE_TEMPLATE = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D041 — Visual informativeness review</title><style>
body{margin:0;background:#f1f4f6;color:#172332;font:15px system-ui;line-height:1.5}header{position:sticky;top:0;background:#172332;color:white;padding:12px 22px;z-index:1}main{max-width:1400px;margin:auto;padding:22px}button,select,input,textarea{font:inherit;padding:8px;border:1px solid #bac6cf;border-radius:5px}button{cursor:pointer}header button{margin-right:7px}article{background:white;border:1px solid #cad3db;border-radius:12px;padding:20px;margin:22px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.condition{min-width:0}.caption{border:1px solid #ced7df;margin:15px 0;padding:15px;border-radius:8px}.caption img{width:100%;max-height:340px;object-fit:contain;background:#f5f7fa}.fact{padding:12px;border-left:3px solid #517798;margin:12px 0;background:#f7f9fb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}summary{cursor:pointer;padding:8px 0}.form{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}.form label{font-size:13px;display:flex;flex-direction:column}.form .wide{grid-column:1/-1}textarea{min-height:48px}.badge{font-size:12px;color:#395b76}nav{display:flex;flex-wrap:wrap;gap:6px}nav a{font-size:12px}.muted{color:#586675}.warn{background:#fff2cf;padding:12px}.good{color:#056c50}code{overflow-wrap:anywhere}@media(max-width:950px){.pair{grid-template-columns:1fr}.form{grid-template-columns:1fr}header{position:static}}</style>
<header><b>D041 · Visual informativeness · 25 businesses</b><br><button id="export">Export annotation drafts JSON</button><button id="reset">Clear local drafts</button><span id="saved"></span></header>
<dialog id="image-dialog"><button id="close-image">Close original image</button><br><img id="zoom-image" style="max-width:90vw;max-height:85vh;object-fit:contain"></dialog><main><p class="warn">Các ô đánh giá đều để trống. Teacher verdict là output của model, chưa phải nhãn human-final. Các cờ từ vựng chỉ để tìm ví dụ; không phải điểm quality. Nhập đánh giá không sửa targets. Draft được lưu trong trình duyệt; hãy export JSON để giữ lại.</p>
<details><summary>Rubric / cách đánh giá</summary><pre id="rubric"></pre></details><details><summary>Thống kê mô tả và denominators</summary><pre id="stats"></pre></details><nav id="nav"></nav><div id="cases"></div></main>
<script id="bundle" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('bundle').textContent);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const key=data.package_id+'|'+data.manual25_sha256;const template=JSON.parse(JSON.stringify(data.annotation_template));let state=JSON.parse(JSON.stringify(template));
try{const prior=JSON.parse(localStorage.getItem(key));if(prior&&prior.package_id===template.package_id&&prior.manual25_sha256===template.manual25_sha256&&prior.rows.length===template.rows.length&&prior.rows.every((r,i)=>r.row_id===template.rows[i].row_id))state=prior;}catch(e){}
const byId=new Map(state.rows.map(r=>[r.row_id,r]));
function save(){state.saved_at=new Date().toISOString();try{localStorage.setItem(key,JSON.stringify(state));document.getElementById('saved').textContent='Draft saved locally';}catch(e){document.getElementById('saved').textContent='Local save unavailable — export JSON';}}
function sel(id,name,title,options){const r=byId.get(id);return '<label>'+title+'<select data-row="'+esc(id)+'" data-field="'+name+'"><option value="">— Chưa đánh giá —</option>'+options.map(([v,t])=>'<option value="'+v+'" '+(String(r[name])===String(v)?'selected':'')+'>'+t+'</option>').join('')+'</select></label>';}
function form(id,scope){const r=byId.get(id);let text='<div class="form">'+sel(id,'factual_support','Factual support (human review)',[['supported','Supported'],['unsupported','Unsupported'],['uncertain','Uncertain'],...(scope==='caption'?[['mixed','Mixed caption']]:[])])+sel(id,'informativeness','Informativeness — chỉ tính thông tin supported',[[0,'0 — Generic'],[1,'1 — Descriptive'],[2,'2 — Informative']])+sel(id,'novelty_vs_reviews','Novelty vs SELECTED reviews',[['covered','Already substantially covered'],['partial','Partially overlapping'],['new','Potentially new'],['uncertain','Uncertain / not assessed']]);
if(scope==='business')text+=sel(id,'bottleneck','Bottleneck hypothesis',[['A','A — Caption bottleneck'],['B','B — Teacher selection bottleneck'],['C','C — Adequate visual information'],['mixed','Mixed'],['uncertain','Uncertain']]);
if(scope==='caption')text+=sel(id,'exclusion_appropriate','Teacher exclusion — chỉ đánh giá nếu có rejected/uncertain claims',[['appropriate','Appropriate'],['overstrict','Unnecessarily strict'],['mixed','Mixed'],['uncertain','Uncertain / not applicable']]);
text+='<label>REV IDs / supporting review references<input data-row="'+esc(id)+'" data-field="review_evidence_ids" value="'+esc((r.review_evidence_ids||[]).join(', '))+'"></label><label class="wide">Human review notes<textarea data-row="'+esc(id)+'" data-field="notes">'+esc(r.notes)+'</textarea></label><label class="wide">Optional ChatGPT second opinion — separate provenance<textarea data-row="'+esc(id)+'" data-field="chatgpt_second_opinion">'+esc(r.chatgpt_second_opinion)+'</textarea></label><label class="wide"><span><input type="checkbox" data-row="'+esc(id)+'" data-field="human_verified" '+(r.human_verified?'checked':'')+'> Tôi đã tự xem bằng chứng và xác nhận đánh giá cuối</span></label></div>';return text;}
function renderCondition(b,condition){const c=b.conditions[condition];let text='<div class="condition"><h3>'+condition+' · '+c.captions.length+' captions · '+c.retained_visual_facts.length+' retained facts</h3><details><summary>Exact [VISUAL_EVIDENCE] target</summary><pre>'+esc(c.visual_target_text||'(Empty visual text — zero captions)')+'</pre></details><details><summary>Descriptive counts</summary><pre>'+esc(JSON.stringify(c.descriptive_counts,null,2))+'</pre></details>';
if(c.zero_caption)text+='<p class="warn">Zero Filtered captions: giữ nguyên empty visual header và review consensus; không có caption thay thế.</p>';
for(const cap of c.captions){text+='<section class="caption"><code>'+esc(cap.caption_id+' / '+cap.photo_marker+' / '+cap.photo_id)+'</code><img data-image="'+esc(cap.image_file)+'" src="'+data.images[cap.image_file]+'" alt="'+esc(cap.photo_id)+'" style="cursor:zoom-in"><p><b>Original BLIP-2 caption:</b> '+esc(cap.caption)+'</p><p><b>Retained exact teacher facts from this caption:</b></p><ul>'+cap.retained_teacher_fact_texts.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>'+(!cap.retained_fact_ids.length?'<p>Không có retained teacher fact.</p>':'')+'<details><summary>Raw teacher claims / verdicts / explanations</summary>'+cap.raw_teacher_assessment.claims.map(cl=>'<p><b>'+esc(cl.verdict)+'</b> '+esc(cl.text)+'<br><small>Caption quote: '+esc(cl.caption_quote)+'<br>Teacher explanation: '+esc(cl.explanation)+'</small></p>').join('')+'</details><details><summary>Lexical diagnostics — not quality judgments</summary><pre>'+esc(JSON.stringify({generic_template_hits:cap.lexical_generic_template_hits,motif_hits:cap.lexical_motif_hits,missing_content_tokens:cap.caption_content_tokens_not_in_its_retained_teacher_facts},null,2))+'</pre></details><details><summary>Assess original caption separately</summary>'+form(cap.audit_row_id,'caption')+'</details></section>';}
text+='<h4>Distinct retained target facts — assess each once</h4>';for(const fact of c.retained_visual_facts){text+='<div class="fact"><p><b>'+esc(fact.text)+'</b></p><small>'+esc(fact.support.map(s=>s.caption_id+' / '+s.photo_id+' / raw claim '+s.raw_claim_index).join(' | '))+'</small><details><summary>Exact supporting quotes and keyword search hints</summary><pre>'+esc(JSON.stringify({support:fact.support,review_keyword_hints_not_entailment:fact.review_keyword_hints_not_entailment},null,2))+'</pre></details>'+form(fact.audit_row_id,'fact')+'</div>';}
text+='<details><summary>Overall business/condition review — A/B/C hypothesis</summary>'+form(b.business_id+'|'+condition+'|BUSINESS','business')+'</details></div>';return text;}
document.getElementById('rubric').textContent=data.rubric;document.getElementById('stats').textContent=JSON.stringify(data.statistics,null,2);document.getElementById('nav').innerHTML=data.businesses.map(b=>'<a href="#'+esc(b.business_id)+'">'+esc(b.business_id)+'</a>').join('');
document.getElementById('cases').innerHTML=data.businesses.map(b=>'<article id="'+esc(b.business_id)+'"><h2>'+esc(b.business_id)+' · '+esc(b.split)+'</h2><details><summary>Shared [REVIEW_CONSENSUS]</summary><pre>'+esc(b.review_consensus)+'</pre></details><details><summary>All '+b.selected_reviews.length+' original selected reviews — compare novelty here</summary>'+b.selected_reviews.map(r=>'<details><summary>'+esc(r.review_marker+' / '+r.review_id+' / '+r.stars+' stars / '+r.date)+'</summary><pre>'+esc(r.text)+'</pre></details>').join('')+'</details><div class="pair">'+CONDITIONS_PLACEHOLDER+'</div></article>').join('');
document.addEventListener('input',e=>{const id=e.target.dataset.row,field=e.target.dataset.field;if(!id)return;const r=byId.get(id);if(field==='human_verified'){r[field]=e.target.checked;r.human_final_provenance=e.target.checked?'User explicitly confirmed after own evidence inspection; optional ChatGPT comparison is separate.':null;}else if(field==='informativeness')r[field]=e.target.value===''?null:Number(e.target.value);else if(field==='review_evidence_ids')r[field]=e.target.value.split(',').map(s=>s.trim()).filter(Boolean);else r[field]=e.target.value||null;save();});
document.addEventListener('click',e=>{const image=e.target.dataset.image;if(image){document.getElementById('zoom-image').src=data.images[image];document.getElementById('image-dialog').showModal();}});document.getElementById('close-image').onclick=()=>document.getElementById('image-dialog').close();
document.getElementById('export').onclick=()=>{save();const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='d041_visual_informativeness_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
document.getElementById('reset').onclick=()=>{if(confirm('Clear only local draft annotations? Original targets stay unchanged.')){try{localStorage.removeItem(key);}catch(e){}location.reload();}};
</script></html>'''


def main():
    def no_network(*args,**kwargs):
        raise RuntimeError('Offline visual audit: network/API calls forbidden')
    urllib.request.urlopen=no_network
    assert not OUT.exists(), 'Preserve an existing export; use a new version'
    before = preserved_snapshot()
    cases,images,caption_rows,fact_rows,business_rows,annotation_rows,source_checks = extract()
    stats = {'package_id':PACKAGE_ID,'businesses':25,'selected_reviews':719,'unique_original_photos':168,
             'sample_unchanged':True,'manual25_sha256':file_sha(SOURCE/'manual_review_25_manifest.json'),
             'conditions':{cond:stats_for(cases,cond) for cond in CONDITIONS},
             'subjective_annotations_created':0,'quality_or_novelty_scores_computed':False,
             'factual_support_or_exclusion_appropriateness_confirmed_by_code':False,
             'caption_vs_teacher_bottleneck_conclusion':'pending human assessment, not inferred from lexical counts',
             'API_calls':0,'source_outputs_or_methods_changed':False}
    example_rows=examples(cases)
    rules={'normalization':'lowercase; collapse whitespace; strip terminal .!? for descriptive counts only',
           'generic_whole_text_templates':GENERIC_PATTERNS,'motif_searches_not_whole_caption_generic_judgments':MOTIFS,
           'content_token_regex':r"[a-z]+(?:'[a-z]+)?",'stopwords':sorted(STOPWORDS),
           'lexical_omission_candidate_rule':'at least2 caption content tokens absent from its retained caption-mapped teacher facts',
           'no_lemma_embeddings_synonym_matching':True,'specificity_cue_words_not_quality_scores':sorted(SPECIFICITY_CUES),
           'review_hints':'top5 selected reviews by count of shared content tokens, then REV numeric index; zero matches imply no novelty judgment',
           'used_for_filtering_or_sampling':False,'subjective_score_generation':False}
    annotation_template={'package_id':PACKAGE_ID,'manual25_sha256':stats['manual25_sha256'],
                         'role':'blank independent human review; optional ChatGPT second opinion separate',
                         'original_source_outputs_must_not_be_modified':True,'rows':annotation_rows}
    immutable(OUT/'cases.json',{'package_id':PACKAGE_ID,'businesses':cases,'subjective_scores_prefilled':False})
    immutable(OUT/'statistics.json',stats);immutable(OUT/'lexical_rules.json',rules);immutable(OUT/'examples_and_candidates.json',example_rows)
    immutable(OUT/'annotations_template.json',annotation_template)
    immutable(OUT/'original_manual25_manifest.json',read(SOURCE/'manual_review_25_manifest.json'))
    immutable(OUT/'source_artifact_sha256.json',source_checks)
    immutable(OUT/'preserved_previous_pilot_and_audit_sha256.json',before)
    csv_export(OUT/'caption_review_worksheet.csv',caption_rows);csv_export(OUT/'fact_review_worksheet.csv',fact_rows)
    csv_export(OUT/'business_condition_counts.csv',business_rows)
    csv_export(OUT/'business_review_worksheet.csv',[{'row_id':r['row_id'],'business_id':r['business_id'],'condition':r['condition'],
               'factual_support':None,'informativeness':None,'novelty_vs_reviews':None,'bottleneck':None,'review_evidence_ids':None,'notes':None,'human_verified':False}
               for r in annotation_rows if r['scope']=='business'])
    paired=[]
    for case in cases:
        u,f=(case['conditions'][c]['descriptive_counts'] for c in CONDITIONS)
        paired.append({'business_id':case['business_id'],'split':case['split'],
                       'unfiltered_captions':u['captions'],'filtered_captions':f['captions'],
                       'unfiltered_target_facts':u['retained_target_facts'],'filtered_target_facts':f['retained_target_facts'],
                       'filtered_minus_unfiltered_captions':f['captions']-u['captions'],
                       'filtered_minus_unfiltered_target_facts':f['retained_target_facts']-u['retained_target_facts'],
                       'unfiltered_facts_per_caption':u['retained_facts_per_caption'],'filtered_facts_per_caption':f['retained_facts_per_caption'],
                       'human_usefulness_difference':None})
    csv_export(OUT/'paired_business_comparison.csv',paired)
    immutable(OUT/'per_caption_descriptive_diagnostics.json',[{'business_id':c['business_id'],'condition':cond,**cap}
              for c in cases for cond in CONDITIONS for cap in c['conditions'][cond]['captions']])
    write_text(OUT/'rubric.md',RUBRIC)
    report=report_text(stats,example_rows);write_text(OUT/'AUDIT_REPORT.md',report)
    write_text(OUT/'README.md','Extract the whole ZIP and open index.html. This offline page also works as one self-contained HTML file. '
               'Inspect all25 cases and complete selected reviews. Read rubric.md and AUDIT_REPORT.md. All subjective fields are blank. '
               'Use CSV worksheets or the interactive page; export page drafts as a separate JSON file. '
               'cases.json is authoritative for exact source text/IDs and relative original JPEG paths. '
               'A teacher verified/rejected verdict is not a human label; lexical heuristics are not quality scores. '
               'Optional ChatGPT second opinions must remain separate from user-confirmed human labels. No original target or method is modified.\n')
    image_data={}
    for relative,original in images.items():
        destination=OUT/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination)
        assert file_sha(destination)==file_sha(original)
        image_data[relative]='data:image/jpeg;base64,'+base64.b64encode(destination.read_bytes()).decode()
    payload={'package_id':PACKAGE_ID,'manual25_sha256':stats['manual25_sha256'],'businesses':cases,'statistics':stats,
             'annotation_template':annotation_template,'images':image_data,'rubric':RUBRIC}
    page=PAGE_TEMPLATE.replace('CONDITIONS_PLACEHOLDER', "['unfiltered','filtered'].map(c=>renderCondition(b,c)).join('')")
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=page.replace('__DATA__',text);write_text(OUT/'index.html',page)
    assert preserved_snapshot()==before
    for relative,h in source_checks.items():assert file_sha(ROOT/relative)==h
    members=sorted(p for p in OUT.rglob('*') if p.is_file())
    checks={p.relative_to(OUT).as_posix():file_sha(p) for p in members}
    immutable(OUT/'export_manifest.json',{'package_id':PACKAGE_ID,'file_sha256':checks,'businesses':25,'caption_rows':len(caption_rows),
                                         'fact_rows':len(fact_rows),'business_review_rows':50,'annotation_rows':len(annotation_rows),
                                         'original_images':168,'API_calls':0,'subjective_labels_created':0})
    archive=OUT/'chatgpt_d041_visual_informativeness_review.zip'
    members.append(OUT/'export_manifest.json')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in members:
            data=p.read_bytes();assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',data),'Secret-like export content'
            z.writestr('chatgpt_d041_visual_informativeness_review/'+p.relative_to(OUT).as_posix(),data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in checks.items():assert digest(z.read('chatgpt_d041_visual_informativeness_review/'+name))==h
        restored=json.loads(z.read('chatgpt_d041_visual_informativeness_review/cases.json'))['businesses']
        assert restored==cases
        for c in restored:
            for condition in CONDITIONS:
                for cap in c['conditions'][condition]['captions']:
                    assert digest(z.read('chatgpt_d041_visual_informativeness_review/'+cap['image_file']))==cap['original_image_sha256']
    immutable(OUT/'export_integrity_report.json',{'status':'PASS','businesses':25,'caption_rows':len(caption_rows),'fact_rows':len(fact_rows),
              'annotation_rows':len(annotation_rows),'all_subjective_fields_initially_blank':True,'previous_artifacts_unchanged':len(before),
              'source_artifacts_verified':len(source_checks),'original_image_count':168,'all_relative_image_paths_resolve':True,
              'source_text_IDs_verdicts_exact':True,'human_annotations_created':0,'API_calls':0,'zip_bytes':archive.stat().st_size,
              'zip_path':archive.relative_to(ROOT).as_posix(),'zip_sha256':file_sha(archive)})
    print(json.dumps({'status':'OFFLINE_ANNOTATION_PACKAGE_COMPLETE','counts':{c:{k:stats['conditions'][c][k] for k in
          ['caption_count','retained_target_fact_count','exact_verified_repetition_removed_by_existing_serializer','caption_generic_lexical_matches',
           'fact_generic_lexical_matches','lexical_omission_candidate_caption_count']} for c in CONDITIONS},
          'zip':str(archive),'zip_bytes':archive.stat().st_size,'human_or_LLM_subjective_labels_created':0},indent=2))


def finalize_prepared_package():
    """Finish this newly prepared audit presentation; previous scientific artifacts are immutable."""
    before=read(OUT/'preserved_previous_pilot_and_audit_sha256.json')
    assert preserved_snapshot()==before
    cases=read(OUT/'cases.json')['businesses'];annotation=read(OUT/'annotations_template.json')
    assert all(r['factual_support'] is None and r['informativeness'] is None and r['novelty_vs_reviews'] is None
               and not r['human_verified'] for r in annotation['rows'])
    stats=read(OUT/'statistics.json');stats['conditions']={c:stats_for(cases,c) for c in CONDITIONS}
    example_rows=examples(cases)
    for name,obj in [('statistics.json',stats),('examples_and_candidates.json',example_rows)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'AUDIT_REPORT.md').write_text(report_text(stats,example_rows),encoding='utf-8')
    images={p.relative_to(OUT).as_posix():'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()
            for p in (OUT/'images').glob('*.jpg')}
    payload={'package_id':PACKAGE_ID,'manual25_sha256':stats['manual25_sha256'],'businesses':cases,
             'statistics':stats,'annotation_template':annotation,'images':images,'rubric':RUBRIC}
    page=PAGE_TEMPLATE.replace('CONDITIONS_PLACEHOLDER',"['unfiltered','filtered'].map(c=>renderCondition(b,c)).join('')")
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (OUT/'index.html').write_text(page.replace('__DATA__',text),encoding='utf-8')
    print('Prepared audit presentation updated; source cases/worksheets/labels unchanged. Repackage after offline UI checks.')


def package_checked_export():
    before=read(OUT/'preserved_previous_pilot_and_audit_sha256.json');assert preserved_snapshot()==before
    members=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in ['export_manifest.json','export_integrity_report.json'])
    checks={p.relative_to(OUT).as_posix():file_sha(p) for p in members}
    manifest=read(OUT/'export_manifest.json');manifest['file_sha256']=checks
    (OUT/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    members.append(OUT/'export_manifest.json');archive=OUT/'chatgpt_d041_visual_informativeness_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in members:
            b=p.read_bytes();assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',b)
            z.writestr('chatgpt_d041_visual_informativeness_review/'+p.relative_to(OUT).as_posix(),b)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in checks.items():assert digest(z.read('chatgpt_d041_visual_informativeness_review/'+name))==h
    report=read(OUT/'export_integrity_report.json');report.update(zip_bytes=archive.stat().st_size,zip_sha256=file_sha(archive),offline_UI_check=read(OUT/'offline_ui_check_report.json')['status'])
    (OUT/'export_integrity_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert preserved_snapshot()==before
    print(json.dumps({'status':'FINAL_EXPORT_INTEGRITY_PASS','zip_bytes':archive.stat().st_size,'subjective_labels_created':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize-presentation',action='store_true');parser.add_argument('--package-checked',action='store_true')
    args=parser.parse_args()
    if args.finalize_presentation:finalize_prepared_package()
    elif args.package_checked:package_checked_export()
    else:main()
