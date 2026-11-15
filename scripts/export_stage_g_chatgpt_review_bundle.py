"""Export the complete Stage G pilot for optional ChatGPT comparison; no API or annotation writes."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import zipfile

from prepare_stage_g_teacher_pilot import ROOT, read, file_sha
from prepare_stage_g_review_evidence_ids import evidence_spans

CONFIG = 'configs/stage_g_teacher_pilot_v3_canonical_ids.json'
OUTPUT = ROOT / 'outputs/stage_g/chatgpt_stage_g_pilot_review_bundle.zip'
REPORT = OUTPUT.with_name('chatgpt_stage_g_pilot_review_bundle_report.json')
PREFIX = 'chatgpt_stage_g_pilot_review_bundle/'

README = """# Stage G pilot: optional independent ChatGPT comparison

This export contains ALL 20 predeclared pilot businesses (10 train / 10 validation),
all 60 condition target texts, their complete selected reviews, exact LED/teacher
captions, all mapped original images, teacher assessments, and provenance.
There is no held-out test data. Nothing was selected or omitted after judging quality.
These are provisional model-generated pseudo-targets, not human reference summaries.

## Files

- `cases.json`: complete structured evidence for all 20 businesses, in frozen pilot order.
- `businesses/<business_id>.json`: the same record per business for manageable batches.
- `images/<photo_id>.jpg`: 128 original JPEG files, unchanged and shared across conditions.
- `index.html`: open after extracting the ZIP to browse targets, reviews, captions and images.
- `rubric.md`: the review scope, source constraints and second-opinion provenance rules.
- `protocol/`: exact actual pilot prompts, schema, active config and Stage G spec.
- `provenance/`: unchanged completion/audit/human-confirmation records and current progress snapshot.
- `review_response_template.json`: optional structure for AI suggestions; it contains no new labels.
- `export_manifest.json`: portable file checksums and export checks.

Read the current consolidated snapshot at the top of `provenance/THESIS_PROGRESS.md`.
Older entries are historical. Production generation for all 2,200 businesses is not
frozen or started. K, thresholds, source budgets and existing labels are already fixed
where declared; this review must not silently change them.

## Copy this request to ChatGPT after attaching the ZIP

Please inspect this Stage G teacher-pilot review bundle using `rubric.md`.
Extract the archive and read `cases.json`; actually open the mapped image files when
checking visual claims. Review the businesses in their listed order, in batches of
five if necessary. State exactly which business IDs, conditions, reviews and images
you inspected. If you cannot extract the ZIP or inspect an image, report that item
as NOT REVIEWED; do not infer visual correctness from the caption or teacher verdict.

Assess review-consensus entailment, mixed/negative opinion preservation, omissions,
overstatement of frequency, caption support and actual image grounding. Separate
factual errors from correct but generic/low-information text and awkward writing.
Check both verified claims entering targets and rejected/uncertain claims omitted
from targets. Cite business/condition/claim index, REV marker/source span or
caption_id/photo_id and the exact relevant quotation for every concern.

Return concrete concerns, uncertain cases, strengths and suggested protocol changes
FOR DISCUSSION ONLY. Use the optional response template if useful. Do not rewrite
targets, change labels/thresholds/K/budgets, fabricate negatives, or claim that your
review is human verification. The researcher inspects evidence and confirms final
decisions. Please explain the findings in Vietnamese.
"""

RUBRIC = """# Active Stage G constraints and review rubric

The exact active review/visual prompts are included in `protocol/` and remain
authoritative. This document organizes inspection questions; it creates no new
scientific threshold, quality score, label or production rule.

## Review consensus

- Judge against the complete SELECTED reviews supplied for this business, not
  unseen reviews from the business's entire original population.
- Consensus must come only from reviews, never photos/captions or outside knowledge.
- Claims must preserve recurring positive/negative opinions and disagreement.
  Mixed claims require support for both sides. Do not force negative claims.
- The current pilot guard requires at least two distinct source reviews for each
  recurring claim. Valid IDs alone do not prove semantic entailment. Inspect the
  cited text AND all selected reviews for omissions/overstated consensus.
- Do not infer population percentages, aggregate ratings or unanimous opinion.
- The identical `[REVIEW_CONSENSUS]` is reused across all three conditions.

## Visual evidence

- A claim must be recoverable from the EXACT provided caption for that condition
  AND supported by its mapped original photo. The photo only verifies that claim.
- A photo-only detail absent from the caption must not enter the target, even if true.
- Incorrect counts, attributes, OCR, objects or spatial relations must not be
  silently repaired into a different photo-only fact.
- Teacher verdict `verified` means model judgment, NOT human verification.
  Unsupported claims should be rejected; unverifiable details remain uncertain.
- Only teacher-verified claims enter `[VISUAL_EVIDENCE]`; rejected/uncertain claims
  remain available for audit. Inspect these verdicts independently.
- Generic but factually image-compatible text is not automatically inconsistent.
  Correct-but-simple claims can be flagged as low information separately from error.
- Do not infer taste, value, service, popularity or business-wide prevalence from a photo.
- Unfiltered/Filtered use their own exact captions/photos. Their visual targets may differ.
- Zero-Filtered-caption businesses retain an empty visual header. No fabricated
  caption/evidence and no automatic business exclusion are required.

## Source/citation implementation already applied

- D-035: immutable source-span IDs resolve exact original review quotations/offsets.
- D-036: identical verified statements across different photos appear once in target
  text, with every supporting photo/caption reference retained. This does not filter
  the Unfiltered input or merge semantically different claims.
- D-037: strict agreeing business/REV marker/span IDs identify canonical original
  review IDs. Three redundant model-reported opaque-ID disagreements across two
  businesses are preserved in raw model output and the audit. Do not confuse this
  documented copying discrepancy with proof of unsupported claim content.
- Claim text/polarity, selected spans, full reviews and raw teacher outputs remain
  unchanged. Wrong marker/span associations and nonexistent spans remain invalid.

## Provenance and scope

The user remains the primary human reviewer. ChatGPT review is only an optional
comparison/adjudication aid after the user's own inspection. Any accepted change
requires an explicit user-confirmed human decision; AI suggestions never become
human labels automatically. No edits are imported by this export.

One existing user-confirmed visual claim is included unchanged: business
EzJI5x25U5GgbYBWhu5ZQQ, Unfiltered IMG_CAP_5, photo HVkUbl7niy2hIADyPh17MA,
`The cup has a straw.` The earlier AI visibility flag was resolved after the user's
enlarged-image inspection. This is ONE claim, not verification of the entire business.
All 20 whole-business targets remain human-unverified. Do not manufacture negative
examples or relabel uncertainty to meet class balance.

This fixed pilot deliberately includes two zero-Filtered cases; it is not a random
estimate of all 2,200 businesses and is separate from the production random manual
inspection subset and official 30-business human evaluation. Review all exported
cases or disclose which are still unreviewed. No test tuning, target regeneration,
threshold/K/source-budget changes, teacher calls or LED training occur here.

The pilot target-token range is 98–382. The value512is currently a planning diagnostic,
not a frozen target cap. API max_output_tokens8192includes structured evidence and
reasoning; it is not the LED target length or source limit. Production length/retry/
model-version provenance/manual-inspection policies still await researcher decisions.
"""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def page(cases):
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Stage G pilot review</title>",
             "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}"
             "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:14px}"
             "img{max-width:100%;max-height:650px}details{margin:16px 0}summary{cursor:pointer}"
             "article{border-top:2px solid #777;margin-top:30px}figure{margin:20px 0}"
             "code{overflow-wrap:anywhere}</style><h1>Stage G: 20-business pilot</h1>",
             "<p>60 provisional targets. ChatGPT/teacher judgments are AI suggestions; whole-business "
             "human review is pending. Read <a href='rubric.md'>rubric.md</a> and "
             "<a href='README.md'>README.md</a>. This page performs no API calls or label writes.</p>"]
    e = html.escape
    for case in cases:
        bid = case['business_id']
        parts += ['<article><h2>' + e(case['split'] + ' / ' + bid) + '</h2>',
                  "<p><a href='businesses/" + e(bid) + ".json'>Complete structured evidence</a></p>"]
        for condition, target in case['targets'].items():
            parts += ['<h3>' + e(condition) + ' target</h3><pre>' + e(target) + '</pre>']
        parts += ['<details><summary>Complete selected reviews and original IDs/order</summary>']
        for review in case['reviews']:
            parts += ['<h4>' + e(review['review_marker'] + ' / ' + review['review_id']) + '</h4>',
                      '<p>' + e(f"Stars: {review['stars']} | Date: {review['date']}") + '</p>',
                      '<pre>' + e(review['text']) + '</pre>']
        parts += ['</details><details><summary>Review claims with source quotes</summary><pre>',
                  e(json.dumps(case['review_consensus_claims'], ensure_ascii=False, indent=2)), '</pre></details>']
        for condition, evidence in case['caption_conditions'].items():
            parts += ['<details><summary>' + e(condition) + ' captions, photos and teacher assessments</summary>']
            if not evidence['captions']:
                parts += ['<p>0 captions; intentionally empty visual evidence.</p>']
            for caption, assessment in zip(evidence['captions'], evidence['teacher_assessments']):
                parts += ['<figure><h4>' + e(caption['caption_id'] + ' / ' + caption['photo_id']) + '</h4>',
                          '<p>' + e(caption['caption']) + '</p>',
                          "<img loading='lazy' src='" + e(caption['image_file']) + "' alt='Original evidence photo'>",
                          '<pre>' + e(json.dumps(assessment, ensure_ascii=False, indent=2)) + '</pre></figure>']
            parts += ['</details>']
        if case['human_confirmations']:
            parts += ['<details><summary>Existing user-confirmed claim (unchanged)</summary><pre>',
                      e(json.dumps(case['human_confirmations'], ensure_ascii=False, indent=2)), '</pre></details>']
        parts += ['</article>']
    return ''.join(parts) + '</html>'


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Existing export preserved; use another version/path instead of overwriting')
    cfg = read(ROOT / CONFIG)
    directory = ROOT / cfg['output_directory']
    execution = directory / 'execution_v1'
    audit = read(execution / 'pilot_integrity_and_tokens_report.json')
    prepared = read(directory / 'preparation_report.json')
    businesses, jobs = read(directory / 'pilot_businesses.json'), read(directory / 'pilot_jobs.json')
    feedback = read(execution / 'human_claim_verifications.json')
    assert audit['businesses_complete'] == 20 and audit['condition_target_texts'] == 60
    protected = dict(audit['artifact_sha256'])
    protected.update(prepared['input_sha256'])
    for path in [directory / 'pilot_businesses.json', directory / 'pilot_jobs.json',
                 execution / 'completion_20_businesses.json', execution / 'pilot_integrity_and_tokens_report.json']:
        protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    for relative, sha in protected.items():
        assert file_sha(ROOT / relative) == sha, 'Original artifact changed: ' + relative
    images = {}
    cases = []
    for business in businesses:
        bid = business['business_id']
        assert business['split'] in {'train', 'validation'}
        target = read(execution / 'provisional_targets' / (bid + '.json'))
        these_jobs = [j for j in jobs if j['business_id'] == bid]
        review_job = next(j for j in these_jobs if j['task'] == 'review_consensus')
        review_cp = read(execution / 'checkpoints' / (review_job['job_id'] + '.json'))
        reviews = [dict(r, evidence_spans=evidence_spans(r)) for r in review_job['evidence']['reviews']]
        conditions = {}
        for condition in ['unfiltered', 'filtered']:
            job = next(j for j in these_jobs if j['condition'] == condition)
            cp = read(execution / 'checkpoints' / (job['job_id'] + '.json'))
            captions = []
            for caption, photo in zip(job['evidence']['captions'], job['images']):
                assert caption['photo_id'] == photo['photo_id']
                image_file = 'images/' + photo['photo_id'] + '.jpg'
                assert file_sha(ROOT / photo['relative_path']) == photo['raw_sha256']
                protected[photo['relative_path']] = photo['raw_sha256']
                if image_file in images:
                    assert images[image_file]['raw_sha256'] == photo['raw_sha256']
                images[image_file] = photo
                captions.append(dict(caption, image_file=image_file,
                                     original_image_sha256=photo['raw_sha256']))
            conditions[condition] = {'job_id': job['job_id'], 'captions': captions,
                'teacher_assessments': cp['output']['assessments'],
                'serialized_visual_claims_with_all_evidence': target['visual_target_claims_with_all_evidence'][condition],
                'empty_caption_condition': not captions, 'whole_business_human_verified': False}
        cases.append({'business_id': bid, 'split': business['split'],
            'zero_filtered_edge_case': business['zero_filtered_edge_case'],
            'whole_business_human_verified': target['human_verified'],
            'targets': target['targets'], 'reviews': reviews,
            'review_consensus_claims': review_cp['resolved_output']['claims'],
            'review_teacher_raw_output': review_cp['output'],
            'review_source_resolution': review_cp['evidence_resolution'],
            'review_citation_metadata_diagnostics': review_cp['citation_metadata_diagnostics'],
            'caption_conditions': conditions,
            'human_confirmations': [r for r in feedback['records'] if r['business_id'] == bid]})
    assert len(cases) == 20 and len(images) == 128
    data = {'schema_version': 'stage_g_portable_second_review_v1', 'export_only': True,
            'requested_model': cfg['requested_model'], 'reported_model': 'gpt-6.1-sol',
            'review_role': 'optional_AI_comparison_not_human_ground_truth',
            'production_method_frozen': False, 'businesses': cases}
    files = {'README.md': README.encode('utf-8'), 'rubric.md': RUBRIC.encode('utf-8'),
             'cases.json': json_bytes(data), 'index.html': page(cases).encode('utf-8')}
    for case in cases:
        files['businesses/' + case['business_id'] + '.json'] = json_bytes(case)
    originals = {
        'protocol/active_pilot_config.json': ROOT / CONFIG,
        'protocol/review_prompt.txt': ROOT / cfg['prompts']['review_consensus'],
        'protocol/visual_prompt.txt': ROOT / cfg['prompts']['visual_evidence'],
        'protocol/output_schema.json': ROOT / cfg['schema'],
        'protocol/active_stage_g_spec.md': ROOT / 'specs/007_teacher_target_construction.md',
        'provenance/THESIS_PROGRESS.md': ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md',
        'provenance/pilot_integrity_and_tokens_report.json': execution / 'pilot_integrity_and_tokens_report.json',
        'provenance/completion_20_businesses.json': execution / 'completion_20_businesses.json',
        'provenance/human_claim_verifications.json': execution / 'human_claim_verifications.json'}
    for relative, source in originals.items():
        files[relative] = source.read_bytes()
        protected[source.relative_to(ROOT).as_posix()] = file_sha(source)
    files['review_response_template.json'] = json_bytes({
        'role': 'AI_second_opinion_only_not_human_final_label',
        'instruction': 'Fill only after actual evidence inspection. No automatic import/label changes.',
        'business_reviews': [{'business_id': c['business_id'], 'split': c['split'],
            'actually_inspected': False, 'conditions_reviewed': [], 'photo_ids_opened': [],
            'review_consensus_concerns': [], 'visual_concerns': [],
            'correct_but_generic_or_style_notes': [], 'not_reviewed_items': [],
            'chatgpt_second_review_suggestion': None, 'human_final_decision': None,
            'human_verified': False} for c in cases]})
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    checksums.update({name: info['raw_sha256'] for name, info in images.items()})
    manifest = {'status': 'EXPORT_ONLY_COMPLETE_PILOT', 'businesses': 20,
        'businesses_by_split': dict(Counter(c['split'] for c in cases)), 'target_texts': 60,
        'selected_reviews': sum(len(c['reviews']) for c in cases), 'unique_original_images': len(images),
        'images_resized': False, 'human_confirmed_claims': feedback['verified_claim_count'],
        'whole_businesses_human_verified': 0, 'held_out_businesses': 0,
        'scientific_outputs_changed': False, 'teacher_API_calls': 0,
        'file_sha256': checksums}
    files['export_manifest.json'] = json_bytes(manifest)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in files.items():
            archive.writestr(PREFIX + name, content)
        for name, info in images.items():
            archive.write(ROOT / info['relative_path'], PREFIX + name)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        expected_names = {PREFIX + name for name in files} | {PREFIX + name for name in images}
        assert set(archive.namelist()) == expected_names
        for name, sha in checksums.items():
            assert hashlib.sha256(archive.read(PREFIX + name)).hexdigest() == sha, name
        exported = json.loads(archive.read(PREFIX + 'cases.json'))
        assert exported == data
        for case, business in zip(exported['businesses'], businesses):
            bid = case['business_id']
            assert bid == business['business_id']
            original_target = read(execution / 'provisional_targets' / (bid + '.json'))
            assert case['targets'] == original_target['targets']
            review_job = next(j for j in jobs if j['business_id'] == bid and j['task'] == 'review_consensus')
            assert [{k:v for k,v in r.items() if k != 'evidence_spans'} for r in case['reviews']] == review_job['evidence']['reviews']
            for condition, item in case['caption_conditions'].items():
                job = next(j for j in jobs if j['business_id'] == bid and j['condition'] == condition)
                assert [{k:v for k,v in c.items() if k not in {'image_file','original_image_sha256'}}
                        for c in item['captions']] == job['evidence']['captions']
                for caption in item['captions']:
                    assert PREFIX + caption['image_file'] in expected_names
        assert [r for c in cases for r in c['human_confirmations']] == feedback['records']
        assert archive.read(PREFIX + 'provenance/human_claim_verifications.json') == (execution / 'human_claim_verifications.json').read_bytes()
    for relative, sha in protected.items():
        assert file_sha(ROOT / relative) == sha, 'Source changed during export: ' + relative
    report = {k:v for k,v in manifest.items() if k != 'file_sha256'}
    report.update({'status': 'EXPORT_INTEGRITY_PASS', 'zip_path': OUTPUT.relative_to(ROOT).as_posix(),
        'zip_bytes': OUTPUT.stat().st_size, 'zip_MiB': round(OUTPUT.stat().st_size / 1024**2, 3),
        'zip_sha256': file_sha(OUTPUT), 'archive_members': len(expected_names),
        'original_artifacts_rechecked': len(protected), 'every_image_present_and_exact': True,
        'review_text_ids_order_unchanged': True, 'caption_text_ids_order_unchanged': True,
        'target_texts_unchanged': True, 'human_confirmation_records_byte_identical': True,
        'original_artifact_sha256': protected})
    REPORT.write_bytes(json_bytes(report))
    print(json.dumps({k:v for k,v in report.items() if k != 'original_artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
