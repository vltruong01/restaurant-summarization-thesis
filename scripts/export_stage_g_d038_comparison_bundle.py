"""Read-only portable D-038 comparison export and offline viewer; never calls an API."""
from pathlib import Path
from collections import Counter
import base64
import copy
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'outputs/stage_g/teacher_pilot_v4_1_grounding_d038_v1'
EXE = NEW / 'execution_v2_quote_recovery'
BASE_ZIP = ROOT / 'outputs/stage_g/chatgpt_stage_g_pilot_review_bundle.zip'
ZIP = ROOT / 'outputs/stage_g/chatgpt_stage_g_d038_comparison_bundle.zip'
PREFIX = 'chatgpt_stage_g_d038_comparison_bundle/'
VIEW = ROOT / 'outputs/stage_g/pilot_comparison_ui_d038_v1'
ASSETS = ROOT / 'scripts/stage_g_comparison_ui'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(content):
    return hashlib.sha256(content).hexdigest()


def jb(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


README = """# Stage G D-038: independent second review

Export-only comparison of the ORIGINAL and REVISED teacher pilots on the SAME
20 businesses (10 train, 10 validation). No held-out test data. The original has
60 provisional target texts; the revised has 19 accepted three-condition sets /
57 texts and ONE blocked business. Structural acceptance is not factual accuracy.

Start with `cases.json`, `reports/comparison_report.md`, `rubric.md`, then open
`images/<photo_id>.jpg`. `businesses/<business_id>.json` contains manageable batches.
`index.html` is a portable comparison page: extract the WHOLE ZIP before opening it.
No image was resized. Exact reviews/captions, teacher outputs and verdicts are retained.
The blocked business has successful components and a separately marked FAILED raw
parsed visual response; it has NO accepted revised targets. Do not promote that response.

## Request to paste into ChatGPT

Please review both Stage G pilots independently using the attached actual images
and complete selected reviews. Read rubric.md and cases.json. Inspect all 20 businesses,
in batches if needed, and list the exact businesses/photos/reviews actually inspected.
If you cannot extract or open an image, say NOT REVIEWED; do not guess from captions.
First compare targets and evidence WITHOUT relying on the existing AI findings;
then use reports/ai_semantic_comparison_findings.json as an optional comparison.
Separate factual errors, unsupported verified judgments, genuine ambiguities,
review overgeneralization/coverage omissions, verdict contradictions, and correct
but simple content. Cite business_id, condition, exact claim, REV/source-span or
IMG_CAP/PHOTO/photo_id and relevant source quotation. Flag improvements AND regressions.
Different claim segmentation changes denominators: raw counts are not error rates.
The failed caption-quote task is an execution/structural issue, separate from factual quality.
Return findings and proposed next steps in Vietnamese FOR RESEARCHER DISCUSSION ONLY.
Do not change labels, targets, K, thresholds, budgets or generate new targets.
Your review is an AI second opinion; the user remains the primary human decision maker.

All target sets remain provisional. No full teacher production protocol is frozen.
The existing one-claim user confirmation is retained only under its original scope;
it does not confer human verification on the revised pilot or other claims.
"""

RUBRIC = """# Active D-038 interpretation / evidence rules

Exact old/new prompts, schema and active Stage G spec are in protocol/. This rubric
summarizes approved rules; it does not create or freeze methodology.

## REVIEW_CONSENSUS
- Selected reviews ONLY: no photo/caption information or external knowledge.
- Preserve positive and negative opinions and genuine disagreements without forcing negatives.
- Do not inflate customer frequency, unanimity or popularity. A valid citation ID is not
  proof of entailment: read source quotes AND complete selected reviews for omissions.
- Each recurring claim still requires support from at least two distinct review sources.
- The review consensus is identical across the three controlled conditions.

## VISUAL_EVIDENCE
- Each claim must be supported by the exact caption supplied to LED AND its mapped photo.
- No photo-only details absent from the caption. No invented ingredients, counts,
  prices, identities, actions or spatial relations.
- Normal plausible descriptions (including qualitative size and ordinary pizza/food
  categories) may be accepted without an unnecessarily strict measurement requirement.
- Generic/simple but factually compatible content is not automatically inconsistent.
- A specific unsupported visual assertion is rejected, or uncertain when genuinely ambiguous.
  A visible fork does not establish a person/hand holding it when no such actor is visible.
- Only supported teacher-verified visual claims enter provisional targets. Inspect
  rejected/uncertain claims too for excessive strictness. Teacher verified is NOT human verified.
- Zero filtered captions retain the agreed empty visual header; never fabricate evidence.

## Consistency and provenance
- Identical photo hash + exact caption + identical claim should normally get the same
  verdict in Unfiltered and Filtered. Flag unexplained contradictions; do not reconcile them.
- Different captions/claims can legitimately differ. Paraphrases and conjunctions need
  explicit semantic inspection, not an assumption of equivalence.
- Existing AI findings are second opinions, not human ground truth or measured error rates.
- ChatGPT review is an optional comparison/adjudication aid after the user's own inspection.
  Final annotation remains a user-confirmed human decision. No automatic label import.
- Preserve existing original straw confirmation unchanged and scoped to ONE original claim.
- Do not manufacture negatives, convert uncertainty to negative, or retune Stage F.
- One blocked revised business is still included. Its raw failed response is diagnostic,
  never an accepted target. Exact quotes and validators were not manually repaired.
- No test data, API calls, threshold/K selection or LED training occur in this export.
- 512 target tokens remains diagnostic, not a frozen target cap; source 5000/review4000/
  caption750 and all Stage F settings remain unchanged.
"""


def render(data):
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    page = (ASSETS / 'index.html').read_text(encoding='utf-8')
    for marker, value in [('__DATA__', encoded), ('__STYLES__', (ASSETS / 'styles.css').read_text(encoding='utf-8')),
                          ('__APP__', (ASSETS / 'app.js').read_text(encoding='utf-8'))]:
        page = page.replace(marker, value)
    assert not any(m in page for m in ['__DATA__', '__STYLES__', '__APP__'])
    restored = json.loads(page.split('<script id="review-data" type="application/json">', 1)[1].split('</script>', 1)[0])
    assert restored == data
    return page.encode('utf-8')


def main(rebuild=False):
    if ZIP.exists() or (VIEW / 'dist/index.html').exists():
        if not rebuild:
            raise FileExistsError('Preserve existing export/viewer; use a new version to rebuild')
        previous = read(ZIP.with_name(ZIP.stem + '_report.json'))
        assert sha(ZIP.read_bytes()) == previous['zip_sha256']
        assert sha((VIEW / 'dist/index.html').read_bytes()) == previous['viewer_sha256']
    protected = {}

    def original(path):
        content = path.read_bytes()
        protected[path.relative_to(ROOT).as_posix()] = sha(content)
        return content

    baseline_zip = original(BASE_ZIP)
    with zipfile.ZipFile(BASE_ZIP) as z:
        prefix = 'chatgpt_stage_g_pilot_review_bundle/'
        baseline = json.loads(z.read(prefix + 'cases.json'))
        manifest = json.loads(z.read(prefix + 'export_manifest.json'))
        for name, expected in manifest['file_sha256'].items():
            assert sha(z.read(prefix + name)) == expected, name
        images = {n[len(prefix):]: z.read(n) for n in z.namelist() if n.startswith(prefix + 'images/')}
        prior_confirmations = z.read(prefix + 'provenance/human_claim_verifications.json')
        old_protocol = {name.split('/')[-1]: z.read(prefix + name) for name in
                        ['protocol/active_pilot_config.json', 'protocol/review_prompt.txt',
                         'protocol/visual_prompt.txt', 'protocol/output_schema.json']}
    jobs = json.loads(original(NEW / 'pilot_jobs.json'))
    completion = json.loads(original(EXE / 'completion_20_attempted_businesses.json'))
    findings = json.loads(original(NEW / 'ai_semantic_comparison_findings.json'))
    old_jobs = read(ROOT / 'outputs/stage_g/teacher_pilot_v3_canonical_review_ids_v1/pilot_jobs.json')
    strip = lambda j: {k: v for k, v in j.items() if k != 'prompt_sha256'}
    assert [strip(j) for j in jobs] == [strip(j) for j in old_jobs]
    cases = []
    failed = []
    for before in baseline['businesses']:
        bid = before['business_id']
        assert before['split'] in ['train', 'validation']
        selected_jobs = [j for j in jobs if j['business_id'] == bid]
        review_job = next(j for j in selected_jobs if j['task'] == 'review_consensus')
        assert review_job['evidence']['reviews'] == [{k: v for k, v in r.items() if k != 'evidence_spans'} for r in before['reviews']]
        review_cp = json.loads(original(EXE / 'checkpoints' / (review_job['job_id'] + '.json')))
        target_path = EXE / 'provisional_targets' / (bid + '.json')
        target = json.loads(original(target_path)) if target_path.exists() else None
        conditions = {}
        for condition in ['unfiltered', 'filtered']:
            job = next(j for j in selected_jobs if j['condition'] == condition)
            cp = json.loads(original(EXE / 'checkpoints' / (job['job_id'] + '.json')))
            captions = copy.deepcopy(before['caption_conditions'][condition]['captions'])
            assert job['evidence']['captions'] == [{k: v for k, v in c.items() if k not in ['image_file', 'original_image_sha256']} for c in captions]
            for caption, image in zip(captions, job['images']):
                assert caption['photo_id'] == image['photo_id']
                assert sha(images[caption['image_file']]) == image['raw_sha256']
                assert sha(original(ROOT / image['relative_path'])) == image['raw_sha256']
            accepted = 'output' in cp
            raw_diagnostic_output = None
            if not accepted:
                response = json.loads(original(EXE / 'raw_responses' / (job['job_id'] + '.json')))
                text = ''.join(c['text'] for o in response['output'] for c in o.get('content', []) if c.get('type') == 'output_text')
                raw_diagnostic_output = json.loads(text)
                failed.append(job['job_id'])
            conditions[condition] = {'job_id': job['job_id'], 'captions': captions,
                'task_status': cp['status'], 'accepted_component': accepted,
                'teacher_assessments': cp.get('output', {}).get('assessments', []),
                'failed_raw_output_diagnostic_only': raw_diagnostic_output,
                'failure_reason': cp.get('local_validation_error'),
                'serialized_visual_claims_with_all_evidence': target['visual_target_claims_with_all_evidence'][condition] if target else None,
                'empty_caption_condition': not captions, 'whole_business_human_verified': False}
        original_version = {k: copy.deepcopy(before[k]) for k in ['targets', 'review_consensus_claims',
            'review_teacher_raw_output', 'review_source_resolution', 'review_citation_metadata_diagnostics', 'caption_conditions']}
        revised = {'complete_target_set_accepted': target is not None, 'targets': target['targets'] if target else None,
            'review_consensus_claims': review_cp['resolved_output']['claims'], 'review_teacher_raw_output': review_cp['output'],
            'review_source_resolution': review_cp['evidence_resolution'],
            'review_citation_metadata_diagnostics': review_cp['citation_metadata_diagnostics'],
            'caption_conditions': conditions, 'whole_business_human_verified': False}
        cases.append({'case_id': bid, 'business_id': bid, 'split': before['split'],
            'zero_filtered_edge_case': before['zero_filtered_edge_case'], 'reviews': copy.deepcopy(before['reviews']),
            'existing_original_human_confirmations': copy.deepcopy(before['human_confirmations']),
            'whole_business_human_verified': False, 'versions': {'original': original_version, 'revised': revised},
            'ai_second_opinion_findings': [f for f in findings['findings'] if f['business_id'] == bid]})
    assert len(cases) == 20 and len(images) == 128 and len(failed) == 1
    assert sum(c['versions']['revised']['complete_target_set_accepted'] for c in cases) == 19
    metadata = {}
    wanted = {c['business_id'] for c in cases}
    for split in ['train', 'validation']:
        path = ROOT / 'outputs/stage_f/production_caption_construction_d033_colab_v1/manifests' / split / 'reviews_only_sources.jsonl'
        for line in original(path).decode('utf-8').splitlines():
            row = json.loads(line)
            if row['business_id'] in wanted:
                head = row['source'].split('\n\n[REVIEWS]', 1)[0]
                metadata[row['business_id']] = dict(l.split('=', 1) for l in head.splitlines() if '=' in l)
    assert set(metadata) == wanted
    data = {'schema_version': 'stage_g_d038_portable_comparison_v1', 'export_only': True,
        'review_role': 'optional_AI_second_opinion_not_human_ground_truth',
        'production_method_frozen': False, 'completion': completion,
        'metadata': metadata, 'businesses': cases, 'images': {n[7:-4]: n for n in images},
        'existing_human_confirmations_apply_only_to_original_scoped_claim': True}
    files = {'README.md': README.encode('utf-8'), 'rubric.md': RUBRIC.encode('utf-8'),
             'cases.json': jb(data), 'index.html': render(data),
             'provenance/original_human_claim_verifications.json': prior_confirmations}
    for case in cases:
        files['businesses/' + case['business_id'] + '.json'] = jb(case)
    for name, content in old_protocol.items():
        files['protocol/original/' + name] = content
    cfg = json.loads(original(ROOT / 'configs/stage_g_teacher_pilot_v4_1_grounding.json'))
    copies = {'protocol/revised/active_pilot_config.json': ROOT / 'configs/stage_g_teacher_pilot_v4_1_grounding.json',
        'protocol/revised/review_prompt.txt': ROOT / cfg['prompts']['review_consensus'],
        'protocol/revised/visual_prompt.txt': ROOT / cfg['prompts']['visual_evidence'],
        'protocol/revised/output_schema.json': ROOT / cfg['schema'],
        'protocol/active_stage_g_spec.md': ROOT / 'specs/007_teacher_target_construction.md',
        'provenance/THESIS_PROGRESS.md': ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md',
        'provenance/completion.json': EXE / 'completion_20_attempted_businesses.json',
        'provenance/valid_targets_integrity_and_tokens_report.json': EXE / 'valid_targets_integrity_and_tokens_report.json',
        'provenance/rerun_preflight_and_preservation.json': NEW / 'rerun_preflight_and_preservation.json'}
    for name in ['comparison_report.json', 'machine_comparison.json', 'ai_semantic_comparison_findings.json',
                 'ai_photo_inspection_notes.json', 'usage_all_attempts.json']:
        copies['reports/' + name] = NEW / name
    for name, path in copies.items():
        files[name] = original(path)
    report_md = original(NEW / 'comparison_report.md').decode('utf-8')
    # Only the exported report's link destinations are adapted, never evidence/claim text.
    files['reports/comparison_report.md'] = report_md.replace('../../../data/processed/yelp_2300_v1/photos/', '../images/').encode('utf-8')
    files['review_response_template.json'] = jb({'role': 'AI_second_opinion_only',
        'business_reviews': [{'business_id': c['business_id'], 'actually_inspected': False,
            'photo_ids_opened': [], 'review_ids_read': [], 'improvements': [], 'regressions': [],
            'ambiguous_cases': [], 'correct_but_simple_content': [], 'not_reviewed_items': [],
            'human_final_decision': None, 'human_verified': False} for c in cases]})
    # No credentials or absolute local image paths should appear in structured portable cases.
    def check_portable(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {'image_file', 'relative_path', 'path'} and isinstance(item, str):
                    assert not re.match(r'(?i)^[a-z]:[/\\]', item)
                check_portable(item)
        elif isinstance(value, list):
            for item in value:
                check_portable(item)
    check_portable(data)
    for name, content in files.items():
        assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}', content), 'Possible credential in ' + name
    checksums = {n: sha(b) for n, b in {**files, **images}.items()}
    manifest = {'status': 'EXPORT_ONLY_D038_COMPARISON', 'businesses': 20,
        'businesses_by_split': dict(Counter(c['split'] for c in cases)), 'original_target_texts': 60,
        'revised_target_texts': 57, 'blocked_businesses': 1, 'task_slots': 60,
        'selected_reviews': sum(len(c['reviews']) for c in cases), 'original_images': len(images),
        'images_resized': False, 'AI_findings': len(findings['findings']),
        'failed_response_promoted': False, 'teacher_API_calls': 0, 'scientific_outputs_changed': False,
        'file_sha256': checksums}
    files['export_manifest.json'] = jb(manifest)
    with zipfile.ZipFile(ZIP, 'w' if rebuild else 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, content in {**files, **images}.items():
            z.writestr(PREFIX + name, content)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name, expected in checksums.items():
            assert sha(z.read(PREFIX + name)) == expected, name
        restored = json.loads(z.read(PREFIX + 'cases.json'))
        assert restored == data
        assert z.read(PREFIX + 'provenance/original_human_claim_verifications.json') == prior_confirmations
        for c, before in zip(restored['businesses'], baseline['businesses']):
            assert c['business_id'] == before['business_id'] and c['reviews'] == before['reviews']
            assert c['versions']['original']['targets'] == before['targets']
            assert c['existing_original_human_confirmations'] == before['human_confirmations']
            for version in c['versions'].values():
                for cond in version['caption_conditions'].values():
                    for caption in cond['captions']:
                        assert sha(z.read(PREFIX + caption['image_file'])) == caption['original_image_sha256']
    local_data = copy.deepcopy(data)
    local_data['images'] = {n[7:-4]: 'data:image/jpeg;base64,' + base64.b64encode(b).decode('ascii') for n, b in images.items()}
    standalone = render(local_data)
    page = VIEW / 'dist/index.html'
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(standalone)
    (VIEW / 'OPEN_STAGE_G_COMPARISON.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n')
    (VIEW / 'HOW_TO_OPEN.txt').write_text('Open OPEN_STAGE_G_COMPARISON.cmd or dist/index.html. Standalone offline; no API/server. Notes are browser-local personal notes, not scientific annotations.\n', encoding='utf-8')
    for relative, expected in protected.items():
        assert sha((ROOT / relative).read_bytes()) == expected, 'Scientific input changed: ' + relative
    assert sha(BASE_ZIP.read_bytes()) == sha(baseline_zip)
    report = dict(manifest, integrity_checks='PASS', original_and_revised_targets_exact=True,
        caption_review_ID_text_order_unchanged=True, human_confirmations_unchanged=True,
        all_case_images_present=True, protected_files_checked=len(protected),
        protected_input_sha256=protected, zip_path=ZIP.relative_to(ROOT).as_posix(),
        zip_bytes=ZIP.stat().st_size, zip_sha256=sha(ZIP.read_bytes()),
        viewer_path=page.relative_to(ROOT).as_posix(), viewer_bytes=len(standalone),
        viewer_sha256=sha(standalone), browser_visual_QA='not_yet_performed')
    ZIP.with_name(ZIP.stem + '_report.json').write_bytes(jb(report))
    (VIEW / 'build_report.json').write_bytes(jb(report))
    print(json.dumps({k: report[k] for k in ['integrity_checks', 'businesses', 'original_target_texts',
          'revised_target_texts', 'blocked_businesses', 'original_images', 'zip_path', 'zip_bytes', 'viewer_path', 'protected_files_checked']}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-export-viewer-only', action='store_true',
                        help='Refresh only this hash-verified derived export/viewer after UI corrections')
    main(parser.parse_args().rebuild_export_viewer_only)
