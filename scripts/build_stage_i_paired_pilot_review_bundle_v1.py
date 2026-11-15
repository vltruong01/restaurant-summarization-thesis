"""Export the prepared paired pilot and existing validation evidence; no model/API calls."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
CLOUD = ROOT / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
NAME = 'stage_i_paired_lora_pilot_preparation_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not DEST.exists() and not ZIP.exists(), 'Preserve existing exports'
    prepared = read(PREP / 'PREPARATION_MANIFEST.json')
    assert sha(ROOT / 'configs/stage_i_paired_lora_pilot_v1.json') == prepared['config_sha256']
    for relative, digest in prepared['files'].items():
        assert sha(PREP / relative) == digest, relative
    for group in ('code', 'upstream'):
        for relative, digest in prepared[group].items():
            assert sha(ROOT / relative) == digest, relative

    mapping = {}

    def include(relative, source):
        name = PurePosixPath(relative)
        assert not name.is_absolute() and '..' not in name.parts and '\\' not in relative
        assert relative not in mapping and source.is_file(), relative
        mapping[relative] = source

    for name in prepared['files']:
        include('preparation/' + name, PREP / name)
    for name in ['PREPARATION_MANIFEST.json', 'approval_example_NOT_APPROVED.json',
                 'READINESS_REPORT.md', 'OFFLINE_VERIFICATION_REPORT.json',
                 'offline_tests_release.log', 'offline_runner_smoke.log']:
        include('preparation/' + name, PREP / name)
    for relative in prepared['code']:
        include('code/' + Path(relative).name, ROOT / relative)
    include('code/' + Path(__file__).name, Path(__file__))
    include('configs/stage_i_paired_lora_pilot_v1.json', ROOT / 'configs/stage_i_paired_lora_pilot_v1.json')
    include('configs/historical_full_training.json', CLOUD / 'configs/stage_i_lora_led_training_approved_v2.json')
    include('configs/frozen_stage_h_manifest.json', CLOUD / 'outputs/stage_h/led_dataset_v1/manifest.json')
    include('configs/frozen_source_template.json', ROOT / 'configs/led_source_template_d032.json')
    include('configs/pinned_LED_config.json', CLOUD / 'assets/models/led-base-16384/config.json')
    include('configs/pinned_LED_config_manifest.json', CLOUD / 'assets/models/led-base-16384.config_manifest.json')
    tokenizer = CLOUD / 'assets/tokenizers/led-base-16384.manifest.json'
    include('assets/tokenizers/led-base-16384.manifest.json', tokenizer)
    tm = read(tokenizer)
    for name, digest in tm['sha256'].items():
        source = CLOUD / tm['relative_directory'] / name
        assert sha(source) == digest
        include('assets/tokenizers/led-base-16384/' + name, source)
    for relative in ['DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md',
                     'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                     'specs/009d_paired_lora_pilot_preparation.md']:
        include('documentation/' + Path(relative).name, ROOT / relative)
    audit = ROOT / 'outputs/stage_i/training_failure_audit_v1'
    for name in ['STAGE_I_TRAINING_FAILURE_AUDIT.md', 'STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md',
                 'architecture_audit.json', 'optimization_audit.json', 'dataset_target_audit.json',
                 'proposed_pilot_manifest.json', 'AUDIT_MANIFEST.json']:
        include('historical/D048/' + name, audit / name)
    include('historical/fixed_validation_subset.json', ROOT / 'outputs/stage_i/decoding_diagnostic_v1/subset_manifest.json')
    include('historical/decoding_configurations.json', ROOT / 'outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json')
    for label, source in [
        ('full_training_baseline', CLOUD / 'outputs/stage_i/runs_v3/filtered/validation_predictions.jsonl'),
        ('greedy_diagnostic_A', ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2/filtered/predictions.jsonl'),
        ('no_repeat_diagnostic_B', ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1/filtered/predictions.jsonl'),
    ]:
        include('historical/predictions/' + label + '.jsonl', source)
    b_eval = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1'
    for name in ['AI_QUALITATIVE_REVIEW.json', 'HEADER_CONSTRAINT_AUDIT.json', 'ngram_lexical_summary.json']:
        include('historical/diagnostic_B/' + name, b_eval / name)

    cases = rows(PREP / 'evaluation_inputs.jsonl')
    sample = read(PREP / 'pilot_manifest.json')
    old_sample = read(audit / 'proposed_pilot_manifest.json')
    assert sample['training_business_ids'] == old_sample['training_business_ids']
    assert len(cases) == 24 and len({c['business_id'] for c in cases}) == 24
    assert all(c['split'] == 'validation' and c['condition'] == 'filtered' for c in cases)
    photo_ids = sorted({photo for case in cases for photo in case['selected_photo_ids']})
    for photo in photo_ids:
        include('images/' + photo + '.jpg', ROOT / 'data/processed/yelp_2300_v1/photos' / (photo + '.jpg'))

    # Inspect text only for accidental secrets; never print a matching value.
    for source in mapping.values():
        if source.suffix.lower() == '.jpg':
            continue
        raw = source.read_bytes()
        encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        text = raw.decode(encoding)
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', text), 'Potential credential in ' + source.name

    DEST.mkdir()
    copied = []
    for relative, source in mapping.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, target)
        assert sha(target) == digest
        copied.append({'path': relative, 'source_workspace_path': source.relative_to(ROOT).as_posix(),
                       'bytes': target.stat().st_size, 'sha256': digest})

    history = {name: rows(DEST / 'historical/predictions' / (name + '.jsonl')) for name in
               ['full_training_baseline', 'greedy_diagnostic_A', 'no_repeat_diagnostic_B']}
    index = []
    full_reviews = []
    for row_number, case in enumerate(cases):
        assert hashlib.sha256(case['source'].encode('utf-8')).hexdigest() == case['source_sha256']
        reviews = []
        for i, review_id in enumerate(case['selected_review_ids'], 1):
            marker = f'REV_{i}'
            match = re.search(r'\[' + marker + r'\] (.*?)(?=\n\n\[REV_\d+\]|\n\n\[VISUAL_CONTEXT\]|\Z)', case['source'], re.S)
            assert match, marker
            reviews.append({'marker': marker, 'review_id': review_id, 'text': match.group(1)})
        captions = re.findall(r'\[(IMG_CAP_\d+)\]\s*(.*?)(?=\n\n\[IMG_CAP_\d+\]|\Z)', case['source'], re.S)
        assert len(captions) == len(case['selected_photo_ids'])
        caps = [{'caption_id': marker, 'text': caption.strip(), 'photo_id': photo,
                 'image_file': 'images/' + photo + '.jpg'}
                for (marker, caption), photo in zip(captions, case['selected_photo_ids'])]
        for cap in caps:
            assert (DEST / cap['image_file']).is_file()
        full_reviews.append({'business_id': case['business_id'], 'reviews': reviews})
        cap_by_id = {cap['caption_id']: cap for cap in caps}
        for assessment in case['evidence_records'].get('assessments', []):
            cap = cap_by_id[assessment['caption_id']]
            assert cap['photo_id'] == assessment['photo_id']
            for claim in assessment['claims']:
                assert claim['caption_quote'] in cap['text']
        for claim in case['shared_review_evidence'].get('claims', []):
            for evidence in claim.get('support', []):
                review = next(r for r in reviews if r['marker'] == evidence['review_marker'])
                assert review['review_id'] == evidence['review_id'] and evidence['quote'] in review['text']
        historical_rows = {}
        for label, values in history.items():
            position = next(i for i, value in enumerate(values) if value['business_id'] == case['business_id'])
            historical_rows[label] = {'file': 'historical/predictions/' + label + '.jsonl', 'row_zero_based': position}
            if label == 'full_training_baseline':
                assert values[position] == case['baseline_prediction']
        index.append({'business_id': case['business_id'], 'subset': case['subset'],
            'case': {'file': 'preparation/evaluation_inputs.jsonl', 'row_zero_based': row_number},
            'full_selected_reviews': {'file': 'evidence/selected_reviews.jsonl', 'row_zero_based': row_number},
            'source_sha256': case['source_sha256'], 'target_sha256': hashlib.sha256(case['target'].encode()).hexdigest(),
            'captions': caps, 'teacher_review_claims_and_evidence': case['shared_review_evidence'],
            'teacher_visual_claims_and_evidence': case['visual_claims_with_evidence'],
            'teacher_caption_verification': case['evidence_records'], 'historical_predictions': historical_rows,
            'teacher_verdicts_are_not_human_ground_truth': True, 'new_paired_pilot_outputs_present': False})
    write(DEST / 'evidence/INDEX.json', {'relative_paths': True, 'businesses': index})
    with (DEST / 'evidence/selected_reviews.jsonl').open('x', encoding='utf-8') as stream:
        for row in full_reviews:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Reproduce train sampling strata without exporting all training reviews or packed arrays.
    training_jsonl = ROOT / 'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/train_filtered.jsonl'
    train_frame = [{'business_id': r['business_id'], 'source_tokens': r['source_tokens'],
                    'caption_count': len(r['selected_photo_ids']), 'source_sha256': r['source_sha256']}
                   for r in rows(training_jsonl)]
    assert len(train_frame) == 2000 and set(sample['training_business_ids']) <= {r['business_id'] for r in train_frame}
    write(DEST / 'evidence/train_sampling_frame.json', {'source_sha256': sha(training_jsonl), 'rows': train_frame})

    (DEST / 'README_FIRST.md').write_text('''# Paired Stage I LoRA pilot — offline preparation review

**No new model forward/backward, inference, GPU feasibility, training or Teacher calls have run. This package does not authorize execution.**

Read `CHATGPT_REVIEW_REQUEST.md`, then `preparation/STAGE_I_PILOT_EVALUATION_PROTOCOL.md`, `preparation/STAGE_I_PAIRED_LORA_PILOT_PLAN.md` and `preparation/READINESS_REPORT.md`. The release has 20 passing offline tests and a passing default CLI smoke; CPU/meta and synthetic gradient-checker results are not actual model-gradient or VRAM evidence.

The new paired arms are decoder-only A (24 modules / 294,912 parameters) and encoder+decoder B (48 / 589,824). Both are fresh from the pinned base, Filtered-only, on the same 512 train businesses, two epochs / 128 updates. All 200 validation businesses supply loss; the fixed 20 random and 4 targeted businesses supply later separate quality reporting. Independent named initialization gives equal common decoder tensors; common epoch orders and optimization are frozen. The old full-training F is a historical reference, not a matched A control.

## Evidence map

- `preparation/evaluation_inputs.jsonl`: exact 24 Filtered sources, Teacher targets, token IDs, review/photo IDs and Teacher evidence. No new paired-pilot outputs.
- `evidence/selected_reviews.jsonl`: complete selected reviews with original REV mappings, extracted without text changes.
- `evidence/INDEX.json`: each business/claim links to exact source/target rows, REV/IMG_CAP/PHOTO mappings, Teacher verification verdicts and historical predictions. `images/` contains 90 original selected validation JPEGs, copied byte-for-byte. Photos provide evidence for independent review; their presence does not certify any Teacher verdict.
- `preparation/reference_unit_candidates.jsonl`: 226 UNVERIFIED starting candidates, not accepted facts, atomic ground truth or a required annotation quota. Confirm/split/add missing source units and freeze them before generating or inspecting new pilot outputs; source review may accompany separately approved training. Blank human templates remain blank.
- `historical/predictions/`: saved full-training F baseline (200 rows) and old greedy/no-repeat diagnostics (24 F rows each). Historical decoding A/B are different experiments from the new paired architecture A/B. Historical AI notes are not human labels or unbiased error-rate estimates.
- `evidence/train_sampling_frame.json` plus fixed manifests permit sampling review without copying all training content. No held-out examples are present.
- `code/`, `configs/`, pinned tokenizer/config assets and `documentation/`: implementation and provenance. `PREPARATION_MANIFEST.json` uses original workspace-relative paths; `EXPORT_REPORT.json` maps exported files to those sources.

Current D-049 supersedes the numerical quality-win cutoffs, separate encoder init seed and practical-reference wording in historical D-048 proposal documents. Historical files remain unchanged. Source-grounded meaningful/mixed/regressed/inconclusive criteria, denominators and paired uncertainty replace automatic numerical wins. B changes both location and capacity; two epochs and one seed limit interpretation. Lower repetition or higher Teacher agreement alone is insufficient.

## Execution boundary

The runner defaults to offline. Explicit separate hash-bound approval is required for feasibility, training and generation. Both reviewed real feasibility reports precede training; both completed pilot trainings and researcher-confirmed reference freeze precede generation. No silent fallback or automatic phase transition. `approval_example_NOT_APPROVED.json` is deliberately false. Historical authorization flags are evidence, not current permission.

This is self-contained for document/evidence/code review, **not a standalone GPU deployment**: pretrained/adapter/optimizer binaries and packed training datasets are omitted. No model download or execution is needed to inspect it. Keep `blinding_key_keep_separate.json` away from future blinded output annotation. Manifest excludes itself; ZIP SHA-256 is supplied externally. No credentials are included.
''', encoding='utf-8')
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text('''# Review the paired architecture pilot before any GPU execution

The researcher authorized offline preparation only after the training-failure audit. Assess whether the current plan is scientifically controlled, technically ready for a bounded feasibility test, and clear enough for independent source-grounded evaluation.

1. Verify actual installed LED/PEFT module names, counts and freezing: 24 decoder self/cross Q/V projections in A; those plus 12 local and 12 global encoder Q/V in B; 294,912 versus 589,824 parameters. Do not equate meta attachment with real gradient flow.
2. Inspect independent initialization, matching decoder hashes, common seed/data order, 512 IDs, two epochs / 128 steps, accumulation/loss scaling, AdamW/scheduler, validation-loss selection and masks/special tokens. Identify defects rather than assuming encoder LoRA is a solution.
3. Review guarded future feasibility: actual branch gradients including initial zero-A/zero-LR behavior, padded 5,120/768, AMP, memory, checkpoint/scheduler/scaler/RNG restore and replay. Distinguish same-process replay from fresh-process recovery; the full Trainer path is not numerically tested. Execution must remain disabled until explicit approval.
4. Review the evaluation criteria before any new outputs: exact/phrase/semantic repetition; specific information/aspects/contrasts and omissions; atomic review/caption support; unsupported cross-caption relations; zero-caption/header/EOS/completeness. Teacher targets and AI notes are not independent truth. Ordinary correct simple visual content is acceptable. Check the 24 source records/photos and reference-candidate construction for missing evidence or biased denominators.
5. Assess whether predefined multidomain meaningful/mixed/regressed/inconclusive criteria and paired business-bootstrap uncertainty replace arbitrary quality-win cutoffs appropriately. Keep random 20 and targeted 4 separate; missing judgments are N/A. Do not invent human reference units or treat unverified candidate labels as final.
6. Check interpretation limits: scope and parameter count change together; common clipping can still affect optimization geometry; two epochs, one seed and repeatedly inspected validation cases cannot prove convergence, generalization or encoder causality. Historical full F is not a matched A baseline.

Return verified strengths, specific bugs/ambiguities, minimal corrections and approve/revise/decline recommendations for the **bounded feasibility phase only**. Do not execute code for training/inference, authorize full retraining, run new decoder experiments, modify datasets/Teacher targets/filters/checkpoints, or inspect held-out data. No automatic quality winner or production method adoption.
''', encoding='utf-8')
    report = {'status': 'EXPORT_ONLY_INTEGRITY_PASS', 'copied_files': copied, 'businesses': 24,
              'random_businesses': sum(c['subset'] == 'random' for c in cases),
              'targeted_businesses': sum(c['subset'] == 'targeted' for c in cases),
              'original_images': len(photo_ids), 'original_image_bytes': sum(mapping['images/' + photo + '.jpg'].stat().st_size for photo in photo_ids),
              'source_target_token_and_evidence_records_unchanged': True,
              'verified_reference_units': 0, 'new_paired_pilot_outputs': 0,
              'new_model_API_GPU_training_calls': 0, 'held_out_examples': 0,
              'runtime_weights_and_packed_training_data_included': False,
              'all_image_caption_review_ID_and_quote_links_validated': True,
              'release_tests_passed': 20, 'runtime_execution_approved': False}
    write(DEST / 'EXPORT_REPORT.json', report)
    payload = [{'path': f.relative_to(DEST).as_posix(), 'bytes': f.stat().st_size, 'sha256': sha(f)}
               for f in sorted(DEST.rglob('*')) if f.is_file()]
    write(DEST / 'MANIFEST.json', {'manifest_excludes_itself': True, 'files': payload})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        expected_names = {NAME + '/' + item['path'] for item in payload} | {NAME + '/MANIFEST.json'}
        assert set(archive.namelist()) == expected_names
        for item in payload:
            data = archive.read(NAME + '/' + item['path'])
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
        for entry in index:
            for cap in entry['captions']:
                assert NAME + '/' + cap['image_file'] in expected_names
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    result = {'status': 'ZIP_CRC_SHA256_AND_EVIDENCE_LINKS_PASS', 'zip': ZIP.relative_to(ROOT).as_posix(),
              'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size / 1048576, 'sha256': sha(ZIP),
              'payload_files': len(payload), 'cases': 24, 'images': len(photo_ids)}
    write(PREP / 'REVIEW_BUNDLE_EXPORT_REPORT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
