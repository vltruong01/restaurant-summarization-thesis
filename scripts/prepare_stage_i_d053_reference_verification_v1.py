"""Prepare the offline D-053 source-only annotation package; no model execution.

Reads accepted validation evidence and existing D-052 checkpoint hashes only.
Candidate claims remain suggestions. This program cannot create human labels.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
OLD = ROOT / 'outputs/stage_i/d052_source_reference_review_v1'
D052 = ROOT / 'outputs/stage_i/d052_paired_pilot_v1'
OUT = ROOT / 'outputs/stage_i/d053_reference_verification_v1'
CONFIG = ROOT / 'configs/stage_i_d053_evaluation_v1.json'
PHOTOS = ROOT / 'data/processed/yelp_2300_v1/photos'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


GUIDE = r'''# Stage I — D-053 Reference Verification Guide

**Status: annotation preparation only. No new Pilot A/B summaries exist in this package.**

## Mở và sử dụng

Giải nén toàn bộ ZIP, mở `SOURCE_REFERENCE_ANNOTATION.html` bằng Chrome/Edge. Giữ thư mục `images/` cạnh HTML. Không cần cài model, chạy notebook hoặc gọi API. Trang hoạt động offline; không upload nội dung lên dịch vụ nào.

1. Nhập tên/mã người kiểm tra. Chọn business; nhãn `random` hoặc `targeted` và condition `Filtered` được giữ cố định.
2. Đọc toàn bộ selected reviews, không chỉ quote gợi ý. Với visual claim, đọc caption và xem đúng ảnh mang PHOTO ID tương ứng.
3. Chọn một trong **Supported / Unsupported / Contradicted / Uncertain / Needs revision**. Không có nhãn được chọn sẵn. Ghi lý do, importance, informativeness và corrected claim text khi cần.
4. Chỉ xác nhận Supported khi ý được đúng nguồn cho phép hỗ trợ. Một câu liệt kê nhiều món/ý kiến phải tách thành các reference units nguyên tử do bạn xác nhận, không coi cả danh sách là một fact. Dùng chức năng thêm/tách unit; chọn evidence và quote nguyên văn cho từng ý.
5. Corrected text là bản nháp, không tự trở thành reference. Với Needs revision, xác nhận các unit đã sửa được nguồn hỗ trợ, hoặc ghi rõ lý do loại khỏi reference. Giữ nguyên quyết định về claim gốc trong lịch sử. Không sửa caption hoặc teacher target.
6. Dùng ô “missing important information” để ghi các ý còn thiếu, rồi **Add** unit được nguồn hỗ trợ nếu ý đó nên nằm trong denominator coverage. Ghi chú một mình không tự tạo reference unit.
7. Rà soát các aspect chính, ý tích cực/tiêu cực, ý kiến đối lập, chi tiết đặc trưng và độ đầy đủ nguồn; xác nhận completeness cho từng business. Việc sửa quyết định/unit phải bỏ xác nhận completeness cũ để bạn rà lại.
8. Xuất progress JSON thường xuyên và dùng Import để tiếp tục. Browser autosave chỉ là tiện ích; nếu chưa gửi file về project, Codex không biết tiến độ trên trình duyệt của bạn.
9. Khi đủ 226 quyết định rõ ràng và 24 business completeness confirmations, xuất JSON final bằng xác nhận rõ ràng của bạn. Các trường hợp Uncertain vẫn được giữ riêng, không đổi thành nhãn âm. Final references chỉ chứa các unit Supported đã được bạn xác nhận; tổng số unit không cần bằng 226.
10. Gửi file annotation cuối để kiểm tra schema/ID/quote/bindings offline. Xác nhận trong HTML không tự cho phép generation. Cần duyệt riêng trước khi sinh 48 summaries (24 businesses × 2 pilots).

## Nhãn của candidate claim

| Label | Definition |
| --- | --- |
| Supported | The permitted evidence directly supports the proposition or an ordinary faithful paraphrase. |
| Unsupported | The evidence does not establish the claim; plausibility alone is insufficient. This does not prove it false in the real world. |
| Contradicted | The evidence conflicts with a specific asserted detail, count, identity, relationship, category or opinion. |
| Uncertain | Evidence or interpretation is genuinely ambiguous; preserve separately. |
| Needs revision | Partly correct or too imprecise/multi-fact for the intended reference. Keep the original claim; explicitly verify corrected/split units separately, or document exclusion. |

Candidate labels are not student-output error labels or model accuracy results. They never change the original 226 AI/Teacher-derived suggestions. The source review, human decisions and finalized atomic units are separate records.

**Importance:** `core` = important information the summary should preserve; `supporting` = useful additional detail. Importance/informativeness are optional descriptive fields on candidate decisions (a multi-fact candidate may need units with different priorities). Both fields are mandatory on each final supported atomic unit. Negative/uncertain candidates need not acquire artificial importance values. Final-unit priorities must be confirmed before seeing new student outputs.

**Informativeness:** `useful_specific`, `useful_general`, `low_information` are descriptive annotations, not a numeric quality score. Ordinary but correct/simple content is not a factual error. Low information density alone does not make a claim unsupported. Preserve the existing `restaurant_specific` and review-polarity fields on final units.

## Evidence boundaries

**Review units:** use selected customer review text only, with the exact `REV_n`, original review ID and quote. Reviews establish what customers expressed, not independently verified facts about the restaurant. Preserve opposing opinions and do not exaggerate how many reviewers agree. Do not import caption/photo details into review consensus.

**Visual units:** the information must be recoverable from the exact supplied `IMG_CAP_n` caption and corroborated by its mapped original photograph (`PHOTO`/photo_id). A photo cannot supply extra ingredients/counts/actions/relationships absent from the caption. Multiple captions must not be merged into an unsupported same-object relationship. Metadata and review facts cannot be relabeled as visual evidence. The zero-caption business has no eligible visual reference units.

Correct but simple descriptions remain acceptable. Reasonable ordinary visual interpretation is allowed; unsupported specific ingredients, object identities, counts, actions and relationships are not automatically verified. Do not invent negative examples or force class balance.

## Protocol provenance and narrow D-053 clarification

The D-049 evaluation protocol, fixed cases, frozen generation settings and D-052 checkpoints remain byte-identical. D-049's sentence placing independent photo verification outside that original protocol is retained as history. The explicit D-053 instruction now requires researcher photo corroboration during **source-reference preparation**. This is an approved additive verification requirement, not a new image scorer, threshold, label change or permission to add photo-only facts. Student visual claims still require supplied-caption support.

Teacher targets are not included in the annotation UI. Candidate provenance remains visible, but their labels are blank/unverified. Teacher material, if inspected elsewhere, is auxiliary and not independent factual ground truth. No historical or new student outputs are displayed in the source worksheet.

## Offline validation

From the project root, validate a saved progress/final annotation using:

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\stage_i_d053_annotations.py --package outputs/stage_i/d053_reference_verification_v1 --annotations "PATH_TO_YOUR_ANNOTATIONS.json"
```

Add `--final` to require full candidate review, confirmed atomic references and all business-completeness decisions. Follow `code/` in the review ZIP for portable source inspection; the original project command uses `scripts/`. Validation checks structure, bindings, exact quotations, domains, provenance fields and readiness. It cannot prove that a claimed human confirmation is truthful or independently verify the image/claim semantics.

## Later comparison — prepared, not executed

Only the matched fresh A/B best checkpoints are eligible: decoder-only A and encoder+decoder B, both selected at step 128. The historical full F checkpoint is context, not the matched comparator. Both future arms must use identical frozen validation inputs and the complete accepted GenerationConfig (`max_length=768`, beam 4, deterministic sampling off, original no-repeat setting 0). No decoding choice is selected based on the expected winner.

Repetition/structure/token termination can be measured automatically on future raw outputs. Semantic redundancy, informativeness coverage, unsupported/contradicted claims, cross-caption errors, missing important information and semantic completeness require explicit human judgments and source evidence. Missing or unverified judgments are N/A, not zero errors. Keep random 20 and targeted 4 separate. Use the D-049 paired-business bootstrap for eligible random-sample measures, not targeted prevalence estimates or claim-level pseudo-replication. No weighted composite or automatic architecture winner is produced.

**Stop boundary:** no new training, inference, generation, Teacher API calls, held-out access or accepted-data/checkpoint modification. After researcher verification and separate approval, a later task may generate exactly 48 summaries and compare their actual quality.
'''


def verify_inputs():
    prep_manifest = read(PREP / 'PREPARATION_MANIFEST.json')
    frozen = {}
    for name, expected in prep_manifest['files'].items():
        path = PREP / name
        require(sha(path) == expected, 'D049 preparation changed: ' + name)
        frozen[path.relative_to(ROOT).as_posix()] = expected
    for group in ['code', 'upstream']:
        for name, expected in prep_manifest[group].items():
            path = ROOT / name
            require(sha(path) == expected, 'D049 binding changed: ' + name)
            frozen[name] = expected
    old_manifest = read(OLD / 'SHA256_MANIFEST.json')
    for name, expected in old_manifest['files'].items():
        path = OLD / name
        require(sha(path) == expected, 'D052 source worksheet changed: ' + name)
        frozen[path.relative_to(ROOT).as_posix()] = expected
    training = read(D052 / 'D052_TRAINING_VALIDATION.json')
    require(training['status'] == 'PAIRED_TRAINING_AND_OFFLINE_AUDIT_PASS_NO_GENERATION', 'D052 training not accepted')
    require(training['new_generation_calls'] == 0, 'Unexpected D052 predictions')
    for arm in training['architectures']:
        path = ROOT / arm['best_adapter']['path']
        require(sha(path) == arm['best_adapter']['sha256'], 'D052 best adapter changed')
        frozen[path.relative_to(ROOT).as_posix()] = arm['best_adapter']['sha256']
        for checkpoint in arm['checkpoints']:
            for name, info in checkpoint['files'].items():
                path = ROOT / checkpoint['path'] / name
                require(sha(path) == info['sha256'], 'D052 checkpoint changed: ' + str(path))
                frozen[path.relative_to(ROOT).as_posix()] = info['sha256']
    for name in ['D052_TRAINING_VALIDATION.json', 'STAGE_I_D052_PAIRED_TRAINING_REPORT.md',
                 'D052_CONDITIONAL_REPRODUCIBILITY_DECISION.md', 'POST_EXECUTION_INTEGRITY.json']:
        path = D052 / name
        frozen[path.relative_to(ROOT).as_posix()] = sha(path)
    cases = read(OLD / 'source_cases.json')
    candidates = rows(PREP / 'reference_unit_candidates.jsonl')
    raw = {row['business_id']: row for row in rows(PREP / 'evaluation_inputs.jsonl')}
    manifest = read(PREP / 'pilot_manifest.json')
    expected_ids = manifest['generation_evaluation_random20'] + manifest['generation_evaluation_targeted4']
    require([case['business_id'] for case in cases] == expected_ids, 'Fixed evaluation IDs/order changed')
    require(len(cases) == len(raw) == 24 and len(set(expected_ids)) == 24, 'Case count/duplicates')
    require(len(candidates) == len({row['unit_id'] for row in candidates}) == 226, 'Candidate count/duplicate IDs')
    require(sum(row['domain'] == 'review' for row in candidates) == 136, 'Review candidate count')
    require(sum(row['domain'] == 'visual' for row in candidates) == 90, 'Visual candidate count')
    byid = {case['business_id']: case for case in cases}
    image_hashes = {}
    original_photo_checks = 0
    for case in cases:
        bid = case['business_id']; source = raw[bid]
        require(case['condition'] == 'filtered' and case['split'] == 'validation', 'Wrong condition/split')
        require(case['subset'] == ('random' if bid in manifest['generation_evaluation_random20'] else 'targeted'), 'Subset role changed')
        require(case['source'] == source['source'] and case['source_sha256'] == hashlib.sha256(case['source'].encode()).hexdigest(), 'Source text/hash mismatch')
        require(case['selected_review_ids'] == source['selected_review_ids'] == [r['review_id'] for r in case['reviews']], 'Review IDs/order mismatch')
        require(case['selected_photo_ids'] == source['selected_photo_ids'] == [r['photo_id'] for r in case['captions']], 'Photo IDs/order mismatch')
        parsed_reviews = re.findall(r'\[(REV_\d+)\] (.*?)(?=\n\n\[REV_\d+\]|\n\n\[VISUAL_CONTEXT\]|\Z)', case['source'], re.S)
        require(parsed_reviews == [(r['marker'], r['text']) for r in case['reviews']], 'Full review text/order mismatch')
        caption_block = case['source'].split('[VISUAL_CONTEXT]', 1)[-1]
        parsed_caps = re.findall(r'\[(IMG_CAP_\d+)\]\n(.*?)(?=\n\n\[IMG_CAP_\d+\]|\Z)', caption_block, re.S)
        require([(m, text.strip()) for m, text in parsed_caps] == [(r['caption_id'], r['text']) for r in case['captions']], 'Caption text/order mismatch')
        for cap in case['captions']:
            relative = 'images/' + cap['photo_id'] + '.jpg'
            require(cap['image_file'] == relative, 'Unsafe/nonportable photo path')
            copied = OLD / relative
            require(copied.is_file(), 'Missing photo: ' + relative)
            digest = sha(copied)
            require(old_manifest['files'].get(relative) == digest, 'Photo copy changed')
            image_hashes[relative] = digest
            original = PHOTOS / (cap['photo_id'] + '.jpg')
            if original.is_file():
                require(sha(original) == digest, 'Original photo mismatch')
                original_photo_checks += 1
    evidence_count = 0
    for candidate in candidates:
        require(candidate['business_id'] in byid and candidate['domain'] in ['review', 'visual'], 'Candidate domain/business mismatch')
        require(candidate['human_verified'] is False and candidate.get('support_status') is None and candidate.get('confirmed_by') is None, 'Candidate suggestion was promoted to human annotation')
        case = byid[candidate['business_id']]
        require(candidate['evidence'], 'Candidate has no evidence')
        for evidence in candidate['evidence']:
            if candidate['domain'] == 'review':
                allowed = {r['marker']: r for r in case['reviews']}
                row = allowed.get(evidence.get('marker'))
                require(row and evidence.get('review_id') == row['review_id'] and evidence.get('quote') and evidence['quote'] in row['text'], 'Candidate REV evidence/quote mismatch')
            else:
                allowed = {r['caption_id']: r for r in case['captions']}
                row = allowed.get(evidence.get('marker'))
                require(row and evidence.get('photo_id') == row['photo_id'] and evidence.get('quote') and evidence['quote'] in row['text'], 'Candidate caption/photo/quote mismatch')
            evidence_count += 1
    return cases, candidates, frozen, image_hashes, original_photo_checks, evidence_count


def main():
    require(not OUT.exists(), 'Preserve previous D053 package; revisions need a new namespace')
    from stage_i_d053_annotations import make_initial_state, validate_annotations
    from stage_i_d053_evaluation import future_annotation_shell
    cases, candidates, frozen, image_hashes, original_photo_checks, evidence_count = verify_inputs()
    require(CONFIG.is_file(), 'Prepare frozen evaluation configuration first')
    bindings = {'case_inputs_sha256': sha(PREP / 'evaluation_inputs.jsonl'),
                'protocol_sha256': sha(PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),
                'candidate_sha256': sha(PREP / 'reference_unit_candidates.jsonl'),
                'source_cases_sha256': sha(OLD / 'source_cases.json'),
                'evaluation_config_sha256': sha(CONFIG)}
    bundle_id = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    state = make_initial_state(cases, candidates, bindings, bundle_id)
    initial_check = validate_annotations(state, cases, candidates, bindings, final=False)
    require(initial_check['valid'] and not initial_check['ready_for_freeze'], 'Initial state must be valid, unverified and incomplete')
    OUT.mkdir(parents=True)
    (OUT / 'images').mkdir()
    for relative, digest in image_hashes.items():
        shutil.copyfile(OLD / relative, OUT / relative)
        require(sha(OUT / relative) == digest, 'Copied photo differs')
    for source, name in [(OLD / 'source_cases.json', 'source_cases.json'),
                         (PREP / 'reference_unit_candidates.jsonl', 'candidates_UNVERIFIED.jsonl'),
                         (PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md', 'D049_FROZEN_EVALUATION_PROTOCOL.md'),
                         (PREP / 'reference_freeze_template.json', 'D049_reference_freeze_template_UNVERIFIED.json'),
                         (PREP / 'expected_generation_config.json', 'FROZEN_GENERATION_CONFIG.json'),
                         (PREP / 'pilot_manifest.json', 'D049_FROZEN_PILOT_MANIFEST.json'),
                         (CONFIG, 'FROZEN_EVALUATION_CONFIG.json')]:
        shutil.copyfile(source, OUT / name)
        require(sha(source) == sha(OUT / name), 'Copy mismatch: ' + name)
    write_new(OUT / 'bindings.json', {'bundle_id': bundle_id, 'bindings': bindings})
    write_new(OUT / 'initial_annotations_UNVERIFIED.json', state)
    write_new(OUT / 'INITIAL_ANNOTATION_VALIDATION.json', initial_check)
    write_new(OUT / 'FUTURE_OUTPUT_ANNOTATION_SHELL_UNVERIFIED.json', future_annotation_shell(cases, read(CONFIG)))
    shutil.copyfile(ROOT / 'scripts/STAGE_I_D053_FUTURE_EVALUATION_USAGE.md', OUT / 'FUTURE_EVALUATION_USAGE.md')
    (OUT / 'STAGE_I_D053_REFERENCE_VERIFICATION_GUIDE.md').write_text(GUIDE, encoding='utf-8')
    payload = {'bundle_id': bundle_id, 'bindings': bindings, 'cases': cases, 'candidates': candidates,
               'initial_working_state': state, 'protocol': (PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md').read_text(encoding='utf-8'),
               'd053_guide': GUIDE, 'freeze_template': read(PREP / 'reference_freeze_template.json')}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = (ROOT / 'scripts/stage_i_d053_annotation_ui.html').read_text(encoding='utf-8')
    require(template.count('__PAYLOAD__') == 1, 'UI payload placeholder mismatch')
    (OUT / 'SOURCE_REFERENCE_ANNOTATION.html').write_text(template.replace('__PAYLOAD__', data), encoding='utf-8')
    index = [{'claim_id': row['unit_id'], 'business_id': row['business_id'], 'condition': 'filtered',
              'subset': next(c['subset'] for c in cases if c['business_id'] == row['business_id']),
              'domain': row['domain'], 'caption_or_review_evidence': row['evidence'],
              'human_verified': False} for row in candidates]
    write_new(OUT / 'annotation_index.json', index)
    completeness = [{'business_id': c['business_id'], 'condition': c['condition'], 'subset': c['subset'],
                     'reviews': len(c['reviews']), 'captions': len(c['captions']),
                     'review_candidates': sum(x['business_id'] == c['business_id'] and x['domain'] == 'review' for x in candidates),
                     'visual_candidates': sum(x['business_id'] == c['business_id'] and x['domain'] == 'visual' for x in candidates),
                     'human_confirmed_candidates': 0, 'source_completeness_confirmed': False} for c in cases]
    write_new(OUT / 'BUSINESS_ANNOTATION_READINESS.json', completeness)
    report = {'status': 'OFFLINE_PREPARATION_READY_HUMAN_VERIFICATION_PENDING', 'businesses': len(cases),
              'random': sum(c['subset'] == 'random' for c in cases), 'targeted': sum(c['subset'] == 'targeted' for c in cases),
              'condition': 'filtered', 'candidate_units': len(candidates), 'review_candidates': 136, 'visual_candidates': 90,
              'complete_selected_reviews': sum(len(c['reviews']) for c in cases), 'original_photos': len(image_hashes),
              'original_photo_source_file_matches_checked': original_photo_checks,
              'evidence_id_and_exact_quote_checks': evidence_count,
              'candidate_decisions_by_label': {k: 0 for k in ['supported', 'unsupported', 'contradicted', 'uncertain', 'needs_revision']},
              'pending_candidates': len(candidates), 'human_verified_candidates': 0, 'confirmed_reference_units': 0,
              'business_completeness_confirmations': 0, 'ready_for_reference_freeze': False, 'ready_for_generation': False,
              'zero_caption_businesses': [c['business_id'] for c in cases if not c['captions']],
              'ai_candidates_copied_byte_identically': sha(PREP / 'reference_unit_candidates.jsonl') == sha(OUT / 'candidates_UNVERIFIED.jsonl'),
              'source_cases_copied_byte_identically': sha(OLD / 'source_cases.json') == sha(OUT / 'source_cases.json'),
              'semantic_claim_support_independently_verified': False, 'no_student_outputs_or_teacher_final_targets_in_ui': True,
              'new_training_runs': 0, 'new_model_inference_or_generation': 0, 'teacher_API_calls': 0, 'held_out_data_access': False,
              'evaluation_metrics_on_new_pilot_outputs': 'NOT_RUN_NO_PREDICTIONS',
              'D049_protocol_preserved': True,
              'D053_photo_corroboration': 'Explicit user-confirmed visual-source check in addition to supplied-caption support; no automated photo judge',
              'bindings': bindings, 'bundle_id': bundle_id, 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_new(OUT / 'ANNOTATION_READINESS_REPORT.json', report)
    (OUT / 'ANNOTATION_READINESS_REPORT.md').write_text(
        '# D-053 annotation preparation and readiness\n\n'
        '**Package prepared; human source verification and generation approval are pending.**\n\n'
        'Verified fixed 24 validation cases (20 random / 4 targeted), 226 unchanged AI candidates '
        '(136 review / 90 visual), full selected reviews, exact evidence IDs/quotes and original photo mappings. '
        'These are binding/integrity checks, not factual endorsement of candidate claims.\n\n'
        f"- Full selected reviews: {report['complete_selected_reviews']}; original photo copies: {len(image_hashes)}.\n"
        f'- Exact evidence-binding checks: {evidence_count}; original-source photo matches: {original_photo_checks}.\n'
        '- Human-verified candidate decisions: 0/226; confirmed reference units: 0; complete businesses: 0/24.\n'
        '- Uncertain cases remain separate. Corrected text is a draft until separately confirmed as a supported atomic unit.\n'
        '- Future quality metrics are prepared only. No A/B predictions, factual accuracy scores or architecture winner are reported.\n\n'
        'Use SOURCE_REFERENCE_ANNOTATION.html and the guide. Save progress JSON regularly. '
        'Only explicit human confirmations may enter final references; final export must pass offline validation. '
        'Teacher/caption candidates are not independent factual ground truth.\n\n'
        '**Stop:** no training, inference, Teacher requests, held-out use, accepted-data edits or checkpoint overwrite. '
        'A later separately approved task may generate 48 summaries after reference freeze.\n', encoding='utf-8')
    (OUT / 'README_FIRST.md').write_text(
        '# D-053 source-based reference verification\n\n'
        'Open `SOURCE_REFERENCE_ANNOTATION.html` with its sibling `images/` folder. '
        'Read `STAGE_I_D053_REFERENCE_VERIFICATION_GUIDE.md` before labeling.\n\n'
        'The package contains fixed validation sources and 226 unverified candidate suggestions, '
        'not new Pilot A/B outputs or human labels. Review, edit/split, add missing information '
        'and explicitly confirm your decisions. Save progress; final references require offline validation '
        'and separate permission before any new generation.\n\n'
        'See `ANNOTATION_READINESS_REPORT.md`, `bindings.json` and the frozen evaluation/generation configs. '
        'Five candidate-label decisions are separate from finalized supported atomic reference units. '
        'The original D-049 protocol and D-052 files remain unchanged; D-053 explicitly adds original-photo '
        'corroboration of caption-recoverable visual units.\n', encoding='utf-8')
    mismatches = [name for name, digest in frozen.items() if sha(ROOT / name) != digest]
    require(not mismatches, 'Immutable artifact changed during preparation: ' + str(mismatches))
    write_new(OUT / 'IMMUTABLE_INPUT_PRESERVATION.json', {'status': 'PASS', 'files': frozen,
                                                       'all_hashes_unchanged': True, 'new_predictions': 0})
    print(json.dumps({k: report[k] for k in ['status', 'businesses', 'candidate_units', 'human_verified_candidates',
                                          'confirmed_reference_units', 'original_photos', 'ready_for_generation']}, indent=2))


if __name__ == '__main__':
    main()
