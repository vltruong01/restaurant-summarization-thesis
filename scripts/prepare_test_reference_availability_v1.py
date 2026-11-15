"""Record-only reference provenance discovery and source-only completion proposal."""
from pathlib import Path
import collections
import hashlib
import json
import os
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/test_reference_availability_v1'
ZIP = ROOT / 'outputs/cloud_handoff/TEST_REFERENCE_AVAILABILITY_V1_REVIEW.zip'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def doc(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def main():
    assert not OUT.exists() and not ZIP.exists(), 'Preserve existing version'
    OUT.mkdir(parents=True)
    old_inventory = ROOT / 'outputs/stage_i/led_large_matched_signoff_v3/REFERENCE_METADATA_INVENTORY.json'
    prior = read(old_inventory)
    extra = []
    errors = []
    scopes = []
    pattern = re.compile(r'reference|human|annotat|gold|stage[_ -]?k', re.I)
    for name in ('approvals', 'data', 'logs', '_archive_old', 'cloud', 'configs', 'schemas', 'prompts'):
        root = ROOT / name
        count = 0
        if root.is_dir():
            def error(e):
                errors.append({'path': str(e.filename), 'error': str(e)})
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=error):
                dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
                for filename in files:
                    count += 1
                    p = Path(directory) / filename
                    if not p.is_symlink() and pattern.search(filename):
                        s = p.stat()
                        extra.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': s.st_size,
                                      'mtime_ns': s.st_mtime_ns, 'payload_opened': False})
        scopes.append({'root': str(root), 'exists': root.is_dir(), 'files_examined': count, 'truncated': False})
    write('SEARCH_SCOPE_AND_METADATA.json', {
        'audit_date_local': '2026-10-25', 'prior_exhaustive_filename_inventory': {
            'path': old_inventory.relative_to(ROOT).as_posix(), 'sha256': sha(old_inventory),
            'files_examined': sum(s['files_examined'] for s in prior['scope']),
            'scope': prior['scope'], 'candidate_count': prior['candidate_count'], 'errors': prior['errors']},
        'additional_filename_only_scopes': scopes, 'additional_errors': errors,
        'additional_candidates': extra, 'reference_payloads_opened': 0,
        'TEST_inputs_opened': 0, 'model_prediction_files_opened': 0,
        'external_accounts_or_browser_drafts_searched': False,
        'limitations': 'Only declared local roots and explicitly selected project records; no archive payload inspection. No global absence claim.'})
    # Explicit metadata/report allowlist. No reference text, source case or prediction payload.
    records = [
        ('outputs/stage_i/d053_final_references_v1/REFERENCE_FREEZE_MANIFEST.json', 'Freeze manifest for the known D-053 validation reference chain'),
        ('outputs/stage_i/d053_final_references_v1/RESEARCHER_APPROVAL_RECORD.json', 'Explicit researcher curated approval for24 cases, no held-out authorization'),
        ('outputs/stage_i/d053_final_references_v1/D053_FINAL_REFERENCE_READINESS.json', '24businesses,576scoring units; held_out_access=false'),
        ('outputs/stage_i/d052_paired_pilot_v1/SOURCE_REFERENCE_READINESS.json', 'Historical24-case reference preparation; not100-test completion'),
        ('outputs/stage_i/d058_preparation_v1/DATASET_AND_REFERENCE_READINESS.json', '2000TRAIN/200validation and24 development evaluation cases'),
        ('outputs/stage_i/main_rouf_shuffled_readiness_v1/REFERENCES_AND_EVALUATION_AUDIT.json', 'Known development24; test references not found in its canonical audit scope'),
        ('outputs/stage_i/main_rouf_shuffled_readiness_v2/METRICS_AND_REFERENCES.json', 'Known validation24; no approved100-test binding'),
        ('outputs/stage_i/led_large_matched_completion_v2/TEST_REFERENCE_INDEX_PENDING.json', '100expected; actual verified coverage/approval hashes null'),
        ('outputs/stage_i/led_large_matched_signoff_v3/REFERENCE_COVERAGE_APPROVAL_INDEX.json', 'Current coverage unknown, not zero'),
    ]
    evidence = []
    data = {}
    for rel, finding in records:
        p = ROOT / rel
        d = read(p)
        data[rel] = d
        evidence.append({'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size,
                         'artifact_type': 'KNOWN_METADATA_OR_READINESS_REPORT', 'finding': finding,
                         'not_test_reference_payload': True})
    approval = data[records[1][0]]
    readiness = data[records[2][0]]
    assert approval['approved_scope']['businesses'] == readiness['businesses'] == 24
    assert readiness['held_out_access'] is False
    write('PROVENANCE_EVIDENCE_INDEX.json', {'records': evidence,
        'known_D053': {'scope': 'VALIDATION_ONLY', 'businesses': 24,
            'record_type': approval['record_type'], 'researcher_identity_as_recorded': approval['researcher_identity'],
            'approval_recorded_at_utc': approval['recorded_at_utc'],
            'counts': approval['approved_scope'], 'TEST_coverage_established': False,
            'authorship_caution': 'Researcher-reviewed curated development references; do not relabel as independently authored TEST references.'},
        'test_reference_independent_author': None, 'test_reference_approval_receipt': None})
    # Locate policy and experiment-history evidence in known project documents only.
    patterns = [r'test.{0,100}(reference|human)', r'reference.{0,100}(test|held.out)',
                r'Stage K', r'Human references must', r'Written directly from',
                r'not limited by any BLIP', r'Annotator [12]', r'without seeing model predictions']
    regex = re.compile('|'.join(patterns), re.I)
    excerpts = []
    for rel in ('THESIS_PROGRESS.md', 'DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md',
                'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                'thesis_sdd_docs/LED_COMPLETION_DECISION_MEMO_20261018.md'):
        p = ROOT / rel
        for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
            if regex.search(line):
                excerpts.append({'path': rel, 'line': i, 'text': line})
    write('PROJECT_RECORD_EXCERPTS.json', {'scope': 'Known protocol/progress/decision documents, not source inputs or predictions',
        'excerpts': excerpts, 'interpretation': 'Historical plans or not-found statements are not completion receipts; stale/provisional status retained as history.'})
    write('TEST_REFERENCE_AVAILABILITY_RECEIPT.json', {
        'status': 'NOT_YET_VERIFIED', 'expected_businesses': 100,
        'verified_TEST_reference_coverage': None, 'verified_approved_TEST_reference_count': None,
        'missing_count': None, 'verified_TEST_independent_author_provenance': None,
        'approved_TEST_reference_payload_sha256': None,
        'D053_VALIDATION_reference_businesses': 24, 'D053_usable_as_TEST_gold': False,
        'reason': 'No dedicated100-test reference manifest plus independent authorship and approval binding established in searched records; filenames and development-reference mirrors are insufficient.',
        'asserts_no_references_exist_anywhere': False, 'replacement_references_generated': 0,
        'TEST_inputs_opened': 0, 'model_predictions_opened': 0,
        'researcher_manages_calendar': True, 'new_deadline_requested': False,
        'GPU_authorization_granted': False, 'source_access_for_reference_creation_authorized': False})
    doc('README_VI.md', '''# Reference TEST: NOT YET VERIFIED

Đã kiểm tra thêm manifest, provenance D-053 và hồ sơ thí nghiệm. Chưa tìm được chuỗi bằng chứng xác lập bộ reference TEST độc lập đủ100doanh nghiệp; **không kết luận chúng không tồn tại**. Coverage TEST và số còn thiếu giữ null.

D-053 có receipt duyệt24doanh nghiệp validation. Các bản sao trong experiment bundles không trở thành reference TEST. Không dùng Teacher targets hoặc D-053 làm test gold.

`PROVENANCE_EVIDENCE_INDEX.json` và `PROJECT_RECORD_EXCERPTS.json` nêu bằng chứng; `SEARCH_SCOPE_AND_METADATA.json` ghi phạm vi và giới hạn. Không mở input TEST hoặc file dự đoán mô hình; không tạo reference, gọi Teacher hoặc chạy GPU.

`SOURCE_ONLY_REFERENCE_CREATION_PLAN.md` là đề xuất Stage K để duyệt, chưa thực hiện. Reference visual TEST được viết từ ảnh gốc theo protocol, độc lập với caption BLIP-2; khác phạm vi reference development D-053. Tách reference này khỏi đánh giá model-visible grounding theo từng condition.

Bạn tự quản lý lịch; báo cáo không đặt ngày bảo vệ hoặc hỏi lại thời hạn. Dừng để review trước khi mở nguồn TEST, soạn reference hoặc cấp quyền thực thi khác.
''')
    doc('FINDINGS_AND_EVIDENCE.md', '''# Findings and confidence

| Finding | Evidence | Interpretation |
|---|---|---|
| Expected final gold is100held-out businesses | Original Stage K in CODEX_THESIS_IMPLEMENTATION_PLAN.md | Planned requirement, not proof of completion |
| D-053 covers24 development/validation businesses | Explicit approval record + readiness + freeze manifest | Researcher-reviewed development reference provenance; never test coverage |
| Later development reports reuse D-053 | D-058 readiness and main-family reference audits | Repeated archive copies do not create new independent gold |
| Previous canonical audits did not establish test references | main readiness V1/V2 and completion/signoff index | Audit-scoped negative evidence; not a global nonexistence claim |
| No authenticated100-test manifest/independent-author/approval binding located in reviewed records | Discovery scope + provenance index | Final status NOT YET VERIFIED; unknown covered/approved/missing counts |
| Original Stage K visual gold comes from original photos, without BLIP-caption restriction | Stage K protocol and extracted source lines | Preserve source-independent gold; do not transplant D-053 caption-safe restriction into test reference creation |

No unknown candidate reference payload, held-out input, model prediction or archive member was opened to infer completion. The search combines the recent exhaustive local output/Downloads filename inventory with additional project filename traversal and explicit allowlisted protocol/readiness/provenance records. Filesystem metadata and known reports cannot certify semantic source support or rule out off-machine/unsubmitted references. External accounts, browser drafts and private collaborator storage remain outside scope.

No tolerance, model, checkpoint, target, evaluation criterion, RO/U/F/Shuffled policy or historical receipt was modified. This is a reference dependency audit, not an authorization or a new research technique.
''')
    doc('SOURCE_ONLY_REFERENCE_CREATION_PLAN.md', '''# Stage K source-only, model-output-blind reference plan — PROPOSED

## Prerequisites and scope

First ask the existing reference custodian, if any, for a dedicated metadata-only index: test split; expected100; unique covered/approved/unresolved counts; reference version; source/payload hashes; independent author/checker identities and dates; prediction-blinding declaration; approval receipt/hash. Reconcile real coverage after authentication. Preserve any valid previously approved references byte-for-byte. Unknown coverage is not100missing and is not a license to regenerate everything.

Only after separate explicit authorization may an independent coordinator open the locked100-business TEST source frame and construct source-only author packets. This proposal grants no input access. Use the existing fixed business membership and selected reviews without re-selection, source changes or outcome-based filtering. If required selected-source artifacts do not yet exist, report the dependency and obtain the already-frozen source-construction authorization; do not invent sources or retune selection. Hash source IDs/text/photo bytes and packet versions under authorized access. Include no predictions, Teacher targets, student checkpoint outputs, metrics, condition labels or suggested reference answers.

## Roles and independence

Assign a real human drafter and a different independent human checker, plus a researcher/adjudicator. Do not fabricate their names, completion or signatures. Both authors/checkers attest that they have not seen evaluated TEST model outputs; record any prior exposure and let researcher assess the conflict before assignment. Reference construction is distinct from the later three-annotator output-rating study. Do not assume that24reviewed development cases establish either test authorship or rater availability.

Drafter writes references from sources. Checker independently inspects sources and the draft, checks support/completeness/balance and records agreement or requested corrections. Resolve disagreements before locking; retain all versions and evidence. No automatic acceptance, synthetic human, model-generated replacement text or Teacher call. Any later proposal for assisted drafting requires separate methodological approval; it is not included here.

## Content policy preserved from Stage K

`[REVIEW_CONSENSUS]`: authored from the exact selected customer reviews. Cover important recurring business-specific aspects, preserve meaningful positive/negative disagreement on the same aspect, distinguish one-off reports from consensus, and describe reviewers' opinions rather than independently verified real-world facts. Every substantive proposition has source review/offset support; do not invent causal explanations or stitch unrelated accounts into a relationship. Keep concise coherent synthesis without a new forced length policy, artificial balance, every minor detail or a new salience optimization cycle.

`[VISUAL_EVIDENCE]`: authored directly from useful observable information in the approved original-photo source scope. Original Stage K explicitly does NOT limit independent human visual gold to any BLIP-2 caption set. Do not show captions as surrogate evidence or tailor references to what a particular model can produce. Bind statements to photo IDs; avoid uncertain object identities, imagined relations/processes and review-derived visual claims. Preserve uncertainty; when no useful visual source exists, record empty/N/A applicability with source-derived reason under researcher-approved rules, not a fabricated description.

This clarifies the scope difference: D-053's24validation units were deliberately caption-safe; that restriction is not silently carried over to Stage K. Reference completeness from photos and inference grounding are different questions. Later condition-aware evaluation separately asks whether a model assertion was supported by its actual visible reviews/captions and corroborated by photos. A photo-only fact in gold does not make an unsupported model assertion grounded. RO visual scoring remains N/A; zero-caption visual assertions remain unsupported. Do not build outcome-dependent per-model gold or replace recipient-business gold with Shuffled donor gold.

Before authoring, researcher confirms the existing original-photo scope and reference guideline; if the protocol does not identify the photo set unambiguously, stop for that one scope decision rather than choose a new sample. No actual photos are opened here. Do not silently introduce Core/Supporting weights or new automatic metric denominators into the fixed final evaluation; evidence notes may support later qualitative assessment without changing the accepted metric lock.

## Records and approval

Keep per-case draft/check/adjudication statuses, writer/checker attestations, exact source links, uncertain/excluded propositions, and raw revision history. Use IDs only after actual source access is authorized; no test IDs are materialized now. Preserve an immutable final two-section reference file, independent authorship ledger, annotation guideline, disagreement/resolution notes and version SHA-256 manifest, as required by Stage K. Reviewer/researcher approval must be explicit and attributable to actual actions; recording a timestamp does not imply a human inspected content then.

Acceptance: exactly100unique locked test businesses with complete source bindings and two-person review; no unresolved blocking claim or fake coverage; all reference/provenance hashes match; no Teacher/validation substitution; independent model-output-blind attestations; explicit researcher freeze approval. Freeze before reference-based final scoring and preferably before producing/releasing any TEST outputs to the reference team. Model-output blindness must persist even if an operator has separately generated outputs. Any subsequent necessary reference correction is a versioned source-supported erratum applied consistently, with reason and approval, never a metric-driven edit.

## Work plan and stop rules

Researcher controls timing. First establish missing count M and real human capacity. Measure drafting/checking workload on a separately approved source-only practice workflow; plan M*(draft_minutes+check_minutes)/60 plus adjudication/admin, with no invented throughput. Preserve the accepted final evaluation reserve and fresh authorization checks when those operations are later considered; no calendar assertions are fabricated by this audit.

STOP if source access is not approved, source scope/coverage is ambiguous, independent authorship cannot be established, support is unresolved or researcher freeze is absent. If no verified existing gold is found, this remains a proposal for the100-business program, not permission to create it. No reference file, TEST input packet, actual100-case index/mapping, inference or final score is generated here. Return for review before proceeding.
''')
    write('REFERENCE_FREEZE_CHECKLIST_PENDING.json', {
        'status': 'PROPOSED_NOT_EXECUTED', 'scope_expected': '100unique held-out Yelp businesses',
        'existing_coverage_authenticated': None, 'source_access_approved': False,
        'actual_business_rows': [], 'reference_authors_confirmed': False,
        'prediction_blinding_attested': None, 'source_packet_manifest_sha256': None,
        'reference_payload_sha256': None, 'independent_check_ledger_sha256': None,
        'unresolved_blockers': None, 'researcher_freeze_approved': False,
        'reference_generation_calls': 0, 'human_actions_claimed': 0})
    (OUT / 'BUILD_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    entries = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write('MANIFEST.json', {'files': entries, 'path_base': 'archive_root'})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            z.write(p, p.name)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name, expected in entries.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    print(json.dumps({'status': 'NOT_YET_VERIFIED', 'archive': str(ZIP), 'sha256': sha(ZIP),
        'manifest_entries': len(entries), 'records_examined': len(records),
        'project_record_excerpts': len(excerpts), 'additional_filenames': sum(s['files_examined'] for s in scopes),
        'additional_candidates': len(extra), 'additional_errors': errors,
        'TEST_inputs_or_predictions_opened': 0}, indent=2))


if __name__ == '__main__':
    main()
