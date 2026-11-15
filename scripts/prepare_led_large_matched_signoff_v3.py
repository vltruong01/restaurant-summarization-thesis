"""Offline signoff supplement; never reads held-out payloads or imports model code."""
from pathlib import Path
import collections
import hashlib
import json
import os
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/stage_i/led_large_matched_completion_v2'
OUT = ROOT / 'outputs/stage_i/led_large_matched_signoff_v3'
ARCHIVE = ROOT / 'outputs/cloud_handoff/LED_LARGE_MATCHED_SIGNOFF_V3_OFFLINE_REVIEW.zip'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def doc(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def verify_manifest(path):
    entries = read(path)['files']
    if not isinstance(entries, dict):
        raise ValueError('Unexpected manifest schema')
    for rel, digest in entries.items():
        p = (ROOT / rel).resolve()
        assert p.is_relative_to(ROOT.resolve()), rel
        assert p.is_file() and sha(p) == digest, rel
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'verified_files': len(entries), 'status': 'SHA256_PASS'}


def fixture_statistics(mappings):
    """Rates refer to positive-caption recipients; zero cases excluded explicitly."""
    positive = [m for m in mappings if m['caption_count'] > 0]
    tier = collections.Counter(m['tier'] for m in positive)
    donors = collections.Counter((m['split'], m['donor_id']) for m in positive)
    n = len(positive)
    counts = list(donors.values())
    hhi = sum((x / n) ** 2 for x in counts) if n else None
    return {'businesses': len(mappings), 'positive_caption_recipients': n,
            'zero_caption_unchanged': len(mappings) - n,
            'tier_counts': dict(tier),
            'category_match_rate': sum(bool(set(m.get('recipient_tags', [])) & set(m.get('donor_tags', []))) for m in positive) / n if n else None,
            'fallback_rate': tier['DECLARED_SAME_COUNT_FALLBACK'] / n if n else None,
            'distinct_donors': len(donors), 'max_donor_reuse_count': max(counts, default=0),
            'max_donor_assignment_share': max(counts) / n if n else None,
            'donor_assignment_HHI': hhi, 'effective_donor_count_inverse_HHI': 1 / hhi if hhi else None,
            'denominator_note': 'Assignment shares among positive-caption recipients; donor identified within split. Not caption-token-weighted.',
            'unavailable_recipients_in_completed_fixture': sum(m['eligible_count'] == 0 for m in positive),
            'strata_present': dict(collections.Counter(f"{m['split']}|captions={m['caption_count']}" for m in mappings))}


def inventory():
    """Filenames/stat only. Never open candidates or list archive members."""
    roots = [ROOT / 'outputs', ROOT / 'thesis_sdd_docs', ROOT / 'docs',
             ROOT / 'specs', Path('C:/Users/VUONGLOCTRUONG/Downloads')]
    pattern = re.compile(r'reference|annotation|annotator|human_eval|human-eval|gold|adjudicat', re.I)
    items, scopes, errors, skipped = [], [], [], []
    for root in roots:
        scope = {'root': str(root), 'exists': root.is_dir(), 'files_examined': 0,
                 'truncated': False, 'directory_errors': 0}
        if root.is_dir():
            def error(exc):
                errors.append({'root': str(root), 'path': str(exc.filename), 'error': str(exc)})
                scope['directory_errors'] += 1
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=error):
                for name in list(dirs):
                    p = Path(directory) / name
                    if p.is_symlink() or p == OUT:
                        dirs.remove(name)
                        skipped.append(str(p))
                for name in files:
                    scope['files_examined'] += 1
                    p = Path(directory) / name
                    if p.is_symlink() or not pattern.search(name):
                        continue
                    try:
                        s = p.stat()
                        context = str(p).lower()
                        if 'd053' in context or 'validation' in context or 'blinded_quality_review' in context:
                            classification = 'KNOWN_DEVELOPMENT_CONTEXT_NAME_NOT_TEST_COVERAGE'
                        elif p.name.lower().startswith('test_') or p.suffix.lower() == '.py':
                            classification = 'CODE_OR_UNIT_TEST_NAME_NOT_REFERENCE_APPROVAL'
                        elif 'matched_completion' in context or 'matched_family' in context:
                            classification = 'KNOWN_OFFLINE_PREPARATION_NAME_NOT_REFERENCE_PAYLOAD'
                        else:
                            classification = 'UNOPENED_CANDIDATE_REQUIRES_METADATA_PROVENANCE'
                        items.append({'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
                                      'classification': classification, 'contents_opened': False,
                                      'archive_members_opened': False})
                    except OSError as exc:
                        errors.append({'path': str(p), 'error': str(exc)})
        scopes.append(scope)
    return {'audit_date_local': '2026-10-25', 'method': 'Exhaustive filename/stat traversal within listed roots; no file-content search, candidate payload reads or archive-member inspection.',
            'scope': scopes, 'errors': errors, 'skipped_directories': skipped,
            'candidate_count': len(items), 'classification_counts': dict(collections.Counter(x['classification'] for x in items)),
            'candidates': items, 'test_payloads_opened': 0, 'external_accounts_searched': False,
            'conclusion': 'No independently approved 100-test reference binding established by metadata available to this supplement. Filename inventory cannot establish global absence or semantic validity.'}


def main():
    assert not OUT.exists() and not ARCHIVE.exists(), 'Versioned output already exists; preserve it'
    OUT.mkdir(parents=True)
    proof = [verify_manifest(OLD / name) for name in ('PORTABLE_MANIFEST.json', 'PREFLIGHT_MANIFEST.json')]
    oldzip = ROOT / 'outputs/cloud_handoff/LED_LARGE_MATCHED_COMPLETION_V2_OFFLINE_REVIEW.zip'
    assert sha(oldzip) == '06e0bdb1da82941c9a4638ad4db03b43c528e37bb7d372efe5f5491c6b8cf73b'
    with zipfile.ZipFile(oldzip) as z:
        assert z.testzip() is None
    # Read frozen TRAIN/validation fixture only; no actual test rows.
    fixture = read(OLD / 'SHUFFLED_FIXTURE_RESULTS.json')
    assert all(m['split'] in ('train', 'validation') for m in fixture['mappings'])
    stats = {'scope': '22 existing TRAIN/validation fixtures only; no held-out mapping or feasibility claim',
             'policy_sha256': fixture['policy_sha256'], 'overall': fixture_statistics(fixture['mappings']),
             'per_split': {s: fixture_statistics([m for m in fixture['mappings'] if m['split'] == s]) for s in sorted({m['split'] for m in fixture['mappings']})},
             'actual_test_category_match_rate': None, 'actual_test_fallback_rate': None,
             'actual_test_donor_concentration': None, 'actual_test_unavailable_strata': None,
             'hard_stop': 'No eligible donor for any positive-caption recipient blocks production; preserve unavailable split/count/category and budget-rejection evidence, never alter policy or resample.'}
    write('SHUFFLED_FIXTURE_STATISTICS.json', stats)
    audit = inventory()
    write('REFERENCE_METADATA_INVENTORY.json', audit)
    write('REFERENCE_COVERAGE_APPROVAL_INDEX.json', {
        'expected_test_businesses': 100, 'verified_unique_test_coverage': None,
        'verified_independently_approved_test_coverage': None, 'missing_test_count': None,
        'status': 'BLOCKED_REFERENCE_BINDING_NOT_ESTABLISHED_NOT_GLOBAL_ABSENCE',
        'payloads_opened': False,
        'known_exclusions': [{'artifact': 'D-053 approved references', 'scope': '24 validation businesses', 'usable_as_test_gold': False},
                             {'artifact': 'Original Teacher train/validation targets', 'usable_as_test_gold': False},
                             {'artifact': 'Blank synthetic annotation packets', 'usable_as_independent_reference': False}],
        'required_metadata_before_claiming_coverage': ['reference_version', 'split=test', 'expected/unique/approved/unresolved counts',
            'source_manifest_sha256', 'reference_payload_sha256', 'independent_author_ids and authorship_dates',
            'approval_receipt_sha256 and approver', 'prediction_blinding/provenance attestation'],
        'declared_metadata_rows': [], 'researcher_supplied_dedicated_index': False,
        'limitations': 'No external account inventory. Candidate filenames are not proof of approval; no unauthorized payload opened. Unknown is not zero.'})
    write('ACCEPTED_V2_PRESERVATION_RECEIPT.json', {'review_date_local': '2026-10-25',
        'checks': proof, 'accepted_zip_sha256': sha(oldzip), 'zip_crc_pass': True,
        'accepted_code_or_artifacts_modified': False, 'F_checkpoint_initialization_unchanged': True,
        'F_preservation_evidence': 'Embedded accepted V2 LOCAL_PRESERVATION.json records all 50 F components checked; current supplement rechecks selected F750 11 files below.',
        'selected_F750': verify_manifest(OLD / 'HISTORICAL_CHECKPOINT_FILES.json'),
        'new_GPU_operations': 0, 'new_training_updates': 0, 'new_generation_calls': 0})
    for short in ('RO', 'U'):
        dest = OUT / f'{short}_GATE_ONLY_SIGNOFF'
        dest.mkdir()
        for src, name in [(OLD / f'{short}_GPU_AUTHORIZATION_READINESS.json', 'READINESS.json'),
                          (OLD / f'PENDING_APPROVALS/{short}_GATE_ONLY_PENDING.json', 'APPROVAL_PENDING.json')]:
            shutil.copyfile(src, dest / name)
        a = read(dest / 'APPROVAL_PENDING.json')
        assert a['GPU_feasibility_approved'] is False and a['training_authorized'] is False
        write(f'{short}_GATE_ONLY_SIGNOFF/SIGNOFF_CHECKLIST.json', {
            'status': 'READY_FOR_RESEARCHER_SIGNOFF_NOT_EXECUTABLE_APPROVAL', 'binding': a['binding'],
            'pending_fields': ['explicit gate-only approval', 'actual UTC authorization and deadline <=48 hours',
                'absolute researcher stop UTC', 'current >=14-day evaluation/writing/defense reserve confirmation',
                'dedicated RTX3090 execution host'],
            'scope': {'discarded_successful_updates': 3, 'TRAIN_technical_generation_calls': 2,
                      'training_updates': 0, 'test_access': False, 'auto_train': False, 'retry': False},
            'host_checks_required': ['accepted V1/V2 chain and exact bindings', 'full pinned parent/model/tokenizer and F750 preservation hashes',
                 '90 CPU tests (26+35+29) per condition on pinned host', 'immutable condition data/config and fresh namespace',
                 '456 finite trainable gradients, bounded AMP/state restore, >=15% conservative VRAM headroom'],
            'after_gate': 'Export PASS/FAIL evidence; STOP. Separate explicit training authorization and reviewed accepted-gate handoff required.'})
    adjudication = read(OLD / 'SHUFFLED_ADJUDICATION_PENDING.json')
    adjudication.update(review_recommendation='CONDITIONAL_ACCEPTANCE_IS_NOT_RESEARCHER_POLICY_ACCEPTANCE',
        researcher_explicit_acceptance=None, researcher_accepted_at_utc=None,
        with_replacement_not_one_to_one=True, test_source_mapping_authorized=False,
        required_production_statistics=['category_match_rate', 'fallback_rate', 'donor_reuse_counts',
            'maximum_donor_share', 'assignment_HHI', 'effective_donor_count', 'unavailable_strata_and_budget_rejections'],
        actual_test_feasibility='UNVERIFIED', policy_file_sha256=sha(OLD / 'SHUFFLED_POLICY_PROPOSED.json'))
    write('SHUFFLED_POLICY_ADJUDICATION_PENDING.json', adjudication)
    shutil.copyfile(OLD / 'SHUFFLED_POLICY_PROPOSED.json', OUT / 'SHUFFLED_POLICY_PROPOSED_UNCHANGED.json')
    write('HUMAN_COORDINATION_PENDING.json', {
        'coordinator': 'Researcher; actual enrollment unconfirmed', 'actual_defense_date': None,
        'actual_source_access_and_sampling_authorization': False,
        'annotators': [{'slot': x, 'is_confirmed_real_annotator': False, 'identity_attestation': None,
                        'independence_conflict_disclosure': None, 'availability_hours': None,
                        'ready_date': None, 'consent_and_source_privacy_attestation': None,
                        'assigned_output_rating_sets': 120, 'completed_sets_verified_here': 0} for x in ('A01', 'A02', 'A03')],
        'businesses': 30, 'raw_conditions_per_business': 4, 'rating_sets_total': 360,
        'six_dimension_fields': 2160, 'structural_RO_visual_NA': 90, 'expected_numeric_fields': 2070,
        'actual_ratings_created': 0, 'fixture_packets_are_not_enrollment_or_annotation': True,
        'communication_to_annotators_sent': False, 'actual_schedule_confirmed': False})
    write('CRITICAL_PATH_PENDING.json', {
        'planning_date_local': '2026-10-25', 'actual_defense_date': None,
        'minimum_protected_days': 14, 'latest_family_freeze': 'Actual defense date minus at least14 days; unresolved until actual date confirmed',
        'absolute_dates_computed': False, 'primary_blockers': ['reference coverage/provenance unknown',
            'three real annotators and workload availability unconfirmed', 'actual defense date unknown',
            'Shuffled exact policy pending and held-out donor feasibility unverified',
            'RO/U host gates and separately authorized training not executed'],
        'fallback_is_researcher_decision': True, 'base_fallback_does_not_solve_reference_or_annotation_dependencies': True})
    doc('README_VI.md', '''# Matched LED-large: hồ sơ ký duyệt và các phụ thuộc đánh giá

Đã giữ nguyên gói V2 được chấp nhận. Supplement này chỉ kiểm kê metadata, xác minh hash và chuẩn bị hồ sơ quản lý; không đổi code/config khoa học, không chạy GPU, không mở payload held-out.

**A — RO/U:** hai thư mục `RO_GATE_ONLY_SIGNOFF` và `U_GATE_ONLY_SIGNOFF` có binding chính xác, template chưa ký và checklist. Mỗi gate chỉ 3 update bỏ đi + 2 technical traces TRAIN, rồi STOP; không cấp quyền training.

**B — Shuffled:** `SHUFFLED_POLICY_ADJUDICATION_PENDING.json` vẫn PENDING. Đề xuất là caption reassignment có hoàn lại, không phải permutation một-một. `SHUFFLED_FIXTURE_STATISTICS.json` chỉ mô tả fixture TRAIN/validation, không chứng minh khả thi trên TEST.

**C — Reference:** chưa xác lập binding reference độc lập được duyệt cho đủ 100 test businesses. Không được hiểu là đã chứng minh không tồn tại reference. Xem index coverage/provenance và kiểm kê metadata; không có payload test nào được mở.

**D — Human/schedule:** chưa xác nhận ba annotator thật và ngày bảo vệ. Có kế hoạch 360 output-rating sets, nhưng không có rating hoặc người tham gia giả. Những trường này giữ PENDING; không tự ghi xác nhận dự trữ 14 ngày.

Gói V2 nguyên bản được đính kèm tại `ACCEPTED_V2.zip` để reviewer xem code, config, 50-file manifest, CPU evidence và các fixture. Model weights, parent workspace và V1 overlay vẫn là dependency ngoài ZIP. Đây không phải archive triển khai GPU độc lập. Dừng để researcher review và ký riêng gate RO/U.
''')
    doc('GATE_ONLY_EXECUTION_AND_SIGNOFF.md', '''# Separate gate-only signoff: frozen V2 implementation

No new runner, configuration or numerical setting is introduced. Common selection remains epoch3/step750 for RO/U/F, informed by earlier F development and fixed before RO/U outputs. Preserve Original CE F and all checkpoints; fresh RO/U initialize from the pretrained backbone, never F.

Each signoff folder contains an exact byte copy of the accepted unsigned V2 approval and readiness receipt. Actual researcher approval text, UTC timestamp, <=48h deadline/absolute stop, current >=14-day reserve and dedicated RTX3090 host must be supplied before execution. Current review acceptance grants none of these execution rights. RO and U require separate approvals and fresh namespaces; no namespace reuse after a failed attempt.

Stage the unchanged V1 overlay and original ZIP, V2 files with their relative paths, complete pinned DecoderPolicyV2 parent/model/tokenizer/runtime, and F750 full state listed by the accepted manifest. Selected F is hash-checked only. Run the accepted host preflight separately per condition: all90 CPU tests (26+35+29), full integrity, source/label/mask/config bindings. Local historical PASS is not host PASS.

```sh
# CPU ONLY; example host paths, must match staged workspace.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/run_led_large_rou_gate_only_v2.py preflight --condition reviews_only --parent-root /workspace/LED_LARGE_DECODER_POLICY_V2_WORKSPACE --receipt /workspace/RO_HOST_CPU.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/run_led_large_rou_gate_only_v2.py preflight --condition unfiltered --parent-root /workspace/LED_LARGE_DECODER_POLICY_V2_WORKSPACE --receipt /workspace/U_HOST_CPU.json
```

After explicit authorization only, use the accepted runner's `feasibility --condition ... --parent-root ... --approval <fresh-signed-file> --execute`. The runner itself reruns CPU preflight, binds the receipt to approval, then launches the gate worker. PENDING files intentionally fail. Capture complete launcher stdout/stderr. See embedded V2 RO_U_GPU_READINESS.md for original command examples and exact staging dependencies.

Each gate: exactly3 successful discarded native-CE updates, inherited24 nominal/<=88 actual microbatches including bounded AMP recovery, exactly2 locked TRAIN beam4 technical traces, finite456 LoRA tensors, state restoration and >=15% conservative measured VRAM headroom. Stop on integrity, numerical/resource/hardware/timing failure; no fallback, retry or altered settings. Preserve successful-step/scaler/memory evidence and technical raw tokens/text; export failure evidence even on partial completion.

After either PASS or FAIL, STOP for independent review. Gate PASS is not training authorization. The old V1 train CLI expects a same-binding/same-approval gate; do not relabel or copy a V2 gate report to satisfy it. A narrowly reviewed accepted-gate authorization handoff is still required before separately approved full training. No additional gate repetition is implicitly authorized. Final test inference and source-only mapping require their own approvals.
''')
    doc('SHUFFLED_POLICY_REVIEW.md', '''# Exact Shuffled policy remains pending

Canonical policy SHA-256 is recorded in the adjudication receipt. The byte-identical proposed JSON is attached; researcher acceptance must cover the exact allowlist and rules, not merely a policy name.

- Tier1: at least one exact normalized allowlisted cuisine/category overlap; only fixed NFKC/trim/casefold, no semantic matching.
- Same split, no self-match, identical positive Filtered caption count, all fixed source/review/caption budgets. Include complete visual-scaffolding token increment, not captions alone.
- Only when no eligible Tier1 donor exists: same-split/same-count budget-feasible fallback. No new heuristic.
- Deterministic locked seed42/hash ordering; donor reuse permitted. Zero-caption recipients unchanged; no feasible donor causes whole-mapping hard stop.
- This is WITH-REPLACEMENT reassignment, NOT one-to-one or distribution-preserving permutation. Reuse changes caption frequency/effective diversity; disclose this limitation in F-versus-Shuffled interpretation.

Fixture statistics use positive-caption recipients as denominators. Report exact-category match and fallback rates separately, per-split and overall donor-use counts, maximum assignment share, HHI=sum(p_d^2), effective donor count=1/HHI, unavailable split/count/category strata and budget-infeasible edges. Assignment shares are not caption-token shares. The existing fixture happened to reuse no donor; the policy still permits reuse. Absent fixture strata and actual held-out feasibility remain unknown.

Before any actual TEST mapping: obtain exact policy acceptance AND separate source-only mapping authorization bound to policy/source metadata. Do not open test payloads or create mappings under this supplement. Lock one mapping before model outputs; no test-output-dependent resampling or unavailable-stratum repair. A no-donor result requires stop/researcher review and applies to both base and large families.
''')
    doc('INDEPENDENT_TEST_REFERENCE_STATUS_AND_COMPLETION.md', '''# Independent test-reference status: binding NOT ESTABLISHED

The new filename/stat inventory removes the prior200,000-file cap within the listed local roots. Review its exact scope/errors/skips. It reads no candidate payloads, archive members, test IDs or annotations. Approved/reference words in a filename do not prove100-business coverage, independence or approval. D-053 is a known24-business validation artifact and cannot be substituted. The original Teacher supervision also cannot serve as independent test gold. No external/cloud account is searched.

The coverage/approval index deliberately leaves verified test coverage, approved count and missing count NULL. Reporting zero would falsely imply a global absence audit; reporting100 would fabricate coverage. Current evidence is sufficient to block final reference-based scoring, not to assert that no references exist.

Researcher action: locate a dedicated metadata-only index/approval receipt or ask its custodian for one. Needed: reference version, test split, expected100 unique businesses, covered/approved/unresolved counts, source and payload hashes, independent author identities/dates, approval identity/receipt/hash and prediction-blinding attestation. Do not put actual reference text into this metadata handoff. Dedicated metadata can establish declared coverage/provenance; actual source support remains separately reviewable.

If coverage is incomplete, obtain separate source-only access authorization first. An independent coordinator reconciles the locked100-business frame against the authenticated coverage index, retains valid approved references unchanged, and assigns only missing/unapproved cases to independent authors. Authors see only frozen selected reviews and permitted source photos/evidence, never any model predictions, Teacher drafts or D-053 validation references. Keep review-consensus primary gold separate from condition-specific visual assessment; do not make one condition's captions the common review gold.

Use the existing reference methodology/rubric without silently changing it. Record exact source support, disagreements/isolated claims, uncertainty, independent authorship and second-person review; preserve draft/adjudication history and researcher approval. Semantic disputes remain unresolved rather than automatically confirmed. Seal versioned source/reference/approval hashes and a100-case coverage index before final scoring. No reference drafting or source adjudication has been performed here.

Capacity is a real dependency. Let M be genuinely missing/unapproved cases, t_a authored minutes and t_r independent review minutes measured on a small separately authorized source-only practice set. Plan hours=M*(t_a+t_r)/60 plus adjudication/admin, not an invented throughput. Allocate named owners and availability before committing final dates. Never promise completion of100 missing references in three days without evidence. Existing approved references may reduce M only after verified provenance/coverage, not filename inference. Keep the authors and evaluators blind to model outputs during gold creation; disclose any prior prediction exposure.

Hard stop: unknown/insufficient approved coverage, prediction-contaminated authorship, missing support/adjudication or absent source-access approval. Do not substitute Teacher targets, validation gold, synthetic human labels or model-generated references to meet a deadline.
''')
    doc('HUMAN_COORDINATION_AND_QUALITY_CONTROL.md', '''# Real human-study coordination plan — enrollment pending

The researcher must confirm three actual independent annotators (A01/A02/A03 aliases acceptable), availability dates/hours, language/source-reading competence, conflicts/prior model exposure, consent and privacy handling. Slots and blank fixtures are not people. No recruitment messages were sent and no confirmations or ratings are invented. Coordinator and independent-reference author/reviewer responsibilities must also be assigned. Researcher contacts people; no external messaging is authorized here.

Keep the locked representative30-business source-only sample: review-length median halves x Basic usable-photo groups0/1–2/3–5, proportional largest-remainder allocation, seed42 with fixed ID tie rules. No output-based case selection. Actual test sampling needs separate metadata access; none is performed. Four raw selected750 condition outputs per business,120 tasks per rater,360 output-rating sets in total. Preserve every raw output without repair. The accepted V2 helper creates opaque HMAC aliases/order with a fresh private admin secret; condition keys stay separate. Do not send the whole engineering review ZIP or synthetic private key to annotators. Blinding is partial: source absence, RO N/A and Shuffled mismatches may reveal condition.

Use the original six1–5 dimensions: review faithfulness, aspect coverage, coherence, non-redundancy, usefulness and combined visual grounding/contribution. Higher is better. RO visual is N/A (90 structural fields); total2160 fields,2070 expected numeric. Non-RO zero-caption outputs are assessed for appropriate abstention, never rewarded for unsupported visuals. Keep Core/mixed-opinion errors as evidence-linked diagnostic notes, not new outcome scales. All source grounding respects each condition's actual visible reviews/captions and corresponding photos; photos cannot rescue claims absent from model-visible captions. Shuffled donor evidence is not genuine recipient visual evidence.

Coordinate TRAIN-only calibration before final packets: explain rubric, practice and clarify interpretation without inspecting/rating final outputs. Freeze the rubric. Then each annotator works independently without AI/model-assisted suggested scores or discussion of final cases. Offer breaks, privacy-safe source access and technical support; no preference feedback or condition unblinding. Preserve120 unique tasks per raw export, original timestamps/attestations, provenance and checksums. Corrections/adjudication become new files, never overwrite independent raw scores. Missing applicable scores block complete analysis; do not impute, replace annotators or fabricate agreement silently.

Workload: at5/10/15 minutes per output-rating set,120 tasks require10/20/30 hours per annotator respectively, plus TRAIN calibration and breaks. These are planning scenarios, not measured rates or guarantees. Obtain actual availability and source-reading time before booking a four-day annotation window. All three work in parallel; elapsed capacity depends on the slowest available annotator. Record escalation/withdrawal with researcher review instead of manufacturing completion.

Preserve primary review-consensus metrics and supplementary visual evaluation, fixed paired-business bootstrap contrasts and original ordinal Krippendorff agreement. Analyse human criterion means across three raters per business/condition before paired business resampling; publish missing/N/A denominators and per-dimension ordinal agreement, no automatic quality labels. Accepted V2 ANALYSIS_LOCK.json remains authoritative and unchanged. No actual scoring is executed here.
''')
    doc('CRITICAL_PATH_AND_RESEARCHER_DECISIONS.md', '''# Critical path, planning date25 October2026

Actual defense date and three-annotator availability are unconfirmed. Absolute execution/evaluation dates cannot responsibly be computed yet. Let D be the confirmed defense date. Latest completed-family/protocol freeze is no later than D-14days;14days is the minimum protected evaluation/analysis/writing/defense reserve, not a promise that unfinished reference work fits within it. Obtain D and capacity now. Do not reuse historical authorization timestamps or invent the reserve confirmation.

| Sequence | Owner and dependency | Exit / stop |
|---|---|---|
| Immediately, parallel | Researcher: exact Shuffled adjudication, defense date, three real raters and reference custodian; coordinator: authenticated reference metadata | Unknown gold/capacity remains a critical blocker even if GPU gates pass |
| After separate RO/U gate-only signoff | Operator: full pinned-host integrity and90 CPU tests per condition, then two independent locked3090gates |3discardedupdates+2TRAINtraces each, export report and STOP; no auto train |
| After independent gate review and separate training signoff | Operator: reviewed accepted-gate handoff, two independent native-CE fresh5epoch/1250update runs, all10 checkpoints, selected750 | No retries/settings changes or opportunistic checkpoint choice; preserve F |
| Before D-14 at latest | Researcher/coordinator: family and protocols sealed; reference completion capacity credible; accepted donor rules; annotators scheduled | If large-family infeasible, researcher decides whole matched-base fallback. No mixed model sizes |
| Reserve early window, with separate source-only authorization | Independent reference team: reconcile/complete/adjudicate gold; coordinator:30sample, one source-only Shuffled mapping, target-free400-input packing | No donor/reference approval or budget feasibility -> STOP; no outcome resampling |
| After explicit final-inference authorization | Operator:100x4 raw test outputs, integrity backup, blinded packets | No held-out tuning, decoding rescue or checkpoint switch |
| Reserve middle window | Three independent raters:120sets each; analyst: separately authorized pinned metrics and paired bootstrap/ordinal agreement | Complete approved gold and real ratings required; report all failures |
| Reserve final window | Researcher: grounded case tables, results/limitations, supervisor review, thesis, slides and rehearsals | No optional optimization consuming reserve |

Provisional allocation within an already viable14-day reserve: days1–3 source/mapping/reference finalization, day4 inference/packet handoff, days5–8 parallel ratings/metrics, days9–10 analysis, days11–12 writing/review, days13–14 defense preparation/buffer. Revise or extend this schedule if reference/annotation capacity does not support it; do not compress required evidence to force this template. Authors may start earlier after separate authorization and must stay prediction-blind.

Resource planning only: historical F training4.11hours suggests roughly8–12hours for both full trainings plus host setup/gates/backup; RO/U actual gate times/VRAM remain unmeasured. Final400 inference estimate1–3hours is extrapolated from historical F and is not a reservation quote or permission. Use actual gate measurements before committing cost. Rental price/available host/absolute dates are unknown.

Fallback trigger: if confirmed calendar leaves <14 protected days after family freeze, locked gate/training fails without authorized retry, or primary dependencies cannot fit, stop and ask researcher to choose calendar/scope or the complete preserved matched-base family. A smaller model does NOT solve missing independent references, unavailable donors or absent annotators. No automatic fallback GPU work or evaluation is authorized. CAD/Contrastive remain closed, Boundary paused, and no new optimization is introduced.
''')
    doc('REVIEW_REQUEST.md', '''# Decisions required next

1. Sign RO and U gate-only approvals separately, if desired, with current UTC <=48h windows, absolute stops, real reserve confirmation and dedicated RTX3090 host. This package does not sign them or authorize training.
2. Accept/reject the exact canonical Shuffled policy, including allowlist, same-count fallback, with-replacement reuse, zero-caption unchanged, full budget accounting and no-donor hard stop. Source-only TEST mapping requires an additional explicit approval.
3. Supply authenticated100-test reference metadata/provenance or authorize independent source-only completion after coverage reconciliation. Current coverage is UNKNOWN, not fabricated zero or100.
4. Confirm actual defense date and three real independent annotators with availability/conflicts; agree a feasible reference/annotation timetable preserving at least14 days.

Review the accepted V2 code/evidence via the embedded immutable archive. No GPU, test payload, final inference, reference generation, human scores or training was performed in this supplement. STOP for independent review.
''')
    shutil.copyfile(oldzip, OUT / 'ACCEPTED_V2.zip')
    shutil.copyfile(Path(__file__), OUT / 'BUILD_SOURCE.py')
    write('DELIVERY_STATUS.json', {'status': 'OFFLINE_SIGNOFF_READY_DEPENDENCIES_PENDING',
        'date_local': '2026-10-25', 'RO_and_U_gate_approvals': 'PENDING_EXPLICIT_SEPARATE_SIGNOFF',
        'Shuffled_policy': 'PENDING', 'reference_coverage': 'NOT_ESTABLISHED',
        'real_annotators': 'UNCONFIRMED', 'defense_date': None, 'new_GPU_calls': 0,
        'test_payloads_opened': 0, 'new_training_updates': 0, 'automatic_continuation': False,
        'accepted_V2_unchanged': True})
    entries = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write('MANIFEST.json', {'path_base': 'archive_root', 'files': entries})
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for rel, digest in entries.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == digest
    ARCHIVE.with_suffix('.zip.sha256').write_text(sha(ARCHIVE) + '  ' + ARCHIVE.name + '\n', encoding='ascii')
    print(json.dumps({'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'portable_entries': len(entries),
        'files_inventoried': sum(x['files_examined'] for x in audit['scope']), 'inventory_errors': len(audit['errors']),
        'fixture_statistics': stats['overall'], 'preservation': proof}, indent=2))


if __name__ == '__main__':
    main()
