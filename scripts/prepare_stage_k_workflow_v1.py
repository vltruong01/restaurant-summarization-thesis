"""Prepare a synthetic-only Stage K interface and unsigned source-access proposal."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/stage_k_workflow_preparation_v1'
APP = OUT / 'APP'
ZIP = ROOT / 'outputs/cloud_handoff/STAGE_K_SOURCE_ONLY_WORKFLOW_V1_OFFLINE_REVIEW.zip'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def doc(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def main():
    assert OUT.is_dir() and not ZIP.exists(), 'Preserve versioned package'
    old = ROOT / 'outputs/cloud_handoff/TEST_REFERENCE_AVAILABILITY_V1_REVIEW.zip'
    assert sha(old) == 'c0b16d475680b3393d05a012b431021363edc34fec0499cd7b6004230369118c'
    with zipfile.ZipFile(old) as z:
        assert z.testzip() is None
        files = json.loads(z.read('MANIFEST.json'))['files']
        for name, expected in files.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    shutil.copyfile(old, OUT / 'ACCEPTED_REFERENCE_AVAILABILITY_V1.zip')
    write('ACCEPTED_AUDIT_FREEZE_RECEIPT.json', {
        'status': 'FROZEN_RESEARCHER_ACCEPTED', 'accepted_archive_sha256': sha(old),
        'portable_entries_verified': len(files), 'original_archive_modified': False,
        'researcher_acceptance': 'Explicit current researcher message accepts metadata-only availability audit; TEST readiness NOT YET VERIFIED.',
        'acceptance_record_date_local': '2026-10-25', 'actual_message_UTC': None,
        'timestamp_note': 'No actual chat UTC timestamp is exposed; none fabricated.'})
    provenance_path = ROOT / 'outputs/stage_i/test_reference_availability_v1/PROVENANCE_EVIDENCE_INDEX.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for item in provenance['records']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    write('COVERAGE_AND_CUSTODIAN_STATUS.json', {
        'status': 'NOT_YET_VERIFIED', 'expected_test_businesses': 100,
        'verified_coverage': None, 'approved_count': None, 'missing_count': None,
        'authenticated_custodian_index_found': False, 'custodian_identity': None,
        'custodian_question_sent_to_researcher_in_chat': True, 'custodian_reply_received': False,
        'external_messages_sent': False, 'existing_metadata_evidence_reverified': len(provenance['records']),
        'known_VALIDATION_only': {'businesses': 24, 'scope': 'D-053', 'not_TEST_gold': True},
        'source': {'accepted_provenance_index': str(provenance_path.relative_to(ROOT)), 'sha256': sha(provenance_path)},
        'interpretation': 'No authenticated TEST index established in reviewed artifacts; unknown is not zero. No replacement generation authorized.'})
    write('CUSTODIAN_COVERAGE_INDEX_TEMPLATE_PENDING.json', {
        'schema': 'METADATA_ONLY_TEST_REFERENCE_COVERAGE_V1', 'status': 'UNFILLED_NOT_EVIDENCE',
        'split': 'test', 'expected_businesses': 100, 'reference_version': None,
        'unique_covered_businesses': None, 'approved_businesses': None, 'unresolved_businesses': None,
        'membership_manifest_sha256': None, 'selected_review_source_manifest_sha256': None,
        'approved_original_photo_scope_manifest_sha256': None, 'reference_payload_sha256': None,
        'writer_provenance_ledger_sha256': None, 'independent_checker_provenance_ledger_sha256': None,
        'model_output_blinding_attestation_sha256': None, 'approval_receipt_sha256': None,
        'custodian_alias': None, 'authorship_dates': None, 'approval_identity_and_date': None,
        'payload_text_included': False, 'actual_business_ids_included': False})
    doc('CUSTODIAN_REQUEST_TEMPLATE.md', '''# Metadata-only request — not sent externally

Do you hold previously written independent Yelp TEST references? Please provide a dedicated metadata-only index or approval receipt with:

- Exact split and reference version; expected100, unique covered/approved/unresolved counts.
- Membership, fixed selected-review and approved original-photo source manifest hashes; reference payload hash.
- Human writer/checker identities or aliases, authorship dates, independence/conflict disclosure and output-blind attestations.
- Researcher approval identity/date/receipt and ledger hashes.

Do not send TEST input text, reference text, model predictions or photos in this first handoff. Do not reconstruct or generate missing references merely to fill the index. A custodian's claim must be authenticated to original artifacts and approval provenance, not accepted from a filename. Existing valid references stay unchanged; unknown counts remain unknown. This template is for researcher/custodian coordination; no outbound message was sent.
''')
    bindings = {p.relative_to(APP).as_posix(): sha(p) for p in sorted(APP.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write('PRACTICE_IMPLEMENTATION_BINDINGS.json', {'scope': 'SYNTHETIC_ONLY_NOT_PRODUCTION_TEST_READER', 'files': bindings})
    write('SOURCE_ONLY_TEST_AUTHORIZATION_PENDING.json', {
        'experiment': 'STAGE_K_SOURCE_ONLY_REFERENCE_PREPARATION_V1', 'status': 'PROPOSED_UNSIGNED',
        'researcher_explicit_authorization': None, 'source_access_approved': False,
        'human_reference_authoring_approved': False, 'authorized_at_utc': None, 'stop_at_utc': None,
        'actual_business_membership_manifest_sha256': None, 'fixed_selected_review_manifest_sha256': None,
        'approved_original_photo_scope_manifest_sha256': None,
        'authenticated_existing_coverage_index_sha256': None,
        'confirmed_writer_aliases': [], 'confirmed_independent_checker_aliases': [],
        'reference_guideline_sha256': None,
        'reference_guideline_binding_note': 'Guideline hash is populated after the proposed guideline file is written below. Source scope and production loader remain unbound.',
        'proposed_scope': ['Authenticate/reconcile existing metadata-only100-business coverage',
            'After explicit source approval, read only locked TEST membership, fixed selected original reviews and approved original-photo scope',
            'Prepare source-only writer/checker packets; exclude every model output and Teacher target',
            'After explicit authoring approval, real humans draft/check/correct only missing or unapproved references',
            'Record actual researcher decisions and versioned freeze; do not auto-approve'],
        'forbidden': ['Model prediction access', 'Teacher/API generation', 'GPU operations', 'Final inference/scoring',
            'Actual Shuffled donor mapping', 'Cohort/review/photo/caption policy changes', 'Validation-gold substitution',
            'Outcome-driven reference changes', 'Automatic replacement of existing approved references'],
        'expected_businesses': 100, 'maximum_population': 100, 'actual_authoring_subset': 'Only missing/unapproved coverage after authenticated reconciliation; not presumed100',
        'source_scope_unresolved': True, 'production_loader_review_required': True,
        'practice_app_can_execute_this_contract': False, 'automatic_continuation': False})
    write('HUMAN_RESOURCE_READINESS_PENDING.json', {
        'status': 'PENDING_REAL_PEOPLE_AND_MEASURED_WORKLOAD',
        'writer_slots': [{'alias': 'W01', 'real_identity_confirmed': False, 'availability_hours': None, 'prior_prediction_exposure': None}],
        'independent_checker_slots': [{'alias': 'C01', 'real_identity_confirmed': False, 'independent_of_writer': None, 'availability_hours': None, 'prior_prediction_exposure': None}],
        'researcher_approver': 'Actual researcher; approval actions not yet performed',
        'roles_are_enrollment_placeholders_not_fabricated_authors': True,
        'custodian': None, 'missing_count_M': None, 'measured_draft_minutes_per_case': None,
        'measured_check_minutes_per_case': None, 'measured_adjudication_minutes': None,
        'estimated_total_person_hours': None, 'productivity_measurement_completed': False,
        'actual_TEST_references_drafted': 0, 'independent_human_checks_claimed': 0})
    doc('README_VI.md', '''# Stage K — giao diện thực hành và đề xuất quyền truy cập nguồn

Audit availability V1 đã được đóng băng, đính kèm nguyên bản và xác minh9/9SHA-256. Chưa có metadata index TEST xác thực từ custodian: **NOT YET VERIFIED**, coverage/approved/missing đều chưa biết, không phải0.

Giao diện `APP` dùng đúng hai case giả lập và một hình minh họa tự tạo. Nó hỗ trợ nguồn → draft → independent check/correction → researcher approval → versioned practice freeze; không có loader TEST hoặc dự đoán. Các trường ban đầu trống. Ledger append-only, hash-chain; sửa draft làm hết hiệu lực check/approval của phiên bản cũ. Alias/role thực hành không phải người tham gia thật hoặc annotation đã duyệt.

Xem `RUN_PRACTICE_AND_PROVENANCE.md` để mở localhost bằng Python. `SOURCE_ONLY_TEST_AUTHORIZATION_PENDING.json` là đề xuất chưa ký, không thực thi được bằng app này. Phạm vi nguồn thật, người viết/checker, coverage cũ và production loader còn phải được xác nhận trước khi researcher cấp quyền.

Giữ đúng Stage K: REVIEW_CONSENSUS từ selected original reviews; VISUAL_EVIDENCE từ ảnh gốc đã duyệt, không giới hạn bởi caption BLIP-2. Đánh giá grounding theo evidence mô hình nhìn thấy là bước riêng. Không thay dataset, Teacher, checkpoint hoặc metric protocol.

Kiểm thử CPU/JS và HTTP preview có receipt riêng. Browser rendering chưa xác minh vì lệnh Chrome headless bị automatic approval review chặn. Không có nguồn TEST, reference thật, score hoặc GPU nào được thực hiện. Dừng để review.
''')
    doc('STAGE_K_WORKFLOW_AND_REFERENCE_GUIDELINE.md', '''# Proposed Stage K reference workflow

## Reconcile before writing

First authenticate any custodian's metadata-only coverage index using original manifest/version/receipt and human authorship/blinding provenance. Current status NOT YET VERIFIED is not proof that100references are missing. Keep valid existing approved references unchanged. Before drafting, resolve the unique missing/unapproved subset against the frozen100-business membership after separately authorized source access. An index alone does not prove semantic support; suspicious provenance or duplicate/mismatched identities blocks acceptance. Do not open held-out payloads under this preparation instruction.

## Scope and two-section content

Use exact frozen selected original reviews for `[REVIEW_CONSENSUS]`. Important business-specific themes and positive/negative disagreement on the same aspect must be considered; isolated reports must not become consensus, and customer opinions must not be asserted as independently verified business facts. Cite exact review IDs/offset quotations; do not invent explanations or combine unrelated accounts into causal relationships. Preserve coherent useful synthesis without a new length target or forcing every minor detail into the reference.

Use only the researcher-approved original-photo scope for `[VISUAL_EVIDENCE]`, independently of BLIP-2 caption presence or quality. Do not silently pick a new photo subset; original-photo scope must be explicitly bound before source access. Write directly observable useful information, cite image IDs, keep uncertainty and avoid unsupported object identities, ingredients, processes or relationships. Missing BLIP captions do not imply absent human visual gold. No useful/available photo evidence may justify an empty visual section with a source-derived reason under the existing protocol. Photo availability must be established from approved sources, never inferred from a caption count.

The reference has exactly the original two section headers. Practice software inserts them once; bodies cannot inject extra headers. Evidence/notes are separate metadata, not new scoring sections. No new Core weighting, salience optimization, metric denominator or evaluation criterion is introduced. Existing D-053 development references/Teacher targets are never supplied as TEST gold or authoring drafts.

## Human stages and revision dependencies

1. Source coordinator binds immutable membership, selected reviews and approved original-photo packet hashes. Packets contain no evaluated outputs, metrics, checkpoint preferences, condition labels, Teacher pseudo-targets or suggested answers. Production source access remains unapproved.
2. Real writer W01 attests output blindness and creates a draft with review quotations and inspected-photo IDs. Actual human content judgments remain attributable to W01; software only checks structure and quotation locations.
3. Different real checker C01 independently inspects full original sources and the exact draft revision. Record supported claims, omissions, opinion balance, uncertainty and exact source-supported correction requests. No overwritten writer text or automatic semantic pass.
4. Writer makes corrections as a new immutable revision. Old drafts/checks stay in history; any edit invalidates dependent current check and approval. Checker checks the new revision explicitly. Disagreements go to actual researcher adjudication; unresolved blocking support issues prevent freeze.
5. Researcher reviews current accepted check and source-supported reference, explicitly records completeness/support and blinding provenance, and approves that revision only. No inferred approval from a checkbox inserted by software or an alias.
6. Freeze versioned two-section references plus source manifest, writer/checker/adjudication ledgers, guideline, unresolved/excluded notes, actual approval receipt and SHA-256 manifest. Preserve all prior versions and independent raw decisions. Final completion needs100unique approved source-bound businesses, no unresolved blocking claim and authenticated human provenance. Do not silently omit cases or fabricate absence/approval counts.

The practice app uses synthetic role codes only; it cannot prove real identity, independence, honest attestation or source semantics. Before production, researcher must assign actual people, review the production identity/access/binding mechanism and explicitly authorize source-only access and human authoring. Existing approved references must be imported as preserved versions only after provenance reconciliation, not regenerated by practice software.

## Two distinct evaluation questions

Independent human reference quality/coverage uses original selected reviews and approved original photos. Separately, model-visible grounding checks whether a model assertion was recoverable from its condition's actual reviews/captions and corroborated by the appropriate photo. A true photo-only detail in human gold does not retroactively ground an unsupported model assertion. RO visual scoring stays N/A; zero-caption visual assertions remain unsupported. Shuffled donor evidence does not become genuine recipient-business gold. Keep primary review-consensus metrics and supplementary visual evaluation as already locked. None are computed here.

## Freeze and stop rules

Freeze references before final scoring and keep reference authors blind to all model predictions throughout creation, including when an operator later has separately generated outputs. Do not correct gold in response to a model's score or missing claim. Any unavoidable source-supported correction is a new researcher-approved erratum applied consistently. Stop for missing source authorization, ambiguous photo scope, unverifiable old coverage, writer/checker conflict, compromised blindness, unsupported claim, missing approval or hash mismatch. No automatic replacement, model API, Teacher call or generated substitute is allowed.
''')
    doc('RUN_PRACTICE_AND_PROVENANCE.md', '''# Local synthetic practice

Use standard Python3.10+; SQLite and HTTP modules are in the standard library. No model, CUDA or third-party dependency is needed. From the project root:

```powershell
# Read-only preview: stores no drafts or approvals.
python outputs/stage_i/stage_k_workflow_preparation_v1/APP/workflow.py --preview --port 8765
```

Open http://127.0.0.1:8765 in a browser. Ctrl+C stops the server. For an interactive synthetic rehearsal, choose a NEW ledger file in an existing directory:

```powershell
python outputs/stage_i/stage_k_workflow_preparation_v1/APP/workflow.py --ledger outputs/stage_i/stage_k_workflow_preparation_v1/practice_session_001.sqlite --port 8765
```

Existing ledger paths are refused; never overwrite an earlier practice record. The terminal prints separate ephemeral writer/checker/researcher practice codes. Paste the applicable code into the interface. Select exact review text and click Cite, inspect/cite image evidence, save a draft, then use the checker code to accept or request corrections and the researcher code to approve. Both cases need current accepted checks/approval before a practice freeze. Export preserves all events and source bindings. All role IDs and decisions are explicitly synthetic, even if a real person clicks them. They do not count as enrollment or TEST annotations. No rehearsal actions are prefilled/delivered as actual human decisions.

Practice source files are SHA-256-bound and the session stops if their binding changes. Sources, drafts, checks and approvals are separate records. SQLite UPDATE/DELETE triggers enforce append-only edits through the application; each event hashes canonical payload+previous hash with source binding, actor, action and actual recording UTC. Revision hashes tie checks/approvals to the exact content. Export the ledger and freeze as new files, retain the SQLite original, hash both and give anchor copies to an independent custodian before any future real use. Never share role codes in final study evidence.

**Limits:** this is tamper-evident local practice, not a cryptographically authenticated human-signature or tamper-proof archive. An administrator with database/code control can drop triggers and rebuild a chain; independently retained anchors and identity/access controls are necessary for production. Practice codes distinguish roles, not real people. Browser-local unsaved text is not a recorded draft. After this preparation, an actual production loader, real identity assignments, restart/import policy, source binding and reference approval contract must be reviewed before researcher-authorized TEST use; editing `split` or replacing the fixture is not an authorization route. The current loader rejects TEST/non-synthetic packets.

No network listening beyond loopback, no CORS, session nonce plus per-role code on mutations. This is a local rehearsal tool, not a public multi-user service; do not expose the port to the internet. User authentication/privacy arrangements for actual independent participants remain pending researcher coordination. Nothing here authorizes uploading actual sources to an external service.
''')
    doc('HUMAN_RESOURCE_AND_WORKLOAD_PLAN.md', '''# Real people and workload — pending confirmation

Researcher has been asked whether a reference custodian exists and whether two real people can fill W01/C01. No reply confirming them is available at package creation. Placeholders are not recruited people. Do not message others without explicit instruction. Researcher controls the calendar; no further defense-date request or invented deadline is introduced by this package. Existing future GPU/time contracts are not altered or auto-confirmed.

Minimum reference team: real human writer; different independent human checker; actual researcher to adjudicate/approve; coordinator/custodian for source access and immutable versions. Confirm aliases/identities privately, competence for source reading and photo uncertainty, conflicts/relationship to the project, prior evaluated-output exposure, output-blinding and data-handling attestations, availability hours, and responsibility for each final version. Writer/checker cannot be the same person. Three independent output-rating annotators are a later, separate dependency; these roles and the synthetic practice actors do not establish that study's enrollment.

Before estimating labor, authenticate existing coverage and determine M genuinely missing/unapproved cases. Use the provided synthetic cases for interface/rubric rehearsal without viewing TEST. Record actual human times for reading/writing/checking/corrections, task difficulty and exclusions. Synthetic illustration/simple text timings validate workflow familiarity only; they do NOT establish realistic100-business productivity. After source-only access is approved, measure a small fixed initial source-only batch without seeing predictions and update capacity transparently. These cases remain part of the fixed cohort; no case replacement or outcome-based batch selection.

Let t_w and t_c be measured mean minutes for actual-scale drafting and independent checking, plus total adjudication/admin hours H_a. Proposed person-hours = M*(t_w+t_c)/60+H_a. Writer load=M*t_w/60; checker load=M*t_c/60. Elapsed duration depends on available hours per person and sequential checking dependencies; do not divide combined work by an imaginary team or assume all work is parallel. Current M,t_w,t_c,H_a and availability are UNKNOWN, so the numerical workload estimate remains UNVERIFIED. Report measured range/uncertainty, not fabricated throughput. Preserve real timings separately from synthetic software test elapsed times.

Readiness exit: authenticated coverage subset, confirmed distinct writer/checker and actual researcher, agreed source scope/guideline, explicit source-only/authoring permission, private assignment/blinding plan, measured credible capacity and retention/access controls. Any missing dependency blocks claims that reference construction is ready to execute.
''')
    doc('PROPOSED_SOURCE_ONLY_ACCESS_REQUEST.md', '''# Proposed researcher authorization — NOT GRANTED

Request a bounded Stage K source-only authorization after independent review of this preparation:

1. Reconcile any custodian's authenticated metadata-only index with the fixed100-business test membership. Covered/approved/missing counts are unknown until verified; preserve valid existing approved references unchanged.
2. Authorize access only to that locked membership, already fixed selected original reviews and explicitly approved original-photo scope. Bind exact source manifests/versions/hashes before payload access. If the photo scope or selected-review artifact is unresolved, stop for that decision. Do not read alternate review selections, model outputs, Teacher targets or validation gold as reference inputs.
3. Prepare source-only packets and, if explicitly approved, permit actual W01/C01 humans to draft/check/correct only the reconciled missing/unapproved references. Scope ceiling is the same100businesses; no new cohort, reference population or resampling. Researcher must approve each final version; authoring authorization is not automatic content approval.
4. Confirm real writer/checker identities/independence/blinding and private source access; review/bind a production-capable loader and identity mechanism before use. Current synthetic code is not a production reader and cannot execute the pending receipt. Do not bypass it by changing fixture flags.
5. Preserve review and photo-only visual reference sections, independent evidence ledger, revisions, human decisions and final researcher-approved hashes. Keep model-visible-caption grounding separate.

Excluded: any model prediction access by reference team, GPU/model inference/training, Teacher/API generation, test scoring, Shuffled donor mapping, source/caption/selection changes, gold substitution or automatic continuation. No explicit approval, exact source binding or real human confirmation is supplied by this package. `SOURCE_ONLY_TEST_AUTHORIZATION_PENDING.json` remains unsigned. STOP for the researcher decision before any actual TEST source payload is opened or reference is written.
''')
    doc('EVALUATION_CRITICAL_PATH.md', '''# Parallel readiness without new execution

Researcher manages the calendar. Milestones/dependencies below do not invent dates, productivity or a reserve confirmation.

| Track | Preparation now | Next explicit permission / exit |
|---|---|---|
| Independent references | Freeze accepted audit; request authentic custodian index; source-only synthetic interface; confirm real W01/C01 | Source manifests/photo scope, real people and reviewed production loader, then explicit TEST source access/authoring authorization |
| RO/U | Accepted separate gate-only signoff packages remain unchanged | Fresh RO/U gate-only approvals; full pinned-host integrity+90CPUtests per condition;3discardedupdates+2TRAINtraces; STOP for review; full training separately gated |
| Shuffled | Exact proposed policy remains pending; with-replacement reuse/concentration and no-donor limitations preserved | Researcher policy adjudication, then separate source-only TEST mapping permission; no outcome-based resampling |
| Human output study | Locked30representativebusinesses,3independentrater plan,120tasks each; private condition key separate | Real raters/capacity, separately authorized source sampling, final raw outputs; no synthetic scores |
| Final inference/metrics | Locked review-consensus primary metrics, supplementary visual evaluation, paired bootstrap and ordinal agreement unchanged | Matched selected750family, complete authenticated independently approved references, accepted source-only mapping and separate final inference/scoring authorization |

Reference metadata/human recruitment and source-scope decisions can proceed in parallel with gate readiness. Do not wait for model outputs to write or revise gold. Reference authors never see evaluated outputs; output-rating annotators receive them only in the separate blinded rating phase. Final gold must be frozen before scoring. Missing references are an evaluation blocker regardless of model size; a base fallback cannot solve author availability, provenance or donor feasibility.

Actual GPU gate results determine measured resource/time estimates for RO/U; current preparation does not execute them. When committing actual execution, retain the existing protected evaluation/writing/defense reserve and mandatory approval checks. If capacity/provenance or source scope cannot be resolved, stop for a researcher calendar/scope decision, not fabricated references, simplified gold, model-size mixing or extra optimization.
''')
    # Correct guideline binding after its real file has been materialized.
    auth = json.loads((OUT / 'SOURCE_ONLY_TEST_AUTHORIZATION_PENDING.json').read_text(encoding='utf-8'))
    auth['reference_guideline_sha256'] = sha(OUT / 'STAGE_K_WORKFLOW_AND_REFERENCE_GUIDELINE.md')
    auth['reference_guideline_binding_note'] = 'Proposed guideline only. Actual source manifests, real people and production loader are not yet bound or approved.'
    write('SOURCE_ONLY_TEST_AUTHORIZATION_PENDING.json', auth)
    (OUT / 'BUILD_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_workflow'], cwd=APP, capture_output=True, text=True)
    (OUT / 'CPU_TESTS.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    require_tests = re.search(r'Ran (\d+) tests', result.stderr)
    assert result.returncode == 0 and require_tests and int(require_tests.group(1)) == 19
    js = re.search(r'<script>([\s\S]*?)</script>', (APP / 'index.html').read_text(encoding='utf-8')).group(1)
    syntax = subprocess.run(['node', '--check'], input=js, capture_output=True, text=True)
    assert syntax.returncode == 0
    (OUT / 'JS_SYNTAX.log').write_text(syntax.stdout + syntax.stderr, encoding='utf-8')
    write('OFFLINE_TEST_RECEIPT.json', {'CPU_tests': 19, 'CPU_returncode': result.returncode,
        'JS_syntax_returncode': syntax.returncode, 'test_data': 'synthetic in-memory; no persisted test annotations',
        'actual_TEST_sources_or_references_created': 0,
        'render_validation': 'UNVERIFIED_HEADLESS_CHROME_LAUNCH_BLOCKED_BY_AUTOMATIC_APPROVAL_REVIEW'})
    write('DELIVERY_STATUS.json', {'scope': 'OFFLINE_SYNTHETIC_WORKFLOW_ONLY', 'TEST_reference_readiness': 'NOT_YET_VERIFIED',
        'authenticated_custodian_index_found': False, 'human_authors_confirmed': False,
        'real_annotations_or_approvals_created': 0, 'TEST_inputs_opened': 0, 'model_predictions_opened': 0,
        'GPU_operations': 0, 'actual_Shuffled_mapping_created': False, 'automatic_continuation': False})
    entries = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.sqlite', '.sqlite-journal') and p.name != 'MANIFEST.json'}
    write('MANIFEST.json', {'path_base': 'archive_root', 'files': entries})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for rel in [*entries, 'MANIFEST.json']:
            z.write(OUT / rel, rel)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for rel, expected in entries.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    print(json.dumps({'archive': str(ZIP), 'manifest_entries': len(entries), 'sha256': sha(ZIP),
        'CPU_tests': 19, 'JS_syntax': 'PASS', 'reference_status': 'NOT_YET_VERIFIED'}, indent=2))


if __name__ == '__main__':
    main()
