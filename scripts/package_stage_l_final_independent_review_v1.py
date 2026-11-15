"""Package existing evidence and run read-only audits; no evaluation or approval."""
from contextlib import closing
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

from s1000_final_eval_core_v1 import file_hash, digest, read, write, require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_l/final_reference_independent_review_20261101_v1'
PACKAGE = OUT/'PACKAGE'
EXPORT = ROOT/'outputs/stage_l/approved_test100_final_export_v1'
APP = ROOT/'outputs/stage_l/test100_human_references_v1'
M = ROOT/'outputs/stage_m/human_primary_execution_20261101_v1'


def db_snapshot():
    with closing(sqlite3.connect((APP/'PRIVATE_STATE/references.sqlite3').as_uri()+'?mode=ro', uri=True)) as c:
        # Never read password hashes, salts, sessions or a private identity roster.
        refs = [(b, v, json.loads(payload)) for b, v, payload in c.execute('SELECT business,version,payload FROM refs ORDER BY business')]
        events = [(seq, b, json.loads(event), sha) for seq, b, event, sha in c.execute('SELECT seq,business,event_json,event_hash FROM events ORDER BY seq')]
        used = {r['author']['human_id'] for _, _, r in refs} | {r['checker']['human_id'] for _, _, r in refs}
        roles = [{'id': identity, 'role': role, 'account_created_at_utc': created} for identity, role, created in c.execute('SELECT id,role,created FROM users ORDER BY id') if identity in used]
    return {'refs': refs, 'events': events, 'pseudonymous_role_extract': roles}


def main():
    OUT.mkdir(parents=True, exist_ok=False); PACKAGE.mkdir()
    originals = {}; source_by_member = {}

    def add(path, member):
        path = Path(path); target = PACKAGE/member
        require(not target.exists(), 'Duplicate member: '+str(member))
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        require(file_hash(path) == file_hash(target), 'Copy differs')
        originals[path.relative_to(ROOT).as_posix()] = file_hash(path)
        source_by_member[target.relative_to(PACKAGE).as_posix()] = path.relative_to(ROOT).as_posix()

    live_before = db_snapshot()
    for path in sorted(EXPORT.rglob('*')):
        if path.is_file(): add(path, Path('FINAL_EXPORT')/path.relative_to(EXPORT))
    add(ROOT/'outputs/final_evaluation/s1000_preparation_v1/COHORT100_FREEZE.json', Path('COHORT/COHORT100_FREEZE.json'))
    for path in sorted((APP/'SOURCE_PACKAGE').rglob('*')):
        if path.is_file() and (path.name == 'ANNOTATION_PACKAGE_MANIFEST.json' or 'SOURCES' in path.relative_to(APP/'SOURCE_PACKAGE').parts or 'TEST100_REFERENCE_TEMPLATE' in path.relative_to(APP/'SOURCE_PACKAGE').parts):
            add(path, Path('SOURCE_EVIDENCE')/path.relative_to(APP/'SOURCE_PACKAGE'))
    provenance = {
        'offline_checker_import_20261101_v1': ['ORIGINAL_CHECKER_DECISIONS.json', 'IMPORT_RECEIPT.json', 'POST_IMPORT_VERIFICATION.json'],
        'batch_author_submission_20261031_v1': ['REPORTED_AUTHOR_CONFIRMATION.json', 'BATCH_SUBMISSION_RECEIPT.json'],
        'final_reference_closeout_20261101_v1': ['RESEARCHER_REPORTED_CONFIRMATION.json', 'APPROVAL_RECEIPT.json', 'MUNGO_BEFORE.json', 'MUNGO_AFTER.json'],
        'mungo_checker_requested_revision_20261101_v1': ['BEFORE_REFERENCE.json', 'AFTER_REFERENCE.json', 'EDIT_RECEIPT.json', 'REVISED_VISUAL_EVIDENCE.txt'],
        'evidence_simplification_20261031_v1': ['CANDIDATE_EVIDENCE_ORIGINAL.json', 'IMPORT_RECEIPT_eaf3b95841ad.json', 'LIVE_VERIFICATION.json'],
    }
    for folder, names in provenance.items():
        for name in names: add(ROOT/'outputs/stage_l'/folder/name, Path('PROVENANCE')/folder/name)
    for folder, patterns in [
        ('four_batch_draft_review_20261030_v1', ['REVIEW_20_BUSINESSES_VI.md', 'SOURCE_FREEZE_VERIFICATION.json']),
        ('batches021_050_feedback_20261030_v1', ['INVENTORY.json']),
        ('batches051_100_feedback_20261031_v1', ['INVENTORY.json', 'STRUCTURAL_CHECK.json'])]:
        for name in patterns: add(ROOT/'outputs/stage_l'/folder/name, Path('PROVENANCE/PRIOR_AI_ASSISTED_REVIEW')/folder/name)
    for name in ['EXECUTION_RELEASE.json', 'PROCESS_RECEIPT.json', 'PROCESS_COMPLETION.json', 'PROTECTED_BEFORE_SCORING.json', 'PRE_SCORING_INPUT_VERIFICATION.json', 'REGRESSION_TEST_RECEIPT.json']:
        add(M/name, Path('STAGE_M_BINDINGS')/name)
    for name in ['COMPLETION.json', 'RESULT_INTEGRITY_VERIFICATION.json']:
        add(ROOT/'outputs/stage_m/human_primary_scored_v1'/name, Path('STAGE_M_BINDINGS')/name)
    for name in ['s1000_final_eval_core_v1.py', 'validate_s1000_human_references_v1.py', 'stage_l_reference_app_v1.py', 'stage_m_evaluation_v1.py', 'import_stage_l_offline_checker_v1.py', 'submit_stage_l_confirmed_drafts_v1.py', 'import_stage_l_candidate_evidence_v1.py']:
        add(ROOT/'scripts'/name, Path('ORIGINAL_SCRIPTS')/name)
    add(ROOT/'scripts/verify_stage_l_final_review_package_v1.py', Path('VERIFY_REFERENCE_PACKAGE.py'))
    with (ROOT/'outputs/stage_m/human_primary_scored_v1/PER_BUSINESS_COMPONENT_METRICS.csv').open(encoding='utf8', newline='') as f:
        input_rows = list(csv.DictReader(f))
    # Explicitly derived projection: no scores, model outputs or new evaluation.
    write(PACKAGE/'STAGE_M_BINDINGS/DERIVED_REFERENCE_USAGE_BINDINGS.json', {'derived_from': 'outputs/stage_m/human_primary_scored_v1/PER_BUSINESS_COMPONENT_METRICS.csv', 'original_file_sha256': file_hash(ROOT/'outputs/stage_m/human_primary_scored_v1/PER_BUSINESS_COMPONENT_METRICS.csv'), 'metric_values_intentionally_excluded': True, 'rows': [{k: r[k] for k in ['condition', 'business_id', 'reference_business_id', 'task', 'reference_type', 'reference_sha256']} for r in input_rows]})
    write(PACKAGE/'AUDIT/DERIVED_PARTICIPANT_ROLE_EXTRACT.json', {'source': 'Read-only SQLite SELECT id,role,created restricted to reference participants; no passwords/salts read', 'participants': live_before['pseudonymous_role_extract'], 'independent_real_identity_authentication': False})
    spec = importlib.util.spec_from_file_location('reference_only_audit', PACKAGE/'VERIFY_REFERENCE_PACKAGE.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    result = verifier.audit(PACKAGE, check_package_manifest=False)
    require(not result['failed_checks'], 'Reference audit failed: '+str(result['failed_checks']))
    exported = {r['business_id']: r for r in read(EXPORT/'APPROVED_REFERENCES.json')}
    exported_events = [json.loads(line) for line in (EXPORT/'REVISION_EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    live_matches = all(exported.get(b) == r and r['history'][-1]['version'] == v for b, v, r in live_before['refs']) and len(live_before['refs']) == 100
    live_chain_matches = [{'event': event, 'sha256': sha} for _, _, event, sha in live_before['events']] == exported_events
    require(live_matches and live_chain_matches, 'Read-only live state differs from final export')
    roles = {r['id']: r['role'] for r in live_before['pseudonymous_role_extract']}
    require(all(roles[r['author']['human_id']] == 'author' and roles[r['checker']['human_id']] == 'checker' for r in exported.values()), 'Pseudonymous role mismatch')
    write(PACKAGE/'AUDIT/DERIVED_LIVE_STATE_OBSERVATION.json', {'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'access_mode': 'SQLite mode=ro; no workflow API or write query', 'live_refs_equal_frozen_export': live_matches, 'live_events_equal_frozen_export_journal': live_chain_matches, 'reference_records': len(live_before['refs']), 'events': len(live_before['events']), 'pseudonymous_roles_match_reference_workflow': True, 'human_identity_authentication': 'NOT_ESTABLISHED'})
    write(PACKAGE/'AUDIT/READ_ONLY_REFERENCE_AUDIT.json', result)
    for key, name in [('derived_reference_approval_ledger', 'DERIVED_APPROVAL_LEDGER.json'), ('derived_declaration_event_ledger', 'DERIVED_DECLARATION_LEDGER.json'), ('derived_mungo_timeline', 'DERIVED_MUNGO_TIMELINE.json')]:
        write(PACKAGE/'AUDIT'/name, {'derived_from': 'FINAL_EXPORT/REVISION_EVENTS.jsonl and unchanged final export; not replacement approval records', 'rows': result[key]})
    disclosure = '''# AI assistance and approval-record limitations

The researcher describes the reference prose as human-authored and the checker as a genuinely different person. The package records those declarations without independently authenticating them.

- Model-assisted feedback was provided on earlier drafts, with illustrative wording and targeted review/photo checks. Available existing feedback records are included under PRIOR_AI_ASSISTED_REVIEW. These are incomplete records of the overall interaction, not an exhaustive reconstruction of every assistance event.
- The existing 100-business evidence-suggestion JSON was produced with AI assistance. Imported literal quotations, IDs and exact spans were mechanically matched; that did not establish semantic entailment, consensus or complete coverage. Its original false photo-inspection flags are preserved in the candidate file. Later true flags in submitted references record reported human confirmations, not AI observation as a substitute for human inspection.
- Initial draft intake, candidate-evidence intake, bulk author submissions and offline checker-decision import were administrative operations by Codex on researcher instructions. The journal actor metadata and comments disclose this. They are not authenticated participant web clicks.
- The official offline checker file requested one revision for Mungo. The final phrase was changed from "a sepia dock bannered 'LITTLE TONY TOMATOES'" to "a sepia-toned building facade with a 'LITTLE TONY TOMATOES' sign" with Codex editorial assistance. A new photo pointer was added. A later researcher-reported author/checker confirmation led to submission and final approval. Preserve this limited editorial assistance in methodological reporting; do not call the whole dataset verified unaided human writing.
- The original checker comment also refers to a separate final Word version. That Word document was not independently inspected or substituted for the actual app record. Only the final exported version and its hash enter the freeze.
- Several old receipt fields named statement_verbatim/current_user_statement_verbatim contain question-mark substitutions from character encoding. Their original bytes are retained, not silently corrected. They cannot establish the exact original Vietnamese wording. English scope qualifications and journal confirmation bindings remain readable; these do not repair verbatim fidelity.
- The file named PROTECTED_BEFORE_SCORING.json is timestamped after the scorer process started. It is an in-run hash observation, not a pre-launch snapshot. EXECUTION_RELEASE.json predates launch and binds the aggregate/freeze; the Stage M completion receipt separately records exact hashes for all105 reference JSON inputs.
- Hashes and a local hash-linked journal establish recorded consistency, not digital signatures, an externally anchored time chain, complete original-image inspection, semantic accuracy, full opinion/scene coverage or unaided authorship. A temporary unobserved modification followed by reversion cannot be ruled out solely from local hash snapshots.

No new checker decisions, confirmations, reference text or historical audit records were generated by this packaging task. Newly derived ledgers and the read-only audit are labelled as observations of existing records. Original JPEGs are excluded, so an offline reviewer can verify photo IDs/hash bindings but cannot independently inspect image pixels from this ZIP alone.
'''
    (PACKAGE/'AI_ASSISTANCE_AND_LIMITATIONS.md').write_text(disclosure, encoding='utf8')
    readme = '''# Stage L final reference independent review

Purpose: inspect the exact100-reference export used by completed Stage M, its recorded approvals and freeze bindings BEFORE reviewing metric outcomes. No metric tables, prediction text, JPEG archives, model weights, private DB, participant roster, passwords, API keys or session tokens are included.

## Start here

1. Read AI_ASSISTANCE_AND_LIMITATIONS.md and AUDIT/READ_ONLY_REFERENCE_AUDIT.json. The status concerns recorded integrity, not authenticated unaided human authorship.
2. FINAL_EXPORT/ is a byte-for-byte copy of the108-file final export:100 annotation JSONs, aggregate, readable Review/Visual text, evidence links, approval tracker, original revision-event journal, manifest and freeze. Do not edit it.
3. COHORT/ and SOURCE_EVIDENCE/ bind the100 TEST IDs to exact selected reviews, photo IDs/hashes and original blank templates. The immutable source manifest's zero-reference/preparation status is historical; it is not the final approval ledger. Original photos are not included.
4. PROVENANCE/ preserves the original offline checker file, author/researcher confirmation receipts, Mungo before/after states and editorial receipt, AI-assisted evidence suggestions/import receipt and available earlier feedback. AUDIT/ derived ledgers are navigation aids, not new approvals. Participant roles use author01/checker01 only; no private identity roster is supplied.
5. STAGE_M_BINDINGS/ contains unchanged execution/completion receipts and exact reference-input hashes. A separately labelled reference-usage projection omits metric values. All105 JSON-input hashes are compared to the final export; release aggregate/freeze hashes and the108-file in-run observation are also checked. No evaluation was rerun.

## Reproduce the read-only checks

Extract the ZIP. With Python3.11 or later, from its root run:

```text
python VERIFY_REFERENCE_PACKAGE.py --package .
```

This prints a fresh JSON audit to stdout using the standard library and the included original validator. It does not edit files, open the application database, evaluate outputs, make network calls or create approvals. ORIGINAL_SCRIPTS/ retains existing verification/workflow/scorer source for inspection; do not run the scorer for this review.

## Scope of the result

The actual packaged checks confirm100 unique TEST businesses,200 nonempty reference components, valid literal evidence IDs/quotes/source hashes, recorded roles/declarations,404 linked workflow events, exact final-version approvals, unchanged observed content and matching Stage M input hashes. Identity, actual human inspection, unaided authorship, continuous nonrepudiable history and full semantic correctness remain NOT_ESTABLISHED. The original Vietnamese receipt-text corruption and the in-run timing of the nominal BEFORE_SCORING snapshot are disclosed explicitly.

PACKAGE_MANIFEST.json hashes every package file except itself. Existing original manifests/receipts are preserved alongside this new packaging manifest. The archive is reopened, extracted, every member is rehashed and the portable audit is rerun before delivery. This package is for research review, not a replacement reference set or a declaration of independent human evaluation of model outputs.
'''
    (PACKAGE/'README.md').write_text(readme, encoding='utf8')
    # Privacy checks are limited to the allowlisted package, not a general claim
    # that no secret can ever appear in unstructured public-source text.
    forbidden = []; profile_paths = []
    for p in PACKAGE.rglob('*'):
        if not p.is_file(): continue
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.sqlite3', '.db', '.safetensors', '.bin', '.zip']:
            forbidden.append(p.relative_to(PACKAGE).as_posix())
        if p.suffix.lower() in ['.json', '.jsonl', '.txt', '.md', '.csv']:
            text = p.read_text(encoding='utf-8-sig')
            if re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]+', text): profile_paths.append(p.relative_to(PACKAGE).as_posix())
            if re.search(r'(?i)"(?:password|salt|api_key|session_token)"\s*:', text): forbidden.append(p.relative_to(PACKAGE).as_posix())
    require(not forbidden and not profile_paths, 'Privacy allowlist check: '+str(forbidden+profile_paths))
    write(PACKAGE/'AUDIT/PACKAGING_SCOPE_AND_ORIGINAL_PRESERVATION.json', {'task': 'Read-only audit and evidence packaging only', 'original_files_hashed': len(originals), 'no_original_file_changes': all(file_hash(ROOT/name) == value for name, value in originals.items()), 'source_file_sha256': originals, 'exact_original_copy_mapping': source_by_member, 'forbidden_binary_or_credential_fields_found': forbidden, 'private_profile_paths_found': profile_paths, 'privacy_check_scope': 'Allowlisted files; no private DB/credential columns copied; public Yelp source text and public source-repository provenance retained unchanged', 'new_approvals': 0, 'reference_edits': 0, 'evaluation_runs': 0})
    require(live_before == db_snapshot(), 'Live reference/event/role state changed during packaging; inspect before claiming read-only preservation')
    files = {p.relative_to(PACKAGE).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
    write(PACKAGE/'PACKAGE_MANIFEST.json', {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'files': files, 'manifest_self_hash': 'Archive sidecar; not recursively included'})
    # Run the standalone verifier with only the extracted package available as
    # its input; no Stage M model runtime or application service is used.
    archive = OUT/'STAGE_L_FINAL_REFERENCE_INDEPENDENT_REVIEW.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(PACKAGE).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_l_independent_review_') as tmp:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and set(names) == set(files) | {'PACKAGE_MANIFEST.json'}, 'ZIP member set')
            require(z.testzip() is None, 'ZIP CRC')
            z.extractall(tmp)
        for name, value in files.items():
            p = Path(tmp)/name
            require(p.stat().st_size == value['bytes'] and file_hash(p) == value['sha256'], 'Extracted member SHA mismatch')
        require(file_hash(Path(tmp)/'PACKAGE_MANIFEST.json') == file_hash(PACKAGE/'PACKAGE_MANIFEST.json'), 'Manifest member differs')
        run = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(tmp)/'VERIFY_REFERENCE_PACKAGE.py'), '--package', tmp], capture_output=True, text=True, encoding='utf8')
        (OUT/'EXTRACTED_PACKAGE_VERIFIER_STDOUT.json').write_text(run.stdout, encoding='utf8')
        (OUT/'EXTRACTED_PACKAGE_VERIFIER_STDERR.log').write_text(run.stderr, encoding='utf8')
        require(run.returncode == 0, 'Extracted portable audit failed: '+run.stderr)
        extracted_audit = json.loads(run.stdout)
        require(not extracted_audit['failed_checks'], 'Extracted package audit flags')
    require(live_before == db_snapshot() and all(file_hash(ROOT/name) == value for name, value in originals.items()), 'Original inputs or live state changed')
    checksum = file_hash(archive)
    (OUT/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n', encoding='ascii')
    write(OUT/'DELIVERY_VERIFICATION.json', {'status': 'PASS_PACKAGING_AND_RECORDED_INTEGRITY_WITH_LIMITATIONS', 'archive_bytes': archive.stat().st_size, 'archive_sha256': checksum, 'members': len(files)+1, 'reopened_extracted_and_all_member_sha256_verified': True, 'portable_audit_exit_code': run.returncode, 'portable_audit_status': extracted_audit['status'], 'unchanged_original_files': len(originals), 'live_reference_event_and_pseudonymous_role_state_unchanged': True, 'human_authenticity': 'NOT_ESTABLISHED', 'verbatim_receipt_text_fidelity': 'NOT_ESTABLISHED', 'evaluation_runs': 0, 'new_approvals': 0})
    print(json.dumps({'zip': str(archive), 'sha256': checksum, 'bytes': archive.stat().st_size, 'members': len(files)+1, 'recorded_integrity': extracted_audit['status'], 'failed_checks': extracted_audit['failed_checks']}, ensure_ascii=False))


if __name__ == '__main__': main()
