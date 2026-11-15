"""Read-only, offline reference/provenance audit. Never evaluates predictions.

Runs with the Python standard library and the included original reference
validator. Prints a fresh observation; it never updates a reference or journal.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit(root, check_package_manifest=True):
    root = Path(root).resolve()
    sys.path.insert(0, str(root/'ORIGINAL_SCRIPTS'))
    from s1000_final_eval_core_v1 import digest, reference_content
    from validate_s1000_human_references_v1 import validate_all
    problems = []
    checks = {}

    def check(name, passed, detail):
        checks[name] = {'status': 'PASS' if passed else 'FAIL', 'detail': detail}
        if not passed:
            problems.append(name)

    export = root/'FINAL_EXPORT'
    source = root/'SOURCE_EVIDENCE'
    freeze = read(export/'REFERENCE_FREEZE.json')
    cohort = read(root/'COHORT/COHORT100_FREEZE.json')['business_ids']
    refs = read(export/'APPROVED_REFERENCES.json')
    ids = [r['business_id'] for r in refs]
    check('exact_100_unique_TEST_businesses', len(ids) == len(set(ids)) == len(cohort) == len(set(cohort)) == 100 and set(ids) == set(cohort), {'businesses': len(ids), 'unique_ids': len(set(ids))})
    check('both_required_components_present', all(isinstance(r.get(k), str) and r[k].strip() for r in refs for k in ['review_consensus', 'visual_evidence']), {'nonempty_components': sum(bool(r.get(k, '').strip()) for r in refs for k in ['review_consensus', 'visual_evidence'])})
    manifest = read(source/'ANNOTATION_PACKAGE_MANIFEST.json')
    check('source_manifest_binding', sha(source/'ANNOTATION_PACKAGE_MANIFEST.json') == freeze['source_manifest_sha256'], {'sha256': sha(source/'ANNOTATION_PACKAGE_MANIFEST.json')})
    check('source_and_cohort_membership', {r['business_id'] for r in manifest['frozen_sources']} == set(cohort), {'source_businesses': len(manifest['frozen_sources'])})
    validated, validation = validate_all(source, export/'ANNOTATIONS', approved=True)
    check('original_reference_evidence_validator', not validation['errors'], validation)
    check('aggregate_equals_individual_annotations', {r['business_id']: r for r in refs} == {r['business_id']: r for r in validated}, {'annotation_files': len(list((export/'ANNOTATIONS').glob('*.json')))})
    check('aggregate_and_content_freeze', sha(export/'APPROVED_REFERENCES.json') == freeze['reference_file_sha256'] and all(freeze['content_hashes'].get(r['business_id']) == digest(reference_content(r)) for r in refs), {'aggregate_file_sha256': sha(export/'APPROVED_REFERENCES.json'), 'freeze_file_sha256': sha(export/'REFERENCE_FREEZE.json')})
    export_manifest = read(export/'EXPORT_MANIFEST.json')['files']
    check('original_export_member_hashes', all((export/name).is_file() and sha(export/name) == value for name, value in export_manifest.items()), {'files_checked': len(export_manifest)})
    events = [json.loads(line) for line in (export/'REVISION_EVENTS.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    previous = None; versions = {}; latest = {}; chain_ok = True; approval_ok = True; transition_ok = True
    declarations = []; mungo = []
    for wrapped in events:
        event = wrapped['event']; b = event['business_id']
        chain_ok &= wrapped['sha256'] == digest(event) and event['previous_event_sha256'] == previous
        chain_ok &= event['version'] == versions.get(b, 0)+1
        state = event['reference']
        chain_ok &= event['content_sha256'] == digest(reference_content(state)) and state['history'][-1]['content_sha256'] == event['content_sha256']
        parent = latest.get(b) or read(source/'TEST100_REFERENCE_TEMPLATE'/f'{b}.json')
        chain_ok &= event['prior_reference_sha256'] == digest(parent)
        chain_ok &= state['history'][:-1] == parent['history']
        action = event['action']; actor = event['actor']
        if action in ['save', 'submit']:
            transition_ok &= parent['status'] in ['NOT_STARTED', 'DRAFT', 'NEEDS_REVISION'] and actor['role'] == 'author'
            transition_ok &= state['status'] == ('DRAFT' if action == 'save' else 'SUBMITTED') and state['author']['human_id'] == actor['id']
        elif action in ['approve', 'request_revision', 'reject']:
            transition_ok &= parent['status'] == 'SUBMITTED' and actor['role'] == 'checker' and actor['id'] != parent['author']['human_id']
            transition_ok &= reference_content(parent) == reference_content(state)
            transition_ok &= state['status'] == ('APPROVED' if action == 'approve' else 'NEEDS_REVISION')
            if action == 'approve':
                transition_ok &= state['checker']['human_id'] == actor['id']
        else:
            transition_ok = False
        previous = wrapped['sha256']; versions[b] = event['version']; latest[b] = state
        if event['action'] in ['submit', 'approve']:
            required = ['human_written', 'sources_checked', 'no_predictions_or_teacher'] if event['action'] == 'submit' else ['independent_check', 'sources_checked', 'no_predictions_or_teacher', 'coverage_and_component_separation_checked']
            approval_ok &= all(event.get('attestations', {}).get(k) is True for k in required)
            approval_ok &= event['actor']['role'] == ('author' if event['action'] == 'submit' else 'checker')
            declarations.append({'business_id': b, 'version': event['version'], 'action': event['action'], 'recorded_at_utc': event['at_utc'], 'actor': event['actor'], 'declarations_as_recorded': event['attestations'], 'original_event_sha256': wrapped['sha256']})
        if b == '-78Va3llKcPbL6SW4is8lw':
            mungo.append({'version': event['version'], 'action': event['action'], 'recorded_at_utc': event['at_utc'], 'content_sha256': event['content_sha256'], 'actor': event['actor'], 'comment': event['comment'], 'original_event_sha256': wrapped['sha256']})
    check('original_event_hash_chain_and_versions', bool(chain_ok), {'events': len(events), 'chain_head_sha256': previous})
    check('allowed_workflow_transitions_no_edits_after_approval', bool(transition_ok), {'scope': 'Recorded state transitions only; not continuous external monitoring'})
    check('recorded_role_and_declaration_workflow', bool(approval_ok), {'submit_or_approve_events': len(declarations), 'human_authenticity_proven': False})
    check('final_export_equals_last_recorded_states', all(latest.get(r['business_id']) == r for r in refs), {'businesses': len(refs)})
    ledger = []
    for r in refs:
        b = r['business_id']; event = next(w['event'] for w in reversed(events) if w['event']['business_id'] == b)
        approval_ok &= r['status'] == 'APPROVED' and event['action'] == 'approve' and r['checker']['reviewed_content_sha256'] == digest(reference_content(r))
        ledger.append({'business_id': b, 'reference_version': versions[b], 'reference_file_sha256': sha(export/'ANNOTATIONS'/f'{b}.json'), 'reference_content_sha256': digest(reference_content(r)), 'source_bundle_sha256': r['source_bundle_sha256'], 'author_id': r['author']['human_id'], 'author_recorded_at_utc': r['author']['at_utc'], 'checker_id': r['checker']['human_id'], 'checker_recorded_at_utc': r['checker']['at_utc'], 'approval_event_recorded_at_utc': event['at_utc'], 'status': r['status'], 'approval_recording_mechanism': event['actor'].get('recorded_by', 'not separately specified')})
    check('approval_binds_final_reference_version', bool(approval_ok), {'approved_final_versions': sum(r['status'] == 'APPROVED' for r in refs)})
    with (export/'REFERENCE_APPROVAL_TRACKER.csv').open(encoding='utf8', newline='') as f:
        tracker = {r['business_id']: r for r in csv.DictReader(f)}
    check('export_approval_ledger_versions', set(tracker) == set(ids) and all(int(tracker[b]['final_version']) == versions[b] for b in ids), {'rows': len(tracker)})
    completion = read(root/'STAGE_M_BINDINGS/COMPLETION.json')
    recorded = completion['reference_files']
    actual_json = {p.relative_to(export).as_posix(): sha(p) for p in export.rglob('*.json')}
    check('Stage_M_exact_reference_input_hashes', actual_json == recorded, {'json_reference_files_checked': len(recorded), 'mismatches': [name for name, value in recorded.items() if actual_json.get(name) != value]})
    protected = read(root/'STAGE_M_BINDINGS/PROTECTED_BEFORE_SCORING.json')
    prefix = 'outputs/stage_l/approved_test100_final_export_v1/'
    observations = {name[len(prefix):]: value for name, value in protected['files'].items() if name.startswith(prefix)}
    check('Stage_M_in_run_export_observation', all((export/name).is_file() and sha(export/name) == value for name, value in observations.items()), {'all_export_files_checked': len(observations), 'recorded_at_utc': protected['recorded_at_utc']})
    release = read(root/'STAGE_M_BINDINGS/EXECUTION_RELEASE.json')
    check('Stage_M_release_freeze_and_aggregate_bindings', release['reference_freeze_sha256'] == sha(export/'REFERENCE_FREEZE.json') and release['approved_aggregate_sha256'] == sha(export/'APPROVED_REFERENCES.json') and completion['researcher_release_sha256'] == sha(root/'STAGE_M_BINDINGS/EXECUTION_RELEASE.json'), {'release_sha256': sha(root/'STAGE_M_BINDINGS/EXECUTION_RELEASE.json')})
    start = read(root/'STAGE_M_BINDINGS/PROCESS_RECEIPT.json')['started_at_utc']
    timing = {'source_freeze_at_utc': freeze.get('frozen_at_utc'), 'release_recorded_at_utc': release['recorded_at_utc'], 'scorer_started_at_utc': start, 'protected_snapshot_at_utc': protected['recorded_at_utc'], 'snapshot_precedes_process_start': dt.datetime.fromisoformat(protected['recorded_at_utc']) <= dt.datetime.fromisoformat(start)}
    if check_package_manifest:
        files = read(root/'PACKAGE_MANIFEST.json')['files']
        check('package_file_hashes', all((root/name).is_file() and (root/name).stat().st_size == binding['bytes'] and sha(root/name) == binding['sha256'] for name, binding in files.items()), {'files_checked': len(files)})
    return {'status': 'PASS_RECORDED_INTEGRITY_WITH_EXPLICIT_AUTHENTICITY_LIMITATIONS' if not problems else 'FAIL', 'checks': checks, 'failed_checks': problems,
        'observed_post_approval_consistency': 'Final content matches final approval hashes, the hash-linked journal, in-run Stage M observation and Stage M completion input hashes. No modification is detected at these recorded observations.',
        'not_established': {'actual_human_identity': 'NOT_ESTABLISHED', 'actual_source_inspection': 'NOT_ESTABLISHED', 'unaided_human_authorship': 'NOT_ESTABLISHED; AI assistance is documented', 'complete_semantic_factuality_and_coverage': 'NOT_ESTABLISHED by structural/hash checks', 'continuous_tamper_proof_history_or_external_nonrepudiation': 'NOT_ESTABLISHED; all receipts and the chain are local records', 'lossless_original_Vietnamese_confirmation_text': 'NOT_ESTABLISHED; question-mark substitutions occur in preserved fields named statement_verbatim/current_user_statement_verbatim'},
        'timing_qualification': timing, 'photo_pixels_available_in_package': False, 'new_decisions_or_reference_edits_created': False,
        'derived_reference_approval_ledger': ledger, 'derived_declaration_event_ledger': declarations, 'derived_mungo_timeline': mungo}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = audit(args.package)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['failed_checks'] else 0)
