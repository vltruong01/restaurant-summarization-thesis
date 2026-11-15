"""Continue unaffected frozen jobs; retain one unresolved business without relabeling outputs."""
import sys
from pathlib import Path
import stage_g_coletoken_production_v1 as ledger

JOB = 'train_0mLgF2T8wVDj7NFtoRtpQQ_review_consensus'
BUSINESS = '0mLgF2T8wVDj7NFtoRtpQQ'
RECOVERY = ledger.RUN / 'operator_review_retry_v1'
CONTROL = ledger.RUN / 'deferred_business_continuation_v1'
SCRIPT = Path(__file__).resolve()


def main():
    ledger.verify_manifest()
    original_inputs = ledger.checked_inputs
    cfg, jobs, businesses, imports = original_inputs()
    for directory in (ledger.EXEC, RECOVERY / 'execution_v1'):
        cp = ledger.read(directory / 'checkpoints' / (JOB + '.json'))
        assert cp['status'] == 'FAILED_STOP_NO_AUTOMATIC_RETRY'
        assert cp['local_validation_error'] == 'Teacher returned another business ID'
    deferred_jobs = [j for j in jobs if j['business_id'] == BUSINESS]
    passed = 0
    for j in deferred_jobs:
        if j['job_id'] != JOB:
            ledger.verify_checkpoint(j, cfg, ledger.EXEC, ledger.CONFIG); passed += 1
    assert passed == 2
    manifest_path = CONTROL / 'authorization_manifest.json'
    preserved = {p.relative_to(ledger.ROOT).as_posix(): ledger.file_sha(p)
                 for root in (ledger.EXEC, RECOVERY) for p in root.rglob('*')
                 if p.is_file() and (BUSINESS in p.name or root == RECOVERY)}
    if not manifest_path.exists():
        ledger.immutable(manifest_path, {
            'approved_by': 'user', 'authorization': 'Continue previously approved production; preserve unresolved business',
            'businesses_planned': 2200, 'business_deferred_not_excluded_from_dataset': BUSINESS,
            'failed_job': JOB, 'completed_visual_components_preserved': 2,
            'automatic_retries': 0, 'maximum_concurrency': 5, 'maximum_active_businesses': 2,
            'frozen_sampling_and_jobs_manifests_changed': False, 'scientific_settings_changed': False,
            'runner_sha256': ledger.file_sha(SCRIPT), 'preserved_artifact_sha256': preserved,
            'completion_policy': 'Remain incomplete until all 2200 businesses pass; no success or training with missing business',
            'created_utc': ledger.now()})
    manifest = ledger.read(manifest_path)
    assert manifest['runner_sha256'] == ledger.file_sha(SCRIPT)
    for rel, sha in manifest['preserved_artifact_sha256'].items():
        assert ledger.file_sha(ledger.ROOT / rel) == sha

    def scheduled_inputs():
        config, all_jobs, all_businesses, imported = original_inputs()
        # Scheduling only; never rewrite the frozen cohort/job manifests.
        return config, [j for j in all_jobs if j['business_id'] != BUSINESS], \
            [b for b in all_businesses if b['business_id'] != BUSINESS], imported

    ledger.checked_inputs = scheduled_inputs
    original_docs = ledger.update_docs

    def publish_docs(state):
        state['deferred_businesses'] = [BUSINESS]
        state['deferred_tasks'] = [{'job_id': JOB, 'reason': 'Wrong business ID in original and isolated retry'}]
        state['accepted_tasks'] += passed
        state['operator_retry_API_attempts'] = 1
        state['total_continuation_API_attempts'] = state['new_API_attempts'] + 1
        if state['status'] == 'RUNNING': state['status'] = 'RUNNING_WITH_DEFERRED_BUSINESS'
        ledger.atomic(ledger.EXEC / 'execution_progress.json', state)
        original_docs(state)
        pointer = ledger.read(ledger.POINTER)
        pointer.update(status=state['status'], updated_utc=state['updated_utc'],
                       deferred_businesses=[BUSINESS], execution_entrypoint=SCRIPT.relative_to(ledger.ROOT).as_posix(),
                       deferred_execution_manifest=manifest_path.relative_to(ledger.ROOT).as_posix())
        ledger.atomic(ledger.POINTER, pointer)

    ledger.update_docs = publish_docs
    import run_stage_g_coletoken_production5_v1 as runner
    try:
        runner.main()
    except Exception as error:
        state = ledger.read(ledger.EXEC / 'execution_progress.json')
        # The inherited runner intentionally cannot declare 2199 businesses complete.
        if state['businesses_completed'] == 2199 and state['accepted_tasks'] == 6599:
            state.update(status='INCOMPLETE_ONE_DEFERRED_BUSINESS', phase='awaiting_failed_business_resolution',
                         stop_reason='One review-consensus component remains invalid; full final audit not run')
        else:
            state.update(status='STOPPED_REQUIRES_REVIEW',
                         stop_reason='Continuation failure: ' + type(error).__name__)
        state.update(current_jobs=[], active_businesses=[], updated_utc=ledger.now())
        # Already published accepted_tasks includes the two retained visual components.
        state['accepted_tasks'] -= passed
        publish_docs(state)
        ledger.atomic(CONTROL / 'execution_stop.json', state)
        if state['status'] != 'INCOMPLETE_ONE_DEFERRED_BUSINESS': raise


if __name__ == '__main__': main()
