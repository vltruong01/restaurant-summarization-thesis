"""Prepare the explicitly approved single HTTP520 recovery; preserve v3."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rewrite(source, old, new, count=1):
    assert source.count(old) == count, 'Unexpected recovery source anchor'
    return source.replace(old, new)


common = (ROOT / 'scripts/stage_g_d041_recovery_v3.py').read_text(encoding='utf-8')
common = common.replace('recovery_v3', 'recovery_v4')
common = rewrite(common, 'from run_stage_g_d041_continuation_v2 import check_inputs as previous_check',
                 'from stage_g_d041_recovery_v3 import check_inputs as previous_check')
common = rewrite(common, 'from run_stage_g_d041_continuation_v2 import FOLDER as PREVIOUS',
                 'from stage_g_d041_recovery_v3 import FOLDER as PREVIOUS')
common = rewrite(common, 'HISTORICAL_FAILED_ATTEMPTS = 2', 'HISTORICAL_FAILED_ATTEMPTS = 3')
common = rewrite(common, 'assert len(failures) == 2', 'assert len(failures) == 1')
common = rewrite(common, "['protocol_snapshot', 'concurrency4_trial_v1', 'concurrency4_continuous_v1']",
                 "['protocol_snapshot', 'concurrency4_trial_v1', 'concurrency4_continuous_v1', 'previous_failed_attempts']")
common = rewrite(common, "['checkpoints', 'raw_responses', 'raw_requests', 'request_intents', 'provisional_targets']",
                 "['checkpoints', 'raw_responses', 'raw_requests', 'request_intents', 'provisional_targets', 'stream_diagnostics']")
common = rewrite(common, "branch in ['checkpoints', 'request_intents']", "branch in ['checkpoints', 'request_intents', 'stream_diagnostics']")
common = rewrite(common, 'assert paid == 142 and len(pending) == 144', 'assert paid == 58 and len(pending) == 59')
common = common.replace("'maximum_operator_retries': 2", "'maximum_operator_retries': 1")
common = common.replace("'maximum_unstarted_paid_calls': 140", "'maximum_unstarted_paid_calls': 57")
common = common.replace("'maximum_new_paid_calls': 142", "'maximum_new_paid_calls': 58")
common = common.replace("'maximum_total_cohort_attempts_including_previous_failures': 294", "'maximum_total_cohort_attempts_including_previous_failures': 295")
common = common.replace("'imported_accepted_tasks': 156", "'imported_accepted_tasks': 241")
common = common.replace("'imported_complete_businesses': 51", "'imported_complete_businesses': 79")
common = common.replace("'imported_complete_API_responses': 150", "'imported_complete_API_responses': 234")
common = common.replace("'previous_total_API_attempts': 152", "'previous_total_API_attempts': 237")
common = common.replace("'operator_retries_authorized': 2", "'operator_retries_authorized': 1")
common = common.replace("'unstarted_paid_calls': 140", "'unstarted_paid_calls': 57")
common = common.replace('USER_APPROVED_TWO_SERIAL_OPERATOR_RECOVERIES_THEN_UNSTARTED_CONCURRENCY4', 'USER_APPROVED_ONE_SERIAL_HTTP520_RECOVERY_THEN_UNSTARTED_CONCURRENCY4')
common = common.replace('User approved diagnostic logging, one serial recovery attempt for each of two failures, then continuation if both pass.',
                        'User explicitly requested fixing the HTTP520 request and continuing; one serial operator retry then57 unstarted paid calls if recovery passes.')
common = common.replace('recovery of two failures', 'recovery of one HTTP520 failure')

runner = (ROOT / 'scripts/run_stage_g_d041_recovery_v3.py').read_text(encoding='utf-8')
runner = runner.replace('recovery_v3', 'recovery_v4')
runner = rewrite(runner, 'assert len(retry_ids) == 2', 'assert len(retry_ids) == 1')
runner = rewrite(runner, ".glob('*.json'))) + 2,", ".glob('*.json'))) + 3,")
runner = runner.replace("'historical_failed_attempts_preserved': 2", "'historical_failed_attempts_preserved': 3")
runner = runner.replace("'imported_API_request_intents': 152", "'imported_API_request_intents': 237")
runner = runner.replace('previous_calls + 2', 'previous_calls + 3')
runner = runner.replace("'maximum_total_cohort_attempts_including_operator_retries': 294", "'maximum_total_cohort_attempts_including_operator_retries': 295")
runner = runner.replace('total + 2', 'total + 3')
runner = runner.replace("'all_two_passed': len(recovery_results) == 2", "'all_recoveries_passed': len(recovery_results) == 1")
runner = runner.replace('two serial recoveries', 'one serial HTTP520 recovery')
runner = runner.replace('Two operator-approved serial recovery attempts', 'One operator-approved serial HTTP520 recovery attempt')
runner = runner.replace('two preserved previous failures', 'three preserved previous failed attempts')
runner = runner.replace('Previous51 complete business targets', 'Previous79 complete business targets')
runner = runner.replace('two serial operator retries', 'one serial HTTP520 operator retry')
runner = runner.replace('two user-approved serial recoveries', 'one user-approved serial HTTP520 recovery')
for name, source in [('stage_g_d041_recovery_v4.py', common), ('run_stage_g_d041_recovery_v4.py', runner)]:
    p = ROOT / 'scripts' / name
    assert not p.exists()
    compile(source, str(p), 'exec')
    p.write_text(source, encoding='utf-8')
print('V4 single-request recovery runtime derived; no API/credential access.')
