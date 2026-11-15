"""Check actual provider error shapes without making requests."""
import json
import unittest
from datetime import datetime
from pathlib import Path
import stage_g_mirai_retry_policy_v2 as policy

FINISHED='2026-10-12T04:58:37+00:00'
EPOCH=datetime.fromisoformat(FINISHED).timestamp()


def diagnostic(status, message, kind='upstream_error'):
    return {'http_status': status, 'provider_http_error_body_redacted': json.dumps({'error': {'type': kind, 'message': message}}),
            'finished_utc': FINISHED}


class ProviderTests(unittest.TestCase):
    def test_no_active_upstream_credentials_requires_provider_repair(self):
        plan=policy.retry_plan(diagnostic(403,'No active credentials for provider: example'),1,epoch=EPOCH)
        self.assertEqual(plan['reason'],'upstream_credentials_unavailable')
        self.assertEqual(plan['action'],'stop')
        self.assertTrue(plan['external_provider_repair_required'])
    def test_compound_reset_and_account_word(self):
        plan=policy.retry_plan(diagnostic(404,'[403]: Insufficient account balance (reset after 1m 58s)'),1,epoch=EPOCH)
        self.assertEqual(plan['action'],'retry')
        self.assertEqual(plan['wait_seconds'],118)
    def test_code_variant(self):
        self.assertEqual(policy.retry_plan(diagnostic(404,'INSUFFICIENT_BALANCE (reset after 2m)'),1,epoch=EPOCH)['wait_seconds'],120)
    def test_plain_404_not_retried(self):
        self.assertEqual(policy.retry_plan(diagnostic(404,'route not found'),1)['action'],'defer')
    def test_direct_account_billing_not_retried(self):
        self.assertEqual(policy.retry_plan(diagnostic(403,'insufficient balance (reset after 2m)','billing_error'),1)['action'],'stop')
    def test_missing_reset_not_retried(self):
        self.assertEqual(policy.retry_plan(diagnostic(404,'Insufficient account balance'),1)['action'],'defer')
    def test_unknown_timeout_not_resent(self):
        self.assertEqual(policy.retry_plan({'error_type':'TimeoutError'},1)['action'],'defer')
    def test_attempt_limit(self):
        self.assertEqual(policy.retry_plan(diagnostic(404,'insufficient balance (reset after 2m)'),3)['reason'],'attempt_limit')
    def test_retry_after_and_backoff(self):
        d=diagnostic(429,'rate limited')
        self.assertEqual(policy.retry_plan(d,1,'90',epoch=EPOCH)['wait_seconds'],90)
        self.assertEqual(policy.retry_plan(d,2,epoch=EPOCH)['wait_seconds'],60)
    def test_elapsed_reset_no_duplicate_wait(self):
        d=diagnostic(404,'insufficient balance (reset after 2m)')
        self.assertEqual(policy.retry_plan(d,1,epoch=EPOCH+180)['wait_seconds'],0)
    def test_bounded_wait(self):
        self.assertEqual(policy.retry_plan(diagnostic(404,'insufficient balance (reset after 1h)'),1,epoch=EPOCH)['action'],'defer')
    def test_recorded_failure_required(self):
        with self.assertRaises(ValueError):policy.retry_plan({},0)
    def test_frozen_config_and_concurrency(self):
        import run_stage_g_mirai_retry6_trial_v4 as runner
        old=runner.frozen.read(runner.old.CONFIG);new=runner.frozen.read(runner.CONFIG)
        for key in runner.frozen.FIXED_KEYS:self.assertEqual(old[key],new[key],key)
        self.assertEqual(old['identifier_guard'],new['identifier_guard'])
        self.assertEqual(new['execution']['maximum_concurrency'],6)
        self.assertFalse(new['full_generation_enabled'])
    def test_unavailable_provider_does_not_trigger_paid_calls(self):
        from unittest.mock import patch
        import run_stage_g_mirai_retry6_trial_v4 as runner
        with patch.object(runner, 'PROVIDER_RECOVERED', False), patch.object(runner.old.transport, 'call') as paid:
            with self.assertRaises(AssertionError):runner.main()
            paid.assert_not_called()
    def test_actual_six_failure_shapes_classified_offline(self):
        base=Path(__file__).resolve().parents[1]/'outputs/stage_g/teacher_production_d042_v1/mirai_retry6_trial_v3/execution_v1/attempts/attempt_01/stream_diagnostics'
        counts={'stop':0,'retry':0}
        for path in base.glob('*.json'):
            d=json.loads(path.read_text(encoding='utf-8'))
            if 'http_status' in d:counts[policy.retry_plan(d,1,epoch=datetime.fromisoformat(d['finished_utc']).timestamp())['action']]+=1
        self.assertEqual(counts,{'stop':4,'retry':2})


if __name__=='__main__':unittest.main()
