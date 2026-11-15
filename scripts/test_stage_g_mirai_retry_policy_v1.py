"""Offline operational recovery tests; no credentials or paid calls."""
import json
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path
import tempfile

import stage_g_mirai_retry_policy_v1 as policy
import run_stage_g_mirai_retry6_trial_v3 as runner

FINISHED = '2026-10-12T04:48:15+00:00'
EPOCH = datetime.fromisoformat(FINISHED).timestamp()


def diagnostic(status=404, kind='upstream_error', message='[403]: insufficient balance (reset after 2m)'):
    return {'http_status': status, 'provider_http_error_body_redacted': json.dumps({'error': {'type': kind, 'message': message}}),
            'finished_utc': FINISHED}


class PolicyTests(unittest.TestCase):
    def test_explicit_wrapped_upstream_billing_waits_two_minutes(self):
        plan = policy.retry_plan(diagnostic(), 1, epoch=EPOCH)
        self.assertEqual(plan['action'], 'retry')
        self.assertEqual(plan['wait_seconds'], 120)
    def test_ordinary_404_not_retried(self):
        self.assertEqual(policy.retry_plan(diagnostic(message='route not found'), 1)['action'], 'defer')
    def test_account_billing_terminal_even_with_reset_words(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=403), 1)['action'], 'stop')
    def test_auth_terminal(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=401), 1)['action'], 'stop')
    def test_outer_error_type_required(self):
        self.assertEqual(policy.retry_plan(diagnostic(kind='billing_error'), 1)['action'], 'defer')
    def test_no_reset_billing_not_retried(self):
        self.assertEqual(policy.retry_plan(diagnostic(message='insufficient balance'), 1)['action'], 'defer')
    def test_historical_elapsed_cooldown_not_waited_twice(self):
        self.assertEqual(policy.retry_plan(diagnostic(), 1, epoch=EPOCH+180)['wait_seconds'], 0)
    def test_initial_plus_two_attempt_limit(self):
        self.assertEqual(policy.retry_plan(diagnostic(), 3, epoch=EPOCH)['reason'], 'attempt_limit')
    def test_first_and_second_backoff(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=503), 1, epoch=EPOCH)['wait_seconds'], 30)
        self.assertEqual(policy.retry_plan(diagnostic(status=503), 2, epoch=EPOCH)['wait_seconds'], 60)
    def test_retry_after_seconds(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=429), 1, '90', epoch=EPOCH)['wait_seconds'], 90)
    def test_retry_after_date(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=429), 1, 'Mon, 12 Oct 2026 04:50:15 GMT', epoch=EPOCH)['wait_seconds'], 120)
    def test_timeout_without_terminal_provider_error_not_retried(self):
        self.assertEqual(policy.retry_plan({'error_type': 'TimeoutError'}, 1)['action'], 'defer')
    def test_excessive_provider_cooldown_is_not_ignored(self):
        self.assertEqual(policy.retry_plan(diagnostic(status=429), 1, '1200', epoch=EPOCH)['action'], 'defer')
    def test_frozen_settings_unchanged(self):
        new = runner.frozen.read(runner.CONFIG)
        old = runner.frozen.read(runner.old.CONFIG)
        for key in runner.frozen.FIXED_KEYS:
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['identifier_guard'], old['identifier_guard'])
        self.assertEqual(new['execution']['maximum_concurrency'], 6)
        self.assertFalse(new['full_generation_enabled'])
    def test_selection_excludes_prior_successes_and_attempted_businesses(self):
        jobs=[]; businesses=[]
        for bid in ('old_a', 'old_b', 'attempted', 'new_a', 'new_b'):
            businesses.append({'business_id': bid, 'split': 'train'})
            jobs += [{'job_id': bid+'_'+str(i), 'business_id': bid, 'split': 'train'} for i in range(3)]
        old_ids=[j['job_id'] for j in jobs[:6]]
        previous={'results': [{'job_id': jid, 'content_valid': i != 5} for i, jid in enumerate(old_ids)]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'attempted/checkpoints').mkdir(parents=True)
            (root/'attempted/checkpoints/attempted_0.json').write_text('{}')
            with patch.object(runner.frozen, 'read', return_value=diagnostic()), \
                 patch.object(runner.old.prior.parent, 'load_attempts'), \
                 patch.object(runner.old.prior.parent, 'read_index', return_value={}), \
                 patch.object(runner.old.prior.parent, 'location', side_effect=lambda j, _: root/j['business_id']):
                selected, reused, plan=runner.select_jobs(jobs, businesses, set(), previous)
        self.assertEqual(selected[0]['job_id'], old_ids[5])
        self.assertEqual(len(selected), 6)
        self.assertEqual(set(reused), set(old_ids[:5]))
        self.assertTrue(all(j['business_id'] in ('new_a', 'new_b') for j in selected[1:]))


if __name__ == '__main__':
    unittest.main()
