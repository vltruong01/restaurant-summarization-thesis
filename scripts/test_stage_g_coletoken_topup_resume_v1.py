"""Offline hash-scoped billing recovery checks; no credentials or API calls."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import resume_stage_g_coletoken_after_topup_v1 as runner


class BillingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / 'checkpoints/job.json'
        self.cp = {'status': 'FAILED', 'http_status': 403, 'error_type': 'HTTPError',
                   'binding': {'provider': 'coletoken'}, 'failed_utc': '2026-10-12T00:00:00+00:00'}
        runner.parent.frozen.immutable(self.path, self.cp)
        self.entry = {'checkpoint_path': 'checkpoints/job.json',
                      'checkpoint_sha256': runner.parent.frozen.file_sha(self.path)}
        self.attempts = [{'directory': self.root}]
        self.epoch = datetime.fromisoformat(self.cp['failed_utc']).timestamp()
    def tearDown(self):
        self.temporary.cleanup()
    def decide(self, state=('fatal', 0), whitelist=True, count=1, epoch=None):
        with patch.object(runner, 'ROOT', self.root), \
             patch.object(runner, 'AUTHORIZED', {'job': self.entry} if whitelist else {}), \
             patch.object(runner, 'ORIGINAL_DECISION', return_value=state):
            return runner.billing_resume_decision('job', self.attempts*count, self.epoch+120 if epoch is None else epoch)
    def test_explicit_hash_bound_billing_checkpoint_recovers(self):
        self.assertEqual(self.decide(), ('ready', 0))
    def test_unlisted_auth_failure_remains_fatal(self):
        self.assertEqual(self.decide(whitelist=False), ('fatal', 0))
    def test_new_failure_does_not_inherit_recovery_permission(self):
        self.path.write_text('{}', encoding='utf-8')
        self.assertEqual(self.decide(), ('fatal', 0))
    def test_success_not_regenerated(self):
        self.assertEqual(self.decide(state=('accepted', 0)), ('accepted', 0))
    def test_unresolved_intent_not_resent(self):
        self.assertEqual(self.decide(state=('unresolved', 0)), ('unresolved', 0))
    def test_exhausted_limit_not_reset(self):
        self.assertEqual(self.decide(count=3), ('exhausted', 0))
    def test_cooldown_preserved(self):
        self.assertEqual(self.decide(count=2, epoch=self.epoch+10), ('cooldown', 50))
    def test_unlisted_exhausted_review_stays_deferred(self):
        self.assertEqual(self.decide(state=('exhausted', 0), whitelist=False), ('exhausted', 0))


if __name__ == '__main__':
    unittest.main()
