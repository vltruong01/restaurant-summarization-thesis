"""Offline scheduler checks. No API calls or project output mutations."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continue_stage_g_production6_defer_errors_20261012_v1 as overlay


class DeferredSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.jid = 'offline_fixture'

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, number=1, **values):
        folder = self.folder / str(number)
        overlay.parent.frozen.immutable(folder / 'checkpoints' / (self.jid + '.json'), {
            'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY', 'error_type': 'ValueError', **values})
        return {'directory': folder, 'historical': number == 1}

    def test_only_untouched_job_is_ready(self):
        self.assertEqual(overlay.defer_decision(self.jid, []), ('ready', 0))

    def test_recorded_schema_failure_not_resent(self):
        self.assertEqual(overlay.defer_decision(self.jid, [self.attempt()]), ('deferred', 0))

    def test_provider_limit_failure_not_resent(self):
        self.assertEqual(overlay.defer_decision(self.jid, [self.attempt(error_type='RuntimeError')]), ('deferred', 0))

    def test_previous_retry_success_is_reused(self):
        attempts = [self.attempt(), self.attempt(2, status=overlay.parent.PASS)]
        self.assertEqual(overlay.defer_decision(self.jid, attempts), ('accepted', 0))

    def test_historical_repeated_failure_stays_deferred(self):
        attempts = [self.attempt(i) for i in (1, 2, 3)]
        self.assertEqual(overlay.defer_decision(self.jid, attempts, epoch=1e20), ('deferred', 0))

    def test_unknown_outcome_not_resent(self):
        folder = self.folder / 'unknown'
        overlay.parent.frozen.immutable(folder / 'request_intents' / (self.jid + '.json'), {'status': 'REQUEST_STARTED'})
        self.assertEqual(overlay.defer_decision(self.jid, [{'directory': folder}]), ('unresolved', 0))

    def test_auth_and_billing_still_fatal(self):
        for status in (401, 402, 403):
            with self.subTest(status=status):
                self.assertEqual(overlay.defer_decision(self.jid, [self.attempt(status, http_status=status)]), ('fatal', 0))

    def test_alias_mismatch_still_fatal(self):
        self.assertEqual(overlay.defer_decision(self.jid, [self.attempt(
            local_validation_error='Reported model differs from requested alias; review provider provenance')]), ('fatal', 0))

    def test_failed_bytes_preserved(self):
        attempt = self.attempt()
        path = attempt['directory'] / 'checkpoints' / (self.jid + '.json')
        before = path.read_bytes()
        overlay.defer_decision(self.jid, [attempt])
        self.assertEqual(path.read_bytes(), before)

    def test_execution_overlay_keeps_frozen_config(self):
        policy = overlay.parent.frozen.read(overlay.POLICY)
        self.assertEqual(policy['frozen_request_config'], overlay.parent.CONFIG.relative_to(overlay.ROOT).as_posix())
        self.assertEqual(policy['automatic_retries'], 0)
        self.assertEqual(policy['maximum_concurrency'], 6)
        self.assertFalse(policy['scientific_payload_or_validation_changed'])

    def test_configure_does_not_mutate_payload_or_validation(self):
        original_payload = overlay.parent.guard.payload_for_job
        original_validator = overlay.parent.verify_checkpoint
        originals = {k: getattr(overlay.parent, k) for k in ('retry_decision', 'MAX_RETRIES', 'SCRIPT', 'update_docs')}
        with patch.multiple(overlay.parent, **originals):
            overlay.configure()
            self.assertIs(overlay.parent.guard.payload_for_job, original_payload)
            self.assertIs(overlay.parent.verify_checkpoint, original_validator)
            self.assertEqual(overlay.parent.MAX_RETRIES, 0)


if __name__ == '__main__':
    unittest.main()
