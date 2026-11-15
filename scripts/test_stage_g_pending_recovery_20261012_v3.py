"""Offline regression tests: known-failure retry bounds and no duplicate calls."""
import unittest
from unittest.mock import patch
import continue_stage_g_pending_recovery_20261012_v3 as runner


class BoundedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.old_allow, self.old_attempts = runner.ALLOW, runner.parent.ATTEMPTS
        runner.ALLOW = {}
        runner.parent.ATTEMPTS = {}
        runner.PENDING_CACHE = None
        self.patcher = patch.object(runner.parent, 'checkpoint', side_effect=lambda jid, a: a.get('cp'))
        self.checkpoint_mock = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        runner.ALLOW, runner.parent.ATTEMPTS = self.old_allow, self.old_attempts
        runner.PENDING_CACHE = None

    def add(self, jid, count=1, cp=None):
        attempts = [{'cp': cp or {'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY',
                                  'failed_utc': '2000-01-01T00:00:00+00:00'}} for _ in range(count)]
        runner.parent.ATTEMPTS[jid] = attempts
        runner.ALLOW[jid] = {'recorded_attempts': count, 'maximum_recorded_attempts': 3}
        return attempts

    def test_three_ready_only(self):
        for jid in 'abcde':
            self.add(jid)
        self.assertEqual([j for j in runner.ALLOW
                          if runner.recovery_decision(j, runner.parent.ATTEMPTS[j])[0] == 'ready'], list('abc'))

    def test_success_never_resent(self):
        attempts = self.add('a', 2, {'status': runner.parent.PASS})
        self.assertEqual(runner.recovery_decision('a', attempts)[0], 'accepted')

    def test_initial_plus_two_retries_only(self):
        attempts = self.add('a')
        self.assertEqual(runner.recovery_decision('a', attempts)[0], 'ready')
        attempts.append(dict(attempts[-1]))
        self.assertEqual(runner.recovery_decision('a', attempts)[0], 'ready')
        attempts.append(dict(attempts[-1]))
        self.assertEqual(runner.recovery_decision('a', attempts)[0], 'deferred')

    def test_unknown_never_resent(self):
        attempts = self.add('a')
        attempts[-1]['cp'] = None
        self.assertEqual(runner.recovery_decision('a', attempts)[0], 'unresolved')

    def test_classic_thirty_sixty_second_cooldown(self):
        for count, delta in [(1, 30), (2, 60)]:
            runner.PENDING_CACHE = None
            attempts = self.add('a', count, {'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY',
                                            'failed_utc': '2026-10-12T00:00:00+00:00'})
            failed = runner.datetime.fromisoformat(attempts[-1]['cp']['failed_utc']).timestamp()
            state, seconds = runner.recovery_decision('a', attempts, failed + 1)
            self.assertEqual((state, seconds), ('cooldown', delta - 1))

    def test_fatal_errors_stop(self):
        for cp in [{'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY', 'http_status': 402},
                   {'status': 'FAILED_STOP_NO_AUTOMATIC_RETRY',
                    'local_validation_error': 'Reported model differs from requested alias; review provider provenance'}]:
            attempts = self.add('a', cp=cp)
            self.assertEqual(runner.recovery_decision('a', attempts)[0], 'fatal')

    def test_new_jobs_wait_then_resume_and_new_errors_defer(self):
        attempts = self.add('a')
        self.assertEqual(runner.recovery_decision('new', [])[0], 'deferred')
        attempts.extend([dict(attempts[0]), dict(attempts[0])])
        self.assertEqual(runner.recovery_decision('new', [])[0], 'ready')
        self.assertEqual(runner.recovery_decision('other', [dict(attempts[0])])[0], 'deferred')

    def test_pending_file_reads_cached_per_batch(self):
        self.add('a')
        runner.remaining_recoveries()
        first = self.checkpoint_mock.call_count
        for _ in range(100):
            runner.remaining_recoveries()
        self.assertEqual(self.checkpoint_mock.call_count, first)
        runner.PENDING_CACHE = None
        runner.remaining_recoveries()
        self.assertGreater(self.checkpoint_mock.call_count, first)


if __name__ == '__main__':
    unittest.main()
