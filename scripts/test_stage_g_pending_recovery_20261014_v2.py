"""Offline checks of outage handling and the recovery-before-fresh-work barrier."""
import unittest

import test_stage_g_pending_recovery_20261012_v8 as base
import continue_stage_g_pending_recovery_20261014_v2 as runner

base.runner = runner


class RecoverySafety(base.RecoverySafety):
    def setUp(self):
        self.service_globals = runner.SERVICE_GATE_JOB, runner.SERVICE_UNAVAILABLE
        runner.SERVICE_GATE_JOB = None
        runner.SERVICE_UNAVAILABLE = False
        super().setUp()

    def tearDown(self):
        super().tearDown()
        runner.SERVICE_GATE_JOB, runner.SERVICE_UNAVAILABLE = self.service_globals

    def test_exhausted_recovery_does_not_unlock_fresh_jobs(self):
        self.attempt('failed')
        self.authorize('failed')
        self.attempt('failed', 2)
        self.attempt('failed', 3)
        self.assertEqual(runner.fresh_window(), frozenset())
        self.assertEqual(runner.recovery_decision('fresh1', [])[0], 'deferred')

    def test_unknown_recovery_does_not_unlock_fresh_jobs(self):
        self.attempt('unknown')
        self.authorize('unknown')
        self.attempt('unknown', 2, known=False)
        self.assertEqual(runner.fresh_window(), frozenset())

    def test_all_passed_recoveries_unlock_only_five_fresh_jobs(self):
        self.attempt('repaired')
        self.authorize('repaired')
        self.attempt('repaired', 2, status=runner.parent.PASS)
        self.assertEqual(len(runner.fresh_window()), 5)

    def test_service_gate_admits_only_one_actual_recovery(self):
        for jid in ('a', 'b', 'c'):
            self.attempt(jid)
            self.authorize(jid)
        runner.SERVICE_GATE_JOB = 'a'
        self.assertEqual(runner.recovery_decision('a', runner.parent.ATTEMPTS['a'])[0], 'ready')
        self.assertEqual(runner.recovery_decision('b', runner.parent.ATTEMPTS['b'])[0], 'deferred')
        self.assertEqual(runner.recovery_decision('fresh1', [])[0], 'deferred')

    def test_passed_service_gate_allows_bounded_recovery(self):
        for jid in ('a', 'b', 'c', 'd', 'e'):
            self.attempt(jid)
            self.authorize(jid)
        runner.SERVICE_GATE_JOB = 'a'
        self.attempt('a', 2, status=runner.parent.PASS)
        ready = [jid for jid in ('b', 'c', 'd', 'e')
                 if runner.recovery_decision(jid, runner.parent.ATTEMPTS[jid])[0] == 'ready']
        self.assertEqual(ready, ['b', 'c', 'd'])

    def test_503_blocks_further_failed_and_fresh_dispatches(self):
        self.attempt('failed')
        self.authorize('failed')
        runner.SERVICE_UNAVAILABLE = True
        self.assertEqual(runner.recovery_decision('failed', runner.parent.ATTEMPTS['failed'])[0], 'fatal')
        self.assertEqual(runner.recovery_decision('fresh1', [])[0], 'fatal')
        self.attempt('accepted', status=runner.parent.PASS)
        self.assertEqual(runner.recovery_decision('accepted', runner.parent.ATTEMPTS['accepted'])[0], 'accepted')


if __name__ == '__main__':
    unittest.main()
