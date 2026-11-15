"""Offline tests for bounded recovery; no credentials or API calls."""
import unittest
from unittest.mock import patch

import continue_stage_g_pending_recovery_20261012_v1 as runner


class RecoveryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.saved_allow = runner.ALLOW
        self.saved_attempts = runner.parent.ATTEMPTS
        runner.ALLOW = {}
        runner.parent.ATTEMPTS = {}
        self.mock = patch.object(runner.parent, 'checkpoint', side_effect=lambda jid, a: a.get('cp'))
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        runner.ALLOW = self.saved_allow
        runner.parent.ATTEMPTS = self.saved_attempts

    def failure(self, jid, count=1, **fields):
        attempts = [{'cp': dict(status='FAILED_STOP_NO_AUTOMATIC_RETRY', **fields)} for _ in range(count)]
        runner.parent.ATTEMPTS[jid] = attempts
        return attempts

    def test_only_first_three_confirmed_failures_are_ready(self):
        for jid in ('a','b','c','d','e'):
            self.failure(jid)
            runner.ALLOW[jid] = {'recorded_attempts': 1}
        ready = [j for j in runner.ALLOW if runner.recovery_decision(j,runner.parent.ATTEMPTS[j])[0]=='ready']
        self.assertEqual(ready,['a','b','c'])

    def test_no_new_jobs_until_recovery_is_dispatched(self):
        self.failure('a')
        runner.ALLOW['a'] = {'recorded_attempts':1}
        runner.parent.ATTEMPTS['new'] = []
        self.assertEqual(runner.recovery_decision('new',[])[0],'deferred')

    def test_exactly_one_extra_attempt_even_for_exhausted_failure(self):
        attempts = self.failure('a',3)
        runner.ALLOW['a'] = {'recorded_attempts':3}
        self.assertEqual(runner.recovery_decision('a',attempts)[0],'ready')
        attempts.append({'cp': {'status':'FAILED_STOP_NO_AUTOMATIC_RETRY'}})
        self.assertEqual(runner.recovery_decision('a',attempts)[0],'deferred')

    def test_unknown_outcome_is_never_resent(self):
        runner.parent.ATTEMPTS['a'] = [{'cp':None}]
        runner.ALLOW['a'] = {'recorded_attempts':1}
        self.assertEqual(runner.recovery_decision('a',runner.parent.ATTEMPTS['a'])[0],'unresolved')

    def test_successful_component_is_never_resent(self):
        runner.parent.ATTEMPTS['a'] = [{'cp':{'status':runner.parent.PASS}}]
        runner.ALLOW['a'] = {'recorded_attempts':1}
        self.assertEqual(runner.recovery_decision('a',runner.parent.ATTEMPTS['a'])[0],'accepted')

    def test_billing_or_alias_failure_still_stops(self):
        for fields in ({'http_status':402}, {'local_validation_error':'Reported model differs from requested alias; review provider provenance'}):
            attempts = self.failure('a',**fields)
            runner.ALLOW['a'] = {'recorded_attempts':1}
            self.assertEqual(runner.recovery_decision('a',attempts)[0],'fatal')

    def test_after_bounded_pass_new_jobs_continue_and_new_failures_defer(self):
        attempts = self.failure('a',2)
        runner.ALLOW['a'] = {'recorded_attempts':1}
        runner.parent.ATTEMPTS['new'] = []
        self.assertEqual(runner.recovery_decision('a',attempts)[0],'deferred')
        self.assertEqual(runner.recovery_decision('new',[])[0],'ready')
        attempts = self.failure('later')
        self.assertEqual(runner.recovery_decision('later',attempts)[0],'deferred')


if __name__=='__main__':
    unittest.main()
