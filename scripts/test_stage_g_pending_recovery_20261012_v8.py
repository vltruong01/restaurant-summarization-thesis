"""Offline scheduler safety checks; synthetic checkpoints, no API or target edits."""
import json
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import continue_stage_g_pending_recovery_20261012_v8 as runner


class RecoverySafety(unittest.TestCase):
    def setUp(self):
        self.checkpoints = {}
        self.codes = {}
        self.old_globals = (runner.ALLOW, runner.QUOTA_GATE_JOB, runner.PENDING_CACHE,
                            runner.FRESH_CACHE, runner.FRESH_JOB_ORDER, runner.IMPORTED_IDS,
                            runner.parent.ATTEMPTS)
        runner.ALLOW = {}
        runner.QUOTA_GATE_JOB = None
        runner.PENDING_CACHE = None
        runner.FRESH_CACHE = None
        runner.FRESH_JOB_ORDER = ('fresh1', 'fresh2', 'fresh3', 'fresh4', 'fresh5', 'fresh6')
        runner.IMPORTED_IDS = frozenset()
        runner.parent.ATTEMPTS = defaultdict(list)
        self.cp_patch = patch.object(runner.parent, 'checkpoint', side_effect=
                                     lambda jid, attempt: self.checkpoints.get((jid, attempt['directory'])))
        self.code_patch = patch.object(runner, 'failure_code', side_effect=
                                       lambda jid, attempt: self.codes.get((jid, attempt['directory'])))
        self.cp_patch.start()
        self.code_patch.start()

    def tearDown(self):
        self.code_patch.stop()
        self.cp_patch.stop()
        (runner.ALLOW, runner.QUOTA_GATE_JOB, runner.PENDING_CACHE,
         runner.FRESH_CACHE, runner.FRESH_JOB_ORDER, runner.IMPORTED_IDS,
         runner.parent.ATTEMPTS) = self.old_globals

    def attempt(self, jid, number=1, code=None, status='FAILED', http_status=None, error=None, known=True):
        directory = runner.ROOT / 'synthetic_offline_checkpoints' / str(number)
        item = {'directory': directory, 'historical': False}
        runner.parent.ATTEMPTS[jid].append(item)
        if known:
            self.checkpoints[(jid, directory)] = {'status': status, 'http_status': http_status,
                'local_validation_error': error, 'failed_utc': '2000-01-01T00:00:00+00:00'}
        self.codes[(jid, directory)] = code
        runner.PENDING_CACHE = None
        return item

    def authorize(self, jid):
        attempts = runner.parent.ATTEMPTS[jid]
        runner.ALLOW[jid] = {'recorded_attempts': len(attempts),
            'maximum_recorded_attempts': len(attempts) + 2,
            'provider_failure_code': self.codes.get((jid, attempts[-1]['directory'])),
            'checkpoint_path': (attempts[-1]['directory'] / 'checkpoints' / (jid+'.json')).relative_to(runner.ROOT).as_posix()}

    def test_nested_quota_code_and_direct_upstream_code(self):
        self.assertEqual(runner.provider_error_code('{"error":{"code":"insufficient_quota"}}'), 'insufficient_quota')
        self.assertEqual(runner.provider_error_code({'code': 'upstream_http2_stream_error'}), 'upstream_http2_stream_error')

    def test_non_json_rate_limit_does_not_invent_quota(self):
        self.assertIsNone(runner.provider_error_code('rate limit'))
        self.assertEqual(runner.provider_error_code({'error': {'code': 'rate_limit_exceeded'}}), 'rate_limit_exceeded')

    def test_only_one_old_quota_job_is_admitted_before_gate_pass(self):
        for jid in ('gate', 'other'):
            self.attempt(jid, code='insufficient_quota', http_status=429)
            self.authorize(jid)
        runner.QUOTA_GATE_JOB = 'gate'
        self.assertEqual(runner.recovery_decision('gate', runner.parent.ATTEMPTS['gate'])[0], 'ready')
        self.assertEqual(runner.recovery_decision('other', runner.parent.ATTEMPTS['other'])[0], 'deferred')
        self.assertEqual(runner.recovery_decision('fresh1', [])[0], 'deferred')

    def test_new_quota_failure_is_fatal_even_after_renewal_authorization(self):
        self.attempt('gate', code='insufficient_quota', http_status=429)
        self.authorize('gate')
        self.attempt('gate', 2, code='insufficient_quota', http_status=429)
        self.assertEqual(runner.recovery_decision('gate', runner.parent.ATTEMPTS['gate'])[0], 'fatal')

    def test_unlisted_old_quota_is_fatal(self):
        self.attempt('not_authorized', code='insufficient_quota', http_status=429)
        self.assertEqual(runner.recovery_decision('not_authorized', runner.parent.ATTEMPTS['not_authorized'])[0], 'fatal')

    def test_validated_gate_unlocks_at_most_three_recoveries(self):
        for jid in ('gate', 'a', 'b', 'c', 'd'):
            self.attempt(jid, code='insufficient_quota' if jid == 'gate' else None)
            self.authorize(jid)
        runner.QUOTA_GATE_JOB = 'gate'
        self.attempt('gate', 2, status=runner.parent.PASS)
        self.assertTrue(runner.quota_gate_passed())
        ready = [jid for jid in ('a','b','c','d') if runner.recovery_decision(jid, runner.parent.ATTEMPTS[jid])[0] == 'ready']
        self.assertEqual(ready, ['a', 'b', 'c'])
        self.assertEqual(runner.recovery_decision('fresh1', [])[0], 'deferred')

    def test_failed_gate_stops_when_bounded_attempts_are_exhausted(self):
        self.attempt('gate', code='insufficient_quota')
        self.authorize('gate')
        runner.QUOTA_GATE_JOB = 'gate'
        self.attempt('gate', 2)
        self.attempt('gate', 3)
        self.assertEqual(runner.recovery_decision('gate', runner.parent.ATTEMPTS['gate'])[0], 'fatal')

    def test_unknown_outcome_is_not_resent(self):
        self.attempt('unknown', known=False)
        self.authorize('unknown')
        self.assertEqual(runner.recovery_decision('unknown', runner.parent.ATTEMPTS['unknown'])[0], 'unresolved')

    def test_authentication_and_model_provenance_stops_remain(self):
        for status in (401, 402, 403):
            jid = 'auth' + str(status)
            self.attempt(jid, http_status=status)
            self.authorize(jid)
            self.assertEqual(runner.recovery_decision(jid, runner.parent.ATTEMPTS[jid])[0], 'fatal')
        self.attempt('alias', error='Reported model differs from requested alias; review provider provenance')
        self.authorize('alias')
        self.assertEqual(runner.recovery_decision('alias', runner.parent.ATTEMPTS['alias'])[0], 'fatal')

    def test_accepted_checkpoint_is_never_resent(self):
        self.attempt('accepted', status=runner.parent.PASS)
        self.authorize('accepted')
        self.assertEqual(runner.recovery_decision('accepted', runner.parent.ATTEMPTS['accepted'])[0], 'accepted')

    def test_fresh_window_admits_five_and_skips_imported_or_attempted(self):
        runner.IMPORTED_IDS = frozenset(['fresh1'])
        self.attempt('fresh2')
        self.assertEqual(runner.select_fresh_window(runner.FRESH_JOB_ORDER, runner.parent.ATTEMPTS, runner.IMPORTED_IDS),
                         ('fresh3', 'fresh4', 'fresh5', 'fresh6'))
        self.assertEqual(sum(runner.recovery_decision(jid, [])[0] == 'ready' for jid in runner.FRESH_JOB_ORDER[2:]), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
