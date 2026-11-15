"""Offline continuation regression checks; no keys or network calls."""
import gzip
import json
import unittest
from copy import deepcopy
from pathlib import Path

import stage_g_production_d042 as frozen
import stage_g_business_id_guard_v1 as guard
import continue_stage_g_guarded_production6_v1 as continuation
import run_stage_g_teacher_pilot_v4_1_recovery as legacy


class SixWorkerContinuationTests(unittest.TestCase):
    def test_runner_changes_only_scheduling(self):
        source = (frozen.ROOT / 'scripts/run_stage_g_production_d042.py').read_text(encoding='utf-8')
        for old, new in {
            '"""Approved D042 teacher-only production: smoke gate, four bounded workers, no retries."""':
                '"""Versioned D042 six-worker drained batches; frozen target logic and zero automatic retries."""',
            "'maximum_concurrency': 4": "'maximum_concurrency': 6",
            'ThreadPoolExecutor(max_workers=4)': 'ThreadPoolExecutor(max_workers=6)',
            'batch = pending[:4]': 'batch = pending[:6]',
        }.items():
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new)
        self.assertEqual(source, (frozen.ROOT / 'scripts/run_stage_g_production_d042_six_worker_v1.py').read_text(encoding='utf-8'))

    def test_scientific_settings_unchanged(self):
        new = frozen.read(continuation.CONFIG)
        parent = frozen.read(continuation.parent.CONFIG)
        for field in frozen.FIXED_KEYS + ['identifier_guard']:
            self.assertEqual(new[field], parent[field], field)
        self.assertEqual(new['execution']['maximum_concurrency'], 6)
        self.assertEqual(new['generation_proposal']['automatic_retries'], 0)

    def test_invalid_source_spans_still_rejected_without_edits(self):
        jobs = frozen.read(frozen.FOLDER / 'production_jobs.json')
        job = next(j for j in jobs if j['job_id'] == continuation.FAILED_JOB)
        cfg = frozen.read(continuation.CONFIG)
        raw = frozen.read(continuation.parent.OUT / 'raw_responses' / (job['job_id'] + '.json'))
        output = json.loads(legacy.response_text(raw)); saved = deepcopy(output)
        with self.assertRaisesRegex(ValueError, 'wrong-review source spans'):
            guard.validate_output(job, output, frozen.read(frozen.ROOT / cfg['schema'])[job['task']])
        self.assertEqual(output, saved)
        self.assertEqual(continuation.source_failure_audit(job)['issue_count'], 8)

    def test_reviewed_retry_payload_exactly_matches_previous_attempt(self):
        jobs = frozen.read(frozen.FOLDER / 'production_jobs.json')
        job = next(j for j in jobs if j['job_id'] == continuation.FAILED_JOB)
        cfg = frozen.read(continuation.CONFIG)
        prompt = (frozen.ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
        schema = frozen.read(frozen.ROOT / cfg['schema'])[job['task']]
        old = gzip.decompress((continuation.parent.OUT / 'raw_requests' / (job['job_id'] + '.json.gz')).read_bytes())
        self.assertEqual(old, frozen.canonical(guard.payload_for_job(job, cfg, prompt, schema)))


if __name__ == '__main__': unittest.main(verbosity=2)
