"""Offline guard checks using frozen jobs and preserved API responses; no API/key access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import stage_g_business_id_guard_v1 as guard
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
from prepare_stage_g_teacher_pilot import ROOT, canonical, digest, file_sha, read
from stage_g_canonical_review_ids import citation_diagnostics, resolve_review


class BusinessIdGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / 'configs/stage_g_teacher_production_d042.json'
        cls.cfg = read(cls.config_path)
        cls.schemas = read(ROOT / cls.cfg['schema'])
        cls.jobs = read(ROOT / 'outputs/stage_g/teacher_production_d042_v1/production_jobs.json')
        cls.index = read(ROOT / 'outputs/stage_g/teacher_production_d042_v1/restored_d042_runner_v1/reuse_index.json')
        cls.review_job = next(j for j in cls.jobs if j['task'] == 'review_consensus' and j['job_id'] in cls.index)
        cls.visual_job = next(j for j in cls.jobs if j['task'] == 'visual_evidence' and not j.get('skip_api_empty_captions') and j['job_id'] in cls.index)
        cls.empty_job = next(j for j in cls.jobs if j.get('skip_api_empty_captions'))

    def preserved_output(self, job):
        path = ROOT / self.index[job['job_id']] / 'checkpoints' / (job['job_id'] + '.json')
        return deepcopy(read(path)['output'])

    def test_valid_preserved_review_and_visual_outputs_pass_without_edits(self):
        for job in (self.review_job, self.visual_job):
            with self.subTest(task=job['task']):
                schema = self.schemas[job['task']]
                output = self.preserved_output(job)
                original = deepcopy(output)
                self.assertEqual(guard.validate_output(job, output, schema), legacy.validate_output(job, output, schema))
                self.assertEqual(output, original)

    def test_one_character_or_cross_business_id_is_rejected(self):
        for job in (self.review_job, self.visual_job):
            schema = self.schemas[job['task']]
            alternatives = [job['business_id'][:-1] + ('X' if job['business_id'][-1] != 'X' else 'Y'),
                            next(j['business_id'] for j in self.jobs if j['business_id'] != job['business_id'])]
            for wrong in alternatives:
                with self.subTest(task=job['task'], kind='wrong_id'):
                    output = self.preserved_output(job)
                    output['business_id'] = wrong
                    with self.assertRaises(ValueError):
                        guard.validate_output(job, output, schema)

    def test_guard_changes_only_top_level_business_id_enum(self):
        for job in (self.review_job, self.visual_job):
            original = deepcopy(self.schemas[job['task']])
            guarded = guard.schema_for_job(job, original)
            self.assertEqual(guarded['properties']['business_id']['enum'], [job['business_id']])
            restored = deepcopy(guarded)
            del restored['properties']['business_id']['enum']
            self.assertEqual(restored, original)
            self.assertEqual(original, self.schemas[job['task']])

    def test_concurrent_and_sequential_guards_cannot_mutate_shared_schema(self):
        jobs = self.jobs[:240]
        before = deepcopy(self.schemas)
        def build(job):
            schema = guard.schema_for_job(job, self.schemas[job['task']])
            self.assertEqual(schema['properties']['business_id']['enum'], [job['business_id']])
            schema['properties']['business_id']['enum'].append('LOCAL_TEST_MUTATION')
            schema['required'].append('LOCAL_TEST_MUTATION')
            return schema
        with ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(len(list(pool.map(build, jobs))), len(jobs))
        for job in jobs:
            self.assertEqual(guard.schema_for_job(job, self.schemas[job['task']])['properties']['business_id']['enum'], [job['business_id']])
        self.assertEqual(self.schemas, before)

    def test_guarded_request_diff_is_only_the_business_id_enum(self):
        for job in (self.review_job, self.visual_job):
            with self.subTest(task=job['task']):
                schema = self.schemas[job['task']]
                prompt = (ROOT / self.cfg['prompts'][job['task']]).read_text(encoding='utf-8')
                original = legacy.payload(job, self.cfg, prompt, schema)
                guarded = guard.payload_for_job(job, self.cfg, prompt, schema)
                self.assertEqual(guarded['text']['format']['schema']['properties']['business_id']['enum'], [job['business_id']])
                del guarded['text']['format']['schema']['properties']['business_id']['enum']
                self.assertEqual(guarded, original)

    def test_invalid_source_evidence_still_fails(self):
        review = self.preserved_output(self.review_job)
        self.assertTrue(review['claims'] and review['claims'][0]['support'])
        review['claims'][0]['support'][0]['evidence_span_ids'] = ['REV_999_SEG_999']
        with self.assertRaises(ValueError):
            guard.validate_output(self.review_job, review, self.schemas['review_consensus'])
        visual = self.preserved_output(self.visual_job)
        visual['assessments'][0]['photo_id'] = 'LOCAL_TEST_WRONG_PHOTO'
        with self.assertRaises(ValueError):
            guard.validate_output(self.visual_job, visual, self.schemas['visual_evidence'])

    def fixture_checkpoint(self, directory, job):
        """Construct only a temporary integrity fixture from preserved API output."""
        source = ROOT / self.index[job['job_id']]
        cp = deepcopy(read(source / 'checkpoints' / (job['job_id'] + '.json')))
        schema = self.schemas[job['task']]
        cp['binding'].update(job_sha256=digest(canonical(job)), config_sha256=file_sha(self.config_path),
            schema_sha256=digest(canonical(guard.schema_for_job(job, schema))))
        name = job['job_id'] + '.json'
        raw = read(source / 'raw_responses' / name)
        for part in ('checkpoints', 'raw_responses', 'raw_requests', 'request_intents'):
            (directory / part).mkdir(parents=True, exist_ok=True)
        raw_path = directory / 'raw_responses' / name
        raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        cp['raw_response_sha256'] = file_sha(raw_path)
        prompt = (ROOT / self.cfg['prompts'][job['task']]).read_text(encoding='utf-8')
        body = canonical(guard.payload_for_job(job, self.cfg, prompt, schema))
        (directory / 'raw_requests' / (job['job_id'] + '.json.gz')).write_bytes(gzip.compress(body))
        (directory / 'request_intents' / name).write_text(json.dumps({
            'binding': cp['binding'], 'request_sha256': digest(body)}), encoding='utf-8')
        (directory / 'checkpoints' / name).write_text(json.dumps(cp, ensure_ascii=False), encoding='utf-8')
        return cp

    def test_guarded_checkpoint_integrity_and_old_schema_hash_rejection(self):
        with tempfile.TemporaryDirectory(prefix='stage_g_guard_offline_') as temp:
            directory = Path(temp)
            job = self.review_job
            cp = self.fixture_checkpoint(directory, job)
            self.assertEqual(guard.verify_checkpoint(job, self.cfg, directory, self.config_path), cp)
            cp['binding']['schema_sha256'] = job['schema_sha256']
            (directory / 'checkpoints' / (job['job_id'] + '.json')).write_text(json.dumps(cp), encoding='utf-8')
            with self.assertRaises(AssertionError):
                guard.verify_checkpoint(job, self.cfg, directory, self.config_path)

    def test_archived_unguarded_request_cannot_pass_guarded_verification(self):
        with tempfile.TemporaryDirectory(prefix='stage_g_guard_offline_') as temp:
            directory = Path(temp)
            job = self.review_job
            self.fixture_checkpoint(directory, job)
            prompt = (ROOT / self.cfg['prompts'][job['task']]).read_text(encoding='utf-8')
            original = canonical(legacy.payload(job, self.cfg, prompt, self.schemas[job['task']]))
            (directory / 'raw_requests' / (job['job_id'] + '.json.gz')).write_bytes(gzip.compress(original))
            with self.assertRaises(AssertionError):
                guard.verify_checkpoint(job, self.cfg, directory, self.config_path)

    def test_frozen_empty_visual_condition_is_still_valid_without_api(self):
        job = self.empty_job
        output = {'business_id': job['business_id'], 'condition': job['condition'], 'assessments': []}
        schema = self.schemas['visual_evidence']
        with tempfile.TemporaryDirectory(prefix='stage_g_guard_offline_') as temp:
            directory = Path(temp)
            (directory / 'checkpoints').mkdir()
            cp = {'status': 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW',
                'binding': {'job_sha256': digest(canonical(job)), 'config_sha256': file_sha(self.config_path),
                    'prompt_sha256': job['prompt_sha256'], 'schema_sha256': digest(canonical(guard.schema_for_job(job, schema))),
                    'base_url': 'https://www.coletoken.de/v1', 'provider': 'coletoken', 'requested_model': self.cfg['requested_model']},
                'api_called': False, 'creation': 'deterministic_empty_visual_condition',
                'output': output, 'validation': guard.validate_output(job, output, schema), 'usage': None}
            (directory / 'checkpoints' / (job['job_id'] + '.json')).write_text(json.dumps(cp), encoding='utf-8')
            self.assertEqual(guard.verify_checkpoint(job, self.cfg, directory, self.config_path), cp)


if __name__ == '__main__':
    unittest.main(verbosity=2)
