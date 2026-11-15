"""Offline provider-only swap checks; temporary fixtures never become teacher outputs."""
import gzip
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import stage_g_production_d042 as frozen
import stage_g_business_id_guard_v1 as original
import stage_g_mirai_guard_v1 as mirai
import run_stage_g_mirai_guarded6_after_retry_v1 as trial
from test_stage_g_business_id_guard_v1 import BusinessIdGuardTests


class MiraiSwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BusinessIdGuardTests.setUpClass()
        cls.helper=BusinessIdGuardTests('test_valid_preserved_review_and_visual_outputs_pass_without_edits')
        cls.cfg=frozen.read(trial.CONFIG)
        cls.previous=frozen.read(trial.parent.CONFIG)
    def test_validator_source_changes_only_provider_binding(self):
        text=Path(original.__file__).read_text(encoding='utf-8')
        expected=text.replace("'https://www.coletoken.de/v1', 'coletoken', 'gpt-6.1-sol'",
                              "'https://api.miraiapi.com/v1', 'miraiapi', 'gpt-6.1-sol'")
        self.assertEqual(Path(mirai.__file__).read_text(encoding='utf-8'),expected)
    def test_scientific_configuration_identical(self):
        for key in frozen.FIXED_KEYS:self.assertEqual(self.cfg[key],self.previous[key],key)
        self.assertEqual(self.cfg['identifier_guard'],self.previous['identifier_guard'])
        self.assertFalse(self.cfg['full_generation_enabled'])
        self.assertEqual(self.cfg['operational_provider_trial']['maximum_calls'],6)
    def test_review_and_visual_payloads_byte_identical(self):
        for job in (self.helper.review_job,self.helper.visual_job):
            prompt=(frozen.ROOT/self.cfg['prompts'][job['task']]).read_text(encoding='utf-8')
            schema=self.helper.schemas[job['task']]
            self.assertEqual(frozen.canonical(mirai.payload_for_job(job,self.cfg,prompt,schema)),
                             frozen.canonical(original.payload_for_job(job,self.previous,prompt,schema)))
    def test_mirai_provenance_passes_same_evidence_checks_and_coletoken_guard_rejects(self):
        job=self.helper.review_job
        with tempfile.TemporaryDirectory(prefix='mirai_offline_fixture_') as temporary:
            directory=Path(temporary)
            cp=self.helper.fixture_checkpoint(directory,job)
            config=directory/'config.json';frozen.atomic(config,self.cfg)
            cp['binding'].update(base_url=trial.BASE_URL,provider=trial.PROVIDER,config_sha256=frozen.file_sha(config))
            frozen.atomic(directory/'checkpoints'/(job['job_id']+'.json'),cp)
            intent=directory/'request_intents'/(job['job_id']+'.json');intent_content=frozen.read(intent)
            intent_content['binding']=cp['binding'];frozen.atomic(intent,intent_content)
            self.assertEqual(mirai.verify_checkpoint(job,self.cfg,directory,config),cp)
            with self.assertRaises(AssertionError):original.verify_checkpoint(job,self.cfg,directory,config)
    def test_wrong_evidence_still_rejected(self):
        job=self.helper.review_job;output=self.helper.preserved_output(job)
        output['claims'][0]['support'][0]['evidence_span_ids']=['REV_999_SEG_999']
        with self.assertRaises(ValueError):mirai.validate_output(job,output,self.helper.schemas[job['task']])
    def test_trial_selection_excludes_used_empty_and_test_businesses(self):
        businesses=[{'business_id':b,'split':split} for b,split in [('used','train'),('empty','train'),('test','test'),('new1','train'),('new2','validation')]]
        jobs=[{'business_id':b['business_id'],'job_id':b['business_id']+'_'+condition,
               'skip_api_empty_captions':b['business_id']=='empty' and condition=='filtered'}
              for b in businesses for condition in ('shared','unfiltered','filtered')]
        with tempfile.TemporaryDirectory() as tmp,patch.object(trial.parent,'read_index',return_value={'used_shared':{}}),patch.object(trial.parent,'location',return_value=Path(tmp)):
            selected,selected_jobs=trial.select_businesses(jobs,businesses,{})
            self.assertEqual([b['business_id'] for b in selected],['new1','new2'])
            self.assertEqual(len(selected_jobs),6)


if __name__=='__main__':
    unittest.main(defaultTest='MiraiSwapTests')
