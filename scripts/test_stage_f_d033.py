"""Final freeze and complete-pool guards; no synthetic fixtures become output data."""
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from stage_f_d033_protocol import validate_compatibility,ROOT
from stage_f_verification_workflow import read,sha


class D033Tests(unittest.TestCase):
    def test_authorized_delta_is_only_K_and_production_input_scope(self):
        r=validate_compatibility()
        self.assertEqual(r['K'],5);self.assertTrue(r['K_frozen'])
        self.assertEqual(r['caption_thresholds'],{'relevance':.08,'redundancy':.59,'grouping':.46})
        self.assertFalse(r['test_tuning']);self.assertFalse(r['test_teacher_targets'])

    def test_unchanged_D026_recipe_and_all_photos_not_K_preselection(self):
        a=yaml.safe_load((ROOT/'configs/stage_f_production_d033.yaml').read_text(encoding='utf-8'))
        b=yaml.safe_load((ROOT/'configs/stage_f_validation_caption_pool.yaml').read_text(encoding='utf-8'))
        for key in ['seed','model_id','requested_revision','generation','device','basic_quality_policy']:
            self.assertEqual(a[key],b[key])
        folder=ROOT/'outputs/stage_f/production_caption_construction_d033_v1'
        plan=read(folder/'complete_photo_plan.json')
        reused=read(folder/'reused_caption_sources.json')
        pending=read(folder/'generation_train_test/pilot_photo_plan.json')
        self.assertEqual(len(plan),30027);self.assertEqual(len(pending),27511)
        self.assertEqual({r['photo_id'] for r in plan},set(reused)|{r['photo_id'] for r in pending})
        self.assertFalse(set(reused)&{r['photo_id'] for r in pending})
        for pid,item in reused.items():
            record=read(ROOT/item['checkpoint_path'])
            self.assertEqual(record['photo_id'],pid);self.assertEqual(sha(ROOT/item['checkpoint_path']),item['checkpoint_sha256'])
            self.assertEqual(record['model_id'],b['model_id'])

    def test_legacy_threshold_CLI_cannot_refit_after_D033(self):
        import calibrate_stage_f_thresholds as legacy
        with patch('sys.argv',['calibrate','--labels','UNREAD_TEST_ONLY.json']),patch('stage_f_d033_protocol.active_d033',return_value=True), \
             patch('stage_f_d033_protocol.freeze_method') as freeze,patch.object(legacy,'threshold_search') as fit:
            legacy.main();freeze.assert_called_once();fit.assert_not_called()


if __name__=='__main__':unittest.main()
