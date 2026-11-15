"""Cloud migration guards without model loads, downloads or modifying scientific outputs."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
import stage_f_cloud_d033 as cloud
import generate_stage_f_validation_captions as original
from verify_stage_f_cloud_package_d033 import safe_relative


class CloudTests(unittest.TestCase):
    def test_complete_partition_no_K_photo_preselection(self):
        allrows=cloud.legacy.read(cloud.OUT/'complete_photo_plan.json');reused=cloud.legacy.read(cloud.OUT/'reused_caption_sources.json')
        pending=cloud.legacy.read(cloud.GEN/'pilot_photo_plan.json')
        self.assertEqual(len(allrows),30027);self.assertEqual(len(reused),2709);self.assertEqual(len(pending),27318)
        self.assertFalse(set(reused)&{r['photo_id'] for r in pending})
        self.assertEqual(set(reused)|{r['photo_id'] for r in pending},{r['photo_id'] for r in allrows})
        self.assertTrue(all(r['split']=='train' for r in pending[:100]))
        from collections import Counter
        self.assertEqual(Counter(r['split'] for r in pending),{'train':26182,'test':1136})

    def test_generator_and_all_scientific_settings_unchanged(self):
        self.assertIs(cloud.generate_locked,original.generate_locked)
        cfg=cloud.configuration();self.assertEqual(cfg['generation']['batch_size'],1);self.assertEqual(cfg['generation']['num_beams'],3)
        for change in [('generation','num_beams',4),('device','quantization','int8'),('source_diagnostics','K',8)]:
            bad=copy.deepcopy(cfg);bad[change[0]][change[1]]=change[2]
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'bad.yaml';p.write_text(yaml.safe_dump(bad),encoding='utf-8')
                with patch.object(cloud,'CONFIG',p),self.assertRaises(AssertionError):cloud.configuration()

    def test_local_checkpoint_provenance_not_rewritten_to_cloud(self):
        parent=cloud.legacy.read(cloud.LOCAL/'reused_caption_sources.json');reused=cloud.legacy.read(cloud.OUT/'reused_caption_sources.json')
        for pid,item in parent.items():self.assertEqual(reused[pid],item)
        for pid,item in reused.items():
            p=cloud.ROOT/item['checkpoint_path'];self.assertEqual(cloud.legacy.sha(p),item['checkpoint_sha256'])
        self.assertNotEqual(cloud.GEN,cloud.LOCAL/'generation_train_test')

    def test_processing_uses_original_core_with_cloud_roots_only(self):
        import construct_stage_f_production_d033 as core
        with patch.object(core,'OUT',core.OUT),patch.object(core,'GEN',core.GEN),patch.object(cloud,'check_execution_seal'),patch.object(core,'score_production') as score:
            cloud.process_step('score');score.assert_called_once();self.assertEqual(core.OUT,cloud.OUT);self.assertEqual(core.GEN,cloud.GEN)

    def test_archive_paths_reject_traversal_and_absolute_windows_names(self):
        for name in ['../x','/root/x','C:/x','data\\photo.jpg']:
            with self.assertRaises(AssertionError):safe_relative(name)
        self.assertEqual(safe_relative('data/processed/photo.jpg'),'data/processed/photo.jpg')


if __name__=='__main__':unittest.main()
