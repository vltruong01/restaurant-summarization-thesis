"""Meaningful corruption/overflow/attention/approval tests. CPU only; no model training."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage_hi_led_data import (ROOT, read, sha, PackedLEDDataset, LEDCompleteCollator,
                               execution_gate, generation_format, settings_signature)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read(ROOT/'configs/stage_i_lora_led_v1.json')

    def test_all_gpu_scopes_are_denied_before_execution(self):
        for mode in ['feasibility','train']:
            with self.assertRaisesRegex(ValueError,'not approved'):
                execution_gate(self.cfg,mode)
        cfg=copy.deepcopy(self.cfg)
        cfg['approval']['length_and_training_settings']=True
        with self.assertRaisesRegex(ValueError,'not authorized'): execution_gate(cfg,'feasibility')

    def test_authorization_cannot_silently_change_scientific_signature(self):
        cfg=copy.deepcopy(self.cfg); cfg['approval']['full_training']=True
        self.assertEqual(settings_signature(cfg),settings_signature(self.cfg))
        cfg['generation']['num_beams']=1
        self.assertNotEqual(settings_signature(cfg),settings_signature(self.cfg))

    def test_boundary_padding_preserves_every_content_token(self):
        f={'input_ids':[0]+[17]*4998+[2],'labels':[0]+[19]*766+[2]}
        short={'input_ids':[0,21,2],'labels':[0,23,2]}
        batch=LEDCompleteCollator()( [f,short] )
        self.assertEqual(tuple(batch['input_ids'].shape),(2,5120))
        self.assertEqual(tuple(batch['labels'].shape),(2,768))
        self.assertEqual(batch['input_ids'][0,:5000].tolist(),f['input_ids'])
        self.assertEqual(batch['labels'][0].tolist(),f['labels'])
        self.assertTrue((batch['labels'][1,3:]==-100).all())
        self.assertEqual(batch['global_attention_mask'].sum(1).tolist(),[1,1])
        self.assertFalse(batch['global_attention_mask'][:,1:].any())
        self.assertFalse(batch['attention_mask'][0,5000:].any())
        self.assertFalse(batch['attention_mask'][1,3:].any())

    def test_source_and_label_overflows_fail_instead_of_truncating(self):
        c=LEDCompleteCollator()
        for f in [{'input_ids':[0]+[7]*4999+[2],'labels':[0,2]},
                  {'input_ids':[0,2],'labels':[0]+[7]*767+[2]}]:
            before=copy.deepcopy(f)
            with self.assertRaisesRegex(ValueError,'overflow'): c([f])
            self.assertEqual(f,before)

    def test_missing_eos_is_not_a_valid_complete_label(self):
        with self.assertRaisesRegex(ValueError,'BOS/EOS'):
            LEDCompleteCollator()([{'input_ids':[0,4,2],'labels':[0,4]}])

    def test_packed_tampering_and_invalid_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'fixture.npz'
            arrays=dict(input_ids=np.array([0,4,2],dtype=np.int32),labels=np.array([0,5,2],dtype=np.int32),
                        input_offsets=np.array([0,3]),label_offsets=np.array([0,3]),business_ids=np.array(['fixture']))
            np.savez_compressed(p,**arrays)
            info=dict(path='fixture.npz',sha256=sha(p),rows=1)
            self.assertEqual(PackedLEDDataset(info,5000,768,root)[0]['labels'],[0,5,2])
            arrays['label_offsets']=np.array([0,2]); np.savez_compressed(p,**arrays)
            with self.assertRaisesRegex(ValueError,'hash'): PackedLEDDataset(info,5000,768,root)
            info['sha256']=sha(p)
            with self.assertRaisesRegex(ValueError,'offsets'): PackedLEDDataset(info,5000,768,root)

    def test_generation_sections_and_zero_caption_representation(self):
        ro='[REVIEW_CONSENSUS]\nMixed food opinions.'
        self.assertTrue(generation_format(ro,'reviews_only'))
        self.assertFalse(generation_format(ro,'unfiltered'))
        empty=ro+'\n\n[VISUAL_EVIDENCE]\n'
        self.assertFalse(generation_format(empty,'filtered'))
        self.assertTrue(generation_format(empty,'filtered',allow_empty_visual=True))
        self.assertFalse(generation_format('[REVIEW_CONSENSUS]\n\n\n[VISUAL_EVIDENCE]\nFood.','unfiltered'))
        self.assertTrue(generation_format(empty+'A plate is visible.','unfiltered'))


if __name__=='__main__': unittest.main(verbosity=2)
