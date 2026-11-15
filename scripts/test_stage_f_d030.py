"""D-030 behavioral guards: soft CLIP, untouched definitions/baseline, pre-K fail closed."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from stage_f_d030_protocol import (CONFIG,BASELINE,assert_exact_authorized_delta,
                                  relevance_eligible,unfiltered_fixed_seed_order,assert_complete_hard_thresholds)
from stage_f_math import quality_order,image_duplicate_reduce,threshold_search


class D030Tests(unittest.TestCase):
    def test_only_explicit_authorized_config_delta(self):
        old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));new=yaml.safe_load((CONFIG.parents[1]/'outputs/stage_f/methodology_amendment_d031/previous_state/configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
        self.assertTrue(assert_exact_authorized_delta(old,new))
        mutations=[lambda c:c['models']['CLIP'].update(revision='changed'),
                   lambda c:c['quality_order'].reverse(),
                   lambda c:c['calibration_search']['image_hamming_grid'].append(16),
                   lambda c:c['relevance']['positive_prototypes'].append('new prototype'),
                   lambda c:c['K_candidates'].append(15),
                   lambda c:c['thresholds'].update(CLIP=.22)]
        for mutate in mutations:
            edited=copy.deepcopy(new);mutate(edited)
            with self.assertRaises(AssertionError):assert_exact_authorized_delta(old,edited)

    def test_low_CLIP_never_hard_rejects_relevant_caption(self):
        rows=[{'photo_id':'low','clip_score':-.99,'relevance_margin':.08},
              {'photo_id':'high','clip_score':.99,'relevance_margin':.08}]
        self.assertEqual(relevance_eligible(rows,[0,1],.08),[0,1])
        self.assertEqual(quality_order(rows),[1,0])

    def test_high_CLIP_does_not_override_relevance_gate(self):
        rows=[{'photo_id':'a','clip_score':.99,'relevance_margin':.0799994},
              {'photo_id':'b','clip_score':-.99,'relevance_margin':.0799996}]
        self.assertEqual(relevance_eligible(rows,[0,1],.08),[1])

    def test_duplicate_representative_uses_existing_quality_only(self):
        rows=[{'photo_id':'a','clip_score':-.99,'relevance_margin':.3},
              {'photo_id':'b','clip_score':.4,'relevance_margin':.2},
              {'photo_id':'unique_low','clip_score':-.999,'relevance_margin':.4}]
        kept,removed=image_duplicate_reduce(rows,['same','same','unique'],[0,0,2**64-1],0)
        self.assertEqual(kept,[1,2]);self.assertEqual(removed,{0:1})
        self.assertIn(2,relevance_eligible(rows,kept,.08))

    def test_unfiltered_order_ignores_quality_scores_and_substantive_filters(self):
        rows=[{'photo_id':str(i),'clip_score':-.9+i*.2,'relevance_margin':-.1} for i in range(6)]
        before=[r['photo_id'] for r in unfiltered_fixed_seed_order(rows)]
        edited=copy.deepcopy(rows)
        for r in edited:r['clip_score']=.99;r['relevance_margin']=1.2
        self.assertEqual(before,[r['photo_id'] for r in unfiltered_fixed_seed_order(edited)])
        self.assertEqual(set(before),{str(i) for i in range(6)})

    def test_CLIP_threshold_and_unresolved_image_criterion_cannot_look_complete(self):
        cfg_hash=__import__('stage_f_verification_workflow').sha(CONFIG)
        good={'CLIP_mode':'soft_only','status':'D030_REMAINING_HARD_THRESHOLDS_FROZEN','protocol_sha256':cfg_hash,
              'thresholds':{'relevance':.08,'redundancy':.59,'grouping':.46,'near_image_duplicate':12}}
        assert_complete_hard_thresholds(good)
        missing=copy.deepcopy(good);missing['thresholds']['near_image_duplicate']=None
        with self.assertRaises(AssertionError):assert_complete_hard_thresholds(missing)
        wrong=copy.deepcopy(good);wrong['thresholds']['CLIP']=.22
        with self.assertRaises(AssertionError):assert_complete_hard_thresholds(wrong)

    def test_original_pHash_grid_cannot_be_frozen_as_better_than_chance(self):
        best,_=threshold_search([22]*16+[20]*124,[True]*16+[False]*124,[0,4,8,12],False,10)
        self.assertEqual(best['balanced_accuracy'],.5)
        self.assertFalse(best['balanced_accuracy']>.5)

    def test_legacy_CLI_routes_D030_without_consistency_fit_or_label_import(self):
        import calibrate_stage_f_thresholds as legacy
        with patch('sys.argv',['calibrate','--labels','TEST_ONLY.json']), \
             patch('stage_f_d033_protocol.active_d033',return_value=False), \
             patch('stage_f_d032_protocol.active_d032',return_value=False), \
             patch('stage_f_d031_protocol.active_d031',return_value=False), \
             patch('stage_f_d030_protocol.active_d030',return_value=True), \
             patch('calibrate_stage_f_d030_remaining.run') as remaining, \
             patch.object(legacy,'write') as mutation,patch.object(legacy,'threshold_search') as old_search:
            legacy.main();remaining.assert_called_once_with(Path('TEST_ONLY.json'))
            mutation.assert_not_called();old_search.assert_not_called()

    def test_filtering_stops_before_loading_K_scores_when_thresholds_missing(self):
        import apply_stage_f_frozen_filters as filters
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(filters,'OUT',Path(folder)), \
             patch.object(filters,'require_validated_amendment'),patch.object(filters,'load_file') as embeddings:
            with self.assertRaisesRegex(RuntimeError,'required hard threshold unresolved'):filters.main()
            embeddings.assert_not_called()


if __name__=='__main__':unittest.main()
