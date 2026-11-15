"""D-031: pixel equality only, phash collisions retained, unchanged caption filters and pre-K report."""
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import yaml
from stage_f_d031_protocol import CONFIG,BASELINE,THRESHOLDS,assert_exact_authorized_delta,exact_duplicate_reduce
from apply_stage_f_frozen_filters import filter_business
from report_stage_f_pre_k_d031 import summarize


def row(pid,i,clip=.4,relevance=.2,basic=True):
    return {'photo_id':pid,'embedding_index':i,'clip_score':clip,'relevance_margin':relevance,
            'basic_eligible':basic,'score_status':'scored' if basic else 'basic_failure_not_scored'}


class D031Tests(unittest.TestCase):
    def test_only_explicit_authorized_delta(self):
        old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));new=yaml.safe_load((CONFIG.parents[1]/'outputs/stage_f/methodology_amendment_d032/previous_state/configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
        self.assertTrue(assert_exact_authorized_delta(old,new))
        for key,value in [('relevance',.09),('redundancy',.60),('grouping',.47),('near_image_duplicate',35),('CLIP',.22)]:
            bad=copy.deepcopy(new);bad['thresholds'][key]=value
            with self.assertRaises(AssertionError):assert_exact_authorized_delta(old,bad)
        bad=copy.deepcopy(new);bad['calibration_search']['image_hamming_grid'].append(35)
        with self.assertRaises(AssertionError):assert_exact_authorized_delta(old,bad)

    def test_exact_representative_uses_existing_quality_order(self):
        rows=[row('z',0,.1),row('b',1,.8),row('a',2,.8),row('unique',3,-.99)]
        kept,removed=exact_duplicate_reduce(rows,['same','same','same','different'])
        self.assertEqual(kept,[2,3]);self.assertEqual(removed,{1:2,0:2})

    def test_phash_collision_with_different_pixels_does_not_remove_caption(self):
        rows=[row('a',0,-.99),row('b',1,.9)]
        plan={'a':{'pixel_sha256':'one','phash64_hex':'0'},'b':{'pixel_sha256':'two','phash64_hex':'0'}}
        result=filter_business('TEST_ONLY',rows,np.eye(2,dtype=np.float32),plan,THRESHOLDS)
        self.assertEqual(result['after_exact_duplicate_count'],2)
        self.assertEqual(result['post_redundancy_count'],2)
        self.assertEqual(result['image_duplicate_removed_to_representative'],{})
        self.assertEqual(set(result['filtered_diversity_ranking']),{'a','b'})

    def test_unfiltered_does_not_receive_exact_relevance_or_redundancy_removals(self):
        rows=[row('a',0,.9),row('b',1,.8),row('c',2,.7,.01),row('bad',3,.6,basic=False)]
        plan={r['photo_id']:{'pixel_sha256':'same' if r['photo_id'] in ['a','b'] else r['photo_id'],'phash64_hex':'0'} for r in rows}
        result=filter_business('TEST_ONLY',rows,np.eye(4,dtype=np.float32),plan,THRESHOLDS)
        self.assertEqual(set(result['unfiltered_fixed_seed_ranking']),{'a','b','c'})
        self.assertEqual(result['basic_pool_count'],3);self.assertEqual(result['after_exact_duplicate_count'],2)
        self.assertEqual(result['after_relevance_count'],1);self.assertEqual(result['filtered_diversity_ranking'],['a'])

    def test_empty_business_is_retained_with_zero_counts(self):
        result=filter_business('TEST_EMPTY',[row('bad',0,basic=False)],np.eye(1,dtype=np.float32),{},THRESHOLDS)
        report=summarize([result])
        self.assertEqual(len(report['businesses']),1)
        self.assertEqual(report['summary']['coarser_semantic_groups']['zero_businesses'],1)
        self.assertEqual(report['summary']['after_relevance_filtering']['businesses_below_K']['3'],1)
        self.assertIsNone(report['K_selected'])

    def test_changed_phash_diagnostics_cannot_change_production_selection(self):
        rows=[row('a',0),row('b',1,.5)]
        a={'a':{'pixel_sha256':'one','phash64_hex':'0'},'b':{'pixel_sha256':'two','phash64_hex':'0'}}
        b=copy.deepcopy(a);b['b']['phash64_hex']='ffffffffffffffff'
        first=filter_business('TEST_ONLY',rows,np.eye(2,dtype=np.float32),a,THRESHOLDS)
        second=filter_business('TEST_ONLY',rows,np.eye(2,dtype=np.float32),b,THRESHOLDS)
        first.pop('diagnostic_phash64_by_basic_photo_id');second.pop('diagnostic_phash64_by_basic_photo_id')
        self.assertEqual(first,second)

    def test_reference_group_denominator_and_nested_diversity_unchanged(self):
        rows=[row('a',0,.5),row('b',1,.4),row('c',2,.3)]
        # a/b same coarser group but complementary below redundancy threshold.
        vectors=np.array([[1,0,0],[.5,np.sqrt(.75),0],[0,0,1]],dtype=np.float32)
        plan={r['photo_id']:{'pixel_sha256':r['photo_id'],'phash64_hex':'0'} for r in rows}
        result=filter_business('TEST_ONLY',rows,vectors,plan,THRESHOLDS)
        self.assertEqual(result['post_redundancy_count'],3);self.assertEqual(result['available_reference_groups'],2)
        self.assertEqual(result['filtered_diversity_ranking'],['a','c','b'])
        self.assertIsNone(result['K']);self.assertFalse(result['final_caption_set'])

    def test_legacy_calibration_routes_to_carry_without_threshold_search(self):
        import calibrate_stage_f_thresholds as legacy
        with patch('sys.argv',['calibrate','--labels','TEST_ONLY.json']), \
             patch('stage_f_d033_protocol.active_d033',return_value=False), \
             patch('stage_f_d032_protocol.active_d032',return_value=False), \
             patch('stage_f_d031_protocol.active_d031',return_value=True), \
             patch('stage_f_d031_protocol.carry_frozen_caption_thresholds') as carry, \
             patch.object(legacy,'threshold_search') as search,patch.object(legacy,'write') as mutation:
            legacy.main();carry.assert_called_once_with(Path('TEST_ONLY.json'))
            search.assert_not_called();mutation.assert_not_called()


if __name__=='__main__':unittest.main()
