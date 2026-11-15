"""D054 negative controls; CPU-only, no fabricated researcher annotations."""
import copy,json,unittest
from pathlib import Path
from stage_i_d053_evaluation import validate_predictions,structural_diagnostics,token_termination,human_metrics,paired_bootstrap
from stage_i_d054_text import unique_segments
from verify_stage_i_d054_bundle import same_json_data

ROOT=Path(__file__).resolve().parents[1]
class Controls(unittest.TestCase):
    def test_metric_span_JSON_roundtrip(self):
        a={'spans':[('[REV_1]',0,7)],'ratio':0.123456789}
        self.assertTrue(same_json_data(a,json.loads(json.dumps(a))))
    def test_metric_float_difference_not_relaxed(self):
        self.assertFalse(same_json_data({'metric':0.1},{'metric':0.1000000001}))
    def test_metric_bool_not_coerced_to_number(self):
        self.assertFalse(same_json_data({'metric':True},{'metric':1}))
    def test_initial_decoder_start_is_not_eos(self):
        c={'max_length':768,'eos_token_id':2,'decoder_start_token_id':2,'bos_token_id':0,'pad_token_id':1,'forced_eos_token_id':2}
        self.assertFalse(token_termination([2,0,12],c)['eos_after_decoder_start'])
    def test_forced_cap_is_not_early_eos(self):
        c={'max_length':768,'eos_token_id':2,'decoder_start_token_id':2,'bos_token_id':0,'pad_token_id':1,'forced_eos_token_id':2}
        d=token_termination([2]+[15]*766+[2],c)
        self.assertTrue(d['eos_at_cap_may_be_forced']);self.assertFalse(d['eos_before_length_cap'])
    def test_missing_headers_not_silently_repaired(self):
        self.assertFalse(structural_diagnostics('Food is good.',4)['section_headers_compliant'])
    def test_zero_caption_requires_empty_visual(self):
        t='[REVIEW_CONSENSUS]\nCustomers liked food.\n\n[VISUAL_EVIDENCE]\n'
        self.assertTrue(structural_diagnostics(t,0)['empty_visual_structurally_compliant'])
        self.assertFalse(structural_diagnostics(t+'A plate.',0)['empty_visual_structurally_compliant'])
    def test_source_markers_are_not_target_headers(self):
        self.assertFalse(structural_diagnostics('[REV_1] Food. [IMG_CAP_1] Plate.',2)['section_headers_compliant'])
    def test_segment_offsets_preserve_original(self):
        t='Food is good. Food is good.\nBad service! tail'
        for s in unique_segments(t):
            for p in s['occurrences']:self.assertEqual(t[p['start']:p['end']],s['text'])
        self.assertEqual(len(unique_segments(t)[0]['occurrences']),2)
    def test_no_human_truth_from_existing_references(self):
        self.assertIsNone(human_metrics(None,{}, {},[{'human_verified':True}])['metrics'])
    def test_bootstrap_preserves_pairing(self):
        r=paired_bootstrap([{'business_id':str(i),'A':i*10,'B':i*10+2} for i in range(20)])
        self.assertEqual(r['mean_paired_difference'],2)
    def test_duplicate_business_pairs_rejected(self):
        with self.assertRaises(ValueError):paired_bootstrap([{'business_id':'a','A':1,'B':2}]*2)
    def test_incomplete_predictions_rejected(self):
        with self.assertRaises(ValueError):validate_predictions([],[],{})

if __name__=='__main__':unittest.main()
