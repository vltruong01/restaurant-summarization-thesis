"""CPU-only regression checks for D055 forensic definitions; never loads weights."""
import json, pathlib, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_stage_i_d055_offline as a

class TinyTokenizer:
    def __call__(self,text,add_special_tokens=True):return {'input_ids':[0,10,11,12,2]}
    def convert_ids_to_tokens(self,x):return str(x)

class ForensicTests(unittest.TestCase):
    def test_exact_header_prefix(self):
        x=a.prefix_diagnostic([2,0,10,11,12,50,2],[0,10,11,12,51,2],TinyTokenizer())
        self.assertTrue(x['required_header_prefix_complete']);self.assertEqual(x['first_teacher_token_mismatch'],5)
        self.assertIsNone(x['first_header_mismatch'])
    def test_wrong_header(self):
        x=a.prefix_diagnostic([2,0,10,99,2],[0,10,11,12,51,2],TinyTokenizer())
        self.assertFalse(x['required_header_prefix_complete']);self.assertEqual(x['first_header_mismatch'],3)
    def test_wrong_bos(self):
        self.assertEqual(a.prefix_diagnostic([2,10,11],[0,10,11,12,2],TinyTokenizer())['mismatch_stage'],'BOS')
    def test_config_semantic_fields_not_ignored(self):
        self.assertNotEqual(a.compatible_config({'max_length':768}),a.compatible_config({'max_length':769}))
    def test_only_provenance_keys_ignored(self):
        self.assertEqual(a.compatible_config({'x':1,'transformers_version':'1'}),{'x':1})
    def test_cpu_shift_mask_real_cases(self):
        cases=a.rows(a.PREP/'evaluation_inputs.jsonl')
        x=a.mask_tests(cases);self.assertEqual(x['status'],'PASS_CPU_TENSORS_ONLY');self.assertGreater(x['padded_label_count'],0)
    def test_overflow_rejected(self):
        with self.assertRaises(ValueError):a.LEDCompleteCollator(source_limit=2)([{'input_ids':[0,3,2],'labels':[0,2]}])
    def test_missing_eos_rejected(self):
        with self.assertRaises(ValueError):a.LEDCompleteCollator()([{'input_ids':[0,3],'labels':[0,2]}])
    def test_zero_caption_claim_is_not_empty(self):
        x=a.structural_diagnostics('[REVIEW_CONSENSUS]\nCustomers praise food.\n\n[VISUAL_EVIDENCE]\nThere are bagels.',0)
        self.assertTrue(x['section_headers_compliant']);self.assertFalse(x['empty_visual_structurally_compliant'])
    def test_source_marker_confusion_fails(self):
        x=a.structural_diagnostics('[VISUAL_CONTEXT]\nFood.\n\n[VISUAL_EVIDENCE]\n',1)
        self.assertFalse(x['section_headers_compliant'])
    def test_start_eos_not_early_stop(self):
        x=a.token_termination([2,0,10,2],{'max_length':768,'eos_token_id':2,'decoder_start_token_id':2,'bos_token_id':0,'pad_token_id':1,'forced_eos_token_id':2})
        self.assertEqual(x['first_eos_position'],3)
    def test_marker_only_not_quality_pass(self):
        self.assertFalse(a.structural_diagnostics('[REVIEW_CONSENSUS]\n\n[VISUAL_EVIDENCE]\n',0)['section_headers_compliant'])
    def test_known_d054_metrics_reproduced(self):
        p=a.ROOT/'outputs/stage_i/d055_root_cause_v2/HISTORICAL_VS_PILOT_SUMMARY.json'
        self.assertTrue(p.is_file(),'Run the offline audit first')
        d=a.read(p)
        for arm,n in [('A_decoder_only',6),('B_encoder_decoder',12)]:
            self.assertEqual(d[arm]['random']['sentence_repetition'],n);self.assertEqual(d[arm]['random']['structure_compliant'],0)
        self.assertEqual(d['historical_full_F']['all24']['exact_one_each_marker'],21)
    def test_no_gpu_or_models(self):
        self.assertFalse(a.torch.cuda.is_initialized())
        r=a.read(a.ROOT/'outputs/stage_i/d055_root_cause_v2/EXECUTION_RECEIPT.json')
        self.assertTrue(r['all_unchanged']);self.assertEqual(r['model_forward_calls'],0);self.assertEqual(r['generation_calls'],0)

if __name__=='__main__':unittest.main(verbosity=2)
