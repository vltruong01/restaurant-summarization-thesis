"""D056 offline preparation regression/negative controls; no model execution."""
import ast,copy,json,pathlib,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d056 as d
from tokenizers import Tokenizer

class D056Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=d.read(d.CONFIG);cls.cases=d.read(d.OUT/'cases.json')
        cls.tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    def test_exact_case_ids_order(self):self.assertEqual([c['business_id'] for c in self.cases],d.CASES)
    def test_four_model_states(self):self.assertEqual(self.cfg['states'],d.STATES)
    def test_all_token_alignment(self):
        for c in self.cases:
            a=d.alignment(c,self.tok);self.assertEqual(a['rescue_prefix_ids'],[2,0,10975,4629,21735,1215,15299,104,12743,3048,742,50118])
            self.assertEqual(a['teacher_decoder_input_ids'],[2]+c['label_ids'][:-1])
    def test_cache_prefix11_predicts_newline(self):
        for c in self.cases:
            p=d.alignment(c,self.tok)['cache_probes'][0];self.assertEqual(p['expected_label_index'],10);self.assertEqual(p['expected_id'],50118)
    def test_cache_prefix12_predicts_first_body(self):
        for c in self.cases:
            p=d.alignment(c,self.tok)['cache_probes'][1];self.assertEqual(p['reference_logit_row'],11);self.assertEqual(p['expected_id'],c['label_ids'][11])
    def test_initial_decoder_eos_not_label_eos(self):
        a=d.alignment(self.cases[0],self.tok);self.assertEqual(a['teacher_decoder_input_ids'][0],2);self.assertEqual(a['token_positions'][0]['expected_id'],0)
    def test_visual_boundary_present_all_cases(self):
        for c in self.cases:
            a=d.alignment(c,self.tok);self.assertIn(a['visual_marker_label_start'],a['critical_logit_rows']);self.assertIn(len(c['label_ids'])-1,a['critical_logit_rows'])
    def test_zero_caption_target_retained(self):
        a=d.alignment(self.cases[2],self.tok);self.assertTrue(a['zero_caption']);self.assertEqual(self.cases[2]['target'].split(d.VISUAL)[-1].strip(),'')
    def test_false_zero_caption_target_rejected(self):
        c=copy.deepcopy(self.cases[2]);c['target']+='fabricated visual';self.assertRaises(ValueError,d.alignment,c,self.tok)
    def test_bad_bos_rejected(self):
        c=copy.deepcopy(self.cases[0]);c['label_ids'][0]=7;self.assertRaises(ValueError,d.alignment,c,self.tok)
    def test_duplicate_visual_marker_rejected(self):
        c=copy.deepcopy(self.cases[0]);c['label_ids'][-1:-1]=self.tok.encode(d.VISUAL,add_special_tokens=False).ids;self.assertRaises(ValueError,d.alignment,c,self.tok)
    def test_budget_teacher_ceiling(self):
        b=d.Budget(self.cfg['ceilings']);b.take('teacher_forced',24);self.assertRaises(ValueError,b.take,'teacher_forced');self.assertEqual(b.used['teacher_forced'],24)
    def test_budget_cache_ceiling(self):
        b=d.Budget(self.cfg['ceilings']);b.take('cache_control',4*3*2*2*3);self.assertRaises(ValueError,b.take,'cache_control')
    def test_budget_traces_ceiling(self):
        b=d.Budget(self.cfg['ceilings']);b.take('traces',36);self.assertRaises(ValueError,b.take,'traces')
    def test_budget_no_negative_increment(self):self.assertRaises(ValueError,d.Budget(self.cfg['ceilings']).take,'traces',-1)
    def test_configuration_frozen_screens(self):self.assertTrue(d.check_config(self.cfg))
    def test_tolerance_mutation_rejected(self):
        c=copy.deepcopy(self.cfg);c['cache_screens']['FP32']['atol']=1;self.assertRaises(ValueError,d.check_config,c)
    def test_length_mutation_rejected(self):
        c=copy.deepcopy(self.cfg);c['generation']['max_length']=769;self.assertRaises(ValueError,d.check_config,c)
    def test_vector_identity(self):
        x=d.vector_comparison([0.,1.,2.],[0.,1.,2.],2,1e-4,1e-4);self.assertTrue(x['allclose_screen_pass']);self.assertEqual(x['KL_reference_actual'],0)
    def test_vector_precision_screens_separate(self):
        self.assertFalse(d.vector_comparison([0.,1.,2.],[.02,1.,2.],2,1e-4,1e-4)['allclose_screen_pass'])
        self.assertTrue(d.vector_comparison([0.,1.,2.],[.02,1.,2.],2,.05,.005)['allclose_screen_pass'])
    def test_relative_floor_finite(self):self.assertTrue(np.isfinite(d.vector_comparison([0.,1.],[.001,1.],1,1e-4,1e-4)['max_relative_with_floor1e8']))
    def test_nonfinite_rejected(self):self.assertRaises(ValueError,d.vector_comparison,[0.,1.],[float('nan'),1.],1,1e-4,1e-4)
    def test_vector_shapes_rejected(self):self.assertRaises(ValueError,d.vector_comparison,[0.,1.],[0.],1,1e-4,1e-4)
    def test_approval_template_denies_execution(self):self.assertRaises(ValueError,d.approval_gate,d.OUT/'GPU_AUTHORIZATION_TEMPLATE.json',d.CONFIG,d.OUT/'PREFLIGHT_MANIFEST.json')
    def test_foreign_approval_denied(self):
        a=d.read(d.OUT/'GPU_AUTHORIZATION_TEMPLATE.json');a['GPU_execution_approved']=True;a['researcher_explicit_authorization']='synthetic test only';a['approval_source']='researcher_message';a['config_sha256']='bad'
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp)/'test.json';p.write_text(json.dumps(a),encoding='utf-8');self.assertRaises(ValueError,d.approval_gate,p,d.CONFIG,d.OUT/'PREFLIGHT_MANIFEST.json')
    def test_cli_no_execute_cannot_load_torch(self):
        code="import sys,builtins,runpy;sys.path.insert(0,'scripts');old=builtins.__import__;"+"\ndef guard(name,*a,**k):\n if name.split('.')[0] in ['torch','transformers','peft']:raise RuntimeError('FORBIDDEN_MODEL_IMPORT')\n return old(name,*a,**k)\nbuiltins.__import__=guard\nsys.argv=['run_stage_i_d056_gpu_diagnostic.py'];runpy.run_path('scripts/run_stage_i_d056_gpu_diagnostic.py',run_name='__main__')"
        p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('separate explicit GPU approval',p.stderr);self.assertNotIn('FORBIDDEN_MODEL_IMPORT',p.stderr)
    def test_no_training_operations_in_prepared_runner(self):
        tree=ast.parse((ROOT/'scripts/run_stage_i_d056_gpu_diagnostic.py').read_text(encoding='utf-8'))
        forbidden={'backward','train','step','save_pretrained','save_model','zero_grad'}
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden])
    def test_scope_manifest_preservation(self):self.assertEqual(d.verify_manifest(d.OUT/'PREFLIGHT_MANIFEST.json')['model_forward_calls'],0)
    def test_d051_still_fail(self):self.assertEqual(self.cfg['D051_strict_replay_status'],'FAIL_UNCHANGED')
    def test_no_gpu_output_namespace(self):self.assertFalse((ROOT/self.cfg['future_output']).exists())
    def test_trace_initial_eos_ignored(self):self.assertIsNone(d.trace_observation([2],[2,0,10],False)['selected_EOS_new_token_index'])
    def test_trace_actual_eos(self):self.assertEqual(d.trace_observation([2],[2,0,10,2],False)['selected_EOS_new_token_index'],2)
    def test_trace_censor_without_fabricated_eos(self):
        t=d.trace_observation([2],[2]+[10]*64,True);self.assertIsNone(t['selected_EOS_new_token_index']);self.assertTrue(t['run_reached_diagnostic_censor'])
    def test_trace_selected_earlier_eos_and_run_censor_independent(self):
        t=d.trace_observation([2],[2,0,2],True);self.assertEqual(t['selected_hypothesis_termination'],'model_EOS');self.assertTrue(t['run_reached_diagnostic_censor'])
    def test_trace_65_tokens_rejected(self):self.assertRaises(ValueError,d.trace_observation,[2],[2]+[10]*65,True)
    def test_trace_wrong_prefix_rejected(self):self.assertRaises(ValueError,d.trace_observation,[2,0,10],[2,0,11],False)
    def test_expected_adapter_key_counts(self):
        self.assertEqual(len(d.expected_adapter_keys('A_decoder_only')),48);self.assertEqual(len(d.expected_adapter_keys('B_encoder_decoder')),96)
    def test_actual_checkpoint_bindings(self):
        self.assertTrue(all(x['best_matches_selected_checkpoint'] for x in d.adapter_checkpoint_bindings(self.cfg).values()))
if __name__=='__main__':unittest.main(verbosity=2)
