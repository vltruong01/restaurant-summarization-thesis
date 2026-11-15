"""D056 offline preparation regression/negative controls; no model execution."""
import ast,copy,json,pathlib,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d056_v2 as d
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
            p=d.alignment(c,self.tok)['cache_probes'][1];self.assertEqual(p['expected_label_index'],10);self.assertEqual(p['expected_id'],50118)
    def test_cache_prefix3_predicts_RE(self):
        for c in self.cases:
            p=d.alignment(c,self.tok)['cache_probes'][0];self.assertEqual(p['reference_logit_row'],2);self.assertEqual(p['expected_id'],4629)
    def test_teacher_forced_retains_p12_first_body(self):
        for c in self.cases:
            a=d.alignment(c,self.tok);self.assertIn(11,a['critical_logit_rows']);self.assertEqual(a['token_positions'][11]['expected_id'],c['label_ids'][11])
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
        code="import sys,builtins,runpy;sys.path.insert(0,'scripts');old=builtins.__import__;"+"\ndef guard(name,*a,**k):\n if name.split('.')[0] in ['torch','transformers','peft']:raise RuntimeError('FORBIDDEN_MODEL_IMPORT')\n return old(name,*a,**k)\nbuiltins.__import__=guard\nsys.argv=['run_stage_i_d056_gpu_diagnostic_v2.py'];runpy.run_path('scripts/run_stage_i_d056_gpu_diagnostic_v2.py',run_name='__main__')"
        p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('separate explicit GPU approval',p.stderr);self.assertNotIn('FORBIDDEN_MODEL_IMPORT',p.stderr)
    def test_no_training_operations_in_prepared_runner(self):
        tree=ast.parse((ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py').read_text(encoding='utf-8'))
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
class D056AmendmentControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=d.read(d.CONFIG);cls.cases=d.read(d.OUT/'cases.json')
    def test_previous_probe_pair_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['cache_prefix_lengths']=[11,12];self.assertRaises(ValueError,d.check_config,cfg)
    def test_cross_precision_screen_drift_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['cross_precision_screen']['max_abs_gt']=.2;self.assertRaises(ValueError,d.check_config,cfg)
    def test_unrelated_setting_drift_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['seed']=43;self.assertRaises(ValueError,d.check_config,cfg)
    def test_v1_files_unchanged(self):
        for path,value in d.read(d.OUT/'PARENT_V1_PRESERVATION_MANIFEST.json').items():self.assertEqual(d.sha(ROOT/path),value,path)
    def test_masks_real_collator_body(self):
        report=d.offline_mask_bindings(self.cases);self.assertFalse(report['actual_torch_imported']);self.assertEqual([x['arrays']['input_ids']['shape'] for x in report['cases']],[[1,4096]]*3)
    def test_source_text_mutation_rejected(self):
        cases=copy.deepcopy(self.cases);cases[0]['source']+='x';self.assertRaises(ValueError,d.offline_mask_bindings,cases)
    def test_incomplete_target_rejected_by_real_collator(self):
        cases=copy.deepcopy(self.cases);cases[0]['label_ids'][-1]=3;self.assertRaises(ValueError,d.offline_mask_bindings,cases)
    def test_mask_shim_cleaned_after_error(self):
        cases=copy.deepcopy(self.cases);cases[0]['input_ids']=[];self.assertRaises(ValueError,d.offline_mask_bindings,cases);self.assertNotIn('torch',sys.modules)
    def test_historical_fingerprint_convention(self):
        from validate_stage_i_d056_v2 import historical_fingerprint_function,Parameter,Tensor
        old,_=historical_fingerprint_function();a=np.arange(12,dtype=np.float32).reshape(3,4).T
        self.assertEqual(old(Parameter(a)),d.tensor_fingerprint(Parameter(a)));self.assertNotEqual(d.tensor_fingerprint(Parameter(a)),d.tensor_fingerprint(Tensor(a)))
    def test_actual_saved_adapter_payloads(self):
        from validate_stage_i_d056_v2 import fingerprint_audit
        report=fingerprint_audit(self.cfg);self.assertEqual(sum(x['actual_saved_tensors_checked'] for x in report['real_adapter_payload_checks'].values()),144)
    def test_actual_runtime_static_bindings(self):
        from validate_stage_i_d056_v2 import validate_prepared_runtime
        report=validate_prepared_runtime(self.cfg,self.cases);self.assertFalse(report['model_loaded'])
    def test_wrong_runtime_version_rejected(self):
        from unittest.mock import patch
        from validate_stage_i_d056_v2 import validate_prepared_runtime
        with patch('validate_stage_i_d056_v2.importlib.metadata.version',return_value='wrong'):
            self.assertRaises(ValueError,validate_prepared_runtime,self.cfg,self.cases)
    def test_scalar_fingerprint_preserves_shape(self):
        import hashlib
        from validate_stage_i_d056_v2 import Parameter
        a=np.array(0,dtype=np.float32)
        expected=hashlib.sha256(b'Parameter:'+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()+b';').hexdigest()
        self.assertEqual(d.tensor_fingerprint(Parameter(a)),expected)
    def test_fingerprint_dtype_and_shape_are_bound(self):
        from validate_stage_i_d056_v2 import Parameter
        a=np.arange(12,dtype=np.float32)
        self.assertNotEqual(d.tensor_fingerprint(Parameter(a)),d.tensor_fingerprint(Parameter(a.astype(np.float16))))
        self.assertNotEqual(d.tensor_fingerprint(Parameter(a)),d.tensor_fingerprint(Parameter(a.reshape(3,4))))
    def test_independent_processed_ranking(self):
        raw=np.arange(20,dtype=np.float32)[None,:];processed=-raw
        x=d.separate_candidates(raw,processed);self.assertEqual(x['raw_top10_ids'][0][0],19);self.assertEqual(x['processed_top10_ids'][0][0],0)
    def test_candidate_ties_ascending_id(self):
        x=d.separate_candidates(np.zeros((1,20)),np.zeros((1,20)));self.assertEqual(x['processed_top10_ids'][0],list(range(10)))
    def test_candidate_float_values_preserved(self):
        a=np.arange(20,dtype=np.float32)[None,:]/7;x=d.separate_candidates(a,a)
        for token,value in zip(x['raw_top10_ids'][0],x['raw_top10_logits'][0]):self.assertEqual(np.float32(value),a[0,token])
    def test_masked_processed_tokens_json_safe(self):
        a=np.arange(20,dtype=np.float32)[None,:];p=a.copy();p[0,19]=-np.inf
        x=d.separate_candidates(a,p);self.assertEqual(x['processed_scores_at_raw_top10'][0][0],'-inf');json.dumps(x,allow_nan=False)
    def test_all_processed_masked_rejected(self):self.assertRaises(ValueError,d.separate_candidates,np.zeros((1,20)),np.full((1,20),-np.inf))
    def test_nan_processed_rejected(self):self.assertRaises(ValueError,d.separate_candidates,np.zeros((1,20)),np.full((1,20),np.nan))
    def test_inf_raw_rejected(self):self.assertRaises(ValueError,d.separate_candidates,np.full((1,20),np.inf),np.zeros((1,20)))
    def test_candidate_shape_mismatch_rejected(self):self.assertRaises(ValueError,d.separate_candidates,np.zeros((1,20)),np.zeros((2,20)))
    def beam(self,**changes):
        kw=dict(prompt=[2],sequence=[2,0,42,2],ancestry=[0,1,2],transitions=[-.3,-.6,-.9],direct_scores=[-.3,-.6,-.9],raw_selected_logits=[2.,3.,4.],returned_score=-.6,length_penalty=1.,score_steps=64,censor_seen=True,num_beams=4)
        kw.update(changes);return d.beam_trace_audit(**kw)
    def test_beam_prior_eos_and_later_censor(self):
        x=self.beam();self.assertTrue(x['selected_earlier_than_last_global_iteration']);self.assertEqual(x['selected_finalization_reason'],'observed_selected_EOS');self.assertFalse(x['finalizer_appended_EOS'])
    def test_beam_censor_no_eos(self):
        x=self.beam(sequence=[2]+[42]*64,ancestry=[0]*64,transitions=[-.3]*64,direct_scores=[-.3]*64,raw_selected_logits=[2.]*64,returned_score=-.3)
        self.assertEqual(x['selected_finalization_reason'],'custom_64_token_censor');self.assertIsNone(x['selected_EOS_new_token_index'])
    def test_beam_reconstruction_uses_generated_length(self):self.assertAlmostEqual(self.beam()['reconstructed_sequence_score'],-.6)
    def test_beam_direct_transition_mismatch_reported(self):self.assertFalse(self.beam(direct_scores=[-.4,-.6,-.9])['transition_gather_exact_match'])
    def test_beam_ancestry_gap_rejected(self):self.assertRaises(ValueError,self.beam,ancestry=[0,-1,2])
    def test_beam_row_outside_scope_rejected(self):self.assertRaises(ValueError,self.beam,ancestry=[0,4,2])
    def test_beam_prefix_count_off_by_one_rejected(self):self.assertRaises(ValueError,self.beam,ancestry=[0,1])
    def test_beam_initial_start_not_eos(self):self.assertEqual(self.beam()['selected_EOS_new_token_index'],2)
    def test_unexplained_non_eos_finalization_rejected(self):self.assertRaises(ValueError,self.beam,sequence=[2,0,42,43],censor_seen=False)
    def test_internal_eos_rejected(self):self.assertRaises(ValueError,self.beam,sequence=[2,0,2,43])
    def test_no_torch_or_model_imports(self):self.assertFalse(any(x in sys.modules for x in ['torch','transformers','peft']))
    def test_legacy_approval_not_accepted(self):self.assertRaises(ValueError,d.approval_gate,ROOT/'outputs/stage_i/d056_diagnostic_preparation_v1/GPU_AUTHORIZATION_TEMPLATE.json',d.CONFIG,d.OUT/'PREFLIGHT_MANIFEST.json')

if __name__=='__main__':unittest.main(verbosity=2)
