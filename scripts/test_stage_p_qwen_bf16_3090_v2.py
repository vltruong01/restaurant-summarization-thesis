"""CPU contracts only. No generated summaries, GPU allocation or human ratings."""
import importlib.util,io,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2'
loader=importlib.util.spec_from_file_location('qwen_runner',ROOT/'scripts/stage_p_qwen_bf16_3090_v2.py');m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)
class Contracts(unittest.TestCase):
    def test_exact_UUID_with_or_without_nvidia_prefix(self):
        actual='2df66163-ec71-02c3-6ab0-dc432c0cb759'
        self.assertEqual(m.normalized_uuid(actual),m.normalized_uuid('GPU-'+actual))
        self.assertNotEqual(m.normalized_uuid(actual),m.normalized_uuid('GPU-2df66163-ec71-02c3-6ab0-dc432c0cb750'))
        with self.assertRaises(ValueError):m.normalized_uuid('NOT_A_UUID')
    def test_frozen_complete_isolated_inputs(self):
        s,p,rows=m.inputs(P);self.assertEqual(len(rows),30);self.assertEqual(s['models'][m.MODEL]['model_id'],'Qwen/Qwen3.5-9B');self.assertFalse(s['enable_thinking']);self.assertIsNone(s['quantization'])
    def test_largest_input_pilot_precedes_all_other_calls(self):
        s,p,rows=m.inputs(P);calls=m.read(P/'RENDERED_CALLS_30.json');b,order=m.execution_order(rows,calls)
        self.assertEqual(b,'TA14622324_1001674');self.assertEqual([r['business_id'] for r in order[:2]],[b,b]);self.assertEqual([r['task'] for r in order[:2]],['review','visual']);self.assertEqual(len({r['call_id'] for r in order}),30)
        self.assertEqual(max(r['input_tokens'] for r in calls),3583)
    def test_no_GPU_UUID_rejected_before_torch_or_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='stage_p_contract_') as td:
            dest=Path(td)/'never_created'
            with self.assertRaisesRegex(RuntimeError,'GPU UUID required'):m.execute(P,dest,None)
            self.assertFalse(dest.exists())
    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='stage_p_contract_') as td:
            with self.assertRaisesRegex(RuntimeError,'No retry/resume/overwrite'):m.execute(P,Path(td),'NOT_A_REAL_GPU_TEST_MARKER')
    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='stage_p_contract_') as td:
            t=Path(td)
            for n in ['FROZEN_SPEC.json','PROMPTS.json','INFERENCE_INPUTS.jsonl']:shutil.copy2(P/n,t/n)
            with (t/'INFERENCE_INPUTS.jsonl').open('a',encoding='utf8') as f:f.write('\n')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):m.inputs(t)
    def test_no_prompt_or_tokenization_change_since_preparation(self):
        old=m.read(ROOT/'outputs/stage_p/qwen_comparison_v1/VALIDATION/CPU_COMPATIBILITY.json')['models'][m.MODEL]['calls'];old={r['call_id']:r for r in old}
        for r in m.read(P/'RENDERED_CALLS_30.json'):
            self.assertEqual(r['input_ids_sha256'],old[r['call_id']]['input_ids_sha256']);self.assertEqual(r['rendered_prompt_sha256'],old[r['call_id']]['rendered_prompt_sha256'])
    def test_new_authorization_disables_fallback_and_human_collection(self):
        plan=m.read(P/'EXECUTION_PLAN.json');self.assertFalse(plan['automatic_fallback']);self.assertFalse(plan['automatic_rental']);self.assertFalse(plan['human_evaluation']);self.assertTrue(plan['full_batch_requires_pilot_pass']);self.assertTrue(plan['pilot_calls_reused_in_final'])
if __name__=='__main__':
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    m.put(P/'CPU_CONTRACT_TESTS.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,details=stream.getvalue(),GPU_calls=0,model_outputs_created=0,human_ratings_created=0));print(stream.getvalue());raise SystemExit(0 if r.wasSuccessful() else 1)
