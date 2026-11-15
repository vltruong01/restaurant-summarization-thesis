"""CPU/meta regression and negative controls; no model forward/backward/generation."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_runtime_v2 as runtime
from run_stage_i_led_large_colab_v2 import MemoryMeter


class FullNotebookTests(unittest.TestCase):
    def test_frozen_config_equivalence(self):
        self.assertTrue(runtime.check_config(runtime.read(runtime.CONFIG)))

    def test_changed_learning_rate_rejected(self):
        c = runtime.read(runtime.CONFIG);c['optimization']['learning_rate'] = 1e-4
        with self.assertRaises(ValueError):runtime.check_config(c)

    def test_changed_scope_rejected(self):
        c = runtime.read(runtime.CONFIG);c['expected_modules'] = 114
        with self.assertRaises(ValueError):runtime.check_config(c)

    def test_no_execution_flag_stops_before_imports(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'run'
            p = subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_led_large_colab_v2.py'),
                                'train','--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('GPU disabled',p.stderr)
            self.assertFalse(output.exists())

    def test_false_approval_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as d:
            approval = Path(d)/'a.json';approval.write_text('{"GPU_execution_approved":false}')
            p = subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_led_large_colab_v2.py'),
                    'feasibility','--execute','--approval',str(approval),'--output',str(Path(d)/'run')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('No explicit researcher',p.stderr)
            self.assertFalse((Path(d)/'run').exists())

    def test_pure_memory_headroom(self):
        r = MemoryMeter.estimate([dict(total=1000,used=650,reserved=600)],700)
        self.assertAlmostEqual(r['headroom_fraction'],.25)

    def test_memory_empty_rejected(self):
        with self.assertRaises(ValueError):MemoryMeter.estimate([],0)

    def test_low_headroom_not_disguised(self):
        r = MemoryMeter.estimate([dict(total=1000,used=900,reserved=850)],850)
        self.assertLess(r['headroom_fraction'],.15)

    def test_exact_generation_metadata(self):
        g = runtime.generation(runtime.read(runtime.CONFIG))
        self.assertEqual(g.num_beams,4);self.assertEqual(g.max_length,768)
        self.assertIsNone(g.max_new_tokens);self.assertEqual(g.no_repeat_ngram_size,0)
        self.assertFalse(g.do_sample)

    def test_d058_training_arguments_retained(self):
        c = runtime.read(runtime.CONFIG);a = runtime.training_arguments(c,Path('out'))
        self.assertEqual(a['num_train_epochs'],5);self.assertEqual(a['gradient_accumulation_steps'],8)
        self.assertFalse(a['load_best_model_at_end']);self.assertEqual(a['save_total_limit'],5)
        self.assertEqual(a['save_strategy'],'epoch');self.assertTrue(a['fp16']);self.assertFalse(a['bf16'])

    def test_deadline_cannot_extend_or_expire(self):
        from datetime import datetime,timezone
        a = dict(execution_authorized_at_utc='2026-10-20T00:00:00+00:00',
                 execution_deadline_utc='2026-10-22T00:00:00+00:00',defense_reserve_at_least_14_days_confirmed=True)
        self.assertTrue(runtime.check_deadline(a,datetime(2026,10,21,tzinfo=timezone.utc)))
        a['execution_deadline_utc'] = '2026-10-23T00:00:00+00:00'
        with self.assertRaises(ValueError):runtime.check_deadline(a,datetime(2026,10,21,tzinfo=timezone.utc))

    def test_researcher_reserve_required(self):
        from datetime import datetime,timezone
        a = dict(execution_authorized_at_utc='2026-10-20T00:00:00+00:00',execution_deadline_utc='2026-10-21T00:00:00+00:00')
        with self.assertRaises(ValueError):runtime.check_deadline(a,datetime(2026,10,20,1,tzinfo=timezone.utc))

    def test_actual_meta_228_module_attachment(self):
        import torch
        from transformers import LEDConfig,LEDForConditionalGeneration
        from peft import LoraConfig,get_peft_model
        mc = runtime.read(runtime.LARGE_PREP/'remote_metadata/official/config.json')
        expected = runtime.read(runtime.LARGE_PREP/'META_LORA_INVENTORY.json')
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')), \
             patch.object(LEDForConditionalGeneration,'forward',side_effect=AssertionError('Forward forbidden')), \
             patch.object(LEDForConditionalGeneration,'generate',side_effect=AssertionError('Generation forbidden')):
            with torch.device('meta'):
                model = LEDForConditionalGeneration(LEDConfig(**mc))
                with patch('peft.tuners.lora.model.is_bnb_available',return_value=False), \
                     patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
                    model = get_peft_model(model,LoraConfig(**runtime.read(runtime.CONFIG)['lora'],target_modules='all-linear'))
            r = runtime.inspect_model(model,expected)
            self.assertEqual(r['trainable_parameters'],4915200);self.assertEqual(r['modules'],228)
            self.assertTrue(all(p.device.type == 'meta' for p in model.parameters()))

    def test_callback_really_serializes_456_gradient_records(self):
        import torch
        ps = []
        for i in range(228):
            for adapter in ('A','B'):
                p = torch.nn.Parameter(torch.ones(1));p.grad = torch.ones(1)
                ps.append((f'base_model.model.led.fake.{i}.lora_{adapter}.default.weight',p))
        model = SimpleNamespace(named_parameters=lambda: iter(ps))
        memory = SimpleNamespace(check=lambda:None,summary=lambda:dict(headroom_fraction=.5))
        state = SimpleNamespace(global_step=0,max_steps=1250,epoch=0.)
        with tempfile.TemporaryDirectory() as d, patch.object(runtime,'check_deadline',return_value='ok'):
            record = runtime.callback_class()(Path(d),{},memory)
            record.on_train_begin(None,state,None)
            record.on_pre_optimizer_step(None,state,None,model=model)
            obj = json.loads((Path(d)/'gradient_checks.jsonl').read_text())
            self.assertEqual(len(obj['gradients']),456);self.assertEqual(obj['step'],1)
            record.trainer = SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=False))
            record.on_optimizer_step(None,state,None,optimizer=SimpleNamespace(state={0:{'step':1}}))
            state.global_step=1
            record.on_step_end(None,state,None,lr_scheduler=SimpleNamespace(get_last_lr=lambda:[.00001]))
            self.assertEqual(json.loads((Path(d)/'progress.jsonl').read_text())['step'],1)

    def test_callback_checkpoint_dependencies(self):
        state = SimpleNamespace(global_step=250)
        with tempfile.TemporaryDirectory() as d, patch.object(runtime,'binding',return_value={'test_only':True}):
            folder = Path(d);p = folder/'checkpoint-250';p.mkdir()
            for n in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'):
                (p/n).write_text('test placeholder, not a scientific checkpoint')
            callback = runtime.callback_class()(folder,{},None)
            callback.on_save(None,state,None)
            proof = json.loads((p/'checkpoint_integrity.json').read_text())
            self.assertEqual(len(proof['files']),7)
            self.assertTrue((folder/'CHECKPOINT_RECEIPTS.jsonl').exists())

    def test_callback_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            callback = runtime.callback_class()(Path(d),{},None)
            with self.assertRaises(ValueError):callback.on_save(None,SimpleNamespace(global_step=250),None)

    def test_initialization_inventory_complete(self):
        x = runtime.read(runtime.PREP/'INITIALIZATION_HASHES.json')
        self.assertEqual(len(x['per_module_sha256']),228)
        self.assertEqual(x['trainable_parameters'],4915200)

    def test_no_top_level_torch_import_in_gate_modules(self):
        for name in ('stage_i_led_large_runtime_v2.py','run_stage_i_led_large_colab_v2.py'):
            tree = ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
            imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
            text = '\n'.join(ast.unparse(n) for n in imports)
            self.assertNotIn('import torch',text);self.assertNotIn('from transformers',text)

    def test_every_full_notebook_code_cell_compiles(self):
        from build_led_large_full_train_colab_v2 import notebook
        nb = notebook();cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertGreaterEqual(len(cells),10)
        for cell in cells:ast.parse(''.join(cell['source']))
        text = '\n'.join(''.join(c['source']) for c in cells)
        self.assertIn("run_logged('feasibility')",text)
        self.assertIn("run_logged('train',extra)",text)
        self.assertIn("run_logged('evaluate')",text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
