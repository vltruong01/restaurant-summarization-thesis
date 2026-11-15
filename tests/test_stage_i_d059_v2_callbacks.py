"""Execute the exact nested Record class offline, using stub boundary dependencies.

No copied callback implementation, Transformer import, tensors, model or CUDA.
This regression reproduces v1's NameError and calls every v2 callback method.
"""
import ast,copy,json,runpy,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime,timezone
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d059_v2 as d


def callback_from_actual_runner(path,out):
    ns=runpy.run_path(str(path),run_name='d059_offline_callback_regression')
    node=next(n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(n,ast.ClassDef) and n.name=='Record')
    gradients={'synthetic.adapter.lora_B.default.weight':{'finite':True,'nonzero':True,'norm':0.125}}
    gradient_fn=Mock(return_value=gradients)
    deadline={'execution_authorized_at_utc':'2026-10-18T15:10:48.941817+00:00',
        'improvement_deadline_utc':'2026-10-25T15:10:48.941817+00:00'}
    ns.update(TrainerCallback=object,modeout=out,gradient_record=gradient_fn,
        trainer=NS(accelerator=NS(optimizer_step_was_skipped=False)),
        memory=lambda:{'synthetic_allocated':101,'synthetic_reserved':202},approval=deadline,
        check_deadline=lambda a:d.check_deadline(a,datetime(2026,10,19,tzinfo=timezone.utc)))
    # Compile and invoke the unmodified class AST extracted from the actual runner.
    code=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]))
    exec(compile(code,str(path),'exec'),ns)
    return ns['Record'](),ns,gradient_fn,gradients


class CallbackRegression(unittest.TestCase):
    def test_v1_actual_callback_reproduces_missing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback,ns,gradient,expected=callback_from_actual_runner(ROOT/'scripts/run_stage_i_d059_v1.py',Path(tmp))
            self.assertNotIn('json',ns)
            with self.assertRaisesRegex(NameError,"name 'json' is not defined"):
                callback.on_pre_optimizer_step(None,NS(global_step=0),None,model=object())
            self.assertEqual((Path(tmp)/'gradient_checks.jsonl').read_text(encoding='utf-8'),'')
    def test_v2_actual_gradient_callback_serializes_early_and_later_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback,ns,gradient,expected=callback_from_actual_runner(d.RUNNER,Path(tmp))
            sentinel=object()
            callback.on_pre_optimizer_step(None,NS(global_step=0),None,model=sentinel)
            callback.on_pre_optimizer_step(None,NS(global_step=2),None,model=sentinel)
            rows=[json.loads(x) for x in (Path(tmp)/'gradient_checks.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows,[{'step':1,'branches':expected},{'step':3,'branches':expected}])
            self.assertEqual([x.kwargs['expect_later_A'] for x in gradient.call_args_list],[False,True])
    def test_progress_callback_serializes_scheduler_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback,ns,*_=callback_from_actual_runner(d.RUNNER,Path(tmp))
            scheduler=NS(get_last_lr=lambda:[0.0002,0.0002])
            callback.on_step_end(None,NS(global_step=8,epoch=.032),None,lr_scheduler=scheduler)
            callback.on_step_end(None,NS(global_step=9,epoch=.036),None,lr_scheduler=scheduler)
            rows=[json.loads(x) for x in (Path(tmp)/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows,[{'step':8,'epoch':.032,'lr':[.0002,.0002],'synthetic_allocated':101,'synthetic_reserved':202}])
            with self.assertRaisesRegex(ValueError,'Update budget exceeded'):
                callback.on_step_end(None,NS(global_step=1251),None)
    def test_optimizer_callback_checks_amp_and_exact_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback,ns,*_=callback_from_actual_runner(d.RUNNER,Path(tmp))
            optimizer=NS(state={'a':{'step':3},'b':{'step':3}})
            callback.on_optimizer_step(None,NS(global_step=2),None,optimizer=optimizer)
            ns['trainer'].accelerator.optimizer_step_was_skipped=True
            with self.assertRaisesRegex(ValueError,'AMP update skipped'):
                callback.on_optimizer_step(None,NS(global_step=2),None,optimizer=optimizer)
            ns['trainer'].accelerator.optimizer_step_was_skipped=False
            with self.assertRaisesRegex(ValueError,'Optimizer step differs'):
                callback.on_optimizer_step(None,NS(global_step=1),None,optimizer=optimizer)
    def test_train_begin_checks_schedule_and_original_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback,ns,*_=callback_from_actual_runner(d.RUNNER,Path(tmp))
            callback.on_train_begin(None,NS(max_steps=1250),None)
            with self.assertRaisesRegex(ValueError,'Unexpected schedule horizon'):
                callback.on_train_begin(None,NS(max_steps=1251),None)
    def test_checkpoint_callback_writes_real_json_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);callback,ns,*_=callback_from_actual_runner(d.RUNNER,folder)
            checkpoint=folder/'checkpoint-250';checkpoint.mkdir()
            needed=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json']
            for name in needed:(checkpoint/name).write_bytes(('SYNTHETIC CALLBACK FIXTURE ONLY '+name).encode())
            callback.on_save(None,NS(global_step=250),None)
            proof=json.loads((checkpoint/'checkpoint_integrity.json').read_text(encoding='utf-8'))
            self.assertEqual(proof['config_sha256'],d.sha(d.CONFIG))
            self.assertEqual(proof['files'],{name:d.sha(checkpoint/name) for name in needed})
            with self.assertRaisesRegex(ValueError,'Unexpected epoch checkpoint'):
                callback.on_save(None,NS(global_step=249),None)
    def test_checkpoint_callback_rejects_missing_required_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);callback,ns,*_=callback_from_actual_runner(d.RUNNER,folder)
            (folder/'checkpoint-500').mkdir()
            with self.assertRaisesRegex(ValueError,'Incomplete checkpoint'):
                callback.on_save(None,NS(global_step=500),None)
            self.assertFalse((folder/'checkpoint-500/checkpoint_integrity.json').exists())
    def test_all_record_callbacks_called_without_heavy_runtime(self):
        names={'on_train_begin','on_pre_optimizer_step','on_optimizer_step','on_step_end','on_save'}
        tree=ast.parse(d.RUNNER.read_text(encoding='utf-8'))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Record')
        self.assertEqual({n.name for n in node.body if isinstance(n,ast.FunctionDef)},names)
        self.assertFalse(any(x in sys.modules for x in ('torch','transformers','peft')))


if __name__=='__main__':unittest.main(verbosity=2)
