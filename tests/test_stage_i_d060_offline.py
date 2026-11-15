"""Offline contracts/negative controls and actual callback invocation with doubles."""
import ast,copy,json,runpy,sys,tempfile,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d060 as d

def callback(out):
    ns=runpy.run_path(str(d.RUNNER),run_name='d060_offline_test')
    node=next(n for n in ast.walk(ast.parse(d.RUNNER.read_text())) if isinstance(n,ast.ClassDef) and n.name=='Record')
    gpu=NS(reset_peak_memory_stats=Mock(),mem_get_info=lambda:(800,1000),memory_reserved=lambda:100,memory_allocated=lambda:50,max_memory_reserved=lambda:100)
    ns.update(TrainerCallback=object,modeout=out,gradient_record=Mock(return_value={'synthetic':{'finite':True,'nonzero':True,'norm':1.}}),
              optimizer=NS(state={}),trainer=NS(accelerator=NS(optimizer_step_was_skipped=False)),memory=lambda:{'synthetic':True},
              torch=NS(cuda=gpu),approval={},check_deadline=lambda a:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),str(d.RUNNER),'exec'),ns)
    return ns['Record'](),ns

class Contracts(unittest.TestCase):
    def test_config(self):self.assertTrue(d.check_config(d.read(d.CONFIG)))
    def test_no_architecture_or_decoding_change(self):
        for key in ('lora','generation','budgets','revision'):
            c=d.read(d.CONFIG);c[key]=None
            with self.assertRaises(ValueError):d.check_config(c)
    def test_no_schedule_or_scope_expansion(self):
        for key,value in [('expected_modules',48),('evaluation_epochs',[1,2,3]),('training',{}),('viability_hours',73),('fresh_optimizer',False)]:
            c=d.read(d.CONFIG);c[key]=value
            with self.assertRaises(ValueError):d.check_config(c)
    def test_false_approval_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json';p.write_text(json.dumps({'decision':'D-060','GPU_execution_approved':False}))
            with self.assertRaisesRegex(ValueError,'Explicit GPU approval'):d.gate(p,'train')
        self.assertFalse(any(x in sys.modules for x in ('torch','transformers','peft')))
    def test_wrong_authorization_hash_rejected(self):
        a={'decision':'D-060','GPU_execution_approved':True,'approval_source':'researcher_message','researcher_explicit_authorization':'fixture','config_sha256':'wrong'}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json';p.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'Approval hash mismatch'):d.gate(p,'train')
    def test_deadline_expiry_and_extension(self):
        start=datetime(2026,10,19,tzinfo=timezone.utc);a={'execution_authorized_at_utc':start.isoformat(),'viability_deadline_utc':(start+timedelta(hours=72)).isoformat()}
        d.check_deadline(a,start+timedelta(hours=71))
        with self.assertRaisesRegex(ValueError,'deadline reached'):d.check_deadline(a,start+timedelta(hours=72))
        a['viability_deadline_utc']=(start+timedelta(hours=73)).isoformat()
        with self.assertRaisesRegex(ValueError,'Clock extended'):d.check_deadline(a,start)
    def test_original_deadline_cap(self):
        a={'execution_authorized_at_utc':'2026-10-24T00:00:00+00:00','viability_deadline_utc':'2026-10-25T15:10:48.941817+00:00'}
        d.check_deadline(a,datetime(2026,10,24,tzinfo=timezone.utc))
    def test_old_source_does_not_get_reinitialized(self):
        code=d.RUNNER.read_text();helper=(ROOT/'scripts/stage_i_d060.py').read_text()
        self.assertNotIn('initialize_named_adapters(',code+helper)
        self.assertIn('is_trainable=True',helper);self.assertIn('torch.equal',helper)
        self.assertIn("torch_device='cpu'",helper)
        self.assertNotIn('optimizer.load_state_dict',code);self.assertIn('resume_from_checkpoint=None',code)
        self.assertIn('get_linear_schedule_with_warmup(optimizer,38,750)',code)
    def test_cpu_evidence(self):
        p=d.read(d.PREP/'CPU_ADAPTER_COMPATIBILITY.json')
        self.assertEqual(p['adapter_trainable_count'],294912);self.assertEqual(p['loaded']['loaded_tensor_count'],48)
        self.assertTrue(p['backbone_historical_fingerprints_match']);self.assertEqual(p['forward_backward_generation_optimizer_step_calls'],0)
        self.assertEqual(p['optimizer_initial_state_entries'],0);self.assertFalse(p['GPU_feasibility_claimed'])
    def test_alignment(self):
        p=d.read(d.PREP/'DATA_ALIGNMENT.json');self.assertEqual(p['exact_source_and_target_encodes'],13200)
        self.assertEqual(p['stats']['train_filtered']['zero_captions'],9);self.assertEqual(p['stats']['validation_filtered']['zero_captions'],1)
    def test_orders_and_fixed_cases(self):
        x=d.read(d.PREP/'EPOCH_ORDERS.json');old=d.read(ROOT/'outputs/stage_i/d059_preparation_v2/EPOCH_ORDERS.json')
        self.assertEqual(x['indices'],old['indices'][:3]);self.assertEqual(x['training_business_ids'],old['training_business_ids'])
        self.assertEqual(d.read(d.PREP/'EVALUATION_MANIFEST.json'),d.read(ROOT/'outputs/stage_i/d059_preparation_v2/EVALUATION_MANIFEST.json'))
    def test_all_frozen_bindings_unchanged(self):
        for rel,h in d.read(d.PREP/'PROTECTED_INPUTS.json')['files'].items():
            p=Path(rel);self.assertEqual(d.sha(p if p.is_absolute() else ROOT/p),h,rel)

class Callbacks(unittest.TestCase):
    def test_actual_gradient_callback_serializes_trained_A_and_B(self):
        with tempfile.TemporaryDirectory() as td:
            cb,ns=callback(Path(td));cb.on_pre_optimizer_step(None,NS(global_step=0),None,model=object())
            self.assertEqual(json.loads((Path(td)/'gradient_checks.jsonl').read_text())['step'],1)
            self.assertTrue(ns['gradient_record'].call_args.kwargs['expect_later_A'])
    def test_begin_and_fresh_optimizer_guard(self):
        with tempfile.TemporaryDirectory() as td:
            cb,ns=callback(Path(td));cb.on_train_begin(None,NS(max_steps=750),None)
            ns['optimizer'].state={'inherited':1}
            with self.assertRaisesRegex(ValueError,'Optimizer must be fresh'):cb.on_train_begin(None,NS(max_steps=750),None)
    def test_amp_and_memory_and_steps(self):
        with tempfile.TemporaryDirectory() as td:
            cb,ns=callback(Path(td));cb.on_optimizer_step(None,NS(global_step=0),None,optimizer=NS(state={'x':{'step':1}}))
            ns['torch'].cuda.mem_get_info=lambda:(100,1000)
            with self.assertRaisesRegex(ValueError,'headroom'):cb.on_optimizer_step(None,NS(global_step=1),None,optimizer=NS(state={'x':{'step':2}}))
            ns['trainer'].accelerator.optimizer_step_was_skipped=True
            with self.assertRaisesRegex(ValueError,'AMP update skipped'):cb.on_optimizer_step(None,NS(global_step=0),None)
    def test_progress_and_budget(self):
        with tempfile.TemporaryDirectory() as td:
            cb,ns=callback(Path(td));cb.on_step_end(None,NS(global_step=8,epoch=.032),None,lr_scheduler=NS(get_last_lr=lambda:[0.]))
            self.assertEqual(json.loads((Path(td)/'progress.jsonl').read_text())['step'],8)
            with self.assertRaisesRegex(ValueError,'budget exceeded'):cb.on_step_end(None,NS(global_step=751),None)
    def test_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);cb,ns=callback(out);p=out/'checkpoint-250';p.mkdir()
            for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'):(p/name).write_text('SYNTHETIC ONLY')
            cb.on_save(None,NS(global_step=250),None);self.assertTrue((p/'checkpoint_integrity.json').exists())
            with self.assertRaisesRegex(ValueError,'Unexpected epoch checkpoint'):cb.on_save(None,NS(global_step=1000),None)
            (out/'checkpoint-500').mkdir()
            with self.assertRaisesRegex(ValueError,'Incomplete checkpoint'):cb.on_save(None,NS(global_step=500),None)

if __name__=='__main__':unittest.main(verbosity=2)
