"""CPU synthetic tests: native skip integration and Trainer accounting. No LED or GPU execution."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_vast_dynamic_v2 as m
from led_large_dynamic_amp_v2 import Recovery,POLICY,capture_rng,restore_rng,install_trainer
import torch


class FakeScaler:
    """Faithful CPU test double of native AMP skip/backoff, NOT a replacement GPU scaler."""
    def __init__(self):self.value=65536.;self.inf=False;self.growth=0
    def get_scale(self):return self.value
    def is_enabled(self):return True
    def scale(self,loss):return loss*self.value
    def unscale_(self,opt):
        self.inf=False
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.div_(self.value)
                    self.inf |= not torch.isfinite(p.grad).all().item()
    def step(self,opt,closure=None):
        if not self.inf:return opt.step()
    def update(self):
        if self.inf:self.value*=.5;self.growth=0
        else:self.growth+=1
        self.inf=False
    def state_dict(self):return dict(scale=self.value,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=self.growth)
    def load_state_dict(self,state):self.value=state['scale'];self.growth=state['_growth_tracker']


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.2)) for _ in range(456)])
    def forward(self,x=None,labels=None):return {'loss':sum(v*v for v in self.p)*x.float().mean()}


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.no_gpu=patch.object(torch.cuda,'is_available',return_value=False);self.no_gpu.start()
        self.forbid_gpu=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden in offline tests'));self.forbid_gpu.start()
        torch.set_num_threads(1)
    def tearDown(self):self.forbid_gpu.stop();self.no_gpu.stop()

    def fixture(self,d):
        from accelerate import Accelerator
        from accelerate.optimizer import AcceleratedOptimizer
        Accelerator(cpu=True)
        model=Fixture();raw=torch.optim.AdamW(model.parameters(),lr=2e-4);scaler=FakeScaler()
        opt=AcceleratedOptimizer(raw,device_placement=False,scaler=scaler)
        recovery=Recovery(Path(d),3,lambda:None,m.write,m.append)
        return model,raw,opt,scaler,recovery

    def test_initial_scale_and_only_declared_scientific_diff(self):
        old=m.read(ROOT/'configs/stage_i_led_large_vast_fresh_v1.json');new=m.read(m.CONFIG)
        new.pop('dynamic_overflow_policy');new['vast_fresh']=old['vast_fresh'];new['amp_scaler']=old['amp_scaler']
        self.assertEqual(old,new);self.assertEqual(m.AMP['init_scale'],65536.)
        self.assertEqual(m.read(m.CONFIG)['amp_scaler']['backoff_factor'],.5)

    def test_real_scaler_configuration_cpu_only(self):
        from torch.cuda.amp import grad_scaler
        with patch.object(grad_scaler,'amp_definitely_not_available',return_value=False):
            s=grad_scaler.GradScaler();a=SimpleNamespace(scaler=s)
            m.configure_amp_scaler(a,m.read(m.CONFIG));self.assertIs(s,a.scaler)
            self.assertEqual(s.state_dict()['scale'],65536.)

    def test_backoff_preserves_weights_optimizer_and_scheduler_until_success(self):
        with tempfile.TemporaryDirectory() as d:
            model,raw,opt,s,r=self.fixture(d);sched=torch.optim.lr_scheduler.LambdaLR(raw,lambda _:1.)
            r.begin();rng_draws=[]
            def backward():
                rng_draws.append(torch.rand(1).item())
                for _ in range(8):r.microbatch()
                for p in model.parameters():p.grad=torch.ones_like(p)*s.get_scale()
                if s.get_scale()>16384.:next(model.parameters()).grad.fill_(float('inf'))
            backward();before=[p.detach().clone() for p in model.parameters()];epoch=sched.last_epoch
            def skip():
                opt.step();self.assertTrue(opt.step_was_skipped)
                self.assertEqual(sched.last_epoch,epoch)
            r.finish(model,opt,s,lambda:s.unscale_(raw),skip,backward)
            self.assertEqual(s.get_scale(),16384.);self.assertEqual(r.overflow_skips,2)
            self.assertTrue(all(torch.equal(p,b) for p,b in zip(model.parameters(),before)))
            self.assertEqual(len(raw.state),0);self.assertEqual(r.microbatches,24)
            self.assertEqual(len(set(rng_draws)),1,'RNG restored for identical window')
            opt.step();self.assertFalse(opt.step_was_skipped);sched.step();r.commit(s)
            self.assertEqual({int(x['step']) for x in raw.state.values()},{1})
            self.assertEqual(r.successful_updates,1);self.assertEqual(sched.last_epoch,epoch+1)
            log=[json.loads(x) for x in (Path(d)/'AMP_DYNAMIC.jsonl').read_text().splitlines()]
            self.assertEqual([x['event'] for x in log],['native_skip_backoff','native_skip_backoff','successful_update'])

    def test_persistent_overflow_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            model,raw,opt,s,r=self.fixture(d);r.begin()
            def backward():
                for _ in range(8):r.microbatch()
                for p in model.parameters():p.grad=torch.full_like(p,float('nan'))
            backward()
            with self.assertRaisesRegex(ValueError,'Bounded AMP'):
                r.finish(model,opt,s,lambda:s.unscale_(raw),opt.step,backward)
            self.assertEqual(r.overflow_skips,8);self.assertEqual(s.get_scale(),256.)
            self.assertEqual(r.microbatches,72);self.assertEqual(len(raw.state),0)
            self.assertEqual(r.successful_updates,0)

    def test_missing_grad_is_fatal_not_backoff(self):
        with tempfile.TemporaryDirectory() as d:
            model,raw,opt,s,r=self.fixture(d);r.begin()
            with self.assertRaisesRegex(ValueError,'Missing adapter gradient'):
                r.finish(model,opt,s,lambda:s.unscale_(raw),opt.step,lambda:None)
            self.assertEqual(s.get_scale(),65536.)

    def test_nonfinite_clipping_norm_stops(self):
        with tempfile.TemporaryDirectory() as d:
            model,raw,opt,s,r=self.fixture(d);r.begin()
            for p in model.parameters():p.grad=torch.full_like(p,1e38)
            with self.assertRaises(RuntimeError):r.finish(model,opt,s,lambda:None,opt.step,lambda:None)
            self.assertEqual(r.overflow_skips,0)

    def test_bad_native_skip_that_updates_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,raw,opt,s,r=self.fixture(d);r.begin()
            for p in model.parameters():p.grad=torch.full_like(p,float('inf'))
            def wrong():
                s.value*=.5
                with torch.no_grad():next(model.parameters()).add_(1.)
            with self.assertRaisesRegex(ValueError,'changed adapter weights'):
                r.finish(model,opt,s,lambda:None,wrong,lambda:None)

    def test_microbatch_and_update_budget_guards(self):
        with tempfile.TemporaryDirectory() as d:
            r=Recovery(Path(d),3,lambda:None,m.write,m.append);r.microbatches=88
            with self.assertRaisesRegex(ValueError,'microbatch ceiling'):r.microbatch()
            r.pending=False
            with self.assertRaisesRegex(ValueError,'No finite'):r.commit(FakeScaler())

    def test_previous_failed_gate_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'feasibility';p.mkdir();(p/'REPORT.json').write_text('{}');(p/'FAILURE_STOP_NO_RETRY.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Failed gate'):m.require_gate(Path(d),{})

    def test_no_approval_or_old_approval_rejected(self):
        with self.assertRaisesRegex(ValueError,'GPU disabled'):m.authorize(None,'train',False)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';a=m.read(ROOT/'VAST_RESEARCHER_AUTHORIZATION.json');a['experiment']='LED_LARGE_VAST_FRESH_V1';p.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'provenance'):m.authorize(p,'train',True)

    def test_end_to_end_real_trainer_two_successful_updates_after_overflow(self):
        """Actual Transformers4.51.3+Accelerate1.6 loop; CPU-only scaler test double."""
        from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainerCallback
        from torch.utils.data import Dataset
        class Data(Dataset):
            def __len__(self):return 16
            def __getitem__(self,i):return dict(x=torch.tensor([float(i+1)]),labels=torch.tensor([1]))
        with tempfile.TemporaryDirectory() as d:
            scaler=FakeScaler();model=Fixture();records=[]
            # Inject only synthetic gradient overflow above32768; forward loss stays finite.
            next(model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')) if scaler.value>32768. else g)
            class Trainer(Seq2SeqTrainer):
                def create_accelerator_and_postprocess(self):
                    super().create_accelerator_and_postprocess();self.accelerator.scaler=scaler
                    self.accelerator.native_amp=True
                    # CPU fixture only: provide equivalent public unscale operation.
                    def unscale(optimizer):
                        while hasattr(optimizer,'optimizer'):optimizer=optimizer.optimizer
                        scaler.unscale_(optimizer)
                    self.accelerator.unscale_gradients=unscale
                def training_step(self,model,inputs,num_items_in_batch=None):
                    r=self.amp_recovery
                    if not r.window:r.begin()
                    r.window.append({k:v.detach().cpu().clone() for k,v in inputs.items()});r.microbatch()
                    return super().training_step(model,inputs,num_items_in_batch)
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    out=model(**inputs)
                    if not torch.isfinite(out['loss']):raise ValueError('Nonfinite forward loss')
                    return (out['loss'],out) if return_outputs else out['loss']
            class CB(TrainerCallback):
                def on_optimizer_step(self,args,state,control,**kw):
                    self_t.assertFalse(t.accelerator.optimizer_step_was_skipped)
                    t.amp_recovery.commit(scaler)
                    records.append((state.global_step,{int(v['step']) for v in t.optimizer.state.values()}))
            self_t=self
            args=Seq2SeqTrainingArguments(output_dir=d,use_cpu=True,per_device_train_batch_size=1,gradient_accumulation_steps=8,
                num_train_epochs=1,save_strategy='no',report_to='none',disable_tqdm=True,dataloader_pin_memory=False,max_grad_norm=1.)
            t=Trainer(model=model,args=args,train_dataset=Data(),callbacks=[CB()]);t.model_accepts_loss_kwargs=False
            install_trainer(t,Path(d),lambda:None,m.write,m.append)
            result=t.train()
            self.assertEqual(t.state.global_step,2);self.assertEqual(t.amp_recovery.successful_updates,2)
            self.assertEqual(t.amp_recovery.overflow_skips,1);self.assertEqual(t.amp_recovery.microbatches,24)
            self.assertEqual(records,[(0,{1}),(1,{2})]);self.assertEqual(t.lr_scheduler.last_epoch,2)
            self.assertEqual(t.state.epoch,1.)


if __name__=='__main__':unittest.main()
