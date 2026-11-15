"""Authorization and pinned Trainer accumulation tests; no GPU or optimizer step."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys
import types
from unittest.mock import patch
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_led_large_contrastive_candidate_v1 as runner
import led_large_contrastive_sequential_v1 as seq
from test_led_large_contrastive_sequential_v1 import TinyConditional,tiny_batch,joint

class AuthorizationTests(unittest.TestCase):
    def args(self,**changes):
        return types.SimpleNamespace(**dict(dict(execute=True,approval=Path('approval.json'),decisions=Path('decisions.json'),
            mode='gate',arm='pair',numerical_receipt=None),**changes))
    def approval(self):
        now=datetime.now(timezone.utc)
        return dict(experiment='LED_LARGE_CONTRASTIVE_PAIR_QUALIFICATION_V1',GPU_execution_approved=True,
            approval_source='researcher_message',config_sha256='h',runner_sha256='h',objective_sha256='h',preflight_sha256='h',
            scopes=['gate'],protected_thesis_days_confirmed=True,authorized_utc=now.isoformat(),
            deadline_utc=(now+timedelta(hours=24)).isoformat(),researcher_decisions_sha256='h')
    def test_execute_flag_required_before_file_io(self):
        with patch.object(runner,'read',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'explicit GPU'):runner.authorization(self.args(execute=False),{})
    def test_offline_receipt_cannot_authorize_gpu(self):
        with patch.object(runner,'read',return_value=dict(experiment='LED_LARGE_CONTRASTIVE_PAIR_QUALIFICATION_V1',GPU_execution_approved=False)):
            with self.assertRaisesRegex(ValueError,'approval absent'):runner.authorization(self.args(),{})
    def test_expired_receipt_stops(self):
        a=self.approval();now=datetime.now(timezone.utc)
        a.update(authorized_utc=(now-timedelta(hours=40)).isoformat(),deadline_utc=(now-timedelta(hours=1)).isoformat())
        with patch.object(runner,'read',return_value=a),patch.object(runner,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'not active'):runner.authorization(self.args(),{})
    def test_pending_positives_cannot_be_promoted(self):
        a=self.approval();d=dict(decider_role='researcher',independently_reviewed=True,positives={'one':dict(decision='PENDING')})
        with patch.object(runner,'read',side_effect=[a,d]),patch.object(runner,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'eight positives'):runner.authorization(self.args(),dict(fit=['one'],pair_ids=[]))
    def test_training_blocked_without_full_numerical_receipt(self):
        a=self.approval();d=dict(decider_role='researcher',independently_reviewed=True,
            positives={'one':dict(decision='ACCEPT',sha256='h')},pairs={'P01':dict(decision='ACCEPT',sha256='h')})
        with patch.object(runner,'read',side_effect=[a,d]),patch.object(runner,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'numerical evidence absent'):runner.authorization(self.args(),dict(fit=['one'],pair_ids=['P01']))

class PinnedTrainerAccumulation(unittest.TestCase):
    def test_native_trainer_eight_example_scaling_without_optimizer(self):
        from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments
        root=Path(__file__).resolve().parents[1]
        p,n=tiny_batch();inputs=dict(positive=p,negative=n)
        model=TinyConditional();model.train()
        class Trainer(Seq2SeqTrainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                return seq.loss(model,**inputs)
        args=Seq2SeqTrainingArguments(output_dir=str(root/'outputs/stage_i/led_large_contrastive_pair_qualification_v1/CPU_TRAINER_FIXTURE'),
            use_cpu=True,per_device_train_batch_size=1,gradient_accumulation_steps=8,report_to=[],remove_unused_columns=False)
        trainer=Trainer(model=model,args=args)
        trainer.model_accepts_loss_kwargs=False
        self.assertEqual(trainer.accelerator.gradient_accumulation_steps,1)
        state=seq.capture_rng(torch.device('cpu'));params=tuple(model.parameters());expected=[torch.zeros_like(x) for x in params]
        for _ in range(8):
            values=torch.autograd.grad(joint(model,p,n)/8,params)
            expected=[a+b for a,b in zip(expected,values)]
        end=seq.capture_rng(torch.device('cpu'));seq.restore_rng(state,torch.device('cpu'))
        for _ in range(8):trainer.training_step(model,inputs)
        self.assertTrue(all(torch.allclose(a,b.grad,atol=1e-6,rtol=1e-5) for a,b in zip(expected,params)))
        self.assertTrue(seq.same_rng(end,seq.capture_rng(torch.device('cpu'))))
        self.assertIsNone(trainer.optimizer)

if __name__=='__main__':unittest.main()
