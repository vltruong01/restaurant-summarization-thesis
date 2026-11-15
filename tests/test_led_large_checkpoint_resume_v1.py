"""Offline resume contracts; no GPU/model forward/backward or optimizer update."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_checkpoint_resume_v1 as recovery
import stage_i_led_large_full_train_v5 as v5


class ResumeTests(unittest.TestCase):
    def test_branch_preserves_original_and_exact_checkpoint(self):
        # Small complete checkpoint fixture tests branch semantics; source hash validation is separate.
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';out=Path(d)/'recovered';train=source/'train';train.mkdir(parents=True)
            cp=train/'checkpoint-250';cp.mkdir()
            (cp/'trainer_state.json').write_text(json.dumps(dict(global_step=250,epoch=1.,max_steps=1250)))
            for name in ('BINDING.json','AUTHORIZATION.json','INITIAL_MODEL_BINDINGS.json','ACTUAL_TRAINING_ARGUMENTS.json',
                         'ACTUAL_AMP_INITIALIZATION.json','ACCEPTED_V4_GATE.json','CHECKPOINT_RECEIPTS.jsonl'):
                (train/name).write_text('{}\n')
            logs={}
            for name in ('progress.jsonl','gradient_checks.jsonl'):
                logs[name]=b''.join((json.dumps(dict(step=i))+'\n').encode() for i in range(1,279))
                (train/name).write_bytes(logs[name])
            with patch.object(recovery,'verify_source',return_value={'source_files':{}}), \
                 patch.object(recovery,'sha',return_value='offline-fixture'), \
                 patch.object(recovery,'binding',return_value={'version':5}):
                ledger=recovery.prepare(source,out)
                self.assertEqual(ledger['superseded_uncheckpointed_steps'],list(range(251,279)))
                self.assertEqual(ledger['cumulative_observed_update_budget_including_discarded_branch'],1278)
                self.assertTrue(ledger['no_fresh_training'])
                for name,raw in logs.items():
                    self.assertEqual((train/name).read_bytes(),raw)
                    self.assertEqual((out/'recovery_evidence'/('ORIGINAL_'+name)).read_bytes(),raw)
                    active=[json.loads(s) for s in (out/'train'/name).read_text().splitlines()]
                    self.assertEqual([x['step'] for x in active],list(range(1,251)))
                self.assertEqual((out/'train/checkpoint-250/trainer_state.json').read_bytes(),(cp/'trainer_state.json').read_bytes())
                with self.assertRaisesRegex(ValueError,'already exists'):
                    recovery.prepare(source,out)

    def test_unknown_or_missing_log_steps_refused(self):
        with self.assertRaisesRegex(ValueError,'Unexpected original log order'):
            recovery.retained_prefix(b'{"step":1}\n{"step":279}\n',250)

    def test_recovery_authority_is_actual_message_not_fabricated_UI(self):
        approval=v5.read(recovery.PREP/'RECOVERY_AUTHORIZATION.json')
        self.assertEqual(approval['approval_source'],'researcher_message')
        self.assertTrue(approval['resume_after_interruption_approved'])
        self.assertEqual(approval['recovery_recomputed_updates_max'],28)
        self.assertEqual(approval['execution_deadline_utc'],'2026-10-21T17:11:48+00:00')
        self.assertEqual(approval['max_training_updates'],1250)
        self.assertEqual(approval['scopes'],['train','evaluate'])

    def test_pinned_trainer_restore_path_and_epoch_sampler(self):
        import torch
        from transformers import Seq2SeqTrainer
        from accelerate.data_loader import prepare_data_loader
        from stage_i_paired_pilot import PairedEpochSampler
        import inspect
        source=inspect.getsource(Seq2SeqTrainer._inner_training_loop)
        for text in ('self._load_optimizer_and_scheduler(resume_from_checkpoint)',
                     'self._load_scaler(resume_from_checkpoint)','self._load_rng_state(resume_from_checkpoint)',
                     'epoch_dataloader.set_epoch(epoch)'):
            self.assertIn(text,source)
        orders=[list(range(8)),list(reversed(range(8)))]
        loader=torch.utils.data.DataLoader(list(range(8)),batch_size=1,sampler=PairedEpochSampler(orders))
        actual=prepare_data_loader(loader,device=torch.device('cpu'),rng_types=[])
        actual.set_epoch(1)
        self.assertEqual([int(batch.item()) for batch in actual],orders[1])

    def test_training_runner_is_unchanged_and_uses_real_resume(self):
        tree=ast.parse(v5.RUNNER.read_text())
        train=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train')
        calls=[x for x in ast.walk(train) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='train']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].keywords[0].arg,'resume_from_checkpoint')
        self.assertEqual(v5.binding(),v5.read(recovery.PREP/'SOURCE_BINDINGS.json')['v5_execution_bindings'])


if __name__=='__main__':unittest.main()
