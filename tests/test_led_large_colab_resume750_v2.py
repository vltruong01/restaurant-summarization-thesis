"""CPU-only recovery fixtures; no model loading, GPU operations or production file writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/stage_i_led_large_colab_resume750_v2.py'
spec = importlib.util.spec_from_file_location('resume750', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.drive = Path(self.temp.name)
        self.source = self.drive/'runs'/m.SOURCE_NAME
        train = self.source/'train'
        train.mkdir(parents=True)
        self.binding = {'fixture_binding': 'NOT_SCIENTIFIC_DATA'}
        self.v = SimpleNamespace(binding=lambda:self.binding, preflight=lambda:None,
            validate_gate=lambda p:None, check_deadline=lambda a:None)
        for name in m.METADATA:
            m.write_new(train/name, {})
        (train/'BINDING.json').write_text(json.dumps(self.binding))
        (train/'AUTHORIZATION.json').write_text(json.dumps({'execution_deadline_utc':m.DEADLINE,
            'execution_authorized_at_utc':'2026-10-19T20:41:00+00:00'}))
        receipts = []
        for step in (250,500,750):
            cp = train/f'checkpoint-{step}'
            cp.mkdir()
            for name in m.FULL_STATE:
                (cp/name).write_bytes(b'CPU_FIXTURE_NOT_A_MODEL')
            (cp/'trainer_state.json').write_text(json.dumps(dict(global_step=step,epoch=step/250,max_steps=1250)))
            m.write_new(cp/'checkpoint_integrity.json',dict(binding=self.binding,global_step=step,
                files={n:m.sha(cp/n) for n in m.FULL_STATE}))
            receipts.append(dict(step=step,checkpoint_integrity_sha256=m.sha(cp/'checkpoint_integrity.json')))
        (train/'CHECKPOINT_RECEIPTS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in receipts))
        for name in ('progress.jsonl','gradient_checks.jsonl'):
            (train/name).write_text(''.join(json.dumps(dict(step=s))+'\n' for s in range(1,891)))
        m.write_new(self.source/'RECOVERY_LINEAGE.json',dict(restored_checkpoint_step=250,
            original_last_logged_step=278,additional_recomputed_updates_max=28,execution_deadline_utc=m.DEADLINE))

    def test_preserves_source_and_every_full_checkpoint(self):
        before = m.snapshot_files(self.source)
        with patch.object(m,'frozen',return_value=self.v):
            output = m.prepare(self.drive)
        self.assertEqual(m.snapshot_files(self.source),before)
        for step in (250,500,750):
            for p in (self.source/'train'/f'checkpoint-{step}').iterdir():
                self.assertEqual(m.sha(p),m.sha(output/'train'/f'checkpoint-{step}'/p.name))
        ledger=m.read(output/'RECOVERY_LINEAGE.json')
        self.assertEqual((ledger['additional_recomputed_updates_max'],ledger['cumulative_observed_optimizer_updates_max']),(140,1418))
        self.assertEqual(len((output/'train/progress.jsonl').read_text().splitlines()),750)
        self.assertEqual(len((output/'recovery_evidence/source_snapshot/train/progress.jsonl').read_text().splitlines()),890)
        self.assertEqual(m.read(output/'RECOVERY_APPROVAL.json')['execution_deadline_utc'],m.DEADLINE)

    def test_corrupt_optimizer_stops_before_branch_creation(self):
        (self.source/'train/checkpoint-750/optimizer.pt').write_bytes(b'CORRUPT')
        with patch.object(m,'frozen',return_value=self.v),self.assertRaisesRegex(ValueError,'hash differs'):
            m.prepare(self.drive)
        self.assertFalse((self.source.parent/m.OUTPUT_NAME).exists())

    def test_numerical_failure_is_not_retried(self):
        m.write_new(self.source/'train/FAILURE_STOP_NO_RETRY.json',{'error':'fixture'})
        with self.assertRaisesRegex(ValueError,'failed numerical'):
            m.verify_source(self.source,self.v)

    def test_newer_checkpoint_does_not_silently_fall_back(self):
        (self.source/'train/checkpoint-1000').mkdir()
        with self.assertRaisesRegex(ValueError,'Checkpoint set differs'):
            m.verify_source(self.source,self.v)

    def test_disagreeing_logs_are_preserved(self):
        p=self.source/'train/gradient_checks.jsonl'
        with p.open('a') as f:f.write('{"step":891}\n')
        with self.assertRaisesRegex(ValueError,'histories disagree'):
            m.verify_source(self.source,self.v)

    def test_old250_snapshot_is_not_recovered(self):
        p=self.source/'train/progress.jsonl'
        p.write_text(''.join(json.dumps(dict(step=s))+'\n' for s in range(1,279)))
        with self.assertRaisesRegex(ValueError,'Unexpected progress'):
            m.verify_source(self.source,self.v)

    def test_preparation_idempotent_but_started_branch_rejected(self):
        with patch.object(m,'frozen',return_value=self.v):
            out=m.prepare(self.drive)
            self.assertEqual(m.prepare(self.drive),out)
            m.write_new(out/'train/RESUME_AUTHORIZATION_STEP_750.json',{})
            with self.assertRaisesRegex(ValueError,'already launched'):
                m.prepare(self.drive)

    def test_incomplete_preparation_not_overwritten(self):
        (self.source.parent/m.OUTPUT_NAME).mkdir()
        with patch.object(m,'frozen',return_value=self.v),self.assertRaisesRegex(ValueError,'Incomplete CPU'):
            m.prepare(self.drive)

    def test_launcher_always_passes_explicit_resume750_to_unchanged_runner(self):
        with patch.object(m,'frozen',return_value=self.v):
            out=m.prepare(self.drive)
        self.v.authorize=lambda *args:None
        process=SimpleNamespace(stdout=iter(['CPU mocked launcher only\n']),wait=lambda:0)
        with patch.object(m,'frozen',return_value=self.v), \
                patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}), \
                patch.object(m.subprocess,'Popen',return_value=process) as start:
            m.launch(self.drive,'train')
        cmd=start.call_args.args[0]
        self.assertEqual(cmd[1:3],['scripts/run_stage_i_led_large_full_train_v5.py','train'])
        self.assertEqual(cmd[-2:],['--resume',str(out/'train/checkpoint-750')])
        with patch.object(m,'frozen',return_value=self.v), \
                patch.object(m.subprocess,'Popen') as start, self.assertRaisesRegex(ValueError,'already launched'):
            m.launch(self.drive,'train')
        start.assert_not_called()

if __name__=='__main__':
    unittest.main()
