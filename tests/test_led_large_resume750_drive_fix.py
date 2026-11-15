"""CPU-only mount/path fixtures; never mount Drive or launch model code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

CELL=Path(__file__).resolve().parents[1]/'outputs/cloud_handoff/LED_LARGE_RESUME750_DRIVE_FIX.py'
spec=importlib.util.spec_from_file_location('drivefix',CELL)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class DrivePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.mount=Path(self.temp.name)
        (self.mount/'MyDrive').mkdir()
        self.gate=b'CPU fixture, not an accepted scientific gate'
        self.patch=patch.object(m,'_EXPECTED_GATE_SHA',hashlib.sha256(self.gate).hexdigest())
        self.patch.start();self.addCleanup(self.patch.stop)

    def fixture(self,parent):
        source=parent/'runs/F_full_train_v5_resume_checkpoint250_v1/train'
        source.mkdir(parents=True)
        (source/'BINDING.json').write_text(json.dumps(m._EXPECTED_V5_BINDING))
        gate=parent/'runs/F_gradient_gate_v4/feasibility/REPORT.json'
        gate.parent.mkdir(parents=True);gate.write_bytes(self.gate)
        return parent

    def test_standard_project_path(self):
        p=self.fixture(self.mount/'MyDrive/LED-Large_multimodal')
        self.assertEqual(m._resolve_colab_drive(self.mount,False),p)

    def test_runs_shortcut_directly_in_mydrive(self):
        p=self.fixture(self.mount/'MyDrive')
        self.assertEqual(m._resolve_colab_drive(self.mount,False),p)

    def test_known_shortcut_target_parent(self):
        p=self.fixture(self.mount/'.shortcut-targets-by-id/1swRC1tz3u83pgbjybt6wn0MfZsOaH5Z1')
        self.assertEqual(m._resolve_colab_drive(self.mount,False),p)

    def test_missing_mount_or_runs_not_created(self):
        before=list(self.mount.rglob('*'))
        with self.assertRaisesRegex(ValueError,'SHORTCUT'):
            m._resolve_colab_drive(self.mount,False)
        self.assertEqual(list(self.mount.rglob('*')),before)

    def test_two_distinct_copies_not_silently_selected(self):
        self.fixture(self.mount/'MyDrive/LED-Large_multimodal')
        self.fixture(self.mount/'MyDrive')
        with self.assertRaisesRegex(ValueError,'nhiều bản source'):
            m._resolve_colab_drive(self.mount,False)

    def test_wrong_binding_rejected(self):
        p=self.fixture(self.mount/'MyDrive')
        (p/'runs/F_full_train_v5_resume_checkpoint250_v1/train/BINDING.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHORTCUT'):
            m._resolve_colab_drive(self.mount,False)

    def test_wrong_gate_rejected(self):
        p=self.fixture(self.mount/'MyDrive')
        (p/'runs/F_gradient_gate_v4/feasibility/REPORT.json').write_bytes(b'Wrong gate')
        with self.assertRaisesRegex(ValueError,'SHORTCUT'):
            m._resolve_colab_drive(self.mount,False)

    def test_unrelated_drive_folders_not_searched(self):
        self.fixture(self.mount/'MyDrive/unrelated_project')
        with self.assertRaisesRegex(ValueError,'SHORTCUT'):
            m._resolve_colab_drive(self.mount,False)

if __name__=='__main__':unittest.main()
