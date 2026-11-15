"""Delivery-only offline regression tests. No model or GPU imports/execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / 'outputs/cloud_handoff/LED-Large_multimodal'
NOTEBOOK = json.loads((DELIVERY / 'LED_LARGE_MULTIMODAL_F_FULL_TRAIN.ipynb').read_text(encoding='utf-8'))
CELLS = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


class DeliveryTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        self.assertEqual(len(CELLS), 11)
        for cell in CELLS:
            compile(cell, '<delivery-notebook>', 'exec')

    def test_drive_folder_consistent(self):
        text = '\n'.join(CELLS)
        self.assertIn("Path('/content/drive/MyDrive/LED-Large_multimodal')", text)
        self.assertNotIn("Path('/content/drive/MyDrive/thesis_led_large", text)
        self.assertNotIn("Path('/content/drive/MyDrive/LED_LARGE_F_V2", text)
        self.assertIn("RUN_OUT=DRIVE_DIR/'runs/F_run_v2'", text)

    def test_scientific_launch_and_data_cells_unchanged(self):
        old = json.loads((ROOT / 'outputs/cloud_handoff/LED_LARGE_F_FULL_TRAIN_COLAB_V2.ipynb').read_text(encoding='utf-8'))
        before = [''.join(c['source']) for c in old['cells'] if c['cell_type'] == 'code']
        # mount, setup, tests/preflight, downloader, gate/train/eval/report/backup unchanged.
        for i in (0, 2, 3, 4, 6, 7, 8, 9, 10):
            self.assertEqual(CELLS[i], before[i])

    def test_manifest_and_actual_approval_preserved(self):
        m = json.loads((DELIVERY / 'DELIVERY_SHA256_MANIFEST.json').read_text(encoding='utf-8'))
        for name, digest in m['files'].items():
            with (DELIVERY / name).open('rb') as f:
                self.assertEqual(hashlib.file_digest(f, 'sha256').hexdigest(), digest)
        self.assertEqual((DELIVERY / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json').read_bytes(),
                         (ROOT / 'outputs/cloud_handoff/LED_LARGE_F_V2_RESEARCHER_APPROVAL.json').read_bytes())

    def fixture(self, tmp):
        d = Path(tmp); drive = d / 'drive'; drive.mkdir(); extract = d / 'work'
        archive = drive / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
        with zipfile.ZipFile(archive, 'x') as z:
            z.writestr('example.txt', 'fixture only, no scientific content')
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (drive / (archive.name + '.sha256')).write_text(digest + '  ' + archive.name, encoding='utf-8')
        (drive / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json').write_bytes(
            (DELIVERY / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json').read_bytes())
        code = CELLS[1].replace('/content/drive/MyDrive/LED-Large_multimodal', drive.as_posix())
        code = code.replace('/content/led_large_f_multimodal_v2', extract.as_posix())
        # Test-fixture digest only; delivered notebook retains trusted actual payload digest.
        code = code.replace('91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f', digest)
        return drive, extract, code

    def test_extraction_and_existing_directory_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive, extracted, code = self.fixture(tmp)
            exec(code, {})
            self.assertTrue((extracted / 'example.txt').is_file())
            with self.assertRaisesRegex(AssertionError, 'Existing extracted directory'):
                exec(code, {})

    def test_missing_receipt_reports_required_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive, extracted, code = self.fixture(tmp)
            (drive / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip.sha256').unlink()
            with self.assertRaisesRegex(AssertionError, 'Thieu file'):
                exec(code, {})
            self.assertFalse(extracted.exists())

    def test_wrong_receipt_stops_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive, extracted, code = self.fixture(tmp)
            (drive / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip.sha256').write_text('wrong V1', encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'Wrong V2 checksum'):
                exec(code, {})
            self.assertFalse(extracted.exists())

    def test_changed_approval_stops_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive, extracted, code = self.fixture(tmp)
            (drive / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'Approval bytes differ'):
                exec(code, {})
            self.assertFalse(extracted.exists())


if __name__ == '__main__':
    unittest.main()
