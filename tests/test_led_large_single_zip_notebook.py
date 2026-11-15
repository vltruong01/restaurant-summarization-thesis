"""Transport-only single-ZIP tests. No installer, model or GPU execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'outputs/cloud_handoff'
nb=json.loads((H/'LED_LARGE_V3_ALL_IN_ONE.ipynb').read_text(encoding='utf-8'))
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']


class SingleZipTests(unittest.TestCase):
    def fixture(self,d):
        d=Path(d);drive=d/'drive';drive.mkdir();work=d/'work';files=d/'unpacked'
        inner=d/'inner.zip'
        with zipfile.ZipFile(inner,'x') as z:z.writestr('configs/stage_i_led_large_colab_v2.json','{}')
        payload=inner.read_bytes();data_hash=hashlib.sha256(payload).hexdigest()
        patch_data=(H/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip').read_bytes()
        manifest={'files':{'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip':data_hash,
                          'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip':hashlib.sha256(patch_data).hexdigest()}}
        bundle=drive/'LED_LARGE_V3_ALL_IN_ONE.zip'
        with zipfile.ZipFile(bundle,'x') as z:
            z.writestr('LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip',payload)
            z.writestr('LED_LARGE_GRADIENT_GATE_V3_PATCH.zip',patch_data)
            z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest))
        text=codes[1].replace('/content/drive/MyDrive/LED-Large_multimodal',drive.as_posix())
        text=text.replace('/content/led_large_v3_all_in_one_files',files.as_posix()).replace('/content/led_large_f_multimodal_v2',work.as_posix())
        replacements={
            'e8352a1ceb9b76c5cadb7a33d98b22ea2675cb0dbbfef581b7934757c09b5446':hashlib.sha256(bundle.read_bytes()).hexdigest(),
            '91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f':data_hash,
            'f6dd97c282657a386de8dbfae225b58b720a909f7b1ee98041d348f83d5ad9ac':hashlib.sha256(b'{}').hexdigest()}
        for old,new in replacements.items():text=text.replace(old,new)
        return work,files,text

    def test_cells_compile_and_scientific_cells_unchanged(self):
        self.assertEqual(len(codes),8)
        for s in codes:compile(s,'<single_zip>','exec')
        old=json.loads((H/'LED_LARGE_GRADIENT_GATE_V3_FRESH_RUNTIME.ipynb').read_text(encoding='utf-8'))
        old=[''.join(c['source']) for c in old['cells'] if c['cell_type']=='code']
        for i in (0,2,4,5,6,7):self.assertEqual(codes[i],old[i])
        self.assertEqual(codes[3],old[3].replace("patch=DRIVE/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'","patch=BUNDLE_FILES/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'"))

    def test_one_zip_restores_both_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            work,files,text=self.fixture(d);exec(text,{})
            self.assertTrue((work/'configs/stage_i_led_large_colab_v2.json').exists())
            self.assertTrue((files/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip').exists())

    def test_existing_valid_data_reused(self):
        with tempfile.TemporaryDirectory() as d:
            work,files,text=self.fixture(d);exec(text,{})
            (work/'keep.txt').write_text('keep');exec(text,{})
            self.assertEqual((work/'keep.txt').read_text(),'keep')

    def test_changed_local_patch_stops(self):
        with tempfile.TemporaryDirectory() as d:
            work,files,text=self.fixture(d);exec(text,{})
            (files/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'Missing/changed local bundle'):exec(text,{})

    def test_changed_outer_zip_stops(self):
        with tempfile.TemporaryDirectory() as d:
            work,files,text=self.fixture(d)
            (Path(d)/'drive/LED_LARGE_V3_ALL_IN_ONE.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'All-in-one ZIP hash mismatch'):exec(text,{})
            self.assertFalse(work.exists());self.assertFalse(files.exists())

    def test_approval_unchanged_not_fabricated(self):
        a=json.loads((ROOT/'outputs/stage_i/led_large_gradient_gate_v3/GPU_APPROVAL_TEMPLATE_NOT_APPROVED.json').read_text())
        self.assertFalse(a['GPU_execution_approved'])
        self.assertIn('LED_LARGE_GRADIENT_GATE_V3_APPROVAL.json',codes[6])
        self.assertNotIn('GPU_execution_approved=True','\n'.join(codes))


if __name__=='__main__':unittest.main()
