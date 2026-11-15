"""Bootstrap regression only: no installs, downloads, models or GPU execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'outputs/cloud_handoff'
nb=json.loads((H/'LED_LARGE_GRADIENT_GATE_V3_FRESH_RUNTIME.ipynb').read_text(encoding='utf-8'))
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']


class FreshRuntimeTests(unittest.TestCase):
    def fixture(self,tmp):
        d=Path(tmp);drive=d/'drive';drive.mkdir();work=d/'work';local=d/'outer'
        inner=d/'inner.zip'
        with zipfile.ZipFile(inner,'x') as z:z.writestr('configs/stage_i_led_large_colab_v2.json','{}')
        raw=inner.read_bytes();inner_hash=hashlib.sha256(raw).hexdigest()
        manifest={'files':{'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip':inner_hash}}
        outer=drive/'LED-Large_multimodal_UPLOAD_V1.zip'
        with zipfile.ZipFile(outer,'x') as z:
            z.writestr('LED-Large_multimodal/LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip',raw)
            z.writestr('LED-Large_multimodal/DELIVERY_SHA256_MANIFEST.json',json.dumps(manifest))
        text=codes[1].replace('/content/drive/MyDrive/LED-Large_multimodal',drive.as_posix())
        text=text.replace('/content/led_large_f_multimodal_v2',work.as_posix()).replace('/content/led_large_gate_v3_source',local.as_posix())
        for old,new in [('bec6ac43c99cded0aaf1081a47302b0a030b442dcd9d04ec404ad298e0b4bdae',hashlib.sha256(outer.read_bytes()).hexdigest()),
                        ('91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f',inner_hash),
                        ('f6dd97c282657a386de8dbfae225b58b720a909f7b1ee98041d348f83d5ad9ac',hashlib.sha256(b'{}').hexdigest())]:
            text=text.replace(old,new)
        return work,text

    def test_compile_and_unchanged_execution_cells(self):
        self.assertEqual(len(codes),8)
        for s in codes:compile(s,'<fresh_colab>','exec')
        old=json.loads((H/'LED_LARGE_GRADIENT_GATE_V3.ipynb').read_text(encoding='utf-8'))
        old=[''.join(c['source']) for c in old['cells'] if c['cell_type']=='code']
        self.assertEqual(codes[6],old[3]);self.assertEqual(codes[7],old[4]);self.assertEqual(codes[4],old[2])

    def test_fresh_nested_zip_restore(self):
        with tempfile.TemporaryDirectory() as d:
            work,text=self.fixture(d);ns={};exec(text,ns)
            self.assertTrue((work/'configs/stage_i_led_large_colab_v2.json').is_file())
            self.assertFalse(ns['PYTHON'].exists())

    def test_existing_workspace_retained(self):
        with tempfile.TemporaryDirectory() as d:
            work,text=self.fixture(d);ns={};exec(text,ns)
            marker=work/'keep.txt';marker.write_text('keep')
            exec(text,{})
            self.assertEqual(marker.read_text(),'keep')

    def test_changed_workspace_stops(self):
        with tempfile.TemporaryDirectory() as d:
            work,text=self.fixture(d);exec(text,{})
            (work/'configs/stage_i_led_large_colab_v2.json').write_text('changed')
            with self.assertRaisesRegex(AssertionError,'Frozen config changed'):exec(text,{})

    def test_missing_venv_bootstraps_exact_pins(self):
        with tempfile.TemporaryDirectory() as d:
            work,text=self.fixture(d);ns={};exec(text,ns);calls=[]
            def fake_run(args,**kwargs):
                calls.append(args)
                if 'venv' in args:
                    ns['PYTHON'].parent.mkdir(parents=True);ns['PYTHON'].write_text('fixture')
            with patch.object(subprocess,'run',side_effect=fake_run):exec(codes[2],ns)
            self.assertEqual(len(calls),5)
            self.assertIn('3.11',calls[1]);self.assertIn('torch==2.1.0+cu118',calls[2])
            self.assertIn('led_large_colab_core.txt',calls[3][-1]);self.assertEqual(calls[4][-1],'check')

    def test_existing_venv_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as d:
            _,text=self.fixture(d);ns={};exec(text,ns)
            ns['PYTHON'].parent.mkdir(parents=True);ns['PYTHON'].write_text('fixture')
            with patch.object(subprocess,'run') as run:exec(codes[2],ns)
            self.assertEqual(run.call_count,1);self.assertEqual(run.call_args.args[0][-1],'check')

    def test_partial_venv_preserved_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            work,text=self.fixture(d);ns={};exec(text,ns);(work/'.venv').mkdir()
            with patch.object(subprocess,'run') as run:
                with self.assertRaisesRegex(AssertionError,'Partial venv'):exec(codes[2],ns)
            run.assert_not_called()


if __name__=='__main__':unittest.main()
