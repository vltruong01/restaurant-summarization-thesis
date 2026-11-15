"""Colab migration and checkpoint restore guards; no GPU/model execution."""
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import stage_f_colab_d033 as colab
import colab_stage_f_checkpoints as cp


class ColabTests(unittest.TestCase):
    def test_same_science_different_output_namespace(self):
        with patch.object(colab.cloud,'OUT',colab.cloud.OUT),patch.object(colab.cloud,'GEN',colab.cloud.GEN),patch.object(colab.cloud,'CONFIG',colab.cloud.CONFIG),patch.object(colab.cloud,'preflight',colab.cloud.preflight):
            colab.bind();cfg=colab.cloud.check_execution_seal()
            self.assertEqual(cfg['generation']['batch_size'],1);self.assertEqual(cfg['generation']['num_beams'],3)
            self.assertEqual(cfg['source_diagnostics']['K'],5)
            self.assertEqual(len(colab.cloud.legacy.read(colab.GEN/'pilot_photo_plan.json')),27318)
            self.assertEqual(len(colab.cloud.legacy.read(colab.OUT/'reused_caption_sources.json')),2709)

    def test_checkpoint_roundtrip_without_touching_other_namespaces(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as bank:
            root=Path(a);file=root/cp.RELATIVE/'generation_remaining/caption_checkpoints/fixture.json';file.parent.mkdir(parents=True)
            file.write_bytes(b'{"raw_caption":"fixture only"}\n');known={};cp.backup(root,bank,known)
            restored=cp.restore(b,bank);target=Path(b)/cp.RELATIVE/'generation_remaining/caption_checkpoints/fixture.json'
            self.assertEqual(target.read_bytes(),file.read_bytes());self.assertEqual(restored,known)
            target.write_bytes(b'changed')
            with self.assertRaises(AssertionError):cp.restore(b,bank)

    def test_partial_score_chunk_not_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);chunk=root/cp.RELATIVE/'scores/chunk_000';chunk.mkdir(parents=True)
            (chunk/'caption_scores.json').write_text('partial')
            self.assertEqual(list(cp.state_files(root)),[])
            (chunk/'chunk_receipt.json').write_text('{}')
            self.assertEqual(len(list(cp.state_files(root))),2)

    def test_mutable_report_archive_uses_exact_hashed_snapshot_bytes(self):
        import zipfile
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as bank:
            root=Path(a);file=root/cp.RELATIVE/'generation_remaining/generation_report.json';file.parent.mkdir(parents=True)
            file.write_bytes(b'{"completed":100}\n')
            real=zipfile.ZipFile.writestr;mutated=False
            def concurrent_update(z,name,data,*args,**kwargs):
                nonlocal mutated
                if name.endswith('generation_report.json'):
                    file.write_bytes(b'{"completed":101}\n');mutated=True
                return real(z,name,data,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'writestr',concurrent_update):cp.backup(root,bank,{})
            self.assertTrue(mutated);cp.restore(b,bank)
            self.assertEqual((Path(b)/cp.RELATIVE/'generation_remaining/generation_report.json').read_bytes(),b'{"completed":100}\n')


if __name__=='__main__':unittest.main()
