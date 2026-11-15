"""Offline transfer guards: no ML runtime/model execution required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name+'.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TransferTests(unittest.TestCase):
    def test_path_traversal(self):
        v = load('verify_led_large_colab_bundle_v1')
        for path in ('../secret', '/tmp/a', 'C:/secret', 'a\\b'):
            with self.assertRaises(ValueError):
                v.safe_path(ROOT, path)

    def test_negative_checksum(self):
        v = load('download_led_large_pinned_v1')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'weight';p.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                v.verify_file(p, expected='0'*64, size=3)
            with self.assertRaises(ValueError):
                v.verify_file(p, size=4)

    def test_no_torch_in_preflight_or_downloader(self):
        import ast
        for name in ('verify_led_large_colab_bundle_v1', 'download_led_large_pinned_v1'):
            tree = ast.parse((ROOT/'scripts'/(name+'.py')).read_text(encoding='utf-8'))
            imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any(n and n.startswith(('torch', 'transformers', 'peft')) for n in imports))

    def test_config_remains_offline(self):
        c = json.loads((ROOT/'configs/stage_i_led_large_proposal_v1.json').read_text(encoding='utf-8'))
        self.assertFalse(c['training_authorized'])
        self.assertFalse(c['GPU_execution_approved'])
        self.assertEqual(c['expected_trainable_parameters'], 4915200)

    def test_optional_download_exact_revision(self):
        m = load('download_led_large_pinned_v1')
        self.assertEqual(m.REVISION, 'cd59d11c3528415b7dda4dfc95cc8f138aceda2e')
        self.assertEqual(m.SIZE, 1839633783)


if __name__ == '__main__':
    unittest.main()
