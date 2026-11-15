"""Supplemental synthetic portability tests; separate from the frozen83 tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import check_stage_i_d053_bundle_readiness_v2 as checker


class PortableReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'ordinary.txt').write_bytes(b'synthetic')

    def tearDown(self):
        self.temp.cleanup()

    def test_included_file_hash_is_verified(self):
        digest = hashlib.sha256(b'synthetic').hexdigest()
        self.assertEqual(checker.verified_file(self.root, 'ordinary.txt', digest), (self.root / 'ordinary.txt').resolve())

    def test_modified_included_file_is_rejected(self):
        with self.assertRaises(ValueError):
            checker.verified_file(self.root, 'ordinary.txt', '0' * 64)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(ValueError):
            checker.safe_file(self.root, 'missing.txt')

    def test_parent_traversal_is_rejected(self):
        for relative in ('../outside', r'..\outside', 'x/../../outside'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                checker.safe_file(self.root, relative)

    def test_absolute_paths_are_rejected_cross_platform(self):
        for relative in ('/outside', r'C:\outside\file', r'\\server\share\file'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                checker.safe_file(self.root, relative)

    def test_missing_portable_map_does_not_fall_back_to_project(self):
        with self.assertRaises(ValueError):
            checker.check(self.root)

    def test_changed_frozen_config_metadata_is_rejected(self):
        (self.root / 'PORTABLE_BINDING_MAP.json').write_text(json.dumps({'frozen_config_sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'Frozen configuration hash changed'):
            checker.check(self.root)

    def test_only_two_explicit_independence_exclusions_are_allowed(self):
        self.assertEqual(checker.WITHHELD, {'case_inputs', 'blinding_key'})
        self.assertEqual(checker.ARMS, {'A_decoder_only', 'B_encoder_decoder'})


if __name__ == '__main__':
    unittest.main()
