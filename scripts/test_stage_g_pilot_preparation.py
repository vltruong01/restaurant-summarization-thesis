"""Offline failure checks for evidence extraction and immutable pilot bindings."""
import tempfile
import unittest
from pathlib import Path
from prepare_stage_g_teacher_pilot import split_reviews, digest, immutable, local_path


class PilotInputSafety(unittest.TestCase):
    def record(self):
        texts = ['First complete review.\nSecond line.', 'Second complete review.']
        block = '\n\n'.join(f'[REV_{i}] stars=4 date=2020-01-01 00:00:00\n{text}'
                             for i, text in enumerate(texts, 1))
        return {'review_block': block, 'review_block_sha256': digest(block.encode()),
                'selected_review_ids': ['review-A', 'review-B'],
                'review_text_sha256_in_order': [digest(t.encode()) for t in texts]}

    def test_complete_multiline_review_and_original_order(self):
        result = split_reviews(self.record())
        self.assertEqual([r['review_id'] for r in result], ['review-A', 'review-B'])
        self.assertEqual(result[0]['text'], 'First complete review.\nSecond line.')
        self.assertEqual(result[1]['text'], 'Second complete review.')

    def test_changed_text_rejected(self):
        record = self.record()
        record['review_block'] = record['review_block'].replace('complete', 'edited', 1)
        with self.assertRaises(ValueError):
            split_reviews(record)

    def test_missing_review_id_rejected(self):
        record = self.record()
        record['selected_review_ids'].pop()
        with self.assertRaises(ValueError):
            split_reviews(record)

    def test_out_of_order_markers_rejected(self):
        record = self.record()
        record['review_block'] = record['review_block'].replace('[REV_1]', '[REV_3]')
        with self.assertRaises(ValueError):
            split_reviews(record)

    def test_changed_checkpoint_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.json'
            immutable(path, {'model': 'original'})
            immutable(path, {'model': 'original'})
            with self.assertRaises(ValueError):
                immutable(path, {'model': 'changed'})

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            local_path('../outside-project.json')


if __name__ == '__main__':
    unittest.main()
