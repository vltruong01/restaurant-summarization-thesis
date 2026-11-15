import copy
import unittest
from stage_g_visual_target_serialization import visual_claims


class TargetSerialization(unittest.TestCase):
    def assessment(self, caption, photo, text='The photo shows a restaurant.', verdict='verified'):
        return {'caption_id': caption, 'photo_id': photo, 'claims': [
            {'text': text, 'caption_quote': 'a restaurant', 'verdict': verdict}]}

    def test_two_photos_one_target_statement_all_evidence_preserved(self):
        original = [self.assessment('IMG_CAP_1', 'p1'), self.assessment('IMG_CAP_2', 'p2')]
        before = copy.deepcopy(original)
        result = visual_claims(original)
        self.assertEqual(len(result), 1)
        self.assertEqual([x['photo_id'] for x in result[0]['support']], ['p1', 'p2'])
        self.assertEqual(original, before)

    def test_first_text_preserved_when_case_or_spacing_varies(self):
        result = visual_claims([self.assessment('A', 'p1'),
            self.assessment('B', 'p2', 'the  photo shows a restaurant.')])
        self.assertEqual(result[0]['text'], 'The photo shows a restaurant.')
        self.assertEqual(len(result[0]['support']), 2)

    def test_rejected_or_uncertain_claims_not_promoted(self):
        result = visual_claims([self.assessment('A', 'p1', verdict='rejected'),
            self.assessment('B', 'p2', verdict='uncertain')])
        self.assertEqual(result, [])

    def test_repetition_within_one_caption_still_rejected(self):
        a = self.assessment('A', 'p1');a['claims'].append(copy.deepcopy(a['claims'][0]))
        with self.assertRaises(ValueError): visual_claims([a])

    def test_similar_but_different_claims_are_not_semantically_filtered(self):
        result = visual_claims([self.assessment('A', 'p1', 'A sandwich is shown.'),
            self.assessment('B', 'p2', 'A sandwich with meat is shown.')])
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
