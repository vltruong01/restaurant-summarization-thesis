"""Focused Stage M.1 adapters; no model, TEST refs, or real human work."""
import unittest
from stage_m1_existing_teacher_metrics_v1 import target_parts,joint_parts
from s1000_final_eval_core_v1 import paired

class Adapters(unittest.TestCase):
    def test_common_review_and_different_visual_are_not_conflated(self):
        r='[REVIEW_CONSENSUS]\nSoup praised.'
        u=r+'\n\n[VISUAL_EVIDENCE]\nA red bowl.';f=r+'\n\n[VISUAL_EVIDENCE]\nA table.'
        self.assertEqual(target_parts(r,'reviews_only')[0],target_parts(u,'unfiltered')[0]);self.assertEqual(target_parts(u,'unfiltered')[0],target_parts(f,'filtered')[0])
        self.assertNotEqual(target_parts(u,'unfiltered')[1],target_parts(f,'filtered')[1])
    def test_empty_visual_retained_and_RO_visual_not_applicable(self):
        r='[REVIEW_CONSENSUS]\nSoup.'
        self.assertEqual(target_parts(r,'reviews_only'),('Soup.',''))
        self.assertEqual(target_parts(r+'\n\n[VISUAL_EVIDENCE]\n','filtered'),('Soup.',''))
    def test_invalid_target_headers_rejected(self):
        for value,condition in [('Soup.','filtered'),('[REVIEW_CONSENSUS]\nSoup.','filtered'),('[REVIEW_CONSENSUS]\nSoup.\n\n[VISUAL_EVIDENCE]\nTable.','reviews_only')]:
            with self.assertRaises(ValueError):target_parts(value,condition)
    def test_joint_component_views_are_exact_existing_policy(self):
        r,v,flag=joint_parts('[REVIEW_CONSENSUS]\nSoup good.\n\n[VISUAL_EVIDENCE]\nA table.')
        self.assertEqual((r,v,flag),('Soup good.','A table.',None))
    def test_malformed_joint_not_repaired_or_excluded(self):
        text='[REVIEW_CONSENSUS]\nSoup good.'
        r,v,flag=joint_parts(text);self.assertEqual(r,text);self.assertEqual(v,'');self.assertIsNotNone(flag)
        text='[REVIEW_CONSENSUS]x[VISUAL_EVIDENCE]y[VISUAL_EVIDENCE]z';self.assertEqual(joint_parts(text)[0],text)
    def test_same_reference_paired_CI_fixed_seed(self):
        a=paired([.9,.8,.7],[.8,.7,.6],B=500,P=1)
        self.assertEqual(a,paired([.9,.8,.7],[.8,.7,.6],B=500,P=1));self.assertAlmostEqual(a['mean_difference'],.1)
        self.assertAlmostEqual(a['CI95'][0],.1);self.assertAlmostEqual(a['CI95'][1],.1)

if __name__=='__main__':unittest.main(verbosity=2)
