"""Meaningful guards for no chaining, broader coverage, diversity and missing human labels."""
import unittest
import numpy as np
from stage_f_math import complete_link_groups, redundancy_reduce, farthest_first,threshold_search,image_duplicate_reduce


class StageFTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'photo_id':p,'clip_score':s,'relevance_margin':.2} for p,s in [('a',.4),('b',.3),('c',.2)]]

    def test_complete_link_does_not_merge_transitive_chain(self):
        m=np.array([[1,.9,.6],[.9,1,.9],[.6,.9,1]])
        g=complete_link_groups(self.rows,m,.8)
        self.assertNotEqual(g[0],g[2]);self.assertEqual(len(set(g.values())),2)

    def test_coarser_groups_can_contain_nonduplicate_representatives(self):
        m=np.array([[1,.7,.1],[.7,1,.1],[.1,.1,1]])
        kept,_=redundancy_reduce(self.rows,m,.9);g=complete_link_groups(self.rows,m,.6)
        self.assertEqual(len(kept),3);self.assertEqual(g[0],g[1]);self.assertNotEqual(g[0],g[2])
        order=farthest_first(self.rows,m,kept)
        self.assertEqual(order,[0,2,1]);self.assertEqual(len({g[i] for i in order[:2]}),2)

    def test_duplicate_mapping_and_deterministic_quality_ties(self):
        m=np.array([[1,.95,.2],[.95,1,.3],[.2,.3,1]])
        kept,mapping=redundancy_reduce(self.rows,m,.9)
        self.assertEqual(kept,[0,2]);self.assertEqual(mapping,{1:0})
        self.assertEqual(farthest_first(self.rows,m,kept),[0,2])

    def test_missing_human_classes_never_freeze(self):
        with self.assertRaises(ValueError):threshold_search([.2,.3],[True,True],[.2,.3])

    def test_image_greedy_does_not_delete_transitive_chain(self):
        kept,mapping=image_duplicate_reduce(self.rows,['x','y','z'],[0,1,3],1)
        self.assertEqual(kept,[0,2]);self.assertEqual(mapping,{1:0})

    def test_balanced_accuracy_search_not_arbitrary_cutoff(self):
        selected,_=threshold_search([.1,.2,.8,.9],[False,False,True,True],[.1,.5,.8,.9],minimum_class=2)
        self.assertEqual(selected['threshold'],.8);self.assertEqual(selected['balanced_accuracy'],1)


if __name__=='__main__':unittest.main()
