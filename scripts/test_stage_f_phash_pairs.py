"""Diagnostic guards for distance direction, denominators and exact case partitions."""
import unittest
from analyze_stage_f_phash_pairs import metrics,point_ids


class PHashDiagnosticTests(unittest.TestCase):
    def test_duplicate_means_distance_at_or_below_cutoff(self):
        r=metrics([2,10,5,16],[True,True,False,False],5)
        self.assertEqual([r[k] for k in ['TP','TN','FP','FN']],[1,1,1,1])
        self.assertEqual(r['same_visual_content_recall'],.5)
        self.assertEqual(r['distinct_content_specificity'],.5)
        self.assertEqual(r['balanced_accuracy'],.5)

    def test_class_denominators_not_total_pairs(self):
        r=metrics([2,20,25,30,40],[True,False,False,False,False],20)
        self.assertEqual(r['distinct_pairs_marked_duplicate_pct'],25)
        self.assertEqual(r['positive_pairs_missed_pct'],0)
        self.assertEqual(r['precision'],.5)

    def test_extreme_cutoffs_and_no_division_by_zero(self):
        lower=metrics([22,30],[True,False],0);upper=metrics([22,30],[True,False],64)
        self.assertEqual(lower['precision'],0)
        self.assertEqual(lower['balanced_accuracy'],.5)
        self.assertEqual(upper['balanced_accuracy'],.5)
        with self.assertRaises(AssertionError):metrics([22],[True],12)

    def test_ids_partition_and_odd_cutoff_identical_decisions(self):
        rows=[{'case_id':str(i),'phash_hamming':v,'positive':i<2} for i,v in enumerate([22,34,20,40])]
        a=point_ids(rows,34);b=point_ids(rows,35)
        self.assertEqual(a,b)
        self.assertEqual(a['TP'],['0','1']);self.assertEqual(a['FP'],['2']);self.assertEqual(a['TN'],['3'])
        self.assertEqual(set(sum(a.values(),[])),{'0','1','2','3'})


if __name__=='__main__':unittest.main()
