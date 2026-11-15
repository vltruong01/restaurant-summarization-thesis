"""Diagnostic math/selection guards; synthetic fixtures are not research labels."""
import unittest
from analyze_stage_f_conservative_consistency_gate import table_for, candidates, decisions, compute, GRID
from stage_f_math import threshold_search


class ConservativeGateTests(unittest.TestCase):
    def test_confusion_counts_rates_percent_denominators_and_rounding(self):
        row=table_for([.2199996,.2199994,.25,.20],[True,True,False,False],[.22])[0]
        self.assertEqual([row[k] for k in ['TP','TN','FP','FN']],[1,1,1,1])
        self.assertEqual(row['consistent_recall'],.5)
        self.assertEqual(row['inconsistent_recall_specificity'],.5)
        self.assertEqual(row['consistent_rejected_pct'],50)
        self.assertEqual(row['inconsistent_accepted_pct'],50)

    def test_BA_tie_rule_exactly_matches_canonical_math(self):
        values=[.215,.238,.25,.26,.30];labels=[True,True,True,False,False]
        chosen=candidates(table_for(values,labels))['maximum_balanced_accuracy']
        expected,_=threshold_search(values,labels,GRID,minimum_class=1)
        self.assertEqual(chosen['threshold'],expected['threshold'])
        self.assertEqual(chosen['balanced_accuracy'],expected['balanced_accuracy'])

    def test_highest_recall_feasible_not_best_precision_or_first_feasible(self):
        scores=[.22]*19+[.235]+[.25];labels=[True]*20+[False]
        selected=candidates(table_for(scores,labels))
        self.assertEqual(selected['highest_recall_at_least_95pct']['threshold'],.22)
        # .23 passes only 1/20 positives; it cannot be selected for 95%.
        self.assertLess(selected['highest_recall_at_least_95pct']['threshold'],.23)

    def test_no_in_grid_feasible_point_no_silent_fallback(self):
        selected=candidates(table_for([.1,.3],[True,False]))
        self.assertIsNone(selected['highest_recall_at_least_95pct'])
        self.assertIsNone(selected['highest_recall_at_least_90pct'])

    def test_accepted_and_rejected_case_sets_are_exact_partition(self):
        rows=[{'case_id':str(i),'clip_score':v,'positive':i<2} for i,v in enumerate([.21,.24,.26,.20])]
        d=decisions(rows,.22)
        self.assertEqual(d['rejected_human_consistent_case_ids'],['0'])
        self.assertEqual(d['accepted_human_inconsistent_case_ids'],['2'])
        self.assertEqual(set(d['accepted_case_ids'])|set(d['rejected_case_ids']),{'0','1','2','3'})
        self.assertFalse(set(d['accepted_case_ids'])&set(d['rejected_case_ids']))

    def test_bootstrap_reproducible_preserves_classes_and_fixed_refit_separation(self):
        rows=[{'case_id':str(i),'clip_score':v,'positive':i<3} for i,v in enumerate([.22,.24,.30,.23,.29])]
        a=compute(rows,replicates=10,seed=42)
        self.assertEqual(a,compute(rows,replicates=10,seed=42))
        for draws in a[4]['fixed_points'].values():
            for draw in draws:
                self.assertEqual(draw['metrics']['consistent_total'],3)
                self.assertEqual(draw['metrics']['inconsistent_total'],2)
        for fold in a[3]:
            self.assertEqual(fold['remaining_positive'],3)
            self.assertEqual(fold['remaining_negative'],1)
            for rule in ['highest_recall_at_least_95pct','highest_recall_at_least_90pct']:
                self.assertEqual(fold['rules'][rule]['candidate']['threshold'],a[1][rule]['threshold'])


if __name__=='__main__':unittest.main()
