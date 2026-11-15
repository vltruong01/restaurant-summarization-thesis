"""Bitwise pure-distance reuse, context safety and restoration checks."""
import unittest,copy
from pathlib import Path
import yaml
from tokenizers import Tokenizer
import budget_aware_review_selection as base
import budget_aware_review_selection_v4_2 as secondary
from review_scoring_memoization import memoized_review_scoring

ROOT=Path(__file__).resolve().parents[1]


class MemoizationChecks(unittest.TestCase):
    def test_repeated_distances_are_bitwise_original(self):
        original=base.jsd
        pairs=[([40,30,20,10],[i+1,3,2,1]) for i in range(30)]
        values=[original(a,b).hex() for a,b in pairs]
        with memoized_review_scoring():
            for _ in range(3):self.assertEqual([base.jsd(a,b).hex() for a,b in pairs],values)
            self.assertEqual([secondary.jsd(a,b).hex() for a,b in pairs],values)

    def test_error_restores_original_functions(self):
        init=base.Evaluator.__init__;metric=base.jsd;secondary_metric=secondary.jsd
        with self.assertRaises(RuntimeError):
            with memoized_review_scoring():raise RuntimeError('synthetic')
        self.assertIs(base.Evaluator.__init__,init);self.assertIs(base.jsd,metric);self.assertIs(secondary.jsd,secondary_metric)

    def test_cross_source_or_rule_reuse_is_rejected(self):
        cfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_tie_pilot.yaml').read_text(encoding='utf-8'))
        tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        pop=[{'review_id':str(i),'stars':i%5+1,'date':f'2020-01-{i+1:02d} 00:00:00','text':'Nice food and service.'} for i in range(10)]
        with memoized_review_scoring():
            base.Evaluator(pop,cfg,tok,'test');other=copy.deepcopy(cfg);other['search']['rating_weight']=.4
            with self.assertRaises(AssertionError):base.Evaluator(pop,other,tok,'test')


if __name__=='__main__':unittest.main()
