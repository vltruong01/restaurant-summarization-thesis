"""Meaningful edge cases for quota, ordering, tokens and budget feasibility."""
import copy
from pathlib import Path
import sys
import unittest
import yaml
from tokenizers import Tokenizer
from review_selection_pilot import hamilton, jsd, select_initial, fixed_time_bins, formatted_input, adjust_budget, token_count

ROOT = Path(__file__).resolve().parents[1]


def reviews(n=45):
    return [{'business_id':'example','review_id':f'r{i:03}', 'stars':1+i%5,
             'date':f'2020-01-{1+i%28:02} 00:00:00', 'text':('Complete original review with food and service opinions. '* (1+i%4))} for i in range(n)]


class SelectionEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ROOT/'configs/review_selection_pilot.yaml').read_text(encoding='utf-8'))
        cls.tokenizer = Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.tokenizer.no_truncation(); cls.tokenizer.no_padding()

    def test_hamilton_rare_ratings_ties_and_complete_population(self):
        self.assertEqual(hamilton({1:1,5:999},30),{1:0,2:0,3:0,4:0,5:30})
        self.assertEqual(hamilton({1:1,2:1,3:1},2),{1:1,2:1,3:0,4:0,5:0})
        self.assertEqual(hamilton({2:2,4:7},9),{1:0,2:2,3:0,4:7,5:0})
        with self.assertRaises(ValueError): hamilton({1:2},3)

    def test_jsd_known_values_symmetry_and_zeros(self):
        self.assertEqual(jsd([1,0],[0,1]),1.0)
        self.assertEqual(jsd([1,0,0],[1,0,0]),0.0)
        self.assertAlmostEqual(jsd([1,3],[2,5]),jsd([2,5],[1,3]))
        with self.assertRaises(ValueError): jsd([0,0],[1,0])

    def test_all_reviews_kept_and_same_date_ordering(self):
        pool = reviews(10)
        for r in pool: r['date']='2020-01-01 00:00:00'
        for method in ('random','rating_time'):
            selected = select_initial(list(reversed(pool)),method,self.cfg,'example')
            self.assertEqual([r['review_id'] for r in selected],[r['review_id'] for r in pool])
        mapping,counts = fixed_time_bins(pool,5)
        self.assertEqual(counts,[2,2,2,2,2]); self.assertEqual(len(mapping),10)

    def test_thirty_selection_is_order_invariant_and_business_independent(self):
        pool = reviews(75)
        for method in ('random','rating_time'):
            a = select_initial(pool,method,self.cfg,'example')
            select_initial(reviews(90),method,self.cfg,'another_business')
            b = select_initial(list(reversed(pool)),method,self.cfg,'example')
            self.assertEqual(a,b); self.assertEqual(len(a),30)
            self.assertEqual(len({r['review_id'] for r in a}),30)

    def test_exact_markers_special_tokens_and_no_text_truncation(self):
        pool = reviews(2)
        expected = '\n\n'.join(f'[REV_{i}] stars={int(r["stars"])} date={r["date"]}\n{r["text"]}' for i,r in enumerate(pool,1))
        self.assertEqual(formatted_input(pool,self.cfg),expected)
        self.assertEqual(token_count(pool,self.cfg,self.tokenizer),len(self.tokenizer.encode(expected,add_special_tokens=False).ids)+2)

    def test_removal_respects_explicit_budget_and_never_empty(self):
        pool = reviews(45); original = copy.deepcopy(pool)
        initial = select_initial(pool,'rating_time',self.cfg,'example')
        budget = token_count(initial,self.cfg,self.tokenizer)//2
        selected,trace,ok = adjust_budget(pool,initial,budget,self.cfg,self.tokenizer)
        self.assertTrue(ok); self.assertLessEqual(token_count(selected,self.cfg,self.tokenizer),budget)
        self.assertEqual(len(selected)+len(trace),len(initial)); self.assertEqual(pool,original)
        self.assertTrue(all(r['text'] in [v['text'] for v in original] for r in selected))
        selected,trace,ok = adjust_budget(pool,initial,1,self.cfg,self.tokenizer)
        self.assertFalse(ok); self.assertEqual(len(selected),1)


if __name__=='__main__': unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
