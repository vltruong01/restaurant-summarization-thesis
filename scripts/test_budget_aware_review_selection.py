"""Long-review/cardinality/regression checks with actual pinned LED tokenization."""
import copy
from pathlib import Path
import sys
import unittest
import yaml
from tokenizers import Tokenizer
from review_selection_pilot import select_initial,adjust_budget,token_count
from budget_aware_review_selection import search_population,choose_candidate,nondominated

ROOT=Path(__file__).resolve().parents[1]


def pool(n,text_function):
    return [{'business_id':'unit','review_id':f'r{i:03}','stars':5,'date':f'2020-01-{1+i%28:02} 00:00:00','text':text_function(i)} for i in range(n)]


class BudgetAwareEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'configs/review_selection_budget_pilot.yaml').read_text(encoding='utf-8'))
        cls.tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.tokenizer.no_truncation(); cls.tokenizer.no_padding()

    def settings(self,budget):
        cfg=copy.deepcopy(self.cfg); cfg['budget']['exploratory_review_token_scenarios']=[budget]
        cfg['search']['local_iterations']=2
        return cfg

    def test_count_preference_requires_comparable_both_metrics(self):
        a={'ids':('a',),'rating_jsd':.1,'temporal_jsd':.1,'count':8,'weighted_cost':.1,'tokens':2000}
        b={'ids':('b',),'rating_jsd':.1005,'temporal_jsd':.1005,'count':12,'weighted_cost':.1005,'tokens':2100}
        self.assertEqual(choose_candidate([a,b],self.cfg)['count'],12)
        b.update(rating_jsd=.11,temporal_jsd=.09,weighted_cost=.1)
        self.assertEqual(choose_candidate([a,b],self.cfg)['count'],8)
        b.update(rating_jsd=.09,temporal_jsd=.09,weighted_cost=.09)
        self.assertEqual(nondominated([a,b]),[b])

    def test_eight_whole_long_reviews_overflow_but_business_not_dropped(self):
        population=pool(10,lambda i:'Delicious food, attentive service and fair prices. '*150)
        self.assertGreater(token_count(population[:8],self.cfg,self.tokenizer),5000)
        cfg=self.settings(3500)
        ev,chosen,frontiers,sensitivity,stats=search_population(population,cfg,self.tokenizer,'unit')
        answer=chosen[3500]
        self.assertIsNotNone(answer); self.assertLess(answer['count'],5)
        self.assertLessEqual(answer['tokens'],3500)
        self.assertTrue(all(ev.by_id[rid]['text']==population[int(rid[1:])]['text'] for rid in answer['ids']))

    def test_full_population_supplies_reviews_outside_initial_seed(self):
        population=pool(60,lambda i:('Good food and service. '*100 if i%2==0 else 'Good food and service.'))
        cfg=self.settings(1400)
        original=select_initial(population,'rating_time',cfg,'unit')
        deletion,_,ok=adjust_budget(population,original,1400,cfg,self.tokenizer)
        ev,chosen,frontiers,_,_=search_population(population,cfg,self.tokenizer,'unit',[('delete_only',deletion)] if ok else [])
        answer=chosen[1400]
        self.assertGreater(answer['count'],len(deletion)); self.assertEqual(answer['count'],30)
        self.assertTrue(set(answer['ids'])-set(r['review_id'] for r in original))

    def test_determinism_date_ties_and_population_order(self):
        population=pool(12,lambda i:'Complete review text. '* (1+i%3))
        for r in population:r['date']='2020-01-01 00:00:00'
        cfg=self.settings(500)
        a=search_population(population,cfg,self.tokenizer,'unit')[1][500]
        b=search_population(list(reversed(population)),cfg,self.tokenizer,'unit')[1][500]
        self.assertEqual(a['ids'],b['ids']); self.assertEqual(a['tokens'],b['tokens'])

    def test_no_whole_review_fits_reports_infeasible_not_empty_valid_input(self):
        population=pool(10,lambda i:'Complete long review text. '*30)
        cfg=self.settings(1)
        ev,chosen,frontiers,sensitivity,_=search_population(population,cfg,self.tokenizer,'unit')
        self.assertIsNone(chosen[1]); self.assertEqual(frontiers[1],[])
        self.assertTrue(all(row['status']=='budget_infeasible' and row['count']==0 for row in sensitivity))

    def test_only_rare_rating_review_fits_is_not_reported_infeasible(self):
        population=pool(40,lambda i:'Complete long review text. '*200)
        population[-1].update(stars=1,text='Poor service.')
        cfg=self.settings(100)
        ev,chosen,_,_,_=search_population(population,cfg,self.tokenizer,'unit')
        answer=chosen[100]
        self.assertIsNotNone(answer)
        self.assertEqual(answer['ids'],('r039',))
        self.assertLessEqual(answer['tokens'],100)


if __name__=='__main__': unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
