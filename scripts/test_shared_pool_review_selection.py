"""Meaningful v4 constraints and token-neutral selection regressions."""
import copy, unittest
from pathlib import Path
import yaml
from tokenizers import Tokenizer
from budget_aware_review_selection_v4_1 import choose_candidate, search_population
from review_selection_pilot import formatted_input

ROOT=Path(__file__).resolve().parents[1]


class V41Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'configs/review_selection_shared_pool_pilot.yaml').read_text())
        cls.tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.tokenizer.no_truncation();cls.tokenizer.no_padding()

    def pool(self,count=12):
        return [{'review_id':f'r{i:03d}','business_id':'synthetic','stars':i%5+1,
                 'date':f'2020-01-{i%28+1:02d} 12:00:00','text':('Food service texture price. '*(i%6+1))}
                for i in range(count)]

    def test_tie_priority_does_not_depend_on_tokens(self):
        a={'ids':('a',),'count':1,'rating_jsd':.01,'temporal_jsd':.02,'weighted_cost':.015,'tokens':30}
        b={**a,'ids':('b',),'tokens':400}
        original=choose_candidate([a,b],self.cfg,'tie')['ids']
        a['tokens']=400;b['tokens']=30
        self.assertEqual(original,choose_candidate([b,a],self.cfg,'tie')['ids'])

    def test_source_order_determinism_and_original_boundaries(self):
        pool=self.pool(36);cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[1200]
        first=search_population(pool,cfg,self.tokenizer,'synthetic')
        reverse=search_population(list(reversed(pool)),cfg,self.tokenizer,'synthetic')
        self.assertEqual(first[1:],reverse[1:])
        selected=first[1][1200];self.assertIsNotNone(selected)
        text=formatted_input([first[0].by_id[rid] for rid in selected['ids']],cfg)
        self.assertLessEqual(len(self.tokenizer.encode(text).ids),1200)
        for rid in selected['ids']:self.assertIn(first[0].by_id[rid]['text'],text)

    def test_all_source_reviews_retained_when_small_pool_fits(self):
        pool=self.pool();cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[3500]
        _,selected,_,_,stats=search_population(pool,cfg,self.tokenizer,'synthetic')
        self.assertEqual(selected[3500]['count'],12)
        self.assertFalse(stats[3500]['rescue_used'])

    def test_rare_only_fit_singleton_rescue_and_fewer_than_ten_valid(self):
        pool=self.pool(12)
        for r in pool:r['stars']=5;r['text']='Lengthy repeated account. '*300
        pool[-1]['stars']=1;pool[-1]['text']='Coffee was cold.'
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[60]
        _,selected,_,_,stats=search_population(pool,cfg,self.tokenizer,'synthetic')
        self.assertEqual(selected[60]['ids'],('r011',));self.assertEqual(selected[60]['count'],1)
        self.assertTrue(stats[60]['rescue_used'])

    def test_truly_infeasible_has_no_empty_valid_input(self):
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[5]
        _,selected,frontier,sensitivity,_=search_population(self.pool(),cfg,self.tokenizer,'synthetic')
        self.assertIsNone(selected[5]);self.assertEqual(frontier[5],[])
        self.assertTrue(all(r['status']=='budget_infeasible' for r in sensitivity))

    def test_uniform_and_length_strata_candidates_are_not_shortest_defaults(self):
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[3500]
        ev,_,_,_,_=search_population(self.pool(36),cfg,self.tokenizer,'synthetic')
        variants={c['origin'].get('variant') for c in ev.audit if c['origin']['type']=='seed'}
        self.assertNotIn('rating_time_low_cost',variants);self.assertNotIn('joint_distribution_low_cost',variants)
        self.assertIn('rating_time_rotating_length_strata',variants)
        offers=ev.incoming([],0)
        self.assertTrue(any(ev.length_cost[rid]>min(ev.length_cost[x] for x in ev.by_id) for rid in offers))


    def test_budget_order_does_not_change_shared_discovery(self):
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[700,1100,1500]
        first=search_population(self.pool(36),cfg,self.tokenizer,'order')
        cfg['budget']['exploratory_review_token_scenarios']=[1500,700,1100]
        other=search_population(self.pool(36),cfg,self.tokenizer,'order')
        self.assertEqual(first[1:],other[1:])
        self.assertEqual(first[0].shared_candidates,other[0].shared_candidates)

    def test_every_feasible_incumbent_remains_in_shared_pool(self):
        pool=self.pool(36);cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[700,1100]
        extra=[('v3_incumbent',pool[:10])]
        ev,choices,frontiers,_,_=search_population(pool,cfg,self.tokenizer,'incumbent',extra)
        ids=ev.canonical([r['review_id'] for r in pool[:10]])
        self.assertIn(ids,[c['ids'] for c in ev.shared_candidates])
        for b,c in choices.items():
            incumbent=next(v for v in ev.shared_candidates if v['ids']==ids)
            if incumbent['tokens']<=b:
                dominates=(incumbent['rating_jsd']<=c['rating_jsd']+1e-12 and incumbent['temporal_jsd']<=c['temporal_jsd']+1e-12 and incumbent['count']>=c['count'] and (incumbent['rating_jsd']<c['rating_jsd']-1e-12 or incumbent['temporal_jsd']<c['temporal_jsd']-1e-12 or incumbent['count']>c['count']))
                self.assertFalse(dominates)

    def test_lower_budget_candidate_is_considered_at_higher_budget(self):
        from unittest.mock import patch
        import budget_aware_review_selection_v4_1 as module
        def c(ids,count,tokens,rating):
            return {'ids':tuple(ids),'count':count,'tokens':tokens,'rating_jsd':rating,'temporal_jsd':0.,'weighted_cost':rating/2,'origin':{'type':'seed'},'path':[]}
        seed=c(['a'],1,20,.1);low=c(['a','b','c'],3,80,.001);high=c(['a','b'],2,90,.0011)
        class FakeEvaluator:
            def __init__(self,*args):pass
            def seed_candidates(self):return [seed],seed
            def improve(self,anchor,budget):return ([low] if budget==100 else [high]),[]
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[100,200]
        with patch.object(module,'Evaluator',FakeEvaluator):
            ev,choices,_,_,_=search_population([],cfg,None,'coverage')
        self.assertEqual(choices[100]['ids'],low['ids'])
        self.assertEqual(choices[200]['ids'],low['ids'])
        self.assertIn(low['ids'],[x['ids'] for x in ev.shared_candidates])


if __name__=='__main__':unittest.main(verbosity=2)

