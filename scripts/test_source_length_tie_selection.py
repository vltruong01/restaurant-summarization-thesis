"""Secondary distribution objectives and hard primary/whole-budget invariants."""
import copy,unittest
from pathlib import Path
import yaml
from tokenizers import Tokenizer
from budget_aware_review_selection_v4_2 import length_profile,length_score,secondary_choice,signature,search_population
from budget_aware_review_selection_v4_1 import discovery_config
from review_selection_pilot import formatted_input

ROOT=Path(__file__).resolve().parents[1]


class LengthTieChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'configs/review_selection_source_length_tie_pilot.yaml').read_text(encoding='utf-8'))
        cls.tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));cls.tok.no_truncation();cls.tok.no_padding()

    def population(self,n=36):
        return [{'review_id':f'r{i:03d}','business_id':'synthetic','stars':i%5+1,'date':f'2020-01-{i%28+1:02d} 12:00:00','text':'Food service texture price. '*(1+i%9)} for i in range(n)]

    def test_balanced_distribution_beats_shortest_and_longest(self):
        lengths={f'r{i}':10*(i+1) for i in range(16)};p=length_profile(lengths)
        sets=[{'ids':tuple(f'r{i}' for i in indices)} for indices in [[0,1,2,3],[12,13,14,15],[1,5,9,13]]]
        chosen=secondary_choice(sets,p,discovery_config(self.cfg),'synthetic')
        self.assertEqual(chosen,sets[2]);self.assertEqual(p[2],[3,4,4,5])
        self.assertLess(length_score(chosen['ids'],p),min(length_score(c['ids'],p) for c in sets[:2]))

    def test_short_source_and_equal_lengths_are_valid(self):
        p=length_profile({f'r{i}':5 for i in range(12)})
        self.assertEqual(p[2],[0,0,0,12]);self.assertEqual(length_score(['r0','r1'],p),0)

    def test_equal_lengths_not_split_by_id(self):
        p=length_profile({'a':10,'b':10,'c':10,'d':20,'e':20})
        self.assertEqual(p[1]['a'],p[1]['c']);self.assertEqual(p[1]['d'],p[1]['e'])

    def test_primary_invariants_and_complete_budget(self):
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[700,1100]
        pop=self.population();result=search_population(pop,cfg,self.tok,'synthetic')
        byid={r['review_id']:r for r in pop}
        for b,methods in result['choices'].items():
            control=methods['v4_2_primary_control']
            for c in methods.values():
                self.assertEqual(c['count'],control['count']);self.assertEqual(c['rating_jsd'],control['rating_jsd']);self.assertEqual(c['temporal_jsd'],control['temporal_jsd'])
                text=formatted_input([byid[rid] for rid in c['ids']],cfg)
                self.assertLessEqual(len(self.tok.encode(text).ids),b)
                for rid in c['ids']:self.assertIn(byid[rid]['text'],text)
            profile=length_profile(result['body_lengths'])
            self.assertLessEqual(length_score(methods['v4_2_length_balanced']['ids'],profile),length_score(control['ids'],profile)+1e-12)

    def test_source_budget_and_candidate_order_determinism(self):
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[700,1100]
        pop=self.population();a=search_population(pop,cfg,self.tok,'synthetic')
        cfg['budget']['exploratory_review_token_scenarios']=[1100,700]
        b=search_population(list(reversed(pop)),cfg,self.tok,'synthetic')
        self.assertEqual(a,b)

    def test_small_source_all_retained(self):
        result=search_population(self.population(12),self.cfg,self.tok,'small')
        self.assertTrue(all(c['count']==12 for methods in result['choices'].values() for c in methods.values()))

    def test_less_than_ten_and_infeasible_are_reportable(self):
        pop=self.population(12)
        for r in pop:r['text']='Long account of the restaurant. '*300;r['stars']=5
        pop[-1]['text']='Coffee cold.';pop[-1]['stars']=1
        cfg=copy.deepcopy(self.cfg);cfg['budget']['exploratory_review_token_scenarios']=[60,5]
        result=search_population(pop,cfg,self.tok,'rescue')
        self.assertTrue(all(c['ids']==('r011',) for c in result['choices'][60].values()))
        self.assertTrue(all(c is None for c in result['choices'][5].values()))


if __name__=='__main__':unittest.main()
