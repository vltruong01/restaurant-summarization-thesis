"""Focused source-only assignment/statistical regression tests; no human data."""
import ast
import copy
import unittest
from types import SimpleNamespace
from stage_m2_validation_shuffled_v1 import adapted_mapping_function, permutation_cycles, cycle_bootstrap
from s1000_final_eval_core_v1 import text_hash, read, digest
from pathlib import Path


class TinyTokenizer:
    def encode(self,text):
        return SimpleNamespace(ids=[0]+list(range(3,3+len(text.split())))+[2])


def fixture(b,count,category='Pizza'):
    source=f'[BUSINESS_INFORMATION]\nname={b}\ncategories={category}, Restaurants\n\n[REVIEWS]\n[REV_1] Good food.\n\n[VISUAL_CONTEXT]\n'
    source+='\n'.join(f'[IMG_CAP_{i+1}] Food photo {b} {i}.' for i in range(count))
    return dict(business_id=b,split='validation',source=source,source_sha256=text_hash(source),selected_review_ids=[b+'_r'],selected_photo_ids=[b+'_p'+str(i) for i in range(count)])


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn,cls.original,cls.derived=adapted_mapping_function()
        cls.policy=read(Path(__file__).resolve().parents[1]/'outputs/final_evaluation/s1000_preparation_v1/SHUFFLED_POLICY.json')
    def run_mapping(self,rows):return type(self).fn(rows,TinyTokenizer(),self.policy)
    def test_derangement_one_to_one_exact_counts_and_zero(self):
        rows=[fixture('A',1),fixture('B',1),fixture('C',2),fixture('D',2),fixture('Z',0)]
        result=self.run_mapping(rows);by={r['recipient_id']:r for r in result['mappings']}
        self.assertEqual({by[b]['donor_id'] for b in 'ABCD'},set('ABCD'))
        self.assertTrue(all(by[b]['donor_id']!=b for b in 'ABCD'))
        self.assertEqual(by['A']['donor_id'],'B');self.assertEqual(by['C']['donor_id'],'D')
        self.assertIsNone(by['Z']['donor_id']);self.assertFalse(by['Z']['visual_source_changed'])
        self.assertTrue(all(r['review_identical_to_F'] for r in by.values()))
    def test_order_invariance_single_fixed_seed(self):
        rows=[fixture(b,1) for b in 'ABCDE']
        self.assertEqual(self.run_mapping(rows)['mappings'],self.run_mapping(rows[::-1])['mappings'])
    def test_categories_preferred_globally(self):
        rows=[fixture('A',1,'Pizza'),fixture('B',1,'Pizza'),fixture('C',1,'Seafood'),fixture('D',1,'Seafood')]
        m=self.run_mapping(rows)['mappings'];self.assertTrue(all(r['matching_tier']=='SHARED_CATEGORY' for r in m))
    def test_singleton_stops_no_relaxation(self):
        with self.assertRaisesRegex(ValueError,'NO_FEASIBLE_ONE_TO_ONE_GROUP'):
            self.run_mapping([fixture('A',1),fixture('Z',0)])
    def test_no_cross_split_or_target_fields(self):
        rows=[fixture('A',1),fixture('B',1)];rows[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'VALIDATION source cohort only'):self.run_mapping(rows)
        rows=[fixture('A',1),fixture('B',1)];rows[0]['target']='must never affect mapping'
        with self.assertRaisesRegex(ValueError,'Source-only schema'):self.run_mapping(rows)
    def test_only_split_literal_and_function_name_adapted(self):
        old=ast.parse(self.original)
        old.body[0].name='validation_shuffled_mapping'
        for node in ast.walk(old):
            if isinstance(node,ast.Constant):
                if node.value=='test':node.value='validation'
                elif node.value=='TEST source cohort only':node.value='VALIDATION source cohort only'
        self.assertEqual(ast.dump(old),ast.dump(ast.parse(self.derived)))
    def test_cycles_and_fixed_seed_bootstrap(self):
        mappings=[dict(recipient_id=a,donor_id=b) for a,b in [('A','B'),('B','A'),('C','D'),('D','C'),('Z',None)]]
        cycles=permutation_cycles(mappings);self.assertEqual(sorted(map(len,cycles)),[1,2,2])
        differences={b:0.0 if b=='Z' else .2 for b in 'ABCDZ'}
        self.assertEqual(cycle_bootstrap(differences,cycles,B=200),cycle_bootstrap(differences,cycles,B=200))
        lo,hi=cycle_bootstrap(differences,cycles,B=200);self.assertLessEqual(lo,.16);self.assertGreaterEqual(hi,.16)


if __name__=='__main__':unittest.main(verbosity=2)
