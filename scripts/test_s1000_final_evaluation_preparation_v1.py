"""Focused CPU integrity/statistics regression tests; all prose/data below synthetic."""
import unittest,copy,sys
from pathlib import Path
from s1000_final_eval_core_v1 import *
from score_s1000_automatic_prepared_v1 import descriptive_score,strip_header
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tok=Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'assets/tokenizers/led-base-16384/tokenizer.json'))
        cls.policy=dict(version='TEST_FIXTURE_V1',seed=42,cuisine_subcategory_allowlist=['Italian','Chinese'])
    def fixture(self,b,category='Italian',n=1):
        source=f'[BUSINESS_INFORMATION]\nname={b}\ncategories={category}\ncity=X\nstate=X\n\n[REVIEWS]\n[REV_1] stars=4 date=2020-01-01\nGood food but slow service.\n\n[VISUAL_CONTEXT]\n'+'\n\n'.join(f'[IMG_CAP_{i+1}]\nA plate with food {b}.' for i in range(n))
        return dict(business_id=b,split='test',source=source,source_sha256=text_hash(source),selected_review_ids=[b+'r'],selected_photo_ids=[b+str(i) for i in range(n)])
    def test_one_to_one_and_reviews_unchanged(self):
        rows=[self.fixture(b) for b in 'abc']+[self.fixture('z',n=0)]
        m=shuffled_mapping(rows,self.tok,self.policy);pos=[r for r in m['mappings'] if r['donor_id']]
        self.assertEqual(len({r['donor_id'] for r in pos}),3)
        for r in m['mappings']:
            self.assertNotEqual(r['recipient_id'],r['donor_id']);self.assertTrue(r['review_identical_to_F'])
            self.assertNotIn('[REV_',r['visual_task_source'])
        self.assertEqual(m,shuffled_mapping(rows[::-1],self.tok,self.policy))
        self.assertIsNone(m['mappings'][-1]['donor_id'])
    def test_category_preference_global(self):
        rows=[self.fixture('a'),self.fixture('b'),self.fixture('c','Chinese'),self.fixture('d','Chinese')]
        m=shuffled_mapping(rows,self.tok,self.policy)
        self.assertTrue(all(r['matching_tier']=='SHARED_CATEGORY' for r in m['mappings']))
    def test_singleton_caption_stratum_stops(self):
        with self.assertRaisesRegex(ValueError,'NO_FEASIBLE'):shuffled_mapping([self.fixture('a'),self.fixture('b',n=2)],self.tok,self.policy)
    def test_forbid_payloads_and_corruption(self):
        a=[self.fixture('a'),self.fixture('b')];a[0]['reference']='forbidden'
        with self.assertRaises(ValueError):shuffled_mapping(a,self.tok,self.policy)
        a=[self.fixture('a'),self.fixture('b')];a[0]['source']+='tampered'
        with self.assertRaises(ValueError):shuffled_mapping(a,self.tok,self.policy)
    def refs(self):
        source=dict(business_id='x',reviews=[dict(review_id='r',text='Good food but slow service.')],photos=[dict(photo_id='p')])
        r=dict(business_id='x',source_bundle_sha256=digest(source),review_consensus='Good food, slow service.',visual_evidence='A plate.',
            review_claims=[dict(output_span='Good food',evidence=[dict(review_id='r',quote='Good food')])],
            visual_claims=[dict(output_span='A plate.',evidence=[dict(photo_id='p',direct_image_viewed=True)])],status='APPROVED',
            author=dict(human_id='H1',at_utc='2026-01-01T00:00:00+00:00',human_written_attestation=True,predictions_seen=False),
            checker=dict(human_id='H2',at_utc='2026-01-01T01:00:00+00:00',independent_check_attestation=True,predictions_seen=False),history=[])
        h=digest(reference_content(r));r['checker']['reviewed_content_sha256']=h;r['history']=[dict(content_sha256=h)]
        return r,source
    def test_objective_valid_reference_synthetic_only(self):
        r,s=self.refs();self.assertEqual(validate_reference(r,s,True),[])
    def test_same_author_checker_rejected(self):
        r,s=self.refs();r['checker']['human_id']='H1';self.assertIn('two distinct humans required',validate_reference(r,s,True))
    def test_stale_approval_rejected(self):
        r,s=self.refs();r['review_consensus']+=' Wrong.';self.assertIn('checker approved another content version',validate_reference(r,s,True))
    def test_wrong_evidence_and_inexact_quote_rejected(self):
        r,s=self.refs();r['review_claims'][0]['evidence'][0]['quote']='not in source';r['visual_claims'][0]['evidence'][0]['photo_id']='donor'
        e=validate_reference(r,s,True);self.assertIn('review quote not exact',e);self.assertIn('foreign or missing evidence ID',e)
    def test_draft_not_final(self):
        r,s=self.refs();r['review_consensus']='';r['status']='NOT_STARTED';self.assertTrue(validate_reference(r,s,True))
    def test_rouge_identical_and_disjoint(self):
        a=descriptive_score('A red plate.','A red plate.','A red plate.');self.assertAlmostEqual(a['rouge1_fmeasure'],1)
        b=descriptive_score('Blue sky','red plate','red plate');self.assertEqual(b['rouge1_fmeasure'],0)
        self.assertEqual(strip_header('[WRONG]\ntext','review'),'[WRONG]\ntext')
    def test_repetition_and_empty(self):
        self.assertGreater(diagnostics('Good food. Good food.')['duplicate_sentence_fraction'],0)
        self.assertTrue(diagnostics('')['empty'])
    def test_bootstrap_preserves_business_pair(self):
        r=paired([1,2,3],[0,1,2],B=500,P=1000)
        self.assertEqual(r['CI95'],[1.,1.]);self.assertEqual(r,paired([1,2,3],[0,1,2],B=500,P=1000))
        with self.assertRaises(ValueError):paired([1,2],[1])
    def test_holm_monotonic_adjustment(self):self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
    def test_ordinal_agreement(self):
        self.assertAlmostEqual(ordinal_alpha([[1,1,1],[3,3,3],[5,5,5]]),1.)
        self.assertIsNone(ordinal_alpha([[3,3,3],[3,3,3]]))
        self.assertLess(ordinal_alpha([[1,3,5],[5,1,3],[3,5,1]]),0.)
if __name__=='__main__':unittest.main(verbosity=2)
