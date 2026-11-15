"""Synthetic offline policy checks; no models, network, GPU or human labels."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage_i_source_consistency_v1 import *


class Tok:
    def encode(self,a,b=None,**kwargs):return [0]+a.split()+(b.split() if b else [])+[2]


class Fake:
    et=Tok();nt=Tok()
    def __init__(self,low=False):self.calls=[];self.low=low
    def embeds(self,texts):return np.array([[1.,0.] for t in texts])
    def nli(self,pairs):
        self.calls+=pairs
        def pred(a,b):
            aa,bb=a.lower(),b.lower()
            difference=('rude' in aa)!=('rude' in bb)
            if 'friendly' in aa and 'rude' in aa:difference=False
            p=.05 if self.low or difference else .95
            return {'entailment':p,'contradiction':.01,'neutral':.99-p,'raw_logits':[0,1,0]}
        return [pred(a,b) for a,b in pairs]


def config(**kw):
    return {**{'evidence_embedding_tokens':220,'review_retrieval_top_k':3,'review_entailment_min':.8,
        'visual_entailment_min':.9,'review_entailment_margin':.6,'visual_entailment_margin':.8,
        'MMR_lambda':.65,'semantic_cosine_min':.85,'mutual_entailment_min':.9,
        'review_sentences_max':10,'visual_sentences_max':5,'output_LED_tokens_max':768},**kw}


def case(raw,visual=''):
    return {'business_id':'synthetic','subset':'synthetic','source':'[BUSINESS_INFORMATION]\nname=X\n\n[REVIEWS]\n[REV_1] stars=5 date=2000\nFriendly service. Rude service.\n\n[VISUAL_CONTEXT]\n'+visual,
        'source_sha256':'synthetic','selected_photo_ids':['photo1','photo2'] if 'IMG_CAP_2' in visual else ['photo1'] if visual else [],
        'raw_prediction':raw,'raw_prediction_sha256':'synthetic'}


class Policies(unittest.TestCase):
    def test_zero_caption_visual_removed_without_fabricating(self):
        r=select(case('[REVIEW_CONSENSUS]\nFriendly service.\n\n[VISUAL_EVIDENCE]\nA bagel is on a plate.'),Fake(),config(),Tok())
        self.assertEqual(r['selected_sentences']['visual'],[])
        self.assertTrue(any(x['decision']=='DROP_ZERO_CAPTION_VISUAL' for x in r['decisions']))
        self.assertEqual(r['new_content_tokens_generated'],0)

    def test_auxiliary_model_does_not_receive_gold_fields(self):
        c=case('[REVIEW_CONSENSUS]\nFriendly service.');c['target']='secret gold'
        with self.assertRaisesRegex(ValueError,'prohibited'):select(c,Fake(),config(),Tok())

    def test_all_visual_premises_are_single_caption_not_reviews(self):
        f=Fake();v='[IMG_CAP_1]\nA glass is on a table.\n\n[IMG_CAP_2]\nWine bottles are on a rack.'
        select(case('[REVIEW_CONSENSUS]\n\n[VISUAL_EVIDENCE]\nA glass is on a table.',v),f,config(),Tok())
        self.assertTrue(f.calls)
        self.assertTrue(all('\n' not in a and 'Friendly' not in a for a,b in f.calls))

    def test_malformed_header_normalization_is_declared_not_hidden(self):
        raw='[VISUAL_CONTEXT]\nFriendly service.\n\n[VISUAL_EVIDENCE]\n'
        r=select(case(raw),Fake(),config(),Tok())
        self.assertIn('[VISUAL_CONTEXT]',r['original_markers']);self.assertTrue(r['canonical_serialization_applied'])
        self.assertIn('Friendly service.',r['text'])

    def test_near_duplicate_mutual_entailment_removed(self):
        raw='[REVIEW_CONSENSUS]\nFriendly service. The staff provide friendly service.\n\n[VISUAL_EVIDENCE]\n'
        r=select(case(raw),Fake(),config(),Tok())
        self.assertEqual(len(r['selected_sentences']['review']),1)
        self.assertTrue(any(d['decision']=='DROP_MODEL_MUTUAL_ENTAILMENT_REDUNDANCY' for d in r['decisions']))

    def test_opposing_opinions_not_removed_by_cosine_alone(self):
        raw='[REVIEW_CONSENSUS]\nFriendly service. Rude service.\n\n[VISUAL_EVIDENCE]\n'
        r=select(case(raw),Fake(),config(),Tok())
        self.assertEqual(len(r['selected_sentences']['review']),2)

    def test_incomplete_last_fragment_is_not_completed_by_code(self):
        raw='[REVIEW_CONSENSUS]\nFriendly service. The scallops including'
        r=select(case(raw),Fake(),config(),Tok())
        self.assertNotIn('scallops',r['text'])
        self.assertTrue(any(d['decision']=='DROP_INCOMPLETE_FRAGMENT' for d in r['decisions']))

    def test_insufficient_support_may_leave_empty_and_is_flagged(self):
        r=select(case('[REVIEW_CONSENSUS]\nFriendly service.'),Fake(low=True),config(),Tok())
        self.assertEqual(r['selected_sentences']['review'],[])
        self.assertEqual(r['coverage_warning'],'LESS_THAN_THREE_REVIEW_SENTENCES_RETAINED')

    def test_source_long_sentence_explicitly_excluded_not_cut(self):
        records=[{'text':' '.join(['word']*230)+'.','marker':'REV_1','source_start':0,'source_end':1200,'domain':'review'}]
        accepted,rejected=pack_evidence(records,Tok(),220)
        self.assertEqual(accepted,[]);self.assertEqual(len(rejected),1)

    def test_sentence_tokens_are_exact_raw_spans(self):
        raw='[REVIEW_CONSENSUS]\n  Friendly service. Rude service.\n\n[VISUAL_EVIDENCE]\n'
        r=select(case(raw),Fake(),config(),Tok())
        for group in r['selected_sentences'].values():
            for u in group:self.assertEqual(raw[u['start']:u['end']],u['text'])

    def test_budget_drops_whole_units_without_partial_sentence(self):
        raw='[REVIEW_CONSENSUS]\nFriendly service. Rude service.\n\n[VISUAL_EVIDENCE]\n'
        r=select(case(raw),Fake(),config(output_LED_tokens_max=8),Tok())
        self.assertLessEqual(len(r['serialized_token_ids']),8)
        self.assertTrue(all(s in raw for s in [u['text'] for u in r['selected_sentences']['review']]))

    def test_photo_binding_mismatch_stops(self):
        with self.assertRaisesRegex(ValueError,'alignment'):source_records('[REVIEWS]\n[REV_1]\nText.\n[VISUAL_CONTEXT]\n[IMG_CAP_1]\nA plate.',[])


if __name__=='__main__':unittest.main()
