"""Production adapter tests; actual prior records remain read-only, new witnesses are fixtures."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from stage_f_verification_workflow import read
import construct_stage_f_production_d033 as production
import prepare_stage_f_production_reviews_d033 as review_adapter
from source_budget_audit_template import counts,pinned_tokenizer,review_block


class ProductionTests(unittest.TestCase):
    def record(self):
        sources=read(production.OUT/'reused_caption_sources.json');pid,item=next(iter(sources.items()))
        r=read(production.ROOT/item['checkpoint_path']);r['_record_path']=item['checkpoint_path'];r['_record_sha256']=item['checkpoint_sha256']
        return pid,r

    def test_zero_filtered_and_whole_caption_prefix_no_fabrication(self):
        pid,r=self.record();pool={'filtered_diversity_ranking':[],'unfiltered_fixed_seed_ranking':[pid]}
        m=production.condition_manifest(r['business_id'],r['split'],pool,{pid:r},'filtered')
        self.assertEqual(m['captions'],[]);self.assertTrue(m['empty_caption_business'])
        u=production.condition_manifest(r['business_id'],r['split'],pool,{pid:r},'unfiltered')
        self.assertEqual(u['captions'][0]['raw_caption'],r['raw_caption'])
        self.assertEqual(u['captions'][0]['normalized_caption'],r['normalized_caption'])
        self.assertEqual(u['retained_caption_count'],1)

    def test_test_review_adapter_preserves_complete_frozen_discovery_on_resume(self):
        root=production.ROOT
        pop=[{'business_id':'SYNTHETIC_NOT_DATA','review_id':f'r{i:02d}','stars':i%5+1,
              'date':f'2020-01-{i+1:02d} 12:00:00','text':f'The food and service were pleasant and this is a complete review number {i}.'} for i in range(12)]
        info={'name':'FIXTURE','split':'test','volume_band':'10-30','sample_group':'fixture'}
        with tempfile.TemporaryDirectory(prefix='d033_fixture_',dir=root/'outputs/stage_f') as tmp,patch.object(review_adapter,'OUT',Path(tmp)):
            a=review_adapter.test_worker(('SYNTHETIC_NOT_DATA',pop,info,'FIXTURE_ONLY'))
            b=review_adapter.test_worker(('SYNTHETIC_NOT_DATA',list(reversed(pop)),info,'FIXTURE_ONLY'))
            self.assertEqual(a,b);self.assertEqual(a['review_budget'],4000)
            self.assertEqual(len(a['selected_review_ids']),12)
            self.assertEqual(len(list(Path(tmp).glob('test_witnesses/*/business_runs/*.json'))),4)

    def test_empty_visual_header_and_overflow_is_recorded_not_truncated(self):
        tok,_=pinned_tokenizer();business={'name':'Fixture','categories':'Restaurants','city':'City','state':'ST'}
        reviews=[{'date':'2020-01-01','review_id':'r','stars':5,'text':'A complete unchanged review.'}]
        block=review_block(reviews)
        cost,source=counts(tok,business,block,[])
        self.assertIn('[VISUAL_CONTEXT]\n',source);self.assertEqual(cost['caption_count'],0)
        long='word '*6000;overflow,text=counts(tok,business,block,[long])
        self.assertTrue(overflow['caption_budget_overflow']);self.assertTrue(overflow['source_budget_overflow'])
        self.assertIn(long,text);self.assertIn(block,text)


if __name__=='__main__':unittest.main()
