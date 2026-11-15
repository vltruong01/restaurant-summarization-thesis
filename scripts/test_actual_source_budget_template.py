"""Whole-source accounting guards; synthetic fixtures never become research inputs."""
import unittest
from source_budget_audit_template import pinned_tokenizer,review_block,base_source,assemble,counts


class BudgetTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tok,_=pinned_tokenizer()

    def setUp(self):
        self.business={'name':'Cafe Example','categories':'Restaurants, Cafes','city':'Example','state':'PA','stars':4.5,'review_count':100}
        self.reviews=[{'review_id':'a','stars':4,'date':'2020-01-01 10:00:00','text':'Keep the entire review. Literal </s> and [REV_99].\nSecond line stays.'}]

    def test_exact_metadata_fields_and_original_review_text(self):
        block=review_block(self.reviews);base=base_source(self.business,block)
        self.assertIn(self.reviews[0]['text'],base)
        self.assertTrue(base.startswith('[BUSINESS_INFORMATION]\nname=Cafe Example\ncategories='))
        self.assertNotIn('review_count=',base);self.assertNotIn('stars=4.5',base)
        self.assertEqual(sum(self.tok.encode(base,add_special_tokens=True).special_tokens_mask),2)

    def test_no_silent_review_reordering(self):
        reversed_rows=[{'review_id':'b','stars':5,'date':'2021','text':'Later.'},{'review_id':'a','stars':4,'date':'2020','text':'Earlier.'}]
        with self.assertRaises(AssertionError):review_block(reversed_rows)

    def test_same_review_base_for_all_caption_condition_contracts(self):
        block=review_block(self.reviews);base=base_source(self.business,block)
        self.assertEqual(assemble(base,None),base)
        for cap in [['Unfiltered caption.'],['Filtered caption.'],['A donor caption from another business.']]:
            source=assemble(base,cap)
            self.assertTrue(source.startswith(base+'\n\n[VISUAL_CONTEXT]\n'))
            self.assertIn(block,source)

    def test_exact_telescoping_accounting_includes_markers_boundaries(self):
        m,source=counts(self.tok,self.business,review_block(self.reviews),['a plate of salad.','a glass of water.'])
        self.assertEqual(m['full_source_tokens'],m['review_block_tokens_with_special']+m['marginal_metadata_review_header_overhead']+m['caption_block_increment_tokens'])
        self.assertEqual(m['caption_block_increment_tokens'],m['visual_header_separator_increment_tokens']+sum(m['complete_caption_prefix_increments']))
        self.assertGreater(m['caption_block_increment_tokens'],m['caption_body_standalone_sum'])
        self.assertIn('[IMG_CAP_2]\na glass of water.',source)

    def test_overflow_is_recorded_without_truncation_or_caption_removal(self):
        original='intact caption '*1000
        m,source=counts(self.tok,self.business,review_block(self.reviews),[original])
        self.assertTrue(m['caption_budget_overflow']);self.assertIn(original,source);self.assertEqual(m['caption_count'],1)
        business=dict(self.business,name='very long metadata '*2000)
        m,source=counts(self.tok,business,review_block(self.reviews),[])
        self.assertTrue(m['source_budget_overflow']);self.assertIn(business['name'],source)

    def test_empty_visual_condition_keeps_header(self):
        m,source=counts(self.tok,self.business,review_block(self.reviews),[])
        self.assertTrue(source.endswith('[VISUAL_CONTEXT]\n'));self.assertEqual(m['caption_count'],0)
        self.assertGreater(m['caption_block_increment_tokens'],0)


if __name__=='__main__':unittest.main()
