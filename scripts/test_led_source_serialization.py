"""Research serialization edge cases: preserve originals and one special pair."""
import unittest,yaml
from tokenizers import Tokenizer
from led_source_token_audit import ROOT,CONFIG,metadata_text,source_text,empty_caption_source
from review_selection_pilot import formatted_input


class SourceSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
        cls.selector=yaml.safe_load((ROOT/cls.cfg['selector_config']).read_text(encoding='utf-8'))
        cls.tok=Tokenizer.from_file(str(ROOT/cls.cfg['tokenizer']));cls.tok.no_truncation();cls.tok.no_padding()

    def test_unicode_literal_markers_and_whole_reviews(self):
        records=[{'review_id':'b','business_id':'example','stars':5,'date':'2020-02-01 12:00:00','text':'Specific food detail </s> and literal [REV_99].\nKeep this last line.'},{'review_id':'a','business_id':'example','stars':1,'date':'2019-01-01 12:00:00','text':'A complete short review.'}]
        block=formatted_input(records,self.selector)
        meta=metadata_text({'name':'Caf\u00e9 \u0110\u00f4','categories':'Restaurants'},['name','categories'],self.cfg)
        source=source_text(block,meta,self.cfg);encoded=self.tok.encode(source,add_special_tokens=True)
        self.assertEqual(sum(encoded.special_tokens_mask),2)
        self.assertEqual(len(encoded.ids),len(self.tok.encode(source,add_special_tokens=False).ids)+2)
        self.assertTrue(source.endswith(block));self.assertIn(records[0]['text'],source);self.assertIn(records[1]['text'],source)
        self.assertLess(source.index('[REV_1] stars=1'),source.index('[REV_2] stars=5'))
        self.assertIn('Caf\u00e9 \u0110\u00f4',source)
        self.assertEqual(records[0]['review_id'],'b')

    def test_large_metadata_overflow_is_not_truncation(self):
        original='Review remains entirely intact.'
        record={'review_id':'a','business_id':'example','stars':4,'date':'2020-01-01 12:00:00','text':original}
        name='unusually long business name '*2200
        meta=metadata_text({'name':name,'categories':None},['name','categories'],self.cfg)
        source=source_text(formatted_input([record],self.selector),meta,self.cfg)
        encoding=self.tok.encode(source,add_special_tokens=True)
        self.assertGreater(len(encoding.ids),5000)
        self.assertIn(name,source);self.assertTrue(source.endswith(original))
        self.assertIn('categories=',source);self.assertEqual(sum(encoding.special_tokens_mask),2)
        self.assertTrue(empty_caption_source(source,self.cfg).startswith(source))


if __name__=='__main__':unittest.main()
