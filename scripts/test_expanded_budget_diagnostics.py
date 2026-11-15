"""Regression checks for the expanded runner's count cache and Windows writes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
import orjson,yaml
from tokenizers import Tokenizer
from run_expanded_budget_diagnostics import write_json,worker,ROOT
from run_budget_review_pilot import run_business
from token_length_cache import TokenLengthCache


class ExpandedRegression(unittest.TestCase):
    def test_temporary_windows_reader_lock_does_not_lose_checkpoint(self):
        actual=Path.replace;attempts=[]
        def locked_then_replace(path,target):
            attempts.append(1)
            if len(attempts)<3:raise PermissionError('simulated Windows reader lock')
            return actual(path,target)
        with TemporaryDirectory() as directory:
            target=Path(directory)/'checkpoint.json';target.write_text('{"completed":1}')
            with patch.object(Path,'replace',locked_then_replace):write_json(target,{'completed':2})
            self.assertEqual(json.loads(target.read_text()),{'completed':2})
            self.assertEqual(len(attempts),3)

    def test_exact_count_cache_separates_special_tokens_and_hides_fake_ids(self):
        tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
        tok.no_truncation();tok.no_padding();memo=TokenLengthCache(tok);text='Complete original review. '
        for special in [True,False,True,False]:
            self.assertEqual(len(memo.encode(text,add_special_tokens=special).ids),len(tok.encode(text,add_special_tokens=special).ids))
        self.assertEqual((memo.hits,memo.misses),(2,2))
        with self.assertRaises(TypeError):list(memo.encode(text).ids)

    def test_expanded_worker_preserves_full_original_search_payload(self):
        cfg=yaml.safe_load((ROOT/'configs/review_selection_expanded.yaml').read_text(encoding='utf-8'))
        pop=[{'business_id':'unit','review_id':f'r{i:02}','date':f'2020-01-{i+1:02} 00:00:00','stars':1+i%5,
              'text':'Original complete review about food and service. '*(1+i)} for i in range(12)]
        info={'name':'Unit business','split':'train','volume_band':'10-30','sample_group':'all_train_validation'}
        tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_truncation();tok.no_padding()
        expected=json.loads(orjson.dumps(run_business('unit',pop,cfg,tok,info),option=orjson.OPT_NON_STR_KEYS))
        got=json.loads(orjson.dumps(worker(('unit',pop,cfg,info,None,'fresh_expanded_search')),option=orjson.OPT_NON_STR_KEYS))
        for key in ['frontiers','candidate_audit','search_statistics','previews','sensitivity','long_review_stress']:
            self.assertEqual(got[key],expected[key],key)
        for a,b in zip(got['runs'],expected['runs']):
            for key,value in b.items():self.assertEqual(a[key],value,key)
        profile=got['length_profile'];self.assertEqual(len(profile['ten_cheapest_singleton_probe_ids']),10)
        self.assertEqual(profile['individual_body_tokens_min'],min(len(tok.encode(r['text'],add_special_tokens=False).ids) for r in pop))
        self.assertEqual(profile['individual_body_tokens_max'],max(len(tok.encode(r['text'],add_special_tokens=False).ids) for r in pop))


if __name__=='__main__':unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
