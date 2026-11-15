"""Focused meaningful invariants for the source-separated intervention; no model calls."""
import unittest,json,copy,sys
from pathlib import Path
from tokenizers import Tokenizer
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from prepare_s1000_shuffled_control_v1 import prepare_mapping
from led_large_shuffled_production_v2 import sha_text
class SourceSeparatedShuffled(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=Tokenizer.from_file(str(R/'assets/tokenizers/led-base-16384/tokenizer.json'))
  cls.policy=json.loads((R/'outputs/stage_i/led_large_matched_signoff_v3/SHUFFLED_POLICY_PROPOSED_UNCHANGED.json').read_text())
 def fixture(self,bid,caption,category='Italian',split='validation'):
  n=bool(caption);source=f'[BUSINESS_INFORMATION]\nname={bid}\ncategories={category}\ncity=X\nstate=X\n\n[REVIEWS]\n[REV_1] stars=5.0 date=2020-01-01\nGood pasta.\n\n[VISUAL_CONTEXT]\n'+('[IMG_CAP_1]\n'+caption if n else '')
  return dict(business_id=bid,split=split,source=source,source_sha256=sha_text(source),selected_review_ids=[bid+'r'],selected_photo_ids=[bid+'p'] if n else [])
 def sample(self):return [self.fixture('a','a plate of pasta.'),self.fixture('b','a bowl of soup.'),self.fixture('c','a plate of salad.','Mexican'),self.fixture('z','')]
 def test_only_visual_changes_and_cuisine_preference(self):
  x=prepare_mapping(self.sample(),self.tok,self.policy);m={r['recipient_id']:r for r in x['mappings']}
  self.assertEqual(m['a']['donor_id'],'b');self.assertEqual(m['b']['donor_id'],'a')
  for key,r in m.items():
   self.assertTrue(r['review_input_identical_to_F']);self.assertEqual(r['evaluation_target_business_id'],key)
   self.assertNotIn('[REV_',r['visual_task_source']);self.assertNotIn('[BUSINESS_INFORMATION]',r['visual_task_source'])
 def test_zero_not_fabricated(self):
  z=next(x for x in prepare_mapping(self.sample(),self.tok,self.policy)['mappings'] if x['recipient_id']=='z')
  self.assertIsNone(z['donor_id']);self.assertEqual(z['visual_task_source'],'[TASK_VISUAL_EVIDENCE]\n[VISUAL_CONTEXT]\n')
 def test_seed_determinism_and_input_order_invariance(self):
  x=self.sample();first=prepare_mapping(x,self.tok,self.policy)
  self.assertEqual(first,prepare_mapping(x,self.tok,self.policy))
  self.assertEqual(first['mappings'],prepare_mapping(list(reversed(x)),self.tok,self.policy)['mappings'])
 def test_no_self_or_missing_donor(self):
  with self.assertRaises(ValueError):prepare_mapping([self.fixture('a','a plate.')],self.tok,self.policy)
 def test_TEST_mapping_pending(self):
  x=self.sample()
  for r in x:r['split']='test'
  with self.assertRaisesRegex(ValueError,'TEST mapping not committed'):prepare_mapping(x,self.tok,self.policy)
 def test_reference_field_rejected(self):
  x=self.sample();x[0]['reference']='must not inspect'
  with self.assertRaises(ValueError):prepare_mapping(x,self.tok,self.policy)
 def test_no_silent_caption_count_relaxation(self):
  x=self.sample()[:2];x[1]['selected_photo_ids'].append('extra')
  with self.assertRaises(ValueError):prepare_mapping(x,self.tok,self.policy)
 def test_corrupted_source_rejected(self):
  x=self.sample();x[0]['source']+=' changed'
  with self.assertRaises(ValueError):prepare_mapping(x,self.tok,self.policy)
if __name__=='__main__':unittest.main()
