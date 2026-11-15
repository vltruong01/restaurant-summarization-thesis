"""Synthetic verification guards only; never calls deletion main."""
from pathlib import Path
import tempfile,unittest,hashlib
from extra_transfer_cleanup_20261029_v1 import LiveVerifier,check_archive,TARGETS
class ExtraChecks(unittest.TestCase):
 def fixture(self,base):
  target=base/TARGETS[0];target.parent.mkdir(parents=True);target.write_bytes(b'synthetic archive')
  keeper=base/'keepers/evidence.txt';keeper.parent.mkdir();keeper.write_bytes(b'scientific evidence')
  h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  row=dict(path=TARGETS[0],eligible=True,errors=[],bytes=target.stat().st_size,sha256=h(target),files=1,
   members=[dict(keep='keepers/evidence.txt',bytes=keeper.stat().st_size,sha256=h(keeper))])
  return target,keeper,row
 def test_live_valid_and_no_deletion(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);a,k,r=self.fixture(b);self.assertEqual(check_archive(r,LiveVerifier(b)),a.resolve());self.assertTrue(a.exists() and k.exists())
 def test_changed_keeper_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);a,k,r=self.fixture(b);k.write_bytes(b'changed contents')
   with self.assertRaises(ValueError):check_archive(r,LiveVerifier(b))
 def test_exact_container_duplicate_guard(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);a,k,r=self.fixture(b);keep=b/'keepers/copy.zip';keep.write_bytes(a.read_bytes())
   r.update(verification_kind='EXACT_DUPLICATE_CONTAINER',keep_archive='keepers/copy.zip',files=0,members=[])
   self.assertEqual(check_archive(r,LiveVerifier(b)),a.resolve())
   keep.write_bytes(b'wrong same bytes!')
   with self.assertRaises(ValueError):check_archive(r,LiveVerifier(b))
 def test_circular_keeper_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);a,k,r=self.fixture(b);r['members'][0]['keep']=TARGETS[1]
   with self.assertRaises(ValueError):check_archive(r,LiveVerifier(b))
 def test_escape_and_unapproved_target_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);a,k,r=self.fixture(b)
   with self.assertRaises(ValueError):LiveVerifier(b).path('../outside')
   r['path']='outputs/stage_k/s1000_test100_phase1_v1/STAGE_K_TEST100_PHASE1_REVIEW.zip'
   with self.assertRaises(ValueError):check_archive(r,LiveVerifier(b))
if __name__=='__main__':unittest.main(verbosity=2)
