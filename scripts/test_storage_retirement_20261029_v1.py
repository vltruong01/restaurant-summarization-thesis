"""Local synthetic pack/restore integrity tests, no project deletion or HF upload."""
from pathlib import Path
import tempfile,unittest,copy,hashlib
from archive_retired_runs_private_hf_20261029_v1 import pack_group,verify_tar,RemoteBytes
class ArchiveTests(unittest.TestCase):
    def fixture(self,base):
        workspace=base/'workspace';root=workspace/'outputs/retired_fixture';root.mkdir(parents=True)
        (root/'witness.txt').write_bytes(b'Original failed-run evidence\n');(root/'nested').mkdir();(root/'nested/empty').mkdir()
        (root/'nested/unicode_ảnh.bin').write_bytes(bytes(range(256))*7)
        return workspace,dict(id='fixture',path='outputs/retired_fixture'),base/'audit'
    def test_roundtrip_all_bytes_timestamps_and_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);w,g,o=self.fixture(b);m,a,mp=pack_group(g,o,workspace=w);dest=b/'restored';dest.mkdir()
            data=a.read_bytes();n=verify_tar(RemoteBytes([data[i:i+37] for i in range(0,len(data),37)]),m,dest)
            self.assertEqual(n,2)
            for r in m['files']:
                self.assertEqual((dest/r['path']).read_bytes(),(w/r['path']).read_bytes())
                self.assertEqual((dest/r['path']).stat().st_mtime_ns,r['mtime_ns'])
            self.assertTrue((dest/'outputs/retired_fixture/nested/empty').is_dir())
    def test_member_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w,g,o=self.fixture(Path(tmp));m,a,_=pack_group(g,o,workspace=w);m['files'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):verify_tar(RemoteBytes([a.read_bytes()]),m)
    def test_source_changed_after_pack_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w,g,o=self.fixture(Path(tmp));pack_group(g,o,workspace=w);(w/g['path']/'witness.txt').write_text('Changed source')
            with self.assertRaises(ValueError):pack_group(g,o,workspace=w)
    def test_archive_whole_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w,g,o=self.fixture(Path(tmp));m,a,_=pack_group(g,o,workspace=w);m['archive_sha256']='0'*64
            with self.assertRaises(ValueError):verify_tar(RemoteBytes([a.read_bytes()]),m)
if __name__=='__main__':unittest.main(verbosity=2)
