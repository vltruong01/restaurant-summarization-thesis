"""Image validity and identity distinctions required by the controlled design."""
from pathlib import Path
import tempfile,unittest
from PIL import Image,PngImagePlugin
from photo_integrity_audit import inspect_photo,digest_file,pixel_digest,phash


class PhotoIntegrityTests(unittest.TestCase):
    def row(self,path,index=0):return {'photo_index':index,'photo_id':path.stem,'business_id':'example','split':'train','label':'food','relative_path':path.name,'bytes':path.stat().st_size,'sha256':digest_file(path)}

    def test_corrupt_or_truncated_matching_bytes_are_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'broken.jpg';p.write_bytes(b'not an image')
            self.assertFalse(inspect_photo((self.row(p),directory))['readable'])
            Image.new('RGB',(64,64),'red').save(p,'JPEG');p.write_bytes(p.read_bytes()[:200])
            self.assertFalse(inspect_photo((self.row(p),directory))['readable'])
            row=self.row(p);row['sha256']='0'*64
            with self.assertRaises(AssertionError):inspect_photo((row,directory))

    def test_metadata_bytes_do_not_equal_normalized_pixel_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Image.new('RGB',(32,16),(120,50,180));a=Path(directory)/'one.png';b=Path(directory)/'two.png'
            image.save(a);meta=PngImagePlugin.PngInfo();meta.add_text('comment','different file metadata');image.save(b,pnginfo=meta)
            first=inspect_photo((self.row(a),directory));second=inspect_photo((self.row(b,1),directory))
            self.assertTrue(first['readable'] and second['readable'])
            self.assertNotEqual(first['raw_sha256'],second['raw_sha256'])
            self.assertEqual(first['pixel_sha256'],second['pixel_sha256'])
            self.assertEqual(first['phash64_hex'],second['phash64_hex'])

    def test_solid_color_perceptual_collision_is_not_pixel_identity(self):
        red=Image.new('RGB',(32,32),'red');blue=Image.new('RGB',(32,32),'blue')
        self.assertEqual(phash(red)[0],phash(blue)[0])
        self.assertNotEqual(pixel_digest(red),pixel_digest(blue))

    def test_exif_transpose_is_in_memory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'oriented.jpg';image=Image.new('RGB',(16,32),'green');exif=Image.Exif();exif[274]=6;image.save(p,exif=exif)
            original=digest_file(p);result=inspect_photo((self.row(p),directory))
            self.assertEqual((result['width'],result['height']),(16,32))
            self.assertEqual((result['normalized_width'],result['normalized_height']),(32,16))
            self.assertEqual(digest_file(p),original)


if __name__=='__main__':unittest.main()
