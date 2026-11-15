"""Offline fixtures test resumable transfer and refusal to publish corrupt weights."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import download_caption_checkpoint as transfer


class Response:
    def __init__(self, data, start, end, corrupt=False):
        self.status_code=206
        self.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'}
        self.data=data[start:end+1]
        if corrupt:
            self.data=bytes([self.data[0]^1])+self.data[1:]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,size):
        for i in range(0,len(self.data),size):yield self.data[i:i+size]


class TransferTests(unittest.TestCase):
    def fixture(self, root):
        out=root/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
        (out/'resume_20261006').mkdir(parents=True)
        revision='a'*40
        payloads={'model-00001-of-00002.safetensors':bytes(range(256))*(2**14)+b'first_end',
                  'model-00002-of-00002.safetensors':b'fixture_not_model_weights'*100}
        files=[{'filename':name,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
               for name,data in payloads.items()]
        inventory={'model_id':'Salesforce/blip2-opt-2.7b-coco','resolved_commit':revision,
                   'weight_files':files,'total_bytes':sum(len(p) for p in payloads.values())}
        (out/'model_revision.json').write_text(json.dumps({'resolved_commit':revision}))
        (out/'resume_20261006/checkpoint_download_inventory.json').write_text(json.dumps(inventory))
        cache=root/'assets/model_cache'
        cache.mkdir(parents=True)
        return out,revision,payloads,files

    def test_corrupt_resume_chunk_is_fetched_again_and_final_bytes_match(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as directory:
            root=Path(directory)
            assert root.resolve().is_relative_to(transfer.ROOT/'outputs')
            out,revision,payloads,files=self.fixture(root)
            staging=root/'assets/model_cache/range_downloads'/revision
            staging.mkdir(parents=True)
            first=files[0]
            original=payloads[first['filename']][:4*2**20]
            damaged=bytes([original[0]^1])+original[1:]
            (staging/(first['filename']+'.part')).write_bytes(damaged)
            state={'resolved_commit':revision,'chunk_bytes':4*2**20,'files':{
                first['filename']:{**{k:first[k] for k in ['size_bytes','sha256']},
                                  'chunks':{'0':{'bytes':len(original),'sha256':hashlib.sha256(original).hexdigest()}}}}}
            (staging/'progress.json').write_text(json.dumps(state))
            requested=[]
            def fetch(url,headers,**kwargs):
                name=url.split('?')[0].rsplit('/',1)[1]
                start,end=map(int,headers['Range'].removeprefix('bytes=').split('-'))
                requested.append((name,start))
                return Response(payloads[name],start,end)
            with patch.object(transfer,'ROOT',root),patch.object(transfer,'OUT',out),patch.object(transfer.requests,'get',fetch),patch('sys.argv',['transfer','--workers','2','--max-seconds','30']):
                self.assertEqual(transfer.main(),0)
            self.assertIn((first['filename'],0),requested)
            target=root/'assets/model_cache/models--Salesforce--blip2-opt-2.7b-coco/snapshots'/revision
            for name,data in payloads.items():self.assertEqual((target/name).read_bytes(),data)
            report=json.loads((out/'weight_download_progress.json').read_text())
            self.assertTrue(report['weights_SHA256_verified'])
            self.assertEqual(report['model_loads'],0)

    def test_full_LFS_checksum_rejects_wrong_body_before_cache_publication(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as directory:
            root=Path(directory)
            assert root.resolve().is_relative_to(transfer.ROOT/'outputs')
            out,revision,payloads,files=self.fixture(root)
            def fetch(url,headers,**kwargs):
                name=url.split('?')[0].rsplit('/',1)[1]
                start,end=map(int,headers['Range'].removeprefix('bytes=').split('-'))
                return Response(payloads[name],start,end,corrupt=True)
            with patch.object(transfer,'ROOT',root),patch.object(transfer,'OUT',out),patch.object(transfer.requests,'get',fetch),patch('sys.argv',['transfer','--workers','2','--max-seconds','30']):
                with self.assertRaisesRegex(AssertionError,'Full weight SHA256 mismatch'):transfer.main()
            target=root/'assets/model_cache/models--Salesforce--blip2-opt-2.7b-coco/snapshots'/revision
            self.assertFalse((target/files[0]['filename']).exists())


if __name__=='__main__':unittest.main()
