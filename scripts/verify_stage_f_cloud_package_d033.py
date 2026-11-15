"""Standard-library-only verification before any package installation or model execution."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='cloud/stage_f_d033/package_manifest.json'


def digest_stream(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_relative(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,name
    return name


def verify(root=ROOT,archive=None):
    if archive:
        with zipfile.ZipFile(archive) as z:
            names=z.namelist();assert len(names)==len(set(names))
            assert all(n.startswith('yelp_dataset/') for n in names)
            for n in names:safe_relative(n)
            data=json.loads(z.read('yelp_dataset/'+MANIFEST));expected={'yelp_dataset/'+MANIFEST}|{'yelp_dataset/'+n for n in data['files']}
            assert set(names)==expected,'ZIP members differ from declared file list'
            for i,(name,entry) in enumerate(data['files'].items(),1):
                safe_relative(name)
                assert z.getinfo('yelp_dataset/'+name).file_size==entry['bytes'],name
                with z.open('yelp_dataset/'+name) as stream:assert digest_stream(stream)==entry['sha256'],name
                if i%5000==0:print(f'ZIP verified {i}/{len(data["files"])}',flush=True)
    else:
        data=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
        for i,(name,entry) in enumerate(data['files'].items(),1):
            p=root/safe_relative(name);assert p.is_file() and p.stat().st_size==entry['bytes'],name
            with p.open('rb') as stream:assert digest_stream(stream)==entry['sha256'],name
            if i%5000==0:print(f'Files verified {i}/{len(data["files"])}',flush=True)
    assert data['cohort_businesses']==2300 and data['full_photo_pool']==30027 and data['reused_caption_records']==2709 and data['pending_generation']==27318
    report={'status':'PASS_PORTABLE_PACKAGE_BYTE_INTEGRITY_NOT_CLOUD_GPU_EXECUTION',
            'files':len(data['files']),'payload_bytes':sum(v['bytes'] for v in data['files'].values()),
            'photos':30027,'reused_caption_records':2709,'pending_generation':27318,'model_inference_executed':False}
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path);a=p.parse_args();verify(archive=a.zip)
