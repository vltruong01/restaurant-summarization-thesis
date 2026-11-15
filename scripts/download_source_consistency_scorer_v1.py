"""Public, pinned model download with upstream LFS/Git checksum verification.

No auth token, inference provider, Teacher call, model loading or GPU operation.
Uses ordinary files because Windows Hub symlinks failed with WinError1314.
"""
import hashlib,json,urllib.request
from pathlib import Path

REPO='cross-encoder/nli-deberta-v3-small'
REV='e9890682d9e4279b7ae6d0fcfb435a43206280ec'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/experimental_scorers/nli_deberta_small_e989068'
NAMES={'README.md','added_tokens.json','config.json','model.safetensors','special_tokens_map.json','spm.model','tokenizer_config.json'}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true',timeout=60) as r:
        metadata=json.load(r)
    assert metadata['sha']==REV
    report={'model_id':REPO,'revision':REV,'public_anonymous_download':True,
        'previous_Hub_transport_failure':'WinError1314 symlink privilege; no inference occurred',
        'files':{}}
    for item in metadata['siblings']:
        name=item['rfilename']
        if name not in NAMES:continue
        path=OUT/name
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true',timeout=120) as response,path.open('xb') as dest:
            while block:=response.read(1024*1024):dest.write(block)
        assert path.stat().st_size==item['size']
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if 'lfs' in item:assert digest==item['lfs']['sha256']
        else:assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==item['blobId']
        report['files'][name]={'sha256':digest,'bytes':len(data),'upstream_identity_verified':True}
        print(json.dumps({'downloaded':name,'bytes':len(data),'checksum':'PASS'}),flush=True)
    assert set(report['files'])==NAMES
    (OUT/'DOWNLOAD_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
