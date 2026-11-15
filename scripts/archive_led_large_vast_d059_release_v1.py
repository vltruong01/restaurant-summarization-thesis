"""Create the single release ZIP and stream-verify every member against the manifest."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='LED_LARGE_VAST_D059_F_V1'
SOURCE=ROOT/'outputs/cloud_handoff'/NAME
PREP=ROOT/'outputs/stage_i/led_large_vast_d059_v1_preparation'

def sha_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    archive=SOURCE.with_suffix('.zip')
    manifest=json.loads((SOURCE/'VAST_SHA256_MANIFEST.json').read_text())['files']
    assert not archive.exists()
    assert not any('__pycache__' in Path(n).parts or n.endswith('.pyc') for n in manifest)
    stored={'.bin','.npz','.jpg','.jpeg','.png','.zip','.gz','.safetensors'}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for i,rel in enumerate(manifest,1):
            method=zipfile.ZIP_STORED if Path(rel).suffix.lower() in stored else zipfile.ZIP_DEFLATED
            z.write(SOURCE/rel,NAME+'/'+rel,compress_type=method,compresslevel=1 if method==zipfile.ZIP_DEFLATED else None)
            if i%2000==0:print('Archived',i,'/',len(manifest),flush=True)
        z.write(SOURCE/'VAST_SHA256_MANIFEST.json',NAME+'/VAST_SHA256_MANIFEST.json')
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={NAME+'/'+n for n in manifest}|{NAME+'/VAST_SHA256_MANIFEST.json'}
        for i,(rel,expected) in enumerate(manifest.items(),1):
            with z.open(NAME+'/'+rel) as f:assert sha_stream(f)==expected,rel
            if i%2000==0:print('SHA256/CRC verified',i,'/',len(manifest),flush=True)
    with archive.open('rb') as f:digest=sha_stream(f)
    Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    receipt=dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,files=len(manifest)+1,
        every_member_SHA256_and_CRC_verified=True,local_bytecode_excluded=True,top_level_folder=NAME,
        GPU_execution=False)
    (PREP/'DELIVERED_ZIP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
