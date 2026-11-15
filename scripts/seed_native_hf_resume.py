"""Verify/copy contiguous custom chunks into Hub-native incomplete blobs; retain originals."""
from pathlib import Path
import hashlib
import json
import os
from filelock import FileLock

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'


def main():
    inventory=json.loads((OUT/'resume_20261006/checkpoint_download_inventory.json').read_text())
    revision=inventory['resolved_commit']
    staging=ROOT/'assets/model_cache/range_downloads'/revision
    state=json.loads((staging/'progress.json').read_text())
    assert state['resolved_commit']==revision
    chunk_bytes=state['chunk_bytes']
    cache=ROOT/'assets/model_cache'
    repo=cache/'models--Salesforce--blip2-opt-2.7b-coco'
    records=[]
    for spec in inventory['weight_files']:
        chunks=state['files'][spec['filename']]['chunks']
        count=0
        while str(count) in chunks:count+=1
        prefix=sum(chunks[str(i)]['bytes'] for i in range(count))
        destination=repo/'blobs'/(spec['sha256']+'.incomplete')
        source=staging/(spec['filename']+'.part')
        lock=cache/'.locks'/repo.name/(spec['sha256']+'.lock')
        lock.parent.mkdir(parents=True,exist_ok=True)
        destination.parent.mkdir(parents=True,exist_ok=True)
        record={'filename':spec['filename'],'contiguous_prefix_bytes':prefix,'source_kept':True,
                'destination':destination.relative_to(ROOT).as_posix(),'chunks_verified':0}
        with FileLock(lock,timeout=0):
            existing=destination.stat().st_size if destination.exists() else 0
            if prefix:
                temp=destination.with_suffix('.native_seed.tmp')
                if temp.exists():raise RuntimeError('Existing migration temp; inspect before proceeding.')
                with source.open('rb') as original,temp.open('xb') as target:
                    for i in range(count):
                        item=chunks[str(i)]
                        original.seek(i*chunk_bytes)
                        block=original.read(item['bytes'])
                        assert len(block)==item['bytes'] and hashlib.sha256(block).hexdigest()==item['sha256']
                        target.write(block)
                        record['chunks_verified']+=1
                    target.flush();os.fsync(target.fileno())
                if existing:
                    with destination.open('rb') as native,temp.open('rb') as expected:
                        remaining=min(existing,prefix)
                        while remaining:
                            size=min(1024*1024,remaining)
                            assert native.read(size)==expected.read(size),'Existing native prefix differs; refused overwrite'
                            remaining-=size
                if existing<prefix:
                    temp.replace(destination)
                else:
                    temp.unlink()
            record['native_resume_bytes']=destination.stat().st_size if destination.exists() else 0
        records.append(record)
    report={'status':'native_resume_seed_verified_original_chunks_retained','model_id':inventory['model_id'],
            'resolved_commit':revision,'files':records,'native_seed_bytes':sum(r['native_resume_bytes'] for r in records),
            'full_weights_verified':False,'model_loads':0,'captions':0,'source_dataset_changed':False}
    path=OUT/'native_hub_20261007/native_seed_report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
