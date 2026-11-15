"""Import checksum-verified owned Colab outputs only; preserve original local/validation evidence."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,io,json,zipfile
import blip2_caption_pilot as legacy

ROOT=Path(__file__).resolve().parents[1]
REL='outputs/stage_f/production_caption_construction_d033_colab_v1'
OUT=ROOT/REL
REVIEW=ROOT/'outputs/stage_f/colab_result_review_20261010'


def run(archive):
    latest={};caption_hashes={};bundles=0
    with zipfile.ZipFile(archive) as outer:
        members=sorted(n for n in outer.namelist() if '/colab_checkpoints/checkpoints_' in n and n.endswith('.zip'))
        assert members
        for i,member in enumerate(members,1):
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                manifest=json.loads(inner.read('checkpoint_manifest.json'))
                assert manifest['namespace']==REL
                for name,h in manifest['files'].items():
                    p=PurePosixPath(name)
                    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and name.startswith(REL+'/')
                    data=inner.read(name);assert hashlib.sha256(data).hexdigest()==h
                    if '/caption_checkpoints/' in name:
                        if name in caption_hashes:assert caption_hashes[name]==h,'Conflicting completed caption bytes'
                        caption_hashes[name]=h
                    latest[name]=(member,h)
            bundles+=1
    assert len(caption_hashes)==27318,'Full Colabcaption export required; do not import incomplete export as completion'
    completed_name=REL+'/completion_report.json';assert completed_name in latest
    grouped={}
    for name,(member,h) in latest.items():grouped.setdefault(member,[]).append((name,h))
    copied=0
    with zipfile.ZipFile(archive) as outer:
        for member,rows in sorted(grouped.items()):
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                for name,h in rows:
                    target=ROOT/name;assert target.resolve().is_relative_to(OUT.resolve())
                    data=inner.read(name);assert hashlib.sha256(data).hexdigest()==h
                    if target.exists():assert legacy.sha(target)==h,f'Existing project output differs: {name}'
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.import.tmp');temp.write_bytes(data);temp.replace(target)
                    copied+=1
                    if copied%5000==0:print(f'Imported verified latest files {copied}/{len(latest)}',flush=True)
    completion=legacy.read(OUT/'completion_report.json')
    for name,h in completion['artifact_sha256'].items():assert legacy.sha(OUT/name)==h,name
    report={'status':'PASS_COMPLETE_COLAB_OUTPUT_IMPORT_PENDING_INDEPENDENT_SOURCE_AUDIT','archive':str(archive),
            'archive_sha256':legacy.sha(archive),'checkpoint_bundles':bundles,'latest_files_imported':len(latest),
            'new_caption_checkpoints':len(caption_hashes),'final_manifest_checksums_passed':len(completion['artifact_sha256']),
            'original_local_validation_records_modified':False,'model_inference_executed':False,'teacher_training':False}
    legacy.save(REVIEW/'complete_import_report.json',report);print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args();run(a.archive)
