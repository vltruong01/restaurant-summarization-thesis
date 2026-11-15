"""Read-only dependency/content checks before removing redundant local copies."""
from pathlib import Path, PurePosixPath
import collections, hashlib, json, os, random, shutil, tarfile, time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_cleanup'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    archive=ROOT/'data/yelp_photos/yelp_photos';extracted=archive.parent
    assert archive.is_file() and (extracted/'photos').is_dir()
    samples=[];rng=random.Random(42);names=set();members=0;photo_count=0;body_bytes=0;metadata=[]
    with tarfile.open(archive,'r:') as tar, (OUT/'removed_archive_member_manifest.jsonl').open('w',encoding='utf-8') as manifest:
        for member in tar:
            if member.isdir():continue
            assert member.isfile(),(member.name,'non-regular archive member')
            parts=PurePosixPath(member.name).parts
            assert parts and not PurePosixPath(member.name).is_absolute() and '..' not in parts
            target=extracted.joinpath(*parts).resolve()
            assert target.is_relative_to(extracted.resolve()) and target.is_file(),member.name
            assert target.stat().st_size==member.size,member.name
            assert member.name not in names,member.name
            names.add(member.name);members+=1;body_bytes+=member.size
            row={'name':member.name,'bytes':member.size,'offset_data':member.offset_data}
            manifest.write(json.dumps(row)+'\n')
            if parts[0]=='photos':photo_count+=1
            else:metadata.append(row)
            if len(samples)<512:samples.append(row)
            else:
                index=rng.randrange(members)
                if index<512:samples[index]=row
            if members%50000==0:print(f'Archive names/sizes checked {members}.',flush=True)
    actual_photos={p.relative_to(extracted).as_posix() for p in (extracted/'photos').iterdir() if p.is_file()}
    archived_photos={name for name in names if PurePosixPath(name).parts[0]=='photos'}
    assert actual_photos==archived_photos
    checked=[]
    with archive.open('rb') as stream:
        for row in sorted({r['name']:r for r in [*samples,*metadata]}.values(),key=lambda r:r['offset_data']):
            stream.seek(row['offset_data']);raw=stream.read(row['bytes']);assert len(raw)==row['bytes']
            digest=hashlib.sha256(raw).hexdigest();assert sha(extracted/row['name'])==digest,row['name']
            checked.append({'name':row['name'],'sha256':digest})
    partial=ROOT/'data/processed/.yelp_2300_v1.serial_attempt';canonical=ROOT/'data/processed/yelp_2300_v1'
    assert partial.is_dir() and canonical.is_dir()
    partial_manifest=[];matching=0;preserved=[]
    preservation=OUT/'discarded_serial_attempt_provenance'
    for directory,_,files in os.walk(partial):
        for name in files:
            path=Path(directory)/name;relative=path.relative_to(partial);current=canonical/relative
            old_hash=sha(path);same=current.is_file() and sha(current)==old_hash
            row={'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':old_hash,'canonical_byte_equal':same}
            if same:matching+=1
            else:
                # Preserve small provenance/config differences; unknown data blocks deletion.
                assert path.stat().st_size<5*1024*1024 and relative.parts[0] not in ['photos','reviews'],relative
                target=preservation/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                assert sha(target)==old_hash;row['preserved_at']=target.relative_to(ROOT).as_posix();preserved.append(row['preserved_at'])
            partial_manifest.append(row)
    partial_bytes=sum(r['bytes'] for r in partial_manifest)
    (OUT/'removed_serial_attempt_manifest.json').write_text(json.dumps(partial_manifest,indent=2)+'\n',encoding='utf-8')
    report={'status':'PASS','date':'2026-10-06','authorization':'User requests deletion of unnecessary project files/data to free disk space.',
        'archive':{'path':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'regular_members':members,'photo_members':photo_count,'extracted_member_bytes':body_bytes,'all_extracted_names_sizes_match':True,'sample_seed':42,'sampled_members':len(samples),'sample_plus_metadata_sha256_checks':checked,'all_members_hash_checked':False},
        'serial_attempt':{'path':partial.relative_to(ROOT).as_posix(),'bytes':partial_bytes,'files':len(partial_manifest),'canonical_equal_files':matching,'different_small_provenance_preserved':preserved},
        'delete_targets':[archive.relative_to(ROOT).as_posix(),partial.relative_to(ROOT).as_posix()],
        'keep':['all extracted raw Yelp photos/photos.json/agreement','all five raw Yelp JSON files','current yelp_2300_v1 and yelp_2300_english_v1/splits','all accepted/intermediate scientific outputs and witnesses/manifests','tokenizer/environment/proposals/code/specs/docs/notebooks'],
        'compression':'Transparent NTFS compression of text/JSON/CSV/SQLite research artifacts; paths and file contents unchanged; no ZIP relocation or dependency removal.',
        'elapsed_seconds':round(time.perf_counter()-started,3)}
    (OUT/'pre_deletion_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['archive','serial_attempt']},indent=2))
    print(json.dumps({'archive_members':members,'photos':photo_count,'sampled':len(samples),'partial_files':len(partial_manifest),'matching':matching,'provenance_preserved':len(preserved),'deletable_bytes':archive.stat().st_size+partial_bytes},indent=2))


if __name__=='__main__':main()
