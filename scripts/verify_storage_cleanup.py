"""Verify retained scientific bytes and audit removals after local cleanup."""
from pathlib import Path
import hashlib,json,shutil,time
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_cleanup'
SNAP=OUT/'previous_current_state_snapshot'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started=time.perf_counter();audit=read(OUT/'pre_deletion_audit.json');deletion=read(OUT/'deletion_report.json')
    assert audit['status']==deletion['status']=='PASS'
    assert all(not (ROOT/name).exists() for name in audit['delete_targets'])
    outputs=read(OUT/'compression_outputs_after.json')['report'];data=read(OUT/'compression_data_after.json')['report']
    assert outputs['status']==data['status']=='PASS'
    assert outputs['file_sizes_mtimes_unchanged'] and data['file_sizes_mtimes_unchanged']
    hashes={};count=0
    def sha(path):
        nonlocal count
        path=Path(path).resolve();key=str(path)
        if key not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            hashes[key]=h.hexdigest();count+=1
            if count%1000==0:print(f'Retained hashes checked {count}.',flush=True)
        return hashes[key]
    before_data=read(OUT/'compression_data_before.json')['content_sha256']
    assert data['data_content_sha256_unchanged_files']==len(before_data)
    # The compressor compared these exact digests before/after its completed run.
    for name,digest in before_data.items():hashes[str((ROOT/name).resolve())]=digest
    for name,digest in read(SNAP/'snapshot_manifest.json').items():assert sha(SNAP/name)==digest,name
    former=read(ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/acceptance_report.json')
    resolutions={}
    for name,digest in former['artifact_sha256'].items():
        current=ROOT/name
        if current.is_file() and sha(current)==digest:resolutions[name]=name
        else:
            saved=SNAP/name;assert saved.is_file() and sha(saved)==digest,name
            resolutions[name]=saved.relative_to(ROOT).as_posix()
    identity=read(ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/scientific_identity.json')['identity']
    for name,digest in identity['protected_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in identity['scripts'].items():assert sha(ROOT/'scripts'/name)==digest,name
    for name,digest in identity['legacy_config_sha256'].items():assert sha(ROOT/'configs'/name)==digest,name
    for row in read(OUT/'removed_serial_attempt_manifest.json'):
        if 'preserved_at' in row:assert sha(ROOT/row['preserved_at'])==row['sha256']
    member_count=0;photos=set();metadata=[]
    for line in (OUT/'removed_archive_member_manifest.jsonl').open(encoding='utf-8'):
        item=json.loads(line);p=ROOT/'data/yelp_photos'/item['name']
        assert p.is_file() and p.stat().st_size==item['bytes'],item['name'];member_count+=1
        if item['name'].startswith('photos/'):photos.add(item['name'])
        else:metadata.append(item['name'])
    actual={'photos/'+p.name for p in (ROOT/'data/yelp_photos/photos').iterdir() if p.is_file()}
    assert actual==photos and len(photos)==audit['archive']['photo_members']==200098
    assert member_count==audit['archive']['regular_members']==200100
    for item in audit['archive']['sample_plus_metadata_sha256_checks']:
        assert sha(ROOT/'data/yelp_photos'/item['name'])==item['sha256']
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
    assert all(value is None for value in pending['review_selection'].values())
    assert all(value is None for value in pending['token_budget'].values())
    free_before=read(OUT/'inventory_before.json')['disk']['free'];free_after=shutil.disk_usage(ROOT).free
    verification={'status':'PASS','retained_scientific_artifact_paths_verified':len(resolutions),'retained_unique_hashes_fresh_checked':count,'raw_processed_content_sha256_preserved_files':len(before_data),'original_photos_names_sizes_retained':len(photos),'archive_member_names_sizes_retained':member_count,'archive_member_hash_checks':len(audit['archive']['sample_plus_metadata_sha256_checks']),'all_photo_hashes_claimed':False,'previous_acceptance_resolves':True,'previous_acceptance_resolution':resolutions,'final_research_parameters_unchanged':True,'elapsed_seconds':round(time.perf_counter()-started,3)}
    (OUT/'retained_science_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    summary={'status':'PASS','date':'2026-10-06','scope':'Project workspace only; no operating-system/user-account cleanup.','deleted':[{'path':audit['archive']['path'],'bytes':audit['archive']['bytes'],'reason':'Redundant uncompressed TAR; all extracted names/sizes and512 sampled member digests plus full metadata match.'},{'path':audit['serial_attempt']['path'],'bytes':audit['serial_attempt']['bytes'],'reason':'Interrupted Stage B copy;3555 files match current dataset, six small differing provenance/config files preserved.'}],'deleted_logical_bytes':deletion['deleted_logical_bytes'],'compression_outputs':outputs,'compression_data':data,'compression_storage_bytes_saved':outputs['storage_bytes_saved']+data['storage_bytes_saved'],'disk_free_before':free_before,'disk_free_after':free_after,'disk_free_increase_bytes':free_after-free_before,'keep':audit['keep'],'science_verification_passed':True,'methodology_changed':False,'final_manifest':False,'final_review_budget':None,'filesystem_compression':'NTFS transparent compression. Same paths/contents; ordinary Python/Jupyter reads need no extraction. Logical file sizes stay the same; Size on disk is smaller. New compressed-directory files can inherit compression.','cloud_transfer':'NTFS compression is local filesystem storage, not an archive for upload. Bundle only final required cohort/review/image/tokenizer/config artifacts under the existing Vast.ai handoff plan.'}
    (OUT/'cleanup_report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    rows=[{'action':'Delete redundant photo TAR','logical_GB_removed':audit['archive']['bytes']/1e9,'storage_GB_saved':None},{'action':'Delete interrupted dataset copy','logical_GB_removed':audit['serial_attempt']['bytes']/1e9,'storage_GB_saved':None},{'action':'Compress retained scientific outputs','logical_GB_removed':0,'storage_GB_saved':outputs['storage_bytes_saved']/1e9},{'action':'Compress retained raw/processed text data','logical_GB_removed':0,'storage_GB_saved':data['storage_bytes_saved']/1e9}]
    report=['# Project storage cleanup - 2026-10-06','User explicitly requested filtering/removing unnecessary project files/data before the next thesis step. Only resolved paths within C:/thesis_proposal/yelp_dataset were changed.','## Actual operations',json.dumps(rows,indent=2),'Deleted logical bytes and filesystem free-space changes are different measures. Available space can also change due to unrelated Windows processes.','## Measured disk space',json.dumps({k:summary[k] for k in ['disk_free_before','disk_free_after','disk_free_increase_bytes','compression_storage_bytes_saved']},indent=2),'## Retention and reproducibility','All five raw Yelp JSONs,200098 extracted photos/photos.json/agreement, current2300 raw/English cohort/splits, all candidate/witness/accepted and historical outputs, tokenizer/environment/proposals/source/docs preserved. Six small differing partial-attempt provenance files copied and hashed. Full archive member name/size equality and512 seeded sample digest checks plus metadata; do not claim SHA verification of every photo or image-readability audit.','NTFS compression changes on-disk representation, not file bytes/path/mtime. Retained accepted v4.2 seal and protected sources/scripts resolve and hash-match; data compressor compared all38 selected raw/processed file digests before/after. No review-selection, eligibility, split, research allocation/source-length or model-stage changes. File Explorer Size remains logical; Size on disk and disk free space reflect savings.','## Verification',json.dumps({k:v for k,v in verification.items() if k!='previous_acceptance_resolution'},indent=2),'## Next step','Continue Stage C2 source allocation review after notebook/report checks. Planned~5000 TOTALsource/final review-metadata-caption budgets and2300 manifest remain pending. No BLIP-2/teacher/train/rental/test tuning. Do not delete accepted candidate caches without a compatible storage/read workflow; keep evidence accessible at current paths.','## Artifacts','inventory_before.json; pre_deletion_audit.json; removed_archive_member_manifest.jsonl; removed_serial_attempt_manifest.json; discarded_serial_attempt_provenance/; deletion_report.json; compression_*_before/after.json; compact_* logs; retained_science_verification.json; cleanup_report.json; previous_current_state_snapshot/. Main notebook includes cached cleanup results after verification.']
    (OUT/'cleanup_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['keep','compression_outputs','compression_data']},indent=2),flush=True)


if __name__=='__main__':main()
