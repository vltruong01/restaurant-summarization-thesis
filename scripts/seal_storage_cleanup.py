"""Seal completed cleanup, updated handoff and cached notebook checks."""
from pathlib import Path
import hashlib,json,shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_cleanup'
SNAP=OUT/'previous_current_state_snapshot'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cleanup=read(OUT/'cleanup_report.json');verification=read(OUT/'retained_science_verification.json')
    assert cleanup['status']==verification['status']=='PASS'
    assert all(not (ROOT/row['path']).exists() for row in cleanup['deleted'])
    # Guard already-hashed frozen artifacts after the handoff/notebook update.
    metadata_checks=0
    for group in ['outputs','data']:
        before=read(OUT/f'compression_{group}_before.json')['files']
        for name,expected in before.items():
            state=(ROOT/name).stat()
            assert state.st_size==expected['logical_bytes'] and state.st_mtime_ns==expected['mtime_ns'],name
            metadata_checks+=1
    former=read(ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/acceptance_report.json')
    resolutions=verification['previous_acceptance_resolution']
    for name,digest in read(SNAP/'snapshot_manifest.json').items():assert sha(SNAP/name)==digest,name
    # Only these authorized mutable paths changed since full retained-byte verification.
    for name in ['THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md','README.md','DECISIONS.md',
                 'scripts/create_unified_dataset_notebook.py','notebooks/00_thesis_dataset_filtering_review_selection.ipynb',
                 'outputs/unified_dataset_inspection/notebook_execution_report.json','outputs/unified_dataset_inspection/acceptance_report.json']:
        if name in former['artifact_sha256']:
            digest=former['artifact_sha256'][name]
            assert sha(SNAP/name)==digest,name
            resolutions[name]=(SNAP/name).relative_to(ROOT).as_posix()
    verification['previous_acceptance_resolution']=resolutions
    verification['post_handoff_retained_metadata_checks']=metadata_checks
    (OUT/'retained_science_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    execution=read(ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json')
    assert execution['status']=='PASS' and not execution['errors']
    assert execution['executed_code_cells']==17 and execution['image_outputs']==8
    assert execution['raw_file_metadata_unchanged'] and execution['protected_source_metadata_unchanged']
    old_book_acceptance=read(SNAP/'outputs/unified_dataset_inspection/acceptance_report.json')
    figures=0
    for name,digest in old_book_acceptance['artifact_sha256'].items():
        if name.endswith('.png'):assert sha(ROOT/name)==digest,name;figures+=1
    assert figures==8
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
    assert all(value is None for value in pending['review_selection'].values())
    assert all(value is None for value in pending['token_budget'].values())
    book_paths=[ROOT/'notebooks/00_thesis_dataset_filtering_review_selection.ipynb',ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json',*sorted((ROOT/'outputs/unified_dataset_inspection/figures').glob('*.png'))]
    book_report={'status':'PASS','date':'2026-10-06','code_cells':17,'image_outputs':8,'previous_eight_figures_unchanged':True,'source_metadata_unchanged':True,'run_all_scope':'cached read-only evidence including storage cleanup','artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in book_paths}}
    book_acceptance=ROOT/'outputs/unified_dataset_inspection/acceptance_report.json'
    book_acceptance.write_text(json.dumps(book_report,indent=2)+'\n',encoding='utf-8')
    disk_current=shutil.disk_usage(ROOT).free
    report={'status':'PASS','date':'2026-10-06','scope':'User-authorized project storage maintenance only','deleted_targets':cleanup['deleted'],'deleted_logical_bytes':cleanup['deleted_logical_bytes'],'compression_storage_bytes_saved':cleanup['compression_storage_bytes_saved'],'disk_free_before':cleanup['disk_free_before'],'disk_free_at_cleanup_measurement':cleanup['disk_free_after'],'disk_free_current':disk_current,'disk_free_increase_current':disk_current-cleanup['disk_free_before'],'retained_scientific_artifact_paths_verified':verification['retained_scientific_artifact_paths_verified'],'post_handoff_retained_metadata_checks':metadata_checks,'raw_processed_content_sha256_preserved_files':verification['raw_processed_content_sha256_preserved_files'],'original_photos_names_sizes_retained':200098,'archive_member_hash_checks':verification['archive_member_hash_checks'],'all_photo_hashes_claimed':False,'partial_different_provenance_files_preserved':6,'previous_acceptance_resolves':True,'notebook_cells':17,'notebook_image_outputs':8,'previous_eight_figures_unchanged':True,'methodology_changed':False,'final_manifest':False,'final_review_budget':None,'system_restore_or_external_user_files_deleted':False}
    paths=[p for p in OUT.rglob('*') if p.is_file() and p!=OUT/'acceptance_report.json']
    paths += book_paths+[book_acceptance]+[ROOT/name for name in ['THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md','README.md','DECISIONS.md','docs/STORAGE_POLICY.md','scripts/audit_storage_cleanup.py','scripts/remove_audited_redundant_copies.ps1','scripts/compress_research_storage.py','scripts/verify_storage_cleanup.py','scripts/seal_storage_cleanup.py','scripts/create_unified_dataset_notebook.py','configs/research_pending.yaml']]
    report['artifact_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    (OUT/'acceptance_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':main()
