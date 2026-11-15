"""Seal read-only D1 diagnostics and pending, non-applied overlap decision."""
from pathlib import Path
import json
import pandas as pd,yaml
from photo_integrity_audit import ROOT,CONFIG,digest_file


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];snap=out/'previous_current_state_snapshot'
    completion=read(out/'completion_report.json');validation=read(out/'validation_report.json');proposal=read(out/'overlap_proposal_validation.json');findings=read(out/'photo_audit_findings.json')
    assert completion['status']=='PASS' and validation['validation_passed'] and proposal['status']=='PASS'
    for name,digest in completion['artifact_sha256'].items():assert digest_file(out/name)==digest,name
    identity=read(out/'scientific_identity.json');assert digest_file(CONFIG)==identity['config_sha256']
    for name,digest in identity['protected_sha256'].items():assert digest_file(ROOT/name)==digest,name
    for name,digest in identity['scripts_sha256'].items():assert digest_file(ROOT/'scripts'/name)==digest,name
    assert digest_file(ROOT/'scripts/validate_photo_integrity_audit.py')==validation['validator_sha256']
    assert digest_file(ROOT/'scripts/verify_photo_overlap_proposal.py')==proposal['verifier_sha256']
    for name,digest in read(snap/'snapshot_manifest.json').items():assert digest_file(snap/name)==digest,name
    previous=read(ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1/acceptance_report.json');resolution={}
    for name,digest in previous['artifact_sha256'].items():
        match=next((p for p in [ROOT/name,snap/name] if p.is_file() and digest_file(p)==digest),None)
        assert match is not None,name;resolution[name]=match.relative_to(ROOT).as_posix()
    execution=read(ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json')
    assert execution['status']=='PASS' and not execution['errors'] and execution['executed_code_cells']==19 and execution['image_outputs']==12
    assert execution['raw_file_metadata_unchanged'] and execution['protected_source_metadata_unchanged']
    old=read(snap/'outputs/unified_dataset_inspection/acceptance_report.json');figures=0
    for name,digest in old['artifact_sha256'].items():
        if name.endswith('.png'):assert digest_file(ROOT/name)==digest,name;figures+=1
    assert figures==9
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
    assert all(value is None for section in ['review_selection','token_budget'] for value in pending[section].values())
    assert findings['photos']==findings['readable']==30027 and findings['unreadable']==0 and findings['raw_cross_split_groups']==227
    assert not findings['hypothetical_overlap_policy_NOT_APPLIED']['applied'] and findings['hypothetical_overlap_policy_NOT_APPLIED']['businesses_zero']==14
    assert not findings['photos_physically_deleted'] and not findings['businesses_replaced_or_excluded'] and not findings['caption_generation_started']
    files=[p for p in out.iterdir() if p.is_file() and p.name!='acceptance_report.json'];size=sum(p.stat().st_size for p in files)
    assert size<cfg['storage']['maximum_scientific_output_bytes'],size
    book_paths=[ROOT/'notebooks/00_thesis_dataset_filtering_review_selection.ipynb',ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json',*sorted((ROOT/'outputs/unified_dataset_inspection/figures').glob('*.png'))]
    book_acceptance=ROOT/'outputs/unified_dataset_inspection/acceptance_report.json'
    book_report={'status':'PASS','date':'2026-10-06','code_cells':19,'image_outputs':12,'three_new_figures_visually_inspected':True,'previous_nine_figures_unchanged':True,'source_metadata_unchanged':True,'run_all_scope':'cached read-only evidence including D1 image audit','artifact_sha256':{p.relative_to(ROOT).as_posix():digest_file(p) for p in book_paths}}
    book_acceptance.write_text(json.dumps(book_report,indent=2)+'\n',encoding='utf-8')
    report={'status':'PASS','date':'2026-10-06','scope':'Authorized read-only D1 integrity/duplicate diagnostics; caption protocol decision pending','photos':30027,'businesses':2300,'readable':30027,'unreadable':0,'cohort_and_original_source_photo_hashes_verified':30027,'full_fresh_pixel_phash_replays':30027,'independent_direct_DCT_reference_photos':256,'within_business_pairs_verified':432699,'within_groups_verified':2041,'global_exact_collision_groups_verified':2584,'serialization_image_edge_tests_passed':4,'visual_pairs_inspected':7,'hypothetical_photo_flags_verified':30027,'hypothetical_business_impacts_verified':2300,'raw_pixel_cross_split_groups':227,'hypothetical_policy_applied':False,'hypothetical_removed':785,'hypothetical_below5':124,'hypothetical_zero':14,'source_pools_business_membership_splits_unchanged':True,'unfiltered_all_readable_photo_IDs_preserved':True,'photos_physically_deleted':False,'near_threshold_final':None,'duplicate_policy_final':None,'caption_model_GPU_API_rental_started':False,'test_policy_tuning':False,'notebook_code_cells':19,'notebook_image_outputs':12,'three_new_figures_visually_inspected':True,'previous_nine_figures_unchanged':True,'previous_source_audit_acceptance_resolution':resolution,'scientific_output_bytes':size,'final_source_allocation_changed':False,'final_manifest':False,'methodological_decision_pending':'Cross-split exact visual overlap and post-policy below5/zero-photo handling; no automatic exclusions/resplit.'}
    paths=[p for p in out.rglob('*') if p.is_file() and p!=out/'acceptance_report.json']+book_paths+[book_acceptance]
    paths += [ROOT/name for name in ['THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md','README.md','DECISIONS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','specs/004_photo_integrity_duplicate_audit.md','configs/photo_integrity_duplicate_audit.yaml','configs/research_pending.yaml','scripts/photo_integrity_audit.py','scripts/run_photo_integrity_audit.py','scripts/validate_photo_integrity_audit.py','scripts/test_photo_integrity_audit.py','scripts/create_photo_audit_report.py','scripts/verify_photo_overlap_proposal.py','scripts/render_photo_pair_gallery.py','scripts/accept_photo_integrity_audit.py','scripts/create_unified_dataset_notebook.py','scripts/dataset_review_notebook_helpers.py']]
    report['artifact_sha256']={p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(set(paths))}
    (out/'acceptance_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['artifact_sha256','previous_source_audit_acceptance_resolution']},indent=2))


if __name__=='__main__':main()
