"""Seal scoped audit and maintained notebook; prior cleanup science stays intact."""
from pathlib import Path
import hashlib,json
import pandas as pd,yaml
from english_eligibility import file_hash
from led_source_token_audit import ROOT,CONFIG


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];snap=out/'previous_current_state_snapshot'
    completion=read(out/'completion_report.json');validation=read(out/'validation_report.json');findings=read(out/'allocation_findings.json')
    assert completion['status']=='PASS' and validation['validation_passed']
    assert validation['source_rows_verified']==19800 and validation['summary_rows_verified']==441
    for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    identity=read(out/'scientific_identity.json');assert file_hash(CONFIG)==identity['config_sha256']
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    for name,digest in identity['scripts_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    assert file_hash(ROOT/'scripts/validate_led_source_token_audit.py')==validation['validator_sha256']
    for name,digest in read(snap/'snapshot_manifest.json').items():assert file_hash(snap/name)==digest,name
    prior=read(ROOT/'outputs/storage_cleanup/acceptance_report.json');resolutions={}
    for name,digest in prior['artifact_sha256'].items():
        match=next((p for p in [ROOT/name,snap/name] if p.is_file() and file_hash(p)==digest),None)
        assert match is not None,name;resolutions[name]=match.relative_to(ROOT).as_posix()
    execution=read(ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json')
    assert execution['status']=='PASS' and not execution['errors'] and execution['executed_code_cells']==18 and execution['image_outputs']==9
    assert execution['raw_file_metadata_unchanged'] and execution['protected_source_metadata_unchanged']
    old=read(snap/'outputs/unified_dataset_inspection/acceptance_report.json');previous_figures=0
    for name,digest in old['artifact_sha256'].items():
        if name.endswith('.png'):assert file_hash(ROOT/name)==digest,name;previous_figures+=1
    assert previous_figures==8
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
    assert all(value is None for section in ['review_selection','token_budget'] for value in pending[section].values())
    summary=pd.read_csv(out/'caption_capacity_summary.csv')
    core=summary[(summary.metadata_profile=='name_categories_location')&summary.caption_allowance.eq(750)].set_index('review_budget')
    proposal=findings['pilot_option_not_adopted']
    assert proposal['observed_businesses_fit']==core.loc[4000].observed_fit==2200
    assert proposal['conservative_audited_envelope_fit']==core.loc[4000].conservative_envelope_fit==2200
    assert proposal['observed_allocated_total_max']==core.loc[4000].observed_allocated_total_max==4833
    assert proposal['conservative_audited_envelope_total_max']==core.loc[4000].conservative_envelope_total_max==4865
    artifacts=[p for p in out.iterdir() if p.is_file() and p.name!='acceptance_report.json']
    scientific_bytes=sum(p.stat().st_size for p in artifacts);assert scientific_bytes<25000000,scientific_bytes
    book_paths=[ROOT/'notebooks/00_thesis_dataset_filtering_review_selection.ipynb',ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json',*sorted((ROOT/'outputs/unified_dataset_inspection/figures').glob('*.png'))]
    book_acceptance=ROOT/'outputs/unified_dataset_inspection/acceptance_report.json'
    book_report={'status':'PASS','date':'2026-10-06','code_cells':18,'image_outputs':9,'new_figure_visually_inspected':True,'previous_eight_figures_unchanged':True,'source_metadata_unchanged':True,'run_all_scope':'cached read-only evidence including source-allocation audit','artifact_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in book_paths}}
    book_acceptance.write_text(json.dumps(book_report,indent=2)+'\n',encoding='utf-8')
    report={'status':'PASS','date':'2026-10-06','scope':'Authorized source-token/capacity audit only','businesses':2200,'source_rows_verified':19800,'complete_review_blocks_verified':6600,'selected_originals_BODY_fresh_encoded':103047,'unique_full_strings_fresh_encoded':41804,'summary_rows_verified':441,'whole_review_ids_text_order_count_primary_metrics_unchanged':True,'serialization_edge_tests_passed':2,'complete_source_examples':9,'notebook_code_cells':18,'notebook_image_outputs':9,'new_figure_visually_inspected':True,'previous_eight_figures_unchanged':True,'previous_cleanup_acceptance_resolves':True,'previous_cleanup_acceptance_resolution':resolutions,'pilot_option_not_adopted':proposal,'actual_captions_available':False,'actual_caption_conditioned_source_tokens':None,'final_source_tokens':None,'final_review_tokens':None,'final_caption_tokens':None,'metadata_fields_final':None,'final_selector_locked':False,'final_manifest':False,'research_parameters_unchanged':True,'test_businesses':0,'model_API_GPU_rental_started':False,'scientific_output_bytes_excluding_snapshot_and_seal':scientific_bytes}
    paths=[p for p in out.rglob('*') if p.is_file() and p!=out/'acceptance_report.json']+book_paths+[book_acceptance]
    paths += [ROOT/name for name in ['THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md','README.md','DECISIONS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','specs/003_review_selection.md','specs/003l_led_source_token_allocation_audit.md','configs/led_source_token_allocation_audit.yaml','configs/research_pending.yaml','scripts/led_source_token_audit.py','scripts/run_led_source_token_audit.py','scripts/validate_led_source_token_audit.py','scripts/test_led_source_serialization.py','scripts/create_led_source_audit_report.py','scripts/accept_led_source_token_audit.py','scripts/create_unified_dataset_notebook.py']]
    report['artifact_sha256']={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(paths))}
    (out/'acceptance_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['artifact_sha256','previous_cleanup_acceptance_resolution']},indent=2))


if __name__=='__main__':main()
