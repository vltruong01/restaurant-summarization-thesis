"""Seal expanded diagnostics after independent validation and notebook review."""
from pathlib import Path
import argparse, hashlib, json, math
import pandas as pd
import orjson, yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_c/review_selection_source_length_expanded_v1'
SNAP = OUT / 'previous_primary_notebook_snapshot'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_summary(frame, path, dimensions):
    summary = pd.read_csv(path, keep_default_na=False)
    groups = frame.groupby(dimensions, dropna=False)
    assert len(summary) == len(groups), str(path)
    for _, actual in summary.iterrows():
        key = tuple(actual[d] for d in dimensions)
        part = groups.get_group(key if len(key) > 1 else key[0])
        feasible = part[part.selection_status.eq('feasible')]
        expected = dict(
            businesses=len(part), retained_mean=part.final_retained_review_count.mean(),
            retained_min=part.final_retained_review_count.min(), retained_max=part.final_retained_review_count.max(),
            below10=part.below_10_retained_flag.sum(), below5=part.below_5_retained_flag.sum(),
            infeasible=part.selection_status.ne('feasible').sum(),
            tokens_min=feasible.tokens_after_budget_enforcement.min(), tokens_max=feasible.tokens_after_budget_enforcement.max(),
            rating_jsd_mean=feasible.rating_jsd_after.mean(), temporal_jsd_mean=feasible.temporal_jsd_after.mean(),
            length_jsd_mean=feasible.length_jsd.mean(), median_mean_ratio=feasible.selected_source_mean_ratio.median(),
            source_short20_macro=feasible.source_lt20_fraction.mean(), selected_short20_macro=feasible.selected_lt20_fraction.mean(),
            exactly30_under1500=(part.final_retained_review_count.eq(30) & part.tokens_after_budget_enforcement.lt(1500)).sum())
        for name, low, high in [('25-30',25,30),('20-24',20,24),('15-19',15,19),('10-14',10,14),('5-9',5,9),('<5',0,4)]:
            expected['band_' + name] = part.final_retained_review_count.between(low,high).sum()
        for column, value in expected.items():
            assert math.isclose(float(actual[column]), float(value), rel_tol=1e-12, abs_tol=1e-12), (path.name,key,column)
    return len(summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--science-only', action='store_true')
    parser.add_argument('--figure-reviewed', action='store_true')
    args=parser.parse_args()
    validation=read(OUT/'validation_report.json'); assert validation['validation_passed']
    completion=read(OUT/'completion_report.json')
    for name, sha in completion['artifact_sha256'].items():
        assert digest(OUT/name)==sha, name
    identity=read(OUT/'scientific_identity.json')['identity']
    for name, sha in identity['protected_sha256'].items(): assert digest(ROOT/name)==sha, name
    for name, sha in identity['scripts'].items(): assert digest(ROOT/'scripts'/name)==sha, name
    assert digest(ROOT/'configs/review_selection_source_length_expanded.yaml')==identity['config_sha256']
    for name, sha in identity['legacy_config_sha256'].items(): assert digest(ROOT/'configs'/name)==sha, name
    old_partial=read(OUT/'pre_memoization_run/partial_artifact_sha256.json')
    for name, sha in old_partial.items(): assert digest(OUT/name)==sha, name
    records=[orjson.loads(line) for line in (OUT/'selection_runs.jsonl').open('rb')]
    assert len(records)==39600
    frame=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in records])
    assert frame.business_id.nunique()==2200
    summary_rows=verify_summary(frame,OUT/'budget_method_summary.csv',['method','review_budget'])
    grouped={d:verify_summary(frame,OUT/f'summary_by_{d}.csv',['method','review_budget',d])
             for d in ['split','volume_band','expanded_witness_origin']}
    before_after=pd.read_csv(OUT/'primary_before_after_summary.csv'); assert len(before_after)==9
    for _, actual in before_after.iterrows():
        part=frame[frame.method.eq(actual['method']) & frame.review_budget.eq(actual.review_budget)]
        assert len(part)==2200 and int(actual.businesses)==2200
        for label,column in [('initial_eligible_count_mean','initial_eligible_review_count'),('initial_candidate_count_mean','initial_candidate_count')]:
            assert math.isclose(actual[label],part[column].mean(),abs_tol=1e-12)
        for stage in ['before','after']:
            values=part['tokens_'+stage+'_budget_enforcement']
            assert int(actual['tokens_'+stage+'_min'])==values.min() and int(actual['tokens_'+stage+'_max'])==values.max()
        for label in ['rating','temporal']:
            before=part[label+'_jsd_before']; after=part[label+'_jsd_after']; delta=after-before
            assert math.isclose(actual[label+'_jsd_before_mean'],before.mean(),abs_tol=1e-12)
            assert math.isclose(actual[label+'_jsd_after_mean'],after.mean(),abs_tol=1e-12)
            assert actual[label+'_improved_vs_reference']==(delta<-1e-12).sum()
            assert actual[label+'_equal_vs_reference']==(delta.abs()<=1e-12).sum()
            assert actual[label+'_worse_vs_reference']==(delta>1e-12).sum()
    indexed={(r['business_id'],r['method'],r['review_budget']):r for r in records}
    assert len(indexed)==39600
    initial_checks=baseline_checks=pair_checks=0
    for bid in frame.business_id.unique():
        previous=read(ROOT/'outputs/stage_c/review_selection_train_validation_v3/business_runs'/(hashlib.sha256(bid.encode()).hexdigest()+'.json'))
        old={r['review_budget']:r for r in previous['runs'] if r['method']=='budget_aware'}
        if not old:
            old={r['review_budget']:r for r in previous['runs'] if r['method']=='budget_aware_rating_time'}
        assert len(old)==3, sorted(set(r['method'] for r in previous['runs']))
        for budget in [3500,4000,4500]:
            reference=old[budget]
            for method in ['v4_2_primary_control','v4_2_existing_length_tie','v4_2_length_balanced']:
                row=indexed[bid,method,budget]
                for field in ['initial_reference_review_ids','initial_eligible_review_count','initial_candidate_count','tokens_before_budget_enforcement','rating_jsd_before','temporal_jsd_before']:
                    assert row[field]==reference[field], (bid,method,budget,field)
                assert row['final_model_input'] is False and row['methodology_locked'] is False and row['final_review_budget'] is None
                initial_checks+=1
            baseline=indexed[bid,'v3_budget_aware',budget]
            for field in ['selected_review_ids','final_retained_review_count','tokens_after_budget_enforcement','rating_jsd_after','temporal_jsd_after']:
                assert baseline[field]==reference[field], (bid,budget,field)
            baseline_checks+=1
            control=indexed[bid,'v4_2_primary_control',budget]; final=indexed[bid,'v4_2_length_balanced',budget]
            for field in ['final_retained_review_count','rating_jsd_after','temporal_jsd_after']:
                assert final[field]==control[field],(bid,budget,field)
            assert final['length_jsd']<=control['length_jsd']+1e-12
            pair_checks+=1
    science={'status':'PASS','summary_rows_verified':summary_rows,'grouped_summary_rows_verified':grouped,'before_after_summary_rows_verified':len(before_after),
             'matched_initial_reference_rows_verified':initial_checks,'immutable_v3_rows_equal':baseline_checks,
             'primary_count_rating_time_pairs_equal':pair_checks,'previous_partial_artifacts_unchanged':len(old_partial)}
    (OUT/'aggregate_acceptance_checks.json').write_text(json.dumps(science,indent=2)+'\n',encoding='utf-8')
    if args.science_only:
        print(json.dumps(science,indent=2));return
    assert args.figure_reviewed, 'Visually inspect the new plot before sealing.'
    for name, sha in read(SNAP/'snapshot_manifest.json').items(): assert digest(SNAP/name)==sha, name
    previous_acceptance=read(ROOT/'outputs/stage_c/review_selection_source_length_tie_pilot_v1/acceptance_report.json')
    resolutions={}
    for name, sha in previous_acceptance['artifact_sha256'].items():
        found=next((p for p in [ROOT/name,SNAP/name] if p.is_file() and digest(p)==sha),None)
        assert found is not None, name
        resolutions[name]=found.relative_to(ROOT).as_posix()
    execution=read(ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json')
    assert execution['status']=='PASS' and not execution['errors']
    assert execution['executed_code_cells']==16 and execution['image_outputs']==8
    assert execution['protected_source_metadata_unchanged'] and execution['raw_file_metadata_unchanged']
    old_figures=read(SNAP/'outputs/unified_dataset_inspection/acceptance_report.json')['artifact_sha256']
    previous_figures=0
    for name, sha in old_figures.items():
        if name.endswith('.png'):
            assert digest(ROOT/name)==sha, name
            previous_figures+=1
    assert previous_figures==7
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text(encoding='utf-8'))
    assert all(v is None for v in pending['review_selection'].values())
    assert all(v is None for v in pending['token_budget'].values())
    notes=read(OUT/'qualitative_review_notes.json'); prepared=read(OUT/'qualitative_reviews_to_read.json')
    assert notes['read_reviews']==len(prepared)
    assert {r['review_id'] for r in prepared}=={rid for case in notes['cases'] for rid in case['read_review_ids']}
    assert all(case['status']=='assistant_read_complete' for case in notes['cases'])
    notebook=ROOT/'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
    nb_artifacts=[notebook,ROOT/'outputs/unified_dataset_inspection/notebook_execution_report.json',
                  *sorted((ROOT/'outputs/unified_dataset_inspection/figures').glob('*.png'))]
    nb_report={'status':'PASS','date':'2026-10-06','code_cells':16,'image_outputs':8,
               'new_figure_visually_inspected':True,'previous_seven_figures_unchanged':True,
               'source_metadata_unchanged':True,'run_all_scope':'cached read-only evidence',
               'artifact_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in nb_artifacts}}
    nb_report_path=ROOT/'outputs/unified_dataset_inspection/acceptance_report.json'
    nb_report_path.write_text(json.dumps(nb_report,indent=2)+'\n',encoding='utf-8')
    findings=read(OUT/'expanded_findings.json')
    report={**science,'date':'2026-10-06','scope':'authorized expanded2200 train/validation diagnostics only',
            'businesses':2200,'source_originals':575749,'runs_verified':39600,'new_method_runs':19800,
            'independent_validation':validation,'findings':findings,'notebook_code_cells':16,'notebook_figures':8,
            'new_figure_visually_inspected':True,'previous_seven_figures_unchanged':True,
            'previous_acceptance_resolution':resolutions,'accepted_science_and_sources_unchanged':True,
            'qualitative_reviews_read':len(prepared),'qualitative_businesses':len(notes['cases']),
            'final_review_budget':None,'final_manifest':False,'final_methodology_locked':False,'model_stage_run':False}
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='acceptance_report.json']
    paths.extend([ROOT/name for name in ['THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md','README.md','DECISIONS.md',
                'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','specs/003_review_selection.md',
                'specs/003k_source_length_balance_expanded_validation.md','configs/review_selection_source_length_expanded.yaml',
                'configs/research_pending.yaml','scripts/run_source_length_expanded.py','scripts/validate_source_length_expanded.py',
                'scripts/review_scoring_memoization.py','scripts/test_review_scoring_memoization.py',
                'scripts/create_source_length_expanded_report.py','scripts/create_unified_dataset_notebook.py',
                'scripts/dataset_review_notebook_helpers.py','scripts/accept_source_length_expanded.py']])
    paths.extend([*nb_artifacts,nb_report_path])
    report['artifact_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}
    (OUT/'acceptance_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['artifact_sha256','previous_acceptance_resolution']},indent=2))


if __name__=='__main__': main()
