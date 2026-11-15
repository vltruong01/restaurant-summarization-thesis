"""User-requested exploratory diagnostic only; no protocol or annotation mutations."""
from pathlib import Path
from collections import Counter
import argparse
import platform
import json
import numpy as np
import yaml
from analyze_stage_f_consistency_stability import (
    ROOT, BASE, ASSISTED, POLICY, validate_declared_prefix, consistency_rows, metrics, summary,
)
from stage_f_verification_workflow import read, write, sha, active_ids
from stage_f_math import threshold_search

OUT = ROOT/'outputs/stage_f/consistency_conservative_gate_diagnostic_v1'
GRID = [i/100 for i in range(20,36)]
RULES = {'maximum_balanced_accuracy': None, 'highest_recall_at_least_95pct': .95,
         'highest_recall_at_least_90pct': .90}
BOOTSTRAP = 2000
SEED = 42
CORRECTED = 'IMAGE_NvOxiwPwBMO6Gkv5zb4-Fw'


def table_for(scores, labels, grid=GRID):
    positives=sum(labels); negatives=len(labels)-positives
    assert positives and negatives, 'Both actual classes required; no pseudo-labels'
    result=[]
    for threshold in grid:
        m=metrics(scores,labels,threshold)
        result.append({'threshold':threshold,'TP':m['tp'],'TN':m['tn'],'FP':m['fp'],'FN':m['fn'],
                       'consistent_recall':m['positive_recall'],
                       'inconsistent_recall_specificity':m['negative_specificity'],
                       'precision':m['positive_precision'],'balanced_accuracy':m['balanced_accuracy'],
                       'consistent_rejected_count':m['fn'],'consistent_rejected_pct':100*m['fn']/positives,
                       'inconsistent_accepted_count':m['fp'],'inconsistent_accepted_pct':100*m['fp']/negatives,
                       'consistent_total':positives,'inconsistent_total':negatives})
    return result


def candidates(table):
    # The BA objective and tie order are the existing frozen definition.
    result={'maximum_balanced_accuracy':max(table,key=lambda r:(r['balanced_accuracy'],r['precision'],r['consistent_recall'],r['threshold']))}
    for rule,target in RULES.items():
        if target is None:continue
        eligible=[r for r in table if r['consistent_recall']>=target]
        result[rule]=max(eligible,key=lambda r:r['threshold']) if eligible else None
    return result


def decisions(rows, threshold):
    groups={'TP':[],'TN':[],'FP':[],'FN':[]}
    for row in rows:
        passed=round(float(row['clip_score']),6)>=threshold
        group='TP' if passed and row['positive'] else 'FP' if passed else 'FN' if row['positive'] else 'TN'
        groups[group].append(row['case_id'])
    for ids in groups.values():ids.sort()
    return {'accepted_case_ids':sorted(groups['TP']+groups['FP']),
            'rejected_case_ids':sorted(groups['TN']+groups['FN']),
            'accepted_human_consistent_case_ids':groups['TP'],
            'rejected_human_inconsistent_case_ids':groups['TN'],
            'accepted_human_inconsistent_case_ids':groups['FP'],
            'rejected_human_consistent_case_ids':groups['FN']}


def distributions(records, original_candidate):
    feasible=[r for r in records if r['candidate'] is not None]
    freqs=Counter(r['candidate']['threshold'] for r in feasible)
    result={'draws':len(records),'feasible_draws':len(feasible),'infeasible_draws':len(records)-len(feasible),
            'selected_threshold_distribution':[{'threshold':t,'count':n,'fraction_all_draws':n/len(records)} for t,n in sorted(freqs.items())],
            'same_as_full_fit_fraction_all_draws':(freqs[original_candidate['threshold']]/len(records)) if original_candidate else None,
            'interpretation':'Empirical conditional sensitivity, not held-out accuracy or a guaranteed population confidence interval'}
    if feasible:
        result['refitted_threshold']=summary([r['candidate']['threshold'] for r in feasible])
        for key in ['balanced_accuracy','consistent_recall','inconsistent_recall_specificity','precision']:
            result['resample_fit_'+key]=summary([r['candidate'][key] for r in feasible])
            result['original_reference_'+key]=summary([r['original_reference'][key] for r in feasible])
        result['original_prediction_flips']=summary([r['original_prediction_flips'] for r in feasible])
    return result


def compute(rows, replicates=BOOTSTRAP, seed=SEED):
    scores=[r['clip_score'] for r in rows]; labels=[r['positive'] for r in rows]
    table=table_for(scores,labels); full=candidates(table)
    positive=np.flatnonzero(labels); negative=np.flatnonzero(np.logical_not(labels))
    rng=np.random.default_rng(seed); fixed={k:[] for k in RULES}; refitted={k:[] for k in RULES}
    full_grid_refits=[]; lono=[]
    for draw in range(replicates):
        indices=np.concatenate((rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)))
        ss=[scores[int(i)] for i in indices]; yy=[labels[int(i)] for i in indices]
        resampled=table_for(ss,yy); selected=candidates(resampled)
        for rule in RULES:
            initial=full[rule]
            if initial is not None:
                at_fixed=next(r for r in resampled if r['threshold']==initial['threshold'])
                fixed[rule].append({'replicate':draw+1,'threshold_fixed':initial['threshold'],'metrics':at_fixed})
            record={'replicate':draw+1,'candidate':selected[rule]}
            if selected[rule] is not None:
                record['original_reference']=next(r for r in table if r['threshold']==selected[rule]['threshold'])
                record['threshold_delta']=selected[rule]['threshold']-initial['threshold'] if initial else None
                record['original_prediction_flips']=sum((round(float(s),6)>=selected[rule]['threshold'])!=(round(float(s),6)>=initial['threshold']) for s in scores) if initial else None
            refitted[rule].append(record)
        # Report the inherited -1..1 grid separately; do not rewrite it to the focus range.
        original_best,_=threshold_search(ss,yy,[i/100 for i in range(-100,101)],minimum_class=1)
        full_grid_refits.append({'replicate':draw+1,**original_best})
    for removed in negative:
        record={'removed_case':rows[int(removed)],'remaining_positive':len(positive),'remaining_negative':len(negative)-1,'rules':{}}
        if len(negative)==1:
            record['status']='UNIDENTIFIABLE_ONE_CLASS_FOLD';lono.append(record);continue
        keep=[i for i in range(len(rows)) if i!=removed]
        fold_table=table_for([scores[i] for i in keep],[labels[i] for i in keep]); refit=candidates(fold_table)
        for rule,initial in full.items():
            chosen=refit[rule]
            r={'candidate':chosen,'fixed_threshold_fold_metrics':next(t for t in fold_table if t['threshold']==initial['threshold']) if initial else None}
            if chosen is not None:
                r['threshold_delta']=chosen['threshold']-initial['threshold'] if initial else None
                r['original_reference']=next(t for t in table if t['threshold']==chosen['threshold'])
                r['original_prediction_flips']=sum((round(float(s),6)>=chosen['threshold'])!=(round(float(s),6)>=initial['threshold']) for s in scores) if initial else None
                r['decision_case_ids']=decisions(rows,chosen['threshold'])
            record['rules'][rule]=r
        record['status']='FITTED';lono.append(record)
    boot_summary={}
    for rule in RULES:
        fixed_summary={}
        if fixed[rule]:
            for metric in ['balanced_accuracy','consistent_recall','inconsistent_recall_specificity','precision',
                           'consistent_rejected_count','consistent_rejected_pct','inconsistent_accepted_count','inconsistent_accepted_pct']:
                fixed_summary[metric]=summary([r['metrics'][metric] for r in fixed[rule]])
            target=RULES[rule]
            fixed_summary['target_recall_met_fraction']=sum(r['metrics']['consistent_recall']>=target for r in fixed[rule])/replicates if target else None
        boot_summary[rule]={'fixed_operating_point_metrics':fixed_summary,'refit_selection_rule':distributions(refitted[rule],full[rule])}
        if RULES[rule] is not None:
            valid=[r for r in refitted[rule] if r['candidate'] is not None]
            boot_summary[rule]['refit_selection_rule']['original_reference_target_recall_met_fraction_feasible']=sum(r['original_reference']['consistent_recall']>=RULES[rule] for r in valid)/len(valid) if valid else None
    raw={'fixed_points':fixed,'refitted_rules':refitted,'inherited_full_grid_BA_refits':full_grid_refits}
    return table,full,boot_summary,lono,raw


def report_markdown(report):
    lines=['# CLIP conservative consistency-gate diagnostic', '',
           '**Review only: no threshold chosen or frozen, no protocol amendment, no annotation expansion and no K-study.**','',
           'The corrected IMAGE_NvOxiwPwBMO6Gkv5zb4-Fw is human-verified uncertain and excluded. '
           'Use112verified consistent and5verified inconsistent image-caption cases only; all13uncertain excluded. '
           'All source cases belong to the40validation-calibration businesses; no train/test/K-study outcomes used.','',
           '## Threshold table','',
           'Accept means rounded6-decimal CLIP cosine >= threshold. TP=consistent accepted; TN=inconsistent rejected; '
           'FP=inconsistent accepted; FN=consistent rejected. Rejection percentages use112as denominator; '
           'inconsistent-acceptance percentages use5. Precision uses TP/(TP+FP). BA=(consistent recall+inconsistent specificity)/2.','',
           '| Cutoff | TP | TN | FP | FN | Consistent recall | Inconsistent specificity | Precision | BA | Consistent rejected n (%) | Inconsistent accepted n (%) |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['threshold_table']:
        lines.append(f"| {r['threshold']:.2f} | {r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} | {r['consistent_recall']:.2%} | {r['inconsistent_recall_specificity']:.2%} | {r['precision']:.2%} | {r['balanced_accuracy']:.2%} | {r['consistent_rejected_count']} ({r['consistent_rejected_pct']:.2f}%) | {r['inconsistent_accepted_count']} ({r['inconsistent_accepted_pct']:.2f}%) |")
    lines+=['','## Three comparison points (not final selections)','',
            'BA maximum uses the unchanged BA/precision/recall/higher-cutoff tie order. The two recall-constrained '
            'points are the highest feasible cutoffs on the requested0.20..0.35 diagnostic grid. '
            'No default below-grid fallback is used if a bootstrap draw has no feasible cutoff.','']
    for rule,point in report['operating_points'].items():
        lines += [f'### {rule}','']
        if point is None:lines+=['No feasible point on the requested diagnostic grid.',''];continue
        lines += [f"Threshold **{point['threshold']:.2f}**; BA {point['balanced_accuracy']:.2%}; consistent recall {point['consistent_recall']:.2%}; inconsistent specificity {point['inconsistent_recall_specificity']:.2%}.",'',
                  f"Consistent rejected: {point['consistent_rejected_count']}/{point['consistent_total']} ({point['consistent_rejected_pct']:.2f}%). Inconsistent accepted: {point['inconsistent_accepted_count']}/{point['inconsistent_total']} ({point['inconsistent_accepted_pct']:.2f}%).",'']
        for label,key in [('Human-consistent captions rejected (FN)','rejected_human_consistent_case_ids'),('Human-inconsistent captions accepted (FP)','accepted_human_inconsistent_case_ids'),('Human-inconsistent captions rejected (TN)','rejected_human_inconsistent_case_ids')]:
            ids=point['decision_case_ids'][key]
            lines += [f'**{label}:**','', '\n'.join('- `'+c+'`' for c in ids) if ids else 'None.','']
        for title,key in [('ALL rejected IDs','rejected_case_ids'),('ALL accepted IDs','accepted_case_ids')]:
            lines+=['<details>',f'<summary>{title}</summary>','', '\n'.join('- `'+c+'`' for c in point['decision_case_ids'][key]) or 'None.','', '</details>','']
    lines+=['## Bootstrap (2,000 stratified draws, seed42)','',
            'Each draw resamples the112consistent and5inconsistent cases independently with replacement, preserving class counts. '
            'Report BOTH metrics at the fixed full-data points and re-estimated operating-point thresholds. '
            'Refit metrics on resamples are apparent calibration metrics; original-reference metrics evaluate on the same original calibration set, not an independent holdout. '
            'Central percentile intervals are conditional sensitivity summaries, not guaranteed population CIs.','',
            '| Rule | Fixed cutoff | Fixed recall mean [2.5,97.5 pctile] | Fixed specificity mean [2.5,97.5] | Fixed BA mean [2.5,97.5] |',
            '|---|---:|---|---|---|']
    def interval(s):return f"{s['mean']:.2%} [{s['percentile_2_5']:.2%}, {s['percentile_97_5']:.2%}]"
    for rule,s in report['bootstrap_summary'].items():
        if report['operating_points'][rule] is None:continue
        f=s['fixed_operating_point_metrics']
        lines.append(f"| {rule} | {report['operating_points'][rule]['threshold']:.2f} | {interval(f['consistent_recall'])} | {interval(f['inconsistent_recall_specificity'])} | {interval(f['balanced_accuracy'])} |")
    for rule,s in report['bootstrap_summary'].items():
        r=s['refit_selection_rule'];lines+=['',f'### Refit rule: {rule}','',
            f"Feasible {r['feasible_draws']}/{r['draws']}; infeasible {r['infeasible_draws']}. Same as original point: {r['same_as_full_fit_fraction_all_draws']:.2%}.",'',
            '| Refit cutoff | Draws | Fraction of all draws |','|---:|---:|---:|']
        lines += [f"| {v['threshold']:.2f} | {v['count']} | {v['fraction_all_draws']:.2%} |" for v in r['selected_threshold_distribution']]
        if r['feasible_draws']:
            t=r['refitted_threshold']
            lines += ['',f"Threshold mean {t['mean']:.4f}, median {t['median']:.2f}, range {t['min']:.2f}..{t['max']:.2f}; empirical2.5/97.5percentiles {t['percentile_2_5']:.2f}..{t['percentile_97_5']:.2f}.",
                      f"Refit BA: {interval(r['resample_fit_balanced_accuracy'])}. Original-reference BA: {interval(r['original_reference_balanced_accuracy'])}.",'']
        if RULES[rule] is not None:
            lines += [f"At the fixed original cutoff, resample recall meets the requested target in {s['fixed_operating_point_metrics']['target_recall_met_fraction']:.2%} of draws. "
                      f"The refitted cutoff meets the target when re-evaluated on the original set in {r['original_reference_target_recall_met_fraction_feasible']:.2%} of feasible draws.",'']
    lines+=['## Leave one verified negative out','',
            'All112positives remain in each of5folds, with4negatives. Recall-constrained cutoff choices necessarily stay the same because only the positive score distribution defines their recall constraint; this is not independent evidence of generalization.','',
            '| Removed case | Rule | Refit threshold | Change | Fold consistent recall | Fold specificity | Fold BA | Original-reference prediction flips |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for fold in report['leave_one_negative_out']:
        if fold['status']!='FITTED':continue
        for rule,r in fold['rules'].items():
            c=r['candidate']
            if c:lines.append(f"| {fold['removed_case']['case_id']} | {rule} | {c['threshold']:.2f} | {r['threshold_delta']:+.2f} | {c['consistent_recall']:.2%} | {c['inconsistent_recall_specificity']:.2%} | {c['balanced_accuracy']:.2%} | {r['original_prediction_flips']} |")
    lines+=['','![Diagnostic curves and bootstrap cutoff frequencies](conservative_gate_diagnostics.png)','',
            '## Interpretation for review','',
            'On these verified cases, the high-recall operating points reject some human-consistent captions yet catch none of the five observed inconsistent captions. '
            'The BA-optimal point rejects38of112consistent captions and catches3of5inconsistent captions. '
            'This observed tradeoff does not demonstrate a useful conservative consistency gate on this calibration subset. '
            'It does not establish that CLIP can never be useful, nor does it measure performance on the complete validation pool. '
            'Precision near95%must be interpreted against a consistent-class prevalence of112/117=95.73%; accepting everything already gives that precision.','',
            'Only five negative cases are observed; one case changes specificity by20percentage points on the full set and25points in LONO folds. '
            'Bootstrap repeats these same cases; business dependencies and deliberate score-quantile sampling are not removed by resampling. '
            'Human-confirmed labels remain ground truth for this diagnostic, including uncertainty exclusions. No scorer, model, version, grid config or protocol is changed. '
            'No threshold is automatically chosen; review these results before any Stage F methodology amendment.','',
            'Exact accepted/rejected IDs and original caption/score/label records are in operating_points.json and diagnostic_report.json; every bootstrap draw is in bootstrap_replicates.json.','']
    return '\n'.join(lines)


def plot_report(table, boot_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    x=[r['threshold'] for r in table]
    for key,label in [('consistent_recall','Consistent recall'),('inconsistent_recall_specificity','Inconsistent specificity'),('balanced_accuracy','Balanced accuracy')]:
        axes[0].plot(x,[r[key] for r in table],marker='o',markersize=3,label=label)
    axes[0].axhline(.95,color='grey',linestyle=':',linewidth=.8);axes[0].axhline(.90,color='grey',linestyle='--',linewidth=.8)
    axes[0].set(xlabel='Diagnostic CLIP cutoff',ylabel='Rate',title='112 consistent / 5 inconsistent (verified only)',ylim=(-.02,1.03));axes[0].legend(fontsize=8)
    for rule,s in boot_summary.items():
        d=s['refit_selection_rule']['selected_threshold_distribution']
        axes[1].plot([r['threshold'] for r in d],[r['fraction_all_draws'] for r in d],marker='o',label=rule.replace('_',' '))
    axes[1].set(xlabel='Refitted cutoff',ylabel='Fraction of bootstrap draws',title='2,000 stratified draws, seed42');axes[1].legend(fontsize=8)
    fig.suptitle('Diagnostic comparison only — no threshold selected/frozen, no K-study')
    fig.savefig(OUT/'conservative_gate_diagnostics.png',dpi=180);plt.close(fig)


def run():
    assert not (OUT/'diagnostic_report.json').exists(), 'Use a new version for a new diagnostic run'
    OUT.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/'configs/stage_f_validation_protocol.yaml';workflow=ROOT/'configs/stage_f_human_verification.yaml'
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    paths=[cfg_path,workflow,POLICY,ASSISTED/'latest_human_export.json',ASSISTED/'active_selection.json',
           ASSISTED/'verification_order.json',ASSISTED/'ai_suggestions.json',BASE/'business_roles.json',
           BASE/'human_check/annotation_dataset.json',BASE/'human_check/private_calibration_scores.json',
           BASE/'scores_calibration/caption_scores.json',BASE/'scores_calibration/scoring_report.json',
           ROOT/'scripts/stage_f_math.py',ROOT/'specs/006b_stage_f_validation_execution.md',ROOT/'DECISIONS.md']
    # Seal existing captions, scores, labels, configs and previous diagnostic files.
    original_names=read(ROOT/'outputs/stage_f/consistency_stability_v1/protected_before_sha256.json')
    paths.extend(ROOT/name for name in original_names)
    for folder in ['consistency_stability_v1','consistency_stability_v2']:
        paths.extend(p for p in (ROOT/'outputs/stage_f'/folder).rglob('*') if p.is_file())
    before={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(paths))}
    write(OUT/'protected_before_sha256.json',before)
    settings={'status':'EXPLORATORY_DIAGNOSTIC_NOT_PROTOCOL_AMENDMENT','diagnostic_grid':GRID,
              'inherited_frozen_grid':cfg['calibration_search']['CLIP_grid'],
              'rules':RULES,'bootstrap_replicates':BOOTSTRAP,'bootstrap_seed':SEED,
              'bootstrap_unit':'case','bootstrap_scheme':'independent_class_resampling_with_replacement_preserve_observed_counts',
              'fixed_point_metrics_and_refit_rules_both_reported':True,
              'no_feasible_draw':'record_null_and_infeasible_count_no_out_of_grid_fallback',
              'LONO':'each_verified_negative_once_keep_all_positives',
              'threshold_automatically_chosen':False,'protocol_amended':False,'K_selection':False,
              'case_expansion':False,'models':cfg['models'],'score_definition':cfg['CLIP_score'],
              'rounding_decimals':cfg['decision_round_decimals'],'scoring_configuration_sha256':sha(cfg_path)}
    write(OUT/'diagnostic_settings.json',settings)
    dataset=read(BASE/'human_check/annotation_dataset.json');labels=read(ASSISTED/'latest_human_export.json')
    order=read(ASSISTED/'verification_order.json');current=read(ASSISTED/'active_selection.json');ai=read(ASSISTED/'ai_suggestions.json')
    assert order['workflow_sha256']==sha(workflow) and order['role_manifest_sha256']==sha(BASE/'business_roles.json')
    assert order['private_stratification_scores_sha256']==sha(BASE/'human_check/private_calibration_scores.json')
    scoring=read(BASE/'scores_calibration/scoring_report.json')
    assert scoring['config_sha256']==sha(cfg_path) and scoring['overlength']==0
    assert cfg['decision_round_decimals']==6 and cfg['CLIP_score']=='L2_normalized_image_dot_L2_normalized_caption'
    assert cfg['calibration_search']['criterion']=='maximum_balanced_accuracy'
    assert cfg['calibration_search']['ties']==['higher_positive_precision','higher_positive_recall','higher_threshold']
    actual,prefix=validate_declared_prefix(dataset,order,current,ai,labels)
    assert actual[CORRECTED]['human_verified'] is True and actual[CORRECTED]['human_final_label']['consistency']=='uncertain'
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')}
    rows,uncertain,pending=consistency_rows(dataset,actual,read(BASE/'human_check/private_calibration_scores.json'),roles,set(active_ids(order,current)))
    assert CORRECTED in uncertain and CORRECTED not in {r['case_id'] for r in rows}
    source_scores={r['photo_id']:r for r in read(BASE/'scores_calibration/caption_scores.json')}
    for r in rows:
        original=source_scores[r['photo_id']]
        assert original['score_status']=='scored' and original['split']=='validation'
        assert original['clip_score']==r['clip_score'] and original['business_id']==r['business_id']
        assert original['normalized_caption']==r['caption'] and original['raw_caption']==r['raw_caption']
    table,points,boot,lono,raw=compute(rows)
    for rule,p in points.items():
        if p is not None:points[rule]={**p,'decision_case_ids':decisions(rows,p['threshold'])}
    original_best,original_table=threshold_search([r['clip_score'] for r in rows],[r['positive'] for r in rows],[i/100 for i in range(-100,101)],minimum_class=1)
    original_freq=Counter(r['threshold'] for r in raw['inherited_full_grid_BA_refits'])
    report={'status':'DIAGNOSTIC_ONLY_REVIEW_REQUIRED_NO_THRESHOLD_CHOSEN','verified_annotation_update':'already_applied_explicit_user_correction_no_second_rewrite',
            'corrected_case':actual[CORRECTED],'counts':{'consistent':sum(r['positive'] for r in rows),'inconsistent':sum(not r['positive'] for r in rows),'fit_total':len(rows),'uncertain_excluded':len(uncertain),'pending_deferred':len(pending),'total_human_verified':sum(r['human_verified'] for r in actual.values())},
            'annotator':labels['annotator'],'verified_declared_prefix':prefix,'fit_cases':rows,
            'fit_business_count':len({r['business_id'] for r in rows}),'uncertain_excluded_case_ids':uncertain,'pending_deferred_case_ids':pending,
            'scorer_information':settings,'threshold_table':table,'operating_points':points,
            'bootstrap_summary':boot,'leave_one_negative_out':lono,
            'inherited_full_grid_reference':{'full_fit':original_best,'grid':cfg['calibration_search']['CLIP_grid'],
                                            'bootstrap_threshold_distribution':[{'threshold':t,'count':n,'proportion':n/BOOTSTRAP} for t,n in sorted(original_freq.items())],
                                            'not_replaced_by_diagnostic_grid':True},
            'protocol_amended':False,'threshold_chosen':False,'threshold_frozen':False,'K_study_run':False,'case_expansion':False,'heldout_test_used':False,
            'AI_suggestions_used_as_labels':False,'source_annotation_modified':False,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},
            'integrity':{'analysis_code_sha256':sha(Path(__file__)),'protected_input_count':len(before),'input_sha256':before}}
    assert before=={name:sha(ROOT/name) for name in before},'Protected source/protocol/artifact changed'
    assert not (BASE/'frozen_thresholds.json').exists()
    report['integrity']['all_protected_files_byte_unchanged']=True
    write(OUT/'diagnostic_report.json',report);write(OUT/'threshold_table.json',table)
    write(OUT/'operating_points.json',points);write(OUT/'bootstrap_replicates.json',raw)
    write(OUT/'leave_one_negative_out.json',lono);write(OUT/'inherited_full_grid_results.json',original_table)
    (OUT/'diagnostic_report.md').write_text(report_markdown(report),encoding='utf-8')
    plot_report(table,boot)
    print(json.dumps({'counts':report['counts'],'points':{k:{kk:v for kk,v in p.items() if kk!='decision_case_ids'} if p else None for k,p in points.items()},
                      'bootstrap_refits':{k:v['refit_selection_rule'] for k,v in boot.items()},
                      'lono_thresholds':{k:[r['rules'][k]['candidate']['threshold'] for r in lono] for k in RULES},
                      'report':str(OUT/'diagnostic_report.md')}))
    return report


if __name__=='__main__':run()
