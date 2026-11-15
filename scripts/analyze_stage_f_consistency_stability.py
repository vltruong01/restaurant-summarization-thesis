"""D-029 report-only Consistency sensitivity; never import labels or freeze cutoffs."""
from pathlib import Path
from collections import Counter
import argparse
import json
import platform
import numpy as np
import yaml
from stage_f_math import threshold_search
from stage_f_verification_workflow import read, write, sha, active_ids, validate_export

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/stage_f/validation_calibration_v1'
ASSISTED = ROOT / 'outputs/stage_f/ai_assisted_calibration_v1'
POLICY = ROOT / 'configs/stage_f_consistency_stability.yaml'


def active_refinement():
    return POLICY.is_file() and yaml.safe_load(POLICY.read_text(encoding='utf-8')).get('active') is True


def validate_declared_prefix(dataset, order, current_selection, ai, labels):
    """An already confirmed older prefix remains usable after a pending batch opens."""
    expected = {c['case_id']: c for c in dataset['cases']}
    declared = labels['active_case_ids']
    assert len(declared) == len(set(declared)) and set(declared).issubset(expected)
    prefix = {'image_cases': sum(expected[c]['kind'] == 'image_caption' for c in declared),
              'pair_cases': sum(expected[c]['kind'] == 'caption_pair' for c in declared)}
    assert prefix['image_cases'] <= current_selection['image_cases']
    assert prefix['pair_cases'] <= current_selection['pair_cases']
    assert declared == active_ids(order, prefix), 'Label export is not the frozen deterministic prefix'
    return validate_export(dataset, order, prefix, ai, labels), prefix


def consistency_rows(dataset, actual, scores, roles, active):
    rows = []; uncertain = []; pending = []
    for case in dataset['cases']:
        if case['kind'] != 'image_caption':
            continue
        cid = case['case_id']; row = actual[cid]
        assert roles[case['business_id']] == 'calibration', 'Non-calibration source rejected'
        if not row['human_verified']:
            if cid in active: pending.append(cid)
            continue
        label = row['human_final_label']['consistency']
        if label == 'uncertain':
            uncertain.append(cid); continue
        assert label in ('consistent', 'inconsistent')
        value = float(scores[cid]['clip_score']); assert np.isfinite(value)
        rows.append({'case_id': cid, 'photo_id': case['photo_ids'][0],
                     'business_id': case['business_id'], 'caption': case['normalized_captions'][0],
                     'raw_caption': case['raw_captions'][0], 'human_final_label': label,
                     'human_verified': True, 'clip_score': value,
                     'decision_score': round(value, 6), 'positive': label == 'consistent'})
    return rows, uncertain, pending


def metrics(scores, labels, threshold):
    scores = [round(float(s), 6) for s in scores]
    predictions = [s >= threshold for s in scores]
    positive = sum(labels); negative = len(labels) - positive
    assert positive and negative, 'Balanced accuracy requires both actual classes'
    tp = sum(p and y for p, y in zip(predictions, labels))
    tn = sum(not p and not y for p, y in zip(predictions, labels))
    fp = negative - tn; fn = positive - tp
    return {'balanced_accuracy': (tp / positive + tn / negative) / 2,
            'positive_precision': tp / (tp + fp) if tp + fp else 0.,
            'positive_recall': tp / positive, 'negative_specificity': tn / negative,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'percentile_2_5': float(np.percentile(values, 2.5)),
            'percentile_97_5': float(np.percentile(values, 97.5))}


def analyze_rows(rows, grid, replicates=2000, seed=42):
    """Use the canonical frozen search for every fit, including every resample."""
    scores = [r['clip_score'] for r in rows]; truth = [r['positive'] for r in rows]
    best, table = threshold_search(scores, truth, grid, minimum_class=1)
    best = {**best, **metrics(scores, truth, best['threshold'])}
    positive = np.flatnonzero(truth); negative = np.flatnonzero(np.logical_not(truth))
    rng = np.random.default_rng(seed); bootstrap = []; lono = []
    full_prediction = np.asarray([round(float(s), 6) >= best['threshold'] for s in scores])
    for index in range(replicates):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        draw_scores = [scores[int(i)] for i in sample]
        draw_labels = [truth[int(i)] for i in sample]
        fit, _ = threshold_search(draw_scores, draw_labels, grid, minimum_class=1)
        fit = {**fit, **metrics(draw_scores, draw_labels, fit['threshold'])}
        reference = metrics(scores, truth, fit['threshold'])
        flips = sum((round(float(s), 6) >= fit['threshold']) != p for s, p in zip(scores, full_prediction))
        bootstrap.append({'replicate': index + 1, 'selected_threshold': fit['threshold'],
                          'threshold_delta': fit['threshold'] - best['threshold'],
                          'resample_fit': fit, 'original_reference': reference,
                          'original_prediction_flips': int(flips)})
    for removed in negative:
        keep = [i for i in range(len(rows)) if i != removed]
        record = {'removed': rows[int(removed)]}
        if len(negative) == 1:
            record.update(status='UNIDENTIFIABLE_ONE_CLASS_FOLD', selected_threshold=None)
        else:
            kept_scores = [scores[i] for i in keep]; kept_truth = [truth[i] for i in keep]
            fit, _ = threshold_search(kept_scores, kept_truth, grid, minimum_class=1)
            reference = metrics(scores, truth, fit['threshold'])
            flipped = [r['case_id'] for r, p in zip(rows, full_prediction)
                       if (round(float(r['clip_score']), 6) >= fit['threshold']) != p]
            record.update(status='FITTED', selected_threshold=fit['threshold'],
                          threshold_delta=fit['threshold'] - best['threshold'],
                          absolute_threshold_delta=abs(fit['threshold'] - best['threshold']),
                          fold_fit={**fit, **metrics(kept_scores, kept_truth, fit['threshold'])},
                          original_reference=reference, original_prediction_flips=len(flipped),
                          flipped_case_ids=flipped,
                          removed_case_predicted_positive=rows[int(removed)]['decision_score'] >= fit['threshold'])
        lono.append(record)
    frequencies = Counter(r['selected_threshold'] for r in bootstrap)
    result = {'full_fit': best, 'bootstrap_summary': {
        'threshold': summary([r['selected_threshold'] for r in bootstrap]),
        'resample_fit_balanced_accuracy': summary([r['resample_fit']['balanced_accuracy'] for r in bootstrap]),
        'original_reference_balanced_accuracy': summary([r['original_reference']['balanced_accuracy'] for r in bootstrap]),
        'original_prediction_flips': summary([r['original_prediction_flips'] for r in bootstrap]),
        'threshold_distribution': [{'threshold': t, 'count': n, 'proportion': n / replicates}
                                   for t, n in sorted(frequencies.items())],
        'same_as_full_fit_fraction': frequencies[best['threshold']] / replicates},
        'leave_one_negative_out': lono}
    return result, table, bootstrap


def render_report(report, out):
    full = report['full_fit']; boot = report['bootstrap_summary']; lono = report['leave_one_negative_out']
    lines = ['# Consistency stability — D-029, provisional researcher review', '',
             '**Report only. No threshold frozen, K selected, labels changed or cases expanded.**', '',
             f"Source: {report['counts']['positive']} positive, {report['counts']['negative']} negative; "
             f"{report['counts']['uncertain_excluded']} verified uncertain excluded. "
             f"{report['counts']['opened_pending_deferred']} already opened image cases deferred.", '',
             '## Full verified non-uncertain calibration fit', '',
             f"Provisional threshold **{full['threshold']:.2f}**; balanced accuracy **{full['balanced_accuracy']:.4f}**; "
             f"positive precision {full['positive_precision']:.4f}; positive recall {full['positive_recall']:.4f}; "
             f"negative specificity {full['negative_specificity']:.4f}.", '',
             f"TP/TN/FP/FN = {full['tp']}/{full['tn']}/{full['fp']}/{full['fn']}.", '',
             'CLIP L2 cosine, Python score rounding to 6 decimals, pass >= cutoff; grid -1..1 by .01. '
             'Maximize balanced accuracy; ties precision, recall, then higher threshold, all unchanged.', '',
             '## Stratified bootstrap', '',
             f"{report['bootstrap']['replicates']} independent class-resampled draws, seed {report['bootstrap']['seed']}; "
             'each draw preserves observed class counts. Every fit uses the original canonical search.', '',
             '| Statistic | Mean | Median | Min | Max | 2.5 percentile | 97.5 percentile |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key in ['threshold', 'resample_fit_balanced_accuracy', 'original_reference_balanced_accuracy', 'original_prediction_flips']:
        s = boot[key]
        lines.append('| ' + key + ' | ' + ' | '.join(f'{s[k]:.4f}' for k in ['mean', 'median', 'min', 'max', 'percentile_2_5', 'percentile_97_5']) + ' |')
    lines += ['', f"Fraction choosing full-fit threshold: {boot['same_as_full_fit_fraction']:.2%}.", '',
              '| Selected threshold | Draws | Proportion |', '|---:|---:|---:|']
    lines += [f"| {r['threshold']:.2f} | {r['count']} | {r['proportion']:.2%} |" for r in boot['threshold_distribution']]
    lines += ['', '![Bootstrap distributions](stability_distributions.png)', '',
              '## Leave one negative out', '',
              '| Removed case | Photo ID | CLIP score | Threshold | Delta | Fold BA | Original-reference BA | Prediction flips |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in lono:
        c = r['removed']
        if r['status'] != 'FITTED':
            lines.append(f"| {c['case_id']} | {c['photo_id']} | {c['decision_score']:.6f} | unidentifiable | | | | |"); continue
        lines.append(f"| {c['case_id']} | {c['photo_id']} | {c['decision_score']:.6f} | {r['selected_threshold']:.2f} | "
                     f"{r['threshold_delta']:+.2f} | {r['fold_fit']['balanced_accuracy']:.4f} | "
                     f"{r['original_reference']['balanced_accuracy']:.4f} | {r['original_prediction_flips']} |")
    fitted = [r for r in lono if r['status'] == 'FITTED']
    lines += ['', '## Observed effect on the verified calibration cases', '',
              f"The provisional full fit rejects {full['fn']} of {full['positive_labels']} human-consistent captions "
              f"and still passes {full['fp']} of {full['negative_labels']} human-inconsistent captions. "
              'High positive precision alone is insufficient evidence of a good consistency filter in this imbalanced set.', '',
              f"Among leave-one-negative-out fits, the largest original-set prediction change is "
              f"{max((r['original_prediction_flips'] for r in fitted), default=0)} cases. "
              'A small numeric cutoff change can therefore have a substantial filtering effect. '
              'The bootstrap shows wider cutoff sensitivity than leaving out one negative at a time. '
              'These observations support caution before freezing; they are not an automatic expansion trigger.', '']
    lines += ['', 'Detailed removed captions, business IDs, scores, full grid and replicate results are in the JSON companions.', '',
              '## Interpretation limits and review', '',
              'The bootstrap percentile range is a conditional empirical sensitivity summary, not a guaranteed population confidence interval. '
              f"Resampling repeats the same {report['counts']['negative']} observed negatives; it cannot provide new independent negative evidence. "
              'Cases were deliberately stratified and may share a business or image, so case exchangeability is an assumption. '
              'All balanced accuracies here are apparent calibration metrics (resample fit or original-set reference), not held-out estimates.', '',
              'No post-hoc binary stability cutoff is assigned. Review the distribution and case influence before deciding whether to accept '
              'the provisional threshold or request further Consistency verification. Natural negative rarity alone no longer triggers expansion. '
              'Relevance/pairwise criteria are not expanded. No K-study or held-out test labels/scores were used for fitting.', '',
              'ChatGPT remains only an optional comparison/adjudication aid after the user’s own inspection. '
              'All fitted labels remain explicitly user-confirmed human decisions.', '']
    (out / 'stability_report.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_report(result, bootstrap, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    freq = result['bootstrap_summary']['threshold_distribution']
    axes[0].bar([r['threshold'] for r in freq], [r['count'] for r in freq], width=.007, color='#2563eb')
    axes[0].axvline(result['full_fit']['threshold'], color='#dc2626', linestyle='--', label='Provisional full fit')
    axes[0].set(title='Stratified bootstrap thresholds', xlabel='CLIP cutoff', ylabel='Draws'); axes[0].legend(fontsize=8)
    axes[1].hist([r['resample_fit']['balanced_accuracy'] for r in bootstrap], bins=30, color='#64748b', alpha=.65, label='Resample fit')
    axes[1].hist([r['original_reference']['balanced_accuracy'] for r in bootstrap], bins=30, color='#059669', alpha=.6, label='Original reference')
    axes[1].set(title='Calibration BA sensitivity', xlabel='Balanced accuracy', ylabel='Draws'); axes[1].legend(fontsize=8)
    fitted = [r for r in result['leave_one_negative_out'] if r['status'] == 'FITTED']
    axes[2].bar(range(1, len(fitted)+1), [r['threshold_delta'] for r in fitted], color='#b45309')
    axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].set(title='Leave one negative out', xlabel='Negative case index (report order)', ylabel='Cutoff change')
    fig.suptitle('D-029: report only; no frozen threshold or K; 2,000 draws, seed 42')
    fig.savefig(out / 'stability_distributions.png', dpi=180); plt.close(fig)


def run(labels_path=None):
    assert active_refinement(), 'D-029 must be active before fitting'
    cfg = yaml.safe_load(POLICY.read_text(encoding='utf-8'))
    labels_path = Path(labels_path) if labels_path else ROOT / cfg['label_source']
    output = ROOT / cfg['output_directory']; output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / cfg['scoring_protocol']; workflow_path = ROOT / cfg['verification_protocol']
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    assert protocol['calibration_search']['CLIP_grid'] == {'start': -1., 'stop': 1., 'step': .01}
    assert protocol['calibration_search']['criterion'] == 'maximum_balanced_accuracy'
    assert protocol['calibration_search']['ties'] == ['higher_positive_precision', 'higher_positive_recall', 'higher_threshold']
    assert protocol['decision_round_decimals'] == 6 and protocol['CLIP_score'] == 'L2_normalized_image_dot_L2_normalized_caption'
    dataset = read(BASE / 'human_check/annotation_dataset.json'); labels = read(labels_path)
    order = read(ASSISTED / 'verification_order.json'); current = read(ASSISTED / 'active_selection.json')
    ai = read(ASSISTED / 'ai_suggestions.json')
    assert order['workflow_sha256'] == sha(workflow_path)
    assert order['role_manifest_sha256'] == sha(BASE / 'business_roles.json')
    assert order['private_stratification_scores_sha256'] == sha(BASE / 'human_check/private_calibration_scores.json')
    scoring = read(BASE / 'scores_calibration/scoring_report.json')
    assert scoring['config_sha256'] == sha(protocol_path) and scoring['overlength'] == 0
    actual, declared = validate_declared_prefix(dataset, order, current, ai, labels)
    roles = {r['business_id']: r['role'] for r in read(BASE / 'business_roles.json')}
    assert all(roles[c['business_id']] == 'calibration' for c in dataset['cases'])
    private = read(BASE / 'human_check/private_calibration_scores.json')
    rows, uncertain, pending = consistency_rows(dataset, actual, private, roles, set(active_ids(order, current)))
    protected = [labels_path, ASSISTED / 'active_selection.json', ASSISTED / 'verification_order.json',
                 ASSISTED / 'ai_suggestions.json', BASE / 'human_check/annotation_dataset.json',
                 BASE / 'human_check/private_calibration_scores.json', protocol_path, workflow_path]
    before = {str(p.resolve()): sha(p) for p in protected}
    frozen = BASE / 'frozen_thresholds.json'; frozen_before = sha(frozen) if frozen.exists() else None
    replicates = cfg['bootstrap']['replicates']; seed = cfg['bootstrap']['seed']
    assert replicates == 2000 and seed == 42 and cfg['automatic_threshold_freeze'] is False and cfg['K_selection'] is False
    result, grid, bootstrap = analyze_rows(rows, [i / 100 for i in range(-100, 101)], replicates, seed)
    report = {'decision': 'D-029', 'status': 'PROVISIONAL_STABILITY_REPORT_RESEARCHER_REVIEW_REQUIRED',
              'threshold_frozen': False, 'case_expansion_performed': False, 'K_selection_run': False,
              'heldout_test_used': False, 'K_study_used_for_fit': False, 'AI_labels_used_for_fit': False,
              'human_labels_modified': False, 'annotator': labels['annotator'],
              'counts': {'positive': sum(r['positive'] for r in rows), 'negative': sum(not r['positive'] for r in rows),
                         'fit_total': len(rows), 'uncertain_excluded': len(uncertain), 'opened_pending_deferred': len(pending)},
              'uncertain_case_ids_excluded': uncertain, 'pending_case_ids_deferred': pending,
              'source_declared_prefix': declared, 'current_selection_unchanged': current,
              'fit_business_count': len({r['business_id'] for r in rows}), 'fit_cases': rows,
              'bootstrap': {'replicates': replicates, 'seed': seed, 'generator': 'numpy.default_rng.PCG64',
                            'scheme': 'case_level_separate_positive_negative_with_replacement_observed_counts'},
              'stability_binary_acceptance_gate': None, 'additional_cases_require_researcher_review': True,
              'score_definition': protocol['CLIP_score'], 'model': protocol['models']['CLIP'],
              'grid': protocol['calibration_search']['CLIP_grid'], 'objective': protocol['calibration_search']['criterion'],
              'tie_breaking': protocol['calibration_search']['ties'], 'score_rounding_decimals': 6,
              'integrity': {'inputs_sha256': before, 'refinement_sha256': sha(POLICY),
                            'analysis_code_sha256': sha(Path(__file__)), 'canonical_math_sha256': sha(Path(__file__).with_name('stage_f_math.py'))},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__}, **result}
    write(output / 'full_grid.json', grid)
    write(output / 'bootstrap_replicates.json', bootstrap)
    write(output / 'leave_one_negative_out.json', result['leave_one_negative_out'])
    plot_report(result, bootstrap, output)
    assert before == {str(p.resolve()): sha(p) for p in protected}, 'Protected inputs changed'
    assert frozen_before == (sha(frozen) if frozen.exists() else None), 'Frozen threshold changed'
    report['integrity']['protected_inputs_byte_unchanged'] = True
    write(output / 'stability_report.json', report); render_report(report, output)
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'full_fit': result['full_fit'],
                      'bootstrap_summary': result['bootstrap_summary'],
                      'lono_thresholds': [r['selected_threshold'] for r in result['leave_one_negative_out']],
                      'report': str(output / 'stability_report.md')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--labels', type=Path)
    run(parser.parse_args().labels)
