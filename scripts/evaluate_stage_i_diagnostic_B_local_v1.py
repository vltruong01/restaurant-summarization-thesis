"""CPU-only paired Baseline/A/B evaluation; never generate or rewrite predictions."""
from collections import Counter
import ast
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer
from run_stage_i_decoding_diagnostic_B_local_v1 import checked_inputs, sha, read, require
from stage_i_decoding_diagnostics import (CONDITIONS, captions, evaluate, read_jsonl, write_json,
                                        write_csv, words, sections, distribution)

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/decoding_diagnostic_v1'
CLOUD = ROOT / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
A = ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2'
B = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1'
OUT = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1'
CONTROL = ROOT / 'outputs/stage_i/decoding_B_execution_control_v1'


def ngram_metrics(tokens, prefix, orders):
    metrics = {}
    for n in orders:
        counts = Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
        total = sum(counts.values())
        duplicates = total - len(counts)
        metrics.update({f'{prefix}_{n}gram_repeated': duplicates > 0,
                        f'{prefix}_{n}gram_occurrences': total,
                        f'{prefix}_{n}gram_duplicate_occurrences': duplicates,
                        f'{prefix}_{n}gram_duplicate_fraction': duplicates / total if total else None,
                        f'{prefix}_distinct_{n}': len(counts) / total if total else None})
    return metrics


def generate_calls(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate']


def main():
    require(not OUT.exists(), 'Preserve earlier evaluation; use a new namespace')
    manifest, cases = checked_inputs(PREP, CLOUD)
    baseline = {c: [r['baseline_prediction'] for r in cases if r['condition'] == c] for c in CONDITIONS}
    prediction_sets = {'baseline': baseline}
    completion_records, environments, configs = {}, {}, {}
    raw_B = []
    tokenizer = Tokenizer.from_file(str(ROOT / 'assets/tokenizers/led-base-16384/tokenizer.json'))
    for name, folder in [('greedy_A', A), ('beam_ngram_B', B)]:
        completions = list((folder / 'attempts').glob('*/completion_report.json'))
        require(len(completions) == 1, 'Expected exactly one successful attempt per diagnostic')
        completion = read(completions[0])
        require(completion['status'] == 'ALL72_DIAGNOSTIC_PREDICTIONS_COMPLETE', 'Incomplete matrix')
        completion_records[name] = completion
        environments[name] = read(completions[0].parent / 'environment.json')
        configs[name] = {c: read(completions[0].parent / f'{c}_generation_config.json') for c in CONDITIONS}
        prediction_sets[name] = {}
        for c in CONDITIONS:
            require(sha(folder / c / 'predictions.jsonl') == completion['predictions_sha256'][c], 'Prediction checksum mismatch')
            preds = read_jsonl(folder / c / 'predictions.jsonl')
            require(len(preds) == 24 and [p['business_id'] for p in preds] == manifest['execution_business_ids'], 'IDs/order mismatch')
            prediction_sets[name][c] = preds
            for p in preds:
                require(p['condition'] == c and 0 < len(p['token_ids']) <= 768, 'Prediction condition/length mismatch')
                require(tokenizer.decode(p['token_ids'], skip_special_tokens=True) == p['text'], 'Token/text mismatch')
                raw = read(folder / c / 'cases' / (hashlib.sha256(p['business_id'].encode()).hexdigest() + '.json'))
                require(raw['prediction'] == p, 'Raw case differs from JSONL')
                if name == 'beam_ngram_B':
                    raw_B.append(raw)
            cfg = configs[name][c]
            require(not cfg['offline_reconstruction_differences'], 'Actual config drift')
            g = cfg['actual_effective']
            require(g['max_length'] == 768 and g['max_new_tokens'] is None and g['do_sample'] is False, 'Length/sampling drift')
            require(g['num_beams'] == (1 if name == 'greedy_A' else 4), 'Beam drift')
            require(g['no_repeat_ngram_size'] == (0 if name == 'greedy_A' else 4), 'Ngram drift')
    approval = read(ROOT / 'outputs/stage_i/decoding_approval_B_local_v1.json')
    binding = read(B / 'run_binding.json')
    require(binding['subset_manifest_sha256'] == sha(PREP / 'subset_manifest.json'), 'B subset binding mismatch')
    require(binding['local_runtime_runner_sha256'] == sha(ROOT / 'scripts/run_stage_i_decoding_diagnostic_B_local_v1.py') == approval['local_runtime_runner_sha256'], 'B code binding mismatch')
    require(generate_calls(ROOT / 'scripts/run_stage_i_decoding_diagnostic_B_local_v1.py') == generate_calls(ROOT / 'scripts/run_stage_i_decoding_diagnostic_local_v2.py'), 'Generate-call AST changed')
    result = evaluate(cases, [p for c in CONDITIONS for p in prediction_sets['beam_ngram_B'][c]], OUT, 'beam_ngram_B')
    # The shared evaluator is preserved; extend its metrics equally for all settings and teacher.
    summaries = {
        'baseline': read(PREP / 'baseline_evaluation/summary.json'),
        'greedy_A': read(ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1/summary.json'),
        'beam_ngram_B': result}
    rows = []
    for role in ['random', 'targeted']:
        for condition in CONDITIONS:
            for name, summary in summaries.items():
                group = summary['subsets'][role][condition]
                row = {'subset': role, 'condition': condition, 'experiment': name, 'outputs': group['outputs']}
                for key in ['exact_repeated_sentence', 'consecutive_repeated_word', 'section_headers_compliant', 'eos_before_length_cap', 'cap_reached', 'visual_text_without_caption_input']:
                    row[key + '_count'] = group[key]['count']
                row.update({'length_' + k: v for k, v in group['generated_length'].items()})
                for key in ['teacher_agreement_full_rouge1_fmeasure', 'teacher_agreement_full_rouge2_fmeasure', 'teacher_agreement_full_rougeL_fmeasure', 'teacher_agreement_review_rougeL_recall']:
                    row[key] = group[key]['macro_mean']
                row['zero_caption_applicable'] = group['zero_caption_empty_visual_compliance']['applicable']
                row['zero_caption_compliant'] = group['zero_caption_empty_visual_compliance']['compliant']
                rows.append(row)
    write_csv(OUT / 'baseline_A_B_comparison.csv', rows)
    indexes = {name: {(p['condition'], p['business_id']): p for pp in groups.values() for p in pp} for name, groups in prediction_sets.items()}
    extended, comparisons = [], []
    for c in cases:
        key = c['condition'], c['business_id']
        comp = {'business_id': c['business_id'], 'condition': c['condition'], 'subset': c['subset'],
                'source': c['source'], 'teacher': c['target'], 'captions': captions(c),
                'selected_review_ids': c['selected_review_ids'], 'selected_photo_ids': c['selected_photo_ids'],
                'review_evidence': c['shared_review_evidence'], 'teacher_visual_evidence': c['visual_claims_with_evidence'],
                'source_sha256': c['source_sha256']}
        for name in ['baseline', 'greedy_A', 'beam_ngram_B', 'teacher']:
            text = c['target'] if name == 'teacher' else indexes[name][key]['text']
            ids = c['label_ids'] if name == 'teacher' else indexes[name][key]['token_ids']
            row = {'business_id': c['business_id'], 'condition': c['condition'], 'subset': c['subset'], 'experiment': name,
                   'lexical_word_count': len(words(text)),
                   **ngram_metrics(words(text), 'lexical', [1, 2, 3, 4]),
                   **ngram_metrics(ids, 'full_BPE_including_specials', [4])}
            extended.append(row)
            if name != 'teacher':
                comp[name] = text
        comparisons.append(comp)
    write_csv(OUT / 'ngram_lexical_metrics.csv', extended)
    write_json(OUT / 'baseline_A_B_cases.json', comparisons)
    ngsummary = []
    for role in ['random', 'targeted']:
        for c in CONDITIONS:
            for name in ['baseline', 'greedy_A', 'beam_ngram_B', 'teacher']:
                rr = [r for r in extended if r['subset'] == role and r['condition'] == c and r['experiment'] == name]
                vals = {'subset': role, 'condition': c, 'experiment': name, 'outputs': len(rr)}
                for metric in ['lexical_2gram_duplicate_fraction', 'lexical_3gram_duplicate_fraction', 'lexical_4gram_duplicate_fraction',
                               'lexical_distinct_1', 'lexical_distinct_2', 'lexical_distinct_4', 'full_BPE_including_specials_4gram_duplicate_fraction']:
                    numbers = [r[metric] for r in rr if r[metric] is not None]
                    vals[metric + '_macro_mean'] = statistics.mean(numbers) if numbers else None
                vals['outputs_with_lexical_repeated_4gram'] = sum(r['lexical_4gram_repeated'] for r in rr)
                vals['outputs_with_full_BPE_repeated_4gram'] = sum(r['full_BPE_including_specials_4gram_repeated'] for r in rr)
                ngsummary.append(vals)
    write_json(OUT / 'ngram_lexical_summary.json', {'definitions': {
        'lexical': 'Same lowercase header-stripped word regex as the accepted adjacent-word diagnostic; punctuation ignored; spans section boundaries.',
        'duplicate_fraction': '(ngram occurrences - unique ngrams) / ngram occurrences; null for no opportunities; macro per output.',
        'distinct_n': 'Unique ngrams / ngram occurrences. Distinct-1 equals type-token ratio and is length-dependent, not semantic informativeness.',
        'BPE': 'Exact frozen token IDs including decoder-start and section markers, n=4; the actual B constraint applies to this sequence, not normalized lexical words.',
        'interpretation': 'BPE 4gram elimination is expected by construction; lexical variants or unsupported filler may remain.'}, 'rows': ngsummary})
    state = read(CONTROL / 'orchestration.json')
    require(state['status'] == 'PROCESS_FINISHED' and state['exit_code'] == 0, 'Runner did not finish normally')
    with (CONTROL / 'gpu_metrics.csv').open(encoding='utf-8') as f:
        gpu = list(csv.DictReader(f))
    bpred = list(indexes['beam_ngram_B'].values())
    runtime = {'device': environments['beam_ngram_B']['device'], 'baseline_device': 'RTX3090', 'A_device': environments['greedy_A']['device'],
               'start_utc': state['start_utc'], 'end_utc': state['end_utc'], 'pipeline_seconds_including_model_load_monitoring_tail': state['orchestration_seconds'],
               'B_generation_seconds': sum(p['seconds'] for p in bpred),
               'B_generation_seconds_by_condition': {c: sum(p['seconds'] for p in bpred if p['condition'] == c) for c in CONDITIONS},
               'torch_peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in raw_B),
               'torch_peak_reserved_GiB': max(r['peak_reserved_GiB'] for r in raw_B),
               'sampled_whole_GPU_peak_MiB': max(float(r['memory_used_MiB']) for r in gpu),
               'sampling_period_seconds': 5, 'samples': len(gpu),
               'sampling_cannot_establish_exact_instantaneous_device_peak': True,
               'no_OOM_generation_failure_or_retry': not list((B / 'attempts').glob('*/failure_report.json')),
               'hardware_comparison_caveat': 'Baseline RTX3090; A and B RTX3050. No isolated cross-hardware speed or bitwise identity claim.'}
    backup = read(CLOUD / 'BACKUP_MANIFEST.json')
    entries = backup['files']
    for e in entries:
        require(sha(CLOUD / e['path']) == e['sha256'], 'Imported scientific archive changed: ' + e['path'])
    integrity = {'status': 'PASS', 'B_outputs': 72, 'A_outputs_reused': 72, 'baseline_outputs_reused': 72,
                 'fixed_businesses': 24, 'random': 20, 'targeted': 4, 'no_overlap': True,
                 'existing_imported_archive_files_unchanged': len(entries),
                 'same_checkpoint_dataset_source_tokenizer_code_bindings': True,
                 'actual_B_generation_configs': configs['beam_ngram_B'], 'B_approval': approval,
                 'only_decoding_difference_from_baseline': {'no_repeat_ngram_size': {'baseline': 0, 'B': 4}},
                 'generation_length_resolution': 'Explicit researcher selection of max_length=768 including decoder-start; max_new_tokens remains null',
                 'environments': environments, 'runtime': runtime,
                 'no_training_teacher_calls_heldout_access_prediction_or_dataset_rewrite': True,
                 'full_BPE_4gram_repeated_B_outputs': sum(r['full_BPE_including_specials_4gram_repeated'] for r in extended if r['experiment'] == 'beam_ngram_B')}
    write_json(OUT / 'execution_integrity_and_runtime.json', integrity)
    write_json(OUT / 'paired_summary.json', {'rows': rows, 'runtime': runtime,
                 'BERTScore': {'status': 'NOT_COMPUTED', 'reason': 'No installed/pinned BERTScore scorer/backbone in the reviewed evaluation protocol; no unreported scorer/model choice or download. ROUGE is available; BERTScore is an explicitly reported limitation.'}})
    # Preserve the shared evaluator's B-only viewer and add the complete four-column view.
    (OUT / 'COMPARISON.html').rename(OUT / 'B_ONLY_COMPARISON.html')
    # Self-contained static four-column viewer: sources/evidence text embedded, no external scripts.
    esc = html.escape
    parts = ['<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">',
             '<meta http-equiv=Content-Security-Policy content="default-src \'none\'; style-src \'unsafe-inline\'">',
             '<title>Stage I Baseline / A / B</title><style>body{font:15px system-ui;margin:24px;background:#f3f5f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 system-ui;max-height:520px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:12px}article{background:white;padding:16px}details{margin:20px 0}summary{cursor:pointer}@media(max-width:1200px){.grid{grid-template-columns:repeat(2,minmax(220px,1fr))}}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>',
             '<h1>Diagnostic B: actual Baseline / A / B comparison</h1><p>Same 24 validation businesses, 20 random + 4 targeted, 72 outputs per setting. Teacher agreement is not factual accuracy. Human labels remain blank. Caption support is distinct from original-photo verification. B only changes no_repeat_ngram_size to 4 from saved beam4 baseline; all settings retain max_length=768.</p>']
    for c in comparisons:
        parts.append(f'<details><summary>{esc(c["subset"])} / {esc(c["business_id"])} / {esc(c["condition"])}</summary><div class=grid>')
        for label, field in [('Teacher', 'teacher'), ('Saved beam4', 'baseline'), ('A: greedy', 'greedy_A'), ('B: beam4 no-repeat4', 'beam_ngram_B')]:
            parts.append(f'<article><h3>{label}</h3><pre>{esc(c[field])}</pre></article>')
        parts.append(f'</div><details><summary>Complete selected reviews + supplied captions</summary><pre>{esc(c["source"])}</pre></details>')
        parts.append(f'<details><summary>REV / IMG_CAP / PHOTO teacher evidence</summary><pre>{esc(json.dumps({k:c[k] for k in ["selected_review_ids", "captions", "review_evidence", "teacher_visual_evidence"]},ensure_ascii=False,indent=2))}</pre></details></details>')
    parts.append('</html>')
    (OUT / 'COMPARISON.html').write_text('\n'.join(parts), encoding='utf-8')
    checked_inputs(PREP, CLOUD)
    print(json.dumps({'status': 'ALL72_B_INTEGRITY_AND_CPU_METRICS_PASS', 'runtime': runtime, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
