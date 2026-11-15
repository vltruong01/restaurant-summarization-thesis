"""Offline D-051 saved-evidence comparisons; no model, CUDA, or acceptance decision.

Creates a new analysis directory and never changes execution evidence. A failed
allclose in gradients or update deltas is descriptive, not a new readiness gate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from stage_i_d051_numeric import (branch_name, compare_nested_states,
                                 compare_tensor_maps, compare_tensors)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/stage_i/d051_replay_investigation_v1/execution'
RTOL, ATOL = 1e-5, 1e-6
RNG_KEYS = ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def norms(row):
    original = row.get('original_squared_sum')
    replay = row.get('replay_squared_sum')
    return {'original_l2_norm': math.sqrt(original) if original is not None else None,
            'replay_l2_norm': math.sqrt(replay) if replay is not None else None}


def numeric_csv(path, rows):
    fields = ['name', 'branch', 'module', 'tensors', 'numel', 'exact', 'allclose',
              'different_tensors', 'failing_tensors', 'failing_elements', 'failing_pct',
              'max_abs', 'rms', 'max_relative_symmetric', 'relative_l2',
              'original_l2_norm', 'replay_l2_norm', 'squared_error_sum',
              'original_squared_sum', 'replay_squared_sum', 'all_finite',
              'near_zero_both_count_at_atol', 'near_zero_max_abs', 'rtol', 'atol']
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, row in rows.items():
            writer.writerow({'name': name, 'rtol': RTOL, 'atol': ATOL, **row, **norms(row)})


def export_map(out, name, original, replay):
    comparison = compare_tensor_maps(original, replay, rtol=RTOL, atol=ATOL)
    write(out / (name + '.json'), comparison)
    numeric_csv(out / (name + '_per_tensor.csv'), comparison['per_tensor'])
    numeric_csv(out / (name + '_per_module.csv'), comparison['per_module'])
    return comparison


def fingerprints(a, b):
    left, right = a or {}, b or {}
    keys = sorted(set(left) | set(right))
    fields = {key: {'original': left.get(key), 'replay': right.get(key),
                    'exact': key in left and key in right and left[key] == right[key]}
              for key in keys}
    return {'exact': a is not None and b is not None and all(v['exact'] for v in fields.values()),
            'fields': fields, 'original_present': a is not None, 'replay_present': b is not None}


def gradient_fingerprints(a, b):
    compared = fingerprints(a, b)
    by_branch = {}
    for name, row in compared['fields'].items():
        group = by_branch.setdefault(branch_name(name), {'tensors': 0, 'exact': 0,
                                                      'different': 0, 'mismatching_names': []})
        group['tensors'] += 1
        group['exact'] += row['exact']
        group['different'] += not row['exact']
        if not row['exact']:
            group['mismatching_names'].append(name)
    return {**compared, 'per_branch': by_branch}


def compare_microbatches(rows):
    grouped = {'original_step3': {}, 'replayed_step3': {}}
    duplicates = []
    for row in rows:
        label = row.get('label')
        if label not in grouped:
            continue
        index = row['microbatch_in_step']
        if index in grouped[label]:
            duplicates.append({'label': label, 'microbatch_in_step': index})
        grouped[label][index] = row
    original, replay = grouped.values()
    common = sorted(set(original) & set(replay))
    comparisons = []
    first_backward = None
    for index in common:
        a, b = original[index], replay[index]
        grad = gradient_fingerprints(a.get('accumulated_scaled_gradient_fingerprints'),
                                     b.get('accumulated_scaled_gradient_fingerprints'))
        if not grad['exact'] and first_backward is None:
            first_backward = {'microbatch_in_step': index, 'business_id': a.get('business_id'),
                              'per_branch': grad['per_branch']}
        loss = compare_tensors(torch.tensor(a['loss'], dtype=torch.float64),
                               torch.tensor(b['loss'], dtype=torch.float64)) if a.get('loss') is not None and b.get('loss') is not None else None
        before = fingerprints(a.get('state_before_forward'), b.get('state_before_forward'))
        after = fingerprints(a.get('state_after_forward'), b.get('state_after_forward'))
        backward = fingerprints(a.get('state_after_backward'), b.get('state_after_backward'))
        row = {'microbatch_in_step': index,
               'original_business_id': a.get('business_id'), 'replay_business_id': b.get('business_id'),
               'business_exact': a.get('business_id') == b.get('business_id'),
               'input_fingerprint_exact': a.get('input_fingerprint') == b.get('input_fingerprint'),
               'source_tokens_exact': a.get('source_tokens') == b.get('source_tokens'),
               'label_tokens_exact': a.get('label_tokens') == b.get('label_tokens'),
               'source_shape_exact': a.get('source_shape') == b.get('source_shape'),
               'label_shape_exact': a.get('label_shape') == b.get('label_shape'),
               'original_loss': a.get('loss'), 'replay_loss': b.get('loss'), 'loss_comparison': loss,
               'full_logits_fingerprint_exact': a.get('logits_fingerprint') is not None and
                                                  a.get('logits_fingerprint') == b.get('logits_fingerprint'),
               'original_logits_fingerprint': a.get('logits_fingerprint'),
               'replay_logits_fingerprint': b.get('logits_fingerprint'),
               'state_before_forward': before, 'state_after_forward': after,
               'state_after_backward': backward, 'accumulated_scaled_gradients': grad,
               'RNG_before_forward_exact': all(before['fields'].get(k, {}).get('exact', False) for k in RNG_KEYS),
               'RNG_after_forward_exact': all(after['fields'].get(k, {}).get('exact', False) for k in RNG_KEYS),
               'RNG_after_backward_exact': all(backward['fields'].get(k, {}).get('exact', False) for k in RNG_KEYS)}
        comparisons.append(row)
    def all_rows(field):
        return bool(comparisons) and all(row[field] for row in comparisons)
    return {'original_microbatch_indices': sorted(original), 'replay_microbatch_indices': sorted(replay),
            'missing_from_replay': sorted(set(original) - set(replay)),
            'extra_in_replay': sorted(set(replay) - set(original)), 'duplicate_indices': duplicates,
            'paired_microbatches': len(comparisons), 'rows': comparisons,
            'all_inputs_exact': all_rows('input_fingerprint_exact') and all_rows('business_exact'),
            'all_logits_exact': all_rows('full_logits_fingerprint_exact'),
            'all_losses_exact': bool(comparisons) and all(r['loss_comparison'] is not None and r['loss_comparison']['exact'] for r in comparisons),
            'all_rng_before_forward_exact': all_rows('RNG_before_forward_exact'),
            'all_rng_after_forward_exact': all_rows('RNG_after_forward_exact'),
            'all_rng_after_backward_exact': all_rows('RNG_after_backward_exact'),
            'first_accumulated_backward_gradient_hash_mismatch': first_backward,
            'interpretation_boundary': 'Fingerprints locate first observed differences; they do not establish a particular CUDA kernel cause.'}


def named_optimizer_states(state, names_by_group):
    groups = state['param_groups']
    if len(groups) != len(names_by_group):
        raise ValueError('Optimizer group/name mapping differs')
    by_field, absent = {}, []
    for group, names in zip(groups, names_by_group):
        if len(group['params']) != len(names):
            raise ValueError('Optimizer parameter/name ordering differs')
        for parameter_id, name in zip(group['params'], names):
            if parameter_id not in state['state']:
                absent.append(name)
                continue
            for field, value in state['state'][parameter_id].items():
                if isinstance(value, torch.Tensor):
                    by_field.setdefault(field, {})[name] = value
    return by_field, absent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.execution.resolve()
    out = args.output.resolve() if args.output else source.parent / 'analysis'
    labels = ['original_step3_start', 'original_step3_end', 'replayed_step3_start', 'replayed_step3_end']
    gradient_labels = [label + '_' + stage + '_gradients' for label in ('original_step3', 'replayed_step3') for stage in ('preclip', 'postclip')]
    required = [source / (name + '.pt') for name in labels + gradient_labels]
    required += [source / 'microbatch_complete.jsonl', source / 'optimizer_parameter_order.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {path.name: digest(path) for path in required}
    loaded = {name: torch.load(source / (name + '.pt'), map_location='cpu', weights_only=False)
              for name in labels + gradient_labels}
    rows = [json.loads(line) for line in (source / 'microbatch_complete.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    parameter_order = read(source / 'optimizer_parameter_order.json')
    out.mkdir(parents=True, exist_ok=False)
    a0, a1, b0, b1 = [loaded[name] for name in labels]
    entry = compare_nested_states(a0, b0)
    write(out / 'entry_state_comparison.json', entry)
    weights = export_map(out, 'adapter_weight_comparison', a1['adapters'], b1['adapters'])
    preclip = export_map(out, 'preclip_gradient_comparison', loaded['original_step3_preclip_gradients'], loaded['replayed_step3_preclip_gradients'])
    postclip = export_map(out, 'postclip_gradient_comparison', loaded['original_step3_postclip_gradients'], loaded['replayed_step3_postclip_gradients'])
    da = {name: value - a0['adapters'][name] for name, value in a1['adapters'].items()}
    db = {name: value - b0['adapters'][name] for name, value in b1['adapters'].items()}
    deltas = export_map(out, 'adapter_update_delta_comparison', da, db)
    write(out / 'update_magnitude_context.json', {
        'arithmetic': 'Native source tensor dtype subtraction of step3_end minus its step3_start; no model values modified',
        'note': 'Update-delta tolerance is descriptive, not a replacement for original final-weight replay criterion.',
        'weights': {'overall': weights['overall'], 'per_branch': weights['per_branch']},
        'update_deltas': {'overall': deltas['overall'], 'per_branch': deltas['per_branch']},
        'per_module': {name: {'weights': {**row, **norms(row)},
                              'update_deltas': {**deltas['per_module'][name], **norms(deltas['per_module'][name])}}
                       for name, row in weights['per_module'].items()}})
    write(out / 'end_optimizer_state_comparison.json', compare_nested_states(a1['optimizer'], b1['optimizer']))
    write(out / 'end_scheduler_scaler_RNG_comparison.json', compare_nested_states(
        {key: a1[key] for key in ('scheduler', 'scaler', 'rng')},
        {key: b1[key] for key in ('scheduler', 'scaler', 'rng')}))
    original_fields, absent_a = named_optimizer_states(a1['optimizer'], parameter_order)
    replay_fields, absent_b = named_optimizer_states(b1['optimizer'], parameter_order)
    moment_summaries = {}
    for field in sorted(set(original_fields) | set(replay_fields)):
        if not field.replace('_', '').isalnum():
            raise ValueError('Unsafe optimizer state-field name')
        moments = export_map(out, 'optimizer_post_' + field,
                             original_fields.get(field, {}), replay_fields.get(field, {}))
        moment_summaries[field] = {'overall': moments['overall'], 'per_branch': moments['per_branch']}
    write(out / 'optimizer_parameter_mapping.json', {'parameter_order': parameter_order,
        'original_absent_optimizer_states': absent_a, 'replay_absent_optimizer_states': absent_b,
        'named_moment_summaries': moment_summaries})
    micro = compare_microbatches(rows)
    write(out / 'microbatch_comparison.json', micro)
    optional = {}
    for label in ('original_step3', 'replayed_step3'):
        path = source / (label + '_update.json')
        if path.is_file():
            hashes[path.name] = digest(path)
            optional[label] = {key: value for key, value in read(path).items()
                               if key not in ('gradient_norms', 'entry_fingerprints', 'end_fingerprints')}
    write(out / 'loss_clip_scaler_step_context.json', optional)
    summary = {'analysis_only_no_readiness_classification': True, 'rtol': RTOL, 'atol': ATOL,
               'entry_state_exact': entry['exact'],
               'final_adapters': {'overall': weights['overall'], 'per_branch': weights['per_branch']},
               'preclip_gradients': {'overall': preclip['overall'], 'per_branch': preclip['per_branch']},
               'postclip_gradients': {'overall': postclip['overall'], 'per_branch': postclip['per_branch']},
               'update_deltas': {'overall': deltas['overall'], 'per_branch': deltas['per_branch']},
               'microbatch_observations': {key: value for key, value in micro.items() if key != 'rows'}}
    write(out / 'NUMERIC_ANALYSIS_SUMMARY.json', summary)
    changed = [name for name, saved in hashes.items() if digest(source / name) != saved]
    write(out / 'SOURCE_INTEGRITY.json', {'source_file_sha256': hashes,
        'changed_during_analysis': changed, 'source_immutable': not changed,
        'analyzer_sha256': digest(Path(__file__)), 'numeric_helper_sha256': digest(Path(__file__).with_name('stage_i_d051_numeric.py'))})
    write(out / 'ANALYSIS_MANIFEST.json', {'files': {path.name: digest(path) for path in out.iterdir() if path.is_file()},
        'CPU_only': True, 'model_loaded': False, 'GPU_or_API_execution': False,
        'readiness_or_methodology_decision': False})
    print(json.dumps({'analysis_directory': str(out), 'entry_state_exact': entry['exact'],
                      'final_adapter_allclose': weights['overall']['allclose'],
                      'source_immutable': not changed, 'CPU_only': True}))


if __name__ == '__main__':
    main()
