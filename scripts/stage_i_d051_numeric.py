"""CPU-only numerical diagnostics for D-051; never changes replay tolerances.

Acceptance uses torch.isclose(original, replay) in the original dtype. Numerical
magnitudes use FP64. relative_l2 is ||replay-original|| / max(||original||,
relative_floor); max_relative_symmetric uses max(|a|, |b|, relative_floor).
Nonfinite metrics are null, never a silently finite approximation.
"""
import math
import numbers
import re

import numpy as np
import torch


def compare_tensors(original, replay, rtol=1e-5, atol=1e-6, relative_floor=1e-12):
    """Compare detached values without modifying tensors, gradients, or RNG.

    Dtype/shape mismatch is a structural error, not a numeric cast or a PASS.
    Empty tensors have zero error. near_zero_max_abs is zero when no elements
    satisfy abs(original)<=atol AND abs(replay)<=atol.
    """
    if not isinstance(original, torch.Tensor) or not isinstance(replay, torch.Tensor):
        raise TypeError('Both inputs must be tensors')
    if original.shape != replay.shape or original.dtype != replay.dtype:
        raise ValueError('Tensor shape/dtype mismatch')
    if original.is_complex() or replay.is_complex():
        raise ValueError('Complex tensor diagnostics are not supported')
    if rtol < 0 or atol < 0 or not math.isfinite(relative_floor) or relative_floor <= 0:
        raise ValueError('Invalid tolerance or relative-error floor')
    a = original.detach().cpu().contiguous()
    b = replay.detach().cpu().contiguous()
    close = torch.isclose(a, b, rtol=rtol, atol=atol)
    n = a.numel()
    failing = int((~close).sum().item())
    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
    finite = bool(finite_a.all().item() and finite_b.all().item())
    x, y = a.to(torch.float64), b.to(torch.float64)
    near = finite_a & finite_b & (x.abs() <= atol) & (y.abs() <= atol)
    near_count = int(near.sum().item())
    result = {'exact': bool(torch.equal(a, b)), 'allclose': bool(close.all().item()),
              'numel': n, 'failing_elements': failing,
              'failing_pct': 100.0 * failing / n if n else 0.0,
              'dtype': str(a.dtype), 'shape': list(a.shape),
              'rtol': rtol, 'atol': atol, 'relative_floor': relative_floor,
              'all_finite': finite,
              'nonfinite_original': int((~finite_a).sum().item()),
              'nonfinite_replay': int((~finite_b).sum().item()),
              'near_zero_both_count_at_atol': near_count,
              'near_zero_max_abs': float((x[near] - y[near]).abs().max().item()) if near_count else 0.0}
    if finite:
        diff = y - x
        error_sum = float(diff.square().sum().item())
        original_sum = float(x.square().sum().item())
        replay_sum = float(y.square().sum().item())
        denominator = torch.maximum(x.abs(), y.abs()).clamp_min(relative_floor)
        result.update(max_abs=float(diff.abs().max().item()) if n else 0.0,
                      rms=math.sqrt(error_sum / n) if n else 0.0,
                      max_relative_symmetric=float((diff.abs() / denominator).max().item()) if n else 0.0,
                      relative_l2=math.sqrt(error_sum) / max(math.sqrt(original_sum), relative_floor),
                      squared_error_sum=error_sum, original_squared_sum=original_sum,
                      replay_squared_sum=replay_sum)
    else:
        result.update({k: None for k in ('max_abs', 'rms', 'max_relative_symmetric',
                      'relative_l2', 'squared_error_sum', 'original_squared_sum', 'replay_squared_sum')})
    return result


def module_name(name):
    """Map PEFT runtime/state-dict LoRA A/B names to the same LED projection."""
    name = name.removeprefix('base_model.model.')
    return re.split(r'\.lora_[AB](?:\.|$)', name, maxsplit=1)[0]


def branch_name(name):
    module = module_name(name)
    if '.encoder.layers.' in module:
        return 'encoder_global' if module.endswith(('query_global', 'value_global')) else 'encoder_local'
    if '.decoder.layers.' in module:
        return 'decoder'
    return 'other'


def _aggregate(rows, relative_floor):
    n = sum(r['numel'] for r in rows)
    failed = sum(r['failing_elements'] for r in rows)
    finite = all(r['all_finite'] for r in rows)
    result = {'tensors': len(rows), 'numel': n,
              'exact': all(r['exact'] for r in rows),
              'allclose': all(r['allclose'] for r in rows),
              'failing_tensors': sum(not r['allclose'] for r in rows),
              'different_tensors': sum(not r['exact'] for r in rows),
              'failing_elements': failed, 'failing_pct': 100.0 * failed / n if n else 0.0,
              'all_finite': finite,
              'nonfinite_original': sum(r['nonfinite_original'] for r in rows),
              'nonfinite_replay': sum(r['nonfinite_replay'] for r in rows),
              'near_zero_both_count_at_atol': sum(r['near_zero_both_count_at_atol'] for r in rows),
              'near_zero_max_abs': max((r['near_zero_max_abs'] for r in rows), default=0.0)}
    if finite:
        error_sum = math.fsum(r['squared_error_sum'] for r in rows)
        original_sum = math.fsum(r['original_squared_sum'] for r in rows)
        replay_sum = math.fsum(r['replay_squared_sum'] for r in rows)
        result.update(max_abs=max((r['max_abs'] for r in rows), default=0.0),
                      rms=math.sqrt(error_sum / n) if n else 0.0,
                      max_relative_symmetric=max((r['max_relative_symmetric'] for r in rows), default=0.0),
                      relative_l2=math.sqrt(error_sum) / max(math.sqrt(original_sum), relative_floor),
                      squared_error_sum=error_sum, original_squared_sum=original_sum,
                      replay_squared_sum=replay_sum)
    else:
        result.update({k: None for k in ('max_abs', 'rms', 'max_relative_symmetric',
                      'relative_l2', 'squared_error_sum', 'original_squared_sum', 'replay_squared_sum')})
    return result


def compare_tensor_maps(original, replay, rtol=1e-5, atol=1e-6, relative_floor=1e-12):
    """Summarize matched tensor names; report missing/extra names explicitly.

    Counts and numeric aggregates cover common keys only. A key mismatch forces
    overall exact/allclose false without inventing missing tensor values.
    """
    missing = sorted(set(original) - set(replay))
    extra = sorted(set(replay) - set(original))
    by_tensor, by_module, by_branch = {}, {}, {}
    for name in sorted(set(original) & set(replay)):
        row = compare_tensors(original[name], replay[name], rtol, atol, relative_floor)
        row.update(module=module_name(name), branch=branch_name(name))
        by_tensor[name] = row
        by_module.setdefault(row['module'], []).append(row)
        by_branch.setdefault(row['branch'], []).append(row)
    modules = {k: {**_aggregate(v, relative_floor), 'branch': v[0]['branch']}
               for k, v in by_module.items()}
    overall = _aggregate(list(by_tensor.values()), relative_floor)
    overall['keys_equal'] = not missing and not extra
    overall['exact'] &= overall['keys_equal']
    overall['allclose'] &= overall['keys_equal']
    return {'per_tensor': by_tensor, 'per_module': modules,
            'per_branch': {k: _aggregate(v, relative_floor) for k, v in by_branch.items()},
            'overall': overall, 'missing_from_replay': missing, 'extra_in_replay': extra,
            'rtol': rtol, 'atol': atol, 'relative_floor': relative_floor}


def compare_nested_states(original, replay, rtol=1e-5, atol=1e-6, relative_floor=1e-12):
    """Recursively inspect optimizer/scaler/RNG state with explicit leaf paths.

    Tensor leaves use compare_tensors. Integer/bool/metadata values require exact
    equality; float scalars also expose tolerant numerical comparisons. The
    `exact` and `allclose` statuses remain separate, never a restored-state waiver.
    """
    leaves, structural = {}, []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            try:
                leaves[path] = {'kind': 'tensor', **compare_tensors(a, b, rtol, atol, relative_floor)}
            except (ValueError, TypeError) as exc:
                structural.append({'path': path, 'reason': str(exc),
                                   'original_shape': list(a.shape), 'replay_shape': list(b.shape),
                                   'original_dtype': str(a.dtype), 'replay_dtype': str(b.dtype)})
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype:
                structural.append({'path': path, 'reason': 'NumPy shape/dtype mismatch'})
                return
            exact = bool(np.array_equal(a, b))
            if np.issubdtype(a.dtype, np.number):
                row = compare_tensors(torch.tensor(a.astype(np.float64)), torch.tensor(b.astype(np.float64)),
                                      rtol, atol, relative_floor)
                leaves[path] = {'kind': 'numpy_array', **row, 'exact': exact,
                                'original_numpy_dtype': str(a.dtype)}
                # RNG/state integer arrays are discrete metadata. Their numeric
                # deltas may be informative, but a tolerance must not waive a
                # different RNG state (large uint32 values can be np.isclose).
                if np.issubdtype(a.dtype, np.integer):
                    leaves[path]['allclose'] = exact
            else:
                leaves[path] = {'kind': 'numpy_array', 'exact': exact, 'allclose': exact}
            return
        if type(a) is not type(b):
            structural.append({'path': path, 'reason': 'Type mismatch',
                               'original_type': type(a).__name__, 'replay_type': type(b).__name__})
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b), key=repr):
                child = path + '[' + repr(key) + ']'
                if key not in a or key not in b:
                    structural.append({'path': child, 'reason': 'Missing from original' if key not in a else 'Missing from replay'})
                else:
                    visit(a[key], b[key], child)
            return
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                structural.append({'path': path, 'reason': 'Sequence length mismatch',
                                   'original_length': len(a), 'replay_length': len(b)})
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}[{index}]')
            return
        if isinstance(a, numbers.Real) and not isinstance(a, (bool, numbers.Integral)):
            leaves[path] = {'kind': 'numeric_scalar',
                            'original': float(a) if math.isfinite(a) else None,
                            'replay': float(b) if math.isfinite(b) else None,
                            'original_nonfinite_repr': None if math.isfinite(a) else repr(a),
                            'replay_nonfinite_repr': None if math.isfinite(b) else repr(b),
                            **compare_tensors(torch.tensor(float(a), dtype=torch.float64),
                                              torch.tensor(float(b), dtype=torch.float64),
                                              rtol, atol, relative_floor)}
            return
        try:
            exact = bool(a == b)
        except (ValueError, TypeError):
            exact = False
        # Normalize NumPy scalar metadata so json.dumps needs no custom encoder.
        plain_a = a.item() if isinstance(a, np.generic) else a
        plain_b = b.item() if isinstance(b, np.generic) else b
        leaves[path] = {'kind': 'exact_metadata', 'original': plain_a, 'replay': plain_b,
                        'exact': exact, 'allclose': exact}

    visit(original, replay, 'root')
    return {'exact': not structural and all(row['exact'] for row in leaves.values()),
            'allclose': not structural and all(row['allclose'] for row in leaves.values()),
            'leaves_compared': len(leaves), 'per_path': leaves,
            'different_paths': [path for path, row in leaves.items() if not row['exact']],
            'failing_paths': [path for path, row in leaves.items() if not row['allclose']],
            'structural_mismatches': structural,
            'rtol': rtol, 'atol': atol, 'relative_floor': relative_floor}
