"""Narrow V2 RNG guard and durable per-call receipts; no model/scorer imports."""
from datetime import datetime, timezone
from pathlib import Path
import os
import json
import time
import traceback
from stage_i_led_large_nbest_v1 import require, unpad

KEYS = {'python', 'numpy', 'cpu', 'cuda'}
LED_SOURCE_SHA256 = 'dceefdc0ef6012aab03b4e97dfdd7c17cdf2831359335551b82a57803efa833a'


def component_fingerprints(state, fingerprint):
    require(set(state) == KEYS, 'Unexpected RNG components')
    return dict(python=fingerprint(state['python']), numpy=fingerprint(state['numpy']),
                cpu=fingerprint(state['cpu']), cuda=[fingerprint(x) for x in state['cuda']])


def expected_cpu_state(before, torch, draws=12):
    require(draws == 12, 'Only pinned twelve-layer encoder is covered')
    require(torch.get_default_dtype() == torch.float32 and torch.empty(0).device.type == 'cpu',
            'Pinned default CPU float32 RNG draw assumption differs')
    generator = torch.Generator(device='cpu')
    generator.set_state(before)
    for _ in range(draws):
        torch.rand([], generator=generator, device='cpu', dtype=torch.float32)
    return generator.get_state()


def rng_report(before, after, restored, torch, fingerprint):
    expected = dict(before, cpu=expected_cpu_state(before['cpu'], torch))
    b = component_fingerprints(before, fingerprint)
    a = component_fingerprints(after, fingerprint)
    e = component_fingerprints(expected, fingerprint)
    r = component_fingerprints(restored, fingerprint)
    transition = {key: a[key] == e[key] for key in sorted(KEYS)}
    restoration = {key: r[key] == b[key] for key in sorted(KEYS)}
    return dict(policy='PINNED_LED_EVAL_CPU12_RESTORE_V2', expected_encoder_scalar_draws=12,
                before=b, expected_pre_restore=e, actual_pre_restore=a, actual_post_restore=r,
                expected_transition_matches=transition, exact_restoration_matches=restoration,
                status='PASS' if all(transition.values()) and all(restoration.values()) else 'FAIL')


def durable_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def memory_record(memory):
    record = dict(samples=list(memory.samples), sampler_failure=memory.failure)
    try:
        memory.sample()
        record.update(samples=list(memory.samples), summary=memory.summary())
    except Exception as error:
        record['measurement_error'] = type(error).__name__ + ': ' + str(error)
    return record


def recorded_generation(*, folder, key, generate, decode, screen, torch, memory,
                        capture, restore, fingerprint, synchronize, metadata,
                        expected_sequences, on_returned, eval_check):
    """Always flush available raw/timing/memory/RNG evidence before raising a guard.

    generate() is the unchanged caller's no_grad/autocast/model.generate closure.
    Failures are recorded, never retried. Restoration failures remain hard stops.
    """
    folder = Path(folder)
    before = capture()
    durable_write(folder/'calls'/f'{key}_STARTED.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), before=component_fingerprints(before, fingerprint),
        expected_sequences=expected_sequences))
    started = time.perf_counter()
    elapsed = None
    generated = None
    rows = []
    raw = None
    errors = []
    diagnostics = dict(utc_started=datetime.now(timezone.utc).isoformat(), metadata=metadata,
                       expected_sequences=expected_sequences, automatic_retry=False)
    try:
        synchronize()
        started = time.perf_counter()
        generated = generate()
        synchronize()
        elapsed = time.perf_counter() - started
    except BaseException as error:
        elapsed = time.perf_counter() - started
        errors.append('Generation failed: ' + type(error).__name__ + ': ' + str(error))
        diagnostics['generation_traceback'] = traceback.format_exc()
    # Capture before restoration, including on partial failure. Never describe an
    # unavailable component as unchanged after a CUDA/device failure.
    after = restored = None
    try:
        after = capture()
        diagnostics['observed_pre_restore'] = component_fingerprints(after, fingerprint)
    except BaseException as error:
        errors.append('Post-call RNG capture failed: ' + str(error))
    try:
        restore(before)
        restored = capture()
        diagnostics['observed_post_restore'] = component_fingerprints(restored, fingerprint)
    except BaseException as error:
        errors.append('RNG restoration failed: ' + str(error))
    if after is not None and restored is not None:
        try:
            diagnostics['rng'] = rng_report(before, after, restored, torch, fingerprint)
            if diagnostics['rng']['status'] != 'PASS':
                errors.append('Unexpected RNG transition or inexact restoration')
        except BaseException as error:
            errors.append('RNG verification failed: ' + str(error))
    if generated is not None:
        try:
            raw = generated.detach().cpu().tolist()
            on_returned(len(raw))
            for i, seq in enumerate(raw):
                row = dict(candidate_id=f'c{i}', generation_rank=i, token_ids=seq, **metadata)
                try:
                    row['text'] = decode(seq)
                    row['technical_screen'] = screen(unpad(seq), row['text'], 768)
                    if row['technical_screen']['status'] != 'PASS_TECHNICAL_ONLY':
                        errors.append(f'Technical generation collapse: sequence {i}')
                except BaseException as error:
                    row['decoding_or_screen_error'] = type(error).__name__ + ': ' + str(error)
                    errors.append(f'Decoding/screen failed: sequence {i}: {error}')
                rows.append(row)
            if len(raw) != expected_sequences:
                errors.append('Unexpected returned sequence count')
        except BaseException as error:
            errors.append('Raw token extraction failed: ' + str(error))
    try:
        require(eval_check(), 'Model left eval mode')
    except BaseException as error:
        errors.append('Eval-state check failed: ' + str(error))
    diagnostics.update(seconds=elapsed, raw_returned_sequences=len(raw) if raw is not None else None,
                       raw_token_ids=raw, memory=memory_record(memory))
    if diagnostics['memory'].get('measurement_error') or diagnostics['memory']['sampler_failure']:
        errors.append('Memory diagnostics failed')
    # These durable files exist before the memory guard, caller identity guard,
    # or any recorded error is allowed to terminate the pipeline.
    durable_write(folder/'raw'/f'{key}.json', rows)
    durable_write(folder/'calls'/f'{key}.json', dict(diagnostics, status='EVIDENCE_FLUSHED', errors=errors))
    try:
        memory.check()
    except BaseException as error:
        errors.append('Memory guard failed: ' + str(error))
    last_memory = dict(samples=list(memory.samples), sampler_failure=memory.failure,
                       summary_before_guard=diagnostics['memory'].get('summary'))
    if errors:
        durable_write(folder/'calls'/f'{key}_FAILURE.json', dict(
            status='STOP_NO_RETRY', errors=errors, memory=last_memory))
        raise ValueError('; '.join(errors))
    durable_write(folder/'calls'/f'{key}_PASS.json', dict(status='PASS',
                  RNG='EXPECTED_CPU12_AND_EXACT_RESTORATION', memory=last_memory))
    return rows, elapsed
