"""D-050 observation wrapper around unchanged D-049 feasibility; never trains a pilot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

import run_stage_i_paired_lora_pilot_v1 as prepared
from stage_i_paired_pilot import ARMS, ROOT, execution_gate, require, sha, write_new


OUT = ROOT / 'outputs/stage_i/d050_gpu_feasibility_v1'
CONFIG = ROOT / 'configs/stage_i_paired_lora_pilot_v1.json'


def fingerprint(value):
    """Compare nested optimizer/RNG states without changing their contents or RNG."""
    digest = hashlib.sha256()

    def visit(item):
        digest.update(type(item).__name__.encode() + b':')
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode() + str(item.shape).encode() + item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b';')

    visit(value)
    return digest.hexdigest()


def parameters(model):
    return {name: {'sha256': fingerprint(p), 'shape': list(p.shape),
                   'requires_grad': p.requires_grad, 'numel': p.numel()}
            for name, p in model.named_parameters()}


def states(optimizer, scheduler, scaler):
    return {'optimizer': fingerprint(optimizer.state_dict()),
            'scheduler': fingerprint(scheduler.state_dict()),
            'scaler': fingerprint(scaler.state_dict()),
            'torch_rng': fingerprint(torch.get_rng_state()),
            'cuda_rng': fingerprint(torch.cuda.get_rng_state_all()),
            'numpy_rng': fingerprint(np.random.get_state()),
            'python_rng': fingerprint(prepared.random.getstate())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--approval', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config, prep, _, approval = execution_gate(CONFIG, 'feasibility', args.arm, args.approval, args.execute)
    require(approval.get('decision') == 'D-050' and approval.get('scopes') == ['feasibility'], 'D-050 feasibility only')
    require(approval.get('observation_wrapper_sha256') == sha(Path(__file__)), 'Observation code changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    folder = OUT / args.arm
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model_ref = []
    before = {}
    updates = []
    gradients = []
    microbatches = []
    previous_adapter_hashes = {}
    stops = threading.Event()
    telemetry = []

    def event(name, **value):
        record = {'utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.perf_counter() - started, 'event': name, **value}
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps({'arm': args.arm, **record}), flush=True)

    def monitor():
        while not stops.is_set():
            try:
                response = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                telemetry.append({'seconds': time.perf_counter() - started,
                                  'csv_memory_MiB_total_MiB_util_percent': response.stdout.strip(),
                                  'exit_code': response.returncode})
            except Exception as exc:
                telemetry.append({'seconds': time.perf_counter() - started, 'error_type': type(exc).__name__})
            stops.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    original_load = prepared.load_gpu
    original_gradient = prepared.gradient_record
    original_optimizer = prepared.optimizer_scheduler
    original_collator = prepared.LEDCompleteCollator
    original_save = prepared.save_feas_checkpoint
    original_restore = prepared.restore_feas_checkpoint

    def observed_load(*call_args, **call_kwargs):
        model, scope, initialization = original_load(*call_args, **call_kwargs)
        model_ref.append(model)
        before.update(parameters(model))
        write_new(folder / 'parameter_hashes_before.json', before)
        properties = torch.cuda.get_device_properties(0)
        write_new(folder / 'runtime.json', {'gpu': properties.name, 'total_memory': properties.total_memory,
            'capability': list(torch.cuda.get_device_capability()), 'torch': torch.__version__,
            'torch_CUDA': torch.version.cuda, 'scope': scope, 'initialization': initialization,
            'config_sha256': sha(CONFIG), 'approval_sha256': sha(args.approval),
            'wrapper_sha256': sha(Path(__file__)), 'D049_binding': prepared.binding(CONFIG, prep, args.arm)})
        event('model_loaded', modules=len(scope['modules']), parameters=scope['trainable_parameters'],
              backbone_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        return model, scope, initialization

    def observed_gradient(model, *call_args, **call_kwargs):
        result = original_gradient(model, *call_args, **call_kwargs)
        gradients.append(result)
        write_new(folder / 'gradients' / f'update_call_{len(gradients)}.json', result)
        groups = {}
        for name, values in result.items():
            group = 'encoder_global' if '_global.' in name else 'encoder_local' if '.encoder.layers.' in name else 'decoder'
            matrix = 'A' if '.lora_A.' in name else 'B'
            key = group + '_' + matrix
            target = groups.setdefault(key, {'tensors': 0, 'nonzero': 0, 'all_finite': True})
            target['tensors'] += 1
            target['nonzero'] += values['nonzero']
            target['all_finite'] &= values['finite']
        event('gradients_checked', update_call=len(gradients), groups=groups)
        return result

    def observed_optimizer(*call_args, **call_kwargs):
        optimizer, scheduler = original_optimizer(*call_args, **call_kwargs)
        previous_adapter_hashes.update({name: fingerprint(p) for name, p in model_ref[0].named_parameters() if p.requires_grad})

        def after_step(opt, unused_args, unused_kwargs):
            current = {name: fingerprint(p) for name, p in model_ref[0].named_parameters() if p.requires_grad}
            changed = [name for name in current if current[name] != previous_adapter_hashes[name]]
            record = {'physical_optimizer_call': len(updates) + 1,
                      'parameter_steps': sorted({int(state['step']) for state in opt.state.values()}),
                      'used_lr': opt.param_groups[0]['lr'], 'scheduler_last_epoch_before_scheduler_step': scheduler.last_epoch,
                      'changed_parameter_names': changed, 'adapter_hashes': current}
            updates.append(record)
            write_new(folder / 'optimizer_updates' / f'call_{len(updates)}.json', record)
            previous_adapter_hashes.update(current)
            event('optimizer_update', physical_call=len(updates), steps=record['parameter_steps'],
                  lr=record['used_lr'], changed_adapter_tensors=len(changed))

        optimizer.register_step_post_hook(after_step)
        return optimizer, scheduler

    class ObservedCollator:
        def __init__(self, *call_args, **call_kwargs):
            self.inner = original_collator(*call_args, **call_kwargs)

        def __call__(self, features):
            require(all(f['input_ids'][0] == 0 and f['input_ids'][-1] == 2 and
                        f['labels'][0] == 0 and f['labels'][-1] == 2 for f in features), 'Source/label BOS or EOS differs')
            batch = self.inner(features)
            item = {'physical_microbatch': len(microbatches) + 1,
                    'actual_source_lengths': [len(f['input_ids']) for f in features],
                    'actual_label_lengths': [len(f['labels']) for f in features],
                    'source_tensor_shape': list(batch['input_ids'].shape),
                    'label_tensor_shape': list(batch['labels'].shape),
                    'global_tokens': batch['global_attention_mask'].sum().item(),
                    'ignored_label_padding': int((batch['labels'] == -100).sum())}
            microbatches.append(item)
            with (folder / 'microbatches.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item) + '\n')
            if (len(microbatches) - 1) % 8 == 0:
                event('accumulation_block_started', physical_microbatch=len(microbatches))
            return batch

    def observed_save(model, optimizer, scheduler, scaler, checkpoint, bind, step):
        snapshot = states(optimizer, scheduler, scaler)
        original_save(model, optimizer, scheduler, scaler, checkpoint, bind, step)
        require(snapshot == states(optimizer, scheduler, scaler), 'Save operation changed runtime state')
        write_new(folder / 'checkpoint_saved_state_fingerprints.json', snapshot)
        event('checkpoint_saved', step=step)

    def observed_restore(model, optimizer, scheduler, scaler, checkpoint, bind):
        saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
        expected = {key: fingerprint(saved[key]) for key in
                    ['optimizer', 'scheduler', 'scaler', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng']}
        result = original_restore(model, optimizer, scheduler, scaler, checkpoint, bind)
        actual = states(optimizer, scheduler, scaler)
        checks = {key: actual[key] == expected[key] for key in expected}
        write_new(folder / 'checkpoint_restored_state_fingerprints.json',
                  {'expected': expected, 'actual': actual, 'exact_matches': checks})
        require(all(checks.values()), 'Restored optimizer/scheduler/scaler/RNG state differs')
        previous_adapter_hashes.update({name: fingerprint(p) for name, p in model.named_parameters() if p.requires_grad})
        event('checkpoint_restored', exact_state_matches=checks)
        return result

    prepared.load_gpu = observed_load
    prepared.gradient_record = observed_gradient
    prepared.optimizer_scheduler = observed_optimizer
    prepared.LEDCompleteCollator = ObservedCollator
    prepared.save_feas_checkpoint = observed_save
    prepared.restore_feas_checkpoint = observed_restore
    error = None
    try:
        event('D050_feasibility_start', scope='Three updates plus one replay; no pilot training or generation')
        prepared.feasible(CONFIG, config, prep, args.arm, approval)
        require(len(microbatches) == 32 and len(updates) == 4, 'Unexpected bounded execution count')
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        event('execution_stopped', error_type=type(exc).__name__, message=str(exc))
    finally:
        if model_ref:
            after = parameters(model_ref[0])
            write_new(folder / 'parameter_hashes_after.json', after)
            frozen = [name for name, info in before.items() if not info['requires_grad']]
            changed_frozen = [name for name in frozen if before[name]['sha256'] != after[name]['sha256']]
            changed_trainable = [name for name, info in before.items()
                                 if info['requires_grad'] and info['sha256'] != after[name]['sha256']]
            integrity = {'frozen_parameter_tensors': len(frozen), 'changed_frozen_parameters': changed_frozen,
                         'changed_trainable_parameters': changed_trainable,
                         'only_expected_adapters_changed': not changed_frozen and all('.lora_A.' in name or '.lora_B.' in name for name in changed_trainable),
                         'frozen_parameters_have_no_gradients': all(p.grad is None for p in model_ref[0].parameters() if not p.requires_grad)}
            write_new(folder / 'actual_parameter_update_integrity.json', integrity)
            if changed_frozen:
                error = error or {'type': 'FrozenBackboneChanged', 'message': 'Unexpected frozen parameter change'}
        stops.set()
        thread.join(timeout=12)
        write_new(folder / 'device_telemetry.json', telemetry)
        write_new(folder / 'observation_completion.json', {'status': 'OBSERVATION_PASS' if error is None else 'AFFECTED_EXECUTION_STOPPED',
            'error': error, 'seconds_including_load_and_observation': time.perf_counter() - started,
            'microbatches_observed': len(microbatches), 'optimizer_calls_observed': len(updates),
            'gradient_checks_observed': len(gradients), 'generation_calls': 0, 'pilot_training_started': False,
            'config_sha256': sha(CONFIG), 'prepared_runner_sha256': sha(Path(prepared.__file__)),
            'observer_does_not_change_training_settings_or_prepared_math': True})
    if error:
        raise RuntimeError(error['type'] + ': ' + error['message'])


if __name__ == '__main__':
    main()
