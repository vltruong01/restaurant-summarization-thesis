"""D-051: one B-only replay investigation, unchanged D-050 scientific settings."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from peft import get_peft_model_state_dict

import run_stage_i_paired_lora_pilot_v1 as prepared
from run_stage_i_d050_feasibility_v1 import fingerprint, parameters, states
from stage_i_paired_pilot import ROOT, execution_gate, read, require, sha, write_new, verify_mask_batch


OUT = ROOT / 'outputs/stage_i/d051_replay_investigation_v1'
CONFIG = ROOT / 'configs/stage_i_paired_lora_pilot_v1.json'
ARM = 'B_encoder_decoder'


def cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def save_tensor_evidence(path, value):
    require(not path.exists(), 'Preserve existing tensor evidence')
    torch.save(cpu_clone(value), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config, prep, _, approval = execution_gate(CONFIG, 'feasibility', ARM, args.approval, args.execute)
    require(approval.get('decision') == 'D-051' and approval.get('arms') == [ARM] and approval.get('scopes') == ['feasibility'], 'D-051 B only')
    require(approval.get('diagnostic_runner_sha256') == sha(Path(__file__)), 'Unreviewed diagnostic code')
    require(approval.get('numeric_helper_sha256') == sha(ROOT/'scripts/stage_i_d051_numeric.py'), 'Unreviewed numeric comparison code')
    require(approval.get('maximum_physical_optimizer_calls') == 4, 'Scope changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run = OUT / 'execution'
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = optimizer = scheduler = scaler = None
    initial_parameters = None
    updates = []
    error = None
    outcome = None
    micro_count = 0

    def event(name, **value):
        item = {'utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter()-started, 'event': name, **value}
        with (run / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def memory():
        free, total = torch.cuda.mem_get_info()
        return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
                'device_free': free, 'device_total': total, 'device_used': total-free}

    def snapshot(name):
        value = {'adapters': get_peft_model_state_dict(model), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'rng': {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                         'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}}
        # Snapshot contains mutable state_dict tensors: clone all of them immediately.
        save_tensor_evidence(run / (name+'.pt'), value)
        fp = {'adapters': fingerprint(get_peft_model_state_dict(model)), **states(optimizer, scheduler, scaler)}
        write_new(run / (name+'_fingerprints.json'), fp)
        return fp

    try:
        event('D051_start', arm=ARM, distinct_steps=3, replay_updates=1, maximum_microbatches=32)
        train, _, _, mc, cloud = prepared.datasets(config, prep)
        model, scope, initialization = prepared.load_gpu(config, prep, ARM, approval, mc, cloud)
        initial_parameters = parameters(model)
        write_new(run/'parameter_hashes_before.json', initial_parameters)
        flags = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                 'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 'gradient_checkpointing': model.is_gradient_checkpointing, 'use_cache': model.config.use_cache}
        require(flags['deterministic_algorithms'] is False, 'Unexpected deterministic policy; stop without changing it')
        write_new(run/'runtime.json', {'scope': scope, 'initialization': initialization, 'flags': flags,
            'gpu': torch.cuda.get_device_name(), 'GPU_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch': torch.__version__, 'CUDA': torch.version.cuda, 'runner_sha256': sha(Path(__file__)),
            'config_sha256': sha(CONFIG), 'approval_sha256': sha(args.approval)})
        control_ids = read(prep/'feasibility_cases.json')['business_ids']
        features = [train[train.business_ids.index(b)] for b in control_ids]
        collator = prepared.LEDCompleteCollator(5000,768,1024,pad_source_to=5120,pad_label_to=768)
        optimizer, scheduler = prepared.optimizer_scheduler(model,config)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        name_by_identity = {id(p): n for n,p in model.named_parameters()}
        write_new(run/'optimizer_parameter_order.json', [[name_by_identity[id(p)] for p in group['params']] for group in optimizer.param_groups])
        bind = {**prepared.binding(CONFIG,prep,ARM), 'decision':'D-051','diagnostic_runner_sha256':sha(Path(__file__))}
        write_new(run/'binding.json',bind)
        torch.cuda.reset_peak_memory_stats()
        event('model_ready',modules=len(scope['modules']),trainable_parameters=scope['trainable_parameters'],flags=flags)

        def update(step, label):
            nonlocal micro_count
            model.train()
            optimizer.zero_grad(set_to_none=True)
            entry_fp = snapshot(label+'_start') if step == 3 else None
            losses = []
            micros = []
            for j in range(8):
                index = ((step-1)*8+j)%len(features)
                feature = features[index]
                batch = collator([feature])
                input_fingerprint = fingerprint(batch)
                require(feature['input_ids'][0] == feature['labels'][0] == 0 and feature['input_ids'][-1] == feature['labels'][-1] == 2,'BOS/EOS differs')
                verify_mask_batch(batch,[feature])
                batch = {k:v.cuda() for k,v in batch.items()}
                before_rng = states(optimizer,scheduler,scaler) if step == 3 else None
                with torch.autocast('cuda',dtype=torch.float16):
                    output = model(**batch,use_cache=False)
                    loss = output.loss
                scalar = float(loss.detach())
                finite = bool(torch.isfinite(loss).item())
                # Log before the finite assertion. Logit hashes observe complete forward output.
                logits_fingerprint = fingerprint(output.logits) if step == 3 else None
                item = {'label':label,'step':step,'microbatch_in_step':j+1,'physical_microbatch':micro_count+1,
                    'business_id':control_ids[index],'input_fingerprint':input_fingerprint,
                    'source_tokens':len(feature['input_ids']),'label_tokens':len(feature['labels']),
                    'source_shape':list(batch['input_ids'].shape),'label_shape':list(batch['labels'].shape),
                    'loss':scalar if finite else None,'loss_nonfinite_repr':None if finite else repr(scalar),
                    'finite_loss':finite,'logits_fingerprint':logits_fingerprint,
                    'state_before_forward':before_rng,'state_after_forward':states(optimizer,scheduler,scaler) if step == 3 else None,
                    'memory_after_forward':memory()}
                with (run/'microbatch_forward.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
                require(finite,'Nonfinite training loss; evidence preserved')
                losses.append(scalar)
                scaler.scale(loss/8).backward()
                micro_count += 1
                # FP32 accumulated scaled gradients: do not alter them for observation.
                if step == 3:
                    gradient_fingerprints = {n:fingerprint(p.grad) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}
                    item['accumulated_scaled_gradient_fingerprints'] = gradient_fingerprints
                    item['state_after_backward'] = states(optimizer,scheduler,scaler)
                item['memory_after_backward'] = memory()
                with (run/'microbatch_complete.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
                micros.append(item)
                del output, loss, batch
            scaler.unscale_(optimizer)
            if step == 3:
                save_tensor_evidence(run/(label+'_preclip_gradients.pt'),{n:p.grad for n,p in model.named_parameters() if p.requires_grad})
            gradients = prepared.gradient_record(model,expect_later_A=step>=3)
            write_new(run/(label+'_gradient_norms.json'),gradients)
            clipping_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            if step == 3:
                save_tensor_evidence(run/(label+'_postclip_gradients.pt'),{n:p.grad for n,p in model.named_parameters() if p.requires_grad})
            used_lr = optimizer.param_groups[0]['lr']
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            # Capture before step-count assertions or final replay decisions.
            end_fp = snapshot(label+'_end') if step == 3 else None
            record = {'label':label,'step':step,'losses':losses,'mean_loss':sum(losses)/8,'used_lr':used_lr,
                'clipping_norm_before':float(clipping_norm),'scaler_before':scale_before,'scaler_after':scaler.get_scale(),
                'optimizer_parameter_steps':sorted({int(s['step']) for s in optimizer.state.values()}),
                'scheduler_last_epoch_after':scheduler.last_epoch,'memory_after_update':memory(),
                'entry_fingerprints':entry_fp,'end_fingerprints':end_fp,'gradient_norms':gradients}
            write_new(run/(label+'_update.json'),record)
            updates.append(record)
            event('update_completed',label=label,step=step,mean_loss=record['mean_loss'],used_lr=used_lr,
                  peak_allocated_MiB=memory()['peak_allocated']/2**20,peak_reserved_MiB=memory()['peak_reserved']/2**20)
            require(record['optimizer_parameter_steps']==[step],'Skipped AMP or optimizer update')
            return record

        update(1,'step1')
        update(2,'step2')
        # Identical prepared checkpoint procedure; isolated destination/binding only.
        before_save = states(optimizer,scheduler,scaler)
        prepared.save_feas_checkpoint(model,optimizer,scheduler,scaler,run/'checkpoint_step2',bind,2)
        write_new(run/'checkpoint_save_fingerprints.json',{'before':before_save,'after':states(optimizer,scheduler,scaler)})
        update(3,'original_step3')
        saved = torch.load(run/'checkpoint_step2/state.pt',map_location='cpu',weights_only=False)
        expected = {k:fingerprint(saved[k]) for k in ['optimizer','scheduler','scaler','torch_rng','cuda_rng','numpy_rng','python_rng']}
        prepared.restore_feas_checkpoint(model,optimizer,scheduler,scaler,run/'checkpoint_step2',bind)
        restored = states(optimizer,scheduler,scaler)
        matches = {k:restored[k]==v for k,v in expected.items()}
        write_new(run/'checkpoint_restore_fingerprints.json',{'expected':expected,'restored':restored,'exact':matches})
        require(all(matches.values()),'Incorrect checkpoint state restoration')
        update(3,'replayed_step3')
        # All numerical evidence is durable before comparisons/assertions.
        from stage_i_d051_numeric import compare_tensor_maps, compare_nested_states
        first = torch.load(run/'original_step3_end.pt',map_location='cpu',weights_only=False)
        second = torch.load(run/'replayed_step3_end.pt',map_location='cpu',weights_only=False)
        comparison = compare_tensor_maps(first['adapters'],second['adapters'])
        write_new(run/'adapter_replay_comparison.json',comparison)
        write_new(run/'end_state_comparison.json',compare_nested_states(first,second))
        outcome = 'STRICT_REPLAY_PASS' if comparison['overall']['allclose'] else 'STRICT_REPLAY_TOLERANCE_FAILURE'
        write_new(run/'strict_replay_result.json',{'outcome':outcome,'rtol':1e-5,'atol':1e-6,
            'tolerances_changed':False,'additional_correction_runs':0,'microbatches':micro_count,
            'distinct_steps':3,'physical_optimizer_updates':len(updates)})
        event('strict_replay_evaluated',outcome=outcome)
    except Exception as exc:
        error = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        write_new(run/'failure.json',error)
        event('affected_execution_stopped',error_type=type(exc).__name__,message=str(exc))
    finally:
        if model is not None:
            write_new(run/'GPU_MEMORY_FINAL.json',memory())
            after = parameters(model)
            write_new(run/'parameter_hashes_after.json',after)
            frozen_changed = [n for n,v in initial_parameters.items() if not v['requires_grad'] and after[n]['sha256']!=v['sha256']] if initial_parameters else None
            write_new(run/'backbone_integrity.json',{'frozen_changed':frozen_changed,
                'frozen_gradients_absent':all(p.grad is None for p in model.parameters() if not p.requires_grad),
                'only_adapter_parameters_trainable':all('.lora_A.' in n or '.lora_B.' in n for n,p in model.named_parameters() if p.requires_grad)})
        write_new(run/'completion.json',{'outcome':outcome,'error':error,'seconds':time.perf_counter()-started,
            'microbatches':micro_count,'physical_optimizer_calls':len(updates),'pilot_training_started':False,
            'generation_calls':0,'scientific_config_sha256':sha(CONFIG),'diagnostic_runner_sha256':sha(Path(__file__))})
    if error:
        raise RuntimeError(error['type']+': '+error['message'])


if __name__ == '__main__':
    main()
