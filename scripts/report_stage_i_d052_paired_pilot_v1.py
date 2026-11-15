"""Offline D-052 training evidence audit/report; no Torch, model or GPU imports.

Requires both bounded training arms and the root post-execution integrity audit
to finish first. Creates new reports only; never changes saved training evidence.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d052_paired_pilot_v1'
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
ARMS = ('A_decoder_only', 'B_encoder_decoder')
CONFIG = ROOT / 'configs/stage_i_paired_lora_pilot_v1.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def branch(name):
    if '.decoder.layers.' in name:
        return 'decoder'
    if '.encoder.layers.' in name:
        return 'encoder_global' if '_global.' in name else 'encoder_local'
    raise ValueError('Unexpected adapter gradient name: ' + name)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def check_sha_map(folder, mapping):
    for name, digest in mapping.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()), 'Unsafe saved checkpoint member: ' + name)
        require(path.is_file() and sha(path) == digest, 'Checkpoint hash mismatch: ' + str(path))


def source_reference_readiness():
    """Describe saved source-review artifacts; browser localStorage is not evidence."""
    worksheet = ROOT / 'outputs/stage_i/d052_source_reference_review_v1'
    exported = read(worksheet / 'EXPORT_REPORT.json')
    initial = read(worksheet / 'initial_working_state_UNVERIFIED.json')
    candidates = rows(worksheet / 'reference_unit_candidates_UNVERIFIED.jsonl')
    require(len(candidates) == 226 and exported['candidates'] == 226, 'Source candidate count differs')
    require(sha(worksheet / 'reference_unit_candidates_UNVERIFIED.jsonl') == sha(PREP / 'reference_unit_candidates.jsonl'),
            'Prepared source-reference candidates changed')
    verified = sum(row.get('human_verified') is True for row in candidates)
    require(verified == 0 and all(row.get('human_verified') is False for row in candidates),
            'Unexpected candidate-label change; review explicit human evidence separately')
    require(exported['human_verified_candidates'] == 0 and exported['final_units'] == 0 and initial['units'] == [],
            'Unverified worksheet initial state differs')
    require(initial['researcher_confirmed_complete'] is False and
            all(row.get('human_verified') is False for row in initial['business_reviews'].values()),
            'Initial source-completeness worksheet already contains confirmation')
    known_final_locations = [worksheet / 'stage_i_source_reference_user_confirmed.json',
                            OUT / 'stage_i_source_reference_user_confirmed.json',
                            PREP / 'stage_i_source_reference_user_confirmed.json',
                            ROOT / 'outputs/stage_i/stage_i_source_reference_user_confirmed.json']
    present = [path.relative_to(ROOT).as_posix() for path in known_final_locations if path.is_file()]
    status = ('EXPLICIT_REFERENCE_FILE_PRESENT_REQUIRES_SEPARATE_VALIDATION' if present
              else 'SOURCE_REFERENCE_HUMAN_VERIFICATION_PENDING')
    evidence = [worksheet / name for name in ['EXPORT_REPORT.json', 'initial_working_state_UNVERIFIED.json',
                                              'reference_unit_candidates_UNVERIFIED.jsonl']]
    return {'status': status, 'candidate_units': len(candidates),
            'human_confirmed_candidates_in_saved_initial_artifacts': verified,
            'human_confirmed_final_units_in_saved_initial_artifacts': len(initial['units']),
            'human_confirmed_businesses_in_saved_initial_artifacts': 0,
            'source_review_businesses': exported['businesses'], 'original_photos': exported['photos'],
            'known_final_reference_files_present': present,
            'known_final_reference_locations_checked': [path.relative_to(ROOT).as_posix() for path in known_final_locations],
            'validated_human_source_reference_freeze_available': False,
            'candidate_suggestions_are_not_human_ground_truth': True,
            'browser_localStorage_or_unsubmitted_downloads_inspected': False,
            'scope_note': 'Counts describe unchanged saved initial worksheet artifacts. User browser progress or unsubmitted files are unknown, never inferred as human labels. A separately supplied final reference still requires offline schema/evidence validation.',
            'evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in evidence},
            'new_predictions_authorized_by_this_report': False}


def validate_arm(arm, config, manifest, epoch_orders):
    audit = OUT / arm
    train = ROOT / config['run_root'] / arm / 'train'
    done = read(train / 'completion.json')
    observed = read(audit / 'observation_completion.json')
    state = read(train / 'trainer_state.json')
    runtime = read(audit / 'runtime.json')
    arguments = read(audit / 'actual_training_arguments.json')
    order_audit = read(audit / 'ACTUAL_ORDER_VERIFICATION.json')
    frozen = read(audit / 'FROZEN_BACKBONE_INTEGRITY.json')
    memory = read(audit / 'GPU_MEMORY_FINAL.json')
    before = read(audit / 'parameter_hashes_before.json')
    after = read(audit / 'parameter_hashes_after.json')
    micro = rows(audit / 'microbatch_loss_and_order.jsonl')
    updates = rows(audit / 'optimizer_scheduler_scaler_memory.jsonl')
    gradients = rows(train / 'gradients.jsonl')
    events = rows(audit / 'events.jsonl')

    require(done['status'] == 'PILOT_TRAINING_COMPLETE_NO_GENERATION', arm + ' training incomplete')
    require(observed['status'] == 'TRAINING_AND_AUDIT_PASS' and observed['error'] is None, arm + ' wrapper audit failed')
    require(done['arm'] == arm and done['steps'] == 128 and done['epochs'] == 2, arm + ' duration differs')
    require(state['global_step'] == 128 and float(state['epoch']) == 2, arm + ' Trainer duration differs')
    require(observed['actual_optimizer_steps'] == 128 and observed['actual_microbatches'] == 1024,
            arm + ' actual training count differs')
    require(observed['validation_batches'] == 400, arm + ' validation count differs')
    require(done['generation_calls'] == observed['generation_calls'] == 0, arm + ' unauthorized generation')
    require(not any((train.parent / name).exists() for name in ['generation', 'generate']), arm + ' unexpected generation directory')
    require(observed['D051_replay_still_FAIL'] and observed['conditional_authorization_only'], arm + ' acceptance semantics differ')
    require(runtime['config_sha256'] == done['binding']['config_sha256'] == sha(CONFIG), arm + ' configuration hash drift')
    require(runtime['wrapper_sha256'] == sha(ROOT / 'scripts/run_stage_i_d052_paired_pilot_v2.py'),
            arm + ' final run was not produced by the reviewed corrected v2 wrapper')
    require(done['epoch_order_sha256'] == sha(PREP / 'epoch_orders.json'), arm + ' order binding differs')
    require(done['binding']['preparation_manifest_sha256'] == sha(PREP / 'PREPARATION_MANIFEST.json'), arm + ' preparation binding differs')

    expected_modules, expected_parameters = (24, 294912) if arm == ARMS[0] else (48, 589824)
    scope = runtime['scope']
    require(len(scope['modules']) == expected_modules and len(set(scope['modules'])) == expected_modules, arm + ' module count differs')
    intended = {f'led.decoder.layers.{layer}.{attention}.{projection}' for layer in range(6)
                for attention in ['self_attn', 'encoder_attn'] for projection in ['q_proj', 'v_proj']}
    if arm == ARMS[1]:
        intended |= {f'led.encoder.layers.{layer}.self_attn.longformer_self_attn.{projection}' for layer in range(6)
                     for projection in ['query', 'value', 'query_global', 'value_global']}
    require(set(scope['modules']) == intended, arm + ' actual attention adapter targets differ')
    require(scope['trainable_parameters'] == expected_parameters, arm + ' trainable count differs')
    require(set(before) == set(after), arm + ' parameter names changed')
    trainable = [name for name, item in before.items() if item['requires_grad']]
    frozen_names = [name for name, item in before.items() if not item['requires_grad']]
    require(len(trainable) == expected_modules * 2, arm + ' trainable tensor count differs')
    require(sum(before[name]['numel'] for name in trainable) == expected_parameters, arm + ' actual parameter count differs')
    require(all('lora_A.' in name or 'lora_B.' in name for name in trainable), arm + ' nonadapter trainable parameter')
    require(all(before[name]['sha256'] == after[name]['sha256'] for name in frozen_names), arm + ' frozen parameter changed')
    require(all(before[name]['requires_grad'] == after[name]['requires_grad'] for name in before), arm + ' trainability changed')
    require(not frozen['changed_frozen_tensors'] and frozen['frozen_gradients_absent'], arm + ' backbone integrity failed')

    expected_args = {
        'num_train_epochs': 2, 'per_device_train_batch_size': 1, 'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 8, 'learning_rate': 2e-4, 'weight_decay': 0.01,
        'optim': 'adamw_torch',
        'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.05,
        'fp16': True, 'bf16': False, 'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'eval_strategy': 'epoch', 'save_strategy': 'epoch', 'save_total_limit': 2,
        'load_best_model_at_end': True, 'metric_for_best_model': 'eval_loss', 'greater_is_better': False,
        'seed': 42, 'data_seed': 42, 'full_determinism': False, 'remove_unused_columns': False,
        'label_names': ['labels'], 'predict_with_generate': False, 'label_smoothing_factor': 0.0,
        'logging_steps': 8, 'dataloader_num_workers': 0,
    }
    for key, value in expected_args.items():
        require(arguments.get(key) == value, arm + ' actual TrainingArguments differ: ' + key)
    require(not runtime['deterministic_algorithms'], arm + ' deterministic policy unexpectedly changed')
    attached = [event for event in events if event['event'] == 'trainer_audit_attached']
    require(len(attached) == 1 and 'AuditedTrainer' in attached[0]['class_MRO'], arm + ' observer MRO attachment not demonstrated')
    started = [event for event in events if event['event'] == 'training_started']
    require(len(started) == 1 and attached[0]['seconds'] <= started[0]['seconds'], arm + ' observer not attached before training')

    expected_order = [bid for epoch in epoch_orders['business_ids'] for bid in epoch]
    require(len(micro) == 1024 and [row['business_id'] for row in micro] == expected_order,
            arm + ' actual microbatch order differs')
    require([row['physical_microbatch'] for row in micro] == list(range(1, 1025)), arm + ' microbatch indices differ')
    require(order_audit['exact_epoch_plans_match'] and order_audit['validation_all200_twice'], arm + ' saved order audit failed')
    require(order_audit['train_microbatches'] == 1024 and order_audit['validation_batches'] == 400, arm + ' saved order counts differ')
    require(order_audit['train_ids_order_sha256'] == hashlib.sha256(json.dumps(expected_order).encode()).hexdigest(), arm + ' order digest differs')
    require(len(updates) == len(gradients) == 128, arm + ' update/gradient records missing')

    gradient_summary = []
    for index, (update, grad) in enumerate(zip(updates, gradients), 1):
        require(update['step'] == update['global_step'] == grad['next_step'] == index, arm + ' step sequence differs')
        require(update['optimizer_steps'] == [index] and update['scheduler_last_epoch'] == index,
                arm + ' optimizer or scheduler step differs')
        require(not update['AMP_optimizer_skipped'], arm + ' skipped AMP update')
        used_factor = (index - 1) / 7 if index - 1 < 7 else (128 - (index - 1)) / 121
        next_factor = index / 7 if index < 7 else max(0, 128 - index) / 121
        require(close(update['used_lr'], 2e-4 * used_factor), arm + ' used LR differs')
        require(len(update['next_lr']) == 2 and all(close(x, 2e-4 * next_factor) for x in update['next_lr']), arm + ' next LR differs')
        for key in ['scaler_before', 'scaler_after']:
            require(update[key] is not None and math.isfinite(update[key]) and update[key] > 0, arm + ' missing/nonfinite scaler')
        require(set(grad['gradients']) == set(trainable), arm + ' gradient parameter geometry differs')
        grouped = {}
        for name, value in grad['gradients'].items():
            require(value['finite'] and math.isfinite(value['norm']) and value['norm'] >= 0, arm + ' nonfinite gradient')
            require(value['nonzero'] or ('.lora_A.' in name and index <= 2), arm + ' inactive intended gradient')
            item = grouped.setdefault(branch(name), {'tensors': 0, 'nonzero_A': 0, 'nonzero_B': 0, 'squared_norm': 0.0})
            item['tensors'] += 1
            item['nonzero_A' if '.lora_A.' in name else 'nonzero_B'] += int(value['nonzero'])
            item['squared_norm'] += value['norm'] ** 2
        for name, item in grouped.items():
            item['L2_from_recorded_tensor_norms'] = math.sqrt(item.pop('squared_norm'))
            gradient_summary.append({'arm': arm, 'step': index, 'branch': name, **item})

    train_losses = [8 * row['training_step_return_loss_after_accumulation_scaling'] for row in micro]
    require(all(math.isfinite(value) for value in train_losses), arm + ' nonfinite microbatch loss')
    eval_entries = [entry for entry in state['log_history'] if 'eval_loss' in entry]
    require([entry['step'] for entry in eval_entries] == [64, 128], arm + ' validation epochs differ')
    require(all(math.isfinite(entry['eval_loss']) for entry in eval_entries), arm + ' nonfinite validation loss')
    val_events = [event for event in events if event['event'] == 'validation_completed']
    require([event['step'] for event in val_events] == [64, 128] and
            [event['validation_examples_seen'] for event in val_events] == [200, 400], arm + ' observed validation events differ')
    best_entry = min(eval_entries, key=lambda entry: entry['eval_loss'])
    best_step = best_entry['step']
    require(close(done['best_eval_loss'], best_entry['eval_loss']) and close(state['best_metric'], best_entry['eval_loss']), arm + ' best-loss rule differs')
    require(Path(done['best_checkpoint']).name == Path(state['best_model_checkpoint']).name == f'checkpoint-{best_step}', arm + ' best checkpoint identity differs')

    checkpoints = []
    for step in [64, 128]:
        folder = train / f'checkpoint-{step}'
        proof = read(folder / 'checkpoint_integrity.json')
        audit_record = read(audit / f'checkpoint_{step}_audit.json')
        require(proof['binding'] == done['binding'], arm + ' checkpoint binding differs')
        check_sha_map(folder, proof['files'])
        check_sha_map(folder, audit_record['sha256'])
        require(all((folder / name).is_file() for name in audit_record['required_files']), arm + ' checkpoint state missing')
        checkpoints.append({'step': step, 'path': folder.relative_to(ROOT).as_posix(),
                            'files': {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                                      for path in sorted(folder.iterdir()) if path.is_file()}})
    best_adapter = train / 'best_adapter/adapter_model.safetensors'
    require(sha(best_adapter) == done['best_adapter_sha256'] == sha(train / f'checkpoint-{best_step}/adapter_model.safetensors'),
            arm + ' saved best adapter differs from selected epoch checkpoint')

    epochs = []
    for epoch, entry in enumerate(eval_entries, 1):
        losses = train_losses[(epoch - 1) * 512:epoch * 512]
        epochs.append({'epoch': epoch, 'optimizer_step': entry['step'], 'training_microbatches': len(losses),
                       'training_mean_microbatch_loss': statistics.fmean(losses),
                       'training_loss_min': min(losses), 'training_loss_max': max(losses),
                       'validation_businesses': 200, 'validation_loss': entry['eval_loss']})
    return {
        'arm': arm, 'status': 'OFFLINE_TRAINING_EVIDENCE_PASS', 'train_directory': train.relative_to(ROOT).as_posix(),
        'modules': expected_modules, 'trainable_parameters': expected_parameters, 'trainable_tensors': len(trainable),
        'frozen_tensors': len(frozen_names), 'frozen_parameters': sum(before[name]['numel'] for name in frozen_names),
        'changed_frozen_tensors': [], 'frozen_gradients_absent': True,
        'changed_adapter_tensors': sum(before[name]['sha256'] != after[name]['sha256'] for name in trainable),
        'canonical_initial_backbone_hashes': {name.removeprefix('base_model.model.').replace('.base_layer.', '.'): before[name]['sha256']
                                             for name in frozen_names},
        'initial_decoder_hashes': read(audit / 'initial_decoder_hashes.json'),
        'runtime': runtime, 'verified_training_arguments': expected_args,
        'epochs': epochs, 'best_step': best_step, 'best_eval_loss': best_entry['eval_loss'],
        'best_adapter': {'path': best_adapter.relative_to(ROOT).as_posix(), 'sha256': sha(best_adapter)},
        'checkpoints': checkpoints, 'optimizer_updates': len(updates), 'microbatches': len(micro),
        'validation_batches': observed['validation_batches'], 'AMP_skipped_updates': 0,
        'scaler_values_observed': sorted({row[key] for row in updates for key in ['scaler_before', 'scaler_after']}),
        'zero_gradient_initialization_behavior': 'A matrices may be zero at steps1/2; B finite/nonzero from step1; all intended A/B nonzero from step3',
        'gradient_records': gradient_summary, 'memory': memory,
        'training_loop_seconds': done['seconds'], 'wrapper_total_seconds': observed['seconds'],
        'generation_calls': 0, 'checkpoint_states_preserved_not_replayed_in_D052': True,
        'source_label_mask_verification': 'Audited every training/validation batch against exact accepted token arrays; no generation',
        'trainer_state_sha256': sha(train / 'trainer_state.json'),
    }


def fmt(value):
    return f'{value:.6f}'


def main():
    targets = [OUT / name for name in ['D052_TRAINING_VALIDATION.json', 'PER_EPOCH_LOSS_COMPARISON.csv',
                                      'GRADIENT_BRANCH_SUMMARY.csv', 'STAGE_I_D052_PAIRED_TRAINING_REPORT.md',
                                      'SOURCE_REFERENCE_READINESS.json']]
    require(all(not path.exists() for path in targets), 'Preserve previous reports; use a new version for revisions')
    require(read(OUT / 'POST_EXECUTION_INTEGRITY.json')['status'] == 'PASS', 'Post-execution preservation audit missing/not PASS')
    config = read(CONFIG)
    interrupted = OUT / 'interrupted_attempt01'
    interrupted_manifest = read(interrupted / 'MANIFEST.json')
    for item in interrupted_manifest['files']:
        require(sha(interrupted / item['path']) == item['sha256'], 'Discarded attempt evidence changed')
    require(not list((interrupted / 'core_train').glob('checkpoint-*')) and
            not (interrupted / 'core_train/best_adapter').exists() and
            not (interrupted / 'core_train/completion.json').exists(), 'Discarded attempt unexpectedly produced selected checkpoints')
    partial_log = (interrupted / 'execution.log').read_bytes()
    partial_encoding = 'utf-16' if partial_log.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    partial_progress = [int(value) for value in re.findall(r'(\d+)/128', partial_log.decode(partial_encoding))]
    require(partial_progress and max(partial_progress) == 27, 'Discarded observed progress differs; update report interpretation')
    discarded = {'directory': interrupted.relative_to(ROOT).as_posix(), 'observed_last_progress': '27/128',
                 'reason': 'Supplemental audit subclass not attached due to stale Transformers lazy module object',
                 'core_scientific_math_and_finite_gradient_checks_were_active': True,
                 'selected_checkpoint_produced': False, 'weights_reused_in_corrected_runs': False,
                 'corrected_runs': 'Independent fresh v2 A and B, each exactly128 updates/two epochs',
                 'correction': 'Warm exact imports, resolve live module, assert audit MRO and guard before Trainer.train',
                 'counted_in_matched_results': False, 'scientific_hyperparameters_changed': False,
                 'manifest_sha256': sha(interrupted / 'MANIFEST.json')}
    manifest = read(PREP / 'pilot_manifest.json')
    orders = read(PREP / 'epoch_orders.json')
    require(len(manifest['training_business_ids']) == len(set(manifest['training_business_ids'])) == 512, 'Train manifest differs')
    require(len(manifest['validation_loss_business_ids']) == len(set(manifest['validation_loss_business_ids'])) == 200, 'Validation manifest differs')
    require(not set(manifest['training_business_ids']) & set(manifest['validation_loss_business_ids']), 'Split overlap')
    require(len(orders['business_ids']) == 2 and all(len(epoch) == 512 and set(epoch) == set(manifest['training_business_ids'])
                                                  for epoch in orders['business_ids']), 'Epoch plans differ')
    results = [validate_arm(arm, config, manifest, orders) for arm in ARMS]
    require(results[0]['initial_decoder_hashes'] == results[1]['initial_decoder_hashes'], 'Paired decoder initialization differs')
    require(results[0]['canonical_initial_backbone_hashes'] == results[1]['canonical_initial_backbone_hashes'],
            'Paired initial pretrained backbone hashes differ')
    require(len(results[0]['initial_decoder_hashes']) == 24, 'Initial decoder module hashes incomplete')
    for key in ['gpu', 'torch', 'CUDA', 'deterministic_algorithms', 'cudnn_deterministic', 'cudnn_benchmark',
                'matmul_allow_tf32', 'cudnn_allow_tf32']:
        require(results[0]['runtime'][key] == results[1]['runtime'][key], 'Unmatched runtime: ' + key)
    comparison = []
    for index in range(2):
        a, b = [result['epochs'][index] for result in results]
        comparison.append({'epoch': index + 1, 'optimizer_step': a['optimizer_step'],
                           'A_training_mean_microbatch_loss': a['training_mean_microbatch_loss'],
                           'B_training_mean_microbatch_loss': b['training_mean_microbatch_loss'],
                           'A_validation_loss': a['validation_loss'], 'B_validation_loss': b['validation_loss'],
                           'B_minus_A_validation_loss': b['validation_loss'] - a['validation_loss']})
    reference_readiness = source_reference_readiness()
    status = {
        'status': 'PAIRED_TRAINING_AND_OFFLINE_AUDIT_PASS_NO_GENERATION', 'decision': 'D-052',
        'D051_strict_replay_gate': 'FAIL_UNCHANGED_RESEARCHER_CONDITIONALLY_ACCEPTED_FOR_D052_ONLY',
        'discarded_operational_attempt': discarded,
        'corrected_wrapper': 'scripts/run_stage_i_d052_paired_pilot_v2.py',
        'config_sha256': sha(CONFIG), 'preparation_manifest_sha256': sha(PREP / 'PREPARATION_MANIFEST.json'),
        'post_execution_integrity_sha256': sha(OUT / 'POST_EXECUTION_INTEGRITY.json'),
        'matched_decoder_initialization': True, 'matched_train512_order': True, 'validation_all200_per_epoch': True,
        'new_generation_calls': 0, 'held_out_examples_accessed_by_reporting': 0,
        'quality_evaluation_performed': False, 'source_reference_human_verification': reference_readiness['status'],
        'source_reference_readiness': reference_readiness,
        'source_reference_candidates_are_not_human_ground_truth': True, 'candidate_units': 226,
        'training_loss_definition': 'Arithmetic mean of 512 per-microbatch returned losses times8 per epoch. The accepted Trainer returns mean token loss divided by accumulation8; this is an example/microbatch mean, not a pooled-token mean or rounded logging mean.',
        'uncertainty': 'One paired seed/two epochs; loss difference is descriptive. No quality confidence interval or significance test, no new predictions.',
        'epoch_comparison': comparison, 'architectures': results,
        'further_generation_requires_separate_approval_and_human_source_reference_freeze': True,
    }
    write_new(targets[4], reference_readiness)
    write_new(targets[0], status)
    with targets[1].open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    grad_rows = [row for result in results for row in result['gradient_records']]
    with targets[2].open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grad_rows[0]))
        writer.writeheader(); writer.writerows(grad_rows)

    lines = [
        '# Stage I - D-052 paired Filtered LoRA training report', '',
        '**Both two-epoch training arms and offline technical audits completed. No new student summaries were generated.**', '',
        'D-051 strict replay remains **FAIL** at rtol1e-5/atol1e-6. The researcher explicitly accepted its documented small encoder-backward variability for this exploratory experiment. This report is not a retrospective replay PASS or an approval for further execution.', '',
        '## Discarded operational attempt and instrumentation correction', '',
        'An early A attempt was stopped at observed progress27/128 before its first epoch/checkpoint because the supplemental audit Trainer subclass had not attached. The original D-049 training math/configuration and finite-gradient checks were active, but the additional order/optimizer/scaler evidence was missing. Its raw logs, partial core records, original wrapper and authorization are preserved under interrupted_attempt01. No selected checkpoint was produced and no runtime weights were reused.', '',
        'CPU inspection identified stale Transformers lazy-module import wiring. The versioned v2 wrapper warms the exact imports, patches the live module and verifies the AuditedTrainer MRO with a fail-closed guard before training. This is an instrumentation/import correction; no scientific settings or original D-049 training math changed. All tables below cover only the independently restarted, fully audited v2 A/B runs, each128 updates/two epochs. The discarded prefix is not a matched experimental arm or an extension to either completed pilot.', '',
        '## Matched experiment and integrity', '',
        f"Both arms use pinned `{config['model_id']}` revision `{config['revision']}`, Filtered only, the same fixed512 training businesses, all200 validation businesses, seed42 and exact frozen two-epoch business ordering. Each arm completed128 optimizer updates,1024 physical training microbatches and400 teacher-forced validation batches. No held-out inputs or Teacher API calls were needed.", '',
        'Each model was initialized independently from the pretrained backbone with matching decoder adapter hashes across the24 shared modules. Historical full-F and D-050/D-051 weights were not used as starting points. No source/target truncation, caption/filter changes or accepted dataset changes occurred; see POST_EXECUTION_INTEGRITY.json and batch-verification code.', '',
        '| Arm | Attention projections | Trainable parameters | Adapter tensors changed | Frozen tensors unchanged | Best validation step |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for result in results:
        lines.append(f"| {result['arm']} | {result['modules']} | {result['trainable_parameters']:,} | {result['changed_adapter_tensors']}/{result['trainable_tensors']} | {result['frozen_tensors']} | {result['best_step']} |")
    lines += ['', 'A adapts decoder self/cross-attention Q/V. B additionally adapts encoder local/global Q/V. All pretrained parameters remain frozen and byte-identical; adapter attachment and trainability are verified against actual runtime scope records.', '',
              '## Per-epoch loss', '',
              '| Epoch / optimizer step | A train mean | B train mean | A validation loss | B validation loss | B minus A validation |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in comparison:
        lines.append(f"| {row['epoch']} / {row['optimizer_step']} | {fmt(row['A_training_mean_microbatch_loss'])} | {fmt(row['B_training_mean_microbatch_loss'])} | {fmt(row['A_validation_loss'])} | {fmt(row['B_validation_loss'])} | {fmt(row['B_minus_A_validation_loss'])} |")
    lines += ['', 'Training figures are reconstructed from all saved microbatch returns multiplied by accumulation8, then averaged over each512-example epoch. They are not the rounded logging summaries and are not a pooled-token average. Validation figures are the saved teacher-forced eval_loss on the same200 businesses. Each arm selects its own minimum-validation-loss checkpoint under the predeclared rule.', '',
              '**Lower validation loss measures teacher-target predictive agreement, not independent factual accuracy, informative summaries or causal proof of an encoder benefit. No summary quality, repetition, grounding or informativeness comparison has yet been performed.** One seed and two epochs provide no quality uncertainty estimate; loss differences are descriptive.', '',
              '## Optimization and actual gradients', '',
              'Matched AdamW LR2e-4, weight decay0.01, betas0.9/0.999, epsilon1e-8, clipping1, physical batch1/accumulation8, FP16 AMP, FP32 weights, non-reentrant activation checkpointing, linear128-step scheduler/warmup7, rank8/alpha16/dropout0. Every recorded optimizer and scheduler step and used/next LR matches the expected schedule; no AMP update was skipped.', '',
              'All recorded adapter gradients are finite. LoRA A matrices may have zero gradients at steps1/2 because B starts at zero and the initial scheduled LR is zero; this expected behavior is not treated as a missing-gradient defect. Intended A/B gradients are nonzero from step3 onward. Frozen backbone gradients remain absent.', '',
              '| Arm / branch | Step1 nonzero A/B tensors | Step2 nonzero A/B | Step3 nonzero A/B | Step128 nonzero A/B |',
              '| --- | ---: | ---: | ---: | ---: |']
    for result in results:
        branches = sorted({row['branch'] for row in result['gradient_records']})
        for name in branches:
            records = {row['step']: row for row in result['gradient_records'] if row['branch'] == name}
            values = [f"{records[step]['nonzero_A']}/{records[step]['nonzero_B']}" for step in [1, 2, 3, 128]]
            lines.append('| ' + result['arm'] + ' / ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'Gradient summaries contain norms of the recorded per-tensor gradients at the Trainer pre-optimizer callback (after the Trainer clipping stage); they are not independently measured unscaled preclip gradients. The full128-step branch CSV and raw gradients.jsonl preserve all observations.', '',
              '## Checkpoints, runtime and memory', '',
              '| Arm | Training-loop seconds | Wrapper total seconds | Peak allocated MiB | Peak reserved MiB | Best checkpoint / eval loss |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for result in results:
        lines.append(f"| {result['arm']} | {result['training_loop_seconds']:.2f} | {result['wrapper_total_seconds']:.2f} | {result['memory']['peak_allocated']/2**20:.2f} | {result['memory']['peak_reserved']/2**20:.2f} | checkpoint-{result['best_step']} / {fmt(result['best_eval_loss'])} |")
    lines += ['', 'The same GPU/software versions and numerical-execution flags are checked across arms. Times include the stated observation overhead and validation/checkpoint work; they are not pure-kernel benchmarks. Allocated/reserved values are PyTorch allocator maxima, not whole-device VRAM or a guarantee for other sequence/batch configurations.', '',
              'Epoch1/2 checkpoints independently retain adapter weights/config, optimizer, scheduler, scaler, RNG, Trainer state and integrity manifests. Every saved hash is rechecked, and each best_adapter binary matches its selected checkpoint exactly. No D-052 restore/replay or fresh-process recovery was performed; preserved state files do not establish strict replay reproducibility.', '']
    for result in results:
        lines.append(f"- {result['arm']} best adapter: `{result['best_adapter']['path']}`; SHA-256 `{result['best_adapter']['sha256']}`.")
    lines += ['', 'Full per-checkpoint file identities/hashes, optimizer/scaler details and runtime bindings are in D052_TRAINING_VALIDATION.json and the raw arm directories.', '',
              '## Source-reference readiness and stop', '',
              'The frozen20-random/4-targeted source-based evaluation roles and D-049 evaluation protocol remain unchanged. The226 candidate reference units are suggestions, not human-confirmed facts. The researcher must explicitly verify/finalize atomic source-supported information units before new student predictions are generated or inspected. Teacher targets remain auxiliary references, not independent factual ground truth. Any source-reference interface/status included in the bundle must retain this distinction.', '',
              f"Saved source worksheet readiness: **{reference_readiness['status']}**. Its226 unchanged candidates contain **0 human confirmations**, the saved initial final-unit list is empty, and no business is confirmed complete in the saved initial state. The checked explicit final-reference locations contain {len(reference_readiness['known_final_reference_files_present'])} submitted files; no validated human reference freeze is available to this report. Browser localStorage and unsubmitted downloads were not inspected and cannot be inferred as annotation evidence. See SOURCE_REFERENCE_READINESS.json for checked paths and exact source hashes.", '',
              'This pilot changes both adapter location and parameter count; it cannot isolate location alone. Two epochs may be insufficient, and a mixed/negative result would not establish encoder LoRA ineffectiveness. No inference is made for RO/U, held-out businesses, long-run convergence or actual generated quality.', '',
              '**Stop after D-052 training/reporting.** New predictions require separate researcher approval and verified source-reference freeze. Do not extend training, run decoding experiments or change accepted methodology automatically.', '']
    with targets[3].open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'status': status['status'], 'report': targets[3].relative_to(ROOT).as_posix(),
                      'epoch_comparison': comparison, 'new_GPU_model_or_API_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
