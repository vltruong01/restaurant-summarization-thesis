"""Minimal full-training handoff after the accepted v4 gate. Import is CPU-only."""
from pathlib import Path
from stage_i_led_large_runtime_v2 import *
import stage_i_led_large_runtime_v2 as frozen

CONFIG = ROOT/'configs/stage_i_led_large_full_train_v5.json'
PREP = ROOT/'outputs/stage_i/led_large_full_train_v5_preparation'
RUNNER = ROOT/'scripts/run_stage_i_led_large_full_train_v5.py'
DEADLINE = '2026-10-21T17:11:48+00:00'
GATE_BINDING = dict(
    config_sha256='36bf45f86d070eb8432ee7f42cb557616da54b6c2f605affb98655a39ed86568',
    runner_sha256='27e57f6242ac99e918c35141256d1b308f0cf5e8eeb88820f1b62d5e1c568c5e',
    preflight_manifest_sha256='05644eefe532fa5fba975baed97b53ad6a01af81642f862c75640d871e644b7e')
GATE_SHA = '3bfbf2248b676c3fa568af77f205905370390b946d33af15aa2b49301d6d51cc'
AMP = dict(init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)


def binding():
    return dict(config_sha256=sha(CONFIG), runner_sha256=sha(RUNNER),
                preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))


def preflight():
    original = frozen.preflight()  # Original runtime pins, tokenizer, data and dependency hashes.
    c = read(CONFIG)
    expected = dict(original, amp_scaler=AMP, accepted_gate_binding=GATE_BINDING,
                    accepted_gate_report_sha256=GATE_SHA, full_training_version=5)
    require(c == expected, 'Unexpected training/protocol amendment')
    for rel,digest in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel) == digest, 'Full-training dependency changed: '+rel)
    return c


def authorize(path, mode, execute):
    require(execute and path is not None, 'GPU disabled: explicit full-training approval required')
    a = read(path)
    require(a.get('GPU_execution_approved') is True and a.get('training_authorized') is True
            and a.get('researcher_explicit_authorization'), 'Full-training approval missing')
    require(a.get('experiment') == 'LED_LARGE_F_FULL_TRAIN_V5'
            and a.get('approval_source') in ('researcher_notebook_cell', 'researcher_message'), 'Wrong approval source/scope')
    require(a.get('scopes') == ['train','evaluate'] and mode in a['scopes'], 'No additional GPU scope authorized')
    require(all(a.get(k) == v for k,v in binding().items()), 'Full-training approval hash mismatch')
    require(a.get('max_training_updates') == 1250 and a.get('evaluation_outputs_max') == 48
            and a.get('new_gate_updates') == 0 and a.get('new_gate_generation_calls') == 0, 'Budget changed')
    require(a.get('accepted_gate_report_sha256') == GATE_SHA and a.get('amp_scaler') == AMP, 'Accepted gate/AMP differs')
    require(a.get('execution_deadline_utc') == DEADLINE
            and a.get('original_authorization_utc') == '2026-10-19T17:11:48+00:00', 'Historical clock changed')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False, 'Additional scope forbidden')
    check_deadline(a)
    return dict(a, _approval_sha256=sha(path))


def validate_gate(gate_output):
    """Read-only reuse of v4; never relabel its binding or reuse its adapters."""
    p = Path(gate_output)/'feasibility'
    require(sha(p/'REPORT.json') == GATE_SHA, 'Accepted gate report differs')
    r = read(p/'REPORT.json')
    require(r['binding'] == GATE_BINDING == read(p/'BINDING.json') and r['status'] == 'PASS', 'v4 gate mismatch')
    require(r['optimizer_updates'] == 3 and r['microbatches'] == 24 and r['generation_calls'] == 2
            and r['replay_updates'] == 0 and r['headroom_fraction'] >= .15, 'v4 gate scope/memory differs')
    require(r['deadline_utc'] == DEADLINE and r['state_restore_exact'] and r['adapter_restore_exact']
            and r['frozen_backbone_unchanged'] and r['D051_strict_replay'] == 'FAIL_UNCHANGED', 'v4 gate integrity differs')
    require(not (p/'FAILURE_STOP_NO_RETRY.json').exists(), 'Failed gate cannot permit training')
    initial = read(p/'INITIAL_MODEL_BINDINGS.json')
    require(initial['initialization'] == read(frozen.PREP/'INITIALIZATION_HASHES.json')['per_module_sha256'],
            'Gate initialization differs from fresh training initialization')
    require(initial['scope'] == dict(modules=228, trainable_tensors=456, trainable_parameters=4915200,
                                     frozen_backbone_parameters=459801600), 'Wrong gate adapter scope')
    require(sha(p/'checkpoint_step3/integrity.json') ==
            '69fe359f09026609b7228094618f0a706eac8184965d59db6c243d30253aa66e',
            'Reviewed gate checkpoint receipt differs')
    proof = read(p/'checkpoint_step3/integrity.json')
    require(proof['binding'] == GATE_BINDING, 'Foreign gate checkpoint')
    require({'README.md','adapter_model.safetensors','adapter_config.json','state.pt'} == set(proof['files']),
            'Incomplete gate checkpoint inventory')
    for name,digest in proof['files'].items():
        require(Path(name).name == name and sha(p/'checkpoint_step3'/name) == digest,
                'Existing gate checkpoint missing/changed: '+name)
    return r


def require_gate(out, approval):
    gate_output = Path(approval['_gate_output']).resolve()
    require(gate_output != Path(out).resolve(), 'Full training needs a fresh namespace separate from gate')
    r = validate_gate(gate_output)
    require(approval['accepted_gate_report_sha256'] == GATE_SHA
            and approval['execution_deadline_utc'] == r['deadline_utc'], 'Gate authorization differs')
    return r


def configure_amp_scaler(accelerator, c):
    """Set the existing Accelerate scaler through its public state API before any backward.

    The same object is retained for AcceleratedOptimizer/checkpoint serialization.
    No optimizer, loss math, precision, growth factors or skip guards are changed.
    """
    scaler = accelerator.scaler
    require(scaler is not None and scaler.is_enabled(), 'FP16 GradScaler required; no fallback')
    require(getattr(scaler, '_scale', None) is None and getattr(scaler, '_growth_tracker', None) is None,
            'Scaler already used before initialization; no reset allowed')
    require(c['amp_scaler'] == AMP, 'Unexpected AMP configuration')
    before = scaler.state_dict()
    expected_before = dict(scale=65536.0, growth_factor=2.0, backoff_factor=.5, growth_interval=2000, _growth_tracker=0)
    require(before == expected_before, 'Pinned Trainer scaler defaults differ')
    after = dict(before, scale=c['amp_scaler']['init_scale'])
    scaler.load_state_dict(after)
    require(accelerator.scaler is scaler and scaler.state_dict() == after, 'Actual Trainer AMP scale differs')
    return dict(before=before, after=scaler.state_dict(), same_scaler_object=True,
                initialized_before_any_forward_backward=True, accepted_v4_gate_report_sha256=GATE_SHA)


def callback_class():
    from transformers import TrainerCallback
    class Record(TrainerCallback):
        def __init__(self, folder, approval, memory):
            self.folder, self.approval, self.memory = folder, approval, memory
            self.trainer = None

        def on_train_begin(self, args, state, control, **kw):
            require(state.max_steps == 1250 and state.global_step <= 1250, 'Schedule differs')
            check_deadline(self.approval)
            require(self.trainer.accelerator.scaler.is_enabled(), 'Training AMP disabled')

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            from stage_i_paired_pilot import gradient_record
            gradients = gradient_record(model, expect_later_A=state.global_step>=2)
            require(len(gradients) == 456, 'Missing large adapter gradients')
            append(self.folder/'gradient_checks.jsonl', dict(step=state.global_step+1, gradients=gradients))

        def on_optimizer_step(self, args, state, control, optimizer=None, **kw):
            require(self.trainer is not None and not self.trainer.accelerator.optimizer_step_was_skipped, 'AMP skipped update')
            require({int(v['step']) for v in optimizer.state.values()} == {state.global_step+1}, 'Optimizer state step differs')

        def on_step_end(self, args, state, control, lr_scheduler=None, **kw):
            check_deadline(self.approval);self.memory.check()
            require(state.global_step <= 1250, 'Update ceiling exceeded')
            append(self.folder/'progress.jsonl', dict(step=state.global_step, epoch=state.epoch,
                   lr=lr_scheduler.get_last_lr(), utc=datetime.now(timezone.utc).isoformat(), **self.memory.summary()))

        def on_save(self, args, state, control, **kw):
            require(state.global_step in (250,500,750,1000,1250), 'Only five approved epoch checkpoints')
            p = self.folder/f'checkpoint-{state.global_step}'
            needed = ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                      'scaler.pt','rng_state.pth','trainer_state.json')
            require(all((p/n).is_file() for n in needed), 'Incomplete epoch checkpoint')
            write(p/'checkpoint_integrity.json', dict(binding=binding(), global_step=state.global_step,
                  files={q.name:sha(q) for q in p.iterdir() if q.is_file()}))
            append(self.folder/'CHECKPOINT_RECEIPTS.jsonl', dict(step=state.global_step,
                   checkpoint_integrity_sha256=sha(p/'checkpoint_integrity.json')))
    return Record
