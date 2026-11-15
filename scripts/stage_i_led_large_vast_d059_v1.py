"""RTX3090 fresh F: explicit65536/bounded-overflow numerical-policy amendment."""
from stage_i_led_large_full_train_v5 import *
import stage_i_led_large_full_train_v5 as historical

CONFIG = ROOT/'configs/stage_i_led_large_vast_d059_v1.json'
PREP = ROOT/'outputs/stage_i/led_large_vast_d059_v1_preparation'
RUNNER = ROOT/'scripts/run_stage_i_led_large_vast_d059_v1.py'
from led_large_dynamic_amp_v2 import POLICY
AMP=dict(AMP,init_scale=65536.0)
DEADLINE = None
EXECUTION_WINDOW = {'policy': 'researcher_managed_for_this_new_run', 'researcher_message': 'không cần để ý tôi sẽ lo', 'historical_deadlines_unchanged': True, 'execution_deadline_utc': None}
EXPERIMENT = 'LED_LARGE_VAST_D059_F_V1'

def binding():
    return dict(config_sha256=sha(CONFIG),runner_sha256=sha(RUNNER),
                preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))

def preflight():
    old = historical.preflight()
    c = read(CONFIG)
    extra = c['vast_fresh']
    require(c == dict(old, vast_fresh=extra, amp_scaler=AMP, dynamic_overflow_policy=POLICY, execution_window=EXECUTION_WINDOW, methodology_reference=c['methodology_reference'], experiment=EXPERIMENT, run_root='runs/F_large_d059_v1'), 'Scientific config differs from v5')
    require(extra == dict(experiment=EXPERIMENT, GPU='NVIDIA GeForce RTX 3090', capability=[8,6],
        initialization='fresh_official_backbone_and_fresh_named_seeded_LoRA_no_Colab_checkpoint',
        output='runs/F_large_d059_v1', deadline_utc=DEADLINE,
        gate_updates=3, gate_microbatches=88, nominal_gate_microbatches=24, gate_generation_calls=2,
        prior_T4_gate_is_historical_only=True), 'Unexpected hardware/namespace/budget amendment')
    validate_d059_reference(c)
    for rel,h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel) == h, 'Vast dependency changed: '+rel)
    binary = ROOT/c['local_model_directory']/'pytorch_model.bin'
    require(binary.stat().st_size == 1839633783 and sha(binary) == c['weights_sha256'], 'Backbone bytes differ')
    return c

def authorize(path, mode, execute):
    require(execute and path is not None, 'GPU disabled: explicit bound execution approval required')
    a = read(path)
    require(a.get('GPU_execution_approved') is True and a.get('training_authorized') is True,
            'Researcher execution approval required')
    require(a.get('experiment') == EXPERIMENT and a.get('approval_source') == 'researcher_message'
            and a.get('researcher_explicit_authorization'), 'Wrong authorization provenance')
    require(a.get('scopes') == ['feasibility','train','evaluate'] and mode in a['scopes'], 'Wrong scope')
    require(all(a.get(k) == v for k,v in binding().items()), 'Authorization hash mismatch')
    require(a.get('execution_window') == EXECUTION_WINDOW, 'Researcher clock ownership differs')
    require(a.get('dynamic_overflow_policy') == POLICY, 'Wrong dynamic overflow policy')
    require(a.get('max_training_updates') == 1250 and a.get('evaluation_outputs_max') == 48
            and a.get('feasibility_updates_max') == 3 and a.get('feasibility_microbatches_max') == 88 and a.get('nominal_feasibility_microbatches') == 24
            and a.get('technical_generation_calls_max') == 2, 'Execution budget differs')
    require(a.get('execution_deadline_utc') == DEADLINE and a.get('amp_scaler') == AMP,
            'Original deadline/AMP changed')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False
            and a.get('conditional_train_after_feasibility_pass') is True, 'Additional scope forbidden')
    check_deadline(a)
    return dict(a, _approval_sha256=sha(path))

def require_gate(out, a):
    p = Path(out)/'feasibility'
    r = read(p/'REPORT.json')
    require(not (p/'FAILURE_STOP_NO_RETRY.json').exists(), 'Failed gate cannot authorize train')
    require(r['status'] == 'PASS' and r['binding'] == binding() == read(p/'BINDING.json'), 'Foreign/failed gate')
    require(r['approval_sha256'] == a['_approval_sha256'] and r['deadline_utc'] == DEADLINE,
            'Gate authorization differs')
    require(r['optimizer_updates'] == 3 and r['nominal_microbatches'] == 24 and r['microbatches'] <= 88 and r['generation_calls'] == 2
            and r['actual_forward_backward_microbatches'] <= POLICY['gate_microbatches_max']
            and r['overflow_skips'] <= POLICY['max_overflow_skips'] and r['dynamic_overflow_policy'] == POLICY
            and r['replay_updates'] == 0 and r['headroom_fraction'] >= .15, 'Gate budget/memory differs')
    require(r['state_restore_exact'] and r['adapter_restore_exact'] and r['frozen_backbone_unchanged']
            and r['D051_strict_replay'] == 'FAIL_UNCHANGED', 'Gate restoration/integrity differs')
    initial = read(p/'INITIAL_MODEL_BINDINGS.json')
    require(initial['initialization'] == read(frozen.PREP/'INITIALIZATION_HASHES.json')['per_module_sha256'],
            'Gate fresh initialization differs')
    require(initial['scope'] == dict(modules=228, trainable_tensors=456, trainable_parameters=4915200,
                                    frozen_backbone_parameters=459801600), 'Adapter scope differs')
    proof = read(p/'checkpoint_step3/integrity.json')
    require(proof['binding'] == binding() and set(proof['files']) ==
            {'README.md','adapter_model.safetensors','adapter_config.json','state.pt'}, 'Incomplete gate checkpoint')
    for n,h in proof['files'].items():
        require(Path(n).name == n and sha(p/'checkpoint_step3'/n) == h, 'Gate checkpoint changed: '+n)
    return r

def configure_amp_scaler(accelerator, c):
    """Set the existing Accelerate scaler through its public state API before any backward.

    The same object is retained for AcceleratedOptimizer/checkpoint serialization.
    Native dynamic scaling starts at65536; finite-forward loss, missing-gradient and bounded-backoff guards remain.
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
                initialized_before_any_forward_backward=True, historical_T4_AMP_reference_sha256=GATE_SHA)

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
            require(self.trainer is not None and not self.trainer.accelerator.optimizer_step_was_skipped, 'Unexpected skipped outer update')
            self.trainer.amp_recovery.commit(self.trainer.accelerator.scaler)
            require({int(v['step']) for v in optimizer.state.values()} == {state.global_step+1}, 'Optimizer state step differs')

        def on_step_end(self, args, state, control, lr_scheduler=None, **kw):
            check_deadline(self.approval);self.memory.check()
            require(state.global_step <= 1250, 'Update ceiling exceeded')
            append(self.folder/'progress.jsonl', dict(step=state.global_step, epoch=state.epoch,
                   lr=lr_scheduler.get_last_lr(), utc=datetime.now(timezone.utc).isoformat(), **self.memory.summary()))

        def on_save(self, args, state, control, **kw):
            require(state.global_step in (250,500,750,1000,1250), 'Only five approved epoch checkpoints')
            p = self.folder/f'checkpoint-{state.global_step}'
            write(p/'AMP_DYNAMIC_STATE.json',self.trainer.amp_recovery.state(self.trainer.accelerator.scaler))
            needed = ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                      'scaler.pt','rng_state.pth','trainer_state.json')
            require(all((p/n).is_file() for n in needed), 'Incomplete epoch checkpoint')
            write(p/'checkpoint_integrity.json', dict(binding=binding(), global_step=state.global_step,
                  files={q.name:sha(q) for q in p.iterdir() if q.is_file()}))
            append(self.folder/'CHECKPOINT_RECEIPTS.jsonl', dict(step=state.global_step,
                   checkpoint_integrity_sha256=sha(p/'checkpoint_integrity.json')))
    return Record




def validate_d059_reference(c):
    reference=ROOT/'configs/stage_i_d059_expanded_lora_v2.json'
    expected=dict(config='configs/stage_i_d059_expanded_lora_v2.json',config_sha256=sha(reference),
        primary_quality_benchmark='D059_Expanded_epoch3',secondary_quality_benchmark='D059_Expanded_epoch5',
        comparison_type='practical_unmatched_base_vs_large_not_isolated_causal_architecture_effect')
    require(c['methodology_reference']==expected,'D059 reference changed')
    d=read(reference)
    for key in ('seed','lora','optimization','generation','budgets','runtime_pins','training','evaluation_epochs',
                'stage_h_manifest','final_references'):
        require(c[key]==d[key],'D059 methodological setting changed: '+key)
    require(c['model_id']=='allenai/led-large-16384' and c['expected_modules']==228
            and c['expected_trainable_parameters']==4915200,'LED-large adapter inventory differs')
    return True

def check_deadline(a, now=None):
    # Only this new namespace follows the researcher's direct clock-management decision.
    # Every historical config/authorization/deadline file remains unchanged.
    require(a.get('experiment')==EXPERIMENT and a.get('execution_window')==EXECUTION_WINDOW,
            'Wrong new-run scope or clock provenance')
    require(a.get('execution_deadline_utc') is None,'Unexpected fixed deadline in new researcher-managed run')
    require(a.get('max_training_updates')==1250 and a.get('evaluation_outputs_max')==48,
            'Time ownership must not remove update/generation ceilings')
    return None
