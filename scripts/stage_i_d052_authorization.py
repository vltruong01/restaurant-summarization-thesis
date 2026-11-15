"""D-052 training-only conditional authorization; no CUDA or model execution.

The D-049 execution gate is intentionally untouched. This separate gate accepts
the researcher's narrowly scoped exception while retaining the original strict
replay failures as failures. Every call first revalidates D-049 preparation.
"""
from pathlib import Path

from stage_i_paired_pilot import ROOT, ARMS, checked_preparation, read, require, sha

CONFIG = ROOT / 'configs/stage_i_paired_lora_pilot_v1.json'
REQUEST = Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/7baa816d-643b-46d3-97c3-da03de0616a6/Pasted text.txt')
DECISION = 'outputs/stage_i/d052_paired_pilot_v1/D052_CONDITIONAL_REPRODUCIBILITY_DECISION.md'
EVIDENCE = {
    'd050_a': 'outputs/stage_i/paired_lora_pilot_runs_v1/A_decoder_only/feasibility/report.json',
    'd050_b': 'outputs/stage_i/d050_gpu_feasibility_v1/B_encoder_decoder/observation_completion.json',
    'd051_readiness': 'outputs/stage_i/d051_replay_investigation_v1/READINESS_ASSESSMENT.json',
    'd051_strict': 'outputs/stage_i/d051_replay_investigation_v1/execution/strict_replay_result.json',
    'd051_comparison': 'outputs/stage_i/d051_replay_investigation_v1/execution/adapter_replay_comparison.json',
    'd051_binding': 'outputs/stage_i/d051_replay_investigation_v1/execution/binding.json',
}
SCOPE = {
    'condition': 'filtered', 'training_businesses': 512,
    'validation_businesses': 200, 'epochs': 2,
    'optimizer_updates_per_arm': 128, 'generation_calls': 0,
}
ACCEPTANCE = {
    'strict_replay_status': 'FAIL_UNCHANGED',
    'original_tolerance': {'rtol': 1e-5, 'atol': 1e-6},
    'exploratory_pilot_only': True, 'long_run_harmlessness_not_proven': True,
    'no_retrospective_pass': True,
}


def _exact_file(record, expected_path, message):
    require(isinstance(record, dict), message + ': missing record')
    actual = Path(record.get('path', ''))
    expected = Path(expected_path)
    require(actual.resolve() == expected.resolve(), message + ': different path')
    require(expected.is_file() and record.get('sha256') == sha(expected), message + ': hash changed')


def _original_binding(config_path, prep, arm):
    return {'config_sha256': sha(config_path),
            'preparation_manifest_sha256': sha(prep/'PREPARATION_MANIFEST.json'),
            'arm': arm,
            'runner_sha256': sha(ROOT/'scripts/run_stage_i_paired_lora_pilot_v1.py'),
            'helper_sha256': sha(ROOT/'scripts/stage_i_paired_pilot.py')}


def conditional_training_gate(approval_path, arm, execute, wrapper_path):
    """Return (unchanged_config, preparation_path, approval) for fresh training.

    This function cannot authorize generation, resume, a different dataset or a
    synthetic replacement feasibility PASS. No CUDA availability check occurs.
    """
    require(execute is True, 'D052 execution requires explicit --execute')
    require(arm in ARMS, 'D052 unknown arm')
    # Hash/version/scope checks from the accepted offline preparation are kept.
    config, prep, _ = checked_preparation(CONFIG)
    require(approval_path is not None and Path(approval_path).is_file(), 'D052 approval missing')
    approval = read(approval_path)
    require(approval.get('decision') == 'D-052' and approval.get('approved') is True,
            'D052 explicit research approval missing')
    require(approval.get('scopes') == ['train'] and approval.get('arms') == list(ARMS),
            'D052 authorizes only the two matched training arms')
    require(approval.get('scientific_scope') == SCOPE, 'D052 scientific scope changed')
    require(approval.get('conditional_reproducibility_acceptance') == ACCEPTANCE,
            'D052 does not accept a retrospective PASS or revised tolerance')
    for key, path in (
        ('config_sha256', CONFIG),
        ('preparation_manifest_sha256', prep/'PREPARATION_MANIFEST.json'),
        ('protocol_sha256', prep/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),
        ('wrapper_sha256', Path(wrapper_path)),
        ('authorization_module_sha256', Path(__file__)),
    ):
        require(path.is_file() and approval.get(key) == sha(path), 'D052 binding changed: ' + key)
    _exact_file(approval.get('user_request'), REQUEST, 'D052 user request')
    _exact_file(approval.get('decision_record'), ROOT/DECISION, 'D052 decision')
    require(config['condition'] == 'filtered' and config['training'] == {
        'businesses': 512, 'epochs': 2, 'steps': 128, 'steps_per_epoch': 64, 'warmup_steps': 7},
        'D052 accepted training scope changed')
    require(config['optimization']['physical_batch_size'] == 1 and
            config['optimization']['gradient_accumulation_steps'] == 8,
            'D052 batch settings changed')
    manifest = read(prep/'pilot_manifest.json')
    require(len(manifest['training_business_ids']) == 512 and
            len(manifest['validation_loss_business_ids']) == 200, 'D052 dataset count changed')
    evidence = approval.get('evidence', {})
    require(set(evidence) == set(EVIDENCE.values()), 'D052 evidence set changed')
    for rel in EVIDENCE.values():
        require((ROOT/rel).is_file() and evidence[rel] == sha(ROOT/rel), 'D052 evidence hash changed: ' + rel)
    a = read(ROOT/EVIDENCE['d050_a'])
    require(a.get('status') == 'TECHNICAL_FEASIBILITY_PASS' and a.get('arm') == ARMS[0] and
            a.get('replay_allclose') is True and a.get('replay_tolerance') == ACCEPTANCE['original_tolerance'],
            'D052 A feasibility evidence incompatible')
    require(a.get('binding') == _original_binding(CONFIG, prep, ARMS[0]), 'D052 A feasibility binding changed')
    b = read(ROOT/EVIDENCE['d050_b'])
    require(b.get('status') == 'AFFECTED_EXECUTION_STOPPED' and
            b.get('error', {}).get('message') == 'Checkpoint/RNG replay not numerically reproducible' and
            b.get('config_sha256') == sha(CONFIG), 'D052 B strict failure was changed')
    r = read(ROOT/EVIDENCE['d051_readiness'])
    require(r.get('assessment') == 'CONDITIONAL' and r.get('strict_replay_gate') == 'FAIL_UNCHANGED' and
            r.get('criterion_amendment_made') is False and
            r.get('strict_original_tolerance') == ACCEPTANCE['original_tolerance'] and
            r.get('long_run_harmlessness_proven') is False, 'D052 D051 assessment changed')
    strict = read(ROOT/EVIDENCE['d051_strict'])
    require(strict.get('outcome') == 'STRICT_REPLAY_TOLERANCE_FAILURE' and
            strict.get('rtol') == 1e-5 and strict.get('atol') == 1e-6 and
            strict.get('tolerances_changed') is False, 'D052 strict replay criterion changed')
    numeric = read(ROOT/EVIDENCE['d051_comparison'])
    n = numeric['overall']
    require(n['numel'] == 589824 and n['failing_elements'] == 18 and n['allclose'] is False and
            n['all_finite'] is True and n['max_abs'] == 3.723427653312683e-6 and
            numeric['rtol'] == 1e-5 and numeric['atol'] == 1e-6,
            'D052 bounded numerical evidence changed')
    original = _original_binding(CONFIG, prep, ARMS[1])
    actual = read(ROOT/EVIDENCE['d051_binding'])
    require(all(actual.get(k) == v for k, v in original.items()), 'D052 D051 original binding changed')
    runroot = ROOT/config['run_root']
    require(not (runroot/arm/'train').exists(), 'D052 fresh run only; training directory already exists')
    require(all(not (runroot/a/'generation').exists() for a in ARMS), 'D052 generation already present; stop for review')
    if arm == ARMS[0]:
        require(not (runroot/ARMS[1]/'train').exists(), 'D052 A must precede B')
    else:
        aout = runroot/ARMS[0]/'train'
        require((aout/'completion.json').is_file(), 'D052 A must complete before B')
        completed = read(aout/'completion.json')
        require(completed.get('status') == 'PILOT_TRAINING_COMPLETE_NO_GENERATION' and
                completed.get('steps') == 128 and completed.get('epochs') == 2 and
                completed.get('arm') == ARMS[0] and completed.get('generation_calls') == 0 and
                completed.get('binding') == _original_binding(CONFIG, prep, ARMS[0]), 'D052 A completion incompatible')
        weights = aout/'best_adapter/adapter_model.safetensors'
        require(weights.is_file() and sha(weights) == completed.get('best_adapter_sha256'), 'D052 A checkpoint changed')
        receipt = read(aout/'d052_authorization_binding.json')
        require(receipt == {'decision': 'D-052', 'approval_sha256': sha(approval_path),
                            'wrapper_sha256': sha(wrapper_path)}, 'D052 A lacks matching conditional authorization')
        observed = read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/A_decoder_only/observation_completion.json')
        require(observed.get('status') == 'TRAINING_AND_AUDIT_PASS' and observed.get('error') is None and
                observed.get('actual_optimizer_steps') == 128, 'D052 A observation audit must pass before B')
    return config, prep, approval
