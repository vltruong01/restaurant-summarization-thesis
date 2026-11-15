"""Additive offline gate-only delivery builder. Refuses to replace an existing seal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

R = Path(__file__).resolve().parents[1]
P = R/'outputs/stage_i/led_large_boundary_global_feasibility_v1'
OLD = R/'outputs/stage_i/led_large_boundary_global_v1'
CONFIG = R/'configs/stage_i_led_large_boundary_global_feasibility_v1.json'
ZIP = R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1_OFFLINE_REVIEW.zip'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2); f.write('\n')


def initialize():
    archive = R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_V1_OFFLINE_REVIEW.zip'
    assert sha(archive) == 'b450b50bd4e668316580f90c8600d9d0d4b5d9d069e7a53baaf61a0e1d5f85de'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        files = json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(files) == 58
        assert all(hashlib.sha256(z.read(n)).hexdigest() == digest for n, digest in files.items())
        for n, digest in files.items():
            if n != 'READ_THIS_FIRST.md': assert sha(R/n) == digest, n
    frozen = {n: digest for n, digest in files.items() if n != 'READ_THIS_FIRST.md'}
    frozen.update({p.relative_to(R).as_posix(): sha(p) for p in OLD.iterdir() if p.is_file()})
    write(P/'FROZEN_V1_RECEIPT.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        archive=archive.relative_to(R).as_posix(), archive_sha256=sha(archive), verified_archive_entries=58,
        frozen_files=frozen, preservation='All original V1 payloads and supplemental receipts immutable; no replacement seal'))
    old = read(R/'configs/stage_i_led_large_boundary_global_v1.json')
    write(CONFIG, dict(experiment='LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1', GPU_execution_approved=False,
        preparation_only=True, run_root='runs/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1',
        frozen_V1_config_sha256=sha(R/'configs/stage_i_led_large_boundary_global_v1.json'),
        frozen_V1_preflight_sha256=sha(OLD/'PREFLIGHT_MANIFEST.json'),
        arms=['bos_only', 'boundary'], arm_order='bos_only_then_boundary_no_restart',
        intended_difference='encoder_global_attention_mask_only', maximum_global_tokens=36,
        train_stress_business_ids=read(OLD/'STRESS_CASES.json')['train_business_ids'],
        gate_successful_updates_per_arm=3, gate_nominal_microbatches_per_arm=24,
        gate_actual_microbatches_max_per_arm=88, technical_beam4_calls_per_arm=2,
        full_cache_stress=dict(source=[1,5120], beams=4, decoder_cache_tokens=768,
            encoder_forwards=1, decoder_forwards=2, generate_calls=0, optimizer_updates=0),
        amp_scaler=old['shared']['amp_scaler'], dynamic_overflow_policy=old['shared']['dynamic_overflow_policy'],
        minimum_memory_headroom=.15, maximum_runtime_seconds=7200, maximum_authorization_hours=48,
        full_training_authorized=False, evaluation_authorized=False, automatic_retry=False,
        held_out_access=False, Teacher_calls=False, stop_after_both_gates=True,
        historical_science_configuration='V1 runtime config bytes reused; their future training fields confer no permission. Explicit out argument selects only this new gate namespace.'))
    print('Initialized additive frozen-parent receipt and gate configuration:', P)


def seal():
    import run_led_large_boundary_global_feasibility_v1 as gate
    import os
    import subprocess
    import sys
    assert read(P/'CPU_TEST_RESULTS.json')['status'] == 'PASS'
    assert read(P/'INHERITED_CPU_TEST_RESULTS.json')['status'] == 'PASS'
    technical = [CONFIG, R/'scripts/run_led_large_boundary_global_feasibility_v1.py', Path(__file__),
        R/'tests/test_led_large_boundary_global_feasibility_v1.py', P/'FROZEN_V1_RECEIPT.json',
        P/'FEASIBILITY_SPEC.md', P/'SCHEDULE_COST_ASSESSMENT.md', P/'EXECUTION_COMMANDS.md',
        P/'CPU_TEST_RESULTS.json', P/'INHERITED_CPU_TEST_RESULTS.json']
    write(P/'PREFLIGHT_MANIFEST.json', dict(files={p.relative_to(R).as_posix(): sha(p) for p in technical},
        scope='Gate-only overlay plus independently checked immutable V1 and full V2 parent chain'))
    bindings = {arm: gate.binding(arm) for arm in ('bos_only','boundary')}
    write(P/'EXECUTION_BINDINGS.json', bindings)
    template = dict(experiment=gate.EXPERIMENT, GPU_execution_approved=False,
        approval_source='researcher_message', researcher_explicit_authorization=None,
        bindings=bindings, execution_authorized_at_utc=None, execution_deadline_utc=None,
        researcher_absolute_stop_utc=None, defense_reserve_at_least_14_days_confirmed=None,
        **gate.expected_contract())
    write(P/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json', template)
    gate.verify_overlay()
    parent = R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
    command = [sys.executable, str(R/'scripts/run_led_large_boundary_global_feasibility_v1.py'),
        'preflight', '--parent-root', str(parent)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1'))
    with (P/'PINNED_GATE_ONLY_PREFLIGHT.log').open('x', encoding='utf-8') as f:
        f.write(result.stdout+'\n'+result.stderr)
    write(P/'PINNED_GATE_ONLY_PREFLIGHT.json', dict(status='PASS' if result.returncode==0 else 'FAIL',
        command=command, returncode=result.returncode, CUDA_VISIBLE_DEVICES='', model_loads=0,
        new_optimizer_updates=0, new_generations=0))
    assert result.returncode == 0, result.stdout+result.stderr
    frozen = read(P/'FROZEN_V1_RECEIPT.json')
    assert all(sha(R/n) == d for n,d in frozen['frozen_files'].items())
    assert sha(R/frozen['archive']) == frozen['archive_sha256']
    protected = {
        'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750/adapter_model.safetensors':
            '507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60',
        'outputs/cloud_handoff/LED_LARGE_CAD_V1_VAST_CHATGPT_REVIEW.zip':
            'c5e4f3a4c4f0fd420d74135e9f30ffda1b5f5c562d8db73b1678d2cfc9f942e8'}
    assert all(sha(R/n)==d for n,d in protected.items())
    write(P/'HISTORICAL_PRESERVATION.json', dict(files=protected, unchanged=True,
        contrastive_and_RO_U_F_Shuffled='No writes; neither code path nor target namespace touched'))
    write(P/'READINESS.json', dict(status='GO_FOR_INDEPENDENT_GATE_ONLY_AUTHORIZATION_REVIEW',
        GPU_execution='BLOCKED_PENDING_FRESH_EXPLICIT_RESEARCHER_APPROVAL',
        full_training='FORBIDDEN_BY_THIS_CONTRACT_EVEN_AFTER_GATE_PASS',
        actual_boundary_memory='UNMEASURED', quality_improvement='UNTESTED',
        current_preparation_GPU_operations=0, current_preparation_optimizer_updates=0,
        current_preparation_generation_calls=0, held_out_access=False,
        original_V1_archive_unchanged=True, original_CE3_and_CAD_unmodified=True,
        thesis_priority='Main RO/U/F/Shuffled takes precedence; Boundary optional architecture ablation'))
    bundle = set(technical) | set(P.glob('*')) | {R/n for n in frozen['frozen_files']}
    manifest = {p.relative_to(R).as_posix(): sha(p) for p in sorted(bundle) if p.is_file()}
    assert not ZIP.exists()
    with zipfile.ZipFile(ZIP, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in manifest: z.write(R/n, n)
        first = (P/'FEASIBILITY_SPEC.md').read_bytes()
        z.writestr('READ_THIS_FIRST.md', first)
        manifest['READ_THIS_FIRST.md'] = hashlib.sha256(first).hexdigest()
        z.writestr('SHA256_MANIFEST.json', json.dumps(dict(files=manifest), indent=2))
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest() == d for n,d in manifest.items())
    with ZIP.with_suffix('.zip.sha256').open('x', encoding='ascii') as f: f.write(sha(ZIP)+'  '+ZIP.name+'\n')
    write(P/'DELIVERY_RECEIPT.json', dict(archive=str(ZIP), sha256=sha(ZIP), bytes=ZIP.stat().st_size,
        manifest_entries=len(manifest), GPU_execution_approved=False,
        binding_config=bindings['bos_only']['config_sha256'], binding_runner=bindings['bos_only']['runner_sha256'],
        binding_preflight=bindings['bos_only']['preflight_manifest_sha256']))
    print(json.dumps(read(P/'DELIVERY_RECEIPT.json'), indent=2))


if __name__ == '__main__':
    import sys
    {'initialize':initialize, 'seal':seal}[sys.argv[1]]()
