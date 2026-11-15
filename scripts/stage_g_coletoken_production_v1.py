"""Versioned Coletoken continuation ledger. Frozen D042 science and prior artifacts stay unchanged."""
from collections import defaultdict
from pathlib import Path
from stage_g_production_d042 import (
    ROOT, FOLDER as ORIGINAL, OUT as ORIGINAL_EXEC, PILOT_OUT, PILOT_CONFIG,
    CONFIG, FIXED_KEYS, read, immutable, atomic, canonical, file_sha, now,
    check_inputs as original_inputs, verify_checkpoint, build_target,
)

RUN = ORIGINAL / 'coletoken_production_concurrency5_v1'
EXEC = RUN / 'execution_v1'
OVERLAY = ROOT / 'configs/stage_g_coletoken_production_concurrency5_v1.json'
POINTER = ROOT / 'outputs/stage_g/active_production_execution_pointer.json'
RECOVERY = ORIGINAL / 'coletoken_concurrency5_recovery_v1/execution_v1'
INDEX = None


def reuse_index():
    global INDEX
    if INDEX is None: INDEX = read(RUN / 'reuse_index.json')
    return INDEX


def location(job, imports):
    entry = reuse_index().get(job['job_id'])
    return ROOT / entry['directory'] if entry else EXEC


def checked_inputs():
    cfg, jobs, businesses, imports = original_inputs()
    overlay = read(OVERLAY)
    assert all(cfg[k] == overlay[k] for k in FIXED_KEYS)
    assert overlay['execution']['provider'] == 'coletoken'
    assert overlay['execution']['base_url'] == 'https://www.coletoken.de/v1'
    assert overlay['execution']['maximum_concurrency'] == 5
    assert overlay['execution']['maximum_active_businesses'] == 2
    return cfg, jobs, businesses, imports


def prepare():
    cfg, jobs, businesses, imports = checked_inputs()
    assert not (RUN / 'manifest.json').exists()
    trial = read(RECOVERY.parent / 'report.json')
    assert trial['passed'] == 5 and trial['failed'] == 0
    assert trial['returned_models'] == {'gpt-6.1-sol': 5}
    index = {}; hashes = {}; history = []
    for j in jobs:
        jid = j['job_id']
        original = PILOT_OUT if jid in imports else ORIGINAL_EXEC
        recovery_path = RECOVERY / 'checkpoints' / (jid + '.json')
        directory = RECOVERY if recovery_path.exists() else original
        cp_path = directory / 'checkpoints' / (jid + '.json')
        if not cp_path.exists(): continue
        cp = read(cp_path)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            raise ValueError('Unrecovered original Coletoken failure: ' + jid)
        config_path = PILOT_CONFIG if jid in imports else CONFIG
        verify_checkpoint(j, cfg, directory, config_path)
        index[jid] = {'directory': directory.relative_to(ROOT).as_posix(),
                      'config_path': config_path.relative_to(ROOT).as_posix(),
                      'source': 'unchanged_D041' if jid in imports else 'Coletoken_five_request_recovery' if directory == RECOVERY else 'original_D042_Coletoken'}
        paths = [cp_path]
        if cp.get('api_called'):
            paths += [directory / 'raw_responses' / cp_path.name,
                      directory / 'request_intents' / cp_path.name,
                      directory / 'raw_requests' / (jid + '.json.gz')]
        for p in paths: hashes[p.relative_to(ROOT).as_posix()] = file_sha(p)
    assert len(index) == 402 and len(imports) == 300
    for name in ('execution_control_concurrency10_trial_v1', 'execution_control_concurrency10_new_key_trial_v2',
                 'mirai_concurrency10_trial_v1', 'mirai_concurrency9_stream_trial_v1',
                 'mirai_concurrency6_stream_trial_v1', 'mirai_grouped6_recovery_v1',
                 'mirai_grouped6_recovery_v2', 'coletoken_concurrency5_recovery_v1'):
        for p in (ORIGINAL / name).rglob('*'):
            if p.is_file(): hashes[p.relative_to(ROOT).as_posix()] = file_sha(p)
        history.append((ORIGINAL / name).relative_to(ROOT).as_posix())
    immutable(RUN / 'reuse_index.json', index)
    bybiz = defaultdict(list)
    for j in jobs: bybiz[j['business_id']].append(j)
    completed = [b['business_id'] for b in businesses if all(j['job_id'] in index for j in bybiz[b['business_id']])]
    assert len(completed) == 134
    immutable(RUN / 'manifest.json', {
        'approved_by': 'user', 'authorization': 'Continue teacher-target production after successful Coletoken five-request trial',
        'provider': 'coletoken', 'base_url': 'https://www.coletoken.de/v1', 'requested_model': cfg['requested_model'],
        'maximum_concurrency': 5, 'maximum_active_businesses': 2, 'automatic_retries': 0,
        'businesses': 2200, 'by_split': {'train': 2000, 'validation': 200}, 'held_out_test': 0,
        'initial_complete_businesses': 134, 'initial_condition_aligned_targets': 402,
        'imported_D041_businesses': 100, 'initial_reused_component_checkpoints': len(index),
        'maximum_new_API_calls': sum(j['job_id'] not in index and not j.get('skip_api_empty_captions') for j in jobs),
        'config_sha256': file_sha(CONFIG), 'overlay_sha256': file_sha(OVERLAY),
        'reuse_index_sha256': file_sha(RUN / 'reuse_index.json'),
        'code_sha256': {rel: file_sha(ROOT / rel) for rel in (
            'scripts/stage_g_coletoken_production_v1.py', 'scripts/run_stage_g_coletoken_production5_v1.py',
            'scripts/audit_stage_g_coletoken_production5_v1.py', 'scripts/stage_g_teacher_sse_transport_v1.py')},
        'previous_artifact_sha256': hashes, 'preserved_history': history,
        'scientific_settings_changed': False, 'Mirai_outputs_imported': False,
        'original_failed_checkpoints_overwritten': False, 'Codex_configuration_or_authentication_access': False,
        'teacher_content_origin': 'teacher_API_only_frozen_serialization', 'prepared_utc': now()})
    print({'preflight': 'PASS', 'initial_businesses': 134, 'initial_targets': 402, 'maximum_concurrency': 5,
           'maximum_active_businesses': 2, 'API_calls': 0}, flush=True)


def verify_manifest():
    m = read(RUN / 'manifest.json')
    assert file_sha(CONFIG) == m['config_sha256']
    assert file_sha(OVERLAY) == m['overlay_sha256']
    assert file_sha(RUN / 'reuse_index.json') == m['reuse_index_sha256']
    for rel, sha in {**m['code_sha256'], **m['previous_artifact_sha256']}.items():
        assert file_sha(ROOT / rel) == sha, rel
    return m


def save_completed_business(business, jobs_by_business, imports):
    content = {}
    for j in jobs_by_business[business['business_id']]:
        p = location(j, imports) / 'checkpoints' / (j['job_id'] + '.json')
        if not p.exists(): return None
        cp = read(p)
        if cp['status'] != 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW': return None
        content[j['condition']] = cp.get('resolved_output', cp['output'])
    target = build_target(business, content)
    for directory in (PILOT_OUT / 'provisional_targets', ORIGINAL_EXEC / 'targets'):
        existing = directory / (business['business_id'] + '.json')
        if existing.exists():
            assert read(existing) == target
            return existing
    p = EXEC / 'targets' / (business['business_id'] + '.json')
    immutable(p, target)
    return p


def update_docs(state):
    marker = '**D-042 live production status:**'
    note = (f"{marker} {state['businesses_completed']}/2200 complete businesses, {state['targets_completed']} targets; "
            f"Coletoken gpt-6.1-sol, up to5 requests /2 business slots; status={state['status']}; "
            f"new continuation API attempts={state['new_API_attempts']}; stop reason={state.get('stop_reason')}. "
            "100 D041 and134 initially completed businesses reused without regeneration; Mirai outputs remain separate. "
            "Authoritative pointer: outputs/stage_g/active_production_execution_pointer.json. "
            "Active progress: outputs/stage_g/teacher_production_d042_v1/coletoken_production_concurrency5_v1/execution_v1/execution_progress.json. "
            "Same frozen scientific inputs/prompts/schema; API-only content; no held-out targets or LED training. "
            "Automated integrity is separate from independent factual verification.\n")
    p = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
    lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
    lines = [l for l in lines if not l.startswith(marker)]
    lines.insert(1, '\n' + note + '\n')
    p.write_text(''.join(lines), encoding='utf-8')
