"""Separate gate-only authorization layer; accepted V1 science is unchanged."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = Path('outputs/stage_i/led_large_matched_completion_v2')
CONDITIONS = ('reviews_only', 'unfiltered')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')


def config(condition):
    require(condition in CONDITIONS, 'RO/U only')
    return ROOT / PREP / f'{condition}_GATE_CONFIG.json'


def binding(condition):
    return dict(gate_config_sha256=sha(config(condition)),
                runner_sha256=sha(ROOT/'scripts/run_led_large_rou_gate_only_v2.py'),
                helper_sha256=sha(ROOT/'scripts/led_large_rou_gate_only_v2.py'),
                preflight_manifest_sha256=sha(ROOT/PREP/'PREFLIGHT_MANIFEST.json'),
                accepted_V1_binding=read(config(condition))['accepted_V1_binding'])


def verify_files():
    for rel, digest in read(ROOT/PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe path')
        require(sha(ROOT/rel) == digest, 'V2 integrity mismatch: '+rel)


def deadline(a, now=None):
    times = [datetime.fromisoformat(a[k]) for k in
             ('authorized_at_utc', 'deadline_utc', 'researcher_absolute_stop_utc')]
    require(all(t.tzinfo and t.utcoffset().total_seconds() == 0 for t in times), 'Actual UTC clocks required')
    start, end, stop = times
    require(start <= (now or datetime.now(timezone.utc)) < end <= min(start+timedelta(hours=48), stop),
            'Expired/invalid gate-only clock')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True, 'Reserve must be freshly confirmed')
    require(a.get('max_successful_gate_updates') == 3 and a.get('max_generation_calls') == 2
            and a.get('max_training_updates') == 0, 'Gate-only ceilings cannot change')
    return end.isoformat()


def authorize(path, condition):
    a = read(path)
    c = read(config(condition))
    require(a.get('GPU_feasibility_approved') is True and a.get('training_authorized') is False,
            'New explicit gate-only approval required; this review is not authorization')
    require(a.get('experiment') == c['experiment'] and a.get('condition') == condition
            and a.get('binding') == binding(condition), 'Wrong condition/hash-bound approval')
    require(a.get('approval_source') == 'researcher_message' and a.get('researcher_explicit_authorization'),
            'Missing explicit researcher authorization')
    expected = dict(scopes=['feasibility'], held_out_access=False, Teacher_calls=False, automatic_train=False,
                    automatic_retry=False, fresh_only=True, max_actual_microbatches=88,
                    evaluation_outputs_max=0, early_sanity_outputs_max=0)
    require(all(a.get(k) == v for k, v in expected.items()), 'Forbidden scope or numerical budget')
    deadline(a)
    # Aliases for unchanged native gate receipt fields, not a grant of training.
    return dict(a, _approval_sha256=sha(path), execution_deadline_utc=a['deadline_utc'])


def cpu_preflight(condition, parent, *, run_tests=True):
    import subprocess
    import os
    verify_files()
    import led_large_matched_family_v1 as accepted
    require(accepted.ROOT == ROOT and accepted.binding(condition) == read(config(condition))['accepted_V1_binding'],
            'Accepted V1 condition binding changed')
    require(sha(ROOT/'outputs/cloud_handoff/LED_LARGE_MATCHED_FAMILY_V1_OFFLINE_REVIEW.zip') ==
            read(ROOT/PREP/'ACCEPTED_V1_CHAIN.json')['archive_sha256'], 'Accepted V1 archive changed')
    chain = read(ROOT/PREP/'ACCEPTED_V1_CHAIN.json')
    portable = ROOT/'outputs/stage_i/led_large_matched_family_v1/PORTABLE_MANIFEST.json'
    require(sha(portable) == chain['portable_manifest_sha256'], 'Accepted V1 portable manifest changed')
    for rel, digest in read(portable)['files'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(ROOT/rel) == digest,
                'Accepted V1 portable member changed: '+rel)
    c, parent_report = accepted.cpu_preflight(condition, Path(parent), full_parent=True)
    receipt = dict(status='CPU_PREFLIGHT_PASS_NOT_GPU_PASS', condition=condition, binding=binding(condition),
                   accepted_parent_report=parent_report, native_gate_science_unchanged=True,
                   CPU_tests=[], GPU_operations=0, model_generations=0, training_updates=0)
    if run_tests:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', LED_MATCHED_PARENT_ROOT=str(Path(parent).resolve()))
        suites = [(ROOT/'tests', 'test_led_large_matched_family_v1.py', 26),
                  (Path(parent)/'tests', 'test_led_large_decoder_policy_v2.py', 35),
                  (ROOT/'tests', 'test_led_large_matched_completion_v2.py', None)]
        import re
        for directory, pattern, expected_count in suites:
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(directory.resolve()), '-p', pattern, '-v']
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                                    encoding='utf-8', errors='replace', timeout=1800)
            log = result.stdout+'\n'+result.stderr
            match = re.search(r'Ran (\d+) tests?', log)
            count = int(match.group(1)) if match else 0
            receipt['CPU_tests'].append(dict(command=command, returncode=result.returncode, tests=count, log=log))
            require(result.returncode == 0 and count > 0 and (expected_count is None or count == expected_count),
                    'CPU suite failed: '+pattern+'\n'+log)
    # Metadata for F checkpoint integrity is itself bound by the new preflight;
    # verify all actual components when present at the separately declared path.
    manifest = read(ROOT/PREP/'HISTORICAL_CHECKPOINT_FILES.json')
    for rel, digest in manifest['files'].items():
        require(sha(ROOT/rel) == digest, 'Original F checkpoint component changed: '+rel)
    receipt['original_F_checkpoint_components_verified'] = len(manifest['files'])
    return c, receipt


def install_gate_context(condition, parent):
    import led_large_matched_family_v1 as accepted
    b, c = accepted.install_condition_context(condition, parent)
    import stage_i_led_large_decoder_policy_v2 as h
    b.binding = h.binding = lambda: binding(condition)
    b.check_deadline = h.check_deadline = deadline
    return b, c
