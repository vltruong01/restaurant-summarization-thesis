"""Gate-only overlay. No training/evaluation entry point; fresh approval required.

Reuses the frozen V1 mask routing and native V2 CE/AMP/state gate unchanged.
The parent process enforces a two-hour wall limit on its isolated worker group.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import csv
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import run_led_large_boundary_global_v1 as old
from led_large_boundary_global_v1 import ARMS, require

R = Path(__file__).resolve().parents[1]
P = R/'outputs/stage_i/led_large_boundary_global_feasibility_v1'
CONFIG = R/'configs/stage_i_led_large_boundary_global_feasibility_v1.json'
read, sha, write = old.read, old.sha, old.write
EXPERIMENT = 'LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1'


def binding(arm):
    require(arm in ARMS, 'Unknown attention arm')
    return dict(config_sha256=sha(CONFIG), runner_sha256=sha(Path(__file__)),
        preflight_manifest_sha256=sha(P/'PREFLIGHT_MANIFEST.json'), attention_arm=arm,
        runtime_config_sha256=sha(old.P/f'RUNTIME_CONFIG_{arm}.json'),
        anchor_manifest_sha256=sha(old.P/'ANCHOR_ROWS.jsonl'),
        frozen_V1_preflight_sha256=sha(old.P/'PREFLIGHT_MANIFEST.json'))


def check_time(a, now=None):
    clocks = [datetime.fromisoformat(a[k]) for k in
        ('execution_authorized_at_utc', 'execution_deadline_utc', 'researcher_absolute_stop_utc')]
    start, end, absolute = clocks
    require(all(x.tzinfo is not None and x.utcoffset().total_seconds() == 0 for x in clocks),
        'Explicit UTC authorization/deadline required')
    require(start <= (now or datetime.now(timezone.utc)) < end <= min(start+timedelta(hours=48), absolute),
        'Invalid/expired maximum 48-hour authorization')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True,
        'Researcher reserve confirmation required; never infer from historical dates')
    if '_runtime_stop_monotonic' in a:
        require(time.monotonic() < a['_runtime_stop_monotonic'], 'Two-hour feasibility wall cap reached')


def expected_contract():
    c = read(CONFIG)
    return dict(approved_arms=list(ARMS), scopes=['feasibility'],
        training_authorized=False, evaluation_authorized=False,
        conditional_train_after_both_complete_gates_pass=False,
        max_training_updates=0, evaluation_outputs_max=0, early_sanity_outputs_max=0,
        early_sanity_generation_approved=False, feasibility_updates_max=3,
        feasibility_updates_suite_max=6, feasibility_microbatches_max=88,
        technical_generation_calls_max=2, all_generation_calls_max=4,
        full_cache_stress_approved=True, fresh_only=True, automatic_retry=False,
        automatic_mask_search=False, held_out_access=False, Teacher_calls=False,
        RO_U_training_authorized=False, minimum_memory_headroom=0.15,
        maximum_runtime_seconds=7200, maximum_authorization_hours=48,
        amp_scaler=c['amp_scaler'], dynamic_overflow_policy=c['dynamic_overflow_policy'],
        stop_after_feasibility=True)


def authorize(path, execute):
    require(execute and path is not None, 'No GPU permission: explicit gate-only researcher approval required')
    a = read(path)
    require(a.get('experiment') == EXPERIMENT and a.get('GPU_execution_approved') is True,
        'Wrong experiment or unapproved template')
    require(a.get('approval_source') == 'researcher_message' and
        isinstance(a.get('researcher_explicit_authorization'), str) and
        a['researcher_explicit_authorization'].strip(), 'Fresh explicit researcher decision required')
    expected = expected_contract()
    require(all(type(a.get(k)) is type(v) and a[k] == v for k, v in expected.items()),
        'Gate-only scope/budget/numerical policy differs')
    require(a.get('bindings') == {arm: binding(arm) for arm in ARMS}, 'Exact fresh gate bindings differ')
    allowed = set(expected) | {'experiment', 'GPU_execution_approved', 'approval_source',
        'researcher_explicit_authorization', 'bindings', 'execution_authorized_at_utc',
        'execution_deadline_utc', 'researcher_absolute_stop_utc',
        'defense_reserve_at_least_14_days_confirmed'}
    require(set(a) == allowed, 'Unknown/missing authorization fields; no implicit additional permission')
    check_time(a)
    return dict(a, _approval_sha256=sha(path))


def verify_overlay():
    c = read(CONFIG)
    require(c['experiment'] == EXPERIMENT and c['GPU_execution_approved'] is False,
        'Preparation config must never self-authorize')
    for rel, digest in read(P/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(R/rel) == digest, 'Feasibility preparation changed: '+rel)
    frozen = read(P/'FROZEN_V1_RECEIPT.json')
    for rel, digest in frozen['frozen_files'].items():
        require(sha(R/rel) == digest, 'Frozen Boundary V1 changed: '+rel)
    require(sha(old.CONFIG) == c['frozen_V1_config_sha256'] and
        sha(old.P/'PREFLIGHT_MANIFEST.json') == c['frozen_V1_preflight_sha256'], 'Wrong V1 parent')
    return c


def preflight(parent):
    c = verify_overlay()
    spec, configs, index, h, b = old.preflight(parent)
    require(spec['shared']['amp_scaler'] == c['amp_scaler'] and
        spec['shared']['dynamic_overflow_policy'] == c['dynamic_overflow_policy'], 'AMP inheritance drift')
    require(read(old.P/'STRESS_CASES.json')['train_business_ids'] == c['train_stress_business_ids'],
        'Locked TRAIN-only stress cases differ')
    return c, configs, index, h, b


def gpu_identity():
    """Called only in the approved worker, never in offline preflight/tests."""
    p = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = list(csv.reader(io.StringIO(p.stdout.strip()), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == 5, 'Dedicated single-GPU host required')
    uuid, name, bus, memory, driver = [s.strip() for s in rows[0]]
    require(uuid.startswith('GPU-') and name == 'NVIDIA GeForce RTX 3090', 'Dedicated RTX 3090 required')
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    pids = [int(s.strip()) for s in q.stdout.splitlines() if s.strip()]
    require(all(pid == os.getpid() for pid in pids), 'Foreign GPU compute workload; do not run this gate')
    return dict(uuid=uuid, name=name, pci_bus_id=bus, memory_total_MiB=int(memory), driver=driver)


@contextmanager
def route(h, b, index, arm, observer):
    # Native runtime config/science and mask implementation remain byte-identical.
    with patch.object(old, 'binding', binding), patch.object(old, 'check_time', check_time):
        with old.route_parent(h, b, index, arm, observer):
            yield


def estimate_schedule(reports, timings):
    estimates = {}
    for arm in ARMS:
        stamps = [x['elapsed_seconds'] for x in timings[arm] if x['kind'] == 'successful_update']
        intervals = [b-a for a, b in zip(stamps, stamps[1:])]
        # First update includes setup; only two later, synchronized update intervals exist.
        sec = sum(intervals)/len(intervals) if len(intervals) == 2 else None
        estimates[arm] = dict(native_gate_seconds=reports[arm]['seconds'],
            measured_later_update_intervals_seconds=intervals,
            approximate_1250_update_compute_hours=None if sec is None else sec*1250/3600,
            caveat='Two warm-up gate intervals only; overflow replay included. Excludes epoch validation, checkpoint I/O, sanity, final generation and setup. Not training approval.')
    return estimates


def run_pair(suite, a, configs, index, h, b, identity=gpu_identity):
    """Single fixed serial pair. Any exception stops the suite; no continuation API."""
    reports, timings = {}, {}
    initial = identity()
    write(suite/'DEDICATED_GPU.json', initial)
    for arm in ARMS:
        check_time(a)
        require(identity() == initial, 'Dedicated GPU identity changed between arms')
        out = suite/arm
        require(not out.exists(), 'Fresh arm namespace required; no retry')
        start = time.monotonic()
        timings[arm] = []
        mask_log = suite/f'{arm}_MASKS.jsonl'
        timing_log = suite/f'{arm}_TIMINGS.jsonl'
        def append(path, row):
            with path.open('a', encoding='utf-8') as f:
                f.write(old.json.dumps(row, separators=(',', ':'))+'\n'); f.flush()
        def observe(row):
            append(mask_log, row)
        native_write = b.write
        def timed_write(path, value):
            result = native_write(path, value)
            if Path(path).name in ('update_1.json', 'update_2.json', 'update_3.json'):
                row = dict(kind='successful_update', step=value['step'], elapsed_seconds=time.monotonic()-start)
                timings[arm].append(row); append(timing_log, row)
            return result
        with route(h, b, index, arm, observe), patch.object(b, 'write', timed_write):
            b.feasible(configs[arm], a, out)
            old.cache_stress(configs[arm], a, out, b)
            reports[arm] = h.require_gate(out, a)
        resource = read(out/'RESOURCE_GATE.json')
        require(resource['binding'] == binding(arm) and resource['approval_sha256'] == a['_approval_sha256']
            and resource['status'] == 'PASS' and resource['headroom_fraction'] >= .15
            and resource['optimizer_updates'] == 0 and resource['new_generate_calls'] == 0
            and resource['cache_tokens'] == 768 and resource['full_cache_batch'] == 4,
            'Missing/failed full-cache resource gate')
        require(identity() == initial, 'GPU identity/workload changed during arm')
        write(suite/f'{arm}_COMPLETE.json', dict(status='PASS', binding=binding(arm),
            elapsed_seconds=time.monotonic()-start, GPU=initial))
    with patch.object(old, 'binding', binding):
        old.require_both_gates(suite, a, h)
    check_time(a)
    summary = dict(status='TECHNICAL_FEASIBILITY_PASS_STOP_FOR_RESEARCHER_DECISION',
        full_training_authorized=False, evaluation_authorized=False,
        discarded_optimizer_updates=6, TRAIN_technical_generation_calls=4,
        full_cache_stress_models=2, gate_adapters_forbidden_for_training=True,
        GPU=initial, approval_sha256=a['_approval_sha256'],
        arm_reports=reports, resource_reports={x: read(suite/x/'RESOURCE_GATE.json') for x in ARMS},
        schedule_estimates=estimate_schedule(reports, timings),
        quality_improvement='UNTESTED', next_action='STOP: new researcher decision and separate training contract required')
    write(suite/'FEASIBILITY_COMPLETE_STOP_FOR_RESEARCHER_REVIEW.json', summary)
    return summary


def worker_from_cli():
    parent, approval, suite, stop = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
    a = authorize(approval, True)
    a['_runtime_stop_monotonic'] = stop
    try:
        check_time(a)
        _, configs, index, h, b = preflight(parent)
        check_time(a)
        run_pair(suite, a, configs, index, h, b)
    except BaseException as exc:
        if not (suite/'STOP_FAILURE.json').exists():
            write(suite/'STOP_FAILURE.json', dict(status='FAIL_STOP_NO_RETRY', error=str(exc),
                traceback=traceback.format_exc(), automatic_retry=False, full_training_authorized=False))
        raise


def execute(parent, approval, a):
    """Linux dedicated host only. Process-group watchdog also bounds a hung CUDA call."""
    require(os.name == 'posix', 'GPU execution requires the dedicated Linux RTX3090 host')
    c = verify_overlay()
    suite = parent/c['run_root']
    require(suite.parent.resolve() == (parent/'runs').resolve(), 'Unsafe output namespace')
    suite.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    remaining = (datetime.fromisoformat(a['execution_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    budget = min(a['maximum_runtime_seconds'], remaining)
    require(budget > 0, 'Authorization expired before launch')
    write(suite/'START.json', dict(experiment=EXPERIMENT, approval_sha256=a['_approval_sha256'],
        actual_utc=datetime.now(timezone.utc).isoformat(), watchdog_seconds=budget,
        training_authorized=False, evaluation_authorized=False))
    write(suite/'RESEARCHER_AUTHORIZATION.json', read(approval))
    code = 'import run_led_large_boundary_global_feasibility_v1 as g; g.worker_from_cli()'
    command = [sys.executable, '-u', '-c', code, str(parent), str(approval.resolve()), str(suite), str(start+budget)]
    env = dict(os.environ, PYTHONPATH=str(R/'scripts'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (suite/'WORKER.log').open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, cwd=R, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException as exc:
            write(suite/'STOP_FAILURE.json', dict(status='LAUNCH_FAILURE_NO_RETRY',
                error=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
            raise
        try:
            rc = process.wait(timeout=max(.01, budget-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            if not (suite/'STOP_FAILURE.json').exists():
                write(suite/'STOP_FAILURE.json', dict(status='WATCHDOG_STOP_NO_RETRY',
                    error='Execution wall cap or authorization deadline reached', automatic_retry=False,
                    termination_attempted=True))
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            write(suite/'WATCHDOG_EXIT.json', dict(returncode=process.returncode, worker_pid=process.pid))
            raise ValueError('Gate-only worker stopped at wall cap/deadline; preserve partial evidence')
    write(suite/'PROCESS_EXIT.json', dict(returncode=rc, seconds=time.monotonic()-start))
    complete = (suite/'FEASIBILITY_COMPLETE_STOP_FOR_RESEARCHER_REVIEW.json').is_file()
    if (rc != 0 or not complete) and not (suite/'STOP_FAILURE.json').exists():
        write(suite/'STOP_FAILURE.json', dict(status='INCOMPLETE_STOP_NO_RETRY',
            returncode=rc, complete_receipt_present=complete, automatic_retry=False))
    require(rc == 0 and not (suite/'STOP_FAILURE.json').exists() and complete,
        'Incomplete/failed feasibility; inspect WORKER.log and STOP_FAILURE; no retry')
    print('FEASIBILITY COMPLETE. STOP. No training/evaluation was authorized. Report:', suite)


def main():
    parser = argparse.ArgumentParser(description='Boundary gate-only overlay; no full-training/evaluation command')
    parser.add_argument('mode', choices=('preflight', 'feasibility'))
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    parent = args.parent_root.resolve()
    if args.mode == 'preflight':
        require(not args.execute and args.approval is None, 'CPU preflight does not consume GPU approval')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        preflight(parent)
        print('OFFLINE_GATE_ONLY_PREFLIGHT_PASS', {x: binding(x) for x in ARMS})
    else:
        a = authorize(args.approval, args.execute)
        execute(parent, args.approval, a)


if __name__ == '__main__':
    main()
