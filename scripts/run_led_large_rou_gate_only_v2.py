"""Only preflight or three-update discarded feasibility. There is no train mode."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import subprocess
import sys
import json
import led_large_rou_gate_only_v2 as g


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['preflight', 'feasibility'])
    p.add_argument('--condition', choices=g.CONDITIONS, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--receipt', type=Path)
    p.add_argument('--approval', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--host-preflight-sha', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.mode == 'preflight':
        _, report = g.cpu_preflight(a.condition, a.parent_root, run_tests=True)
        if a.receipt: g.write(a.receipt, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    g.require(a.execute and a.approval, 'No GPU without new gate-only approval and --execute')
    approval = g.authorize(a.approval, a.condition)
    if not a.worker:
        _, report = g.cpu_preflight(a.condition, a.parent_root, run_tests=True)
        target = g.ROOT/read_config(a.condition)['output_namespace']
        g.require(not target.exists(), 'Fresh gate-only namespace required; no retries')
        report.update(approval_sha256=approval['_approval_sha256'], created_utc=datetime.now(timezone.utc).isoformat())
        preflight_path = target.parent/(target.name+'_HOST_CPU_PREFLIGHT.json')
        g.write(preflight_path, report)
        remaining = (datetime.fromisoformat(approval['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
        g.require(remaining > 0, 'Deadline passed during CPU preflight')
        command = [sys.executable, __file__, *sys.argv[1:], '--worker', '--host-preflight-sha', g.sha(preflight_path)]
        try:
            subprocess.run(command, check=True, timeout=remaining)
        except BaseException as exc:
            g.write(target.parent/(target.name+'_WORKER_STOP_NO_RETRY.json'), dict(error=str(exc), automatic_retry=False))
            raise
        return
    # The parent process reproduced all CPU suites; worker rechecks bound bytes
    # and full parent in a fresh interpreter before touching CUDA.
    target = g.ROOT/read_config(a.condition)['output_namespace']
    host = target.parent/(target.name+'_HOST_CPU_PREFLIGHT.json')
    g.require(a.host_preflight_sha and host.is_file() and g.sha(host) == a.host_preflight_sha,
              'Worker requires exact completed host CPU receipt')
    proof = g.read(host)
    g.require(proof['binding'] == g.binding(a.condition) and proof['approval_sha256'] == approval['_approval_sha256']
              and proof['status'] == 'CPU_PREFLIGHT_PASS_NOT_GPU_PASS', 'Foreign/missing host checks')
    g.require([x['tests'] for x in proof['CPU_tests']] == [26,35,29]
              and all(x['returncode'] == 0 for x in proof['CPU_tests']), 'All three pinned CPU suites required')
    g.cpu_preflight(a.condition, a.parent_root, run_tests=False)
    b, c = g.install_gate_context(a.condition, a.parent_root)
    target = g.ROOT/read_config(a.condition)['output_namespace']
    g.require(not target.exists(), 'Fresh gate-only namespace required')
    b.feasible(c, approval, target)
    g.write(target/'STOP_AFTER_GATE.json', dict(training_authorized=False, automatic_train=False,
            next_action='Independent gate review, then separately approved training handoff',
            historical_training_binding=read_config(a.condition)['accepted_V1_binding']))


def read_config(condition):
    return g.read(g.config(condition))


if __name__ == '__main__':
    main()
