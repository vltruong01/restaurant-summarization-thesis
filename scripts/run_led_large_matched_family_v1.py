"""Fresh condition-specific RO/U gate or training. No final-inference entry point."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import os
import led_large_matched_family_v1 as m


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['preflight', 'feasibility', 'train'])
    p.add_argument('--condition', required=True, choices=m.CONDITIONS)
    p.add_argument('--parent-root', required=True, type=Path)
    p.add_argument('--approval', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--receipt', type=Path)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.mode != 'preflight' and not a.worker:
        m.require(a.execute and a.approval, 'Fresh execution receipt and --execute required')
        m.verify_overlay()
        approval = m.authorize(a.approval, a.condition, a.mode)
        # Independent process for CPU verification, including parent imports.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        command = [sys.executable, __file__, 'preflight', '--condition', a.condition,
                   '--parent-root', str(a.parent_root)]
        subprocess.run(command, env=env, check=True)
        m.check_deadline(approval)
        # Whole worker wallclock is bounded by the remaining receipt deadline.
        from datetime import datetime, timezone
        remaining = (datetime.fromisoformat(approval['execution_deadline_utc']) - datetime.now(timezone.utc)).total_seconds()
        child = [sys.executable, __file__, *sys.argv[1:], '--worker']
        try:
            subprocess.run(child, check=True, timeout=remaining)
        except BaseException as exc:
            m.write(m.ROOT / m.PREP / ('WORKER_STOP_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'),
                    dict(error=str(exc), automatic_retry=False, condition=a.condition, mode=a.mode))
            raise
        return
    c, report = m.cpu_preflight(a.condition, a.parent_root, full_parent=True)
    if a.mode == 'preflight':
        if a.receipt:
            m.write(a.receipt, report)
        print(json.dumps(report, indent=2))
        return
    m.require(a.execute and a.approval, 'Explicit execution required even in worker')
    approval = m.authorize(a.approval, a.condition, a.mode)
    b, c = m.install_condition_context(a.condition, a.parent_root)
    output = m.ROOT / c['run_root']
    m.require(not output.resolve().is_relative_to(a.parent_root.resolve()), 'Output cannot enter immutable parent')
    if a.mode == 'feasibility':
        m.require(not output.exists(), 'Fresh namespace required; no retries')
        b.feasible(c, approval, output)
    else:
        m.require(not (output / 'train').exists(), 'Fresh training only; no automatic resume')
        b.train(c, approval, output)


if __name__ == '__main__':
    main()
