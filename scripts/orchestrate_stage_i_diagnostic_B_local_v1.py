"""Run the single approved B matrix locally; monitor memory and stop on any runner failure."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'outputs/stage_i/decoding_B_execution_control_v1'


def main():
    state_path = CONTROL / 'orchestration.json'
    if state_path.exists():
        raise FileExistsError('Preserve previous execution; do not start duplicate inference')
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_stage_i_decoding_diagnostic_B_local_v1.py'),
               '--preparation', str(ROOT / 'outputs/stage_i/decoding_diagnostic_v1'),
               '--experiment', 'beam_ngram_B', '--output', str(ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1'),
               '--approval', str(ROOT / 'outputs/stage_i/decoding_approval_B_local_v1.json'), '--execute']
    started = time.perf_counter()
    state = {'status': 'STARTING', 'start_utc': datetime.now(timezone.utc).isoformat(),
             'command': command, 'runtime_profile': 'local_rtx3050_6gb', 'automatic_retry_or_fallback': False}
    with state_path.open('x', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    with (CONTROL / 'runner.log').open('x', encoding='utf-8') as log, (CONTROL / 'gpu_metrics.csv').open('x', newline='', encoding='utf-8') as mf:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        state.update(status='RUNNING', pid=child.pid)
        state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        writer = csv.DictWriter(mf, fieldnames=['utc', 'elapsed_seconds', 'memory_used_MiB', 'memory_total_MiB', 'utilization_gpu_percent'])
        writer.writeheader()
        while True:
            sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if sample.returncode == 0:
                used, total, util = sample.stdout.strip().splitlines()[0].split(',')
                writer.writerow({'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - started,
                                 'memory_used_MiB': used.strip(), 'memory_total_MiB': total.strip(), 'utilization_gpu_percent': util.strip()})
                mf.flush()
            code = child.poll()
            if code is not None:
                break
            time.sleep(5)
    state.update(status='PROCESS_FINISHED', exit_code=code, end_utc=datetime.now(timezone.utc).isoformat(),
                 orchestration_seconds=time.perf_counter() - started)
    state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(state), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
