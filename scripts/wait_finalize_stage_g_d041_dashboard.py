"""Lightweight one-shot postprocessing for the already running bounded D041 pilot.

No API calls, credential access, retries or production continuation.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / 'outputs/stage_g/teacher_extended_pilot_d041_v1'
report_path = folder / 'execution_postprocessing_status.json'
deadline = time.monotonic() + 12 * 3600
while not (folder / 'execution_v1/completion.json').exists():
    if time.monotonic() >= deadline:
        report_path.write_text(json.dumps({'status': 'WAIT_TIMEOUT_NO_API_ACTION',
            'API_calls': 0, 'teacher_process_not_stopped_or_retried': True}, indent=2), encoding='utf-8')
        sys.exit(1)
    time.sleep(5)
if (folder / 'human_review_bundle_report.json').exists():
    status = {'status': 'ALREADY_EXPORTED_NO_MUTATION', 'API_calls': 0}
else:
    with (folder / 'execution_postprocessing.log').open('wb') as log:
        result = subprocess.run([sys.executable, str(root / 'scripts/finalize_stage_g_d041_dashboard.py')],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT)
    status = {'status': 'COMPLETED' if result.returncode == 0 else 'POSTPROCESSING_FAILED_REVIEW_LOG',
              'exit_code': result.returncode, 'API_calls': 0,
              'scientific_protocol_changed': False, 'full_production_started': False}
report_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
print(json.dumps(status), flush=True)
