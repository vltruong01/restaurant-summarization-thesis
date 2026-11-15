"""Stop only the verified D041 serial worker between paid requests (Windows).

Briefly suspend the worker to inspect a stable intent/checkpoint boundary.
An active request is always resumed; it is never interrupted or resent.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import time
from pathlib import Path
from prepare_stage_g_teacher_pilot import immutable

root = Path(__file__).resolve().parents[1]
folder = root / 'outputs/stage_g/teacher_extended_pilot_d041_v1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pid', type=int, required=True)
args = parser.parse_args()
query = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {args.pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
result = subprocess.run(['powershell', '-NoProfile', '-Command', query], capture_output=True, text=True, check=True)
identity = json.loads(result.stdout)
assert identity['ProcessId'] == args.pid
assert 'run_stage_g_extended_pilot_d041_dashboard.py --execute-pilot' in identity['CommandLine']
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
nt = ctypes.WinDLL('ntdll')
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
nt.NtSuspendProcess.argtypes = nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
nt.NtSuspendProcess.restype = nt.NtResumeProcess.restype = wintypes.LONG
handle = kernel.OpenProcess(0x0800 | 0x0001 | 0x1000, False, args.pid)
if not handle: raise ctypes.WinError(ctypes.get_last_error())
out = folder / 'execution_v1'
deadline = time.monotonic() + 240
suspended = False
try:
    while time.monotonic() < deadline:
        if (out / 'completion.json').exists():
            raise ValueError('Execution already completed; do not replace completion')
        if nt.NtSuspendProcess(handle) < 0:
            raise RuntimeError('Cannot suspend the verified worker')
        suspended = True
        try:
            active = [p.stem for p in (out / 'request_intents').glob('*.json')
                      if not (out / 'checkpoints' / p.name).exists()]
            if not active:
                if not kernel.TerminateProcess(handle, 0):
                    raise ctypes.WinError(ctypes.get_last_error())
                suspended = False
                report = {'status': 'USER_APPROVED_SCHEDULER_HANDOFF_AT_SAVED_RESPONSE_BOUNDARY',
                          'worker_pid': args.pid, 'active_requests_at_stop': [],
                          'paid_requests_interrupted': 0, 'API_calls': 0,
                          'request_intents_saved': len(list((out / 'request_intents').glob('*.json'))),
                          'checkpoints_saved': len(list((out / 'checkpoints').glob('*.json'))),
                          'target_content_modified': False}
                immutable(folder / 'concurrency5_serial_handoff.json', report)
                print(json.dumps(report), flush=True)
                break
        finally:
            if suspended:
                if nt.NtResumeProcess(handle) < 0:
                    raise RuntimeError('Cannot resume the verified worker')
                suspended = False
        time.sleep(0.02)
    else:
        raise TimeoutError('No idle boundary observed; active request left running')
finally:
    if suspended: nt.NtResumeProcess(handle)
    kernel.CloseHandle(handle)
