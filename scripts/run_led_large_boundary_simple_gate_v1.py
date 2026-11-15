"""Operational adapter only: frozen Boundary gate, simplified provider safety.

No new training/evaluation entry point. The original scientific runners remain
byte-identical. Exactly one two-arm feasibility attempt; no automatic retry.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys, time, signal, traceback
from unittest.mock import patch
from led_large_gpu_execution_simple_v1 import SimpleSafety, acquire_run_lock, validate_processes


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


class GateSafety:
    """Idle until first model load, then one stable opaque NVIDIA compute PID.

    This is provider assurance + cooperative locking, NOT PID authentication.
    """
    def __init__(self, simple, identity):
        self.simple=simple; self.identity=identity; self.model_seen=False
    def identity_check(self):
        if self.model_seen:
            result=self.simple.check()
        else:
            snap=self.simple.snapshot()
            validate_processes(snap['compute_pids'], before_model=True)
            result={k:snap[k] for k in ('uuid','name','memory_total_MiB','driver')}
        if result != self.identity: raise ValueError('Assigned GPU identity changed')
        return result
    def after_load(self):
        self.model_seen=True
        return self.identity_check()


def setup(root):
    root=Path(root).resolve(); ops=root/'ops'; overlay=root/'overlay'; parent=root/'parent'
    sys.path[:0]=[str(overlay/'scripts'),str(parent/'scripts')]
    import run_led_large_boundary_global_feasibility_v1 as g
    import hashlib
    for rel, digest in read(ops/'OPS_MANIFEST.json')['files'].items():
        p=(ops/rel).resolve()
        if not p.is_relative_to(ops) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Operational binding mismatch: '+rel)
    envelope=read(ops/'EXECUTION_APPROVAL.json')
    if envelope['instance_id']!=57562275 or os.environ.get('CONTAINER_ID')!='57562275':
        raise ValueError('Wrong authorized instance')
    if not envelope['gate_only_approved'] or envelope['automatic_retry'] or envelope['full_training_authorized']:
        raise ValueError('Wrong operational authorization scope')
    for rel,digest in envelope['code_bindings'].items():
        if hashlib.sha256((ops/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError('Approved new code differs: '+rel)
    if envelope['frozen_gate_bindings']!=read(ops/'GATE_APPROVAL.json')['bindings']:
        raise ValueError('Frozen scientific gate binding differs')
    if hashlib.sha256((ops/'RESEARCHER_REQUEST.txt').read_bytes()).hexdigest()!=envelope['request_sha256']:
        raise ValueError('Researcher request binding differs')
    a=g.authorize(ops/'GATE_APPROVAL.json',True)
    return root,ops,overlay,parent,g,a,envelope


def worker(root, stop):
    root,ops,overlay,parent,g,a,envelope=setup(root)
    a['_runtime_stop_monotonic']=float(stop)
    suite=parent/read(g.CONFIG)['run_root']
    fd=None
    try:
        g.check_time(a)
        _,configs,index,h,b=g.preflight(parent)
        fd=acquire_run_lock('/workspace/.led_large_experiment.lock')
        write(suite/'COMMON_RUN_LOCK.json',dict(acquired=True,cooperative_only=True))
        safety=SimpleSafety(envelope,lambda name,value:write(suite/name,value))
        identity=safety.before_model()
        guarded=GateSafety(safety,identity)
        native_load=b.load_model; native_check=b.MemoryMeter.check
        def load(*args,**kwargs):
            result=native_load(*args,**kwargs)
            guarded.after_load()
            return result
        def check(meter):
            native_check(meter)
            guarded.identity_check()
        write(suite/'OPERATIONAL_ADAPTATION.json',dict(
            original_scientific_code_unchanged=True,
            original_raw_PID_identity_function_not_called=True,
            explicit_run_pair_identity_argument=True,common_lock=True,
            model_load_and_memory_check_workload_sampling=True,
            limitation='Provider assurance and cooperative lock, not authenticated PID ownership or continuous isolation',
            new_runner_sha256=envelope['code_bindings']['run_led_large_boundary_simple_gate_v1.py']))
        with patch.object(b,'load_model',load),patch.object(b.MemoryMeter,'check',check):
            g.run_pair(suite,a,configs,index,h,b,identity=guarded.identity_check)
    except BaseException as exc:
        if not (suite/'STOP_FAILURE.json').exists():
            write(suite/'STOP_FAILURE.json',dict(status='FAIL_STOP_NO_RETRY',error=str(exc),
                traceback=traceback.format_exc(),automatic_retry=False,full_training_authorized=False))
        raise
    finally:
        if fd is not None: os.close(fd)


def execute(root):
    root,ops,overlay,parent,g,a,envelope=setup(root)
    suite=parent/read(g.CONFIG)['run_root']
    if suite.parent.resolve()!=(parent/'runs').resolve():raise ValueError('Unsafe output root')
    suite.mkdir(parents=True,exist_ok=False)
    budget=min(7200,(datetime.fromisoformat(a['execution_deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
    if budget<=0:raise ValueError('Expired fresh gate-only approval')
    start=time.monotonic()
    write(suite/'START.json',dict(actual_utc=datetime.now(timezone.utc).isoformat(),
        watchdog_seconds=budget,training_authorized=False,evaluation_authorized=False,
        approval_sha256=a['_approval_sha256']))
    write(suite/'RESEARCHER_AUTHORIZATION.json',read(ops/'GATE_APPROVAL.json'))
    command=[sys.executable,'-u',str(Path(__file__)),'worker',str(root),str(start+budget)]
    env=dict(os.environ,PYTHONPATH=str(ops)+os.pathsep+str(overlay/'scripts')+os.pathsep+str(parent/'scripts'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (suite/'WORKER.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=overlay,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=process.wait(timeout=max(.01,budget-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            write(suite/'WATCHDOG_STOP_NO_RETRY.json',dict(status='TIMEOUT_STOP',automatic_retry=False))
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            rc=process.returncode
    complete=(suite/'FEASIBILITY_COMPLETE_STOP_FOR_RESEARCHER_REVIEW.json').is_file()
    write(suite/'PROCESS_EXIT.json',dict(returncode=rc,seconds=time.monotonic()-start,complete_receipt=complete))
    if rc!=0 or not complete or (suite/'STOP_FAILURE.json').exists():
        raise ValueError('Incomplete/failed gate; no retry or continuation')


if __name__=='__main__':
    if len(sys.argv)<3 or sys.argv[1] not in ('worker','execute'):
        raise SystemExit('Only a bounded worker/execute gate is supported')
    if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
    else:execute(sys.argv[2])
