"""CPU preparation and explicit launcher for the interrupted Colab v5 trajectory.

The scientific runner/config remain unchanged. This helper never starts fresh training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = 'F_full_train_v5_resume_checkpoint250_v1'
OUTPUT_NAME = 'F_full_train_v5_resume_checkpoint750_v2'
STEP = 750
DEADLINE = '2026-10-21T17:11:48+00:00'
METADATA = ('BINDING.json', 'AUTHORIZATION.json', 'INITIAL_MODEL_BINDINGS.json',
            'ACTUAL_TRAINING_ARGUMENTS.json', 'ACTUAL_AMP_INITIALIZATION.json',
            'ACCEPTED_V4_GATE.json', 'CHECKPOINT_RECEIPTS.jsonl')
FULL_STATE = {'adapter_model.safetensors', 'adapter_config.json', 'training_args.bin',
              'optimizer.pt', 'scheduler.pt', 'scaler.pt', 'rng_state.pth', 'trainer_state.json'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_new(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def frozen():
    import stage_i_led_large_full_train_v5 as v
    return v

def copy_bytes(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as a, dest.open('xb') as b:
        shutil.copyfileobj(a, b, length=1024*1024)

def snapshot_files(source):
    files = {}
    # Only this known run; no datasets, held-out cases or unrelated Drive files.
    for p in sorted(source.rglob('*')):
        require(not p.is_symlink(), 'Symlink inside run; stop and inspect: ' + str(p))
        if p.is_file():
            files[p.relative_to(source).as_posix()] = sha(p)
    return files

def log_prefix(path, checkpoint):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    require(rows and [x['step'] for x in rows] == list(range(1, len(rows)+1)),
            'Noncontiguous/truncated source log: ' + str(path))
    # Explicitly bound to this later interruption, never the old278-step snapshot.
    require(890 <= rows[-1]['step'] < 1000, 'Unexpected progress; inspect before changing recovery scope')
    return raw, b''.join(line for line, row in zip(lines, rows) if row['step'] <= checkpoint), rows[-1]['step']

def verify_source(source, v):
    train = source/'train'
    require(source.is_dir() and train.is_dir(), 'Missing mounted recovery run; do not create replacement Drive folders')
    require(not any((train/n).exists() for n in ('COMPLETION.json', 'FAILURE_STOP_NO_RETRY.json')),
            'Completed or failed numerical run cannot be resumed by this repair')
    require(read(train/'BINDING.json') == v.binding(), 'Source scientific binding changed')
    checkpoints = {p.name for p in train.glob('checkpoint-*')}
    require(checkpoints == {'checkpoint-250', 'checkpoint-500', 'checkpoint-750'},
            'Checkpoint set differs; no automatic fallback or rollback to250')
    receipts = [json.loads(s) for s in (train/'CHECKPOINT_RECEIPTS.jsonl').read_text().splitlines()]
    require([r['step'] for r in receipts] == [250,500,750], 'Checkpoint receipt history differs')
    for step, row in zip((250,500,750), receipts):
        p = train/f'checkpoint-{step}'
        proof = read(p/'checkpoint_integrity.json')
        require(sha(p/'checkpoint_integrity.json') == row['checkpoint_integrity_sha256'], 'Receipt history hash differs')
        require(proof['binding'] == v.binding() and proof['global_step'] == step,
                'Foreign checkpoint binding/step')
        require(FULL_STATE.issubset(proof['files']), 'Incomplete full-state checkpoint')
        for name, digest in proof['files'].items():
            require(Path(name).name == name and name not in ('.','..') and '\\' not in name and ':' not in name,
                    'Unsafe checkpoint member')
            require(sha(p/name) == digest, 'Checkpoint member hash differs: '+name)
        state = read(p/'trainer_state.json')
        require(state['global_step'] == step and state['epoch'] == step/250 and state['max_steps'] == 1250,
                'Checkpoint trajectory differs')
    logs = {n:log_prefix(train/n, STEP) for n in ('progress.jsonl','gradient_checks.jsonl')}
    require(len({x[2] for x in logs.values()}) == 1, 'Progress/gradient histories disagree; preserve evidence')
    prior = read(source/'RECOVERY_LINEAGE.json')
    require(prior['restored_checkpoint_step'] == 250 and prior['original_last_logged_step'] == 278
            and prior['additional_recomputed_updates_max'] == 28
            and prior['execution_deadline_utc'] == DEADLINE, 'Prior recovery lineage differs')
    return logs

def paths(drive):
    drive = Path(drive)
    require(drive.is_dir() and (drive/'runs').is_dir(), 'Drive/runs unavailable: remount correct account/shortcut first')
    source = (drive/'runs'/SOURCE_NAME).resolve()
    # Derive the sibling from the existing resolved source, avoiding mixed shortcut paths.
    output = source.parent/OUTPUT_NAME
    require(source != output and source.parent == output.parent, 'Wrong resolved recovery paths')
    return source, output, (drive/'runs/F_gradient_gate_v4').resolve()

def prepare(drive):
    v = frozen()
    v.preflight()
    source, output, gate = paths(drive)
    v.validate_gate(gate)
    v.check_deadline(read(source/'train/AUTHORIZATION.json'))
    if output.exists():
        verify_prepared(output, v, before_launch=True)
        print('Existing unlaunched preparation verified:', output, flush=True)
        return output
    logs = verify_source(source, v)
    inventory = snapshot_files(source)
    last = logs['progress.jsonl'][2]
    output.mkdir()
    (output/'train').mkdir()
    # Preserve ALL source files and prior lineage as a separately hashed evidence snapshot.
    for rel in inventory:
        copy_bytes(source/rel, output/'recovery_evidence/source_snapshot'/rel)
    for step in (250,500,750):
        prefix = f'train/checkpoint-{step}/'
        for rel in inventory:
            if rel.startswith(prefix):
                copy_bytes(source/rel, output/rel)
    for name in METADATA:
        copy_bytes(source/'train'/name, output/'train'/name)
    for name, (_, prefix, _) in logs.items():
        with (output/'train'/name).open('xb') as f:
            f.write(prefix)
    ledger = dict(source_run=str(source), recovery_run=str(output), restored_checkpoint_step=750,
        original_last_logged_step=last, additional_recomputed_updates_max=last-750,
        prior_recomputed_updates=28, cumulative_observed_optimizer_updates_max=1250+28+last-750,
        source_files=inventory, v5_execution_bindings=v.binding(), original_logs_preserved=True,
        adapter_optimizer_scheduler_scaler_rng_restored=True, exact_CUDA_replay_not_claimed=True,
        D051_strict_replay='FAIL_UNCHANGED', execution_deadline_utc=DEADLINE,
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    write_new(output/'RECOVERY_LINEAGE.json', ledger)
    approval = {k:value for k,value in read(source/'train/AUTHORIZATION.json').items() if not k.startswith('_')}
    approval.update(v.binding())
    approval.update(approval_source='researcher_message',
        researcher_explicit_authorization='chạy tiếp thế nào hãy sửa notebooj để chạy tiếp từ checlpoint; '
            'trong thời gian đó bản colab cũng gặp lỗi nếu có thể gửi lại notebook đã sửa',
        repair_record_created_at_utc=datetime.now(timezone.utc).isoformat(),
        timestamp_note='Repair record creation time, not a claimed exact chat authorization timestamp.',
        resume_after_interruption_approved=True,
        resume_checkpoint_integrity_sha256=sha(output/'train/checkpoint-750/checkpoint_integrity.json'),
        recovery_rollback_explicitly_disclosed=True, recovery_original_last_step=last,
        recovery_checkpoint_step=750, recovery_recomputed_updates_max=last-750,
        cumulative_observed_optimizer_updates_max=ledger['cumulative_observed_optimizer_updates_max'],
        execution_deadline_utc=DEADLINE)
    write_new(output/'RECOVERY_APPROVAL.json', approval)
    require(snapshot_files(source) == inventory, 'Source changed during CPU copy; stop, preserve partial branch')
    files = snapshot_files(output)
    write_new(output/'PREPARED_FILES.json', files)
    verify_prepared(output, v, before_launch=True)
    print(f'READY: checkpoint750 ->1250; recompute {last-750} updates; prior28 preserved.', flush=True)
    print('Output:', output, 'Deadline:', DEADLINE, flush=True)
    return output

def verify_prepared(output, v, before_launch=False):
    require((output/'PREPARED_FILES.json').is_file(), 'Incomplete CPU preparation; preserve folder, do not retry/overwrite')
    train = output/'train'
    if before_launch:
        require(not (output/'launcher_logs/train.log').exists()
                and not (train/'RESUME_AUTHORIZATION_STEP_750.json').exists()
                and not (train/'COMPLETION.json').exists()
                and not (train/'FAILURE_STOP_NO_RETRY.json').exists(), 'This recovery already launched; no silent retry')
        for rel, digest in read(output/'PREPARED_FILES.json').items():
            require(sha(output/rel) == digest, 'Prepared content changed: '+rel)
    ledger = read(output/'RECOVERY_LINEAGE.json')
    require(ledger['v5_execution_bindings'] == v.binding() and ledger['restored_checkpoint_step'] ==750
            and ledger['execution_deadline_utc'] == DEADLINE, 'Recovery lineage/binding differs')
    return ledger

def launch(drive, mode):
    v = frozen()
    v.preflight()
    source, output, gate = paths(drive)
    ledger = verify_prepared(output, v, before_launch=mode=='train')
    if mode == 'train':
        require(snapshot_files(source) == ledger['source_files'],
                'Interrupted source advanced/changed; stop to avoid concurrent trajectories')
    v.validate_gate(gate)
    approval = output/'RECOVERY_APPROVAL.json'
    v.authorize(approval, mode, True)
    logdir = output/'launcher_logs'
    logdir.mkdir(exist_ok=True)
    if mode == 'evaluate':
        complete = read(output/'train/COMPLETION.json')
        require(complete['steps']==1250 and complete['epochs']==5, 'Training incomplete')
        require(not (output/'evaluate').exists(), 'Evaluation already attempted; no automatic repeat')
    log = logdir/(mode+'.log')
    cmd = [sys.executable, 'scripts/run_stage_i_led_large_full_train_v5.py', mode, '--execute',
           '--approval',str(approval),'--gate-output',str(gate),'--output',str(output)]
    if mode == 'train':
        cmd += ['--resume',str(output/'train/checkpoint-750')]
    env = dict(os.environ, PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    # Do not inherit the CPU-preflight CUDA_VISIBLE_DEVICES='' into an actual launch.
    require(env.get('CUDA_VISIBLE_DEVICES') != '', 'GPU hidden by CPU-preflight environment; restart launcher cell normally')
    with log.open('x',encoding='utf-8') as f:
        process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            f.write(line);f.flush()
        status = process.wait()
    require(status==0, f'{mode} stopped ({status}); preserve evidence; no retry.')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','train','evaluate'))
    ap.add_argument('--drive',type=Path,required=True)
    a = ap.parse_args()
    if a.mode=='prepare':
        prepare(a.drive)
    else:
        launch(a.drive,a.mode)

if __name__=='__main__':
    main()
