"""CPU-only, explicit checkpoint-250 recovery branch; original run remains immutable."""
import json
from pathlib import Path
import shutil
from stage_i_led_large_full_train_v5 import ROOT, read, sha, require, write, binding

PREP=ROOT/'outputs/stage_i/led_large_checkpoint250_resume_v1'


def verify_source(source):
    source=Path(source).resolve()
    spec=read(PREP/'SOURCE_BINDINGS.json')
    require(read(source/'train/BINDING.json')==binding()==spec['v5_execution_bindings'], 'Source runner binding differs')
    require(not (source/'train/COMPLETION.json').exists() and not (source/'train/FAILURE_STOP_NO_RETRY.json').exists(),
            'Completed or numerical-failure run must not be automatically recovered')
    require({p.name for p in (source/'train').glob('checkpoint-*')}=={'checkpoint-250'},
            'Checkpoint set changed; inspect newer evidence rather than rolling back')
    for rel,digest in spec['source_files'].items():
        require(sha(source/rel)==digest, 'Source changed since reviewed interrupted snapshot: '+rel)
    proof=read(source/'train/checkpoint-250/checkpoint_integrity.json')
    require(proof['binding']==binding() and proof['global_step']==250, 'Wrong recovery checkpoint')
    for name,digest in proof['files'].items():
        require(Path(name).name==name and sha(source/'train/checkpoint-250'/name)==digest,
                'Checkpoint component missing/changed: '+name)
    progress=[json.loads(s) for s in (source/'train/progress.jsonl').read_text().splitlines()]
    gradients=[json.loads(s) for s in (source/'train/gradient_checks.jsonl').read_text().splitlines()]
    require([x['step'] for x in progress]==list(range(1,279))
            and [x['step'] for x in gradients]==list(range(1,279)), 'Unexpected additional or missing updates')
    require(read(source/'train/checkpoint-250/trainer_state.json')['global_step']==250, 'Trainer state step differs')
    return spec


def retained_prefix(raw, step):
    """Prefix in a NEW recovery branch only; never edit the original 278-step logs."""
    lines=raw.splitlines(keepends=True)
    rows=[json.loads(line) for line in lines]
    require([x['step'] for x in rows]==list(range(1,279)), 'Unexpected original log order')
    return b''.join(line for line,row in zip(lines,rows) if row['step']<=step)


def prepare(source, output):
    source=Path(source).resolve();output=Path(output).resolve()
    require(source!=output and source.parent==output.parent, 'Recovery must be a separate sibling run')
    spec=verify_source(source)
    require(not output.exists(), 'Recovery branch already exists; do not overwrite or repeat it')
    output.mkdir()
    train=output/'train';train.mkdir()
    evidence=output/'recovery_evidence';evidence.mkdir()
    # Keep every original checkpoint byte and receipt, including old v5 binding.
    shutil.copytree(source/'train/checkpoint-250',train/'checkpoint-250')
    for name in ('BINDING.json','AUTHORIZATION.json','INITIAL_MODEL_BINDINGS.json','ACTUAL_TRAINING_ARGUMENTS.json',
                 'ACTUAL_AMP_INITIALIZATION.json','ACCEPTED_V4_GATE.json','CHECKPOINT_RECEIPTS.jsonl'):
        shutil.copy2(source/'train'/name,train/name)
    for name in ('progress.jsonl','gradient_checks.jsonl'):
        raw=(source/'train'/name).read_bytes()
        (evidence/('ORIGINAL_'+name)).write_bytes(raw)
        (train/name).write_bytes(retained_prefix(raw,250))
    state=read(train/'checkpoint-250/trainer_state.json')
    require(state['global_step']==250 and state['epoch']==1.0 and state['max_steps']==1250,
            'Recovery trajectory differs')
    ledger=dict(status='EXPLICIT_INTERRUPTION_RECOVERY_NOT_NEW_EXPERIMENT',
        source_run=str(source),recovery_run=str(output),restored_checkpoint_step=250,
        original_last_logged_step=278,superseded_uncheckpointed_steps=list(range(251,279)),
        original_logs_preserved_at_source=True,original_full_logs_copied_to='recovery_evidence/',
        recovery_active_logs_are_explicit_checkpoint_prefix=True,original_checkpoint_bytes_unchanged=True,
        additional_recomputed_updates_max=28,trajectory_final_step=1250,
        cumulative_observed_update_budget_including_discarded_branch=1278,
        recover_adapter_optimizer_scheduler_scaler_rng=True,additional_epochs=4,
        no_fresh_training=True,initialized_from_checkpoint_not_gate_or_random=True,
        D051_strict_replay='FAIL_UNCHANGED',exact_CUDA_replay_not_claimed=True,
        source_binding_manifest_sha256=sha(PREP/'SOURCE_BINDINGS.json'),v5_execution_bindings=binding(),
        original_authorization_utc='2026-10-19T17:11:48+00:00',execution_deadline_utc='2026-10-21T17:11:48+00:00')
    write(output/'RECOVERY_LINEAGE.json',ledger)
    for rel,digest in spec['source_files'].items():
        require(sha(source/rel)==digest,'Original run changed while preparing recovery; stop')
    return ledger
