"""Export existing D-050 evidence only. No GPU, model, API or data transformation."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d050_gpu_feasibility_v1'
NAME = 'stage_i_d050_gpu_feasibility_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    assert not DEST.exists() and not ZIP.exists(), 'Preserve prior exports'
    comparison = json.loads((OUT / 'AB_COMPARISON.json').read_text())
    assert comparison['status'] == 'A_PASS_B_REPLAY_BLOCKER_PILOT_NOT_READY'
    sources = {}

    def include(relative, source):
        assert source.is_file() and relative not in sources
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        sources[relative] = source

    for path in OUT.rglob('*'):
        if path.is_file():
            include('execution_evidence/' + path.relative_to(OUT).as_posix(), path)
    for arm in ['A_decoder_only', 'B_encoder_decoder']:
        raw = ROOT / 'outputs/stage_i/paired_lora_pilot_runs_v1' / arm / 'feasibility'
        for path in raw.rglob('*'):
            if path.is_file():
                include('temporary_feasibility_checkpoints/' + arm + '/' + path.relative_to(raw).as_posix(), path)
    for name in ['run_stage_i_d050_feasibility_v1.py', 'report_stage_i_d050_feasibility_v1.py',
                 'build_stage_i_d050_review_bundle_v1.py', 'run_stage_i_paired_lora_pilot_v1.py',
                 'stage_i_paired_pilot.py', 'stage_hi_led_data.py', 'stage_i_decoding_diagnostics.py',
                 'test_stage_i_paired_lora_pilot_v1.py']:
        include('code/' + name, ROOT / 'scripts' / name)
    prep = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
    for name in ['pilot_manifest.json', 'epoch_orders.json', 'PREPARATION_MANIFEST.json',
                 'initialization_hashes.json', 'module_attachment_verification.json',
                 'STAGE_I_PAIRED_LORA_PILOT_PLAN.md', 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md',
                 'offline_tests_release.log']:
        include('D049_context/' + name, prep / name)
    include('configs/stage_i_paired_lora_pilot_v1.json', ROOT / 'configs/stage_i_paired_lora_pilot_v1.json')
    for name in ['DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md',
                 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                 'specs/009d_paired_lora_pilot_preparation.md', 'specs/009e_d050_gpu_feasibility.md']:
        include('current_documentation/' + Path(name).name, ROOT / name)
    cloud = ROOT / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
    for name in ['assets/models/led-base-16384.config_manifest.json', 'assets/models/led-base-16384/config.json',
                 'assets/tokenizers/led-base-16384.manifest.json', 'outputs/stage_h/led_dataset_v1/manifest.json',
                 'BACKUP_MANIFEST.json']:
        include('frozen_manifests/' + name, cloud / name)
    for source in sources.values():
        if source.suffix in ['.pt', '.safetensors']:
            continue
        data = source.read_bytes()
        encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', data.decode(encoding)), 'Possible credential in ' + source.name
    DEST.mkdir()
    origins = []
    for relative, source in sources.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, target)
        assert sha(target) == digest
        origins.append({'path': relative, 'source_workspace_path': source.relative_to(ROOT).as_posix(),
                        'bytes': target.stat().st_size, 'sha256': digest})
    write(DEST / 'INDEX.json', {
        'report': 'execution_evidence/STAGE_I_D050_GPU_FEASIBILITY_REPORT.md',
        'readiness': 'execution_evidence/PAIRED_PILOT_READINESS_AFTER_D050.md',
        'AB_comparison': 'execution_evidence/AB_COMPARISON.json',
        'replay_comparison': 'execution_evidence/REPLAY_COMPARISON.json',
        'train_controls': 'execution_evidence/REAL_TRAIN_CONTROLS.json',
        'architectures': {arm: {'observation_directory': 'execution_evidence/' + arm,
                              'prepared_runner_output_and_step2_checkpoint': 'temporary_feasibility_checkpoints/' + arm,
                              'console_log': 'execution_evidence/' + arm + '_execution.log'}
                          for arm in ['A_decoder_only', 'B_encoder_decoder']},
        'paths_relative_to_bundle_root': True, 'new_pilot_targets_or_predictions': 0})
    (DEST / 'README_FIRST.md').write_text('''# D-050 GPU feasibility review — A PASS, B replay blocker

**The approved bounded execution is complete and stopped. No pilot training or generation.**

Read `CHATGPT_REVIEW_REQUEST.md`, `execution_evidence/STAGE_I_D050_GPU_FEASIBILITY_REPORT.md` and `execution_evidence/PAIRED_PILOT_READINESS_AFTER_D050.md` first. `INDEX.json` links code, raw execution logs, real train controls, masks, branch gradients, optimizer updates, replay hashes and checkpoint states.

Each architecture independently used the unchanged D-049 protocol on local RTX3050 Laptop6GB: three distinct optimizer steps plus one replay of step3 after restoring step2, 32 microbatches. Actual scopes24/48 and parameters294,912/589,824 match. Real decoder/local/global gradients are finite/nonzero by step3. Backbone parameters remain byte-identical and gradient-free; accepted data and historical checkpoints remain unchanged.

A passes exact replay. B restores model/optimizer/scheduler/scaler/RNG exactly but fails unchanged rtol1e-5/atol1e-6. All48 decoder hashes match;22/24 local and16/24 global encoder hashes differ. Small gradient-norm drift is not a parameter-error estimate. No tolerance or scientific setting was changed. Encoder backward numerical nondeterminism is a hypothesis, not a proven cause.

B numerical loss history and Torch allocator maxima were not retained because the prepared routine writes them after the failed replay assertion. Null fields are explicit. Both arms completed finite-loss assertions and all intended gradients/updates without OOM/NaN/Inf/skipped steps. Device-memory figures are 2-second samples. No new GPU replay was run to fill missing fields.

## Package scope

- All D-050 logs, code/config snapshots, observer state fingerprints and before/after parameter hashes are preserved verbatim.
- `temporary_feasibility_checkpoints/` contains isolated step2 adapters/optimizer/RNG checkpoints and the original prepared-runner PASS or failure files. They are disposable feasibility evidence and MUST NOT warmstart the matched pilot. Checkpoint files are trusted locally generated artifacts with SHA manifests, not externally supplied models.
- `REAL_TRAIN_CONTROLS.json` contains the exact three accepted train examples with full source/Teacher target, input/label token IDs and existing evidence IDs; no new labels or targets. No held-out examples.
- `D049_context/` documents the unchanged paired plan, fixed512 IDs, evaluation protocol and prior20 offline tests. Historical text saying GPU execution was unapproved is superseded only for this completed D-050 scope. Training/generation remain unapproved.
- Current documentation records the blocker. The earlier approved execution spec/config is also snapshotted as provenance. The D-050 approval JSON permits feasibility only, not a new run or pilot training.

This package is self-contained for evidence/code review, not a standalone model deployment: pretrained weights, full NPZ datasets and historical training checkpoints are omitted. Original workspace paths in logs/manifests are provenance; all ZIP paths are relative. No credentials or Teacher API calls.

Next proposal is a separately approved bounded B replay with continuous loss/memory logging and persisted original/replayed tensor values. Do not loosen tolerance or enable a new deterministic policy automatically. Stop for researcher review. The full Trainer loop, fresh-process restart and summary quality remain untested.
''', encoding='utf-8')
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text('''# Independently review D-050 before authorizing any more GPU work

Please assess the actual evidence, not only the labels PASS/FAIL.

1. Confirm the prepared runner/config are unchanged; attached scopes/counts, frozen backbone hashes, shared decoder initialization, complete real data/masks and optimizer accumulation/scheduler behavior are correct. Account for zero initial B, initial LR0, delayed A gradients and AdamW decay of A at step2 despite zero gradients.
2. Check that real gradients reach all intended encoder local/global and decoder A/B adapters by step3. Distinguish this success and practical GPU fit from full replay/readiness acceptance.
3. Inspect exact model/optimizer/scheduler/scaler/RNG restore fingerprints, Adam step1/2/3/replayed3 and the failed B allclose assertion. All decoder hashes match; differences localize to encoder. Does this favor a numerical-backward hypothesis or suggest a checkpoint/implementation bug? Do not call the cause proven.
4. Explain what can and cannot be inferred from gradient-norm differences and tensor hashes. Numeric B losses, allocator maxima and elementwise replay error distributions are missing; do not fabricate them or dismiss the failed gate as harmless. A passed exact replay on the same environment.
5. Review the minimal proposed correction: persist losses/memory in real time/finally and original/replayed tensors before asserting; only after explicit approval, repeat a bounded B replay under unchanged scientific settings/tolerance to quantify errors. Consider deterministic-kernel diagnosis separately, with a consistent future policy across both arms; do not silently change precision, data, attention, LoRA scope, batch, LR or tolerance.
6. Recommend a precise next bounded verification scope. Do not approve the512-business pilot yet without resolving the documented blocker or an explicitly justified researcher-reviewed criterion amendment. Full Trainer and fresh-process recovery remain untested; encoder capacity/location and two-epoch limits still apply.

Return confirmed facts, possible alternative causes, limitations and the smallest justified next action. Do not execute training/inference, regenerate Teacher targets, alter accepted data/checkpoints/filters, inspect held-out data or run decoding experiments. No summary-quality or encoder-causality claim is established by D-050.
''', encoding='utf-8')
    write(DEST / 'EXPORT_REPORT.json', {'status': 'OFFLINE_EXPORT_PASS', 'source_origins': origins,
        'A_status': 'PASS', 'B_status': 'REPLAY_BLOCKER', 'pilot_ready': False,
        'GPU_calls_during_export': 0, 'additional_replays': 0,
        'existing_targets_modified': False, 'held_out_examples': 0})
    payload = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(DEST.rglob('*')) if p.is_file()]
    write(DEST / 'MANIFEST.json', {'manifest_excludes_itself': True, 'files': payload})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(payload) + 1
        for item in payload:
            data = archive.read(NAME + '/' + item['path'])
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
        index = json.loads(archive.read(NAME + '/INDEX.json'))
        for key in ['report', 'readiness', 'AB_comparison', 'replay_comparison', 'train_controls']:
            assert NAME + '/' + index[key] in archive.namelist()
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    result = {'status': 'ZIP_CRC_SHA256_AND_INDEX_PASS', 'zip': ZIP.relative_to(ROOT).as_posix(),
              'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size / 2**20, 'sha256': sha(ZIP),
              'payload_files': len(payload), 'A': 'PASS', 'B': 'REPLAY_BLOCKER', 'additional_GPU_execution': 0}
    write(OUT / 'BUNDLE_EXPORT_RECEIPT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
