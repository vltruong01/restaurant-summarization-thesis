"""Portable D-051 evidence export only; no model/GPU/API execution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d051_replay_investigation_v1'
NAME = 'stage_i_d051_replay_investigation_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    assert not DEST.exists() and not ZIP.exists(), 'Preserve previous exports'
    assert (OUT / 'STAGE_I_D051_REPLAY_INVESTIGATION_REPORT.md').is_file()
    assert read(OUT / 'POST_EXECUTION_INTEGRITY.json')['status'] == 'PASS'
    sources = {}

    def include(relative, source):
        assert source.is_file() and relative not in sources
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        sources[relative] = source

    for source in OUT.rglob('*'):
        if source.is_file():
            include('D051/' + source.relative_to(OUT).as_posix(), source)
    previous = ROOT / 'outputs/stage_i/d050_gpu_feasibility_v1'
    for name in ['STAGE_I_D050_GPU_FEASIBILITY_REPORT.md', 'AB_COMPARISON.json', 'REPLAY_COMPARISON.json',
                 'GRADIENT_BRANCH_SUMMARY.json', 'POST_EXECUTION_INTEGRITY.json']:
        include('D050_historical/' + name, previous / name)
    for name in ['run_stage_i_d051_replay_v1.py', 'stage_i_d051_numeric.py', 'test_stage_i_d051_numeric.py',
                 'analyze_stage_i_d051_replay_v1.py', 'build_stage_i_d051_review_bundle_v1.py',
                 'report_stage_i_d051_replay_v1.py',
                 'run_stage_i_d050_feasibility_v1.py', 'run_stage_i_paired_lora_pilot_v1.py',
                 'stage_i_paired_pilot.py', 'stage_hi_led_data.py', 'stage_i_decoding_diagnostics.py']:
        include('code/' + name, ROOT / 'scripts' / name)
    for relative in ['DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md',
                     'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                     'specs/009f_d051_replay_investigation.md', 'configs/stage_i_paired_lora_pilot_v1.json']:
        include('current_documentation/' + Path(relative).name, ROOT / relative)
    previous_inputs = previous / 'REAL_TRAIN_CONTROLS.json'
    include('accepted_inputs/REAL_TRAIN_CONTROLS.json', previous_inputs)
    prep = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
    for name in ['feasibility_cases.json', 'pilot_manifest.json', 'PREPARATION_MANIFEST.json',
                 'module_attachment_verification.json', 'STAGE_I_PAIRED_LORA_PILOT_PLAN.md',
                 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md']:
        include('D049_context/' + name, prep / name)
    for source in sources.values():
        if source.suffix.lower() in ['.pt', '.safetensors']:
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
    (DEST / 'README_FIRST.md').write_text('''# D-051 — Encoder + Decoder LoRA replay investigation

Read `D051/STAGE_I_D051_REPLAY_INVESTIGATION_REPORT.md` and `D051/TRAINING_READINESS_AFTER_D051.md` for the result and limits. This package preserves one authorized B-only execution and its original/replayed tensors. **No paired-pilot training, new generation or corrective experiment was run.**

The frozen D049/D050 scientific configuration and original rtol1e-5/atol1e-6 criterion are unchanged. Three distinct optimizer steps plus replay of step3 from saved step2: four physical update calls/32 microbatches. Detailed evidence is saved before the strict replay verdict; process exit success only means diagnostics finished, not necessarily tolerance PASS.

## Evidence

- `D051/execution/`: original/replayed entry/exit `.pt` files with independent CPU clones of adapters, optimizer/scheduler/scaler/RNG; pre/postclip gradient tensors; checkpoint files; microbatch inputs/RNG/loss/full-logit fingerprints and memory; frozen-backbone integrity and runtime flags. `.pt` files are trusted locally generated evidence with SHA manifests, not external model downloads.
- `D051/analysis/`: per-tensor/module/branch numerical JSON/CSV, update-delta and gradient/Adam comparisons, restored-entry equality, microbatch stage localization. Relative errors use documented near-zero safeguards; failed-element counts use the original torch.isclose direction/dtype.
- `accepted_inputs/`: exact three real train controls with selected source/Teacher target/token IDs. Existing content only; no held-out examples.
- `D051/SOURCE_LEVEL_NONDETERMINISM_HYPOTHESIS.md` and `source_evidence/`: pinned source chain and alternatives. Source inspection is a mechanism candidate, not a profiler-confirmed kernel attribution.
- `D050_historical/` and `D049_context/`: unchanged prior evidence and plan. Earlier approval/status text is historical. The separate D051 approval permits only the completed bounded B replay.
- `code/`, snapshots and execution logs make instrumentation and numeric aggregation inspectable. The original prepared runner/checkpoint helpers and scientific config remain unchanged.

The strict replay gate and researcher-facing PASS/CONDITIONAL/FAIL readiness interpretation are separate. CONDITIONAL must never be reported as a retrospective tolerance PASS or automatic training approval. One replay cannot establish long-run convergence, quality or harmlessness across hardware/seeds. No deterministic policy or tolerance was silently changed.

This is self-contained for evidence/code review, not a standalone training deployment: the full pretrained backbone, original full datasets and historical trained adapters are omitted. D051 temporary adapters/checkpoints must NOT warmstart future pilots. All ZIP member paths are relative; original workspace paths in raw provenance remain historical references. No credentials/API calls.
''', encoding='utf-8')
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text('''# Independent review of D-051 numerical replay evidence

Please review the report and actual saved comparisons before deciding whether the next matched pilot is scientifically defensible.

1. Confirm original D050 scientific settings/data/order/seed/precision/LoRA/checkpoint procedure and rtol1e-5/atol1e-6 are unchanged. Check actual48 modules/589,824 parameters, gradient flow, source/label masks and frozen-backbone integrity.
2. Verify original/replayed step3 entry state equality, then pair the eight microbatches: inputs/RNG, full-logit hashes and losses, accumulated backward gradients, pre/postclip tensors, Adam state and update deltas. Identify the first observed divergence and distinguish checkpoint restoration, forward, backward incl checkpoint recomputation, clipping and optimizer effects.
3. Audit allclose orientation, element denominators, FP64 magnitude aggregation, max/RMS/relative-L2 and near-zero safeguards. Relative percentages near zero and parameter hash differences alone do not establish meaningful divergence. Inspect error relative to the actual update, not only the total weight magnitude.
4. Evaluate the source-level overlapping-attention/index_add hypothesis critically. Do not claim a particular CUDA kernel is proven without runtime attribution; consider alternatives and whether the evidence rules them out.
5. Distinguish strict tolerance PASS/FAIL from researcher-facing CONDITIONAL readiness. Are deviations negligible for this observed update, or unexplained/significant? Do not silently waive the original criterion. One short replay cannot prove long-run harmlessness, summary quality or encoder causality.
6. Recommend whether to approve/revise/hold the512-business/two-epoch paired pilot, and any minimal additional correction requiring separate approval. No pilot is automatically authorized by this bundle. Do not run new GPU experiments, enable deterministic algorithms, change precision/architecture/optimizer/tolerance, modify accepted targets/data, or inspect held-out data.

Return verified facts, unresolved uncertainties, any numerical/reporting errors and a bounded recommendation. Teacher targets and prior AI commentary are not independent factual ground truth; this investigation evaluates numerical replay, not summary quality.
''', encoding='utf-8')
    write(DEST / 'EXPORT_REPORT.json', {'status': 'OFFLINE_EXPORT_ONLY_PASS', 'copied_sources': origins,
        'additional_GPU_calls': 0, 'new_targets_or_predictions': 0, 'held_out_examples': 0,
        'pretrained_and_historical_checkpoint_binaries_included': False,
        'D051_temporary_tensor_evidence_included': True})
    payload = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(DEST.rglob('*')) if p.is_file()]
    write(DEST / 'MANIFEST.json', {'files': payload, 'manifest_excludes_itself': True})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(payload)+1
        for item in payload:
            data = archive.read(NAME + '/' + item['path'])
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP)+'  '+ZIP.name+'\n', encoding='ascii')
    result = {'status': 'ZIP_CRC_SHA256_PASS', 'zip': ZIP.relative_to(ROOT).as_posix(),
              'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size/2**20, 'sha256': sha(ZIP),
              'payload_files': len(payload), 'additional_GPU_calls': 0}
    write(OUT / 'BUNDLE_EXPORT_RECEIPT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
