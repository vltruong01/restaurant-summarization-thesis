"""Offline portable D-052 review export; no Torch/model/GPU/API imports.

Includes both small adapter checkpoints and their preserved training states,
not the pretrained backbone or full accepted training NPZ arrays.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d052_paired_pilot_v1'
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
NAME = 'stage_i_d052_paired_training_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')
ARMS = ('A_decoder_only', 'B_encoder_decoder')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    require(not DEST.exists() and not ZIP.exists(), 'Preserve previous exports; version revisions separately')
    validation = read(OUT / 'D052_TRAINING_VALIDATION.json')
    require(validation['status'] == 'PAIRED_TRAINING_AND_OFFLINE_AUDIT_PASS_NO_GENERATION', 'Both training audits required')
    require(read(OUT / 'POST_EXECUTION_INTEGRITY.json')['status'] == 'PASS', 'Preservation audit required')
    require((OUT / 'STAGE_I_D052_PAIRED_TRAINING_REPORT.md').is_file(), 'Report missing')
    reference_readiness = read(OUT / 'SOURCE_REFERENCE_READINESS.json')
    require(validation['source_reference_readiness'] == reference_readiness, 'Reference readiness report differs')
    require(reference_readiness['candidate_units'] == 226 and
            reference_readiness['human_confirmed_candidates_in_saved_initial_artifacts'] == 0 and
            reference_readiness['human_confirmed_final_units_in_saved_initial_artifacts'] == 0 and
            reference_readiness['browser_localStorage_or_unsubmitted_downloads_inspected'] is False,
            'Unverified saved worksheet status differs; inspect explicit human evidence separately')
    for relative, digest in reference_readiness['evidence_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Saved source-reference readiness evidence changed: ' + relative)
    config = read(ROOT / 'configs/stage_i_paired_lora_pilot_v1.json')
    sources = {}

    def include(relative, source):
        source = Path(source)
        require(source.is_file(), 'Missing required review evidence: ' + str(source))
        require(relative not in sources and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Unsafe/duplicate export path: ' + relative)
        require(source.stat().st_size < 100 * 2**20, 'Unexpectedly large review file; inspect before export: ' + str(source))
        sources[relative] = source

    # Current D052 evidence, including the source-only annotation interface/status
    # when present. Earlier source snapshots are labeled historical in README.
    for source in sorted(OUT.rglob('*')):
        if source.is_file() and source.name not in ('BUNDLE_EXPORT_RECEIPT.json',):
            include('D052/' + source.relative_to(OUT).as_posix(), source)
    for arm in ARMS:
        folder = ROOT / config['run_root'] / arm / 'train'
        require(read(folder / 'completion.json')['generation_calls'] == 0, 'Unauthorized generation in training completion')
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                include('paired_training/' + arm + '/' + source.relative_to(folder).as_posix(), source)

    for name in ['STAGE_I_PILOT_EVALUATION_PROTOCOL.md', 'STAGE_I_PAIRED_LORA_PILOT_PLAN.md',
                 'pilot_manifest.json', 'epoch_orders.json', 'evaluation_inputs.jsonl',
                 'reference_unit_candidates.jsonl', 'reference_freeze_template.json', 'business_annotation_template.json',
                 'initialization_hashes.json', 'module_attachment_verification.json', 'PREPARATION_MANIFEST.json',
                 'expected_generation_config.json', 'OFFLINE_CHECKS.json', 'READINESS_REPORT.md']:
        include('D049_frozen_context/' + name, PREP / name)

    for name in ['STAGE_I_D051_REPLAY_INVESTIGATION_REPORT.md', 'TRAINING_READINESS_AFTER_D051.md',
                 'SOURCE_LEVEL_NONDETERMINISM_HYPOTHESIS.md', 'analysis/NUMERIC_ANALYSIS_SUMMARY.json',
                 'execution/strict_replay_result.json']:
        include('D051_historical/' + name, ROOT / 'outputs/stage_i/d051_replay_investigation_v1' / name)
    for source in sorted((ROOT / 'outputs/stage_i/d051_replay_investigation_v1/source_evidence').rglob('*')):
        if source.is_file():
            include('D051_historical/source_evidence/' + source.name, source)

    required_code = ['run_stage_i_d052_paired_pilot_v1.py', 'run_stage_i_d052_paired_pilot_v2.py',
                     'stage_i_d052_authorization.py', 'test_stage_i_d052_authorization.py',
                     'build_stage_i_d052_source_reference_review_v1.py',
                     'report_stage_i_d052_paired_pilot_v1.py', 'build_stage_i_d052_review_bundle_v1.py',
                     'run_stage_i_paired_lora_pilot_v1.py', 'stage_i_paired_pilot.py',
                     'prepare_stage_i_paired_lora_pilot_v1.py', 'test_stage_i_paired_lora_pilot_v1.py',
                     'stage_hi_led_data.py', 'stage_i_decoding_diagnostics.py',
                     'run_stage_i_d050_feasibility_v1.py']
    for name in required_code:
        include('code/' + name, ROOT / 'scripts' / name)
    # Include any accompanying offline D052 authorization/tests/reference UI
    # scripts without requiring a particular interface filename in advance.
    for source in sorted((ROOT / 'scripts').glob('*d052*.py')):
        relative = 'code/' + source.name
        if relative not in sources:
            include(relative, source)
    for source in sorted((ROOT / 'specs').glob('*d052*.md')):
        include('current_documentation/' + source.name, source)
    for relative in ['DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md',
                     'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                     'configs/stage_i_paired_lora_pilot_v1.json']:
        include('current_documentation/' + Path(relative).name, ROOT / relative)

    worksheet = ROOT / 'outputs/stage_i/d052_source_reference_review_v1'
    require((worksheet / 'SOURCE_REFERENCE_REVIEW.html').is_file() and
            (worksheet / 'EXPORT_REPORT.json').is_file(), 'Build the source-only reference worksheet before final export')
    worksheet_manifest = read(worksheet / 'SHA256_MANIFEST.json')
    for relative, digest in worksheet_manifest['files'].items():
        require(sha(worksheet / relative) == digest, 'Source-reference worksheet payload changed: ' + relative)
    for source in sorted(worksheet.rglob('*')):
        if source.is_file():
            include('source_reference_worksheet/' + source.relative_to(worksheet).as_posix(), source)

    binary_suffixes = {'.pt', '.pth', '.safetensors', '.bin', '.npz', '.jpg', '.jpeg', '.png', '.webp'}
    for source in sources.values():
        if source.suffix.lower() in binary_suffixes:
            continue
        raw = source.read_bytes()
        encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        text = raw.decode(encoding)
        require(not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', text), 'Potential API credential in export: ' + source.name)

    DEST.mkdir()
    origins = []
    for relative, source in sources.items():
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, destination)
        require(sha(destination) == digest, 'Copied source changed: ' + str(source))
        origins.append({'path': relative, 'source_workspace_path': source.relative_to(ROOT).as_posix(),
                        'bytes': destination.stat().st_size, 'sha256': digest})

    (DEST / 'README_FIRST.md').write_text('''# D-052 - paired Filtered LED + LoRA training evidence

Start with `D052/STAGE_I_D052_PAIRED_TRAINING_REPORT.md` and `D052/D052_TRAINING_VALIDATION.json`. Both fresh models were trained for the separately approved two-epoch exploratory comparison. **No new student summaries, quality evaluations, Teacher calls or held-out evaluations were performed.**

D-051 strict replay remains FAIL at rtol1e-5/atol1e-6. `D052/D052_CONDITIONAL_REPRODUCIBILITY_DECISION.md` records explicit researcher acceptance of the bounded encoder-backward numerical limitation for this experiment only; it is not a retroactive PASS or a silently relaxed criterion.

## Contents and interpretation

- `D052/`: current report, raw arm audits/events, actual TrainingArguments/runtime, per-microbatch loss/order, optimizer/LR/scaler/memory, original/final parameter hashes, frozen-backbone and checkpoint audits, authorization and preservation evidence. `snapshots/` is historical pre-execution provenance; its older status text must not override current final reports.
- `D052/interrupted_attempt01/`: the discarded early A attempt, stopped at observed progress27/128 before an epoch checkpoint because a stale Transformers lazy module prevented supplemental observer attachment. No weights were reused. Both final matched runs are fresh v2 executions; the discarded prefix is excluded from all matched loss tables and is not an epoch extension. The versioned correction and original raw traces are preserved.
- `paired_training/{arm}/`: completion/Trainer state, gradient logs, both epoch checkpoints and best adapters. Adapter/model-config, optimizer, scheduler, scaler and RNG files are retained. These are trusted locally generated PyTorch files with hashes; inspect code and hashes before deserializing. D052 saves state but does not claim a new strict or fresh-process replay PASS.
- `D049_frozen_context/`: exact512 train IDs, two epoch permutations,200 validation loss IDs and frozen20-random/4-targeted evaluation roles, complete24-case sources/Teacher targets/evidence,226 **unverified** reference-unit candidates and blank human-confirmation templates. Teacher targets and candidate units are not independent factual ground truth.
- `source_reference_worksheet/SOURCE_REFERENCE_REVIEW.html`: offline source-only human verification UI with actual supplied-caption photos and review evidence. Human confirmation and separate generation approval are still required before producing or inspecting new student predictions. Existing historical outputs, if present in frozen context, are not new D052 outputs. Its initial unverified state is not a claim that a human has finished annotation.
- `D052/SOURCE_REFERENCE_READINESS.json`: exact saved-artifact readiness snapshot:226 unchanged candidates,0 human-confirmed candidates/final units/business-completeness decisions in the saved initial worksheet, and explicit final-reference locations checked. Browser localStorage and unsubmitted downloads were not inspected. They must never be invented as human annotations; any separately supplied final reference requires offline evidence/schema validation before predictions.
- `D051_historical/`: earlier strict FAIL, numeric scale, source-level mechanism hypothesis and licensed source excerpts. The exact CUDA kernel remains unproven; one-step drift cannot establish long-run harmlessness.
- `code/` and `current_documentation/`: implemented training/authorization/audit/export code and final method status. The package contains inspectable code; running it is not authorized by this export.

## Research limits

The paired comparison changes both adapter location and parameter count. Two epochs/512 examples/one seed are a screening budget. Lower teacher-forced validation loss is not evidence of improved informative, coherent or evidence-grounded generated summaries; no such quality comparison has yet been run. No automatic winner, longer training or new decoding experiment is authorized.

Read per-epoch training-loss definitions carefully: the report reconstructs mean microbatch losses from saved Trainer returns after accumulation scaling, rather than relying on rounded logging means. Source/target lengths, data order, optimization and decoder initialization remain matched. Checkpoint selection is independently minimum validation loss for each arm.

In the gradient table, a slash such as `0/24` gives the separate counts of nonzero `lora_A` and `lora_B` gradient tensors; it is not a fraction of failed gradients. The architecture arm names A/B and adapter matrices `lora_A`/`lora_B` are distinct concepts.

## Portability and integrity

`MANIFEST.json` and `SHA256SUMS.txt` cover all other bundle payloads, except their own integrity-index files. `EXPORT_REPORT.json` maps copied files to original workspace paths. ZIP member paths are relative; absolute paths in raw logs/configuration remain historical provenance. The external `.zip.sha256` verifies the archive itself.

This is self-contained for code/evidence review, not a standalone training deployment: pretrained LED binaries and full accepted NPZ datasets are intentionally omitted. Included adapter checkpoints are the actual new D052 pilots, not historical full-F or feasibility weights. No Codex credentials or Teacher API keys are included.

**Stop boundary:** review these completed training results, verify source-reference units, then obtain separate approval before generation, extra training or any other experiment.
''', encoding='utf-8')
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text('''# Independent review of D-052 paired pilot training

Please assess the existing evidence without generating or changing model targets, labels, data or checkpoints.

1. Verify matched architecture A (24 projections/294,912 parameters) and B (48/589,824), correct encoder local/global and decoder self/cross Q/V attachment, independent pretrained initializations and exact shared decoder initialization.
2. Verify the same fixed512 F training businesses, exact two-epoch ordering, all200 validation businesses,128 optimizer updates, batch1/accumulation8, frozen seed/precision/checkpointing/AdamW/LR schedule and complete-source/label mask audits. Check loss-scaling definitions and per-epoch mean calculations against raw returns.
3. Check finite intended gradients, expected initial LoRA A zeros, absent frozen gradients, unchanged backbone and accepted data, no AMP skips, optimizer/scheduler/scaler records, actual GPU memory, and both complete epoch checkpoints. Validate hashes and selected minimum-loss best adapters.
   Also verify the discarded27/128 operational attempt is disclosed separately: it produced no selected checkpoint and did not warmstart either corrected v2 run. Confirm v2 observer attachment/MRO and pre-training guard; do not count discarded losses as a matched arm or training extension.
4. Distinguish conditional researcher acceptance of D051 numerical backward variability from the original strict replay gate, which remains FAIL. Do not infer a new replay PASS from saved state files or successful pilot training. Evaluate whether any new numerical warning remains unresolved.
5. Interpret validation-loss differences cautiously: there are no new summaries or independent quality labels yet. Do not claim an encoder quality benefit, causal location effect or generalization to RO/U/test. Increased parameter count, one seed and a short screening budget remain limitations.
6. Review source-reference readiness independently of student outputs. The226 candidate units must never be treated as human ground truth; explicit researcher verification and source completeness confirmation are still required. Preserve the separate20-random/4-targeted reporting roles and predefined repetition/informativeness/grounding/structure rubric.
7. Return factual audit findings, any inconsistencies, limitations and the next action requiring researcher approval. Do not run generation, extra epochs, GPU experiments, Teacher calls or held-out evaluation. A proposed next evaluation requires separate generation approval after source-reference freeze.
''', encoding='utf-8')
    write_new(DEST / 'EXPORT_REPORT.json', {
        'status': 'OFFLINE_EXPORT_ONLY_PASS', 'copied_sources': origins,
        'source_files_copied_unchanged': True, 'new_GPU_model_or_API_calls': 0,
        'new_student_predictions': 0, 'held_out_examples': 0,
        'new_pilot_adapter_and_training_state_files_included': True,
        'pretrained_backbone_and_full_training_npz_included': False,
        'D051_strict_replay': 'FAIL_UNCHANGED_CONDITIONALLY_ACCEPTED_FOR_D052_ONLY',
    })
    payload = [{'path': path.relative_to(DEST).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(DEST.rglob('*')) if path.is_file()]
    write_new(DEST / 'MANIFEST.json', {'files': payload, 'excludes': ['MANIFEST.json', 'SHA256SUMS.txt']})
    (DEST / 'SHA256SUMS.txt').write_text(''.join(row['sha256'] + '  ' + row['path'] + '\n' for row in payload), encoding='utf-8')
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        require(len(archive.namelist()) == len(payload) + 2, 'Unexpected ZIP member count')
        require(all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in archive.namelist()), 'Unsafe ZIP path')
        for item in payload:
            data = archive.read(NAME + '/' + item['path'])
            require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'ZIP payload hash failed')
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    result = {'status': 'ZIP_CRC_SHA256_PASS', 'zip': ZIP.relative_to(ROOT).as_posix(),
              'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size / 2**20, 'sha256': sha(ZIP),
              'payload_files': len(payload), 'new_GPU_model_or_API_calls': 0, 'new_student_predictions': 0}
    write_new(OUT / 'BUNDLE_EXPORT_RECEIPT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
