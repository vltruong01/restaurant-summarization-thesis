# Instructions for Codex on the Vast.ai instance

## Scope and current authorization

This is the Stage H/I handoff, prepared on 2026-10-15. Stage H is accepted. Stage I is implemented and offline-validated; actual GPU feasibility and full training have not been performed. The shipped 768-token decoder limit and additional optimization/decoding defaults are proposals, and all execution approval fields are false.

Use the latest explicit researcher instructions in the current chat when deciding what execution is authorized. Record an approval already given rather than asking for it again. Being given this ZIP or access to an instance alone does not approve the pending scientific settings or full training. When the instruction is only to set up/check this package, complete environment setup and offline preflight, report results and stop.

The configured teacher is not needed for Stage I. Do not request Teacher API keys, call any Teacher API, edit Codex configuration/authentication, regenerate captions/targets, change selected reviews/splits, or inspect held-out test results. No held-out inputs/targets or raw photos are shipped.

## 1. Read these files before execution

All paths are relative to the extracted package root `stage_hi_v2/`; work from that directory.

1. `cloud/stage_hi/START_HERE.md` and `cloud/stage_hi/README.md`.
2. The overview and latest checkpoint in `thesis_sdd_docs/THESIS_PROGRESS.md`.
3. `specs/008_led_dataset_construction.md` and `specs/009_lora_led_training_preparation.md`.
4. D-044/D-045 in `DECISIONS.md` and `configs/stage_i_lora_led_v1.json`.
5. `outputs/stage_h/led_dataset_v1/ACCEPTANCE_REPORT.md`, its manifest and target-limit report, and `outputs/stage_i/preparation_v1/TRAINING_READINESS_REPORT.md`.

The six NPZ files are complete tokenized input-target datasets: 2,000 train and 200 validation businesses for each of RO/U/F. Original Stage F/G text, evidence and absolute historical paths remain referenced as provenance. Those original files are intentionally not copied here. Do not rebuild Stage H or require the full original Windows project to run Stage I. The portable reader checks the shipped arrays, tokenizer, model config and accepted report hashes directly.

## 2. Verify archive and inspect the provided machine

From the upload folder, before extracting into a fresh directory:

```bash
sha256sum -c stage_hi_vast_bundle.zip.sha256
unzip stage_hi_vast_bundle.zip
cd stage_hi_v2
sha256sum -c cloud/stage_hi/transfer.sha256
```

Stop on any checksum failure. Do not edit a checksum to accept unexpected content. If the target extraction directory already exists, inspect it and preserve previous runs; do not overwrite it blindly.

Record a compact inventory in `outputs/stage_i/cloud_operator_log_v1/` (use a new numbered namespace if it already contains an earlier attempt). Avoid dumping environment variables, which may contain credentials.

```bash
mkdir -p outputs/stage_i/cloud_operator_log_v1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3.11 --version
nproc
free -h
df -h .
```

Record OS, Python executable/version, actual GPU/VRAM and driver, host RAM/CPU, disk availability and whether the working directory/checkpoints are on durable storage. The first proposed test uses one 24GB RTX3090/4090-class GPU, at least 32GB host RAM, four vCPUs and 50GB free persistent disk. The runner requires at least 23 GiB reported device memory. A smaller device or unsuitable runtime is a blocker for this unchanged test profile; do not silently reduce sequence lengths or change precision. Multiple GPUs do not pool memory in this single-GPU runner.

## 3. Build the pinned environment and run offline preflight

Use a fresh virtual environment. If Python 3.11 or the compatible driver is missing, report the precise environment issue; do not substitute an unreviewed package stack.

```bash
python3.11 -m venv .venv-led
source .venv-led/bin/activate
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/stage_hi_training.txt
python -m pip check
python -m pip freeze > outputs/stage_i/cloud_operator_log_v1/pip_freeze.txt
```

Run from Bash with pipeline failure propagation; preserve stdout/stderr. Choose a fresh preflight report namespace on a repeated check.

```bash
set -euo pipefail
python scripts/run_stage_i_lora_led.py --mode offline-preflight --report-directory outputs/stage_i/cloud_preflight_v1 2>&1 | tee outputs/stage_i/cloud_operator_log_v1/offline_preflight.log
```

Expected report: `outputs/stage_i/cloud_preflight_v1/offline_preflight_report.json`, status `STAGE_I_OFFLINE_PREFLIGHT_PASS_GPU_UNTESTED`. Check 36 fast/slow tokenizer ID comparisons, all six datasets, preserved labels, BOS-only global attention, padding masks and decoder-start alignment. Meta-device architecture checking attaches LoRA without loading pretrained weights or doing model forward/backward.

Literal `q_proj`/`v_proj`, rank 8 / alpha 16, matches 24 decoder attention projections, with 294,912 trainable adapter parameters. Do not expand to encoder `query`/`value` projections. Logical source/review/caption limits remain 5,000/4,000/750 and K=5; source tensors may pad to 5,120 because the pinned attention window is 1,024. The proposed label limit 768 preserves all observed labels (max 656 including BOS/EOS); proposed generation `max_length=768` also counts the decoder-start token.

**Stop here when settings/GPU execution have not been approved.** Return the inventory, checksum outcome, preflight report, exact proposed config and remaining approvals. An offline PASS does not establish GPU compatibility, VRAM sufficiency, factual target quality or successful generation.

## 4. Bounded GPU feasibility — only after explicit approval

Preserve `configs/stage_i_lora_led_v1.json`. After the researcher approves the pending settings and bounded GPU probe, make `configs/stage_i_lora_led_feasibility_approved_v1.json` as a versioned runtime copy. Record the approval in the operator log and progress log. Change only the approval/status metadata needed for that scope:

- `approval.length_and_training_settings = true`.
- `approval.gpu_feasibility = true`.
- `approval.full_training = false`.

Keep the scientific values and reviewed Stage H manifest checksum identical. Do not bypass `execution_gate` or alter code to force a pass. Verify the original transfer checksums before creating approved runtime copies; new approval metadata is documented separately from the immutable transferred baseline.

```bash
set -euo pipefail
python scripts/run_stage_i_lora_led.py --config configs/stage_i_lora_led_feasibility_approved_v1.json --mode feasibility --execute 2>&1 | tee outputs/stage_i/cloud_operator_log_v1/feasibility.log
```

This downloads the pinned LED backbone and tests the three conditions on deterministic train controls. It does not start the full 2,000-business training runs. Inspect:

- `outputs/stage_i/runs_v1/feasibility/sample_manifest.json`.
- `outputs/stage_i/runs_v1/feasibility/report.json`, or `failure_report.json` if execution failed.
- Per-condition artifacts in that feasibility directory.

Report finite loss/gradients, complete-token masks/global attention, 5,120/768 stress shapes, peak allocated/reserved VRAM and the 15% headroom check, loss trajectory, real generation sections/EOS/cap warnings, checkpoint state restoration and the resumed optimizer step. Compatibility status `GPU_FEASIBILITY_PASS_PENDING_RESEARCHER_REVIEW` still requires review of output format, warnings and loss behavior. It is not full-training authorization.

On OOM, non-finite loss, missing checkpoint state, format/cap warning or headroom failure, preserve all artifacts and report the finding. Do not silently change budgets, K, labels, precision, LoRA scope, beam count or thresholds. A failed feasibility namespace is preserved; a reviewed repeat must use a fresh output namespace. No automatic full-training launch follows feasibility.

Stress microstep timing is not a full-epoch speed measurement. Report it as such; use measured real-data training/validation and final generation timing before making a firm rental-time forecast.

## 5. Full training and recovery — later, separately authorized

Only after the researcher reviews successful feasibility and authorizes full training, create `configs/stage_i_lora_led_training_approved_v1.json` from the approved feasibility config. Retain identical scientific values, enable `approval.full_training`, and bind:

- `approved_feasibility_report`: the actual relative report path.
- `approved_feasibility_report_sha256`: SHA-256 of that exact file.

The runner checks the report status and scientific settings signature. The three conditions must share optimizer, schedule, effective batch size, limits, decoding and checkpoint-selection rules. Run serially on the single GPU; do not launch competing training workers. Use an already available durable terminal session for long runs and preserve logs; reconnecting to SSH is not a reason to launch a second process.

```bash
set -euo pipefail
python scripts/run_stage_i_lora_led.py --config configs/stage_i_lora_led_training_approved_v1.json --mode train --condition reviews_only --execute 2>&1 | tee outputs/stage_i/cloud_operator_log_v1/train_reviews_only.log
python scripts/run_stage_i_lora_led.py --config configs/stage_i_lora_led_training_approved_v1.json --mode train --condition unfiltered --execute 2>&1 | tee outputs/stage_i/cloud_operator_log_v1/train_unfiltered.log
python scripts/run_stage_i_lora_led.py --config configs/stage_i_lora_led_training_approved_v1.json --mode train --condition filtered --execute 2>&1 | tee outputs/stage_i/cloud_operator_log_v1/train_filtered.log
```

For an interrupted condition, reuse the same approved config and append `--resume-from-checkpoint` with its actual existing `outputs/stage_i/runs_v1/<condition>/checkpoint-<step>` directory. Choose a checkpoint containing `trainer_state.json` and optimizer/scheduler/RNG state; do not invent a step number, resume across conditions, reset the optimizer, or start a replacement run over an existing run directory. Keep data/settings/code bindings unchanged. If state is incomplete, report it before proposing recovery.

Each completed condition saves `completion_report.json`, `best_adapter/` and `validation_predictions.jsonl` under its output directory, alongside Trainer/checkpoint artifacts. Check that validation-loss selection was used, all 200 validation predictions are present, and generation/length/format warnings are reported without rewriting predictions. Do not run held-out test evaluation in this handoff.

## 6. Return artifacts and update the shared progress log

After the authorized scope ends, update `thesis_sdd_docs/THESIS_PROGRESS.md` with actual commands/config paths, environment, data/code/config hashes, runtime, memory, pass/fail checks, warnings, report/checkpoint paths and next action. Preserve historical entries and distinguish measured outcomes from plans. Record any approved scientific amendment in `DECISIONS.md` and the active spec before implementation; ordinary environment or operational notes belong in the progress/operator logs.

Return a concise outcome plus these artifacts as applicable: cloud inventory/pins, preflight, feasibility/failure report and sample manifest, approved runtime configs/approval note, complete logs, condition completion reports, best adapters, latest resumable checkpoints/Trainer state and validation predictions. Sync results to researcher-controlled durable storage before releasing the instance. Do not include credentials or overwrite/delete local Stage F/G provenance.
