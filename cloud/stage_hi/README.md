# Stage H/I portable handoff — preparation only

Codex execution guide: read `CODEX_HANDOFF.md` in this directory. Handoff v2 preserves the runtime/data/configuration and adds detailed execution, logging, failure/recovery and reporting instructions.

No GPU was rented, no data uploaded, no pretrained LED weights loaded and no training performed during preparation. Proposed settings need review; do not execute paid/GPU steps until explicitly approved. Current config approval fields are false.

First evaluate **one24GB RTX3090/4090-class card**, physical batch1, accumulation8. Use >=32GB host RAM, >=4vCPU, >=50GB free persistent disk, Python3.11 and a CUDA11.8-compatible NVIDIA driver.24GB sufficiency is untested. Two24GB GPUs do not automatically provide a48GB model-memory pool; this runner is single-GPU.

Transfer only the reviewed packed train/validation data, frozen tokenizer/model config, code/config and reports listed in `transfer_manifest.json` (created after offline acceptance). No original photos, caption model caches, teacher raw requests, API credentials, Codex configuration or held-out data are needed. Original full evidence stays in durable local Stage F/G artifacts. The manifest preserves references to Stage G originals; the GPU reader verifies packed and tokenizer hashes without requiring a second copy of all raw evidence.

Install the pinned dependencies in a new environment **after rental is approved**:

```bash
python3.11 -m venv .venv-led
source .venv-led/bin/activate
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/stage_hi_training.txt
python -m pip check
```

The cloud operator should first verify every transfer checksum, then run the offline preflight in a **new report namespace**, preserving the transferred local report. Record Python, packages, driver, CUDA, device and data hashes. No historical report is overwritten:

```bash
sha256sum -c cloud/stage_hi/transfer.sha256
python scripts/run_stage_i_lora_led.py --mode offline-preflight --report-directory outputs/stage_i/cloud_preflight_v1
```

After explicit length/settings and GPU-probe approval, run:

```bash
python scripts/run_stage_i_lora_led.py --mode feasibility --execute
```

This downloads only the pinned LED backbone then runs all three conditions on the same preselected train controls. It checks real complete inputs/labels and padded5120/768 shapes, finite forward/backward, actual LoRA gradients, AdamW updates, saved/restored adapters/optimizer/scaler/RNG, a resumed update, generation sections/cap/EOS diagnostics and peak memory. Padding is masked, with valid-token lengths separately reported; no synthetic teacher target is introduced. No automatic source shortening, quantization, module expansion, precision switch or OOM fallback.

Inspect `outputs/stage_i/runs_v1/feasibility/report.json` plus any failure report. Review actual headroom under training and beam4 generation (target >=15%), overfit loss/format and checkpoint recovery. Warmed stress times are conservative shape timings, not a measured whole training epoch. Obtain real-data epoch/validation measurements before making a firm rental-time estimate.

Only after successful reviewed GPU feasibility **and separate full-training authorization**, bind the approved feasibility report/hash and approved settings in the config. Train RO/U/F separately with the same reviewed config:

```bash
python scripts/run_stage_i_lora_led.py --mode train --condition reviews_only --execute
python scripts/run_stage_i_lora_led.py --mode train --condition unfiltered --execute
python scripts/run_stage_i_lora_led.py --mode train --condition filtered --execute
```

Resume uses `--resume-from-checkpoint outputs/stage_i/runs_v1/<condition>/checkpoint-<step>` and requires matching data/settings/code; it restores optimizer/scheduler/scaler/RNG via Trainer. Epoch checkpoints bound restart loss to at most the unfinished epoch. The best validation-loss adapter and latest resumable checkpoint are retained. Keep durable storage and sync checkpoints/logs/configs before terminating an instance. No test references/results are used in this workflow.

**Illustrative time calculation, not a speed forecast:** with batch1 and five epochs, each condition has10,000 train microbatches and1,000 validation examples; all three have30,000/3,000. At measured2s/train microbatch and1s/validation example this is17.5hours plus checkpoint/generation/transfer overhead; at1s/0.5s it is8.75hours. Actual GPU measurements and early stopping determine the duration. Gradient accumulation reduces optimizer updates (proposed250/epoch), not the number of forward/backward microbatches.

Primary references: [LED and global attention](https://huggingface.co/docs/transformers/model_doc/led), [pinned backbone configuration](https://huggingface.co/allenai/led-base-16384/blob/38335783885b338d93791936c54bb4be46bebed9/config.json), [LoRA](https://huggingface.co/docs/peft/package_reference/lora), [generation accounting](https://huggingface.co/docs/transformers/main_classes/text_generation), [RTX3090 memory specification](https://www.nvidia.com/en-eu/geforce/graphics-cards/30-series/rtx-3090/). Local code behavior is pinned to Transformers4.51.3/PEFT0.15.2, not assumed from the latest docs.
