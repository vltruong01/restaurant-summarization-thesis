# Stage Specification — LoRA-LED preparation and later training

**Spec ID:** 009  
**Stage:** I  
**Status:** Original preparation preserved; externally authorized cloud training now verified complete, output quality pending researcher review  
**Last updated:** 2026-10-15  
**Execution environment:** Local offline preflight and read-only return audit; imported RTX3090 cloud execution

**Post-training diagnostic preparation, D-046 (2026-10-15):** See [spec009b](009b_validation_decoding_diagnostic.md). Existing checkpoints are preserved. The researcher requested20 random validation businesses plus4 known-warning cases, saved beam4 baseline and proposed greedy A, with B separately proposed after A review. Offline data/configuration/evaluation preparation PASS; no additional GPU inference, paid action, teacher call, held-out access or retraining is authorized/run. Production decoding is unchanged; explicit diagnostic/cost approval is the next gate.

**Execution evidence addendum, 2026-10-15:** The original preparation scope and proposed controls below remain historical provenance. The returned cloud archive contains researcher authorization, successful LED forward/backward/checkpoint feasibility, an approved runtime configuration using the proposed768 cap and common scientific settings, and three completed five-epoch/1,250-step runs. The v2 runner adds only `label_names=['labels']` for validation-loss reporting under PEFT. Data/checkpoint integrity PASS; substantial output repetition and a zero-caption visual-grounding failure require review. See `../outputs/stage_i/cloud_return_20261015_v1/audit_v1/AUDIT_REPORT.md` and the latest progress checkpoint. This addendum changes no model, target, decoding or optimization parameter and authorizes no additional execution. Local draft approval flags and immutable cloud artifacts are preserved.

## 1. Objective and scope

Prepare one reproducible runner for RO/U/F with equivalent optimization/decoding, full labels and validation-loss checkpoint selection. Prepare, but do not execute, GPU feasibility and overfit procedures. No teacher API, test tuning, rental/upload or full training in this task.

## 2. Inputs and outputs

Inputs: accepted spec008 packed dataset manifest/hashes, pinned tokenizer/model config, `configs/stage_i_lora_led_v1.json`. Outputs: offline readiness/architecture/collator report, executable training/feasibility runner, dependency pins, and Vast.ai handoff instructions. Future execution writes per-condition configuration/environment, adapter checkpoints, optimizer/scheduler/scaler/RNG state, loss/time/memory/generation logs and a completion report.

## 3. Planned controls inherited from the master plan

- Backbone `allenai/led-base-16384`, same pinned revision as the audited tokenizer.
- Literal LoRA module names `q_proj`, `v_proj`, r8, alpha16, AdamW, lr2e-4, weight decay0.01, physical batch1–2; seed42.
- Important architecture clarification: LED encoder projections are `query/value` and `query_global/value_global`. Literal q_proj/v_proj match **decoder self-attention and decoder cross-attention only** (24 linear modules in the six-layer base decoder). Do not silently add encoder targets. Report matched names and trainable parameter count.
- Same logical source limit5000, review4000, caption750 and K5; original attention_window1024 per encoder layer. GPU source tensors may pad to5120: padding does not change logical content budgets.
- Right-padding; global attention on the first real BOS token only; padding never has local/global attention. Labels contain full BOS/text/EOS and only added padding is -100. Do not add marker vocabulary tokens or resize embeddings.
- Gradient checkpointing with non-reentrant implementation, use_cache=False during training; cache enabled only for generation. No quantization or automatic module/budget changes on OOM.
- Select the best checkpoint by validation loss, not generated test performance. Separate training early stopping from beam-search early stopping.

## 4. Explicit proposals requiring review

The draft config proposes: labels<=768 including BOS/EOS; generation max_length768 including initial decoder-start (=at most767 new tokens); physical batch1/accumulation8/effective8; epochs5; linear scheduler/warmup ratio0.05; max_grad_norm1; LoRA dropout0/bias none; FP16 autocast with FP32 base weights and loss scaling; validation/save each epoch; early-stopping patience2; deterministic beam4/do_sampleFalse/length_penalty1/no_repeat_ngram0. These defaults are **prepared proposals, not frozen research decisions**. Limits and all optimization/decoding settings must be approved once, equivalently for RO/U/F, before executing the production runner. Batch2 can be assessed later with accumulation4, preserving effective batch8, if approved; it is not silently selected.

768 preserves all known labels (max656), even after conservatively allowing one extra decoder-start token for generation; both are below the model's1024 decoder-position capacity. It does not guarantee natural EOS or correct section format in generated summaries. Log cap-reaching sequences and validate section structure without rewriting predictions.

## 5. Offline acceptance versus GPU acceptance

Offline: complete data integrity, exact tokenizer/shifted-label/padding checks, meta-device adapter attachment/count, runner CLI/approval gates, dependency inventory and unit tests. Meta inspection does not load pretrained weights or perform model forward/backward.

GPU A2 after approval: one24GB device first, actual frozen backbone/revision, physical batch1, real long/zero-caption examples for all conditions, source tensor5120 and padded decoder768 stress; report valid-token counts separately from padded shapes. Finite forward loss, nonzero finite adapter gradients, optimizer step, checkpoint round-trip including optimizer/scaler/RNG, small fixed-train-sample overfit loss and generation format, measured peak allocated/reserved/free VRAM and step times. Test planned beam4 generation separately. An initial pretrained output lacking section tags is a baseline diagnostic; overfit behavior is assessed separately. Retain failures/OOM reports; do not fall back automatically to shorter inputs, QLoRA or different modules.

Training requires a successful reviewed GPU report as well as explicit full-training approval. Runtime manifest/config/data hashes bind checkpoints; reject incompatible resume. Trainer saves optimizer/scheduler/RNG, plus adapters, and resume uses that state rather than reinitializing an optimizer. Keep the best-validation checkpoint and latest resumable checkpoint; sync durable files before releasing a rental.

## 6. Feasibility/time plan

Propose one RTX3090/4090-class24GB GPU, >=32GB host RAM, >=4vCPU, >=50GB free persistent disk, compatible CUDA11.8 driver/runtime and Python3.11. No promise that24GB fits. A second GPU does not automatically combine memory. Target >=15% VRAM headroom under training **and** beam-generation tests; actual measurements decide acceptance. Time estimates must come from warmed-up real-data microsteps and a full validation timing; report projections for2000/200 examples, all three conditions and proposed5 epochs, not an invented throughput guarantee.

## 7. Stops

Config remains unapproved by default; GPU actions fail closed before weights are loaded. Stop on data/hash mismatch, loss/gradient non-finiteness, missing LoRA matches, OOM, bad recovery or incompatible settings. Structural PASS is not human factual certification. No test references/results or teacher test targets are used. Do not train until the researcher reviews offline deliverables and explicitly authorizes the next execution scope.

## 8. Actual offline result — 2026-10-14

Offline preflight PASS: 36 fresh HF-fast/slow token-ID round-trip checks on selected real rows from all six datasets; full collator content/mask/shift checks; meta-only adapter attachment confirms24 decoder projections,294,912 adapter parameters (0.1819% of augmented model). All seven injected overflow/corruption/attention/authorization/format tests PASS. Model weights were not downloaded or materialized; no forward/backward, GPU memory benchmark, generation, checkpoint recovery or optimizer/training step ran. The executable runner uses explicit LED finite-loss checks and avoids passing the unsupported num_items_in_batch kwarg through PEFT to LED. Actual runtime behavior awaits the approved GPU feasibility procedure. Reports under `outputs/stage_i/preparation_v1/`; draft scientific settings remain unapproved.
