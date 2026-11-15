# D-059 technical restart v2 — offline repair only

Researcher approved a narrowly scoped missing-json-import repair and offline preparation. **Renewed GPU execution is NOT authorized.** D059's prior3-update gate is historical PASS; its original F run failed before optimizer update1. This is technical incompleteness, not a quality NO-GO. Preserve D051 strict replay FAIL.

## Exact changes

1. New `scripts/run_stage_i_d059_v2.py` adds `json` to the existing import line. Its only other difference is `from stage_i_d059_v2 import *`, routing to new bindings. All training/evaluation statements, nested Trainer/Record methods, loss, optimizer, schedule, gradients, checkpoints and decoding remain byte-for-byte identical.
2. New `scripts/stage_i_d059_v2.py` is the old helper with CONFIG/PREP/RUNNER paths changed, added exact v1→v2 scientific-config equivalence checks, and an explicit fixed original clock/deadline check. Adapter attachment, initialization, gradients, masks, 15% memory criterion, serialization and numerical behavior are unchanged. Old files are not edited.
3. New `configs/stage_i_d059_expanded_lora_v2.json` differs only in `run_root=outputs/stage_i/d059_expanded_filtered_v2` and transparent `technical_restart` provenance. All scientific keys are identical to v1, including114modules/1,843,200parameters, F2000/200, seed42/orders, fiveepochs/1250updates/warmup63, D058 AdamW/LR/AMP/batch settings,5epoch saves/no best-loss reload and beam4/max_length768. No quantization, retuning, output repair or fallback.

Original execution clock: **2026-10-18T15:10:48.941817+00:00**. Final deadline: **2026-10-25T15:10:48.941817+00:00**. Restart authorization receipt time must be recorded separately; it cannot replace this clock. Expiry retains the existing stop-loss.

## Regression and integrity evidence

Offline tests execute the exact Record class extracted by AST from each runner; they do not substitute a copied callback implementation. They import the real runner's top-level namespace without calling main, and inject test-only boundary doubles for TrainerCallback, gradient computation, scheduler, memory and trainer state. v1's actual callback must reproduce NameError. v2's actual gradient callback must write parseable JSON with expected step/gradient values. Also invoke on_train_begin, on_optimizer_step, on_step_end and on_save, checking schedule/deadline, AMP/optimizer-step guards, progress JSON and complete checkpoint-file/hash manifests plus missing-file rejection. No Torch/Transformers/PEFT model or CUDA runtime is needed. Synthetic checkpoint files exist only in temporary test directories and are explicitly marked fixtures.

Retain the original29 offline contracts against v2 paths and add exact-diff, no-clock-reset, stale-authorization/stale-feasibility and historical-evidence checks. Existing meta inventory, CPU serialization,2200 token-alignment results, fixed24 case manifest and five training orders are reused by exact hash; no new model construction or source annotation is required for an import repair. Preserve all195 captured original v1 files plus original runtime/data bindings. The old incident/authorization/traceback/runner and historical gate remain immutable.

## Proposed renewed hash-bound execution scope — future approval required

1. Rerun actual-environment preflight under the new config/preflight/runner SHA256 triplet.
2. Exactly one new gate: max3optimizer updates/24microbatches, no generation/replay; same real training examples, padded5120/768,114module/1,843,200parameter and gradient/frozen-base/AMP/mask/state-restoration checks; at least15% measured headroom. Do not reuse the v1 PASS or loosen its hash checks.
3. If and only if that gate passes, one fresh Filtered training run, no resume from v1 or feasibility adapters: max5epochs/1250updates, all5epoch checkpoints, same optimizer/schedule/precision/order/budgets. Preserve stop-on-failure/no automatic retry.
4. Exactly24 fixed validation summaries at epoch3 and24 at epoch5 with original beam4 settings; compare to saved historical F and D058epoch5. Use D053 references and original substantive quality rubric, separate random20/targeted4, AI/software/human provenance. No automatic quality GO.
5. Stop for independent review. No RO/U execution, Teacher calls, held-out access, new experiments, settings/threshold/target changes, or clock reset. Any later matched RO/U requires separate researcher approval and must fit the original deadline.

The authorization template stays false. Final new hashes, exact diffs and callback invocation logs are in `outputs/stage_i/d059_preparation_v2/`. This document is preparation, not a GPU execution record.
