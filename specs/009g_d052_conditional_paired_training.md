# D-052 ? Conditional paired training pilot

Researcher-approved execution of D049 fresh F-only512x2 A/B training with the unchanged scientific config and exact data/order/optimization. D051 strict replay remains FAIL, quantified limitation accepted only for this exploratory pilot. Original gates/reports are preserved; separate versioned hash-bound train-only gate checks exact approval and evidence, never syntheticPASS.

Run A then B on local GPU with128 actual updates, finite gradients/loss, optimizer/scheduler/AMP skip/memory monitoring, complete masks/content, frozen base hashes and exact shared decoder initialization/runtime. Preserve checkpoint64/128 states and choose minimum validation loss. Stop on material anomaly without fallback; do not repeat or extend automatically.

Source-only reference verification is independent of new outputs; candidates stay unverified until researcher confirms. No generation permission, held-out use, Teacher calls, frozen-data/filter/checkpoint overwrite, policy/tolerance/precision/batch/hyperparameter changes or financial work. Stop after training, validation and portable review bundle. Versioned decision: outputs/stage_i/d052_paired_pilot_v1/D052_CONDITIONAL_REPRODUCIBILITY_DECISION.md.


## Verified execution and stop boundary (2026-10-16)

Both fresh Filtered-only models completed the approved 512-business, two-epoch pilot: 128 actual optimizer updates, 1,024 training microbatches and 400 teacher-forced validation batches per arm. The same 200 validation businesses were evaluated at steps 64 and 128. Actual A: 24 projections / 294,912 trainable parameters; B: 48 projections / 589,824, including encoder local/global Q/V. All 24 shared decoder initializations and both exact epoch orders match.

| Epoch | A train mean | B train mean | A validation loss | B validation loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.990661 | 3.840852 | 3.186374 | 2.994199 |
| 2 | 3.355705 | 3.167322 | 2.955577 | 2.757420 |

Both arms select checkpoint-128 by minimum validation loss. B minus A validation loss at epoch 2 is -0.198157. This is descriptive teacher-target predictive agreement, not evidence that generated summaries are more informative or factually grounded. No new summaries or quality comparison were produced.

Every intended adapter tensor has finite gradients; `lora_A` gradients are expected to be zero at steps 1/2 because `lora_B` starts at zero and the initial scheduled LR is zero. All intended `lora_A`/`lora_B` tensors have nonzero gradients from step 3. Pretrained parameters remain frozen and byte-identical. No NaN/Inf, OOM or AMP-skipped update occurred; optimizer/scheduler steps, LR schedule, masks and complete input/label sequences pass actual execution checks.

Same local RTX 3050 Laptop GPU / pinned environment. PyTorch peak allocated/reserved: A 1468.44/2630.00 MiB; B 1825.34/3070.00 MiB. Epoch checkpoints preserve adapter, optimizer, scheduler, scaler, RNG and Trainer states with verified hashes. This is state preservation, not a new strict replay or fresh-process recovery PASS.

D-051 strict replay remains FAIL under rtol=1e-5 / atol=1e-6; its documented encoder-backward variability is accepted only for this exploratory experiment. No tolerance, deterministic-policy, precision, data, filtering, optimization or decoding change was made. The post-execution audit verified all 14 preparation, 6 original code, 22 upstream, 275 imported archive, 7 cached-backbone and 197 D050/D051/prior-feasibility hash entries unchanged.

Operational provenance: an initial A attempt was stopped at observed 27/128 before any epoch checkpoint because the supplemental observer did not attach through a stale Transformers lazy-module import. Raw evidence is preserved in interrupted_attempt01. Versioned wrapper v2 fixes import/observer wiring and checks attachment before training; no scientific training math changed. No partial weights were reused. Only the fresh, fully audited 128-update A/B runs enter the paired tables.

Source-reference readiness: the portable source-only worksheet covers the fixed 20 random + 4 targeted cases, 652 complete reviews and 90 original photos. Its 226 candidate units remain unverified in the saved initial artifacts; no user-confirmed reference file has been submitted at the checked locations. Browser progress is unknown. Explicit researcher verification/freeze and separate approval are required before any new predictions. Teacher targets/candidates are auxiliary, not independent factual ground truth.

Two epochs, one seed and the simultaneous change in adaptation location/parameter count limit interpretation. Do not infer RO/U, test or long-run quality results. No held-out use, Teacher requests, accepted-data edits, historical checkpoint overwrite, new decoding or automatic training extension. Stop after training/reporting.

Evidence: `outputs/stage_i/d052_paired_pilot_v1/STAGE_I_D052_PAIRED_TRAINING_REPORT.md`, `D052_TRAINING_VALIDATION.json`, `PER_EPOCH_LOSS_COMPARISON.csv`, `GRADIENT_BRANCH_SUMMARY.csv`, `POST_EXECUTION_INTEGRITY.json`, and the source-only `outputs/stage_i/d052_source_reference_review_v1/SOURCE_REFERENCE_REVIEW.html`. Review bundle: `outputs/stage_i/stage_i_d052_paired_training_chatgpt_bundle_v1.zip`.

The original frozen D-049 configuration retains its historical offline-prepared status fields. D-052 execution permission is exclusively the separate hash-bound train-only approval; it does not authorize generation or disable the original replay gate.
