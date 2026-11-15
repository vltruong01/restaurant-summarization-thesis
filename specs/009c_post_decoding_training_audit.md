# Stage I: post-decoding implementation audit and proposed scope pilot

**ID:**009c / D-048. **Date:**2026-10-16, Asia/Saigon.
**Status:** Offline audit authorized/completed. New feasibility, model inference and training **not authorized**.

## Authorized work

Inspect actual successful training code/configuration, saved optimizer/scheduler/scaler/checkpoints and existing train/validation data and Baseline/A/B outputs. CPU tensor reads, source-code inspection, token/mask arithmetic, descriptive text metrics and meta-only architecture/PEFT injection are permitted. No pretrained model forward/backward/generate, Teacher requests, held-out data, dataset/weight edits or more decoding trials. Preserve previous artifacts and record evidence versus hypotheses.

## Verified audit

Successful runs use24 decoder self/cross q_proj/v_proj adapters,48 tensors/294,912 parameters, rank8/alpha16; encoder base and all other base weights are frozen. Encoder modules use query/value/query_global/value_global names, which do not match q_proj/v_proj suffixes. Meta inspection verifies counts and names; checkpoint/optimizer evidence verifies actual successful updates. All three runs reached1,250 optimizer steps/five epochs; best validation-loss checkpoint is1250. Actual args, LR/scaler/scheduler states, complete source/label bindings and masks show no identified critical training bug. Original zero-caption format-check weakness is an acceptance diagnostic issue; strict existing diagnostic checks must be used for future acceptance.

All6,600 teacher targets have zero exact repeated sentences under the common rule, while students repeat heavily. Repetitive target-adjacent words/templates, source-specific omissions, inherited Teacher/caption limitations, sparse nine zero-F train cases, capacity/location and teacher-forcing/free-running behavior are investigated separately. No causal mechanism is established. ROUGE is teacher agreement; AI qualitative notes are not human truth.

## Proposed experiment, awaiting separate approval

Two fresh **F-only** arms, with512 identical train businesses from the saved proportional caption-count/source-length sample(seed20261016), two epochs,128 optimizer steps each, batch1/accum8. Never train on validation. All200 existing validation businesses supply loss; generation only on the same random20+targeted4 validation F cases per arm,48 future outputs total. Existing full F checkpoint remains the primary practical reference, but is not compute-matched to the short pilot. Do not compare expanded short B to old full A alone as causal evidence.

A retains current decoder-only scope. B adds encoder local/global query/value, totaling48 projections/96 tensors/589,824 trainable parameters, same rank8/alpha16/dropout0/bias none. Exact full-match module regex and complete names are in the proposed config/architecture audit. Both keep the accepted datasets/targets/token budgets/source template/K5/backbone/precision, AdamW2e-4 and baseline beam4/no-repeat0/max_length768. The shorter common scheduler has7 warmup steps versus63 for the old full run; record this difference. Shared fresh decoder initialization must be tensor-equal across arms despite changed module-initialization order. Fresh scope-bound optimizer/RNG/checkpoints only; no cross-scope resume.

Future feasibility must first verify full5120/768 padded shapes/masks, every intended trainable branch and numerical/checkpoint behavior, actual VRAM/headroom and exact settings; stop on failure without fallback. LoRA A gradients can initially be zero when B matrices initialize zero. No new-scope GPU compatibility is claimed now.

Predeclared proposed quality gates cover technical integrity, exact sections/EOS and zero-F behavior, >=30% relative reduction in macro adjacent-word pair fraction, >=10 percentage-point gain in source-verified useful-unit coverage without worse contrast/unsupported-claim rate, and comparison with the full reference. These are proposed criteria for approval, not accepted research parameters. Reference content units require source verification before pilot outputs; absent verification makes grounding/coverage inconclusive. Lower repetition/loss/ROUGE alone cannot pass. Full details and risks are in the proposal report; no automatic escalation to RO/U/F retraining.

## Acceptance of this offline task

Write versioned `STAGE_I_TRAINING_FAILURE_AUDIT.md`, `STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md`, machine evidence/manifests and a self-contained ChatGPT review ZIP. Update progress/decision log. Stop after packaging, awaiting researcher review. No new training/decoding method is frozen by preparing this proposal.
