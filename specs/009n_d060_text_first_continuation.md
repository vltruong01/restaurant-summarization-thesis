# D-060 — Final text-first LoRA continuation hypothesis

Status: OFFLINE PREPARATION ONLY. The researcher selected this hypothesis after
D-059; this supersedes the earlier prohibition on *preparing* another hypothesis,
not the requirement for separate GPU authorization. No D-059 quality GO/NO-GO is
assigned by Codex. D-051 strict replay FAIL remains unchanged.

Keep the pinned LED-base backbone and decoder Q/V-only LoRA: 24 modules,
294,912 parameters, rank8/alpha16/dropout0. Load the exact trained historical RO
best adapter (step1250, five epochs) with is_trainable=True. Do not randomly
reinitialize it, merge it into the backbone, stack a second adapter, or import
historical optimizer/scheduler/scaler/RNG states.

Propose one Filtered Stage-2 run: train2000/validation200, maximum3epochs,
750updates, batch1/accumulation8. Fresh AdamW2e-4,wd0.01,betas0.9/0.999,
eps1e-8,clip1; fresh linear750/warmup38 (ceil0.05x750). FP32 storage/FP16 AMP,
nonreentrant gradient checkpointing, source token0 global attention. Preserve
source5000/reviews4000/captions750/target768/K5 and all original complete tokens.
Use the first three frozen D058 epoch permutations. Keep all three checkpoints,
no loss early-stop or automatic best-loss reload. Do not resume old training.

Evaluate exactly24 fixed Filtered validation cases at Stage-2 epochs1 and3:
48 new outputs, original beam4/max_length768 including decoder-start, all other
effective GenerationConfig values unchanged. Compare saved historicalF,
D058epoch5 and D059Expanded epochs3/5; no benchmark regeneration. Report random20
and targeted4 separately with D053 approved references/D058 rubric.

The detailed preparation plan defines material quality gains, initialization/
duration/validation-reuse confounds, nonautomatic researcher GO, and a strict
72-hour viability decision window from any future execution approval, capped
by the unchanged original D059 deadline2026-10-25T15:10:48.941817+00:00.
Failure, expiry or inconclusive quality ends optimization; use original matched
RO/U/F and proceed to formal evaluation. No follow-up experiment is automatic.

If GO, separately authorize matched Stage-2 RO/U/F from identical copies of the
same RO adapter with identical schedules, data order, precision and one common
selected Stage-2 epoch. RO gets additional RO training, not merely reuse of its
Stage-1 endpoint. Reuse the approved F continuation if its settings match exactly.
Primary comparisons must never mix new F with historical RO/U.

This preparation allows CPU adapter load/serialization, hashing, tokenization
and tests, without model forward/backward/generation or optimizer updates.
Original datasets, targets, references, checkpoints and historical decisions are
read-only. No Teacher calls or held-out contents; no review-count/learning-rate/
rank/architecture/decoder search. Execute only after new hash-bound approval.
