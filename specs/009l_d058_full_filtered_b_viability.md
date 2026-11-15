# D-058 — One full-data Filtered-only B viability attempt

Status: **PROPOSED / OFFLINE PREPARATION ONLY — researcher execution approval pending**.
The researcher accepts D057 findings and closes the current root-cause inquiry.
Weak learned early transitions are supported; additional exposure is not a proven
cure. Historical full F remains a practical, imperfect benchmark. D051 strict
replay FAIL and all previous artifacts remain unchanged.

## Implementation and bounded budget

One fresh `allenai/led-base-16384` revision
`38335783885b338d93791936c54bb4be46bebed9` with the existing B scope:24 decoder,
12 encoder local and12 encoder global Q/V projections;48 modules/589,824 trainable
adapter parameters. Rank8/alpha16/dropout0, FP32 backbone frozen, FP16 AMP,
non-reentrant gradient checkpointing, token0 global attention, no truncation.
Use original Stage H Filtered NPZ/Stage G JSONL,2000 train/200 validation; source
5000/reviews4000/captions750/K5, complete labels<=768. No target regeneration.

Reuse existing collation, keyed fresh adapter initialization, per-epoch CPU
`torch.randperm(2000, seed=42+epoch)` ordering, LoRA attachment checks, optimizer,
accumulation-loss contract and evaluation metrics. Prepared runner:
`scripts/run_stage_i_d058_v1.py`; no new diagnostic framework or feasibility run.

AdamW2e-4/weight decay0.01/betas0.9,0.999/epsilon1e-8; batch1, accumulation8,
clip1, validation batch1. Five planned epochs are a hard ceiling:250 optimizer
updates/epoch,1250 total;10,000 training microbatches;1000 validation-loss batches
(200 after each epoch). No distributed execution or batch/precision substitution.

Explicit proposed adaptations from the512x2 pilot, requiring approval:

| Item | Short pilot | D058 proposal |
| --- | --- | --- |
| Training cohort/epochs |512 /2|2000 /5 maximum|
| Fresh linear scheduler |128 updates /7 warmup|1250 updates /63 warmup (ceil0.05x1250)|
| Loss-based early stopping |patience2|disabled within this single bounded run|
| Checkpoint retention |2|5 epoch snapshots so epochs3/5 cannot be deleted|
| Automatic best-loss reload |yes|no; assess exact epoch3/5 snapshots|

Disabling loss early stopping and automatic best reload is an explicit checkpoint
policy proposal, not a silent optimization change. This provides the two requested
quality checkpoints. Epochs1/2/4 are recovery/loss records only, never new decoding
candidates. No short-pilot resume, scheduler restart, LR search or extra epoch.

Save optimizer/scheduler/scaler/RNG state and integrity receipt per epoch. No
automatic restart on failure. A technical failure is an incomplete experiment,
not proof of scientific failure; stop/report without starting another attempt.

Local time planning only: prior B512x2 training/validation took2194.33s. Scaling
the old train+validation workload gives roughly4.7–6h before inference; reserve
about4–7h for training, not a guarantee or fresh benchmark. Extra48 beam4 outputs
may add substantial time. Prior B stress training fit the local6GB GPU, but full
run memory/time remain untested. No GPU is loaded or rented during preparation.

## Frozen evaluation scope

Use the exact24 existing validation inputs (20 random +4 targeted, unchanged IDs,
order, source/token hashes), D053 final approved references and saved historical
full F predictions. Generate only24 outputs at epoch3/step750 and24 at
epoch5/step1250: **48 new outputs maximum**. Comparison has72 existing/new records,
not72 new generations. No historical baseline regeneration or Pilot A generation.

Original beam4, do_sample=False, max_length=768 **including decoder-start**,
no_repeat_ngram_size=0, length_penalty1, early_stopping=True, forced_eos_token_id2,
FP16 generation and complete effective GenerationConfig remain unchanged. At most
36,816 newly emitted token slots (48x767); no prompt rescue or64-token censor.

Treat historical full F as a usability benchmark, **not a matched causal
architecture control**. Fresh named adapter initialization, architecture/location,
parameter count and execution history differ. No encoder-only causal claim.

## Predefined assessment and go/no-go

Reuse the frozen D049/D053 definitions, with no new embedding scorer, quality
threshold tuning or composite score. Report random20 as primary, targeted4
individually/separately. Do not pool them into a prevalence estimate.

For each output classify `usable`, `minor_repair`, `unusable` or `uncertain`, with
evidence and provenance. Usable means coherent and semantically complete review
synthesis, required structure, no degeneration dominating the body, useful
restaurant-specific source-supported information, meaningful opinion contrasts,
and permitted visual evidence/empty visual behavior. A valid header, low loss,
generic correctness or lower repetition alone is insufficient. Review statements
remain attributed opinions; simple correct visual content is not an error.

- **Software:** exact sentence/word/lexical and BPE repetition; marker format;
  lengths; EOS after decoder-start; cap hits and EOS-at-cap possibly forced.
  Do not count initial decoder-start2 or forced terminal EOS as natural completion.
- **Model-assisted inspection:** optional evidence-linked suggestions for claim
  support, semantic redundancy, coverage, contrasts and usability. Keep explicitly
  provisional; no extra paid judge/API or embedding/GPU experiment is included.
- **Researcher-confirmed:** final usability, claim support/severity, semantic
  proposition groups, Core coverage, useful visuals/contrasts and semantic ending.
  Reuse D053 source references; do not redo source annotation. Missing output
  verification is N/A, never zero error or automatic approval of suggestions.

Core coverage is descriptive and evidence-linked; not every Core detail must be
verbatim. Supporting omissions are not automatic errors. Excluded88 units and
two uncertain parent claims remain outside scoring. Avoid counting all664
proposals as the denominator. Preserve positive/negative perspectives and caption
scope: no photo-only information or new cross-caption relationships.

For paired cases use `material_B_win`, `material_baseline_win`, `minor_or_tie`,
`uncertain`, anchored in multiple quality domains and exact evidence IDs. A
material win fixes an unusable summary or recovers useful important grounded
content/coherence without a compensating material grounding/contrast loss.
Cosmetic rephrasing, only header recovery, length/loss/ROUGE shifts are minor.
Show usable counts, wins/losses/ties and applicable denominators; existing paired
business bootstrap(seed20261016,5000,95% percentile) is descriptive for random20,
not a mechanical significance gate. No targeted4 inferential CI.

**GO:** researcher explicitly confirms a clear, practically substantial net
improvement in usable summaries on random20, spanning semantic repetition,
informative Core/opinion synthesis and grounding, with no compensating material
error pattern and no unresolved severe targeted/zero-caption regression. The
written decision must identify recovered and regressed businesses with evidence.
This qualitative rubric preserves the approved no-rigid-percentage-threshold
policy; it is not automatic software acceptance.

**NO-GO:** neither fixed checkpoint meets that substantive criterion; results are
only loss/header/repetition gains, minor/mixed, or still unconvincing. Inconclusive
quality does not authorize more LED optimization. Missing researcher review means
PENDING, not GO. Any persistent important grounding/zero-caption failure must be
addressed in the decision rather than hidden in averages.

Assess epochs3 and5 with the same rubric; if both qualify without a meaningful
quality difference, prefer epoch3. If only one qualifies, nominate it. Do not
decode extra checkpoints or choose by validation loss alone. Researcher confirms
the final practical decision; these24 repeatedly examined cases limit generality.

## Strict stop-loss and exact authorization

One fresh B training run,<=5 epochs/1250 updates plus the two fixed24-case
evaluations; no extra diagnostics, seed/schedule/architecture/decoding sweeps.
Do not interrupt training at epoch3 to tune epoch5: preserve one declared schedule.
Unexpected technical/nonfinite/OOM/integrity failure stops without fallback.

If NO-GO, stop the current LED-LoRA optimization route and return a concise
alternative-model strategy for a separate decision; no automatic D059/D060
numerical diagnostics. If GO, only propose a fair RO/U/F study with the same
selected architecture/protocol across conditions; do not start it automatically.

Current approval is **offline preparation only**. Future researcher authorization
must explicitly approve the declared schedule/checkpoint-policy adaptations and
bind config, preflight manifest and runner SHA256. No held-out100 access, Teacher
API, annotation regeneration, frozen artifact/checkpoint modification, rental,
extra run, or unapproved resume. Stop after execution/quality report for review.
