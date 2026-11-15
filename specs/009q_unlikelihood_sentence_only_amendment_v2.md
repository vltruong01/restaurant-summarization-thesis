# Unlikelihood pilot v2: ONE sentence-only offline amendment

Status: researcher-authorized OFFLINE revision only. V1 accepted as a hypothesis,
not GPU authorized. Preserve spec009p/v1 and all scientific artifacts unchanged.

## Exact amendment

The principal observed defect is sentence/semantic repetition. Replace the v1
alternation with512sentence pseudo-repetition views for the exact same512train
businesses. Word-negative objective is disabled in v2. The prior256sentence views
must be identical. Same builder/seed/limits; select candidates by the existing
deterministic policy. Any business lacking a safe view is reported and blocks
readiness; no truncation, substitution from another business, invented sentence
or fallback to words. Independently derive the entire penalty mask, including
all gold-prefix conflicts and special-token exclusions, not merely check set bits.

Unchanged: historical F adapter, decoder-only24modules/294912parameters, full source
and original positive labels,512IDs/order, seed42, AdamW2e-4/decay.01, FP16, batch1/
accum8,2epochs/128updates per arm/warmup7,CE control/CE+0.1UL, checkpoints64/128,
200validation CE, final epoch2only,24development cases per arm,48outputs total,
beam4/max_length768. No model/optimizer execution during preparation.

## Gate observation, not optimization change

Within each existing3-update/24-microbatch gate, measure CE and UL gradients on
all48trainable LoRA tensors using autograd.grad(retain_graph=True), leaving .grad
buffers and RNG unchanged. Use current AMP scale for separate traversals, then
unscale diagnostic values. Record raw UL norm,0.1weighted norm,effective norm,
CE norm, ratio, cosine,combined norm and per-tensor missing/zero counts; also
record0.1UL/CE loss ratio. Control receives effective weight0 but still reports
counterfactual0.1weighted signal, never applies it. Training updates remain3each.

At most48additional autograd traversals per gate arm. No additional explicit
forward calls/generation/replay/optimizer updates. Gradient checkpoint recomputation
can increase runtime and memory; report actual gate use. These measurements are
per-microbatch pre-clipping gradients, not accumulated-step or AdamW update fractions.
Existing pre-optimizer gradient log in pinned Trainer occurs AFTER norm clipping;
do not conflate it with component probe norms. Frozen-base integrity/AMP checks,
finite numerical criteria and15%measured memory headroom are unchanged.

UL loss nonzero with zero gradient is explicitly reported. Weighted gradient norm
below1%of CE receives a DESCRIPTIVE weak-signal flag, not a new numerical pass/fail
gate or automatic loss-weight tuning. Report microbatch distribution; a high norm
does not imply improved quality. No extra gate updates to obtain stronger signal.
Any unexpected nonfinite, mutation, integrity, mask or memory failure stops execution.

## Quality comparison

Compare CE and UL as the matched continuation contrast. Also reuse the24saved
D059 Expanded-LoRA epoch3 predictions with exact source/subset/decoding bindings.
This is a NON-CAUSAL practical benchmark: different architecture, parameter count,
training duration and initialization. No new D059 generation or rerun.

GO requires researcher-confirmed meaningful reduction of harmful repetition without
loss of Core/business-specific information, positive-negative opinion balance,
factual grounding, visual usefulness or structural/semantic completion. Exact
repeat rate alone, lower loss or header compliance never establishes GO. Audit
regressions by business, separate random20/targeted4, keep AI labels provisional.
No requirement that every Core unit appears verbatim or every Supporting unit appears.
The24cases remain repeatedly inspected DEVELOPMENT cases, not independent holdout.
Exact sentence-copy negatives do not directly supervise paraphrase redundancy or
grounding; improvements on those dimensions remain empirical questions.

## Authorization and stopping

New hash-bound researcher GPU approval, explicit exception to historical final-
attempt restrictions, and a new UTC execution/review window of at most48hours
are indispensable. V1 approval cannot authorize v2; templates remainfalse.
No historical deadline is changed/reset. Preserve D051strict replayFAIL.
If pilot fails or remains inconclusive, stop without an additional experiment.
No automatic RO/U, Teacher, held-out contents, decoder changes or artifact edits.
