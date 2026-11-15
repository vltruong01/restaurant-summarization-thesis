# Proposed next step: learn to avoid repetition while preserving information

Offline preparation only,2026-10-19. User authorized implementing the offline
proposal; no new GPU/model experiment has started. Historical evidence is preserved.

## Why this intervention

Hard NLI/MMR reduced text but left17/24 empty review sections and retained factual
errors. It is not adopted. Decoder guards prevent selected surface forms but do
not teach evidence-grounded synthesis. Welleck et al. propose token/sequence
Unlikelihood to suppress undesirable generations while retaining likelihood
training: https://arxiv.org/abs/1908.04319 . Xu et al.'s DITTO uses pseudo-repeated
sentences and tests summarization on CNN/DailyMail:
https://arxiv.org/abs/2206.02369 . Neither establishes our failure cause, guarantees
LED/Yelp transfer, nor proves better factual grounding. Our proposed negative-prefix
objective differs from both published training procedures and must be disclosed.

Measurement caution: the earlier legacy adjacent-word flags include sentence
boundaries and legitimate names; narrow literal forbidden whitespace stutters
were already0 in the guard audit. Do not invent a word-stutter improvement from
those flags. Sentence/semantic loops and lost Core information are the main
remaining defects. Exact-copy UL does not directly supervise paraphrased redundancy
or hallucinated relationships; success on those dimensions remains unproven.

## Minimal implementation plan

1. Freeze512 hash-sampled TRAIN businesses, positive labels, one safe negative view
   per business where possible, evaluation IDs and exact integrity bindings.
2. After NEW approval only: gate both arms3updates each, no generation/replay.
3. If both gates PASS: two fresh continuations of same historical trained full-F
   adapter, CE control versus CE+0.1UL. Two epochs/128updates each, same full inputs,
   seed/order/optimization; save64/128, evaluate only128. No hyperparameter search.
4. Compare48 raw outputs,20random and4targeted separately, using D053 evidence and
   existing quality rubric. No inference-time deletion or repair. Stop for review.

Architecture stays decoder-only24modules/294912parameters. Using historical F makes
the two arms matched for initialization, unlike comparisons across D058/D059/D060.
It tests a corrective continuation, not training from scratch or whether UL would
work with Expanded LoRA. New schedule128/warmup7 and auxiliary loss need approval.

## Budget and hardware

Gate total:6updates/48positive microbatches, at most48negative forwards,0generation.
Pilot total:256updates/2048positive microbatches, at most2048negative forwards;
200validation examples after each of2epochs per arm =800CE-only forwards.
Generation:48outputs, beam4,max_length768 (includes decoder-start); at most36816
newly emitted-token slots, unchanged semantics. No model/scorer/API generation now.

D060 measured3469.41s for6000positive training microbatches +600validation batches
on the local accepted setup; peak allocated1.54GB/reserved2.42GB. A very rough
linear anchor for this pilot's4096maximum training forwards +800validation forwards
is around43minutes, before gate/generation/I/O; this is NOT a benchmark or guarantee.
The negative decoder prefixes differ in length and both loss graphs coexist.
Plan operationally for roughly1-3hours total on comparable hardware, with the
actual local GPU tested first. No paid GPU assumption. Memory is UNMEASURED for UL;
old peaks are context only, not a sufficiency claim. Require15% measured headroom;
on failure stop without changing precision/batch/budgets.

## Decision and time protection

Primary criterion: reviewer-confirmed useful, business-specific, grounded summaries
with reduced harmful repetition and preserved Core/opinion information. Check all
regressions by business, not aggregate repetition alone. AI labels stay provisional.
These24cases are development cases. New confirmation subset is NOT yet claimed
unused/independent; audit prior exposure before selecting one, no held-out access.

PROPOSED timebox: at most48hours from a future explicit approval, including researcher
quality review. This is a proposal, not an extension of the old clocks. Historical
D060 deadline2026-10-22T07:11:22.846767+00:00 and D059deadline
2026-10-25T15:10:48.941817+00:00 remain recorded unchanged. The researcher must explicitly
authorize an exception to prior final-attempt restrictions and state a new UTC end
time. Pending review at expiry is inconclusive; no automatic repeat/variant.

## Exact authorization to be considered later

Use AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json in preparation output. Approval must
bind config, preflight and runner hashes; approve both specified arms,6gateupdates,
256trainingupdates and48validation generations, new UTC start/end and explicit
prior-stop-loss amendment. Gate PASS does not imply quality GO. No RO/U, Teacher,
held-out access, frozen artifact changes, decoder search or further experiments.
The preparation config remains GPU_execution_approved=false.

## Offline commands

From repository root in the pinned .venv-caption-pilot environment:

    python -m unittest discover -s tests -p test_stage_i_unlikelihood_v1.py -v
    python scripts/run_stage_i_unlikelihood_pilot_v1.py preflight

The prepared runner refuses GPU modes without a separately supplied, explicit,
hash-bound researcher approval. Full runtime, AMP, PEFT attachment, actual UL
gradient flow and checkpoint replay are NOT verified by synthetic offline tests.
