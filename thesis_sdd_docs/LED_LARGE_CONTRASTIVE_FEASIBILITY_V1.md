# LED-large: source-verified contrastive feasibility proposal V1

**Decision requested: approve this methodology for bounded pair preparation and implementation review, not GPU execution.** No new training, inference, negative examples or Teacher targets have been produced. The included CPU checks use synthetic tensors only.

## 1. Current evidence and limits

Accept the user-supplied independent N-best V2 review: Phase A is technically successful; the semantic-diversity and quality-headroom gates are **not established**. Close the current beam4 reranking branch. Preserve the 60 V2 sequences, the separate failed V1 call, source packets, RNG receipts and full checkpoints. Do not convert the old model-assisted case preferences into human-confirmed labels.

The measured pool has identical review sections for 9/12 businesses, only five clear profile differences, and one clear grounding repair (Mesa c2). Eleven of twelve zero-caption candidates assert unsupported visuals. These findings motivate testing a different learning signal, but do not prove that contrastive learning will fix it. LED-large V2 Original CE epoch 3 remains the provisional Filtered checkpoint; the established matched LED-base family remains the primary thesis baseline until a complete matched large family exists.

CLIFF supplies evidence that carefully constructed factual-error negatives can help summarization, but its representation loss and projection head are not the proposed implementation. [Cao and Wang, 2021](https://aclanthology.org/2021.emnlp-main.532/). BRIO supports combining sequence-ranking and token CE; it ranks with reference-based ROUGE, whereas this proposal requires source-verified pair labels. This is an adaptation, not a reproduction or an established Yelp result. [Liu et al., 2022](https://aclanthology.org/2022.acl-long.207/).

## 2. One objective; original supervision retained

For source x, let y+ be the **unchanged original frozen Filtered Teacher target**, after the accepted removal of its initial BOS in the training-label view. Use it as a contrastive positive only if its complete content passes source adjudication. A negative y- is a separately versioned, minimally corrupted copy, never a replacement Teacher target. Include EOS and section headers; exclude only -100 padding. Both sequences use their own once-shifted teacher-forcing prefixes, decoder start EOS=2, and the original tokenizer. Do not score both alternatives under the positive prefix.

Define, with V(y) the valid token positions:

    sθ(y|x) = mean[t in V(y)] log pθ(yt | x, y<t)
    LCE = -sθ(y+|x)
    Lpair = softplus((m - sθ(y+|x) + sθ(y-|x)) / τ)
    L = LCE + λ Lpair

**One fixed proposal:** λ=0.10, m=0.10 nats per valid token, τ=1.0. These are conservative engineering presets, not optimized or literature-validated values. No weight, margin or temperature sweep. At λ=0 the scalar and gradients equal ordinary CE. Average per-example losses over the eight-example accumulation window, matching the parent's batch-one CE reduction. Keep section weights uniform: this is not WCE, UL, DPO, an auxiliary classifier or a new architecture.

Use FP32 selected-token log probabilities computed as chosen logit minus logsumexp, in token blocks if needed; never exponentiate full sequence probabilities or use log(1-p). Length normalization reduces raw-length dominance, but does not eliminate length, fluency or synthetic-artifact shortcuts. Log scores, lengths and defect types separately. Positive CE anchors informative generation; the preference term alone could reduce negative likelihood without improving positive content.

## 3. Pair admissibility: the main scientific precondition

Use only the existing twelve TRAIN source packets. `TRAIN_ONLY_PILOT_MANIFEST.json` locks eight fit businesses (two zero-caption, six captioned) and four probe businesses (one zero-caption, three captioned). Within each stratum, sort SHA256(`LED_CONTRASTIVE_FEASIBILITY_V1:` + business_id); take the first two/six for fit. The split uses source metadata, not output quality. All twelve have already been inspected and were in CE training; probe means excluded from the new preference updates, **not unseen or held-out**.

After separate approval, examine at most **24 proposed edits**, seeking exactly **16 admissible pairs** over all eight fit businesses, one to three pairs each:

| Pair category | Required evidence and minimum |
|---|---|
| Core omission | At least four pairs across at least three businesses. Delete one supported, recurring business-specific fact already present in y+. Quote at least two distinct visible reviews for recurrence. The negative is incomplete, not necessarily false. |
| Same-aspect disagreement loss | At least four pairs across at least three businesses. Preserve an existing positive/negative contrast in y+; the negative removes or overgeneralizes one side. Quote opposing reviews about the **same** aspect. Distinguish minority omission from false consensus. |
| Grounding error | At least four pairs, including both zero-caption fit cases and at least two captioned businesses. Introduce exactly one unsupported visual/entity/relation claim. Every visual relation in y+ needs support in one specific caption; independent captions do not establish a joint scene. |

The remaining four pairs may use these same categories, with a fixed source-audited assignment before any optimization. Do not force a category onto an unsuitable case, replace businesses after seeing model effects, or expand the edit budget to manufacture a pass. If eight fully admissible positives or the pair quotas cannot be obtained, stop and report the bottleneck.

For every pair, store original-target hash, complete positive/negative token IDs, one minimal text diff, defect category, exact review/caption markers and character spans, reviewer rationale, and explicit independent researcher acceptance. Preserve the original target bytes and label IDs. Automated or model-assisted support checks are provisional; a high NLI score is not verification. NLI ranking itself has documented reliability limits. [Falke et al., 2019](https://aclanthology.org/P19-1213/).

No new LED/LLM negative generation is proposed. A demonstrably wrong claim already present in the saved N-best outputs may motivate an edit; do not label an entire lower-ranked beam negative automatically. Full-positive source checking prevents teaching its other errors as truth. No new factual text may be inserted into y+ to repair a deficient Teacher target. If the desired Core item or disagreement is absent from the immutable target, this objective has no direct positive signal to teach it; disclose that limitation or reject the pair.

Use no D-053 references, original photos, validation labels or held-out information for pair construction. Source support means the exact truncated **model-visible** reviews and condition-specific Filtered captions. A true fact found only elsewhere is inadmissible. Zero captions means no supported visual assertion. Do not treat all reviews as unanimous, mixed opinions as contradictions to erase, or every short summary as bad. Log shorter omission negatives and longer insertion negatives separately; inspect for length/style shortcuts.

## 4. Numerical feasibility on RTX 3090

The successful CE run recorded conservative peak usage 17,684,430,848 bytes (**16.47 GiB**, 30.10% headroom), not the roughly 4.6 GiB observed in N-best inference. Two retained training graphs might exceed 24 GB. Fit is **unproven**; do not assume memory doubles exactly or claim the old headroom transfers.

Propose sequential recomputation, with no second model or projection head. Scout y+ and y- under no_grad but the unchanged training/dropout mode, saving separate full RNG states before each and after the pair. Let

    q = stop_gradient[(λ/τ) sigmoid((m - s+ + s-)/τ)]
    ∇L = -(1+q) ∇s+ + q ∇s-

Replay each scout RNG state, retain one differentiable graph at a time, and backward its contribution. Compare scout/replayed FP32 scores (atol=rtol=1e-6); require exact RNG restoration and CPU-gradient equivalence before deployment. Log the true scout objective, **not** the linear surrogate as if it were the loss value. The synthetic CPU checks establish the derivative identity, not LED/CUDA replay correctness. If the pinned runtime cannot satisfy replay tests, stop; do not silently use stale coefficients or substitute a detached negative branch.

Training RNG consumption must be replayed, not assumed equal to the twelve eval-only encoder draws. Generation uses the accepted eval guard. Preserve model modes, cache, adapter activation and checkpointed-dropout RNG. Verify Python, NumPy, CPU/CUDA states and parameter gradients in the future implementation. No exact CUDA trajectory claim; D-051 strict replay FAIL remains preserved.

Use the existing FP16 family policy, proposed unchanged: initial scale 65536, factor 2 growth/0.5 backoff, growth interval 2000; at most eight backoffs per job and floor 256. Keep one shared scale across both contributions and all eight pairs. Unscale once after the full window, check all 456 intended LoRA tensors, clip norm 1, then step optimizer/scheduler once. Replay an overflowed whole window under the bounded policy; never step after only the positive contribution. No independently scaled component-gradient probes. PyTorch requires a constant scale throughout an effective batch and checks at effective-batch granularity. [PyTorch 2.1 AMP guidance](https://docs.pytorch.org/docs/2.1/notes/amp_examples.html).

FP32 loss arithmetic and bounded q (0≤q≤0.1) do **not** bound all network gradients or repair nonfinite model activations. Stop on nonfinite forward values, replay mismatch, missing gradients, exhausted AMP bounds, corruption or <15% measured headroom. No automatic retry, precision fallback, source truncation or batch change outside the declared overflow policy.

## 5. Bounded future pilot and fair control

This proposal is **not** a fresh five-epoch replacement run. Start two isolated branches from the same pinned original CE checkpoint-750 adapter: CE-only continuation and CE+pair continuation. Load weights identically; use fresh matched optimizer/scheduler/scaler states, not the historical optimizer state. This is a proposed methodological choice requiring approval. Keep the original checkpoint untouched. Comparing only against the saved CE3 output would confound the objective with extra training; the CE-only continuation is mandatory.

Keep backbone revision, 228 Expanded-LoRA modules/456 tensors, r8/alpha16/dropout0, all source and label policies, seed42, physical batch1 and accumulation8. For this small continuation only, propose fixed LR **2e-5**, AdamW betas .9/.999, epsilon1e-8, decay.01, clipping1, linear schedule with one warmup update. Both arms use the same positive-business order and 16-update schedule. Extra negative forwards consume stochastic draws; seed equality does not establish identical training trajectories. Do not present this schedule as identical to the original five-epoch schedule.

1. Complete source/pair adjudication and pinned CPU implementation/replay tests; obtain fresh hash-bound GPU approval, real UTC deadline within 48 hours, and a newly confirmed protected thesis-evaluation reserve. Historical approvals do not transfer.
2. On one RTX3090, run each arm's gate for three successful updates, nominal24/max88 example microbatches, at least15% headroom. Generate the same four TRAIN probes once per gated arm for technical sanity. Discard both gate branches after recording evidence.
3. Only under explicit conditional authorization, restart both pilot branches from CE3 and run **16 successful updates per arm**, nominal128/max192 example microbatches each. Each update visits all eight fit businesses once; rotate each business's accepted pairs in a locked order. No early best-loss selection. Save full states at updates8 and16; keep terminal16 as the sole pilot endpoint.
4. Generate the same four TRAIN probes once per final arm with unchanged beam4/top1 settings. Total ceiling: **38 successful updates including gates, 16 generation calls/outputs**, no validation inference. Keep score/margin diagnostics on the fit pairs; preference margins do not certify generated factuality.

Technical collapse criteria remain V2: ≥5 consecutive BOS, blank/special-token-dominated generation or broken state continuity stop; isolated BOS and cap hits are warnings unless collapse is also present. A full, useful review section with an empty zero-caption visual section is valid. Do not erase unsupported assertions by postprocessing.

## 6. Cost, interpretation and stop decision

CE's observed 14,808.9-second/1250-update runtime implies about11.85 seconds per update on that historical host. It is only a planning anchor. Two no_grad scouts plus two recomputed forward/backward passes may cost roughly **2.5–4 times CE** per contrastive update; annotation, IO and hardware vary. Plan **15–45 GPU minutes**, cap active GPU execution at **one hour** including gates/inference, within the separately approved 48-hour window. This is an estimate, not a benchmark. Charge estimate = actual hourly rental rate × active instance hours; do not invent a current Vast price. Pair-source adjudication likely dominates researcher time. Add no model-weight download to this compact review ZIP.

Primary result is numerical/data feasibility, not quality GO. Independently review blinded control-versus-contrastive outputs for all four TRAIN probes against sources: supported Core items retained/omitted, same-aspect disagreement, unsupported assertions, zero-caption compliance, caption usefulness, semantic repetition and usability. Keep the saved N1 outputs as an unchanged descriptive anchor. Include short-but-useful and long-but-factually-wrong cases in the review rubric; do not select by length, preference loss or validation CE.

Proposed practical expansion signal: at least3/4 substantive paired wins, no new serious grounding/zero-caption regression, and no material Core/opinion-balance regression. This is a small-sample resource-allocation rule, **not statistical proof or final quality GO**. Failing to improve after16 updates does not establish a model-capability or contrastive-learning NO-GO. If source qualification or numerical feasibility fails, or no convincing source-grounded signal emerges, archive the attempt and prioritize completing the matched family rather than opening another search.

Any larger training/validation experiment needs a new proposal and authorization. No Phase B/C reranker, beam changes, sampling, Teacher regeneration, large negative corpus, WCE mixing or held-out access. If contrastive continuation later earns independent quality GO, first define the same condition-aware objective, pair-adjudication budget, initialization/continuation schedule and checkpoint rule for RO/U/F. Never substitute only improved F into the primary caption-effect comparison. Shuffled must reuse the exact matched F adapter. Otherwise retain CE3 and complete the established matched RO/U/F/Shuffled evaluation.
