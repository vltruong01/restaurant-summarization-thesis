# Actual-caption source-budget freeze audit — 2026-10-09

Authorized audit before K-study; this is NOT production D-032 approval. Keep active budgets TBD until the researcher reviews the results. Audit all200validation businesses, reporting calibration40/K-study160 separately. Do not read held-out review/caption data, generate models/captions, evaluate semantic coverage, compare K performance, select K, or mutate source reviews/captions/filtering rules.

## Single immutable audit-candidate template

Use exact original business strings (missing/null -> empty string), four fields only, in this order:

```text
[BUSINESS_INFORMATION]
name={name}
categories={categories}
city={city}
state={state}

[REVIEWS]
{shared selected review block}

[VISUAL_CONTEXT]
[IMG_CAP_1]
{normalized caption 1}

[IMG_CAP_2]
{normalized caption 2}
```

Sections and caption entries are separated by exactly two newline characters; a marker and its body are separated by one newline. No artificial final newline is added after the final body. Reviews-only stops after the unchanged review block; caption-conditioned sources always include the visual-context header, even when zero eligible captions remain. Markers are literal text, not newly registered tokenizer special tokens. Do not serialize photo IDs or add business aggregate ratings/counts/addresses/attributes/new fields.

Reuse the accepted v4.2 length-balanced4000scenario's exact selected review IDs, text/stars/date/order and frozen formatter: `[REV_i] stars={stars} date={date}\n{complete text}`, double-newline separators, date_then_review_id order. No new selection/search, truncation or review/caption removal for source fitting. The same business metadata/review base is used by Reviews-only, Unfiltered, Filtered and future Shuffled. No shuffled donor assignments are created by this audit; its review contract is checked by the generic serializer, while actual shuffled-source token validation remains required when that control is instantiated.

## Caption constructions: budget stress only

Use max candidate10 ONLY, without coverage/performance comparisons. Unfiltered takes the first up-to10Basic-eligible normalized captions from the already defined fixed-seed order; Filtered takes the first up-to10from the frozen D-031diversity ranking. Existing160K-study rankings must match saved fixed pools exactly. For40calibration businesses construct ranks using unchanged D-031rules and original scores/embeddings; do not retune criteria or report K effectiveness. Do not duplicate captions when fewer than10exist, and do not drop a selected caption to fit a token budget.

## Exact token accounting and provisional limits

Use allenai/led-base-16384 revision38335783885b338d93791936c54bb4be46bebed9 and the verified local tokenizer assets. Encode complete assembled source with BOS/EOS added once, no padding/truncation. Independently check exact token-ID equality against the pinned local slow LEDTokenizer.

Provisional values under audit: full source5000; existing review block4000; common incremental entire caption block750. Caption increment is `tokens(full caption-conditioned source) - tokens(exact reviews-only base)`, including visual header, caption markers, separators and BPE boundary effects. It is not the sum of standalone caption-body token counts. Exact telescoping components are review-block tokens, base-minus-review marginal overhead, and full-minus-base caption increment. Also record standalone component counts and successive full-prefix increments for transparent boundary accounting; never add BOS/EOS twice.

Report min/mean/median/p95/max for caption increments and complete sources, count >750caption increments and >5000full sources, separately for Unfiltered/Filtered over all200and the160K-study role. Report reviews-only base and metadata overhead, both actual fits and declared4000+750overhead envelope. Preserve hash/ID/order/text checks, and export a few complete worst-case sources rather than duplicate the full dataset.

If any required construction overflows, record the cases/components and stop; do not change5000/4000/750, review/caption text/counts, K grid or filtering parameters automatically. If all required constructions have zero overflow, return D-032 PROPOSAL with exact template/overflow policy and evidence SHA for researcher approval. No active budget/template production freeze before approval, no K-study execution in this audit.
