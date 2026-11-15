# C2 v4 exploratory pilot — reduce systematic short-review preference

2026-10-05. User authorizes the next selection revision and requests one notebook combining past/current/planned dataset and filtering work. Preserve v3 and length-audit evidence. Implement a separately versioned pilot, not a final all-2,300 manifest or a new full train/validation sweep.

## Unchanged core

At most 30 complete detector-English originals under exact LED review-block budget; no internal truncation, minimum ten selected, automatic exclusion or source-length increase. Business source eligibility >=10 English and >=5 local photos remains unchanged. Full business English pool, chronological output, seed42, five full-population temporal rank bins, base2 rating/time JSD, weights0.5/0.5, comparability0.001 per metric and sensitivity0/0.001/0.005. Exploratory review budgets3500/4000/4500; ~5000 total source target still includes pending metadata/captions. Semantic selection is disabled; no new minimum review length, quality classifier or token-filling objective.

## Explicit revision, provisional search settings

1. Remove shortest-review seed variants as normal candidates. Generate three independent uniform rating/chronological-stratum draws per cardinality, two joint-star/time greedy draws with uniformly shuffled review order inside each cell, and one rating/chronological seed that rotates through four equal-frequency length strata. Strata sample a review uniformly, never choose the shortest/longest or treat length as content quality.
2. Joint-cell greedy construction still minimizes incremental rating/time cost; its ties use seed-derived priorities independent of token lengths. Every candidate is assembled/tokenized exactly.
3. Incoming proposals cover one seeded uniform available review and one uniformly sampled available review from each of four within-cell length rank strata (deduplicated). If source<=30, expose all unselected originals. No shortest-available default.
4. Candidate, anchor and local-proposal ties after rating/time/count use stable SHA256-derived priorities over seed/version/business/canonical IDs, independent of token length. Preserve the original count preference within declared rating/time similarity. Candidate ranking does not maximize used tokens or semantic scores.
5. Local quality/count proposals and a same-count one-for-one replacement ranking are shortlisted without token-minimizing ties. Exact whole-block feasibility decides acceptance; six iterations, three anchors, four exact proposals per ranking remain bounded exploratory settings.
6. Keep the minimum-cost singleton solely as a feasibility rescue: exclude it from normal search whenever any ordinary feasible seed exists for that budget. If none exists, use it to start additions/replacements. It guarantees that at least one complete original is found if any fits; it is not a shortest-set objective. Record rescue usage.
7. Reuse the two accepted v3 delete-only baselines as feasible comparison candidates without modifying their algorithm/seeds/results. This may carry some budget-related length pressure; report provenance rather than assert bias is eliminated.

This is a combined candidate-generation/tie-breaking revision; comparison is not an ablation of individual mechanisms. It may still prefer shorter subsets when fitting more complete reviews requires it. Bounded search has no global optimum or semantic-quality guarantee. Final method/budget stays pending regardless of pilot outcomes.

## Pilot and comparisons

Union the accepted56 pilot businesses, ten length-audit examples, and four additional seeded train/validation businesses from each source-count band10–30/31–100/101–300/301+ excluding already chosen IDs. Record actual union count, reasons/splits/source hashes. Include flagged/long/short cases; this purposive-plus-stratified pilot does not estimate whole-cohort quality. All v3 comparisons come from the accepted expanded v3 results for identical businesses/budgets.

Report v3/v4 count/tokens/retained bins/<10/<5/infeasible, rating/time differences and per-business regressions, selected/source BODY length ratios and within-source percentiles, short-review fractions, same-count comparisons, tolerance sensitivity and rescue usage. Full-text examples include low-token cases and worse metrics/low-count flags. Length shifts do not demonstrate content quality; record limited qualitative comparison after execution. No frozen model evidence or teacher targets.

## One primary notebook

`notebooks/00_thesis_dataset_filtering_review_selection.ipynb` is the maintained user-facing entry point. It includes raw pool/eligibility/cohort/photo/review statistics, English filtering policy/counts/filtered originals, v1/v3 history, expanded budget and length diagnostics, v4 pilot comparison, one configurable business viewer for originals/language labels/v3/v4 selections, and completed/current/planned stages. Read verified cached outputs; running the notebook must not trigger raw full scans/search/model calls. Old notebooks remain provenance; future work updates notebook00 rather than requiring the user to switch notebooks. Scientific plots exported separately.

## Acceptance

- [x] Versioned spec/config/new scripts; prior accepted scientific scripts/config/artifacts unchanged.
- [x] Meaningful synthetic checks: token-neutral ties, order determinism, full-original boundaries, rare-rating singleton rescue, genuinely infeasible and <10 valid inputs.
- [x] Source/pilot membership and exact original IDs/fields/text verified; only train/validation; independent exact tokens/rating/time/length/flags/provenance checks.
- [x] Fresh reversed-population v4 searches reproduce every pilot case; same-frontier ranking/tolerance checks; baselines agree with accepted v3.
- [x] One consolidated notebook executed with protected sources; scientific plots inspected; statuses/pending stages clear.
- [x] Report/progress/decision/plans updated; no final budget/methodology/manifest, model training or rental.

Accepted diagnostic evidence: `outputs/stage_c/review_selection_length_balanced_pilot_v1/validation_report.json` / `pilot_review_report.md`; notebook00 execution report under `outputs/unified_dataset_inspection/`. Actual union80 (69train/11validation),40,407 originals,960 runs; independent3,763 frontier/420 move/720 tolerance checks and fresh all80 full-payload equality (240 v4 selections). Six meaningful synthetic tests PASS. Notebook00 11/11 code cells PASS, four figures/previews inspected, sources protected; no automatic pipeline execution from notebook.

Same80-business median selected/source mean BODY ratios v3=0.417/0.423/0.429 versus v4=0.808/0.855/0.876 at3500/4000/4500. V4 mean retained24.04/25.59/26.61 versus v3=26.53/27.08/27.15; count10–30 and no<10/<5/infeasible at any scenario. At3500 short20 macro fraction13.53%→1.35% (source1.05%); 31 rating/16 temporal regressions versus v3, reported individually. El Vez3500 retains30, tokens1154→3420, rating/time JSD identical. Longer input does not certify content quality; assistant read15 purposively selected v4 originals across three example businesses, not the full source or formal human evaluation.

Bounded search has two count-decrease cases from3500→4000 and two from4000→4500; independent per-budget candidate coverage and quality-first ranking do not guarantee count monotonicity. Record before final method, do not hide as source failure. Stop after this pilot; no expanded v4 sweep/final manifest/source change.

Execution note: initial80 caches complete (~61.3s); Windows default cp1252 aggregate-read error handled by rerunning unchanged driver with `PYTHONUTF8=1`, reusing all80 caches. Warm aggregate6.776s excludes first compute. Fresh validator uses explicit UTF8;111.8s. No scientific parameter/ID change.
