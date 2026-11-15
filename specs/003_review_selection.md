# Stage C — Full English corpus, review diagnostics and representative input selection

**Status:** C1/review-selection diagnostics through003k and LED source-capacity audit003l validated; final template/selector/allocations/C2 manifest pending; primary notebook00  
**Date:** 2026-10-05  
**Environment:** Local CPU, validated A1; A2 not required

## Objective and authorization

The user authorizes the next step after validated cohort/English business eligibility. C1 exhaustively classifies all 601,493 source reviews of the fixed 2,300 businesses, calculates exact English counts and LED lengths, exports the full English review pool, and creates a Stage C diagnostics notebook. C2 will freeze <=30 representative English reviews and shared input budgets only after its still-pending rules are recorded. The earlier threshold gate proves eligibility but does not replace exhaustive C1 classification.

## Inputs

Immutable v1 source review partitions and business split; D-006 detector/config and verified threshold cache; pinned LED tokenizer assets/revision/checksums; A1 runtime lock. Source records, images and v1 split remain unchanged. Cohort IDs, selection positions and split are never regenerated.

## C1 methodology

- Reuse langdetect 1.0.9/seed 42, stripped length >=20, first 2,000 characters, accepted top label en, no confidence cutoff. Only language probing is stripped/truncated: exported review text and tokenization retain the complete original text.
- Classify every cohort review, reusing prior labels only with matching source ID/business/text hash and exact detector policy. Cache new language/token metadata in a separate resumable SQLite database with a pinned scientific-input identity.
- Tokenize each complete original review with the local fast LED tokenizer; no truncation/padding. Save counts without special tokens and the explicit number of added BOS/EOS tokens. Review-length sums are diagnostics, not exact token counts of a concatenated marked model input.
- Stream all three source files; check each byte SHA-256/count, unique review IDs, business/split/text/stars/date fields, and unchanged source stats. All records have one language/token metadata row.
- Export all English original records to a separately versioned review pool. Keep unknown/non-English metadata separately for inspection; do not silently delete from the immutable source.
- Save exact business/split/star/year counts and token quantiles; language-error examples and diagnostics do not claim human annotation. Do not tune selection parameters on test data or model outcomes.

## Outputs

- `outputs/stage_c/review_diagnostics_v1/`: resumable metadata DB, progress, scientific/config identity, review-level CSV, per-business/rating/year/token diagnostics, hashes, completion/validation reports and figures.
- `data/processed/yelp_2300_english_v1/`: original-record English review JSONL by split, identical business split, metadata/provenance/checksums. Photos remain in v1 and are referenced explicitly; this is not a finalized <=30-review/LED training dataset or full GPU handoff package.
- `notebooks/03_english_review_selection_diagnostics.ipynb`, executed outputs; progress/decision/SDD updates.

## C2 — Budget-constrained representative selection (revised 2026-10-05)

### Authorization and execution boundary

The user requested this specification revision and a review of the updated logic **before a new full C2 pilot**, then instructed implementation. The revised search and explicit exploratory conventions are recorded in `specs/003c_budget_constrained_pilot.md` and `configs/review_selection_budget_pilot.yaml`; implementation/pilot acceptance is reported separately. Existing pilot v1 remains historical initial-selection evidence. Its delete-only adjustment routine is a labeled comparator, not the revised search. No final selection manifest for all 2,300 businesses may be frozen from this exploratory pilot.

### Population, eligibility and cardinality

For each business let R be its complete C1 detector-English review population and N=|R|. Rating targets and fixed temporal diagnostic bins are calculated from R, including reviews that are individually too long to fit a configured budget. Do not redefine the target population as only short/individually-fitting reviews.

- Business eligibility is N>=10 English reviews plus >=5 associated unique local photos BEFORE review selection.
- **30 is an upper bound, not a required final count.** A valid final input may contain 7–8 reviews, fewer than 5, or another positive count <=min(N,30), provided the configured budget is respected. There is **no requirement for >=10 selected reviews**.
- N<=30 does not bypass the token constraint. Keeping all reviews is permissible only when the exact assembled input fits.
- A valid review input contains at least one complete review. If no complete review fits, flag `budget_infeasible`; preserve the business and record the case, without exporting an empty valid model input or changing eligibility/design automatically.

### Revised formulation

**Select the most representative set of complete reviews, with at most 30 reviews, subject to the configured review-token budget.**

Candidate S is feasible only if S is a subset of R, its IDs are unique, 1<=|S|<=min(N,30), and `T_LED(format(S)) <= B_review`. T_LED is exact pinned-tokenizer length of the configured chronologically ordered review-block template, including its markers/separators/rating/date fields and the documented special-token convention. No word/character estimate or sum of individual review lengths can certify feasibility.

This supersedes treating one initially selected 30-review set as the only search pool and repeatedly deleting until it fits. The unconstrained M=min(N,30) selection remains a BEFORE-budget diagnostic reference/seed, not a mandatory final size or restriction on replacement candidates.

### Objectives and comparison rules

1. Primary objectives: preserve the full English population's 1–5-star distribution and temporal coverage; fixed seed/config must reproduce IDs and order. Rating JSD (base 2) is a rating proxy, not a direct aspect-sentiment label.
2. Review count is a secondary objective: among subsets with similar rating AND temporal representativeness, prefer more complete reviews. A smaller set of long reviews must not win solely because it was sampled first when a larger set of shorter reviews has similar or better rating/time representation.
3. Clear dominance: if A has no worse rating JSD and no worse temporal JSD than B, and has at least as many reviews, with one strict improvement, prefer A. Candidate sets with opposing tradeoffs are not automatically equivalent.
4. “Similar” requires explicit per-metric tolerances/comparison rules in a draft config before executing candidate ranking. Quality-equivalence tolerances are **TBD**; the old pilot's 1e-12 numerical tie tolerance is NOT a substantive similarity tolerance. Rating/time tradeoff weights or ordering are also explicit draft decisions, not silent defaults. If these remain unresolved, diagnostics may show the nondominated candidate alternatives, but may not declare one the final selected input.
5. Review length is a budget cost, not a standalone quality objective. Do not globally prefer the shortest reviews, prioritize long reviews, drop latest/oldest/longest blindly, or claim a mathematically optimal solution from a heuristic search.
6. No direct semantic-diversity optimization. Semantic centroid cosine and nearest-selected coverage, if an embedding model/config is enabled, are diagnostics only; otherwise report null/not-computed. More reviews do not by themselves prove semantic/aspect coverage.

### Proposed draft search logic to review before implementation

1. Compute full-population star proportions and fixed full-population temporal bins/date ranges once. Build the unbudgeted rating/time reference and random reference for before/after comparison, each with M=min(N,30).
2. Consider candidate cardinalities k from 1 through min(N,30). Hamilton quotas for k and chronological segments provide search targets/initial seeds; they are not unconditional hard quotas if the token constraint makes them infeasible. Never require one review from every rating or temporal bin.
3. Search with whole-review additions, deletions and replacements using the full R candidate pool, not only the reference set. In particular, evaluate replacing long reviews with shorter reviews of comparable rating/time placement and using released token space to add further complete reviews. Also allow other documented moves so a restricted same-bin replacement rule does not silently exclude better feasible alternatives.
4. For every candidate comparison/move, sort by (date, review_id), rebuild/re-tokenize the entire formatted block, enforce <=30 and the explicit budget, and recalculate rating/time diagnostics against the original fixed targets. Never edit/truncate text inside a review. Log proposed/accepted removals, replacements and additions and their exact token/metric effects.
5. Compare feasible candidates using primary rating/time representation and the declared similarity-plus-more-reviews rule. Keep nondominated alternatives for diagnostic review. Fixed SHA-256-derived seeds and deterministic tie resolution must not depend on business processing order.
6. Bounded-search strategy, number of starts/iterations, similarity tolerances and rating/time comparison rules must be documented before the revised pilot. No exhaustive/global-optimality guarantee is assumed. The review-block template and candidate-search settings remain draft until pilot review.
7. Export pilot selections/alternatives and diagnostics only. After reviewing the pilot, and only after final rules/allocations are explicitly recorded, create one frozen manifest shared by Reviews-only, Unfiltered-caption, Filtered-caption and Shuffled-caption conditions. Do not independently reselect per condition.

### Budget allocation and unresolved dependencies

`B_review` remains **TBD**. Approximately 5,000 LED tokens is the planned OVERALL source target, not automatically the review budget. Business metadata, structural markers, captions and special tokens need an explicit shared allocation/accounting convention. The same fixed review evidence must fit every condition's reserved source space. Do not double count or omit markers/special tokens across blocks.

No real-budget enforcement results may be reported when no budget is configured. Any sensitivity budget must be explicitly designated exploratory; it does not lock final review/metadata/caption allocations. Exact total/review/metadata-marker/caption budgets, candidate-search rules, rating/time tradeoff/similarity rules and embedding identity remain recorded dependencies. If a solution requires changing source length, allocation or another research-design parameter, record the proposed change as TBD/blocked for that decision and **stop for user approval**; do not alter it automatically. This is a user-imposed design-change gate, not a reason to pause authorized documentation work.

### Required per-business diagnostics

Save for each method/scenario and any proposed final pilot candidate:

- `initial_eligible_review_count` = N and `candidate_pool_count` = N.
- `initial_candidate_count` = M of the explicitly labeled unbudgeted reference; it does not limit the search to those M IDs.
- `final_retained_review_count`, exact `tokens_before_budget_enforcement` for that reference and `tokens_after_budget_enforcement` for the feasible candidate.
- Rating JSD and temporal JSD BEFORE and AFTER budget enforcement, full/selected star/time distributions, fixed-bin identity and original/selected date ranges/span coverage.
- Semantic centroid similarity/nearest-selected coverage when enabled; null/not-computed otherwise.
- Selected IDs/order; removed IDs, replaced-out/replaced-in ID pairs, added IDs, reasons and exact token/representation effects. Distinguish differences versus the reference from the step-by-step search trace.
- `below_10_retained_flag`, `below_5_retained_flag`, budget/method/model/tokenizer/seed/algorithm/config identity, selection status and whether methodology is frozen (false throughout pilot).

Flag **every case with fewer than 10 final retained reviews**, including 7–8-review inputs, for human review. This flag is informational, not an exclusion rule or an instruction to increase input length/truncate/change methodology. Also flag fewer than 5. For an infeasible business, report zero diagnostically with `budget_infeasible`/no valid model-input export; distinguish it from feasible low-count cases in all aggregates.

### Revised pilot and aggregate report

Use train/validation only. Preserve the initial 48-business diagnostic sample/evidence; deliberately include unusually long whole-review cases, especially low-volume businesses that exceed an explicitly configured budget with only 7–8 reviews. Identify them from full individual lengths PLUS exact assembled blocks; add/version stress examples if the prior sample does not expose this case. Include a deterministic synthetic long-review regression case if the real pilot pool lacks a particular extreme, clearly separated from real-corpus results. Do not tune using test businesses.

For each configured budget/method, report retained-review distribution: **25–30, 20–24, 15–19, 10–14, 5–9, <5**. Include zero-budget-infeasible cases with an explicit separate status/count, state denominators, and separately report counts below 10, below 5 and infeasible. Compare rating/time representation before/after enforcement and changes in retained count. No arbitrary JSD/semantic pass/fail thresholds.

Provide **5–10 detailed real-business examples**, including long-review and flagged low-count cases when present, with original/reference/retained text and removed/replaced IDs. Do not claim manual aspect/opinion audit from merely generating the example artifact. Stop after the reviewed pilot report and update root/SDD `THESIS_PROGRESS.md`; do not freeze final selections for all 2,300 businesses.

### Revised C2 pilot acceptance (validated; final methodology pending)

- [x] New search spec/config decisions reviewed before implementation/full pilot; final review allocation still explicitly distinguished from ~5,000 total source tokens.
- [x] Every valid candidate has unique complete original reviews, <=30, exact configured block-token feasibility; no artificial >=10 selected-review requirement.
- [x] Search tests cover long-review 7–8-review overflow, full R replacements/additions, cardinality preference under declared comparable quality, rare ratings/date ties, determinism and infeasible no-empty behavior.
- [x] Before/after star/time diagnostics, candidate counts, exact lengths and removal/replacement/addition provenance independently checked against originals.
- [x] Every <10 and <5 case flagged and reported; retained-count histogram/denominators/infeasible statuses correct; no automatic exclusions or design changes.
- [x] Real long-review cases deliberately included; ten detailed examples available and manual quality review status explicit (pending).
- [x] No test tuning, semantic-diversity optimization, implicit budget/model choice or final 2,300-business manifest; report and progress updated after pilot.

Evidence: `specs/003c_budget_constrained_pilot.md`, `outputs/stage_c/review_selection_budget_pilot_v3/validation_report.json` / `pilot_review_report.md`, notebook 05 (8/8 cells PASS). 56 train/validation businesses, 504 runs and fresh reproducibility checks; retained mean 26.91/27.57/27.66 for exploratory 3500/4000/4500 review blocks, <10/<5 each 0/56. No cohort-wide claim. Manual aspect audit, semantic diagnostic model, final allocation/methodology and all-cohort manifest remain pending; overall Stage C is not complete.

Subsequent user-authorized expansion is accepted under `specs/003d_train_validation_budget_diagnostics.md`: all 2,000 train + 200 validation businesses, 575,749 original English reviews and 19,800 runs, no test tuning. Unchanged scientific settings; new-method retained means 28.27/28.43/28.51, count 10–30 and <10/<5/infeasible zero per exploratory budget. Exact selected block ranges 842–3500 / 842–3999 / 842–4492, original body 4–1966. Full-run independent integrity verification and 288 fresh searches on 32 cases PASS; notebook 06 8/8 cells PASS. Evidence at `outputs/stage_c/review_selection_train_validation_v3/`; manual content review, semantic model/scores, final allocation/methodology and final all-2,300 manifest remain pending.

## C1 acceptance (validated)

- [x] Exactly 601,493 unique review metadata rows; full input hashes/counts and original split match.
- [x] Every business's exhaustive English count meets >=10; English/non-English/unknown partition is complete and exact.
- [x] Exported English text/fields/IDs/order match original records and language decisions; split overlap zero.
- [x] Pinned tokenizer assets/hash verified; sampled independent re-encoding and cached/recomputed language labels agree.
- [x] Corpus/business/star/year/token reports match exported/metadata counts; no sum is mislabeled exact model-input length.
- [x] Notebook executes with meaningful tables/figures; raw/v1/split remain unchanged.
- [x] Documentation distinguishes C1 validated from C2 and overall Stage C pending.

## C1 execution evidence — 2026-10-05

Complete labels: **600,862 en**, **597 other languages**, **34 unknown**; these are detector results, not human ground truth. All 2,300 businesses retain >=10 English reviews (minimum 10). English exports contain 529,631 train / 46,118 validation / 25,113 test reviews with the unchanged 2,000/200/100 business split. Reused 23,031 source-matched prior labels; tokenized every complete original review. Eight CPU workers finished classification/export in 784.519 seconds.

Independent validation passes every original ID/field/text hash/export order and file hashes, language/star/year counts and business token sums/maxima, plus 256 fresh token encodings and 32 fresh language checks (18.885 seconds). Notebook 03 passes **8/8 code cells**, four visually checked figures, 17.401 seconds; protected raw/v1/split/English-pool metadata unchanged. Reports: `outputs/stage_c/review_diagnostics_v1/{completion_report,validation_report,notebook_execution_report}.json`. Scientific identity: `15fcc5144643f7eddc211677c4005a31edfab537eda248a645d5a956bbaef480`.

C1 acceptance is complete. C2 and overall Stage C remain pending; no selected <=30-review manifest, exact marked business input or final budget has been frozen.

User-authorized v4 revision under003f changes candidate sampling/token-neutral ties while preserving primary rules. Validated80-business pilot reduces length shift but has count/rating/time tradeoffs; no expanded v4/final freeze. See `outputs/stage_c/review_selection_length_balanced_pilot_v1/pilot_review_report.md`. One maintained user-facing notebook00 consolidates all completed/current/planned dataset/filtering work and complete-original business viewer; future stages update it, old notebooks remain history.

Latest follow-up under spec 003e: selected/source length audit on all 2,200 train/validation confirms a systematic short-review shift (median mean-length ratios ~0.383 across the three budgets). 199 exactly-30 inputs under1500 tokens per scenario; feasible longer same-star/time-bin counterfactuals exist for all five lowest-token examples. Notebook 07 and independent diagnostic validation PASS; purposive assistant reading is not full source-aspect/human evaluation. See `outputs/stage_c/review_selection_length_audit_v1/length_content_audit_report.md`. Prior integrity/token/rating/time acceptance remains valid but does not accept selected-input content quality. Review candidate construction/token tie-breaking before a new pilot/final manifest; no methodology or source/allocation change made here.

Subsequent C2 work: `specs/003b_review_selection_pilot.md` records a bounded initial pilot on 48 train/validation businesses with 96 random/rating-time runs. Initial implementation/integrity and notebook outputs are validated. Real-corpus budget adjustment, semantic model/diagnostics and manual quality review remain pending; this does not freeze final C2 choices in `configs/research_pending.yaml`.

## Stop conditions

Source/config/cache identity mismatch, invalid joins/IDs/records, tokenizer/hash mismatch, <10 English reviews in any fixed business's SOURCE population, export/count mismatches or existing final output with another identity. A valid budget-constrained selected input with <10 reviews is a review flag, not this source-eligibility failure. Preserve old artifacts and report failures. Do not invent C2 research parameters to bypass a pending decision. Respect the user's review-before-new-pilot and approval-before-required-design-change gates.

## Latest accepted diagnostics - 2026-10-06 (003k)

User-authorized expanded v4.2 verification covers all2,000 train/200 validation businesses and575,749 original English reviews. Every003j scientific setting unchanged; missing2120 v4/v4.1 witnesses generated under frozen rules and accepted80 expanded payload equality verified. All39,600 runs/6,600 primary histogram pairs/1,041,950 independent checks PASS;32 precommitted additional reverse-source current-v4.2 payload replays equal with fixed witnesses. No claim of all2200 fresh replay. Means28.287/28.437/28.518, count10-30,0<10/<5/infeasible; block842-3500/842-4000/842-4500. Refined median BODY ratios0.765/0.880/0.938 and5,628 strictly improved lengthJSD pairs; no baseline primary Pareto-domination, no content-quality proof. Four3500-to4000 count drops already occur in primary control: previous sets fit/remain in higher candidate pool/frontier but fail comparability to a new rating anchor. No automatic design change. Before/after per-business rating/time regressions and residual3500 length shift are reported. Notebook00 section9.5/global original viewers16/16 cells/eight figures PASS, previous seven figures/evidence unchanged; eight cases/30 actual original reads. Evidence: `outputs/stage_c/review_selection_source_length_expanded_v1/expanded_review_report.md`, validation/aggregate acceptance reports and spec003k. Final source allocation/selector/2300 manifest remain pending; planned~5000 total includes metadata/caption space. Test100 excluded, no BLIP-2/teacher/train/rental.

## Source allocation audit accepted - 2026-10-06 (003l)

All2200 train/validation accepted v4.2 review sets retained;6600 original marked block counts and19800 provisional metadata/base source rows freshly encoded and independently verified. Special pair once/full string; no review-marker double count. Three audit-only metadata profiles, no real captions. Name/categories/city/state additional overhead median51/max115; minimum entire-caption incremental capacity1421/917/414.4000+750 fits2200/2200 measured cases with maximum allocation scenario4833/conservative audited bound4865, not adopted/frozen.4500+750 overflows1040 scenarios without changing reviews. Protected inputs/final fields unchanged; no test tuning/model/API. Two serialization edges/103047 selected BODY originals/41804 whole strings/441 summary rows/nine complete examples PASS. Notebook00 section9.6,18/18 cells/nine figures with prior8 unchanged. Final metadata/template, selector, source/review/caption allocations and2300 manifest still pending; actual captions and full condition-source re-encoding/feasibility needed. Evidence at outputs/stage_c/led_source_token_allocation_audit_v1/source_allocation_report.md and spec003l/D-020.
