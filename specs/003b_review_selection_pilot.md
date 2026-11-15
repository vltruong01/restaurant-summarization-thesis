# C2 draft pilot — rating quotas and chronological segments

Date: 2026-10-05. Status: initial pilot implementation validated; budget/semantic/manual-quality extension pending; final C2 methodology not approved/frozen.

**Revision notice (2026-10-05):** This file preserves the historical v1 pilot definitions/results below. The current C2 specification is the budget-constrained revision in `specs/003_review_selection.md`. Its search considers whole-review additions/replacements from the full population and prefers more reviews at comparable rating/time quality. The v1 delete-only routine, numerical tie rule and checked initial-pilot acceptance below do not implement/validate that new search. Do not rerun v1 as the revised full pilot. After user authorization, revised search and real-budget diagnostics were implemented and validated separately under `specs/003c_budget_constrained_pilot.md`, outputs `outputs/stage_c/review_selection_budget_pilot_v3/`, notebook 05. Final methodology/manifest, semantic diagnostics and manual quality review remain pending.

## Scope

The user accepts reviewing an experimental draft before final selection. Implement a bounded train/validation pilot and random-sampling baseline, not selections for all 2,300 businesses. Preserve source data, photos, split and model design. Follow the pasted draft reviewed in the conversation; supplementary implementation definitions below are explicit provisional choices, not final research decisions.

## Population and pilot businesses

Population = all C1 detector-English original reviews of each business. Pilot excludes test entirely. Select 12 businesses in each of four English-count bands (10–30, 31–100, 101–300, >=301), 48 total. Within each band include stress cases for high 5-star proportion, high rating entropy, rare-but-present negative ratings (1/2 stars), long average review lengths, long/short histories, plus at least two validation businesses and seeded random fill. Break ties by business ID. This is a diagnostic stress sample, not a probability sample for estimating whole-cohort prevalence. Save exact IDs and selection reasons before running algorithms. Eight detail examples, two per band, expose original full review texts, selected reviews and metrics.

## Algorithm definitions

- M = min(N, 30). Hamilton star quotas use integer arithmetic: floors of M*n_r/N, then largest remainders, ties ascending rating. No forced rare-rating minimum.
- Sort each rating population by (date, review_id). For quota q, segment i has ranks floor(i*n/q) through floor((i+1)*n/q)-1. Select one review per segment with independent SHA-256-derived seeds from global seed/business/rating/segment/algorithm version. q=0 selects none; q=n keeps all.
- Random baseline chooses M reviews uniformly without replacement from the chronologically sorted population using a business-derived stable seed. Final order is always (date, review_id).
- Diagnostic time bins use five contiguous approximately equal-count rank bins of the FULL English business population, with fixed membership throughout removals. Same-timestamp ties use review_id; this measures chronological rank coverage, not equal calendar-duration coverage. Also report date ranges and date-span coverage.
- JSD is divergence with log base 2, explicitly computed, no arbitrary PASS thresholds. Rating JSD is a rating-proportion diagnostic, not direct aspect sentiment annotation.
- Review-block template for this draft: `[REV_i] stars=<r> date=<original date>\n<complete original text>`, separated by two newlines; pinned LED tokenizer with BOS/EOS once, no truncation/padding. All block markers/dates/ratings/separators are counted. Caption/business-metadata allocations remain outside this pilot's review-block budget and TBD. This formatting is provisional; future final input construction must use the frozen template or rerun exact token validation.

## Budget and semantic dependencies

Final shared review-token allocation remains null. An optional user question asks whether to run explicitly exploratory 3,500/4,000/4,500-token scenarios or only measure unadjusted selections. Until answered, configured scenario list is empty; no budget is assumed and no adjusted set is exported as final.

Implement a callable removal routine for explicit budgets. Pilot cost is 0.5*rating_JSD + 0.5*time_JSD relative to fixed full-population targets, recalculated for every candidate removal. Equal weights are a recorded draft convention for later review. Tie tolerance = 1e-12 (numerical equality only, not a quality threshold); prefer greater exact assembled-token savings, then review_id. Record every candidate's cost/tokens and chosen removal. Keep at least one whole review; flag infeasible if none can fit, never export empty/claim success/drop a business. No semantic score guides selection. This greedy rule is not claimed globally optimal or guaranteed to preserve all rating/time groups.

Embedding model remains null unless the user answers the optional model question. Semantic centroid and nearest-selected-review coverage stay null with an explicit not-computed status; never substitute zero. If approved, pin the model revision/config, document tokenizer length limits and complete-text chunk aggregation, compute full and selected embeddings, centroid cosine and per-review nearest-selected similarities. These are diagnostics only. No global Python installation or implicit model selection.

## Outputs

Draft config `configs/review_selection_pilot.yaml`; algorithms/pilot runner and independent validator; outputs under `outputs/stage_c/review_selection_pilot_v1/` (population/sample provenance, original pilot pool, initial/explicit scenario selections, diagnostic CSV/JSON, removal traces, identity/hashes, reports); notebook 04 and charts/detail examples. No final data folder, no canonical <=30 manifest replacement, no test selection, no GPU rental.

## Acceptance

- [x] Pilot sample has 48 unique train/validation businesses, 12/band, stress-case reasons and >=2 validation/band; no test reviews.
- [x] Initial selections equal min(N,30); IDs unique/subset; exact Hamilton quotas, one/rating-segment; sorted output; fresh computations/reconstruction reproduce IDs/order.
- [x] Selected original texts/fields match exported English sources and C1 metadata; file/tokenizer/split identities verified; source files unchanged.
- [x] Fresh validator checks base-2 JSD, full fixed rank-bin assignments, exact input token counts and any budget traces; meaningful algorithm edge cases pass.
- [x] Diagnostics and random baseline available before adjustment; explicit budget scenarios only when configured; semantic TBD clearly represented.
- [x] Notebook executes, figures inspected, eight full-text inspection examples saved; no empirical quality threshold or final method claimed.
- [x] Progress/SDD records actual pilot scope/results, pending semantic/budget decisions and overall C2 pending.

## Initial pilot execution evidence

48 businesses (40 train / 8 validation), 13,110 original English review records, 96 initial runs. Independent validator PASS, 6 edge-case tests PASS, notebook 04 8/8 code cells PASS (8.788 seconds), three visually inspected figures, source metadata unchanged. Rating JSD mean: random 0.014898, rating/time 0.001373; temporal JSD mean: random 0.013238, rating/time 0.003366. Rating/time has worse temporal JSD in four businesses; this is reported rather than hidden. Rating/time block median 4,018 tokens, max 9,121. No corpus budget adjustment or semantic model run: optional preferences unanswered, scenario list empty/model null. Full report: `outputs/stage_c/review_selection_pilot_v1/pilot_review_report.md`. Initial pilot implementation validated; extended budget/semantic/manual-quality pilot and final methodology remain pending.

Stop on source/config identity mismatches, duplicate IDs, invalid fields, inconsistent sample/quota/ordering/counts, or ambiguous existing outputs. Pilot output is draft even if all implementation checks pass. Report the pilot before freezing the final methodology.
