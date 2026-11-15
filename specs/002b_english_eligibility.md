# Stage B amendment — English eligibility before cohort sampling

**Status:** Validated — English gate, independent verification and notebook PASS  
**Date:** 2026-10-05  
**Environment:** Local CPU, existing preprocessing lock

## Objective and authorization

The user accepts checking the full 10,197 baseline pool before sampling and requests notebook/SDD updates. Require >=10 valid English reviews and >=5 unique locally available associated photos before proposing 2,300 businesses and 2,000/200/100 splits. This narrows original raw-count eligibility; do not force the English-qualified pool to remain 10,197. D-006 records the approved ordering change and operational language settings.

## Inputs and outputs

Inputs: verified original pool CSV/report; raw review/photo files and local photo inventory; current v1 manifests for comparison; `configs/english_eligibility.yaml`.

Outputs only under `outputs/english_eligibility/`: resumable per-review language cache, all-business decisions, passed/failed tables, comparison with v1, proposed seed-42 split and reproducible report/config hashes. Update the existing `02_yelp_2300_dataset_overview.ipynb` to perform this gate before the v1 comparison sections. Preserve all v1 files and canonical v1 split. A proposed split is not a rebuilt physical dataset or finalized Stage C input corpus.

## Detection and qualification

Reuse the existing preliminary language script settings: langdetect 1.0.9, seed 42; strip text; <20 characters => unknown; otherwise classify the first 2,000 characters by top predicted label; only `en` qualifies. No probability cutoff is invented. This is an automatic operational rule, not human-verified language ground truth. Keep review/rating/temporal/token selection pending for Stage C.

Stream the complete raw review file and verify its recorded SHA-256 and all 10,197 baseline per-business raw counts. Recheck unique local photo counts against the verified baseline. For each business, classify reviews in source order until ten English reviews are proved; then skip further language calls but continue raw counting. For any business not reaching ten, classify every available review. Pending batch work must be resolved before skipping any potentially needed review. This exactly decides the >=10 predicate without falsely claiming exact English totals for passed businesses. Save assessed counts, English lower bounds, unassessed counts and exactness flags explicitly.

Worker tasks must be deterministic. Cache is keyed by source/pool/detector-policy identity and review-text hash; config/source changes invalidate reuse. Persist bounded batches for resume. Finished reports can be reused only with matching configuration/source stats and output hashes.

## Sampling preview

After the complete gate passes, sample sorted English-qualified business IDs with fresh `random.Random(42).sample(..., 2300)`, then assign 2,000/200/100 by selection position. Export a proposed split outside the immutable dataset. Compare original cohort failures and membership/split differences. Never silently relabel, overwrite or resample the v1 package. If fewer than 2,300 businesses qualify, stop and report.

## Acceptance

- [x] Every baseline business has a decision; raw full-source hash/counts match prior audit.
- [x] Current unique local photo counts match baseline and all candidates have >=5.
- [x] Every passed business has >=10 observed English reviews; failure exhaustiveness is enforced (zero failures in this run).
- [x] Lower-bound/exact counts are explicitly distinguished; no full English corpus count is invented.
- [x] Proposed cohort has 2,300 unique qualified businesses, exact splits, zero overlap and deterministic reproduction.
- [x] Notebook executes; new report/old-reference scopes are clear; v1/raw files and v1 split unchanged.
- [x] Decisions, master plan and progress logs updated with actual outcome and pending Stage C.

Evidence: `outputs/english_eligibility/eligibility_report.json`, `validation_report.json`, `notebook_execution_report.json`. Full raw scan: 6,990,280 review records, with all 2,566,892 baseline-business counts matching. Assessed 102,103 review languages, including exactly ten `en` observations per business. All 10,197 qualify; zero current-cohort failures. Fresh post-gate draw reproduces the same v1 IDs, positions and split, so a physical cohort rebuild is unnecessary under this unchanged policy. Detection took 191.444 seconds; independent raw/cache evidence verification 56.228 seconds; notebook 12/12 cells PASS in 40.721 seconds, five figure outputs. English totals beyond the observed lower bounds remain unknown; v1 review text is still the raw source pool.

## Remaining work

Language error/manual review diagnostics and exact full English classification of the chosen cohort belong to subsequent input preparation. Freeze representative <=30 reviews and token budgets under spec 003. Image content/readability rules remain Stage D. GPU/model/evaluation design and external/generalization/human-sampling additions remain unchanged.
