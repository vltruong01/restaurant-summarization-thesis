# C2 implementation pilot v3 — complete-review search under token budgets

2026-10-05. Authorized by the user's instruction to implement after reviewing revised spec 003 and the 3,500/4,000/4,500 review-token sensitivity proposal. The ~5,000 total-source target and final review/caption/metadata allocation are unchanged/TBD. No all-cohort manifest, semantic selection or test tuning.

## Explicit exploratory implementation conventions

Reuse the historical 48 train/validation businesses, add eight low-volume (<=100 English reviews) businesses ranked by sum of their eight longest C1 original-review lengths. Scan all allowed businesses' metadata to identify these stress cases, never test. Verify exact assembled seven/eight-review lengths after loading original text. Report stress-case membership separately; sample means/low-count percentages are descriptive, not cohort-wide estimates.

Search considers k=1..min(N,30) with three seeds: rating/time seeded samples; cheapest full review within each rating-specific chronological rank segment; greedy joint star/global-time-cell representation with length used for ties. Also include the exact minimum-cost complete singleton, guaranteeing feasibility whenever any complete review fits, even if only a rare-rating review is short enough. Thus length is a cost inside representation groups, not a global shortest-only sampling policy. All original reviews are candidate sources; target proportions include individually unfit reviews.

Implementation correction before acceptance: v2 preliminary outputs are preserved under `outputs/stage_c/review_selection_budget_pilot_v2/`, but its verification was interrupted because the singleton feasibility guarantee was missing. They are not the accepted pilot. V3 changes the algorithm identity/seed, reruns the pilot and independent validation, and is the current evidence.

For every budget retain feasible seeds and delete-only random/rating-time baselines for comparison. Improve up to three feasible anchors (best weighted rating/time, greatest count, best temporal score), with six deterministic local iterations. Consider additions/deletions and replacements across all star/time cells. Per-cell shortlist scans all unselected reviews: two cheapest available and one independently seeded available review. Low-volume populations (<=30) expose every unselected review. Metadata-only scoring/length estimates may rank/prune structural moves; only exact re-tokenization of the entire final-order block certifies a proposed/accepted subset. Evaluate union of four leading proposals per ranking (weighted quality, high count, cheapest comparable-quality); log exact-evaluated proposals including infeasible ones, seed choices, accepted paths and counts of metadata proposals pruned. This is bounded heuristic search, not exhaustive/global optimization.

Retain nondominated candidates under rating JSD, time JSD and count. Rank anchor by 0.5*rating_JSD+0.5*time_JSD. Among candidates no more than **0.001 worse in each individual metric** than this anchor, prefer the greatest count, then smaller weighted cost, exact tokens and ordered IDs. These are explicit exploratory settings, not acceptance thresholds or final research parameters. Also report decisions under tolerances 0 / 0.001 / 0.005 on the SAME searched candidate frontier, separating ranking sensitivity from search robustness. Stable seeds, chronological (date,ID) final order; no minimum ten-review input. Infeasible status if no complete original review fits; no empty valid input.

Output source populations, exact initial references, per-budget new-method plus random/rating-time delete-only baselines, searched candidates/frontiers/move provenance, tolerance sensitivity, retained-count bins and <10/<5/infeasible flags, pre/post rating/time/length, seven/eight longest-review stress diagnostics, ten HTML detail examples, notebook and independent acceptance. Semantic scores null/model TBD. No claim that positive count/JSD alone proves semantic coverage.

The v1 delete-only routine is preserved and used ONLY as a labeled comparator; the new method can replace/add from the full pool. A search seed is an alternative subset, not an executed edit to the reference. Outputs must distinguish reference-to-final set differences from actual accepted local additions/replacements/deletions.

## Acceptance

- [x] All populations/selected originals and no-test membership verified, historical 48 evidence/source/split untouched; eight real stress additions verified against allowed metadata ranking.
- [x] Every valid selected set fits exact configured budget and <=30; no truncation/minimum-ten or automatic business exclusion/design changes.
- [x] Independent diagnostics recomputation verifies original IDs/order/text, pre/post base-2 JSD, token counts, flags/bin aggregates, exact candidate frontier ranking and tolerance choices.
- [x] Six edge tests cover seven/eight long-review overflow, replacements/additions outside initial seed, larger-count preference at comparable quality, rare-rating-only-fit/date ties, determinism and no-review-fits behavior.
- [x] Fresh identical-input algorithm recomputation reproduces IDs/order for all new-method budgets; accepted paths and frontier candidates verified.
- [x] Notebook/figures/ten detail examples/report reviewed for integrity; manual quality audit status explicit (pending); actual <10/<5 figures reported with sample denominator/statuses.
- [x] Stop after pilot, update progress/spec/decisions, no final 2,300-business selection or implicit final budget/model.

## Accepted execution evidence

V3 outputs: `outputs/stage_c/review_selection_budget_pilot_v3/`; `pilot_review_report.md`, `validation_report.json`, `notebook_execution_report.json`, `manual_examples.html`; notebook 05. 56 businesses (47 train / 9 validation), 13,713 complete English originals, 504 runs. Mean retained 26.91 / 27.57 / 27.66 for 3500 / 4000 / 4500; minimum 10; <10, <5 and infeasible each 0/56 per scenario. This is a descriptive stress sample, not cohort-wide certification.

Independent validation PASS: all 504 run diagnostics and 504 fresh deterministic comparisons, 1,935 frontier candidates and 161 accepted local moves; six edge tests PASS. Notebook 8/8 cells PASS with three visually checked figures and protected source metadata unchanged. Ten-example HTML preserves all 546 original example texts/IDs. Seven/eight longest-review stress blocks exceed 5,000 for 15/56 and 21/56 businesses. Pilot 488.333s; four-process independent validator 105.282s.

Report explicitly records rating/time regressions versus deletion baselines, ranking-tolerance sensitivity and possible shorter-review content bias. Manual quality/semantic diagnostic/model and final allocation/methodology remain pending. No selected-input count floor of ten, final all-cohort manifest or source-length/design change. Stage C remains pending beyond this validated implementation pilot.
