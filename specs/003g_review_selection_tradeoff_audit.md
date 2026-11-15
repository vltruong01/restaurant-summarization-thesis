# C2 v4 tradeoff and candidate-coverage audit

2026-10-06. User authorizes continuing the review of v4 tradeoffs and cross-budget coverage. This stage diagnoses the accepted 80-business pilot; it does not run an expanded selector or freeze model inputs.

## Scope and invariants

Use only the accepted v4 pilot caches, their accepted v3 comparisons and full original English population. Preserve all existing scientific scripts/config/results and train/validation membership. No test tuning, internal truncation, minimum length filter, semantic objective, source-length/allocation change or training. Keep existing rating/time/count comparison and 0.001 tolerances. All counterfactual selected IDs are labelled diagnostic, never exported as final evidence.

## Checks

1. For all 240 v3/v4 pairs, report count and JSD regressions, whether v3 Pareto-dominates v4 in rating/time/count, v3 comparability against the actual v4 quality anchor, and whether the v3 set was in v4's evaluated candidates/frontier. Small absolute JSD is not a new acceptance threshold.
2. For all 160 adjacent-budget transitions, identify count decreases. Check exact fit of the previous selection, presence in the next frontier and comparability. Rerank the next frontier plus the previous selection with the unchanged rule as a diagnostic.
3. On the same 80 businesses, rerank three cached candidate pools: union of all budget frontiers filtered to current exact budget; current frontier plus the accepted current-budget v3 incumbent; and union frontiers plus all feasible v3 incumbents. This is cached candidate reuse, not a fresh v4.1 search. Do not impose monotonic review counts or maximize tokens. Report any remaining count decreases and reintroduced short-review shift.
4. Independently implement Pareto filtering/anchor/choice and verify agreement with v4 on original frontiers and counterfactual pools. Freshly encode every unique diagnostic selected block and recompute rating/time JSD from original full source; verify membership, complete boundaries and budget fit. Hash-check accepted cached evidence before and after.
5. Inspect full-original differences for purposive examples covering largest rating regression, largest time regression and largest count loss, plus a long-review stress case. Record exact IDs actually read and limited observations; no formal human-evaluation or exhaustive aspect recall claim.
6. Extend notebook00 with all comparison tables, cross-budget causes, clearly labelled counterfactuals, full text and future refinement. Run All remains cached/read-only. Preserve its previous executed snapshot before replacement. Update progress/decisions and distinguish scientific validation from suitability for final use.

## Stop condition

Complete the diagnostic review, recommend a concrete next pilot refinement, and stop before expanded v4/full-cohort manifests. Unresolved research parameters stay null. Changes to source length/allocation or primary objectives need approval; ordinary candidate-coverage fixes can be prepared under a separate versioned pilot spec when that next step is authorized.
