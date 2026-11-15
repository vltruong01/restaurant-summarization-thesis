# C2 v4.1 shared feasible candidate pool pilot

2026-10-06. User authorizes the next step after reviewing003g. Run a fresh separately versioned pilot on the exact same80 train/validation businesses, not expanded2,200/final2,300.

## Controlled correction

Preserve full detector-English source, complete reviews/cap30, exact LED review-block tokenization, five source time bins, base2 rating/time JSD, weights0.5/0.5, existing count preference/comparability0.001, sensitivity0/0.001/0.005, budgets3500/4000/4500, seed42 and all v4 candidate/local-search settings. Sampling and tie hash version remains v4 to isolate candidate coverage; the new selector/output identity is v4.1. No new length/semantic objective, forced count monotonicity, exclusion or source allocation.

1. Freshly generate the same v4 uniform/length-stratum/joint-cell candidates. Include accepted v3 sets for every method and exploratory budget, including its previously omitted budget-aware incumbent. Re-encode originals exactly rather than trusting incumbent scores.
2. Execute the existing bounded local discovery for each budget, in ascending budget order, with the unchanged v4 proposal rules and starts. Collect all exact candidate evaluations into one business-specific shared pool. Do not reuse accepted v4 candidates as discovery input; accepted v4 is a comparison only. No additional discovery iterations or recursive refinement.
3. After discovery at all budgets, deduplicate the shared candidates by chronological IDs. For each budget independently, filter exact feasibility, Pareto-filter rating/time/count, and apply unchanged v4 anchor/comparability/count/token-neutral tie rules. Lower-budget feasible candidates therefore remain available at higher budgets. Quality-anchor movement can still change count; no monotonic/global-optimum guarantee.
4. Minimum-cost singleton remains rescue only when no ordinary seed fits. Exclude it from ordinary final shared pools when an ordinary feasible seed exists for that budget. Final count may be1–30 or infeasible; flag<10/<5 without exclusion. All accepted intermediate local states fit their discovery budget. Because the shared pool explicitly includes all exact proposals, a terminal proposal rejected for discovery-budget overflow can later be chosen at a larger final budget. Validate its final fit, record that overflow and distinguish candidate derivation from an accepted local sequence. The inherited `accepted_local_path` field is a derivation trace for the globally chosen candidate; it does not certify that its terminal proposal was accepted inside local discovery.

## Evidence and acceptance

- Exact same80 businesses/40,407 originals, comparison against immutable v3/v4 baselines; no test tuning or final inputs.
- Meaningful tests for shared-candidate coverage, dominating incumbent, source/budget order reproducibility, complete-original fit, token-neutral ties, small source, singleton rescue/infeasible.
- Independent source/hash/original-text joins, exact token/base2-JSD/length diagnostics, retained bins/min/max/flags, frontier nondominance and full shared-pool reconstruction, unchanged baselines, tolerance choices, local paths and summary/paired tables.
- Fresh reversed-source full rerun of all80 businesses. Report regressions/Pareto domination/cross-budget count drops and length shift including short20 fractions, not just aggregate improvements.
- Extend notebook00 with v4.1 comparisons and viewer; execute read-only, inspect added figure, preserve previous executed snapshot. Update progress/decisions/reports with measured outcomes. Final parameters remain null.

Stop after this pilot. Do not expand/freeze/run models/rent GPU automatically. If correction still has substantive coverage problems, report them for further pilot design; do not silently increase source length or alter primary research objectives.

## Accepted pilot evidence

Implemented/validated2026-10-06:80 businesses/40,407 originals/1,200 runs, nine meaningful synthetic tests,16,770 shared candidates/6,892 frontiers/262 derivation moves/720 sensitivity choices, fresh all80 reversed-source full-payload equality. Notebook00 section9.2 and v4.1 viewer executed13/13 cells PASS, five figures/previews with new plot inspected; protected sources unchanged.

Retained means26.525/27.0875/27.175,0 count decreases/v3 Pareto-domination/<10/<5/infeasible, exact ranges959–3495/959–3994/959–4485. MedianBODYratios0.457/0.498/0.503 and selected short20~11–12% show residual shift; accepted v4 still dominates two4500 cases. Final suitability pending. Prior cached003g gains are not fresh outcomes. Reports/validation in `outputs/stage_c/review_selection_shared_pool_pilot_v1/`; no expanded sweep/freeze/model work.
