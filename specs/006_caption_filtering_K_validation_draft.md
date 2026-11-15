# Stage F / K - proposed validation protocol

Status: **PROPOSED, NOT ADOPTED OR EXECUTED**. User submits a validation-only protocol. Candidate set and selection rule below are a concrete draft for approval; K and all production thresholds/budgets remain unresolved. No new model calls, caption filtering or teacher/training/rental.

## Preconditions and data boundary

Basic Quality is already adopted under D-026. Define and freeze the CLIP/relevance/near-image-duplicate/redundancy/grouping/diversity models, versions, preprocessing, algorithms and thresholds before the K sweep. Threshold development/selection uses validation only under the implementation plan, with its own predeclared search/report. Candidate threshold grids and the criterion for picking them are still TBD; do not invent them from this protocol.

Also hold review IDs/order/whole texts, metadata/source template, common caption-token allocation, full-source ceiling, caption model/prompt/retry/normalization and seed fixed. Those budget/serializer/review-selector choices must be settled for the study; current4000/750/approximately5000 are exploratory evidence, not automatically adopted production allocations. Separate threshold-freeze and K-selection receipts/config hashes.

Use full eligible caption pools for the200 validation businesses. There are2504 original validation photo IDs (before filtering),5-110 per business. Current actual E1 pilot covers only4 validation businesses/12 photos at3 photos/business: it cannot assess K5/8/10. Do not cap generation at10 images, because that would alter the reference denominator and hide saturation. Image/caption exclusions follow fixed Stage F rules; preserve all original records and reasons. The held-out test is not used for tuning, manual threshold decisions, reference groups or K results.

## Fixed candidate comparison

Proposed candidates are exactly `{3,5,8,10}`. Never modify the candidate set after seeing held-out test results. For each business, compute all non-K filter decisions, complete eligible filtered pool and semantic-group IDs **once**; freeze these across K. Define grouping model/revision/algorithm/threshold and the point relative to redundancy reduction explicitly. Reference groups are those available in the final eligible filtered pool before K or token pruning. No K-dependent reclustering or denominator.

If redundancy pruning leaves one representative per semantic group, coverage may largely reduce to a caption/group-count fraction; disclose this and report group-size/pool-size distributions. Coverage is a semantic-group proxy, not proof of factual accuracy, all useful visual information, or better summarization. Keep image-text consistency, unsupported details and filter failure diagnostics separate.

Filtered selection uses one deterministic diversity ranking per business and nested prefixes for increasing K; retain stable group/photo IDs and tie rules. Unfiltered uses only Basic Quality and one fixed-seed ordering per business, also nested prefixes. Same maximum K and common caption-token allocation in both conditions. Do not add semantic filters to Unfiltered or tune its ordering to improve coverage.

If a pool has fewer than K captions, retain all available; never duplicate/fabricate or lower thresholds. If no filtered captions/groups remain, coverage is NA and the business stays in the original cohort. Report the empty-pool rate and keep it in budget/retained-count reports; zero-caption source/teacher/visual-target handling remains a separate TBD before production, never an automatic business exclusion.

## Coverage and selection

For business b with fixed available group count G_b>0:

`coverage_b(K) = distinct fixed reference groups in selected captions / G_b`.

Compute C(K) as the **macro mean over the same positive-group validation businesses for every K**. Report empty-pool count separately, coverage distribution, absolute C(K), and sensitivity by available-group/pool size; low-capacity businesses can saturate early and raise the aggregate. Report the >=10-group subset separately when nonempty, without replacing the primary population.

`C_best = max(C(3), C(5), C(8), C(10))`.

Eligible K must satisfy `C(K) >= 0.95 * C_best` and the agreed token constraints in **both** caption conditions. Choose the **smallest eligible K**. Smaller K is the primary criterion; lower redundancy/token cost only break ties between equivalent candidate subsets at the same K. Putting redundancy before K would contradict the stated smallest-K rule and is not this draft.

Relative95% means95% of the best score within these four candidates, not95% absolute coverage. Example: C_best=97%, threshold=92.15%; the proposed74/91/96/97 table selects K8 if budget-feasible. If C_best is55%, the cutoff is52.25%; report this limitation instead of claiming95% of all visual information.

Keep the relative reference over all four candidates even if one is budget-infeasible. If no feasible candidate reaches the coverage rule, stop with a TBD; do not silently redefine best over only feasible K, lower95%, increase source length or add candidates. Handle all-empty pools/no meaningful coverage similarly. No K selected from the current3-photo pilot.

## Redundancy, tokens and outputs

Before execution, declare the remaining-redundancy metric and threshold. A possible fixed metric is the proportion of selected caption pairs above the frozen semantic-redundancy threshold (0 for fewer than2 selected captions), plus mean/max pair similarity; representation/model/cosine conventions still need specification. Report group counts and whether hard redundancy reduction makes this metric trivially0 for all K.

For each validation business/K/condition report original and basic/filtered pool counts, available/represented groups, coverage, pairwise redundancy, retained caption IDs/count, <K shortage, empty status, normalized caption BODY/block tokens, exact full LED source tokens, common-caption/full-source overflow, and non-K config hashes. No internal caption/review truncation or unrecorded removal to make K look feasible.

Proposed feasibility screen: zero sources over the final agreed limits in either condition. Approximately5000 is diagnostic until adopted; an allowable overflow rate other than0 or caption-removal/packing policy would need a separately explicit rule. An unfiltered random subset can cost more tokens than the filtered subset, so checking Filtered alone is insufficient.

Validation sensitivity table includes every candidate, C(K)/relative-best, redundancy, retained counts/shortages/empty rate, caption/full-source token mean/p95/max and overflow counts for both conditions. Raw-photo shortages before filtering: K3=0,K5=0,K8=86,K10=105 out of200 businesses. These are not usable-caption counts and cannot substitute for the required scores.

After actual validation and acceptance, freeze selectedK, thresholds, group/diversity definitions, seed/source/budget protocol and manifests **before processing/evaluating held-out test**. Report all candidate sensitivities in the thesis; only selected frozen K in main experiments. No automatic cohort/split/test-image policy/3-model4-condition/teacher/external/generalization/human-evaluation change.

Machine draft: `configs/k_validation_protocol_draft.yaml`. Metadata-only evidence: `outputs/stage_f/k_validation_protocol_draft_v1/validation_inventory_report.json`. Active Basic Quality: [spec005b](005b_basic_quality_validation_pilot.md). Authority: [Stage F implementation plan](../thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md).
