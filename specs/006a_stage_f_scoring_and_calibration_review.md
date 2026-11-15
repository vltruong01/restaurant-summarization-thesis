# Stage F scoring, calibration roles and diversity - concrete proposal

Status: **FOR USER REVIEW, NOT FROZEN OR EXECUTED**. User requests review of a definitions-first/full-validation-pool/calibration/diversity/K workflow and explicitly identifies business relevance as an unresolved research decision. No scorer model is chosen, downloaded or run by this proposal. Original200 validation businesses,2504 photo IDs and all existing split/image/basic policies are unchanged.

## Proposed workflow

1. Review/freeze model/scoring definitions, relevance scope/prototypes, redundancy/diversity/coverage algorithms, internal validation role and human-label sampling protocol. Diversity algorithm is fixed here, not after viewing K.
2. Generate complete2504-photo validation caption pool using unchanged COCO/beam3/D-026. Keep original raw/normalized/token/EOS provenance and all failures. No train/test generation at this stage; no generation cap at10 images/business.
3. Compute frozen model embeddings/scores and calibrate thresholds using human labels from only the calibration role. Predeclare threshold grids/selection metric/sample sizes/ties before labeling; currently stillTBD. Human checks are researcher-provided labels, not the assistant's earlier qualitative observations.
4. Freeze thresholds/group definitions/quality ordering and source/review/budget protocol. Do not inspect K-study sensitivities and return to change thresholds/scorers. Label/score coverage inadequacy or empty pools requires a documented new protocol before any test processing.
5. Run K={3,5,8,10} on the K-study role with frozen nested rankings/reference groups, and screen budget in both caption conditions. Select smallest feasibleK with>=95% of best candidate mean coverage; freeze before held-out test.
6. Apply unchanged frozen rules to train, then approved held-out test processing/evaluation and main3-model/4-condition experiments. ProductionA2/GPU handoff/rental, finalK/budget/sharedmanifest remain separate dependencies.

## Model candidates and formulas for review

**CLIP consistency candidate:** `openai/clip-vit-base-patch32`, exact commit/software/preprocessing/dtype to be pinned only after approval. Score `s_clip(image, caption)=dot(L2normalize(image_features), L2normalize(text_features))`, using checkpoint preprocessing on original EXIF-corrected RGB and normalized caption. Use the unscaled pair cosine, not pool-dependent softmax. CLIP supports image/text feature comparison; this proposal is not a validated restaurant-specific consistency classifier or OCR/factual verifier. Freeze complete-text handling: CLIP has a77-token context; record overlength and decide a fixed policy before scoring, never silently truncate or treat inability to score as irrelevance.

**Semantic embedding candidate:** `sentence-transformers/all-mpnet-base-v2`, normalized caption embeddings and cosine, shared by redundancy/diversity and the relevance candidate below. Pin exact model revision/software/precision/normalization after approval; this is not a claim of best restaurant performance or local GPU validation. Suggested FP32 scores/raw storage and six-decimal decision/ranking quantization need review.

Primary API/model references: [OpenAI CLIP](https://github.com/openai/CLIP), [CLIP checkpoint card](https://huggingface.co/openai/clip-vit-base-patch32), [Sentence Transformers model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [cosine semantics](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html). Sources document embeddings/APIs; the domain scoring/threshold/calibration proposal is our research design, not an official validated recipe.

## Business relevance - decision required

Suggested scope is **restaurant/bar/cafe visual-domain relevance**, not proof the depicted content belongs to the exact named business. Original photo-business mapping supplies dataset association; an embedding score does not verify brand/location identity. Await the user's scope preference. If exact business identity is required, define additional evidence/rubric and scorer in a separate review; do not silently substitute a broad-domain score.

Candidate A (recommended for broad scope): use the same SBERT caption embedding against the fixed six positive and six negative prototypes in configs/stage_f_scoring_review.yaml. `relevance_margin=max cosine(caption, positive prototype)-max cosine(caption, negative prototype)`. Threshold is calibrated on labeled calibration captions; the margin is not a probability. No review-text similarity, arbitrary category-name matching or business-specific prompt edits. Menus/screenshots, food/drinks, interior/seating, venue exterior/signage and service/patrons can be relevant. Generic street/personal/device scenes without a dining connection can be irrelevant. Grounding is a separate CLIP/human-consistency label; relevance alone cannot correct unsupported food/brand details. Prototype bias/sparse negative labels must be tested, not assumed away.

Candidate B for review: use the same CLIP image encoder with a frozen positive/negative dining-domain text prompt set to classify visual-domain relevance via a cosine margin. This scores image context rather than caption context and may retain caption text with little relevant information. Do not run A and B and pick one after K results; choose the intended definition/scorer first. Neither verifies exact business identity. No third model is automatically introduced.

## Quality, redundancy and deterministic diversity

Define quality order explicitly: highest CLIP consistency, then highest relevance margin, then ascendingphoto_id. This avoids unspecified best-quality scores or new mixing weights; score precision/tie convention must freeze.

Proposed redundancy reduction: sort by fixed quality order; keep a caption iff its cosine to every kept representative is below the calibrated redundancy threshold. Map each removed caption to its highest-similarity kept representative (stable ID ties) and preserve all records/reasons. This avoids automatically deleting entire transitive connected components; clustering/grouping and pairwise reduction are not interchangeable.

Proposed farthest-first: start at best quality candidate; next caption maximizes its minimum distance to selected captions, with `distance=1-cosine`. Ties use quality order/ID. Compute one full deterministic ranking per business, use nested prefixes for K. No MMRlambda/new diversity threshold. This is a reproducible heuristic, not proof of optimal coverage; residual semantic outliers can be selected despite upstream filters and require calibration diagnostics. Basic-only Unfiltered retains fixed-seed ordering, not this quality/diversity filter.

## Critical coverage decision

If the reference semantic groups are the same groups reduced to one representative each, every retained caption represents its own group. Then `coverage(K)=min(K,G)/G` regardless of farthest-first order. This primarily measures how many nonredundant groups are retained; it cannot demonstrate that the diversity algorithm chose a better subset.

Option A: explicitly accept/report count-based nonredundant-group coverage and avoid claiming diversity quality from it. Report an additional fixed visual/content coverage diagnostic if desired, without changing the K criterion after results.

Option B: separately define **coarser reference semantic groups** (same embedding, grouping algorithm/criterion/threshold frozen and calibrated before K), while redundancy removal uses its stricter duplicate-information criterion. Multiple nonduplicate captions may belong to one broader topic; group coverage can then distinguish selection sets. This adds a grouping definition/calibration parameter, even though farthest-first itself adds no lambda. Annotate broad-topic sameness separately from duplicate information if this option is selected. Avoid opaque transitive chaining or group definitions tuned after K. User decision is pending; do not silently choose OptionB or fabricate group labels.

Macro coverage uses the same positive-group K-study businesses for all K, zero-group casesNA plus reported empty rate, no originalbusiness exclusion. Relative95% is relative to best of four candidates, not95% of all visual information. Smallest eligibleK is primary; redundancy/token costs only tie-break equivalent sameK subsets. The supplied smallestK sentence and redundancy-before-K tie list conflict; this draft retains the smallestK rule.

## Internal validation roles and human calibration

Proposed40 calibration +160 K-study businesses, business-disjoint, deterministicseed42. Ten calibration businesses per raw photo-count band5-7/8-9/10-19/20+ is feasible: available counts86/19/65/30. Roles are not assigned/frozen yet. Check food/drink/inside/outside/menu representation before any human labels (validation metadata has only10 menu photos), save selectedIDs/provenance and keep whole businesses in one role. Audit exact visual duplicates across roles; business-disjoint does not guarantee image-disjoint. Define handling or report limitations before claiming an independent internal comparison; do not alter original2300 splits/image pools automatically.

Human label rubric: consistency={consistent,inconsistent,uncertain}; relevance={relevant,irrelevant,uncertain}; redundancy pairs={same_information,complementary_information,uncertain}; if OptionB, broad-topic pair/group label separate. Use original images and raw/normalized captions; blind scorer numbers and role/K outcomes in annotation view. Caption/pair sample sizes, candidate score strata/seed, candidate threshold grids, optimization criterion/class balance and uncertain-label handling still need review. Sufficient negative/positive coverage is a prerequisite; sparse labels are not permission to pick arbitrary cutoffs.

No caption/pair from K-study is used to fit thresholds. Scores/embeddings may be precomputed under frozen models, but K results are not inspected before threshold freeze. Threshold tuning human checks are distinct from the final representative30-business human evaluation; do not reuse the terminology or imply that final evaluation occurred.

For K study, keep complete review IDs/metadata/source serializer and common caption/full-source budgets fixed in both conditions. Approximately5000 and750 remain exploratory pending allocation approval. Fewer-thanK pools keep all eligible captions without duplication/threshold relaxation. Empty pool/teacher-visual-target handling remainsTBD, no automatic business exclusion. No test data used to choose any of the above.

Concrete decisions needed before execution: relevance scope/scorer/prototype list; model/version/scoring precision; coverage OptionA/B; quality/redundancy/group/diversity rules; calibration/K role IDs and human sampling/threshold search; common source budgets. Machine proposal: configs/stage_f_scoring_review.yaml. Metadata-only evidence: outputs/stage_f/scoring_definition_review_v1/design_review_report.json. Active Basic Quality unchanged under D-026.
