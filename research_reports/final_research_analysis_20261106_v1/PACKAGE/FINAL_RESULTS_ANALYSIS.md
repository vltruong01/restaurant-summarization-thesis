# Final Research Results Analysis

Completed-research analysis, 6 November 2026. This analysis uses the original, recovered Stage P V1 collection, never the later revised collections. It preserves the frozen models and all recorded ratings. Evidence codes below resolve in [EVIDENCE_TRACEABILITY.md](EVIDENCE_TRACEABILITY.md).

## 1. Evidence scope and verification

| Stage | Cohort | Conditions | Main observations | Interpretation boundary |
| --- | --- | --- | --- | --- |
| M | Yelp TEST100 | RO/U/F/S | 700 component-score rows from 400 outputs | Agreement with common frozen human-approved references |
| N | Yelp TEST30, nested in TEST100 | RO/U/F/S | 240 ratings; 1,380 numeric scores; 60 structural Visual N/A cells | Two-evaluator, source-grounded judgments |
| O | TripAdvisor15 | LED RO/U/F | 90 ratings; 510 numeric scores; 30 structural Visual N/A cells | External-domain, reference-free judgments |
| P | Same TripAdvisor15 | LED F-S1000/Qwen-Caption/Qwen-Image | 90 original V1 ratings; 540 scores | New blinded comparison panel; returning evaluators |

Stages are analyzed separately. Ratings, criterion scores and restaurants are different units: the paired unit is the business/restaurant, with both raters retained together. Ninety Stage P records do not constitute 90 independent restaurants or 540 independent experiments. O and P share restaurants and returning evaluator accounts; N is a subset of M. No cross-stage pooled effect or composite winner is calculated. [E01–E05]

The official P archive SHA-256 is `e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5`; its combined rating export is `e47b88f1b4d6ccf0bf1649b5b06ead3f538f97af30324993f5e45cf79a58472f`. Verification independently reconstructs the statistics from original V1 and cross-checks them against execution of the unchanged frozen analysis script. [E01]

All M/N/O/P original archive member manifests and CRC checks passed. Independent calculations matched 42 M metric means, 24 M paired CIs and four primary adjusted tests; N descriptives/agreement and 17 paired tests; O descriptives/agreement and 11 paired tests; and all P descriptives, agreement and 18 contrasts. The included verification receipt identifies exact seeds, source hashes and software. This verifies numerical consistency, not participant identity, every semantic judgment, or fresh execution of ROUGE/BERTScore models. [E02]

## 2. Stage M: final TEST100 reference agreement

All metrics below are macro means on the original **0–1 scale**, including ROUGE. RO has no Visual component and is not scored as zero. These are final TEST100 human-reference results, not VALIDATION200 Teacher-target agreement. [E03]

| Component | Condition | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT P | BERT R | BERT F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| review | RO | 100 | 0.2884 | 0.0369 | 0.1539 | 0.8660 | 0.8531 | 0.8595 |
| review | U | 100 | 0.2859 | 0.0372 | 0.1545 | 0.8657 | 0.8519 | 0.8587 |
| review | F | 100 | 0.2825 | 0.0327 | 0.1511 | 0.8629 | 0.8517 | 0.8572 |
| review | S | 100 | 0.2825 | 0.0327 | 0.1511 | 0.8629 | 0.8517 | 0.8572 |
| visual | U | 100 | 0.2725 | 0.0386 | 0.1714 | 0.8620 | 0.8432 | 0.8525 |
| visual | F | 100 | 0.2561 | 0.0324 | 0.1566 | 0.8522 | 0.8231 | 0.8374 |
| visual | S | 100 | 0.2197 | 0.0212 | 0.1422 | 0.8435 | 0.8161 | 0.8295 |

The four prespecified BERTScore-F1 comparisons use 20,000 paired business bootstrap samples and 50,000 Monte Carlo sign flips, seed 20261028, with Holm correction across four tests. F minus RO Review is −0.002308 (95% CI [−0.005814, −0.000064], raw p=0.058319, Holm p=0.174957). F minus U Review is −0.001546 ([−0.004965, 0.000745], Holm p=0.813384). F minus U Visual is −0.015106 ([−0.041334, 0.002773], Holm p=0.813384). F minus S Visual is +0.007852 ([0.005865, 0.009825], Holm p=0.000080). [E03]

Only the F–S Visual comparison is supported at the adjusted 0.05 threshold. Pointwise bootstrap CIs and multiplicity-adjusted tests answer different inferential questions; the F–RO interval is not a reason to override its nonsignificant adjusted test. U's higher Visual similarity is descriptive evidence against a universal F advantage, not proof of better visual factuality. F and S Review texts are exactly reused, so equal Review metrics are a construction check rather than replicated Review evidence.

ROUGE uses full component text with Porter stemming; BERTScore uses pinned RoBERTa-large, layer 17, IDF off and no baseline rescaling, with a fixed 512-token prefix. One F Review and its S reuse were prefix-truncated; references were not. F/S each have two empty Visual outputs, retained with zero similarity scores. The same recipient-business reference is used for every applicable condition, including S. [E03]

Length/repetition diagnostics show shorter F than U Visual outputs (51.68 versus 67.74 mean words) and lower mean repeated-trigram fraction (0.0605 versus 0.1230). Duplicate-sentence fractions are 0.0032 versus 0.0187. These are descriptive output properties; shorter text can suppress repetition and omit useful content simultaneously. Review repetition does not show the same F advantage. [E04]

## 3. Stage N: Yelp TEST30 human evaluation

Means on the 1–5 rubric; all original 240 records retained. Full SDs, medians, distributions and per-rater results are in the supplied tables. [E05]

| Criterion | RO | U | F | S |
| --- | --- | --- | --- | --- |
| Review Faithfulness | 4.150 | 4.233 | 4.033 | 4.017 |
| Aspect Coverage | 3.383 | 3.383 | 3.083 | 3.050 |
| Coherence | 4.550 | 3.950 | 4.067 | 3.700 |
| Non-Redundancy | 3.950 | 3.117 | 3.367 | 3.083 |
| Usefulness | 3.283 | 3.283 | 3.050 | 3.017 |
| Visual Grounding and Contribution | N/A | 4.000 | 3.900 | 1.233 |

F exceeds S on Visual Grounding and Contribution by 2.667 points (95% CI [2.300, 3.017], raw exploratory p=0.000020); 28 businesses favor F, two tie and none favor S. This is substantially larger than the automatic similarity gap and supports the importance of business-aligned visual evidence. It does not isolate which filtering operation causes the effect. [E06]

F does not establish Visual superiority over U: difference −0.100 ([−0.383, 0.200], raw p=0.582008). F–U Aspect Coverage is −0.300 ([−0.500, −0.133], raw p=0.003300) and Usefulness −0.233 ([−0.417, −0.067], raw p=0.020240). F–U Non-Redundancy is +0.250 ([−0.117, 0.633], raw p=0.233235). These exploratory, unadjusted comparisons indicate coverage/usefulness tradeoffs, rather than consistent F superiority or an established Yelp human repetition benefit. No multiplicity correction was prespecified for these 17 N contrasts; no confirmatory interpretation is added retrospectively. [E06]

Condition-specific quadratic-weighted kappa ranges from 0.460 to 0.952. High raw agreement can coexist with modest kappa when scores concentrate near the floor: S Visual exact agreement is 93.3% but kappa is about 0.565. Six of 180 F/S Review-only score cells differ despite identical Review components, illustrating contextual or measurement variation. Existing audit concerns remain disclosed; the researcher accepted the recorded data without changes. Participant flags remain reports rather than an independently adjudicated factual-error census. [E07]

## 4. Stage O: external TripAdvisor generalization

The Yelp-trained LED checkpoints were applied without additional fine-tuning. These 15 restaurants provide external-domain evidence, but are a geographically narrow English-review convenience cohort. They do not support a population-wide generalization estimate. [E08]

| Criterion | RO | U | F |
| --- | --- | --- | --- |
| Review Faithfulness | 3.500 | 3.500 | 3.700 |
| Aspect Coverage | 3.200 | 3.267 | 3.233 |
| Coherence | 4.433 | 4.000 | 4.167 |
| Non-Redundancy | 4.200 | 3.000 | 3.833 |
| Usefulness | 2.900 | 2.933 | 3.033 |
| Visual Grounding and Contribution | N/A | 3.700 | 3.533 |

F–U Non-Redundancy is +0.833 ([0.400, 1.333], raw exploratory p=0.004740); ten restaurants favor F, four tie and one favors U. This is the clearest external filtering-associated advantage. F–U Visual is −0.167 ([−0.633, 0.300], raw p=0.601748); Review-only F–U and F–RO contrasts do not establish superiority. These are 11 exploratory comparisons, including two supplementary component-budget comparisons, without a prespecified multiplicity correction. [E09]

The models retain some readable and source-related summarization capability, but coverage is around 3.2 and usefulness around 3.0. Recorded comments describe missing mixed opinions, omitted venue details, unsupported service complaints and repeated phrases. Agreement is heterogeneous (condition-specific QWK 0.000–0.904); U/F Visual kappas are approximately 0.536/0.490. These are qualified external results, not evidence of unchanged in-domain performance or a formal cross-domain non-inferiority result. No TripAdvisor Teacher/reference-similarity evaluation is substituted for source-grounded judgments. [E08–E10]

## 5. Stage P: official V1 comparison with Qwen

### 5.1 Descriptives and distributions

Each combined mean uses 30 scores, two per restaurant. SDs below describe those scores, not 30 independent restaurants. P uses the original LED F outputs and the frozen Qwen outputs. [E11]

| Criterion | System | Mean ± sample SD | Median | Scores 1/2/3/4/5 |
| --- | --- | --- | --- | --- |
| Review Faithfulness | LED F original | 3.267 ± 1.143 | 3.0 | 2/5/11/7/5 |
| Review Faithfulness | Qwen-Caption | 4.300 ± 0.702 | 4.0 | 0/0/4/13/13 |
| Review Faithfulness | Qwen-Image | 4.300 ± 0.702 | 4.0 | 0/0/4/13/13 |
| Aspect Coverage | LED F original | 3.000 ± 0.455 | 3.0 | 0/3/24/3/0 |
| Aspect Coverage | Qwen-Caption | 4.533 ± 0.629 | 5.0 | 0/0/2/10/18 |
| Aspect Coverage | Qwen-Image | 4.533 ± 0.629 | 5.0 | 0/0/2/10/18 |
| Coherence | LED F original | 4.067 ± 0.254 | 4.0 | 0/0/0/28/2 |
| Coherence | Qwen-Caption | 5.000 ± 0.000 | 5.0 | 0/0/0/0/30 |
| Coherence | Qwen-Image | 5.000 ± 0.000 | 5.0 | 0/0/0/0/30 |
| Non-Redundancy | LED F original | 3.167 ± 1.085 | 3.0 | 0/11/7/8/4 |
| Non-Redundancy | Qwen-Caption | 4.800 ± 0.407 | 5.0 | 0/0/0/6/24 |
| Non-Redundancy | Qwen-Image | 4.800 ± 0.407 | 5.0 | 0/0/0/6/24 |
| Usefulness | LED F original | 2.933 ± 0.450 | 3.0 | 0/4/24/2/0 |
| Usefulness | Qwen-Caption | 4.400 ± 0.563 | 4.0 | 0/0/1/16/13 |
| Usefulness | Qwen-Image | 4.400 ± 0.563 | 4.0 | 0/0/1/16/13 |
| Visual Grounding and Contribution | LED F original | 3.533 ± 0.819 | 4.0 | 0/3/11/13/3 |
| Visual Grounding and Contribution | Qwen-Caption | 3.533 ± 0.819 | 4.0 | 0/3/11/13/3 |
| Visual Grounding and Contribution | Qwen-Image | 4.167 ± 0.747 | 4.0 | 0/0/6/13/11 |

All 15 Qwen Review components are identical between Caption and Image variants. Their three Review-only criteria therefore do not test an image-derived Review benefit. Coherence and Non-Redundancy assess both components. In this collection the two Qwen variants also receive identical scores on those criteria, although their Visual text differs.

### 5.2 Paired inference under the frozen plan

Within each restaurant, the two within-rater differences are averaged. Twenty thousand whole-restaurant bootstrap samples use seed 20261106; the exact sign-flip test enumerates all 32,768 sign assignments. Holm correction is applied separately to 12 primary tests and six additional exploratory tests. CIs are pointwise percentile intervals, not simultaneous family-wise intervals. The tests remain exploratory because an exact enumeration does not make this observational system comparison randomized. [E12]

| Contrast (left minus right) | Criterion | Difference | Pointwise 95% CI | Exact p | Holm p | Family |
| --- | --- | --- | --- | --- | --- | --- |
| LED F original − Qwen-Caption | Review Faithfulness | -1.033 | [-1.467, -0.567] | 0.002197 | 0.017578 | PRIMARY |
| Qwen-Caption − Qwen-Image | Review Faithfulness | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| LED F original − Qwen-Image | Review Faithfulness | -1.033 | [-1.467, -0.567] | 0.002197 | 0.004395 | ADDITIONAL_EXPLORATORY |
| LED F original − Qwen-Caption | Aspect Coverage | -1.533 | [-1.767, -1.267] | 0.000061 | 0.000732 | PRIMARY |
| Qwen-Caption − Qwen-Image | Aspect Coverage | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| LED F original − Qwen-Image | Aspect Coverage | -1.533 | [-1.767, -1.267] | 0.000061 | 0.000366 | ADDITIONAL_EXPLORATORY |
| LED F original − Qwen-Caption | Coherence | -0.933 | [-1.000, -0.833] | 0.000061 | 0.000732 | PRIMARY |
| Qwen-Caption − Qwen-Image | Coherence | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| LED F original − Qwen-Image | Coherence | -0.933 | [-1.000, -0.833] | 0.000061 | 0.000366 | ADDITIONAL_EXPLORATORY |
| LED F original − Qwen-Caption | Non-Redundancy | -1.633 | [-2.000, -1.233] | 0.000122 | 0.001099 | PRIMARY |
| Qwen-Caption − Qwen-Image | Non-Redundancy | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| LED F original − Qwen-Image | Non-Redundancy | -1.633 | [-2.000, -1.233] | 0.000122 | 0.000366 | ADDITIONAL_EXPLORATORY |
| LED F original − Qwen-Caption | Usefulness | -1.467 | [-1.700, -1.233] | 0.000061 | 0.000732 | PRIMARY |
| Qwen-Caption − Qwen-Image | Usefulness | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| LED F original − Qwen-Image | Usefulness | -1.467 | [-1.700, -1.233] | 0.000061 | 0.000366 | ADDITIONAL_EXPLORATORY |
| LED F original − Qwen-Caption | Visual Grounding and Contribution | 0.000 | [0.000, 0.000] | 1.000000 | 1.000000 | PRIMARY |
| Qwen-Caption − Qwen-Image | Visual Grounding and Contribution | -0.633 | [-1.033, -0.200] | 0.021118 | 0.147827 | PRIMARY |
| LED F original − Qwen-Image | Visual Grounding and Contribution | -0.633 | [-1.033, -0.200] | 0.021118 | 0.021118 | ADDITIONAL_EXPLORATORY |

Qwen-Caption exceeds LED F on all five non-Visual criteria after correction within the primary family. The largest differences concern coverage, non-redundancy and usefulness. Both raters favor Qwen on the mean difference for all five criteria; their repetition-effect magnitudes differ markedly. LED F and Qwen-Caption receive identical Visual scores for every restaurant/rater cell in V1. This exact recorded equality is not an equivalence trial, proof of equal factual accuracy, or equality of overall quality.

Qwen-Image exceeds Qwen-Caption descriptively on Visual by 0.633 points. Expressed as Caption minus Image, the pointwise CI is [−1.033, −0.200], raw exact p=0.021118 and primary-family Holm p=0.147827: the planned primary contrast is inconclusive after correction. The same numerical Image advantage over LED F has additional-family Holm p=0.021118, because that is a different, smaller family of six tests. This ancillary result must not replace the primary Caption–Image conclusion. There is no prespecified composite winner.

### 5.3 Agreement and measurement limitations

| Criterion | System | QWK | Exact agreement | Mean absolute difference | A01 − A02 |
| --- | --- | --- | --- | --- | --- |
| Review Faithfulness | LED F original | 0.842 | 60.0% | 0.400 | 0.133 |
| Review Faithfulness | Qwen-Caption | 0.521 | 53.3% | 0.467 | 0.200 |
| Review Faithfulness | Qwen-Image | 0.521 | 53.3% | 0.467 | 0.200 |
| Aspect Coverage | LED F original | 0.388 | 73.3% | 0.267 | 0.267 |
| Aspect Coverage | Qwen-Caption | 0.302 | 46.7% | 0.533 | 0.000 |
| Aspect Coverage | Qwen-Image | 0.302 | 46.7% | 0.533 | 0.000 |
| Coherence | LED F original | 0.000 | 86.7% | 0.133 | -0.133 |
| Coherence | Qwen-Caption | UNDEFINED | 100.0% | 0.000 | 0.000 |
| Coherence | Qwen-Image | UNDEFINED | 100.0% | 0.000 | 0.000 |
| Non-Redundancy | LED F original | 0.284 | 6.7% | 1.400 | 1.400 |
| Non-Redundancy | Qwen-Caption | 0.000 | 60.0% | 0.400 | -0.400 |
| Non-Redundancy | Qwen-Image | 0.000 | 60.0% | 0.400 | -0.400 |
| Usefulness | LED F original | 0.659 | 86.7% | 0.133 | 0.000 |
| Usefulness | Qwen-Caption | 0.589 | 73.3% | 0.267 | -0.267 |
| Usefulness | Qwen-Image | 0.589 | 73.3% | 0.267 | -0.267 |
| Visual Grounding and Contribution | LED F original | 0.630 | 46.7% | 0.533 | 0.533 |
| Visual Grounding and Contribution | Qwen-Caption | 0.630 | 46.7% | 0.533 | 0.533 |
| Visual Grounding and Contribution | Qwen-Image | 0.818 | 80.0% | 0.200 | 0.200 |

Both Qwen Coherence cells have perfect exact agreement but undefined kappa: every score is 5 and expected disagreement is zero. They are not assigned kappa=1. LED Non-Redundancy has only 6.7% exact agreement and A01−A02 mean difference +1.400, despite both raters favoring Qwen in the paired contrast. QWK and exact agreement are reported together because score concentration can produce low kappa even when disagreements are small. Two raters and 15 restaurants cannot establish reliability across a wider evaluator population. [E13]

Stage P Review lengths average 81.07 words for LED and 132.87 for either Qwen variant; Visual means are 69.00, 52.13 and 65.53 respectively. Longer Qwen Review scope may help coverage but also changes the task budget. Word counts use one shared whitespace rule; model token counts use different native tokenizers and are not interchangeable. [E14]

### 5.4 Participant comments and flags

All 90 V1 records contain comments; 17 records carry serious-grounding flags (A01: 10; A02: 7). These count flagged rating records, not unique errors, independent adjudications or photo-level accuracy. Shared Qwen Review errors can be flagged in both variants. The original certainty labels remain attributed to the rater. [E22]

| System | Rater | Serious flags / 15 | Rater-labeled confirmed | Rater-labeled uncertain |
| --- | --- | --- | --- | --- |
| LED F original | A01 | 5 / 15 | 5 | 0 |
| LED F original | A02 | 4 / 15 | 4 | 0 |
| Qwen-Caption | A01 | 2 / 15 | 2 | 0 |
| Qwen-Caption | A02 | 2 / 15 | 2 | 0 |
| Qwen-Image | A01 | 3 / 15 | 3 | 0 |
| Qwen-Image | A02 | 1 / 15 | 1 | 0 |

The comments describe omitted mixed opinions and venue details, unsupported service complaints, dish-level polarity misattribution, reversed visual actions, sign-reading errors and cross-scene relations. For example, A02 flags Qwen-Caption's “One man pours wine into a green bottle” for `TA14622324_2312071`, citing `TA_PHOTO_0713755e`, `TA_PHOTO_0f821627` and `TA_PHOTO_1ba1c1f3`; A01 flags Qwen-Image's “typically costing around 15 to 30 euros for two people” for `TA14622324_2298952`, citing `TA_REVIEW_523497914`, `TA_REVIEW_789979868` and `TA_REVIEW_673263312`. These are traceable participant-reported examples, not new independent confirmations. Their presence illustrates that higher mean scores do not imply error-free Qwen outputs.

## 6. Integrated interpretation

The strongest consistent result is the harm caused by recipient-misaligned visual evidence: F exceeds S in TEST100 Visual reference agreement and TEST30 Visual human scores. Filtering is associated with shorter, less repetitive Visual outputs and a TripAdvisor Non-Redundancy improvement, but does not consistently improve Visual human scores over U; some Yelp Review coverage/usefulness results favor U. Stage P places this limitation in context: a large pretrained instruction-following baseline is rated better than LED on five criteria, even with the same selected Review/caption evidence. A smaller measured inference footprint is therefore a resource tradeoff accompanied by observed quality deficits. [E03–E15]

This analysis leaves F-S1000 frozen as the previously selected study system; it does not reselect checkpoints from TEST results. Answers to the original research questions and qualified deployment implications are in the companion documents. No new human scores, model calls, references or experiments were produced.
