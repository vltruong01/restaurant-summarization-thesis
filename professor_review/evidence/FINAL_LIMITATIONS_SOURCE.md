# Limitations and Threats to Validity

## 1. Experimental identification and training history

RO/U/F are complete trained systems, not a controlled ablation differing only in one input at inference. Final RO and U continue their own RO750 and U750 parents; F continues CE3-750. Each receives 1,000 new successful updates, but RO has 4,000 Review and no Visual presentations, whereas U/F each have 4,000 Review and 4,000 Visual presentations. RO's four Review microbatches are normalized by the eight-slot accumulation denominator, unlike U/F's four-plus-four composition. Equal update counts do not imply equal task exposure, gradient scaling, compute or prior supervision. RO's own parent provenance must not be replaced with the inaccurate claim that it starts from caption-trained CE3. U/F target construction and previous training histories are condition-dependent. [E17]

Source-separated inference isolates Review and Visual evidence. This helps attribution but limits claims about multimodal fusion into the Review consensus. The fixed Shuffled donor permutation tests recipient alignment; it does not isolate each filtering component or create independent Review generations. RO coherence/non-redundancy is judged on one component and other conditions on two, introducing an output-budget difference. [E03, E05, E17]

## 2. Source independence and benchmark scope

Business IDs are disjoint across TRAIN2000/VALIDATION200/TEST100, but Yelp image content is partly reused across splits. The prior audit records 17 affected TEST businesses, 123 TEST photo IDs and 490 duplicate cross-split pairs. Actual selected duplicate entries number 66 for U and 44 for F; 59 U and 42 F entries overlap their own-condition TRAIN caption sources and retained Visual-label support. This is a relevant visual exposure limitation, not proof of memorization or a measured inflation estimate. All original 100 businesses are retained; no favorable post-hoc TEST83 subset or new subgroup test is introduced. [E19]

TripAdvisor comprises 15 restaurants from a narrow geographic area and available English reviews, with 247 selected reviews and 5,143 collected photos. Public image renditions are not necessarily original-resolution camera files. This is external-domain evidence, not representative sampling of TripAdvisor or all restaurant types. M/N are nested cohorts; O/P reuse the same restaurants and returning raters. Cross-stage numerical differences cannot be treated as independent replication, causal domain effects or pooled evidence. [E08, E20]

## 3. Reference construction and metric validity

TEST100 references are recorded as author/checker approved and frozen, with documented AI feedback, evidence suggestions and minor editorial assistance. Acceptance and hashes do not independently authenticate unaided human authorship, actual source inspection or participant identity. Human-approved references remain fallible. Teacher-target VALIDATION200 agreement is development evidence and must not be substituted for final TEST100 human-reference scores. [E03, E21]

ROUGE/BERTScore quantify reference similarity, not factual correctness or useful visual grounding. Reference scope can include photos beyond the caption bottleneck. BERTScore uses a 512-token prefix, affecting one F Review and its identical S reuse. Empty Visual outputs remain in the primary cohort with zero similarity scores. Shorter outputs can reduce repetition while reducing coverage; metrics and human criteria do not resolve that tradeoff with a single scalar. [E03–E04]

## 4. Human judgment, provenance and agreement

Two evaluator accounts do not provide broad population reliability or independently authenticated identity. Returning evaluators may remember sources or prior outputs despite randomized blinded assignments. Recorded declarations and technical completion are narrower evidence than verified independent behavior. The historical Stage O record includes an earlier test-context declaration, later identified by the researcher as a test; subsequent human participation is researcher-reported rather than independently authenticated. AI assistance in reference preparation and earlier model-assisted audits is disclosed; those audits are not extra human ratings or gold-standard adjudications. [E07, E10, E20–E21]

The official N data retains accepted-as-recorded audit concerns, including two A02 source/comment findings and a partial-source follow-up. No ratings were corrected by the analysis. P uses the recovered original V1 only; later revisions are excluded and do not retrospectively replace this study's record. This preserves provenance but does not erase measurement uncertainty. No participant is declared dishonest or reliable solely from agreement statistics. [E01, E07]

Criterion scores are ordinal. Means are useful descriptive summaries, not proof of equal interval spacing. Visual Grounding and Contribution combines fidelity and usefulness, so its score does not isolate either construct. Review-only criteria should not be interpreted as effects of pixels when Qwen Review is shared. Coherence ceilings, concentrated marginals and systematic rater offsets limit kappa interpretation. P's two all-5 Coherence cells have undefined kappa; LED repetition ratings show substantial rater disagreement. Exact agreement and score differences accompany kappa. [E13]

Participant “confirmed” grounding flags express the participant's judgment. The integration does not independently verify every statement or all photos, and does not promote flags to a factual-error rate. Sparse comments, differing severity judgments, repeated errors in shared outputs and multi-component judgments limit counting. Missing comments do not prove absence of error. [E07, E10, E22]

## 5. Statistical inference

Whole-business pairing retains both raters and avoids treating 540 P scores as independent. However, 15 restaurants and two fixed raters give limited precision and evaluator generalizability. Bootstrap intervals reflect empirical restaurant resampling conditional on these raters, not uncertainty over a population of raters. The sign-flip tests assume exchangeability/symmetry of paired differences; enumerating all signs does not create random assignment of system identity. Fixed shuffled donors and shared images can weaken business independence. [E02, E12]

M has four adjusted primary BERTScore tests. N/O tests are exploratory and unadjusted under their existing protocols; P has separate Holm families of 12 primary and six additional tests. Pointwise CIs are not simultaneous intervals. An interval excluding zero alongside an adjusted nonsignificant p-value is not a license to select the favorable result. Non-significance is not equivalence, and no non-inferiority margin was specified. No composite winner, post-hoc exclusion or TEST-based checkpoint selection is added. [E03, E06, E09, E12]

## 6. Efficiency and baseline fairness

LED and Qwen differ in parameter count, pretraining, task tuning, prompts, lengths, precision, decoding and framework. Qwen-Image reuses its Review and receives 73 caption-selected images; it is not a completely independent raw-photo pipeline. The historical BLIP run processed 5,143 photos on an offloaded RTX 3080 path, whereas summary benchmarks use an RTX 3090. Final filtering and complete pipeline times are missing. Generation-only ratios, allocator memory and one continuation-training time cannot establish end-to-end latency, minimum hardware, energy or cost superiority. [E15, E18]

## 7. Prospective work, not completed experiments

Reasonable next studies would preregister larger geographically diverse cohorts and additional raters; use participant-authenticated source audits; measure grounding separately from visual usefulness; harmonize summary scopes and training exposures; retain leakage-aware source manifests; test filtering components individually; and measure full pipelines on matched hardware with repeated timing. They could include a prospective non-inferiority question and compute/quality acceptance thresholds. None of these studies has been initiated by this analysis.
