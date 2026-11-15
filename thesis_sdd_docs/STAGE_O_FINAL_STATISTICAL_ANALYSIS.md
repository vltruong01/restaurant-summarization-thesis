# Stage O final TripAdvisor statistical analysis

Status: complete for researcher/ChatGPT review, 6 November 2026. Qwen remains deferred.

All 15 restaurants, 45 frozen RO/U/F outputs and both original evaluators are retained: 90 submitted records, 510 numeric scores, 30 structural RO Visual N/A cells. No scores or assignments were changed; no exclusions, training, new inference, reference generation or Qwen work occurred.

| Criterion | RO | U | F |
|---|---:|---:|---:|
| Review faithfulness | 3.500 | 3.500 | 3.700 |
| Aspect coverage | 3.200 | 3.267 | 3.233 |
| Coherence | 4.433 | 4.000 | 4.167 |
| Non-Redundancy | 4.200 | 3.000 | 3.833 |
| Review usefulness | 2.900 | 2.933 | 3.033 |
| Visual grounding and contribution | N/A | 3.700 | 3.533 |

F–U Non-Redundancy: +0.833, pointwise95% paired-business CI [0.400,1.333], raw exploratory sign-flip p=0.004740. Ten restaurants favor F, four tie, one favors U. F–U Visual: -0.167, CI [-0.633,0.300], p=0.601748. Review correctness/coverage/usefulness and applicable F–RO Review contrasts are inconclusive. No consistent overall superiority or equivalence is established.

The frozen amendment determines 20,000 bootstrap draws, 50,000 sign flips, seed20261028, with both raters averaged within restaurant. Fifteen restaurants are the inferential units. All nine frozen-plan exploratory contrasts and two separately labeled supplementary F–RO Coherence/Non-Redundancy contrasts are reported without multiplicity adjustment. No composite winner score is constructed.

Condition-specific QWK ranges 0.000–0.904; F Coherence has low marginal variation (A02 always4). U/F Visual QWK is 0.536/0.490, with A01 systematically higher. Full score-difference distributions and per-rater descriptive results are preserved.

All 90 comments were read. Participant flags remain participant reports, including those labeled confirmed by a rater; ten purposive source/text examples are model-assisted analysis. Three cited photos were independently viewed, not all5,143 images. Earlier test declarations and the direct-access waiver remain in the original export; account/journal evidence does not independently authenticate human identity or complete source inspection. Yelp TEST30 comparison retains its prior limitations and does not pool cohorts or establish Teacher-style independence.

Reproducibility: custom QWK checked against sklearn; means/SD checked with stdlib; every CI reconstructed from the full two-rater tensor; all p-values reconstructed using the original Stage N chunked procedure. The extracted portable script reproduced all25 deterministic table files byte for byte. The ZIP was reopened, extracted, parsed and all108 members verified. All47 bound original files and5,143 photos were rehashed unchanged. Checkpoint weights were not newly loaded/rehashed.

Artifact: `outputs/stage_o/final_human_statistical_analysis_20261106_v1/STAGE_O_FINAL_STATISTICAL_REVIEW.zip`

SHA-256: `2d2ee4d12f53d12c63e42166b00142a952df9055dedee6e265e379edda809b8b`

Size: 1,697,401 bytes. Original rating export SHA-256: `d89c2fc9ed7c228139a1594c9ff7b1f7b2ffda2dfbe18bff435a7ccb6a13df7f`.

Stop for review; no next research stage has begun.
