# Evidence Traceability

All new prose refers to the following internal evidence IDs. Filenames are relative to this package. Exact member byte hashes are in `SOURCE_INVENTORY.json`; final delivered-file hashes are in `PACKAGE_MANIFEST.json`. The review ZIP contains selected small evidence, not every original large experiment archive. Historical scripts/prompts/documents under `SOURCES/` are preserved evidence, not new execution instructions.

## 1. Frozen source archives

| Stage | Original source archive SHA-256 |
| --- | --- |
| M | `edccd741a1b6580926b3d0dd46e42380d03670ddb2c2de11b726cfca5db765dc` |
| N | `f6bdcdb8870a40d06af59de7a71b57f0bae7e660ccb852b68c33318217ea8b42` |
| O | `2d2ee4d12f53d12c63e42166b00142a952df9055dedee6e265e379edda809b8b` |
| P | `e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5` |
| Training provenance | `7ecedb7084ee8034e8180fd73f951c8a1ed1cdbaadcaa6ffef78b3f5c8f1afd3` |

Original archive paths, CRC checks and every-member manifest checks are recorded in `SOURCE_INVENTORY.json`. P is explicitly bound to recovered V1, never the live/revised data. The external ChatGPT statistical ZIP is unavailable and was not verified; independent reproduction was authorized as the replacement for comparing that absent artifact. Source hashes establish bytes and provenance, not semantic correctness or authenticated participant identity.

## 2. Claim-level register

| ID | Claim scope | Source files / row locator |
| --- | --- | --- |
| E01 | Official P V1 identity and scope | [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json); [SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json](SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json) — 90 records,540 scores; original archive/export hashes; external ChatGPT ZIP not inspected |
| E02 | Independent numerical reconstruction | [VERIFIED_TABLES/VERIFICATION_RESULTS.json](VERIFIED_TABLES/VERIFICATION_RESULTS.json); [SCRIPTS/verify_integrated_research_statistics_v1.py](SCRIPTS/verify_integrated_research_statistics_v1.py) — M42means/24CIs/4tests,N17tests,O11tests,P18tests; no source changes |
| E03 | M final TEST100 automatic metrics and inference | [SOURCES/M/RESULTS_ORIGINAL/AGGREGATE_METRICS.csv](SOURCES/M/RESULTS_ORIGINAL/AGGREGATE_METRICS.csv); [SOURCES/M/RESULTS_ORIGINAL/PAIRED_COMPARISONS.csv](SOURCES/M/RESULTS_ORIGINAL/PAIRED_COMPARISONS.csv); [SOURCES/M/SUMMARY/METHODOLOGY.md](SOURCES/M/SUMMARY/METHODOLOGY.md) — Rows condition/task and contrast/metric; raw0–1 scores;4primary tests;700 component rows |
| E04 | M repetition/length | [VERIFIED_TABLES/M_OUTPUT_DIAGNOSTICS_VERIFIED.csv](VERIFIED_TABLES/M_OUTPUT_DIAGNOSTICS_VERIFIED.csv); [SOURCES/M/RESULTS_ORIGINAL/STRUCTURE_REPETITION_LENGTH.md](SOURCES/M/RESULTS_ORIGINAL/STRUCTURE_REPETITION_LENGTH.md) — U/F visual means; words,repeated_3gram_fraction,duplicate_sentence_fraction |
| E05 | N criterion means,SD/distributions/per-rater | [VERIFIED_TABLES/N_DESCRIPTIVES_VERIFIED.csv](VERIFIED_TABLES/N_DESCRIPTIVES_VERIFIED.csv); [SOURCES/N/TABLES/DESCRIPTIVE_STATISTICS.json](SOURCES/N/TABLES/DESCRIPTIVE_STATISTICS.json) — 240records;condition/criterion/panel;Review versus VisualN/A |
| E06 | N paired comparisons | [SOURCES/N/TABLES/PAIRED_COMPARISONS.json](SOURCES/N/TABLES/PAIRED_COMPARISONS.json) — 17rows;contrast/criterion;whole30businesses;unadjusted exploratoryp |
| E07 | N agreement and accepted-as-recorded provenance | [VERIFIED_TABLES/N_AGREEMENT_VERIFIED.csv](VERIFIED_TABLES/N_AGREEMENT_VERIFIED.csv); [SOURCES/N/METHODOLOGY.md](SOURCES/N/METHODOLOGY.md); [SOURCES/N/PROVENANCE/MODEL_ASSISTED_AUDIT_SUMMARY.json](SOURCES/N/PROVENANCE/MODEL_ASSISTED_AUDIT_SUMMARY.json); [SOURCES/N/PROVENANCE/RESEARCHER_ACCEPTANCE_AS_RECORDED.json](SOURCES/N/PROVENANCE/RESEARCHER_ACCEPTANCE_AS_RECORDED.json) — 23condition-specific agreementcells;knownsource/commentconcerns;notindependentlyauthenticated |
| E08 | O source/domain and descriptive results | [VERIFIED_TABLES/O_DESCRIPTIVES_VERIFIED.csv](VERIFIED_TABLES/O_DESCRIPTIVES_VERIFIED.csv); [SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md](SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md); [SOURCES/O/ANALYSIS_CONFIG.json](SOURCES/O/ANALYSIS_CONFIG.json) — 15restaurants,90ratings;English/geographic/conveniencescope;noTripAdvisorfinetune |
| E09 | O paired comparisons | [SOURCES/O/TABLES/PAIRED_COMPARISONS.json](SOURCES/O/TABLES/PAIRED_COMPARISONS.json) — 11exploratorycontrasts;F−Unonred+.833,Visual−.167 |
| E10 | O agreement and qualitative scope | [VERIFIED_TABLES/O_AGREEMENT_VERIFIED.csv](VERIFIED_TABLES/O_AGREEMENT_VERIFIED.csv); [SOURCES/O/COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json](SOURCES/O/COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json); [SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md](SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md) — 17agreementcells;commentsareparticipantjudgments |
| E11 | P official descriptive statistics | [VERIFIED_TABLES/P_DESCRIPTIVES.csv](VERIFIED_TABLES/P_DESCRIPTIVES.csv) — model/criterion/panel;54rows,18combinedcells;means/SD/median/all1–5counts |
| E12 | P paired analysis and multiplicity | [VERIFIED_TABLES/P_PAIRED_COMPARISONS.csv](VERIFIED_TABLES/P_PAIRED_COMPARISONS.csv); [SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json](SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json) — 18contrasts;15pairedrestaurants;20kbootstrap,seed20261106;32768signs;Holm12/6 |
| E13 | P condition-specific agreement | [VERIFIED_TABLES/P_AGREEMENT.csv](VERIFIED_TABLES/P_AGREEMENT.csv) — 18cells;twoQwenCoherencekappasundefined;LEDNonRedexact1/15,bias1.4 |
| E14 | P output budget/shared Review | [VERIFIED_TABLES/P_OUTPUT_LENGTH_CONTEXT.csv](VERIFIED_TABLES/P_OUTPUT_LENGTH_CONTEXT.csv); [SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json](SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json) — SharedReview15/15;whitespacewordmeans81.07/132.87;notnativetokens |
| E15 | Measured generation and missing pipeline costs | [SOURCES/EFFICIENCY/VERIFIED_FACTS.json](SOURCES/EFFICIENCY/VERIFIED_FACTS.json); [SOURCES/EFFICIENCY/EFFICIENCY_MEASUREMENTS.csv](SOURCES/EFFICIENCY/EFFICIENCY_MEASUREMENTS.csv); [SOURCES/EFFICIENCY/EFFICIENCY_COMPARISON.md](SOURCES/EFFICIENCY/EFFICIENCY_COMPARISON.md); [SOURCES/EFFICIENCY/MISSING_MEASUREMENTS.md](SOURCES/EFFICIENCY/MISSING_MEASUREMENTS.md) — Existing audit provenance rows;generation97.931/186.844/constructed379.697;BLIP48642.959;score178.529;E2Eunmeasured |
| E16 | Original research questions | [SOURCES/RESEARCH_QUESTIONS.json](SOURCES/RESEARCH_QUESTIONS.json); [SOURCES/APPROVED_PROPOSAL/DOCUMENT_EXTRACT.json](SOURCES/APPROVED_PROPOSAL/DOCUMENT_EXTRACT.json) — Exact3questions;sourceWordhash;noobjective redefinition |
| E17 | Final model ancestry and LoRA scope | [SOURCES/M/PROTOCOL/MODEL_AND_DECODER_LOCK.json](SOURCES/M/PROTOCOL/MODEL_AND_DECODER_LOCK.json); [SOURCES/TRAINING/LORA_TRAINABLE_PREFLIGHT.json](SOURCES/TRAINING/LORA_TRAINABLE_PREFLIGHT.json); [SOURCES/TRAINING/TRAINING_COMPLETION_REPORT.md](SOURCES/TRAINING/TRAINING_COMPLETION_REPORT.md); [SOURCES/TRAINING/INITIALIZATION_HASHES.json](SOURCES/TRAINING/INITIALIZATION_HASHES.json) — RO/Uown750parents,FCE3parent;4000R+0/4000V;4915200trainable/459801600backbone |
| E18 | Accepted F continuation runtime | [SOURCES/TRAINING/F_S1000_COMPLETION_NATIVE.json](SOURCES/TRAINING/F_S1000_COMPLETION_NATIVE.json) — completion.seconds7465.885813831992;peak_allocated_bytes3665676800;peak_reserved_bytes5572132864;RTX3090 |
| E19 | Known cross-split image exposure | [SOURCES/IMAGE_OVERLAP/EXPOSURE_SUMMARY.json](SOURCES/IMAGE_OVERLAP/EXPOSURE_SUMMARY.json); [SOURCES/IMAGE_OVERLAP/DUPLICATION_CAUSE_AND_IMPACT_SUMMARY.md](SOURCES/IMAGE_OVERLAP/DUPLICATION_CAUSE_AND_IMPACT_SUMMARY.md) — 17businesses,123TESTphotoIDs,490pairs;selectedU66/F44;ownTRAINU59/F42 |
| E20 | Study measurement/provenance scope | [SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json](SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json); [SOURCES/N/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md](SOURCES/N/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md); [SOURCES/O/COLLECTION/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md](SOURCES/O/COLLECTION/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md) — Returningraters;accountrecordsnotidentityauthentication;noindependentpopulationofraters |
| E21 | Reference AI-assistance disclosure | [SOURCES/M/REFERENCE_PROVENANCE/AI_ASSISTANCE_AND_LIMITATIONS.md](SOURCES/M/REFERENCE_PROVENANCE/AI_ASSISTANCE_AND_LIMITATIONS.md) — Author/checkerapprovalandAIassistance;freezehashesnotunaidedauthorshipproof |
| E22 | P participant comments/flags | [VERIFIED_TABLES/P_FROZEN_SCRIPT_REPLAY/RATER_COMMENTS_VERBATIM.json](VERIFIED_TABLES/P_FROZEN_SCRIPT_REPLAY/RATER_COMMENTS_VERBATIM.json); [SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json](SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json) — Verbatimreports;notanindependenterrorcensus |

## 3. Calculation and reporting conventions

M means are independently recomputed from stored per-business similarity scores; embedding/ROUGE models are not rerun. N/O/P means, sample SDs, medians, distributions and condition-specific QWK are independently recomputed from the frozen numeric scores. N/O/M paired intervals and tests are reconstructed with their original Monte Carlo order and seed. P independently enumerates integer-scaled half-point differences and all2^15sign assignments, then matches the unchanged frozen script's output. The bootstrap index hash is stored. Numerical comparisons allow rounding-level tolerance, not replacement data.

Pointwise percentile95%CIs and family-wise p-values are labeled separately. No new exclusion, corrected rating, compositewinner,condition pooling or multiplicity policy is introduced. P additional-family significance does not override an inconclusive primary-family contrast. All later P revisions are excluded.

## 4. Style and artifact preservation

`SOURCES/STYLE_REFERENCE/DOCUMENT_EXTRACT.json` records the numbered stage organization and SHA-256 of the unchanged Word style reference. Its original file is not overwritten or bundled. The discussion follows that organization while avoiding the unprespecified six-criterion composite score found in the reference report. No notebook, final thesis rewrite or presentation is produced.

The new statistical script, report builder and archive verifier are supplied under `SCRIPTS/`. `VALIDATION_REPORT.json` and `SOURCE_PRESERVATION.json` document actual final checks. No database, login/password file, model binary, API credential or original photo archive is included. These exclusions do not alter the frozen source evidence retained in the project.
