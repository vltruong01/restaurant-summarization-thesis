# Stage Specification — Environment and reproducibility

**Spec ID:** 001
**Stage:** A1 local preprocessing / A2 cloud experiments
**Status:** A1 Validated; A2 pending (Stage A overall partial)
**Last updated:** 2026-10-05
**Execution environment:** Local Windows for A1; Vast.ai for A2
**Environment gate required:** A1 before B/C; A2 before GPU-dependent experiments

## 1. Objective

Establish an isolated, reproducible local preprocessing environment and portable configuration before building the source dataset. Validate cloud model feasibility later, after the local dataset milestone, as approved in D-001.

## 2. Scope

### In scope for A1

- Isolated Python environment, preprocessing dependencies, and a resolved dependency lock.
- Global seed, portable paths, and explicit placeholders for unresolved research choices.
- Read-only presence/schema checks on the first record of each core Yelp JSONL source.
- Synthetic smoke checks for language detection, normalization/segmentation, JSONL reading, deterministic random operations, and LED tokenization.
- Export the existing cached LED tokenizer at its exact revision, with file checksums for future transfer.
- Machine-readable environment/acceptance report and progress/decision entries.

### Out of scope for A1

- Cohort construction, real review selection, full raw-data scans, captioning, teacher calls, model training, cloud rental, and uploads.
- Choosing K, thresholds, token allocations, the teacher, external dataset, or the 30-business sampling algorithm.
- Requiring torch/CUDA or loading BLIP-2/LED weights in the local preprocessing environment.

## 3. Inputs

- Master plan, D-001, approved proposal methodology, and existing local preprocessing scripts.
- `data/yelp_academic_dataset_business.json`: business_id, categories.
- `data/yelp_academic_dataset_review.json`: review_id, business_id, text, stars, date.
- `data/yelp_photos/photos.json`: photo_id, business_id, label.
- `data/yelp_photos/photos/`: original images, checked for directory presence only in A1.
- Cached `allenai/led-base-16384` tokenizer revision `38335783885b338d93791936c54bb4be46bebed9`.
- No cohort/split manifest exists yet. Paths resolve relative to the repository/configured data root, independent of the shell working directory.

## 4. Outputs / artifacts

- `configs/project.yaml` and `configs/research_pending.yaml`.
- `requirements/preprocessing.in`, resolved `requirements/preprocessing.lock.txt`, and Python/runtime version record.
- `.venv-preprocess/` (local runtime, excluded from transfer/version control).
- `assets/tokenizers/led-base-16384/` and `assets/tokenizers/led-base-16384.manifest.json`.
- `scripts/validate_local_environment.py`.
- `outputs/environment/local_a1_report.json`.
- Progress and decision logs. No experimental data outputs.

## 5. Methodology rules and invariants

- Raw files and images are read-only. Synthetic fixtures must live in a temporary directory.
- Retain 2,300 businesses and 2,000/200/100 split design; do not implement selection in A1.
- Seed 42 reuses the existing preliminary scripts and must be recorded, rather than silently assuming a new seed.
- Pin the exact tokenizer revision already used by the local cache; record asset hashes and compare cached/exported token IDs.
- Language smoke checks establish capability only. They do not approve a cohort language-filtering policy or rating-population definition.
- Never use held-out test material to select configurations. All smoke text is synthetic.
- A1 completion does not imply A2 or full Stage A completion.

## 6. Configuration

### Fixed values

- Python 3.11.7 for local execution; separately record platform and installed packages.
- CPU preprocessing; no torch dependency is required in the isolated A1 environment.
- Seed: 42, inherited from both existing preprocessing scripts (D-002).
- Tokenizer: `allenai/led-base-16384`, revision `38335783885b338d93791936c54bb4be46bebed9`.
- New isolated dependencies must satisfy package metadata constraints; avoid changing the global Python environment (D-003).
- Default data root: repository `data/`, overridable through `THESIS_DATA_ROOT`.
- Default output root: repository `outputs/`, overridable through `THESIS_OUTPUT_ROOT`.

### TBD / empirical values

- Exact review/marker/metadata/caption token allocations; K and filtering search spaces/criteria; teacher settings and manual inspection size; external dataset; human sampling algorithm.
- English-only filtering order, detector policy, minimum-English-review interpretation, and rating-distribution population: defined by B/C specs before use.
- Cloud GPU/runtime/budget/storage: A2 preparation.
- A1 does not select these values. Validation-selected quantities must later specify search space, validation criterion, and exclusion of held-out test data.

## 7. Implementation tasks

1. Create portable configs, dependency input, isolated venv, and this spec.
2. Resolve/install preprocessing dependencies, freeze the runtime, and run dependency consistency checks.
3. Build a validation script with detailed JSON results, failures, elapsed time, and repeatable synthetic fixtures.
4. Export the cached tokenizer and verify model identity, revision provenance, asset hashes, and token-ID equivalence.
5. Run the same validation twice and compare deterministic signatures.
6. Update acceptance status and logs using actual results.

## 8. Validation / sanity checks

- `pip check` passes in the isolated environment.
- Required preprocessing modules import and versions are recorded.
- Core raw paths exist and first JSONL records contain required fields; no claim of full-dataset validation is made.
- Portable path resolution works from another working directory.
- Synthetic English/Spanish language detection is repeatable at the configured seed.
- spaCy blank English sentencizer splits synthetic text; no downloaded language model or linguistic-quality claim is required.
- Cached and exported LED tokenizers yield equal token IDs for fixed marked review/visual strings; no truncation is applied.
- Two runs produce identical synthetic RNG, language, and tokenizer signatures, and unchanged tokenizer/source-sample hashes.

## 9. Acceptance criteria

### A1

- [x] Isolated environment installed with consistent dependencies and a resolved lock.
- [x] Required modules import; Python/platform/runtime inventory recorded.
- [x] Raw core paths/schema sample checks pass without modifications.
- [x] Portable configuration passes from a different working directory.
- [x] Synthetic language, segmentation, JSONL, and deterministic-RNG checks pass.
- [x] Exact LED tokenizer assets exported with verified checksums and matching token IDs.
- [x] Two runs have identical deterministic signatures.
- [x] Config placeholders preserve unresolved research decisions; logs accurately report A1/A2 status.

Evidence: `outputs/environment/local_a1_report.json`, `local_a1_repeat_report.json`, and `a1_acceptance_summary.json` in the same directory. Both runs pass all six grouped runtime checks; their deterministic signatures match across different working directories. Tokenizer manifest hashes were independently rechecked. Local lock contains 58 packages. Full eligibility and cloud feasibility have not been tested.

### A2 — pending, to be completed before cloud execution

- [ ] Cloud resource/runtime configuration recorded and reproducible.
- [ ] Transferred hashes, locked split/review manifests, and tokenizer behavior match local artifacts.
- [ ] BLIP-2 loading/generation and LED LoRA forward/backward feasibility checks pass at the intended methodology.
- [ ] Peak VRAM/runtime and checkpoint/resume/durable-sync checks recorded.

## 10. Stop conditions

- Required raw input or tokenizer assets are unavailable/corrupted.
- Dependency compatibility cannot be achieved without changing the methodology.
- Tokenizer behavior differs between source and exported assets or deterministic checks disagree.
- A later implementation needs a currently unresolved research parameter: record it and resolve it under the relevant stage spec before dependent work.
- Cloud GPU limitations would require changing methodology: follow master-plan stop conditions.

## 11. Expected logs / reports

- Local A1 JSON report, dependency lock, tokenizer checksum manifest.
- `THESIS_PROGRESS.md` records actual outcomes; `DECISIONS.md` records seed/environment choices.
- Cloud A2 report remains pending until its checks are executed.

## 12. Deviations / amendments

- D-001 authorizes local preparation followed by Vast.ai execution.
- D-002 records reuse of seed 42 and the cached tokenizer revision.
- D-003 records isolation of preprocessing dependencies after detecting a global package constraint mismatch.
- No change to the core experimental design is authorized by this spec.
