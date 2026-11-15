# Stage Specification — Read-only dataset inspection notebook

**Spec ID:** 000
**Stage:** Exploratory raw/cohort inspection supporting A1/B/C
**Status:** Validated for quick raw inspection and full Stage B cohort inspection; notebook raw-audit mode remains opt-in
**Last updated:** 2026-10-05
**Execution environment:** Local Python notebook
**Environment gate required:** Available pandas/orjson/Pillow; optional language/tokenizer/plot capabilities

## 1. Objective

Provide a runnable `.ipynb` for inspecting Yelp files, schemas, samples, images, and data quality. Keep bounded inspection distinct from full-corpus audit and from the future locked experimental dataset.

## 2. Scope

In scope: source inventory, prefix samples, null/type/rating/date/duplicate checks within their stated scope, sample language/token diagnostics, original photo previews, existing preliminary CSVs, and an opt-in streaming full audit of the core files.

Out of scope: writing raw data, choosing the 2,300 businesses, assigning splits, freezing selected reviews, tuning thresholds, generating captions/targets, or renting GPU compute.

## 3. Inputs

`configs/project.yaml`, raw Yelp business/review/photo JSONL files and original photo directory; bounded auxiliary user/tip/check-in samples; existing preliminary CSVs; exported A1 tokenizer assets when available. Resolve data roots portably and support `THESIS_DATA_ROOT`.

## 4. Outputs / artifacts

- `notebooks/01_yelp_data_inspection.ipynb` with executed quick-mode outputs.
- `scripts/data_inspection.py` with streaming and QC helpers.
- Reproducible notebook creation/execution helpers.
- Notebook execution report under `outputs/inspection/`.
- README instructions and progress entry.

## 5. Methodology rules and invariants

- All source access is read-only. Do not convert diagnostics into filtering or selection decisions.
- Prefix samples are not random/representative samples. Do not label absent sample joins as global orphan IDs.
- Full eligibility audit reproduces the original raw-review-count criterion (10 reviews, 5 locally present associated photos). It does not settle the unresolved English-review eligibility policy.
- Existing CSVs are historical results, not newly recomputed counts.
- Photo metadata captions are user/source captions, not BLIP-2 output.
- Image decoding checks cover only the explicitly inspected subset; full image validation remains Stage D.
- Exact full review-ID duplicate checks are optional and use a temporary SQLite index rather than an unbounded RAM set.

## 6. Configuration

Fixed design values come from `configs/project.yaml`. Quick defaults: 5,000 business/photo metadata records, 10,000 reviews, 100 auxiliary records, 50 language samples, 100 tokenizer samples, and 6 image previews. Full audit and full review-duplicate tracking default off. These are inspection limits, not research sample settings.

K, filtering thresholds, English policy, token allocation, teacher, external dataset, and representative human sampling remain TBD under their own stage specs.

## 7. Implementation tasks

1. Implement safe JSONL sampling and streaming QC with explicit scope/corruption reports.
2. Build Vietnamese notebook sections for inventory, schemas, samples, plots, business/photo exploration, language/token lengths, and existing reports.
3. Add opt-in full audit with business/photo referential checks, local file availability, raw eligibility counts, and optional disk-based review-ID duplicate detection.
4. Execute quick mode using the existing Python kernel; exercise the full-audit helper on adversarial synthetic fixtures.
5. Validate notebook structure, execution errors, outputs, and documentation.

## 8. Validation / sanity checks

- Compile all code cells and helper files; validate required notebook keys/cell IDs.
- Run all quick cells with full-audit flags off and capture real kernel outputs.
- Synthetic fixtures cover malformed/non-object JSON, blank lines, duplicate IDs, empty text, invalid ratings/dates, orphan references, missing images, and accommodation exclusion.
- Verify no core raw files are modified during quick execution.

## 9. Acceptance criteria

- [x] Quick notebook executes without errors and includes meaningful tables/plots/photo previews.
- [x] Sample/full scopes and unresolved research choices are explicit.
- [x] Full-audit helper passes synthetic correctness checks; actual full scan is not falsely marked complete.
- [x] Paths work from the notebook directory and notebook source is valid format 4.
- [x] Raw input file metadata remains unchanged; logs accurately describe performed checks.

Evidence: `outputs/inspection/notebook_execution_report.json` reports 12/12 code cells passed (including the saved Stage B report viewer added later), six image outputs, unchanged raw file metadata, and full audit disabled. `synthetic_audit_verification.json` records adversarial fixture checks. A separate valid fixture verified no false quality flag and no invented zero for unchecked full review duplicates. Notebook source, outputs, and cell IDs were checked; the distribution plot was visually inspected.

## 10. Stop conditions

Missing required sources, corrupt notebook/runtime, or an attempted research-policy change. Report a failed optional capability without silently choosing another methodological setting.

## 11. Expected logs / reports

`outputs/inspection/notebook_execution_report.json`, synthetic audit verification, and `THESIS_PROGRESS.md`. No new research decision is made by inspection.

## 12. Deviations / amendments

None. This notebook supports observation and does not implement Stage B/C selection.

### 2026-10-05 — New Stage B cohort overview notebook

The user requests a separate notebook to inspect `yelp_2300_v1`: total reviews/photos, distributions, and businesses with most/fewest reviews/photos. Create `notebooks/02_yelp_2300_dataset_overview.ipynb` using the immutable saved cohort. Default full streaming aggregation scans the 601,493 exported reviews, retaining business-level counters and review IDs rather than the full corpus text; counts must match manifests independently. Include split totals, descriptive statistics/quantiles, ranked extrema with tie counts, plots, rating/date/photo-label profiles, and configurable business detail with bounded original photo previews. These are descriptive observations, not review selection, filtering or model tuning.

Outputs may be written only to `outputs/cohort_inspection/` and the notebook itself. Keep raw sources, dataset files and saved split unchanged. Extend the existing notebook executor with explicit notebook/report/protected-root arguments while preserving its original defaults. Acceptance: all cells run successfully, actual review/photo counts and joins equal Stage B manifests, meaningful tables/plots/photo previews are saved, plot rendering is inspected, and source/dataset file metadata snapshots are unchanged. Report and progress entry must distinguish this full cohort scan from the separate full raw Yelp audit.

Acceptance validated: `outputs/cohort_inspection/notebook_execution_report.json` records **9/9 code cells PASS**, four image outputs and unchanged protected dataset/split/raw-source metadata. Full cohort counts are 2,300 businesses, 601,493 reviews and 30,027 photos; all review and photo metadata file hashes match the package checksum inventory. Extrema: 10–5,254 reviews/business (16 businesses tied at minimum), 5–231 photos/business (417 at minimum). All four rendered figures were visually inspected; six decoded preview images were readable. The unchanged executor defaults also passed the original 12-cell notebook. This does not validate Stage C review selection or Stage D full image integrity.

### Subsequent D-006 amendment — English gate before final sampling

Notebook 02 now also runs the pre-sampling full-pool English gate/proposed sampling under the separate approved `specs/002b_english_eligibility.md`. This spec continues to govern the retained v1 descriptive/reference sections; its earlier no-sampling scope does not prohibit the newly authorized proposal sections. Latest notebook: 25 cells, 12 code cells PASS; report now at `outputs/english_eligibility/notebook_execution_report.json`. All 10,197 businesses qualify and proposed membership/positions/splits match v1. See spec 002b and the current progress log for language settings, lower-bound count semantics and remaining Stage C work.
