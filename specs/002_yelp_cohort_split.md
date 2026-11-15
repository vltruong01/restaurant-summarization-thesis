# Stage Specification — Yelp eligibility audit and source cohort

**Spec ID:** 002  
**Stage:** B  
**Status:** Validated (reviewed eligibility gate and independent acceptance PASS)  
**Last updated:** 2026-10-05  
**Execution environment:** Local Windows, `.venv-preprocess`  
**Environment gate required:** A1 validated; A2 not required

## 1. Objective

Independently scan the original Yelp sources to verify exactly 10,197 eligible restaurants. Only if this count and structural checks pass, create a new reproducible 2,300-business source dataset and immutable 2,000/200/100 business-level split.

## 2. Scope

In scope: full business/review/photo metadata scans; exact review-ID duplicate checking; local photo filename inventory; original eligibility; deterministic cohort/split; all original reviews and locally available associated photo metadata/images for the selected businesses; manifests, checksums, and acceptance evidence.

Out of scope: English detection/filtering, selecting 30 reviews, token budgets, image decoding/deduplication/filtering (Stage D), captions, targets, model training, renting or uploading. This is the Stage B source pool, not the final LED training dataset or completed local B/C handoff milestone.

## 3. Inputs

- `configs/project.yaml`, validated A1 environment and dependency lock.
- Raw business and review JSONL, `data/yelp_photos/photos.json`, original local photo directory.
- Historical `data/processed/language_stats/restaurant_business_language_stats.csv` and eligibility statistics are independent comparison evidence, not selection inputs.
- Input files/images remain read-only. Paths resolve from the project/configuration, independent of the working directory; package paths are relative.

## 4. Outputs / artifacts

- `outputs/cohort/eligibility_audit.json` and full eligible-business CSV.
- `data/processed/yelp_2300_v1/`: source business JSONL, train/validation/test review JSONL and photo metadata JSONL, all associated locally available original photos, eligible and remaining-business CSVs, cohort/photo manifests, configuration snapshots, README, dataset manifest and SHA-256 inventory.
- `data/processed/splits/business_split.csv`, `split_summary.json`: canonical immutable split; exact copy in the package.
- `outputs/cohort/build_report.json`, independent `validation_report.json`.
- Decision and progress entries.

## 5. Methodology rules and invariants

- Eligibility: exact `Restaurants` category; no intersection with `Hotels`, `Resorts`, `Bed & Breakfast`, `Hostels`, `Vacation Rentals`, `Guest Houses`; at least 10 review records in the raw review file; at least 5 associated photo IDs with local files. `Hotels & Travel` alone is not accommodation exclusion.
- Count real review records, not the business metadata's `review_count`. Five photos is an eligibility minimum, not an image cap.
- Verify exactly 10,197 before writing any new cohort/split. Halt on parsing, duplicate-ID, invalid-join or incomplete-scan failures.
- Raw counts define this original pool. English-only experimental inputs will be constructed in Stage C; detector/settings, rating population, and language/selection ordering remain explicitly pending. Do not silently redefine the 10,197 pool or label raw reviews English-only.
- Sort eligible IDs ascending; use a fresh Python `random.Random(42).sample(sorted_ids, 2300)`; positions 0–1999 train, 2000–2199 validation, 2200–2299 test. Record selection position and runtime version; reuse the saved manifest rather than resampling later.
- CSV rows may be sorted by business ID for readability, but split membership follows the recorded selection positions. Remaining 7,897 IDs are separately recorded.
- Business-level joins only. No test/external tuning or selection of the human-evaluation 30.
- Refuse to overwrite any existing cohort or canonical split directory. Build in a staging directory; promote only after build invariants pass. Keep incomplete staging artifacts identifiable after failure.
- Preserve original records and image bytes. Hash artifacts and every copied image; do not decode or apply image-quality policy here.

## 6. Configuration

Fixed: seed 42, expected pool 10,197, cohort 2,300, split 2,000/200/100, minimum raw reviews 10 and local photos 5; Python 3.11.7 and existing preprocessing lock. No model/tokenizer operation required.

Execution-only setting: eight concurrent independent photo-copy/hash tasks with a bounded queue. Metadata output is drained in raw source order. This does not affect sampling, split membership or image contents. The slow initial serial export was stopped and retained in `.yelp_2300_v1.serial_attempt`; it is not an accepted dataset.

TBD for Stage C onward: English handling, representative review sampling, exact token allocation, image processing, K, target generation, external corpus, human-evaluation strata. No empirical tuning occurs in Stage B.

## 7. Implementation tasks

1. Record spec and D-004 before implementation.
2. Run full read-only core audit with exact review-ID duplicate checking; save evidence and historical comparisons.
3. If the gate passes, sample/split and export source records and all local associated photos with portable manifests.
4. Record original source hashes while streaming export; save output hashes, source file stats, and build summary.
5. Independently validate saved split, joins, counts, partition, deterministic reproduction, and hashes.
6. Update status/logs only when acceptance passes.

## 8. Validation / sanity checks

Full scans; no malformed/non-object JSON; unique IDs; valid references; unchanged source sizes/mtime; exact eligible count and historical ID/count comparison. Validate exact split counts, zero overlap, eligible thresholds for every cohort business, remaining-pool partition, raw subset review counts and photo counts against audit, all records belong to their split, original image bytes equal copied bytes at creation, complete package checksum inventory and identical canonical/package split. Compare fresh seed-42 reproduction to saved IDs/positions. Image readability is deferred explicitly to Stage D.

## 9. Acceptance criteria

- [x] Full raw-source audit reproduces 10,197 and passes the reviewed structural/count gate (D-005).
- [x] Exactly 2,300 unique eligible businesses; exact 2,000/200/100 split; zero overlap.
- [x] Remaining 7,897 partition is disjoint and complete.
- [x] All exported review/photo counts and joins match the audited source pool.
- [x] Every original associated local image is preserved with portable path and SHA-256.
- [x] Independent validation confirms deterministic reproduction and all artifact hashes.
- [x] Raw inputs unchanged; manifests/configuration/progress distinguish Stage B from pending C/D/A2.

Acceptance evidence: `outputs/cohort/validation_report.json` (23/23 checks PASS); 601,493 original reviews and 30,027 original images. Split SHA-256: `f3331ce71da531099349ef680e18ab38255f38db8752dcf61e8998c7a191d93b`. The package's specification snapshot retains its exact build-time state; this validated live spec is outside the immutable package.

## 10. Stop conditions

Do not create a cohort if the verified count differs from 10,197 or structural audit fails. Report the discrepancy rather than altering thresholds. Stop on existing output directories, source changes, copy/checksum errors, split/count mismatches or insufficient space. Ask only if a research-policy change is needed; do not make Stage C choices inside Stage B.

## 11. Expected logs / reports

Audit, historical comparison, build report, independent acceptance report; `THESIS_PROGRESS.md` and `DECISIONS.md`.

## 12. Deviations / amendments

### Final English eligibility amendment (D-006)

This spec's validated v1 artifacts remain the original raw-count source cohort. The user subsequently requires English eligibility before final sampling; see `specs/002b_english_eligibility.md`. Do not treat the earlier English-policy deferral as the current final ordering. Full-pool certification and a fresh post-gate draw are compared to v1 without overwriting it. All original-pool IDs qualify under the recorded detector; the post-gate IDs/splits match v1. Exact English corpus classification and input review selection remain Stage C.

The user's 2026-10-05 request explicitly authorizes creating a fresh 2,300-business dataset if the original 10,197 count is verified. D-004 records the scope: freeze the original source cohort while leaving English input construction in Stage C. No model or evaluation design changes.

### 2026-10-05 — Resolved metadata multiplicity (D-005)

The full baseline scan found two repeated photo IDs with different label values, both belonging to businesses outside the eligible pool. Preserve the original failed structural gate in `eligibility_audit.json` and the investigation evidence. A separate `eligibility_verified.json` and `verified_eligible_businesses.csv` recheck eligibility using unique locally available photo IDs. They confirm the same 10,197 businesses and the same 128,472 eligible photos; no affected business is eligible or selected. This resolves the global metadata warning without choosing a label, deleting raw records, changing membership, or performing Stage D image-content deduplication. The builder must require this reviewed gate and reject any such conflicting metadata inside the eligible pool. Final acceptance refers to this reviewed gate, not the preserved baseline failure.
