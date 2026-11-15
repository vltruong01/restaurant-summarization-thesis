# Stage Specification — LED dataset construction

**Spec ID:** 008  
**Stage:** H  
**Status:** Implemented / data acceptance validated; decoder-policy approval pending  
**Last updated:** 2026-10-14  
**Execution environment:** Local CPU; portable artifacts for a later approved GPU run  
**Environment gate:** A1; Stage G complete; no A2/training execution here

## 1. Objective

Reuse the six completed Stage G JSONL datasets without changing any teacher text. Prepare complete deterministic LED input/label IDs for RO/U/F, with original IDs, order, frozen inputs and exact target/evidence provenance. Stage H data acceptance is separate from approval of the student decoder limit and GPU training.

## 2. Scope

In scope: final read-only Stage G audit, Stage F/source equality, split integrity, exact tokenizer accounting, packed token arrays, checksums, statistics, candidate-length audit and collator tests. Out of scope: teacher calls, target editing, Stage F changes, test targets/results, GPU rental/upload, model forward/backward and training.

## 3. Inputs

- `outputs/stage_g/production_manifest_pointer.json` and its SHA-bound completion report, six JSONL files, final validation/length/readiness reports and existing target artifacts.
- `outputs/stage_f/production_manifest_pointer.json`; only train/validation source/caption manifests are opened.
- Frozen `data/processed/yelp_2300_v1/business_split.csv`: test IDs are checked only for exclusion, with no test captions/reviews/references opened.
- D-032 template and tokenizer `allenai/led-base-16384@38335783885b338d93791936c54bb4be46bebed9` with existing local asset hashes.

## 4. Outputs

`outputs/stage_h/led_dataset_v1/`: manifest, validation report, human-readable acceptance report, full per-row token-length CSV, target-limit candidate report and six compressed ragged-array NPZ files. NPZ contains complete input/label IDs and row IDs/hashes, not duplicate source/target/evidence text. JSONL provenance remains in Stage G. Paths are workspace-relative; binary loading forbids pickle.

## 5. Invariants

- Exactly 2,000 train and 200 validation businesses per condition; frozen business membership/order and review/photo IDs.
- Source, target and shared review evidence unchanged; reproduce every source from the existing Stage F source/caption records, not new selection.
- Full source <=5,000; marked review block <=4,000; full caption increment relative to RO <=750; K<=5; no truncation/removal. Markers remain ordinary BPE text, not newly registered special tokens.
- Complete labels include the original tokenizer BOS/EOS. Stored labels have no padding; collator masks only added label padding with -100.
- Preserve the visual header with empty body for the ten zero-Filtered-caption train/validation businesses. No fabricated visual content.
- Shared REVIEW_CONSENSUS byte-equivalent across conditions; final visual sentences must map to teacher-verified claims and exact caption quotes in the existing evidence.
- Candidate target limit 768 is audited, **not approved/frozen**. Construction stores all original labels independently of a cap.
- Training and generation limits are distinct accounting conventions: labels contain BOS/EOS; generation contains the initial decoder-start token as well. Neither an EOS nor reaching a cap certifies a semantically complete generated summary.

## 6. Configuration

`configs/stage_h_led_dataset_v1.json`; budgets/seed42/tokenizer revision inherited. No empirical tuning. Compare diagnostic limits 512/640/768/1024 using train/validation length distributions only. Generation-full-sequence headroom is checked conservatively as label length + one decoder-start token. Decoder limit remains a researcher decision.

## 7. Implementation and validation

Verify hashes before opening upstream datasets; check every row against Stage F and its teacher target artifact; verify exact quotations and condition-specific visual support; audit business/review/photo split intersections; encode without padding/truncation; save complete int32 ragged IDs; reload and compare all arrays and provenance hashes; recheck upstream hashes. Repeat tokenization signatures without creating a second text dataset. Validate the actual training collator on CPU, including mixed lengths, attention-window padding, global attention and label padding.

## 8. Acceptance

- All six datasets pass counts, mapping, exact source/target equality, quotation and token checks.
- All arrays round-trip; no content is truncated; every zero-caption case is valid.
- No train/validation/test-business overlap or selected train/validation review/photo ID overlap.
- Reports explicitly preserve Stage G diagnostic warnings and distinguish structural PASS from factual certification.
- Training stays blocked pending decoder/optimization approval and actual A2 GPU feasibility.

## 9. Stops and provenance

Stop on any hash/mapping/budget/label failure; do not correct content. Preserve previous Stage F/G artifacts. Record D-044 offline construction and proposed D-045 length policy, plus `THESIS_PROGRESS.md`. No scientific parameter is adopted by an offline PASS.

## 10. Actual acceptance result — 2026-10-14

6,600 complete pairs across six files PASS; 144,379 canonical review support entries, 39,159 verified-visual support bindings and 2,200 target-artifact hashes checked. Two full tokenization passes produced identical arrays/signatures and unchanged upstream hashes. All counts/splits/conditions/quotes/budgets/zero-caption invariants pass; zero source/label truncations. Compressed data39,329,935bytes (37.51MiB). Reports: `validation_report.json`, `repeat_validation.json`, `ACCEPTANCE_REPORT.md`. This accepts data construction only. Common target limit and Stage I GPU execution/training remain unapproved.
