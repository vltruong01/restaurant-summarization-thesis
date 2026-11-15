"""Export existing Stage I return evidence; no model execution or output editing."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / "outputs/stage_i/cloud_return_20261015_v1"
CLOUD = RETURN / "stage_hi_v2"
AUDIT = RETURN / "audit_v1"
REFERENCE = ROOT / "outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset"
NAME = "stage_i_chatgpt_review_bundle"
DEST = RETURN / NAME
ZIP = RETURN / "stage_i_chatgpt_review_bundle_v1.zip"
CONDITIONS = ("reviews_only", "unfiltered", "filtered")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if DEST.exists() or ZIP.exists():
        raise FileExistsError("Preserve the existing export; use a new version for another export.")
    sources: dict[str, Path] = {}

    def include(relative: str, source: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        if relative in sources:
            raise ValueError(relative)
        sources[relative] = source

    for name in ("AUDIT_REPORT.md", "report.json", "prediction_diagnostics.csv",
                 "qualitative_notes.json", "review_sample.json", "REVIEW_SAMPLE.html",
                 "FINAL_INTEGRITY_CHECK.json"):
        include(f"audit/{name}", AUDIT / name)
    include("provenance/IMPORT_RECEIPT.json", RETURN / "IMPORT_RECEIPT.json")
    for name in ("authorization.json", "compatibility_fix.json", "feasibility_review.json",
                 "pipeline_status.json", "run_bindings.json", "gpu_metrics.csv", "pip_freeze.txt",
                 "train_reviews_only.log", "train_unfiltered.log", "train_filtered.log"):
        include(f"execution/{name}", CLOUD / "outputs/stage_i/cloud_training_v2" / name)
    include("execution/feasibility_report.json", CLOUD / "outputs/stage_i/runs_v2/feasibility/report.json")
    for condition in CONDITIONS:
        run = CLOUD / "outputs/stage_i/runs_v3" / condition
        include(f"validation/{condition}/teacher_dataset.jsonl", REFERENCE / f"validation_{condition}.jsonl")
        for name in ("validation_predictions.jsonl", "completion_report.json", "runtime_initial.json",
                     "run_binding.json", "trainer_state.json"):
            include(f"validation/{condition}/{name}", run / name)
        include(f"validation/{condition}/adapter_config.json", run / "best_adapter/adapter_config.json")
    for name in ("stage_i_lora_led_training_approved_v2.json", "stage_i_lora_led_v1.json",
                 "led_source_template_d032.json"):
        include(f"configuration/{name}", CLOUD / "configs" / name)
    include("configuration/stage_h_manifest.json", CLOUD / "outputs/stage_h/led_dataset_v1/manifest.json")
    for name in ("run_stage_i_lora_led_cloud_v2.py", "stage_hi_led_data.py"):
        include(f"code/{name}", CLOUD / "scripts" / name)
    include("code/audit_stage_i_cloud_return_v1.py", ROOT / "scripts/audit_stage_i_cloud_return_v1.py")
    include("code/build_stage_i_chatgpt_review_bundle_v1.py", Path(__file__))
    for source in ("thesis_sdd_docs/THESIS_PROGRESS.md", "thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md",
                   "DECISIONS.md", "specs/009_lora_led_training_preparation.md"):
        include(f"documentation/{Path(source).name}", ROOT / source)

    # Scan the explicit text-only whitelist before copying; never include credentials.
    for source in sources.values():
        content = source.read_text(encoding="utf-8-sig")
        if re.search(r"\bsk-[A-Za-z0-9_-]{20,}", content):
            raise RuntimeError(f"Potential credential in whitelisted file: {source.name}")

    references, predictions = {}, {}
    for condition in CONDITIONS:
        def rows(path: Path) -> list[dict]:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        refs = rows(sources[f"validation/{condition}/teacher_dataset.jsonl"])
        preds = rows(sources[f"validation/{condition}/validation_predictions.jsonl"])
        assert len(refs) == len(preds) == 200
        assert len({r["business_id"] for r in refs}) == 200
        assert len({r["business_id"] for r in preds}) == 200
        assert [r["business_id"] for r in refs] == [r["business_id"] for r in preds]
        assert all(r["split"] == "validation" and r["condition"] == condition for r in refs)
        references[condition] = refs
        predictions[condition] = preds
    ids = [r["business_id"] for r in references["reviews_only"]]
    assert all([r["business_id"] for r in references[c]] == ids for c in CONDITIONS)

    DEST.mkdir()
    copied = []
    for relative, source in sources.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_sha = digest(source)
        shutil.copyfile(source, target)
        assert digest(target) == source_sha
        copied.append({"bundle_file": relative, "source_sha256": source_sha,
                       "bytes": target.stat().st_size, "copied_without_modification": True})

    index = {
        "purpose": "Offline independent review of existing Stage I validation outputs",
        "validation_businesses": 200, "condition_aligned_predictions": 600,
        "conditions": list(CONDITIONS),
        "indexing": "Zero-based JSONL record number; business_id is the primary join key.",
        "photos_included": False,
        "photo_scope": "Original caption text/photo IDs and teacher evidence records are included. Original photos are not; do not claim independent image verification.",
        "businesses": [
            {"business_id": bid, "split": "validation", "records": {
                c: {"jsonl_record_number_zero_based": i,
                    "teacher_source_evidence_file": f"validation/{c}/teacher_dataset.jsonl",
                    "student_prediction_file": f"validation/{c}/validation_predictions.jsonl",
                    "selected_review_ids": references[c][i]["selected_review_ids"],
                    "selected_photo_ids": references[c][i]["selected_photo_ids"]}
                for c in CONDITIONS}} for i, bid in enumerate(ids)]}
    save_json(DEST / "INDEX.json", index)

    (DEST / "README_FIRST.md").write_text("""# Stage I production training: independent review bundle

This is an export of existing results, not a new experiment. Open `audit/REVIEW_SAMPLE.html` locally for the comparison viewer. In ChatGPT, extract the ZIP and start with `CHATGPT_REVIEW_REQUEST.md`, `audit/AUDIT_REPORT.md` and `audit/report.json`.

## Included evidence

- All 600 existing validation predictions, in three original JSONL files; decoded text, token IDs and generation flags are unchanged.
- All 600 corresponding original Stage G validation records, with complete source/target text, REV/caption/photo IDs, exact evidence records and provenance. These retain original artifact-path strings as historical provenance; review their embedded evidence rather than resolving old machine paths.
- `INDEX.json` joins every business/condition to its teacher/source/evidence and student-output record using portable relative paths.
- All-600 repetition/format diagnostics; ten fixed-seed random businesses plus four separately marked warning businesses in a static viewer. Random sample seed20261015. Warning examples are not an independent random sample.
- Actual runtime configs, pinned source template, Stage H manifest, Trainer states, completion reports, training logs, GPU measurements, run bindings, feasibility evidence, authorization and original executed code. No adapter weights or optimizer binaries are needed to read the results.
- Current progress/decision/spec context. Earlier dated pending/not-run statements are historical; the newest cloud-return checkpoint supersedes them.

## Interpretation boundaries

Technical training/data checks PASS; generated-output quality requires review. Repetition is an exact textual diagnostic, not a human factual-error rate. Full-target validation losses have different target texts across conditions and cannot determine a final model ranking. Correct but simple descriptions are not automatically factual errors.

Original photos are not included. Captions, photo IDs and teacher evidence are included; assess caption/source grounding without claiming independent photo verification. Teacher verdicts and AI preliminary audit notes are not human ground truth. No held-out test examples/results, credentials, Teacher requests, new generation, target editing or retraining are part of this export.

All source copies are byte-identical. `EXPORT_REPORT.json` lists source-copy hashes; `MANIFEST.json` lists every payload file hash. The external ZIP SHA-256 file verifies transport. No research setting or student prediction was changed to prepare this package.
""", encoding="utf-8")
    (DEST / "CHATGPT_REVIEW_REQUEST.md").write_text("""# Request for independent Stage I quality review

We trained three LoRA-LED models (Reviews-only, Unfiltered, Filtered) on the same frozen2000/200 train/validation businesses. Training is complete; held-out test evaluation has not started. Please inspect this ZIP using the files, not only the preliminary conclusions.

1. Read the audit and actual settings/logs. Confirm what technical checks establish and what they do not establish. Each run has5epochs/1250optimizer steps,294912trainable parameters in decoder q_proj/v_proj adapters, rank8/alpha16, learning rate2e-4, batch1/accum8, beam4, generation max_length768, no_repeat_ngram_size0. Source/review/caption caps5000/4000/750 and K5 are unchanged.
2. Inspect the full600 prediction diagnostics and paired original validation records through INDEX.json. Independently check repetition and loss of specific information. Distinguish correct simple content, unsupported source/caption claims, wrong formatting and generation loops. Give business_id, condition and exact excerpts when making a claim.
3. Examine zero-Filtered-caption business `j3kGicU3mPG_ciIKG1Vuyg`: empty visual input/teacher but generated bagel/tray/cream-cheese text. Also inspect `OpnIXsrqaK-e8Wz5gco7Vw` (loops reaching768), `JcdOQvvx3BqZ5Ga2_9n_Bg` (repetition/unsupported relation), and wrong-header cases. Keep these selected warning examples separate from the fixed random sample.
4. Explain plausible hypotheses for the generation problems without treating an untested cause as established. Distinguish decoding effects from insufficient task learning or limited adaptation scope. Lower validation loss and low VRAM use alone do not establish summary quality.
5. Propose the smallest bounded validation-only diagnostic plan before any retraining. State any proposed decoding/training change explicitly and how equivalent settings across RO/U/F would be preserved. Do not propose a change merely to fill GPU memory or make Filtered win.

Do not edit the existing outputs/targets, treat teacher verification as independent human truth, suggest held-out tuning, or claim original-photo verification because photos are not in this ZIP. Any new execution or scientific change will require researcher review. Return findings, uncertainty, evidence-linked examples and a prioritized diagnostic plan.
""", encoding="utf-8")

    report = {"status": "EXPORT_ONLY_PASS", "copied_source_files": len(copied),
              "validation_businesses": 200, "predictions": 600, "teacher_reference_records": 600,
              "viewer_random_businesses": 10, "viewer_additional_warning_businesses": 4,
              "source_copy_integrity": copied, "photos_included": False,
              "outputs_modified": False, "teacher_API_calls": 0, "new_generation_or_training": False,
              "held_out_test_results_used": False}
    save_json(DEST / "EXPORT_REPORT.json", report)
    payload = [{"path": p.relative_to(DEST).as_posix(), "bytes": p.stat().st_size,
                "sha256": digest(p)} for p in sorted(DEST.rglob("*")) if p.is_file()]
    save_json(DEST / "MANIFEST.json", {"files": payload, "manifest_excludes_itself": True})
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, f"{NAME}/{path.relative_to(DEST).as_posix()}")
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for entry in payload:
            assert hashlib.sha256(archive.read(f"{NAME}/{entry['path']}")).hexdigest() == entry["sha256"]
        for entry in copied:
            assert digest(sources[entry["bundle_file"]]) == entry["source_sha256"]
    checksum = digest(ZIP)
    ZIP.with_suffix(".zip.sha256").write_text(f"{checksum}  {ZIP.name}\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "zip": str(ZIP), "bytes": ZIP.stat().st_size,
                      "MiB": round(ZIP.stat().st_size / 1024**2, 2), "sha256": checksum,
                      "payload_files": len(payload) + 1, "predictions": 600}, indent=2))


if __name__ == "__main__":
    main()
