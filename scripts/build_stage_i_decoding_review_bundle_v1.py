"""Portable export of the existing offline decoding diagnostic; no inference or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage_i/decoding_diagnostic_v1"
CLOUD = ROOT / "outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2"
NAME = "stage_i_decoding_diagnostic_chatgpt_bundle_v1"
DEST = ROOT / "outputs/stage_i" / NAME
ZIP = DEST.with_suffix(".zip")
CONDITIONS = ("reviews_only", "unfiltered", "filtered")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    if DEST.exists() or ZIP.exists():
        raise FileExistsError("Preserve an existing export; choose a fresh version")
    manifest = read(PREP / "PREPARATION_MANIFEST.json")
    subset = read(PREP / "subset_manifest.json")
    sources = {}
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        source = PREP / relative
        assert source.stat().st_size == entry["bytes"] and sha(source) == entry["sha256"]
        sources[f"prepared/{relative.as_posix()}"] = source
    sources["prepared/PREPARATION_MANIFEST.json"] = PREP / "PREPARATION_MANIFEST.json"
    for path, expected in subset["code_bindings"].items():
        assert sha(ROOT / path) == expected, path
        sources[f"code/{Path(path).name}"] = ROOT / path
    for path in ("scripts/test_stage_i_decoding_diagnostics.py", "scripts/build_stage_i_decoding_review_bundle_v1.py"):
        sources[f"code/{Path(path).name}"] = ROOT / path
    for path in ("thesis_sdd_docs/THESIS_PROGRESS.md", "thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md",
                 "DECISIONS.md", "specs/009b_validation_decoding_diagnostic.md"):
        sources[f"documentation/{Path(path).name}"] = ROOT / path
    sources["configuration/stage_i_decoding_diagnostic_v1.json"] = ROOT / "configs/stage_i_decoding_diagnostic_v1.json"
    sources["training_provenance/stage_i_lora_led_training_approved_v2.json"] = CLOUD / "configs/stage_i_lora_led_training_approved_v2.json"
    sources["training_provenance/backbone_config.json"] = ROOT / "assets/models/led-base-16384/config.json"
    sources["training_provenance/original_executed_runner.py"] = CLOUD / "scripts/run_stage_i_lora_led_cloud_v2.py"
    for name in ("AUDIT_REPORT.md", "report.json"):
        sources[f"training_provenance/cloud_return_audit/{name}"] = ROOT / "outputs/stage_i/cloud_return_20261015_v1/audit_v1" / name
    for condition in CONDITIONS:
        run = CLOUD / f"outputs/stage_i/runs_v3/{condition}"
        for name in ("completion_report.json", "runtime_initial.json", "trainer_state.json", "run_binding.json"):
            sources[f"training_provenance/{condition}/{name}"] = run / name
        sources[f"training_provenance/{condition}/adapter_config.json"] = run / "best_adapter/adapter_config.json"

    cases = rows(PREP / "cases.jsonl")
    baseline = {c: {r["business_id"]: r for r in rows(CLOUD / f"outputs/stage_i/runs_v3/{c}/validation_predictions.jsonl")} for c in CONDITIONS}
    references = {c: {r["business_id"]: r for r in rows(ROOT / f"outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/validation_{c}.jsonl")} for c in CONDITIONS}
    population = sorted(baseline["reviews_only"])
    assert len(population) == 200
    assert hashlib.sha256(json.dumps(population).encode()).hexdigest() == subset["population_sorted_ids_sha256"]
    assert random.Random(subset["seed"]).sample(population, 20) == subset["random_business_ids"]
    assert len(cases) == 72 and len(subset["execution_business_ids"]) == 24
    assert subset["overlap_ids"] == []
    for case in cases:
        c, bid = case["condition"], case["business_id"]
        assert case["split"] == "validation"
        assert case["baseline_prediction"] == baseline[c][bid]
        assert all(case[key] == value for key, value in references[c][bid].items())
    assert read(PREP / "approval_EXAMPLE_NOT_APPROVED.json")["approved"] is False
    assert read(PREP / "offline_checks.json")["experiments"]["greedy_A"] == "PREPARED_NOT_EXECUTED"

    # Explicit whitelist only. Export no credentials or model/optimizer binaries.
    for source in sources.values():
        content = source.read_text(encoding="utf-8-sig")
        if re.search(r"\bsk-[A-Za-z0-9_-]{20,}", content):
            raise RuntimeError(f"Potential credential in source: {source.name}")
    source_checks = []
    DEST.mkdir()
    for relative, source in sources.items():
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_sha = sha(source)
        shutil.copyfile(source, destination)
        assert sha(destination) == original_sha
        source_checks.append({"file": relative, "bytes": destination.stat().st_size,
                              "sha256": original_sha, "source_copy_unchanged": True})
    write_json(DEST / "validation_population_ids.json", {"split": "validation", "sorted_business_ids": population,
               "count": 200, "sampling_seed": subset["seed"],
               "reproduction": "random.Random(20261015).sample(sorted_business_ids,20)",
               "note": "Full200 IDs only; complete source/teacher/baseline evidence is included for the selected24 businesses, not the other176."})
    (DEST / "README_FIRST.md").write_text("""# Stage I decoding diagnostic: ChatGPT discussion bundle

**This package contains OFFLINE PREPARATION and EXISTING BASELINE ONLY. Diagnostic A/B have NOT been run.**

Read in order:
1. `CHATGPT_REVIEW_REQUEST.md`
2. `prepared/PREPARATION_REPORT.md`
3. `prepared/EXECUTION_PLAN.md` and `prepared/cost_estimate.json`
4. `prepared/subset_manifest.json` and `prepared/decoding_configurations.json`
5. `prepared/baseline_evaluation/summary.json`, metrics and evidence-linked comparison cases.

Open `prepared/baseline_evaluation/COMPARISON.html` in a browser for the side-by-side viewer. Baseline mode deliberately repeats the SAME saved prediction in the baseline/diagnostic columns; it does not display greedy results.

## Scope

-20 fixed-seed random validation businesses plus4 previously identified targeted warning cases; no overlap,24 unique businesses and72 RO/U/F cases. Same IDs across conditions. `validation_population_ids.json` provides all200 validation IDs so the random draw can be independently reproduced. Do not mix targeted4 rates with random20 prevalence.
-`prepared/cases.jsonl` contains complete selected-review sources, caption/photo IDs, exact original teacher targets/evidence, original token IDs and saved student baseline predictions for the24 businesses. All sources/targets/predictions were compared with authoritative originals during export.
-Full200-per-condition baseline statistics are in `prepared/baseline_all200_context.json`; complete source/prediction records for the other176 businesses are not included here. The earlier full training-review ZIP is a separate preserved artifact.
-Proposed A changes only num_beams4->1; proposed B changes only no_repeat_ngram_size0->4 at beam4. All other explicit decoding settings remain unchanged. Neither is executed or frozen for production.
-Code, offline acceptance tests, exact training config/checkpoint metadata and current method/progress documents are included. Model weights, optimizer binaries, original photos, Teacher credentials, Codex settings and held-out data are excluded. This is a discussion bundle, not a standalone GPU-runtime installation.

## Interpretation and approval

ROUGE measures teacher-target agreement, not factual correctness. Exact repeated sentences, adjacent-word repetition, formatting, empty-visual behavior, review coverage and caption grounding are separate dimensions. Human coverage/contrast/support/factual fields remain blank; lexical matches and teacher verdicts are not independent human truth. Correct simple visual descriptions are not errors merely for lacking detail. Original photos are absent; do not claim independent image verification.

The full original GenerationConfig was not archived. Its offline reconstruction is labeled; a future approved pinned-backbone load must record the actual configuration and stop on unexpected differences. Adapter metadata revision=null does not establish that the original runner loaded the wrong model; its actual config/code explicitly pinned the original revision.

The old approved training config documents completed training ONLY. It is NOT approval for this new diagnostic. The included approval example has approved=false. No new inference, rental, upload, credit spending, teacher requests, retraining, target editing or held-out tuning occurred. Price/time ranges are planning assumptions, not a purchased offer or measured greedy throughput.

All prepared files and source copies are byte-identical. `EXPORT_REPORT.json` records integrity checks; `MANIFEST.json` lists ZIP payload hashes. Some original records/logs retain machine paths as provenance; use the included relative files and embedded source/evidence instead of resolving those old paths.
""", encoding="utf-8")
    (DEST / "CHATGPT_REVIEW_REQUEST.md").write_text("""# Request: independently review the proposed decoding diagnostic before GPU execution

We reviewed completed three-condition LoRA-LED training and found severe repetition and a zero-caption grounding failure. We chose to investigate decoding on validation before considering retraining. Codex has now prepared a controlled diagnostic but has NOT generated any new A/B output.

Please inspect this ZIP and discuss:

1. Is the fixed20-random+4-targeted validation design appropriate? Reproduce the seed20261015 draw from validation_population_ids.json and confirm same IDs across RO/U/F, no overlap and separate reporting denominators.
2. Are proposed A (greedy, num_beams1/do_samplefalse/no_repeat_ngram0) and separate proposed B (beam4/no_repeat_ngram4) controlled relative to the saved baseline? Check complete configs, especially unchanged768 cap, EOS/start behavior, precision, global attention and checkpoint/backbone pinning. No additional configurations have approval.
3. Review saved baseline metrics and source/teacher/student examples. Distinguish repetition, malformed headers, unsupported caption claims, source-specific information loss and correct simple content. Explain the zero-caption F case without treating review-derived food knowledge as permitted visual evidence. Give exact business IDs/conditions/excerpts for factual statements.
4. Are the evaluation definitions and blank human review/visual worksheets adequate? ROUGE is teacher agreement only. Lexical matches do not establish semantic support, factual truth or preservation of important contrasting opinions; do not invent human labels or combined quality scores.
5. Assess configuration findings as evidence, not assumed causes: original full GenerationConfig missing, adapter revision=null but original runner explicitly pinned the base, correct recorded special tokens/global masks and best-checkpoint selection. Which hypotheses can A help distinguish? What remains unresolved without an actual approved run?
6. Review the conservative GPU time/cost assumptions and recommendation to run A first. Suggest the smallest justified diagnostic next step. Do not advocate retraining/module expansion merely to fill VRAM, tune on test, make Filtered win, or accept A solely because repetition declines.

Return a concise assessment, any plan/implementation concerns, and whether to approve A as prepared or request a specific change. A/B results do not exist yet; never describe the duplicated baseline viewer columns as a measured greedy comparison. Do not generate or rewrite teacher/student targets. Actual GPU execution, paid actions and any scientific change remain subject to researcher approval.
""", encoding="utf-8")
    write_json(DEST / "EXPORT_REPORT.json", {"status": "EXPORT_ONLY_PASS", "copied_files": len(source_checks),
               "source_copy_integrity": source_checks, "random_businesses": 20, "targeted_businesses": 4,
               "overlap": [], "unique_businesses": 24, "condition_aligned_cases": 72,
               "full_validation_population_IDs": 200, "source_teacher_and_baseline_unchanged": True,
               "A_B_results_exist": False, "new_model_inference": False, "paid_actions": False,
               "teacher_requests": 0, "held_out_data": False, "photos_included": False})
    payload = [{"path": p.relative_to(DEST).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(DEST.rglob("*")) if p.is_file()]
    write_json(DEST / "MANIFEST.json", {"files": payload, "manifest_excludes_itself": True})
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                archive.write(p, f"{NAME}/{p.relative_to(DEST).as_posix()}")
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for entry in payload:
            assert hashlib.sha256(archive.read(f"{NAME}/{entry['path']}")).hexdigest() == entry["sha256"]
    for relative, source in sources.items():
        assert sha(source) == sha(DEST / relative)
    checksum = sha(ZIP)
    ZIP.with_suffix(".zip.sha256").write_text(f"{checksum}  {ZIP.name}\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "zip": str(ZIP), "bytes": ZIP.stat().st_size,
                      "MiB": round(ZIP.stat().st_size / 1024**2, 2), "sha256": checksum,
                      "payload_files": len(payload) + 1}, indent=2))


if __name__ == "__main__":
    main()
