"""Export completed Diagnostic A evidence only; no evaluation/model/API execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "stage_i_diagnostic_A_chatgpt_bundle_v1"
DEST = ROOT / "outputs/stage_i" / NAME
ZIP = DEST.with_suffix(".zip")
PREP = ROOT / "outputs/stage_i/decoding_diagnostic_v1"
RESULT = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1"
RUN = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2"
CONTROL = ROOT / "outputs/stage_i/decoding_A_execution_control_v1"
CONDITIONS = ("reviews_only", "unfiltered", "filtered")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    if DEST.exists() or ZIP.exists():
        raise FileExistsError("Preserve previous exports; choose a new version")
    sources = {}

    def include(relative, path):
        assert path.is_file() and relative not in sources
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        sources[relative] = path

    for entry in read(RESULT / "EVALUATION_MANIFEST.json")["files"]:
        path = RESULT / entry["path"]
        assert sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
        include(f"results/{entry['path']}", path)
    include("results/EVALUATION_MANIFEST.json", RESULT / "EVALUATION_MANIFEST.json")
    for name in ("cases.jsonl", "subset_manifest.json", "decoding_configurations.json", "baseline_all200_context.json"):
        include(f"inputs/{name}", PREP / name)
    include("inputs/frozen_source_template.json", ROOT / "configs/led_source_template_d032.json")
    include("execution/run_binding.json", RUN / "run_binding.json")
    for name in ("connection_preflight.json", "local_v2_runner.log", "local_v2_gpu_metrics.csv",
                 "local_v2_orchestration.json", "local_v2_runtime_diff.patch", "local_v2_evaluation.log"):
        include(f"execution/control/{name}", CONTROL / name)
    for path in sorted((RUN / "attempts").rglob("*")):
        if path.is_file():
            assert path.suffix == ".json", path.name
            include(f"execution/attempts/{path.relative_to(RUN / 'attempts').as_posix()}", path)
    for condition in CONDITIONS:
        include(f"predictions/{condition}/predictions.jsonl", RUN / condition / "predictions.jsonl")
        paths = sorted((RUN / condition / "cases").glob("*.json"))
        assert len(paths) == 24
        for path in paths:
            include(f"predictions/{condition}/cases/{path.name}", path)
    include("execution/local_A_approval.json", ROOT / "outputs/stage_i/decoding_approval_A_local_v2.json")
    for relative in ("scripts/run_stage_i_decoding_diagnostic.py", "scripts/run_stage_i_decoding_diagnostic_local_v2.py",
                     "scripts/stage_i_decoding_diagnostics.py", "scripts/stage_hi_led_data.py",
                     "scripts/prepare_stage_i_decoding_diagnostic.py", "scripts/test_stage_i_decoding_diagnostics.py",
                     "scripts/evaluate_stage_i_decoding_A_local_v2.py", "scripts/report_stage_i_decoding_A_local_v2.py",
                     "scripts/build_stage_i_diagnostic_A_results_bundle_v1.py"):
        include(f"code/{Path(relative).name}", ROOT / relative)
    for relative in ("thesis_sdd_docs/THESIS_PROGRESS.md", "thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md",
                     "DECISIONS.md", "specs/009b_validation_decoding_diagnostic.md"):
        include(f"documentation/{Path(relative).name}", ROOT / relative)

    subset = read(PREP / "subset_manifest.json")
    for name, expected in subset["preparation_bindings"].items():
        assert sha(PREP / name) == expected
    cases = rows(PREP / "cases.jsonl")
    comparison = read(RESULT / "comparison_cases.json")
    assert len(cases) == len(comparison) == 72
    indexed = {(c["business_id"], c["condition"]): c for c in comparison}
    predictions = {c: rows(RUN / c / "predictions.jsonl") for c in CONDITIONS}
    completion_files = list((RUN / "attempts").glob("*/completion_report.json"))
    assert len(completion_files) == 1
    completion = read(completion_files[0])
    assert completion["status"] == "ALL72_DIAGNOSTIC_PREDICTIONS_COMPLETE"
    for c in CONDITIONS:
        assert len(predictions[c]) == 24 and sha(RUN / c / "predictions.jsonl") == completion["predictions_sha256"][c]
        assert [p["business_id"] for p in predictions[c]] == subset["execution_business_ids"]
    for case in cases:
        key = case["business_id"], case["condition"]
        comp = indexed[key]
        pred = next(p for p in predictions[case["condition"]] if p["business_id"] == case["business_id"])
        assert case["split"] == "validation" and comp["source"] == case["source"] and comp["teacher"] == case["target"]
        assert comp["baseline"] == case["baseline_prediction"]["text"] and comp["diagnostic"] == pred["text"]
    population_path = ROOT / "outputs/stage_i/stage_i_decoding_diagnostic_chatgpt_bundle_v1/validation_population_ids.json"
    population = read(population_path)["sorted_business_ids"]
    assert len(population) == 200
    assert random.Random(subset["seed"]).sample(population, 20) == subset["random_business_ids"]
    include("inputs/validation_population_ids.json", population_path)
    for path in sources.values():
        if re.search(r"\bsk-[A-Za-z0-9_-]{20,}", path.read_text(encoding="utf-8-sig")):
            raise RuntimeError(f"Potential credential in {path.name}")

    DEST.mkdir()
    copied = []
    for relative, source in sources.items():
        destination = DEST / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, destination)
        assert sha(destination) == digest
        copied.append({"file": relative, "bytes": destination.stat().st_size, "sha256": digest, "source_copy_unchanged": True})
    case_positions = {(c["condition"], c["business_id"]): i for i, c in enumerate(cases)}
    index = {"businesses": [{"business_id": bid, "split": "validation",
              "subset": "random" if bid in subset["random_business_ids"] else "targeted",
              "conditions": {c: {"source_teacher_baseline_file": "inputs/cases.jsonl",
                    "source_teacher_baseline_row_zero_based": case_positions[(c, bid)],
                    "baseline_field": "baseline_prediction",
                    "greedy_file": f"predictions/{c}/predictions.jsonl", "greedy_row_zero_based": i}
                    for c in CONDITIONS}} for i, bid in enumerate(subset["execution_business_ids"])],
             "counts": {"random": 20, "targeted": 4, "unique_businesses": 24, "greedy_outputs": 72},
             "all_file_paths_relative_to_bundle_root": True}
    for business in index["businesses"]:
        for record in business["conditions"].values():
            assert (DEST / record["source_teacher_baseline_file"]).is_file() and (DEST / record["greedy_file"]).is_file()
    write_json(DEST / "INDEX.json", index)
    (DEST / "README_FIRST.md").write_text("""# Stage I Diagnostic A: completed results for independent ChatGPT review

**This is the COMPLETED greedy-A result bundle, superseding the earlier preparation-only bundle for this discussion. B has NOT been run.**

Read:
1. `CHATGPT_REVIEW_REQUEST.md`
2. `results/DIAGNOSTIC_REPORT.md`
3. `results/paired_summary.json` and `results/baseline_greedy_comparison.csv`
4. `results/AI_QUALITATIVE_REVIEW.json`
5. `results/COMPARISON.html` (open locally in a browser) or `results/comparison_cases.json`.

## Evidence and navigation

- Same20 fixed-seed random validation businesses plus4 separately targeted warning cases, no overlap,24 unique businesses/72 RO/U/F outputs. All200 validation IDs are included to reproduce sampling; no held-out data. Do not merge targeted rates into random-sample prevalence.
- `inputs/cases.jsonl` includes complete original selected reviews/REV IDs, captions/photo IDs, teacher targets/evidence, frozen token IDs and exact saved beam4 baseline predictions.
- `predictions/{condition}/predictions.jsonl` contains all72 new greedy outputs with token IDs. Raw per-case files preserve warnings and memory measurements. `INDEX.json` joins each case to source/teacher/baseline/greedy by relative path and zero-based JSONL record number.
- `results/COMPARISON.html` now shows actual teacher / saved beam4 / newly generated greedy-A columns, not duplicated baseline placeholders.
- Actual effective generation configs, approval, execution logs, runtime/memory, original code and local-runtime diff are included. The old24GB runner and original preparation remain preserved; user explicitly approved local RTX3050 execution. Code is supplied for inspection; this review ZIP omits model weights and is not a standalone runtime installer.

## Findings to verify rather than assume

Random20 exact-sentence repetition beam4->A: RO5->2, U19->20, F15->20. Zero-caption F remains0/1 compliant. Two cap-loop outputs terminate earlier and regain visual headers, but substantive grounding failures persist. The preliminary recommendation is not to adopt greedy and to discuss B separately before any retraining. Inspect the evidence independently; Codex qualitative judgments are AI assessments, not human-final labels.

Only num_beams4->1 changed in decoding. Same checkpoints, complete inputs, precision, source/token budgets,768 limit, tokenizer and pinned backbone; no quantization/offload or output rewriting. Baseline ran on RTX3090; greedy ran on researcher-approved RTX3050 Laptop6GB. Different hardware is disclosed, so no isolated hardware-independent speed/bitwise claim is made. No inference failure or retries;72 outputs generated once. Pipeline507.77s, summed generation439.13s, sampled whole-GPU peak3460MiB. No new rental/Teacher/API charge; local electricity cost was not measured.

## Interpretation boundaries

Keep formatting, repetition, unsupported caption claims, information coverage and genuinely incorrect review statements separate. Simple supported descriptions are acceptable. ROUGE is teacher-target agreement, not factual accuracy. Original photos are NOT included, so caption grounding must not be reported as independent image verification. Teacher verdicts and AI qualitative notes are not human truth; human worksheet fields remain blank.

Some location information is true in business metadata but unsupported as VISUAL_EVIDENCE; do not call a true metadata fact false merely because it appears under the wrong evidence section. Review-only ingredients/actions in visual output are evidence-binding concerns. No source, target, caption, filter, split, adapter or original prediction was changed for this export. No Teacher calls, new generation/evaluation, B, retraining, paid action or held-out tuning was performed by the exporter.

`EXPORT_REPORT.json` records integrity checks/source hashes; `MANIFEST.json` lists payload hashes; the ZIP has an external SHA256 sidecar. Original records retain machine-path strings as provenance, but all source/evidence text required for review is embedded and the index uses relative paths. Historical offline-only statuses in preserved preparation/documents describe earlier dates; the latest completed checkpoint and actual result files take precedence.
""", encoding="utf-8")
    (DEST / "CHATGPT_REVIEW_REQUEST.md").write_text("""# Independently review completed Diagnostic A before any next experiment

We approved and completed greedy Diagnostic A on the fixed24 validation businesses across RO/U/F, exactly72 outputs. Please inspect this result ZIP, not only its preliminary conclusion.

1. Verify execution integrity and actual generation settings: saved beam4 baseline versus new greedy beam1, do_samplefalse,768 cap, all other settings/checkpoints/inputs/precision unchanged. Note user-approved RTX3090-baseline versus RTX3050-greedy hardware difference and distinguish it from a decoder-parameter change.
2. Compare the random20 and targeted4 separately: exact sentence repetition, adjacent-word repetition, header compliance, lengths, EOS/cap and zero-caption visual behavior. Do not pool targeted cases into prevalence or equate EOS with completeness.
3. Read representative source/teacher/student comparisons and AI qualitative notes critically. Separate supported simple content from repetitive low information, unsupported object/action/ingredient/relationship claims, true metadata placed in the wrong evidence section, and actual review-category errors. Cite business_id, condition and exact REV/IMG_CAP/PHOTO IDs where applicable.
4. Assess important review-specific facts and contrasting opinions, not just ROUGE. Inspect the new pasta-under-pizza classification, review-only bergamot in visual output, persistent zero-caption F statements, recovered but ungrounded Opn visual sections, and the7u F output that now lacks VISUAL_EVIDENCE despite ending before768. Also acknowledge local improvements such as the caption-supported fork detail and removal of one malformed liquid relation.
5. Is the conclusion 'do not adopt greedy' justified? Explain which hypotheses remain plausible: beam effects, generic imitation/insufficient learning, evidence binding or decoder-only LoRA limitations. Do not treat a possible cause as established or claim independent photo verification.
6. Recommend the smallest next step: the already proposed B (beam4/no_repeat_ngram4) under separate approval, or examination of training limitations. B may suppress duplicates by construction while forcing other unsupported details; do not optimize only repetition counts. No retraining/module changes, new parameters, teacher regeneration or held-out tuning are approved.

Return a concise evidence-grounded assessment, any errors in Codex's interpretation, examples of real improvements/regressions, remaining uncertainty and a recommended next decision. Do not rewrite teacher/student targets or manufacture human labels. A is complete; B has not been run, and no next execution is automatically authorized.
""", encoding="utf-8")
    write_json(DEST / "EXPORT_REPORT.json", {"status": "EXPORT_ONLY_PASS", "copied_source_files": len(copied),
               "files": copied, "random_businesses": 20, "targeted_businesses": 4, "overlap": 0,
               "greedy_outputs": 72, "saved_baseline_outputs": 72, "source_teacher_and_predictions_unchanged": True,
               "relative_index_paths_valid": True, "original_photos_included": False,
               "new_generation_or_evaluation_during_export": False, "B_run": False, "teacher_calls": 0,
               "training": False, "held_out_data": False, "paid_actions": False})
    payload = [{"path": p.relative_to(DEST).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(DEST.rglob("*")) if p.is_file()]
    write_json(DEST / "MANIFEST.json", {"files": payload, "manifest_excludes_itself": True})
    with zipfile.ZipFile(ZIP, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, f"{NAME}/{path.relative_to(DEST).as_posix()}")
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for entry in payload:
            data = archive.read(f"{NAME}/{entry['path']}")
            assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    for relative, source in sources.items():
        assert sha(source) == sha(DEST / relative)
    digest = sha(ZIP)
    ZIP.with_suffix(".zip.sha256").write_text(f"{digest}  {ZIP.name}\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "zip": str(ZIP), "bytes": ZIP.stat().st_size,
                      "MiB": round(ZIP.stat().st_size / 1024**2, 2), "sha256": digest,
                      "files": len(payload) + 1, "greedy_outputs": 72}, indent=2))


if __name__ == "__main__":
    main()
