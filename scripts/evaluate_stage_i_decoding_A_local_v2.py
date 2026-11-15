"""Evaluate exactly the approved existing72 local greedy results; never generates text."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer

from stage_i_decoding_diagnostics import CONDITIONS, evaluate, read_jsonl, write_json
from run_stage_i_decoding_diagnostic_local_v2 import checked_inputs, sha

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage_i/decoding_diagnostic_v1"
RUN = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2"
CLOUD = ROOT / "outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2"
OUT = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1"
CONTROL = ROOT / "outputs/stage_i/decoding_A_execution_control_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUT.exists():
        raise FileExistsError("Preserve existing evaluation; use a fresh version")
    manifest, cases = checked_inputs(PREP, CLOUD)
    bindings = read(RUN / "run_binding.json")
    approval = read(ROOT / "outputs/stage_i/decoding_approval_A_local_v2.json")
    assert bindings["experiment"] == "greedy_A"
    assert bindings["local_runtime_runner_sha256"] == sha(ROOT / "scripts/run_stage_i_decoding_diagnostic_local_v2.py") == approval["local_runtime_runner_sha256"]
    assert bindings["subset_manifest_sha256"] == sha(PREP / "subset_manifest.json")
    completions = sorted((RUN / "attempts").glob("*/completion_report.json"))
    assert len(completions) == 1 and read(completions[0])["status"] == "ALL72_DIAGNOSTIC_PREDICTIONS_COMPLETE"
    complete = read(completions[0])
    environment = read(completions[0].parent / "environment.json")
    assert environment["training"] is False and "3050" in environment["device"]
    actual_configs = {c: read(completions[0].parent / f"{c}_generation_config.json") for c in CONDITIONS}
    for c, record in actual_configs.items():
        assert not record["offline_reconstruction_differences"]
        gen = record["actual_effective"]
        assert gen["num_beams"] == 1 and gen["do_sample"] is False and gen["no_repeat_ngram_size"] == 0
        assert gen["max_length"] == 768 and gen["use_cache"] is True
    tokenizer = Tokenizer.from_file(str(ROOT / "assets/tokenizers/led-base-16384/tokenizer.json"))
    predictions, samples = [], []
    for c in CONDITIONS:
        path = RUN / c / "predictions.jsonl"
        assert sha(path) == complete["predictions_sha256"][c]
        pp = read_jsonl(path)
        assert len(pp) == 24 and [p["business_id"] for p in pp] == manifest["execution_business_ids"]
        for pred in pp:
            assert pred["condition"] == c and 0 < len(pred["token_ids"]) <= 768
            assert tokenizer.decode(pred["token_ids"], skip_special_tokens=True) == pred["text"]
            casepath = RUN / c / "cases" / (hashlib.sha256(pred["business_id"].encode()).hexdigest() + ".json")
            record = read(casepath)
            assert record["prediction"] == pred
            samples.append(record)
        predictions.extend(pp)
    assert len(predictions) == 72
    result = evaluate(cases, predictions, OUT, "greedy_A")
    viewer = OUT / "COMPARISON.html"
    presentation = viewer.read_text(encoding="utf-8")
    presentation = presentation.replace(
        "Baseline mode shows the saved prediction twice, without new generation.",
        "This page compares the original teacher reference, saved beam4 baseline and newly generated greedy_A output. Baseline ran on RTX3090; greedy ran on the researcher-approved local RTX3050.")
    viewer.write_text(presentation, encoding="utf-8")
    baseline = read(PREP / "baseline_evaluation/summary.json")
    rows = []
    for group in ("random", "targeted"):
        for c in CONDITIONS:
            b, g = baseline["subsets"][group][c], result["subsets"][group][c]
            row = {"subset": group, "condition": c, "outputs": g["outputs"]}
            for key in ("exact_repeated_sentence", "consecutive_repeated_word", "section_headers_compliant",
                        "eos_before_length_cap", "cap_reached", "visual_text_without_caption_input"):
                row[f"baseline_{key}_count"] = b[key]["count"]
                row[f"greedy_{key}_count"] = g[key]["count"]
                row[f"delta_{key}_count"] = g[key]["count"] - b[key]["count"]
            for key in ("mean", "median", "p95", "max"):
                row[f"baseline_length_{key}"] = b["generated_length"][key]
                row[f"greedy_length_{key}"] = g["generated_length"][key]
            for key in ("teacher_agreement_full_rouge1_fmeasure", "teacher_agreement_full_rouge2_fmeasure",
                        "teacher_agreement_full_rougeL_fmeasure", "teacher_agreement_review_rougeL_recall"):
                row[f"baseline_{key}"] = b[key]["macro_mean"]
                row[f"greedy_{key}"] = g[key]["macro_mean"]
            rows.append(row)
    with (OUT / "baseline_greedy_comparison.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    orchestration = read(CONTROL / "local_v2_orchestration.json")
    assert orchestration["status"] == "PROCESS_FINISHED" and orchestration["exit_code"] == 0
    with (CONTROL / "local_v2_gpu_metrics.csv").open(encoding="utf-8", newline="") as f:
        gpu = list(csv.DictReader(f))
    runtime = {"local_GPU": environment["device"], "baseline_GPU": "RTX3090",
               "hardware_differs_disclosed": True, "exact_cross_device_bitwise_control_not_claimed": True,
               "start_utc": orchestration["start_utc"], "end_utc": orchestration["end_utc"],
               "pipeline_seconds_including_download_import_load_monitor_tail": orchestration["orchestration_seconds"],
               "greedy_generation_seconds_sum": sum(p["seconds"] for p in predictions),
               "baseline_generation_seconds_sum": sum(c["baseline_prediction"]["seconds"] for c in cases),
               "greedy_generation_seconds_by_condition": {c: sum(p["seconds"] for p in predictions if p["condition"] == c) for c in CONDITIONS},
               "torch_peak_allocated_GiB": max(r["peak_allocated_GiB"] for r in samples),
               "torch_peak_reserved_GiB": max(r["peak_reserved_GiB"] for r in samples),
               "whole_GPU_sampled_peak_MiB": max(float(r["memory_used_MiB"]) for r in gpu),
               "whole_GPU_sampled_utilization_mean_percent": statistics.mean(float(r["utilization_gpu_percent"]) for r in gpu),
               "GPU_samples": len(gpu), "GPU_sampling_period_seconds_approximately": 5,
               "new_rental_API_cost_USD": 0, "paid_rental_upload_or_teacher_calls": False,
               "electricity_cost_USD": None, "electricity_note": "Local electricity tariff/incremental device energy not measured; no fabricated total monetary charge",
               "generated_outputs": 72, "B_run": False, "retraining": False, "held_out_used": False}
    write_json(OUT / "execution_integrity_and_runtime.json", {"status": "PASS", "approval": approval,
               "all72_prediction_ID_text_mappings": "PASS", "exact_same_fixed_cases": True,
               "existing_checkpoints_unchanged": True, "frozen_input_config_bindings": "PASS",
               "generation_parameters_changed": {"num_beams": {"baseline": 4, "greedy_A": 1}},
               "actual_generation_configs": actual_configs, "environment": environment, "runtime": runtime})
    write_json(OUT / "paired_summary.json", {"rows": rows, "runtime": runtime,
               "zero_caption_F_baseline": baseline["subsets"]["targeted"]["filtered"]["zero_caption_empty_visual_compliance"],
               "zero_caption_F_greedy": result["subsets"]["targeted"]["filtered"]["zero_caption_empty_visual_compliance"],
               "qualitative_audit_status": "See separately authored AI evidence-linked notes; not human ground truth"})
    checked_inputs(PREP, CLOUD)
    print(json.dumps({"status": "ALL72_INTEGRITY_AND_EVALUATION_PASS", "output": str(OUT),
                      "rows": rows, "runtime": runtime}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
