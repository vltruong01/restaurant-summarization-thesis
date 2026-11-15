"""Offline independent audit of preserved partial guard-pilot incident.

Reads files only; never loads models, uses GPU, retries generation or rewrites
an inference artifact. Writes a new independent audit namespace once.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from stage_i_repetition_pilot_v1 import ROOT, preflight
from stage_i_decoding_diagnostics import aggregate, structure, termination, text_diagnostics, words
from stage_i_d053_evaluation import repetition_diagnostics
from stage_i_repetition_guard_v1 import analyze
from audit_stage_i_repetition_guard_results_v1 import read, require, sha, write


def audit(output):
    config, manifest, cases, preflight_proof = preflight(require_fresh=False)
    run = ROOT / config["run_root"]
    failure_path = run / "FAILURE_STOP_NO_RETRY.json"
    failure = read(failure_path)
    require(failure["error"] == "No finite legal continuation; no fallback", "Different incident")
    require(not (run / "COMPLETION.json").exists() and not (run / "predictions.json").exists(),
            "Unexpected final completion or prediction file")
    require(not (run / "COMPARISON_RECORDS.json").exists(), "Partial run unexpectedly has complete comparisons")
    completed = cases[:5]
    failed = cases[5]
    ids = [c["business_id"] for c in completed]
    require(failed["business_id"] == "3v0tSwHXR9LhpsoGXLgsTA", "Unexpected failed case")
    require({p.name for p in (run / "raw_cases").glob("*.json")} == {i + ".json" for i in ids},
            "Completed raw-case file set differs")
    require({p.name for p in (run / "analysis").glob("*.json")} == {i + ".json" for i in ids},
            "Completed analysis file set differs")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / config["tokenizer_directory"] / "tokenizer.json"))
    generation = read(run / "EFFECTIVE_GENERATION_CONFIG.json")["actual"]
    saved_b_path = ROOT / config["saved_diagnostic_B"]
    saved_b = [json.loads(line) for line in saved_b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(saved_b) == 24 and [p["business_id"] for p in saved_b] == [c["business_id"] for c in cases],
            "Saved B baseline mapping differs")
    by_b = {p["business_id"]: p for p in saved_b}
    records, artifact_hashes = [], {}
    for c in completed:
        path = run / "raw_cases" / (c["business_id"] + ".json")
        p = read(path)
        require(p["business_id"] == c["business_id"] and p["condition"] == "filtered" and
                p["subset"] == c["subset"] and p["source_sha256"] == c["source_sha256"],
                "Partial source/condition/subset differs")
        require(p["raw_prediction_not_rewritten"] is True and p["human_verified"] is False,
                "Unexpected rewrite/human flag")
        check = analyze(p["text"], c["source"], finalize=True)
        analysis_path = run / "analysis" / path.name
        require(check == read(analysis_path) and not check["violations"], "Partial final guard analysis differs")
        artifact_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        artifact_hashes[analysis_path.relative_to(ROOT).as_posix()] = sha(analysis_path)
        for arm, pred in (("historical_F_saved", c["baseline_prediction"]),
                          ("Diagnostic_B_saved", by_b[c["business_id"]]), ("guard_v1_partial", p)):
            require(pred["condition"] == "filtered" and
                    tokenizer.decode(pred["token_ids"], skip_special_tokens=True) == pred["text"],
                    "Saved text/token or condition mismatch")
            d = {**text_diagnostics(pred["text"]), **structure(pred["text"], "filtered", len(c["selected_photo_ids"])),
                 **termination(pred["token_ids"])}
            lexical = words(pred["text"])
            records.append({"business_id": c["business_id"], "subset": c["subset"], "arm": arm,
                "comparison_scope": "SAME_FIRST_FIVE_COMPLETED_CASES_ONLY_NOT_RANDOM20_RESULT",
                "source_sha256": c["source_sha256"], "text": pred["text"], "token_ids": pred["token_ids"],
                "software": d, "accepted_extended_repetition": repetition_diagnostics(pred["text"], pred["token_ids"], generation),
                "type_token_ratio_length_dependent_not_informativeness": len(set(lexical)) / len(lexical),
                "surface_guard_analysis_not_quality_labels": analyze(pred["text"], c["source"], finalize=True),
                "human_verified": False, "semantic_grounding_Core_quality": "NOT_ESTABLISHED_BY_SOFTWARE"})
    arms = ("historical_F_saved", "Diagnostic_B_saved", "guard_v1_partial")
    groups, lexical_groups = {}, {}
    for arm in arms:
        rr = [r for r in records if r["arm"] == arm]
        groups[arm] = aggregate([r["software"] for r in rr])
        ng = {}
        for n in (2, 3, 4):
            values = [r["accepted_extended_repetition"]["lexical_ngrams"][str(n)] for r in rr]
            ng[str(n)] = {"outputs_with_lexical_repeated_ngram": sum(v["duplicate_occurrences"] > 0 for v in values),
                         "macro_duplicate_fraction": statistics.mean(v["duplicate_fraction"] for v in values if v["duplicate_fraction"] is not None),
                         "macro_distinct_n": statistics.mean(v["distinct_n"] for v in values if v["distinct_n"] is not None)}
        lexical_groups[arm] = {"n": len(rr), "lexical_ngrams": ng,
            "mean_type_token_ratio_length_dependent_not_informativeness": statistics.mean(r["type_token_ratio_length_dependent_not_informativeness"] for r in rr)}
    trace_path = run / "interventions" / (failed["business_id"] + ".jsonl")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    terminal = events[-1]
    require(terminal["event"] == "FAIL_CLOSED_NO_LEGAL_TOKEN" and len(terminal["prefix_ids"]) == 767,
            "Terminal event/prefix differs")
    require(len(terminal["rejected"]) == 1 and terminal["rejected"][0]["token_id"] == 2 and
            terminal["rejected"][0]["original_score"] == 0.0, "Terminal forced-EOS candidate differs")
    require({v["rule"] for v in terminal["rejected"][0]["violations"]} == {"duplicate_list_item"},
            "Terminal violation differs")
    require(all(e["source_sha256"] == failed["source_sha256"] for e in events if "source_sha256" in e),
            "Failed trace source mapping differs")
    prefix_text = tokenizer.decode(terminal["prefix_ids"], skip_special_tokens=True)
    eos_assessment = analyze(tokenizer.decode(terminal["prefix_ids"] + [2], skip_special_tokens=True), failed["source"], finalize=True)
    require(eos_assessment["violations"] == terminal["rejected"][0]["violations"],
            "Terminal EOS-finalization analysis did not reproduce")
    refs_path = ROOT / config["final_references"]
    refs = read(refs_path)
    require(refs["bindings"]["case_inputs_sha256"] == sha(ROOT / config["cases"]), "Reference/input binding differs")
    require(len(refs["units"]) == len({u["unit_id"] for u in refs["units"]}) == 576, "Reference IDs/count differ")
    require(dict(Counter(u["approved_reference_role"] for u in refs["units"])) == {"core": 261, "supporting": 315},
            "Approved reference roles differ")
    initial = read(run / "INITIAL_MODEL_BINDINGS.json")
    require(initial["adapter_exact"] and initial["all_parameters_frozen"] and len(initial["backbone_hashes"]) == 295,
            "Initial model binding differs")
    # The runner failed before its final tensor-equality postconditions. On-disk
    # checksums establish saved-file preservation, NOT post-error RAM equality.
    preflight(require_fresh=False)
    report = {"status": "PARTIAL_INCIDENT_INDEPENDENT_SOFTWARE_AUDIT_PASS_NOT_EXPERIMENT_COMPLETION",
        "planned": 24, "attempted": 6, "completed": 5, "failed": 1, "not_attempted": 18, "incomplete_total": 19,
        "completed_business_ids": ids, "failed_business_id": failed["business_id"],
        "not_attempted_business_ids": [c["business_id"] for c in cases[6:]],
        "comparison_scope": "ONLY_SAME_FIVE_COMPLETED_CASES_ALL_FROM_RANDOM_STRATUM; NOT_20_RANDOM_OR_24_FULL_RESULTS",
        "comparison_records": 15, "partial_raw_source_tokens_and_guard_analysis": "PASS",
        "preflight": preflight_proof, "all_bound_scientific_files_unchanged_on_disk": True,
        "initial_model_bindings_recorded": True,
        "post_exception_in_memory_backbone_adapter_equality": "UNAVAILABLE_FINAL_CHECK_NOT_REACHED",
        "completion_and_final_prediction_files_absent": True,
        "forced_EOS_collision": {"prefix_tokens_including_decoder_start": 767,
            "max_length": generation["max_length"], "forced_eos_token_id": generation["forced_eos_token_id"],
            "trace_rejected_candidates": terminal["rejected"],
            "only_finite_EOS_interpretation": "Pinned ForcedEOSTokenLogitsProcessor at max_length-1 permits only EOS; trace records its score0 and rejection. Full original score vector was not separately stored.",
            "prefix_text_tail": prefix_text[-1600:], "prefix_is_diagnostic_evidence_not_final_summary": True,
            "trace_events": len(events), "trace_sha256": sha(trace_path), "EOS_finalization_rule_reproduced_offline": True},
        "reference_sha256": sha(refs_path), "eligible_references": 576, "excluded_references": len(refs["approved_exclusion_log"]),
        "no_factual_or_human_quality_labels_assigned": True, "quality_GO": "NOT_ESTABLISHED",
        "zero_caption_case_attempted": False, "D051_strict_replay": "FAIL_UNCHANGED",
        "failure_sha256": sha(failure_path), "saved_B_file_sha256": sha(saved_b_path),
        "audit_script_sha256": sha(Path(__file__)),
        "metric_caveats": ["Partial first-five success subset cannot estimate full-pilot success or random20 quality.",
                           "Accepted adjacent-word metric ignores punctuation and crosses sentence boundaries.",
                           "Lexical diversity and repetition are not factual grounding or substantive quality.",
                           "An EOS/cap collision is not repaired by this audit; no retry/fallback was executed."]}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "INDEPENDENT_SOFTWARE_AUDIT.json", report)
    write(out / "PARTIAL_COMPARISON_RECORDS.json", records)
    write(out / "SOFTWARE_METRICS_PARTIAL.json", groups)
    write(out / "LEXICAL_NGRAM_SUMMARY_PARTIAL.json", lexical_groups)
    write(out / "FAILED_TERMINAL_EVENT_UNCHANGED.json", terminal)
    write(out / "SOURCE_ARTIFACT_SHA256.json", artifact_hashes)
    write(out / "SHA256_MANIFEST.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({"status": report["status"], "completed": 5, "attempted": 6,
                      "file_bindings": preflight_proof["file_bindings"],
                      "historical_bindings": preflight_proof["preserved_historical_file_bindings"],
                      "partial_metrics": {a: {k: groups[a][k] for k in ("outputs", "exact_repeated_sentence", "consecutive_repeated_word", "section_headers_compliant", "cap_reached")} for a in arms}}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
