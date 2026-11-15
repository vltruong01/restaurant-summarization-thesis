"""Independent offline audit of a COMPLETE F24 repetition-guard pilot.

No polling, models, GPU, network, target editing or human quality labels.
Run only after the orchestrating researcher task confirms completion.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from stage_i_repetition_pilot_v1 import ROOT, preflight
from stage_i_decoding_diagnostics import aggregate, structure, termination, text_diagnostics, words
from stage_i_d053_evaluation import repetition_diagnostics
from stage_i_repetition_guard_v1 import analyze


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def audit(output):
    # This command has no waiting/resume/retry behavior. No run files are read
    # until explicitly invoked by the parent after its completion notice.
    config, manifest, cases, preflight_proof = preflight(require_fresh=False)
    run = ROOT / config["run_root"]
    completion = read(run / "COMPLETION.json")
    require(completion["status"] == "F24_COMPLETE_STOP_FOR_RESEARCHER_QUALITY_REVIEW"
            and completion["generation_calls"] == 24, "Pilot is incomplete")
    require(not (run / "FAILURE_STOP_NO_RETRY.json").exists(), "Execution has failure evidence")
    predictions = read(run / "predictions.json")
    comparisons = read(run / "COMPARISON_RECORDS.json")
    ids = [c["business_id"] for c in cases]
    require(len(predictions) == 24 and [p["business_id"] for p in predictions] == ids,
            "New prediction count or order differs")
    require(len({p["business_id"] for p in predictions}) == 24, "Duplicate predictions")
    require({p.name for p in (run / "raw_cases").glob("*.json")} == {i + ".json" for i in ids},
            "Raw-case file set differs")
    require({p.name for p in (run / "analysis").glob("*.json")} == {i + ".json" for i in ids},
            "Analysis file set differs")
    saved_b_path = ROOT / config["saved_diagnostic_B"]
    saved_b = [json.loads(line) for line in saved_b_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(saved_b) == 24 and [p["business_id"] for p in saved_b] == ids,
            "Saved Diagnostic B count or order differs")
    require(all(p["condition"] == "filtered" for p in saved_b), "Saved B condition differs")
    by_b = {p["business_id"]: p for p in saved_b}
    by_new = {p["business_id"]: p for p in predictions}
    by_comparison = {(r["business_id"], r["arm"]): r for r in comparisons}
    arms = ("historical_F_saved", "Diagnostic_B_saved", "guard_v1")
    require(len(comparisons) == len(by_comparison) == 72 and
            set(by_comparison) == {(i, arm) for i in ids for arm in arms}, "Comparison72 mapping differs")
    expected_evidence = [{k: c[k] for k in ("business_id", "subset", "source", "target", "source_sha256",
                         "selected_review_ids", "selected_photo_ids", "shared_review_evidence",
                         "visual_claims_with_evidence", "evidence_records")} for c in cases]
    require(read(run / "REVIEW_EVIDENCE.json") == expected_evidence, "Review evidence was altered")
    from tokenizers import Tokenizer  # Pinned CPU tokenizer only; no model.
    tokenizer = Tokenizer.from_file(str(ROOT / config["tokenizer_directory"] / "tokenizer.json"))
    generation = read(run / "EFFECTIVE_GENERATION_CONFIG.json")["actual"]
    basic_rows, extended_rows = [], []
    for c in cases:
        new = by_new[c["business_id"]]
        require(new == read(run / "raw_cases" / (c["business_id"] + ".json")), "Raw/final new prediction differs")
        require(new["condition"] == "filtered" and new["subset"] == c["subset"]
                and new["source_sha256"] == c["source_sha256"], "New source/condition/subset mapping differs")
        require(new["raw_prediction_not_rewritten"] is True, "Prediction rewrite flag differs")
        final_analysis = analyze(new["text"], c["source"], finalize=True)
        require(not final_analysis["violations"] and
                read(run / "analysis" / (c["business_id"] + ".json")) == final_analysis,
                "Final guard audit differs or contains a hard violation")
        for arm, p in ((arms[0], c["baseline_prediction"]), (arms[1], by_b[c["business_id"]]), (arms[2], new)):
            require(tokenizer.decode(p["token_ids"], skip_special_tokens=True) == p["text"],
                    "Stored token IDs/text differ")
            d = {**text_diagnostics(p["text"]), **structure(p["text"], "filtered", len(c["selected_photo_ids"])),
                 **termination(p["token_ids"])}
            r = by_comparison[(c["business_id"], arm)]
            require(r["text"] == p["text"] and r["token_ids"] == p["token_ids"] and r["software"] == d
                    and r["subset"] == c["subset"] and r["source_sha256"] == c["source_sha256"],
                    "Comparison data or accepted software diagnostics differ")
            require(r["human_verified"] is False and r["researcher_usable_quality"] is None,
                    "Unexpected human quality labels")
            basic_rows.append({"business_id": c["business_id"], "arm": arm, "subset": c["subset"], **d})
            lexical = words(p["text"])
            extended_rows.append({"business_id": c["business_id"], "arm": arm, "subset": c["subset"],
                "accepted_repetition": repetition_diagnostics(p["text"], p["token_ids"], generation),
                "lexical_type_token_ratio": len(set(lexical)) / len(lexical) if lexical else None,
                "surface_guard_analysis_not_quality_labels": analyze(p["text"], c["source"], finalize=True)})
    existing_groups = read(run / "SOFTWARE_METRICS.json")
    groups, lexical_groups = {}, {}
    for arm in arms:
        for subset in ("random", "targeted"):
            key = arm + "/" + subset
            group = [r for r in basic_rows if r["arm"] == arm and r["subset"] == subset]
            groups[key] = aggregate(group)
            require(groups[key] == existing_groups[key], "Published software aggregate differs")
            group = [r for r in extended_rows if r["arm"] == arm and r["subset"] == subset]
            ng = {}
            for n in (2, 3, 4):
                name = str(n)
                values = [r["accepted_repetition"]["lexical_ngrams"][name] for r in group]
                ng[name] = {"outputs_with_lexical_repeated_ngram": sum(v["duplicate_occurrences"] > 0 for v in values),
                            "macro_duplicate_fraction": statistics.mean(v["duplicate_fraction"] for v in values if v["duplicate_fraction"] is not None),
                            "macro_distinct_n": statistics.mean(v["distinct_n"] for v in values if v["distinct_n"] is not None)}
            lexical_groups[key] = {"n": len(group), "lexical_ngrams": ng,
                "mean_type_token_ratio_length_dependent_not_informativeness": statistics.mean(r["lexical_type_token_ratio"] for r in group if r["lexical_type_token_ratio"] is not None)}
    refs_path = ROOT / config["final_references"]
    refs = read(refs_path)
    require(refs["bindings"]["case_inputs_sha256"] == sha(ROOT / config["cases"]), "Reference/input binding differs")
    require(len({u["unit_id"] for u in refs["units"]}) == len(refs["units"]), "Reference unit IDs duplicated")
    subset_by_id = {c["business_id"]: c["subset"] for c in cases}
    require(set(u["business_id"] for u in refs["units"]) == set(ids), "Reference business scope differs")
    reference_groups = {}
    for subset in ("random", "targeted"):
        units = [u for u in refs["units"] if subset_by_id[u["business_id"]] == subset]
        require(all(u["scoring_eligible"] and u["approved_reference_role"] in ("core", "supporting") for u in units),
                "Ineligible reference entered the eligible unit list")
        reference_groups[subset] = dict(Counter(u["approved_reference_role"] for u in units))
    # Recheck all bound historical/input files after reading results.
    preflight(require_fresh=False)
    report = {"status": "INDEPENDENT_SOFTWARE_INTEGRITY_PASS_NOT_A_QUALITY_GO", "new_predictions": 24,
              "saved_historical_predictions": 24, "saved_B_predictions": 24, "comparisons": 72,
              "raw_final_comparison_identity": True, "accepted_metrics_recomputed_identically": True,
              "all_bound_inputs_and_historical_files_unchanged": True, "preflight": preflight_proof,
              "reference_sha256": sha(refs_path), "reference_eligible_roles_by_subset": reference_groups,
              "excluded_reference_units_outside_denominator": len(refs["approved_exclusion_log"]),
              "semantic_quality_and_grounding": "NOT_ESTABLISHED_BY_THIS_AUDIT",
              "no_new_human_labels_assigned": True, "D051_strict_replay": "FAIL_UNCHANGED",
              "audit_script_sha256": sha(Path(__file__)), "prediction_file_sha256": sha(run / "predictions.json"),
              "saved_B_file_sha256": sha(saved_b_path),
              "metric_caveats": ["Accepted adjacent-word metric ignores punctuation and flattens sentence boundaries.",
                                  "Lexical n-gram repeats can be grammatical and distinct-n is length-dependent.",
                                  "A surface guard PASS does not establish Core coverage, grounding or semantic non-redundancy."]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "INDEPENDENT_SOFTWARE_AUDIT.json", report)
    write(output / "ACCEPTED_METRICS_RECOMPUTED.json", groups)
    write(output / "LEXICAL_NGRAM_SUMMARY.json", lexical_groups)
    write(output / "PER_CASE_REPETITION.json", extended_rows)
    write(output / "SHA256_MANIFEST.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
