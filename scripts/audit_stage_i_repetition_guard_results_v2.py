"""Offline independent v2 results audit; run only after root reports termination.

No polling, model, GPU, network or inference-artifact editing. A fresh audit
namespace records either completed accounting or an explicit unexpected stop.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from stage_i_repetition_pilot_v2 import ROOT, OLD_RUN, preflight
from audit_stage_i_repetition_guard_results_v1 import read, require, sha, write
from stage_i_decoding_diagnostics import aggregate, structure, termination, text_diagnostics, words
from stage_i_d053_evaluation import repetition_diagnostics
from stage_i_repetition_guard_v1 import analyze

ARMS = ("historical_F_saved", "Diagnostic_B_saved", "guard_v2_combined")


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def adjacent_decomposition(text):
    """Descriptive original-regex pairs; never redefine accepted metrics."""
    cleaned = re.sub(r"\[[A-Z_]+\]", lambda m: " " * len(m.group()), text).lower()
    spans = list(re.finditer(r"[a-z0-9]+(?:'[a-z]+)?", cleaned))
    pairs = []
    for a, b in zip(spans, spans[1:]):
        if a.group() != b.group():
            continue
        gap = cleaned[a.end():b.start()]
        category = ("sentence_or_section_boundary" if re.search(r"[.!?\n\r]", gap) else
                    "literal_horizontal_whitespace" if gap.isspace() else "within_sentence_punctuation")
        pairs.append({"word": a.group(), "category": category,
                      "excerpt": text[max(0, a.start()-20):min(len(text), b.end()+30)]})
    return pairs


def emit(output, report, **artifacts):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "INDEPENDENT_SOFTWARE_AUDIT.json", report)
    for name, value in artifacts.items():
        write(out / (name + ".json"), value)
    write(out / "SHA256_MANIFEST.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(report, ensure_ascii=True))


def audit(output):
    c, manifest, cases, reused, proof = preflight(require_fresh=False)
    run = ROOT / c["run_root"]
    completion_path, failure_path = run / "COMPLETION.json", run / "FAILURE_STOP_NO_RETRY.json"
    if not completion_path.exists():
        require(failure_path.exists(), "Run not terminated; this auditor does not poll")
        failure = read(failure_path)
        report = {"status": "UNEXPECTED_STOP_RECORDED_NOT_COMPLETED_EXPERIMENT", "preflight": proof,
            "new_calls_attempted_recorded": failure.get("new_calls_attempted"),
            "raw_case_files_present_not_claimed_complete": len(list((run / "raw_cases").glob("*.json"))),
            "failure_sha256": sha(failure_path), "original_v1_files_and_bound_inputs_unchanged_on_disk": True,
            "final_in_memory_integrity_checks": "NOT_CONFIRMED_ON_UNEXPECTED_STOP",
            "human_quality_GO": "NOT_ESTABLISHED", "D051_strict_replay": "FAIL_UNCHANGED",
            "audit_script_sha256": sha(Path(__file__))}
        emit(output, report, UNEXPECTED_FAILURE_UNCHANGED=failure)
        return
    require(not failure_path.exists(), "Unexpected-stop evidence coexists with completion")
    completion = read(completion_path)
    require(completion["status"] in ("ALL24_OUTPUTS_AVAILABLE_STOP_FOR_REVIEW",
            "ALL24_ACCOUNTED_WITH_MISSING_CONSTRAINT_OUTPUTS_STOP_FOR_REVIEW"), "Unexpected completion status")
    require(completion["new_generation_calls"] == 19 and completion["reused_outputs"] == 5 and
            completion["unique_businesses_accounted"] == 24, "Execution/reuse accounting differs")
    require(completion["checkpoint_unchanged"] and not completion["training"] and
            completion["Teacher_calls"] == 0 and not completion["held_out_used"], "Forbidden scope recorded")
    ids = [x["business_id"] for x in cases]
    case_map = {x["business_id"]: x for x in cases}
    outcomes = read(run / "CASE_OUTCOMES.json")
    require(len(outcomes) == 24 and [x["business_id"] for x in outcomes] == ids and
            len({x["business_id"] for x in outcomes}) == 24, "Outcome24 mapping differs")
    require(all(x["status"] == "REUSED_UNCHANGED_V1" and x["new_generation_calls"] == 0 for x in outcomes[:5]),
            "Reused outcomes differ")
    require(all(x["status"] in ("NEW_V2_OUTPUT", "FAILED_NO_COMPLIANT_OUTPUT") and
                x["new_generation_calls"] == 1 for x in outcomes[5:]), "Remaining19 accounting differs")
    failed_ids = {x["business_id"] for x in outcomes if x["status"] == "FAILED_NO_COMPLIANT_OUTPUT"}
    available_ids = [i for i in ids if i not in failed_ids]
    predictions = read(run / "predictions.json")
    require([x["business_id"] for x in predictions] == available_ids and
            len(predictions) == len({x["business_id"] for x in predictions}), "Available predictions differ")
    require(completion["available_outputs"] == len(predictions) and
            completion["constraint_failed_outputs"] == len(failed_ids), "Available/missing counters differ")
    require({p.stem for p in (run / "raw_cases").glob("*.json")} == set(available_ids) and
            {p.stem for p in (run / "analysis").glob("*.json")} == set(available_ids), "Raw/analysis file sets differ")
    starts = [read(run / "case_starts" / (i + ".json")) for i in ids[5:]]
    require(len(list((run / "case_starts").glob("*.json"))) == 19 and
            [x["business_id"] for x in starts] == ids[5:] and
            [x["new_call"] for x in starts] == list(range(1, 20)), "New generation attempts differ")
    require(all(x["exactly_one_attempt"] and x["source_sha256"] == case_map[x["business_id"]]["source_sha256"] for x in starts),
            "Attempt/source bindings differ")
    require({p.stem for p in (run / "known_constraint_failures").glob("*.json")} == failed_ids,
            "Known missing-case record set differs")
    for i in failed_ids:
        f = read(run / "known_constraint_failures" / (i + ".json"))
        require(f["status"] == "FAILED_NO_COMPLIANT_OUTPUT" and not f["raw_output_fabricated"] and
                f["source_sha256"] == case_map[i]["source_sha256"], "Known case failure differs")
    provenance = read(run / "REUSE_PROVENANCE.json")
    require([x["business_id"] for x in provenance] == ids[:5], "Reuse provenance IDs differ")
    for p, original in zip(provenance, reused):
        i = p["business_id"]
        for kind, key in (("raw_cases", "raw_sha256"), ("analysis", "analysis_sha256")):
            require(sha(run / kind / (i + ".json")) == sha(OLD_RUN / kind / (i + ".json")) == p[key],
                    "Copied v1 evidence bytes differ")
        require(read(run / "raw_cases" / (i + ".json")) == original, "Reused prediction was modified")
    saved_b_path = ROOT / c["saved_diagnostic_B"]
    saved_b = [json.loads(s) for s in saved_b_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    require([x["business_id"] for x in saved_b] == ids, "Saved B IDs/order differ")
    by_b = {x["business_id"]: x for x in saved_b}
    comparisons = read(run / "COMPARISON_RECORDS.json")
    indexed = {(x["business_id"], x["arm"]): x for x in comparisons}
    require(len(comparisons) == len(indexed) == 3 * len(predictions) and
            set(indexed) == {(i, arm) for i in available_ids for arm in ARMS}, "Completed-case comparisons differ")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / c["tokenizer_directory"] / "tokenizer.json"))
    generation = read(run / "EFFECTIVE_GENERATION_CONFIG.json")["actual"]
    expected_evidence = [{k: x[k] for k in ("business_id", "subset", "source", "target", "source_sha256",
                         "selected_review_ids", "selected_photo_ids", "shared_review_evidence",
                         "visual_claims_with_evidence", "evidence_records")} for x in cases]
    require(read(run / "REVIEW_EVIDENCE.json") == expected_evidence, "Scientific evidence changed")
    records = []
    for pred in predictions:
        i, case = pred["business_id"], case_map[pred["business_id"]]
        require(pred == read(run / "raw_cases" / (i + ".json")) and pred["condition"] == "filtered" and
                pred["subset"] == case["subset"] and pred["source_sha256"] == case["source_sha256"], "Raw/final mapping differs")
        analysis = analyze(pred["text"], case["source"], True)
        require(not analysis["violations"] and analysis == read(run / "analysis" / (i + ".json")), "Final surface analysis differs")
        require(pred["human_verified"] is False and pred["raw_prediction_not_rewritten"], "Unexpected human/rewrite flags")
        if i not in ids[:5]:
            returned = read(run / "returned_candidates" / (i + ".json"))
            require(returned["token_ids"] == pred["token_ids"] and returned["text"] == pred["text"], "Accepted output differs from raw return")
            p = pred["completion_proof"]
            eos = termination(pred["token_ids"])
            require(p["actual_finished_beam_verified"] and p["is_sent_finished"] and
                    math.isfinite(p["actual_normalized_sequence_score"]) and
                    p["generated_EOS_position"] == eos["first_eos_position"] and
                    eos["eos_after_decoder_start"] and not eos["nonpadding_tokens_after_first_eos"], "Finished-beam/EOS proof differs")
        for arm, x in ((ARMS[0], case["baseline_prediction"]), (ARMS[1], by_b[i]), (ARMS[2], pred)):
            require(tok.decode(x["token_ids"], skip_special_tokens=True) == x["text"], "Stored token IDs/text differ")
            d = {**text_diagnostics(x["text"]), **structure(x["text"], "filtered", len(case["selected_photo_ids"])),
                 **termination(x["token_ids"])}
            row = indexed[(i, arm)]
            require(row["text"] == x["text"] and row["token_ids"] == x["token_ids"] and row["software"] == d and
                    row["source_sha256"] == case["source_sha256"] and row["subset"] == case["subset"], "Comparison/raw/accepted metric differs")
            require(row["human_verified"] is False and row["researcher_usable_quality"] is None, "Unexpected researcher labels")
            hard = analyze(x["text"], case["source"], True)
            lexical = words(x["text"])
            records.append({"business_id": i, "arm": arm, "subset": case["subset"], "software": d,
                "extended_repetition": repetition_diagnostics(x["text"], x["token_ids"], generation),
                "original_adjacent_metric_decomposition": adjacent_decomposition(x["text"]),
                "unchanged_hard_rule_counts": dict(Counter(v["rule"] for v in hard["violations"])),
                "source_or_grammatical_reduplication_exceptions": hard["source_reduplication_exceptions"],
                "type_token_ratio_length_dependent": len(set(lexical)) / len(lexical) if lexical else None})
    groups, ngroups = {}, {}
    published = read(run / "SOFTWARE_METRICS.json")
    for arm in ARMS:
        for subset in ("random", "targeted"):
            key = arm + "/" + subset
            rr = [x for x in records if x["arm"] == arm and x["subset"] == subset]
            groups[key] = aggregate([x["software"] for x in rr]) if rr else {"outputs": 0, "status": "NO_NEW_OUTPUTS_IN_THIS_SUBSET"}
            require(groups[key] == published[key], "Published aggregate differs")
            ng = {}
            for n in (2, 3, 4):
                vals = [x["extended_repetition"]["lexical_ngrams"][str(n)] for x in rr]
                ng[str(n)] = {"outputs_with_repeat": sum(x["duplicate_occurrences"] > 0 for x in vals),
                    "macro_duplicate_fraction": mean_or_none(x["duplicate_fraction"] for x in vals),
                    "macro_distinct_n": mean_or_none(x["distinct_n"] for x in vals)}
            ngroups[key] = {"n": len(rr), "lexical_ngrams": ng,
                "outputs_with_hard_adjacent_word_violation": sum(x["unchanged_hard_rule_counts"].get("adjacent_word", 0) > 0 for x in rr),
                "outputs_with_hard_list_violation": sum(x["unchanged_hard_rule_counts"].get("duplicate_list_item", 0) > 0 for x in rr),
                "outputs_with_only_original_boundary_adjacent_pairs": sum(bool(x["original_adjacent_metric_decomposition"]) and
                    all(p["category"] == "sentence_or_section_boundary" for p in x["original_adjacent_metric_decomposition"]) for x in rr),
                "mean_type_token_ratio_not_informativeness": mean_or_none(x["type_token_ratio_length_dependent"] for x in rr)}
    refs_path = ROOT / c["final_references"]
    refs = read(refs_path)
    require(refs["bindings"]["case_inputs_sha256"] == sha(ROOT / c["cases"]), "Reference/input hash binding differs")
    require(len({u["unit_id"] for u in refs["units"]}) == len(refs["units"]) == 576, "Frozen reference unit IDs differ")
    preflight(require_fresh=False)
    report = {"status": "INDEPENDENT_V2_ACCOUNTING_AND_SOFTWARE_AUDIT_PASS_NOT_QUALITY_GO", "preflight": proof,
        "planned_cases": 24, "reused_outputs": 5, "new_generation_attempts": 19,
        "available_outputs": len(predictions), "missing_constraint_outputs": len(failed_ids),
        "failed_business_ids": sorted(failed_ids), "comparisons": len(comparisons),
        "paired_denominator_policy": "Only the same available businesses in all three arms; failed cases stay explicitly separate.",
        "all_reused_v1_bytes_unchanged": True, "all_bound_scientific_files_unchanged_on_disk": True,
        "runtime_final_tensor_integrity_assertions_reached": True,
        "runtime_tensor_integrity_independently_reexecuted": False,
        "raw_final_comparison_source_token_alignment": True, "published_metrics_recomputed_identically": True,
        "reference_sha256": sha(refs_path), "Core_Supporting": dict(Counter(u["approved_reference_role"] for u in refs["units"])),
        "excluded_references": len(refs["approved_exclusion_log"]), "no_new_human_labels": True,
        "source_grounding_Core_semantic_quality": "NOT_ESTABLISHED_BY_SOFTWARE", "quality_GO": "PENDING_RESEARCHER",
        "D051_strict_replay": "FAIL_UNCHANGED", "audit_script_sha256": sha(Path(__file__)),
        "predictions_sha256": sha(run / "predictions.json"), "completion_sha256": sha(completion_path),
        "caveats": ["Legacy adjacent-word metrics ignore punctuation and sentence boundaries.",
                    "Lexical distinct-n is length-dependent and does not establish informative content.",
                    "Surface hard-rule compliance does not guarantee semantic non-redundancy or grounding."]}
    emit(output, report, ACCEPTED_METRICS_RECOMPUTED=groups, LEXICAL_AND_ADJACENT_SUMMARY=ngroups,
         PER_CASE_REPETITION=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
