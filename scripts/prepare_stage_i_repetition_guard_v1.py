"""Offline-only packaging of the versioned surface-guard proposal. No Torch/model.

Run once after code/tests have settled. Does not authorize or execute the pilot.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from stage_i_repetition_guard_v1 import analyze
from stage_i_repetition_pilot_v1 import ROOT, CONFIG, PREP, RUNNER, read, rows, sha, write, require


def main():
    require(not CONFIG.exists() and not (PREP / "PREFLIGHT_MANIFEST.json").exists(), "Preparation already exists; never overwrite a released version")
    require(sha(ROOT / "configs/stage_i_d060_ro_to_filtered_v1.json") == "4513dc202e72f8d2589408e573ecf21dba0e8a409950f4c69f95b1a689a22c01", "Original D060 config changed")
    require(sha(ROOT / "outputs/stage_i/d060_preparation_v1/PREFLIGHT_MANIFEST.json") == "c27796dfb001d38190a56f3888a97c6fabf9dbd39ee689bed4b73400f0a32080", "Original D060 manifest changed")
    require(sha(ROOT / "scripts/run_stage_i_d060_v1.py") == "c00f74a61b92fcafd106e2b5cb656865eead232a05a63a70aeb739bbe1fd83c1", "Original D060 runner changed")
    from stage_i_d060 import preflight as historical_preflight
    historical_preflight(allow_existing_run=True)
    old = read(ROOT / "configs/stage_i_d060_ro_to_filtered_v1.json")
    config = {"experiment": "repetition_guard_pilot_v1", "status": "OFFLINE_PREPARED_AWAITING_NEW_HASH_BOUND_APPROVAL",
              "GPU_execution_approved": False, "training_authorized": False,
              "scope": {"condition": "filtered", "businesses": 24, "random": 20, "targeted": 4,
                        "generation_calls_max": 24, "training_updates": 0},
              "model_id": old["model_id"], "revision": old["revision"], "seed": 42,
              "adapter_directory": "outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2/outputs/stage_i/runs_v3/filtered/best_adapter",
              "cases": "outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl",
              "saved_diagnostic_B": "outputs/stage_i/decoding_beam_ngram_B_local_v1/filtered/predictions.jsonl",
              "expected_generation_config": "outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json",
              "backbone_fingerprints": "outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json",
              "final_references": old["final_references"], "runtime_pins": old["runtime_pins"],
              "historical_integrity_manifest": "outputs/stage_i/d060_preparation_v1/PREFLIGHT_MANIFEST.json",
              "tokenizer_directory": "assets/tokenizers/led-base-16384", "generation": old["generation"],
              "budgets": old["budgets"], "CPU_threads": 8, "minimum_memory_headroom": 0.15,
              "precision": "FP32 weights; FP16 autocast; TF32 disabled; no quantization",
              "run_root": "outputs/stage_i/repetition_guard_pilot_v1", "hardware": "local_RTX3050_6GB_batch1_beam4",
              "policy": {"version": "completed_surface_units_v1", "hard_rules": ["duplicate_sentence", "adjacent_word", "duplicate_list_item"],
                         "sentence_scope": "casefolded lexical words within canonical section; completed units only; conservative abbreviation/decimal exceptions",
                         "list_scope": "finished comma/semicolon/and/or items; ignore leading articles; first-clause suffix match with negation safeguard",
                         "literal_word_scope": "closed same word separated by horizontal whitespace within a sentence",
                         "source_exception": "exact LED-visible source reduplication bounded by source maximum run; not verified names",
                         "grammar_ambiguity_exceptions": ["had had", "that that"],
                         "markers": "canonical section histories independent; repeated markers do not reset; unknown syntax flagged; no repair/forcing",
                         "candidate_filter": "eight legal per-row frontier plus ties; lower tail untouched; no renormalization; pinned top8 selection safety assertion",
                         "semantic_and_suffix_evasion": "flag only; no semantic guarantee; reject usability conclusion without researcher review",
                         "no_postprocessing": True},
              "viability_deadline_utc": "2026-10-22T07:11:22.846767+00:00",
              "original_D059_deadline_utc": "2026-10-25T15:10:48.941817+00:00",
              "D051_strict_replay": "FAIL_UNCHANGED", "Teacher_calls_authorized": False,
              "held_out_access": False, "automatic_retry": False, "fallback_authorized": False,
              "RO_U_Shuffled_authorized": False, "sweeps_authorized": False,
              "quality_acceptance": "researcher-confirmed usable improvement in Core/opinion/grounding/visual usefulness/completion, not repetition alone"}
    PREP.mkdir(parents=True, exist_ok=True)
    write(CONFIG, config)
    baseline = rows(ROOT / config["cases"])
    diagnostic_b = {r["business_id"]: r for r in rows(ROOT / config["saved_diagnostic_B"])}
    audit = []
    for c in baseline:
        for arm, prediction in (("historical_F_saved", c["baseline_prediction"]), ("Diagnostic_B_saved", diagnostic_b[c["business_id"]])):
            audit.append({"business_id": c["business_id"], "subset": c["subset"], "arm": arm,
                          "source_sha256": c["source_sha256"], "analysis": analyze(prediction["text"], c["source"], True),
                          "not_factual_error_labels": True, "existing_prediction_unchanged": True})
    write(PREP / "SAVED_OUTPUT_POLICY_AUDIT.json", audit)
    files = [CONFIG, RUNNER, ROOT / "scripts/stage_i_repetition_guard_v1.py", ROOT / "scripts/stage_i_repetition_pilot_v1.py",
             Path(__file__), ROOT / "tests/test_stage_i_repetition_guard_v1.py", ROOT / "tests/test_stage_i_repetition_pilot_v1.py",
             ROOT / config["cases"], ROOT / config["saved_diagnostic_B"] , ROOT / config["expected_generation_config"],
             ROOT / config["backbone_fingerprints"], ROOT / config["final_references"], ROOT / config["historical_integrity_manifest"],
             ROOT / "outputs/stage_i/decoding_diagnostic_v1/subset_manifest.json",
             ROOT / "outputs/stage_i/decoding_diagnostic_v1/cases.jsonl",
             ROOT / "outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json",
             ROOT / "outputs/stage_i/d060_execution_v1/RESEARCHER_EXECUTION_AUTHORIZATION.json",
             ROOT / "scripts/stage_hi_led_data.py", ROOT / "scripts/stage_i_paired_pilot.py",
             ROOT / "scripts/stage_i_d056_v2.py", ROOT / "scripts/stage_i_decoding_diagnostics.py",
             ROOT / "thesis_sdd_docs/FOCUSED_REPETITION_RESEARCH_20261019.md", PREP / "SAVED_OUTPUT_POLICY_AUDIT.json",
             PREP / "PROTOCOL_AND_AUTHORIZATION.md"]
    files += list((ROOT / config["adapter_directory"]).glob("adapter*"))
    files += [p for p in (ROOT / config["tokenizer_directory"]).iterdir() if p.is_file()]
    spec = importlib.util.find_spec("transformers")
    package = Path(next(iter(spec.submodule_search_locations)))
    files += [package / "generation/utils.py", package / "generation/logits_process.py", package / "generation/stopping_criteria.py"]
    def rel(path):
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    write(PREP / "PREFLIGHT_MANIFEST.json", {"experiment": "repetition_guard_pilot_v1", "GPU_execution_approved": False,
          "no_model_load_during_preparation": True, "files": {rel(p): sha(p) for p in files},
          "D051_strict_replay": "FAIL_UNCHANGED"})
    hashes = {"config_sha256": sha(CONFIG), "preflight_manifest_sha256": sha(PREP / "PREFLIGHT_MANIFEST.json"),
              "runner_sha256": sha(RUNNER), "guard_sha256": sha(ROOT / "scripts/stage_i_repetition_guard_v1.py")}
    write(PREP / "AUTHORIZATION_HASHES.json", hashes)
    write(PREP / "approval_template_NOT_APPROVED.json", {"experiment": "repetition_guard_pilot_v1", **hashes,
          "GPU_execution_approved": False, "approval_source": "NOT_AUTHORIZED_TEMPLATE",
          "researcher_explicit_authorization": "", "execution_authorized_at_utc": None,
          "generation_calls_max": 24, "condition": "filtered", "training_authorized": False,
          "Teacher_calls_authorized": False, "held_out_access": False, "automatic_retry": False,
          "fallback_authorized": False, "RO_U_Shuffled_authorized": False,
          "viability_deadline_utc": config["viability_deadline_utc"]})
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
