"""Build the validation-only diagnostic packet and evaluate saved baseline outputs offline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from stage_i_decoding_diagnostics import (CONDITIONS, aggregate, distribution, evaluate,
    read_jsonl, sections, structure, termination, text_diagnostics, write_json)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_i_decoding_diagnostic_v1.json")
    args = parser.parse_args()
    cfg = read(args.config)
    output = ROOT / cfg["preparation_output"]
    if output.exists():
        raise FileExistsError("Preserve existing preparation; use a new version for a new preparation")
    cloud = ROOT / cfg["cloud_root"]
    accepted = read(cloud / cfg["frozen_training_config"])
    manifest = read(cloud / accepted["stage_h_manifest"])
    assert sha(cloud / accepted["stage_h_manifest"]) == accepted["stage_h_manifest_sha256"]
    tm = read(ROOT / accepted["tokenizer_manifest"])
    mm = read(ROOT / accepted["model_config_manifest"])
    mc = read(ROOT / mm["config_file"])
    assert sha(ROOT / mm["config_file"]) == mm["config_sha256"]
    assert mm["revision"] == tm["revision"] == accepted["revision"]
    tokenizer_bindings = {f"{tm['relative_directory']}/{f}": h for f, h in tm["sha256"].items()}
    tokenizer_bindings[mm["config_file"]] = mm["config_sha256"]
    assert all(sha(ROOT / p) == h for p, h in tokenizer_bindings.items())
    tokenizer = Tokenizer.from_file(str(ROOT / tm["relative_directory"] / "tokenizer.json"))
    refs, preds, packs, source_bindings, checkpoints = {}, {}, {}, {}, {}
    for condition in CONDITIONS:
        refpath = ROOT / cfg["teacher_validation_directory"] / f"validation_{condition}.jsonl"
        folder = cloud / cfg["runs_relative_to_cloud"] / condition
        predpath = folder / "validation_predictions.jsonl"
        refs[condition], preds[condition] = read_jsonl(refpath), read_jsonl(predpath)
        source_bindings[refpath.relative_to(ROOT).as_posix()] = sha(refpath)
        source_bindings[predpath.relative_to(ROOT).as_posix()] = sha(predpath)
        assert len(refs[condition]) == len(preds[condition]) == 200
        ids = [r["business_id"] for r in refs[condition]]
        assert len(set(ids)) == 200 and [p["business_id"] for p in preds[condition]] == ids
        assert all(r["condition"] == condition and r["split"] == "validation" for r in refs[condition])
        info = manifest["datasets"][f"validation_{condition}"]
        npzpath = cloud / info["path"]
        assert sha(npzpath) == info["sha256"]
        source_bindings[npzpath.relative_to(ROOT).as_posix()] = info["sha256"]
        with np.load(npzpath, allow_pickle=False) as pack:
            packs[condition] = {k: pack[k].copy() for k in pack.files}
        assert packs[condition]["business_ids"].tolist() == ids
        complete, state, binding, runtime = (read(folder / f) for f in (
            "completion_report.json", "trainer_state.json", "run_binding.json", "runtime_initial.json"))
        assert state["global_step"] == 1250 and state["best_model_checkpoint"].endswith("checkpoint-1250")
        assert complete["best_eval_loss"] == state["best_metric"] == min(x["eval_loss"] for x in state["log_history"] if "eval_loss" in x)
        assert runtime["config"]["generation"] == accepted["generation"]
        assert runtime["config"]["revision"] == accepted["revision"]
        assert binding["runner_sha256"] == sha(cloud / "scripts/run_stage_i_lora_led_cloud_v2.py")
        adapter = folder / "best_adapter"
        ac = read(adapter / "adapter_config.json")
        assert ac["r"] == 8 and ac["lora_alpha"] == 16 and set(ac["target_modules"]) == {"q_proj", "v_proj"}
        with safe_open(adapter / "adapter_model.safetensors", framework="numpy") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        assert len(tensors) == 48 and sum(v.size for v in tensors.values()) == 294912
        assert all("led.decoder." in k and np.isfinite(v).all() for k, v in tensors.items())
        assert sha(adapter / "adapter_model.safetensors") == sha(folder / "checkpoint-1250/adapter_model.safetensors")
        checkpoints[condition] = {"directory": adapter.relative_to(cloud).as_posix(),
            "files": {f: sha(adapter / f) for f in ("adapter_config.json", "adapter_model.safetensors")},
            "best_optimizer_step": 1250, "base_revision_in_adapter_metadata": ac["revision"],
            "base_revision_enforced_by_runner": accepted["revision"]}
    population = sorted(r["business_id"] for r in refs["reviews_only"])
    assert all(sorted(r["business_id"] for r in refs[c]) == population for c in CONDITIONS)
    random_ids = random.Random(cfg["sampling"]["seed"]).sample(population, cfg["sampling"]["random_count"])
    targeted = cfg["sampling"]["targeted"]
    targeted_ids = [x["business_id"] for x in targeted]
    assert len(set(targeted_ids)) == 4 and set(targeted_ids) <= set(population)
    overlap = sorted(set(random_ids) & set(targeted_ids))
    execution_ids = random_ids + [x for x in targeted_ids if x not in random_ids]
    assert len(execution_ids) == 24 and not overlap, "v1 expected draw changed; review overlap rather than silently changing sampling"
    indexed = {c: {r["business_id"]: (i, r, preds[c][i]) for i, r in enumerate(refs[c])} for c in CONDITIONS}
    cases = []
    for condition in CONDITIONS:
        pack = packs[condition]
        for bid in execution_ids:
            i, ref, pred = indexed[condition][bid]
            start, end = pack["input_offsets"][i:i + 2]
            ls, le = pack["label_offsets"][i:i + 2]
            inputs, labels = pack["input_ids"][start:end].tolist(), pack["labels"][ls:le].tolist()
            assert tokenizer.encode(ref["source"]).ids == inputs
            assert tokenizer.encode(ref["target"]).ids == labels
            assert tokenizer.decode(pred["token_ids"], skip_special_tokens=True) == pred["text"]
            assert len(inputs) <= 5000 and len(labels) <= 768
            assert len(ref["selected_photo_ids"]) == int(pack["caption_counts"][i])
            row = dict(ref)
            row.update(subset="random" if bid in random_ids else "targeted", input_ids=inputs,
                       label_ids=labels, baseline_prediction=pred)
            cases.append(row)
    for bid in execution_ids:
        rows = [c for c in cases if c["business_id"] == bid]
        assert len({tuple(c["selected_review_ids"]) for c in rows}) == 1
        assert len({c["source"].split("[VISUAL_CONTEXT]")[0].rstrip() for c in rows}) == 1
        assert len({sections(c["target"])[0] for c in rows}) == 1

    # CPU-only shape/label checks. No model weights, forward, backward or generation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import GenerationConfig, LEDConfig
    from stage_hi_led_data import LEDCompleteCollator
    collator = LEDCompleteCollator(5000, 768, 1024)
    collator_checks = []
    for c in cases:
        batch = collator([{"input_ids": c["input_ids"], "labels": c["label_ids"]}])
        ni, nl = len(c["input_ids"]), len(c["label_ids"])
        assert batch["input_ids"][0, :ni].tolist() == c["input_ids"]
        assert batch["labels"][0, :nl].tolist() == c["label_ids"]
        assert batch["global_attention_mask"][0, 0] == 1 and batch["global_attention_mask"].sum() == 1
        assert torch.all(batch["attention_mask"][0, ni:] == 0) and torch.all(batch["labels"][0, nl:] == -100)
        shifted = batch["labels"].clone()
        shifted[:, 1:] = batch["labels"][:, :-1]
        shifted[:, 0] = 2
        shifted.masked_fill_(shifted == -100, 1)
        assert shifted[0, 0] == 2 and shifted[0, 1:nl].tolist() == c["label_ids"][:-1]
        collator_checks.append({"business_id": c["business_id"], "condition": c["condition"],
                                "source_tensor_length": batch["input_ids"].shape[1],
                                "label_tensor_length": batch["labels"].shape[1]})
    baseline = accepted["generation"]
    assert baseline == {"max_length": 768, "num_beams": 4, "do_sample": False,
                        "length_penalty": 1.0, "no_repeat_ngram_size": 0,
                        "early_stopping": True, "forced_eos_token_id": 2}
    defaults = GenerationConfig.from_model_config(LEDConfig(**mc)).to_dict()
    experiments = {}
    for key, spec in cfg["experiments"].items():
        kwargs = {**baseline, **spec["generation_overrides"]}
        effective = {**defaults, **kwargs, "use_cache": True}
        differences = {k: {"baseline": baseline[k], "experiment": kwargs[k]} for k in baseline if baseline[k] != kwargs[k]}
        experiments[key] = {"mode": spec["mode"], "generation_kwargs": kwargs,
                            "differences_from_baseline": differences,
                            "expected_effective_generation_config_reconstruction": effective}
    assert set(experiments["greedy_A"]["differences_from_baseline"]) == {"num_beams"}
    assert set(experiments["beam_ngram_B"]["differences_from_baseline"]) == {"no_repeat_ngram_size"}
    versions = {p: importlib.metadata.version(p).split("+")[0] for p in (
        "torch", "transformers", "peft", "accelerate", "numpy", "tokenizers", "safetensors")}
    assert versions == {"torch": "2.1.0", "transformers": "4.51.3", "peft": "0.15.2", "accelerate": "1.6.0",
                        "numpy": "1.26.4", "tokenizers": "0.21.1", "safetensors": "0.8.0"}
    generation = {"model_id": accepted["model_id"], "revision": accepted["revision"],
                  "required_runtime_versions": versions, "tokenizer_directory": tm["relative_directory"],
                  "backbone_checks": {k: mc[k] for k in (
                      "attention_window", "max_decoder_position_embeddings", "max_encoder_position_embeddings",
                      "d_model", "encoder_layers", "decoder_layers", "bos_token_id", "pad_token_id", "eos_token_id", "decoder_start_token_id")},
                  "original_full_generation_config_archived": False,
                  "reconstruction_scope": "Pinned model config plus installed defaults plus actual saved explicit generation kwargs. Not an observed complete baseline runtime snapshot.",
                  "experiments": experiments}
    full_rows, teacher_context = [], {}
    for condition in CONDITIONS:
        teacher_metrics = [text_diagnostics(r["target"]) for r in refs[condition]]
        teacher_context[condition] = {"outputs": 200, "repeated_sentence_count": sum(x["exact_repeated_sentence"] for x in teacher_metrics),
                                      "consecutive_repeated_word_count": sum(x["consecutive_repeated_word"] for x in teacher_metrics)}
        for ref, pred in zip(refs[condition], preds[condition]):
            full_rows.append({"condition": condition, **text_diagnostics(pred["text"]), **termination(pred["token_ids"]),
                              **structure(pred["text"], condition, len(ref["selected_photo_ids"]))})
    context = {"saved_baseline_only": True, "conditions": {c: aggregate([r for r in full_rows if r["condition"] == c]) for c in CONDITIONS},
               "teacher_text_diagnostics": teacher_context}
    assert [context["conditions"][c]["exact_repeated_sentence"]["count"] for c in CONDITIONS] == [31, 195, 166]
    assert all(t["repeated_sentence_count"] == 0 for t in teacher_context.values())
    prior_times = [c["baseline_prediction"]["seconds"] for c in cases]
    cost = {"no_rental_or_upload": True, "currency": "USD", "pricing_checked": "2026-10-15",
            "pricing_sources": ["https://vast.ai/pricing", "https://github.com/vast-ai/docs/blob/main/guides/pricing.mdx"],
            "pricing_basis": "Host-set marketplace rates; no offer or reservation selected. Rate scenarios below are planning assumptions, not a fixed quote.",
            "measured_prior_beam4_seconds_for_same72_cases": sum(prior_times),
            "measured_prior_case_seconds": distribution(prior_times),
            "measured_prior_seconds_by_condition": {c: sum(r["baseline_prediction"]["seconds"] for r in cases if r["condition"] == c) for c in CONDITIONS},
            "A_new_generations": 72, "baseline_new_generations": 0, "B_new_generations_authorized": 0,
            "A_time_minutes_planning": {"warm_existing_environment": [15, 45], "cold_setup_downloads_included": [30, 75]},
            "time_uncertainty": "Greedy is unmeasured; loops can use all767new-token steps. Prior beam4 times are a reference, not a greedy speed guarantee.",
            "compute_only_cost_scenarios": [{"assumed_USD_per_hour": rate,
                "warm_USD_range": [round(rate * .25, 4), round(rate * .75, 4)],
                "cold_USD_range": [round(rate * .5, 4), round(rate * 1.25, 4)]} for rate in (.20, .30, .50)],
            "additional_cost_formula": "Host storage_GB_hour_rate * allocated_GB * retained_hours + ingress_GB_rate * ingress_GB + egress_GB_rate * egress_GB; rates must be reviewed before rental.",
            "resource_plan": "One RTX3090-class24GB GPU, >=16GB host RAM (32GB preferred), >=4vCPU, >=20GB free disk, existing exact pinned environment preferred. Three conditions serial, batch1; no training.",
            "approval_required": True}
    output.mkdir(parents=True)
    with (output / "cases.jsonl").open("x", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    write_json(output / "decoding_configurations.json", generation)
    write_json(output / "cost_estimate.json", cost)
    code = {p: sha(ROOT / p) for p in (
        "scripts/prepare_stage_i_decoding_diagnostic.py", "scripts/run_stage_i_decoding_diagnostic.py",
        "scripts/stage_i_decoding_diagnostics.py", "scripts/stage_hi_led_data.py")}
    subset = {"status": "OFFLINE_PREPARED_NO_NEW_INFERENCE", "seed": cfg["sampling"]["seed"],
              "sampling_method": cfg["sampling"]["method"], "population": "All200 existing validation businesses only",
              "population_size": 200, "population_sorted_ids_sha256": hashlib.sha256(json.dumps(population).encode()).hexdigest(),
              "random_business_ids": random_ids, "targeted_businesses": targeted, "overlap_ids": overlap,
              "execution_business_ids": execution_ids, "unique_businesses": len(execution_ids),
              "same_ids_across_conditions": True, "source_file_bindings": source_bindings,
              "config_sha256": sha(args.config), "checkpoints": checkpoints,
              "code_bindings": code, "tokenizer_bindings": tokenizer_bindings,
              "preparation_bindings": {f: sha(output / f) for f in ("cases.jsonl", "decoding_configurations.json")}}
    write_json(output / "subset_manifest.json", subset)
    write_json(output / "approval_EXAMPLE_NOT_APPROVED.json", {"approved": False, "cost_reviewed": False,
               "approved_experiments": [], "greedy_A_results_reviewed": False, "allow_backbone_download": False,
               "subset_manifest_sha256": sha(output / "subset_manifest.json"),
               "decoding_configurations_sha256": sha(output / "decoding_configurations.json"),
               "note": "Separate actual researcher approval is required. This template authorizes nothing."})
    baseline_summary = evaluate(cases, [c["baseline_prediction"] for c in cases], output / "baseline_evaluation", "baseline")
    write_json(output / "baseline_all200_context.json", context)
    audit = {"status": "OFFLINE_DATA_CONFIG_COLLATOR_CHECKS_PASS_NEW_INFERENCE_NOT_RUN",
             "source_label_ID_roundtrips": len(cases), "baseline_prediction_ID_text_roundtrips": len(cases),
             "cpu_collator_checks": collator_checks, "global_attention": "Exactly first real BOS, no padding global attention",
             "label_shift": "Decoder start2 followed by complete original BOS0/text; -100 padding masked",
             "special_tokens": {"BOS": 0, "PAD": 1, "EOS": 2, "decoder_start": 2},
             "precision": "Training and saved-generation code use FP32 base+LoRA weights and FP16 autocast; proposed inference preserves this",
             "known_provenance_gaps": ["Adapter metadata revision=null; proposed loader explicitly pins original base revision",
                                        "Original full model.generation_config was not archived; derived defaults are labeled reconstruction and must be checked against loaded pinned model before GPU generation",
                                        "Original section validator allowed empty visuals but did not require empty for zero-caption F; new evaluation tests this separately without repairing outputs"],
             "no_proven_data_or_special_token_mismatch_found": True,
             "runtime_generation_match_unverified_without_approved_model_load": True,
             "hypotheses_not_established": ["Beam decoding amplifies repetitive high-probability continuations", "Generic target imitation or insufficient task learning", "Decoder-only LoRA limits adaptation", "A hidden generation/runtime mismatch not covered by recorded metadata"],
             "no_model_weights_loaded_into_model": True, "GPU_inference": False, "training": False,
             "teacher_calls": 0, "held_out_data_read": False,
             "experiments": {"baseline": "SAVED_OUTPUTS_EVALUATED", "greedy_A": "PREPARED_NOT_EXECUTED", "beam_ngram_B": "SEPARATE_PROPOSAL_NOT_EXECUTED"}}
    write_json(output / "offline_checks.json", audit)
    assert all(sha(ROOT / path) == expected for path, expected in source_bindings.items())
    print(json.dumps({"status": audit["status"], "directory": str(output), "random": 20, "targeted": 4,
                      "overlap": overlap, "baseline_ROUGE": baseline_summary["rouge"],
                      "prior_beam4_seconds_for72": sum(prior_times)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
