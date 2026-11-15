"""User-approved local RTX3050 Diagnostic B runtime; immutable inputs/checkpoints; no training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid
import warnings

from stage_i_decoding_diagnostics import CONDITIONS, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_inputs(preparation, cloud):
    manifest = read(preparation / "subset_manifest.json")
    for name, expected in manifest["preparation_bindings"].items():
        require(sha(preparation / name) == expected, f"Preparation changed: {name}")
    for name, expected in manifest["code_bindings"].items():
        require(sha(ROOT / name) == expected, f"Reviewed code changed: {name}")
    for name, expected in manifest["tokenizer_bindings"].items():
        require(sha(ROOT / name) == expected, f"Tokenizer changed: {name}")
    for c, info in manifest["checkpoints"].items():
        folder = cloud / info["directory"]
        for name, expected in info["files"].items():
            require(sha(folder / name) == expected, f"Checkpoint changed: {c}/{name}")
    for name, expected in manifest["source_file_bindings"].items():
        require(sha(ROOT / name) == expected, f"Upstream dataset/source changed: {name}")
    cases = read_jsonl(preparation / "cases.jsonl")
    require(len(cases) == manifest["unique_businesses"] * 3 == 72, "Expected complete24x3 matrix")
    keys = {(c["condition"], c["business_id"]) for c in cases}
    require(len(keys) == len(cases) and all(c["split"] == "validation" for c in cases), "Duplicate or non-validation case")
    for c in CONDITIONS:
        require([r["business_id"] for r in cases if r["condition"] == c] == manifest["execution_business_ids"], "Case order/condition differs")
    return manifest, cases


def approval_gate(preparation, approval_path, experiment, execute):
    # Keep this check before importing Torch or loading/downloading a model.
    require(execute, "New GPU inference requires --execute and a separately approved diagnostic run")
    require(experiment in ("greedy_A", "beam_ngram_B"), "Baseline must reuse saved predictions")
    require(approval_path is not None and approval_path.is_file(), "Researcher approval JSON missing")
    approval = read(approval_path)
    require(approval.get("approved") is True, "Researcher has not approved this GPU diagnostic")
    require(approval.get("subset_manifest_sha256") == sha(preparation / "subset_manifest.json"), "Approval is bound to another subset")
    require(approval.get("decoding_configurations_sha256") == sha(preparation / "decoding_configurations.json"), "Approval is bound to another decoding configuration")
    require(experiment in approval.get("approved_experiments", []), "Experiment is not approved")
    if experiment == "beam_ngram_B":
        require(approval.get("greedy_A_results_reviewed") is True, "B is a separate proposal after A review")
    require(experiment == "beam_ngram_B", "This local runner permits B only")
    require(approval.get("length_clarification") == "max_length_768_including_decoder_start", "Generation length not clarified")
    require(approval.get("allow_backbone_download") is False, "Use the already cached pinned backbone only")
    require(approval.get("runtime_profile") == "local_rtx3050_6gb", "Local GPU profile is not approved")
    require(approval.get("local_runtime_runner_sha256") == sha(Path(__file__)), "Local runner differs from approved runtime")
    return approval


def semantic_generation(config):
    return {k: v for k, v in config.items() if k not in ("transformers_version", "_from_model_config", "_commit_hash")}


def infer(preparation, cloud, output, experiment, approval_path, resume):
    approval = approval_gate(preparation, approval_path, experiment, True)
    manifest, cases = checked_inputs(preparation, cloud)
    settings = read(preparation / "decoding_configurations.json")
    chosen = settings["experiments"][experiment]
    binding = {"subset_manifest_sha256": sha(preparation / "subset_manifest.json"),
               "decoding_configurations_sha256": sha(preparation / "decoding_configurations.json"),
               "experiment": experiment, "checkpoint_hashes": manifest["checkpoints"],
               "code_bindings": manifest["code_bindings"],
               "local_runtime_runner_sha256": sha(Path(__file__)), "runtime_profile": "local_rtx3050_6gb"}
    require(not output.resolve().is_relative_to(cloud.resolve()), "Do not write into the immutable cloud import")
    require(not output.resolve().is_relative_to(preparation.resolve()), "Use a separate experiment directory, not the preparation")
    if resume:
        require(output.is_dir() and read(output / "run_binding.json") == binding, "Incompatible resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_binding.json", binding)
    attempt = output / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    write_json(attempt / "approval_snapshot.json", approval)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import LEDForConditionalGeneration, LEDTokenizerFast, set_seed
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_hi_led_data import LEDCompleteCollator
    expected = settings["required_runtime_versions"]
    for pkg, version in expected.items():
        require(importlib.metadata.version(pkg).split("+")[0] == version, f"Runtime drift: {pkg}")
    require(torch.cuda.is_available(), "CUDA required; no CPU fallback")
    require(torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 5.5, "Approved local profile requires the6GB GPU class")
    require("3050" in torch.cuda.get_device_name(0), "Unexpected local GPU; stop for review")
    write_json(attempt / "environment.json", {"device": torch.cuda.get_device_name(0),
               "cuda": torch.version.cuda, "torch": torch.__version__,
               "versions": {p: importlib.metadata.version(p) for p in expected},
               "precision": "FP32 weights; FP16 autocast", "training": False,
               "GPU_vram_GiB": torch.cuda.get_device_properties(0).total_memory / 1024**3,
               "hardware_differs_from_baseline_RTX3090": True, "local_runtime_runner_sha256": sha(Path(__file__))})
    tokenizer = LEDTokenizerFast.from_pretrained(ROOT / settings["tokenizer_directory"], local_files_only=True)
    collator = LEDCompleteCollator(5000, 768, 1024)
    model = None
    try:
        for condition in CONDITIONS:
            condition_cases = [c for c in cases if c["condition"] == condition]
            folder = output / condition
            case_folder = folder / "cases"
            case_folder.mkdir(parents=True, exist_ok=True)
            pending = [c for c in condition_cases if not (case_folder / (hashlib.sha256(c["business_id"].encode()).hexdigest() + ".json")).exists()]
            if pending:
                set_seed(42)
                torch.backends.cuda.matmul.allow_tf32 = False
                base = LEDForConditionalGeneration.from_pretrained(
                    settings["model_id"], revision=settings["revision"], torch_dtype=torch.float32,
                    local_files_only=not approval.get("allow_backbone_download", False))
                for key, expected_value in settings["backbone_checks"].items():
                    require(getattr(base.config, key) == expected_value, f"Backbone mismatch: {key}")
                adapter = cloud / manifest["checkpoints"][condition]["directory"]
                model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
                loaded = get_peft_model_state_dict(model)
                saved = load_file(adapter / "adapter_model.safetensors", device="cpu")
                require(loaded.keys() == saved.keys(), "Adapter tensor names differ")
                require(all(torch.equal(loaded[k].detach().cpu(), saved[k]) for k in saved), "Loaded adapter differs from accepted checkpoint")
                require(not any(p.requires_grad for p in model.parameters()), "Diagnostic must be inference-only")
                require(all(p.dtype == torch.float32 for p in model.parameters()), "Weight precision differs from baseline")
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.config.use_cache = False
                model.eval()
                effective = deepcopy(model.generation_config)
                for key, value in chosen["generation_kwargs"].items():
                    setattr(effective, key, value)
                effective.use_cache = True
                actual = effective.to_dict()
                expected_generation = chosen["expected_effective_generation_config_reconstruction"]
                differences = {k: {"expected": semantic_generation(expected_generation).get(k),
                                   "actual": semantic_generation(actual).get(k)}
                               for k in semantic_generation(actual).keys() | semantic_generation(expected_generation).keys()
                               if semantic_generation(actual).get(k) != semantic_generation(expected_generation).get(k)}
                write_json(attempt / f"{condition}_generation_config.json", {"actual_effective": actual,
                           "offline_reconstruction_differences": differences,
                           "original_baseline_full_generation_config_was_not_archived": True})
                require(not differences, "Unexpected pinned generation defaults; stop for review instead of changing settings")
                model.to("cuda")
                torch.cuda.reset_peak_memory_stats()
            all_predictions = []
            for case in condition_cases:
                file = case_folder / (hashlib.sha256(case["business_id"].encode()).hexdigest() + ".json")
                case_binding = {"condition": condition, "business_id": case["business_id"],
                                "source_sha256": case["source_sha256"], "experiment": experiment}
                if file.exists():
                    result = read(file)
                    require(result["binding"] == case_binding, "Case resume binding differs")
                    pred = result["prediction"]
                else:
                    feature = {"input_ids": case["input_ids"], "labels": case["label_ids"]}
                    batch = {k: v.cuda() for k, v in collator([feature]).items() if k != "labels"}
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            generated = model.generate(**batch, **chosen["generation_kwargs"], use_cache=True)
                    torch.cuda.synchronize()
                    ids = generated[0].cpu().tolist()
                    require(0 < len(ids) <= 768, "Unexpected generation length")
                    pred = {"business_id": case["business_id"], "condition": condition,
                            "token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                            "seconds": time.perf_counter() - started, "tokens_including_decoder_start": len(ids),
                            "cap_reached": len(ids) == 768, "no_prediction_rewrite": True}
                    write_json(file, {"binding": case_binding, "prediction": pred,
                               "warnings": [str(w.message) for w in caught],
                               "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                               "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3})
                require(pred["business_id"] == case["business_id"] and pred["condition"] == condition, "Prediction mapping differs")
                require(tokenizer.decode(pred["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) == pred["text"], "Prediction token/text mismatch")
                all_predictions.append(pred)
                print(json.dumps({"condition": condition, "business_id": case["business_id"],
                                  "complete_in_condition": len(all_predictions), "total": len(condition_cases)}), flush=True)
            final = folder / "predictions.jsonl"
            if final.exists():
                require(read_jsonl(final) == all_predictions, "Existing final predictions differ")
            else:
                with final.open("x", encoding="utf-8") as f:
                    for pred in all_predictions:
                        f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            if model is not None:
                state_after = get_peft_model_state_dict(model)
                require(all(torch.equal(state_after[k].detach().cpu(), saved[k]) for k in saved), "Adapter changed during inference")
                del state_after, loaded, saved, model, base
                model = None
                torch.cuda.empty_cache()
        checked_inputs(preparation, cloud)
        write_json(attempt / "completion_report.json", {"status": "ALL72_DIAGNOSTIC_PREDICTIONS_COMPLETE",
                   "experiment": experiment, "existing_checkpoints_unchanged": True,
                   "teacher_calls": 0, "training": False, "held_out_used": False,
                   "predictions_sha256": {c: sha(output / c / "predictions.jsonl") for c in CONDITIONS}})
    except Exception as exc:
        write_json(attempt / "failure_report.json", {"error_type": type(exc).__name__, "error": str(exc),
                   "no_automatic_fallback": True, "existing_successful_cases_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, default=ROOT / "outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2")
    parser.add_argument("--experiment", choices=("beam_ngram_B",))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        manifest, _ = checked_inputs(args.preparation, args.cloud_root)
        print(json.dumps({"status": "OFFLINE_BINDINGS_PASS_NO_GPU_MODEL_LOAD", "cases": manifest["unique_businesses"] * 3}))
        return
    require(args.output is not None, "Choose a fresh versioned experiment output")
    approval_gate(args.preparation, args.approval, args.experiment, args.execute)
    infer(args.preparation, args.cloud_root, args.output, args.experiment, args.approval, args.resume)


if __name__ == "__main__":
    main()
