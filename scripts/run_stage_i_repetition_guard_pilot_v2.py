"""Authorized technical v2 resumption: reuse5, attempt remaining19 once.

Same hard rules/science; only terminal beam exhaustion and durable reporting
are repaired. Expected no-compliant-output cases are never fabricated/retried.
"""
import argparse
import copy
import html
import json
import os
import time
import traceback
import warnings
import shutil

from stage_i_repetition_pilot_v2 import ROOT, CONFIG, PREP, RUNNER, OLD_RUN, authorization, deadline, preflight, read, rows, sha, require, write


def semantic(value):
    return {k: v for k, v in value.items() if k not in ("transformers_version", "_from_model_config", "_commit_hash")}


def compare_outputs(config, cases, predictions, out):
    """Software measures and evidence for review, never human/factual labels."""
    from stage_i_decoding_diagnostics import text_diagnostics, structure, termination, aggregate
    from stage_i_repetition_guard_v1 import analyze
    saved_b = {r["business_id"]: r for r in rows(ROOT / config["saved_diagnostic_B"])}
    current = {r["business_id"]: r for r in predictions}
    records = []
    complete_cases = [c for c in cases if c['business_id'] in current]
    for c in complete_cases:
        for arm, prediction in (("historical_F_saved", c["baseline_prediction"]),
                                ("Diagnostic_B_saved", saved_b[c["business_id"]]),
                                ("guard_v2_combined", current[c["business_id"]])):
            d = {**text_diagnostics(prediction["text"]),
                 **structure(prediction["text"], "filtered", len(c["selected_photo_ids"])),
                 **termination(prediction["token_ids"])}
            records.append({"business_id": c["business_id"], "subset": c["subset"], "arm": arm,
                            "text": prediction["text"], "token_ids": prediction["token_ids"],
                            "software": d, "guard_analysis_not_factual_labels": analyze(prediction["text"], c["source"], True),
                            "source_sha256": c["source_sha256"], "human_verified": False,
                            "researcher_usable_quality": None})
    groups = {}
    for arm in ("historical_F_saved", "Diagnostic_B_saved", "guard_v2_combined"):
        for subset in ("random", "targeted"):
            group = [r["software"] for r in records if r["arm"] == arm and r["subset"] == subset]
            groups[arm + "/" + subset] = aggregate(group) if group else {'outputs':0,'status':'NO_NEW_OUTPUTS_IN_THIS_SUBSET'}
    write(out / "COMPARISON_RECORDS.json", records)
    write(out / "SOFTWARE_METRICS.json", groups)
    # Evidence is reused verbatim; source/teacher text is not an independent
    # factual verdict. The approved reference file stays separately bound.
    write(out / "REVIEW_EVIDENCE.json", [{k: c[k] for k in ("business_id", "subset", "source", "target", "source_sha256",
          "selected_review_ids", "selected_photo_ids", "shared_review_evidence", "visual_claims_with_evidence", "evidence_records")} for c in cases])
    blocks = []
    for c in complete_cases:
        rr = [r for r in records if r["business_id"] == c["business_id"]]
        blocks.append("<article><h2>" + html.escape(c["business_id"] + " / " + c["subset"]) + "</h2>"
                      + "<details><summary>Frozen source and Teacher comparison</summary><pre>" + html.escape(c["source"] + "\n\nTEACHER (not ground truth)\n" + c["target"]) + "</pre></details>"
                      + "<div class='grid'>" + "".join("<section><h3>" + html.escape(r["arm"]) + "</h3><pre>" + html.escape(r["text"]) + "</pre><details><summary>Software diagnostics</summary><pre>" + html.escape(json.dumps(r["software"], indent=2)) + "</pre></details></section>" for r in rr) + "</div></article>")
    write(out / "REVIEW.html", "<!doctype html><meta charset='utf-8'><title>Repetition guard validation review</title>"
          "<style>body{font:16px system-ui;margin:24px;background:#f4f5f7;color:#17202a}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article,section{background:white;padding:16px;margin-bottom:18px}h1{font-size:25px}</style>"
          "<h1>Completed-case-only comparison:5 reused v1 plus new v2 outputs</h1><p>No researcher labels assigned. Missing/infeasible businesses are excluded from all three arms' paired denominator. See CASE_OUTCOMES.json. Review Core coverage, contrasting opinions, source grounding, filler and zero-caption behavior. Teacher agreement and mechanical repetition are not factual accuracy.</p>" + "".join(blocks))


def execute(approval_path):
    approval = authorization(approval_path)  # MUST precede all model/GPU imports.
    config, manifest, cases, reused, proof = preflight()
    out = ROOT / config["run_root"]
    out.mkdir(parents=True, exist_ok=False)
    write(out / "AUTHORIZATION_AND_CONFIG.json", {"approval": approval, "config": config, "preflight": proof})
    started = time.perf_counter()
    model = None
    outcomes = []
    calls_attempted = 0
    def memory_snapshot():
        # Reporting only; never change allocation/precision or retry a failure.
        if model is None:
            return {'status':'MODEL_NOT_LOADED'}
        try:
            return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        except Exception as logging_error:
            # Preserve the original execution exception if CUDA reporting fails.
            return {'status':'MEMORY_REPORT_UNAVAILABLE','logging_error':str(logging_error)}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import LEDForConditionalGeneration, LEDTokenizerFast, LogitsProcessorList, StoppingCriteriaList, set_seed
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        from unittest.mock import patch
        from stage_hi_led_data import LEDCompleteCollator
        from stage_i_paired_pilot import verify_mask_batch
        from stage_i_d056_v2 import tensor_fingerprint
        from stage_i_repetition_guard_v2 import RepetitionGuardLogitsProcessor, SelectedCandidateSafetyCheck, FinishedBeamObserver, NoCompliantCompletion

        require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
        require("3050" in torch.cuda.get_device_name(0) and torch.cuda.get_device_properties(0).total_memory >= 5.5 * 1024**3,
                "Unexpected hardware; stop for review")
        torch.set_num_threads(config["CPU_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        set_seed(42)
        tokenizer = LEDTokenizerFast.from_pretrained(ROOT / config["tokenizer_directory"], local_files_only=True)
        base = LEDForConditionalGeneration.from_pretrained(config["model_id"], revision=config["revision"],
                                                           torch_dtype=torch.float32, local_files_only=True)
        settings = read(ROOT / "outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json")
        for key, value in settings["backbone_checks"].items():
            require(getattr(base.config, key) == value, "Backbone config mismatch: " + key)
        with patch("peft.tuners.lora.model.is_bnb_available", return_value=False), patch("peft.tuners.lora.model.is_bnb_4bit_available", return_value=False):
            model = PeftModel.from_pretrained(base, ROOT / config["adapter_directory"], is_trainable=False, torch_device="cpu")
        expected_weights = load_file(ROOT / config["adapter_directory"] / "adapter_model.safetensors", device="cpu")
        loaded = get_peft_model_state_dict(model)
        require(set(loaded) == set(expected_weights) and all(torch.equal(v.detach().cpu(), expected_weights[k]) for k, v in loaded.items()), "Adapter load differs")
        require(len(loaded) == 48 and sum(v.numel() for v in loaded.values()) == 294912, "Historical F adapter scope differs")
        require(not any(p.requires_grad for p in model.parameters()), "Inference model must be completely frozen")
        require(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()), "CPU FP32 load differs")
        historical = read(ROOT / config["backbone_fingerprints"])["architectures"][0]["canonical_initial_backbone_hashes"]
        def backbone_hashes():
            return {n.removeprefix("base_model.model.").replace(".base_layer.", "."): tensor_fingerprint(p)
                    for n, p in model.named_parameters() if ".lora_A." not in n and ".lora_B." not in n}
        before = backbone_hashes()
        require(before == historical and len(before) == 295, "Canonical backbone fingerprint mismatch")
        g = copy.deepcopy(model.generation_config)
        for key, value in config["generation"].items():
            setattr(g, key, value)
        g.use_cache = True
        require(semantic(g.to_dict()) == semantic(read(ROOT / config["expected_generation_config"])), "Effective GenerationConfig differs")
        write(out / "EFFECTIVE_GENERATION_CONFIG.json", {"actual": g.to_dict(), "custom_processor": "RepetitionGuardLogitsProcessor_v2_terminal_repair",
                "selected_top8_safety_hook": True, "terminal_beam_repair":'v2_only_prefix767_forcedEOS2',
                "output_collection_flags_changed":False,"historical_baseline_complete_config_was_reconstructed": True})
        write(out / "INITIAL_MODEL_BINDINGS.json", {"adapter_exact": True, "adapter_tensors": 48, "adapter_parameters": 294912,
                "all_parameters_frozen": True, "backbone_hashes": before})
        write(out / "ENVIRONMENT.json", {"GPU": torch.cuda.get_device_name(0), "CUDA": torch.version.cuda,
                "FP32_weights_FP16_autocast": True, "TF32": False, "CPU_threads": torch.get_num_threads(),
                "local_runtime_differs_from_historical_baseline": True, "GPU_total_bytes": torch.cuda.get_device_properties(0).total_memory})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.eval().to("cuda")
        torch.cuda.reset_peak_memory_stats()
        collator = LEDCompleteCollator(5000, 768, 1024)
        predictions = list(reused)
        reuse_provenance=[]
        for case,pred in zip(cases[:5],reused):
            for kind in ('raw_cases','analysis'):
                src=OLD_RUN/kind/(case['business_id']+'.json');dst=out/kind/src.name
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                require(sha(dst)==sha(src),'Imported v1 bytes changed')
            outcomes.append({'business_id':case['business_id'],'subset':case['subset'],'status':'REUSED_UNCHANGED_V1','new_generation_calls':0})
            reuse_provenance.append({'business_id':case['business_id'],'origin_run':str(OLD_RUN.relative_to(ROOT)),
                'raw_sha256':sha(OLD_RUN/'raw_cases'/(case['business_id']+'.json')),
                'analysis_sha256':sha(OLD_RUN/'analysis'/(case['business_id']+'.json')),
                'compatibility':'v1 completed successfully without the terminal-dead-row exception; v2 changes only that terminal handling; result lengths<767; scientific kwargs and analyze identical; historical generation is reused, not rerun'})
        write(out/'REUSE_PROVENANCE.json',reuse_provenance)
        for case in cases[5:]:
            deadline()
            calls_attempted+=1
            require(calls_attempted<=19,'V2 new generation scope exceeded')
            feature = {"input_ids": case["input_ids"], "labels": case["label_ids"]}
            batch = collator([feature])
            verify_mask_batch(batch, [feature])
            batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}
            guard = RepetitionGuardLogitsProcessor(tokenizer, case["source"], trace_path=out / "interventions" / (case["business_id"] + ".jsonl"))
            check = SelectedCandidateSafetyCheck(guard)
            torch.cuda.synchronize()
            start_case = time.perf_counter()
            write(out/'case_starts'/(case['business_id']+'.json'),{'business_id':case['business_id'],
                'source_sha256':case['source_sha256'],'new_call':calls_attempted,'exactly_one_attempt':True})
            try:
                with warnings.catch_warnings(record=True) as caught, FinishedBeamObserver(base,guard) as observer:
                    warnings.simplefilter("always")
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        generated = model.generate(**batch, generation_config=g, logits_processor=LogitsProcessorList([guard]),
                                                   stopping_criteria=StoppingCriteriaList([check]))
                    torch.cuda.synchronize()
                    ids = generated[0].cpu().tolist()
                    write(out/'returned_candidates'/(case['business_id']+'.json'),{'token_ids':ids,
                        'text':tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                        'status':'RAW_RETURN_BEFORE_COMPLETION_VALIDATION_NOT_AUTOMATICALLY_AN_ACCEPTED_OUTPUT'})
                    finish_proof=observer.verify_returned(ids)
                analysis=guard.validate_final(ids)
            except NoCompliantCompletion as exc:
                known={'business_id':case['business_id'],'subset':case['subset'],'status':'FAILED_NO_COMPLIANT_OUTPUT',
                    'error':str(exc),'seconds':time.perf_counter()-start_case,'source_sha256':case['source_sha256'],
                    'terminal_rows_pruned':guard.terminal_rows_pruned,'new_generation_calls':1,
                    'raw_output_fabricated':False,'no_EOS_exemption_or_cap_change':True,**memory_snapshot()}
                write(out/'known_constraint_failures'/(case['business_id']+'.json'),known);outcomes.append(known)
                free,total=torch.cuda.mem_get_info()
                headroom=(free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()-torch.cuda.max_memory_reserved())/total
                require(headroom>=config['minimum_memory_headroom'],'Known failure also violated GPU headroom; stop')
                print(json.dumps({'attempted_new':calls_attempted,'business_id':case['business_id'],
                      'status':known['status'],'continuing_other_businesses_only':True}),flush=True)
                continue
            require(0 < len(ids) <= 768 and ids[0] == 2, "Generation length/start differs")
            pred = {"business_id": case["business_id"], "condition": "filtered", "subset": case["subset"],
                    "source_sha256": case["source_sha256"], "token_ids": ids,
                    "text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "seconds": time.perf_counter() - start_case, "raw_prediction_not_rewritten": True,
                    "guard": {"calls": guard.calls, "tested_candidates": guard.tested_candidates,
                              "blocked_candidates": guard.blocked_candidates,
                              "selected_continuations_checked": guard.selected_continuations_checked,
                              "terminal_rows_pruned":guard.terminal_rows_pruned},
                    "completion_proof":finish_proof,"origin_version":"v2_terminal_repair",
                    "warnings": [str(w.message) for w in caught], "human_verified": False}
            # Preserve raw result even if final safety validation fails.
            write(out / "raw_cases" / (case["business_id"] + ".json"), pred)
            write(out / "analysis" / (case["business_id"] + ".json"), analysis)
            free, total = torch.cuda.mem_get_info()
            headroom = (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() - torch.cuda.max_memory_reserved()) / total
            require(headroom >= config["minimum_memory_headroom"], "Measured GPU headroom below approved minimum")
            predictions.append(pred)
            outcomes.append({'business_id':case['business_id'],'subset':case['subset'],
                'status':'NEW_V2_OUTPUT','new_generation_calls':1,'seconds':pred['seconds'],
                'measured_headroom_fraction':headroom,**memory_snapshot()})
            print(json.dumps({"complete": len(predictions), "of": 24, "business_id": case["business_id"], "seconds": pred["seconds"]}), flush=True)
        require(calls_attempted==19 and len(outcomes)==24,'Unaccounted scope or missing attempt')
        require(all(torch.equal(v.detach().cpu(), expected_weights[k]) for k, v in get_peft_model_state_dict(model).items()), "Adapter changed during inference")
        require(backbone_hashes() == before, "Frozen backbone changed during inference")
        preflight(require_fresh=False)
        predictions.sort(key=lambda p:[c['business_id'] for c in cases].index(p['business_id']))
        write(out/'CASE_OUTCOMES.json',outcomes)
        write(out / "predictions.json", predictions)
        compare_outputs(config, cases, predictions, out)
        write(out / "COMPLETION.json", {"status": "ALL24_OUTPUTS_AVAILABLE_STOP_FOR_REVIEW" if len(predictions)==24 else 'ALL24_ACCOUNTED_WITH_MISSING_CONSTRAINT_OUTPUTS_STOP_FOR_REVIEW',
                "new_generation_calls":calls_attempted,'reused_outputs':5,'available_outputs':len(predictions),
                'constraint_failed_outputs':24-len(predictions),'unique_businesses_accounted':24,
                "seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "checkpoint_unchanged": True,
                "Teacher_calls": 0, "training": False, "held_out_used": False, "D051_strict_replay": "FAIL_UNCHANGED",
                "human_quality_decision": "PENDING", "automatic_next_experiment": False})
    except Exception as exc:
        write(out / "FAILURE_STOP_NO_RETRY.json", {"error": str(exc), "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - started, "no_fallback_or_prediction_rewrite": True,
                'new_calls_attempted':calls_attempted,'case_outcomes_so_far':outcomes,**memory_snapshot()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=str)
    args = parser.parse_args()
    if args.execute:
        execute(args.approval)
    else:
        _, _, _, _, report = preflight()
        print(json.dumps(report))


if __name__ == "__main__":
    main()
