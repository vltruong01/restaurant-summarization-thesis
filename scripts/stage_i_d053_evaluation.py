"""D-053 offline evaluation preparation. No model loading, inference, or factual judge.

CPU procedures preserve D-049 definitions (NumPy only inside paired bootstrap).
Annotation flags are provenance
claims made by the researcher, not facts that software can independently verify.
No student predictions are created by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from stage_i_decoding_diagnostics import text_diagnostics, words

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_i_d053_evaluation_v1.json"
ARMS = ("A_decoder_only", "B_encoder_decoder")
REVIEW = "[REVIEW_CONSENSUS]"
VISUAL = "[VISUAL_EVIDENCE]"
LABELS = ("supported", "contradicted", "not_in_permitted_evidence", "wrong_evidence_domain", "uncertain")
COVERAGE = ("covered", "partly_covered", "omitted", "uncertain")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def ngram_statistics(sequence, n):
    """Overlapping occurrences; duplicated occurrences are sum(count - 1)."""
    grams = [tuple(sequence[i:i+n]) for i in range(max(0, len(sequence)-n+1))]
    counts = Counter(grams)
    duplicate = sum(v-1 for v in counts.values())
    return {"n": n, "opportunities": len(grams), "distinct_ngrams": len(counts),
            "duplicate_occurrences": duplicate, "duplicate_fraction": rate(duplicate, len(grams)),
            "distinct_n": rate(len(counts), len(grams))}


def structural_diagnostics(text, caption_count):
    """Inspect raw text without repairing sections; empty visual is valid for any K."""
    exact_markers = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\[[A-Z_0-9]+\]", text)]
    suspicious = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\[[^\]\n]{1,80}\]", text)]
    r = [m.start() for m in re.finditer(re.escape(REVIEW), text)]
    v = [m.start() for m in re.finditer(re.escape(VISUAL), text)]
    ordered = len(r) == len(v) == 1 and r[0] < v[0]
    marker_lines = text.startswith(REVIEW+"\n") and ordered and "\n\n"+VISUAL+"\n" in text
    review_text = text[len(REVIEW):v[0]].strip() if ordered else None
    visual_text = text[v[0]+len(VISUAL):].strip() if ordered else None
    extras = [m for m in suspicious if m[0] not in (REVIEW, VISUAL)]
    compliant = bool(marker_lines and review_text and not extras)
    zero = caption_count == 0
    return {"section_headers_compliant": compliant,
            "review_marker_count": len(r), "visual_marker_count": len(v),
            "missing_review_marker": not r, "missing_visual_marker": not v,
            "duplicated_marker": len(r) > 1 or len(v) > 1,
            "marker_order_correct": ordered, "required_newlines_present": bool(marker_lines),
            "extra_or_malformed_bracket_markers": extras,
            "raw_marker_spans": exact_markers, "review_section_raw": review_text,
            "visual_section_raw": visual_text, "raw_text_preserved": text,
            "empty_visual_required": zero,
            "empty_visual_structurally_compliant": bool(compliant and not visual_text) if zero else None,
            "empty_visual_with_nonzero_captions_is_not_structure_failure": bool(not zero and compliant and not visual_text),
            "zero_caption_assertions_elsewhere_require_human_review": zero}


def token_termination(token_ids, config):
    """Initial LED start token2 is not a terminating EOS; cap EOS may be forced."""
    if not isinstance(token_ids, list) or not token_ids or any(type(x) is not int or x < 0 for x in token_ids):
        raise ValueError("Generated token IDs must be a nonempty list of nonnegative integers")
    max_length = config["max_length"]
    eos = config["eos_token_id"]
    positions = [i for i, t in enumerate(token_ids) if i > 0 and t == eos]
    first = positions[0] if positions else None
    trailing = token_ids[first+1:] if first is not None else []
    return {"generated_tokens_including_decoder_start": len(token_ids),
            "decoder_start_correct": token_ids[0] == config["decoder_start_token_id"],
            "generated_bos_after_start": len(token_ids) > 1 and token_ids[1] == config["bos_token_id"],
            "eos_after_decoder_start": first is not None, "first_eos_position": first,
            "eos_before_length_cap": first is not None and first < max_length-1,
            "cap_reached": len(token_ids) >= max_length,
            "exceeds_length_cap": len(token_ids) > max_length,
            "eos_at_cap_may_be_forced": first == max_length-1 and config["forced_eos_token_id"] == eos,
            "nonpadding_tokens_after_first_eos": any(t != config["pad_token_id"] for t in trailing),
            "semantic_completeness_requires_human_review": True}


def repetition_diagnostics(text, token_ids=None, generation=None):
    result = text_diagnostics(text)  # accepted sentence/adjacent-word definitions unchanged
    lexical = words(text)
    result["lexical_word_count"] = len(lexical)
    result["adjacent_word_pair_opportunities"] = max(0, len(lexical)-1)
    result["lexical_ngrams"] = {str(n): ngram_statistics(lexical, n) for n in (2, 3, 4)}
    if token_ids is None:
        result["generated_bpe_ngrams"] = None
        result["bpe_status"] = "NOT_EVALUABLE_RAW_TOKEN_IDS_ABSENT"
    else:
        generation = generation or {"decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1, "bos_token_id": 0}
        # Remove only known special tokens; retain all ordinary generated BPE IDs.
        # Never reconstruct token IDs by approximate text re-tokenization.
        content = list(token_ids)
        if content and content[0] == generation["decoder_start_token_id"]:
            content = content[1:]
        content = [t for t in content if t not in (generation["eos_token_id"], generation["pad_token_id"], generation["bos_token_id"])]
        result["generated_bpe_ngrams"] = {str(n): ngram_statistics(content, n) for n in (2, 3, 4)}
        result["bpe_status"] = "EXACT_STORED_GENERATED_CONTENT_IDS_SPECIAL_TOKENS_EXCLUDED"
    return result


def human_output_annotation_template(prediction, units):
    """Empty fields only. No automatic claim segmentation or support decisions."""
    return {"business_id": prediction["business_id"], "arm": prediction["arm"],
            "prediction_text_sha256": text_sha256(prediction["text"]),
            "human_verified": False, "confirmed_by": None,
            "atomic_claims_complete": False, "source_coverage_complete": False,
            "semantic_groups_complete": False, "completeness_and_zero_caption_reviewed": False,
            "claims": [],
            "source_unit_coverage": [{"unit_id": u["unit_id"], "state": None, "human_verified": False,
                                      "confirmed_by": None, "output_claim_ids": [], "notes": ""} for u in units],
            "quality": {"unfinished_clause": None, "incoherent": None,
                        "unresolved_contradiction": None, "important_contrast_preserved": None,
                        "visual_assertion_anywhere": None, "human_verified": False,
                        "confirmed_by": None, "notes": ""},
            "notes": "No automatic assessment. Segment every factual/opinion claim in all raw sections and tails."}


def future_annotation_shell(cases, config):
    """48 blank annotation shells, without creating fake prediction records.

    Source-unit coverage IDs must be populated from the later researcher-confirmed
    reference freeze. Candidate IDs are intentionally not eligible denominators.
    After approved generation, human_output_annotation_template binds each real
    output hash and the actual frozen eligible unit IDs.
    """
    return {"status": "TEMPLATE_ONLY_NO_STUDENT_PREDICTIONS_EXIST", "human_annotations": [
        {"business_id": case["business_id"], "arm": arm, "subset": case["subset"], "condition": "filtered",
         "prediction_text_sha256": None, "human_verified": False, "confirmed_by": None,
         "atomic_claims_complete": False, "source_coverage_complete": False,
         "semantic_groups_complete": False, "completeness_and_zero_caption_reviewed": False,
         "claims": [], "source_unit_coverage": [],
         "quality": {"unfinished_clause": None, "incoherent": None, "unresolved_contradiction": None,
                     "important_contrast_preserved": None, "visual_assertion_anywhere": None,
                     "human_verified": False, "confirmed_by": None, "notes": ""},
         "notes": "Blank shell only. Bind a real output after separately approved generation; coverage uses later frozen verified units, never unverified candidates."}
        for case in cases for arm in ARMS], "expected_future_outputs": config["future_output_count"],
        "no_output_claims_or_support_labels_generated": True}


def _confirmed(obj):
    return obj.get("human_verified") is True and isinstance(obj.get("confirmed_by"), str) and bool(obj["confirmed_by"].strip())


def _evidence_check(evidence, case, required_domain=None):
    if not isinstance(evidence, list):
        return ["evidence must be a list"]
    errors = []
    for item in evidence:
        marker = item.get("marker")
        reviews = {r["marker"]: r for r in case["reviews"]}
        captions = {c["caption_id"]: c for c in case["captions"]}
        if marker in reviews:
            source, domain, idkey = reviews[marker], "review", "review_id"
        elif marker in captions:
            source, domain, idkey = captions[marker], "visual", "photo_id"
        else:
            errors.append("unknown evidence marker")
            continue
        if item.get(idkey) != source[idkey] or not item.get("quote") or item["quote"] not in source["text"]:
            errors.append("evidence ID/quotation mismatch")
        if required_domain and domain != required_domain:
            errors.append("supported claim uses wrong evidence domain")
        if domain == "visual" and required_domain == "visual" and item.get("photo_corroborated_by_researcher") is not True:
            errors.append("visual support requires explicit photo corroboration")
    return errors


def validate_output_annotation(annotation, prediction, case, units):
    """Return errors and readiness, never assign labels or hide missing judgments."""
    errors = []
    if annotation.get("business_id") != case["business_id"] or annotation.get("arm") != prediction["arm"]:
        errors.append("annotation business/arm differs")
    if annotation.get("prediction_text_sha256") != text_sha256(prediction["text"]):
        errors.append("annotation prediction hash differs")
    ready = _confirmed(annotation) and all(annotation.get(k) is True for k in
            ("atomic_claims_complete", "source_coverage_complete", "semantic_groups_complete", "completeness_and_zero_caption_reviewed"))
    claims = annotation.get("claims", [])
    if not isinstance(claims, list):
        return {"valid": False, "ready": False, "errors": ["claims must be a list"]}
    ids = []
    for c in claims:
        ids.append(c.get("claim_id"))
        a, b = c.get("start"), c.get("end")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(prediction["text"]) or prediction["text"][a:b] != c.get("text"):
            errors.append("claim span/quotation mismatch")
        if c.get("domain") not in ("review", "visual", "other") or c.get("support_label") not in LABELS:
            errors.append("invalid claim domain/label")
        if c.get("severity") not in ("none", "minor", "major", "severe", "uncertain"):
            errors.append("severity missing/invalid")
        for flag in ("cross_caption_relationship_error", "generic_content", "low_information"):
            if type(c.get(flag)) is not bool:
                errors.append(f"{flag} must be explicit human boolean")
        if not c.get("semantic_group_id"):
            errors.append("human semantic group missing")
        ready = ready and _confirmed(c)
        evidence = c.get("evidence", [])
        if c.get("support_label") == "supported" and not evidence:
            errors.append("supported claim lacks evidence")
        errors.extend(_evidence_check(evidence, case, c.get("domain") if c.get("support_label") == "supported" else None))
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        errors.append("claim IDs missing/duplicated")
    coverage = annotation.get("source_unit_coverage", [])
    if {u.get("unit_id") for u in coverage} != {u["unit_id"] for u in units} or len(coverage) != len(units):
        errors.append("coverage must account exactly once for every eligible frozen unit")
    for u in coverage:
        if u.get("state") not in COVERAGE:
            ready = False
        ready = ready and _confirmed(u)
        links = u.get("output_claim_ids", [])
        if any(i not in ids for i in links):
            errors.append("coverage refers to unknown output claim")
        if u.get("state") in ("covered", "partly_covered") and not links:
            errors.append("covered unit requires claim/span link")
    quality = annotation.get("quality", {})
    ready = ready and _confirmed(quality)
    for flag in ("unfinished_clause", "incoherent", "unresolved_contradiction", "visual_assertion_anywhere"):
        if type(quality.get(flag)) is not bool:
            ready = False
    if quality.get("important_contrast_preserved") not in (True, False, None):
        errors.append("contrast must be human boolean or not-applicable null")
    return {"valid": not errors, "ready": bool(ready and not errors), "errors": errors}


def _coverage_group(units, states):
    count = Counter(states[u["unit_id"]] for u in units)
    n = len(units)
    return {"eligible_units": n, "counts": {k: count[k] for k in COVERAGE},
            "fully_covered_fraction": rate(count["covered"], n),
            "partial_half_sensitivity": rate(count["covered"] + .5*count["partly_covered"], n),
            "omitted_unit_ids": [u["unit_id"] for u in units if states[u["unit_id"]] == "omitted"],
            "uncertain_unit_ids": [u["unit_id"] for u in units if states[u["unit_id"]] == "uncertain"]}


def human_metrics(annotation, prediction, case, units):
    check = validate_output_annotation(annotation, prediction, case, units) if annotation else None
    if not check or not check["ready"]:
        return {"status": "NOT_EVALUABLE_HUMAN_ANNOTATIONS_INCOMPLETE", "metrics": None,
                "validation": check, "unverified_never_counted_as_zero_errors": True}
    claims = annotation["claims"]
    grounding = {}
    for domain in ("review", "visual", "other"):
        cc = [c for c in claims if c["domain"] == domain]
        counts = Counter(c["support_label"] for c in cc)
        n = len(cc)
        certain = n-counts["uncertain"]
        unsupported = sum(counts[k] for k in ("contradicted", "not_in_permitted_evidence", "wrong_evidence_domain"))
        grounding[domain] = {"claim_count": n, "label_counts": {k: counts[k] for k in LABELS},
                             "non_uncertain_denominator": certain,
                             "support_precision_non_uncertain": rate(counts["supported"], certain),
                             "unsupported_fraction_non_uncertain": rate(unsupported, certain),
                             "uncertainty_all_unsupported_support_precision": rate(counts["supported"], n),
                             "uncertainty_all_supported_support_precision": rate(counts["supported"]+counts["uncertain"], n),
                             "cross_caption_relationship_errors": sum(c["cross_caption_relationship_error"] for c in cc),
                             "severe_error_claim_ids": [c["claim_id"] for c in cc if c["severity"] == "severe" and c["support_label"] != "supported"]}
    states = {r["unit_id"]: r["state"] for r in annotation["source_unit_coverage"]}
    coverage = {"all": _coverage_group(units, states)}
    for field, values in (("domain", ("review", "visual")), ("importance", ("core", "supporting")),
                          ("polarity", ("positive", "negative", "mixed", "neutral")),
                          ("restaurant_specific", (True, False))):
        for v in values:
            coverage[f"{field}:{v}"] = _coverage_group([u for u in units if u.get(field) == v], states)
    groups = Counter(c["semantic_group_id"] for c in claims)
    repeated = sum(v-1 for v in groups.values())
    return {"status": "RESEARCHER_CONFIRMED_ANNOTATIONS_AVAILABLE_NOT_INDEPENDENT_MACHINE_TRUTH",
            "metrics": {"grounding": grounding, "source_coverage": coverage,
                        "semantic_redundancy": {"propositions": len(claims), "groups": len(groups),
                                                "repeated_proposition_instances": repeated, "redundant_fraction": rate(repeated, len(claims))},
                        "generic_claims": sum(c["generic_content"] for c in claims),
                        "low_information_claims": sum(c["low_information"] for c in claims),
                        "useful_restaurant_specific_units_fully_covered": sum(states[u["unit_id"]] == "covered" and u.get("restaurant_specific") is True for u in units),
                        "quality": annotation["quality"],
                        "zero_caption_no_visual_assertion_anywhere": not annotation["quality"]["visual_assertion_anywhere"] if not case["captions"] else None},
            "validation": check}


def _percentile(values, fraction):
    position = (len(values)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def paired_bootstrap(pairs, seed=20261016, resamples=5000, ci=.95, direction="higher_is_better"):
    """Input business_id/A/B; resample businesses, keeping arms paired."""
    if len({p["business_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate business pairs")
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError("Metric direction must be explicit")
    valid = [p for p in pairs if p.get("A") is not None and p.get("B") is not None]
    if any(type(p[a]) not in (int, float) or not math.isfinite(p[a]) for p in valid for a in ("A", "B")):
        raise ValueError("Nonfinite or nonnumeric paired metric; not silently treated as missing")
    diffs = [p["B"]-p["A"] for p in valid]
    interval = None
    if len(diffs) >= 2:
        # Exact predeclared D049 RNG algorithm and percentile implementation.
        # NumPy is already pinned in the accepted runtime; no Torch/model import.
        import numpy as np
        delta = np.asarray(diffs, dtype=float)
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(1)
        interval = np.quantile(means, [(1-ci)/2, 1-(1-ci)/2]).tolist()
    sign = 1 if direction == "higher_is_better" else -1
    return {"direction": direction, "difference_definition": "B_minus_A", "complete_pairs": len(valid),
            "missing_pairs": len(pairs)-len(valid), "A_macro_mean": statistics.mean(p["A"] for p in valid) if valid else None,
            "B_macro_mean": statistics.mean(p["B"] for p in valid) if valid else None,
            "mean_paired_difference": statistics.mean(diffs) if diffs else None,
            "B_wins": sum(d*sign > 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "B_losses": sum(d*sign < 0 for d in diffs),
            "bootstrap": {"seed": seed, "resamples": resamples, "confidence_level": ci,
                          "unit": "business_pair", "method": "numpy_default_rng_integers_then_quantile_linear_D049",
                          "interval": interval,
                          "reason_if_absent": "Fewer than2 complete business pairs" if interval is None else None}}


def blinded_orders(business_ids, seed=20261016):
    """Stable display ordering only; not a new sample or inference."""
    rng = random.Random(seed)
    orders = {}
    for business_id in business_ids:
        arms = list(ARMS)
        rng.shuffle(arms)
        orders[business_id] = {"X": arms[0], "Y": arms[1]}
    return orders


def validate_config(config, root=ROOT):
    errors = []
    for entry in config["bindings"].values():
        p = root / entry["path"]
        if not p.is_file() or sha256(p) != entry["sha256"]:
            errors.append("binding differs: "+entry["path"])
    for arm in ARMS:
        entry = config["arms"][arm]
        if sha256(root/entry["adapter_path"]) != entry["adapter_sha256"]:
            errors.append("adapter differs: "+arm)
    random_ids, target_ids = config["random_business_ids"], config["targeted_business_ids"]
    if len(random_ids) != 20 or len(target_ids) != 4 or len(set(random_ids+target_ids)) != 24:
        errors.append("Expected unchanged20 random and4 disjoint targeted IDs")
    original = read_json(root/config["bindings"]["paired_training_config"]["path"])
    generation = read_json(root/config["bindings"]["generation_config"]["path"])
    if original["generation"] != config["generation_overrides"] or generation != config["effective_generation_config"]:
        errors.append("Frozen generation definitions differ")
    manifest = read_json(root/config["bindings"]["pilot_manifest"]["path"])
    if random_ids != manifest["generation_evaluation_random20"] or target_ids != manifest["generation_evaluation_targeted4"]:
        errors.append("Business IDs/order differ from approved manifest")
    return {"valid": not errors, "errors": errors, "status": config["status"]}


def validate_predictions(predictions, cases, config):
    lookup = {c["business_id"]: c for c in cases}
    keys = [(p.get("arm"), p.get("business_id")) for p in predictions]
    expected = {(arm, business_id) for arm in ARMS for business_id in lookup}
    if len(keys) != 48 or len(set(keys)) != 48 or set(keys) != expected:
        raise ValueError("Future comparison requires exactly48 unique matched predictions, not historical outputs")
    for p in predictions:
        c = lookup[p["business_id"]]
        if p.get("condition") != "filtered" or p.get("subset") != c["subset"] or p.get("source_sha256") != c["source_sha256"]:
            raise ValueError("Prediction input/split/condition binding differs")
        if p.get("adapter_sha256") != config["arms"][p["arm"]]["adapter_sha256"]:
            raise ValueError("Prediction not from paired D052 adapter")
        if p.get("generation_config_sha256") != config["bindings"]["generation_config"]["sha256"]:
            raise ValueError("Prediction generation configuration differs")
        if not isinstance(p.get("text"), str):
            raise ValueError("Raw prediction text missing")
        termination = token_termination(p.get("token_ids"), config["effective_generation_config"])
        if termination["exceeds_length_cap"] or not termination["decoder_start_correct"]:
            raise ValueError("Generated token sequence incompatible with frozen configuration")


def _flatten_numeric(obj, prefix=""):
    result = {}
    for key, value in obj.items():
        name = prefix+key
        if isinstance(value, dict):
            result.update(_flatten_numeric(value, name+"."))
        elif type(value) in (bool, int, float):
            result[name] = float(value)
        elif value is None:
            result[name] = None
    return result


def paired_comparison(per_output, config):
    """Report scalar paired differences and human missingness; no quality winner."""
    indexed = {(r["arm"], r["business_id"]): r for r in per_output}
    result = {"random": {}, "targeted": {}, "composite_quality_score": None,
              "automatic_winner": None, "quality_interpretation": "Researcher multidomain rubric; lower repetition alone is insufficient"}
    for subset, business_ids in (("random", config["random_business_ids"]), ("targeted", config["targeted_business_ids"])):
        scalar = {}
        for business_id in business_ids:
            for arm in ARMS:
                record = indexed[(arm, business_id)]
                numeric = {"auto."+k: v for k, v in _flatten_numeric(record["automatic"]).items()}
                if record["human"]["metrics"] is not None:
                    numeric.update({"human."+k: v for k, v in _flatten_numeric(record["human"]["metrics"]).items()})
                scalar[(arm, business_id)] = numeric
        metric_names = sorted(config["paired_measures"])
        pairs = {name: [{"business_id": b, "A": scalar[(ARMS[0], b)].get(name),
                         "B": scalar[(ARMS[1], b)].get(name)} for b in business_ids] for name in metric_names}
        if subset == "random":
            summaries = {}
            for name, pp in pairs.items():
                direction = config["paired_measures"][name]
                summary = paired_bootstrap(pp, **{k: config["bootstrap"][k] for k in ("seed", "resamples")},
                                           ci=config["bootstrap"]["confidence_level"],
                                           direction=direction if direction != "descriptive_only" else "higher_is_better")
                if direction == "descriptive_only":
                    for k in ("B_wins", "B_losses", "ties"):
                        summary.pop(k)
                    summary["direction"] = "descriptive_only_not_quality_direction"
                summaries[name] = summary
            result[subset] = {"businesses": len(business_ids), "paired_metrics_B_minus_A": summaries,
                              "human_incomplete_outputs": sum(indexed[(arm, b)]["human"]["metrics"] is None for b in business_ids for arm in ARMS)}
        else:
            result[subset] = {"businesses": len(business_ids), "individual_values": pairs,
                              "inferential_CI": None, "pooled_prevalence": None,
                              "human_incomplete_outputs": sum(indexed[(arm, b)]["human"]["metrics"] is None for b in business_ids for arm in ARMS)}
    # Pooled claims are explicitly descriptive; macro paired rates above remain
    # business-level. A missing human assessment contributes no invented counts.
    result["pooled_claim_counts_descriptive_only"] = {}
    for subset in ("random", "targeted"):
        result["pooled_claim_counts_descriptive_only"][subset] = {}
        for arm in ARMS:
            complete = [r for r in per_output if r["arm"] == arm and r["subset"] == subset and r["human"]["metrics"] is not None]
            result["pooled_claim_counts_descriptive_only"][subset][arm] = {
                "human_complete_outputs": len(complete),
                "domains": {domain: {label: sum(r["human"]["metrics"]["grounding"][domain]["label_counts"][label] for r in complete)
                                      for label in LABELS} if complete else None for domain in ("review", "visual", "other")}}
    return result


def load_future_comparison(reference_annotations_path, reference_freeze_path, predictions_path,
                           output_annotations_path=None, config_path=CONFIG):
    """Future CPU-only evaluation after generation is independently approved.

    Requires final source annotations to pass the D053 evidence validator and
    reproduce the frozen reference byte-semantically. This function cannot
    generate predictions. Human quality measures stay N/A until separately
    researcher-confirmed output annotations are complete.
    """
    config = read_json(config_path)
    check = validate_config(config)
    if not check["valid"]:
        raise ValueError("Evaluation bindings invalid: "+str(check["errors"]))
    from stage_i_d053_annotations import validate_annotations, reference_freeze
    cases = read_json(ROOT/config["bindings"]["source_cases"]["path"])
    candidates = read_jsonl(ROOT/config["bindings"]["candidates"]["path"])
    bindings = {"case_inputs_sha256": config["bindings"]["case_inputs"]["sha256"],
                "protocol_sha256": config["bindings"]["d049_protocol"]["sha256"],
                "candidate_sha256": config["bindings"]["candidates"]["sha256"],
                "source_cases_sha256": config["bindings"]["source_cases"]["sha256"],
                "evaluation_config_sha256": sha256(config_path)}
    state = read_json(reference_annotations_path)
    reference_check = validate_annotations(state, cases, candidates, bindings, final=True)
    if not reference_check["valid"] or not reference_check["ready_for_freeze"]:
        raise ValueError("Source references are not final researcher-confirmed evidence")
    frozen = read_json(reference_freeze_path)
    if reference_freeze(state, cases, candidates, bindings) != frozen:
        raise ValueError("Frozen reference does not reproduce verified annotation state")
    predictions = read_jsonl(predictions_path)
    validate_predictions(predictions, cases, config)
    if any(p.get("reference_freeze_sha256") != sha256(reference_freeze_path) for p in predictions):
        raise ValueError("Predictions must bind the previously frozen human reference")
    output_annotations = read_json(output_annotations_path) if output_annotations_path else []
    annotation_keys = [(r["arm"], r["business_id"]) for r in output_annotations]
    if len(annotation_keys) != len(set(annotation_keys)):
        raise ValueError("Duplicate output annotations")
    expected = {(p["arm"], p["business_id"]) for p in predictions}
    if set(annotation_keys)-expected:
        raise ValueError("Output annotations contain unexpected businesses/arms")
    annotations = dict(zip(annotation_keys, output_annotations))
    cases_by_id = {c["business_id"]: c for c in cases}
    per_output = []
    pending_templates = []
    for prediction in predictions:
        case = cases_by_id[prediction["business_id"]]
        units = [u for u in frozen["units"] if u["business_id"] == case["business_id"]]
        annotation = annotations.get((prediction["arm"], case["business_id"]))
        automatic = {"repetition": repetition_diagnostics(prediction["text"], prediction["token_ids"], config["effective_generation_config"]),
                     "structure": structural_diagnostics(prediction["text"], len(case["captions"])),
                     "termination": token_termination(prediction["token_ids"], config["effective_generation_config"])}
        per_output.append({"arm": prediction["arm"], "business_id": case["business_id"], "subset": case["subset"],
                           "prediction_text_sha256": text_sha256(prediction["text"]), "automatic": automatic,
                           "human": human_metrics(annotation, prediction, case, units)})
        if annotation is None:
            pending_templates.append(human_output_annotation_template(prediction, units))
    return {"status": "EXISTING_FUTURE_PREDICTIONS_EVALUATED_CPU_ONLY", "per_output": per_output,
            "paired": paired_comparison(per_output, config), "pending_output_annotation_templates": pending_templates,
            "source_reference_validation": reference_check,
            "ROUGE": "Not computed here; existing approved secondary teacher-agreement implementation may be invoked separately",
            "BERTScore": "NOT_COMPUTED_NO_APPROVED_PINNED_SCORER", "automatic_factual_judgment": False}


def offline_readiness(config):
    return {"status": "OFFLINE_EVALUATION_PREPARED_NO_INFERENCE_PERMISSION", "configuration": validate_config(config),
            "expected_future_outputs": 48, "reference_candidates_are_human_truth": False,
            "actual_new_predictions_evaluated": 0, "GPU_or_model_execution": False,
            "future_requirements": ["Researcher-confirmed source annotation freeze passes evidence validation",
                                    "Separate researcher approval before any Pilot A/B generation",
                                    "Exactly48 actual outputs with source/adapter/generation/token bindings",
                                    "Researcher-confirmed atomic output claim, coverage and semantic-group annotations for human metrics"],
            "ROUGE": "Secondary teacher agreement only; not computed during preparation",
            "BERTScore": "NOT_COMPUTED_NO_APPROVED_PINNED_SCORER", "composite_quality_score": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--readiness", action="store_true", help="CPU only; verify frozen file bindings")
    parser.add_argument("--write-future-shell", type=Path, help="Write48 empty future annotation shells; no model outputs")
    parser.add_argument("--compare-existing", "--evaluate-existing", dest="evaluate_existing", action="store_true", help="Future CPU-only evaluation of already-generated48 outputs; requires verified source freeze")
    parser.add_argument("--reference-annotations", type=Path)
    parser.add_argument("--reference-freeze", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sum((args.readiness, args.write_future_shell is not None, args.evaluate_existing)) != 1:
        parser.error("Choose exactly one offline operation; no model inference exists here")
    config = read_json(args.config)
    if args.write_future_shell:
        if not validate_config(config)["valid"]:
            raise ValueError("Frozen bindings differ")
        report = future_annotation_shell(read_json(ROOT/config["bindings"]["source_cases"]["path"]), config)
        with args.write_future_shell.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "shells": len(report["human_annotations"]), "output": str(args.write_future_shell)}))
        return
    if args.evaluate_existing:
        if not all((args.reference_annotations, args.reference_freeze, args.predictions, args.output)):
            parser.error("Existing evaluation requires --reference-annotations --reference-freeze --predictions --output")
        report = load_future_comparison(args.reference_annotations, args.reference_freeze, args.predictions,
                                        args.output_annotations, args.config)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "outputs": len(report["per_output"]), "output": str(args.output)}))
        return
    report = offline_readiness(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["configuration"]["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
