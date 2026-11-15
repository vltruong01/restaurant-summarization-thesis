"""Synthetic CPU tests only: no trained model, real prediction, or human label creation."""
import copy
import ast
from pathlib import Path
import unittest

import stage_i_d053_evaluation as e


class SyntheticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.generation = {"max_length": 8, "eos_token_id": 2, "decoder_start_token_id": 2,
                           "bos_token_id": 0, "pad_token_id": 1, "forced_eos_token_id": 2}
        self.text = "[REVIEW_CONSENSUS]\nFriendly staff.\n\n[VISUAL_EVIDENCE]\nA plate."
        self.prediction = {"business_id": "SYNTHETIC_ONLY", "arm": e.ARMS[0], "text": self.text}
        self.case = {"business_id": "SYNTHETIC_ONLY", "reviews": [{"marker": "REV_1", "review_id": "r", "text": "Friendly staff."}],
                     "captions": [{"caption_id": "IMG_CAP_1", "photo_id": "p", "text": "A plate."}]}
        self.units = [{"unit_id": "u1", "business_id": "SYNTHETIC_ONLY", "domain": "review", "importance": "core",
                       "polarity": "positive", "restaurant_specific": False},
                      {"unit_id": "u2", "business_id": "SYNTHETIC_ONLY", "domain": "visual", "importance": "supporting",
                       "polarity": "neutral", "restaurant_specific": False}]

    def complete_annotation(self):
        annotation = e.human_output_annotation_template(self.prediction, self.units)
        annotation.update(human_verified=True, confirmed_by="SYNTHETIC TEST FIXTURE",
                          atomic_claims_complete=True, source_coverage_complete=True,
                          semantic_groups_complete=True, completeness_and_zero_caption_reviewed=True)
        for i, (text, domain, marker, idkey, source_id) in enumerate([
                ("Friendly staff.", "review", "REV_1", "review_id", "r"),
                ("A plate.", "visual", "IMG_CAP_1", "photo_id", "p")]):
            start = self.text.index(text)
            annotation["claims"].append({"claim_id": f"c{i}", "start": start, "end": start+len(text), "text": text,
                                         "domain": domain, "support_label": "supported", "severity": "none",
                                         "evidence": [{"marker": marker, idkey: source_id, "quote": text,
                                                       "photo_corroborated_by_researcher": True}],
                                         "semantic_group_id": f"g{i}", "cross_caption_relationship_error": False,
                                         "generic_content": False, "low_information": False,
                                         "human_verified": True, "confirmed_by": "SYNTHETIC TEST FIXTURE"})
        for i, u in enumerate(annotation["source_unit_coverage"]):
            u.update(state="covered", human_verified=True, confirmed_by="SYNTHETIC TEST FIXTURE", output_claim_ids=[f"c{i}"])
        annotation["quality"].update(unfinished_clause=False, incoherent=False, unresolved_contradiction=False,
                                      visual_assertion_anywhere=True, human_verified=True, confirmed_by="SYNTHETIC TEST FIXTURE")
        return annotation

    def test_accepted_exact_sentence_definition(self):
        d = e.repetition_diagnostics("Good food.  GOOD food. Good service!")
        self.assertEqual(d["sentence_count"], 3)
        self.assertEqual(d["duplicate_sentence_instances"], 1)
        self.assertEqual(d["duplicate_sentence_fraction"], 1/3)

    def test_adjacent_words_ignore_punctuation(self):
        d = e.repetition_diagnostics("[REVIEW_CONSENSUS] Good, good food.")
        self.assertEqual(d["consecutive_repeated_word_pairs"], 1)
        self.assertEqual(d["adjacent_word_pair_opportunities"], 2)
        self.assertEqual(d["consecutive_repeated_word_pair_fraction"], .5)

    def test_overlapping_ngrams_count_occurrences_not_types(self):
        n = e.ngram_statistics(["a"]*5, 2)
        self.assertEqual(n["opportunities"], 4)
        self.assertEqual(n["duplicate_occurrences"], 3)
        self.assertEqual(n["distinct_n"], .25)

    def test_empty_ngram_opportunities_are_na(self):
        self.assertIsNone(e.ngram_statistics([], 4)["distinct_n"])

    def test_bpe_raw_ids_required(self):
        self.assertIsNone(e.repetition_diagnostics("a a")["generated_bpe_ngrams"])

    def test_bpe_special_exclusion_preserves_duplicate_ids(self):
        d = e.repetition_diagnostics("irrelevanttext", [2, 0, 70, 70, 70, 2, 1])
        self.assertEqual(d["generated_bpe_ngrams"]["2"]["opportunities"], 2)
        self.assertEqual(d["generated_bpe_ngrams"]["2"]["duplicate_occurrences"], 1)

    def test_nonzero_caption_empty_visual_allowed(self):
        d = e.structural_diagnostics("[REVIEW_CONSENSUS]\nGood.\n\n[VISUAL_EVIDENCE]\n", 3)
        self.assertTrue(d["section_headers_compliant"])
        self.assertTrue(d["empty_visual_with_nonzero_captions_is_not_structure_failure"])

    def test_missing_marker_is_not_silent_empty(self):
        d = e.structural_diagnostics("Review then a plate.", 0)
        self.assertFalse(d["section_headers_compliant"])
        self.assertIsNone(d["visual_section_raw"])
        self.assertFalse(d["empty_visual_structurally_compliant"])

    def test_duplicated_marker_preserved(self):
        raw = self.text+"\n[VISUAL_EVIDENCE]\nAgain."
        d = e.structural_diagnostics(raw, 1)
        self.assertTrue(d["duplicated_marker"])
        self.assertEqual(d["raw_text_preserved"], raw)

    def test_extra_source_marker_fails_structure(self):
        d = e.structural_diagnostics(self.text+"\n[IMG_CAP_1]", 1)
        self.assertFalse(d["section_headers_compliant"])

    def test_malformed_marker_fails_structure(self):
        d = e.structural_diagnostics(self.text+"\n[visual evidence]", 1)
        self.assertFalse(d["section_headers_compliant"])

    def test_wrong_newlines_fail(self):
        self.assertFalse(e.structural_diagnostics(self.text.replace("\n\n[VISUAL", "\n[VISUAL"), 1)["section_headers_compliant"])

    def test_initial_decoder_start_is_not_eos(self):
        self.assertFalse(e.token_termination([2, 0, 10], self.generation)["eos_after_decoder_start"])

    def test_cap_eos_not_natural_termination(self):
        d = e.token_termination([2, 0, 10, 11, 12, 13, 14, 2], self.generation)
        self.assertTrue(d["cap_reached"])
        self.assertTrue(d["eos_at_cap_may_be_forced"])
        self.assertFalse(d["eos_before_length_cap"])

    def test_trailing_nonpad_after_eos(self):
        self.assertTrue(e.token_termination([2, 0, 11, 2, 1, 12], self.generation)["nonpadding_tokens_after_first_eos"])

    def test_empty_tokens_rejected(self):
        with self.assertRaises(ValueError):
            e.token_termination([], self.generation)

    def test_human_empty_template_stays_pending(self):
        a = e.human_output_annotation_template(self.prediction, self.units)
        self.assertFalse(a["human_verified"])
        self.assertEqual(a["claims"], [])
        self.assertIsNone(e.human_metrics(a, self.prediction, self.case, self.units)["metrics"])

    def test_absent_human_is_na_not_perfect_grounding(self):
        d = e.human_metrics(None, self.prediction, self.case, self.units)
        self.assertIsNone(d["metrics"])
        self.assertTrue(d["unverified_never_counted_as_zero_errors"])

    def test_complete_synthetic_human_annotations(self):
        a = self.complete_annotation()
        self.assertTrue(e.validate_output_annotation(a, self.prediction, self.case, self.units)["ready"])
        m = e.human_metrics(a, self.prediction, self.case, self.units)["metrics"]
        self.assertEqual(m["grounding"]["visual"]["support_precision_non_uncertain"], 1)
        self.assertIsNone(m["grounding"]["other"]["support_precision_non_uncertain"])

    def test_span_mismatch_rejected(self):
        a = self.complete_annotation()
        a["claims"][0]["end"] += 1
        self.assertFalse(e.validate_output_annotation(a, self.prediction, self.case, self.units)["valid"])

    def test_wrong_photo_binding_rejected(self):
        a = self.complete_annotation()
        a["claims"][1]["evidence"][0]["photo_id"] = "different"
        self.assertFalse(e.validate_output_annotation(a, self.prediction, self.case, self.units)["valid"])

    def test_photo_corroboration_required_for_visual_support(self):
        a = self.complete_annotation()
        a["claims"][1]["evidence"][0]["photo_corroborated_by_researcher"] = False
        self.assertFalse(e.validate_output_annotation(a, self.prediction, self.case, self.units)["valid"])

    def test_coverage_cannot_omit_denominator(self):
        a = self.complete_annotation()
        a["source_unit_coverage"].pop()
        self.assertFalse(e.validate_output_annotation(a, self.prediction, self.case, self.units)["valid"])

    def test_partial_is_sensitivity_not_primary(self):
        a = self.complete_annotation()
        a["source_unit_coverage"][0]["state"] = "partly_covered"
        m = e.human_metrics(a, self.prediction, self.case, self.units)["metrics"]["source_coverage"]["all"]
        self.assertEqual(m["fully_covered_fraction"], .5)
        self.assertEqual(m["partial_half_sensitivity"], .75)

    def test_uncertain_claim_stays_separate(self):
        a = self.complete_annotation()
        a["claims"][1].update(support_label="uncertain", severity="uncertain")
        d = e.human_metrics(a, self.prediction, self.case, self.units)["metrics"]["grounding"]["visual"]
        self.assertIsNone(d["support_precision_non_uncertain"])
        self.assertEqual(d["label_counts"]["uncertain"], 1)
        self.assertEqual(d["uncertainty_all_unsupported_support_precision"], 0)
        self.assertEqual(d["uncertainty_all_supported_support_precision"], 1)

    def test_human_groups_not_lexical_similarity(self):
        a = self.complete_annotation()
        a["claims"][1]["semantic_group_id"] = a["claims"][0]["semantic_group_id"]
        d = e.human_metrics(a, self.prediction, self.case, self.units)["metrics"]["semantic_redundancy"]
        self.assertEqual(d["redundant_fraction"], .5)

    def test_bootstrap_paired_constant_delta(self):
        pairs = [{"business_id": str(i), "A": i, "B": i+2} for i in range(20)]
        d = e.paired_bootstrap(pairs)
        self.assertEqual(d["bootstrap"]["interval"], [2, 2])
        self.assertEqual(d["mean_paired_difference"], 2)
        self.assertEqual(d["B_wins"], 20)

    def test_bootstrap_reproducible_and_missing_not_zero(self):
        pairs = [{"business_id": "a", "A": 0, "B": 3}, {"business_id": "b", "A": 5, "B": 1}, {"business_id": "c", "A": None, "B": 10}]
        self.assertEqual(e.paired_bootstrap(pairs), e.paired_bootstrap(pairs))
        self.assertEqual(e.paired_bootstrap(pairs)["missing_pairs"], 1)

    def test_bootstrap_single_pair_no_interval(self):
        self.assertIsNone(e.paired_bootstrap([{"business_id": "a", "A": 1, "B": 2}])["bootstrap"]["interval"])

    def test_duplicate_business_pairs_rejected(self):
        with self.assertRaises(ValueError):
            e.paired_bootstrap([{"business_id": "a", "A": 1, "B": 2}]*2)

    def test_bootstrap_matches_frozen_D049_implementation(self):
        # Extract just the original function from immutable source, avoiding its
        # model imports. This checks compatibility rather than a new RNG choice.
        import numpy as np
        source = Path(__file__).with_name("stage_i_paired_pilot.py").read_text(encoding="utf-8")
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "paired_bootstrap")
        namespace = {"np": np, "require": lambda ok, message: None if ok else (_ for _ in ()).throw(ValueError(message))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen_D049_function_only", "exec"), namespace)
        aa, bb = [0, 5, 2, 7, 1], [3, 1, 6, 2, 9]
        frozen = namespace["paired_bootstrap"](aa, bb)
        current = e.paired_bootstrap([{"business_id": str(i), "A": a, "B": b} for i, (a, b) in enumerate(zip(aa, bb))])
        self.assertEqual(current["bootstrap"]["interval"], frozen["CI95"])
        self.assertEqual(current["mean_paired_difference"], frozen["mean_B_minus_A"])

    def test_predictions_require_exactly48_bound_outputs(self):
        cases = [dict(self.case, business_id=str(i), subset="random" if i < 20 else "targeted", source_sha256="source") for i in range(24)]
        config = {"arms": {arm: {"adapter_sha256": arm} for arm in e.ARMS},
                  "bindings": {"generation_config": {"sha256": "generation"}}, "effective_generation_config": self.generation}
        predictions = [{"arm": arm, "business_id": c["business_id"], "subset": c["subset"], "condition": "filtered",
                        "source_sha256": "source", "adapter_sha256": arm, "generation_config_sha256": "generation",
                        "text": self.text, "token_ids": [2, 0, 40, 2]} for arm in e.ARMS for c in cases]
        e.validate_predictions(predictions, cases, config)
        with self.assertRaises(ValueError):
            e.validate_predictions(predictions[:-1], cases, config)
        wrong = copy.deepcopy(predictions)
        wrong[0]["source_sha256"] = "changed"
        with self.assertRaises(ValueError):
            e.validate_predictions(wrong, cases, config)

    def test_targeted_results_have_no_pooled_prevalence_or_CI(self):
        cfg = {"random_business_ids": ["r"], "targeted_business_ids": ["t"],
               "paired_measures": {"auto.value": "lower_is_better"},
               "bootstrap": {"seed": 20261016, "resamples": 5000, "confidence_level": .95}}
        rows = [{"arm": arm, "business_id": b, "subset": subset, "automatic": {"value": 1},
                 "human": {"metrics": None}} for arm in e.ARMS for b, subset in [("r", "random"), ("t", "targeted")]]
        result = e.paired_comparison(rows, cfg)
        self.assertIsNone(result["targeted"]["inferential_CI"])
        self.assertIsNone(result["targeted"]["pooled_prevalence"])
        self.assertIsNone(result["pooled_claim_counts_descriptive_only"]["random"][e.ARMS[0]]["domains"]["review"])

    def test_blind_order_deterministic_not_fixed_arm(self):
        ids = [str(i) for i in range(24)]
        self.assertEqual(e.blinded_orders(ids), e.blinded_orders(ids))
        self.assertEqual({v["X"] for v in e.blinded_orders(ids).values()}, set(e.ARMS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
