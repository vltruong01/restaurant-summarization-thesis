"""Boundary checks for evaluation and fail-closed execution; no model/GPU imports."""
import json
from pathlib import Path
import tempfile
import unittest

from stage_i_decoding_diagnostics import structure, termination, text_diagnostics
from run_stage_i_decoding_diagnostic import approval_gate, sha


class DiagnosticBoundaries(unittest.TestCase):
    def test_decoder_start_is_not_eos_termination(self):
        result = termination([2, 0, 500, 600])
        self.assertFalse(result["eos_after_decoder_start"])
        self.assertTrue(result["decoder_start_correct"])

    def test_normal_early_eos(self):
        result = termination([2, 0, 500, 2])
        self.assertTrue(result["eos_before_length_cap"])
        self.assertFalse(result["cap_reached"])

    def test_forced_eos_at_cap_is_not_natural_completeness(self):
        result = termination([2, 0] + [500] * 765 + [2])
        self.assertTrue(result["cap_reached"])
        self.assertTrue(result["eos_at_cap_may_be_forced"])
        self.assertFalse(result["eos_before_length_cap"])

    def test_nonpadding_after_eos(self):
        self.assertTrue(termination([2, 0, 2, 500])["nonpadding_tokens_after_first_eos"])

    def test_repetition_ignores_case_and_spacing(self):
        self.assertTrue(text_diagnostics("A table is visible. A   TABLE is visible.")["exact_repeated_sentence"])
        self.assertFalse(text_diagnostics("A table is visible. A chair is visible.")["exact_repeated_sentence"])

    def test_word_repetition_includes_comma_lists(self):
        self.assertEqual(text_diagnostics("Food was bland, bland, BLAND.")["consecutive_repeated_word_pairs"], 2)
        self.assertFalse(text_diagnostics("Food was bland but service was helpful.")["consecutive_repeated_word"])

    def test_zero_visual_is_required_not_only_allowed(self):
        prefix = "[REVIEW_CONSENSUS]\nReviews are mixed.\n\n[VISUAL_EVIDENCE]\n"
        self.assertTrue(structure(prefix, "filtered", 0)["empty_visual_compliant"])
        result = structure(prefix + "A bagel is on a tray.", "filtered", 0)
        self.assertTrue(result["section_headers_compliant"])
        self.assertFalse(result["empty_visual_compliant"])
        self.assertTrue(result["visual_text_without_caption_input"])

    def test_wrong_or_missing_header_is_not_accepted(self):
        self.assertFalse(structure("[VISUAL_CONTEXT]\nReviews are mixed.", "unfiltered", 5)["section_headers_compliant"])
        self.assertFalse(structure("[REVIEW_CONSENSUS]\nOnly reviews.", "filtered", 0)["empty_visual_compliant"])

    def test_missing_execute_flag_blocks_before_loading(self):
        with self.assertRaisesRegex(ValueError, "requires --execute"):
            approval_gate(Path("unused"), None, "greedy_A", False)

    def test_approval_scope_and_bindings(self):
        with tempfile.TemporaryDirectory(prefix="stage_i_gate_") as directory:
            root = Path(directory)
            for name in ("subset_manifest.json", "decoding_configurations.json"):
                (root / name).write_text("{}", encoding="utf-8")
            approval = {"approved": False, "cost_reviewed": True, "approved_experiments": ["greedy_A"],
                        "subset_manifest_sha256": sha(root / "subset_manifest.json"),
                        "decoding_configurations_sha256": sha(root / "decoding_configurations.json")}
            path = root / "approval.json"
            path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not approved"):
                approval_gate(root, path, "greedy_A", True)
            approval["approved"] = True
            path.write_text(json.dumps(approval), encoding="utf-8")
            self.assertTrue(approval_gate(root, path, "greedy_A", True)["approved"])
            with self.assertRaisesRegex(ValueError, "not approved"):
                approval_gate(root, path, "beam_ngram_B", True)
            approval["approved_experiments"].append("beam_ngram_B")
            path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "after A review"):
                approval_gate(root, path, "beam_ngram_B", True)
            (root / "subset_manifest.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another subset"):
                approval_gate(root, path, "greedy_A", True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
