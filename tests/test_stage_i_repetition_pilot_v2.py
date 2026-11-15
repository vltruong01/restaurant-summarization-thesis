"""Offline v2 authorization/reuse boundary tests; no Torch/model/GPU/API.

Temporary fixtures are synthetic, not researcher annotations or predictions.
The historical preflight is mocked to isolate new v2 boundary behavior.
"""
import ast
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
pilot = importlib.import_module("stage_i_repetition_pilot_v2")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.prep = self.root / "prep"
        self.runner = self.root / "runner.py"
        self.approval = self.root / "approval.json"
        put(self.config, {"synthetic": True})
        put(self.prep / "PREFLIGHT_MANIFEST.json", {"files": {}})
        self.runner.write_text("# Synthetic test-only runner fixture\n", encoding="utf-8")
        for key, value in (("CONFIG", self.config), ("PREP", self.prep), ("RUNNER", self.runner)):
            p = patch.object(pilot, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.deadline = patch.object(pilot, "deadline")
        self.deadline_mock = self.deadline.start()
        self.addCleanup(self.deadline.stop)
        self.a = {"experiment": "repetition_guard_pilot_v2", "GPU_execution_approved": True,
            "approval_source": "researcher_message", "researcher_explicit_authorization": "Synthetic unit-test authorization only",
            "config_sha256": digest(self.config), "preflight_manifest_sha256": digest(self.prep / "PREFLIGHT_MANIFEST.json"),
            "runner_sha256": digest(self.runner), "new_generation_calls_max": 19, "reused_outputs": 5,
            "retry_failed_case6_once": True, "continue_expected_terminal_infeasibility": True,
            "viability_deadline_utc": "2026-10-22T07:11:22.846767+00:00",
            "execution_authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            **{k: False for k in ("training_authorized", "Teacher_calls_authorized", "held_out_access",
                                  "RO_U_Shuffled_authorized", "automatic_retry", "fallback_authorized", "relax_hard_rules")}}

    def invoke(self, changes=None, remove=None):
        a = {**self.a, **(changes or {})}
        if remove:
            a.pop(remove)
        put(self.approval, a)
        return pilot.authorization(self.approval)

    def test_actual_authorization_accepts_exact_nineteen_plus_five(self):
        self.assertEqual(self.invoke(), self.a)
        self.deadline_mock.assert_called_once_with()

    def test_missing_authorization_rejected_before_bindings(self):
        with self.assertRaisesRegex(ValueError, "authorization missing"):
            pilot.authorization(None)

    def test_changed_runner_cannot_use_old_approval(self):
        put(self.approval, self.a)
        self.runner.write_text("# altered synthetic fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch: runner_sha256"):
            pilot.authorization(self.approval)

    def test_scope_and_fail_closed_authorization_negative_controls(self):
        cases = [{"experiment": "repetition_guard_pilot_v1"}, {"GPU_execution_approved": False},
                 {"new_generation_calls_max": 24}, {"new_generation_calls_max": 20}, {"reused_outputs": 4},
                 {"retry_failed_case6_once": False}, {"continue_expected_terminal_infeasibility": False},
                 {"relax_hard_rules": True}, {"automatic_retry": True}, {"fallback_authorized": True},
                 {"training_authorized": True}, {"Teacher_calls_authorized": True}, {"held_out_access": True},
                 {"RO_U_Shuffled_authorized": True}, {"approval_source": "model_message"},
                 {"viability_deadline_utc": "2026-10-25T15:10:48.941817+00:00"},
                 {"execution_authorized_at_utc": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.invoke(change)

    def test_missing_denial_is_not_approval(self):
        for key in ("automatic_retry", "relax_hard_rules", "Teacher_calls_authorized"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.invoke(remove=key)


class ReuseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = json.loads((PROJECT / "configs/stage_i_repetition_guard_pilot_v1.json").read_text(encoding="utf-8"))
        self.c = {**copy.deepcopy(self.old), "experiment": "repetition_guard_pilot_v2", "run_root": "new_run",
                  "new_generation_calls_max": 19, "reuse_count": 5, "terminal_repair_prefix_tokens": 767}
        self.config = self.root / "config.json"
        self.prep = self.root / "prep"
        self.old_run = self.root / "old_run"
        self.audit = self.root / "outputs/stage_i/repetition_guard_independent_audit_v1/SOURCE_ARTIFACT_SHA256.json"
        self.cases = [{"business_id": "SYNTHETIC_CASE_" + str(i), "condition": "filtered", "subset": "random" if i < 20 else "targeted",
                       "source_sha256": hashlib.sha256(("synthetic source " + str(i)).encode()).hexdigest(),
                       "source": "synthetic source " + str(i)} for i in range(24)]
        self.bound = {}
        for i, case in enumerate(self.cases[:5]):
            p = self.old_run / "raw_cases" / (case["business_id"] + ".json")
            a = self.old_run / "analysis" / p.name
            text = "[REVIEW_CONSENSUS]\nA synthetic opinion.\n\n[VISUAL_EVIDENCE]\n"
            put(p, {"business_id": case["business_id"], "condition": "filtered", "subset": "random",
                    "source_sha256": case["source_sha256"], "token_ids": [2, 0, 100 + i, 2], "text": text})
            put(a, {"violations": [], "flags": [], "source_reduplication_exceptions": []})
            self.bound[p.relative_to(self.root).as_posix()] = digest(p)
            self.bound[a.relative_to(self.root).as_posix()] = digest(a)
        put(self.audit, self.bound)
        put(self.config, self.c)
        put(self.prep / "PREFLIGHT_MANIFEST.json", {"files": {}})
        for key, value in (("ROOT", self.root), ("CONFIG", self.config), ("PREP", self.prep), ("OLD_RUN", self.old_run)):
            p = patch.object(pilot, key, value)
            p.start()
            self.addCleanup(p.stop)
        original = patch.object(pilot, "original_preflight", return_value=(self.old, {}, self.cases, {"fixture_only": True}))
        original.start()
        self.addCleanup(original.stop)
        tokenizer_module = types.ModuleType("tokenizers")
        class FakeTokenizer:
            @staticmethod
            def from_file(path):
                return FakeTokenizer()
            def decode(self, ids, skip_special_tokens=True):
                return "[REVIEW_CONSENSUS]\nA synthetic opinion.\n\n[VISUAL_EVIDENCE]\n"
        tokenizer_module.Tokenizer = FakeTokenizer
        tp = patch.dict(sys.modules, {"tokenizers": tokenizer_module})
        tp.start()
        self.addCleanup(tp.stop)

    def refresh_binding(self, p):
        self.bound[p.relative_to(self.root).as_posix()] = digest(p)
        put(self.audit, self.bound)

    def test_five_reused_outputs_are_returned_without_rewriting(self):
        before = {rel: digest(self.root / rel) for rel in self.bound}
        c, manifest, cases, reused, proof = pilot.preflight()
        self.assertEqual([p["business_id"] for p in reused], [c["business_id"] for c in cases[:5]])
        self.assertEqual(proof["remaining_attempts_max"], 19)
        self.assertEqual(before, {rel: digest(self.root / rel) for rel in self.bound})

    def test_mutated_original_raw_bytes_are_rejected(self):
        p = next((self.old_run / "raw_cases").glob("*.json"))
        p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audited artifact changed"):
            pilot.preflight()

    def test_even_rebound_wrong_business_is_rejected(self):
        p = self.old_run / "raw_cases" / "SYNTHETIC_CASE_0.json"
        x = json.loads(p.read_text(encoding="utf-8"))
        x["business_id"] = "SYNTHETIC_WRONG_BUSINESS"
        put(p, x)
        self.refresh_binding(p)
        with self.assertRaisesRegex(ValueError, "reuse mapping differs"):
            pilot.preflight()

    def test_bound_long_reused_output_rejected_at_terminal_boundary(self):
        p = self.old_run / "raw_cases" / "SYNTHETIC_CASE_0.json"
        x = json.loads(p.read_text(encoding="utf-8"))
        x["token_ids"] = [2] + [100] * 766
        put(p, x)
        self.refresh_binding(p)
        with self.assertRaisesRegex(ValueError, "Terminal repair could affect"):
            pilot.preflight()

    def test_scientific_generation_change_rejected(self):
        self.c["generation"]["num_beams"] = 1
        put(self.config, self.c)
        with self.assertRaisesRegex(ValueError, "Scientific setting changed: generation"):
            pilot.preflight()

    def test_second_attempt_namespace_rejected(self):
        (self.root / self.c["run_root"]).mkdir()
        with self.assertRaisesRegex(ValueError, "namespace exists"):
            pilot.preflight()
        self.assertEqual(len(pilot.preflight(require_fresh=False)[3]), 5)


class RunnerBoundaryTests(unittest.TestCase):
    def test_model_imports_follow_actual_authorization_and_preflight(self):
        path = PROJECT / "scripts/run_stage_i_repetition_guard_pilot_v2.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        auth = next(n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "authorization")
        preflight = next(n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "preflight")
        imports = [n.lineno for n in ast.walk(execute) if isinstance(n, (ast.Import, ast.ImportFrom))
                   and (getattr(n, "module", "") or "").split(".")[0] in {"transformers", "peft", "safetensors"}]
        self.assertTrue(imports)
        self.assertLess(auth, min(imports))
        self.assertLess(preflight, min(imports))

    def test_only_named_known_constraint_handler_continues(self):
        tree = ast.parse((PROJECT / "scripts/run_stage_i_repetition_guard_pilot_v2.py").read_text(encoding="utf-8"))
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        handlers = [n for n in ast.walk(execute) if isinstance(n, ast.ExceptHandler)]
        known = [n for n in handlers if isinstance(n.type, ast.Name) and n.type.id == "NoCompliantCompletion"]
        self.assertEqual(len(known), 1)
        self.assertTrue(any(isinstance(n, ast.Continue) for n in ast.walk(known[0])))
        # Logging helpers may catch an unavailable memory reading. The actual
        # execution-wide error handler must still raise, and only the named
        # case-infeasibility handler may continue the generation loop.
        continuing = [n for n in handlers if any(isinstance(x, ast.Continue) for x in ast.walk(n))]
        self.assertEqual(continuing, known)
        execution_try = [n for n in execute.body if isinstance(n, ast.Try)]
        self.assertEqual(len(execution_try), 1)
        unexpected = [n for n in execution_try[0].handlers if isinstance(n.type, ast.Name) and n.type.id == "Exception"]
        self.assertEqual(len(unexpected), 1)
        self.assertFalse(any(isinstance(n, ast.Continue) for n in ast.walk(unexpected[0])))
        self.assertTrue(any(isinstance(n, ast.Raise) for n in ast.walk(unexpected[0])))


if __name__ == "__main__":
    unittest.main()
