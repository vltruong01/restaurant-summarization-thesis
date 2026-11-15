"""Offline bindings and authorization for one proposed 24-case F decoding run.

Import does not load Torch, a model, or a GPU. No data or targets are generated.
"""
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_i_repetition_guard_pilot_v1.json"
PREP = ROOT / "outputs/stage_i/repetition_guard_preparation_v1"
RUNNER = ROOT / "scripts/run_stage_i_repetition_guard_pilot_v1.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write((value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + "\n")


def preflight(require_fresh=True):
    config, manifest = read(CONFIG), read(PREP / "PREFLIGHT_MANIFEST.json")
    require(config["GPU_execution_approved"] is False and config["training_authorized"] is False,
            "Prepared config is not a researcher authorization")
    require(config["scope"] == {"condition": "filtered", "businesses": 24, "random": 20, "targeted": 4,
                                "generation_calls_max": 24, "training_updates": 0}, "Scope changed")
    for rel, digest in manifest["files"].items():
        path = Path(rel)
        require(sha(path if path.is_absolute() else ROOT / path) == digest, "Integrity mismatch: " + rel)
    inherited = read(ROOT / config["historical_integrity_manifest"])
    for rel, digest in inherited["files"].items():
        path = Path(rel)
        require(sha(path if path.is_absolute() else ROOT / path) == digest, "Frozen historical artifact changed: " + rel)
    for name, version in config["runtime_pins"].items():
        require(importlib.metadata.version(name) == version, "Runtime drift: " + name)
    g = read(ROOT / config["expected_generation_config"])
    require(g["num_beams"] == 4 and not g["do_sample"] and g["max_length"] == 768 and g["max_new_tokens"] is None,
            "Wrong beam or length semantics")
    require(g["renormalize_logits"] is False and g["watermarking_config"] is None and g["no_repeat_ngram_size"] == 0,
            "Unapproved extra score processor")
    settings = read(ROOT / "outputs/stage_i/decoding_diagnostic_v1/decoding_configurations.json")
    require(config["generation"] == settings["experiments"]["baseline"]["generation_kwargs"], "Baseline decoding kwargs changed")
    cases = rows(ROOT / config["cases"])
    manifest_ids = read(ROOT / "outputs/stage_i/decoding_diagnostic_v1/subset_manifest.json")["execution_business_ids"]
    original = [r for r in rows(ROOT / "outputs/stage_i/decoding_diagnostic_v1/cases.jsonl") if r["condition"] == "filtered"]
    require(cases == original and len(cases) == len({r["business_id"] for r in cases}) == 24, "Fixed F cases changed")
    require([r["business_id"] for r in cases] == manifest_ids and all(r["split"] == "validation" for r in cases),
            "Order/split changed")
    require(sum(r["subset"] == "random" for r in cases) == 20 and sum(r["subset"] == "targeted" for r in cases) == 4,
            "Sampling stratum changed")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / config["tokenizer_directory"] / "tokenizer.json"))
    for r in cases:
        require(hashlib.sha256(r["source"].encode()).hexdigest() == r["source_sha256"], "Source hash differs")
        require(tokenizer.encode(r["source"]).ids == r["input_ids"], "Source tokens differ")
        require(tokenizer.encode(r["target"]).ids == r["label_ids"], "Target tokens differ")
        require(0 < len(r["input_ids"]) <= 5000 and 0 < len(r["label_ids"]) <= 768, "Lossless token budget violated")
        require(r["label_ids"][0] == 0 and r["label_ids"][-1] == 2, "BOS/EOS label mismatch")
        require(tokenizer.decode(r["baseline_prediction"]["token_ids"], skip_special_tokens=True) == r["baseline_prediction"]["text"],
                "Saved baseline text/tokens differ")
        if not r["selected_photo_ids"]:
            require(r["source"].endswith("\n\n[VISUAL_CONTEXT]\n") and r["target"].endswith("\n\n[VISUAL_EVIDENCE]\n"),
                    "Zero-caption source/target changed")
    require(sum(not r["selected_photo_ids"] for r in cases) == 1, "Expected fixed single zero-caption case")
    if require_fresh:
        require(not (ROOT / config["run_root"]).exists(), "Run namespace already exists; no retry/resume/overwrite")
    return config, manifest, cases, {"status": "OFFLINE_PASS_NO_MODEL_OR_GPU_EXECUTION", "file_bindings": len(manifest["files"]),
                                   "preserved_historical_file_bindings": len(inherited["files"]), "cases": 24,
                                   "source_and_label_roundtrip": True, "random": 20, "targeted": 4,
                                   "zero_caption": 1, "D051_strict_replay": "FAIL_UNCHANGED"}


def deadline():
    limit = datetime.fromisoformat("2026-10-22T07:11:22.846767+00:00")
    require(datetime.now(timezone.utc) < limit, "Original D060 viability deadline expired; stop")


def authorization(path):
    require(path is not None and Path(path).is_file(), "New hash-bound researcher approval required before any model/GPU operation")
    a = read(path)
    require(a.get("experiment") == "repetition_guard_pilot_v1" and a.get("GPU_execution_approved") is True,
            "This is not the new guard execution approval")
    require(a.get("approval_source") == "researcher_message" and a.get("researcher_explicit_authorization"), "Explicit approval source absent")
    stamp = datetime.fromisoformat(a["execution_authorized_at_utc"])
    require(stamp.utcoffset().total_seconds() == 0 and stamp <= datetime.now(timezone.utc), "Approval UTC timestamp invalid")
    for key, path in (("config_sha256", CONFIG), ("preflight_manifest_sha256", PREP / "PREFLIGHT_MANIFEST.json"), ("runner_sha256", RUNNER)):
        require(a.get(key) == sha(path), "Approval binding mismatch: " + key)
    require(a.get("generation_calls_max") == 24 and a.get("condition") == "filtered", "Approval scope mismatch")
    for key in ("training_authorized", "Teacher_calls_authorized", "held_out_access", "automatic_retry", "fallback_authorized", "RO_U_Shuffled_authorized"):
        require(a.get(key) is False, "Forbidden scope or missing denial: " + key)
    require(a.get("viability_deadline_utc") == "2026-10-22T07:11:22.846767+00:00", "Deadline extended")
    deadline()
    return a
