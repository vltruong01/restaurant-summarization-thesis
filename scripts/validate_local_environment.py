"""Stage A1 checks: CPU preprocessing only; never select experimental data."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

# Tokenizer-only environment: no cloud/model runtime dependency in A1.
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root(value: str, override: str) -> Path:
    path = Path(os.environ.get(override, value))
    return (path if path.is_absolute() else REPO / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/project.yaml")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    report: dict = {
        "stage": "A1",
        "a2_status": "pending",
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "working_directory": str(Path.cwd()),
            "device": "cpu",
            "torch_cuda_runtime": None,
            "torch_cuda_runtime_reason": "Torch/model execution is outside A1.",
        },
        "checks": {},
        "signature_scope": "Synthetic inputs and tokenizer assets, not cohort selection.",
        "raw_check_scope": "Presence and first-record schema/hash only; not full corpus validation.",
    }

    def check(name, action):
        try:
            details = action()
            report["checks"][name] = {"status": "PASS", "details": details}
            return details
        except Exception as error:
            report["checks"][name] = {
                "status": "FAIL", "error": f"{type(error).__name__}: {error}"
            }
            return None

    def load_config():
        import yaml
        config_path = args.config if args.config.is_absolute() else REPO / args.config
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["schema_version"] != 1 or config["reproducibility"]["seed"] != 42:
            raise ValueError("Unexpected config schema/seed; review the spec first.")
        if config["study"]["split_counts"] != {"train": 2000, "validation": 200, "test": 100}:
            raise ValueError("Planned split differs from the approved 2,000/200/100 design.")
        if config["tokenizer"]["model_id"] != "allenai/led-base-16384" or config["tokenizer"]["revision"] != "38335783885b338d93791936c54bb4be46bebed9":
            raise ValueError("Tokenizer identity differs from the Stage A1 spec.")
        return config

    config = check("configuration", load_config)
    output_root = resolve_root("outputs", "THESIS_OUTPUT_ROOT")
    deterministic = {}
    if config is not None:
        output_root = resolve_root(config["paths"]["output_root"], "THESIS_OUTPUT_ROOT")
        data_root = resolve_root(config["paths"]["data_root"], "THESIS_DATA_ROOT")

        def dependencies():
            imports = {
                "numpy": "numpy", "pandas": "pandas", "orjson": "orjson",
                "langdetect": "langdetect", "transformers": "transformers",
                "tokenizers": "tokenizers", "huggingface-hub": "huggingface_hub",
                "PyYAML": "yaml", "tqdm": "tqdm", "Pillow": "PIL", "spacy": "spacy",
            }
            versions = {}
            for package, module in imports.items():
                importlib.import_module(module)
                versions[package] = importlib.metadata.version(package)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            lock = REPO / "requirements/preprocessing.lock.txt"
            if not lock.is_file():
                raise FileNotFoundError("Resolve and save the dependency lock before validation.")
            frozen = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, check=True,
            ).stdout
            if set(frozen.splitlines()) != set(lock.read_text(encoding="utf-8").splitlines()):
                raise ValueError("Installed packages differ from the preprocessing lock.")
            return {"versions": versions, "pip_check": result.stdout.strip(),
                    "lock_sha256": sha256_file(lock)}

        check("dependency_imports_and_consistency", dependencies)

        def paths_and_sources():
            required = {
                "business_jsonl": ["business_id", "categories"],
                "review_jsonl": ["review_id", "business_id", "text", "stars", "date"],
                "photo_jsonl": ["photo_id", "business_id", "label"],
            }
            details = {}
            for key, fields in required.items():
                path = data_root / config["paths"][key]
                before = path.stat()
                with path.open("rb") as stream:
                    first_line = next(line for line in stream if line.strip())
                obj = json.loads(first_line)
                missing = set(fields) - obj.keys()
                if missing:
                    raise ValueError(f"{key} missing fields: {sorted(missing)}")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while reading {key}.")
                details[key] = {
                    "relative_path": config["paths"][key], "bytes": before.st_size,
                    "first_record_sha256": hashlib.sha256(first_line).hexdigest(),
                    "required_fields": fields,
                }
            photo_dir = data_root / config["paths"]["photo_directory"]
            if not photo_dir.is_dir():
                raise FileNotFoundError(photo_dir)
            details["photo_directory"] = config["paths"]["photo_directory"]
            deterministic["raw_first_record_hashes"] = {
                key: details[key]["first_record_sha256"] for key in required
            }
            return details

        check("portable_paths_and_raw_schema_samples", paths_and_sources)

        def synthetic_preprocessing():
            import numpy as np
            import orjson
            import spacy
            from langdetect import DetectorFactory, detect

            seed = config["reproducibility"]["seed"]
            DetectorFactory.seed = seed
            texts = {
                "en": "The restaurant serves fresh seafood and generous portions. "
                      "Several customers praise the dishes, while others report slow service.",
                "es": "El restaurante sirve comida deliciosa y las porciones son generosas. "
                      "Los clientes destacan la calidad de los platos y el ambiente agradable.",
            }
            languages = {language: [detect(text) for _ in range(2)]
                         for language, text in texts.items()}
            if any(results != [expected, expected] for expected, results in languages.items()):
                raise ValueError(f"Unexpected/repeated language output: {languages}")
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            sentences = [sentence.text for sentence in nlp(texts["en"]).sents]
            if len(sentences) != 2:
                raise ValueError("Synthetic sentence boundary check failed.")
            normalized = " ".join("  Fresh\n seafood\t and generous portions.  ".split())
            if normalized != "Fresh seafood and generous portions.":
                raise ValueError("Synthetic normalization check failed.")
            records = [{"id": 1, "text": texts["en"]}, {"id": 2, "text": texts["es"]}]
            with tempfile.TemporaryDirectory(prefix="thesis-a1-") as directory:
                fixture = Path(directory) / "synthetic.jsonl"
                fixture.write_bytes(b"\n".join(orjson.dumps(row) for row in records) + b"\n")
                with fixture.open("rb") as stream:
                    restored = [orjson.loads(line) for line in stream if line.strip()]
            if restored != records:
                raise ValueError("JSONL round-trip changed synthetic records.")
            rng = random.Random(seed)
            signature = {
                "languages": languages, "sentences": sentences, "normalized": normalized,
                "python_rng": [rng.randrange(1000000) for _ in range(10)],
                "numpy_rng": np.random.default_rng(seed).integers(0, 1000000, 10).tolist(),
            }
            deterministic["synthetic_preprocessing"] = signature
            return signature

        check("synthetic_language_segmentation_jsonl_rng", synthetic_preprocessing)

        def tokenizer_export():
            from transformers import AutoTokenizer
            token_config = config["tokenizer"]
            source = AutoTokenizer.from_pretrained(
                token_config["model_id"], revision=token_config["revision"],
                local_files_only=True,
            )
            assets = REPO / token_config["assets_directory"]
            assets.mkdir(parents=True, exist_ok=True)
            source.save_pretrained(assets)
            exported = AutoTokenizer.from_pretrained(assets, local_files_only=True)
            texts = [
                "[BUSINESS_INFORMATION]\nRestaurants\n[REVIEWS]\n[REV_1] Fresh seafood.\n"
                "[REV_2] Slow service.\n[VISUAL_CONTEXT]\n[IMG_CAP_1] Outdoor tables.",
                "A café serves generous portions, but opinions differ about service.",
            ]
            token_ids = []
            for text in texts:
                kwargs = {"add_special_tokens": token_config["add_special_tokens"],
                          "truncation": False}
                original_ids = source.encode(text, **kwargs)
                exported_ids = exported.encode(text, **kwargs)
                if original_ids != exported_ids or not original_ids:
                    raise ValueError("Cached/exported tokenizer IDs differ.")
                token_ids.append(original_ids)
            hashes = {path.name: sha256_file(path) for path in sorted(assets.iterdir())
                      if path.is_file()}
            manifest = {"model_id": token_config["model_id"], "revision": token_config["revision"],
                        "relative_directory": token_config["assets_directory"],
                        "sha256": hashes}
            manifest_path = assets.parent / "led-base-16384.manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            for filename, expected in hashes.items():
                if sha256_file(assets / filename) != expected:
                    raise RuntimeError("Tokenizer export hash verification failed.")
            deterministic["tokenizer"] = {"manifest": manifest, "sample_token_ids": token_ids}
            return {"manifest": manifest, "sample_lengths": [len(ids) for ids in token_ids]}

        check("pinned_tokenizer_export_and_equivalence", tokenizer_export)

        def pending_research_parameters():
            import yaml
            pending = yaml.safe_load((REPO / "configs/research_pending.yaml").read_text(encoding="utf-8"))
            expected_null = [pending["token_budget"]["review_tokens"],
                             pending["caption_filtering"]["maximum_k"],
                             pending["teacher"]["model_and_version"],
                             pending["external_generalization"]["dataset"],
                             pending["human_evaluation"]["stratification_and_allocation_rule"]]
            if any(value is not None for value in expected_null):
                raise ValueError("Research choices unexpectedly populated during A1.")
            return {"unresolved_choices_preserved": True}

        check("research_placeholders", pending_research_parameters)

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        report["runtime"]["gpu_inventory"] = gpu.stdout.strip() if gpu.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        report["runtime"]["gpu_inventory"] = None
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(deterministic, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    report["deterministic_payload"] = deterministic
    failed = [name for name, result in report["checks"].items() if result["status"] != "PASS"]
    report["status"] = "FAIL" if failed else "PASS"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    destination = args.report or output_root / "environment/local_a1_report.json"
    if not destination.is_absolute():
        destination = REPO / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(destination),
                      "failed_checks": failed, "elapsed_seconds": report["elapsed_seconds"],
                      "deterministic_signature": report["deterministic_signature"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
