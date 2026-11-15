"""Run the original 83 D-053 offline tests wholly from an extracted review bundle.

This orchestration script never edits tests, labels, references, accepted inputs,
or model artifacts. Third-party Python dependencies may come from the selected
interpreter; project modules and test fixtures must come from --root only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REQUIRED = (
    "code/test_stage_i_d053_annotations.py",
    "code/test_stage_i_d053_evaluation.py",
    "code/test_stage_i_d053_ui.js",
    "code/stage_i_d053_annotations.py",
    "code/stage_i_d053_evaluation.py",
    "code/stage_i_d053_annotation_ui.html",
    "code/stage_i_decoding_diagnostics.py",
    "code/stage_i_paired_pilot.py",
    "schemas/stage_i_d053_annotation_v1.schema.json",
    "SOURCE_REFERENCE_ANNOTATION.html",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Extracted bundle root, containing code/ and schemas/.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON report; raw logs use the same filename stem.")
    parser.add_argument("--forbid-original-root", type=Path,
                        help="Optional original workspace to exclude from extraction.")
    parser.add_argument("--portable-readiness-script", type=Path,
                        help="Optional standalone CPU checker, relative to bundle root.")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if not root.is_dir():
        parser.error("--root must be an extracted directory")
    if args.forbid_original_root:
        original = args.forbid_original_root.resolve(strict=True)
        if root == original or root.is_relative_to(original):
            parser.error("Extraction must be outside the original workspace")
    if output.exists():
        parser.error("Refusing to overwrite an existing report")
    output.parent.mkdir(parents=True, exist_ok=True)
    bound_files = {}
    for relative in REQUIRED:
        path = (root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            parser.error("Missing or externally linked bundle file: " + relative)
        bound_files[relative] = sha(path)
    if args.portable_readiness_script:
        checker = (root / args.portable_readiness_script).resolve(strict=True)
        if not checker.is_file() or not checker.is_relative_to(root):
            parser.error("Portable readiness checker must be inside extraction")
    else:
        checker = None
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "NODE_OPTIONS", "NODE_PATH",
                "NODE_V8_COVERAGE", "D053_SYNTHETIC_FIXTURE_OUT", "D053_UI_EXPORTED_FIXTURE_OUT",
                "D053_UI_TRANSITION_FIXTURE_OUT"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    report = {
        "schema_version": "D053_EXTRACTED_OFFLINE_TESTS_v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "FAIL", "extracted_root": str(root), "cwd": str(root),
        "python_executable": sys.executable,
        "environment_isolation": {
            "PYTHONPATH_cleared": True, "PYTHONHOME_cleared": True,
            "user_site_disabled": True, "bytecode_writes_disabled": True,
            "HF_offline": True, "CUDA_VISIBLE_DEVICES": "",
            "original_workspace_excluded": args.forbid_original_root is not None,
        },
        "original_test_and_fixture_sha256": bound_files,
        "tests": [], "total_tests": 0, "skipped": 0,
        "human_annotations_created": 0, "actual_reference_units_created": 0,
        "student_predictions_created": 0, "GPU_model_or_API_execution": False,
        "browser_smoke_test": "SEPARATE_NOT_ESTABLISHED_BY_NODE_TESTS",
    }

    def command(name, argv):
        stdout = output.parent / (output.stem + "." + name + ".stdout.log")
        stderr = output.parent / (output.stem + "." + name + ".stderr.log")
        # Reserve both output names before executing any command.
        with stdout.open("x", encoding="utf-8", newline="\n") as out_stream:
            with stderr.open("x", encoding="utf-8", newline="\n") as err_stream:
                started = dt.datetime.now(dt.timezone.utc)
                proc = subprocess.run(argv, cwd=root, env=env, stdout=out_stream,
                                      stderr=err_stream, check=False)
        return {"name": name, "command": [str(x) for x in argv],
                "cwd": str(root), "returncode": proc.returncode,
                "elapsed_seconds": (dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
                "stdout_log": str(stdout), "stderr_log": str(stderr),
                "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr)}, \
            stdout.read_text(encoding="utf-8"), stderr.read_text(encoding="utf-8")

    try:
        probe = r'''
import importlib, importlib.metadata as m, json, pathlib, sys
root = pathlib.Path.cwd().resolve()
sys.path.insert(0, str(root / 'code'))
origins = {}
for name in ('stage_i_d053_annotations', 'stage_i_d053_evaluation', 'stage_i_decoding_diagnostics'):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError('Project module resolved outside extraction: ' + name)
    origins[name] = str(path)
    if hasattr(module, 'ROOT') and pathlib.Path(module.ROOT).resolve() != root:
        raise RuntimeError('Module ROOT differs from extraction: ' + name)
if any(name == 'torch' or name.startswith('torch.') or name == 'transformers' or name.startswith('transformers.') for name in sys.modules):
    raise RuntimeError('Offline dependency probe imported model frameworks')
versions = {name: m.version(name) for name in ('numpy', 'jsonschema')}
if versions['numpy'] != '1.26.4':
    raise RuntimeError('Frozen bootstrap requires NumPy 1.26.4')
print(json.dumps({'status': 'PASS', 'root': str(root), 'project_module_origins': origins,
                  'dependency_versions': versions, 'model_frameworks_imported': False}, indent=2))
'''
        result, stdout, stderr = command("dependency_probe", [sys.executable, "-s", "-X", "utf8", "-c", probe])
        report["dependency_probe"] = result
        if result["returncode"]:
            raise RuntimeError("Extracted dependency/path probe failed; see raw logs")
        report["dependency_and_module_origins"] = json.loads(stdout)
        node = shutil.which("node", path=env.get("PATH"))
        if not node:
            raise RuntimeError("Node.js is required for all 27 original UI tests")
        version, stdout, stderr = command("node_version", [node, "--version"])
        if version["returncode"]:
            raise RuntimeError("Unable to inspect Node.js version")
        report["node"] = {**version, "version": stdout.strip()}
        for name, pattern, expected in (
                ("annotations", "test_stage_i_d053_annotations.py", 22),
                ("evaluation", "test_stage_i_d053_evaluation.py", 34)):
            result, stdout, stderr = command(name, [sys.executable, "-s", "-X", "utf8", "-m", "unittest",
                                                   "discover", "-s", "code", "-p", pattern, "-v"])
            combined = stdout + "\n" + stderr
            counts = re.findall(r"^Ran (\d+) tests? in ", combined, flags=re.MULTILINE)
            skipped = re.findall(r"\bskipped(?:=\d+| ['\"])", combined, flags=re.IGNORECASE)
            skipped_total = re.search(r"\bskipped=(\d+)", combined)
            if skipped_total:
                report["skipped"] += int(skipped_total.group(1))
            result.update(expected_tests=expected,
                          actual_tests=int(counts[-1]) if counts else None,
                          skipped_detected=bool(skipped),
                          status="PASS" if result["returncode"] == 0 and counts
                          and int(counts[-1]) == expected and not skipped
                          and re.search(r"^OK\s*$", combined, flags=re.MULTILINE) else "FAIL")
            report["tests"].append(result)
            report["total_tests"] += result["actual_tests"] or 0
            if result["status"] != "PASS":
                raise RuntimeError(name + " tests did not pass with exact count and zero skips")
        result, stdout, stderr = command("ui", [node, "code/test_stage_i_d053_ui.js", "SOURCE_REFERENCE_ANNOTATION.html"])
        records = []
        for line in stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "tests" in value:
                records.append(value)
        ui = records[-1] if records else {}
        result.update(expected_tests=27, actual_tests=ui.get("tests"), skipped_detected=False,
                      status="PASS" if result["returncode"] == 0 and ui.get("status") == "PASS"
                      and ui.get("tests") == 27 and ui.get("synthetic_only") is True
                      and ui.get("source_files_modified") is False else "FAIL")
        report["tests"].append(result)
        report["total_tests"] += result["actual_tests"] or 0
        if result["status"] != "PASS":
            raise RuntimeError("Original Node UI tests did not pass with exact count27")
        if report["total_tests"] != 83:
            raise RuntimeError("Expected exactly 83 original offline tests")
        for relative, digest in bound_files.items():
            if sha(root / relative) != digest:
                raise RuntimeError("Original test/fixture changed during execution: " + relative)
        report["original_test_and_fixture_preservation"] = "PASS"
        if checker:
            result, stdout, stderr = command("portable_readiness", [sys.executable, "-s", "-X", "utf8", str(checker), "--readiness"])
            report["portable_readiness"] = result
            if result["returncode"]:
                raise RuntimeError("Optional standalone readiness checker failed")
        report["status"] = "PASS"
        report["error"] = None
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    write_new(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "total_tests": report["total_tests"],
                      "skipped": report["skipped"], "report": str(output), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
