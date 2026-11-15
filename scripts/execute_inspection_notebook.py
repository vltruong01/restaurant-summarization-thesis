"""Execute read-only inspection notebooks with an existing local kernel."""

import argparse
import json
from pathlib import Path
from queue import Empty
import sys
import time

from jupyter_client import KernelManager
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, default=Path('notebooks/01_yelp_data_inspection.ipynb'))
    parser.add_argument('--report', type=Path, default=Path('outputs/inspection/notebook_execution_report.json'))
    parser.add_argument('--mode', choices=['quick', 'cohort_full_scan', 'english_eligibility', 'review_diagnostics', 'review_selection_pilot', 'budget_review_pilot', 'expanded_budget_diagnostics', 'selected_review_length_audit', 'unified_dataset_review'], default='quick')
    parser.add_argument('--cell-timeout-seconds', type=int, default=180)
    parser.add_argument('--read-only-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(set(identifiers)) == len(identifiers)
    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from data_inspection import resolve_data_root
    data_root = resolve_data_root(ROOT, config)
    raw_paths = sorted(set(list(data_root.glob("yelp_academic_dataset_*.json")) + [
        data_root / config["paths"]["business_jsonl"], data_root / config["paths"]["review_jsonl"],
        data_root / config["paths"]["photo_jsonl"], data_root / config["paths"]["photo_directory"]]))
    missing_raw_paths = [p for p in raw_paths if not p.exists()]
    archived_raw_allowed = False
    if missing_raw_paths:
        cached_main = args.mode == 'unified_dataset_review' and path.resolve() == (
            ROOT / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb').resolve()
        backup = ROOT / 'outputs/hf_dataset_backup_20261008/remote_verification.json'
        proof = json.loads(backup.read_text(encoding='utf-8')) if backup.is_file() else {}
        archived_raw_allowed = cached_main and proof.get('status') == 'PASS_FULL_DATA_REMOTE_RESTORE_HASH_VERIFIED'
        archived_raw_allowed = archived_raw_allowed and proof.get('private') is True and proof.get('source_files_verified') == 230173
        if not archived_raw_allowed:
            raise FileNotFoundError('Raw inputs missing: restore the verified HF dataset before raw inspection/rebuilding.')
        for relative in ['data/processed/yelp_2300_v1', 'data/processed/yelp_2300_english_v1', 'data/processed/splits']:
            if not (ROOT / relative).is_dir():
                raise FileNotFoundError('Required processed cohort missing: ' + relative)
    protected_roots = [(p if p.is_absolute() else ROOT / p).resolve() for p in args.read_only_root]

    def snapshot():
        protected = set(raw_paths)
        for directory in protected_roots:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            protected.add(directory)
            protected.update(directory.rglob('*'))
        # Preserve absence as a sentinel, so deletion/creation during a run is detected too.
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in protected}

    before = snapshot()
    report = {"mode": args.mode, "full_raw_audit_executed": False,
              "notebook_path": str(path), "protected_roots": list(map(str, protected_roots)),
              "kernel_executable": sys.executable, "cells": [], "errors": [],
              "raw_sources_archived": archived_raw_allowed,
              "missing_raw_paths": list(map(str, missing_raw_paths)),
              "raw_source_checks": 'presence_and_metadata_sentinels_not_a_fresh_raw_scan'}
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable  # In-memory only; no kernel spec file modified.
    client = None
    started = time.perf_counter()
    try:
        manager.start_kernel(cwd=str(path.parent))
        client = manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=40)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{index}", "exec")
            # This executor deliberately validates quick mode only.
            if "RUN_FULL_AUDIT = True" in source or "CHECK_FULL_REVIEW_DUPLICATES = True" in source:
                raise ValueError("Quick executor refuses full-audit enabled configuration.")
            cell["outputs"] = []
            tick = time.perf_counter()
            message_id = client.execute(source, store_history=True, stop_on_error=True)
            idle = False
            while not idle:
                if time.perf_counter() - tick > args.cell_timeout_seconds:
                    raise TimeoutError(f"Cell {index} exceeded {args.cell_timeout_seconds} seconds.")
                try:
                    message = client.get_iopub_msg(timeout=10)
                except Empty:
                    continue
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind, content = message["header"]["msg_type"], message["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    idle = True
                elif kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    cell["outputs"].append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in {"display_data", "execute_result"}:
                    output = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    cell["outputs"].append(output)
                elif kind == "error":
                    error = {"output_type": "error", "ename": content["ename"],
                             "evalue": content["evalue"], "traceback": content["traceback"]}
                    cell["outputs"].append(error)
                    report["errors"].append({"cell": index, "ename": content["ename"], "evalue": content["evalue"]})
            while True:
                response = client.get_shell_msg(timeout=20)
                if response.get("parent_header", {}).get("msg_id") == message_id:
                    break
            elapsed = round(time.perf_counter() - tick, 3)
            report["cells"].append({"index": index, "execution_count": cell["execution_count"],
                                    "status": response["content"]["status"], "elapsed_seconds": elapsed})
            print(f"Cell {index}: {response['content']['status']} ({elapsed}s)", flush=True)
            if response["content"]["status"] != "ok":
                break
    except Exception as error:
        report["errors"].append({"ename": type(error).__name__, "evalue": str(error)})
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client is not None:
            client.stop_channels()
    after = snapshot()
    report["raw_file_metadata_unchanged"] = before == after
    report["protected_source_metadata_unchanged"] = before == after
    expected = sum(cell["cell_type"] == "code" for cell in notebook["cells"])
    report["expected_code_cells"] = expected
    report["executed_code_cells"] = len(report["cells"])
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["image_outputs"] = sum(any(key.startswith("image/") for key in output.get("data", {}))
                                  for cell in notebook["cells"] for output in cell.get("outputs", []))
    report["status"] = "PASS" if not report["errors"] and len(report["cells"]) == expected and before == after else "FAIL"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
