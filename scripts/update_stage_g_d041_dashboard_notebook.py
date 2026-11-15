"""Append a read-only D041 runtime/results view; preserve every prior notebook cell."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
book = nbformat.read(path, as_version=4)
heading = '## 12.9 D041 — dashboard-authorized execution and fixed25 review'
if not any(heading in c.source for c in book.cells):
    book.cells.append(nbformat.v4.new_markdown_cell(heading + '''

Researcher-approved100-business pilot; the prior pricing gate was superseded by explicit approval and Coletoken dashboard monitoring. Same frozen100/25 sample, D040 protocol and Stage F. This cell reads saved artifacts only: no API, credential access or generation. Teacher judgments remain distinct from human-confirmed factual assessment. No full production or training.
'''))
    book.cells.append(nbformat.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import display, Markdown
_d041_root = Path.cwd()
if not (_d041_root / "outputs").exists():
    _d041_root = _d041_root.parent
_d041_folder = _d041_root / "outputs/stage_g/teacher_extended_pilot_d041_v1"
for _rel in ["dashboard_execution_preflight.json", "execution_v1/execution_progress.json", "execution_v1/completion.json", "execution_report.json", "validation_report.json", "human_review_bundle_report.json"]:
    _p = _d041_folder / _rel
    if _p.exists():
        _r = json.loads(_p.read_text(encoding="utf-8"))
        _keys = ["status", "businesses_completed", "businesses_complete", "target_texts", "API_request_intents", "new_API_request_intents", "accepted_tasks", "failed_tasks", "usage", "stop_reason", "technical_flags", "exact_cross_condition", "target_tokens", "cases", "zip_bytes", "zip_path", "viewer_path", "responses_without_usage"]
        display(Markdown("**" + _rel + "**"))
        display({k: _r[k] for k in _keys if k in _r})
    else:
        print(_rel + ": pending")
print("Automated validation does not establish independent factual accuracy. Human review remains pending.")
'''))
    nbformat.write(book, path)
print('Read-only D041 view available; prior cells preserved, no API/key access.')
