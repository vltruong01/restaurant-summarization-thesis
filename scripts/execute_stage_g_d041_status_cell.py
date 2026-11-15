"""Execute only the new read-only notebook status cell, preserving earlier cells."""
from pathlib import Path
import json
import re
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
book = nbformat.read(path, as_version=4)
assert 'D041' in book.cells[-2].source and book.cells[-1].cell_type == 'code'
one = nbformat.v4.new_notebook(cells=[book.cells[-1]])
NotebookClient(one, timeout=60, kernel_name='thesis-caption-pilot',
               resources={'metadata': {'path': str(root)}}).execute()
book.cells[-1] = one.cells[0]
nbformat.write(book, path)
pointer_path = root / 'outputs/stage_g/teacher_extended_pilot_d041_v1/active_execution_pointer.json'
if pointer_path.exists():
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    active_folder = root / pointer['current_namespace']
    completion_path = active_folder / 'execution_v1/completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        pointer['status'] = completion['status']
        pending_path = root / 'configs/research_pending.yaml'
        pending = pending_path.read_text(encoding='utf-8')
        pending = re.sub(r'(?m)^  pilot_protocol_status: .*$',
                         '  pilot_protocol_status: D041_' + active_folder.name + '_finished_researcher_review_pending_production_not_frozen', pending)
        pending_path.write_text(pending, encoding='utf-8')
    elif active_folder.name.startswith('recovery_v'):
        recovery_path = active_folder / 'recovery_attempts.json'
        pointer['status'] = active_folder.name.upper() + ('_RUNNING_CONCURRENCY4' if recovery_path.exists() else '_SERIAL_OPERATOR_RETRIES')
    elif (active_folder / 'concurrency4_continuous_v1/manifest.json').exists():
        pointer['status'] = 'CONTINUATION_RUNNING_CONCURRENCY4'
    elif (active_folder / 'concurrency4_trial_v1/manifest.json').exists() and not (active_folder / 'concurrency4_trial_v1/report.json').exists():
        pointer['status'] = 'CONTINUATION_BOUNDED_CONCURRENCY4_TRIAL_RUNNING'
    elif (active_folder / 'concurrency4_trial_v1/report.json').exists() and json.loads((active_folder / 'concurrency4_trial_v1/report.json').read_text(encoding='utf-8'))['after_trial'] == 'stop_for_review':
        pointer['status'] = 'CONTINUATION_CONCURRENCY4_TRIAL_STOPPED_FOR_REVIEW'
    elif (active_folder / 'execution_v1/execution_progress.json').exists():
        pointer['status'] = 'CONTINUATION_RUNNING_SERIAL1'
    pointer_path.write_text(json.dumps(pointer, indent=2) + '\n', encoding='utf-8')
print('D041 latest status cell executed; earlier cells untouched; no API/key access.')
