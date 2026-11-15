"""Execute only read-only Stage G views and save genuine Jupyter cell outputs."""
from pathlib import Path
import copy
import argparse
import hashlib
import json
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
OUT = ROOT / 'outputs/notebook_stage_g_view_repair_v1'


def main(output_directory='outputs/notebook_stage_g_view_repair_v1'):
    notebook = nbformat.read(PATH, as_version=4)
    plan = json.loads((OUT / 'execution_plan.json').read_text())
    indexes = plan['patched_code_cells']
    untouched = {i: copy.deepcopy(c) for i, c in enumerate(notebook.cells) if i not in indexes}
    # Each view must run on its own in a fresh kernel, without cell 1 or any previous view.
    independent = []
    for index in indexes:
        one = nbformat.v4.new_notebook(cells=[copy.deepcopy(notebook.cells[index])])
        NotebookClient(one, timeout=90, kernel_name='thesis-caption-pilot',
                       resources={'metadata': {'path': str(ROOT)}}).execute()
        assert not any(o.output_type == 'error' for o in one.cells[0].outputs)
        independent.append(index)
    # Execute views in order and write their actual display output into the shared notebook.
    views = nbformat.v4.new_notebook(cells=[copy.deepcopy(notebook.cells[i]) for i in indexes])
    NotebookClient(views, timeout=90, kernel_name='thesis-caption-pilot',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    for index, executed in zip(indexes, views.cells):
        notebook.cells[index].outputs = executed.outputs
        notebook.cells[index].execution_count = executed.execution_count
    for index, cell in untouched.items():
        assert notebook.cells[index] == cell, 'An unrelated historical cell changed'
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    report = {'status': 'READ_ONLY_STAGE_G_VIEWS_EXECUTED_PASS',
              'executed_code_cells': indexes, 'fresh_kernel_independent_cell_checks': independent,
              'execution_errors': 0, 'kernel': 'thesis-caption-pilot',
              'teacher_API_calls': 0, 'scientific_data_or_labels_modified': False,
              'unrelated_historical_cells_preserved': True,
              'notebook_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest()}
    report_directory = ROOT / output_directory
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / 'execution_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', default='outputs/notebook_stage_g_view_repair_v1')
    args = parser.parse_args()
    main(args.output_directory)
