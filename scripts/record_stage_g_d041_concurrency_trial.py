"""Record the user's bounded concurrency approval; no API/credentials/target edits."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
note = '''
**D041 bounded concurrency trial approved (2026-10-11):** The researcher requested a trial of five simultaneous requests. The serial worker was stopped at a stable saved-response boundary: no active paid requests, no interrupted paid calls,46 saved request intents/49 checkpoints. A separate `concurrency5_trial_v1/manifest.json` records the first five unstarted nonempty jobs in the unchanged frozen order, their exact request hashes and effective runtime concurrency5. The original prepared config/run manifest retain their historical concurrency1; the approved runtime scheduling overlay is explicit and does not alter API inference payloads, prompts/model/schema/settings, sources, targets,100/25 sampling, Stage F or the292initial-call ceiling. All five started requests are drained and saved even if another fails; account/rate/network/provenance failure stops further calls with0automatic retries. This is one bounded scheduling/performance trial, not a production parameter freeze or a repeated-input speed experiment. After a successful trial, resume the already authorized serial pilot at concurrency1 using validated checkpoints without rebilling. Trial result: `concurrency5_trial_v1/report.json`. Final report/ZIP must preserve this runtime amendment and distinguish original serial provenance from the five-call phase.
'''
for rel in ['DECISIONS.md','specs/007_teacher_target_construction.md',
            'thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
    path = root / rel; text = path.read_text(encoding='utf-8')
    if '**D041 bounded concurrency trial approved (2026-10-11):**' not in text:
        first, rest = text.split('\n', 1)
        path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
path = root / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
book = nbformat.read(path, as_version=4)
assert '12.9' in book.cells[-2].source
source = book.cells[-1].source
source = source.replace('"dashboard_execution_preflight.json", ', '"concurrency5_trial_v1/manifest.json", "concurrency5_trial_v1/report.json", "dashboard_execution_preflight.json", ')
source = source.replace('"responses_without_usage"]', '"responses_without_usage", "effective_trial_concurrency", "selected_jobs", "maximum_concurrent_workers", "wall_seconds", "results", "estimated_throughput_ratio_vs_previous_task_means", "after_trial"]')
book.cells[-1].source = source
book.cells[-2].source += '\n\nRuntime exception approved by the researcher: one five-request scheduling trial, then return to serial. The manifest/report explicitly record actual concurrency; frozen inference payloads and content remain unchanged.\n'
nbformat.write(book,path)
print('D041 five-request scheduling trial recorded; no API/key/target edits.')
