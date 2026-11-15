"""Document the authorized unstarted-task continuation; no teacher calls/key access."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
note = '''
**D041 continuation_v2 authorized (2026-10-11):** Researcher instructed continue after the five-request trial. Continue only the241 previously unstarted API tasks (246slots including5deterministic empty visual tasks), with concurrency1 and0automatic retries. Reuse byte-identical validated checkpoints/raw responses/requests:51prior paid intents,54checkpoints,50complete raw responses,17complete businesses. Preserve the failed `train_2pj5byDMoMDu-d0bTu2Sew_unfiltered_visual_evidence` attempt unchanged; it is not resent or silently replaced/excluded. This continuation may end with99complete businesses and the original failed business still incomplete unless a separate reviewed recovery is approved; preserve the100cohort and fixed25sample regardless. Output namespace `outputs/stage_g/teacher_extended_pilot_d041_v1/continuation_v2/`, runner `scripts/run_stage_g_d041_continuation_v2.py`; source namespace/previous reports/ZIP/pages remain protected by SHA256. Offline preflight passed before credentials/API: all copied request bodies/checkpoints/raw responses/evidence bindings validated, original input/protocol hashes unchanged. All target sentences remain teacher-API-only with frozen serialization. Prompts/model/alias/inference settings/StageF/K/source budgets remain unchanged. The previous concurrency5 amendment is historical, not the current scheduler. Monitor money directly in Coletoken; no pricing checkpoint or Codex config/auth access. Cumulative initial-call ceiling292 includes51imported intents; new initial-call allowance241. New fatal account/rate/network/stream/provenance errors stop without automatic retry. A graceful stop flag is now supported at saved-response boundaries. After completion, automatically audit and export a NEW fixed25 ZIP/page in continuation_v2, preserving historical partial exports, then stop before production or LED training. Progress checks remain on researcher request only.
'''
for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md','DECISIONS.md',
            'specs/007_teacher_target_construction.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
    path = root / rel; text = path.read_text(encoding='utf-8')
    if '**D041 continuation_v2 authorized (2026-10-11):**' not in text:
        first, rest = text.split('\n', 1)
        path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
path = root / 'configs/research_pending.yaml'
text = path.read_text(encoding='utf-8').replace(
    'D041_pilot_execution_finished_researcher_review_pending_production_not_frozen',
    'D041_continuation_v2_unstarted_tasks_authorized_serial1_previous_failure_preserved_production_not_frozen')
if 'active_pilot_execution_pointer:' not in text:
    text = text.replace('  active_pilot_config: configs/stage_g_teacher_extended_pilot_d041.json',
        '  active_pilot_config: configs/stage_g_teacher_extended_pilot_d041.json\n  active_pilot_execution_pointer: outputs/stage_g/teacher_extended_pilot_d041_v1/active_execution_pointer.json')
path.write_text(text, encoding='utf-8')
path = root / 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb'
book = nbformat.read(path, as_version=4)
assert '12.9' in book.cells[-2].source
source = book.cells[-1].source
marker = '_d041_folder = _d041_root / "outputs/stage_g/teacher_extended_pilot_d041_v1"\n'
if '_d041_pointer =' not in source:
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''_d041_pointer = _d041_folder / "active_execution_pointer.json"
if _d041_pointer.exists():
    _d041_active = json.loads(_d041_pointer.read_text(encoding="utf-8"))
    _d041_folder = _d041_root / _d041_active["current_namespace"]
    print("Active execution:", _d041_folder)
''')
    source = source.replace('for _rel in [', 'for _rel in ["continuation_manifest.json", "continuation_preflight.json", "postprocessing_status.json", ')
    source = source.replace('"after_trial"]', '"after_trial", "effective_concurrency", "new_initial_API_calls_this_continuation", "cumulative_cohort_API_request_intents", "failed_tasks_preserved_not_retried", "unstarted_paid_tasks"]')
    book.cells[-1].source = source
    book.cells[-2].source += '\n\nLatest continuation: serial1, new namespace continuation_v2, validated successful responses reused without rebilling; previous failed request remains unchanged. The active execution pointer selects current status while preserving historical reports.\n'
    nbformat.write(book, path)
print('D041 continuation documented and notebook status pointer updated; no API/key access.')
