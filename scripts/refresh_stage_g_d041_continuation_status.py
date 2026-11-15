"""Update only operational progress wording after the first continuation API call."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'thesis_sdd_docs/THESIS_PROGRESS.md'
text = path.read_text(encoding='utf-8')
if 'Current active execution namespace: `continuation_v2`' not in text:
    text = text.replace('Current state (2026-10-11):', 'Historical state (2026-10-11):')
    first, rest = text.split('\n', 1)
    text = first + '\n\nCurrent active execution namespace: `continuation_v2` under D041, serial1. Authoritative live state is `outputs/stage_g/teacher_extended_pilot_d041_v1/continuation_v2/execution_v1/execution_progress.json` (and `completion.json` when present). The stopped17/100 result and concurrency5 trial below belong to the preserved prior execution; new continuation targets remain API-only and the prior failed request is not automatically retried. Production teacher remains unfrozen.\n' + rest
    path.write_text(text, encoding='utf-8')
path = root / 'configs/research_pending.yaml'
text = path.read_text(encoding='utf-8').replace(
    'D041_continuation_v2_unstarted_tasks_authorized_serial1_previous_failure_preserved_production_not_frozen',
    'D041_continuation_v2_running_serial1_previous_failure_preserved_production_not_frozen')
path.write_text(text, encoding='utf-8')
print('Current progress points to continuation_v2; historical outputs preserved.')
