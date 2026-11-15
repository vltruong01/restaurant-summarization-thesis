"""Record observed trial results only; preserve API outputs and historical exports."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
folder = root / 'outputs/stage_g/teacher_extended_pilot_d041_v1'
report = json.loads((folder / 'concurrency5_trial_v1/report.json').read_text(encoding='utf-8'))
passed = sum(r['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for r in report['results'])
note = f'''
**D041 concurrency5 trial outcome (2026-10-11):** Exactly5 initial calls attempted concurrently; peak5 client workers, wall{report['wall_seconds']}seconds, {passed} structurally/source-valid completed responses and1 failed response-stream task (`train_2pj5byDMoMDu-d0bTu2Sew_unfiltered_visual_evidence`, RuntimeError, HTTP status null, no complete raw response/usage returned). No HTTP429 was recorded and the evidence does not establish that concurrency caused the failure. The previous serial task-type means predict approximately{report['previous_serial_task_mix_estimate_seconds']}seconds for the same task mix, but these are different inputs and one trial failed; do not claim a reliable1.9x successful-throughput gain. All started calls were drained/saved and no failed request was resent. D041 stopped for review at17/100complete businesses,51provisional targets,51total request intents and50complete raw responses; it did not resume serial or proceed to production. The partial random25 review export retains all25 predetermined cases (2complete/23incomplete), explicitly marked incomplete rather than presented as a finished100-business pilot. Its ZIP/document snapshots are preserved as-of-export. Automated validation PASS_WITH_PRESERVED_FAILURES checks saved evidence/schema/serialization, not independent factual correctness. Existing method/labels/StageF and100/25 manifests remain unchanged apart from the already approved bounded runtime scheduling trial. Review `concurrency5_trial_v1/report.json` and `execution_report.json` before an explicitly reviewed recovery or further execution.
'''
for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md','DECISIONS.md',
            'specs/007_teacher_target_construction.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
    path = root / rel; text = path.read_text(encoding='utf-8')
    if '**D041 concurrency5 trial outcome (2026-10-11):**' not in text:
        first, rest = text.split('\n', 1)
        path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
print(json.dumps({'passed': passed, 'failed': 5-passed, 'wall_seconds': report['wall_seconds'],
                  'businesses_complete': report['businesses_complete_after_trial'],
                  'pilot_stopped': report['after_trial'] == 'stop_for_review'}))
