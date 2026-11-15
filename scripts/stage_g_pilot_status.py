"""Read-only live pilot status; no credentials, API calls or target edits."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]


def status(config='configs/stage_g_teacher_pilot_v3_canonical_ids.json'):
    cfg = json.loads((ROOT / config).read_text(encoding='utf-8'))
    folder = ROOT / cfg['output_directory']
    execution = folder / 'execution_v1'
    jobs = json.loads((folder / 'pilot_jobs.json').read_text(encoding='utf-8'))
    states = Counter()
    pending = []
    failures = []
    completed_api = 0
    deterministic_empty = 0
    usages = []
    for job in jobs:
        checkpoint = execution / 'checkpoints' / (job['job_id'] + '.json')
        intent = execution / 'request_intents' / (job['job_id'] + '.json')
        try:
            if checkpoint.exists():
                cp = json.loads(checkpoint.read_text(encoding='utf-8'))
                states[cp['status']] += 1
                if cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                    if cp.get('api_called'):
                        completed_api += 1
                        usages.append(cp.get('usage') or {})
                    else:
                        deterministic_empty += 1
                else:
                    failures.append({'job_id': job['job_id'], 'error_type': cp.get('error_type'),
                                     'http_status': cp.get('http_status'),
                                     'validation_error': cp.get('local_validation_error')})
            elif intent.exists():
                started = json.loads(intent.read_text(encoding='utf-8'))['started_utc']
                pending.append({'job_id': job['job_id'], 'started_utc': started,
                                'elapsed_seconds': round((datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds(), 1)})
                states['STARTED_NO_COMPLETED_CHECKPOINT'] += 1
            else:
                states['NOT_STARTED'] += 1
        except json.JSONDecodeError:
            states['FILE_WRITE_IN_PROGRESS'] += 1
    targets = []
    for path in sorted((execution / 'provisional_targets').glob('*.json')):
        try:
            target = json.loads(path.read_text(encoding='utf-8'))
            targets.append({'business_id': target['business_id'], 'split': target['split'],
                            'human_verified': target['human_verified'],
                            'zero_filtered': target['zero_filtered_edge_case']})
        except json.JSONDecodeError:
            continue
    return {'protocol_version': cfg['protocol_version'], 'model': cfg['requested_model'],
            'planned_businesses': 20, 'businesses_with_all_targets': len(targets),
            'completed_API_tasks': completed_api, 'deterministic_empty_visual_tasks': deterministic_empty,
            'task_states': dict(states), 'pending_or_interrupted_requests': pending,
            'failed_tasks': failures, 'human_verified_businesses': sum(x['human_verified'] for x in targets),
            'input_tokens_completed_tasks': sum(u.get('input_tokens', 0) for u in usages),
            'output_tokens_completed_tasks': sum(u.get('output_tokens', 0) for u in usages),
            'targets': targets, 'full_generation_started': False,
            'note': 'Started-without-checkpoint may be live or interrupted; status does not automatically retry or certify grounding.'}


if __name__ == '__main__':
    print(json.dumps(status(), indent=2))
