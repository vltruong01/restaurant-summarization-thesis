"""Summarize actual E0 attempts and capacity without changing caption policy."""
from collections import Counter
from pathlib import Path
import csv
import json
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def interval(values):
    return {'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values)}


def main():
    cfg = yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))
    out = ROOT/cfg['output_directory']
    generation = read(out/'generation_report.json')
    assert generation['status'] == 'raw_pilot_complete_pending_review'
    plan = read(out/'pilot_photo_plan.json')
    records = [read(out/'caption_checkpoints'/f"{p['photo_id']}.json") for p in plan]
    attempts = [a for r in records for a in r['attempts']]
    initial = [r['attempts'][0] for r in records]
    retries = [r['attempts'][1] for r in records if len(r['attempts']) == 2]
    chosen = [r['attempts'][r['chosen_attempt']-1] for r in records]
    with (out/'pilot_source_token_ledger.csv').open(encoding='utf-8', newline='') as stream:
        ledger = list(csv.DictReader(stream))
    with (out/'raw_caption_statistics.csv').open(encoding='utf-8', newline='') as stream:
        stats = list(csv.DictReader(stream))
    raw_sources = [s for s in ledger if s['condition'] == 'all_raw_pilot_captions_capacity_only']
    summary = {
        'status': 'ACTUAL_24_PHOTO_DIAGNOSTICS_NOT_PRODUCTION_QUALITY_ACCEPTANCE',
        'photos': len(records), 'businesses': len({r['business_id'] for r in records}),
        'splits': dict(Counter(r['split'] for r in records)),
        'attempts': len(attempts), 'initial_attempts': len(initial), 'retry_attempts': len(retries),
        'chosen_surface_eligible': sum(r['surface_eligible'] for r in records),
        'chosen_surface_flags': dict(Counter(f for a in chosen for f in a['surface_flags'])),
        'initial_empty': sum(not a['raw_caption'] for a in initial),
        'retry_empty': sum(not a['raw_caption'] for a in retries),
        'retry_surface_eligible': sum(a['surface_eligible'] for a in retries),
        'chosen_empty': sum(not r['raw_caption'] for r in records),
        'chosen_words': interval([a['word_count'] for a in chosen]),
        'chosen_LED_body_tokens': interval([int(s['LED_body_tokens']) for s in stats]),
        'initial_seconds': interval([a['seconds'] for a in initial]),
        'retry_seconds': interval([a['seconds'] for a in retries]) if retries else None,
        'all_attempt_seconds_total': sum(a['seconds'] for a in attempts),
        'seconds_per_photo_all_attempts': interval([sum(a['seconds'] for a in r['attempts']) for r in records]),
        'peak_GPU_allocated_GiB': max(a['peak_GPU_allocated_GiB'] for a in attempts),
        'peak_GPU_reserved_GiB': max(a['peak_GPU_reserved_GiB'] for a in attempts),
        'raw_full_source_tokens': interval([int(s['actual_full_source_tokens']) for s in raw_sources]),
        'raw_incremental_caption_block_tokens': interval([int(s['actual_incremental_caption_block_tokens']) for s in raw_sources]),
        'source5000_overflows': sum(s['within_diagnostic_total5000'] != 'True' for s in ledger),
        'caption750_overflows': sum(s['within_provisional_caption750'] != 'True' for s in ledger),
        'raw_text_modified': False, 'research_parameters_changed': False,
        'final_caption_policy_K_source_allocation_manifest': False,
        'production_A2_or_caption_quality_passed': False,
        'manual_picture_caption_review': ('separate_assistant_report_complete_not_formal_human_evaluation'
                                          if (out/'manual_caption_review_notes.json').is_file()
                                          else 'separate_report_required'),
        'teacher_LED_training_or_rental_started': False,
    }
    (out/'diagnostic_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
