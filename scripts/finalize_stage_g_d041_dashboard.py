"""Offline D041 execution reporting only; never creates or edits teacher content."""
import json
import runpy
import statistics
from collections import Counter
from pathlib import Path
from run_stage_g_extended_pilot_d041 import check_inputs, ROOT, FOLDER, read, file_sha, immutable
from run_stage_g_extended_pilot_d041_dashboard import preflight


def finalize():
    cfg, pre, jobs, auth = preflight()
    out = FOLDER / 'execution_v1'
    completion = read(out / 'completion.json')
    from audit_stage_g_extended_pilot_d041 import results
    validation = results()
    protected = read(FOLDER / 'dashboard_execution_preflight.json')
    for rel, h in protected['preparation_artifacts_sha256'].items():
        assert file_sha(FOLDER / rel) == h
    flags = read(FOLDER / 'technical_flags.json')
    trial_path = FOLDER / 'concurrency5_trial_v1/report.json'
    trial = read(trial_path) if trial_path.exists() else None
    timings = {}
    job_types = {j['job_id']: j['task'] + ':' + j['condition'] for j in jobs}
    for path in (out / 'checkpoints').glob('*.json'):
        cp = read(path)
        if cp.get('api_called') and cp.get('seconds') is not None:
            timings.setdefault(job_types[path.stem], []).append(cp['seconds'])
    timing_stats = {k: {'n': len(v), 'mean_seconds': round(statistics.mean(v), 3),
                       'median_seconds': round(statistics.median(v), 3),
                       'min_seconds': min(v), 'max_seconds': max(v)} for k, v in timings.items()}
    report = {
        'scope': 'D041_100_business_pilot_only', 'execution': completion,
        'automated_validation_status': validation['status'],
        'offline_technical_preflight_status': protected['status'],
        'sampling_100_and_manual25_unchanged': True,
        'original_preparation_zip_and_page_unchanged': True,
        'D040_scientific_protocol_unchanged': True, 'Stage_F_unchanged': True,
        'runtime_scheduling_trial': trial,
        'runtime_scheduling_disclosure': 'Original serial config retained as provenance. User-approved one-batch concurrency5 trial, then serial1; API inference payload settings unchanged.' if trial else 'serial1_only',
        'teacher_target_content_origin': 'teacher_API_only_verbatim_frozen_serialization',
        'independent_factual_assessment': 'pending_researcher_review',
        'factual_accuracy_estimate': None, 'human_verified_businesses': 0,
        'flag_counts': dict(Counter(f['flag'] for f in flags['flags'])),
        'exact_matching_claim_verdict_audit': validation['exact_cross_condition'],
        'flags_are_diagnostics_not_confirmed_factual_errors': True,
        'zero_filtered_businesses_complete': validation['zero_caption_businesses_complete'],
        'usage_source': 'provider_returned_metadata_only_missing_usage_explicit',
        'request_completion_seconds_by_task': timing_stats,
        'money_monitoring': 'researcher_Coletoken_dashboard_no_pricing_gate',
        'config_sha256': file_sha(ROOT / 'configs/stage_g_teacher_extended_pilot_d041.json'),
        'authorization_sha256': file_sha(FOLDER / 'dashboard_execution_authorization.json'),
        'validation_report_sha256': file_sha(FOLDER / 'validation_report.json'),
        'stop_after_pilot': True, 'production_teacher_frozen': False, 'LED_training_started': False,
    }
    immutable(FOLDER / 'execution_report.json', report)
    lines = [
        '# D-041 extended pilot execution report', '',
        f"Execution status: {completion['status']}. Complete businesses: {completion['businesses_completed']}/100; target texts: {completion['target_texts']}.",
        f"Initial API request intents: {completion['new_API_request_intents']}/292; automatic retries: 0. Provider-reported usage: {json.dumps(completion['usage'])}.",
        f"Automated source/schema/token/serialization validation: {validation['status']}. This is not a factual-accuracy certificate.",
        f"Stop reason: {completion['stop_reason']}. Preserved failed tasks: {len(completion['failed_tasks'])}.",
        f"Technical flag counts: {json.dumps(report['flag_counts'])}.",
        '', 'The fixed 25-business review sample is unchanged, including incomplete cases if any. Review exact claims with original reviews/captions/photos; teacher verdicts are model-produced, not human ground truth.',
        'No target content was written, corrected or paraphrased by Codex. Only approved verbatim teacher sentence serialization and evidence-ID resolution were applied.',
        'D040 protocol and Stage F controls remain unchanged. Monetary usage is monitored by the researcher in Coletoken; no invoice cost is inferred.',
        'Full production and LED training remain stopped pending researcher approval.',
    ]
    md = FOLDER / 'execution_report.md'
    content = '\n'.join(lines) + '\n'
    if md.exists(): assert md.read_text(encoding='utf-8') == content
    else: md.write_text(content, encoding='utf-8')
    note = '\n**D041 actual execution result (2026-10-11):** ' + ' '.join(lines[2:7]) + ' Fixed25 human review remains pending; teacher judgments are not confirmed factual ground truth. See `outputs/stage_g/teacher_extended_pilot_d041_v1/execution_report.json`, `validation_report.json`, `technical_flags.json`, and the self-contained random25 ZIP/page. Stage F/D040/sampling unchanged; stopped before full production or training.\n'
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'DECISIONS.md',
                'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        path = ROOT / rel; text = path.read_text(encoding='utf-8')
        if '**D041 actual execution result (2026-10-11):**' not in text:
            first, rest = text.split('\n', 1)
            if rel.endswith('THESIS_PROGRESS.md'):
                rest = rest.replace('Current state (2026-10-11):', 'Historical execution-start state (2026-10-11):', 1)
                note_for_file = f"\nCurrent state (2026-10-11): D041 pilot stopped after execution; {completion['businesses_completed']}/100 complete, {completion['target_texts']} provisional targets, {completion['new_API_request_intents']} API request intents. Researcher factual review pending; no production approval or training.\n" + note
            else: note_for_file = note
            path.write_text(first + '\n' + note_for_file + '\n' + rest, encoding='utf-8')
    config = ROOT / 'configs/research_pending.yaml'
    text = config.read_text(encoding='utf-8').replace(
        'D041_extended_fixed_D040_100_cases_execution_authorized_dashboard_monitored_production_not_frozen',
        'D041_pilot_execution_finished_researcher_review_pending_production_not_frozen')
    config.write_text(text, encoding='utf-8')
    # Export with the frozen exporter, adding current operational reports only.
    exporter = ROOT / 'scripts/export_stage_g_extended_pilot_d041_review.py'
    original = exporter.read_text(encoding='utf-8')
    old = "'technical_flags.json','pilot_usage_estimate.json']"
    new = "'technical_flags.json','pilot_usage_estimate.json','execution_report.json','execution_report.md','dashboard_execution_authorization.json','dashboard_execution_preflight.json','execution_v1/completion.json','execution_v1/run_manifest.json']"
    if trial:
        new = new[:-1] + ",'concurrency5_serial_handoff.json','concurrency5_trial_v1/authorization.json','concurrency5_trial_v1/manifest.json','concurrency5_trial_v1/report.json']"
    assert original.count(old) == 1
    derived = ROOT / 'scripts/export_stage_g_extended_pilot_d041_dashboard_review.py'
    runtime_marker = "    files['protocol/config.json']=original(CONFIG)"
    assert original.count(runtime_marker) == 1
    runtime_files = runtime_marker + "\n    for runtime in ['run_stage_g_extended_pilot_d041_dashboard.py','authorize_stage_g_d041_dashboard.py','finalize_stage_g_d041_dashboard.py']:\n        files['protocol/runtime/'+runtime]=original(ROOT/'scripts'/runtime)"
    if trial:
        runtime_files += "\n    files['protocol/runtime/run_stage_g_d041_concurrency5_trial.py']=original(ROOT/'scripts/run_stage_g_d041_concurrency5_trial.py')"
    source = '# Derived export-only wrapper source; original exporter preserved. Source SHA256: ' + file_sha(exporter) + '\n' + original.replace(old, new).replace(runtime_marker, runtime_files)
    if derived.exists(): assert derived.read_text(encoding='utf-8') == source
    else: derived.write_text(source, encoding='utf-8')
    runpy.run_path(str(derived), run_name='__main__')
    runpy.run_path(str(ROOT / 'scripts/execute_stage_g_d041_status_cell.py'), run_name='__main__')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': finalize()
