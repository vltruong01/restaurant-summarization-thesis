"""Record the explicit user D041 execution approval; no credentials/API access."""
from run_stage_g_extended_pilot_d041 import check_inputs, FOLDER, CONFIG, ROOT, read, file_sha, immutable

cfg, pre, jobs = check_inputs()
auth = {
    'approved': True, 'approved_by': 'user', 'approval_date': '2026-10-11',
    'business_count': 100, 'production_authorized': False, 'maximum_calls': pre['max_new_API_calls'],
    'sampling_manifest_sha256': file_sha(FOLDER / 'sampling_manifest.json'),
    'manual_review_25_manifest_sha256': file_sha(FOLDER / 'manual_review_25_manifest.json'),
    'config_sha256': file_sha(CONFIG), 'jobs_sha256': pre['jobs_sha256'],
    'cost_monitoring': 'researcher_Coletoken_dashboard',
    'pricing_checkpoint_required': False,
    'authorization_basis': 'User explicitly accepted D041 preparation and authorized the 100-business pilot; separate pricing research, spending cap proposal and additional cost approval are not required. Usage is monitored by the researcher through Coletoken.',
    'automatic_retries': 0, 'teacher_protocol': 'unchanged_D040',
    'Codex_configuration_or_authentication_access': False,
}
immutable(FOLDER / 'dashboard_execution_authorization.json', auth)
note = '''
**D-041 execution authorization update (2026-10-11):** The researcher accepted the preparation bundle and explicitly authorized the unchanged 100-business pilot, with monetary usage monitored directly through the Coletoken dashboard. This supersedes the earlier preparation-only pricing/cost-cap checkpoint for this bounded pilot. No separate pricing investigation, cost estimate, cap proposal or additional paid-approval checkpoint is required. Historical prepared config/status/estimates remain unchanged as provenance; the supplemental `dashboard_execution_authorization.json` binds their exact hashes and the unchanged 100/25 manifests. Execution uses `scripts/run_stage_g_extended_pilot_d041_dashboard.py`, at most 292 initial API calls, concurrency1 and zero automatic retries. Missing provider usage is recorded as unknown, not invented. Account/rate/network failures and unresolved requests still stop without automatic resending. All target sentences come exclusively from the teacher API via frozen serialization. D040 prompts/model/settings/schema and Stage F inputs/K/budgets remain unchanged. No Codex config/auth access, held-out teacher, full-cohort generation or LED training is authorized. Automatic validation is not independent factual or human verification. Stop after the 100-case pilot and preselected25 review package.
'''
for relative in ['DECISIONS.md', 'specs/007_teacher_target_construction.md',
                 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
    path = ROOT / relative
    text = path.read_text(encoding='utf-8')
    if 'D-041 execution authorization update (2026-10-11)' not in text:
        first, rest = text.split('\n', 1)
        path.write_text(first + '\n' + note + '\n' + rest, encoding='utf-8')
path = ROOT / 'thesis_sdd_docs/THESIS_PROGRESS.md'
text = path.read_text(encoding='utf-8')
if 'D-041 execution authorization update (2026-10-11)' not in text:
    first, rest = text.split('\n', 1)
    rest = rest.replace('Current state (2026-10-11):', 'Historical pre-execution preparation state (2026-10-11):', 1)
    path.write_text(first + '\n\nCurrent state (2026-10-11): D041 execution approved; offline technical preflight precedes API execution. Fixed100/25 sampling and D040 protocol unchanged. Coletoken dashboard monitors monetary usage; no further pricing checkpoint. Production teacher remains unfrozen.\n' + note + '\n' + rest, encoding='utf-8')
path = ROOT / 'configs/research_pending.yaml'
text = path.read_text(encoding='utf-8')
text = text.replace('D041_extended_fixed_D040_scope_approved_100_unseen_cases_offline_preparation_paid_execution_pending_cost_authorization_production_not_frozen',
                    'D041_extended_fixed_D040_100_cases_execution_authorized_dashboard_monitored_production_not_frozen')
path.write_text(text, encoding='utf-8')
print('D041 bounded dashboard authorization recorded; no API/key access.')
