"""Create isolated Mirai trial configuration; preserve Coletoken files unchanged."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'scripts/stage_g_business_id_guard_v1.py'
target=root/'scripts/stage_g_mirai_guard_v1.py'
text=source.read_text(encoding='utf-8')
original="'https://www.coletoken.de/v1', 'coletoken', 'gpt-6.1-sol'"
replacement="'https://api.miraiapi.com/v1', 'miraiapi', 'gpt-6.1-sol'"
assert text.count(original)==1 and not target.exists()
target.write_text(text.replace(original,replacement),encoding='utf-8')
cfg=json.loads((root/'configs/stage_g_production6_retry_v1.json').read_text(encoding='utf-8'))
cfg['status']='USER_APPROVED_ISOLATED_MIRAI_SIX_REQUEST_TRIAL'
cfg['output_directory']='outputs/stage_g/teacher_production_d042_v1/mirai_guarded6_after_retry_v1'
cfg['full_generation_enabled']=False
cfg['execution']['provider']='miraiapi'
cfg['execution']['base_url']='https://api.miraiapi.com/v1'
cfg['execution']['runner']='isolated_mirai_six_request_trial_after_bounded_retry_v1'
cfg['execution']['parent_config']='configs/stage_g_production6_retry_v1.json'
cfg['execution']['dedicated_secret_filename']='mirai_api_key_d042_guard_trial_v1.txt'
cfg['operational_provider_trial']={'maximum_calls':6,'maximum_concurrency':6,'automatic_retries_in_trial':0,
    'selection':'first_two_untouched_nonempty_businesses_in_frozen_order',
    'payload_and_validation_changes':'provider_binding_only_no_teacher_payload_change',
    'production_integration_approved':False,'production_resume_after_trial':False}
destination=root/'configs/stage_g_mirai_guarded6_after_retry_v1.json'
assert not destination.exists()
destination.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
print('Mirai-specific provider binding and trial config prepared; historical files unchanged')
