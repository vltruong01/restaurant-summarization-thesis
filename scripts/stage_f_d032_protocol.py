"""Approved D-032 source-budget/template authority; retain immutable D-031 filters."""
from pathlib import Path
import copy
import yaml
from stage_f_verification_workflow import read,write,sha
from source_budget_audit_template import TEMPLATE,pinned_tokenizer

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/stage_f/methodology_amendment_d032'
PREVIOUS=ROOT/'outputs/stage_f/methodology_amendment_d031'
AUDIT=ROOT/'outputs/stage_f/actual_source_budget_freeze_audit_v1'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
CONFIG=ROOT/'configs/stage_f_validation_protocol.yaml'
SOURCE_TEMPLATE=ROOT/'configs/led_source_template_d032.json'
BASELINE=STATE/'previous_state/configs/stage_f_validation_protocol.yaml'
THRESHOLDS={'relevance':.08,'redundancy':.59,'grouping':.46}


def active_d032():return yaml.safe_load(CONFIG.read_text(encoding='utf-8')).get('decision')=='D-032'


def assert_exact_authorized_delta(old,new):
    expected=copy.deepcopy(old)
    expected['decision']='D-032';expected['status']='approved_validated_common_source_budget_template_frozen_K_pending_review'
    expected['budget']={'caption_tokens':750,'full_source_tokens':5000,'review_tokens':4000,
                        'remaining_declared_overhead_capacity':250,'diagnostic_total_target':5000}
    expected['source_serialization']={
        'frozen_template':'configs/led_source_template_d032.json',
        'budget_manifest':'outputs/stage_f/methodology_amendment_d032/source_budget_manifest.json',
        'shared_validation_review_reference':'outputs/stage_f/methodology_amendment_d032/frozen_shared_review_reference.json',
        'caption_accounting':'exact_full_source_minus_identical_reviews_only_base',
        'overflow_policy':'record_and_stop_zero_silent_truncation_or_removal'}
    assert new==expected,'D-032 must not alter any score/filter/group/ranking/K definition'
    assert new['K'] is None and new['K_candidates']==[3,5,8,10]
    return True


def validate_compatibility():
    assert active_d032();old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));new=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert_exact_authorized_delta(old,new)
    template=read(SOURCE_TEMPLATE);audited=read(AUDIT/'audit_template.json')
    extras=['decision','audited_candidate_sha256','serializer_code_sha256','audit_report_sha256','tokenizer_manifest_sha256']
    core={k:v for k,v in template.items() if k not in extras};core['status']=audited['status']
    assert core==audited==TEMPLATE and template['status']=='APPROVED_VALIDATED_FROZEN' and template['decision']=='D-032'
    assert template['audited_candidate_sha256']==sha(AUDIT/'audit_template.json')
    assert template['serializer_code_sha256']==sha(ROOT/'scripts/source_budget_audit_template.py')
    assert template['audit_report_sha256']==sha(AUDIT/'audit_report.json')
    assert template['tokenizer_manifest_sha256']==sha(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    _,manifest=pinned_tokenizer()
    audit=read(AUDIT/'audit_report.json');verification=read(AUDIT/'verification_report.json')
    assert audit['status']=='PASS_ZERO_OVERFLOW_D032_PROPOSAL_REQUIRES_RESEARCHER_APPROVAL'
    assert audit['independent_slow_LEDTokenizer_full_ID_equality_checks']==600 and verification['status']=='PASS_ACTUAL_TOKEN_AUDIT_ONLY_D032_PENDING'
    s=audit['summary']['all_validation_200']
    assert [s[c]['caption_block_tokens']['max'] for c in ['unfiltered','filtered']]==[265,254]
    assert [s[c]['complete_source_tokens']['max'] for c in ['unfiltered','filtered']]==[4310,4312]
    assert all(s[c]['caption_blocks_over750']==s[c]['full_sources_over5000']==0 for c in s)
    assert sha(STATE/'frozen_shared_review_reference.json')==sha(AUDIT/'shared_review_reference.json')
    receipt=read(PREVIOUS/'scoring_compatibility_receipt.json')
    assert receipt['active_protocol_sha256']==sha(BASELINE)
    seal=read(STATE/'protected_before_sha256.json')
    for name in ['scripts/stage_f_math.py','scripts/stage_f_score_validation.py','scripts/source_budget_audit_template.py',
                 'outputs/stage_f/validation_calibration_v1/fixed_filter_pools.json',
                 'outputs/stage_f/validation_calibration_v1/frozen_thresholds.json',
                 'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json']:
        assert sha(ROOT/name)==seal[name],name
    frozen=read(BASE/'frozen_thresholds.json');assert frozen['thresholds']==THRESHOLDS and frozen['protocol_sha256']==sha(BASELINE)
    result={'status':'PASS_D032_BUDGET_TEMPLATE_FREEZE_UNCHANGED_D031_FILTERS',
            'active_protocol_sha256':sha(CONFIG),'previous_D031_protocol_sha256':sha(BASELINE),
            'previous_compatibility_receipt_sha256':sha(PREVIOUS/'scoring_compatibility_receipt.json'),
            'inherited_filter_manifest_sha256':sha(BASE/'frozen_thresholds.json'),
            'fixed_pools_sha256':sha(BASE/'fixed_filter_pools.json'),'frozen_template_sha256':sha(SOURCE_TEMPLATE),
            'frozen_review_reference_sha256':sha(STATE/'frozen_shared_review_reference.json'),
            'audit_report_sha256':sha(AUDIT/'audit_report.json'),'caption_thresholds_unchanged':THRESHOLDS,
            'budgets':{'full_source_tokens':5000,'review_tokens':4000,'caption_tokens':750},
            'tokenizer':manifest,'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only',
            'scoring_reports':receipt['scoring_reports'],'original_scoring_protocol_sha256':receipt['original_scoring_protocol_sha256'],
            'budget_template_formally_frozen':True,'K_not_frozen':True,'heldout_test_used':False}
    path=STATE/'compatibility_receipt.json'
    if path.exists():assert read(path)==result
    else:write(path,result)
    return result


def require_validated_amendment():
    receipt=validate_compatibility();v=read(STATE/'amendment_validation_report.json')
    assert v['status']=='PASS' and v['active_protocol_sha256']==sha(CONFIG)
    assert v['protocol_code_sha256']==sha(Path(__file__)) and v['test_code_sha256']==sha(ROOT/'scripts/test_stage_f_d032.py')
    return receipt


def freeze_budget_manifest():
    receipt=require_validated_amendment()
    result={'status':'D032_APPROVED_VALIDATED_COMMON_SOURCE_BUDGET_TEMPLATE_FROZEN','decision':'D-032',
            'budgets_finalized':True,'budgets':receipt['budgets'],'source_template_sha256':sha(SOURCE_TEMPLATE),
            'source_template_path':'configs/led_source_template_d032.json','source_serializer_sha256':sha(ROOT/'scripts/source_budget_audit_template.py'),
            'review_reference_sha256':sha(STATE/'frozen_shared_review_reference.json'),'filter_threshold_manifest_sha256':sha(BASE/'frozen_thresholds.json'),
            'protocol_sha256':sha(CONFIG),'compatibility_receipt_sha256':sha(STATE/'compatibility_receipt.json'),
            'approval_source':'explicit_researcher_D032_approval_after_actual_caption_validation_audit',
            'overflow_policy':TEMPLATE['overflow_policy'],'K_sensitivity_authorized':[3,5,8,10],
            'K':None,'automatic_K_freeze':False,'shuffled_assignment_audited':False,
            'empirical_evidence':{'validation_businesses':200,'real_BLIP2_captions':True,'maximum_candidate_stress':10,
                'unfiltered_caption_max':265,'filtered_caption_max':254,'unfiltered_source_max':4310,'filtered_source_max':4312,
                'caption_and_source_overflows':0,'independent_token_ID_checks':600,'review_ids_text_order_changed':False,'silent_content_removal_or_truncation':False}}
    path=STATE/'source_budget_manifest.json'
    if path.exists():assert read(path)==result
    else:write(path,result)
    return result


def assert_complete_hard_thresholds(frozen):
    receipt=read(STATE/'compatibility_receipt.json')
    assert frozen['thresholds']==THRESHOLDS and frozen['protocol_sha256']==receipt['previous_D031_protocol_sha256']
    assert sha(BASE/'frozen_thresholds.json')==receipt['inherited_filter_manifest_sha256']


def assert_scoring_binding(role,report):
    r=read(STATE/'compatibility_receipt.json');path=BASE/('scores_'+role)/'scoring_report.json'
    assert sha(path)==r['scoring_reports'][role]['sha256'] and report['config_sha256']==r['original_scoring_protocol_sha256']


if __name__=='__main__':print(validate_compatibility()['status'])
