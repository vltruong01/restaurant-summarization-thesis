"""D-033 finalK5 production method freeze, preserving D-032 budget and all filter evidence."""
from pathlib import Path
import copy
import yaml
from stage_f_verification_workflow import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/stage_f/methodology_amendment_d033'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
PREVIOUS=ROOT/'outputs/stage_f/methodology_amendment_d032'
CONFIG=ROOT/'configs/stage_f_validation_protocol.yaml'
BASELINE=STATE/'previous_state/configs/stage_f_validation_protocol.yaml'


def active_d033():return yaml.safe_load(CONFIG.read_text(encoding='utf-8')).get('decision')=='D-033'


def validate_compatibility():
    assert active_d033();old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));new=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=copy.deepcopy(old);expected.update(decision='D-033',status='approved_validated_K5_final_production_method_frozen',K=5,train_test_caption_generation=True)
    expected['production_split_policy']={'apply_frozen_input_pipeline':['train','validation','test'],
       'test_input_caption_generation_allowed':True,'test_method_tuning_allowed':False,'test_teacher_generation_allowed':False,'human_test_references_used':False}
    assert new==expected,'Only finalK5 and authorized production input scope may change under D-033'
    parent=read(PREVIOUS/'compatibility_receipt.json');assert parent['active_protocol_sha256']==sha(BASELINE)
    source=read(PREVIOUS/'source_budget_manifest.json');assert source['budgets_finalized'] is True
    study=read(ROOT/'outputs/stage_f/k_sensitivity_d032_v1/K_report.json')
    assert study['rule_implied_K']==5 and study['K_frozen'] is False and study['K_candidates']==[3,5,8,10]
    row=next(r for r in study['K_table'] if r['K']==5)
    assert row['budget_feasible_both_conditions'] and row['relative_coverage_vs_best']>=.95
    assert not next(r for r in study['K_table'] if r['K']==3)['meets95pct_best']
    result={'status':'PASS_D033_FINAL_K5_UNCHANGED_D032_SOURCE_D031_FILTERS',
       'active_protocol_sha256':sha(CONFIG),'previous_D032_protocol_sha256':sha(BASELINE),
       'K':5,'K_frozen':True,'budgets':source['budgets'],'source_budget_manifest_sha256':sha(PREVIOUS/'source_budget_manifest.json'),
       'source_template_sha256':sha(ROOT/'configs/led_source_template_d032.json'),
       'filter_threshold_manifest_sha256':sha(BASE/'frozen_thresholds.json'),
       'caption_thresholds':{'relevance':.08,'redundancy':.59,'grouping':.46},'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only',
       'K_sensitivity_evidence_sha256':sha(ROOT/'outputs/stage_f/k_sensitivity_d032_v1/K_report.json'),
       'rule':'smallest_feasible_at95pct_best','relative_coverage_K5':row['relative_coverage_vs_best'],
       'production_scope':['train','validation','test'],'test_tuning':False,'test_teacher_targets':False,'teacher_or_LED_training':False}
    p=STATE/'compatibility_receipt.json'
    if p.exists():assert read(p)==result
    else:write(p,result)
    return result


def require_validated_amendment():
    r=validate_compatibility();v=read(STATE/'amendment_validation_report.json')
    assert v['status']=='PASS' and v['active_protocol_sha256']==sha(CONFIG)
    return r


def freeze_method():
    r=require_validated_amendment()
    cfg=yaml.safe_load(ROOT.joinpath('configs/stage_f_validation_caption_pool.yaml').read_text(encoding='utf-8'))
    manifest={'status':'D033_APPROVED_VALIDATED_FINAL_PRODUCTION_METHOD_FROZEN','decision':'D-033',
       'K':5,'K_frozen':True,'common_budget':r['budgets'],'source_template_sha256':r['source_template_sha256'],
       'image_duplicate_policy':'exact_normalized_pixel_hard_Filtered_only','pHash':'diagnostic_only','CLIP':'soft_quality_order_only',
       'caption_thresholds':r['caption_thresholds'],'diversity':'unchanged_farthest_first_nested_prefix',
       'unfiltered':'Basic_only_existing_fixed_seed_prefix_without_image_dedup_or_substantive_filters',
       'model_id':cfg['model_id'],'model_revision':cfg['requested_revision'],'generation':cfg['generation'],
       'basic_policy_sha256':sha(ROOT/cfg['basic_quality_policy']),
       'generation_code_sha256':sha(ROOT/'scripts/generate_stage_f_validation_captions.py'),
       'generation_recipe':'reuse_exact_D026_generate_locked_not_new_prompt_or_precision_rule',
       'normalized_raw_provenance_required':True,'all_eligible_photos_before_K_selection':True,
       'role_scope':{'train':2000,'validation':200,'test':100},'zero_caption_businesses_allowed':True,
       'test_input_processing_only_no_tuning_or_teacher_reference':True,'no_teacher_or_LED_training':True,
       'production_complete':False,'protocol_sha256':sha(CONFIG),'compatibility_sha256':sha(STATE/'compatibility_receipt.json')}
    p=STATE/'frozen_production_method.json'
    if p.exists():assert read(p)==manifest
    else:write(p,manifest)
    return manifest


def assert_complete_hard_thresholds(frozen):
    assert frozen['thresholds']=={'relevance':.08,'redundancy':.59,'grouping':.46}


def assert_scoring_binding(role,report):
    parent=read(PREVIOUS/'compatibility_receipt.json');path=BASE/('scores_'+role)/'scoring_report.json'
    assert sha(path)==parent['scoring_reports'][role]['sha256'] and report['config_sha256']==parent['original_scoring_protocol_sha256']


if __name__=='__main__':print(validate_compatibility()['status'])
