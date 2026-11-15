"""Exact D-030 config delta and immutable compatibility with pre-amendment scorers."""
from pathlib import Path
import copy
import hashlib
import yaml
from stage_f_verification_workflow import read, write, sha

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/stage_f/methodology_amendment_d030'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
CONFIG=ROOT/'configs/stage_f_validation_protocol.yaml'
BASELINE=STATE/'previous_state/configs/stage_f_validation_protocol.yaml'
HARD_CRITERIA=['relevance','redundancy','grouping','near_image_duplicate']


def active_d030():
    return CONFIG.exists() and yaml.safe_load(CONFIG.read_text(encoding='utf-8')).get('decision')=='D-030'


def assert_exact_authorized_delta(old,new):
    expected=copy.deepcopy(old)
    expected['decision']='D-030'
    expected['status']='CLIP_soft_only_remaining_hard_thresholds_from_verified_calibration'
    expected['CLIP_consistency_policy']={
        'mode':'soft_only','hard_rejection':False,'freeze_consistency_threshold':False,
        'use':'existing_deterministic_quality_order_and_ties_only',
        'interpretation':'broad_semantic_image_caption_alignment_not_factual_correctness',
        'fine_grained_factual_verification':False,
        'evidence_directory':'outputs/stage_f/consistency_conservative_gate_diagnostic_v1'}
    expected['calibration_search']['active_hard_criteria']=HARD_CRITERIA
    expected['calibration_search']['CLIP_grid_purpose']='historical_diagnostic_only_no_production_threshold'
    assert new==expected,'D-030 allows only the explicit CLIP applicability delta; another definition changed'
    assert new['thresholds']['CLIP'] is None and new['quality_order']==old['quality_order']
    return True


def validate_compatibility():
    assert active_d030()
    old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'))
    current=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert_exact_authorized_delta(old,current)
    seal=read(STATE/'protected_before_sha256.json')
    for name in ['scripts/stage_f_score_validation.py','scripts/stage_f_math.py',
                 'configs/stage_f_validation_caption_pool.yaml','configs/stage_f_human_verification.yaml',
                 'outputs/stage_f/validation_calibration_v1/business_roles.json',
                 'outputs/stage_f/validation_calibration_v1/validation_photo_plan.json',
                 'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json']:
        assert sha(ROOT/name)==seal[name],name
    reports={}
    for role in ['calibration','K_study']:
        path=BASE/('scores_'+role)/'scoring_report.json';r=read(path)
        assert r['config_sha256']==sha(BASELINE)
        assert r['models']==current['models'] and r['dtype']=='float32' and r['TF32'] is False
        assert r['code_sha256']==sha(ROOT/'scripts/stage_f_score_validation.py')
        assert r['overlength']==0
        reports[role]={'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),
                       'scoring_protocol_sha256':r['config_sha256'],'scored':r['scored']}
    generation=read(BASE/'generation_report.json')
    assert generation['planned']==generation['completed']==2504
    receipt={'status':'PASS_D030_EXACT_AUTHORIZED_FILTER_POLICY_DELTA_SCORERS_UNCHANGED',
             'active_protocol_path':str(CONFIG.relative_to(ROOT)).replace('\\','/'),
             'active_protocol_sha256':sha(CONFIG),
             'original_scoring_protocol_path':str(BASELINE.relative_to(ROOT)).replace('\\','/'),
             'original_scoring_protocol_sha256':sha(BASELINE),
             'scorer_code_sha256':sha(ROOT/'scripts/stage_f_score_validation.py'),
             'math_code_sha256':sha(ROOT/'scripts/stage_f_math.py'),
             'role_manifest_sha256':sha(BASE/'business_roles.json'),
             'generation_protocol_sha256':sha(ROOT/'configs/stage_f_validation_caption_pool.yaml'),
             'verification_workflow_sha256':sha(ROOT/'configs/stage_f_human_verification.yaml'),
             'scoring_reports':reports,'CLIP_mode':'soft_only','CLIP_threshold_frozen':False,
             'scorer_models_preprocessing_dtype_prototypes_scores_unchanged':True,
             'other_filter_K_budget_definitions_unchanged':True,
             'full_validation_caption_pool_complete':True,
             'K_outcomes_used_for_calibration':False,'heldout_test_used':False}
    path=STATE/'scoring_compatibility_receipt.json'
    if path.exists():assert read(path)==receipt,'Immutable compatibility receipt cannot be replaced'
    else:write(path,receipt)
    return receipt


def require_validated_amendment():
    receipt=validate_compatibility()
    validation=read(STATE/'amendment_validation_report.json')
    assert validation['status']=='PASS' and validation['active_protocol_sha256']==sha(CONFIG)
    assert validation['test_code_sha256']==sha(ROOT/'scripts/test_stage_f_d030.py')
    return receipt


def assert_scoring_binding(role,report):
    receipt=read(STATE/'scoring_compatibility_receipt.json')
    assert receipt['active_protocol_sha256']==sha(CONFIG)
    assert_exact_authorized_delta(yaml.safe_load(BASELINE.read_text(encoding='utf-8')),
                                  yaml.safe_load(CONFIG.read_text(encoding='utf-8')))
    path=BASE/('scores_'+role)/'scoring_report.json'
    assert role in receipt['scoring_reports'] and receipt['scoring_reports'][role]['sha256']==sha(path)
    assert read(path)==report and report['config_sha256']==receipt['original_scoring_protocol_sha256']
    assert sha(BASELINE)==receipt['original_scoring_protocol_sha256'] and report['overlength']==0


def relevance_eligible(rows,indices,threshold):
    # No CLIP cutoff: CLIP enters only the unchanged Stage F quality_order helpers.
    return [i for i in indices if round(float(rows[i]['relevance_margin']),6)>=threshold]


def unfiltered_fixed_seed_order(rows):
    # Same Basic-only fixed-seed ordering as before D-030, without score filtering.
    return sorted(rows,key=lambda r:hashlib.sha256(('42|Unfiltered|'+r['photo_id']).encode()).hexdigest())


def assert_complete_hard_thresholds(frozen):
    assert frozen['CLIP_mode']=='soft_only' and 'CLIP' not in frozen['thresholds']
    assert set(frozen['thresholds'])==set(HARD_CRITERIA)
    assert all(frozen['thresholds'][name] is not None for name in HARD_CRITERIA), 'Required hard threshold still unresolved'
    assert frozen['status']=='D030_REMAINING_HARD_THRESHOLDS_FROZEN'
    assert frozen['protocol_sha256']==sha(CONFIG)
    assert frozen['thresholds']['grouping']<frozen['thresholds']['redundancy']


if __name__=='__main__':
    receipt=validate_compatibility()
    print(receipt['status'])
