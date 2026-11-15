"""D-031 exact-pixel hard removal and pHash diagnostic-only; reuse frozen scores/thresholds."""
from pathlib import Path
import copy
import yaml
from stage_f_verification_workflow import read,write,sha
from stage_f_math import quality_order

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
STATE=ROOT/'outputs/stage_f/methodology_amendment_d031'
PREVIOUS=ROOT/'outputs/stage_f/methodology_amendment_d030'
CONFIG=ROOT/'configs/stage_f_validation_protocol.yaml'
BASELINE=STATE/'previous_state/configs/stage_f_validation_protocol.yaml'
ORIGINAL=PREVIOUS/'previous_state/configs/stage_f_validation_protocol.yaml'
THRESHOLDS={'relevance':.08,'redundancy':.59,'grouping':.46}


def active_d031():
    return CONFIG.exists() and yaml.safe_load(CONFIG.read_text(encoding='utf-8')).get('decision')=='D-031'


def assert_exact_authorized_delta(old,new):
    expected=copy.deepcopy(old)
    expected['decision']='D-031'
    expected['status']='exact_pixel_hard_phash_diagnostic_only_caption_thresholds_frozen'
    expected['image_duplicate_algorithm']='quality_ranked_exact_normalized_pixel_hash_representative_only'
    expected['image_duplicate_policy']={
        'exact_normalized_pixel_hard_removal':True,'representative':'existing_CLIP_relevance_photo_ID_quality_order',
        'phash_role':'diagnostic_only','phash_hard_rejection':False,'freeze_phash_threshold':False,
        'phash_collision_is_not_exact_duplicate':True,'evidence_directory':'outputs/stage_f/phash_pair_diagnostic_v1'}
    expected['calibration_search']['active_hard_criteria']=list(THRESHOLDS)
    expected['calibration_search']['image_hamming_grid_purpose']='historical_diagnostic_only_no_production_cutoff'
    expected['thresholds'].update(THRESHOLDS)
    assert new==expected,'Only the authorized exact/pHash applicability delta and carried caption threshold values are allowed'
    assert new['thresholds']['near_image_duplicate'] is None and new['thresholds']['CLIP'] is None
    assert new['calibration_search']['image_hamming_grid']==[0,4,8,12]
    return True


def validate_compatibility():
    assert active_d031()
    old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert_exact_authorized_delta(old,cfg)
    original=yaml.safe_load(ORIGINAL.read_text(encoding='utf-8'))
    old_receipt=read(PREVIOUS/'scoring_compatibility_receipt.json')
    assert old_receipt['active_protocol_sha256']==sha(BASELINE)
    assert old_receipt['original_scoring_protocol_sha256']==sha(ORIGINAL)
    assert cfg['models']==original['models'] and cfg['relevance']==original['relevance']
    seal=read(STATE/'protected_before_sha256.json')
    for name in ['scripts/stage_f_score_validation.py','scripts/stage_f_math.py',
                 'configs/stage_f_validation_caption_pool.yaml','configs/stage_f_human_verification.yaml',
                 'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json',
                 'outputs/stage_f/validation_calibration_v1/business_roles.json',
                 'outputs/stage_f/validation_calibration_v1/validation_photo_plan.json']:
        assert sha(ROOT/name)==seal[name],name
    reports={}
    for role in ['calibration','K_study']:
        path=BASE/('scores_'+role)/'scoring_report.json';r=read(path)
        assert r['config_sha256']==sha(ORIGINAL) and r['models']==cfg['models']
        assert r['dtype']=='float32' and r['TF32'] is False and r['overlength']==0
        assert r['code_sha256']==sha(ROOT/'scripts/stage_f_score_validation.py')
        reports[role]={'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path)}
    parent=read(PREVIOUS/'frozen_remaining_hard_thresholds.json')
    assert {k:parent['thresholds'][k] for k in THRESHOLDS}==THRESHOLDS
    assert parent['human_labels_sha256']==sha(ROOT/'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json')
    receipt={'status':'PASS_D031_EXACT_AUTHORIZED_DELTA_UNCHANGED_SCORERS_CAPTION_THRESHOLDS',
             'active_protocol_sha256':sha(CONFIG),'previous_D030_protocol_sha256':sha(BASELINE),
             'original_scoring_protocol_path':str(ORIGINAL.relative_to(ROOT)).replace('\\','/'),
             'original_scoring_protocol_sha256':sha(ORIGINAL),
             'previous_compatibility_receipt_sha256':sha(PREVIOUS/'scoring_compatibility_receipt.json'),
             'parent_caption_threshold_manifest_sha256':sha(PREVIOUS/'frozen_remaining_hard_thresholds.json'),
             'scorer_code_sha256':sha(ROOT/'scripts/stage_f_score_validation.py'),
             'math_code_sha256':sha(ROOT/'scripts/stage_f_math.py'),
             'models_preprocessing_dtype_prototypes_embeddings_scores_unchanged':True,
             'caption_thresholds_unchanged':THRESHOLDS,'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only',
             'exact_duplicate_rule':'normalized_pixel_hash_equality_only_existing_quality_order',
             'role_manifest_sha256':sha(BASE/'business_roles.json'),'scoring_reports':reports,
             'human_labels_unchanged':True,'new_threshold_search':False,'heldout_test_used':False}
    path=STATE/'scoring_compatibility_receipt.json'
    if path.exists():assert read(path)==receipt
    else:write(path,receipt)
    return receipt


def require_validated_amendment():
    receipt=validate_compatibility();v=read(STATE/'amendment_validation_report.json')
    assert v['status']=='PASS' and v['active_protocol_sha256']==sha(CONFIG)
    assert v['test_code_sha256']==sha(ROOT/'scripts/test_stage_f_d031.py')
    assert v['protocol_code_sha256']==sha(Path(__file__))
    return receipt


def assert_scoring_binding(role,report):
    receipt=read(STATE/'scoring_compatibility_receipt.json')
    assert receipt['active_protocol_sha256']==sha(CONFIG)
    path=BASE/('scores_'+role)/'scoring_report.json'
    assert sha(path)==receipt['scoring_reports'][role]['sha256'] and read(path)==report
    assert report['config_sha256']==receipt['original_scoring_protocol_sha256'] and report['overlength']==0


def exact_duplicate_reduce(rows,pixel_hashes):
    assert len(rows)==len(pixel_hashes) and all(isinstance(h,str) and h for h in pixel_hashes)
    representatives={};kept=[];removed={}
    for i in quality_order(rows):
        pixel=pixel_hashes[i]
        if pixel not in representatives:representatives[pixel]=i;kept.append(i)
        else:removed[i]=representatives[pixel]
    return kept,removed


def assert_complete_hard_thresholds(frozen):
    assert frozen['status']=='D031_FROZEN_CAPTION_THRESHOLDS_EXACT_PIXEL_HARD_PHASH_DIAGNOSTIC_ONLY'
    assert frozen['protocol_sha256']==sha(CONFIG) and frozen['thresholds']==THRESHOLDS
    assert frozen['CLIP_mode']=='soft_only' and frozen['pHash_mode']=='diagnostic_only'
    assert frozen['image_duplicate_rule']=='exact_normalized_pixel_hash_only'
    assert 'CLIP' not in frozen['thresholds'] and 'near_image_duplicate' not in frozen['thresholds']
    assert frozen['thresholds']['grouping']<frozen['thresholds']['redundancy']


def carry_frozen_caption_thresholds(labels_path=None):
    compatibility=require_validated_amendment()
    current=ROOT/'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json'
    if labels_path is not None:assert sha(Path(labels_path))==sha(current)
    parent=read(PREVIOUS/'frozen_remaining_hard_thresholds.json')
    result={'status':'D031_FROZEN_CAPTION_THRESHOLDS_EXACT_PIXEL_HARD_PHASH_DIAGNOSTIC_ONLY',
            'decision':'D-031','thresholds':THRESHOLDS,'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only',
            'image_duplicate_rule':'exact_normalized_pixel_hash_only','CLIP_production_threshold':None,'pHash_production_threshold':None,
            'protocol_sha256':sha(CONFIG),'scoring_protocol_sha256':compatibility['original_scoring_protocol_sha256'],
            'scoring_compatibility_receipt_sha256':sha(STATE/'scoring_compatibility_receipt.json'),
            'parent_D030_manifest_sha256':sha(PREVIOUS/'frozen_remaining_hard_thresholds.json'),
            'human_labels_sha256':sha(current),'role_manifest_sha256':parent['role_manifest_sha256'],
            'verification_workflow_sha256':parent['verification_workflow_sha256'],
            'AI_only_labels_used_for_thresholds':False,'K_study_labels_or_results_used_to_fit_thresholds':False,
            'recalibration_performed':False,'verified_label_balance':{k:parent['verified_label_balance'][k] for k in THRESHOLDS},
            'search_tables':{k:parent['search_tables'][k] for k in THRESHOLDS},
            'heldout_test_used':False,'K':None,'budgets_finalized':False,'K_sensitivity_permitted':False,
            'authorized_execution':'fixed_pipeline_160_K_study_businesses_preK_distribution_only_then_stop_for_review'}
    assert_complete_hard_thresholds(result)
    for path in [STATE/'frozen_thresholds.json',BASE/'frozen_thresholds.json']:
        if path.exists():assert read(path)==result,'Do not replace an existing frozen threshold manifest'
        else:write(path,result)
    print('D031_CAPTION_THRESHOLDS_CARRIED_UNCHANGED_NO_CLIP_OR_PHASH_CUTOFF')
    return result


if __name__=='__main__':print(validate_compatibility()['status'])
