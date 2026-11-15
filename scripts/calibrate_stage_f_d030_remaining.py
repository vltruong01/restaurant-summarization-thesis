"""D-030: freeze only the four unchanged hard criteria; no CLIP fit or batch expansion."""
from pathlib import Path
from collections import Counter
import json
import yaml
from stage_f_verification_workflow import read,write,sha,CHOICES,canonical_sha
from analyze_stage_f_consistency_stability import validate_declared_prefix
from stage_f_math import threshold_search
from stage_f_d030_protocol import ROOT,BASE,STATE,CONFIG,HARD_CRITERIA,require_validated_amendment

ASSISTED=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'


def run(labels_path=None):
    compatibility=require_validated_amendment()
    labels_path=Path(labels_path) if labels_path else ASSISTED/'latest_human_export.json'
    assert sha(labels_path)==sha(ASSISTED/'latest_human_export.json'),'Import/confirm a new human state separately before freezing'
    labels=read(labels_path);dataset=read(BASE/'human_check/annotation_dataset.json')
    order=read(ASSISTED/'verification_order.json');selection=read(ASSISTED/'active_selection.json')
    ai=read(ASSISTED/'ai_suggestions.json')
    actual,prefix=validate_declared_prefix(dataset,order,selection,ai,labels)
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')}
    assert all(roles[c['business_id']]=='calibration' for c in dataset['cases'])
    private=read(BASE/'human_check/private_calibration_scores.json')
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));search=cfg['calibration_search']
    assert search['active_hard_criteria']==HARD_CRITERIA
    grids={name:[i/100 for i in range(-200,201)] if name=='relevance' else [i/100 for i in range(101)]
           for name in ['relevance','redundancy','grouping']}
    grids['near_image_duplicate']=search['image_hamming_grid']
    searches=[('relevance','relevance','relevance_margin',True),('redundancy','redundancy','semantic_cosine',True),
              ('grouping','broad_topic','semantic_cosine',True),('near_image_duplicate','image_duplicate','phash_hamming',False)]
    thresholds={name:None for name in HARD_CRITERIA};tables={};blockers=[];counts={};fit_ids={}
    for name,key,score,above in searches:
        verified=[cid for cid,r in actual.items() if r['human_verified'] and key in r['human_final_label']]
        uncertain=[cid for cid in verified if actual[cid]['human_final_label'][key]=='uncertain']
        ids=[cid for cid in verified if cid not in set(uncertain)]
        positive=sum(actual[cid]['human_final_label'][key]==CHOICES[key][0] for cid in ids)
        negative=len(ids)-positive
        counts[name]={'positive':positive,'negative':negative,'uncertain_excluded':len(uncertain)}
        fit_ids[name]=ids
        grid=grids[name]
        if name=='grouping':
            if thresholds['redundancy'] is None:
                blockers.append({'criterion':name,'status':'REDUNDANCY_THRESHOLD_UNRESOLVED'});continue
            grid=[g for g in grid if g<thresholds['redundancy']]
        if not grid or min(positive,negative)<search['minimum_each_binary_class']:
            blockers.append({'criterion':name,'status':'PREDECLARED_GRID_OR_CLASS_REQUIREMENT_NOT_MET','counts':counts[name]});continue
        best,table=threshold_search([private[cid][score] for cid in ids],
                                   [actual[cid]['human_final_label'][key]==CHOICES[key][0] for cid in ids],grid,above,
                                   search['minimum_each_binary_class'])
        tables[name]={'best_candidate':best,'grid_results':table,'direction':'pass_at_or_above' if above else 'duplicate_at_or_below',
                      'verified_case_ids':ids,'uncertain_case_ids_excluded':uncertain,'accepted_for_freeze':best['balanced_accuracy']>.5}
        if best['balanced_accuracy']<=.5:
            blockers.append({'criterion':name,'status':'NO_BETTER_THAN_CHANCE_PREDECLARED_THRESHOLD',
                             'required':'balanced_accuracy_strictly_above0.5','best_candidate_not_frozen':best,
                             'grid_unchanged':grid,'automatic_expansion':False,'automatic_grid_or_acceptance_change':False})
        else:thresholds[name]=best['threshold']
    result={'status':'D030_REMAINING_HARD_THRESHOLDS_FROZEN' if not blockers else 'D030_PARTIAL_HARD_THRESHOLDS_FROZEN_REQUIRED_CRITERION_BLOCKED',
            'decision':'D-030','CLIP_mode':'soft_only','CLIP_hard_rejection':False,'CLIP_consistency_threshold_frozen':False,
            'thresholds':thresholds,'frozen_criteria':[name for name in HARD_CRITERIA if thresholds[name] is not None],
            'unresolved_criteria':[name for name in HARD_CRITERIA if thresholds[name] is None],
            'verified_label_balance':counts,'verified_case_ids_by_criterion':fit_ids,'search_tables':tables,'blockers':blockers,
            'human_labels_sha256':sha(labels_path),'annotation_dataset_sha256':dataset['dataset_sha256'],
            'verification_order_sha256':order['manifest_sha256'],'ai_artifact_sha256':canonical_sha(ai),
            'source_declared_prefix':prefix,'active_selection_unchanged':selection,
            'protocol_sha256':sha(CONFIG),'scoring_protocol_sha256':compatibility['original_scoring_protocol_sha256'],
            'scoring_compatibility_receipt_sha256':sha(STATE/'scoring_compatibility_receipt.json'),
            'verification_workflow_sha256':sha(ROOT/'configs/stage_f_human_verification.yaml'),
            'role_manifest_sha256':sha(BASE/'business_roles.json'),'AI_only_labels_used_for_thresholds':False,
            'K_study_labels_or_results_used':False,'heldout_test_used':False,'K':None,'budgets_finalized':False,
            'case_expansion_performed':False,'source_labels_modified':False,'K_study_filtering_permitted':not blockers}
    destination=STATE/'frozen_remaining_hard_thresholds.json'
    if destination.exists():assert read(destination)==result,'Immutable calibrated thresholds cannot be overwritten'
    else:write(destination,result)
    if not blockers:
        complete=BASE/'frozen_thresholds.json'
        if complete.exists():assert read(complete)==result
        else:write(complete,result)
    else:
        assert not (BASE/'frozen_thresholds.json').exists(),'Do not leave a complete-looking threshold file while a required criterion is blocked'
        write(STATE/'remaining_threshold_blockers.json',{'status':'STOP_BEFORE_K_STUDY_FILTERING_REQUIRED_CRITERION_UNRESOLVED',
                                                       'blockers':blockers,'threshold_manifest_sha256':sha(destination)})
    print(json.dumps({'status':result['status'],'CLIP_mode':result['CLIP_mode'],'thresholds':thresholds,'blockers':blockers}))
    return result


if __name__=='__main__':run()
