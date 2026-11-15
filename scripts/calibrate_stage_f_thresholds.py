"""D-028: fit only explicitly verified human-final labels; never AI pseudo-labels."""
from pathlib import Path
import argparse
import shutil
import yaml
from stage_f_math import threshold_search
from stage_f_verification_workflow import (CHOICES, read, write, sha, canonical_sha,
                                           active_ids, validate_export, balance_and_expansion)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
OUT=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--import-only',action='store_true');args=parser.parse_args()
    from stage_f_d033_protocol import active_d033,freeze_method
    if active_d033() and not args.import_only:
        freeze_method()
        return
    from stage_f_d032_protocol import active_d032,freeze_budget_manifest
    if active_d032() and not args.import_only:
        freeze_budget_manifest()
        return
    from stage_f_d031_protocol import active_d031,carry_frozen_caption_thresholds
    if active_d031() and not args.import_only:
        carry_frozen_caption_thresholds(args.labels)
        return
    from stage_f_d030_protocol import active_d030
    if active_d030() and not args.import_only:
        from calibrate_stage_f_d030_remaining import run as freeze_remaining
        freeze_remaining(args.labels)
        return
    from analyze_stage_f_consistency_stability import active_refinement, run
    if active_refinement() and not args.import_only:
        # D-029 supersedes the historic automatic balance-expansion/freeze path.
        # Analyze without importing/replacing even the authoritative human JSON.
        run(args.labels)
        return
    dataset=read(BASE/'human_check/annotation_dataset.json');labels=read(args.labels)
    order=read(OUT/'verification_order.json');selection=read(OUT/'active_selection.json')
    ai=read(OUT/'ai_suggestions.json')
    workflow_path=ROOT/'configs/stage_f_human_verification.yaml'
    workflow=yaml.safe_load(workflow_path.read_text(encoding='utf-8'))
    assert order['workflow_sha256']==sha(workflow_path)
    assert order['role_manifest_sha256']==sha(BASE/'business_roles.json')
    assert order['private_stratification_scores_sha256']==sha(BASE/'human_check/private_calibration_scores.json')
    actual=validate_export(dataset,order,selection,ai,labels)
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')}
    assert all(roles[c['business_id']]=='calibration' for c in dataset['cases'])
    copy=OUT/'human_labels'/(sha(args.labels)+'.json');copy.parent.mkdir(exist_ok=True)
    if not copy.exists():shutil.copy2(args.labels,copy)
    if active_refinement() or active_d030() or active_d031() or active_d032() or active_d033():
        from collections import Counter
        counts={key:Counter(positive=0,negative=0,uncertain=0) for key in CHOICES}
        for row in actual.values():
            if not row['human_verified']:continue
            for key,label in row['human_final_label'].items():
                category='uncertain' if label=='uncertain' else 'positive' if label==CHOICES[key][0] else 'negative'
                counts[key][category]+=1
        balance={'status':('D030_IMPORTED_ONLY_NO_CONSISTENCY_THRESHOLD_OR_EXPANSION' if active_d030() else 'D029_IMPORTED_ONLY_STABILITY_REVIEW_REQUIRED_NO_EXPANSION_OR_FREEZE'),
                 'counts':{k:dict(v) for k,v in counts.items()},'expanded':False,
                 'selection':dict(selection),'automatic_threshold_freeze':False}
    else:
        balance=balance_and_expansion(dataset,order,selection,actual,workflow)
    report={'workflow':'AI_assisted_human_verification_not_independent_annotation',
            'annotator':labels['annotator'],'human_export_sha256':sha(args.labels),
            'AI_labels_used_directly_for_fitting':False,'K_study_labels_used':False,
            'final_30_business_evaluation':False,**balance}
    write(OUT/'human_verification_import_report.json',report)
    write(OUT/'latest_human_export.json',labels)
    if args.import_only:
        print(report['status']);return
    if balance['expanded']:
        next_selection=balance['selection'];next_selection['round']=selection['round']+1
        write(OUT/'selection_history'/f"round_{selection['round']:02d}.json",
              {'previous_selection':selection,'human_export_sha256':sha(args.labels),
               'reason':'verified_class_balance_only','balance':balance})
        write(OUT/'active_selection.json',next_selection)
        from build_stage_f_assisted_human_check import build
        build()
        print('EXPANDED_FOR_CLASS_BALANCE: reopen assisted UI and verify newly selected cases');return
    if balance['status']!='BALANCED_VERIFIED_LABELS_READY_FOR_THRESHOLD_SEARCH':
        print(balance['status']);return
    generation=read(BASE/'generation_report.json')
    assert generation['completed']==generation['planned']==2504
    scoring_cfg=ROOT/'configs/stage_f_validation_protocol.yaml'
    cfg=yaml.safe_load(scoring_cfg.read_text(encoding='utf-8'))
    for role in ['calibration','K_study']:
        report=read(BASE/('scores_'+role)/'scoring_report.json')
        assert report['overlength']==0 and report['config_sha256']==sha(scoring_cfg)
    private=read(BASE/'human_check/private_calibration_scores.json')
    chosen={};tables={};ids=active_ids(order,selection)
    searches=[('CLIP','consistency','clip_score',True,[i/100 for i in range(-100,101)]),
              ('relevance','relevance','relevance_margin',True,[i/100 for i in range(-200,201)]),
              ('redundancy','redundancy','semantic_cosine',True,[i/100 for i in range(101)]),
              ('grouping','broad_topic','semantic_cosine',True,[i/100 for i in range(101)]),
              ('near_image_duplicate','image_duplicate','phash_hamming',False,[0,4,8,12])]
    for parameter,key,score,above,grid in searches:
        values=[];truth=[]
        for cid in ids:
            row=actual[cid]
            assert row['human_verified'] is True
            label=row['human_final_label'].get(key)
            if label and label!='uncertain':values.append(private[cid][score]);truth.append(label==CHOICES[key][0])
        if parameter=='grouping':grid=[g for g in grid if g<chosen['redundancy']]
        if not grid:
            write(OUT/'threshold_search_blocker.json',{'status':'NO_GROUPING_THRESHOLD_BELOW_REDUNDANCY','expansion_allowed':False});return
        best,table=threshold_search(values,truth,grid,above,cfg['calibration_search']['minimum_each_binary_class'])
        if best['balanced_accuracy']<=.5:
            write(OUT/'threshold_search_blocker.json',{'status':'NO_BETTER_THAN_CHANCE_PREDECLARED_THRESHOLD',
                  'criterion':parameter,'best':best,'expansion_allowed':False});return
        chosen[parameter]=best['threshold'];tables[parameter]={'chosen':best,'grid_results':table}
    result={'status':'THRESHOLDS_FROZEN_FROM_ACTUAL_CALIBRATION_LABELS_NOT_K_SELECTION',
            'annotation_method':'AI_assisted_preannotation_with_human_verification_not_independent',
            'thresholds':chosen,'human_labels_sha256':sha(args.labels),
            'annotation_dataset_sha256':dataset['dataset_sha256'],
            'protocol_sha256':sha(scoring_cfg),'verification_workflow_sha256':sha(workflow_path),
            'verification_order_sha256':order['manifest_sha256'],'ai_artifact_sha256':canonical_sha(ai),
            'active_verified_case_ids':ids,'verified_label_balance':balance['counts'],
            'role_manifest_sha256':sha(BASE/'business_roles.json'),'calibration_businesses':40,
            'K_study_labels_or_results_used':False,'AI_only_labels_used_for_thresholds':False,
            'search_tables':tables,'K':None,'budgets_finalized':False}
    destination=BASE/'frozen_thresholds.json'
    if destination.exists():assert read(destination)==result,'Frozen thresholds cannot be replaced'
    else:write(destination,result)
    print('THRESHOLDS_FROZEN_FROM_VERIFIED_HUMAN_FINAL_LABELS; K/budgets pending')


if __name__=='__main__':main()
