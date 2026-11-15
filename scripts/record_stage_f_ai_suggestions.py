"""Record actual assistant visual judgments; no scoring/automatic-label heuristic."""
from pathlib import Path
import argparse
import datetime
import json
import sys
from stage_f_verification_workflow import CHOICES, read, write, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'


def record(payload):
    batches=read(OUT/'ai_review_batches.json');batch=batches[payload['batch']-1]
    assert batch['batch']==payload['batch'] and len(payload['labels'])==len(batch['cases'])
    rows=[]
    for case, symbols in zip(batch['cases'],payload['labels']):
        assert len(symbols)==len(case['label_keys']) and all(s in '+-?' for s in symbols)
        labels={key:CHOICES[key][{'+' :0,'-':1,'?':2}[s]] for key,s in zip(case['label_keys'],symbols)}
        assert not(labels.get('redundancy')=='same_information' and labels.get('broad_topic')=='different_topic')
        rows.append({'case_id':case['case_id'],'ai_suggested_label':labels,
                     'ai_rationale':payload.get('notes',{}).get(str(case['case_index']),
                         'Assistant inspected the displayed original image(s) and caption(s) using the fixed rubric; no scorer values shown.'),
                     'human_final_label':{key:'' for key in case['label_keys']},'human_verified':False})
    artifact={'annotator_type':'AI','annotator':'Codex assistant','model_version':'not_exposed_by_runtime',
              'review_method':'direct_assistant_visual_and_caption_review_no_private_scores',
              'batch':batch['batch'],'business_id':batch['business_id'],
              'gallery_sha256':{p:sha(ROOT/p) for p in batch['panels']},'cases':rows}
    target=OUT/'ai_batch_labels'/f"batch_{batch['batch']:02d}.json"
    if target.exists():assert read(target)==artifact,'Existing AI judgment is immutable; do not overwrite silently'
    else:write(target,artifact)
    all_rows={}
    for p in sorted((OUT/'ai_batch_labels').glob('batch_*.json')):
        for c in read(p)['cases']:
            assert c['case_id'] not in all_rows;all_rows[c['case_id']]=c
    dataset=read(BASE/'human_check/annotation_dataset.json')
    progress={'status':'AI_PREANNOTATION_COMPLETE_HUMAN_VERIFICATION_PENDING' if len(all_rows)==600 else 'AI_PREANNOTATION_IN_PROGRESS',
              'AI_suggested_cases':len(all_rows),'total_cases':600,'human_verified_cases':0,
              'AI_only_used_for_thresholds':False}
    write(OUT/'ai_annotation_progress.json',progress)
    if len(all_rows)==600:
        assert set(all_rows)=={c['case_id'] for c in dataset['cases']}
        result={'schema_version':2,'annotator_type':'AI','annotator':'Codex assistant',
                'model_version':'not_exposed_by_runtime','dataset_sha256':dataset['dataset_sha256'],
                'verification_order_sha256':read(OUT/'verification_order.json')['manifest_sha256'],
                'human_ground_truth':False,'use_for_thresholds_without_human_verification':False,
                'cases':[all_rows[c['case_id']] for c in dataset['cases']]}
        target=OUT/'ai_suggestions.json'
        if target.exists():assert read(target)==result
        else:write(target,result)
    print(json.dumps(progress))


if __name__=='__main__':
    record(json.load(sys.stdin))
