"""Separate D-028 Accept/Correct UI; immutable original queue is never rebuilt."""
from pathlib import Path
import argparse
import base64
import json
import yaml
from stage_f_verification_workflow import read, write, canonical_sha, active_ids, sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'
OUT=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'


def build(ui_only=False):
    dataset=read(BASE/'human_check/annotation_dataset.json');ai=read(OUT/'ai_suggestions.json')
    order=read(OUT/'verification_order.json');selection=read(OUT/'active_selection.json')
    assert len(ai['cases'])==600 and ai['human_ground_truth'] is False
    lookup={c['case_id']:c for c in ai['cases']}
    data={k:v for k,v in dataset.items() if k!='cases'}
    data.update(workflow='AI_assisted_human_verification',verification_order_sha256=order['manifest_sha256'],
                ai_artifact_sha256=canonical_sha(ai),active_case_ids=active_ids(order,selection),
                selection=selection,cases=[{**c,**lookup[c['case_id']]} for c in dataset['cases']])
    images={}
    for c in dataset['cases']:
        if c['case_id'] not in data['active_case_ids']:continue
        for pid,path in zip(c['photo_ids'],c['image_paths']):
            if pid not in images:images[pid]='data:image/jpeg;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode('ascii')
    data['images']=images
    secondary_path=OUT/'active_secondary_review.json'
    if secondary_path.exists():
        secondary=read(secondary_path)
        assert secondary['status']=='EXTERNAL_CHATGPT_SECOND_REVIEW_NOT_HUMAN_GROUND_TRUTH'
        source=ROOT/secondary['source_path']
        import hashlib
        assert hashlib.sha256(source.read_bytes()).hexdigest()==secondary['source_sha256']
        assert set(secondary['cases']).issubset(set(data['active_case_ids']))
        data['secondary_reviews']=secondary['cases']
        data['secondary_review_sha256']=secondary['source_sha256']
    data['imported_human_state']=read(OUT/'latest_human_export.json') if (OUT/'latest_human_export.json').exists() else None
    data['imported_human_export_sha256']=sha(OUT/'latest_human_export.json') if data['imported_human_state'] else None
    refinement=ROOT/'configs/stage_f_consistency_stability.yaml'
    if refinement.is_file():
        data['consistency_refinement']=yaml.safe_load(refinement.read_text(encoding='utf-8'))
    data['CLIP_filtering_policy']=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8')).get('CLIP_consistency_policy')
    data['image_duplicate_policy']=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8')).get('image_duplicate_policy')
    imported=data['imported_human_state']
    verified={c['case_id'] for c in imported['cases'] if c['human_verified']} if imported else set()
    verified.intersection_update(data['active_case_ids'])
    template=Path(__file__).with_name('stage_f_assisted_human_check_template.html').read_text(encoding='utf-8')
    target=OUT/'human_check.html'
    target.write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    if ui_only:
        print('ASSISTED_UI_WORDING_UPDATED_ONLY',target)
        return
    write(OUT/'ui_build_report.json',{'status':'AI_ASSISTED_UI_READY_HUMAN_VERIFICATION_PENDING',
          'active_image_cases':selection['image_cases'],'active_pair_cases':selection['pair_cases'],
          'actual_human_verified_cases':len(verified),'pending_active_cases':len(data['active_case_ids'])-len(verified),
          'second_review_suggestions':len(data.get('secondary_reviews',{})),
          'AI_suggestions':600,'original_queue_sha256':dataset['dataset_sha256'],
          'AI_only_used_for_thresholds':False,'independent_human_annotation':False})
    progress=read(OUT/'ai_annotation_progress.json')
    progress.update(human_verified_cases=len(verified),active_human_verification_cases=len(data['active_case_ids']),
                    pending_active_cases=len(data['active_case_ids'])-len(verified))
    write(OUT/'ai_annotation_progress.json',progress)
    print('ASSISTED_UI_READY',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ui-only',action='store_true',help='Refresh presentation only; preserve scientific reports/progress.')
    build(ui_only=parser.parse_args().ui_only)
