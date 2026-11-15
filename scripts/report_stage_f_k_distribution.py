"""Report fixed-filter K-study support before sensitivity; never select/change K."""
from pathlib import Path
import csv
import numpy as np
from stage_f_verification_workflow import read, write, sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'


def summarize(businesses, candidates=(3,5,8,10)):
    rows=[]
    for r in businesses:
        if r['role']!='K_study':continue
        row={'business_id':r['business_id'],
             'eligible_captions':len(r.get('relevance_eligible_photo_ids',r.get('clip_relevance_eligible_photo_ids',[]))),
             'post_redundancy_captions':len(r['filtered_diversity_ranking']),
             'coarser_groups':r['available_reference_groups']}
        assert 0<=row['coarser_groups']<=row['post_redundancy_captions']<=row['eligible_captions']
        rows.append(row)
    assert len({r['business_id'] for r in rows})==len(rows)
    summary={}
    for key in ['eligible_captions','post_redundancy_captions','coarser_groups']:
        values=np.array([r[key] for r in rows])
        summary[key]={'businesses':len(rows),'mean':float(values.mean()),'median':float(np.median(values)),
                      'p90':float(np.quantile(values,.9)),'p95':float(np.quantile(values,.95)),
                      'max':int(values.max()),'zero_businesses':int((values==0).sum()),
                      'businesses_below_K':{str(k):int((values<k).sum()) for k in candidates}}
    tail=[r['business_id'] for r in rows if r['coarser_groups']>max(candidates)]
    return {'status':'POST_FREEZE_K_STUDY_DISTRIBUTION_NOT_K_SELECTION','K_candidates':list(candidates),
            'businesses':rows,'summary':summary,'quantiles':'numpy_linear',
            'groups_above_max_K_businesses':tail,'grid_adequacy_review_needed':bool(tail),
            'grid_inadequacy_proven':False,'grid_changed':False,'K_selected':None,
            'review_note':'A group-count tail above 10 is flagged for review; it is not proof the grid is inadequate. Do not add K automatically.'}


def main():
    from stage_f_d031_protocol import active_d031
    if active_d031():
        from report_stage_f_pre_k_d031 import main as current_report
        return current_report()
    from stage_f_d030_protocol import require_validated_amendment,assert_complete_hard_thresholds
    require_validated_amendment()
    frozen=BASE/'frozen_thresholds.json';assert frozen.exists(),'Freeze thresholds before K distributions'
    assert_complete_hard_thresholds(read(frozen))
    pools=read(BASE/'fixed_filter_pools.json');assert pools['threshold_manifest_sha256']==sha(frozen)
    result=summarize(pools['businesses']);assert len(result['businesses'])==160
    result['threshold_manifest_sha256']=sha(frozen)
    write(BASE/'K_study_distribution.json',result)
    with (BASE/'K_study_distribution.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['businesses'][0]));writer.writeheader();writer.writerows(result['businesses'])
    print('K distribution ready; review support before sensitivity. K grid unchanged; budgets still required.')


if __name__=='__main__':main()
