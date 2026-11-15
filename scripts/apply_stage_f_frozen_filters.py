"""D-031 fixed filtering on160K-study businesses; no K sensitivity or source mutation."""
from pathlib import Path
from collections import defaultdict
import os
for thread_option in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(thread_option,'1')
import numpy as np
import yaml
from safetensors.numpy import load_file
from stage_f_math import redundancy_reduce,complete_link_groups,farthest_first
from stage_f_verification_workflow import read,write,sha
from stage_f_active_protocol import require_validated_amendment,assert_scoring_binding,assert_complete_hard_thresholds,relevance_eligible,unfiltered_fixed_seed_order
from stage_f_d031_protocol import exact_duplicate_reduce

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/validation_calibration_v1'
STATE=ROOT/'outputs/stage_f/methodology_amendment_d031'


def filter_business(bid,all_rows,vectors,plan,t):
    basic=sorted([r for r in all_rows if r['basic_eligible']],key=lambda r:r['photo_id'])
    assert all(r['score_status']=='scored' for r in basic)
    unfiltered=unfiltered_fixed_seed_order(basic)
    pixels=[plan[r['photo_id']]['pixel_sha256'] for r in basic]
    exact_kept,exact_map=exact_duplicate_reduce(basic,pixels)
    eligible=relevance_eligible(basic,exact_kept,t['relevance'])
    pool=[basic[i] for i in eligible]
    if pool:
        embedded=vectors[[r['embedding_index'] for r in pool]];matrix=embedded@embedded.T
        groups=complete_link_groups(pool,matrix,t['grouping'])
        kept,red_map=redundancy_reduce(pool,matrix,t['redundancy'])
        ranking=farthest_first(pool,matrix,kept)
        available=sorted({groups[i] for i in kept})
    else:groups={};kept=[];red_map={};ranking=[];available=[]
    result={'business_id':bid,'role':'K_study','raw_pool_count':len(all_rows),'basic_pool_count':len(basic),
            'after_exact_duplicate_count':len(exact_kept),'after_relevance_count':len(pool),'post_redundancy_count':len(kept),
            'unfiltered_fixed_seed_ranking':[r['photo_id'] for r in unfiltered],
            'exact_duplicate_kept_photo_ids':[basic[i]['photo_id'] for i in exact_kept],
            'image_duplicate_removed_to_representative':{basic[i]['photo_id']:basic[j]['photo_id'] for i,j in exact_map.items()},
            'relevance_eligible_photo_ids':[r['photo_id'] for r in pool],
            'reference_group_by_photo_id':{pool[i]['photo_id']:g for i,g in groups.items()},
            'reference_group_count_before_redundancy':len(set(groups.values())),
            'semantic_removed_to_representative':{pool[i]['photo_id']:pool[j]['photo_id'] for i,j in red_map.items()},
            'filtered_diversity_ranking':[pool[i]['photo_id'] for i in ranking],
            'available_reference_group_ids':available,'available_reference_groups':len(available),
            'empty_filtered_pool':not available,'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only',
            'pHash_hard_rejection':False,'CLIP_hard_rejection':False,
            'diagnostic_phash64_by_basic_photo_id':{r['photo_id']:plan[r['photo_id']]['phash64_hex'] for r in basic},
            'K':None,'final_caption_set':False,'source_captions_modified':False}
    assert 0<=len(available)<=len(kept)<=len(pool)<=len(exact_kept)<=len(basic)<=len(all_rows)
    return result


def main():
    require_validated_amendment()
    if not (OUT/'frozen_thresholds.json').exists():
        raise RuntimeError('required hard threshold unresolved; validated frozen manifest required before filtering')
    frozen=read(OUT/'frozen_thresholds.json');assert_complete_hard_thresholds(frozen)
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
    assert cfg['decision']=='D-031' and frozen['role_manifest_sha256']==sha(OUT/'business_roles.json')
    assert frozen['verification_workflow_sha256']==sha(ROOT/'configs/stage_f_human_verification.yaml')
    assert frozen['AI_only_labels_used_for_thresholds'] is False
    roles={r['business_id']:r['role'] for r in read(OUT/'business_roles.json')}
    folder=OUT/'scores_K_study';report=read(folder/'scoring_report.json');assert_scoring_binding('K_study',report)
    inputs=[folder/'caption_scores.json',folder/'embeddings.safetensors',OUT/'validation_photo_plan.json',OUT/'business_roles.json']
    before={str(p):sha(p) for p in inputs}
    scores=read(folder/'caption_scores.json');plan={r['photo_id']:r for r in read(OUT/'validation_photo_plan.json')}
    vectors=load_file(str(folder/'embeddings.safetensors'))['SBERT']
    assert vectors.dtype==np.float32 and np.isfinite(vectors).all()
    by_business=defaultdict(list)
    for r in scores:
        assert r['split']=='validation' and r['role']=='K_study' and roles[r['business_id']]=='K_study'
        assert plan[r['photo_id']]['role']=='K_study' and plan[r['photo_id']]['raw_sha256']==r['raw_image_sha256']
        by_business[r['business_id']].append(r)
    assert set(by_business)=={bid for bid,role in roles.items() if role=='K_study'} and len(by_business)==160
    results=[filter_business(bid,rows,vectors,plan,frozen['thresholds']) for bid,rows in sorted(by_business.items())]
    assert before=={str(p):sha(p) for p in inputs}
    data={'status':'D031_FIXED_160_BUSINESS_FILTER_POOLS_NOT_K_SELECTION','decision':'D-031',
          'threshold_manifest_sha256':sha(OUT/'frozen_thresholds.json'),'protocol_sha256':sha(ROOT/'configs/stage_f_validation_protocol.yaml'),
          'businesses':results,'roles_processed':['K_study'],'business_count':160,'original_businesses_excluded':0,
          'CLIP_mode':'soft_only','pHash_mode':'diagnostic_only','K':None,'source_budget':cfg['budget'],
          'source_input_sha256':before,'source_inputs_byte_unchanged':True,'K_sensitivity_run':False,'teacher_training':False}
    for path in [STATE/'fixed_filter_pools.json',OUT/'fixed_filter_pools.json']:
        if path.exists():assert read(path)==data,'Do not replace an existing fixed-pool result'
        else:write(path,data)
    from report_stage_f_k_distribution import main as report_distribution
    report_distribution()
    print('D031_160_FIXED_POOLS_AND_PREK_DISTRIBUTIONS_READY_STOP_BEFORE_K_SENSITIVITY')
    return data


if __name__=='__main__':main()
