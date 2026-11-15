"""Freeze approved Stage F definitions and validation-only business roles before labels."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_f/validation_calibration_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def rank(value):
    return hashlib.sha256(('42|stageF_roles_v1|'+value).encode()).hexdigest()


def main():
    from stage_f_d030_protocol import active_d030
    from stage_f_d031_protocol import active_d031
    from stage_f_active_protocol import validate_compatibility
    if active_d030() or active_d031():
        validate_compatibility()
        print('AMENDED_STAGE_F_REUSE_ORIGINAL_ROLES_CAPTIONS_SCORES_NO_REBUILD')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = ROOT/'data/processed/yelp_2300_v1/photo_metadata/validation.jsonl'
    rows = [json.loads(line) for line in metadata.read_text(encoding='utf-8').splitlines()]
    counts = Counter(r['business_id'] for r in rows)
    ids = set(counts)
    assert len(ids) == 200 and len(rows) == len({r['photo_id'] for r in rows}) == 2504
    bands = {bid: '5-7' if n<8 else '8-9' if n<10 else '10-19' if n<20 else '20+' for bid,n in counts.items()}
    parent = {bid:bid for bid in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b):
        a,b = find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    pixels = defaultdict(set); photos = {}
    pool = ROOT/'outputs/stage_d/photo_integrity_duplicate_audit_v1/unfiltered_readable_photo_pool.csv'
    with pool.open(encoding='utf-8', newline='') as f:
        for p in csv.DictReader(f):
            if p['split'] == 'validation' and p['business_id'] in ids:
                photos[p['photo_id']] = p; pixels[p['pixel_sha256']].add(p['business_id'])
    assert set(photos) == {r['photo_id'] for r in rows}
    for group in pixels.values():
        ordered = sorted(group)
        for bid in ordered[1:]: union(ordered[0],bid)
    components = defaultdict(list)
    for bid in sorted(ids): components[find(bid)].append(bid)
    menu_businesses = {r['business_id'] for r in rows if r['label'] == 'menu'}
    ordered_components = sorted(components.values(), key=lambda g: rank('|'.join(g)))
    selected = set(); used = Counter()
    # Pre-label metadata constraint: calibration contains menu evidence where available.
    first = next(g for g in ordered_components if set(g) & menu_businesses)
    selected.update(first); used.update(bands[b] for b in first)
    for group in ordered_components:
        if selected.intersection(group): continue
        add = Counter(bands[b] for b in group)
        if all(used[band]+n <= 10 for band,n in add.items()):
            selected.update(group); used.update(add)
    assert len(selected) == 40 and all(used[b] == 10 for b in ['5-7','8-9','10-19','20+'])
    roles = [{'business_id':bid,'role':'calibration' if bid in selected else 'K_study',
              'raw_photo_count':counts[bid],'raw_photo_band':bands[bid],
              'exact_pixel_business_component':find(bid)} for bid in sorted(ids)]
    role_by_id = {r['business_id']:r['role'] for r in roles}
    overlaps = [key for key,g in pixels.items() if len({role_by_id[b] for b in g})>1]
    assert not overlaps
    label_counts = {role:dict(Counter(r['label'] for r in rows if role_by_id[r['business_id']]==role)) for role in ['calibration','K_study']}
    assert all(label_counts['calibration'].get(label,0)>0 for label in ['food','drink','inside','outside','menu'])
    review = yaml.safe_load((ROOT/'configs/stage_f_scoring_review.yaml').read_text(encoding='utf-8'))
    cfg = {
        'schema_version':1,'decision':'D-027','status':'definitions_and_roles_frozen_thresholds_pending_human_labels',
        'output_directory':OUT.relative_to(ROOT).as_posix(),'seed':42,
        'models':{'CLIP':{'id':'openai/clip-vit-base-patch32','revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','weight_file':'pytorch_model.bin','weight_sha256':'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'},
                  'SBERT':{'id':'sentence-transformers/all-mpnet-base-v2','revision':'e8c3b32edf5434bc2275fc9bab85f82640a19130','weight_file':'model.safetensors','weight_sha256':'78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658'}},
        'score_dtype':'float32','decision_round_decimals':6,
        'CLIP_score':'L2_normalized_image_dot_L2_normalized_caption','CLIP_max_text_tokens':77,
        'SBERT_pooling':'attention_mask_weighted_mean_then_L2_normalize','SBERT_max_text_tokens':384,
        'overlength':'record_unscored_and_stop_threshold_freeze_no_silent_truncation',
        'relevance':{'scope':'restaurant_bar_cafe_domain_not_exact_business_identity','formula':'max_positive_cosine_minus_max_negative_cosine','positive_prototypes':review['business_relevance']['positive_prototypes'],'negative_prototypes':review['business_relevance']['negative_prototypes']},
        'quality_order':['CLIP_desc','relevance_desc','photo_id_ascending'],
        'redundancy':'quality_ranked_greedy_pairwise_threshold_to_each_kept_representative',
        'reference_grouping':'deterministic_complete_link_agglomeration_before_redundancy_then_available_groups_among_kept_representatives',
        'coverage_option':'coarser_reference_groups_separate_from_duplicate_information',
        'diversity':'farthest_first_max_min_one_minus_cosine_quality_ID_ties_nested_prefix',
        'human_sampling':{'image_caption_max':200,'pair_max_per_calibration_business':10,'pair_total_max':400,'image_caption_sampling':'one_seeded_case_per_business_plus_all_calibration_menu_cases_then_round_robin_5x5_CLIP_relevance_score_quantile_cells','pair_sampling':'five_semantic_rank_and_five_phash_rank_quantile_pairs_per_business_up_to10'},
        'calibration_search':{'CLIP_grid':{'start':-1.0,'stop':1.0,'step':0.01},'relevance_grid':{'start':-2.0,'stop':2.0,'step':0.01},'redundancy_grid':{'start':0.0,'stop':1.0,'step':0.01},'grouping_grid':{'start':0.0,'stop':1.0,'step':0.01},'criterion':'maximum_balanced_accuracy','ties':['higher_positive_precision','higher_positive_recall','higher_threshold'],'grouping_after_redundancy_threshold':'choose_best_group_threshold_strictly_below_redundancy_threshold','uncertain_labels':'excluded_from_binary_optimization_report_separately','minimum_each_binary_class':10,'image_hamming_grid':[0,4,8,12],'minimum_balanced_accuracy':'strictly_above0.5'},
        'image_duplicate_algorithm':'quality_ranked_exact_pixel_or_calibrated_phash_greedy_representative_no_component_chaining',
        'human_rubric':{'broad_topic':'caption_accessible_primary_visual_topic_not_photo_only_details; prepared_food_family,beverages,interior_seating,exterior_signage,menu_promotions,service_patrons; different_dish_details_can_be_complementary_same_topic','same_information_implies_not_different_broad_topic':True,'relevance_scope':'restaurant_bar_cafe_domain_not_named_business_identity'},
        'thresholds':{'CLIP':None,'relevance':None,'redundancy':None,'grouping':None,'near_image_duplicate':None},
        'K_candidates':[3,5,8,10],'K_primary':'smallest_budget_feasible_at_95_percent_best_K_study_macro_coverage','K':None,
        'budget':{'caption_tokens':None,'full_source_tokens':None,'diagnostic_total_target':5000},
        'heldout_test_used':False,'human_final30_evaluation_is_separate':True,
        'train_test_caption_generation':False,'final_manifest':False}
    cfg_path = ROOT/'configs/stage_f_validation_protocol.yaml'
    if cfg_path.exists(): assert yaml.safe_load(cfg_path.read_text()) == cfg
    else: cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),encoding='utf-8')
    business_names = {}
    with (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open(encoding='utf-8') as f:
        for line in f:
            b = json.loads(line)
            if b['business_id'] in ids: business_names[b['business_id']] = b['name']
    plan = []
    for row in rows:
        p = photos[row['photo_id']]
        plan.append({**p,'name':business_names[p['business_id']],'role':role_by_id[p['business_id']],
                    'image_path':'data/processed/yelp_2300_v1/'+p['relative_path'],
                    'caption_id':'VAL_'+p['photo_id'],'final_caption_set':False})
    plan.sort(key=lambda p:(p['role']!='calibration',rank(p['business_id']),rank(p['photo_id'])))
    report = {'status':'ROLES_DEFINITIONS_FROZEN_NO_HUMAN_THRESHOLDS_OR_K','businesses':200,'photos':2504,
              'calibration_businesses':40,'K_study_businesses':160,'calibration_per_photo_band':dict(used),
              'role_photo_counts':dict(Counter(p['role'] for p in plan)),'label_counts_by_role':label_counts,
              'exact_pixel_cross_role_groups':0,'near_or_pretraining_independence_claimed':False,
              'component_sizes':dict(Counter(len(g) for g in components.values())),
              'roles_are_internal_original_dataset_splits_unchanged':True,'human_labels_collected':0,
              'thresholds_K_budgets_frozen':False,'metadata_source_sha256':sha(metadata),'config_sha256':sha(cfg_path)}
    for name,data in [('business_roles.json',roles),('validation_photo_plan.json',plan),('preparation_report.json',report)]:
        data=json.loads(json.dumps(data,ensure_ascii=False))
        path = OUT/name
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            assert previous==data,(name,[k for k in data if previous.get(k)!=data[k]] if isinstance(data,dict) else 'manifest differs')
        else: path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
