"""Pending-policy Source-Separated Shuffled preparation. CPU text/token operations only.

No model/inference entry point. Reuses the existing proposed donor policy on
TRAIN/VALIDATION fixtures; actual TEST assignment awaits researcher adjudication.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1'))
from source_separated_contract import isolated_sources
from led_large_shuffled_production_v2 import build_mapping,sha_text,sha_obj,require

def prepare_mapping(rows,tokenizer,policy):
    require(all(r['split'] in ('train','validation') for r in rows),
            'TEST mapping not committed: donor reuse, fallback, category allowlist and zero policy need researcher approval')
    result=build_mapping(rows,tokenizer,policy)
    byid={r['business_id']:r for r in rows}
    for m in result['mappings']:
        recipient=byid[m['recipient_id']]
        original_review,original_visual=isolated_sources(recipient['source'])
        shuffled_review,shuffled_visual=isolated_sources(m['source'])
        require(shuffled_review==original_review,'Review task changed')
        require('[BUSINESS_INFORMATION]' not in shuffled_visual and '[REV_' not in shuffled_visual,'Visual task contaminated')
        if m['donor_id'] is None:
            require(m['caption_count']==0 and shuffled_visual==original_visual,'Zero-caption input changed')
        else:
            donor=byid[m['donor_id']]
            require(m['donor_id']!=m['recipient_id'] and donor['split']==recipient['split'],'Invalid donor')
            require(shuffled_visual==isolated_sources(donor['source'])[1],'Not exact donor Visual text')
        review_ids=tokenizer.encode(shuffled_review).ids;visual_ids=tokenizer.encode(shuffled_visual).ids
        require(len(review_ids)<=5000 and len(visual_ids)<=750,'Actual task token budget exceeded')
        m.update(evaluation_target_business_id=m['recipient_id'],checkpoint_condition='F-S1000',
            review_task_source_sha256=sha_text(shuffled_review),visual_task_source_sha256=sha_text(shuffled_visual),
            review_task_input_ids_sha256=sha_obj(review_ids),visual_task_input_ids_sha256=sha_obj(visual_ids),
            review_task_source=shuffled_review,visual_task_source=shuffled_visual,
            review_input_identical_to_F=True,recipient_correct_photo_ids=recipient['selected_photo_ids'],
            generated_predictions=False)
    result.update(status='SOURCE_SEPARATED_FIXTURE_ONLY_POLICY_PENDING',no_model_loaded=True,
                  TEST_mapping_created=False,inference_authorized=False,
                  donor_policy_limitation='Existing proposal uses replacement/reuse, not a global permutation; cannot call it distribution-preserving')
    return result
