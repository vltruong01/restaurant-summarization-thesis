"""Offline report assembler for FUTURE paired outputs and saved D059epoch3.

No generation, no automatic factual/human labels; refuses absent paired outputs.
"""
from stage_i_unlikelihood_v2 import *


def validate_benchmark(config):
    index=read(PREP/'D059_EPOCH3_PRACTICAL_BENCHMARK.json')
    require(sha(ROOT/index['source_path'])==index['source_sha256'],'D059 epoch3 predictions changed')
    require(sha(ROOT/index['effective_generation_config_path'])==index['effective_generation_config_sha256'],
            'D059 decoding evidence changed')
    require(index['comparison_type']=='NON_CAUSAL_CROSS_ARCHITECTURE_PRACTICAL_BENCHMARK','Benchmark interpretation changed')
    ev=read(PREP/'EVALUATION_MANIFEST.json')
    saved={r['business_id']:r for r in index['predictions']}
    require(len(saved)==24 and set(saved)==set(ev['business_ids']),'Benchmark IDs differ')
    return saved


def main():
    cfg=read(CONFIG);benchmark=validate_benchmark(cfg)
    ev=read(PREP/'EVALUATION_MANIFEST.json')
    arm_rows={}
    for arm in cfg['arms']:
        p=ROOT/cfg['run_root']/arm/'evaluate/predictions.json'
        require(p.exists(),'Paired outputs absent; no placeholder predictions or extra generation')
        rows=read(p);by_id={r['business_id']:r for r in rows}
        require(len(rows)==len(by_id)==24 and set(by_id)==set(ev['business_ids']),'Wrong paired output IDs')
        require(all(r['arm']==arm for r in rows),'Wrong arm')
        arm_rows[arm]=by_id
    records=[]
    for bid in ev['business_ids']:
        rows=[arm_rows[arm][bid] for arm in cfg['arms']]+[benchmark[bid]]
        require(len({r['source_sha256'] for r in rows})==1,'Comparison source mismatch')
        require(all(r['subset']==ev['subsets'][bid] for r in rows),'Comparison subset mismatch')
        records.append({'business_id':bid,'subset':ev['subsets'][bid],'outputs':rows,
            'matched_comparison':['control_ce','treatment_ul'],
            'non_causal_practical_benchmark':'D059 Expanded-LoRA epoch3',
            'human_verified':False,'quality_assessment':'PENDING_RESEARCHER_REVIEW',
            'required_review_axes':['harmful_semantic_and_exact_repetition','Core_and_business_specific_information',
                'positive_negative_opinion_balance','source_factual_grounding','caption_supported_visual_usefulness',
                'zero_caption_compliance','structure_EOS_semantic_completeness'],
            'GO_rule':'Meaningful reduction in harmful repetition without compensating information/grounding/completeness loss; exact repetition alone insufficient'})
    out=ROOT/cfg['run_root']/'quality_comparison'
    require(not out.exists(),'Preserve comparison report; no overwrite')
    write(out/'COMPARISON_RECORDS.json',records)
    write(out/'REVIEW_STATUS.json',{'status':'EVIDENCE_READY_NOT_QUALITY_GO','new_outputs':48,'saved_benchmark_outputs':24,
        'random':20,'targeted':4,'AI_assessments_provisional':True,'human_verified':False,
        'additional_experiment_if_failure':False,'held_out_access':False})


if __name__=='__main__':main()
