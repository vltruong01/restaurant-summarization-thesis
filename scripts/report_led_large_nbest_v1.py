"""Conditional offline report from saved candidates and source-only judgments.

Oracle judgments are a separate evaluation input, never passed to the reranker.
"""
import argparse
from pathlib import Path
from stage_i_led_large_nbest_v1 import read,write,sha,require,rerank,diversity_gate,verification_audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--judgments',type=Path,required=True);ap.add_argument('--oracle',type=Path,required=True)
    ap.add_argument('--training-assessment',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--judge-cost',type=Path)
    ap.add_argument('--verifier-audit',type=Path)
    args=ap.parse_args();complete=read(args.run/'COMPLETION.json')
    require(complete['status'].startswith('COMPLETE_STOP'),'Failed/incomplete inference cannot enter comparison')
    cases=read(args.cases);case_map={x['business_id']:x for x in cases}
    require(sha(args.cases)==complete['source_only_case_file_sha256'],'Wrong source-only case file')
    if complete['phase']=='training_pilot':
        gate=diversity_gate(read(args.training_assessment),list(case_map))
        require(gate['status'].startswith('PASS'),'Do not prepare/use reranker without TRAIN candidate headroom')
    else:
        # The generation runner already required the completed bound training lock.
        gate={'status':'VALIDATION_AFTER_BOUND_TRAIN_LOCK'}
    judged={x['business_id']:x for x in read(args.judgments)}
    oracle={x['business_id']:x for x in read(args.oracle)}
    records=read(args.run/'CANDIDATES.json')
    require(sha(args.run/'CANDIDATES.json')==complete['candidates_sha256'],'Candidate receipt changed')
    require(set(judged)==set(oracle)==set(case_map)=={r['business_id'] for r in records},'Incomplete/foreign judgments')
    require(not args.output.exists(),'Fresh report namespace required')
    results=[];selected_correct=baseline_correct=regressions=rejected=0
    for r in records:
        bid=r['business_id'];j=judged[bid];gold=oracle[bid];ids={x['candidate_id'] for x in r['candidates']}
        require(gold['assessor_kind']=='researcher' and gold['researcher_confirmed'] is True
                and gold['source_only_review_confirmed'] is True,'Oracle must be separately researcher-confirmed')
        require(set(gold['acceptable_candidate_ids'])<=ids and set(gold['regression_vs_top1_ids'])<=ids,'Foreign oracle IDs')
        require(gold['rationale'].strip(),'Oracle rationale required')
        # This call receives no oracle/reference/Teacher/quality-grade information.
        result=rerank(case_map[bid],r['candidates'],j['inventory'],j['annotations'])
        selected=result['selected_candidate_id'];baseline=r['candidates'][0]['candidate_id']
        selected_correct+=int(selected in gold['acceptable_candidate_ids'])
        baseline_correct+=int(baseline in gold['acceptable_candidate_ids'])
        regressions+=int(selected in gold['regression_vs_top1_ids']);rejected+=int(selected is None)
        results.append(dict(business_id=bid,selection=result,independent_oracle=gold,
                            candidate_diversity=r['diversity'],candidate_generation_seconds=r['candidates_seconds'],
                            original_top1_seconds=r['single_seconds']))
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'PAIRED_RESULTS.json',results)
    n=len(results)
    report=dict(status='DESCRIPTIVE_SOURCE_BASED_REVIEW_NOT_AUTOMATIC_SCIENTIFIC_GO',n=n,
        inference_phase=complete['phase'],training_candidate_headroom=gate,
        selected_in_oracle_acceptable_set=selected_correct,baseline_in_oracle_acceptable_set=baseline_correct,
        reranker_accuracy_all_businesses=selected_correct/n,baseline_oracle_accuracy=baseline_correct/n,
        selected_quality_regressions=regressions,all_rejected_businesses=rejected,
        lexical_diversity_is_not_semantic_diversity=True,
        candidate_generation_cost={'calls':complete['generation_calls'],'returned_sequences':complete['returned_sequences'],
                                  'wall_seconds':complete['seconds']},
        verification_inference_cost=read(args.judge_cost) if args.judge_cost else 'NOT_REPORTED_MUST_MEASURE_SEPARATELY',
        include_all_costs='Generator + evidence inventory/verifier calls + verification audit/researcher minutes; report tokens, latency and actual bill if used.',
        no_rewriting=True,repeated_validation_exposure=True,held_out_access=False)
    write(args.output/'REPORT.json',report)
    if args.verifier_audit:
        audit=read(args.verifier_audit)
        require(complete['phase']=='training_pilot','Use independent TRAIN verifier audit; do not recalibrate from validation')
        write(args.output/'VERIFIER_AUDIT_REPORT.json',dict(metrics=verification_audit(audit,list(case_map)),
              full_verification_cases=audit['claims'],input_sha256=sha(args.verifier_audit),
              FP_definition='supported source claim rejected as error',FN_definition='grounding error accepted as supported'))
    write(args.output/'INPUT_RECEIPT.json',{str(x):sha(x) for x in [args.cases,args.judgments,args.oracle,args.training_assessment,args.run/'CANDIDATES.json',args.run/'COMPLETION.json']})
    print(args.output/'REPORT.json')


if __name__=='__main__':main()
