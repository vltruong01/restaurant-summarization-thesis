"""Finalize evidence-linked pilot comparison and preservation/usage inventory; no model calls."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import shutil

from prepare_stage_g_teacher_pilot import ROOT,read,file_sha,canonical,digest
from run_stage_g_teacher_pilot_v4_1_recovery import validate_output,payload,response_text

NEW=ROOT/'outputs/stage_g/teacher_pilot_v4_1_grounding_d038_v1'
EXEC=NEW/'execution_v2_quote_recovery'


def main():
    machine=read(NEW/'machine_comparison.json')
    audit=read(EXEC/'valid_targets_integrity_and_tokens_report.json')
    inventory=read(EXEC/'completion_20_attempted_businesses.json')
    notes=read(NEW/'ai_semantic_comparison_findings.json')
    photos=read(NEW/'ai_photo_inspection_notes.json')
    cfg=read(ROOT/'configs/stage_g_teacher_pilot_v4_1_grounding.json')
    jobs=read(NEW/'pilot_jobs.json');schemas=read(ROOT/cfg['schema'])
    assert inventory['task_slots_accounted_for']==60 and inventory['businesses_attempted']==20
    assert photos['unique_photos_inspected']==128 and photos['businesses_covered']==20
    controls=read(EXEC/'collection_control_manifest.json')
    assert file_sha(ROOT/'scripts/collect_stage_g_remaining_pilot_tasks.py')==controls['collector_sha256']
    checked=0;failed=[]
    for job in jobs:
        cp=read(EXEC/'checkpoints'/(job['job_id']+'.json'))
        prompt=(ROOT/cfg['prompts'][job['task']]).read_text(encoding='utf-8')
        assert cp['binding']['job_sha256']==digest(canonical(job))
        assert cp['binding']['config_sha256']==file_sha(ROOT/'configs/stage_g_teacher_pilot_v4_1_grounding.json')
        if cp['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            validate_output(job,cp['output'],schemas[job['task']]);checked+=1
        else:
            raw=read(EXEC/'raw_responses'/(job['job_id']+'.json'))
            assert raw['status']=='completed' and raw['model']==cfg['requested_model']
            try:validate_output(job,json.loads(response_text(raw)),schemas[job['task']])
            except ValueError as error:
                assert str(error)==cp['local_validation_error']
                failed.append({'job_id':job['job_id'],'reason':str(error),'failure_reproduced':True})
            else:raise ValueError('Failed task unexpectedly passes unchanged validator')
        if cp.get('api_called') or cp['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            request_sha=digest(canonical(payload(job,cfg,prompt,schemas[job['task']])))
            expected=(cp['import_lineage']['identical_API_request_sha256'] if cp.get('imported_completed_response')
                      else read(EXEC/'request_intents'/(job['job_id']+'.json'))['request_sha256'])
            assert request_sha==expected
    pre=read(NEW/'rerun_preflight_and_preservation.json')
    for relative,expected in pre['original_artifact_sha256'].items():
        assert file_sha(ROOT/relative)==expected,relative
    # Deduplicate copied responses; count actual completed responses, including invalid/superseded attempts.
    responses={}
    for folder in [ROOT/'outputs/stage_g/teacher_pilot_v4_grounding_d038_v1/execution_v1',NEW/'execution_v1',EXEC]:
        for path in (folder/'raw_responses').glob('*.json'):
            data=read(path);key=file_sha(path)
            responses.setdefault(key,{'file':path.relative_to(ROOT).as_posix(),'response_id':data['id'],'usage':data['usage']})
    usage={'unique_completed_API_responses':len(responses),
        'input_tokens':sum(v['usage'].get('input_tokens',0) for v in responses.values()),
        'output_tokens':sum(v['usage'].get('output_tokens',0) for v in responses.values()),
        'reasoning_tokens':sum(v['usage'].get('output_tokens_details',{}).get('reasoning_tokens',0) for v in responses.values()),
        'successful_final_component_responses':inventory['successful_API_task_records'],
        'extra_superseded_or_invalid_responses':len(responses)-inventory['successful_API_task_records'],
        'scope':'D038only; imports counted once; provider-reported usage is not an invoice'}
    primary_ids={'REG_HOTDOG','REG_BACKGROUND_PIZZA','REG_ASPARAGUS','GREENBEAN_CONFLICT','PERSIST_BOAT'}
    primary=[f for f in notes['findings'] if f['finding_id'] in primary_ids]
    flagged=[r for f in primary for r in f['evidence']['revised'] if r['verdict']=='verified']
    # Evidence-linked manual AI findings must match the immutable machine inventory exactly.
    for f in notes['findings']:
        for version,rows in f['evidence'].items():
            source=machine['review_claims_and_source_quotes'][version] if f['photo_id'] is None else machine['claim_texts_and_verdicts'][version]
            for row in rows:assert row in source,(f['finding_id'],version)
    summary={'status':'RERUN_ATTEMPTED20_WITH_FORMAT_FAILURE_AND_SEMANTIC_CONCERNS_RETURN_FOR_RESEARCHER_REVIEW',
        'businesses_attempted':20,'valid_complete_businesses':inventory['businesses_with_valid_three_condition_targets'],
        'valid_target_texts':inventory['valid_target_texts'],'blocked_businesses':machine['missing_revised_businesses'],
        'all_task_slots_checked':60,'successful_task_validator_rechecks':checked,'failed_task_reproduction':failed,
        'original_artifacts_hash_checked':len(pre['original_artifact_sha256']),
        'all_original_artifacts_unchanged':True,'model_generation_schema_and_inputs_unchanged':True,
        'actual_original_photos_AI_inspected':128,'AI_photo_business_coverage':20,
        'AI_strong_unsupported_verified_concern_occurrences':len(flagged),
        'AI_strong_concern_photos':len({r['photo_id'] for r in flagged}),
        'AI_strong_concern_findings':sorted(primary_ids),
        'AI_findings_are_not_human_ground_truth':True,'human_verified_businesses':0,
        'exact_cross_condition_groups':{v:machine['versions'][v]['exact_cross_condition']['shared_claim_groups'] for v in ['original','revised']},
        'exact_cross_condition_conflicts':{v:machine['versions'][v]['exact_cross_condition']['conflict_groups'] for v in ['original','revised']},
        'matched_accepted_cohort_metrics':machine['matched_accepted_business_comparison'],
        'strong_frequency_word_candidates_not_error_counts':{v:machine['versions'][v]['strong_review_prevalence_candidates'] for v in ['original','revised']},
        'target_LED_token_stats':audit['LED_target_tokens_with_special_tokens'],
        'slow_native_target_ID_checks':audit['slow_LED_target_token_ID_checks'],
        'zero_filtered_caption_cases_preserved':2,'source_allocation_or_StageF_changes':False,
        'no_teacher_target_text_manually_rewritten':True,'usage_all_attempts':usage,
        'production_protocol_frozen':False,'full2200_generation_started':False,'held_out_teacher_targets':0,'LED_training_started':False,
        'review_audit_coverage':notes['review_consensus_inspection'],
        'limitations':['One stochastic generation per prompt; before/after differences are not isolated causal prompt effects.',
                      'Only19valid complete target sets; original20summary is not directly comparable without matching19cohort.',
                      'Literal matches miss paraphrase/compound disagreements, which are separately identified by AI inspection.',
                      'All128images inspected byAI; review semantic audit is not exhaustive entailment certification of all557reviews.',
                      'Same requested/returned alias/nullsnapshot retained; immutable gateway backend snapshot not independently attestable.']}
    (NEW/'comparison_report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (NEW/'usage_all_attempts.json').write_text(json.dumps(dict(usage,responses=list(responses.values())),indent=2)+'\n',encoding='utf-8')
    lines=['# Stage G D-038: revised pilot and comparison — 2026-10-11','',
        '**Do not freeze production yet.** Ordinary-description acceptance and the fork gate improved, but factual regressions/uncertainties remain. No original output or human label was edited.','',
        f"All20same train/validation businesses attempted. **{summary['valid_complete_businesses']}valid businesses /{summary['valid_target_texts']}provisional target texts**; one business pWT3WAUvww4iqF-v2htBXA is blocked by exact caption-quote fidelity.59task slots pass unchanged validators;1fails reproducibly. Its valid review/Filtered components remain stored, but no three-condition target set is promoted. The business remains in the frozen cohort.",'',
        'Original v3 artifacts are preserved. Revised prompts: reviewv5 and visualv2; same gpt-6.1-sol alias/nullsnapshot, mediumreasoning,8192maxoutput, streaming, concurrency1,0automaticretries. D038initialreview span-entry error and two failed OCRquote attempts are retained; one operator-reviewed regeneration occurred without prompt/settings changes. Independent planned tasks were collected once after the failure, never repairing/promoting failed output.','',
        '## Validation and comparable inventory','',
        '| Measure | Original matched19 | Revised valid19 |','|---|---:|---:|']
    m=summary['matched_accepted_cohort_metrics']
    for name,key in [('Review claims','review_claims'),('Visual candidate claims','visual_claims')]:
        lines.append(f"| {name} | {m['original_'+key]} | {m['revised_'+key]} |")
    for verdict in ['verified','rejected','uncertain']:
        lines.append(f"| Teacher {verdict} | {m['original_visual_verdict_counts'].get(verdict,0)} | {m['revised_visual_verdict_counts'].get(verdict,0)} |")
    lines+=['','Different claim decomposition: counts are NOT factual correctness rates. Original full20had139review/409visual claims; revised valid19have109review/411visual claims.','',
        '55accepted-target API request SHA checks,1113exact source-span resolutions and57native/slowLED target-ID comparisons PASS. Remaining2valid component API tasks and the failed task are also independently checked; all60slots accounted for.19complete businesses share identical review consensus across conditions. Both zero-Filtered-input businesses keep the agreed empty visual header. No truncation, manual claim rewrite, input/StageF/K/threshold/budget change.','',
        '| Condition | Min target tokens | Mean | Max |','|---|---:|---:|---:|']
    for condition,stat in summary['target_LED_token_stats'].items():
        lines.append(f"| {condition} | {stat['min']} | {stat['mean']:.2f} | {stat['max']} |")
    lines+=['','No target exceeds512;512remains diagnostic, NOT a frozen target cap. Source5000/review4000/caption750unchanged.','',
        '## Factual and consistency findings','',
        f"Actual original images opened for all128uniquephotos/20businesses. AI inspection identifies **{len(flagged)}verified claim occurrences across{summary['AI_strong_concern_photos']}photos** as strong unsupported/wrong-identity concerns: hotdog, backgroundpizzas, asparagus, greenbeans and boat. Several sentences share one wrong identity; these are not9independent images/errors. Additional ambiguous ingredient/actor/plurality cases are listed separately. These are AI second-opinion findings awaiting the user's final judgment, NOT new human labels or calibrated error rates.",'',
        'Machine literal audit: original1conflict among27exactsharedclaimgroups; revised1among20. The old conflict was the personholdingfork; the new exact conflict is greenbeans in FFV8_b-ec5Z0E3xgvkVxKw (Unfilteredrejected vsFilteredverified). Different claim segmentation changes matching opportunities. Case/whitespace-normalized literal results are stored separately and also have1conflict each. No labels were reconciled.','',
        'AI semantic inspection additionally flags related-paraphrase/conjunction disagreements for backgroundpizzas, men-seated posture, forkpresence, margarita identity and lobsterplurality; fork-holder reject/uncertain granularity differs but BOTHexclude the unsupported actor. Different genuinely different propositions are not automatically contradictions.','',
        '## Detailed evidence-linked examples','']
    for finding in notes['findings']:
        lines+=['### '+finding['finding_id']+' — '+finding['business_id'],'',finding['reason'],'']
        if finding['photo_id']:
            pid=finding['photo_id'];lines+=['Photo: `'+pid+'` ([original image](../../../data/processed/yelp_2300_v1/photos/'+pid+'.jpg)).','']
        for version,rows in finding['evidence'].items():
            for row in rows:
                if finding['photo_id']:
                    label=f"{version} / {row['condition']} / [{row['caption_id']}] / [{row['photo_marker']}] / claim{row['claim_index']+1} / {row['verdict']}"
                    lines+=['- **'+label+'**: '+row['claim'],'  - Caption quote: `'+row['caption_quote']+'`']
                else:
                    lines+=['- **'+version+' / review claim'+str(row['claim_index']+1)+'**: '+row['claim']]
                    for support in row['support']:
                        lines+=['  - `'+support['review_marker']+'` / `'+support['review_id']+'`: '+support['quote'].strip().replace('\n',' ')]
        lines+=['','All observations above are AI review comments, not edits/adjudicated human labels.','']
    lines+=['## Review prevalence and opinion preservation','',
        'Strong wording candidates (most/consistently/universally/everyone/overwhelmingly/always) decreased3→0 in the reported inventories. This is a lexical diagnostic, NOT proof of zero overgeneralization. Many/some/frequently still require source interpretation; cited-source count is not a population estimate or distinct-customer count.','',
        'All original20consensus texts and revised19complete+1validreviewcomponent texts were read. Full selectedreviews were read for EzJI5x25U5GgbYBWhu5ZQQ (30reviews), with targeted exact-source checks for additional cases. Firstbusiness revised consensus preserves qualified minority complaints in REV_3/4/9/15/20/30. PRrWtQ3rdINngwESZNefVQ has a new plantain/corn grouping concern and loses bland-food/BYOB disagreements retained by the original. This is not exhaustive semantic certification of all557reviews.','',
        'Correct but simple descriptions (food/plate/table, chairs, ordinary pool/restaurant scenes) are retained and are NOT counted as errors. Largehamburger acceptance improved; pizza-likeflatbread acceptance was already present and is maintained. Remaining possible overly strict cases (taco taxonomy, marble material, unclear physical-scale/context details) are separate discussion items.','',
        '## Usage and stop point','',
        f"{usage['unique_completed_API_responses']}uniquecompleted APIresponses across allD038attempts; {usage['successful_final_component_responses']}successful finalcomponent records and{usage['extra_superseded_or_invalid_responses']}superseded/invalid responses. Copied responses counted once. Reportedinput{usage['input_tokens']}/output{usage['output_tokens']}, includingreasoning{usage['reasoning_tokens']}; this is not billing verification.",'',
        'Stop after this attempted pilot/comparison. No production2200generation, held-outteacher targets, StageFretuning, protocol freeze or LEDtraining. Researcher review is needed for factual regressions/ambiguous cases and the rejected-task quote protocol. Do not silently relax validators or rewrite targets to pass.','',
        'Machine records: `machine_comparison.json`; detailed AI comments: `ai_semantic_comparison_findings.json`; image observations: `ai_photo_inspection_notes.json`; target/token audit: `execution_v2_quote_recovery/valid_targets_integrity_and_tokens_report.json`; all-task completion: `execution_v2_quote_recovery/completion_20_attempted_businesses.json`.']
    (NEW/'comparison_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    snapshot=NEW/'final_protocol_snapshot'
    for relative in ['configs/stage_g_teacher_pilot_v4_1_grounding.json','prompts/stage_g_review_consensus_v5.txt','prompts/stage_g_visual_evidence_v2.txt','schemas/stage_g_teacher_pilot_v3.json','scripts/run_stage_g_teacher_pilot_v4_1_recovery.py','scripts/collect_stage_g_remaining_pilot_tasks.py','scripts/stage_g_canonical_review_ids.py','scripts/stage_g_evidence_id_protocol.py','scripts/stage_g_visual_target_serialization.py']:
        dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.read_bytes()==(ROOT/relative).read_bytes()
        else:shutil.copyfile(ROOT/relative,dest)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['matched_accepted_cohort_metrics','review_audit_coverage','target_LED_token_stats','limitations']},indent=2))


if __name__=='__main__':
    main()
