"""Offline D057 analysis of existing authorized artifacts; no model/API execution."""
import collections,json,math,shutil,sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from stage_i_d056_v2 import *

REPORT=ROOT/'outputs/stage_i/d057_bounded_gpu_diagnostic_v1'

def new(path,value):write(path,value)

def main():
    exit_path=REPORT/'GPU_PROCESS_EXIT.json'
    require(exit_path.exists() or (REPORT/'ORCHESTRATION_STOP.json').exists(),'Execution still active; no final report yet')
    approval=read(REPORT/'RESEARCHER_GPU_AUTHORIZATION.json');cfg=read(CONFIG)
    require(approval['GPU_execution_approved'] is True and approval['training_authorized'] is False,'Wrong execution authorization')
    require(sha(CONFIG)==approval['config_sha256'] and sha(OUT/'PREFLIGHT_MANIFEST.json')==approval['preflight_manifest_sha256']
            and sha(ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py')==approval['GPU_runner_sha256'],'Reviewed bindings changed')
    manifest=verify_manifest(OUT/'PREFLIGHT_MANIFEST.json');rawroot=ROOT/cfg['future_output']
    completion=read(rawroot/'COMPLETION.json') if (rawroot/'COMPLETION.json').exists() else None
    failure=read(rawroot/'FAILURE_STOP_NO_FALLBACK.json') if (rawroot/'FAILURE_STOP_NO_FALLBACK.json').exists() else None
    process=read(exit_path) if exit_path.exists() else None
    cases=read(OUT/'cases.json');byid={x['business_id']:x for x in cases}
    aligns={x['business_id']:x for x in read(OUT/'TOKEN_POSITION_ALIGNMENT.json')}
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    teacher=[];cache=[];cross=[];traces=[];memory=[];integrity=[]
    for state in STATES:
        cp=rawroot/(state+'_COMPLETION.json')
        if cp.exists():memory.append({'state':state,**read(cp)})
        for bid in CASES:
            folder=rawroot/state/bid;c=byid[bid];a=aligns[bid]
            for precision in cfg['precision_modes']:
                path=folder/(precision+'_teacher.json')
                if path.exists():
                    data=read(path);p=folder/(precision+'_teacher_raw.npz')
                    transition_details={}
                    with np.load(p,allow_pickle=False) as pack:
                        require(pack['labels'].tolist()==c['label_ids'],'Saved teacher labels differ')
                        require(pack['decoder_input_ids'].tolist()==a['teacher_decoder_input_ids'],'Saved decoder alignment differs')
                        require(pack['critical_rows'].tolist()==a['critical_logit_rows'],'Critical positions changed')
                        require(np.isfinite(pack['logits']).all(),'Saved teacher logits nonfinite')
                        require(np.array_equal(-pack['correct_logprobs'],np.asarray(data['per_token_NLL'])),'Saved per-token NLL mismatch')
                        differences=[]
                        for j,row in enumerate(pack['critical_rows']):
                            got=log_probs(pack['logits'][j])[c['label_ids'][int(row)]]
                            differences.append(abs(float(got)-float(pack['correct_logprobs'][int(row)])))
                            if int(row) in [2,10,11]:
                                vec=pack['logits'][j];correct=c['label_ids'][int(row)];best=int(np.argmax(vec));other=vec.copy();other[correct]=-np.inf;competitor=int(np.argmax(other))
                                transition_details[str(int(row))]={'correct_id':correct,'correct_token_text':tok.decode([correct],skip_special_tokens=False),
                                    'argmax_id':best,'argmax_token_text':tok.decode([best],skip_special_tokens=False),
                                    'correct_is_argmax':best==correct,'strictly_higher_logit_tokens':int((vec>vec[correct]).sum()),
                                    'best_competing_id':competitor,'best_competing_text':tok.decode([competitor],skip_special_tokens=False)}
                        maxdiff=max(differences)
                        integrity.append({'kind':'teacher','state':state,'business_id':bid,'precision':precision,
                            'raw_vector_rows':len(pack['critical_rows']),'correct_logprob_recomputation_max_abs':maxdiff,
                            'source':p.relative_to(ROOT).as_posix()})
                    important={str(x['row']):x for x in data['critical_positions'] if x['row'] in [2,10,11,a['visual_marker_label_start'],len(c['label_ids'])-1]}
                    teacher.append({'state':state,'business_id':bid,'precision':precision,'mean_NLL':data['mean_NLL'],
                        'important_positions':important,'early_transition_tokens':transition_details,'zero_caption':a['zero_caption'],'file':path.relative_to(ROOT).as_posix()})
                for pfx in [3,11]:
                    path=folder/(precision+f'_cache_p{pfx}.json')
                    if not path.exists():continue
                    data=read(path);p=folder/(precision+f'_cache_p{pfx}_raw.npz')
                    with np.load(p,allow_pickle=False) as pack:
                        require(pack['prefix'].tolist()==a['teacher_decoder_input_ids'][:pfx],'Saved cache prefix mismatch')
                        require(int(pack['correct_id'])==c['label_ids'][pfx-1],'Saved cache prediction alignment mismatch')
                        screen=cfg['cache_screens']['FP32' if precision=='FP32' else 'FP16']
                        recomputed=vector_comparison(pack['uncached_logits'],pack['cached_logits'],int(pack['correct_id']),**screen)
                        require(all(data[k]==value for k,value in recomputed.items()),'Raw cache vectors do not reproduce reported numerical comparison')
                    integrity.append({'kind':'cache','state':state,'business_id':bid,'precision':precision,'p':pfx,
                        'full_vector_recomputation_exact_match':True,'source':p.relative_to(ROOT).as_posix()})
                    memory.append({'state':state,'business_id':bid,'precision':precision,'p':pfx,'memory':data['memory']})
                    cache.append({'state':state,'business_id':bid,'precision':precision,'prefix_length':pfx,
                        **{k:data[k] for k in recomputed},'file':path.relative_to(ROOT).as_posix()})
            path=folder/'CROSS_PRECISION.json'
            if path.exists():
                data=read(path);cross.append({'state':state,'business_id':bid,
                    'critical_rows':len(data['rows']),'flagged_rows':sum(x['cross_precision_investigation_flag'] for x in data['rows']),
                    'argmax_changes':sum(x['argmax_changed'] for x in data['rows']),
                    'max_abs':max(x['max_abs'] for x in data['rows']),
                    'early_transition_rows':[x for x in data['rows'] if x['row'] in [2,10,11]],
                    'file':path.relative_to(ROOT).as_posix()})
            for mode in cfg['trace_modes']:
                path=folder/(mode+'.json')
                if not path.exists():continue
                data=read(path);observation=trace_observation(data['prompt_ids'],data['raw_ids'],data['run_reached_diagnostic_censor'])
                require(all(data[k]==v for k,v in observation.items()),'Trace termination observation mismatch')
                for j,step in enumerate(data['per_step_per_beam_top10']):
                    require(step['step']==j,'Trace steps out of order')
                    for idskey,valskey in [('raw_top10_ids','raw_top10_logits'),('processed_top10_ids','processed_top10_scores')]:
                        require(idskey in step and valskey in step,'Missing independent top10')
                        for ids,vals in zip(step[idskey],step[valskey]):
                            require(len(ids)==len(vals)==10 and len(set(ids))==10,'Invalid top10 geometry')
                            require(all(0<=x<50265 for x in ids),'Trace candidate outside vocabulary')
                            numeric=[float(x) for x in vals];require(all(numeric[k]>=numeric[k+1] for k in range(9)),'Candidate ranking not descending')
                            require(all(ids[k]<ids[k+1] for k in range(9) if numeric[k]==numeric[k+1]),'Candidate tie-break differs')
                beam=data['sequence_score_reconstruction']
                if beam is not None:
                    check=beam_trace_audit(data['prompt_ids'],data['raw_ids'],data['beam_indices'][0],data['selected_transition_scores'][0],
                        beam['selected_processed_scores_direct'],beam['selected_raw_logits'],data['sequences_scores'][0],
                        data['generation_config']['length_penalty'],len(data['per_step_per_beam_top10']),data['run_reached_diagnostic_censor'],4)
                    require(check==beam,'Beam ancestry/reconstruction report mismatch')
                new_ids=data['raw_ids'][len(data['prompt_ids']):];text=tok.decode(data['raw_ids'],skip_special_tokens=True)
                grams=collections.Counter(tuple(new_ids[i:i+4]) for i in range(max(0,len(new_ids)-3)))
                expected_prefix=a['rescue_prefix_ids'];prefix_matches=data['raw_ids'][:len(expected_prefix)]==expected_prefix
                mismatch=next((i for i,(x,y) in enumerate(zip(data['raw_ids'],expected_prefix)) if x!=y),None)
                traces.append({'state':state,'business_id':bid,'mode':mode,**observation,
                    'decoded_text':text,'raw_ids':data['raw_ids'],'prompt_ids':data['prompt_ids'],
                    'has_review_marker':REVIEW in text,'has_visual_marker':VISUAL in text,
                    'matches_correct_12_token_header_prefix':prefix_matches,'first_header_prefix_mismatch_index':mismatch,
                    'header_was_supplied_not_learned':mode=='greedy_correct_header_prefix',
                    'adjacent_equal_token_pairs':sum(x==y for x,y in zip(new_ids,new_ids[1:])),
                    'extra_occurrences_of_exact_4grams':sum(v-1 for v in grams.values() if v>1),
                    'zero_caption':a['zero_caption'],'zero_caption_note':'Censored prefixes cannot establish full-summary empty-visual compliance.' if a['zero_caption'] else None,
                    'beam_reconstruction':beam,'file':path.relative_to(ROOT).as_posix()})
    expected={'teacher_outputs':24,'cache_comparisons':48,'traces':36}
    observed={'teacher_outputs':len(teacher),'cache_comparisons':len(cache),'traces':len(traces)}
    complete=completion is not None and process is not None and process['returncode']==0 and observed==expected
    counts=(completion or failure or {}).get('counts',{})
    require(all(counts.get(k,0)<=v for k,v in cfg['ceilings'].items()),'Recorded count exceeds approved ceiling')
    cache_stats={p:{'comparisons':len([x for x in cache if x['precision']==p]),
        'screen_passes':sum(x['allclose_screen_pass'] for x in cache if x['precision']==p),
        'argmax_changes':sum(x['argmax_changed'] for x in cache if x['precision']==p),
        'max_abs':max([x['max_abs'] for x in cache if x['precision']==p],default=None)} for p in cfg['precision_modes']}
    def memory_value(x,key):return x.get('memory',{}).get(key,0)
    memories={'peak_allocated_bytes':max([memory_value(x,'peak_allocated') for x in memory],default=0),
              'peak_reserved_bytes':max([memory_value(x,'peak_reserved') for x in memory],default=0)}
    runtime=read(rawroot/'APPROVAL_AND_CONFIG.json') if (rawroot/'APPROVAL_AND_CONFIG.json').exists() else {}
    new(REPORT/'TEACHER_FORCED_MEASUREMENTS.json',teacher);new(REPORT/'CACHE_COMPARISONS.json',cache)
    new(REPORT/'CROSS_PRECISION_SUMMARY.json',cross);new(REPORT/'TRACE_INDEX_AND_TEXT.json',traces)
    new(REPORT/'RAW_EVIDENCE_RECOMPUTATION.json',integrity);new(REPORT/'GPU_MEMORY_OBSERVATIONS.json',memory)
    state_observations=[]
    for state in STATES:
        for precision in cfg['precision_modes']:
            group=[x for x in teacher if x['state']==state and x['precision']==precision]
            if not group:continue
            state_observations.append({'state':state,'precision':precision,'n_cases':len(group),
                'mean_P_RE':float(np.mean([x['important_positions']['2']['expected_probability'] for x in group])),
                'mean_P_newline':float(np.mean([x['important_positions']['10']['expected_probability'] for x in group])),
                'correct_argmax_cases_p3':sum(x['early_transition_tokens']['2']['correct_is_argmax'] for x in group),
                'correct_argmax_cases_p11':sum(x['early_transition_tokens']['10']['correct_is_argmax'] for x in group)})
    new(REPORT/'DESCRIPTIVE_EARLY_TRANSITION_OBSERVATIONS.json',state_observations)
    interpretation={
        'H1_underadaptation':'Unresolved causal explanation. Report descriptive early-transition probabilities across states; historical exposure/diversity/schedule are unmatched. No duration-only conclusion.',
        'H5_teacher_forcing_free_generation_divergence':'Compare per-case correct-prefix probabilities and unprefixed versus fixed-prefix traces. Short censored continuations do not establish full-summary rescue or grounded quality.',
        'H4_multi_token_marker_transition':'p3 and p11 probabilities, competing tokens and margins directly measure early transitions. No tokenization/serialization change or marker-cause claim.',
        'H6_pretrained_task_adaptation':'Pretrained is a diagnostic reference; shifts with adapters are observations, not proof that more training alone solves failures.',
        'H7_cache_precision':'Within-precision cache screens and cross-precision flags are reported separately. Screen failure is not automatically a cache defect; a near-tie argmax flip is not proof of instability. D051 backward replay FAIL remains distinct.',
        'H2_training_inference_contract':'Live shape/mask/config/adapter guards must pass before each observed measurement. Disk hashes remain verified. A stopped integrity/runtime contract supersedes mechanism interpretation.',
        'H3_LoRA_capacity':'No capacity ceiling or winning architecture can be selected using three targeted diagnostic cases; B also has twice the adapter parameters.',
        'stopping_policy':'Stop for independent review. Do not launch, authorize, or select a subsequent training/decoding experiment.'}
    new(REPORT/'D055_HYPOTHESIS_INTERPRETATION.json',interpretation)
    summary={'status':'D057_COMPLETE_STOP_FOR_INDEPENDENT_REVIEW' if complete else 'D057_PARTIAL_OR_FAILED_STOP_NO_RETRY',
        'complete':complete,'expected_artifact_counts':expected,'observed_artifact_counts':observed,
        'recorded_budget_counts':counts,'budget_count_semantics':'Runner reserves counters before forwarding; on failure these are attempted/reserved, not guaranteed completed calls.',
        'execution_invocations':1,'retry_invocations':0,'GPU':runtime.get('GPU'),'process_exit':process,
        'failure':failure,'cache_summary':cache_stats,'cross_precision_flagged_rows':sum(x['flagged_rows'] for x in cross),
        'cross_precision_critical_rows':sum(x['critical_rows'] for x in cross),'GPU_memory':memories,
        'source_checkpoint_files_unchanged':True,'bound_files_rechecked':len(manifest['files']),
        'in_memory_weights_verified_states':[x['state'] for x in memory if x.get('weights_unchanged') is True],
        'D051_strict_replay':'FAIL_UNCHANGED','training_updates':0,'Teacher_calls':0,'held_out_access':False,
        'raw_measurements_are_model_observations_not_human_factual_ground_truth':True,
        'no_architecture_winner_or_new_experiment_authorization':True}
    new(REPORT/'D057_SUMMARY.json',summary)
    lines=['# Stage I — D-057 bounded GPU diagnostic report','',f"Status: **{summary['status']}**.",'',
        '## Execution and integrity','',
        'The researcher explicitly authorized this hash-bound D056-v2 run. The original offline preflight passed in the same local Python environment. The frozen runner was invoked once; no retry, settings change, fallback or later experiment was performed.',
        '',f"GPU: {summary['GPU']}. Process return code: {process['returncode'] if process else 'not invoked/unknown'}. All {len(manifest['files'])} preflight file hashes still match. D051 strict replay remains **FAIL_UNCHANGED**.",'',
        '| Saved evidence | Observed | Expected |','| --- | ---: | ---: |']
    lines += [f'| {k} | {observed[k]} | {v} |' for k,v in expected.items()]
    lines += ['',f"Recorded reserved/attempted budget counters: `{counts}`. Per completed cache comparison, three explicit forwards;48 comparisons correspond to144 cache forwards. On failure, reserved counters must not be represented as completed calls.",'',
        f"Peak recorded allocated {memories['peak_allocated_bytes']/1024**2:.1f} MiB; reserved {memories['peak_reserved_bytes']/1024**2:.1f} MiB. Runtime including offline preflight: {process['total_wall_seconds']:.1f}s." if process else 'No completed process timing available.','',
        '## Within-precision cache screens','',
        'Frozen FP32 atol/rtol1e-4/1e-4; FP16 atol0.05/rtol0.005; relative floor1e-8. These screens were not modified. Saved full-vocabulary cache vectors were independently recomputed offline.',
        '', '| Precision | Comparisons | Screen passes | Argmax changes | Maximum absolute difference |','| --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {p} | {v['comparisons']} | {v['screen_passes']} | {v['argmax_changes']} | {v['max_abs']} |" for p,v in cache_stats.items()]
    lines += ['', 'Cross-precision maxAbs>0.1 OR argmax-change flags are separate from within-cache allclose. '
              f"Flagged {summary['cross_precision_flagged_rows']}/{summary['cross_precision_critical_rows']} critical rows. Inspect saved row-level margins/probabilities; flags do not alone establish meaningful numerical instability.",'',
        '## Correct-prefix transition measurements','',
        'P(RE) is teacher-forced row2 (p3); P(newline) row10 (p11); P(first body token) row11 (p12). These probabilities condition on complete correct prefixes, not on free-generation prefixes. No new cutoff defines success.',
        '', '| State | Case | Precision | Mean NLL | P(RE) | P(newline) | P(first body) |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for x in teacher:
        p=x['important_positions'];lines.append(f"| {x['state']} | {x['business_id']} | {x['precision']} | {x['mean_NLL']:.5f} | {p['2']['expected_probability']:.6g} | {p['10']['expected_probability']:.6g} | {p['11']['expected_probability']:.6g} |")
    lines += ['', '## Censored traces and EOS/finalization','',
        'All traces are <=64 newly emitted tokens. Their winner/prefix is not necessarily the winner/prefix of a full768-token beam search. Real selected EOS, custom censor and vectorized finalization are separate. Initial decoder-start2 is not EOS completion. Top10 raw logits and processed scores use independent rankings.',
        '', '| State | Case | Mode | New tokens | Selected EOS index | Global censor | Adjacent identical IDs | Extra 4-grams |','| --- | --- | --- | ---: | ---: | --- | ---: | ---: |']
    for x in traces:lines.append(f"| {x['state']} | {x['business_id']} | {x['mode']} | {x['new_token_count']} | {x['selected_EOS_new_token_index']} | {x['run_reached_diagnostic_censor']} | {x['adjacent_equal_token_pairs']} | {x['extra_occurrences_of_exact_4grams']} |")
    lines += ['', 'Token repetition is descriptive, not summary-quality evidence. Full texts/IDs/ancestry/selected scores are in TRACE_INDEX_AND_TEXT.json and the original raw trace JSON. Teacher-forced visual-boundary/EOS values are saved, including the zero-caption case; censored prefixes cannot prove full-summary zero-caption compliance.', '',
              '## D055 hypotheses and scientific limits','']
    lines += [f'- **{k}:** {v}' for k,v in interpretation.items()]
    if not complete:lines += ['', '## Failure and incomplete evidence','', 'Execution stopped without retry. Do not generalize missing state/case results or interpret absent measurements as PASS.', '', '```json',json.dumps(failure or read(REPORT/'ORCHESTRATION_STOP.json') if (REPORT/'ORCHESTRATION_STOP.json').exists() else failure,indent=2),'```']
    lines += ['', '## Research boundary','',
        'No training, Teacher requests, held-out access, dataset/target/reference/checkpoint modification or architecture/decoding-policy change. The approved pipeline and prior D051/D055 evidence are preserved. Observations do not establish a causal solution, grounded full-summary quality, or a winning architecture. Stop here for independent scientific review.']
    new(REPORT/'STAGE_I_D057_DIAGNOSTIC_REPORT.md','\n'.join(lines))
    print(json.dumps({'status':summary['status'],'observed':observed,'cache_summary':cache_stats,'memory':memories},indent=2))

if __name__=='__main__':main()
