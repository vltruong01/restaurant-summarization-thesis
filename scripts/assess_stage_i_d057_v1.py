"""Evidence-linked D057 scientific assessment and living status, offline only."""
import collections,json
from stage_i_d056_v2 import *

R=ROOT/'outputs/stage_i/d057_bounded_gpu_diagnostic_v1'

def main():
    summary=read(R/'D057_SUMMARY.json');require(summary['complete'],'Review stopped execution before using complete-run assessment')
    early=read(R/'DESCRIPTIVE_EARLY_TRANSITION_OBSERVATIONS.json');traces=read(R/'TRACE_INDEX_AND_TEXT.json')
    teachers=read(R/'TEACHER_FORCED_MEASUREMENTS.json');cross=read(R/'CROSS_PRECISION_SUMMARY.json')
    integrity=read(R/'RAW_EVIDENCE_RECOMPUTATION.json');rawroot=ROOT/read(CONFIG)['future_output']
    completion=read(rawroot/'COMPLETION.json');approval=read(R/'RESEARCHER_GPU_AUTHORIZATION.json')
    beam=[x['beam_reconstruction'] for x in traces if x['beam_reconstruction'] is not None]
    trace_groups=[]
    for state in STATES:
        for mode in read(CONFIG)['trace_modes']:
            group=[x for x in traces if x['state']==state and x['mode']==mode]
            trace_groups.append({'state':state,'mode':mode,'n':len(group),
                'correct_header_and_newline':sum(x['matches_correct_12_token_header_prefix'] for x in group),
                'header_supplied_by_intervention':mode=='greedy_correct_header_prefix',
                'selected_EOS':sum(x['selected_EOS_new_token_index'] is not None for x in group),
                'global_censored':sum(x['run_reached_diagnostic_censor'] for x in group),
                'traces_with_repeated_4grams':sum(x['extra_occurrences_of_exact_4grams']>0 for x in group),
                'mean_extra_4grams':sum(x['extra_occurrences_of_exact_4grams'] for x in group)/len(group)})
    zero=[]
    for x in teachers:
        if x['zero_caption']:
            last=max(map(int,x['important_positions']));position=x['important_positions'][str(last)]
            zero.append({'state':x['state'],'precision':x['precision'],'business_id':x['business_id'],
                'EOS_label_index':last,'EOS_probability_given_correct_complete_target_prefix':position['expected_probability'],
                'EOS_is_argmax':position['top_ids'][0]==2,'file':x['file']})
    findings={
        'verified_observations':[
            {'id':'O1','claim':'Exactly one authorized frozen run completed 24 TF,144 cache forwards,36 traces; all four state weights unchanged and344 bound disk-file hashes match.',
             'evidence':['GPU_PROCESS_EXIT.json','OFFLINE_PREFLIGHT_RESULT.json',str((rawroot/'COMPLETION.json').relative_to(ROOT))]},
            {'id':'O2','claim':'All48 equivalent-prefix comparisons passed their original within-precision screens; no cache argmax changes. MaxAbs FP32=3.62396240234375e-05; FP16=0.0234375.',
             'evidence':['CACHE_COMPARISONS.json','RAW_EVIDENCE_RECOMPUTATION.json']},
            {'id':'O3','claim':'Zero of304 TF critical rows triggered cross-precision maxAbs>0.1/argmax-change screen; largest FP32/AMP difference=0.031287193298339844.',
             'evidence':['CROSS_PRECISION_SUMMARY.json']},
            {'id':'O4','claim':'Under correct teacher prefixes, both short pilots choose correct RE at p3 on3/3 cases, but neither chooses newline at p11 on any case, in either precision.',
             'evidence':['TEACHER_FORCED_MEASUREMENTS.json','DESCRIPTIVE_EARLY_TRANSITION_OBSERVATIONS.json']},
            {'id':'O5','claim':'FP32 mean newline probability is0.0000381024(A),0.000458049(B),0.909430(historical F),0.00000195741(pretrained); n=3 cases/state. These are descriptive, unmatched historical observations.',
             'evidence':['DESCRIPTIVE_EARLY_TRANSITION_OBSERVATIONS.json']},
            {'id':'O6','claim':'All36 traces hit64-new-token censor with no selected EOS. Every pilot prefix-rescue trace still repeats4-grams. Supplied header success is not learned header success.',
             'evidence':['TRACE_INDEX_AND_TEXT.json','TRACE_GROUP_OBSERVATIONS.json']},
            {'id':'O7','claim':'All12 selected beam transition gathers exactly match library transitions; maximum sequence-score reconstruction discrepancy1.4759409705522586e-07 is reported without introducing an acceptance tolerance.',
             'evidence':['TRACE_INDEX_AND_TEXT.json']},
        ],
        'supported_but_not_proven_explanations':[
            'Weak learned early transitions, especially newline, are directly compatible with pilot degeneration and are present even under a correct teacher prefix.',
            'Underadaptation/task adaptation remains compatible with historical-versus-short-pilot contrast, but exposure, schedule, diversity and initialization differ.',
            'Free-generation error amplification can coexist with weak transitions; it is not the sole demonstrated cause because teacher-forced newline is already weak.'
        ],
        'unresolved_or_not_established':[
            'These probes do not prove training duration is the cause or that longer training will solve degeneration.',
            'A cache or FP32/FP16 mismatch was not detected at these points; this does not establish equivalence for every later free-generation prefix or all kernels.',
            'No architecture winner or capacity limit can be inferred from three selected diagnostic cases; B has twice the adapter parameters.',
            'No full-summary grounding, informativeness, visual-section correctness or zero-caption generation compliance is established by64-token censored traces.',
            'D051 strict backward checkpoint replay FAIL remains unchanged and is not invalidated by forward/cache screens.'
        ],
        'next_action':'Independent review of D057 only. No subsequent experiment is proposed or authorized in this completion step.',
        'D051_strict_replay':'FAIL_UNCHANGED',
    }
    write(R/'TRACE_GROUP_OBSERVATIONS.json',trace_groups);write(R/'ZERO_CAPTION_TF_EOS_OBSERVATIONS.json',zero)
    write(R/'VERIFIED_OBSERVATIONS_AND_HYPOTHESES.json',findings)
    lines=['# D-057 — Scientific assessment of bounded evidence','',
        '**Completed once; STOP for independent review.** No training or follow-up experiment is authorized.', '',
        '## Direct observations','',
        'The inference/runtime result is not a full-summary quality PASS. All observations below are limited to three prespecified Filtered validation cases and four frozen states.', '']
    lines += [f"- **{x['id']}:** {x['claim']} Evidence: "+', '.join('`'+p+'`' for p in x['evidence']) for x in findings['verified_observations']]
    lines += ['', '## Descriptive early transitions (n=3 per state and precision)','',
        'Means are over the three selected diagnostic cases. There is no population estimate, architecture selection, or duration-only causal comparison.', '',
        '| State | Precision | Mean P(RE at p3) | Mean P(newline at p11) | Correct p3 argmax | Correct p11 argmax |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for x in early:lines.append(f"| {x['state']} | {x['precision']} | {x['mean_P_RE']:.8g} | {x['mean_P_newline']:.8g} | {x['correct_argmax_cases_p3']}/3 | {x['correct_argmax_cases_p11']}/3 |")
    lines += ['', 'The p3 teacher-forced success does not imply every later marker token or free-generated prefix is correct. The p11 result shows that at least part of the failure already exists under a correct prefix, rather than arising only from free-generation drift.', '',
        '## Prefix rescue and free-generation observations','',
        '| State | Mode | Correct header/newline | Repeated4-gram traces | Censored | EOS |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for x in trace_groups:lines.append(f"| {x['state']} | {x['mode']} | {x['correct_header_and_newline']}/3{' (supplied)' if x['header_supplied_by_intervention'] else ''} | {x['traces_with_repeated_4grams']}/3 | {x['global_censored']}/3 | {x['selected_EOS']}/3 |")
    lines += ['', 'The supplied prefix is a controlled intervention, not an edited target or a learned structural improvement. All six A/B rescued continuations contain repeated4-grams; this does not quantify their full-summary quality or factual accuracy.', '',
        '## Representative observed text (diagnostic prefix only)','']
    bid=CASES[0]
    for state in STATES:
        for mode in ['greedy_unprefixed','greedy_correct_header_prefix']:
            x=next(t for t in traces if t['state']==state and t['business_id']==bid and t['mode']==mode)
            lines += [f'### {state} / {mode} / {bid}', '', '```text',x['decoded_text'],'```','']
    lines += ['These excerpts are observed model output, not claims accepted as restaurant facts. For example, the A rescued prefix asserts "The restaurant is open"; its presence is not factual verification. No automatic grounded-quality grade is assigned. Full sources, teacher target and photo/caption IDs remain in the frozen cases evidence.', '',
        '## Zero-caption case','',
        f'`{CASES[2]}` retains its original empty visual target. Existing TF measurements include the visual marker boundary, newline and terminal EOS. The following is target-conditioned EOS probability, not free-generation compliance:', '',
        '| State | Precision | EOS row | P(EOS given correct prefix) | EOS argmax |','| --- | --- | ---: | ---: | --- |']
    for x in zero:lines.append(f"| {x['state']} | {x['precision']} | {x['EOS_label_index']} | {x['EOS_probability_given_correct_complete_target_prefix']:.8g} | {x['EOS_is_argmax']} |")
    lines += ['', 'No selected trace terminated in EOS within the64-token window. Therefore zero-caption full-generation behavior and review-to-visual passage cannot be accepted from these short outputs. The frozen source/target representation is unchanged.', '',
        '## Interpretation against D055 hypotheses','']
    lines += ['- '+x for x in findings['supported_but_not_proven_explanations']]
    lines += ['', 'Unresolved and excluded conclusions:', '']+['- '+x for x in findings['unresolved_or_not_established']]
    lines += ['', '## Runtime, warnings and integrity','',
        f"Process wall time including offline preflight/imports: {summary['process_exit']['total_wall_seconds']:.3f}s. Prepared runner measurement interval: {completion['seconds']:.3f}s; this includes CPU model loading/hashing and is not GPU-kernel-only time. Peak allocated {summary['GPU_memory']['peak_allocated_bytes']/1024**2:.2f} MiB; reserved {summary['GPU_memory']['peak_reserved_bytes']/1024**2:.2f} MiB.", '',
        'Preserved warnings: TypedStorage deprecation; early_stopping has no greedy effect while remaining frozen; Triton Windows emits "Failed to find CUDA". Torch CUDA operations nevertheless completed on the recorded RTX3050. No application-level fallback or setting adjustment was performed. No fatal runtime, NaN/Inf, OOM or guard failure was reported.', '',
        'Raw logits/probabilities, 48 recomputed cache vector pairs,304 cross-precision rows,36 per-step raw/processed score traces,12 beam ancestries, effective configs, mask/forward records, pre/post hashes and stderr are preserved. All24 teacher critical-vector log-probability recomputations match with maxAbs0. Actual parameter rehashes passed for all four states. Existing frozen files were not changed.', '',
        '## Stop boundary','',findings['next_action']]
    write(R/'STAGE_I_D057_SCIENTIFIC_ASSESSMENT.md','\n'.join(lines))
    write(R/'README_FIRST.md','''# D-057 — Authorized bounded GPU diagnostic, completed once

Read STAGE_I_D057_SCIENTIFIC_ASSESSMENT.md first, then
STAGE_I_D057_DIAGNOSTIC_REPORT.md and the machine-readable evidence tables.

Execution completed exactly24 TF forwards,144 cache forwards at p3/11,36 traces
at64 newly emitted tokens. No retry, training, Teacher call, held-out access,
target edit or new experiment. Original D056-v2 three hashes remain unchanged;
current explicit GPU authorization is recorded separately from earlier scientific
acceptance. D051 strict replay FAIL remains unchanged.

All48 cache comparisons pass their original screens; zero of304 cross-precision
critical rows are flagged. Both short pilots nevertheless give very low newline
probability under correct prefixes; supplied-header traces still repeat. This
supports weak learned transitions, not a proven training-duration cause or an
architecture winner. All36 traces are censored with no selected EOS. They are
not full summaries and do not establish factual grounding/zero-caption compliance.

The portable bundle includes complete new raw measurements, original frozen
config/preflight/runner bytes, sources/teacher targets for the3 cases, tokenizer,
reports and relative SHA256 manifest. Run `python verify_bundle.py <bundle_dir>`
with numpy and tokenizers installed to verify bytes/encodes/cache calculations.
This verification performs no model/GPU calls. The original frozen runtime has
local artifact paths; it is evidence, not a portable rerun authorization. Full
backbone/checkpoint/dataset files are not duplicated in this review ZIP.
''')
    write(R/'CHATGPT_REVIEW_REQUEST.md','''# Independent review of D057 observations

Audit explicit authorization, unchanged three-way hashes, exactly-once execution,
ceilings, forward/mask/adapter bindings and unchanged scientific artifacts.
Recompute raw cache vectors and inspect FP32/FP16 critical positions, correct
probabilities, competitor tokens, beam ancestry/direct transitions/reconstruction,
raw versus processed top10 and EOS versus64-token censor/finalization.

Evaluate the distinction between directly observed weak newline transitions and
hypotheses about underadaptation, free-generation amplification, runtime mismatch
or capacity. Historical full F versus512-business pilot is not a duration-only
comparison. Header rescue does not eliminate repeated body content, and supplied
headers must not be counted as learned success. All36 outputs are censored;
do not infer full-summary grounding, visual compliance or an architecture winner.

Preserve D051 strict replay FAIL. Teacher targets are comparisons, not independent
factual truth. No further training/decoding/data changes are approved or proposed
by completion of this diagnostic. Return independent scientific review first.
''')
    entry=f'''## D-057 — Hash-bound D056-v2 GPU diagnostic completed once; independent review pending

Researcher explicitly authorized bounded execution after separate scientific
acceptance. Original config/preflight/runner SHA256 bindings remain unchanged.
Local RTX3050 completed **24 TF /144 cache forwards at p3/11 /36 traces<=64 new
tokens**, one invocation, zero retries. All344 bound files and all four model
state pre/post tensor fingerprints match. No scientific artifact edits, training,
Teacher requests, held-out access or automatic subsequent experiment.

**Cache:**48/48 original within-precision screens PASS; no argmax changes.
MaxAbs FP32=3.6239624e-5; FP16=0.0234375. **Cross precision:**0/304 critical-row
flags; maxAbs0.0312872, no argmax changes. Saved raw vectors independently
recomputed offline. All12 beam transition gathers exact; sequence-score
reconstruction max discrepancy1.47594e-7, reported without a new tolerance.

**Learned transitions:** under correct prefixes, both short pilots predict RE
at p3 on3/3 cases, but not newline at p11 on any case, in either precision.
FP32 mean P(newline): A0.0000381024; B0.000458049; historicalF0.909430;
pretrained0.00000195741 (n=3 diagnostic cases/state). This supports weak learned
transitions, not a proven duration cause or winning architecture. All six A/B
header-rescue continuations still repeat4-grams. All36 traces censored at64;
zero selected EOS. Full-summary/zero-caption quality is not established.

Runtime including preflight/imports {summary['process_exit']['total_wall_seconds']:.1f}s;
runner interval {completion['seconds']:.1f}s (not GPU-kernel-only timing).
Peak allocated {summary['GPU_memory']['peak_allocated_bytes']/1024**2:.1f}MiB;
reserved {summary['GPU_memory']['peak_reserved_bytes']/1024**2:.1f}MiB.
Warnings preserved; no fatal execution/NaN/Inf/OOM failure. **D051 strict replay
remains FAIL_UNCHANGED.** No tolerances/config/settings changed.

Report: `outputs/stage_i/d057_bounded_gpu_diagnostic_v1/STAGE_I_D057_DIAGNOSTIC_REPORT.md`;
assessment: `outputs/stage_i/d057_bounded_gpu_diagnostic_v1/STAGE_I_D057_SCIENTIFIC_ASSESSMENT.md`;
raw evidence: `outputs/stage_i/d056_gpu_diagnostic_v2`.
Review ZIP: `outputs/stage_i/stage_i_d057_review_bundle_v1.zip`.
Status: **STOP FOR INDEPENDENT SCIENTIFIC REVIEW**; no follow-up execution or
new experiment proposal in this completion step.

'''
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/rel;old=p.read_text(encoding='utf-8');require(entry.splitlines()[0] not in old,'Status already recorded')
        write(R/'documentation_before_completion'/p.name,old)
        head,_,body=old.partition('\n');p.write_text(head+'\n\n'+entry+body.lstrip('\n'),encoding='utf-8',newline='\n')
    files={p.relative_to(ROOT).as_posix():sha(p) for folder in [R,rawroot] for p in folder.rglob('*') if p.is_file()}
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md',
        'scripts/report_stage_i_d057_v1.py','scripts/assess_stage_i_d057_v1.py','scripts/execute_stage_i_d057_authorized_v1.py',
        'scripts/package_stage_i_d057_v1.py','scripts/verify_stage_i_d057_review_bundle_v1.py']:
        files[rel]=sha(ROOT/rel)
    write(R/'D057_EXECUTION_AND_REPORT_SHA256_MANIFEST.json',{'files':files,'reviewed_config_sha256':approval['config_sha256'],
        'reviewed_preflight_sha256':approval['preflight_manifest_sha256'],'reviewed_runner_sha256':approval['GPU_runner_sha256'],
        'scope':'Complete execution and offline analysis; excludes this manifest and later ZIP/verification receipts to avoid circular hashes.'})
    print(json.dumps({'status':'D057_SCIENTIFIC_ASSESSMENT_COMPLETE_STOP_FOR_REVIEW','files_bound':len(files),'GPU_calls_this_script':0,'model_loads_this_script':0},indent=2))

if __name__=='__main__':main()
