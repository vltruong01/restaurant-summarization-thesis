"""Offline incident/results packaging. No retry, training or model/scorer load."""
import base64
import csv
import html
import io
import math
import shutil
import zipfile
from stage_i_unlikelihood_v2 import *

AUDIT=ROOT/'outputs/stage_i/unlikelihood_execution_v2'
REPORT=ROOT/'outputs/stage_i/unlikelihood_execution_report_v2'


def main():
    require((AUDIT/'STOP_ON_FAILURE_NO_RETRY.json').exists(),'This report handles the actual stopped numerical incident only')
    require(not REPORT.exists(),'Preserve completed report')
    REPORT.mkdir()
    cfg=read(CONFIG);manifest=read(PREP/'PREFLIGHT_MANIFEST.json');run=ROOT/cfg['run_root']
    for rel,digest in manifest['files'].items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==digest,'Frozen post-execution binding changed: '+rel)
    gate=run/'control_ce/feasibility'
    failure=read(gate/'FAILURE_STOP_NO_RETRY.json')
    require(failure['error']=='Nonfinite diagnostic gradient; stop','Unexpected incident')
    probes=[json.loads(s) for s in (gate/'UL_GRADIENT_CONTRIBUTION.jsonl').read_text(encoding='utf-8').splitlines()]
    objectives=[json.loads(s) for s in (gate/'objective.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(probes)==len(objectives)==1 and probes[0]['update_before']==0,'Unexpected measurement accounting')
    require(not (gate/'memory.jsonl').exists() and not list(gate.glob('checkpoint-*')),'Unexpected optimizer/checkpoint evidence')
    require(not (run/'treatment_ul').exists(),'A later arm must not execute after failure')
    approval=read(AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json')
    p=probes[0]
    write(REPORT/'ACTUAL_LOSS_AND_UL_SIGNAL.json',{'complete_probe_microbatches':1,'parameter_tensors':48,
        'trainable_parameters':294912,'microbatch_probe':p,
        'first_business_id_inferred_from_frozen_epoch_order':'4H6KdEMRlSrpFktX2k5gxg',
        'failed_next_business_id_inferred_from_frozen_epoch_order':'0Mjhi7hYia7iv8da7NBL3Q',
        'business_ID_provenance':'inferred from frozen epoch0 sampler/batch1; IDs not explicitly written by failed runner',
        'failing_tensor_name':None,'failing_component_CE_or_UL':None,
        'failed_probe_raw_gradients_persisted':False,'gradient_components_for_failed_microbatch':'NOT_AVAILABLE',
        'counterfactual_weighted_UL_not_applied_to_control':True})
    s=io.StringIO();writer=csv.DictWriter(s,fieldnames=['tensor','elements','CE_norm','UL_raw_norm','UL_weighted_0_1_norm',
        'UL_effective_norm','CE_missing','UL_missing','UL_zero_elements']);writer.writeheader()
    for name,value in p['per_tensor'].items():writer.writerow({'tensor':name,**value})
    write(REPORT/'PER_TENSOR_FINITE_PROBE.csv',s.getvalue())
    state={'status':'STOPPED_NUMERICAL_FEASIBILITY_FAILURE_NO_RETRY',
        'recommendation':'NO_GO_CURRENT_PILOT_EXECUTION_END_MODEL_OPTIMIZATION',
        'quality_conclusion':'NOT_EVALUATED; not evidence that UL hypothesis cannot improve summaries',
        'formal_researcher_quality_decision':'PENDING_NOT_ASSIGNED_BY_CODEX',
        'gates_invoked':1,'gate_passes':0,'gates_not_invoked':1,'gate_optimizer_updates_completed':0,
        'complete_logged_microbatches':1,'next_microbatch_failed_in_gradient_probe':True,
        'training_runs_invoked':0,'training_updates':0,'generation_calls':0,'new_predictions':0,
        'expected_predictions_if_gates_passed':48,'new_checkpoints':0,
        'actual_pretrained_and_adapter_binding_match_at_load':True,
        'postrun_in_memory_backbone_verification_reached':False,
        'frozen_disk_bindings_verified_unchanged':len(manifest['files']),
        'peak_VRAM':None,'measured_headroom_fraction':None,'memory_acceptance':'NOT_REACHED_NOT_PASS',
        'checkpoint_save_restore':'NOT_REACHED','treatment_gate':'NOT_STARTED_AFTER_CONTROL_FAILURE',
        'process_seconds':read(AUDIT/'FEASIBILITY_CONTROL_CE_EXIT.json')['seconds'],
        'runner_body_seconds_until_failure':failure['seconds'],
        'timing_note':'Process time includes import/startup; runner body includes CPU/model/GPU work; neither is isolated GPU kernel time',
        'approval_recorded_utc':approval['execution_authorized_at_utc'],'deadline_utc':approval['new_execution_deadline_utc'],
        'GPU_work_stopped_utc':read(AUDIT/'FEASIBILITY_CONTROL_CE_EXIT.json')['UTC'],
        'retries':0,'fallbacks':0,'coefficient_changes':0,'teacher_calls':0,'held_out_access':False,
        'further_model_experiments_started':0,'D051_strict_replay':'FAIL_UNCHANGED'}
    write(REPORT/'EXECUTION_AND_STOP_ACCOUNTING.json',state)
    write(REPORT/'CHECKPOINT_AND_PREDICTION_AVAILABILITY.json',{'new_checkpoints':[],
        'new_predictions':[],'reason':'Control feasibility stopped before first optimizer update; conditional training and48generation scope never opened',
        'saved_D059_epoch3_benchmark_outputs':24,'benchmark_is_not_new_pilot_output':True})
    write(REPORT/'POSTRUN_INTEGRITY.json',{'status':'PASS_DISK_ARTIFACTS_UNCHANGED','verified_bindings':len(manifest['files']),
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
        'runner_sha256':sha(RUNNER),'new_model_experiments':0,'D051_strict_replay':'FAIL_UNCHANGED'})
    text=f'''# Unlikelihood v2 — stopped feasibility execution report

**STOPPED at the first CE-control gate. No retry/fallback or later GPU stage.**

Researcher approval recorded{approval['execution_authorized_at_utc']}; hard deadline
{approval['new_execution_deadline_utc']}. Historical D059/D060 clocks unchanged.
Three exact approved SHA256 +482workspace bindings +runtime pins PASS before execution.
Actual GPU: NVIDIA GeForce RTX3050Laptop6GB. Intended24modules/294912parameters
matched actual PEFT attachment and initial historical F/backbone tensor fingerprints.

## Actual execution

One of two gates invoked; zero gates PASS. A complete first microbatch was recorded.
The next attempted microbatch raised `Nonfinite diagnostic gradient; stop` while
checking separate CE/UL gradients. Trainer was still at global_step0; no optimizer
update was completed. No treatment gate, paired training, checkpoint or generation
followed.0/48new outputs exist; no output or checkpoint has been fabricated.

Process elapsed{state['process_seconds']:.3f}s; runner body until failure
{failure['seconds']:.3f}s (includes setup, not isolated GPU compute). No peak VRAM
or15%headroom acceptance was recorded before the optimizer-step hook. Report them
as unavailable, not0and notPASS. Checkpoint restoration was not reached.

## Finite first-microbatch evidence (not a distribution/quality result)

| Measurement | Actual |
|---|---:|
| CE loss | {p['CE_loss']:.8f} |
| UL scalar loss | {p['UL_loss_raw']:.8f} |
| Counterfactual0.1UL / CE loss | {p['weighted_UL_loss_over_CE']:.6f} |
| CE gradient norm | {p['CE_gradient_norm']:.8f} |
| Raw UL gradient norm | {p['UL_gradient_norm_raw']:.8f} |
| Counterfactual0.1UL / CE gradient norm | {p['weighted_UL_gradient_norm_over_CE']:.6f} |
| CE/UL gradient cosine | {p['CE_UL_gradient_cosine']:.8f} |
| Current AMP scale | {p['amp_scale']:.0f} |
| Trainable tensors measured | {p['parameter_tensors']} |

All48tensors in this FIRST probe had finite/nonzero CE and UL gradients, no missing
UL gradient and no zero UL tensor elements. .grad buffers/RNG were unchanged by
the probe. Control effective UL coefficient was0; counterfactual0.1weighted norm
is observational, NOT an applied UL update. A20.3%norm ratio on one sample does
not establish usefulness, adequate dataset-wide signal, or future optimization
stability.0.1weight and all precision/numerical settings were never changed.

## Verified failure and unresolved cause

The exception proves a nonfinite CE OR UL component gradient was found after the
separate AMP-scaled diagnostic traversals. It does NOT identify the offending
tensor/component: the assertion ran before that failed probe was serialized.
Failed-probe gradients/loss values cannot be recovered from saved evidence.
This is a logging limitation; do not invent the missing measurements.

AMP scaling of separate, unweighted UL probes (scale65536on the first successful
probe) is a possible numerical explanation. Instability in a component graph or
its interaction with checkpoint recomputation is also unresolved. No evidence
here proves a particular cause, weak coefficient, corrupt dataset, OOM, or that
CE-only optimization itself would fail. No extra GPU diagnosis/reproduction was
run, and no approved code/tolerance/scaler/precision was patched.

Initial in-memory frozen-base hashes matched. Final in-memory verification was
not reached; post-exit all482bound disk artifacts still match. D051strict replay
FAIL and all original RO/U/F,D058,D059,D060 evidence remain unchanged.

## Quality review and recommendation

No new summaries exist, so harmful repetition, Core coverage, opinion balance,
factual/visual grounding, zero-caption behavior and completeness cannot be scored
for this pilot. The24saved D059Expanded-LoRA epoch3 outputs are supplied only as
the prespecified practical non-causal benchmark, not as new paired outputs.
No researcher-confirmed output labels or scientific qualityGO/NO-GO is assigned.

**Recommendation:NO-GO for continuing this pilot under the current execution;
quality remains untested. End model optimization under the approved strict stop.**
This numerical feasibility failure is not proof the UL hypothesis is ineffective.
Do not automatically repair/retry, trainRO/U, change decoding or start another
experiment. Return to researcher review and prioritize the final thesis evaluation.
No held-out evaluation is launched by this report.
'''
    write(REPORT/'EXECUTION_REPORT.md',text)
    source_root=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1/source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
    cases=read(source_root/'source_cases.json');references=read(ROOT/cfg['final_references'])
    inputs=[json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    require([c['business_id'] for c in cases]==[c['business_id'] for c in inputs],'Review source IDs changed')
    benchmark=read(PREP/'D059_EPOCH3_PRACTICAL_BENCHMARK.json')
    bench={r['business_id']:r for r in benchmark['predictions']}
    write(REPORT/'SOURCE_CASES.json',cases);write(REPORT/'D059_EPOCH3_SAVED_BENCHMARK.json',benchmark)
    page=['<!doctype html><meta charset="utf-8"><title>UL v2 stopped gate review</title>',
        '<style>body{max-width:1250px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap;background:#f4f5f7;padding:16px}img{max-width:280px;max-height:250px}section{border-top:2px solid #ddd;padding:20px}summary{cursor:pointer;font-weight:bold}</style>',
        '<h1>Unlikelihood v2: feasibility stopped</h1><p>No new CE/UL summaries or checkpoints. Numerical failure before update1. The outputs below are SAVED D059epoch3 benchmark only; no paired quality result.</p>',
        '<details><summary>Actual execution report</summary><pre>'+html.escape(text)+'</pre></details>']
    image_bindings={}
    for c,inp in zip(cases,inputs):
        bid=c['business_id'];require(c['source_sha256']==inp['source_sha256']==bench[bid]['source_sha256'],'Source mapping mismatch')
        units=[u for u in references['units'] if u['business_id']==bid]
        page.append('<section><h2>'+html.escape(c['display_name']+' — '+bid+' / '+c['subset'])+'</h2>')
        page.append('<p>CE: NOT GENERATED. UL: NOT GENERATED. Human output-quality assessment: NOT AVAILABLE.</p>')
        page.append('<details><summary>Original LED-visible source / selected reviews and captions</summary><pre>'+html.escape(c['source'])+'</pre></details>')
        page.append('<details><summary>Original Teacher target (comparison, not independent truth)</summary><pre>'+html.escape(inp['target'])+'</pre></details>')
        page.append('<details><summary>Approved D053 references — retain Core/Supporting/Exclude distinctions</summary><pre>'+html.escape(json.dumps(units,ensure_ascii=False,indent=2))+'</pre></details>')
        page.append('<details><summary>Corresponding original photos / captions</summary>')
        for cap in c['captions']:
            src=source_root/cap['image_file'];raw=src.read_bytes();image_bindings[cap['image_file']]=sha(src)
            dest=REPORT/cap['image_file'];dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest)
            mime='image/png' if raw.startswith(b'\x89PNG') else 'image/webp' if raw[:4]==b'RIFF' else 'image/jpeg'
            page.append('<figure><img src="data:'+mime+';base64,'+base64.b64encode(raw).decode()+'"><figcaption>'+html.escape(cap['caption_id']+' / PHOTO '+cap['photo_id']+': '+cap['text'])+'</figcaption></figure>')
        page.append('</details><h3>Saved D059Expanded-LoRA epoch3 (non-causal context only)</h3><pre>'+html.escape(bench[bid]['text'])+'</pre></section>')
    write(REPORT/'REVIEW.html','\n'.join(page));write(REPORT/'ORIGINAL_PHOTO_BINDINGS.json',image_bindings)
    bundle=ROOT/'outputs/stage_i/unlikelihood_execution_review_bundle_v2';bundle.mkdir(exist_ok=False)
    def cp(src,rel):
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for folder,rel in [(REPORT,'report'),(AUDIT,'execution'),(run,'partial_run'),(PREP,'preparation')]:
        for src in folder.rglob('*'):
            if src.is_file() and 'documentation_before' not in str(src):cp(src,Path(rel)/src.relative_to(folder))
    code=['scripts/stage_i_unlikelihood_v2.py','scripts/run_stage_i_unlikelihood_pilot_v2.py',
          'scripts/execute_stage_i_unlikelihood_authorized_v2.py','scripts/report_stage_i_unlikelihood_execution_v2.py',
          'scripts/prepare_stage_i_unlikelihood_v2.py','scripts/audit_stage_i_unlikelihood_v2.py',
          'scripts/compare_stage_i_unlikelihood_v2.py','tests/test_stage_i_unlikelihood_v2.py',
          'configs/stage_i_unlikelihood_pilot_v2.json','specs/009q_unlikelihood_sentence_only_amendment_v2.md',
          'thesis_sdd_docs/UNLIKELIHOOD_PILOT_AMENDMENT_V2.md']
    for rel in code:cp(ROOT/rel,rel)
    cp(ROOT/cfg['final_references'],'references/FINAL_RESEARCHER_APPROVED_REFERENCES.json')
    cp(ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl','evidence/FROZEN24_DEVELOPMENT_INPUTS_AND_TEACHER_TARGETS.jsonl')
    cp(ROOT/'outputs/stage_i/unlikelihood_review_bundle_v1/evidence/TRAIN512_ORIGINAL_SOURCES_AND_TEACHER_TARGETS.jsonl',
       'evidence/TRAIN512_ORIGINAL_SOURCES_AND_TEACHER_TARGETS.jsonl')
    write(bundle/'README_FIRST.md','''# UL v2 actual execution: numerical gate failure

Read report/EXECUTION_REPORT.md and report/EXECUTION_AND_STOP_ACCOUNTING.json.
Open report/REVIEW.html for24frozen evidence cases/photos/approved references and
SAVED D059epoch3 benchmark outputs. No CE/UL model summaries were generated.
One CE gate invoked; nonfinite diagnostic gradient on the next microbatch after
one finite saved probe.0optimizer updates,0checkpoints,0training,0new predictions.
No treatment gate, retry/fallback/scaler/precision/coefficient change or extra GPU
diagnostic. The failed probe did not persist the offending tensor/component.
Do not manufacture missing evidence or classify scientific quality from no outputs.
The approval and exact bound code, tracebacks and partial logs are preserved.
Peak VRAM/headroom/restoration were not reached, not reported asPASS.
SHA256_MANIFEST.json covers payload. This is a review bundle, not authorization
or a runnable full-training handoff; weights and accepted datasets remain untouched.
''')
    write(bundle/'CHATGPT_REVIEW_REQUEST.md','''# Independent scientific review after stopped approved UL v2 gate

The explicit stop requirement was enforced: first gate failed nonfinite component
gradient before update1; later gate/train/generation never ran. Review actual
finite first-probe CE/UL losses,48per-tensor gradients, AMPscale65536,counterfactual
weighted gradient ratio20.3%and cosine. Control UL effective weight0.
The failed next probe's tensor/component was not serialized; cause remains unknown.
Possible AMP/probe numerical explanations are hypotheses, not verified causes.
No loss-weight/precision adjustment, retry, extra GPU test or new experiment occurred.
Evaluate integrity/accounting and limitations. Quality of new CE/UL summaries is
NOT EVALUATED because0/48exist. Saved D059epoch3 output is practical context only.
Recommendation:NO-GO for continuing this execution; conclude model optimization
and prioritize final thesis evaluation. Researcher formal judgment remains theirs.
Preserve D051FAIL/all historical artifacts and deadlines. This package does not
request or authorize another optimization experiment or held-out execution.
''')
    files={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'SHA256_MANIFEST.json',{'files':files,'payload_files':len(files)})
    archive=ROOT/'outputs/stage_i/stage_i_unlikelihood_execution_review_v2.zip'
    require(not archive.exists(),'Preserve delivered review ZIP')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    receipt={'archive':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),'bytes':archive.stat().st_size,
        'payload_files':len(files),'new_predictions':0,'GPU_work_stopped':True,'status':state['status'],
        'no_retry':True,'formal_researcher_quality_judgment_pending':True}
    write(REPORT/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
