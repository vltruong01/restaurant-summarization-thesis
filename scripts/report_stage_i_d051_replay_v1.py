"""Finalize D-051 saved evidence and readiness assessment offline only."""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from stage_i_d051_numeric import compare_tensor_maps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_i/d051_replay_investigation_v1'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    before=read(OUT/'PRE_EXECUTION_INTEGRITY.json')
    prep=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
    assert sha(ROOT/'configs/stage_i_paired_lora_pilot_v1.json')==before['config_sha256']
    assert sha(prep/'PREPARATION_MANIFEST.json')==before['preparation_manifest_sha256']
    for path,digest in before['prepared_files'].items():assert sha(prep/path)==digest,path
    for group in ['code','upstream','original_archive_files','D050_and_prior_feasibility_files']:
        for path,digest in before[group].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/'scripts/run_stage_i_d051_replay_v1.py')==before['D051_runner_sha256']
    assert sha(ROOT/'scripts/stage_i_d051_numeric.py')==before['numeric_helper_sha256']
    cache=Path.home()/'.cache/huggingface/hub/models--allenai--led-base-16384/snapshots/38335783885b338d93791936c54bb4be46bebed9'
    for name,row in before['cached_backbone_files'].items():assert sha(cache/name)==row['sha256']
    write(OUT/'POST_EXECUTION_INTEGRITY.json',{'status':'PASS','original_archive_files_unchanged':len(before['original_archive_files']),
        'D050_and_prior_feasibility_files_unchanged':len(before['D050_and_prior_feasibility_files']),
        'prepared_artifacts_unchanged':len(before['prepared_files']),'reviewed_code_unchanged':len(before['code']),
        'upstream_files_unchanged':len(before['upstream']),'cached_backbone_files_unchanged':len(before['cached_backbone_files']),
        'D051_runner_and_numeric_helper_bound_to_approval':True,'held_out_access':False,'accepted_content_modified':False})
    run=OUT/'execution';a=OUT/'analysis'
    weights=read(a/'adapter_weight_comparison.json');grads=read(a/'preclip_gradient_comparison.json')
    updates=read(a/'adapter_update_delta_comparison.json');micro=read(a/'microbatch_comparison.json')
    entry=read(a/'entry_state_comparison.json');end=read(a/'end_scheduler_scaler_RNG_comparison.json')
    ctx=read(a/'loss_clip_scaler_step_context.json');memory=read(run/'GPU_MEMORY_FINAL.json')
    completion=read(run/'completion.json');integrity=read(run/'backbone_integrity.json')
    assert entry['exact'] and end['exact'] and not integrity['frozen_changed'] and integrity['frozen_gradients_absent']
    assert completion['microbatches']==32 and completion['physical_optimizer_calls']==4 and completion['error'] is None
    assert micro['all_inputs_exact'] and micro['all_logits_exact'] and micro['all_losses_exact']
    original=torch.load(run/'original_step3_end.pt',map_location='cpu',weights_only=False)['adapters']
    replay=torch.load(run/'replayed_step3_end.pt',map_location='cpu',weights_only=False)['adapters']
    start=torch.load(run/'original_step3_start.pt',map_location='cpu',weights_only=False)['adapters']
    failures=[]
    for name in sorted(original):
        fail=~torch.isclose(original[name],replay[name],rtol=1e-5,atol=1e-6)
        for coordinates in fail.nonzero().tolist():
            index=tuple(coordinates);x=float(original[name][index]);y=float(replay[name][index]);s=float(start[name][index])
            failures.append({'tensor':name,'index':','.join(map(str,coordinates)),'original':x,'replayed':y,'start':s,
                'absolute_error':abs(y-x),'allowed_error_at_replayed_value':1e-6+1e-5*abs(y),
                'original_update':x-s,'replayed_update':y-s,'symmetric_relative_error_floor_1e_minus12':abs(y-x)/max(abs(x),abs(y),1e-12)})
    assert len(failures)==weights['overall']['failing_elements']==18
    with (OUT/'FAILING_ADAPTER_ELEMENTS.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
    clip_checks={}
    for label in ['original_step3','replayed_step3']:
        pre=torch.load(run/(label+'_preclip_gradients.pt'),map_location='cpu',weights_only=False)
        post=torch.load(run/(label+'_postclip_gradients.pt'),map_location='cpu',weights_only=False)
        clip_checks[label]=compare_tensor_maps(pre,post)['overall']
        assert clip_checks[label]['exact']
    write(OUT/'CLIPPING_EFFECT.json',{'pre_post_exact_per_path':clip_checks,'clipping_did_not_change_gradients':True})
    o=weights['overall'];u=updates['overall'];g=grads['overall']
    failing_tensors={n:r for n,r in weights['per_tensor'].items() if not r['allclose']}
    assert all('.lora_A.' in n and '.encoder.layers.' in n for n in failing_tensors)
    write(OUT/'READINESS_ASSESSMENT.json',{'assessment':'CONDITIONAL','strict_replay_gate':'FAIL_UNCHANGED',
        'pilot_execution_approved':False,'criterion_amendment_made':False,'strict_original_tolerance':{'rtol':1e-5,'atol':1e-6},
        'localization':'First observed mismatch in encoder accumulated backward gradients at microbatch1; forward logits/loss and restored entry exact. Includes checkpoint recomputation; exact kernel unproven.',
        'observed_deviations_small_in_this_bounded_run':True,'long_run_harmlessness_proven':False,
        'new_deterministic_policy_or_scientific_setting':False,'next_step':'Researcher review and explicit decision before any matched pilot or corrective experiment.'})
    branch_lines='\n'.join(f"| {name} | {v['numel']:,} | {v['different_tensors']}/{v['tensors']} | {v['failing_elements']} ({v['failing_pct']:.6f}%) | {v['max_abs']:.6g} | {v['rms']:.6g} | {v['relative_l2']:.6g} |" for name,v in weights['per_branch'].items())
    tensor_lines='\n'.join(f"| `{n.removeprefix('base_model.model.led.encoder.layers.')}` | {v['failing_elements']} | {v['max_abs']:.6g} | {v['rms']:.6g} |" for n,v in failing_tensors.items())
    loss_lines='\n'.join(f"| {i+1} | {x:.12f} | {y:.12f} | 0 |" for i,(x,y) in enumerate(zip(ctx['original_step3']['losses'],ctx['replayed_step3']['losses'])))
    optimizer=read(a/'optimizer_post_exp_avg.json')['overall'];optimizer_sq=read(a/'optimizer_post_exp_avg_sq.json')['overall']
    worst_module=max(updates['per_module'].items(),key=lambda pair:pair[1]['relative_l2'])
    report=f'''# Stage I — D-051 Encoder + Decoder LoRA replay investigation

**2026-10-16. Readiness assessment: CONDITIONAL. The unchanged strict replay gate remains FAIL; no pilot training approval or execution.**

One fresh B-only run reproduced D-050's mismatch under the original settings. Checkpoint restoration and complete forward outputs agree, while small differences first appear in encoder backward gradients. The evidence supports bounded numerical backward variability, not a restore bug or gross short-run divergence. It does not prove a particular CUDA kernel or harmlessness over 128 updates.

## Scope and integrity

The same pinned `allenai/led-base-16384` revision38335783885b338d93791936c54bb4be46bebed9, actual48 LoRA projections/589,824 parameters, rank8/alpha16, seed42, frozen train controls/order, batch1/accumulation8, FP32 weights/FP16 AMP, non-reentrant checkpointing, AdamW2e-4/decay0.01/clipping1 and linear128-step schedule/warmup7 are unchanged. Three distinct steps plus replay of step3 from saved step2: four physical optimizer calls /32 microbatches. No retries, extra forward probe, generation, policy/kernel/precision/tolerance change or pilot training.

Scientific config SHA256: `264e120200d9519edaf2cb974291c8bcdff9837be453a23c36041848f8d840e2`. The prepared load/optimizer/checkpoint functions are reused unchanged; the versioned D-051 loop adds observations and durable CPU-cloned evidence. Complete logits are hashed during the approved forward passes, not recomputed in extra passes. Observations can affect execution timing/live-memory residency, so neither timing nor memory is an uninstrumented throughput benchmark.

All161,844,480 pretrained parameters/295 frozen tensors remain unchanged and gradient-free. Intended local/global encoder and decoder adapters have finite/nonzero A/B gradients by step3, accounting for initial zero B and zero scheduled LR. All275 original archive files,91 D-050/prior feasibility files,14 prepared artifacts,22 upstream bindings,six reviewed code files and cached pretrained files match their pre-execution hashes. No accepted data/Teacher/checkpoint modification or held-out access.

## Strict parameter replay comparison

Criterion stays `torch.allclose(original, replayed, rtol=1e-5, atol=1e-6)` in FP32, with the same orientation. `torch.isclose` counts failing elements against `atol + rtol*abs(replayed)`; no criterion was relaxed.

- **{o['failing_elements']}/{o['numel']:,} elements fail: {o['failing_pct']:.8f}%**; seven tensors fail tolerance,38 tensors differ bitwise.
- Maximum absolute difference **{o['max_abs']:.12g}**; RMS **{o['rms']:.12g}**.
- Relative L2 versus total adapter weights **{o['relative_l2']:.12g}**; this is diluted by initially nonzero A matrices, so update-relative scale is also reported.
- Maximum symmetric elementwise relative error **{o['max_relative_symmetric']:.9g}** (171.35%), with denominator `max(abs(original),abs(replayed),1e-12)`. Near-zero/sign-changing values can produce a large relative percentage despite tiny absolute error. The703 elements with both magnitudes<=1e-6 have maximum absolute error1.75900069621e-7 and are not the strict failures. Relative metrics never replace the original allclose gate.

| Branch | Elements | Bitwise-different tensors | Failing elements | Max absolute error | RMS | Weight-relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{branch_lines}

All seven failing tensors are encoder LoRA A matrices: six local and one global. Many encoder B matrices differ bitwise but pass tolerance. `FAILING_ADAPTER_ELEMENTS.csv` lists every failing coordinate/value, original update, replay update and allowed error.

| Encoder tensor (layer prefix removed) | Failing elements | Max absolute error | RMS |
| --- | ---: | ---: | ---: |
{tensor_lines}

Full per-tensor and48-module JSON/CSV comparisons are in `analysis/adapter_weight_comparison*`; actual independent original/replay tensors and states are in `execution/*step3*.pt`.

## Where the paths first differ

1. **Restoration:** all1,151 entry-state leaves are exact, including adapters, optimizer moments/steps/groups, scheduler, scaler and all RNG states. Input/mask fingerprints and each microbatch's before/after-forward/backward RNG agree. No observed restore or data-order error.
2. **Forward:** all8 complete-logit fingerprints and loss scalars are exact. This observes the normal forward outputs; hidden activations recomputed inside checkpoint backward were not independently fingerprinted.
3. **Backward:** already after microbatch1,22 local and16 global encoder gradient fingerprints differ; decoder48 remain exact. Unscaled preclip differences remain encoder-localized. The first observed discrepancy precedes clipping and optimizer update, localizing it to encoder backward **including possible checkpoint recomputation**, not a proven individual operator.
4. **Clipping:** original/replay total norms0.994291603565 and0.994292795658 are below1. Raw pre/postclip tensors are exact within each path, so clipping introduces no observed difference here.
5. **Optimizer:** gradient differences propagate into encoder Adam moments and updates. Decoder gradients, moments and weights are exact. Adam step counters3, scheduler, scaler(scale65536), and RNG remain exact at the end. This is not evidence of an independent optimizer/state-restore fault; no independent synthetic optimizer replay is claimed.

Preclip gradient max absolute error **{g['max_abs']:.12g}**, RMS **{g['rms']:.12g}**, relative L2 **{100*g['relative_l2']:.6f}%**;1,416/589,824 elements fail the same tolerance when applied descriptively to gradients. Postclip comparisons are identical. Gradient-tolerance results are diagnostics, not the parameter acceptance rule.

Post-update Adam exp_avg max error **{optimizer['max_abs']:.12g}**, RMS **{optimizer['rms']:.12g}** (2 elements fail descriptive tolerance); exp_avg_sq max **{optimizer_sq['max_abs']:.12g}**, RMS **{optimizer_sq['rms']:.12g}** (0 fail). Original/replay end-state comparisons and named per-parameter moment CSVs are saved; restore fingerprints and full CPU state snapshots permit verification.

## Loss comparison

| Step-3 microbatch | Original loss | Replay loss | Absolute difference |
| --- | ---: | ---: | ---: |
{loss_lines}

Both means are **5.0361379981040955**. Finite loss/gradient checks pass for all32 microbatches, with no NaN/Inf, OOM or skipped optimizer step. Step3 controls are indices[1,2,0,1,2,0,1,2] in the frozen three-example order. No additional post-update evaluation was authorized or performed.

## Numerical significance and mechanism evidence

The observed update-difference L2 is **{100*u['relative_l2']:.6f}%** of the original whole-adapter step3 update; local encoder **0.060911%**, global **0.035128%**, decoder0. The largest projection-module update-relative L2 is **{100*worst_module[1]['relative_l2']:.6f}%**, at `{worst_module[0]}`. Thus total-weight normalization does not hide the update scale. The largest single coefficient error is3.7234e-6, about6.52% of used LR5.7143e-5; sparse worst-element error is not erased by a small aggregate RMS. Update-relative element percentages near zero are also unstable and are reported with safeguards in the CSV/JSON.

These are small, sparse deviations in this observed update, with identical ordinary forward outputs and exact state restoration. They do **not** establish long-run harmlessness, convergence, reproducibility across devices/seeds or no future effect on generated text. No 128-step drift bound can be inferred from one replay. Differences already appear at backward, rather than arising solely from Adam arithmetic or a failed restore.

The pinned source chain provides a plausible mechanism: LED overlapping `as_strided` sliding attention; Torch overlapping-view backward uses `index_add`; CUDA indexed accumulation can be nondeterministic under the unchanged default policy. See `SOURCE_LEVEL_NONDETERMINISM_HYPOTHESIS.md` and included source excerpts. Runtime evidence strengthens backward localization, but no profiler/operator-isolation test proves this kernel is responsible. Local/global differences may share an upstream backward source. No deterministic algorithms were enabled, no kernel disabled, and no precision change was tested.

## GPU memory and execution

Local RTX3050 Laptop6GB, Torch2.1.0+cu118 / transformers4.51.3 / PEFT0.15.2 / Accelerate1.6.0. Peak allocated **{memory['peak_allocated']/2**20:.3f}MiB**; reserved **{memory['peak_reserved']/2**20:.3f}MiB**. Wall time **{completion['seconds']:.2f}s** including loading, observations and final saved-evidence comparison. The complete105-second run is one bounded execution, not a training-time forecast. Peak statistics and per-microbatch loss/memory are durable even on the failure path.

Source shapes1×5120 and labels1×768 preserve whole accepted content/masks; logical budgets5000/4000/750/K5 are unchanged. Same real train controls: longest source p53i8ogh4pADR41GwaPRjg(4196/421 tokens), longest target WXgV2lOUgas7DzTLeDau-w(4151/619), zero-caption3_rsHAhzuqG0M3-EfMb-7A(2583/122). One global BOS, ignored label pads, decoder start/BOS/EOS handling and complete-token equality are checked.

## Readiness assessment and stop

**CONDITIONAL**, under the researcher's requested classification: strict replay still FAILS, but the quantified short-run deviation is small and localized to backward, with no observed incorrect restore, nonfinite behavior or gross divergence. This assessment does not change the acceptance criterion retrospectively or mark D-050 B as PASS.

A512-business×2-epoch exploratory paired pilot could be scientifically defensible **if the researcher explicitly accepts the documented non-bitwise encoder reproducibility limitation and decides the unresolved strict gate before execution**, using matched unchanged settings, fresh base initializations, full logs and the predefined quality evaluation. It cannot proceed automatically under the existing B strict gate. If strict replay is mandatory, a separately approved correction/diagnostic is needed; none is proposed as already effective or silently applied here.

The full Trainer loop, fresh-process restart and long-run/output quality remain untested. Capacity/location and two-epoch/single-seed limits remain; source-reference freeze and separate generation approval still apply. No full pilot, decoding experiment, Teacher request, held-out evaluation or accepted checkpoint overwrite. All D-050 outputs and original/replayed D-051 evidence remain preserved. Stop after D-051 for researcher review.
'''
    (OUT/'STAGE_I_D051_REPLAY_INVESTIGATION_REPORT.md').write_text(report,encoding='utf-8')
    (OUT/'TRAINING_READINESS_AFTER_D051.md').write_text('''# Training readiness after D-051

**Assessment: CONDITIONAL. Original strict replay gate: FAIL, unchanged. Pilot training: NOT authorized.**

One B-only replay reproduced the mismatch:18/589,824 parameters(0.0030518%) exceed rtol1e-5/atol1e-6; max error3.7234e-6/RMS1.5267e-8. All1,151 restored-entry leaves, all8 complete forward-logit/loss pairs, inputs and RNG states match. First divergence is in encoder backward gradient fingerprints before clipping/optimizer; decoder remains exact. Whole-step update-relative L2 deviation0.03515%, worst module0.12902%. All finite; real encoder/decoder gradients and memory feasibility are demonstrated.

Evidence supports small, localized numerical backward variability for this update; the exact CUDA kernel and harmlessness over128 steps are not proven. This is not a retrospective PASS or new acceptance threshold. No scientific setting, deterministic policy or tolerance changed.

An exploratory matched512-business/two-epoch pilot may be defensible only after the researcher explicitly reviews/accepts this reproducibility limitation and resolves the existing strict gate. Otherwise request a separately approved corrective/diagnostic plan; no correction is assumed effective. No pilot has started.

Fresh initialization, no reuse of feasibility weights, matched settings/order, existing quality protocol, source-reference freeze before outputs and separate generation approval remain. Full Trainer/fresh-process recovery/long-run stability/summary quality are untested; capacity/location and short single-seed screen limits remain.
''',encoding='utf-8')
    print(json.dumps({'assessment':'CONDITIONAL','strict_gate':'FAIL_UNCHANGED','failing_elements':len(failures),
        'max_abs':o['max_abs'],'rms':o['rms'],'update_relative_L2_pct':100*u['relative_l2'],
        'GPU_peak_allocated_MiB':memory['peak_allocated']/2**20,'GPU_peak_reserved_MiB':memory['peak_reserved']/2**20,
        'pilot_training_started':False},indent=2))


if __name__ == '__main__':main()
