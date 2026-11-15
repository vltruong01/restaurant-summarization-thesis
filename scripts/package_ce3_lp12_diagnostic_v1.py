"""Export stopped one-shot attempt truthfully; no inference or guard changes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import zipfile
from tokenizers import Tokenizer

R=Path(__file__).resolve().parents[1]
E=R/'outputs/stage_i/led_large_ce3_lp12_diagnostic_v1'
D=E/'downloaded'
O=E/'portable_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,x):
    p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    assert not O.exists()
    exported=read(D/'EXPORT_MANIFEST.json')['files']
    for rel,digest in exported.items():assert sha(D/rel)==digest,rel
    for rel,digest in read(D/'TRANSFER_MANIFEST.json')['files'].items():assert sha(D/rel)==digest,rel
    old=R/'outputs/stage_i/led_large_nbest_v2_execution_v1/portable_review'
    previous=read(old/'SHA256_MANIFEST.json')['files']
    for rel,digest in previous.items():assert sha(old/rel)==digest,rel
    failure=read(D/'result/FAILURE_STOP_NO_RETRY.json')
    assert failure['generation_calls']==failure['completed_cases']==failure['optimizer_updates']==0
    assert failure['stage']=='GPU_OWNERSHIP_AFTER_MODEL_BEFORE_INFERENCE'
    assert read(D/'result/CHECKPOINT_PRESERVATION.json')['all_components_unchanged']
    O.mkdir()
    # Exact original code/configuration and raw host evidence are retained unmodified.
    shutil.copytree(D,O/'EXECUTION_UNCHANGED')
    for name in ['run_led_large_ce3_lp12_diagnostic_v1.py','test_led_large_ce3_lp12_diagnostic_v1.py',
                 'prepare_deploy_ce3_lp12_diagnostic_v1.py','package_ce3_lp12_diagnostic_v1.py']:
        (O/'LOCAL_CODE').mkdir(exist_ok=True);shutil.copyfile(R/'scripts'/name,O/'LOCAL_CODE'/name)
    tokenizer_path=R/'assets/tokenizers/led-base-16384/tokenizer.json'
    assert sha(tokenizer_path)=='2bb1a22cfbe25b8e5a232b7fc4d7fc5073923b45724a5f813b00811bb6620f66'
    tok=Tokenizer.from_file(str(tokenizer_path))
    cases=read(D/'payload/TRAIN_SOURCE_ONLY_CASES.json');baseline=read(D/'payload/BASELINE_CANDIDATES.json')
    lookup={b['business_id']:b for b in baseline}
    rows=[]
    table=['# Case-level A/B status','',
           'All 12 original A outputs are preserved. B was not generated because the mandatory GPU ownership guard stopped the attempt. NA means not measured, not zero quality.','',
           '| TRAIN business | Captions | A review-body tokens | A visual-body tokens | A raw tokens incl. decoder start/EOS | B output | Quality comparison |',
           '|---|---:|---:|---:|---:|---|---|']
    for c in cases:
        b=lookup[c['business_id']]['single'];text=b['text'];raw=b['token_ids']
        enc=tok.encode(text,add_special_tokens=False)
        assert enc.ids==[i for i in raw if i not in (0,1,2)]
        a='[REVIEW_CONSENSUS]';v='[VISUAL_EVIDENCE]'
        assert text.count(a)==text.count(v)==1 and text.startswith(a)
        at=text.index(v)
        spans=[('review_header',0,len(a)),('review_body',len(a),at),
               ('visual_header',at,at+len(v)),('visual_body',at+len(v),len(text))]
        counts={label:0 for label,_,_ in spans};cross=[]
        for token_id,(start,end) in zip(enc.ids,enc.offsets):
            label=next(label for label,lo,hi in spans if lo<=start<hi)
            counts[label]+=1
            if any(start<boundary<end for boundary in (len(a),at,at+len(v))):
                cross.append(dict(token_id=token_id,start=start,end=end,assigned_to=label))
        assert sum(counts.values())==len(enc.ids)
        name=next(line[5:] for line in c['source'].splitlines() if line.startswith('name='))
        row=dict(business_id=c['business_id'],name=name,split='train',caption_count=len(c['captions']),
                 original_section_tokens=counts,original_raw_token_count=len(raw),
                 original_content_tokens=len(enc.ids),original_EOS_completion=raw[-1]==2,
                 original_length_cap_hit=len(raw)>=768,original_BOS_count=raw.count(0),
                 raw_content_retokenization_exact=True,tokens_crossing_section_boundary=cross,
                 LP12=None,LP12_status='NOT_GENERATED_GPU_OWNERSHIP_STOP',quality_comparison='NOT_ASSESSABLE')
        rows.append(row)
        table.append(f"| {name} | {len(c['captions'])} | {counts['review_body']} | {counts['visual_body']} | {len(raw)} | Not generated | NA |")
        folder=O/'cases'/c['business_id'];folder.mkdir(parents=True)
        (folder/'ORIGINAL_CE3_LP1_0.txt').write_bytes(text.encode())
        (folder/'FROZEN_MODEL_VISIBLE_SOURCE.txt').write_bytes(c['source'].encode())
        write(f'cases/{c["business_id"]}/STATUS.json',row)
    write('SECTION_TOKEN_LENGTH_STATISTICS.json',dict(cases=rows,baseline_tokenizer_sha256=sha(tokenizer_path),
        definition='Pinned tokenizers 0.21.1 tokenizer.json; full decoded A re-encoded without specials equals raw content IDs in every case. Section assignment uses full-text token offset start; bodies include whitespace/newline separators, headers separate; EOS/decoder-start excluded from section counts. Raw totals include start and terminal EOS. Crossing-boundary tokens are explicitly listed.',
        LP12_statistics=None,LP12_not_generated=True,
        baseline_totals={k:sum(row['original_section_tokens'][k] for row in rows) for k in rows[0]['original_section_tokens']}))
    write('LP12_PREDICTIONS_NOT_CREATED.json',dict(returned_sequences=0,predictions=[],
        reason=failure['error'],do_not_interpret_missing_as_empty_generation=True))
    (O/'CASE_LEVEL_AB_COMPARISON.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    ownership=read(D/'result/OWNERSHIP_AFTER_MODEL.json')
    used=read(D/'result/MEMORY_SAMPLES.json')['samples']
    observed_used=max(x['used'] for x in used);observed_total=used[-1]['total']
    write('ARTIFACT_INTEGRITY_REPORT.json',dict(downloaded_export_entries_verified=len(exported),
        original_NBEST_files_reverified_after_attempt=len(previous),
        transfer_entries_verified=len(read(D/'TRANSFER_MANIFEST.json')['files']),
        checkpoint_component_hashes_after_attempt='ALL_UNCHANGED',
        adapter_sha256=read(D/'result/CHECKPOINT_PRESERVATION.json')['adapter_sha256'],
        historical_baseline_config_exact=True,baseline_raw_content_token_identities=12,
        local_CPU_contract_tests_passed=5,local_CPU_contract_tests_failed=0,
        LP12_generation_calls=0,new_optimizer_updates=0))
    report=f'''# LED-large CE3 Length Penalty 1.2 Diagnostic V1

**Execution status: BLOCKED / STOPPED by the unchanged mandatory GPU ownership guard. Scientific recommendation: INCONCLUSIVE — LP1.2 was not evaluated.** There were **zero generation calls, zero LP1.2 outputs and zero optimizer updates**. This is neither a quality failure nor evidence of an improvement from length penalty.

## Authorized scope and exact configuration

The researcher authorized only 12 existing TRAIN businesses, 9 captioned and 3 zero-caption. The Original CE DecoderPolicyV2 epoch-3/step-750 adapter is `507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60`. Original N1 predictions were reused after verification. The full effective baseline configuration matches the historical execution configuration. B changes only length_penalty 1.0 → 1.2; num_beams=4, max_length=768, do_sample=false and every other generation field remain unchanged.

A new hash-bound receipt records first clock observation of the current authorization at 2026-10-24T19:42:53Z, with a conservative maximum 48-hour validity and one-hour worker cap. Exact chat-message UTC is unavailable and was not invented. The worker used a new isolated namespace `/workspace/CE3_LP12_20261024T194253Z`. No prior authorization was reused.

## Successful preflight checks

- 171/171 accepted N-best source-bundle file hashes verified before transfer and again after the stopped attempt.
- All 12 historical N1 raw token arrays match saved predictions and successful call receipts; exact TRAIN identities/source hashes and input/mask geometry match. No resampling occurred.
- Full original code, pretrained weights, tokenizer, checkpoint components and installed runtime source files passed the existing immutable verifier. Original effective generation config also matched the model loader.
- Pinned runtime: Torch 2.1.0+cu118, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, NumPy 1.26.4, tokenizers 0.21.1, safetensors 0.8.0, huggingface-hub 0.36.2.
- Five new CPU contract tests passed, including the one-parameter decoding difference and unchanged ownership-guard rejection of an unequal PID. No new CUDA test/model forward was added.
- Dedicated target identified as RTX3090, UUID GPU-386606c5-3717-8b4f-c0fe-70907f18b717, driver595.91.07. The initial ownership check passed on an idle GPU. The original loader placed FP32 model parameters on GPU with the original CE3 adapter, eval mode, BOS-only global attention and planned FP16 autocast unchanged. Frozen input-tokenization checks passed before the second guard.

## Exact stopping evidence

After model loading and before the first generate call, the worker saved the ownership inventory at {ownership['utc']}:

| Observation | Value |
|---|---|
| Worker PID inside container | 2610 |
| /proc/self/status NSpid | 2610 |
| nvidia-smi compute PID | 2606305 |
| nvidia-smi process name | [Not Found] |
| nvidia-smi compute memory | 2040 MiB |
| GPU memory used in same snapshot | 2050 MiB |
| Unchanged guard condition | every compute PID must equal os.getpid() |
| Raised error | Foreign GPU compute workload; do not run this gate |

The failure stage is `GPU_OWNERSHIP_AFTER_MODEL_BEFORE_INFERENCE`. The existing guard function was imported unchanged from its hash-verified implementation; no namespace allowlist, PID remapping, threshold relaxation or fallback was applied. The rejected PID is now preserved, unlike the older Boundary failure.

A host/container PID namespace mismatch is **plausible but not authenticated** by these observations. They do not establish that PID2606305 was this worker, or conclusively exclude a foreign process. The later idle snapshot cannot establish ownership retroactively. The researcher's explicit stop rule applies, so no inference, retry or ownership investigation was started after the failure.

## Resource and preservation results

The worker stopped after {failure['seconds']:.2f} seconds. Saved memory samples show {observed_used/1024**3:.3f} GiB used out of {observed_total/1024**3:.3f} GiB after model loading. This is **not an inference peak** and cannot establish the required 15% headroom under beam4 decoding; that resource gate remains unverified. No numerical/EOS/length behavior of LP1.2 was measured.

All original checkpoint component hashes were verified unchanged after failure, including the CE3 adapter. No optimizer was constructed, no backward pass occurred, and no training or model-weight edit took place. The one-shot supervisor service exited, with autostart=false and autorestart=false. A read-only post-exit GPU query showed1MiB used,0% utilization and no listed compute process. The instance was retained; no stop/destroy action was requested or performed here.

The worker catches the exception to flush evidence and therefore returned OS code0. **That return code is not a PASS:** FAILURE_STOP_NO_RETRY.json is authoritative. There is no COMPLETION.json, no LP1.2 output and no automatic downstream task. This receipt-semantics limitation is disclosed without rewriting the executed code or historical logs.

## A/B comparison and section lengths

[The 12-case table](CASE_LEVEL_AB_COMPARISON.md) lists every original baseline and the missing B measurement. All12 original texts and frozen model-visible sources are embedded under cases/. [Section token statistics](SECTION_TOKEN_LENGTH_STATISTICS.json) report baseline review/visual/header lengths using the pinned tokenizer, with exact raw-content token identity checks. LP1.2 statistics are explicitly null, not zero. Terminal EOS exists in all saved baseline outputs, with no baseline cap hit; no equivalent claim can be made about B.

Source-supported improvements, regressions, mixed-opinion changes, visual fusion, additional useful facts, verbosity and readability **cannot be compared** without B outputs. No favorable examples, artificial predictions or model-assisted quality scores were fabricated. Teacher-target comparison was not performed because the run stopped before producing a candidate; no new targets were created or altered. A longer-output hypothesis remains untested.

## Decision and stop

**INCONCLUSIVE scientifically; BLOCKED operationally.** Preserve Original CE3 and length_penalty1.0 as the provisional baseline. This failure does not support or reject length_penalty1.2 on summary quality. Any attempt to establish host/container ownership or correct a guard requires a separate researcher decision; none is proposed as an automatic continuation or implemented here. No further decoding value or experiment was tried.

No validation/TEST payload, RO/U, Shuffled, references, human-evaluation activity, Boundary experiment or Source-Bound renderer was executed. Only the existing standalone ownership function from Boundary code was reused as a protection; no Boundary diagnostic/training entry point ran.

## Included evidence

- EXECUTION_UNCHANGED/: exact authorization, source/request attachment, original runner/config, input/baseline packet, raw GPU snapshots, traceback, memory samples and checkpoint-preservation receipt.
- CASE_LEVEL_AB_COMPARISON.md and cases/: complete baseline coverage; B is absent by design after the stop.
- SECTION_TOKEN_LENGTH_STATISTICS.json and ARTIFACT_INTEGRITY_REPORT.json.
- LOCAL_CODE/: exact CPU tests, transfer/launcher and packaging source.
- SHA256_MANIFEST.json: every included file, excluding itself. Historical exported manifest paths are relative to EXECUTION_UNCHANGED/, not the package root.

Stop for independent researcher review. No retry is scheduled.
'''
    (O/'LENGTH_PENALTY_DIAGNOSTIC_REPORT.md').write_text(report,encoding='utf-8')
    write('FINAL_DECISION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        execution='BLOCKED_GPU_OWNERSHIP_GUARD',scientific_decision='INCONCLUSIVE_NO_LP12_GENERATIONS',
        completed_pairs=0,baseline_cases_preserved=12,automatic_retry=False,
        GPU_guard_modified=False,original_adapter_unchanged=True,stop_for_review=True))
    manifest={p.relative_to(O).as_posix():sha(p) for p in sorted(O.rglob('*')) if p.is_file()}
    write('SHA256_MANIFEST.json',dict(files=manifest,algorithm='SHA256',self_excluded=True))
    target=R/'outputs/cloud_handoff/LED_LARGE_CE3_LP12_DIAGNOSTIC_V1_REVIEW.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(O.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(O).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for rel,digest in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    with Path(str(target)+'.sha256').open('x',encoding='ascii') as f:f.write(sha(target)+'  '+target.name+'\n')
    print(json.dumps(dict(zip=str(target),bytes=target.stat().st_size,sha256=sha(target),
                         portable_files=len(manifest),export_files=len(exported),cases=12,LP12_outputs=0),indent=2))


if __name__=='__main__':main()
