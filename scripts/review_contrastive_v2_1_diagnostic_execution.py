"""CPU-only independent recomputation and portable packaging of the single diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2_1_execution_v1'
PREP=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1'
RUN=OUT/'downloaded/runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY/diagnose_pair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    # Deliberate CPU-only analysis of saved tensors, never a new model forward/diagnostic.
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    report=read(RUN/'lambda0/LAMBDA0_COMPARISON.json')
    snapshots=report['raw_snapshots']
    for meta in snapshots.values():assert sha(RUN/'lambda0'/meta['path'])==meta['sha256']
    native=torch.load(RUN/'lambda0/native_gradients.pt',map_location='cpu',weights_only=True)
    delegated=torch.load(RUN/'lambda0/delegated_gradients.pt',map_location='cpu',weights_only=True)
    assert native['parameter_names']==delegated['parameter_names']
    count=total=failures=bitexact=0;maximum=0.;groups={};rows=[]
    for name in native['parameter_names']:
        a=native['gradients'][name];b=delegated['gradients'][name]
        assert a.device.type==b.device.type=='cpu' and a.dtype==b.dtype==torch.float32 and a.shape==b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        close=torch.isclose(b,a,atol=1e-5,rtol=1e-4,equal_nan=False)
        failed=int((~close).sum());error=float((a.double()-b.double()).abs().max())
        row=dict(name=name,elements=a.numel(),failing_elements=failed,allclose=bool(close.all()),bit_exact=bool(torch.equal(a,b)),max_absolute_difference=error)
        rows.append(row);count+=1;total+=a.numel();failures+=failed;maximum=max(maximum,error);bitexact+=row['bit_exact']
        group=('encoder' if '.encoder.' in name else 'decoder')+('_A' if '.lora_A.' in name else '_B')
        g=groups.setdefault(group,dict(tensors=0,failing_tensors=0,elements=0,failing_elements=0,max_abs=0.))
        g['tensors']+=1;g['failing_tensors']+=int(failed>0);g['elements']+=a.numel();g['failing_elements']+=failed;g['max_abs']=max(g['max_abs'],error)
    saved={x['name']:x for x in report['per_tensor']}
    assert len(saved)==len(rows)==456
    for row in rows:
        old=saved[row['name']]
        assert row['failing_elements']==old['failing_element_count'] and row['allclose']==old['allclose']
        assert row['max_absolute_difference']==old['max_absolute_difference']
    ctxa=read(RUN/'lambda0/native_context.json');ctxb=read(RUN/'lambda0/delegated_context.json')
    checks=dict(RNG_starts_equal=ctxa['RNG_start']==ctxb['RNG_start'],RNG_ends_equal=ctxa['RNG_end']==ctxb['RNG_end'],
        modes_start_equal=ctxa['mode_start']==ctxb['mode_start'],native_modes_unchanged=ctxa['mode_start']==ctxa['mode_end'],
        delegated_modes_unchanged=ctxb['mode_start']==ctxb['mode_end'],batch_start_equal=ctxa['batch_start']==ctxb['batch_start'],
        native_batch_unchanged=ctxa['batch_start']==ctxa['batch_end'],delegated_batch_unchanged=ctxb['batch_start']==ctxb['batch_end'])
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    original=read(ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1/positives/EzJI5x25U5GgbYBWhu5ZQQ.json')
    expected=CorrectedLabelCollator()([dict(input_ids=original['input_ids'],labels=original['corrected_labels'])])
    expected_signature={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in expected.items()}
    checks['batch_matches_frozen_Tempo_positive']=ctxa['batch_start']==expected_signature
    y=expected['labels'];shifted=torch.full_like(y,1);shifted[:,1:]=y[:,:-1];shifted[:,0]=2;shifted[shifted==-100]=1
    checks['both_decoder_prefixes_match_corrected_labels']=ctxa['decoder_ids']==ctxb['decoder_ids']==shifted.tolist()
    assert all(checks.values()) and report['preconditions_pass']
    summary=dict(status='INDEPENDENT_CPU_RECOMPUTATION_MATCHES_GPU_DIAGNOSTIC_EVIDENCE',
        tensors=count,total_gradient_elements=total,failing_tensors=sum(not x['allclose'] for x in rows),
        passing_tensors=sum(x['allclose'] for x in rows),bit_exact_tensors=bitexact,
        failing_elements=failures,failing_element_fraction=failures/total,max_absolute_difference=maximum,
        both_native_and_delegated_all_finite=True,native_loss=float(native['loss']),delegated_loss=float(delegated['loss']),
        loss_bit_exact=bool(torch.equal(native['loss'],delegated['loss'])),checks=checks,groups=groups,
        thresholds=dict(atol=1e-5,rtol=1e-4,reference='native'),additional_GPU_calls=0,model_forwards=0,optimizer_updates=0)
    write(OUT/'INDEPENDENT_CPU_RECOMPUTATION.json',dict(summary=summary,per_tensor=rows))
    closeout=read(OUT/'downloaded/operations/FAILURE_CLOSEOUT.json')
    memory=read(RUN/'MEMORY_EVIDENCE.json')['summary']
    final=dict(status='FAIL_STOP_LAMBDA0_OUTSIDE_LOCKED_TOLERANCE',diagnostic_attempts=1,
        authorization_attempt_consumed=True,successful_optimizer_updates=0,generation_calls=0,
        baseline_inference=False,CE_training=False,contrastive_training=False,automatic_retry=False,automatic_continuation=False,
        lambda0_within_tolerance=False,lambda0_bit_exact=False,lambda0_preconditions_pass=True,
        numeric_summary=summary,observed_memory_at_early_stop=memory,
        full_intended_effective_batch_memory_headroom='UNVERIFIED_NOT_REACHED',
        joint_vs_sequential_gradients='NOT_REACHED',negative_own_prefix_GPU_check='NOT_REACHED',
        FP16_eight_example_accumulation='NOT_REACHED',AMP_skip_optimizer_integrity='NOT_REACHED',
        root_cause='UNCONFIRMED',scientific_quality_conclusion='UNASSESSED_NO_TRAINING_OR_GENERATION',
        preservation=closeout,stop_for_independent_ChatGPT_review=True)
    write(OUT/'FINAL_EXECUTION_REPORT.json',final)
    text=f'''# V2.1 single GPU diagnostic — FAIL / STOP

Exactly one fresh RTX3090 diagnostic ran under the locked approved V2.1 contract. It stopped at
LAMBDA0_NATIVE_CE because gradient differences exceed the unchanged atol=1e-5,rtol=1e-4.
No retry, threshold/precision/setting change or continuation occurred. No successful optimizer
update, generation, CE baseline inference, CE training or contrastive training occurred.

## Actual evidence

- Full host preflight:9961 parent manifest files,10 checkpoint components, exact V2.1/adapter/
  adjudication/code/config hashes and pinned dependencies PASS.28 CPU tests PASS on the host.
- Both losses exactly{summary['native_loss']}; native/delegated456-gradient inventories match.
- BOTH gradient sets are entirely finite. All RNG starts/ends, modes, dropout/checkpointing,
  source-batch and decoder-prefix preconditions PASS. Independent CPU comparison also reconstructed
  the exact frozen Tempo positive batch and decoder shift, confirming source/label identity.
- {summary['passing_tensors']}/456 tensors pass allclose;{summary['failing_tensors']}/456 contain failures.
  {summary['failing_elements']:,}/{summary['total_gradient_elements']:,} elements fail ({100*summary['failing_element_fraction']:.4f}%).
  Maximum absolute difference:{summary['max_absolute_difference']}. Per-tensor absolute/relative
  differences, counts, bit-exact flags and all raw gradients are preserved.
- Independent CPU-only recomputation from the downloaded raw .pt files reproduced every tensor's
  allclose result, failure count and max absolute difference. This used no model forward or GPU.
- Observed early-stop headroom:{100*memory['headroom_fraction']:.2f}%, conservative peak used
  {memory['conservative_peak_used_bytes']:,}bytes. This is ONLY the lambda0 stage; it does not qualify
  joint graphs or full intended eight-example FP16 accumulation, which were never reached.

## Interpretation and stop boundary

Unlike V2's insufficient bit-exact-only log, V2.1 now demonstrates actual outside-tolerance
gradient disagreement under the locked comparator. It is NOT a NaN/Inf failure: both gradient
sets are finite. Cause remains unconfirmed. The failed gate uses lambda=0: the nonzero pairwise
contrastive objective has NOT been exercised. GradScaler scaling/AMP recovery and optimizer
integrity checks have NOT been reached; no attribution to loss scale65536 is justified.
Historical D051 strict replay FAIL remains unchanged; CUDA/FP16 nondeterminism or implementation
effects are hypotheses, not established explanations. Do not silently relax tolerances or rerun
native-vs-native/another precision to seek PASS under this consumed authorization.

Joint-vs-sequential gradients, negative own-prefix GPU check, full8-example FP16 accumulation,
AMP/optimizer integrity and full-batch memory qualification remain NOT REACHED/UNVERIFIED.
No scientific-quality conclusion, candidate ranking improvement or generation gain can be drawn.
Current decision is NO-GO for numerical qualification/training, not a quality failure of the method.
Keep untouched Original CEepoch3 and primary RO/U/F/Shuffled comparisons unchanged. Stop for
independent ChatGPT review; no further action or experiment is automatically authorized.

## Preservation and delivery

Both raw gradient files are19,833,056bytes each, SHA-bound alongside full context/RNG signatures,
the456-tensor report, all numerical stages, memory samples, CPU logs, failure traceback, actual
UTC approval, exact command, host/runtime/GPU records and supervisor configuration. All are included.
Postflight verified9961 parent files,81 execution bindings,10 checkpoint components, original
backbone binary and4 historical V2 run files unchanged. Supervisor is EXITED with autorestart
and autostart disabled; no GPU compute processes remain. Instance was not destroyed.
'''
    (OUT/'REVIEW_REPORT.md').write_text(text,encoding='utf-8')
    portable=OUT/'portable_review';portable.mkdir(exist_ok=False)
    shutil.copytree(OUT/'downloaded',portable/'complete_execution_evidence')
    for name in ['FINAL_EXECUTION_REPORT.json','INDEPENDENT_CPU_RECOMPUTATION.json','REVIEW_REPORT.md','DOWNLOAD_VERIFIED.json','REMOTE_EXPORT_RECEIPT.json','V2_1_RESEARCHER_AUTHORIZATION.json','LOCAL_APPROVAL_BINDING_VERIFICATION.json','DEPLOY_AND_START.log','cf_diagnostic_v2_1.conf','HOST_RUN_DIAGNOSTIC.py','HOST_EXPORT_DIAGNOSTIC.py']:
        shutil.copyfile(OUT/name,portable/name)
    shutil.copyfile(PREP/'EXACT_BINDINGS.json',portable/'EXACT_APPROVED_BINDINGS.json')
    shutil.copyfile(Path(__file__),portable/'review_contrastive_v2_1_diagnostic_execution.py')
    shutil.copyfile(ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_OFFLINE_REVIEW.zip',portable/'historical_V2_1_offline_review_unchanged.zip')
    frozen=portable/'frozen_diagnostic_case';frozen.mkdir()
    for folder,name in [('positives','EzJI5x25U5GgbYBWhu5ZQQ'),('pairs','P02')]:
        shutil.copyfile(ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'/folder/(name+'.json'),frozen/(name+'.json'))
    (portable/'README_FIRST_VI.md').write_text(f'''# V2.1 đã thực hiện đúng1 diagnostic và dừng

Đọc REVIEW_REPORT.md. Gate lambda0 FAIL thực sự ngoài tolerance đã khóa:
{summary['failing_tensors']}/456 tensor,{summary['failing_elements']:,} phần tử vượt ngưỡng; cả2 bộ gradient đều hữu hạn.
Loss bằng nhau, các precondition RNG/batch/mode/prefix đều PASS. Nguyên nhân chưa xác nhận.
0update,0generation, không baseline/training/retry. Gate joint/FP16/AMP/full-batch VRAM chưa tới.
Gói giữ cả2 gradient thô và toàn bộ bằng chứng; CPU recomputation đã xác minh lại kết quả.
NO-GO hiện tại là chưa đủ điều kiện số học để train, không phải kết luận chất lượng contrastive.
Giữ CEepoch3 và RO/U/F/Shuffled chính; dừng cho ChatGPT review độc lập.
Authorization trong gói là bằng chứng của lượt đã dùng, KHÔNG cho phép chạy lại hoặc train.
''',encoding='utf-8')
    files={p.relative_to(portable).as_posix():sha(p) for p in portable.rglob('*') if p.is_file()}
    write(portable/'SHA256_MANIFEST.json',dict(files=files))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_GPU_DIAGNOSTIC_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(portable.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(portable).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in files.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),sha256=sha(archive),
        bytes=archive.stat().st_size,manifest_PASS=len(files),raw_gradient_sets=2,raw_gradient_tensors_per_set=456,
        independent_CPU_recomputation='PASS_MATCHES_GPU_EVIDENCE',diagnostic='FAIL_LAMBDA0_OUTSIDE_LOCKED_TOLERANCE',
        diagnostic_attempts=1,successful_optimizer_updates=0,generation_calls=0,automatic_retry=False,
        stop_for_independent_review=True)
    write(OUT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(dict(delivery=receipt,summary=summary)))

if __name__=='__main__':main()
