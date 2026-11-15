"""Package the stopped diagnostic honestly. No model imports or GPU operations."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile,difflib,re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2'
PORTABLE=OUT/'portable_review'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not PORTABLE.exists()
    c=read(OUT/'CONFIG.json');a=read(OUT/'RESEARCHER_AUTHORIZATION.json')
    stages=[json.loads(x) for x in (OUT/'downloaded/runs/CF_EXPLORATORY_V2/diagnose_pair/NUMERICAL_STAGES.jsonl').read_text().splitlines()]
    lam=next(x for x in stages if x['stage']=='LAMBDA0_NATIVE_CE')
    assert lam['loss']==lam['delegated_loss'] and lam['bit_exact'] is False
    assert len(lam['gradients'])==456 and all(g['finite'] for g in lam['gradients'])
    assert read(OUT/'downloaded/operations/STATUS.json')['status']=='STOP_NO_AUTOMATIC_RETRY'
    closeout=read(OUT/'downloaded/operations/FAILURE_CLOSEOUT.json')
    assert closeout['optimizer_updates']==closeout['generation_calls']==0 and closeout['training_namespaces_created']==[]
    record=dict(status='NO_GO_EXECUTION_NOT_NUMERICALLY_QUALIFIED',quality_conclusion='UNASSESSED_NO_TRAINING_OR_NEW_OUTPUTS',
        stop_reason='Lambda0 bit-exact gradient guard failed; equal scalar loss, all456 delegated gradients finite',
        guard_limitation='Bit equality was stricter than the researcher requirement of agreement within declared tolerance. Actual pairwise gradient differences and native reference gradient tensors were not saved. No claim of divergence or NaN is supported.',
        optimizer_updates=0,generation_calls=0,diagnostic_process_attempts=1,automatic_retry=False,
        diagnostic_tests=dict(parent_9961_manifest='PASS',checkpoint_10_components='PASS',CPU_tests_26='PASS',
            positive_decoder_prefix_and_manual_token_CE='PASS',lambda0_scalar_loss='EXACT_EQUAL',
            lambda0_gradient_bit_equality='FAIL',lambda0_gradient_tolerance='UNVERIFIED',
            joint_vs_sequential_gradient='NOT_REACHED',negative_real_model_prefix='NOT_REACHED',
            full8_FP16_accumulation='NOT_REACHED',AMP_overflow_and_optimizer_integrity='NOT_REACHED',
            dropout_RNG_equivalence='NOT_REACHED',measured_15_percent_headroom='UNVERIFIED'),
        native_loss=lam['loss'],delegated_loss=lam['delegated_loss'],delegated_gradient_tensors_finite=456,
        CE3_checkpoint_preserved=True,base_binary_preserved=True,postflight_immutable_bindings_PASS=75,
        primary_thesis_family_unchanged=True,post_adjudication_amendment_disclosed=True)
    write(OUT/'FINAL_EXECUTION_REPORT.json',record)
    report='''# Exploratory Contrastive Pilot V2 — Stopped diagnostic

**Execution decision: NO-GO under the current diagnostic. Scientific quality: NOT ASSESSED.**
One real RTX3090 diagnostic process ran and stopped before any optimizer was constructed or any
training update/generation. No CE or contrastive arm was launched. No tolerance/precision/weight,
checkpointing, objective, data, sampling or decoding setting was changed after the stop; no GPU retry.

## Protocol and preservation

V1 required16 pairs; independent researcher adjudication accepted15 and rejectedP03, hence V1 NO-GO.
V2 is the explicit POST-adjudication exploratory amendment, NOT preregistered. It keeps all8 accepted
immutable positives,15 locked pairs (5Core/6disagreement/4grounding), fixed8fit/4TRAIN probes,
Original CE3 initialization and matched fresh optimizers. P03/P17/P18 stay excluded. See the exact
protocol/config/approval, executed runner, unchanged objective and V1-to-V2 patch. Pair length and
absolute-wording shortcuts remain limitations, not silently repaired supervision.

## What the actual run established

Host preflight passed9961 parent manifest files,10 original checkpoint components and26 pinned-runtime
CPU tests. RTX3090 UUID GPU-5c9887ed-7773-40dd-30ec-13b6c755b8c9, driver580.126.09. On real full LED
weights the positive decoder input was observed directly to start EOS2 then first label tokens;
native loss equaled manual valid-token CE, including headers/EOS. Native and delegated lambda0 losses
were BOTH1.1627447605133057. All456 recorded delegated gradients were finite (maxabs0.2161865234375).

The bit-exact loss/gradient conjunction was FALSE. Given equal scalar losses, at least one gradient
tensor failed torch.equal. The implementation STOP was a stricter criterion than the researcher's
within-tolerance requirement. This guard choice is acknowledged as a diagnostic-design limitation.
The log did NOT retain native reference gradient statistics, per-tensor deltas or raw gradient pairs;
therefore it cannot determine whether deviations exceeded1e-5 absolute/1e-4 relative tolerance, nor
prove NaN, numerical divergence, a contrastive-objective bug or CUDA nondeterminism. Historical
D051 strict replay FAIL remains unchanged and is a reason to investigate, not proof of this cause.

Joint-graph/streamed equivalence, negative GPU prefix, full8-example FP16 accumulation, AMP skip,
optimizer-state integrity and15% memory-headroom qualification were NOT REACHED/NOT ESTABLISHED.
Passing reduced-width CPU tests cannot substitute for them. The pipeline correctly blocked all
baseline/pilot generation and both16-update arms. A proposed instrumentation/guard correction is
provided as a NONEXECUTED patch for review; it has not retroactively converted this attempt to PASS.
No further GPU attempt is implied by this report. Do not reopen the stopped suite automatically.

## Deliverable limits and recommendation

All persisted diagnostic stages, component stats, exact code/config hashes, CPU logs, full traceback,
supervisor/command evidence and failure closeout are included. There are no matched training logs or
new probe outputs because those stages did not run; every one of the4 planned cases is explicitly
listed NOT RUN. Four preexisting N1 baseline outputs are included separately as HISTORICAL references,
not outputs of this attempt and not evidence of a contrastive gain. No independent comparison or
researcher-confirmed quality ratings are fabricated. Objective/ranking and generation improvements
are all unassessed. The4 probes are previously examined TRAIN cases, not a generalization benchmark.

Keep Original CE epoch3 and prioritize the primary RO/U/F/Shuffled thesis comparison. Current evidence
does not justify proceeding with contrastive training or claiming that contrastive learning cannot
work. Resolve the diagnostic guard/instrumentation issue through explicit review before considering
any separately bounded reopening. No broader optimization, new negatives or replacement targets.
'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    source=(ROOT/'scripts/run_led_large_exploratory_contrastive_v2.py').read_text(encoding='utf-8')
    old="""    equal=torch.equal(native,delegated) and all(torch.equal(x,y) for x,y in zip(ga,gb))
    event('LAMBDA0_NATIVE_CE',loss=float(native),delegated_loss=float(delegated),bit_exact=equal,gradients=gradstats(gb))
    require(equal,'Lambda0 native CE loss/gradient mismatch')"""
    new="""    # PROPOSED ONLY. Before any fresh execution, lock and independently review these bindings/tolerances.
    torch.save(dict(native=[g.detach().cpu() for g in ga],delegated=[g.detach().cpu() for g in gb]),folder/'lambda0_raw_gradients.pt')
    differences=[dict(index=i,max_abs_error=float((x-y).abs().max()),allclose=bool(torch.allclose(x,y,atol=1e-5,rtol=1e-4))) for i,(x,y) in enumerate(zip(ga,gb))]
    equal=torch.equal(native,delegated) and all(torch.equal(x,y) for x,y in zip(ga,gb))
    event('LAMBDA0_NATIVE_CE',loss=float(native),delegated_loss=float(delegated),bit_exact=equal,native_gradients=gradstats(ga),delegated_gradients=gradstats(gb),per_tensor=differences)
    require(torch.allclose(native,delegated,atol=1e-5,rtol=1e-4) and all(x['allclose'] for x in differences),'Lambda0 native CE differs beyond locked tolerance')"""
    assert old in source
    proposed=source.replace(old,new)
    (OUT/'PROPOSED_GUARD_INSTRUMENTATION_FIX_NOT_EXECUTED.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),proposed.splitlines(True),fromfile='actual executed runner (preserved)',tofile='proposal only, not executed or rebound')),encoding='utf-8')
    PORTABLE.mkdir()
    for name in ['PROTOCOL_AMENDMENT.md','CONFIG.json','RESEARCHER_AUTHORIZATION.json','RESEARCHER_ADJUDICATION_AMENDMENT.json','PREFLIGHT_MANIFEST.json','V1_TO_V2_RUNNER.patch','CPU_AMENDMENT_TESTS.txt','FINAL_EXECUTION_REPORT.json','FINAL_REPORT.md','PROPOSED_GUARD_INSTRUMENTATION_FIX_NOT_EXECUTED.patch','REMOTE_ARCHIVE_RECEIPT.json','CF_EXPLORATORY_V2_DIAGNOSTIC_EVIDENCE.tar.gz','DEPLOY_AND_START.log']:
        shutil.copyfile(OUT/name,PORTABLE/name)
    shutil.copytree(OUT/'downloaded',PORTABLE/'actual_execution_evidence')
    (PORTABLE/'code').mkdir()
    for rel in ['scripts/run_led_large_exploratory_contrastive_v2.py','scripts/led_large_contrastive_sequential_v1.py','tests/test_exploratory_contrastive_v2.py']:
        shutil.copyfile(ROOT/rel,PORTABLE/'code'/Path(rel).name)
    oldprep=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
    for folder,ids in [('positives',c['fit']),('pairs',c['pair_ids'])]:
        (PORTABLE/folder).mkdir()
        for key in ids:shutil.copyfile(oldprep/folder/(key+'.json'),PORTABLE/folder/(key+'.json'))
    probes={p['business_id']:p for p in read(ROOT/c['TRAIN_source_packets'])}
    historical=ROOT/'outputs/stage_i/led_large_nbest_v2_execution_v1/downloaded/CE_V2_WORKSPACE/runs/F_CE3_nbest_training_v2/raw'
    (PORTABLE/'probe_sources').mkdir();(PORTABLE/'historical_CE3_not_new_outputs').mkdir()
    table=['# All four planned TRAIN probes','',
        'No new outputs: all conditional generation was blocked by the diagnostic stop. Historical N1 files are labeled separately.',
        '', '| Business | Untouched CE3 new evaluation | CE continuation | Contrastive continuation | Quality comparison |',
        '|---|---|---|---|---|']
    case_reports=[]
    for bid in c['probe']:
        case=probes[bid];assert case['split']=='train'
        name=re.search(r'^name=(.*)$',case['source'],re.M).group(1)
        write(PORTABLE/'probe_sources'/(bid+'.json'),case)
        raw=historical/(bid+'_single.json');data=read(raw)
        assert data[0]['source_sha256']==case['source_sha256'] and data[0]['checkpoint_sha256']==c['adapter_sha256']
        shutil.copyfile(raw,PORTABLE/'historical_CE3_not_new_outputs'/raw.name)
        table.append(f'| {name} | NOT RUN | NOT RUN | NOT RUN | NOT ASSESSABLE |')
        case_reports.append(dict(business_id=bid,name=name,Filtered_captions=len(case['captions']),new_baseline='NOT_RUN',CE_arm='NOT_RUN',contrastive_arm='NOT_RUN',
            Core_coverage='UNASSESSED',mixed_opinion_preservation='UNASSESSED',factual_visual_grounding='UNASSESSED',repetition='UNASSESSED',zero_caption_regression='UNASSESSED',
            evidence=case['source_sha256'],historical_baseline_sha256=sha(raw),independent_judgment=None))
    table+=['','There are no examples of pilot gains/regressions or pair-ranking improvement to quote. All are unassessed, not zero-valued quality scores.',
        'All four are previously examined TRAIN businesses. No validation/test access occurred.']
    (PORTABLE/'ALL_FOUR_CASE_COMPARISONS.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write(PORTABLE/'CASE_LEVEL_STATUS.json',case_reports)
    (PORTABLE/'README_FIRST_VI.md').write_text('''# Đọc FINAL_REPORT.md trước

Đây là gói của một diagnostic đã dừng, KHÔNG phải pilot đã train xong.
Protocol V2 sửa đổi SAU adjudication:15 cặp, loạiP03/P17/P18; không preregistered.
Host9961hash,10component checkpoint và26CPUtest PASS. Loss lambda0 bằng nhau,
456 gradient delegated hữu hạn, nhưng phép so bit-exact gradient không đạt.
Phép kiểm tra này chặt hơn yêu cầu trong tolerance; thiếu delta/raw gradient để xác định mức lệch.
Không tự hạ ngưỡng hay rerun. Đã dừng:0update,0generation, hai nhánh không bắt đầu.
Các gate joint/AMP/accumulation/headroom chưa được xác nhận. Không có kết luận chất lượng CF.
Xem actual_execution_evidence/ và bảng ALL_FOUR_CASE_COMPARISONS.md.
historical_CE3_not_new_outputs/ là4 N1 đã có từ N-best V2, không phải output mới hay bằng chứng cải thiện.
Patch guard đề xuất CHƯA được áp dụng/thực thi; không làm receipt FAIL thành PASS.
Giữ CEepoch3 và ưu tiên RO/U/F/Shuffled chính. Không có phê duyệt tự động tiếp tục.
''',encoding='utf-8')
    files={p.relative_to(PORTABLE).as_posix():sha(p) for p in PORTABLE.rglob('*') if p.is_file()}
    write(PORTABLE/'SHA256_MANIFEST.json',dict(files=files))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_DIAGNOSTIC_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(PORTABLE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PORTABLE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,h in files.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,manifest_PASS=len(files),
                 optimizer_updates=0,generation_calls=0,training_executed=False,checkpoint_unchanged=True,decision='STOP_NOT_NUMERICALLY_QUALIFIED_NOT_A_QUALITY_VERDICT')
    write(OUT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
