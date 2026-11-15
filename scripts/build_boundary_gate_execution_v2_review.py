"""Package stopped gate evidence; CPU decoding only of already-saved technical IDs."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

R=Path(__file__).resolve().parents[1]
E=R/'outputs/stage_i/led_large_boundary_global_feasibility_v1_execution_v2'
D=E/'downloaded'
G=D/'gate_run'
REVIEW=E/'portable_review'
ZIP=R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GATE_EXECUTION_V2_REVIEW.zip'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,d):
    p=REVIEW/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')


def main():
    assert not REVIEW.exists() and not ZIP.exists()
    REVIEW.mkdir()
    manifest=read(D/'SHA256_MANIFEST.json')['files']
    assert len(manifest)==144
    for rel,digest in manifest.items():assert sha(D/rel)==digest,rel
    status=read(D/'operations/EXECUTION_STATUS.json')
    native=read(G/'bos_only/feasibility/REPORT.json')
    cache=read(G/'bos_only/RESOURCE_GATE.json')
    failure=read(G/'STOP_FAILURE.json')
    assert status['status']=='FAIL_STOP_NO_RETRY' and native['status']==cache['status']=='PASS'
    assert not (G/'boundary').exists() and not (G/'FEASIBILITY_COMPLETE_STOP_FOR_RESEARCHER_REVIEW.json').exists()
    assert native['optimizer_updates']==3 and native['generation_calls']==2
    for step in range(1,4):
        grad=read(G/f'bos_only/feasibility/UNSCALED_GRADIENTS_step{step}.json')
        assert grad['trainable_tensors']==len(grad['tensors'])==456 and not grad['missing'] and not grad['nonfinite']
        assert all(x['status']=='finite' and x['nan']==x['positive_inf']==x['negative_inf']==0 for x in grad['tensors'].values())
    assert native['headroom_fraction']>=.15 and cache['headroom_fraction']>=.15
    native_init=read(G/'bos_only/feasibility/INITIAL_MODEL_BINDINGS.json')['initialization']
    assert native_init==cache['initialization']
    post=read(D/'operations/POST_EXECUTION_INTEGRITY.json')
    assert post['parent_PASS'] and post['overlay_PASS'] and read(D/'operations/ORIGINAL_CE3_AFTER.json')['unchanged']
    # Decode existing IDs without loading any neural model or making a generate call.
    from tokenizers import Tokenizer
    tokenfile=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/tokenizers/led-base-16384/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenfile))
    technical=[]
    for i in (1,2):
        p=G/f'bos_only/feasibility/train_memory_trace_{i}.json';row=read(p)
        technical.append(dict(row,decoded_text=tokenizer.decode(row['token_ids'],skip_special_tokens=True),
            raw_trace_sha256=sha(p),tokenizer_sha256=sha(tokenfile),
            warning='Fresh3-update discarded gate adapter. Technical resource trace ONLY; not trained-model quality evidence.',new_generation=False))
    write('TECHNICAL_OUTPUTS_CPU_DECODED.json',technical)
    timings=[json.loads(x) for x in (G/'bos_only_TIMINGS.jsonl').read_text().splitlines()]
    ts=[x['elapsed_seconds'] for x in timings if x['kind']=='successful_update']
    intervals=[b-a for a,b in zip(ts,ts[1:])]
    result=dict(suite_status='FAIL_INCOMPLETE_NO_RETRY',failure=failure['error'],
        CPU_tests=82,package_hashes=79,parent_files=10025,
        completed_arms=['bos_only'],Boundary_executed=False,
        successful_discarded_optimizer_updates=3,TRAIN_technical_generate_calls=2,full_cache_tests=1,
        full_training_updates=0,quality_evaluation_calls=0,held_out_access=False,
        bos_native={k:native[k] for k in ['optimizer_updates','microbatches','overflow_skips','scale_final','state_restore_exact',
            'adapter_restore_exact','frozen_backbone_unchanged','seconds','conservative_peak_used_bytes','headroom_fraction']},
        bos_full_cache={k:cache[k] for k in ['source_shape','global_count','full_cache_batch','cache_tokens','peak_allocated_bytes',
            'peak_reserved_bytes','conservative_peak_used_bytes','headroom_fraction']},
        all456gradients_finite_each_of3updates=True,native_and_cache_fresh_initialization_equal=True,
        cross_arm_initialization_comparison='UNVERIFIED_NO_BOUNDARY_ARM',
        GPU_guard_actual_rejected_PID_list='NOT_RECORDED_BY_FROZEN_IMPLEMENTATION',
        PID_namespace_mismatch='PLAUSIBLE_NOT_ESTABLISHED; no new CUDA diagnostic performed',
        bos_later_update_intervals_seconds=intervals,
        bos_extrapolated_1250_update_hours=sum(intervals)/len(intervals)*1250/3600,
        cost_caveat='Only2stress-gate intervals, with possible AMP replay, no epoch I/O/evaluation. Not a training-throughput benchmark. Boundary and paired-training cost unmeasured.',
        complete_driver_wall_seconds=status['wall_seconds'],source_and_original_CE3_unchanged=True,
        instance_action='KEEP_UNCHANGED',next_action='STOP for independent researcher review; no retry or full training')
    write('RESULTS_SUMMARY.json',result)
    # Top-level readable logs/JSON; all binaries and full locked overlay remain in the embedded complete archive.
    for p in sorted(D.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(D)
        if rel.parts[0] not in ('operations','gate_run'):continue
        if p.suffix not in ('.json','.jsonl','.log','.py','.patch','.conf','.md'):continue
        if p.name=='PARENT_TRANSFER_MANIFEST.json':continue
        target=REVIEW/'READABLE_EVIDENCE'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for name in ('DOWNLOAD_VERIFIED.json','REMOTE_EXPORT_RECEIPT.json','READ_ONLY_CLOSEOUT.json','OPERATIONAL_DIFF.patch','AUTHORIZATION_PROVENANCE.json','LOCAL_PACKAGING_CHECK_FAILURE.json'):
        shutil.copyfile(E/name,REVIEW/name)
    write('FULL_EVIDENCE_ARCHIVE_BINDING.json',dict(file='FULL_EXECUTION_EVIDENCE.tar.gz',sha256=sha(E/'BOUNDARY_GLOBAL_FEASIBILITY_V1_COMPLETE_EVIDENCE.tar.gz'),
        bytes=(E/'BOUNDARY_GLOBAL_FEASIBILITY_V1_COMPLETE_EVIDENCE.tar.gz').stat().st_size,
        inner_manifest_entries=144,all_inner_entries_reverified=True,checkpoints_loaded=False))
    # Avoid another disk copy: add immutable archive directly to the ZIP later.
    report='''# Boundary gate V2 — STOP after BOS-only

**Overall: FAIL / INCOMPLETE. Boundary feasibility and quality remain UNVERIFIED.**

The new researcher instruction accepted the gate-first proposal only. Full training was not authorized or run. A fresh namespace/UTC receipt was used; prior failed evidence is unchanged. Scientific package79files and full parent10025files verified before and after the attempt; all82pinned-host CPU tests and full parent/model/runtime/mask preflight passed. The demonstrated old CPU import-path failure was corrected only by setting operational `PYTHONPATH` to the unchanged parent's scripts, with fresh paths/timestamp. Scientific files/hashes are unchanged.

| Check | Actual result |
|---|---|
| BOS native gate |3successful discarded updates;48actual/24nominal microbatches;2TRAIN beam4 technical traces |
| Native unscaled gradients |456/456finite at each of3updates; no missing/nonfinite tensors |
| AMP |3bounded overflow skips; final scale8192; successful optimizer-step accounting PASS |
| State roundtrip |Adapter/optimizer/scheduler/scaler/RNG restore receipts PASS; backbone unchanged |
| BOS native memory |Conservative peak5,078,450,176bytes; headroom79.93% |
| BOS full decoder-cache stress |Source1x5120,4beams,768cache,1global; conservative peak6,619,856,896bytes; headroom73.83% |
| Fresh native/cache initialization |Identical BOS initialization hashes; comparison with Boundary unavailable |
| Boundary arm |NOT STARTED;0updates,0generate calls,0cache test |
| Full training / quality / held-out |0 /0 /no access |

After BOS native+cache completed, the frozen inter-arm GPU ownership check raised `Foreign GPU compute workload; do not run this gate` at `run_pair -> gpu_identity`. It compares every PID reported by nvidia-smi directly with Python `os.getpid()`. The original failing PID list was NOT saved. A host/container PID-namespace mismatch is a plausible explanation, but a real transient foreign workload cannot be ruled out from preserved evidence. The post-exit snapshot has no listed compute PID and1MiB idle memory; it cannot reconstruct the rejected list. Do not describe this as a proven Boundary numerical/memory failure, a confirmed hardware fault, or a demonstrated guard bug.

The mandatory fail-closed rule stopped the suite before Boundary. No guard suppression, changed mask/precision/AMP/threshold, scientific fix, retry, further inference or CUDA diagnostic was attempted. The driver and worker exited; autostart/autorestart remain disabled. Instance57481997 was retained as instructed. Original CE epoch3 adapter remains unchanged.

Driver elapsed229.97s(about3.83min), excluding deployment/transfer/export. BOS native gate182.65s includes setup/restoration/technical generation, not pure update compute. Later update intervals22.659s and11.485s produce a very uncertain arithmetic1250-update extrapolation of~5.93h; only two stress-case intervals, possible AMP replay and no epoch validation/checkpoint cost. Boundary runtime/headroom and any paired training estimate remain UNMEASURED. Do not authorize full training from this partial result or interpret the two fresh3-update technical outputs as summary-quality evidence.

Operational disclosures: before deployment, a local archive/workspace path check incorrectly expected the archive-only README at workspace root; the verification was corrected and preserved in its own receipt, with0GPUcalls. A read-only live report command had a shell-quoting error; it performed no model operation and did not affect the running worker. Neither is a hidden gate retry.

The downloaded complete archive is152,500,863bytes,144/144inner SHA-256 PASS. It contains all original gate checkpoints/state, raw technical IDs, mask/timing/memory/AMP records, locked overlay, logs and approvals. `TECHNICAL_OUTPUTS_CPU_DECODED.json` is only offline decoding of the2existing traces; no new generation. No pickle/tensor checkpoint was loaded during review packaging.

**Next decision: independent review of the stopped suite.** If continuation is considered, the GPU PID-ownership observation/namespace question must be resolved and any operational correction explicitly reviewed with fresh authorization. This report does not implement that change, grant a retry, reopen quality techniques or start training. D051strict replay FAIL and all paused thesis-stage artifacts remain unchanged.
'''
    (REVIEW/'EXECUTION_REPORT.md').write_text(report,encoding='utf-8')
    (REVIEW/'README_VI.md').write_text('''Gate V2 đã dừng đúng fail-closed sau BOS-only. BOS native/cache PASS nhưng guard sở hữuGPU chặn trước Boundary; không có full training hay kết quả chất lượng. Đọc EXECUTION_REPORT.md và RESULTS_SUMMARY.json. Archive evidence đầy đủ được nhúng trong ZIP,144hashđã xác minh. Không tự retry; giữ nguyên instance và CE3.\n''',encoding='utf-8')
    (REVIEW/'BUILD_REVIEW_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    entries={p.relative_to(REVIEW).as_posix():sha(p) for p in sorted(REVIEW.rglob('*')) if p.is_file()}
    archive=E/'BOUNDARY_GLOBAL_FEASIBILITY_V1_COMPLETE_EVIDENCE.tar.gz'
    entries['FULL_EXECUTION_EVIDENCE.tar.gz']=sha(archive)
    write('SHA256_MANIFEST.json',dict(path_base='archive_root',files=entries))
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REVIEW.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(REVIEW).as_posix())
        z.write(archive,'FULL_EXECUTION_EVIDENCE.tar.gz',compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name,digest in entries.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP)+'  '+ZIP.name+'\n',encoding='ascii')
    print(json.dumps(dict(archive=str(ZIP),bytes=ZIP.stat().st_size,manifest_entries=len(entries),sha256=sha(ZIP),summary=result),indent=2))


if __name__=='__main__':main()
