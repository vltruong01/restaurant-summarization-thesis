"""Validate frozen local evidence and package offline V2.1 review. No remote/GPU calls."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1'
OLD=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    c=read(OUT/'CONFIG.json');pending=read(OUT/'DIAGNOSTIC_AUTHORIZATION_PENDING.json')
    assert pending['GPU_execution_approved'] is False and pending['authorized_utc'] is None and pending['deadline_utc'] is None
    assert pending['scopes']==['diagnose'] and not pending['automatic_continuation'] and not pending['training_authorized']
    tests=(OUT/'CPU_TESTS.txt').read_text()
    assert 'Ran 28 tests' in tests and '\nOK' in tests
    for rel,digest in c['files'].items():assert sha(ROOT/rel)==digest,rel
    original_zip=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_DIAGNOSTIC_REVIEW.zip'
    assert sha(original_zip)=='69c93558ab35a926a67bedfb91825bf3f87922ac4065cbc47a153777d472690c'
    with zipfile.ZipFile(original_zip) as z:
        chain=read(OLD/'portable_review/SHA256_MANIFEST.json')['files']
        for rel,digest in chain.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
    files={p.relative_to(ROOT).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file()}
    for rel,digest in read(OLD/'CONFIG.json')['files'].items():assert sha(ROOT/rel)==digest;files[rel]=digest
    backbone=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    assert sha(backbone)==c['host_backbone_sha256']
    files[backbone.relative_to(ROOT).as_posix()]=sha(backbone)
    write(OUT/'PRESERVATION_CHECKS.json',dict(original_V2_portable_entries_PASS=len(chain),old_preparation_execution_and_frozen_files_unchanged=files,
        original_CE3_adapter_sha256=c['science']['adapter_sha256'],original_backbone_sha256=c['host_backbone_sha256'],
        original_V2_review_zip_sha256=sha(original_zip),V2_1_GPU_operations=0,remote_connections=0,optimizer_updates=0,generation_calls=0))
    bindings={k:v for k,v in pending.items() if k.endswith('_sha256')}
    write(OUT/'EXACT_BINDINGS.json',bindings)
    write(OUT/'VALIDATION_RESULTS.json',dict(CPU_unit_regression_tests_PASS=28,CPU_unit_regression_tests_FAIL=0,
        metadata_runtime_pins_checked=True,offline_preflight='PASS',pending_GPU_authorization_rejected=True,
        forbidden_pilot_mode_rejected=True,old_V2_GPU_authorization_rejected_by_test=True,
        preserved_joint_accumulation_and_AMP_block_byte_equivalent=True,
        full_pretrained_LED_CUDA_FP16_equivalence='UNVERIFIED',GPU_memory_headroom='UNVERIFIED',GPU_execution_approved=False,
        no_V2_1_run_namespace_created=not (ROOT/'runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY').exists()))
    (OUT/'REVISION_AND_REVIEW.md').write_text('''# V2.1 diagnostic-only instrumentation correction

**Current authorization: OFFLINE preparation only. No GPU rerun, baseline inference or training.**

## Preserved interpretation of V2

V2 stopped at LAMBDA0_NATIVE_CE. Both scalar losses were exactly1.1627447605133057 and all456
logged delegated gradients were finite. Bit-exact gradient equality failed. Native gradients
and pairwise deltas were not saved, so agreement within tolerance is UNVERIFIED, not proven to
fail outside tolerance. Zero optimizer updates and zero new generations; no quality conclusions.
The original63-entry ZIP, raw stopped-run evidence, code, protocol,8 accepted positives,15 pairs,
excludedP03/P17/P18, CE3 checkpoint and original backbone remain unchanged. V2.1 changes no
scientific settings; the post-adjudication amendment remains disclosed and not preregistered.

## Minimal correction

The unchanged thresholds atol1e-5,rtol1e-4 are locked in helper/config/authorization hashes before
any future execution. Each element passes iff finite and abs(delegated-native)<=atol+rtol*abs(native),
using original dtype torch.isclose semantics with native reference and equal_nan=False. Loss must
also pass; shape/dtype/inventory errors fail. Bit-exact equality is recorded separately, never required
for this lambda0 gate. Nonfinite values hard-fail even if torch.isclose considers matching infinities equal.

Immediately after the native backward, before delegated execution, write native_gradients.pt with
all named CPU gradient tensors and loss, plus native_context.json and SHA256 receipt. Do the same
for delegated. LAMBDA0_COMPARISON.json records all456 per-tensor names/shapes/dtypes, both finite
flags/nonfinite counts, max absolute and max relative differences, failing-element counts, allclose,
bit-exact result, scalar loss comparison, thresholds and preconditions. Relative error is
abs(delta)/abs(native): zero/zero=0; nonzero/zero is the JSON string "inf". Relative error alone
does not determine acceptance when the absolute tolerance covers a small near-zero difference.
Float64 CPU copies are used ONLY to report difference metrics; model/gradient computation precision
and the original-dtype acceptance comparator are unchanged. No modified loss, scoring or precision.

Before EACH path explicitly restore the same complete RNG snapshot. Record Python/NumPy/TorchCPU/
TorchCUDA start/end fingerprints separately; require equal starts AND ends. Capture full module
modes, active adapters, use_cache, dropout settings, checkpointing flags/kwargs and source-batch
tensor hashes before/after each path. Checkpoint kwargs are deep-copied so mutation cannot rewrite
earlier evidence. Require unchanged batches/modes/settings, native and delegated own prefixes, and
manual CE reduction alignment. Raw native evidence survives a delegated-path failure. No file overwrites.

## Unchanged later gates and receipt semantics

The complete real-model joint-vs-sequential, negative own-prefix, eight-example native FP16
accumulation, RNG continuity, injected AMP overflow skip/optimizer integrity and final memory
checks remain byte-equivalent to V2 after lambda0. Snapshot/score objectives and thresholds are
unchanged. The original strict accumulation comparison remains as reviewed; V2.1 does not relax it.
No diagnostic optimizer update succeeds; the injected GradScaler step is required to skip and leave
optimizer/adapter unchanged. Every mandatory gate and at least15% measured VRAM headroom must pass.

Receipt fields are lambda0_within_tolerance, lambda0_bit_exact and lambda0_thresholds; there is no
lambda0_exact=True shorthand. A lambda0 PASS is explicitly INCOMPLETE for the whole diagnostic.
Missing/failed gates or insufficient/unknown memory cannot produce a final PASS. COMPLETION is
written only after MemoryMeter exits successfully and final memory evidence is persisted. Failure
keeps raw evidence and a FAILURE_STOP_NO_RETRY receipt. Timeout, nonfinite, out-of-tolerance,
context/RNG or integrity failures stop without retry, altered settings or continuation.

## Offline validation and remaining uncertainty

28 focused CPU tests PASS: exact/within-tolerance/outside-tolerance comparators, finite/nonfinite,
near-zero relative metrics, inventory mismatch, raw snapshot roundtrip/preservation, RNG/mode/batch
precondition refusal, immutable checkpoint-context snapshots, incomplete receipt and memory failure,
fresh diagnostic-only contract, rejected V2 approval, forbidden training, and unchanged later gates.
CLI CPU preflight PASS; pending authorization rejected; pilot mode rejected. No remote connection,
pretrained model forward, GPU query, inference, training or optimizer update occurred in preparation.
Full real LED/CUDA/FP16 equivalence, AMP and accumulation behavior and headroom remain UNVERIFIED.
These tests validate instrumentation and guards; they do not rehabilitate V2's numerical result.

## Resource and authorization scope

All456 LoRA gradients contain4,915,200 FP32 values: about19.66MB per raw path, about39.32MB total
before serialization overhead; allow roughly45-60MB disk for lambda0 snapshots/context/JSON. CPU
metric calculations add temporary reporting storage. Model precision, batch sizes, checkpointing and
joint reference remain unchanged; GPU peak/headroom and total runtime still require measurement.
Keep original1-hour diagnostic wall cap within a NEW maximum48h approval window. No old timestamp
or failed namespace is reused. DIAGNOSTIC_AUTHORIZATION_PENDING.json is FALSE/null until a fresh
explicit researcher approval is supplied; all execution hashes are in EXACT_BINDINGS.json.

Only preflight/diagnose modes exist in the V2.1 runner; no train or generation entrypoint. A successful
diagnostic ends PASS_DIAGNOSTIC_ONLY_REQUIRES_INDEPENDENT_REVIEW, with0 successful updates,
0 generations and automatic_continuation=false. Stop and export evidence for independent review.
No CE baseline, CE continuation or contrastive continuation follows. Original CE3 remains current F;
primary RO/U/F/Shuffled thesis experiments are unchanged. No further technique or tuning is introduced.
''',encoding='utf-8')
    (OUT/'EXECUTION_CONTRACT_NOT_AUTHORIZED.md').write_text('''# Fresh diagnostic-only authorization request — NOT APPROVED

Requested FUTURE scope: exactly one RTX3090 full-model diagnostic using locked V2.1 hashes.
Zero successful optimizer updates; zero generation calls; no training, baseline or automatic chain.
The current researcher instruction authorizes offline preparation only. The pending JSON has
GPU_execution_approved=false, approval_source=null and authorized_utc/deadline_utc=null.
Following a separate explicit approval, record its actual UTC time and fresh deadline (maximum48h)
in a NEW approval file, retaining all exact hash/budget/namespace restrictions. Do not reuse V2 approval.
No command below was executed during preparation.

On the full hash-verified pinned parent workspace, only after fresh explicit authorization:

```bash
.venv/bin/python -u scripts/run_led_large_exploratory_contrastive_diagnostic_v2_1.py diagnose \\
  --execute --approval NEW_V2_1_DIAGNOSTIC_RESEARCHER_AUTHORIZATION.json \\
  --output runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY/diagnose_pair
```

The exact output parent must not already exist; no automatic retry/resume. Parent/model/checkpoint/
runtime verification remains mandatory. The review ZIP omits large pretrained weights and is not
a standalone execution workspace. Use the existing verified full workspace, not a fresh unpinned install.
After the command, export all raw gradients, context/RNG signatures, comparisons, numerical stages,
memory samples, full traceback if any, exact bindings and final diagnostic receipt. Stop for independent
review regardless of PASS/FAIL. No chained script or training command is supplied by V2.1.
''',encoding='utf-8')
    (OUT/'README_FIRST_VI.md').write_text('''# V2.1 — chỉ sửa instrumentation offline

Đọc REVISION_AND_REVIEW.md và EXACT_BINDINGS.json. Comparator lambda0 giữ nguyên
atol=1e-5,rtol=1e-4; lưu bit-exact riêng, không biến kết quả trong tolerance thành exact.
Lưu đầy đủ cả456 gradient native/delegated, delta/count/finite, RNG signature và model/batch context.
Các gate joint/FP16 accumulation/AMP/RNG/memory15% giữ nguyên. Receipt chỉ PASS sau khi mọi gate
và việc lưu memory evidence hoàn tất. Thiếu bằng chứng hoặc có lỗi đều dừng.

28 kiểm thử CPU PASS; không kết nối remote, không truy cập GPU, không inference/training/update.
Full LED/CUDA/FP16 numerical equivalence vẫn UNVERIFIED. Authorization mới để FALSE/null;
namespace mới. Runner chỉ có preflight/diagnose, không có baseline/generation/train.
Nếu sau này diagnostic PASS cũng phải DỪNG gửi review độc lập; không tự chạy hai nhánh.

historical_V2_review_unchanged.zip là gói63-entry trước, giữ nguyên SHA256 và chỉ để đối chiếu.
Mọi authorization trong gói lịch sử là của V2 đã dừng, không hợp lệ cho V2.1.
Giữ Original CEepoch3 và các so sánh RO/U/F/Shuffled chính nguyên trạng.
''',encoding='utf-8')
    portable=OUT/'portable_review';portable.mkdir(exist_ok=False)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.patch','.txt','.log'):shutil.copyfile(p,portable/p.name)
    (portable/'code').mkdir()
    for rel in ['scripts/run_led_large_exploratory_contrastive_diagnostic_v2_1.py',
                'scripts/led_large_contrastive_diagnostic_evidence_v2_1.py',
                'scripts/run_led_large_exploratory_contrastive_v2.py',
                'scripts/led_large_contrastive_sequential_v1.py',
                'scripts/prepare_exploratory_contrastive_diagnostic_v2_1.py',
                'scripts/package_contrastive_diagnostic_v2_1_review.py',
                'tests/test_contrastive_diagnostic_v2_1.py']:
        shutil.copyfile(ROOT/rel,portable/'code'/Path(rel).name)
    shutil.copyfile(original_zip,portable/'historical_V2_review_unchanged.zip')
    manifest={p.relative_to(portable).as_posix():sha(p) for p in portable.rglob('*') if p.is_file()}
    write(portable/'SHA256_MANIFEST.json',dict(files=manifest))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(portable.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(portable).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,digest in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
    for rel,digest in files.items():assert sha(ROOT/rel)==digest,rel
    assert not (ROOT/c['output_namespace']).parent.exists()
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),manifest_PASS=len(manifest),CPU_tests_PASS=28,bindings=bindings,
        GPU_execution_approved=False,remote_connections=0,GPU_operations=0,successful_optimizer_updates=0,generation_calls=0,
        full_pretrained_LED_CUDA_FP16='UNVERIFIED',prior_V2_chain_verified=True,stop_for_researcher_review=True)
    write(OUT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
