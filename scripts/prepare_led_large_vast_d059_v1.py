"""One versioned, self-contained D059-referenced fresh LED-large F handoff."""
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='LED_LARGE_VAST_D059_F_V1'
DEST=ROOT/'outputs/cloud_handoff'/NAME
PREP='outputs/stage_i/led_large_vast_d059_v1_preparation'
CONFIG='configs/stage_i_led_large_vast_d059_v1.json'
HELPER='scripts/stage_i_led_large_vast_d059_v1.py'
RUNNER='scripts/run_stage_i_led_large_vast_d059_v1.py'
REPORTER='scripts/report_stage_i_led_large_vast_d059_v1.py'
CLOCK=dict(policy='researcher_managed_for_this_new_run',researcher_message='không cần để ý tôi sẽ lo',
           historical_deadlines_unchanged=True,execution_deadline_utc=None)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(rel,value):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    p.write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')

def sources():
    old=read(ROOT/'configs/stage_i_led_large_vast_dynamic_v2.json')
    d059=read(ROOT/'configs/stage_i_d059_expanded_lora_v2.json')
    for key in ('seed','lora','optimization','generation','budgets','runtime_pins','training','evaluation_epochs',
                'stage_h_manifest','final_references'):
        assert old[key]==d059[key], 'D059 methodological mismatch: '+key
    c=json.loads(json.dumps(old))
    c['vast_fresh'].update(experiment=NAME,output='runs/F_large_d059_v1',deadline_utc=None)
    c.update(experiment=NAME,run_root='runs/F_large_d059_v1')
    c['execution_window']=CLOCK
    c['methodology_reference']=dict(config='configs/stage_i_d059_expanded_lora_v2.json',
        config_sha256=sha(ROOT/'configs/stage_i_d059_expanded_lora_v2.json'),
        primary_quality_benchmark='D059_Expanded_epoch3',secondary_quality_benchmark='D059_Expanded_epoch5',
        comparison_type='practical_unmatched_base_vs_large_not_isolated_causal_architecture_effect')
    save(CONFIG,c)
    oldhelper=(ROOT/'scripts/stage_i_led_large_vast_dynamic_v2.py').read_text(encoding='utf-8')
    helper=oldhelper.replace('stage_i_led_large_vast_dynamic_v2','stage_i_led_large_vast_d059_v1')
    helper=helper.replace('led_large_vast_dynamic_v2_preparation','led_large_vast_d059_v1_preparation')
    helper=helper.replace('LED_LARGE_VAST_DYNAMIC_65536_V2',NAME).replace('runs/F_vast3090_dynamic65536_v2','runs/F_large_d059_v1')
    helper=helper.replace("EXPERIMENT =", 'DEADLINE = None\nEXECUTION_WINDOW = '+repr(CLOCK)+'\nEXPERIMENT =',1)
    helper=helper.replace('dict(old, vast_fresh=extra, amp_scaler=AMP, dynamic_overflow_policy=POLICY)',
        "dict(old, vast_fresh=extra, amp_scaler=AMP, dynamic_overflow_policy=POLICY, execution_window=EXECUTION_WINDOW, methodology_reference=c['methodology_reference'], experiment=EXPERIMENT, run_root='runs/F_large_d059_v1')")
    anchor="    for rel,h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():"
    assert helper.count(anchor)==1
    helper=helper.replace(anchor,"    validate_d059_reference(c)\n"+anchor)
    anchor="    require(a.get('dynamic_overflow_policy') == POLICY, 'Wrong dynamic overflow policy')"
    assert helper.count(anchor)==1
    helper=helper.replace(anchor,"    require(a.get('execution_window') == EXECUTION_WINDOW, 'Researcher clock ownership differs')\n"+anchor)
    helper+='''

def validate_d059_reference(c):
    reference=ROOT/'configs/stage_i_d059_expanded_lora_v2.json'
    expected=dict(config='configs/stage_i_d059_expanded_lora_v2.json',config_sha256=sha(reference),
        primary_quality_benchmark='D059_Expanded_epoch3',secondary_quality_benchmark='D059_Expanded_epoch5',
        comparison_type='practical_unmatched_base_vs_large_not_isolated_causal_architecture_effect')
    require(c['methodology_reference']==expected,'D059 reference changed')
    d=read(reference)
    for key in ('seed','lora','optimization','generation','budgets','runtime_pins','training','evaluation_epochs',
                'stage_h_manifest','final_references'):
        require(c[key]==d[key],'D059 methodological setting changed: '+key)
    require(c['model_id']=='allenai/led-large-16384' and c['expected_modules']==228
            and c['expected_trainable_parameters']==4915200,'LED-large adapter inventory differs')
    return True

def check_deadline(a, now=None):
    # Only this new namespace follows the researcher's direct clock-management decision.
    # Every historical config/authorization/deadline file remains unchanged.
    require(a.get('experiment')==EXPERIMENT and a.get('execution_window')==EXECUTION_WINDOW,
            'Wrong new-run scope or clock provenance')
    require(a.get('execution_deadline_utc') is None,'Unexpected fixed deadline in new researcher-managed run')
    require(a.get('max_training_updates')==1250 and a.get('evaluation_outputs_max')==48,
            'Time ownership must not remove update/generation ceilings')
    return None
'''
    compile(helper,HELPER,'exec');save(HELPER,helper)
    oldrunner=(ROOT/'scripts/run_stage_i_led_large_vast_dynamic_v2.py').read_text(encoding='utf-8')
    runner=oldrunner.replace('stage_i_led_large_vast_dynamic_v2','stage_i_led_large_vast_d059_v1')
    runner=runner.replace('runs/F_vast3090_dynamic65536_v2','runs/F_large_d059_v1')
    start=runner.index('    def __exit__(self, *exc):')
    end=runner.index('\n\n\ndef data',start)
    runner=runner[:start]+'''    def __exit__(self, *exc):
        self.stop.set();self.thread.join(timeout=5)
        if self.failure is not None:
            self.exit_failure=self.failure
            if exc[0] is None:
                raise ValueError('GPU memory monitor failed: '+self.failure)
        try:
            self.sample()
        except Exception as error:
            self.exit_failure=type(error).__name__+': '+str(error)
            if exc[0] is None:
                raise  # A memory-query failure still fails an otherwise successful gate/run.
        return False  # Keep the original model exception when cleanup also fails.
'''+runner[end:]
    anchor="                        with torch.autocast('cuda',dtype=torch.float16):\n                            loss=model(**batch,use_cache=False).loss"
    assert runner.count(anchor)==1
    runner=runner.replace(anchor,"""                        append(p/'MICROBATCH_START.jsonl',dict(step=step,microbatch=j+1,
                            business_id=ids[((step-1)*8+j)%len(features)],retry=recovery.replaying,
                            actual_microbatches=recovery.microbatches,
                            input_shapes={k:list(v.shape) for k,v in batch.items()}))
"""+anchor)
    anchor="            if 'memory' in locals():\n                diagnostics['memory_summary']=memory.summary()"
    assert runner.count(anchor)==1
    runner=runner.replace(anchor,"            diagnostics['post_exception_GPU_values_are_unvalidated']=True\n            if 'memory' in locals():\n                diagnostics['memory_cleanup_error']=getattr(memory,'exit_failure',None)\n                diagnostics['memory_summary']=memory.summary()")
    compile(runner,RUNNER,'exec');save(RUNNER,runner)
    report=(ROOT/'scripts/report_stage_i_led_large_vast_dynamic_v2.py').read_text(encoding='utf-8')
    report=report.replace('stage_i_led_large_vast_dynamic_v2','stage_i_led_large_vast_d059_v1')
    anchor="('D058_B_epoch5','outputs/stage_i/d058_full_filtered_B_v1/evaluate/epoch5/predictions.json')"
    assert report.count(anchor)==1
    report=report.replace(anchor,"('D059_Expanded_epoch5','outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch5/predictions.json'),\n                     "+anchor)
    report=report.replace("'Lower repetition, lower validation loss or correct headers alone cannot establish usable summary quality.',",
        "'D059 epochs3/5 are practical base-model benchmarks. This is not an isolated causal backbone/architecture experiment.',\n              'Lower repetition, lower validation loss or correct headers alone cannot establish usable summary quality.',")
    save(REPORTER,report)
    save(PREP+'/RUNNER_DIFF_FROM_PRESERVED_VAST_V2.patch',''.join(difflib.unified_diff(
        oldrunner.splitlines(True),runner.splitlines(True),fromfile='historical_vast_dynamic_v2.py',tofile=RUNNER)))
    save(PREP+'/METHOD_COMPARISON.json',dict(unchanged_D059_keys=['seed','lora','optimization','generation',
        'budgets','runtime_pins','training','evaluation_epochs','stage_h_manifest','final_references'],
        base=dict(backbone=d059['model_id'],modules=114,trainable_parameters=1843200),
        large=dict(backbone=c['model_id'],revision=c['revision'],modules=228,trainable_parameters=4915200),
        inherited_large_adaptations=['verified large inventory/initialization','2 train-input beam4 memory traces',
            'init65536 bounded same-window native AMP backoff, max8 per mode, minimum256'],
        transport_changes=['Vast-only new namespace','researcher-managed time for this run only',
            'preserve primary exception on memory cleanup failure','log input case/shapes before gate forward'],
        no_checkpoint_or_state_import=True,RO_U_automatic=False))
    print('Versioned D059-referenced sources prepared (CPU only).',flush=True)

def build():
    sources()
    assert not DEST.exists(),'Fresh package directory required; do not overwrite prior package'
    DEST.mkdir()
    oldroot=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_DYNAMIC_65536_V2'
    oldmanifest=read(oldroot/'VAST_SHA256_MANIFEST.json')['files']
    skip={'START_HERE_VAST.md','CODEX_VAST_HANDOFF.md','VAST_EXECUTION_BINDINGS.json',
          'VAST_RESEARCHER_AUTHORIZATION.json','SETUP_VAST.sh','RUN_FRESH_F.sh',
          'VAST_READINESS_REPORT.md','COLAB_VS_VAST_COMPARISON.md','VAST_RUNNER_DIFF.patch',
          'NUMERICAL_POLICY_AMENDMENT.md'}
    for rel,h in oldmanifest.items():
        if rel in skip:continue
        src=oldroot/rel;assert sha(src)==h,'Historical package changed: '+rel
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        try:dst.hardlink_to(src)
        except OSError:shutil.copyfile(src,dst)
    extra=[CONFIG,HELPER,RUNNER,REPORTER,'scripts/check_led_large_vast_host_v1.py',
           'scripts/prepare_led_large_vast_d059_v1.py','tests/test_led_large_vast_d059_v1.py',
           'configs/stage_i_d059_expanded_lora_v2.json',
           'outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch3/predictions.json',
           'outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch5/predictions.json']
    for rel in extra:
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(ROOT/rel),'Do not mutate inherited hardlink: '+rel
        else:shutil.copyfile(ROOT/rel,dst)
    bound={rel:sha(DEST/rel) for rel in extra}
    bound['outputs/stage_i/led_large_full_train_v5_preparation/PREFLIGHT_MANIFEST.json']=sha(DEST/'outputs/stage_i/led_large_full_train_v5_preparation/PREFLIGHT_MANIFEST.json')
    save(PREP+'/PREFLIGHT_MANIFEST.json',dict(files=bound,historical_artifacts_unchanged=True,
        mode='fresh_Filtered_only_D059_referenced_LED_large_Vast'))
    (DEST/PREP).mkdir(parents=True,exist_ok=True)
    for name in ('PREFLIGHT_MANIFEST.json','METHOD_COMPARISON.json','RUNNER_DIFF_FROM_PRESERVED_VAST_V2.patch'):
        shutil.copyfile(ROOT/PREP/name,DEST/PREP/name)
    b=dict(config_sha256=sha(DEST/CONFIG),runner_sha256=sha(DEST/RUNNER),preflight_manifest_sha256=sha(DEST/PREP/'PREFLIGHT_MANIFEST.json'))
    save(PREP+'/EXECUTION_BINDINGS.json',b)
    shutil.copyfile(ROOT/PREP/'EXECUTION_BINDINGS.json',DEST/PREP/'EXECUTION_BINDINGS.json')
    a=read(oldroot/'VAST_RESEARCHER_AUTHORIZATION.json')
    now=datetime.now(timezone.utc).isoformat()
    a.update(experiment=NAME,**b,execution_authorized_at_utc=now,approval_recorded_at_utc=now,
        original_authorization_utc=now,execution_deadline_utc=None,execution_window=CLOCK,
        timestamp_note='Local record creation time; exact message timestamp unavailable. This is a new run, not a rewrite of historical authorization clocks.',
        researcher_explicit_authorization=[
            'bản colab đã gặp vấn đề và die, hãy chuyển sang toàn bộ là vastai, sử dụng D-059 làm chuẩn để cùng nhau train lại với model large',
            'RTX3090','không cần để ý tôi sẽ lo'],
        historical_authorizations_preserved=True,fresh_start=True,resume_after_interruption_approved=False,
        historical_large_deadline_utc='2026-10-21T17:11:48+00:00',
        historical_D059_deadline_utc='2026-10-25T15:10:48.941817+00:00')
    for stale in ('amendment_recorded_at_utc',):a.pop(stale,None)
    (DEST/'VAST_RESEARCHER_AUTHORIZATION.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DEST/'VAST_EXECUTION_BINDINGS.json').write_text(json.dumps(b,indent=2)+'\n',encoding='utf-8')
    setup=(oldroot/'SETUP_VAST.sh').read_text(encoding='utf-8')
    setup=setup.replace('uv venv --python 3.11 .venv',"if test ! -d .venv; then uv venv --python 3.11 .venv; fi")
    setup=setup.replace('run_stage_i_led_large_vast_dynamic_v2.py','run_stage_i_led_large_vast_d059_v1.py')
    setup=setup.replace('.venv/bin/python -m unittest discover',"export CUDA_VISIBLE_DEVICES=''\n.venv/bin/python -m unittest discover",1)
    setup=setup.replace('.venv/bin/python scripts/verify_led_large_vast_fresh_v1.py',
        '.venv/bin/python -m unittest discover -s tests -p test_led_large_vast_d059_v1.py -v\n.venv/bin/python scripts/verify_led_large_vast_fresh_v1.py')
    (DEST/'SETUP_VAST.sh').write_text(setup,encoding='utf-8',newline='\n')
    run=(oldroot/'RUN_FRESH_F.sh').read_text(encoding='utf-8')
    run=run.replace('run_stage_i_led_large_vast_dynamic_v2.py','run_stage_i_led_large_vast_d059_v1.py')
    run=run.replace('report_stage_i_led_large_vast_dynamic_v2.py','report_stage_i_led_large_vast_d059_v1.py')
    run=run.replace('F_vast3090_dynamic65536_v2','F_large_d059_v1').replace('vast_dynamic65536_v2.log','large_d059_v1.log')
    run=run.replace('for MODE in feasibility train evaluate; do',
        '$PY scripts/check_led_large_vast_host_v1.py --report launcher_logs/HOST_PREFLIGHT.json\nfor MODE in feasibility train evaluate; do')
    (DEST/'RUN_FRESH_F.sh').write_text(run,encoding='utf-8',newline='\n')
    docs(b)
    # Catch entrypoint collisions before releasing a package or running the deep preflight.
    for rel in ('README_FIRST.md','SHA256_MANIFEST.json','TRANSFER_CONTRACT.json','TRANSFER_DATASETS.json'):
        assert sha(DEST/rel)==oldmanifest[rel], 'Inherited transfer metadata changed: '+rel
    manifest={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'VAST_SHA256_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(package=str(DEST),files=len(manifest),binding=b),indent=2),flush=True)

def docs(binding):
    start=f'''# LED-large F on Vast RTX3090 — D059 reference

Một gói duy nhất. Không notebook, Google Drive, ZIP Colab hay patch rời.
Payload chứa model LED-large chính thức đã pin, tokenizer, F2000train/200validation,
selected reviews/captions, original Teacher targets, ảnh nguồn, D053 references,
fixed24cases và saved D059epoch3/5 benchmarks. Không dùng held-out contents.

## Chạy trên instance RTX3090 đã có

Giải nén vào SSD local, mở Codex tại {NAME}, đọc CODEX_VAST_HANDOFF.md.
Tối thiểu một RTX3090/24GB; chỉ1GPU visible. Dùng môi trường Linux và .venv riêng.
Cần uv trên PATH; nếu chưa có, Codex có thể cài uv từ PyPI trong user environment.
Không tự thuê máy mới, thay driver hoặc reset host. GPU cũ đang lỗi phải được
khôi phục hoặc thay bằng instance người dùng cung cấp trước khi model execution.

```bash
bash SETUP_VAST.sh
tmux new -s led_large_d059
bash RUN_FRESH_F.sh
```

Detach tmux bằng Ctrl-b rồi d; không chạy launcher lần thứ hai. Nếu SSH rớt,
attach lại session hiện có, không khởi động một run trùng. Không supervisor/autorestart.
SETUP chỉ CPU. RUN kiểm tra NVML/GPU idle, gate3successfulupdates rồi mới Ftraining.
Approval JSON có sẵn, nguồn là chỉ thị trực tiếp của researcher trong chat.

## Protocol giữ theo D059

Fresh official allenai/led-large-16384 revisioncd59d11c3528415b7dda4dfc95cc8f138aceda2e.
Không import model/adapter/optimizer/scaler/RNG từ Colab, gate cũ, D059base hay Amazon.
LoRA all-linear r8/alpha16/dropout0; loại LMhead/embeddings; inventory đã kiểm tra:
228modules/456trainabletensors/4,915,200parameters. Backbone459,801,600 frozen.
Seed42/named initialization; positiveCE; F2000/200; budgets5000source/768labels,
4000reviews/750captions/K5. AdamW2e-4,weight_decay.01,betas(.9,.999),eps1e-8;
batch1/accum8,clip1,FP32storage/FP16autocast,nonreentrantcheckpointing;
5epochs/1250successfulupdates,warmup63,linearLR;all5checkpoints,noearlystop/bestlossreload.
Gate3successfulupdates/24nominal/<=88actualmicros,15%headroom,allgradient/frozen
backbone/checkpoint-restoration checks and2train-input beam4memorytraces.
Discard gate state before fresh training. AMP starts65536, bounded nativebackoff
max8/mode,minscale256;same8microbatchwindow/RNG on overflow;no scheduler/globalstep
advance on skips. Missing gradient/nonfinite forward/OOM/integrity failures STOP.
Training maximum10064actualmicrobatches. No numerical/search/decoding variants.
This inherits the previously approved large numerical policy; it is not literally
the old D059base runner or a causal scale experiment.

## Evaluation / practical benchmark

Generate exactly48outputs:epoch3/5 × fixed24(random20,targeted4),originalbeam4,
max_length768,no_repeat_ngram_size0. Compare both large checkpoints to saved
D059Expandedepoch3(primary)/epoch5(secondary),D058Bepoch5,historicalF.
Report random20 andtargeted4 separately. Base vslarge differs in capacity and
history; do not claim a matched causal architecture experiment.
GO requires researcher-confirmed substantive gain in Core/business detail,
source grounding, balanced opinions, useful caption-supported visual information,
semantic repetition, zero-caption behavior and completeness without severe tradeoffs.
Loss, length, headers or exact-repeat reduction alone are insufficient. Software
metrics and provisional AI inspection are separate from human labels.
Only after researcherGO may matched RO/U training be separately authorized;
Shuffled must use that matched F checkpoint. Historical LED-base RO/U/F remain
primary until a complete matched large family exists.

## Timing, evidence and stop

Researcher says “không cần để ý tôi sẽ lo”: timing for THIS new run is researcher-
managed; no new automatic wall-clock deadline. All historical clocks/decisions,
D051strictreplayFAIL, Colab and failedVast evidence remain untouched. Update and
generation ceilings remain enforced. Stop on unexpected failure;no automatic retry,
fallback,resume,RO/U or newexperiment. Memory cleanup preserves the primary
exception and records secondary failures; no attention/mask repair or tolerance change.

Outputs: runs/F_large_d059_v1/. Preserve all checkpoints, receipts andlauncher_logs.
Quality bundle: runs/F_large_d059_v1/LED_LARGE_F_QUALITY_REVIEW.zip.
If storage is not persistent, copy results/evidence outside the instance before
recycle/destroy; this package does not authorize instance deletion or rental.
Pinned package files may not be modified to bypass checks.

Bindings:
```json
{json.dumps(binding,indent=2)}
```
'''
    (DEST/'START_HERE_VAST.md').write_text(start,encoding='utf-8')
    # README_FIRST.md is part of the inherited immutable data-transfer manifest.
    # Keep it byte-identical; the new entrypoint is START_HERE_VAST.md.
    handoff='''# Codex handoff — one fresh Vast LED-large F run using D059 as reference

Read START_HERE_VAST.md. Researcher explicitly requests Vast-only fresh LED-large
training on RTX3090 using D059 as reference and personally manages this NEW run's
deadline. Authorization is embedded and hash-bound; do not ask to recreate JSON.
Do not rewrite historical clocks or relabel prior failures. Scope is Ffirst,
2000/200,5epochs/1250successfulupdates and48fixedvalidationpredictions only.
No RO/U,held-out access,Teacher calls,source/target/reference changes,extra searches.

Verify all package SHA256s, pinned .venv, config/runner/preflight bindings, D059
method mapping and GPU availability. SETUP_VAST.sh is CPUonly. Use tmux then
RUN_FRESH_F.sh once. A previously failed/missing GPU cannot be bypassed. The new
gate must PASS; old T4/Vast gates do not substitute. No old adapter/optimizer import.
Preserve raw primary/secondary errors; no auto-restart or silent fallback.

For disconnect, reconnect to the existing tmux session; do not start duplicate jobs.
After successful train/evaluation, export actual predictions,automatic metrics,
sources/captions/photos,references,checkpoint receipts and runtime logs. Researcher
quality labels stay blank/pending. Stop for independent review; do not launch RO/U.
If the host fails, preserve and back up evidence; no reset/reboot/rental by this scope.
'''
    (DEST/'CODEX_VAST_HANDOFF.md').write_text(handoff,encoding='utf-8')
    # The reference config is preserved verbatim, rather than relabelled as a large config.
    shutil.copyfile(ROOT/PREP/'METHOD_COMPARISON.json',DEST/'D059_METHOD_COMPARISON.json')

def archive():
    p=DEST.with_suffix('.zip');assert DEST.is_dir() and not p.exists(),'New complete archive required'
    manifest=read(DEST/'VAST_SHA256_MANIFEST.json')['files']
    with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,h in manifest.items():
            assert sha(DEST/rel)==h,'Staging changed: '+rel
            z.write(DEST/rel,rel)
        z.write(DEST/'VAST_SHA256_MANIFEST.json','VAST_SHA256_MANIFEST.json')
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'VAST_SHA256_MANIFEST.json'}
    digest=sha(p)
    Path(str(p)+'.sha256').write_text(digest+'  '+p.name+'\n',encoding='utf-8')
    save(PREP+'/DELIVERED_ZIP_RECEIPT.json',dict(archive=str(p),sha256=digest,bytes=p.stat().st_size,
        files=len(manifest)+1,CPU_only_preparation=True,new_GPU_execution=False))
    print('ARCHIVE:',p,'SHA256:',digest,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('sources','build','archive'))
    mode=ap.parse_args().mode
    {'sources':sources,'build':build,'archive':archive}[mode]()
