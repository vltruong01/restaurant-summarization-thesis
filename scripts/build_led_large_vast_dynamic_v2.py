"""Offline versioned amendment of the existing self-contained3090 handoff."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='LED_LARGE_VAST_DYNAMIC_65536_V2'
DEST=ROOT/'outputs/cloud_handoff'/NAME
PREP='outputs/stage_i/led_large_vast_dynamic_v2_preparation'
HELPER='scripts/stage_i_led_large_vast_dynamic_v2.py'
RUNNER='scripts/run_stage_i_led_large_vast_dynamic_v2.py'
CONFIG='configs/stage_i_led_large_vast_dynamic_v2.json'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(rel,value):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    s=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    p.write_text(s+'\n',encoding='utf-8',newline='\n')


def code():
    from led_large_dynamic_amp_v2 import POLICY
    old=json.loads((ROOT/'configs/stage_i_led_large_vast_fresh_v1.json').read_text())
    c=json.loads(json.dumps(old));c['amp_scaler']['init_scale']=65536.0
    c['vast_fresh'].update(experiment=NAME,output='runs/F_vast3090_dynamic65536_v2',gate_microbatches=88,nominal_gate_microbatches=24)
    c['dynamic_overflow_policy']=POLICY
    save(CONFIG,c)
    helper=(ROOT/'scripts/stage_i_led_large_vast_fresh_v1.py').read_text()
    helper=helper.replace('RTX3090 independent fresh F: versioned transport; unchanged v5 science.',
                          'RTX3090 fresh F: explicit65536/bounded-overflow numerical-policy amendment.')
    helper=helper.replace('stage_i_led_large_vast_fresh_v1','stage_i_led_large_vast_dynamic_v2')
    helper=helper.replace('led_large_vast_fresh_v1_preparation','led_large_vast_dynamic_v2_preparation')
    helper=helper.replace('LED_LARGE_VAST_FRESH_V1',NAME).replace('runs/F_vast3090_fresh_v1','runs/F_vast3090_dynamic65536_v2')
    helper=helper.replace('gate_microbatches=24, gate_generation_calls=2','gate_microbatches=88, nominal_gate_microbatches=24, gate_generation_calls=2')
    helper=helper.replace("EXPERIMENT =", "from led_large_dynamic_amp_v2 import POLICY\nAMP=dict(AMP,init_scale=65536.0)\nEXPERIMENT =",1)
    helper=helper.replace("dict(old, vast_fresh=extra)","dict(old, vast_fresh=extra, amp_scaler=AMP, dynamic_overflow_policy=POLICY)")
    helper=helper.replace("require(a.get('max_training_updates')", "require(a.get('dynamic_overflow_policy') == POLICY, 'Wrong dynamic overflow policy')\n    require(a.get('max_training_updates')",1)
    helper=helper.replace("a.get('feasibility_microbatches_max') == 24", "a.get('feasibility_microbatches_max') == 88 and a.get('nominal_feasibility_microbatches') == 24")
    helper=helper.replace("and r['microbatches'] == 24 and r['generation_calls'] == 2", "and r['microbatches'] == 24 and r['generation_calls'] == 2\n            and r['actual_forward_backward_microbatches'] <= POLICY['gate_microbatches_max']\n            and r['overflow_skips'] <= POLICY['max_overflow_skips'] and r['dynamic_overflow_policy'] == POLICY")
    helper=helper.replace("require(self.trainer is not None and not self.trainer.accelerator.optimizer_step_was_skipped, 'AMP skipped update')",
        "require(self.trainer is not None and not self.trainer.accelerator.optimizer_step_was_skipped, 'Unexpected skipped outer update')\n            self.trainer.amp_recovery.commit(self.trainer.accelerator.scaler)")
    helper=helper.replace("r['microbatches'] == 24", "r['nominal_microbatches'] == 24 and r['microbatches'] <= 88")
    helper=helper.replace("needed = ('adapter_model.safetensors'", "write(p/'AMP_DYNAMIC_STATE.json',self.trainer.amp_recovery.state(self.trainer.accelerator.scaler))\n            needed = ('adapter_model.safetensors'",1)
    helper=helper.replace('No optimizer, loss math, precision, growth factors or skip guards are changed.',
        'Native dynamic scaling starts at65536; finite-forward loss, missing-gradient and bounded-backoff guards remain.')
    compile(helper,HELPER,'exec');save(HELPER,helper)
    runner=(ROOT/'scripts/run_stage_i_led_large_vast_fresh_v1.py').read_text()
    runner=runner.replace('Full F training/evaluation after accepted v4 gate; no gate rerun or adapter reuse.',
                          'Fresh3090 F with new bounded dynamic65536 gate; no checkpoint or adapter reuse.')
    runner=runner.replace('stage_i_led_large_vast_fresh_v1','stage_i_led_large_vast_dynamic_v2')
    runner=runner.replace('runs/F_vast3090_fresh_v1','runs/F_vast3090_dynamic65536_v2')
    runner=runner.replace('import traceback','import traceback\nfrom led_large_dynamic_amp_v2 import Recovery, POLICY, install_trainer',1)
    runner=runner.replace("                require(torch.isfinite(output.loss).item(), 'Nonfinite train/validation loss')",
        "                if not torch.isfinite(output.loss).item():\n                    write(folder/'NONFINITE_FORWARD_LOSS.json',dict(loss=str(float(output.loss.detach())),\n                        phase='train' if model.training else 'validation',global_step=self.state.global_step,\n                        scale=float(self.accelerator.scaler.get_scale()),automatic_recovery=False))\n                    raise ValueError('Nonfinite train/validation forward loss; STOP')")
    anchor='            def _load_rng_state(self, checkpoint):'
    insert='''            def training_step(self, model, inputs, num_items_in_batch=None):
                recovery=self.amp_recovery
                if not recovery.window:recovery.begin()
                recovery.window.append({k:v.detach().cpu().clone() for k,v in inputs.items()})
                require(len(recovery.window)<=8,'Accumulation window exceeds8')
                recovery.microbatch()
                return super().training_step(model,inputs,num_items_in_batch)

'''
    assert runner.count(anchor)==1;runner=runner.replace(anchor,insert+anchor)
    anchor='            callback.trainer = trainer;trainer.model_accepts_loss_kwargs = False;set_seed(42)'
    assert runner.count(anchor)==1
    runner=runner.replace(anchor,anchor+"\n            install_trainer(trainer,folder,lambda:(check_deadline(a),memory.check()),write,append)")
    runner=runner.replace("generation_calls=0,**memory.summary()", "generation_calls=0,amp_dynamic=trainer.amp_recovery.state(trainer.accelerator.scaler),**memory.summary()")
    runner=runner.replace('GradScaler(init_scale=1024.0','GradScaler(init_scale=65536.0')
    anchor='''            model.train()
            for step in range(1,4):'''
    assert runner.count(anchor)==1
    runner=runner.replace(anchor,'''            recovery=Recovery(p,3,lambda:(check_deadline(a),memory.check()),write,append)
            model.train()
            for step in range(1,4):''')
    start=runner.index("                opt.zero_grad(set_to_none=True);losses = []")
    end=runner.index("                grads = gradient_record(model,expect_later_A=step>=3)",start)
    block='''                opt.zero_grad(set_to_none=True);recovery.begin();losses=[]
                def backward_window():
                    losses.clear()
                    for j in range(8):
                        recovery.microbatch();check_deadline(a)
                        feature=features[((step-1)*8+j)%len(features)]
                        batch={k:v.cuda() for k,v in pack([feature]).items()}
                        with torch.autocast('cuda',dtype=torch.float16):
                            loss=model(**batch,use_cache=False).loss
                        if not torch.isfinite(loss).item():
                            write(p/'NONFINITE_FORWARD_LOSS.json',dict(loss=str(float(loss.detach())),
                                step=step,microbatch=j+1,scale=float(scaler.get_scale()),automatic_recovery=False))
                            raise ValueError('Nonfinite forward CE loss; not recoverable AMP overflow')
                        losses.append(float(loss.detach()))
                        scaler.scale(loss/8).backward()
                        append(p/'MICROBATCH_EVIDENCE.jsonl',dict(step=step,microbatch=j+1,
                            actual_microbatches=recovery.microbatches,retry=recovery.replaying,
                            CE_loss=float(loss.detach()),scale=float(scaler.get_scale())))
                        memory.check()
                backward_window()
                def native_skip():
                    scaler.step(opt);scaler.update()
                recovery.finish(model,opt,scaler,lambda:scaler.unscale_(opt),native_skip,backward_window)
                write(p/f'UNSCALED_GRADIENTS_step{step}.json',inspect_gradients(model,detailed=True))
'''
    runner=runner[:start]+block+runner[end:]
    # Recovery has already clipped finite gradients once, exactly at the original clip location.
    runner=runner.replace("                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)\n",'')
    anchor="                require({int(s['step']) for s in opt.state.values()} == {step}, 'AMP skipped update')"
    runner=runner.replace(anchor,anchor+"\n                recovery.commit(scaler)")
    runner=runner.replace("microbatches=24,generation_calls=2,replay_updates=0,state_restore_exact=True,",
        "microbatches=24,generation_calls=2,replay_updates=0,state_restore_exact=True,\n              actual_forward_backward_microbatches=recovery.microbatches,overflow_skips=recovery.overflow_skips,\n              dynamic_overflow_policy=POLICY,scale_final=float(scaler.get_scale()),")
    runner=runner.replace("diagnostics['scaler_state']=scaler.state_dict()", "diagnostics['scaler_state']=scaler.state_dict()\n                if 'recovery' in locals():diagnostics['amp_dynamic']=recovery.state(scaler)")
    runner=runner.replace('microbatches=24,generation_calls=2,replay_updates=0,state_restore_exact=True,', 'microbatches=recovery.microbatches,nominal_microbatches=24,generation_calls=2,replay_updates=0,state_restore_exact=True,')
    compile(runner,RUNNER,'exec');save(RUNNER,runner)
    report=(ROOT/'scripts/report_stage_i_led_large_vast_fresh_v1.py').read_text().replace('stage_i_led_large_vast_fresh_v1','stage_i_led_large_vast_dynamic_v2')
    report=report.replace("'ACCEPTED_VAST_GATE.json')", "'ACCEPTED_VAST_GATE.json','AMP_DYNAMIC.jsonl')")
    save('scripts/report_stage_i_led_large_vast_dynamic_v2.py',report)


def build():
    from led_large_dynamic_amp_v2 import POLICY
    code()
    if DEST.exists():raise ValueError('New package already exists; preserve it')
    DEST.mkdir()
    oldroot=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_FRESH_V1'
    oldmanifest=json.loads((oldroot/'VAST_SHA256_MANIFEST.json').read_text())['files']
    # Preserve old immutable sources/configs/preflight bytes; supersede only entrypoint documents.
    skip={'START_HERE_VAST.md','CODEX_VAST_HANDOFF.md','VAST_EXECUTION_BINDINGS.json',
          'VAST_RESEARCHER_AUTHORIZATION.json','SETUP_VAST.sh','RUN_FRESH_F.sh',
          'VAST_READINESS_REPORT.md','COLAB_VS_VAST_COMPARISON.md','VAST_RUNNER_DIFF.patch'}
    for rel,h in oldmanifest.items():
        if rel in skip:continue
        src=oldroot/rel
        if sha(src)!=h:raise ValueError('Old package changed: '+rel)
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        # Hard links save local staging space; no original file is edited via these links.
        try:dst.hardlink_to(src)
        except OSError:shutil.copyfile(src,dst)
    extra=[HELPER,RUNNER,CONFIG,'scripts/led_large_dynamic_amp_v2.py',
           'scripts/report_stage_i_led_large_vast_dynamic_v2.py','scripts/build_led_large_vast_dynamic_v2.py',
           'tests/test_led_large_dynamic_amp_v2.py']
    for rel in extra:
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    bound={rel:sha(DEST/rel) for rel in extra}
    bound['outputs/stage_i/led_large_full_train_v5_preparation/PREFLIGHT_MANIFEST.json']=sha(DEST/'outputs/stage_i/led_large_full_train_v5_preparation/PREFLIGHT_MANIFEST.json')
    save(PREP+'/PREFLIGHT_MANIFEST.json',dict(files=bound,original_scientific_artifacts_unchanged=True,
        amendment='init_scale65536_native_bounded_backoff_same_window_RNG_retry_no_missing_successful_updates',
        dynamic_overflow_policy=POLICY))
    (DEST/PREP).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/PREP/'PREFLIGHT_MANIFEST.json',DEST/PREP/'PREFLIGHT_MANIFEST.json')
    b=dict(config_sha256=sha(DEST/CONFIG),runner_sha256=sha(DEST/RUNNER),preflight_manifest_sha256=sha(DEST/PREP/'PREFLIGHT_MANIFEST.json'))
    save(PREP+'/EXECUTION_BINDINGS.json',b)
    shutil.copyfile(ROOT/PREP/'EXECUTION_BINDINGS.json',DEST/PREP/'EXECUTION_BINDINGS.json')
    a=json.loads((oldroot/'VAST_RESEARCHER_AUTHORIZATION.json').read_text())
    a.update(experiment=NAME,**b,amp_scaler=json.loads((DEST/CONFIG).read_text())['amp_scaler'],
        dynamic_overflow_policy=POLICY,feasibility_microbatches_max=88,nominal_feasibility_microbatches=24,
        feasibility_attempt_microbatches_max=88,training_attempt_microbatches_max=10064,
        researcher_explicit_authorization=['vậy sửa zip để tôi sẽ thử dùng 3090 trong vast ai với khởi đầu là 65536','ok hãy làm'],
        amendment_recorded_at_utc=datetime.now(timezone.utc).isoformat())
    # Original start and absolute end stay unchanged, rather than resetting any clock.
    (DEST/'VAST_RESEARCHER_AUTHORIZATION.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    setup=(oldroot/'SETUP_VAST.sh').read_text().replace('test_led_large_vast_fresh_v1.py','test_led_large_dynamic_amp_v2.py').replace('run_stage_i_led_large_vast_fresh_v1.py','run_stage_i_led_large_vast_dynamic_v2.py')
    (DEST/'SETUP_VAST.sh').write_text(setup,encoding='utf-8',newline='\n')
    run=(oldroot/'RUN_FRESH_F.sh').read_text().replace('run_stage_i_led_large_vast_fresh_v1.py','run_stage_i_led_large_vast_dynamic_v2.py').replace('report_stage_i_led_large_vast_fresh_v1.py','report_stage_i_led_large_vast_dynamic_v2.py').replace('F_vast3090_fresh_v1','F_vast3090_dynamic65536_v2').replace('vast_fresh_v1.log','vast_dynamic65536_v2.log')
    (DEST/'RUN_FRESH_F.sh').write_text(run,encoding='utf-8',newline='\n')
    docs(b)
    print(json.dumps(dict(package=str(DEST),binding=b),indent=2),flush=True)


def docs(b):
    text=f'''# LED-large F — RTX3090, dynamic loss scaling từ65.536 (v2)

**Dùng ZIP này thay cho ZIP Vast v1 init_scale1024.** V1 và Colab được giữ nguyên.
Đây là một F run mới từ backbone/LoRA mới; không resume/import adapter Colab/gate.
Gói chứa toàn bộ F2000/200,2,200Teacherartifacts,7,578photos,NPZ/JSONL/tokenizer,
official pinned LED-large binary,approvedreferences,fixed24cases và code/dependencies.
Không cần ZIP cũ/patch/model download thêm. Những README/notebook Colab cũ trong
payload là provenance lịch sử,không phải hướng dẫn thực thi v2.

## Chạy trên máy RTX3090 đã được người dùng cung cấp
Linux x86_64,1visibleRTX3090/24GB,driverCUDA11.8-compatible,RAM nên32GB,disk30GB.
Giải nén vào SSD local,mở Codex tại thư mục {NAME},đọc CODEX_VAST_HANDOFF.md.
Không tự thuê instance khác. Cần uv trên PATH và mạng để cài dependency pins.
```bash
bash SETUP_VAST.sh
# dùng tmux/screen để SSH disconnect không kết thúc training
bash RUN_FRESH_F.sh
```
Approval JSON mới đã nằm trong ZIP,ghi đúng yêu cầu chat;không cần tạo file thủ công.
Deadline cũ vẫn2026-10-21T17:11:48Z (=00:11:48 ngày22/10 giờ Việt Nam),không gia hạn.

## Thay đổi được ghi nhận rõ
- GradScaler **init_scale65536**,growth2,backoff0.5,growth_interval2000;FP16autocast,
  FP32storedweights giữ nguyên.65536 là initial scale,không khóa scale cố định.
- Cho phép tối đa8nativeoverflow skips/mode(gate,train),scale không thấp hơn256.
  Chỉ recover nếu loss forward hữu hạn nhưng gradient nonfinite;missinggradients,
  nonfiniteforwardloss,weights/statechange,OOM/integrity failure vẫn STOP.
- Mỗi overflow: nativeGradScaler bỏ qua AdamW,giảm scale;kiểm tra optimizer/weights
  không thay đổi;zero grads;khôi phục RNG đầu accumulation window và tính lại đúng
  8microbatches. Không bỏ business/đổi data order/tiến scheduler/globalstep lúc skip.
- Giới hạn này là calibration/backward retry có ghi log,không auto-retry toàn run.
  Không giả là3updates/24microbatches cố định khi có overflow:
  **Gate3successfulupdates,24nominal microbatches,<=88actualmicrobatches**;
  **Train1250successfulupdates,10000nominal,<=10064actualmicrobatches**.
  Scheduler vẫn63warmup/1250successfulupdates;full training vẫn5epochs.
- Transformers4.51.3 tự tăngglobalstep sau skippedouterstep. Do đó hook xử lý
  bounded backoff trước outeroptimizerstep;outerstep phải thành công mới được đếm.
  Đây là versioned implementation amendment,không sửa code/checkpoints cũ.

## Phần giữ nguyên
OfficialLED-large revisioncd59d11c3528415b7dda4dfc95cc8f138aceda2e;
228LoRAmodules/456tensors/4,915,200params,r8/alpha16/dropout0,seed42;
F2000train/200validation;source5000/label768;AdamW2e-4,batch1/accum8;
all5epochcheckpoints,noearlystop/bestlossreload. Gate phảiPASS với>=15%headroom,
frozenbackbone,all456gradients,AMPandexactcheckpointroundtrip;2technicaltrain-input
beam4traces max_length768,0replay optimizer updates. Gate states discarded.
Sau fulltrain:sinh48validationoutputs(epoch3/5 ×24cases),beam4/max_length768giữ nguyên.
Random20/targeted4 báo riêng. Không RO/U/Teacher/test/targetcaptionref modifications.
D051 strict replayFAIL và D058–D060/Unlikelihood evidence giữ nguyên,không relabel.

## Log và diễn giải
AMP_DYNAMIC.jsonl ghi skip,scale,microbatches,successfulupdates;AMP_overflow_* giữ
gradient evidence;checkpoint thêmAMP_DYNAMIC_STATE.json vàscaler.pt có state thực.
Missing/zero-gradient rules LoRA A theo số successfulupdates,không theo số retry.
Giảm overflow không chứng minh summary tốt hơn. Actual3090feasibility chưa chạy local.
CPUfixtures test recovery logic không phải LED GPU feasibility hoặc replayPASS.
So T4/1024 với3090/dynamic65536 thay cảhardware/runtime vàinitialscale,không phải
matched causal loss-scale comparison. Không tự chạy thêm1024arm trên3090.

## Sau chạy
Giữ toàn bộ runs/F_vast3090_dynamic65536_v2 vàlauncher_logs,cả5checkpoints/fullstate.
QualityZIP: runs/F_vast3090_dynamic65536_v2/LED_LARGE_F_QUALITY_REVIEW.zip.
Dừng cho researcherreview;AIassessments chỉprovisional. Không autolaunch experiment.
VAST_SHA256_MANIFEST.json kiểm tra payload;VAST_EXECUTION_BINDINGS.json chứa:
```json
{json.dumps(b,indent=2)}
```
'''
    (DEST/'START_HERE_VAST.md').write_text(text,encoding='utf-8')
    handoff='''# Codex Vast handoff — dynamic65536 v2

Read START_HERE_VAST.md first. Use only new dynamic_v2 entrypoints; historical
Colab/v1 scripts and documents are frozen evidence. Researcher explicitly asks
init_scale65536 with bounded native GradScaler backoff. No GPU work executed here.
Validate full package SHA manifest, SETUP_VAST.sh tests, exact approval/config/runner/
preflight binding and original deadline before RUN_FRESH_F.sh once in tmux.
The fresh single3090 run is independent of Colab; never initialize from its adapter
or gate state. Do not rent another GPU or alter scientific inputs or settings.
Gate3successfulupdates/<=88actualmicros,then conditional F1250successfulupdates/
5epochs/<=10064actualmicros,then48epoch3/5outputs. Max8overflowbackoffs per mode;
minimumscale256. Overflows replay only the same8microbatch window with RNG reset,
never advance optimizer/scheduler/globalstep on skips. All counters/evidence logged.
Do not remove numerical/memory/integrity/restore gates, increase limits or silently
retry a failed run. ForwardNaN/missinggradient/OOM remain fatal. No resume authorized.
Preserve D051 strict replayFAIL and previous failures unchanged. No RO/U/Teacher/test.
After48outputs, export software metrics,source/caption/photo/reference review and
all runtime/checkpoint receipts. Quality decisions stay researcher-pending. Back up
full run and logs before instance removal. Do not start another experiment.
'''
    (DEST/'CODEX_VAST_HANDOFF.md').write_text(handoff,encoding='utf-8')
    (DEST/'VAST_EXECUTION_BINDINGS.json').write_text(json.dumps(b,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':build()
