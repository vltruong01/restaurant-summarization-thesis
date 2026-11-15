"""Build an isolated, self-contained RTX3090 fresh-F handoff. CPU file work only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'outputs/cloud_handoff/led_large_full_train_colab_verification_v2/extracted'
NAME = 'LED_LARGE_VAST_FRESH_V1'
DEST = ROOT/'outputs/cloud_handoff'/NAME
PREP = 'outputs/stage_i/led_large_vast_fresh_v1_preparation'
HELPER = 'scripts/stage_i_led_large_vast_fresh_v1.py'
RUNNER = 'scripts/run_stage_i_led_large_vast_fresh_v1.py'
CONFIG = 'configs/stage_i_led_large_vast_fresh_v1.json'
DEADLINE = '2026-10-21T17:11:48+00:00'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(rel, value):
    p = ROOT/rel
    p.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    p.write_text(text+'\n', encoding='utf-8', newline='\n')


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def prepare_code():
    v5 = (ROOT/'scripts/stage_i_led_large_full_train_v5.py').read_text()
    helper = '''"""RTX3090 independent fresh F: versioned transport; unchanged v5 science."""
from stage_i_led_large_full_train_v5 import *
import stage_i_led_large_full_train_v5 as historical

CONFIG = ROOT/'configs/stage_i_led_large_vast_fresh_v1.json'
PREP = ROOT/'outputs/stage_i/led_large_vast_fresh_v1_preparation'
RUNNER = ROOT/'scripts/run_stage_i_led_large_vast_fresh_v1.py'
EXPERIMENT = 'LED_LARGE_VAST_FRESH_V1'

def binding():
    return dict(config_sha256=sha(CONFIG),runner_sha256=sha(RUNNER),
                preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))

def preflight():
    old = historical.preflight()
    c = read(CONFIG)
    extra = c['vast_fresh']
    require(c == dict(old, vast_fresh=extra), 'Scientific config differs from v5')
    require(extra == dict(experiment=EXPERIMENT, GPU='NVIDIA GeForce RTX 3090', capability=[8,6],
        initialization='fresh_official_backbone_and_fresh_named_seeded_LoRA_no_Colab_checkpoint',
        output='runs/F_vast3090_fresh_v1', deadline_utc=DEADLINE,
        gate_updates=3, gate_microbatches=24, gate_generation_calls=2,
        prior_T4_gate_is_historical_only=True), 'Unexpected hardware/namespace/budget amendment')
    for rel,h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel) == h, 'Vast dependency changed: '+rel)
    binary = ROOT/c['local_model_directory']/'pytorch_model.bin'
    require(binary.stat().st_size == 1839633783 and sha(binary) == c['weights_sha256'], 'Backbone bytes differ')
    return c

def authorize(path, mode, execute):
    require(execute and path is not None, 'GPU disabled: explicit bound execution approval required')
    a = read(path)
    require(a.get('GPU_execution_approved') is True and a.get('training_authorized') is True,
            'Researcher execution approval required')
    require(a.get('experiment') == EXPERIMENT and a.get('approval_source') == 'researcher_message'
            and a.get('researcher_explicit_authorization'), 'Wrong authorization provenance')
    require(a.get('scopes') == ['feasibility','train','evaluate'] and mode in a['scopes'], 'Wrong scope')
    require(all(a.get(k) == v for k,v in binding().items()), 'Authorization hash mismatch')
    require(a.get('max_training_updates') == 1250 and a.get('evaluation_outputs_max') == 48
            and a.get('feasibility_updates_max') == 3 and a.get('feasibility_microbatches_max') == 24
            and a.get('technical_generation_calls_max') == 2, 'Execution budget differs')
    require(a.get('execution_deadline_utc') == DEADLINE and a.get('amp_scaler') == AMP,
            'Original deadline/AMP changed')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False
            and a.get('conditional_train_after_feasibility_pass') is True, 'Additional scope forbidden')
    check_deadline(a)
    return dict(a, _approval_sha256=sha(path))

def require_gate(out, a):
    p = Path(out)/'feasibility'
    r = read(p/'REPORT.json')
    require(not (p/'FAILURE_STOP_NO_RETRY.json').exists(), 'Failed gate cannot authorize train')
    require(r['status'] == 'PASS' and r['binding'] == binding() == read(p/'BINDING.json'), 'Foreign/failed gate')
    require(r['approval_sha256'] == a['_approval_sha256'] and r['deadline_utc'] == DEADLINE,
            'Gate authorization differs')
    require(r['optimizer_updates'] == 3 and r['microbatches'] == 24 and r['generation_calls'] == 2
            and r['replay_updates'] == 0 and r['headroom_fraction'] >= .15, 'Gate budget/memory differs')
    require(r['state_restore_exact'] and r['adapter_restore_exact'] and r['frozen_backbone_unchanged']
            and r['D051_strict_replay'] == 'FAIL_UNCHANGED', 'Gate restoration/integrity differs')
    initial = read(p/'INITIAL_MODEL_BINDINGS.json')
    require(initial['initialization'] == read(frozen.PREP/'INITIALIZATION_HASHES.json')['per_module_sha256'],
            'Gate fresh initialization differs')
    require(initial['scope'] == dict(modules=228, trainable_tensors=456, trainable_parameters=4915200,
                                    frozen_backbone_parameters=459801600), 'Adapter scope differs')
    proof = read(p/'checkpoint_step3/integrity.json')
    require(proof['binding'] == binding() and set(proof['files']) ==
            {'README.md','adapter_model.safetensors','adapter_config.json','state.pt'}, 'Incomplete gate checkpoint')
    for n,h in proof['files'].items():
        require(Path(n).name == n and sha(p/'checkpoint_step3'/n) == h, 'Gate checkpoint changed: '+n)
    return r

'''
    # Preserve public-state scaler initialization and all gradient/checkpoint callback checks.
    helper += function(v5, 'configure_amp_scaler').replace('accepted_v4_gate_report_sha256=GATE_SHA',
        "historical_T4_AMP_reference_sha256=GATE_SHA")+'\n'+function(v5, 'callback_class')
    save(HELPER, helper)
    original = (ROOT/'scripts/run_stage_i_led_large_full_train_v5.py').read_text()
    runner = original[:original.index('\ndef main():')]
    runner = runner.replace('from stage_i_led_large_full_train_v5 import (', 'from stage_i_led_large_vast_fresh_v1 import (')
    # Resume is deliberately not exposed; training and inference implementations stay unchanged.
    gatecode = (ROOT/'scripts/run_stage_i_led_large_gradient_gate_v4.py').read_text()
    runner += '\nfrom led_large_gradient_evidence_v3 import inspect_gradients, finite_value, snapshot_raw_gradients, snapshot_rng\n\n'
    runner += function(gatecode, 'feasible')
    runner = runner.replace("check_gpu_same(gate,torch);trainset", "check_gpu_same(gate,torch);trainset")
    runner = runner.replace("write(folder/'ACCEPTED_V4_GATE.json'", "write(folder/'ACCEPTED_VAST_GATE.json'")
    runner = runner.replace("gate_output=a['_gate_output']", "gate_output=str(out)")
    # Explicit hardware check before any model load. Store the exact actual profile for subsequent modes.
    runner = runner.replace("train,_ = data(c);ids", "check_required_gpu(torch);train,_ = data(c);ids")
    runner = runner.replace("gate = require_gate(out,a);folder", "check_required_gpu(torch);gate = require_gate(out,a);folder")
    runner = runner.replace("gate = require_gate(out,a)\n    complete", "check_required_gpu(torch);gate = require_gate(out,a)\n    complete")
    runner += '''
def check_required_gpu(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA GPU required')
    require(torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090'
            and list(torch.cuda.get_device_capability()) == [8,6], 'Required RTX3090 hardware differs')
    require(torch.cuda.get_device_properties(0).total_memory >= 23*1024**3, 'Expected RTX3090 24GB memory missing')

def main():
    ap = argparse.ArgumentParser(description='Fresh independent RTX3090 F run; never resume Colab')
    ap.add_argument('mode', choices=('preflight','feasibility','train','evaluate'))
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--approval', type=Path)
    ap.add_argument('--output', type=Path, default=ROOT/'runs/F_vast3090_fresh_v1')
    args = ap.parse_args()
    c = preflight() # Runtime + complete frozen input/model hash checks, CPU only.
    if args.mode == 'preflight':
        print('CPU preflight PASS', binding(), flush=True);return
    a = authorize(args.approval,args.mode,args.execute)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out = args.output.resolve()
    require(out == ROOT/'runs/F_vast3090_fresh_v1', 'Use the isolated fixed fresh namespace')
    check_deadline(a)
    if args.mode == 'feasibility':
        require(not out.exists(), 'Fresh gate namespace required; no automatic retry')
        feasible(c,a,out)
    elif args.mode == 'train':
        train(c,a,out) # Same freshly initialized backbone/adapters; no gate adapter reuse.
    else:
        evaluate(c,a,out)

if __name__ == '__main__':
    main()
'''
    compile(runner, RUNNER, 'exec')
    save(RUNNER, runner)
    report = (ROOT/'scripts/report_stage_i_led_large_full_train_v5.py').read_text()
    report = report.replace('from stage_i_led_large_full_train_v5 import ', 'from stage_i_led_large_vast_fresh_v1 import ')
    report = report.replace("'ACCEPTED_V4_GATE.json'", "'ACCEPTED_VAST_GATE.json'")
    save('scripts/report_stage_i_led_large_vast_fresh_v1.py', report)
    cfg = json.loads((ROOT/'configs/stage_i_led_large_full_train_v5.json').read_text())
    cfg['vast_fresh'] = dict(experiment=NAME,GPU='NVIDIA GeForce RTX 3090',capability=[8,6],
        initialization='fresh_official_backbone_and_fresh_named_seeded_LoRA_no_Colab_checkpoint',
        output='runs/F_vast3090_fresh_v1',deadline_utc=DEADLINE,gate_updates=3,gate_microbatches=24,
        gate_generation_calls=2,prior_T4_gate_is_historical_only=True)
    save(CONFIG,cfg)


def build():
    prepare_code()
    if DEST.exists():
        raise ValueError('Fresh package directory required: '+str(DEST))
    DEST.mkdir()
    original = json.loads((WORK/'SHA256_MANIFEST.json').read_text())['files']
    bound = dict(original)
    # Recursively include all inherited execution preflights and their exact dependencies.
    queue = ['outputs/stage_i/led_large_full_train_v5_preparation/PREFLIGHT_MANIFEST.json']
    seen = set()
    while queue:
        rel = queue.pop()
        if rel in seen: continue
        seen.add(rel)
        src = ROOT/rel if (ROOT/rel).exists() else WORK/rel
        bound[rel] = sha(src)
        for name,h in json.loads(src.read_text())['files'].items():
            if name in bound and bound[name] != h: raise ValueError('Conflicting frozen hash: '+name)
            bound[name] = h
            # Historical manifests inside provenance are inert evidence, not runtime
            # dependencies. Follow only manifests actually invoked by v5 preflight.
            if name in ('outputs/stage_i/led_large_colab_preparation_v2/PREFLIGHT_MANIFEST.json',
                        'outputs/stage_i/led_large_gradient_gate_v4_preparation/PREFLIGHT_MANIFEST.json'):
                queue.append(name)
    # Add verified official weights, initialization record and required local overlays.
    extras = [HELPER,RUNNER,CONFIG,'scripts/report_stage_i_led_large_vast_fresh_v1.py',
              'tests/test_led_large_vast_fresh_v1.py','scripts/build_led_large_vast_fresh_v1.py',
              'scripts/verify_led_large_vast_fresh_v1.py','SHA256_MANIFEST.json',
              'assets/models/led-large-16384/pytorch_model.bin',
              'assets/models/led-large-16384/config.json',
              'outputs/stage_i/led_large_colab_preparation_v2/INITIALIZATION_HASHES.json']
    for rel in extras:
        src = ROOT/rel if (ROOT/rel).exists() else WORK/rel
        bound[rel] = sha(src)
    for rel,h in sorted(bound.items()):
        src = ROOT/rel if (ROOT/rel).exists() and sha(ROOT/rel) == h else WORK/rel
        if not src.is_file() or sha(src) != h: raise ValueError('Missing/changed dependency: '+rel)
        p = DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
    manifest = dict(files={k:bound[k] for k in extras+list(seen)},
        inherited_frozen_v5_preflight=True, original_payload_files=len(original),
        model_binary_included=True, GPU_executed_during_preparation=False)
    save(PREP+'/PREFLIGHT_MANIFEST.json',manifest)
    p = DEST/PREP;p.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/PREP/'PREFLIGHT_MANIFEST.json',p/'PREFLIGHT_MANIFEST.json')
    bindings = dict(config_sha256=sha(DEST/CONFIG),runner_sha256=sha(DEST/RUNNER),
                    preflight_manifest_sha256=sha(p/'PREFLIGHT_MANIFEST.json'))
    save(PREP+'/EXECUTION_BINDINGS.json',bindings)
    shutil.copyfile(ROOT/PREP/'EXECUTION_BINDINGS.json',p/'EXECUTION_BINDINGS.json')
    now = datetime.now(timezone.utc).isoformat()
    approval = dict(experiment=NAME,GPU_execution_approved=True,training_authorized=True,
        approval_source='researcher_message',researcher_explicit_authorization=[
            'train lại hoàn toàn riêng với colab để so sánh',
            'Vast.ai RTX 3090, độc lập với lượt Colab hiện tại'],
        approval_recorded_at_utc=now, execution_authorized_at_utc=now,
        timestamp_note='Record creation time; exact chat receipt timestamp unavailable. Historical deadline NOT extended.',
        original_authorization_utc='2026-10-19T17:11:48+00:00',execution_deadline_utc=DEADLINE,
        scopes=['feasibility','train','evaluate'],max_training_updates=1250,evaluation_outputs_max=48,
        feasibility_updates_max=3,feasibility_microbatches_max=24,technical_generation_calls_max=2,
        conditional_train_after_feasibility_pass=True,amp_scaler=json.loads((ROOT/CONFIG).read_text())['amp_scaler'],
        defense_reserve_at_least_14_days_confirmed=True,defense_reserve_source='Preserved existing approved reserve and earlier absolute deadline',
        held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,
        rental_or_instance_creation_authorized_by_this_file=False,**bindings)
    (DEST/'VAST_RESEARCHER_AUTHORIZATION.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    setup = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use one visible GPU. Install in an isolated venv; do not change system ML libraries.
command -v uv >/dev/null || { echo 'Install uv or provide it on PATH, then rerun setup. No GPU work started.'; exit 1; }
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'torch==2.1.0+cu118' --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r requirements/led_large_colab_core.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m unittest discover -s tests -p test_led_large_vast_fresh_v1.py -v
.venv/bin/python scripts/verify_led_large_vast_fresh_v1.py --tokenize
.venv/bin/python scripts/run_stage_i_led_large_vast_fresh_v1.py preflight
'''
    (DEST/'SETUP_VAST.sh').write_text(setup,encoding='utf-8',newline='\n')
    run = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=runs/F_vast3090_fresh_v1
test ! -e "$OUT" || { echo 'Run already exists: preserve evidence; no automatic retry.'; exit 1; }
mkdir -p launcher_logs
test ! -e launcher_logs/vast_fresh_v1.log || { echo 'Launcher log exists: inspect before any retry.'; exit 1; }
exec > >(tee launcher_logs/vast_fresh_v1.log) 2>&1
PY=.venv/bin/python
$PY scripts/run_stage_i_led_large_vast_fresh_v1.py preflight
for MODE in feasibility train evaluate; do
  $PY -u scripts/run_stage_i_led_large_vast_fresh_v1.py "$MODE" --execute --approval VAST_RESEARCHER_AUTHORIZATION.json --output "$OUT"
done
$PY scripts/report_stage_i_led_large_vast_fresh_v1.py --output "$OUT"
echo 'Finished 48 outputs. Stop for researcher review; no RO/U or further experiment.'
'''
    (DEST/'RUN_FRESH_F.sh').write_text(run,encoding='utf-8',newline='\n')
    docs(DEST, bindings, len(original), now)
    print(json.dumps(dict(package=str(DEST),binding=bindings,files=len(bound)),indent=2),flush=True)


def docs(dest, bindings, n, now):
    text = f'''# Bắt đầu tại đây — LED-large F mới trên Vast.ai RTX3090

Đây là lượt train **mới hoàn toàn, độc lập với Colab** theo yêu cầu nghiên cứu.
Không lấy checkpoint250, adapter feasibility hay optimizer/RNG Colab làm khởi tạo.
Giữ nguyên lượt Colab đang chạy và toàn bộ kết quả cũ. Gói không tự thuê GPU.

## Gói đã có gì?
Toàn bộ F train2000/validation200, NPZ/JSONL đã chốt, nguồn review/caption,
Teacher targets/provenance, ảnh gốc, tokenizer, D053 references, fixed24 validation,
các benchmark đã lưu, backbone LED-large chính thức đã pin (binary 1,839,633,783 bytes),
script train/evaluate/report và runtime pins. Không có held-out contents/API keys.
{n} tệp payload gốc giữ byte-exact. Không phải tải lại backbone hoặc tự ghép ZIP cũ.
Các README/Colab notebook cũ trong payload là hồ sơ lịch sử; hướng dẫn này ưu tiên
cho lượt Vast. Không chạy notebook hoặc lệnh T4 cũ.

## Máy và lệnh chạy
Người dùng tự chọn/thuê **một RTX3090 24GB**, Linux x86_64, NVIDIA driver hỗ trợ CUDA11.8,
Python3.11 qua uv, RAM hệ thống nên32GB, disk trống nên30GB (gói+giải nén+venv+checkpoints).
Đây là dự trù triển khai, không phải VRAM/runtime đã đo trên3090. Cần mạng cho dependencies.
Upload đúng một ZIP, unzip vào SSD cục bộ ví dụ /workspace/LED_LARGE_VAST_FRESH_V1.
Mở Codex trong thư mục đó và đọc CODEX_VAST_HANDOFF.md.

```bash
cd /workspace/LED_LARGE_VAST_FRESH_V1
bash SETUP_VAST.sh
# Chạy trong tmux/screen để mất kết nối SSH không làm chết tiến trình:
bash RUN_FRESH_F.sh
```
Nếu chưa có uv, Codex cài uv từ nguồn chính thức hoặc dùng uv sẵn có; không đổi ML pins.
SETUP chỉ CPU kiểm tra; RUN mới chạy GPU. Giữ một GPU visible. Không cài bitsandbytes/quantization.
Approval JSON đã ghi yêu cầu thật trong chat và hash mới; không cần tạo file phê duyệt thủ công.
Không dùng approval T4 cũ. Timestamp {now} là lúc lập record, không giả là timestamp UI.

## Phạm vi và kiểm soát
1. Gate RTX3090 mới:3 optimizer updates/24 microbatches,2 technical train-input beam4 traces
   tối đa768token,0replay. Kiểm tra456gradients,AMP1024,masks,backbone,checkpoint round-trip,
   ít nhất15% headroom. T4 PASS chỉ là lịch sử;3090 chưa được kiểm tra.
2. Chỉ sau PASS:load backbone/LoRA mới lại từ đầu,228modules/4,915,200trainableparameters,
   r8/alpha16/dropout0;FP32stored/FP16AMP1024;F2000/200;5epochs/1250updates,warmup63,
   AdamW2e-4,batch1/accum8,seed42,cùng EPOCH_ORDERS và nguồn5000/label768.
3. Giữ5checkpoint250/500/750/1000/1250;khôngearly-stop/bestlossreload.
4. Sinh đúng48developmentoutputs epoch3/5,mỗi24(random20+targeted4),beam4/max_length768.
5. Xuất HTML/metrics/qualityZIP và dừng. Không train RO/U, không test/Teacher requests.

**Deadline gốc không đổi: {DEADLINE} (00:11:48 ngày22/10 giờ Việt Nam).**
Nếu đã quá hạn hoặc không đủ thời gian, dừng và báo; không tự gia hạn hoặc rút ngắn protocol.
Mọi failure đều giữ logs/checkpoints; không tự retry/fallback/bỏ gradient checks/đổi precision.
Không tự resume sau disconnect. Run namespace cố định: runs/F_vast3090_fresh_v1.

## So sánh với Colab
Cùng backbone/data/seed/order/schedule/AMP/decoding;khác GPU/driver và có thể cuDNN,
vì vậy không cam kết bitwise-identical hoặc coi GPU nhanh hơn là chất lượng tốt hơn.
Ghi walltime/update,validation/generation/checkpoint I/O riêng. ETA3090 chưa đo.
Colab hiện chưa được chứng minh hoàn tất;gói không có48outputsColab chưa tồn tại.
Khi cả hai hoàn tất, so sánh cùng epoch3/5 và business IDs, tách random20/targeted4,
đánh giá Corecoverage,opinioncontrasts,grounding,semanticrepetition,visualusefulness,
zero-caption và completeness với references. Softwareflags không phải humanlabels.
Không chọn checkpoint chỉ từ loss/headers/repetition. D051strictreplayFAIL giữ nguyên.

## Kết quả cần tải về
Giữ nguyên toàn bộ runs/F_vast3090_fresh_v1 và launcher_logs (gồm optimizer/scheduler/scaler/RNG
và5checkpoints), cộng chất lượng runs/F_vast3090_fresh_v1/LED_LARGE_F_QUALITY_REVIEW.zip.
Backup trước khi hủy instance; không đưa bí mật/key vào archive.

## Integrity
VAST_SHA256_MANIFEST.json kiểm tra toàn bộ payload mới (không tự-hash chính nó).
VAST_EXECUTION_BINDINGS.json và VAST_RESEARCHER_AUTHORIZATION.json chứa3hash mới:
```json
{json.dumps(bindings,indent=2)}
```
Không sửa scientific files. Offline chuẩn bị không xác nhận actual3090feasibility/training.
'''
    (dest/'START_HERE_VAST.md').write_text(text,encoding='utf-8',newline='\n')
    handoff = '''# Prompt bàn giao Codex trên Vast.ai

Read START_HERE_VAST.md before all historical README/Colab instructions.
Researcher requested a fresh independent RTX3090 Filtered training run to compare
with the ongoing Colab run. Preserve Colab; do not import/resume its adapter,
optimizer, scheduler, scaler or RNG. Approval source is the two actual researcher
messages recorded transparently in VAST_RESEARCHER_AUTHORIZATION.json; no UI actions fabricated.
Do not create/rent another instance or alter deadlines. Work only on this uploaded instance.

First verify VAST_SHA256_MANIFEST.json with Python hashlib, then SETUP_VAST.sh,
the new regression suite and CPU preflight in this extracted workspace. Do not edit
immutable configs/dependency files to make checks pass. Inspect START_HERE_VAST.md
and the exact new binding. The original T4 gate does not authorize3090 training.

Run RUN_FRESH_F.sh once, preferably in tmux. It enforces new3090 feasibility before
fresh full F training, then48outputs and source-based review export. Same scientific
protocol as Colab v5; do not change batch/AMP/LR/order/checkpoint/beam settings.
Stop on failure/expiry, preserve full evidence and report the actual cause. No retries,
fallback or automatic resumption. Do not patch failed Unlikelihood or reopen D051.
No RO/U, Teacher requests, held-out access, target/caption/ref changes or other experiments.

At completion give actual stage runtimes, memory headroom,1250update/5checkpoint receipts,
48predictions splitrandom20/targeted4,softwaremetric reports and reviewZIP with original
sources/captions/photos/approvedreferences. AI quality opinions remain provisional.
Archive full run and launcher logs for user download before shutting down instance.
Do not declare researcher GO/NO-GO or claim bitwise reproducibility between T4 and3090.
'''
    (dest/'CODEX_VAST_HANDOFF.md').write_text(handoff,encoding='utf-8',newline='\n')
    (dest/'VAST_EXECUTION_BINDINGS.json').write_text(json.dumps(bindings,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    build()
