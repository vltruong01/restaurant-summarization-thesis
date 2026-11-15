"""Allowlisted F-only data transfer, not a GPU execution or training authorization."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/cloud_handoff/led_large_f_colab_v1'
PREP = 'outputs/stage_i/led_large_preparation_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        f.write('\n')


def notebook():
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=s.splitlines(True)))
    def code(s):
        compile(s, '<colab_cell>', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=s.splitlines(True), execution_count=None, outputs=[]))
    md('''# LED-large — Filtered-only Colab handoff v1
Chỉ thiết lập CPU, giải nén và kiểm tra offline. Chưa có lệnh train/generate.
Upload ZIP và `.zip.sha256` vào MyDrive. Không cần API key.
Đọc README_FIRST.md và CODEX_HANDOFF.md trước khi làm tiếp.
T4 là ứng viên kiểm tra; chưa xác nhận đủ VRAM. RO/U chưa được phép chạy.''')
    code('''from google.colab import drive
drive.mount('/content/drive')
''')
    code('''from pathlib import Path, PurePosixPath
import hashlib, zipfile, stat
archive = Path('/content/drive/MyDrive/LED_LARGE_F_COLAB_V1.zip')  # sửa nếu cần
receipt = Path(str(archive) + '.sha256')
expected = receipt.read_text().split()[0]
with archive.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
assert actual == expected, 'ZIP checksum mismatch; STOP'
ROOT = Path('/content/led_large_f_colab_v1')
assert not ROOT.exists(), 'Existing directory; inspect before reusing. No overwrite.'
with zipfile.ZipFile(archive) as z:
    for entry in z.infolist():
        p = PurePosixPath(entry.filename)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in entry.filename and '\\\\' not in entry.filename
        assert not stat.S_ISLNK(entry.external_attr >> 16)
    z.extractall(ROOT)
print(ROOT)
''')
    md('''## Môi trường Python riêng
Không thay Torch của kernel Colab. Các lệnh chạy bằng Python 3.11 trong venv riêng.
Không tự nâng phiên bản nếu cài đặt thất bại; giữ log và báo lại.
uv là công cụ bootstrap; runtime ML được ghim trong requirements.''')
    code('''import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', str(ROOT / '.venv')], check=True)
PYTHON = ROOT / '.venv/bin/python'
subprocess.run([str(PYTHON), '-m', 'pip', 'install', 'torch==2.1.0+cu118', '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True)
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements/led_large_colab_core.txt')], check=True)
subprocess.run([str(PYTHON), '-m', 'pip', 'check'], check=True)
(ROOT / 'transfer_runs').mkdir(exist_ok=True)
with (ROOT / 'transfer_runs/installed_freeze.txt').open('w') as f:
    subprocess.run([str(PYTHON), '-m', 'pip', 'freeze'], stdout=f, check=True)
''')
    code('''env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
for pattern in ['test_led_large_colab_transfer_v1.py', 'test_stage_i_led_large_preparation_v1.py']:
    subprocess.run([str(PYTHON), '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'], cwd=ROOT, env=env, check=True)
subprocess.run([str(PYTHON), 'scripts/verify_led_large_colab_bundle_v1.py', '--tokenize', '--runtime', '--report', 'transfer_runs/COLAB_OFFLINE_PREFLIGHT.json'], cwd=ROOT, env=env, check=True)
''')
    md('''## Tùy chọn: tải trọng số công khai đúng revision, chỉ tải file
Không load model, không forward, không dùng GPU. Khoảng 1,84GB; không cần Teacher key.
Nếu chưa cần tải, giữ DOWNLOAD_MODEL=False.''')
    code('''DOWNLOAD_MODEL = False
if DOWNLOAD_MODEL:
    subprocess.run([str(PYTHON), 'scripts/download_led_large_pinned_v1.py', '--download'], cwd=ROOT, check=True)
''')
    md('''## DỪNG trước GPU
Đây là gói dữ liệu + notebook preflight, chưa phải runner LED-large đã được duyệt.
Gửi CODEX_HANDOFF.md cho Codex để chuẩn bị runner từ mã D-059 tham khảo, kiểm tra
228 modules / 4.915.200 parameters, cập nhật hash bindings và trình duyệt trước GPU.
Không chạy runner LED-base cũ với cách đổi mỗi model_id. Không train RO/U.
Sau khi được duyệt: gate 3 updates/24 microbatches + tối đa 2 train-input memory traces;
chỉ train F nếu PASS, cùng cấu hình khoa học và >=15% measured VRAM headroom.
Lưu checkpoints/logs vào Drive hoặc sao lưu có kiểm tra; phiên Colab có thể ngắt.
Không tự resume/retry hoặc đổi batch/precision/length khi chưa nằm trong runner được duyệt.''')
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), colab=dict(name='LED_LARGE_F_COLAB_V1.ipynb')), cells=cells)


def main():
    assert not (DEST/'SHA256_MANIFEST.json').exists(), 'Preserve frozen handoff; select a new version instead'
    assert not (DEST.parent/'LED_LARGE_F_COLAB_V1.zip').exists(), 'Preserve delivered archive'
    DEST.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    def cp(rel):
        src = ROOT / rel
        assert src.is_file(), rel
        dest = DEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        assert sha(src) == sha(dest), 'Copy differs: ' + rel
        source_hashes[rel] = sha(src)

    cfg = read(ROOT/'configs/stage_i_led_large_proposal_v1.json')
    hm = read(ROOT/cfg['stage_h_manifest'])
    cp('configs/stage_i_led_large_proposal_v1.json')
    cp('configs/stage_i_d059_expanded_lora_v2.json')
    cp(cfg['stage_h_manifest'])
    cp('outputs/stage_h/led_dataset_v1/validation_report.json')
    cp('configs/led_source_template_d032.json')
    cp('assets/tokenizers/led-base-16384.manifest.json')
    for p in (ROOT/cfg['source_tokenizer']).iterdir():
        if p.is_file():
            cp(p.relative_to(ROOT).as_posix())
    datasets = {name: info for name, info in hm['datasets'].items() if name.endswith('_filtered')}
    photos, artifacts, index = set(), set(), []
    for name, info in datasets.items():
        cp(info['path']); cp(info['stage_g_jsonl'])
        rows = [json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
        for row in rows:
            artifacts.add(row['target_artifact'])
            assert sha(ROOT/row['target_artifact']) == row['target_artifact_sha256']
            photos.update(row['selected_photo_ids'])
            index.append({k: row[k] for k in ('business_id', 'split', 'condition', 'source_sha256',
                         'selected_review_ids', 'selected_photo_ids', 'target_artifact', 'target_artifact_sha256')})
    assert len(index) == 2200 and len(artifacts) == 2200
    print(f'F-only data selected: {len(index)} pairs, {len(photos)} original photos, {len(artifacts)} Teacher artifacts', flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(cp, sorted(artifacts)))
        rels = ['data/processed/yelp_2300_v1/photos/'+pid+'.jpg' for pid in sorted(photos)]
        for i, _ in enumerate(pool.map(cp, rels)):
            if i % 1000 == 0:
                print(f'Copied and hashed photos {i+1}/{len(photos)}', flush=True)
    # Original reference protocol, consent provenance, exclusions and source mappings.
    refdir = 'outputs/stage_i/d053_final_references_v1/'
    for name in ('FINAL_RESEARCHER_APPROVED_REFERENCES.json', 'SCORING_POPULATIONS_AND_INTERPRETATION.json',
                 'REFERENCE_DISTRIBUTIONS_BY_BUSINESS.json', 'RESEARCHER_APPROVAL_RECORD.json',
                 'RESEARCHER_APPROVAL_MESSAGE.txt', 'PRESERVED_TWO_UNCERTAIN_EXCLUSIONS.json',
                 'CAPTION_AND_OMISSION_RESOLUTION_AUDIT.json', 'COMPLETE_FINAL_REFERENCE_VALIDATION.json',
                 'D053_FINAL_REFERENCE_READINESS.json', 'D053_FINAL_REFERENCE_ACCEPTANCE_REPORT.md'):
        cp(refdir+name)
    base = 'outputs/stage_i/paired_lora_pilot_preparation_v1/'
    for name in ('evaluation_inputs.jsonl', 'expected_generation_config.json', 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'):
        cp(base+name)
    for p in (ROOT/PREP).rglob('*'):
        if p.is_file() and 'documentation_before' not in p.parts and 'untrained_CPU_serialization_probe' not in p.parts:
            cp(p.relative_to(ROOT).as_posix())
    # Benchmarks are existing outputs, never generated by the transfer builder.
    benchmarks = [
        'outputs/stage_i/cloud_return_20261015_v1/stage_i_chatgpt_review_bundle/validation/filtered/validation_predictions.jsonl',
        'outputs/stage_i/d058_full_filtered_B_v1/evaluate/epoch5/predictions.json',
        'outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch3/predictions.json',
        'outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch5/predictions.json']
    for rel in benchmarks:
        cp(rel)
    # Data/collation/init/evaluation references; BASE training guards are NOT rewritten or executable as large.
    code = ['scripts/stage_hi_led_data.py', 'scripts/stage_i_paired_pilot.py',
            'scripts/stage_i_d053_evaluation.py', 'scripts/stage_i_d059_v2.py',
            'scripts/run_stage_i_d059_v2.py', 'scripts/stage_i_d058.py',
            'scripts/run_stage_i_d050_feasibility_v1.py', 'scripts/run_stage_i_paired_lora_pilot_v1.py',
            'scripts/prepare_stage_i_led_large_v1.py', 'scripts/finalize_stage_i_led_large_offline_v1.py',
            'scripts/verify_led_large_colab_bundle_v1.py', 'scripts/download_led_large_pinned_v1.py',
            'scripts/build_led_large_filtered_colab_bundle_v1.py',
            'tests/test_led_large_colab_transfer_v1.py', 'tests/test_stage_i_led_large_preparation_v1.py',
            'thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md']
    for rel in code:
        cp(rel)
    for rel in ('DECISIONS.md', 'thesis_sdd_docs/THESIS_PROGRESS.md', 'thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md'):
        cp(rel)
    write(DEST/'TRANSFER_DATASETS.json', dict(datasets=datasets, condition='filtered',
          upstream_manifest=cfg['stage_h_manifest'], upstream_manifest_sha256=sha(ROOT/cfg['stage_h_manifest']),
          subset_is_transport_only=True, frozen_sources_modified=False))
    write(DEST/'TRAIN_VALIDATION_EVIDENCE_INDEX.jsonl', '\n'.join(json.dumps(x, ensure_ascii=False) for x in index))
    write(DEST/'ORIGINAL_PHOTO_SHA256.json', {pid: source_hashes['data/processed/yelp_2300_v1/photos/'+pid+'.jpg'] for pid in sorted(photos)})
    contract = dict(package='LED_LARGE_F_COLAB_V1', created_utc=datetime.now(timezone.utc).isoformat(),
        request_scope='Filtered-only preparation; RO/U only after researcher-confirmed F quality and separate approval',
        condition='filtered', train_businesses=2000, validation_businesses=200,
        training_validation_pairs=2200, selected_original_photos=len(photos),
        original_teacher_artifacts=2200, teacher_artifact_note='Preserved whole business files contain historical RO/U/F text; only F packs/JSONLs are execution data',
        GPU_execution_approved=False, training_authorized=False, RO_U_training_authorized=False,
        held_out_content_included=False, teacher_API_credentials_included=False,
        selected_hardware_candidate='Colab T4; memory/throughput UNTESTED; 24GB is not a measured minimum',
        minimum_measured_headroom=.15, pretrained_weights_included=False,
        pretrained_download_script='scripts/download_led_large_pinned_v1.py',
        training_runner_status='LARGE_RUNNER_ADAPTATION_REVIEW_AND_EXPLICIT_GPU_APPROVAL_REQUIRED',
        historical_training_code='Engineering reference only; hardcoded LED-base guards prohibit direct execution as large',
        D051_strict_replay='FAIL_UNCHANGED', Unlikelihood='CLOSED_NUMERICAL_FEASIBILITY_FAILURE_NOT_QUALITY_NO_GO')
    write(DEST/'TRANSFER_CONTRACT.json', contract)
    write(DEST/'SOURCE_COPY_SHA256.json', source_hashes)
    write(DEST/'requirements/led_large_colab_core.txt', '\n'.join(k+'=='+v for k,v in cfg['runtime_pins'].items() if k != 'torch'))
    write(DEST/'LED_LARGE_F_COLAB_V1.ipynb', notebook())
    write(DEST/'README_FIRST.md', '''# LED-large: Filtered-only Colab data handoff

Gói dùng để chuẩn bị và kiểm tra LED-large F trước; KHÔNG tự chạy train.

## Có trong ZIP
- Đúng 2.000 train + 200 validation, F-only: 2 JSONL nguyên bản (nguồn, Teacher target,
  REV/IMG_CAP/PHOTO evidence) + 2 NPZ token IDs đóng băng, không truncate/retokenize để train.
- Toàn bộ ảnh gốc được chọn cho F, 2.200 hồ sơ Teacher nguyên bản và evidence index.
  Hồ sơ Teacher được giữ nguyên nên còn chứa cả nội dung RO/U lịch sử; không có pack/JSONL RO/U để train.
- Tokenizer đóng băng; config/revision chính xác của LED-large; inventory 228 modules,
  4.915.200 LoRA parameters; seed, thứ tự 5 epoch, 3 train stress cases.
- 24 development cases (20 random + 4 targeted), D-053 references đã duyệt, benchmark cũ,
  evaluation protocol, hướng dẫn cho Codex, notebook và kiểm tra portable.
- Không có held-out test, API key, model checkpoint đã train, hoặc UL training tiếp.

## Cách dùng
1. Upload ZIP và `.zip.sha256` lên MyDrive. Mở LED_LARGE_F_COLAB_V1.ipynb trong Colab.
2. Chạy lần lượt các cell setup/offline. Notebook tạo Python 3.11 venv riêng, không thay Torch kernel.
3. Chạy full preflight `--tokenize`, 23 tests; lưu báo cáo và pip freeze. Nếu lỗi thì dừng.
4. Có thể bật cell tải checkpoint công khai ~1,84GB (chỉ download/hash, không GPU).
5. Đưa CODEX_HANDOFF.md cho Codex để hoàn thiện/pin runner LED-large từ mã tham khảo D-059.
   Trình config/runner/preflight hashes và xin duyệt GPU trước gate hoặc training.

Đây là gói dữ liệu và chuẩn bị Colab đầy đủ cho F, CHƯA phải training runner LED-large
được duyệt. Đổi mỗi model_id trong runner LED-base là sai: có kiểm tra 114 modules,
tensor hashes/backbone/base shape và authorization cũ. Không bỏ qua các kiểm tra này.

T4 là lựa chọn feasibility đầu tiên; chưa có đo VRAM. Giữ batch1, accum8, FP32 parameter
storage + FP16 AMP, checkpointing, source5000/padding5120 và labels768, beam4.
Chỉ tiếp tục sau gate PASS và >=15% measured memory headroom. Không tự giảm length,
đổi precision, quantize, hoặc chuyển GPU để vượt lỗi. Colab có thể ngắt phiên: runner
cuối cùng phải có kế hoạch checkpoint/Drive và resume được duyệt, không bỏ qua failure.

## Tính toàn vẹn và quyền thực thi
SHA256_MANIFEST.json kiểm tra file trong gói. SOURCE_COPY_SHA256.json ghi hash tại máy gốc.
TRANSFER_DATASETS.json chỉ chọn 2 F datasets từ manifest Stage H nguyên bản, không đổi artifact.
Các manifest lịch sử có thể tham chiếu RO/U/test metadata hoặc đường dẫn gốc KHÔNG đi kèm;
đây là provenance, không phải dependency cho portable preflight. Dùng verifier mới trong notebook.
Không chạy lại các script chuẩn bị cũ để ghi đè evidence; không dùng lịch sử approval cho LED-large.
Chưa duyệt GPU/F/RO/U. Nếu F tốt, researcher duyệt riêng matched RO/U cùng protocol/epoch.
Giữ LED-base RO/U/F làm baseline chính đến khi có đủ matched LED-large family.
''')
    write(DEST/'CODEX_HANDOFF.md', '''# Task for Codex in Colab — OFFLINE preparation first

Researcher asked to transfer ONLY F first. The ZIP is a complete F train/validation data
handoff, not an instruction to execute GPU training. Read TRANSFER_CONTRACT.json and the
LED-large proposal. Preserve all scientific data/reference bytes and historical evidence.

1. Run portable verifier --tokenize and both included test files from extracted root.
   Verify source copies, dataset counts/IDs, 9+1 zero-F behavior, exact labels and all photos.
   Full NPZ/tokenizer equality is required; no source shortening or dataset regeneration.
2. Retain pinned official allenai/led-large-16384 revision
   cd59d11c3528415b7dda4dfc95cc8f138aceda2e. Amazon is assessed only, not a fallback.
   Download via supplied script and verify official binary SHA256 before CPU tensor verification.
   No model weights are present in this ZIP; no pre-trained LED-base adapter is an initialization.
3. Prepare ONE versioned LED-large F runner using D-059 engineering references. The historical
   base runners are reference code only, not a turnkey dependency chain. Do not directly invoke
   their historical authorization/preflight or replace 114 with 228 and weaken checks.
   Use actual module names/shapes in META_LORA_INVENTORY.json; PEFT all-linear exact228 /
   456 adapter tensors /4915200 parameters. Exclude lm_head/embeddings. Check actual CPU
   backbone tensor shapes/hash bindings and explicit seed-by-name A/zeroB initialization.
4. Proposal settings: full F2000/200, five epochs/1250 updates, warmup63, AdamW2e-4/wd.01,
   batch1/accum8, FP32 storage +FP16AMP, non-reentrant gradient checkpointing, seed42,
   frozen existing tokenizer and EPOCH_ORDERS.json. Keep every epoch checkpoint; no best-loss
   reload/early stopping. Evaluate ONLY epochs3/5,24 fixed inputs each, beam4/max_length768.
   Actual effective GenerationConfig must be compared to frozen expected config, excluding
   metadata-only keys; do not inherit a different hub generation policy.
5. Propose T4 feasibility first. Gate max3updates/24microbatches, train stress examples
   at padded5120x768, actual masks/grads/AMP/frozen-backbone and restore checks,
   >=15% measured total-memory headroom; max2 TRAIN-input beam4 memory traces. No quality
   generation during gate. Separate actual allocated/reserved/total-device memory. Use measured
   timing to assess F+review48h and proposed familyfreeze2026-10-26T00:00:00Z; actual deadline
   and defense date still need confirmation. Do not reset historical D059/D060 clocks.
6. Prepare checkpoint persistence/resume plan for Colab disconnects before approval. A full
   restore must preserve optimizer/scheduler/scaler/RNG/sampler position and unchanged budget;
   no silent restart or extra optimization. No resume implementation has been accepted yet.
7. Return final runner/config/preflight hashes and bounded authorization scope BEFORE GPU.
   This package does NOT authorize forward/backward/generate/train. No RO/U/test/Teacher calls.

Quality: approved D-053 Core/supporting/excluded distinctions; report random20 and targeted4
separately. Teacher targets are not independent factual ground truth. AI labels provisional.
GO requires researcher-confirmed Core coverage, opinion synthesis, grounded visuals and usable
complete summaries without severe regressions, not loss/repetition/header improvements alone.
Compare saved historicalF/D059e3 as practical non-causal benchmarks. No automatic RO/U.
Preserve D051 strict replay FAIL, D058-D060 and UL numerical failure; no UL patch/retry.
''')
    write(DEST/'COLAB_SETUP_SOURCES.md', '''# Setup sources
uv managed Python and isolated environments:
https://docs.astral.sh/uv/concepts/python-versions/
https://docs.astral.sh/uv/guides/integration/jupyter/
Model provenance is pinned in outputs/stage_i/led_large_preparation_v1/REMOTE_PROVENANCE.json.
Bootstrap uv itself is not a scientific runtime pin. Save its version and complete pip freeze.
The seven ML runtime pins are preserved from the proposal; transitive wheels/CUDA driver
in Colab are not yet executed/validated. Stop on install/import incompatibility, no version fallback.
''')
    # Reject credentials in allowlisted small text payloads without printing any matched value.
    for p in DEST.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.py', '.md', '.txt', '.ipynb'):
            assert not re.search(rb'sk-[A-Za-z0-9]{32,}', p.read_bytes()), 'Credential-like string; stop packaging'
    paths = [p for p in sorted(DEST.rglob('*')) if p.is_file()]
    with ThreadPoolExecutor(max_workers=12) as pool:
        manifest = dict(zip((p.relative_to(DEST).as_posix() for p in paths), pool.map(sha, paths)))
    write(DEST/'SHA256_MANIFEST.json', dict(files=manifest, payload_files=len(manifest)))
    archive = DEST.parent / 'LED_LARGE_F_COLAB_V1.zip'
    assert not archive.exists(), 'Preserve old archive'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(DEST).as_posix())
    digest = sha(archive)
    write(Path(str(archive)+'.sha256'), digest+'  '+archive.name)
    write(DEST.parent/'LED_LARGE_F_COLAB_V1_RECEIPT.json', dict(archive=archive.name,
        sha256=digest, bytes=archive.stat().st_size, payload_files=len(manifest),
        train=2000, validation=200, condition='filtered', original_photos=len(photos),
        copied_original_files=len(source_hashes), GPU_execution_approved=False))
    shutil.copy2(DEST/'LED_LARGE_F_COLAB_V1.ipynb', DEST.parent/'LED_LARGE_F_COLAB_V1.ipynb')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest,
                         files=len(manifest), photos=len(photos)), indent=2), flush=True)


if __name__ == '__main__':
    main()
