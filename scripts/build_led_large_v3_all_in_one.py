"""Transport-only consolidation: one uploaded ZIP plus one notebook, unchanged v3 science."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cloud_handoff'
DATA='LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
PATCH='LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'
ARCHIVE='LED_LARGE_V3_ALL_IN_ONE.zip'
NOTEBOOK='LED_LARGE_V3_ALL_IN_ONE.ipynb'
DATA_SHA='91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f'
PATCH_SHA='66e9488571eff7ba295b9a093256316fbdea110a4cc8be842107fa6f0aa24de1'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,obj):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,indent=2,ensure_ascii=False));f.write('\n')


BOOTSTRAP='''from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,stat,os,subprocess,sys
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
BUNDLE=DRIVE/'LED_LARGE_V3_ALL_IN_ONE.zip'
BUNDLE_FILES=Path('/content/led_large_v3_all_in_one_files')
ROOT=Path('/content/led_large_f_multimodal_v2')
PYTHON=ROOT/'.venv/bin/python'

def checksum(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_extract(archive,dest):
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            assert info.filename not in seen and not p.is_absolute() and '..' not in p.parts and ':' not in info.filename
            assert chr(92) not in info.filename and not stat.S_ISLNK(info.external_attr >> 16)
            seen.add(info.filename)
        z.extractall(dest)

assert BUNDLE.is_file(), f'Chi can upload LED_LARGE_V3_ALL_IN_ONE.zip vao {DRIVE}; khong can giai nen'
assert checksum(BUNDLE)=='__BUNDLE_SHA__', 'All-in-one ZIP hash mismatch; STOP'
with zipfile.ZipFile(BUNDLE) as z:
    manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))['files']
    assert set(z.namelist())==set(manifest)|{'BUNDLE_MANIFEST.json'}
    for name in manifest:
        assert PurePosixPath(name).name==name and chr(92) not in name, 'Unsafe bundle path'
if not BUNDLE_FILES.exists():
    safe_extract(BUNDLE,BUNDLE_FILES)
for name,digest in manifest.items():
    assert (BUNDLE_FILES/name).is_file() and checksum(BUNDLE_FILES/name)==digest, 'Missing/changed local bundle member: '+name
assert checksum(BUNDLE_FILES/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip')=='91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f'
assert checksum(BUNDLE_FILES/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip')=='66e9488571eff7ba295b9a093256316fbdea110a4cc8be842107fa6f0aa24de1'
if not ROOT.exists():
    safe_extract(BUNDLE_FILES/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip',ROOT)
    print('Fresh runtime: original frozen data/code restored automatically.')
else:
    print('Existing workspace retained; verification follows.')
assert (ROOT/'configs/stage_i_led_large_colab_v2.json').is_file(), 'Partial workspace; STOP and inspect'
assert checksum(ROOT/'configs/stage_i_led_large_colab_v2.json')=='f6dd97c282657a386de8dbfae225b58b720a909f7b1ee98041d348f83d5ad9ac', 'Frozen config changed'
print('Single ZIP verified; workspace:',ROOT)
'''


def build():
    assert sha(OUT/DATA)==DATA_SHA and sha(OUT/PATCH)==PATCH_SHA
    guide='''# LED-large v3 — ONE ZIP upload

Upload LED_LARGE_V3_ALL_IN_ONE.zip and open LED_LARGE_V3_ALL_IN_ONE.ipynb in Colab.
Use MyDrive/LED-Large_multimodal. No manual extraction and no separate patch ZIP,
checksum receipt or data ZIP upload. The notebook verifies/extracts both internal
ZIPs itself, creates the pinned Python3.11 venv when absent, runs original preflight,
verifies/downloads public LED-large weights and stops at the separate GPU approval gate.

Original frozen F2000/200 data,228LoRA modules/4915200trainable parameters, model,
tokenizer/targets/references, precision and gate rules remain unchanged. The v3 repair
records missing-versus-nonfinite gradients; it is not a verified numerical cure.
The supplied approval template remains false. No new GPU approval is fabricated.
No main training or evaluation is available in the diagnostic notebook. Gate<=3updates/
24microbatches and<=2train-input memory traces only after explicit v3 hash-bound approval.
>=15% memory headroom; no fallback/replay/automatic retry. Preserve old failed runs.
Original deadline2026-10-21T17:11:48+00:00 remains unchanged.

Weights~1.84GB are fetched from the pinned public source, not loaded as a model during
setup. Checkpoint and GPU numerical compatibility remain unverified until the approved
diagnostic execution. No Teacher key, Codex config or test data is needed.
'''
    manifest={DATA:DATA_SHA,PATCH:PATCH_SHA,'README_FIRST.md':hashlib.sha256(guide.encode('utf-8')).hexdigest()}
    archive=OUT/ARCHIVE
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED,allowZip64=True) as z:
        z.write(OUT/DATA,DATA);z.write(OUT/PATCH,PATCH)
        z.writestr('README_FIRST.md',guide)
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(dict(files=manifest,GPU_execution_approved=False,science_changed=False),indent=2))
    digest=sha(archive)
    nb=json.loads((OUT/'LED_LARGE_GRADIENT_GATE_V3_FRESH_RUNTIME.ipynb').read_text(encoding='utf-8'))
    codes=[c for c in nb['cells'] if c['cell_type']=='code']
    codes[1]['source']=BOOTSTRAP.replace('__BUNDLE_SHA__',digest).splitlines(True)
    s=''.join(codes[3]['source'])
    assert "patch=DRIVE/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'" in s
    codes[3]['source']=s.replace("patch=DRIVE/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'", "patch=BUNDLE_FILES/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'").splitlines(True)
    nb['cells'][0]['source']='''# LED-large v3 — ONE ZIP, ONE notebook
Upload only LED_LARGE_V3_ALL_IN_ONE.zip to MyDrive/LED-Large_multimodal.
Open this notebook in Colab, select T4 and run preparation cells in order.
No manual unzip or separate patch/data/checksum upload. Both internal ZIPs are verified
and extracted automatically. Fresh and already prepared runtimes are supported.
This supersedes the earlier multi-ZIP upload instructions; old artifacts remain historical.
Same frozen data/science/v3 hashes/deadline. GPU approval remains a separate boundary;
this notebook does not create human authorization or run main training.
'''.splitlines(True)
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),NOTEBOOK,'exec')
        elif ''.join(c['source']).startswith('## 1.'):
            c['source']='## 1. Open the single ZIP and restore the original workspace automatically\n'.splitlines(True)
    nb['metadata']['colab']['name']=NOTEBOOK
    save(OUT/NOTEBOOK,nb)
    report_dir=ROOT/'outputs/stage_i/led_large_gradient_gate_v3/all_in_one_delivery_v1';report_dir.mkdir()
    save(report_dir/'DELIVERY_RECEIPT.json',dict(archive=ARCHIVE,bytes=archive.stat().st_size,sha256=digest,
             notebook=NOTEBOOK,notebook_sha256=sha(OUT/NOTEBOOK),v3_bindings_changed=False,approval=False,GPU_execution=False))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,notebook=NOTEBOOK),indent=2))


if __name__=='__main__':build()
