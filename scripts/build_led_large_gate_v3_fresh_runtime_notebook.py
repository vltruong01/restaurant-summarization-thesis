"""Notebook transport fix: bootstrap a fresh/reset Colab before the unchanged v3 gate."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cloud_handoff'
NAME='LED_LARGE_GRADIENT_GATE_V3_FRESH_RUNTIME.ipynb'

BOOTSTRAP='''from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,stat,os,subprocess,sys
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
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

if not ROOT.exists():
    archive=DRIVE/'LED-Large_multimodal_UPLOAD_V1.zip'
    assert archive.is_file(), f'Upload ZIP goc vao {archive}; khong can giai nen'
    assert checksum(archive)=='bec6ac43c99cded0aaf1081a47302b0a030b442dcd9d04ec404ad298e0b4bdae', 'Original upload ZIP hash mismatch'
    local=Path('/content/led_large_gate_v3_source')
    assert not local.exists(), 'Existing partial extraction: inspect; no overwrite'
    safe_extract(archive,local)
    package=local/'LED-Large_multimodal'
    manifest=json.loads((package/'DELIVERY_SHA256_MANIFEST.json').read_text(encoding='utf-8'))['files']
    for name,digest in manifest.items():
        assert PurePosixPath(name).name==name and chr(92) not in name
        assert checksum(package/name)==digest, 'Delivery checksum mismatch: '+name
    data=package/'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
    assert checksum(data)=='91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f', 'Frozen V2 ZIP mismatch'
    safe_extract(data,ROOT)
    print('Fresh/reset runtime: original frozen workspace restored.')
else:
    print('Workspace exists; verify it, preserving all files.')
assert (ROOT/'configs/stage_i_led_large_colab_v2.json').is_file(), 'Partial/unexpected workspace; STOP'
assert checksum(ROOT/'configs/stage_i_led_large_colab_v2.json')=='f6dd97c282657a386de8dbfae225b58b720a909f7b1ee98041d348f83d5ad9ac', 'Frozen config changed'
print('Workspace ready:',ROOT)
'''

SETUP='''# Restore the same isolated Python 3.11 environment; no model execution.
venv=ROOT/'.venv'
if not PYTHON.is_file():
    assert not venv.exists(), 'Partial venv exists: STOP and inspect; do not overwrite'
    subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.11','--seed',str(venv)],check=True)
    assert PYTHON.is_file(), 'Python 3.11 venv creation failed'
    subprocess.run([str(PYTHON),'-m','pip','install','torch==2.1.0+cu118','--index-url','https://download.pytorch.org/whl/cu118'],check=True)
    subprocess.run([str(PYTHON),'-m','pip','install','-r',str(ROOT/'requirements/led_large_colab_core.txt')],check=True)
else:
    print('Existing venv retained; full runtime pin checks follow. No reinstallation.')
subprocess.run([str(PYTHON),'-m','pip','check'],check=True)
print('Python executable:',PYTHON)
'''


def build():
    nb=json.loads((OUT/'LED_LARGE_GRADIENT_GATE_V3.ipynb').read_text(encoding='utf-8'))
    old=[c for c in nb['cells'] if c['cell_type']=='code']
    def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
    def code(s):
        compile(s,NAME,'exec')
        return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
    install=''.join(old[1]['source'])
    install=install[install.index("patch=DRIVE/"):]
    install=install.replace("assert not (ROOT/name).exists(), 'Preserve existing files; patch already installed or conflict: '+name",
        "assert not (ROOT/name).exists() or checksum(ROOT/name)==digest, 'Existing patch file differs; STOP: '+name")
    install=install.replace("with dest.open('xb') as f:f.write(z.read(name))",
        "if not dest.exists():\n            with dest.open('xb') as f:f.write(z.read(name))")
    nb['cells']=[md('''# LED-large gradient gate v3 — fresh or existing Colab runtime
This notebook replaces the old v3 notebook's assumption that the v2 venv still exists.
Keep TWO existing ZIP files unchanged in MyDrive/LED-Large_multimodal:
LED-Large_multimodal_UPLOAD_V1.zip and LED_LARGE_GRADIENT_GATE_V3_PATCH.zip.
No manual extraction. Restore data -> Python 3.11 -> identical v3 patch -> full preflight
-> pinned public model download/verification -> NEW approval -> one diagnostic gate.
No old v2 gate is invoked; no main training. Existing Drive failures/logs remain untouched.
GPU execution remains separately gated; this notebook does not create approval or reset clocks.
'''),old[0],md('## 1. Restore original workspace automatically if runtime was reset'),code(BOOTSTRAP),
      md('## 2. Create the pinned Python environment only when absent'),code(SETUP),
      md('## 3. Install/verify the exact v3 patch without overwriting files'),code(install),
      md('## 4. CPU/meta tests and inherited full integrity preflight'),old[2],
      md('## 5. Verify/download the exact public backbone — no model load or GPU execution'),
      code("subprocess.run([str(PYTHON),'scripts/download_led_large_pinned_v1.py','--download'],cwd=ROOT,check=True)\n"),
      md('''## 6. Separate GPU authorization boundary
The unchanged v3 gate requires LED_LARGE_GRADIENT_GATE_V3_APPROVAL.json in the Drive folder.
Do not use the historical v2 approval. This notebook does not grant GPU approval.
Original deadline stays2026-10-21T17:11:48+00:00. No main training follows the gate.
'''),old[3],md('## 7. Export evidence after PASS or FAIL; do not rerun the GPU cell'),old[4]]
    nb['metadata']['colab']['name']=NAME
    with (OUT/NAME).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(nb,f,indent=2,ensure_ascii=False);f.write('\n')
    print(OUT/NAME)


if __name__=='__main__':build()
