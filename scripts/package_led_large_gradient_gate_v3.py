"""Small diagnostic overlay ZIP and Colab notebook; no automatic GPU authorization."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_gradient_gate_v3'
OUT=ROOT/'outputs/cloud_handoff'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p, value):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False));f.write('\n')


def build():
    report=json.loads((PREP/'OFFLINE_VALIDATION_REPORT.json').read_text())
    assert report['tests']==55 and report['status']=='PASS_OFFLINE_ONLY'
    b=json.loads((PREP/'EXECUTION_BINDINGS.json').read_text())
    save(PREP/'README_FIRST.md', '''# LED-large gradient gate v3 — diagnostic evidence repair only

The original gate cannot distinguish missing gradients from NaN/Inf. This version
adds read-only per-microbatch CE/scaler/gradient evidence, all456 tensor finite/NaN/Inf
counts after unscale, raw gradient snapshots on failure, memory/scaler/RNG evidence,
and completed-update accounting before the original failure gate terminates.
It does NOT fix an established numerical cause: the cause remains unconfirmed.

Original scientific v2 config, model/data/order, seed, LoRA, CE, FP16 precision,
default GradScaler, accumulation, optimizer/schedule and strict assertions stay the
same. Initialization zeroA is still permitted; no skip/relax/retry/precision fallback.
Original v2 failure, runner, authorization and data remain untouched. All inherited
original v2 preflight checks remain mandatory. No training or evaluation CLI is
available in this diagnostic-only runner. No quality GO/NO-GO claim is made.

Upload PATCH ZIP and the new diagnostic notebook into MyDrive/LED-Large_multimodal.
Use the existing Colab runtime/workspace after the v2 failure. The notebook checks
the patch SHA, verifies frozen baseline files, then installs only new versioned files.
No manual unzip. Never overwrite or delete the old failed run. If runtime was reset,
prepare the original direct-ZIP notebook through setup/preflight/model download only;
do not rerun its old feasibility cell.

The approval template remains false. A new hash-bound approval is needed before the
v3 GPU gate, preserving deadline2026-10-21T17:11:48+00:00. No clock reset. Authorized
proposal: one fresh gate<=3updates/24microbatches and<=2train-input memory traces,
>=15% headroom, no replay; no main training or predictions. Stop after evidence export,
whether PASS or FAIL. Export includes raw .pt gradients, not accepted model checkpoints.
Only load those tensors from this trusted run; offline evaluation of them is not training.

55 CPU/meta offline tests PASS, including13 actual synthetic-gradient tests and42
inherited tests, on the extracted original payload plus overlay. Full inherited v2
and new overlay preflight PASS. No LED forward/backward/GPU execution performed here.
All original scientific artifacts and D051 strict replay FAIL remain preserved.
''')
    save(PREP/'READINESS_AND_AUTHORIZATION.md',
         '# Gate-only v3: ready for separate approval\n\n'+json.dumps(b,indent=2)+
         '\n\nScope: one fresh diagnostic gate <=3 updates/24 microbatches, <=2 original train-input generation memory traces; >=15% measured memory headroom; no replay, main training, evaluation, Teacher calls or held-out access.\n'+
         'Keep original deadline2026-10-21T17:11:48+00:00. No execution approval is fabricated.\n'+
         'Read-only instrumentation changes timing but no intended math/settings/RNG state; exact GPU replay is not claimed. Numerical cause and actual T4 behavior remain unresolved pending explicitly approved execution.\n')
    files=[ROOT/p for p in ('scripts/run_stage_i_led_large_gradient_gate_v3.py','scripts/led_large_gradient_evidence_v3.py',
                           'configs/stage_i_led_large_gradient_gate_v3.json','tests/test_led_large_gradient_evidence_v3.py')]
    files += [p for p in PREP.iterdir() if p.is_file()]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    archive=OUT/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('PATCH_SHA256_MANIFEST.json',json.dumps({'files':manifest},indent=2))
    digest=sha(archive);save(Path(str(archive)+'.sha256'),digest+'  '+archive.name)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'<diagnostic_notebook>','exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('''# LED-large v3 — preserve numerical evidence before stopping
This is a diagnostic-only repair, not a proven numerical cure. Keep the existing
failed v2 run. Upload LED_LARGE_GRADIENT_GATE_V3_PATCH.zip to the existing Drive folder.
No manual unzip, scientific-setting change, main training or automatic retry.
GPU gate still requires NEW explicit approval; the provided template is false.
''')
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    md('## 1. Install new files only; CPU/file operations, no GPU execution')
    code('''from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,stat,os,subprocess
DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')
ROOT=Path('/content/led_large_f_multimodal_v2')
PYTHON=ROOT/'.venv/bin/python'
assert PYTHON.is_file(), 'Existing prepared v2 workspace required; do not rerun old gate'
patch=DRIVE/'LED_LARGE_GRADIENT_GATE_V3_PATCH.zip'
with patch.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest() == '__DIGEST__', 'Patch ZIP hash mismatch; STOP'
with zipfile.ZipFile(patch) as z:
    manifest=json.loads(z.read('PATCH_SHA256_MANIFEST.json'))['files']
    assert set(z.namelist())==set(manifest)|{'PATCH_SHA256_MANIFEST.json'}
    for name,digest in manifest.items():
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in name and chr(92) not in name
        assert not stat.S_ISLNK(z.getinfo(name).external_attr >> 16)
        assert not (ROOT/name).exists(), 'Preserve existing files; patch already installed or conflict: '+name
        assert hashlib.sha256(z.read(name)).hexdigest()==digest
    inherited=json.loads(z.read('outputs/stage_i/led_large_gradient_gate_v3/PREFLIGHT_MANIFEST.json'))['files']
    for name,digest in inherited.items():
        if name not in manifest:
            with (ROOT/name).open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==digest, 'Original v2 binding mismatch: '+name
    for name in manifest:
        dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(z.read(name))
print('New diagnostic files installed; original v2 files remain unchanged.')
'''.replace('__DIGEST__',digest))
    code('''OFFLINE_ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p','test_led_large_gradient_evidence_v3.py','-v'],cwd=ROOT,env=OFFLINE_ENV,check=True)
subprocess.run([str(PYTHON),'-c',"import sys;sys.path.insert(0,'scripts');import run_stage_i_led_large_gradient_gate_v3 as m;m.preflight_patch();print('Full preflight PASS')"],cwd=ROOT,env=OFFLINE_ENV,check=True)
print((ROOT/'outputs/stage_i/led_large_gradient_gate_v3/EXECUTION_BINDINGS.json').read_text())
''')
    md('''## 2. GPU approval boundary
New file: MyDrive/LED-Large_multimodal/LED_LARGE_GRADIENT_GATE_V3_APPROVAL.json.
Obtain explicit approval of the new bindings before running this cell. Never edit
the historical v2 approval. No automatic main training follows this gate.
''')
    code('''APPROVAL=DRIVE/'LED_LARGE_GRADIENT_GATE_V3_APPROVAL.json'
assert APPROVAL.is_file(), 'New hash-bound GPU diagnostic approval not yet supplied; STOP'
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import run_stage_i_led_large_gradient_gate_v3 as m;m.authorize_patch(Path(sys.argv[1]),True);print('New approval PASS')",str(APPROVAL)],cwd=ROOT,check=True)
RUN=DRIVE/'runs/F_gradient_gate_v3'
LOG=DRIVE/'runs/F_gradient_gate_v3_launcher.log'
assert not RUN.exists() and not LOG.exists(), 'Fresh diagnostic run required; no retries'
LOG.parent.mkdir(parents=True,exist_ok=True)
with LOG.open('x') as f:
    p=subprocess.Popen([str(PYTHON),'scripts/run_stage_i_led_large_gradient_gate_v3.py','feasibility','--execute','--approval',str(APPROVAL),'--output',str(RUN)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        print(line,end='');f.write(line);f.flush()
    status=p.wait()
assert status==0, 'Gate stopped. Preserve evidence; run ONLY the export cell below, not main training.'
print('Gate PASS. Stop and export for independent review; no automatic F training.')
''')
    md('## 3. Export evidence after PASS or FAIL — CPU only; do not restart the GPU cell')
    code('''from datetime import datetime,timezone
RUN=DRIVE/'runs/F_gradient_gate_v3'
LOG=DRIVE/'runs/F_gradient_gate_v3_launcher.log'
assert RUN.is_dir(), 'No diagnostic execution artifacts to export'
out=DRIVE/('LED_LARGE_GRADIENT_V3_EVIDENCE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl','.log','.pt','.txt'}:
            z.write(p,'run/'+p.relative_to(RUN).as_posix())
    if LOG.is_file():z.write(LOG,'launcher.log')
print('Send this ZIP for review:',out)
''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
             language_info=dict(name='python'),colab=dict(name='LED_LARGE_GRADIENT_GATE_V3.ipynb')),cells=cells)
    save(OUT/'LED_LARGE_GRADIENT_GATE_V3.ipynb',nb)
    save(PREP/'DELIVERY_RECEIPT.json',dict(archive=archive.name,archive_sha256=digest,bytes=archive.stat().st_size,
         notebook='LED_LARGE_GRADIENT_GATE_V3.ipynb',notebook_sha256=sha(OUT/'LED_LARGE_GRADIENT_GATE_V3.ipynb'),
         binding=b,approval=False,GPU_executed=False))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,bindings=b),indent=2))


if __name__=='__main__':build()
