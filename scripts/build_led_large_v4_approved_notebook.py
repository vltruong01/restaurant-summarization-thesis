"""Embed approved v4 overlay/approval; reuse the existing single data ZIP unchanged."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'outputs/cloud_handoff'
NAME='LED_LARGE_V4_ALL_IN_ONE_APPROVED.ipynb'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build():
    nb=json.loads((H/'LED_LARGE_V3_ALL_IN_ONE.ipynb').read_text(encoding='utf-8'))
    prep='outputs/stage_i/led_large_gradient_gate_v4_preparation'
    rels=['scripts/run_stage_i_led_large_gradient_gate_v4.py',
          'configs/stage_i_led_large_gradient_gate_v4.json',
          'tests/test_led_large_v4_amp_candidate.py',
          prep+'/PREFLIGHT_MANIFEST.json',prep+'/EXECUTION_BINDINGS.json',
          prep+'/GPU_APPROVAL_TEMPLATE_NOT_APPROVED.json']
    payload=dict(files={rel:base64.b64encode((ROOT/rel).read_bytes()).decode('ascii') for rel in rels},
                 hashes={rel:sha(ROOT/rel) for rel in rels})
    raw=json.dumps(payload,separators=(',',':')).encode('utf-8')
    encoded=base64.b64encode(zlib.compress(raw,9)).decode('ascii')
    patch='''# Versioned v4 overlay embedded here: no second uploaded ZIP.
import base64,zlib
raw=zlib.decompress(base64.b64decode(__ENCODED__))
assert hashlib.sha256(raw).hexdigest()==__HASH__, 'Embedded v4 overlay hash mismatch'
overlay=json.loads(raw)
decoded={name:base64.b64decode(data) for name,data in overlay['files'].items()}
assert set(decoded)==set(overlay['hashes'])
for name,data in decoded.items():
    rel=PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name and chr(92) not in name
    assert hashlib.sha256(data).hexdigest()==overlay['hashes'][name], 'v4 file hash mismatch: '+name
    dest=ROOT/name
    assert not dest.exists() or checksum(dest)==overlay['hashes'][name], 'Existing v4 file differs: '+name
inherited=json.loads(decoded['outputs/stage_i/led_large_gradient_gate_v4_preparation/PREFLIGHT_MANIFEST.json'])['files']
for name,digest in inherited.items():
    if name not in decoded:
        assert checksum(ROOT/name)==digest, 'Frozen dependency differs: '+name
for name,data in decoded.items():
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with dest.open('xb') as f:f.write(data)
print('Approved v4 files ready; old runners, data and failed runs preserved.')
'''.replace('__ENCODED__',repr(encoded)).replace('__HASH__',repr(hashlib.sha256(raw).hexdigest()))
    approval=json.loads((H/'LED_LARGE_GRADIENT_GATE_V4_APPROVAL.json').read_text(encoding='utf-8'))
    assert approval['GPU_execution_approved'] is True and approval['training_authorized'] is False
    record='''# Actual conversation approval, recorded before execution; no new self-approval.
APPROVAL=DRIVE/'LED_LARGE_GRADIENT_GATE_V4_APPROVAL.json'
approved_record=json.loads(__APPROVAL__)
if APPROVAL.exists():
    assert json.loads(APPROVAL.read_text(encoding='utf-8'))==approved_record, 'Existing v4 approval differs; STOP'
else:
    with APPROVAL.open('x',encoding='utf-8') as f:
        json.dump(approved_record,f,ensure_ascii=False,indent=2);f.write(chr(10))
subprocess.run([str(PYTHON),'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import run_stage_i_led_large_gradient_gate_v4 as m;m.authorize_patch(Path(sys.argv[1]),True);print('Actual researcher v4 approval PASS')",str(APPROVAL)],cwd=ROOT,check=True)
print('Original deadline unchanged:',approved_record['execution_deadline_utc'])
'''.replace('__APPROVAL__',repr(json.dumps(approval,ensure_ascii=False)))
    def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
    def code(s):
        compile(s,NAME,'exec')
        return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
    old=[c for c in nb['cells'] if c['cell_type']=='code']
    checks=''.join(old[4]['source']).replace('test_led_large_gradient_evidence_v3.py','test_led_large_v4_amp_candidate.py')
    checks=checks.replace('run_stage_i_led_large_gradient_gate_v3','run_stage_i_led_large_gradient_gate_v4')
    checks=checks.replace('outputs/stage_i/led_large_gradient_gate_v3/EXECUTION_BINDINGS.json',prep+'/EXECUTION_BINDINGS.json')
    gate=''.join(old[6]['source'])
    gate=gate.replace('LED_LARGE_GRADIENT_GATE_V3_APPROVAL.json','LED_LARGE_GRADIENT_GATE_V4_APPROVAL.json')
    gate=gate.replace('run_stage_i_led_large_gradient_gate_v3','run_stage_i_led_large_gradient_gate_v4')
    gate=gate.replace('F_gradient_gate_v3','F_gradient_gate_v4')
    export=''.join(old[7]['source']).replace('F_gradient_gate_v3','F_gradient_gate_v4').replace('LED_LARGE_GRADIENT_V3_EVIDENCE_','LED_LARGE_GRADIENT_V4_EVIDENCE_')
    nb['cells']=[md('''# LED-large v4 — approved initial AMP scale correction
Use the SAME LED_LARGE_V3_ALL_IN_ONE.zip in MyDrive/LED-Large_multimodal. Do not rename it
or upload another ZIP. This notebook embeds the small v4 overlay and actual approval.
Fresh and existing runtimes supported. Select T4; run cells in order.
Only numerical change: GradScaler init_scale1024 instead of65536. FP16,model,data,
optimizer,LoRA and all strict checks remain unchanged. This is a bounded gate, not
full training. Scope<=3updates/24microbatches and<=2originaltrain-input memory traces.
Original deadline2026-10-21T17:11:48+00:00 remains unchanged. No automatic retry.
Stop after evidence export on PASS or FAIL. Do not rerun v2/v3 gates or overwrite failures.
'''),old[0],md('## 1. Verify the existing single ZIP and restore workspace if needed'),old[1],
    md('## 2. Same pinned Python environment'),old[2],
    md('## 3. Restore/verify unchanged v3 dependencies without running its GPU gate'),old[3],
    md('## 4. Install embedded, hash-bound v4 files'),code(patch),
    md('## 5. Offline v4 tests and full inherited integrity preflight'),code(checks),
    md('## 6. Verify/download the one pinned official backbone; no model execution'),old[5],
    md('## 7. Record the actual researcher approval automatically'),code(record),
    md('## 8. Execute the approved gate ONCE; do not start main training'),code(gate),
    md('## 9. Export after PASS or FAIL; if the preceding cell failed, run ONLY this export cell'),code(export)]
    nb['metadata']['colab']['name']=NAME
    dest=H/NAME
    with dest.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(nb,f,indent=2,ensure_ascii=False);f.write('\n')
    receipt=dict(notebook=NAME,notebook_sha256=sha(dest),same_existing_zip=True,embedded_overlay_files=len(rels),
                 approved_scope='one bounded v4 gate only',training_authorized=False,GPU_execution=False)
    with (ROOT/'outputs/stage_i/led_large_gradient_gate_v4_authorization_v1/NOTEBOOK_RECEIPT.json').open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':build()
