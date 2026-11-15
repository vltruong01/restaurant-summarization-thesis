"""Create a transport-only notebook that opens the unchanged consolidated ZIP in Colab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cloud_handoff'
DEST = BASE / 'LED_LARGE_MULTIMODAL_DIRECT_ZIP.ipynb'

BOOTSTRAP = '''from pathlib import Path, PurePosixPath
import hashlib, zipfile, stat, json

# Upload ONLY the original consolidated ZIP to this existing Drive folder.
DRIVE_DIR = Path('/content/drive/MyDrive/LED-Large_multimodal')
UPLOAD_ZIP = DRIVE_DIR / 'LED-Large_multimodal_UPLOAD_V1.zip'
EXPECTED_UPLOAD_SHA256 = 'bec6ac43c99cded0aaf1081a47302b0a030b442dcd9d04ec404ad298e0b4bdae'
EXPECTED_ZIP_SHA256 = '91afd457d467dd41cf27cc402493139139d1a5f417183bb1f84031e0d2f9cf7f'
EXPECTED_APPROVAL_SHA256 = 'ca3b06f9647f9cfa1888ee3fd29ba11cc1c9b91db08cdca33a234c3535491ef2'
LOCAL_UPLOAD = Path('/content/led_large_direct_upload_v1')
ROOT = Path('/content/led_large_f_multimodal_v2')
RUN_OUT = DRIVE_DIR / 'runs/F_run_v2'
assert UPLOAD_ZIP.is_file(), f'Upload {UPLOAD_ZIP.name} vao {DRIVE_DIR}; khong can giai nen'
assert not LOCAL_UPLOAD.exists() and not ROOT.exists(), 'Existing local extraction: STOP and inspect; no overwrite or silent restart'

def file_sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def safe_extract_zip(archive, destination):
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            assert info.filename not in seen, 'Duplicate ZIP member; STOP'
            seen.add(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in info.filename, 'Unsafe ZIP path; STOP'
            assert chr(92) not in info.filename and not stat.S_ISLNK(info.external_attr >> 16), 'Unsafe ZIP member; STOP'
        z.extractall(destination)

assert file_sha256(UPLOAD_ZIP) == EXPECTED_UPLOAD_SHA256, 'Consolidated upload ZIP hash mismatch; STOP'
safe_extract_zip(UPLOAD_ZIP, LOCAL_UPLOAD)
UPLOAD_FILES = LOCAL_UPLOAD / 'LED-Large_multimodal'
manifest = json.loads((UPLOAD_FILES / 'DELIVERY_SHA256_MANIFEST.json').read_text(encoding='utf-8'))
for name, digest in manifest['files'].items():
    assert PurePosixPath(name).name == name and chr(92) not in name, 'Unsafe manifest path'
    assert file_sha256(UPLOAD_FILES / name) == digest, 'Delivery member hash mismatch: ' + name
archive = UPLOAD_FILES / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip'
receipt = UPLOAD_FILES / 'LED_LARGE_F_FULL_TRAIN_COLAB_V2.zip.sha256'
APPROVAL = UPLOAD_FILES / 'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'
parts = receipt.read_text(encoding='utf-8').split()
assert len(parts) == 2 and parts[0] == EXPECTED_ZIP_SHA256 and parts[1] == archive.name, 'Wrong V2 checksum receipt; STOP'
assert file_sha256(archive) == EXPECTED_ZIP_SHA256, 'Frozen payload ZIP hash mismatch; STOP'
assert file_sha256(APPROVAL) == EXPECTED_APPROVAL_SHA256, 'Researcher approval bytes changed; STOP'
approval = json.loads(APPROVAL.read_text(encoding='utf-8'))
assert approval.get('GPU_execution_approved') is True, 'Researcher approval missing'
safe_extract_zip(archive, ROOT)
print('Both ZIPs and approval verified; extracted automatically in Colab:', ROOT)
print('Persistent checkpoints and logs:', RUN_OUT)
print('Original approval deadline (not reset):', approval['execution_deadline_utc'])
'''


def build():
    nb = json.loads((BASE / 'LED-Large_multimodal/LED_LARGE_MULTIMODAL_F_FULL_TRAIN.ipynb').read_text(encoding='utf-8'))
    nb['metadata']['colab']['name'] = DEST.name
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    cells[1]['source'] = BOOTSTRAP.splitlines(True)
    s = ''.join(cells[5]['source'])
    s = s.replace("APPROVAL=DRIVE_DIR/'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'", "APPROVAL=UPLOAD_FILES/'LED_LARGE_F_V2_RESEARCHER_APPROVAL.json'")
    cells[5]['source'] = s.splitlines(True)
    nb['cells'][0]['source'] = '''# LED-large F — run directly from the consolidated ZIP
Upload LED-Large_multimodal_UPLOAD_V1.zip (unchanged) into MyDrive/LED-Large_multimodal.
Open THIS notebook in Colab; select T4 and run cells in order. No manual unzip, checksum
upload or approval-file upload is needed. The notebook verifies the published outer ZIP
digest, then automatically extracts the upload files and frozen data/code ZIP into /content.
Existing GPU authorization, runner/config bindings, scope and deadline remain unchanged.
Training checkpoints/logs persist in MyDrive/LED-Large_multimodal/runs/F_run_v2.
If a previous GPU run has already started, STOP before starting another: no automatic restart.
'''.splitlines(True)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), DEST.name, 'exec')
            c['execution_count'] = None; c['outputs'] = []
        elif ''.join(c['source']).startswith('## 4.'):
            c['source'] = '''## 4. Existing approval automatically extracted from the verified ZIP
The included conversation approval is used without editing or re-dating it.
This cell checks unchanged execution hashes and the original deadline before GPU work.
No separate approval upload or manual extraction is required.
If any prior gate/train run started, preserve its logs and stop before creating a duplicate.
'''.splitlines(True)
    with DEST.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False); f.write('\n')
    print(DEST)


if __name__ == '__main__':
    build()
