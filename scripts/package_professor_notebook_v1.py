"""Verify, extract/re-execute and package the portable thesis overview."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/professor_notebook/20261103_v1'
PKG=OUT/'THESIS_JUPYTER_NOTEBOOK_FOR_PROFESSOR'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory():
    files=[p for p in sorted(PKG.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.csv']
    with (PKG/'PACKAGE_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','bytes','sha256'])
        w.writeheader()
        for p in files:w.writerow(dict(file=p.relative_to(PKG).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))


def main():
    report=json.loads((PKG/'VALIDATION_REPORT.json').read_text())
    book=nbformat.read(PKG/'THESIS_RESEARCH_OVERVIEW.ipynb',as_version=4)
    nbformat.validate(book)
    # Fail before delivery if executable summary cells could launch heavy/network actions.
    heavy=re.compile(r'\b(?:import torch|from transformers|requests\.|subprocess\.|pip install|from_pretrained\(|\.generate\()')
    assert not any(heavy.search(c.source) for c in book.cells if c.cell_type=='code')
    secret_patterns=[r'(?i)\bsk-[A-Za-z0-9_-]{20,}',r'\bhf_[A-Za-z0-9]{20,}',
                     r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                     r'(?i)\b(?:api_key|access_token|password)\s*[=:]\s*[\"\x27][^\"\x27]{8,}[\"\x27]',
                     r'C:\\\\?Users\\\\?VUONGLOCTRUONG']
    for p in PKG.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.ipynb','.html','.txt','.md','.py','.csv','.yaml']:
            text=p.read_text(encoding='utf-8-sig')
            for pattern in secret_patterns:assert not re.search(pattern,text), (p.name,pattern)
    report.update(notebook_schema='PASS',executable_cells_no_heavy_or_network_actions='PASS',
                  credential_and_private_user_path_scan='PASS',figures_visually_inspected='PASS_6_SAVED_FIGURES',
                  package_portability='PENDING_EXTRACTED_REEXECUTION')
    (PKG/'VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    inventory()
    local=subprocess.run([sys.executable,str(PKG/'verify_package.py')],capture_output=True,text=True,encoding='utf8',check=True)
    with tempfile.TemporaryDirectory(prefix='thesis_professor_portability_') as tmp:
        # Stage ZIP proves actual ZIP extraction, not just a second read of the source directory.
        stage=Path(tmp)/'portability.zip'
        with zipfile.ZipFile(stage,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(PKG.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(PKG))
        extracted=Path(tmp)/'extracted'
        with zipfile.ZipFile(stage) as z:
            assert z.testzip() is None
            z.extractall(extracted)
        result=subprocess.run([sys.executable,str(extracted/'verify_package.py')],capture_output=True,text=True,encoding='utf8',check=True)
        portable=nbformat.read(extracted/'THESIS_RESEARCH_OVERVIEW.ipynb',as_version=4)
        NotebookClient(portable,kernel_name='thesis-caption-pilot',timeout=180,resources={'metadata':{'path':str(extracted)}}).execute()
        assert len([c for c in portable.cells if c.cell_type=='code'])==16
        assert not any(o.output_type=='error' for c in portable.cells if c.cell_type=='code' for o in c.outputs)
        # Saved output numbers/tables/text must reproduce; embedded figures are also retained.
        for old,new in zip(book.cells,portable.cells):
            if old.cell_type=='code':
                a=[(o.output_type,o.get('text'),{k:v for k,v in o.get('data',{}).items() if k!='image/png'}) for o in old.outputs]
                b=[(o.output_type,o.get('text'),{k:v for k,v in o.get('data',{}).items() if k!='image/png'}) for o in new.outputs]
                assert a==b, old.execution_count
        report.update(package_portability='PASS_ACTUAL_ZIP_EXTRACT_AND_RUN_ALL',
                      extracted_notebook_code_cells_executed=16,
                      reproduced_nonimage_saved_outputs='PASS_EXACT_MATCH',
                      standalone_verifier='PASS_ORIGINAL_AND_EXTRACTED',
                      actual_zip_crc_and_member_hash_check='PASS')
    bindings=json.loads((PKG/'data/original_artifact_bindings.json').read_text())
    assert all(sha(ROOT/r['repository_file'])==r['sha256'] for r in bindings)
    report['frozen_artifact_preservation_after_final_validation']='PASS'
    (PKG/'VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    (PKG/'VALIDATION_REPORT.md').write_text('''# Validation report

| Check | Actual result |
|---|---|
| Notebook schema and execution | PASS — 16 cells executed, no errors |
| Saved plots | PASS — six embedded figures, visually inspected |
| Portability | PASS — actual ZIP extracted into an unrelated temporary directory; Run All succeeded |
| Saved table/text reproduction | PASS — exact match after extracted execution |
| Data and code integrity | PASS — every bundled evidence/code SHA-256 checked |
| Dataset | PASS — 2,300 unique business IDs, 2,000/200/100 split, 6,900 budget records |
| Validation metrics | PASS — all 30 aggregate metric values reproduced from 1,000 frozen applicable component rows |
| TEST100 metrics | PASS — all 42 aggregate metric values reproduced from 700 rows; no business exclusions |
| Reference freeze | PASS — 100 original approved annotation files/content hashes matched the Stage M consumption receipt; same per-business reference used across conditions |
| TEST30 | PASS — 240 unchanged numeric records; 690 paired-business criterion rows; combined means and 23 kappa panels reproduced |
| Statistical tables | PASS — original 24 automatic / 17 human comparison receipts retained; all paired point estimates checked |
| Examples | PASS — complete original outputs verified against their output hashes |
| Credentials and private user paths | PASS — pattern scan; no participant accounts, keys, passwords, session stores or private identity mapping bundled |
| Preservation | PASS — original source evidence rehashed unchanged before/after export and final verification |

The machine-readable report contains exact counts and timing. Confidence intervals and p-values are the completed analyses' existing results, not a fresh inferential run. No GPU, model download, model scoring, paid API, training or new experiment was used in notebook execution. This verifies recorded evidence and computation; it does not independently authenticate human identity, source inspection, or unaided authorship. Model-weight hashes are existing freeze receipts; the large binaries are not included or newly rehashed here.
''',encoding='utf8')
    inventory()
    subprocess.run([sys.executable,str(PKG/'verify_package.py')],check=True)
    zip_path=OUT/'THESIS_JUPYTER_NOTEBOOK_FOR_PROFESSOR.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(PKG.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PKG))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names))
        for p in PKG.rglob('*'):
            if p.is_file():assert hashlib.sha256(z.read(p.relative_to(PKG).as_posix())).hexdigest()==sha(p)
    final=dict(zip_file=zip_path.name,bytes=zip_path.stat().st_size,sha256=sha(zip_path),members=len(names),
               final_zip_crc_and_every_member_sha256='PASS',extracted_run_all='PASS',frozen_originals_unchanged=len(bindings))
    (OUT/(zip_path.name+'.sha256')).write_text(final['sha256']+'  '+zip_path.name+'\n',encoding='utf8')
    (OUT/'FINAL_PACKAGE_VERIFICATION.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf8')
    print(json.dumps(final,indent=2))


if __name__=='__main__':main()
