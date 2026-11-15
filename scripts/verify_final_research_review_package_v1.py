"""Portable read-only review-package verifier, using only the standard library."""
from pathlib import Path
import argparse,hashlib,json

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf8'))
def verify(root):
 root=root.resolve();manifest=load(root/'PACKAGE_MANIFEST.json');entries=manifest['files']
 expected={r['file'] for r in entries}|{'PACKAGE_MANIFEST.json'}
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
 assert len(entries)==len(expected)-1 and expected==actual,'Missing, duplicate or unexpected files'
 for row in entries:
  path=(root/row['file']).resolve();assert path.is_relative_to(root)
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['file']
 inventory=load(root/'SOURCE_INVENTORY.json')
 for row in inventory['member_copies']:assert sha(root/row['package_file'])==row['sha256']
 assert inventory['official_P_source_sha256']=='e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5'
 checks=load(root/'VERIFIED_TABLES/VERIFICATION_RESULTS.json');assert checks['status']=='PASS'
 assert checks['P']['official_records']==90 and checks['P']['paired_tests']==18 and len(checks['P']['undefined_QWK_cells'])==2
 assert checks['M']['businesses']==100 and checks['N']['records']==240 and checks['O']['records']==90
 assert not checks['P']['external_ChatGPT_statistical_ZIP_verified']
 assert load(root/'VALIDATION_REPORT.json')['status']=='PASS_NUMERICAL_AND_PACKAGE_CHECKS_WITH_DISCLOSED_LIMITATIONS'
 required=['FINAL_RESULTS_ANALYSIS.md','RESEARCH_QUESTIONS_FINDINGS.md','QUALITY_EFFICIENCY_DISCUSSION.md','LIMITATIONS_AND_THREATS_TO_VALIDITY.md','FINAL_THESIS_DISCUSSION_DRAFT.md','EVIDENCE_TRACEABILITY.md']
 for file in required:assert (root/file).stat().st_size>1000
 assert not any(p.suffix.lower() in {'.sqlite','.sqlite3','.db','.safetensors','.pt','.jpg','.jpeg'} for p in root.rglob('*') if p.is_file())
 return dict(status='PASS',files_checked=len(entries),source_copies_checked=len(inventory['member_copies']),official_P='original recovered V1',no_model_calls=True,scope='Hashes and recorded numerical receipt; semantic correctness and participant identity are not independently authenticated')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path('.'));a=p.parse_args();print(json.dumps(verify(a.package),indent=2))
