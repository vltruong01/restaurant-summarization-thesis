"""Package existing audit evidence; never query or change study records."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path('outputs/stage_n/full240_rating_quality_audit_20261102_v1')

VERIFY_SCRIPT='''"""Offline byte/binding verification only; does not rerun semantic audit."""
import collections,hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((p/n).read_text(encoding='utf-8-sig'))
def h(b):return hashlib.sha256(b).hexdigest()
def digest(x):return h(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
manifest=read('PACKAGE_MANIFEST.json')
for x in manifest['members']:
 b=(p/x['file']).read_bytes()
 assert len(b)==x['bytes'] and h(b)==x['sha256'],x['file']
fs=read('ITEM_LEVEL_AUDIT_240.json');gs=read('GROUPED_ITEMS.json');ss=read('SOURCES.json')
assert len(fs)==240 and len(gs)==120 and len(ss)==30
assert len({x['alias'] for x in fs})==240
bygroup={g['group_id']:g for g in gs}
for rater in ['A01','A02']:assert {x['position'] for x in fs if x['rater']==rater}==set(range(1,121))
for f in fs:
 g=bygroup[f['group_id']];r=next(r for r in g['ratings'] if r['alias']==f['alias'])
 assert r['rating']==f['original_rating_payload'] and r['version']==f['original_version']
 assert digest(r['rating'])==f['original_payload_canonical_sha256']
 assert h(g['review_output'].encode())==f['review_output_sha256']
 assert (None if g['visual_output'] is None else h(g['visual_output'].encode()))==f['visual_output_sha256']
for receipt in read('SOURCE_INTEGRITY_RECEIPTS.json'):
 s=ss[receipt['business_id']];assert digest(s)==receipt['canonical_sha256']
 assert h(s['selected_review_block'].encode())==receipt['review_block_sha256']
for e in read('FOLLOWUP_EVIDENCE.json'):
 for photo in e['original_photos']:assert h((p/photo['file']).read_bytes())==photo['sha256']
summary=read('AUDIT_SUMMARY.json')
for rater in ['A01','A02']:
 assert dict(collections.Counter(f['assessment'] for f in fs if f['rater']==rater))==summary['classification_counts'][rater]
print(json.dumps({'result':'PASS','hash_verified_members':len(manifest['members']),'records':240,'output_groups':120,'businesses':30,'semantic_audit_rerun':False}))
'''


def main():
    readme='''# Stage N — Full240 rating-quality audit

Coordinator/researcher review material. Contains both participants' judgments;
do not send this cross-rater material back to an ongoing blinded rater panel.

- `BAO_CAO_KIEM_TRA_240_VI.md`: conclusion and all240 record-level findings.
- `CAC_MUC_CAN_XEM_LAI_VI.md`:3 priority source follow-ups, optional calibration
  and explicitly uncertain details. No replacement numerical grades.
- `ITEM_LEVEL_AUDIT_240.json`: exact original rating payloads and audit findings.
- `GROUPED_ITEMS.json`:120 exact outputs,2 ratings each; no condition identity key.
- `SOURCES.json`:842 unedited selected reviews and photo manifests for30 businesses.
- `FOLLOWUP_EVIDENCE.json`/`FOLLOWUP_PHOTOS/`: exact excerpts and18 unchanged JPEGs
  for follow-up; the other325 JPEGs are hash-bound in source receipts, not included.
- `SOURCE_INTEGRITY_RECEIPTS.json`, `REUSE_VERIFICATION_80.json`,
  `SNAPSHOT_RECEIPT.json`, `AUDIT_SUMMARY.json`: actual coverage/integrity evidence.
- `RUBRIC.json`: frozen rating criteria.
- `VERIFY_REVIEW_PACKAGE.py`: run with Python3 to check package hashes and bindings.

All240 records were reviewed, with80 unchanged full payloads reusing earlier
findings and160 new rating records inspected. Source checking remains bounded:
21 full business review transcripts plus targeted source checking for9 others;
all343 photo images viewed across prior labelled contact sheets and targeted
original-image reopenings. This is model-assisted plausibility audit, not an
exhaustive every-claim gold standard or authentication of human identity/source
inspection. Scores/rater identities were visible to the auditor.

Two confirmed old A02 comment/source problems remain recorded at positions1/3.
A02 position5 explicitly documents8/34 photo checking and requires follow-up.
Taco count at A01-019/A02-055 remains uncertain due overlapping shells; an
intermediate assertion that A01 was certainly wrong was retracted. Earlier
audit certainty that three tacos were verified is correspondingly qualified.

No human data, declarations, locks, model outputs or sources were amended.
No agreement statistics or model-winner comparison was run. Private SQLite,
password hashes, passwords, sessions and condition key are excluded.

`PACKAGE_MANIFEST.json` lists every other member's bytes/SHA256. It excludes
itself to avoid a circular hash. ZIP hash/extraction-verification receipt is
outside the ZIP. Offline verification rechecks bindings; it cannot reproduce
the semantic inspection or independently verify live database state.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf8')
    (ROOT/'VERIFY_REVIEW_PACKAGE.py').write_text(VERIFY_SCRIPT,encoding='utf8')
    rubric=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE/RUBRIC.json')
    (ROOT/'RUBRIC.json').write_bytes(rubric.read_bytes())
    names=['README.md','BAO_CAO_KIEM_TRA_240_VI.md','CAC_MUC_CAN_XEM_LAI_VI.md','ITEM_LEVEL_AUDIT_240.json',
        'GROUPED_ITEMS.json','SOURCES.json','FOLLOWUP_EVIDENCE.json','SOURCE_INTEGRITY_RECEIPTS.json',
        'REUSE_VERIFICATION_80.json','SNAPSHOT_RECEIPT.json','AUDIT_SUMMARY.json','RUBRIC.json','VERIFY_REVIEW_PACKAGE.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'FOLLOWUP_PHOTOS').glob('*.jpg'))]
    assert len(names)==31 and len(set(names))==31
    def h(b):return hashlib.sha256(b).hexdigest()
    manifest={'format':'stage_n_full240_audit_package_manifest_v1','excludes_itself':True,
        'members':[{'file':n,'bytes':(ROOT/n).stat().st_size,'sha256':h((ROOT/n).read_bytes())} for n in names]}
    (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    names.append('PACKAGE_MANIFEST.json')
    archive=ROOT/'STAGE_N_FULL240_RATING_QUALITY_AUDIT.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in names:z.write(ROOT/name,name)
    with tempfile.TemporaryDirectory(prefix='stage_n_full240_verify_') as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(names) and set(z.namelist())==set(names)
            z.extractall(tmp)
        completed=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(tmp)/'VERIFY_REVIEW_PACKAGE.py')],capture_output=True,text=True,encoding='utf8')
        assert completed.returncode==0,(completed.stdout,completed.stderr)
        receipt={'verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'archive':archive.name,
            'bytes':archive.stat().st_size,'sha256':h(archive.read_bytes()),'unique_zip_members':len(names),
            'crc_reopen_extract_all_member_hash_and_json_parse':'PASS','offline_verifier_result':json.loads(completed.stdout),
            'private_database_and_condition_key_excluded':True,'human_scores_modified':False}
    (ROOT/'PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
