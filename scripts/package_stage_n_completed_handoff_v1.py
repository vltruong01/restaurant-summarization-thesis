"""Create an accepted-collection review handoff without changing human records."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

AUDIT=Path('outputs/stage_n/full240_rating_quality_audit_20261102_v1')
ROOT=Path('outputs/stage_n/completed_human_evaluation_handoff_20261102_v1')
STUDY=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(data):return hashlib.sha256(data).hexdigest()
def rows(path,table):
    c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:return c.execute('SELECT * FROM '+table+' ORDER BY '+('alias' if table=='ratings' else 'seq')).fetchall()
    finally:c.close()


REPORT='''# Stage N — Completed Human Evaluation: Review Handoff

## Completion and researcher decision

A01 and A02 have completed their recorded human-evaluation panels:120 submitted
items per participant,240 submissions total. Each participant assessed the same
30 TEST businesses with four frozen outputs per business. The researcher has
accepted the complete collection, retaining the original scores and comments.

This is the Stage N rating of model outputs. It is separate from Stage L's100
approved TEST references and Stage M's reference-based automatic evaluation.

## What was assessed

The study contains RO, U, Filtered and Shuffled conditions. Each participant had
independently randomized item order and aliases. The review material here keeps
the condition-to-alias key separate from the delivered data. There are120 paired
output groups, not240 distinct businesses or240 independently generated outputs.

The six1–5 criteria are Review Faithfulness, Aspect Coverage, Coherence,
Non-Redundancy, Usefulness, and Visual Grounding and Contribution. Review
Faithfulness, Aspect Coverage and Usefulness judge Review only; Coherence and
Non-Redundancy judge all applicable components. Visual is judged against the
recipient business's original photographs. Structural absence is N/A; an empty
but applicable Visual component remains scoreable.

## Recorded dataset and verification

| Item | Count |
|---|---:|
| TEST businesses |30|
| Output groups |120|
| A01 submitted records |120|
| A02 submitted records |120|
| Total recorded ratings |240|
| Numeric1–5 criterion scores |1,380|
| Structural Visual N/A scores |60|
| Applicable empty Visual ratings |4|
| Selected source reviews |842|
| Source photographs hash-bound |343|
| Original journal events |241|

Actual read-only checks verify assignment completeness, ownership, frozen output
and source bindings, score ranges/N/A rules, SQLite integrity and the original
event hash/version/state chain. The handoff also rechecks that the live240 rating
rows and241 events still equal the audited snapshot. The81 serious-issue flags
have valid output-span/source-ID schemas; this is not a claim that all81 are
independently confirmed semantic errors.

## Package structure

- `RATINGS_240.json`: exact submitted payloads, original versions, aliases,
  item positions, business IDs and paired output-group IDs. No substituted scores.
- `COHORT30.json`: frozen membership and selection rule.
- `RESEARCHER_ACCEPTANCE_AS_RECORDED.json`: researcher acceptance receipt.
- `AUDIT/`: complete preserved earlier review package, including original
  outputs, reviews, photo hashes, rubric and model-assisted audit records.
  Eighteen targeted original JPEGs are included; all343 are hash-bound, but
  the remaining325 photographs are not bundled here.
- `ORIGINAL_AUDIT_PACKAGE_RECEIPT.json`: receipt for the unchanged prior ZIP.
- `VERIFY_PACKAGE.py`, `PACKAGE_MANIFEST.json`: offline member-hash and
  record/binding verification. The manifest excludes itself to avoid circularity.
- `PROMPT_FOR_CHATGPT.txt`: proposed independent-review request.

## Interpretation and provenance

Original participant scores/comments, reference data, outputs and checkpoints
are unchanged. Researcher acceptance is the disposition of the recorded
collection; it does not replace the separate audit record. Detailed audit
findings and uncertainties remain available under `AUDIT/` for traceability.
The audit is model-assisted and does not independently authenticate participant
identity or source inspection. Existing direct-access/declaration limitations
remain documented in the study record; no declarations were created here.

No inter-rater agreement, paired condition statistics or model ranking has been
computed for this handoff. Do not infer a winning model from collection completion
or audit triage counts. A condition mapping and actual prespecified statistical
analysis are required before model-specific human-evaluation conclusions.

No credentials, password hashes, sessions, private SQLite or condition identity
key are included. This package contains both participants' judgments and is for
researcher/coordinator review, not an ongoing blinded rating panel.
'''

PROMPT='''Please review the attached STAGE_N_HUMAN_EVALUATION_COMPLETED_REVIEW.zip.

A01 and A02 completed120 recorded human-evaluation items each:30 TEST businesses
times four outputs per participant,240 total submissions. The researcher has
accepted the complete collection with original scores and comments retained.
Stage L's100 reference records and Stage M's automatic results are separate.

Read STAGE_N_COMPLETION_REPORT.md and verify the manifest first. Inspect
RATINGS_240.json, the frozen cohort/rubric and supporting AUDIT records as needed.

Please:
1. Check completeness, score/N/A handling, pairing and provenance.
2. Distinguish researcher acceptance, recorded human ratings, model-assisted
   audit judgments and independently established facts.
3. Explain what conclusions are currently justified and propose the next
   appropriate agreement/paired-analysis step using the existing protocol.
4. Keep the original cohort and all240 records. Do not replace human scores with
   AI grades, selectively remove records, or infer a model winner without actual
   condition mapping and statistical results.

The condition-to-alias key is deliberately not included. If model-specific
analysis requires it or additional protocol evidence, identify exactly what is
missing instead of guessing. Detailed audit findings are retained separately
under AUDIT/; the main handoff reports completion and researcher acceptance.

Do not initiate training, inference, new evaluation collection or another
research stage. Return a concise review and any necessary analysis prerequisites.
'''

VERIFY='''"""Verify handoff bytes and exact recorded payload bindings; no rescoring."""
import hashlib,json,pathlib,subprocess,sys
p=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((p/n).read_text(encoding='utf-8-sig'))
m=read('PACKAGE_MANIFEST.json')
for x in m['members']:
 b=(p/x['file']).read_bytes()
 assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],x['file']
ratings=read('RATINGS_240.json');audit=read('AUDIT/ITEM_LEVEL_AUDIT_240.json')
assert len(ratings)==240 and len({r['alias'] for r in ratings})==240
indexed={r['alias']:r for r in audit}
for r in ratings:
 a=indexed[r['alias']]
 assert r['rating']==a['original_rating_payload'] and r['version']==a['original_version']
 assert r['rater']==a['rater'] and r['position']==a['position'] and r['group_id']==a['group_id']
for name in ['A01','A02']:assert {r['position'] for r in ratings if r['rater']==name}==set(range(1,121))
assert len(read('COHORT30.json')['business_ids'])==30
assert read('RESEARCHER_ACCEPTANCE_AS_RECORDED.json')['decision']=='ACCEPT_ALL_240_RECORDED_RATINGS_AS_IS'
result=subprocess.run([sys.executable,'-X','utf8','-B',str(p/'AUDIT/VERIFY_REVIEW_PACKAGE.py')],capture_output=True,text=True,encoding='utf-8')
assert result.returncode==0,(result.stdout,result.stderr)
print(json.dumps({'result':'PASS','hash_verified_handoff_members':len(m['members']),'exact_original_rating_payloads':240,'nested_audit_verification':json.loads(result.stdout),'scores_changed':False}))
'''


def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    live_db=STUDY/'PRIVATE_STATE/ratings.sqlite3';snap=AUDIT/'PRIVATE_SNAPSHOT.sqlite3'
    assert rows(live_db,'ratings')==rows(snap,'ratings')
    assert rows(live_db,'events')==rows(snap,'events')
    acceptance=read(AUDIT/'RESEARCHER_ACCEPTANCE_AS_RECORDED.json')
    prior=read(AUDIT/'PACKAGE_VERIFICATION.json');original=AUDIT/prior['archive']
    assert sha(original.read_bytes())==prior['sha256']==acceptance['unchanged_reference_review_zip_sha256']
    with zipfile.ZipFile(original) as z:
        assert z.testzip() is None
        original_members=z.namelist()
        assert len(original_members)==32 and all(not n.startswith('/') and '..' not in Path(n).parts for n in original_members)
        for name in original_members:
            path=ROOT/'AUDIT'/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(z.read(name))
    findings=read(AUDIT/'ITEM_LEVEL_AUDIT_240.json')
    ratings=[{'rater':f['rater'],'position':f['position'],'alias':f['alias'],
        'business_id':f['business_id'],'business_name':f['business_name'],'group_id':f['group_id'],
        'version':f['original_version'],'rating':f['original_rating_payload'],
        'review_output_sha256':f['review_output_sha256'],'visual_output_sha256':f['visual_output_sha256']} for f in findings]
    (ROOT/'RATINGS_240.json').write_text(json.dumps(ratings,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for source,name in [(AUDIT/'RESEARCHER_ACCEPTANCE_AS_RECORDED.json','RESEARCHER_ACCEPTANCE_AS_RECORDED.json'),
            (AUDIT/'PACKAGE_VERIFICATION.json','ORIGINAL_AUDIT_PACKAGE_RECEIPT.json'),(STUDY/'COHORT30.json','COHORT30.json')]:
        (ROOT/name).write_bytes(source.read_bytes())
    (ROOT/'STAGE_N_COMPLETION_REPORT.md').write_text(REPORT,encoding='utf8')
    (ROOT/'PROMPT_FOR_CHATGPT.txt').write_text(PROMPT,encoding='utf8')
    (ROOT/'VERIFY_PACKAGE.py').write_text(VERIFY,encoding='utf8')
    names=['RATINGS_240.json','RESEARCHER_ACCEPTANCE_AS_RECORDED.json','ORIGINAL_AUDIT_PACKAGE_RECEIPT.json',
        'COHORT30.json','STAGE_N_COMPLETION_REPORT.md','PROMPT_FOR_CHATGPT.txt','VERIFY_PACKAGE.py']
    names+=['AUDIT/'+name for name in original_members]
    assert len(names)==len(set(names))==39
    manifest={'format':'stage_n_completed_human_evaluation_handoff_v1','manifest_excludes_itself':True,
        'members':[{'file':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha((ROOT/name).read_bytes())} for name in names]}
    (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    names.append('PACKAGE_MANIFEST.json')
    archive=ROOT/'STAGE_N_HUMAN_EVALUATION_COMPLETED_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in names:z.write(ROOT/name,name)
    with tempfile.TemporaryDirectory(prefix='stage_n_completed_handoff_') as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==40 and set(z.namelist())==set(names)
            z.extractall(tmp)
        result=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(tmp)/'VERIFY_PACKAGE.py')],capture_output=True,text=True,encoding='utf8')
        assert result.returncode==0,(result.stdout,result.stderr)
    assert rows(live_db,'ratings')==rows(snap,'ratings') and rows(live_db,'events')==rows(snap,'events')
    receipt={'verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'archive':archive.name,
        'bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes()),'unique_members':40,
        'crc_reopen_extract_all_member_hash_and_json_checks':'PASS','offline_verification':json.loads(result.stdout),
        'live240_ratings_and241_events_unchanged':True,'original_audit_zip_unchanged':True,
        'condition_key_private_db_passwords_and_sessions_excluded':True,'statistics_or_model_ranking_run':False}
    (ROOT/'HANDOFF_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
