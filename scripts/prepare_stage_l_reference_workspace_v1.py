"""Materialize the existing blank source-only package and verify all original photos."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET
from stage_l_reference_app_v1 import Store, new_json, file_hash, now

REL='outputs/stage_l/test100_human_references_v1'
GUIDE='''# Human reference annotation guidelines

Stage L constructs one common, model-independent English reference per frozen TEST business. This is reference writing, not evaluation of model outputs. No reference is supplied by an AI. Source hashes and the fixed cohort must remain unchanged.

## Sources and independence

Read ALL selected reviews with their IDs and inspect ALL original photos indexed for the business. The frozen scope is 100 businesses, 2,817 selected reviews and 1,136 original photos. These are original-byte-equivalent Yelp images retained locally, not generated images. Review source text and review order are unchanged. The photo scope is not restricted to either U or F caption subsets. The interface provides no captions by default. Do not search for additional reviews or photos. Photo/business association does not establish correspondence with a particular review.

Neither author nor checker may access evaluated RO/U/F/S TEST predictions, model identity keys, Stage K scores/diagnostics or Teacher targets as answers. Give participants only the source workspace and these guidelines. Do not give them the Stage K review ZIP or full administrator project. Participants who have already seen these outputs must disclose it and must not make a false blinding attestation; the coordinator must find eligible participants or report the independence limitation before final freeze.

## Review Consensus

Write recurring positive and negative opinions grounded only in the selected reviews. Preserve sentiment balance, useful business-specific dishes/details and important minority complaints, with proportionate attribution such as "some reviewers" where appropriate. An individual report is not universal consensus. Retain negation and distinctions between an opinion and an observable fact. Do not infer experiences from photos, erase important mixed opinions, invent specifics or copy Teacher/model prose. Aim for concise coherent English, avoid repetition and generic boilerplate. There is no new hard word limit: length/repetition checks are warnings, not automatic editing rules.

## Visual Evidence

Inspect every original photo in the frozen business scope. Describe useful directly observable objects, attributes, scene details and relationships. Keep distinct scenes separate; do not combine people, actions or objects across photos. Do not guess ingredients, precise dish identities, provenance, actions or context unsupported by pixels. Do not import review sentiment into Visual Evidence. Cover useful recurring and distinctive details without mechanically listing repetitive images. An absent F caption is not an absent original photo and does not require an empty human Visual reference. Do not use Shuffled donor photos. If an image is unclear, log its photo ID and uncertainty in notes and avoid making an uncertain detail a factual reference claim.

## Writing and evidence links

Enter component text without section markers; the export supplies [REVIEW_CONSENSUS] and [VISUAL_EVIDENCE]. Select a review excerpt and use its evidence button, or copy an exact excerpt into the linking panel. Enter an exact span from your own reference. Add multiple evidence links to the same span when the opinion recurs. Visual links require the correct photo ID and truthful direct-image inspection confirmation. Use the JSON evidence editor to inspect/remove/correct links. Evidence IDs remain metadata, outside the final prose. Maintain uncertainty and revision comments separately. Saving preserves drafts even while content is incomplete; submission requires valid linked evidence for both components.

## Independent checking and finality

A distinct real human checks the exact submitted version against all original sources, evidence, component separation, opinion balance and major omissions. Discuss disagreements and record resolution. Unsupported facts or missing important aspects must be sent back with comments. The checker may request revision or reject a version; both return it to NEEDS_REVISION without erasing the rejected text or feedback. An author revises and resubmits. Approve only after an actual independent check. Approval records exact content, IDs and UTC timestamps. Approved text is locked; do not silently change it.

Automated validation cannot authenticate human identities, certify factuality, prove complete coverage or replace photo inspection. The coordinator keeps a private roster verifying that author and checker IDs correspond to different eligible real people. No genuine references or approvals existed when this workflow was prepared. After all 100 are genuinely authored, independently checked and approved, run the final export validator. Stage M automatic metrics need a separate release. Human evaluation of outputs is later and separate.
'''

WORKFLOW='''# Reference author and checker workflow

NOT_STARTED -> DRAFT -> SUBMITTED -> APPROVED. SUBMITTED -> NEEDS_REVISION -> DRAFT -> SUBMITTED records requested corrections or rejection. Rejection is recorded as an explicit event, not deletion. No new REJECTED status is introduced into the approved five-status tracker.

The coordinator registers actual participant accounts using `add-user`. IDs are pseudonyms, not invented real-person identities; keep an eligibility/identity roster privately. Authors can edit only their own drafts. Checkers cannot change prose, impersonate authors or approve their own authored record. Coordinators can inspect progress but cannot author or approve via a coordinator role. A server login is required to read sources or original photos. No general filesystem/static directory is exposed. API mutation requires a session and same-origin CSRF token. The server binds only to 127.0.0.1.

Author: read all sources, write plain English, add exact span-to-review/quote and span-to-photo evidence, record uncertainties, save/resume, then truthfully attest human authorship, source inspection and blindness before submission. Invalid evidence blocks submission, not structural draft review. Checker: independently read sources and claims; assess factuality, polarity, proportionate attribution, major omissions, visual relationships and scene separation; request revision with comments or approve the exact version with explicit attestations. Different IDs do not by themselves prove different real people.

Draft writes are SQLite transactions with synchronous FULL; reference state and a full revision event commit together. Concurrent edits require the expected server version; stale saves are rejected. Every accepted action records actor, role, timestamp, content SHA256, prior reference hash, complete snapshot, comments and the previous event hash. This detects ordinary accidental corruption/tampering, not malicious rewriting by an administrator controlling the database. The browser autosaves with a local recovery draft; server save failures are visible. Do not clear browser storage while a save is failing. Browser recovery is private participant work: use separate OS/browser profiles, and clear recovery after confirming server persistence. It is not an independent backup. Use the database backup utility regularly, before moving machines, and after approval sessions.

Approved records are locked; this version does not provide an approval-reopening shortcut. If an approved reference is found wrong, stop final freeze and involve the coordinator/researcher rather than editing SQLite manually. Preserve the old database and establish an explicitly logged correction process before any replacement.

Deployment boundary: loopback authentication prevents casual app access, not access by someone who can read the whole administrator disk. For real independence, run the portable source-only bundle on a separate annotator OS account/machine without Stage K outputs or Teacher directories. Do not expose the development server to the public internet. On the same trusted machine, sign out and use separate participant browser profiles. Coordinator identity roster, credentials, drafts and real approval history stay outside the ChatGPT review ZIP.
'''

README='''# Stage L human reference workbench

Working source-only interface for real human authors and independent checkers. Actual reference progress at delivery: 0/100 authored, 0/100 submitted, 0/100 approved. Synthetic workflow tests are separate and are not human work.

## Start on this Windows project

Open PowerShell in `C:\\thesis_proposal\\yelp_dataset`. Register each real participant (choose their own distinct human ID and password; do not use these placeholder labels literally):

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py add-user --workspace outputs/stage_l/test100_human_references_v1 --id REAL_AUTHOR_ID --role author
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py add-user --workspace outputs/stage_l/test100_human_references_v1 --id REAL_CHECKER_ID --role checker
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py serve --workspace outputs/stage_l/test100_human_references_v1
```

Open http://127.0.0.1:8765 . The source workspace is ready; no real accounts are precreated. All image hashes and image decoding are checked on start. PIL/Pillow is the only non-stdlib dependency (already available in the project environment). The first launch may take a few seconds for image checks. `--port` selects another local port. Ctrl+C stops the server; committed drafts remain in SQLite. Sign out between participants and preferably use separate OS/browser profiles. Setup helpers are also supplied as CMD files.

Use business search/status filter, review search, full original-image viewer, separate component editors and the evidence linking panel. Each successful save shows its server version. Incomplete drafts remain editable; invalid evidence/format blocks submission. Checker actions appear only for submitted versions to a registered checker. Rejection and revision requests require feedback. Never share your password. Comments and previous text stay in history.

## Verify, back up and export

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py verify --workspace outputs/stage_l/test100_human_references_v1
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py backup --workspace outputs/stage_l/test100_human_references_v1 --destination PRIVATE_BACKUPS/reference_session_001.sqlite3
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py export --workspace outputs/stage_l/test100_human_references_v1 --destination PRIVATE_EXPORTS/draft_session_001
```

Destinations must be NEW. Draft export includes a readable inspection document, clean machine-readable records, compatible per-business annotation JSONs, approval CSV and full revision events. It is explicitly not final gold. Before final export, all 100 actual human references must pass checks and independent approval:

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py export --workspace outputs/stage_l/test100_human_references_v1 --destination PRIVATE_EXPORTS/final_001 --final
```

Final export adds APPROVED_REFERENCES.json and REFERENCE_FREEZE.json, with original-compatible `ANNOTATIONS/` usable by the existing Stage M scorer's `--references` argument. Export writes no model scores. No final freeze exists at delivery. All final component markers are supplied only in readable export. Reference SHA256 uses the existing canonical reference_content contract, including business/source/evidence bindings; exact serialized-file SHA256 is also retained. Private database backups include credentials and must not be uploaded as review evidence. Restore with the server stopped to a new workspace copy, preserve the old database, then verify its journal and sources before use.

## Portable source-only use

The review ZIP includes scripts, web UI, guidelines, exact source JSONs, blank templates and small integrity receipts; no original image archive, model predictions, scores or accounts. Keep the included directory tree. On an independent participant machine with Python 3.11+ and Pillow, materialize the 1,136 exact original photos using the existing `materialize_reference_photos_v1.py` on the coordinator machine, verify its receipt, then transfer this source-only image folder privately under Yelp terms. Use `configure_stage_l_portable_v1.py` to create a NEW workspace bound to those photos, without altering frozen source JSONs. Do not distribute the whole thesis project. See SOURCE_ACCESS_INSTRUCTIONS.md in SOURCE_PACKAGE for original image provenance.

No GPU work, reference generation, Stage M metrics or human model-output evaluation occurs here. Actual human writing and independent checking are the next required actions. Separate researcher release remains necessary for Stage M after final approval.
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path.cwd());a=p.parse_args();project=a.project.resolve();out=project/REL
    if out.exists():raise ValueError('Refuse to overwrite an existing annotation workspace')
    package=project/'outputs/final_evaluation/s1000_preparation_v1/TEST100_REFERENCE_ANNOTATION_PACKAGE'
    out.mkdir(parents=True);shutil.copytree(package,out/'SOURCE_PACKAGE')
    config={'version':'stage_l_reference_v1','source_manifest_sha256':file_hash(package/'ANNOTATION_PACKAGE_MANIFEST.json'),
        'expected_businesses':100,'expected_reviews':2817,'expected_photos':1136,'project_root':str(project),'portable_photo_root':None}
    new_json(out/'CONFIG.json',config)
    for name,text in [('HUMAN_REFERENCE_ANNOTATION_GUIDELINES.md',GUIDE),('REFERENCE_AUTHOR_CHECKER_WORKFLOW.md',WORKFLOW),('README_SETUP_VI.md',README)]:
        (out/name).write_text(text,encoding='utf8')
    schema={'component_fields':['review_consensus','visual_evidence'],'plain_text_no_markers':True,
        'review_claim':{'output_span':'exact substring supplied by human','evidence':[{'review_id':'same-business frozen ID','quote':'exact excerpt','note':'optional uncertainty/detail'}]},
        'visual_claim':{'output_span':'exact substring supplied by human','evidence':[{'photo_id':'same-business frozen ID','direct_image_viewed':'must be boolean true after actual inspection','note':'optional observable detail'}]},
        'statuses':['NOT_STARTED','DRAFT','SUBMITTED','NEEDS_REVISION','APPROVED'],
        'semantic_verification':'Human author plus distinct human checker, never certified by automated schema'}
    new_json(out/'EVIDENCE_LINKING_SCHEMA.json',schema)
    store=Store(out);new_json(out/'SOURCE_INTEGRITY_RECEIPT.json',store.verify_sources(True))
    new_json(out/'INITIAL_REFERENCE_PROGRESS.json',store.audit());store.export(out/'INITIAL_DRAFT_EXPORT',False)
    evidence=out/'EVIDENCE';evidence.mkdir();proposal=Path('C:/thesis_proposal/thesis_proposal_v16.docx')
    with zipfile.ZipFile(proposal) as z:
        root=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paras=[''.join(t.text or '' for t in par.findall('.//w:t',ns)) for par in root.findall('.//w:p',ns)]
    indices=[623,665,730,791,929,936,938,940,942,944,946]
    new_json(evidence/'PROPOSAL_V16_REFERENCE_PROTOCOL_EXCERPTS.json',{'path':str(proposal),'sha256':file_hash(proposal),'paragraphs':[{'zero_based_paragraph':i,'text':paras[i]} for i in indices]})
    protected={}
    for rel in ['outputs/final_evaluation/s1000_preparation_v1','outputs/stage_k/s1000_test100_phase1_v1']:
        base=project/rel
        for f in base.rglob('*'):
            if f.is_file() and f.stat().st_size<30_000_000 and not any(x in f.parts for x in ['METRIC_CACHE','METRIC_RUNTIME','REVIEW_PACKAGE']):protected[str(f.relative_to(project)).replace('\\','/')]=file_hash(f)
    new_json(evidence/'PROTECTED_ARTIFACTS_BEFORE.json',protected)
    runtime=r'C:\Users\VUONGLOCTRUONG\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
    (out/'START_ANNOTATION.cmd').write_text('@echo off\r\ncd /d C:\\thesis_proposal\\yelp_dataset\r\n.venv-caption-pilot\\Scripts\\python.exe -X utf8 -B scripts\\stage_l_reference_app_v1.py serve --workspace outputs\\stage_l\\test100_human_references_v1\r\npause\r\n',encoding='utf8')
    (out/'REGISTER_PARTICIPANT.cmd').write_text('@echo off\r\ncd /d C:\\thesis_proposal\\yelp_dataset\r\nset /p PARTICIPANT_ID=Real participant pseudonymous ID: \r\nset /p PARTICIPANT_ROLE=Role (author or checker): \r\n.venv-caption-pilot\\Scripts\\python.exe -X utf8 -B scripts\\stage_l_reference_app_v1.py add-user --workspace outputs\\stage_l\\test100_human_references_v1 --id "%PARTICIPANT_ID%" --role "%PARTICIPANT_ROLE%"\r\npause\r\n',encoding='utf8')
    print(json.dumps({'workspace':str(out),'source_counts':store.verify_sources(False),'progress':store.audit()},indent=2))

if __name__=='__main__':main()
