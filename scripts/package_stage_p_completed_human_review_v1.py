"""Package existing completed ratings and frozen evidence; no new scoring/analysis."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/PACKAGE'
COLLECTION=ROOT/'outputs/stage_p/human_collection_complete_20261106T070046Z_v1'
DEST=ROOT/'outputs/stage_p/completed_human_evaluation_handoff_20261106_v1'
PACKAGE=DEST/'PACKAGE'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
copied=[]
def copy(source,relative):
    target=PACKAGE/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert sha(source)==sha(target)
    copied.append(dict(source=source.relative_to(ROOT).as_posix(),package_file=relative,sha256=sha(source)))

def main():
    global COLLECTION,DEST,PACKAGE
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',type=Path,default=COLLECTION)
    parser.add_argument('--destination',type=Path,default=DEST)
    args=parser.parse_args()
    COLLECTION=args.collection.resolve()
    DEST=args.destination.resolve()
    PACKAGE=DEST/'PACKAGE'
    assert not DEST.exists(),'Existing handoff must not be overwritten.'
    completion=read(COLLECTION/'COMPLETION_VERIFICATION.json')
    assert completion['status']=='PASS'
    events=completion['event_records']
    flags={r:completion['per_rater'][r]['serious_grounding_flags'] for r in ['A01','A02']}
    export_sha=sha(COLLECTION/'STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')
    PACKAGE.mkdir(parents=True)
    live=ROOT/'outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
    database=live/'STATE/ratings.sqlite3'
    db_before=sha(database)
    for name in ['A01_FINAL_45.json','A02_FINAL_45.json','STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json','RATING_EVENT_JOURNAL.json']:
        copy(COLLECTION/name,'RATINGS/'+name)
    copy(COLLECTION/'COMPLETION_VERIFICATION.json','EVIDENCE/COMPLETION_VERIFICATION.json')
    copy(live/'WORKSPACE/COLLECTION_OPEN.json','EVIDENCE/COLLECTION_OPEN.json')
    for file in sorted((SOURCE/'COORDINATOR_ONLY').iterdir()):
        if file.is_file():copy(file,'FROZEN_STUDY/COORDINATOR_ONLY/'+file.name)
    w=SOURCE/'EVALUATOR_APP/WORKSPACE'
    for name in ['CONFIG.json','BLINDED_ITEMS.json','RUBRIC.json','TWO_RATER_PROTOCOL_AMENDMENT.md']:
        copy(w/name,'FROZEN_STUDY/EVALUATOR_APP/WORKSPACE/'+name)
    for binding in read(w/'CONFIG.json')['sources'].values():
        copy(w/binding['file'],'FROZEN_STUDY/EVALUATOR_APP/WORKSPACE/'+binding['file'])
    for file in sorted((SOURCE/'STATISTICS').iterdir()):
        if file.is_file():copy(file,'STATISTICS/'+file.name)
    copy(ROOT/'scripts/stage_p_revision_provenance_v1.py','STATISTICS/stage_p_revision_provenance_v1.py')
    if (live/'STATE/IMPORTS').exists():
        for file in sorted((live/'STATE/IMPORTS').glob('*.json')):copy(file,'EVIDENCE/REVISION_IMPORTS/'+file.name)
        copy(ROOT/'scripts/stage_p_revision_aware_study_v1.py','EVIDENCE/stage_p_revision_aware_study_v1.py')
    copy(ROOT/'scripts/verify_stage_p_completed_review_package_v1.py','VERIFY_PACKAGE.py')
    copy(Path(__file__),'REPRODUCIBILITY/package_stage_p_completed_human_review_v1.py')
    write(PACKAGE/'EVIDENCE/COPY_PROVENANCE.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),copied_files=copied,reference_free=True,old_Stage_O_ratings_included=False,ratings_modified=0))
    (PACKAGE/'README.md').write_text('''# Stage P completed human evaluation: independent review

This is a coordinator/ChatGPT review handoff, NOT an evaluator distribution. It includes the true system identity key. Do not give it to either evaluator for further blinded collection.

## Completed collection

15 frozen TripAdvisor restaurants; original accepted LED F-S1000, Qwen-Caption and Qwen-Image; two accounts A01/A02, each45/45 submitted assignments. Total90 NEW records and540 integer1-5 scores across the unchanged six criteria. No missing items or drafts. The137 recorded journal events match final rating states. All original scores/comments/timestamps are preserved. The older Stage O LED RO/U/F90-record panel remains separate and is not reused here.

## Files

- RATINGS/: exact final coordinator export, two final blinded per-rater exports and recorded event journal.
- FROZEN_STUDY/COORDINATOR_ONLY/: true alias/system mapping, original frozen45 outputs, position balance/seeds, exact source pointers, bindings and prespecified analysis plan.
- FROZEN_STUDY/EVALUATOR_APP/WORKSPACE/: frozen90 blinded assignments, configuration, unchanged rubric/protocol and exact15 source JSON files containing247 selected reviews and5,143 photo metadata/hash records.
- EVIDENCE/: original completion verification, actual operational collection-open record and byte-for-byte copy provenance. Direct account access was authorized by the researcher; no participant declarations were created. Account IDs do not independently establish human identity.
- STATISTICS/: existing validator and prepared analysis script/configuration. Analysis has NOT been run for this handoff.
- VERIFY_PACKAGE.py and PACKAGE_MANIFEST.json: offline read-only verification of package bytes, completed records, source/output/mapping bindings and journal.

## Independent verification

Extract the ZIP to a new folder. Run `python -B VERIFY_PACKAGE.py` using Python3.11+ (standard library only). The verification prints actual PASS/check counts and writes no ratings. File manifest excludes only itself. The adjacent delivery receipt records actual extraction and every-member hash checks. Optional future analysis uses NumPy; see STATISTICS/requirements.txt and the frozen plan before running it. Do not change/exclude ratings to obtain a preferred result.

Final90 export SHA256: e47b88f1b4d6ccf0bf1649b5b06ead3f538f97af30324993f5e45cf79a58472f.

## Interpretation and limitations

A01 reported10 grounding flags and A02 reported7. These17 flags are participant reports, not17 independently confirmed semantic errors. Comments and uncertain/confirmed labels are original participant judgments. Technical integrity does not independently authenticate participant identity, authorship, source inspection, blindness or judgment quality. No statistical results or model ranking are supplied.

All15 Qwen-Caption/Qwen-Image Review texts are identical; different ratings on Review-only criteria cannot establish different generated Review content. Existing analysis plan treats15 restaurants as paired clusters, retains both raters and uses the frozen comparisons/settings. Source preparation, model provenance and supervision differences remain potential confounders; these external reference-free human judgments are distinct from Teacher-target mimicry metrics.

The package intentionally excludes original JPEG pixels, model weights, the application, account database/password hashes, plaintext credentials and private backups. It contains exact photo IDs/hashes and source pointers, but cannot independently support visual semantic verification without opening the separately preserved original photos. Photo SHA256 verification at preparation/release is recorded; the completion audit repeated presence/size only. No fabricated declarations, approvals or replacement audit records are added. Collection completion is researcher-reported and technically verified, not independent authentication of human participation.
''',encoding='utf8')
    (PACKAGE/'PROMPT_FOR_CHATGPT.txt').write_text('''Stage P three-system human evaluation collection is complete. Please independently inspect this entire package before proposing the final statistical analysis.

Verify the exact90 submitted records (A01=45,A02=45),540 valid scores,15 restaurants, frozen model assignments, original45 outputs, rubric/source bindings,137-event journal and file hashes. Distinguish technical completion from independently authenticated human participation. Preserve original ratings/comments and keep the older Stage O90-record panel separate.

Read EVIDENCE/COMPLETION_VERIFICATION.json, then the final exports, true assignment key, unchanged rubric and frozen analysis plan. Use VERIFY_PACKAGE.py if execution is available. If execution or file access fails, state which checks remain unperformed; do not claim PASS based only on a manifest.

There are10 A01 and7 A02 grounding flags; they are reported judgments, not automatically confirmed factual errors. Original image pixels are excluded, so do not claim to have visually verified photo-based judgments from this ZIP. All15 Qwen Review pairs share identical text. Do not choose a winner from summary lengths or isolated flags.

Report whether collection evidence is sufficient to proceed with the frozen statistical plan, any real integrity issues, and the exact remaining limitations. No new training, inference, references, altered ratings or fabricated provenance are authorized by this review request.
''',encoding='utf8')
    # Render current recorded counts and export binding; never reuse old findings.
    for name in ['README.md','PROMPT_FOR_CHATGPT.txt']:
        path=PACKAGE/name
        text=path.read_text(encoding='utf8')
        text=text.replace('137-event',str(events)+'-event').replace('The137 recorded',f'The{events} recorded')
        text=text.replace('e47b88f1b4d6ccf0bf1649b5b06ead3f538f97af30324993f5e45cf79a58472f',export_sha)
        text=text.replace('A01 reported10 grounding flags and A02 reported7. These17 flags',f'A01 reported{flags["A01"]} grounding flags and A02 reported{flags["A02"]}. These{sum(flags.values())} flags')
        text=text.replace('not17 independently',f'not{sum(flags.values())} independently')
        text=text.replace('There are10 A01 and7 A02 grounding flags;',f'There are{flags["A01"]} A01 and{flags["A02"]} A02 grounding flags;')
        path.write_text(text,encoding='utf8')
    receipts=[read(p) for p in sorted((live/'STATE/IMPORTS').glob('*.receipt.json'))] if (live/'STATE/IMPORTS').exists() else []
    if receipts:
        with (PACKAGE/'README.md').open('a',encoding='utf8') as f:
            f.write('\n## Imported portable A02 revisions\n\nThe current coordinator dataset includes researcher-authorized administrative imports from the exact returned A02 file. Source/baseline files and receipts are in EVIDENCE/REVISION_IMPORTS/. The original journal prefix is retained; import events are explicitly labeled and retain remote submission timestamps separately from local import time. No new local human submissions are inferred. See receipts for actual before/after field changes; packaging itself changes no scores.\n')
        with (PACKAGE/'PROMPT_FOR_CHATGPT.txt').open('a',encoding='utf8') as f:f.write('\nThis revision includes an explicit portable A02 import. Inspect EVIDENCE/REVISION_IMPORTS/ source,baseline and receipt files. The validator checks every submitted-state import transition against those archived SHA256-bound files. Do not treat import timestamps as new human rating times.\n')
    write(PACKAGE/'EVIDENCE/CURRENT_EXPORT_NOTICE.json',dict(
        format='stage_p_current_reexport_v2',researcher_statement='The researcher reports the evaluators revised their evaluation and requests a new ZIP.',
        current_snapshot=COLLECTION.name,current_snapshot_verified_at_utc=completion['verified_at_utc'],
        records=90,scores=540,journal_events=events,flags_by_rater=flags,current_export_sha256=export_sha,
        current_snapshot_checks='PASS',score_or_comment_changes_made_by_packaging=0,
        verified_portable_import_receipts=receipts,
        previous_record_level_changes_verified=bool(receipts),
        comparison_limitation=('The current portable A02 import has an exact archived baseline and verified before/after changes in the import receipt. This establishes this specific revision only, not every historical revision.' if receipts else 'The earlier loose coordinator export available at packaging contains zero records and its available ZIP has no expected RATINGS member. Exact before/after score changes cannot be independently reconstructed from those available files. They are preserved unchanged. Current records, frozen assignments and journal were verified independently.')))
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    write(PACKAGE/'PACKAGE_MANIFEST.json',dict(format='stage_p_completed_human_review_manifest_v1',files=[dict(file=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    result=subprocess.run([sys.executable,'-B',str(PACKAGE/'VERIFY_PACKAGE.py')],capture_output=True,text=True,encoding='utf8',check=True)
    native_validation=json.loads(result.stdout)
    archive=DEST/'STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip'
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in files:z.write(path,path.relative_to(PACKAGE).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_p_completed_review_') as temporary:
        extracted=Path(temporary)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names=z.namelist()
            assert len(names)==len(set(names))==len(files)
            assert set(names)=={p.relative_to(PACKAGE).as_posix() for p in files}
            for path in files:assert hashlib.sha256(z.read(path.relative_to(PACKAGE).as_posix())).hexdigest()==sha(path)
            z.extractall(extracted)
        result=subprocess.run([sys.executable,'-B',str(extracted/'VERIFY_PACKAGE.py')],capture_output=True,text=True,encoding='utf8',check=True)
        extracted_validation=json.loads(result.stdout)
        assert native_validation==extracted_validation
    assert sha(database)==db_before,'Live database changed during packaging; inspect before publication.'
    archive_hash=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(archive_hash+'  '+archive.name+'\n',encoding='ascii')
    receipt=dict(status='PASS',verified_at_utc=datetime.now(timezone.utc).isoformat(),archive=archive.name,bytes=archive.stat().st_size,sha256=archive_hash,members=len(files),zip_crc='PASS',every_zip_member_sha256='PASS',extracted_offline_verifier=extracted_validation,native_offline_verifier=native_validation,live_database_bytes_unchanged=True,private_database_or_credentials_included=False,ratings_modified=0,statistical_analysis_performed=False)
    write(DEST/'DELIVERY_VALIDATION.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
