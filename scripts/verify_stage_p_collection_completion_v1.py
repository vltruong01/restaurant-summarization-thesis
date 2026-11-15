"""Read-only completion check and consistent backup of the actual local study."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT/'outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
PACKAGE = ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/PACKAGE'
sys.path.insert(0, str(LIVE/'APP'))
from stage_p_human_evaluation_v1 import Study, CRITERIA
try:
    from stage_p_revision_aware_study_v1 import RevisionAwareStudy as Study
except ImportError:
    pass
from human_app_core_v1 import digest, file_hash
from validate_stage_p_human_exports_v1 import validate

def read_only_database():
    return sqlite3.connect((LIVE/'STATE/ratings.sqlite3').as_uri()+'?mode=ro', uri=True)

def fingerprint(connection):
    tables = ['users','participants','ratings','events','participant_declaration_history']
    return digest({t:connection.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in tables})

def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'outputs/stage_p'/('human_collection_complete_'+stamp+'_v1')
    private = out/'PRIVATE_BACKUP'
    private.mkdir(parents=True,exist_ok=False)
    source = read_only_database()
    try:
        before = fingerprint(source)
        with sqlite3.connect(private/'ratings.sqlite3') as backup:
            source.backup(backup)
        after_backup = fingerprint(source)
        assert before == after_backup, 'Database changed while snapshotting; preserve backup and inspect.'
    finally:
        source.close()
    if (LIVE/'STATE/IMPORTS').exists():
        shutil.copytree(LIVE/'STATE/IMPORTS',private/'IMPORTS')
    # Initialization and any SQLite journal mode operations affect only the backup.
    study = Study(LIVE/'WORKSPACE',preview=False,state_dir=private)
    audit = study.audit()
    assert audit['status']=='PASS',audit
    export = study.export({'id':'COORDINATOR','role':'coordinator'},final=True)
    assert audit['stored_rating_records']==90 and audit['counts']=={'DRAFT':0,'SUBMITTED':90}
    save(out/'STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json',export)
    per_rater = []
    for identity in ['A01','A02']:
        path = out/(identity+'_FINAL_45.json')
        save(path,study.rater_export({'id':identity,'role':'rater'},final=True))
        per_rater.append(path)
    records, validation = validate(PACKAGE,per_rater)
    assert len({r['business_id'] for r in records})==15
    assert sum(type(r[k]) is int and 1<=r[k]<=5 for r in records for k in CRITERIA)==540
    # Verify the original journal independently against the exported snapshot.
    with study.connect() as db:
        events = [json.loads(r[0]) for r in db.execute('SELECT event_json FROM events ORDER BY seq')]
        declarations = db.execute('SELECT COUNT(*) FROM participants').fetchone()[0]
        declaration_history = db.execute('SELECT COUNT(*) FROM participant_declaration_history').fetchone()[0]
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
        users = [dict(r) for r in db.execute('SELECT id,role FROM users ORDER BY id')]
    assert integrity=='ok'
    save(out/'RATING_EVENT_JOURNAL.json',events)
    # All bound source JSON, output text, assignments, rubric, code and runtime
    # bytes are checked here. Original photos have full hashes in the preparation
    # verifier; this completion audit checks their presence and byte lengths only.
    manifest=json.loads((LIVE/'IMMUTABLE_MANIFEST.json').read_text(encoding='utf8'))
    byte_hash_checked=0
    photo_metadata_checked=0
    for row in manifest['files']:
        path=LIVE/row['file']
        assert path.is_file() and path.stat().st_size==row['bytes'],row['file']
        if row['file'].replace('\\','/').startswith('WORKSPACE/PHOTOS/'):
            photo_metadata_checked+=1
        else:
            assert file_hash(path)==row['sha256'],row['file']
            byte_hash_checked+=1
    assert photo_metadata_checked==5143
    prior = {
      'outputs/stage_o/human_collection_complete_20261106_v1/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json':'d89c2fc9ed7c228139a1594c9ff7b1f7b2ffda2dfbe18bff435a7ccb6a13df7f',
      'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/STAGE_P_THREE_SYSTEM_HUMAN_EVAL_READY.zip':'afea0830e245f4c99f947215bcd785f3c7fb889a636724fb44465c74d46f1488',
      'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/STAGE_P_BLINDED_EVALUATOR_WINDOWS.zip':'0fa783544f165909589ce20291b3f5403378b08852ee4a582dbbc51d1925bdc4'
    }
    for name,expected in prior.items(): assert file_hash(ROOT/name)==expected,name
    with read_only_database() as connection:
        after = fingerprint(connection)
    assert before==after,'Live records changed during the audit; snapshot remains preserved.'
    summary={identity:{'submitted':sum(r['annotator']==identity for r in records),
        'numeric_scores':sum(r['annotator']==identity for r in records)*6,
        'nonempty_comments':sum(r['annotator']==identity and bool(r['evidence_notes'].strip()) for r in records),
        'serious_grounding_flags':sum(r['annotator']==identity and r['serious_grounding_issue'] for r in records)} for identity in ['A01','A02']}
    report=dict(status='PASS',verified_at_utc=datetime.now(timezone.utc).isoformat(),study_id=study.config['study_id'],
        restaurants=15,records=90,numeric_scores=540,criteria=CRITERIA,per_rater=summary,
        missing_assignments=0,duplicate_aliases=0,drafts=0,invalid_scores=0,event_records=len(events),
        journal_audit=audit,export_validator=validation,sqlite_integrity=integrity,users=users,
        participant_declaration_records=declarations,declaration_history_records=declaration_history,
        frozen_nonphoto_files_sha256_checked=byte_hash_checked,photo_presence_size_checked=photo_metadata_checked,
        photo_full_sha256_repeated_in_this_audit=False,
        protected_prior_artifacts=prior,live_database_logical_sha256_before=before,live_database_logical_sha256_after=after,
        scores_changed=0,ratings_created=0,statistical_analysis_performed=False,
        limitations=['Technical completion and recorded provenance verified; not independent authentication of participant identity, source inspection or rating quality.',
                    'Grounding flags are participant reports; no new semantic error confirmation is made.',
                    'Photo byte hashes were verified at preparation/release; this completion check repeats only existence and size.'])
    save(out/'COMPLETION_VERIFICATION.json',report)
    (out/'README.md').write_text(f'# Stage P actual collection completion\n\nBoth accounts submitted45/45 items:90 records,540 valid integer scores,15 restaurants. No drafts or missing items. The journal contains{len(events)} recorded events including any explicitly identified administrative imports; earlier event prefixes are preserved. See COMPLETION_VERIFICATION.json for the actual checks and scope.\n\nA01_FINAL_45.json and A02_FINAL_45.json are final blinded per-rater exports; STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json is the neutral coordinator export. RATING_EVENT_JOURNAL.json preserves recorded transitions. PRIVATE_BACKUP contains a consistent SQLite snapshot including password hashes: keep it private and exclude it from public/evaluator handoffs. No plaintext credentials are stored.\n\nThis is a completion and integrity check, not a statistical analysis or independent certification of human authorship/source inspection/rating quality. Old Stage O data remain separate.\n',encoding='utf8')
    # Flush/close any WAL in the backup before hashing the portable snapshot.
    with sqlite3.connect(private/'ratings.sqlite3') as connection: connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    files=[p for p in out.rglob('*') if p.is_file() and not p.name.endswith(('-shm','-wal'))]
    save(out/'ARTIFACT_HASHES.json',{'files':[dict(file=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(files)]})
    print(json.dumps({'status':'PASS','folder':str(out),'records':90,'scores':540,'events':len(events),'per_rater':summary,'export_sha256':file_hash(out/'STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')},indent=2))

if __name__=='__main__':main()
