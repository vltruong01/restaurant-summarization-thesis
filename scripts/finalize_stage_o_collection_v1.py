"""Read-only live collection check, consistent private backup and original export.

No rating edits, semantic regrading, statistics, training or inference.
"""
from pathlib import Path
import collections,contextlib,datetime,hashlib,json,sqlite3,sys

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP'
D=ROOT/'outputs/stage_o/human_collection_complete_20261106_v1'
sys.path.insert(0,str(APP/'scripts'))
from stage_o_human_evaluation_v1 import Study,CRITERIA

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
    assert not D.exists(),'Preserve any previous completion snapshot; do not overwrite'
    state=D/'PRIVATE_BACKUP';state.mkdir(parents=True)
    target=state/'ratings.sqlite3';source=APP/'WORKSPACE/PRIVATE_STATE/ratings.sqlite3'
    with contextlib.closing(sqlite3.connect('file:'+source.resolve().as_posix()+'?mode=ro',uri=True)) as src,contextlib.closing(sqlite3.connect(target)) as dst:src.backup(dst)
    study=Study(APP/'WORKSPACE',state_dir=state);audit=study.audit();issues=[];summary={}
    with study.connect() as c:
        integrity=c.execute('PRAGMA integrity_check').fetchone()[0]
        allrows=[dict(x) for x in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]
        events=c.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    for who in ['A01','A02']:
        rows=[r for r in allrows if r['rater']==who];statuses=collections.Counter();conditions=collections.Counter();numeric=na=0
        expected={i['alias'] for i in study.key if i['annotator']==who}
        if {r['alias'] for r in rows}!=expected:issues.append({'rater':who,'issue':'Missing or extra assigned items'})
        for row in rows:
            p=json.loads(row['payload']);i=study.items.get(row['alias']);statuses[p.get('status')]+=1
            if not i or i['annotator']!=who:issues.append({'alias':row['alias'],'issue':'Wrong assignment owner'});continue
            conditions[i['condition']]+=1
            if p.get('status')!='SUBMITTED' or not p.get('submitted_at_utc'):issues.append({'alias':row['alias'],'issue':'Not submitted or missing timestamp'})
            if set(p.get('scores',{}))!=set(CRITERIA):issues.append({'alias':row['alias'],'issue':'Wrong criterion fields'})
            for criterion in CRITERIA:
                value=p.get('scores',{}).get(criterion)
                if criterion==CRITERIA[-1] and i['condition']=='RO':
                    if value!='N/A':issues.append({'alias':row['alias'],'issue':'RO Visual must be N/A'})
                    else:na+=1
                elif type(value) is not int or not 1<=value<=5:issues.append({'alias':row['alias'],'criterion':criterion,'issue':'Missing or invalid score'})
                else:numeric+=1
            sourceinfo=study.sources[i['business_id']];allowed=set(sourceinfo['selected_review_ids'])|{p['photo_id'] for p in sourceinfo['photos']}
            if any(x not in allowed for x in p.get('evidence_source_ids',[])):issues.append({'alias':row['alias'],'issue':'Evidence ID outside restaurant'})
            if p.get('serious_grounding_issue'):
                if not (p.get('evidence_notes','').strip() and p.get('issue_span') and p['issue_span'] in i['review_text']+'\n'+(i['visual_text'] or '') and p.get('evidence_source_ids') and p.get('issue_certainty') in ['uncertain','confirmed']):issues.append({'alias':row['alias'],'issue':'Incomplete serious-issue evidence fields'})
        if conditions!={'RO':15,'U':15,'F':15}:issues.append({'rater':who,'issue':'Unbalanced restaurant/condition coverage'})
        summary[who]=dict(assigned=45,submitted=statuses['SUBMITTED'],drafts=statuses['DRAFT'],not_started=45-len(rows),condition_counts=dict(conditions),numeric_scores=numeric,RO_visual_NA=na)
    if integrity!='ok' or audit['status']!='PASS':issues.append({'issue':'SQLite or event journal integrity failed'})
    old=ROOT/'outputs/stage_o/human_collection_status_20261105_v1/A01_BACKUP_RECEIPT.json'
    prior_comparison='NOT_AVAILABLE'
    if old.exists():
        oldpath=Path(read(old)['file']);assert sha(oldpath)==read(old)['sha256']
        with contextlib.closing(sqlite3.connect('file:'+oldpath.as_posix()+'?mode=ro',uri=True)) as c:
            previous=c.execute("SELECT alias,rater,version,payload FROM ratings WHERE rater='A01' ORDER BY alias").fetchall()
        current=[tuple(r[k] for k in ['alias','rater','version','payload']) for r in allrows if r['rater']=='A01']
        prior_comparison='UNCHANGED' if current==previous else 'DIFFERENT'
        if current!=previous:issues.append({'issue':'A01 differs from previously verified completion backup'})
    report=dict(status='PASS' if not issues else 'ATTENTION',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),raters=summary,
        stored_rating_records=len(allrows),event_records=events,SQLite_integrity_check=integrity,journal_status=audit['status'],journal_errors=audit['errors'],
        issues=issues,A01_vs_prior_completion_backup=prior_comparison,rating_changes=0,artificial_ratings_created=0,Qwen_started=False,
        scope='Completeness, ownership, score ranges, Visual applicability, evidence-field validity, timestamps and journal integrity; not semantic regrading or independent participant authentication')
    put(D/'COMPLETION_VERIFICATION.json',report)
    if issues:print(json.dumps(report,indent=2));raise SystemExit('ATTENTION: backup preserved; final export not created')
    payload=study.export({'id':'COORDINATOR','role':'coordinator'},final=True)
    assert payload['final'] is True and len(payload['records'])==90 and payload['synthetic'] is False
    export=D/'STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json';put(export,payload)
    assert len(read(export)['records'])==90
    # Original trial declaration and actual access waiver are retained separately.
    user_auth=APP/'WORKSPACE'/payload['direct_access_authorization']['authorization_record']
    put(D/'PROVENANCE/DIRECT_ACCESS_AUTHORIZATION_SOURCE.json',read(user_auth))
    put(D/'PROVENANCE/CONFIG.json',read(APP/'WORKSPACE/CONFIG.json'))
    put(D/'PROVENANCE/RUBRIC.json',read(APP/'WORKSPACE/RUBRIC.json'))
    receipt=dict(status='PASS_COMPLETE_90_SUBMITTED_RECORDS',export_file=export.name,export_sha256=sha(export),export_bytes=export.stat().st_size,
        backup_file='PRIVATE_BACKUP/ratings.sqlite3',backup_sha256=sha(target),backup_bytes=target.stat().st_size,
        config_sha256=payload['config_sha256'],protocol_sha256=payload['amendment_sha256'],cohort_freeze_sha256=payload['cohort_freeze_sha256'],
        contains_private_hashed_accounts_in_backup=True,backup_not_for_distribution=True,ratings_changed=0,analysis_or_Qwen_started=False)
    put(D/'EXPORT_AND_BACKUP_RECEIPT.json',receipt)
    print(json.dumps(report,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
