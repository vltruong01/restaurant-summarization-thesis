"""Validate and atomically import exact researcher-supplied A02 revisions."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,shutil,sqlite3,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/'outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
SOURCE=Path(r'C:\Users\VUONGLOCTRUONG\Downloads\A02_REVISED_RESULTS_FINAL.json')
BASELINE=ROOT/'outputs/stage_p/a02_portable_revision_20261106_v1/STAGE_P_A02_REVISION_APP/WORKSPACE/A02_BASELINE_FINAL_45.json'
sys.path.insert(0,str(LIVE/'APP'))
from stage_p_human_evaluation_v1 import Study
from stage_p_revision_provenance_v1 import CRITERIA,EXTRA,digest,payload
from stage_p_revision_aware_study_v1 import RevisionAwareStudy
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def rows(study,rater):
    with study.connect() as db:return [tuple(x) for x in db.execute('SELECT * FROM ratings WHERE rater=? ORDER BY alias',(rater,))]

def validate(study,data,baseline):
    assert data['format']=='stage_p_a02_portable_revision_v1' and data['final'] is True and data['rater']=='A02'
    for key,value in [('study_id',study.config['study_id']),('study_format',study.config['format']),('config_sha256',sha(LIVE/'WORKSPACE/CONFIG.json')),('blinded_items_sha256',sha(LIVE/'WORKSPACE/BLINDED_ITEMS.json')),('amendment_sha256',sha(LIVE/'WORKSPACE/TWO_RATER_PROTOCOL_AMENDMENT.md'))]:assert data[key]==baseline[key]==value,key
    assert data['synthetic'] is False and data['records_sha256']==digest(data['records'])
    revision=data['portable_revision'];assert revision['baseline_file_sha256']==sha(BASELINE) and revision['baseline_records_sha256']==baseline['records_sha256']==digest(baseline['records'])
    original={r['alias']:r for r in baseline['records']};current={r['alias']:r for r in data['records']}
    expected={r['alias'] for r in study.key if r['annotator']=='A02'}
    assert len(data['records'])==len(current)==len(original)==45 and set(current)==set(original)==expected
    for alias,row in current.items():
        item=study.items[alias];old=original[alias]
        for key in ['alias','annotator','position','business_id','condition']:assert row[key]==old[key]
        assert row['annotator']=='A02' and row['business_id']==item['business_id'] and row['position']==item['position'] and row['condition']==item['condition']
        assert row['status']=='SUBMITTED' and type(row['version']) is int and row['version']>=1
        datetime.fromisoformat(row['submitted_at_utc'])
        assert all(type(row[c]) is int and 1<=row[c]<=5 for c in CRITERIA)
        assert type(row['serious_grounding_issue']) is bool and isinstance(row['evidence_notes'],str) and len(row['evidence_notes'])<=20000
        s=study.sources[row['business_id']];allowed=set(s['selected_review_ids'])|{p['photo_id'] for p in s['photos']}
        assert isinstance(row['evidence_source_ids'],list) and set(row['evidence_source_ids'])<=allowed
        assert row['issue_certainty'] in ['none','uncertain','confirmed']
        if row['serious_grounding_issue']:assert row['evidence_notes'].strip() and row['issue_span'] and row['issue_span'] in item['review_text']+'\n'+item['visual_text'] and row['evidence_source_ids'] and row['issue_certainty']!='none'
        stored,_=study.rating({'id':'A02','role':'rater'},alias);assert stored==payload(old),'Conflict: coordinator A02 changed after portable baseline'
    states={};versions={};previous=None
    for event in revision['revision_journal']:
        alias=event['alias'];assert alias in expected and event['rater']=='A02' and event['action'] in ['save','submit']
        assert event['previous_event_sha256']==previous and event['version']==versions.get(alias,0)+1
        prior=states.get(alias)
        if prior is None:prior=payload(original[alias]);prior['status']='BASELINE';prior['submitted_at_utc']=''
        else:assert prior['status']!='SUBMITTED'
        assert event['prior_state_sha256']==digest(prior)
        assert event['payload']['status']==('SUBMITTED' if event['action']=='submit' else 'DRAFT')
        previous=digest(event);states[alias]=event['payload'];versions[alias]=event['version']
    assert len(states)==revision['newly_submitted_count']==45 and revision['inherited_original_count']==0
    assert set(revision['newly_saved_or_submitted_aliases'])==expected and not revision['retained_original_aliases']
    for alias,state in states.items():assert state==payload(current[alias]) and versions[alias]==current[alias]['version']
    remote_audit=revision['journal_audit'];assert remote_audit['status']=='PASS' and remote_audit['event_chain_head_sha256']==previous
    fields=CRITERIA+EXTRA[:5]
    changes=[dict(alias=r['alias'],business_id=r['business_id'],fields={k:dict(before=original[r['alias']][k],after=r[k]) for k in fields if original[r['alias']][k]!=r[k]}) for r in data['records'] if any(original[r['alias']][k]!=r[k] for k in fields)]
    assert {r['alias']:r for r in changes}=={r['alias']:r for r in revision['changes']}
    return changes

def apply(study,data):
    with study.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        for row in data['records']:
            alias=row['alias'];old,version=study.rating({'id':'A02','role':'rater'},alias,db)
            baseline=next(r for r in read(BASELINE)['records'] if r['alias']==alias)
            assert old==payload(baseline),'Conflict during transaction'
            last=db.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            event=dict(alias=alias,rater='A02',version=version+1,action='import_a02_portable_revision',recorded_at_utc=now(),payload=payload(row),prior_state_sha256=digest(old),previous_event_sha256=last[0] if last else None,
                import_provenance=dict(operator_action='researcher_authorized_file_import',source_file_sha256=sha(SOURCE),baseline_file_sha256=sha(BASELINE),source_record_version=row['version'],authority='Researcher supplied the A02 returned file and explicitly asked to save it into the local website. This is an administrative import, not a new local human submission.'))
            db.execute('UPDATE ratings SET version=?,payload=? WHERE alias=? AND rater=?',(version+1,json.dumps(event['payload']),alias,'A02'))
            assert db.execute('SELECT changes()').fetchone()[0]==1
            db.execute('INSERT INTO events(event_json,event_hash) VALUES(?,?)',(json.dumps(event),digest(event)))

def main():
    data=read(SOURCE);baseline=read(BASELINE);study=RevisionAwareStudy(LIVE/'WORKSPACE',state_dir=LIVE/'STATE')
    assert study.audit()['status']=='PASS'
    changes=validate(study,data,baseline)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'outputs/stage_p'/('a02_revision_import_'+stamp+'_v1');out.mkdir()
    source_sha=sha(SOURCE);baseline_sha=sha(BASELINE)
    shutil.copy2(SOURCE,out/'A02_REVISED_RESULTS_FINAL.json');shutil.copy2(BASELINE,out/'A02_BASELINE_FINAL_45.json')
    private=out/'PRIVATE_BACKUP';private.mkdir()
    with study.connect() as src,sqlite3.connect(private/'ratings_before.sqlite3') as dst:src.backup(dst)
    a01_before=rows(study,'A01');old_events=[]
    with study.connect() as db:old_events=[tuple(x) for x in db.execute('SELECT * FROM events ORDER BY seq')]
    # First replay the exact import into an isolated copy, never synthetic ratings.
    with tempfile.TemporaryDirectory(prefix='stage_p_real_revision_shadow_') as td:
        shadow=Path(td);shutil.copy2(private/'ratings_before.sqlite3',shadow/'ratings.sqlite3');(shadow/'IMPORTS').mkdir()
        shutil.copy2(SOURCE,shadow/'IMPORTS'/(source_sha+'.json'));shutil.copy2(BASELINE,shadow/'IMPORTS'/(baseline_sha+'.json'))
        trial=RevisionAwareStudy(LIVE/'WORKSPACE',state_dir=shadow);apply(trial,data);assert trial.audit()['status']=='PASS'
        assert rows(trial,'A01')==a01_before
        for row in data['records']:assert trial.rating({'id':'A02','role':'rater'},row['alias'])[0]==payload(row)
        assert len(trial.export({'id':'COORDINATOR','role':'coordinator'},final=True)['records'])==90
    imports=LIVE/'STATE/IMPORTS';imports.mkdir(exist_ok=True)
    for source,binding in [(SOURCE,source_sha),(BASELINE,baseline_sha)]:
        target=imports/(binding+'.json')
        if target.exists():assert sha(target)==binding
        else:shutil.copy2(source,target)
    apply(study,data)
    audit=study.audit();assert audit['status']=='PASS' and audit['verified_portable_import_transitions']==45
    assert rows(study,'A01')==a01_before
    with study.connect() as db:assert [tuple(x) for x in db.execute('SELECT * FROM events ORDER BY seq LIMIT ?',(len(old_events),))]==old_events
    for row in data['records']:assert study.rating({'id':'A02','role':'rater'},row['alias'])[0]==payload(row)
    summary=dict(status='PASS',imported_at_utc=now(),source_file_sha256=source_sha,baseline_file_sha256=baseline_sha,
        source_file='A02_REVISED_RESULTS_FINAL.json',imported_A02_records=45,content_changed_records=len(changes),
        changed_score_cells=sum(k in CRITERIA for r in changes for k in r['fields']),changed_comment_records=sum('evidence_notes' in r['fields'] for r in changes),
        A01_rows_byte_identical=True,prior_journal_prefix_unchanged=True,prior_event_count=len(old_events),current_event_count=len(old_events)+45,
        new_events_are_administrative_imports=True,original_remote_submission_timestamps_preserved=True,changes=changes,audit=audit,
        archive_folder=out.relative_to(ROOT).as_posix(),statistical_analysis_performed=False)
    save(out/'IMPORT_RECEIPT.json',summary);save(imports/(source_sha+'.receipt.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['changes','audit']},indent=2))

if __name__=='__main__':main()
