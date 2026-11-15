"""Standalone stdlib verifier: reads package only; no ratings or opinions written."""
from pathlib import Path
import collections,hashlib,json
root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=read('PACKAGE_MANIFEST.json')
for row in manifest['files']:
    p=(root/row['file']).resolve();assert p.is_relative_to(root) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['file']
x=read('STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json');receipt=read('EXPORT_AND_BACKUP_RECEIPT.json')
assert sha(root/'STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json')==receipt['export_sha256']
assert x['final'] is True and x['synthetic'] is False and len(x['records'])==90 and x['audit']['status']=='PASS'
assert sha(root/'PROVENANCE/CONFIG.json')==x['config_sha256'] and sha(root/'PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md')==x['amendment_sha256']
config=read('PROVENANCE/CONFIG.json');assert config['expected_records']==90 and config['conditions']==['RO','U','F'] and len(config['sources'])==15
assert config['cohort_freeze_sha256']==x['cohort_freeze_sha256']
assignment=read('ASSIGNMENT_MANIFEST_90.json');by={r['alias']:r for r in assignment};assert len(by)==90
criteria=[r['id'] for r in read('PROVENANCE/RUBRIC.json')['criteria']];numeric=na=0
exports={r['alias']:r for r in x['records']};assert set(exports)==set(by)
for who in ['A01','A02']:
    rows=[r for r in x['records'] if r['annotator']==who];assert len(rows)==45
    assert {(r['business_id'],r['condition']) for r in rows}=={(b,c) for b in config['sources'] for c in config['conditions']}
    assert sorted(r['position'] for r in rows)==list(range(1,46))
for r in x['records']:
    assert r['status']=='SUBMITTED' and r['submitted_at_utc']
    assert all(r[k]==by[r['alias']][k] for k in ['annotator','position','business_id','condition'])
    for criterion in criteria:
        if criterion==criteria[-1] and r['condition']=='RO':assert r[criterion]=='N/A';na+=1
        else:assert type(r[criterion]) is int and 1<=r[criterion]<=5;numeric+=1
journal=read('RATING_EVENT_JOURNAL_138.json');states={};versions={};head=None
assert len(journal)==138
for row in journal:
    e=json.loads(row['event_json']);a=e['alias'];assert a in by
    assert digest(e)==row['event_hash'] and e['previous_event_sha256']==head
    assert e['rater']==by[a]['annotator'] and e['version']==versions.get(a,0)+1
    if a in states:assert states[a]['status']!='SUBMITTED' and e['prior_state_sha256']==digest(states[a])
    else:
        initial=dict(scores={k:'N/A' if k==criteria[-1] and by[a]['condition']=='RO' else None for k in criteria},evidence_notes='',serious_grounding_issue=False,issue_span='',evidence_source_ids=[],issue_certainty='none',status='NOT_STARTED',submitted_at_utc='')
        assert e['prior_state_sha256']==digest(initial)
    states[a]=e['payload'];versions[a]=e['version'];head=row['event_hash']
raw=read('RAW_RATING_ROWS_90.json');assert len(raw)==90 and set(states)==set(exports)
for row in raw:
    a=row['alias'];p=json.loads(row['payload']);r=exports[a]
    assert row['rater']==r['annotator'] and row['version']==versions[a]==r['version'] and p==states[a]
    for k in criteria:assert p['scores'][k]==r[k]
    for k in ['status','submitted_at_utc','evidence_notes','issue_span','evidence_source_ids','issue_certainty','serious_grounding_issue']:assert p[k]==r[k]
assert head==x['audit']['event_chain_head_sha256'] and numeric==510 and na==30
for file in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json']:
    record=read('PROVENANCE/'+file);name=Path(record['authorization_record']).name
    assert sha(root/'PROVENANCE'/name)==record['authorization_record_sha256']
print(json.dumps(dict(status='PASS',members_verified=len(manifest['files']),restaurants=15,submitted_records=90,A01=45,A02=45,numeric_scores=510,RO_visual_NA=30,
    journal_events=138,event_chain_reconstructed=True,raw_payloads_match_export=True,participant_identity_or_source_inspection_independently_authenticated=False,ratings_or_statistics_modified=False),indent=2))
