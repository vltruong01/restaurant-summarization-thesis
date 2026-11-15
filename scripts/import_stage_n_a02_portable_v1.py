"""Validate an A02 portable return; import only with explicit --apply, never overwrite."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sqlite3

from stage_n_human_evaluation_v1 import CRITERIA
from s1000_final_eval_core_v1 import read, digest, file_hash


def need(ok, message):
    if not ok: raise ValueError(message)


def blank(item):
    return {'scores':{c:'N/A' if c==CRITERIA[-1] and item['visual_text'] is None else None for c in CRITERIA},
        'evidence_notes':'','serious_grounding_issue':False,'issue_span':'','evidence_source_ids':[],
        'issue_certainty':'none','status':'NOT_STARTED','submitted_at_utc':''}


def validate_payload(item, source, payload, action):
    need(set(payload)==set(blank(item)), 'Unexpected/missing rating fields')
    need(action in ['save','submit'] and payload['status']==('SUBMITTED' if action=='submit' else 'DRAFT'), 'Action/status mismatch')
    scores=payload['scores'];need(isinstance(scores,dict) and set(scores)==set(CRITERIA), 'Six original criteria required')
    for c,value in scores.items():
        if c==CRITERIA[-1] and item['visual_text'] is None: need(value=='N/A','Structural Visual N/A mismatch')
        else: need(type(value) is int and 1<=value<=5 or action=='save' and value is None,'Invalid applicable score')
    notes=payload['evidence_notes'];span=payload['issue_span'];ids=payload['evidence_source_ids'];flag=payload['serious_grounding_issue'];certainty=payload['issue_certainty']
    need(isinstance(notes,str) and len(notes)<=20000 and isinstance(span,str),'Invalid notes/span')
    need(type(flag) is bool and certainty in ['none','uncertain','confirmed'],'Invalid issue fields')
    valid=set(source['selected_review_ids'])|{p['photo_id'] for p in source['photos']}
    need(isinstance(ids,list) and all(isinstance(x,str) and x in valid for x in ids),'Foreign recipient evidence ID')
    if flag and action=='submit':
        need(notes.strip() and span and span in item['review_text']+'\n'+(item['visual_text'] or '') and ids and certainty!='none','Incomplete serious-grounding evidence')
    if action=='submit':
        t=dt.datetime.fromisoformat(payload['submitted_at_utc']);need(t.tzinfo is not None,'Timezone missing')
    else: need(payload['submitted_at_utc']=='','Draft has submission timestamp')


def validate_return(workspace, distribution, returned):
    root=Path(workspace);dist=Path(distribution);path=Path(returned)
    need(path.stat().st_size<30_000_000,'Return file too large')
    value=read(path);h=value.get('content_sha256');without={k:v for k,v in value.items() if k!='content_sha256'}
    need(h==digest(without),'Return checksum mismatch')
    need(value.get('format')=='stage_n_A02_portable_return_v1' and value.get('rater')=='A02' and value.get('final') is True,'Final A02 return required')
    binding=read(dist/'DATA/BINDING.json');baseline=read(dist/'DATA/BASELINE.json')
    need(value['binding']==binding and value['binding_sha256']==file_hash(dist/'DATA/BINDING.json'),'Distribution binding mismatch')
    need(value['baseline_sha256']==file_hash(dist/'DATA/BASELINE.json')==binding['baseline_sha256'],'Baseline binding mismatch')
    for rel,key in [('CONFIG.json','original_config_sha256'),('COORDINATOR_ONLY/PRIVATE_ITEMS.json','original_assignment_key_sha256'),('RUBRIC.json','original_rubric_sha256'),('COHORT30.json','original_cohort_sha256'),('LIVE_RELEASE.json','original_release_sha256'),('DIRECT_RATER_ACCESS.json','original_direct_access_sha256')]:
        need(file_hash(root/rel)==binding[key], 'Original freeze/release changed: '+rel)
    cfg=read(root/'CONFIG.json');key=read(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')
    items={x['alias']:x for x in key if x['annotator']=='A02'}
    need(len(items)==binding['expected_records'],'Unexpected assignment count')
    sources={b:read(root/v['file']) for b,v in cfg['sources'].items()}
    need({b:digest(s) for b,s in sources.items()}==binding['source_canonical_sha256'],'Original sources changed')
    package_items=read(dist/'DATA/ITEMS.json')
    need(file_hash(dist/'DATA/ITEMS.json')==binding['blinded_items_sha256'],'Blinded assignment file changed')
    for p in package_items:
        need(p['alias'] in items and all(items[p['alias']][k]==v for k,v in p.items()),'Aliases/texts differ from original')
    need(len(package_items)==len(items) and {p['alias'] for p in package_items}==set(items),'Blinded assignment coverage')
    events=value['events'];seed=baseline['events']
    need(events[:len(seed)]==seed,'Preserved baseline journal changed')
    versions={};states={};previous=None
    for event in events:
        a=event['alias'];need(a in items and event['rater']=='A02','Foreign journal item/rater')
        old=states.get(a,blank(items[a]));need(old['status']!='SUBMITTED','Post-submission modification refused')
        need(event['previous_event_sha256']==previous and event['prior_state_sha256']==digest(old),'Journal hash/prior-state mismatch')
        need(type(event['version']) is int and event['version']==versions.get(a,0)+1,'Journal version mismatch')
        t=dt.datetime.fromisoformat(event['recorded_at_utc']);need(t.tzinfo is not None,'Event timezone missing')
        validate_payload(items[a],sources[items[a]['business_id']],event['payload'],event['action'])
        states[a]=event['payload'];versions[a]=event['version'];previous=digest(event)
    rows=value['records'];need(len(rows)==len(items) and {r['alias'] for r in rows}==set(items),'Complete unique panel required')
    for r in rows:
        need(set(r)=={'alias','rater','version','payload'} and r['rater']=='A02','Return row schema/rater')
        need(r['payload']['status']=='SUBMITTED' and r['payload']==states.get(r['alias']) and r['version']==versions[r['alias']],'Return row/journal mismatch')
    baseline_rows=sorted(baseline['ratings'],key=lambda r:r['alias'])
    need(digest(baseline_rows)==baseline['original_a02_states_sha256'],'Baseline snapshot changed')
    return value,baseline,items


def db_rows(c,rater):
    return [{'alias':r['alias'],'rater':r['rater'],'version':r['version'],'payload':json.loads(r['payload'])}
        for r in c.execute('SELECT * FROM ratings WHERE rater=? ORDER BY alias',(rater,))]


def process(workspace, distribution, returned, apply=False):
    root=Path(workspace).resolve();path=Path(returned).resolve()
    value,baseline,items=validate_return(root,distribution,path)
    returned_sha=file_hash(path);db=root/'PRIVATE_STATE/ratings.sqlite3'
    c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=30);c.row_factory=sqlite3.Row
    try:
        c.execute('BEGIN')
        current=db_rows(c,'A02');a01_before=digest(db_rows(c,'A01'))
        imported=[json.loads(r['event_json']) for r in c.execute('SELECT event_json FROM events')
            if json.loads(r['event_json']).get('portable_origin',{}).get('return_file_sha256')==returned_sha]
    finally:c.rollback();c.close()
    final_rows=sorted(value['records'],key=lambda r:r['alias'])
    if imported:
        need(current==final_rows,'Previously imported return conflicts with current A02 state')
        return {'status':'ALREADY_IMPORTED','return_sha256':returned_sha,'records':len(final_rows),'no_changes':True}
    need(current==sorted(baseline['ratings'],key=lambda r:r['alias']), 'A02 changed on the original site after distribution; leave both copies intact for coordinator reconciliation')
    report={'status':'VALIDATED_NOT_IMPORTED','rater':'A02','records':len(final_rows),'new_journal_events':len(value['events'])-len(baseline['events']),
        'return_sha256':returned_sha,'binding_sha256':value['binding_sha256'],'A01_state_sha256':a01_before,
        'participant_identity_independently_authenticated':False,'timestamps':'Participant-device timestamps preserved; not independent authentication'}
    if not apply:return report
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_dir=root/'COORDINATOR_ONLY/PORTABLE_RETURNS'/returned_sha;receipt_dir.mkdir(parents=True,exist_ok=True)
    raw=receipt_dir/'ORIGINAL_A02_RETURN.json'
    if raw.exists():need(file_hash(raw)==returned_sha,'Receipt file conflict')
    else:
        with raw.open('xb') as f:f.write(path.read_bytes())
    backup=receipt_dir/('PRE_IMPORT_'+stamp+'.sqlite3')
    with sqlite3.connect(db) as source,sqlite3.connect(backup) as target:source.backup(target)
    c=sqlite3.connect(db,timeout=30);c.row_factory=sqlite3.Row
    try:
        c.execute('BEGIN IMMEDIATE')
        need(db_rows(c,'A02')==current,'A02 changed during import; no records imported')
        a01_locked=digest(db_rows(c,'A01'))
        last=c.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone();previous=last[0] if last else None
        imported_at=dt.datetime.now(dt.timezone.utc).isoformat()
        for event in value['events'][len(baseline['events']):]:
            e=copy.deepcopy(event)
            e['previous_event_sha256']=previous
            e['portable_origin']={'return_file_sha256':returned_sha,'portable_event_sha256':digest(event),
                'binding_sha256':value['binding_sha256'],'imported_at_utc':imported_at,
                'recording_location':'A02 participant computer; original device timestamps retained'}
            c.execute('INSERT INTO events(event_json,event_hash) VALUES(?,?)',(json.dumps(e),digest(e)));previous=digest(e)
            c.execute('INSERT OR REPLACE INTO ratings VALUES(?,?,?,?)',(e['alias'],'A02',e['version'],json.dumps(e['payload'])))
        need(db_rows(c,'A02')==final_rows,'Imported state differs from exact return')
        need(digest(db_rows(c,'A01'))==a01_locked,'A01 changed unexpectedly')
        c.commit()
    except BaseException:c.rollback();raise
    finally:c.close()
    report.update(status='IMPORTED',imported_at_utc=imported_at,private_pre_import_backup_sha256=file_hash(backup),A01_unchanged=True)
    with (receipt_dir/('IMPORT_RECEIPT_'+stamp+'.json')).open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path);p.add_argument('--distribution',required=True,type=Path);p.add_argument('--return-file',required=True,type=Path)
    p.add_argument('--apply',action='store_true',help='Explicitly import a validated complete return; default is read-only validation')
    args=p.parse_args();print(json.dumps(process(args.workspace,args.distribution,args.return_file,args.apply),indent=2))
