"""Read-only completion/integrity check; no model comparison or rating changes."""
import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sqlite3

from s1000_final_eval_core_v1 import read, digest, file_hash
from stage_n_human_evaluation_v1 import CRITERIA, RATERS

def blank(item):
    return {'scores':{k:'N/A' if k==CRITERIA[-1] and item['visual_text'] is None else None for k in CRITERIA},
        'evidence_notes':'','serious_grounding_issue':False,'issue_span':'','evidence_source_ids':[],
        'issue_certainty':'none','status':'NOT_STARTED','submitted_at_utc':''}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--rater',choices=RATERS,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.workspace.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh audit output directory')
    config=read(root/'CONFIG.json');key=read(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json');all_items={r['alias']:r for r in key}
    expected={a:r for a,r in all_items.items() if r['annotator']==args.rater};sources={b:read(root/v['file']) for b,v in config['sources'].items()}
    binding_errors=[]
    for name,h in config['file_bindings'].items():
        if file_hash(root/name)!=h:binding_errors.append(name)
    for b,v in config['sources'].items():
        if digest(sources[b])!=v['canonical_sha256']:binding_errors.append('source:'+b)
    out.mkdir(parents=True);private=out/'PRIVATE';private.mkdir()
    db=root/'PRIVATE_STATE/ratings.sqlite3'
    conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=15);conn.row_factory=sqlite3.Row
    try:
        conn.execute('BEGIN')
        quick=[r[0] for r in conn.execute('PRAGMA quick_check')]
        db_rows=[dict(r) for r in conn.execute('SELECT alias,rater,version,payload FROM ratings ORDER BY rater,alias')]
        events=[dict(r) for r in conn.execute('SELECT seq,event_json,event_hash FROM events ORDER BY seq')]
        participants=[dict(r) for r in conn.execute('SELECT id,payload FROM participants ORDER BY id')]
        backup=private/'ratings_snapshot.sqlite3'
        with sqlite3.connect(backup) as target:conn.backup(target)
    finally:conn.rollback();conn.close()
    rows=[r for r in db_rows if r['rater']==args.rater];actual={r['alias']:r for r in rows}
    issues=[];journal_errors=[];states={};versions={};previous=None
    for r in events:
        try:
            e=json.loads(r['event_json']);a=e['alias'];item=all_items.get(a)
            if item is None:journal_errors.append({'event':r['seq'],'issue':'Unknown assignment'});continue
            old=states.get(a,blank(item))
            if digest(e)!=r['event_hash'] or e['previous_event_sha256']!=previous:journal_errors.append({'event':r['seq'],'issue':'Event hash chain'})
            if e['version']!=versions.get(a,0)+1 or e['rater']!=item['annotator']:journal_errors.append({'event':r['seq'],'issue':'Version/rater mismatch'})
            if e['prior_state_sha256']!=digest(old) or old['status']=='SUBMITTED':journal_errors.append({'event':r['seq'],'issue':'Invalid or post-submission transition'})
            if e['action'] not in ['save','submit'] or e['payload']['status']!=('SUBMITTED' if e['action']=='submit' else 'DRAFT'):journal_errors.append({'event':r['seq'],'issue':'Action/status mismatch'})
            previous=r['event_hash'];states[a]=e['payload'];versions[a]=e['version']
        except (KeyError,TypeError,ValueError) as error:journal_errors.append({'event':r['seq'],'issue':'Malformed event: '+str(error)})
    if set(states)!={r['alias'] for r in db_rows}:journal_errors.append({'issue':'Rating/journal item set mismatch'})
    for r in db_rows:
        if states.get(r['alias'])!=json.loads(r['payload']) or versions.get(r['alias'])!=r['version']:journal_errors.append({'alias':r['alias'],'issue':'Stored row differs from journal'})
    missing=[expected[a]['position'] for a in set(expected)-set(actual)];unexpected=sorted(set(actual)-set(expected))
    pair_counts=Counter((r['business_id'],r['condition']) for r in expected.values());business_counts=Counter(r['business_id'] for r in expected.values())
    cohort=read(root/'COHORT30.json')['business_ids']
    assignment_ok=(len(expected)==120 and len(pair_counts)==120 and all(v==1 for v in pair_counts.values()) and set(business_counts)==set(cohort) and all(v==4 for v in business_counts.values()))
    status=Counter();numeric=0;na=0;notes=0;serious=0;visual=Counter();timestamps=[];checked=[];snapshot=[]
    for row in rows:
        item=expected.get(row['alias']);p=json.loads(row['payload']);errors=[]
        if item is None:issues.append({'alias':row['alias'],'errors':['Unexpected assignment']});continue
        status[p.get('status','MISSING')]+=1
        if p.get('status')!='SUBMITTED':errors.append('Not submitted')
        try:
            t=dt.datetime.fromisoformat(p['submitted_at_utc'])
            if t.tzinfo is None:raise ValueError('Timezone missing')
            timestamps.append(t)
        except (KeyError,TypeError,ValueError):errors.append('Missing or invalid actual submission timestamp')
        scores=p.get('scores',{})
        if set(scores)!=set(CRITERIA):errors.append('Score fields differ from the six-criterion rubric')
        for criterion in CRITERIA:
            value=scores.get(criterion)
            if criterion==CRITERIA[-1] and item['visual_text'] is None:
                if value!='N/A':errors.append('Inapplicable Visual must be N/A')
                else:na+=1
            elif type(value) is not int or not 1<=value<=5:errors.append('Invalid/missing applicable score: '+criterion)
            else:numeric+=1
        v=item['visual_text'];visual['not_applicable' if v is None else 'empty_applicable' if not v.removeprefix('[VISUAL_EVIDENCE]\n').strip() else 'nonempty']+=1
        note=p.get('evidence_notes');span=p.get('issue_span');ids=p.get('evidence_source_ids');certainty=p.get('issue_certainty');flag=p.get('serious_grounding_issue')
        if not isinstance(note,str) or len(note)>20000:errors.append('Invalid comment')
        elif note.strip():notes+=1
        if not isinstance(span,str):errors.append('Invalid evidence span')
        if type(flag) is not bool or certainty not in ['none','uncertain','confirmed']:errors.append('Invalid issue flag/certainty')
        src=sources[item['business_id']];valid_ids=set(src['selected_review_ids'])|{photo['photo_id'] for photo in src['photos']}
        if not isinstance(ids,list) or any(not isinstance(x,str) or x not in valid_ids for x in ids):errors.append('Evidence IDs are outside recipient source')
        if flag is True:
            serious+=1
            if not (isinstance(note,str) and note.strip() and isinstance(span,str) and span and span in item['review_text']+'\n'+(v or '') and isinstance(ids,list) and ids and certainty in ['uncertain','confirmed']):errors.append('Serious grounding flag lacks required exact span/source IDs/explanation/certainty')
        if errors:issues.append({'alias':row['alias'],'position':item['position'],'business_id':item['business_id'],'errors':errors})
        checked.append({'alias':row['alias'],'position':item['position'],'business_id':item['business_id'],'status':p.get('status'),'record_validation':'PASS' if not errors else 'FAIL'})
        snapshot.append({'alias':row['alias'],'annotator':row['rater'],'position':item['position'],'business_id':item['business_id'],'version':row['version'],'rating':p})
    snapshot.sort(key=lambda r:r['position']);snapshot_path=private/(args.rater+'_SUBMISSIONS_SNAPSHOT.json')
    snapshot_path.write_text(json.dumps({'rater':args.rater,'records':snapshot,'scope':'Read-only snapshot; no scores rewritten; no model comparison'},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tz=dt.timezone(dt.timedelta(hours=7));valid=(quick==['ok'] and not binding_errors and assignment_ok and not missing and not unexpected and len(rows)==120 and not issues and not journal_errors and numeric==690 and na==30)
    progress=Counter((r['rater'],json.loads(r['payload']).get('status')) for r in db_rows)
    provenance={name:read(root/name) for name in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json'] if (root/name).exists()}
    report={'status':'PASS' if valid else 'FAIL','rater':args.rater,'checked_at_Asia_Saigon':dt.datetime.now(dt.timezone.utc).astimezone(tz).isoformat(),
        'expected_records':120,'stored_records':len(rows),'status_counts':dict(status),'businesses':len({r['business_id'] for r in snapshot}),'four_assignments_per_business':assignment_ok,
        'valid_numeric_scores':numeric,'expected_numeric_scores':690,'correct_structural_NA':na,'expected_structural_NA':30,'visual_component_counts':dict(visual),
        'nonempty_comment_records':notes,'serious_grounding_flags':serious,'missing_positions':sorted(missing),'unexpected_aliases':unexpected,'record_validation_issues':issues,
        'sqlite_quick_check':quick,'journal_validation':'PASS' if not journal_errors else 'FAIL','journal_issues':journal_errors,'journal_events':len(events),'journal_head_sha256':previous,
        'frozen_binding_validation':'PASS' if not binding_errors else 'FAIL','frozen_binding_errors':binding_errors,
        'all_rater_progress_as_of_snapshot':[{'rater':r,'status':st,'records':n} for (r,st),n in sorted(progress.items())],
        'submission_timestamps_Asia_Saigon':{'first':min(timestamps).astimezone(tz).isoformat() if timestamps else None,'last':max(timestamps).astimezone(tz).isoformat() if timestamps else None,'not_interpreted_as_active_work_duration':True},
        'checks_by_item':checked,'preserved_private_snapshot_sha256':file_hash(snapshot_path),'preserved_private_DB_backup_sha256':file_hash(backup),
        'config_sha256':file_hash(root/'CONFIG.json'),'private_assignment_key_sha256':file_hash(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json'),
        'release_and_direct_access_provenance':provenance,'read_only_database_access':True,'ratings_changed':False,'scores_imputed':False,'model_quality_comparison_run':False,
        'limitations':['Completion and technical consistency of recorded submissions do not independently authenticate participant identity, source inspection, honesty or semantic correctness of judgments.',
            'Participant declaration/coordinator gate was waived on direct researcher instruction; do not infer absent personal declarations.',
            'Two-rater agreement and paired model comparison require the complete genuine two-rater panel and have not been run here.']}
    report_path=out/(args.rater+'_COMPLETION_CHECK.json');report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    concise={k:v for k,v in report.items() if k not in ['checks_by_item','release_and_direct_access_provenance']}
    print(json.dumps(concise,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
