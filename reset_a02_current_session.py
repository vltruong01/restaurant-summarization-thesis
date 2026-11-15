import json,sqlite3,hashlib,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path.cwd().resolve(); sys.path.insert(0,str(ROOT/'scripts'))
from human_app_core_v1 import digest
stage=ROOT/'outputs/stage_p'; live=stage/'three_system_human_evaluation_live_20261106_v1'; complete=stage/'human_collection_complete_20261106T070046Z_v1'; hand=stage/'completed_human_evaluation_handoff_20261106_v1'; local=ROOT/'outputs/a02_eval45_8768_20261106'
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def remove(p):
 p=p.resolve();assert p.is_relative_to(ROOT/'outputs')
 if p.is_file():p.unlink();removed.append(str(p.relative_to(ROOT)))
removed=[];dbresults=[]
for db in [live/'STATE/ratings.sqlite3',complete/'PRIVATE_BACKUP/ratings.sqlite3']:
 with sqlite3.connect(db) as c:
  c.execute('PRAGMA secure_delete=ON');c.execute('BEGIN IMMEDIATE')
  before=c.execute("SELECT * FROM ratings WHERE rater!='A02' ORDER BY alias").fetchall()
  accounts=c.execute('SELECT * FROM users ORDER BY id').fetchall()
  count=c.execute("SELECT count(*) FROM ratings WHERE rater='A02'").fetchone()[0]
  c.execute("DELETE FROM ratings WHERE rater='A02'")
  previous=None;deleted=0
  for seq,raw in c.execute('SELECT seq,event_json FROM events ORDER BY seq').fetchall():
   event=json.loads(raw)
   if event['rater']=='A02':c.execute('DELETE FROM events WHERE seq=?',(seq,));deleted+=1;continue
   event['previous_event_sha256']=previous;previous=digest(event)
   c.execute('UPDATE events SET event_json=?,event_hash=? WHERE seq=?',(json.dumps(event),previous,seq))
  assert before==c.execute("SELECT * FROM ratings WHERE rater!='A02' ORDER BY alias").fetchall()
  assert accounts==c.execute('SELECT * FROM users ORDER BY id').fetchall()
  c.commit();c.execute('PRAGMA wal_checkpoint(TRUNCATE)');c.execute('VACUUM')
  assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
  dbresults.append({'database':str(db.relative_to(ROOT)),'deleted_A02_ratings':count,'deleted_A02_events':deleted,'other_raters_and_accounts_unchanged':True})
for base in [complete,hand/'PACKAGE/RATINGS']:
 remove(base/'A02_FINAL_45.json')
 p=base/'RATING_EVENT_JOURNAL.json';events=json.loads(p.read_text());write(p,[e for e in events if e['rater']!='A02'])
 p=base/'STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json';d=json.loads(p.read_text());d['records']=[r for r in d['records'] if r['annotator']!='A02'];d.update(final=False,record_count=len(d['records']),submitted_count=len(d['records']),invalidated_final_artifact=True,reset_notice='A02 results deleted by explicit user request; waiting for revised criteria.',records_sha256=digest(d['records']));d['audit']={'status':'RESET_PENDING_NEW_RUBRIC','stored_rating_records':len(d['records'])};write(p,d)
for pattern in ['batch*.json','submitted_*.json','FINAL_VERIFIED_A02_45.json','KET_QUA_A02_8768.md']:
 for p in local.glob(pattern):remove(p)
# Source snapshots may embed ratings; retain sources but remove all saved assessment payloads.
for p in local.glob('item_*.json'):
 d=json.loads(p.read_text(encoding='utf-8-sig'));d.pop('rating',None);d.pop('version',None);write(p,d)
remove(local/'items.json')
report={'status':'RESET_A02_PENDING_NEW_RUBRIC','reset_at_utc':datetime.now(timezone.utc).isoformat(),'authorization':'User explicitly requested deleting old A02 results before new criteria.','scope':'Current Stage P / port 8768 and its completion, handoff and local scoring copies; other historical sessions unchanged.','A02':{'submitted':0,'draft':0,'numeric_scores':0,'comments':0},'collection_complete':False,'new_ratings_created':0,'other_raters_and_sources_preserved':True,'database_reset':dbresults,'journal_reset_disclosed':True,'removed_files':removed,'next_task_optional_evidence_comments':'Leave blank; awaiting new user instructions.'}
for p in [live/'STATE/A02_RESET_VERIFICATION_20261106.json',complete/'A02_RESET_VERIFICATION_20261106.json',complete/'COMPLETION_VERIFICATION.json',hand/'PACKAGE/EVIDENCE/COMPLETION_VERIFICATION.json',hand/'DELIVERY_VALIDATION.json',hand/'PACKAGE/PACKAGE_MANIFEST.json']:
 write(p,report)
for p in [complete/'README.md',hand/'PACKAGE/README.md',hand/'PACKAGE/PROMPT_FOR_CHATGPT.txt']:
 p.write_text('A02 results reset by user request. Collection is incomplete; previous completion claims are withdrawn. Waiting for revised criteria and a new grading request. Source reviews, photos and frozen outputs are preserved. See A02 reset verification.\n',encoding='utf-8')
write(complete/'ARTIFACT_HASHES.json',{'status':'RESET_PENDING_NEW_RUBRIC','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in complete.glob('*.json') if p.name!='ARTIFACT_HASHES.json'}})
z=hand/'STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
 for p in (hand/'PACKAGE').rglob('*'):
  if p.is_file():archive.write(p,p.relative_to(hand))
Path(str(z)+'.sha256').write_text(hashlib.sha256(z.read_bytes()).hexdigest()+'  '+z.name+'\n')
print(json.dumps({'databases':dbresults,'removed_files_count':len(removed),'status':report['status']}))
