"""Local source-only human-reference workbench. No model or metric imports.

SQLite commits each revision and its hash-linked event atomically. Existing
S1000 reference contracts are reused for the Stage M handoff.
"""
import argparse
import copy
from contextlib import contextmanager, closing
import csv
import datetime as dt
import getpass
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote

from s1000_final_eval_core_v1 import (canonical, digest, file_hash, read,
    reference_content, validate_reference, diagnostics)

STATUSES = ['NOT_STARTED', 'DRAFT', 'SUBMITTED', 'NEEDS_REVISION', 'APPROVED']
SOURCE_KEYS = {'business_id','split','business_metadata','selected_review_ids',
    'selected_review_block','review_block_sha256','reviews','photos','photo_scope',
    'contains_predictions','contains_teacher_targets','captions_provided'}
ATTESTATIONS = ['human_written', 'sources_checked', 'no_predictions_or_teacher']
CHECK_ATTESTATIONS = ['independent_check', 'sources_checked',
    'no_predictions_or_teacher', 'coverage_and_component_separation_checked']

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def need(ok, message):
    if not ok: raise ValueError(message)
def json_bytes(x): return (json.dumps(x, ensure_ascii=False, indent=2)+'\n').encode('utf8')
def new_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: f.write(json_bytes(obj)); f.flush(); os.fsync(f.fileno())
def safe_under(root, rel):
    root=Path(root).resolve(); p=(root/rel).resolve()
    need(p.is_relative_to(root) and p!=root, 'Path outside allowed source root')
    return p

class Store:
    def __init__(self, workspace):
        self.root=Path(workspace).resolve(); self.config=read(self.root/'CONFIG.json')
        self.package=self.root/'SOURCE_PACKAGE'
        need(file_hash(self.package/'ANNOTATION_PACKAGE_MANIFEST.json')==self.config['source_manifest_sha256'], 'Source manifest changed')
        self.manifest=read(self.package/'ANNOTATION_PACKAGE_MANIFEST.json')
        self.frame={r['business_id']:r for r in self.manifest['frozen_sources']}
        need(len(self.frame)==len(self.manifest['frozen_sources'])==self.config['expected_businesses'], 'Cohort count')
        self.photo_index={}; self.sources={}
        for b,row in self.frame.items():
            s=read(safe_under(self.package,row['source_file']))
            need(digest(s)==row['sha256'] and s['business_id']==b and s['split']=='test', 'Frozen source mismatch')
            need(set(s)==SOURCE_KEYS and not any(s[k] for k in ['contains_predictions','contains_teacher_targets','captions_provided']), 'Source-only contract')
            need(s['selected_review_ids']==[r['review_id'] for r in s['reviews']], 'Review ID/order mismatch')
            need(len(set(s['selected_review_ids']))==len(s['reviews']), 'Duplicate review ID')
            need(all(r['business_id']==b for r in s['reviews']), 'Foreign review')
            need(hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256'], 'Review block changed')
            need(len({p['photo_id'] for p in s['photos']})==len(s['photos']), 'Duplicate photo ID')
            for ph in s['photos']:
                need(ph['sha256']==ph['original_source_sha256'], 'Photo differs from original source digest')
                self.photo_index[b,ph['photo_id']]=ph
            self.sources[b]=s
        self.db=self.root/'PRIVATE_STATE'/'references.sqlite3'
        self.db.parent.mkdir(exist_ok=True)
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, role TEXT NOT NULL, salt TEXT NOT NULL, password TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS refs(business TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, business TEXT NOT NULL, event_json TEXT NOT NULL, event_hash TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=15); c.row_factory=sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON'); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA synchronous=FULL')
        try:
            with c: yield c
        finally:c.close()

    def source(self,b):
        need(b in self.sources,'Unknown business'); return self.sources[b]

    def blank(self,b):
        r=read(self.package/'TEST100_REFERENCE_TEMPLATE'/f'{b}.json')
        need(r['business_id']==b and r['source_bundle_sha256']==self.frame[b]['sha256'],'Blank template source binding changed')
        need(r['status']=='NOT_STARTED' and not any(r[k] for k in ['review_consensus','visual_evidence','review_claims','visual_claims','history','notes']), 'Template must remain blank')
        need(not r['author']['human_id'] and not r['checker']['human_id'] and not r['author']['human_written_attestation'] and not r['checker']['independent_check_attestation'],'Template contains human provenance')
        return r

    def get(self,b,c=None):
        self.source(b)
        if c is None:
            with self.connect() as conn: return self.get(b,conn)
        row=c.execute('SELECT * FROM refs WHERE business=?',(b,)).fetchone()
        return (json.loads(row['payload']),row['version']) if row else (self.blank(b),0)

    def add_user(self,identity,role,password):
        need(re.fullmatch(r'[A-Za-z0-9_.-]{2,64}',identity) and role in ['author','checker','coordinator'],'Invalid ID or role')
        need(len(password)>=12,'Password must contain at least 12 characters')
        salt=secrets.token_hex(16); value=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),200000).hex()
        with self.connect() as c: c.execute('INSERT INTO users VALUES(?,?,?,?,?)',(identity,role,salt,value,now()))

    def login(self,identity,password):
        with self.connect() as c: row=c.execute('SELECT * FROM users WHERE id=?',(identity,)).fetchone()
        salt=bytes.fromhex(row['salt']) if row else b'unknown-user-salt'
        value=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,200000).hex()
        need(row is not None and hmac.compare_digest(row['password'],value),'Invalid credentials')
        return {'id':row['id'],'role':row['role']}

    def errors(self,r,approved=False):
        try:
            need(isinstance(r.get('review_claims'),list) and isinstance(r.get('visual_claims'),list),'Claim lists required')
            for k in ['review_claims','visual_claims']:
                need(all(isinstance(c,dict) and isinstance(c.get('output_span'),str) and isinstance(c.get('evidence'),list)
                    and all(isinstance(e,dict) for e in c['evidence']) for c in r[k]),'Malformed claim/evidence')
            errors=validate_reference(r,self.source(r['business_id']),approved)
            if any(e.get('direct_image_viewed') is not True for cl in r['visual_claims'] for e in cl['evidence']):
                errors.append('visual claim requires a truthful direct-image inspection confirmation')
            for k in ['review_consensus','visual_evidence']:
                if re.search(r'```|<\s*/?[A-Za-z][^>]*>|\[[A-Z_]{3,}\]',r.get(k,'')):
                    errors.append(k+': use plain component text, no embedded headings/HTML/code fences')
            return errors
        except (KeyError,TypeError,ValueError,AttributeError) as e: return ['Malformed reference: '+str(e)]

    def warnings(self,r):
        out=[]
        for key in ['review_consensus','visual_evidence']:
            d=diagnostics(r[key]); n=d['words']
            if n and (n<25 or n>450): out.append(f'{key}: {n} words, inspect length; no automatic cutoff')
            if d['repeated_3gram_fraction']>.25: out.append(key+': repeated trigrams exceed 25%; inspect manually')
        return out

    def apply(self,b,actor,action,body):
        self.source(b)
        need(action in ['save','submit','request_revision','reject','approve'],'Unknown action')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            r,version=self.get(b,c)
            need(type(body.get('expected_version')) is int and body['expected_version']==version,'Version conflict: reload; unsaved browser draft is retained')
            comment=body.get('comment','')
            need(isinstance(comment,str) and len(comment)<=20000,'Invalid comment')
            if action in ['save','submit']:
                need(actor['role']=='author','Author account required')
                need(r['status'] in ['NOT_STARTED','DRAFT','NEEDS_REVISION'],'Reference is locked for independent checking or already approved')
                need(not r['author']['human_id'] or r['author']['human_id']==actor['id'],'Assigned to another author')
                draft=body.get('draft',{})
                for key in ['review_consensus','visual_evidence','notes']:
                    value=draft.get(key,''); need(isinstance(value,str) and len(value)<=100000,'Invalid text'); r[key]=value
                for key in ['review_claims','visual_claims']:
                    need(isinstance(draft.get(key),list),'Claims must be arrays'); r[key]=draft[key]
                    need(all(isinstance(cl,dict) and isinstance(cl.get('output_span'),str) and isinstance(cl.get('evidence'),list)
                        and all(isinstance(e,dict) for e in cl['evidence']) for cl in r[key]),'Malformed claim structure; browser recovery retained')
                r['author']={'human_id':actor['id'],'at_utc':now(),'human_written_attestation':False,'predictions_seen':False}
                r['checker']=self.blank(b)['checker']; r['status']='DRAFT'
                if action=='submit':
                    need(all(body.get('attestations',{}).get(k) is True for k in ATTESTATIONS),'Author attestations required')
                    need(r['review_consensus'].strip() and r['visual_evidence'].strip() and r['review_claims'] and r['visual_claims'],'Both components and evidence are required')
                    need(not self.errors(r),'Invalid evidence or reference structure: '+str(self.errors(r)))
                    r['author']['human_written_attestation']=True; r['status']='SUBMITTED'
            else:
                need(actor['role']=='checker','Checker account required')
                need(r['status']=='SUBMITTED','Only a submitted reference may be checked')
                need(actor['id']!=r['author']['human_id'],'Author and checker must be different people')
                if action in ['request_revision','reject']:
                    need(comment.strip(),'Explain the requested correction or rejection')
                    r['status']='NEEDS_REVISION'; r['checker']=self.blank(b)['checker']
                else:
                    need(all(body.get('attestations',{}).get(k) is True for k in CHECK_ATTESTATIONS),'Checker attestations required')
                    r['checker']={'human_id':actor['id'],'at_utc':now(),'independent_check_attestation':True,
                        'predictions_seen':False,'reviewed_content_sha256':digest(reference_content(r))}
                    r['status']='APPROVED'
            content_hash=digest(reference_content(r)); timestamp=now()
            previous=c.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            event={'business_id':b,'version':version+1,'action':action,'actor':actor,'at_utc':timestamp,
                'comment':comment,'content_sha256':content_hash,'previous_event_sha256':previous[0] if previous else None,
                'attestations':body.get('attestations',{}),'prior_reference_sha256':digest(self.get(b,c)[0])}
            r['history'].append({'version':version+1,'content_sha256':content_hash,'supplied_by_human_id':actor['id'],
                'recorded_at_utc':timestamp,'event':action,'comment':comment,'parent_reference_state_sha256':event['prior_reference_sha256']})
            if action=='approve': need(not self.errors(r,True),'Final integrity rules: '+str(self.errors(r,True)))
            event['reference']=copy.deepcopy(r)
            c.execute('INSERT OR REPLACE INTO refs VALUES(?,?,?)',(b,version+1,json.dumps(r,ensure_ascii=False)))
            c.execute('INSERT INTO events(business,event_json,event_hash) VALUES(?,?,?)',(b,json.dumps(event,ensure_ascii=False),digest(event)))
        return {'reference':r,'version':version+1,'errors':self.errors(r),'warnings':self.warnings(r)}

    def photo_path(self,b,pid):
        need((b,pid) in self.photo_index,'Unknown photo binding')
        ph=self.photo_index[b,pid]
        if self.config.get('portable_photo_root'):
            path=safe_under(self.config['portable_photo_root'],pid+'.jpg')
        else: path=safe_under(self.config['project_root'],ph['path'])
        need(path.is_file() and path.stat().st_size==ph['bytes'] and file_hash(path)==ph['sha256'],'Photo unavailable or changed')
        return path

    def verify_sources(self,photos=True):
        reviews=0; photo_count=0
        for b,s in self.sources.items():
            row=self.frame[b]
            need(digest(read(safe_under(self.package,row['source_file'])))==row['sha256'],'Source changed on disk')
            need(len(s['reviews'])==row['reviews'] and len(s['photos'])==row['photos'],'Manifest count mismatch')
            for ph in s['photos']:
                if photos:
                    path=self.photo_path(b,ph['photo_id'])
                    from PIL import Image
                    with Image.open(path) as im: im.verify()
                    with Image.open(path) as im:
                        im.load()
                        need(im.width>0 and im.height>0,'Invalid original image dimensions')
                photo_count+=1
            reviews+=len(s['reviews'])
        need(reviews==self.config['expected_reviews'] and photo_count==self.config['expected_photos'],'Source totals mismatch')
        return {'status':'PASS','businesses':len(self.frame),'selected_reviews':reviews,'indexed_original_photos':photo_count,
            'all_photo_sha256_and_image_decode_checked':photos,'source_manifest_sha256':self.config['source_manifest_sha256'],
            'predictions_teacher_scores_exposed':False,'checked_at_utc':now()}

    def audit(self):
        errors=[]; prev=None; latest={}; versions={}
        with self.connect() as c:
            for row in c.execute('SELECT * FROM events ORDER BY seq'):
                ev=json.loads(row['event_json']); b=ev['business_id']
                if row['event_hash']!=digest(ev) or ev['previous_event_sha256']!=prev: errors.append('Audit hash chain')
                if ev['version']!=versions.get(b,0)+1 or ev['reference']['history'][-1]['content_sha256']!=digest(reference_content(ev['reference'])): errors.append('Audit version binding')
                prev=row['event_hash']; latest[b]=ev['reference']; versions[b]=ev['version']
            rows=c.execute('SELECT * FROM refs').fetchall()
            if {row['business'] for row in rows}!=set(latest):errors.append('Missing/extra state row')
            for row in rows:
                if json.loads(row['payload'])!=latest.get(row['business']) or row['version']!=versions.get(row['business']): errors.append('State differs from journal')
        counts={k:0 for k in STATUSES}; invalid={}; duplicates=[]; texts={}; authored=0
        for b in self.frame:
            r,v=self.get(b); counts[r['status']]+=1; authored+=bool(r['review_consensus'].strip() or r['visual_evidence'].strip())
            ee=self.errors(r,r['status']=='APPROVED')
            if ee:invalid[b]=ee
            text=' '.join((r['review_consensus']+'\n'+r['visual_evidence']).casefold().split())
            if text:
                if text in texts: duplicates.append([texts[text],b])
                texts[text]=b
        return {'status':'PASS_WORKFLOW_INTEGRITY' if not errors and not invalid and not duplicates else 'NOT_READY',
            'counts':counts,'nonempty_drafts':authored,'independently_approved_records':counts['APPROVED'],
            'journal_errors':errors,'reference_errors':invalid,'duplicate_reference_pairs':duplicates,
            'event_chain_head_sha256':prev,'semantic_factuality_certified':False,
            'identity_limitation':'Local accounts and truthful attestations do not independently authenticate two real people.'}

    def export(self,destination,final=False):
        dest=Path(destination); need(not dest.exists(),'Export destination must be new')
        report=self.audit()
        need(not report['journal_errors'],'Journal corruption')
        if final:
            need(report['status']=='PASS_WORKFLOW_INTEGRITY' and report['counts']['APPROVED']==len(self.frame),'All references must be independently approved and structurally valid')
            self.verify_sources(True)
        dest.mkdir(parents=True)
        annotations=dest/'ANNOTATIONS'; annotations.mkdir()
        refs=[]; clean=[]; tracker=[]; readable=[]
        for b in sorted(self.frame):
            r,v=self.get(b); refs.append(r); new_json(annotations/f'{b}.json',r)
            row={'business_id':b,'review_reference':r['review_consensus'],'visual_reference':r['visual_evidence'],
                'supporting_review_ids':sorted({e.get('review_id','') for cl in r['review_claims'] for e in cl.get('evidence',[])}),
                'supporting_photo_ids':sorted({e.get('photo_id','') for cl in r['visual_claims'] for e in cl.get('evidence',[])}),
                'author_identifier':r['author']['human_id'],'checker_identifier':r['checker']['human_id'],
                'approval_status':r['status'],'final_version':v,'source_manifest_sha256':self.config['source_manifest_sha256'],
                'reference_sha256':digest(reference_content(r)),'reference_file_sha256':file_hash(annotations/f'{b}.json')}
            clean.append(row); tracker.append({k:row[k] for k in ['business_id','approval_status','author_identifier','checker_identifier','final_version','reference_sha256','reference_file_sha256']})
            readable.append(f"# {b} | {r['status']} | version {v}\n\n[REVIEW_CONSENSUS]\n{r['review_consensus']}\n\n[VISUAL_EVIDENCE]\n{r['visual_evidence']}\n")
        new_json(dest/'WORKFLOW_VALIDATION.json',report); new_json(dest/'REFERENCES_INSPECTION.json',clean)
        (dest/'REFERENCES_READABLE.md').write_text('\n\n'.join(readable),encoding='utf8')
        with (dest/'REFERENCE_APPROVAL_TRACKER.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(tracker[0])); w.writeheader(); w.writerows(tracker)
        with self.connect() as c:
            with (dest/'REVISION_EVENTS.jsonl').open('w',encoding='utf8') as f:
                for row in c.execute('SELECT event_json,event_hash FROM events ORDER BY seq'):f.write(json.dumps({'event':json.loads(row[0]),'sha256':row[1]},ensure_ascii=False)+'\n')
        if final:
            from validate_s1000_human_references_v1 import validate_all
            _,legacy=validate_all(self.package,annotations,approved=True)
            need(not legacy['errors'],'Existing Stage M validator rejected references')
            new_json(dest/'APPROVED_REFERENCES.json',refs)
            new_json(dest/'REFERENCE_FREEZE.json',{'businesses':len(refs),'source_manifest_sha256':self.config['source_manifest_sha256'],
                'reference_file_sha256':file_hash(dest/'APPROVED_REFERENCES.json'),'human_written_and_second_checked':True,
                'objective_validation':legacy,'content_hashes':{r['business_id']:digest(reference_content(r)) for r in refs},'frozen_at_utc':now()})
        new_json(dest/'EXPORT_MANIFEST.json',{'final_frozen':final,'files':{str(p.relative_to(dest)).replace('\\','/'):file_hash(p) for p in sorted(dest.rglob('*')) if p.is_file()}})
        return report

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,store,web):
        self.store=store; self.web=Path(web); self.sessions={}; self.lock=threading.Lock(); self.failures={}
        super().__init__(address,Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass # Never log draft prose, tokens or passwords.
    def respond(self,status,data,ctype='application/json; charset=utf-8',cookie=None):
        raw=json_bytes(data) if ctype.startswith('application/json') else data
        self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:self.send_header('Set-Cookie',cookie)
        self.end_headers(); self.wfile.write(raw)

    def context(self):
        need(self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'],'Invalid Host')
        cookie=self.headers.get('Cookie',''); match=re.search(r'(?:^|;\s*)stage_l_session=([a-f0-9]+)(?:;|$)',cookie)
        key=match[1] if match else ''; session=self.server.sessions.get(key)
        if session and session['expires']<dt.datetime.now().timestamp():session=None
        return key,session

    def do_GET(self):
        try:
            key,session=self.context(); path=unquote(urlsplit(self.path).path)
            if path in ['/','/app.js','/app.css']:
                name={'/':'index.html','/app.js':'app.js','/app.css':'app.css'}[path]
                return self.respond(200,(self.server.web/name).read_bytes(),{'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/app.css':'text/css; charset=utf-8'}[path])
            need(session is not None,'Login required'); store=self.server.store
            if path=='/api/session': return self.respond(200,{'actor':session['actor'],'csrf':session['csrf']})
            if path=='/api/cases':
                cases=[]
                for b,s in store.sources.items():
                    r,v=store.get(b);cases.append({'business_id':b,'name':s['business_metadata'].get('name',b),'reviews':len(s['reviews']),'photos':len(s['photos']),'status':r['status'],'version':v,'author':r['author']['human_id']})
                return self.respond(200,{'cases':cases,'progress':store.audit()['counts']})
            if path.startswith('/api/case/'):
                b=path[len('/api/case/'):]; s=copy.deepcopy(store.source(b));r,v=store.get(b)
                s['photos']=[{'photo_id':p['photo_id'],'sha256':p['sha256'],'url':f'/api/photo/{b}/{p["photo_id"]}'} for p in s['photos']]
                return self.respond(200,{'source':s,'reference':r,'version':v,'errors':store.errors(r),'warnings':store.warnings(r)})
            if path.startswith('/api/photo/'):
                b,pid=path[len('/api/photo/'):].split('/');return self.respond(200,store.photo_path(b,pid).read_bytes(),'image/jpeg')
            if path=='/api/guidelines': return self.respond(200,(store.root/'HUMAN_REFERENCE_ANNOTATION_GUIDELINES.md').read_text('utf8'))
            return self.respond(404,{'error':'Not found'})
        except (ValueError,KeyError,FileNotFoundError) as e:self.respond(400,{'error':str(e)})

    def do_POST(self):
        try:
            key,session=self.context(); origin=self.headers.get('Origin')
            need(origin in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'],'Same-origin request required')
            need(self.headers.get('Content-Type','').split(';')[0]=='application/json','JSON required')
            n=int(self.headers.get('Content-Length','0'));need(0<n<=2000000,'Request size')
            body=json.loads(self.rfile.read(n)); path=urlsplit(self.path).path
            if path=='/api/login':
                stamp=dt.datetime.now().timestamp();failures=self.server.failures.setdefault(self.client_address[0],[])
                failures[:]=[x for x in failures if stamp-x<60];need(len(failures)<10,'Too many login attempts; wait one minute')
                try:actor=self.server.store.login(body.get('identity',''),body.get('password',''))
                except ValueError:failures.append(stamp);raise
                token=secrets.token_hex(32);session={'actor':actor,'csrf':secrets.token_hex(32),'expires':stamp+12*3600}
                self.server.sessions[token]=session
                return self.respond(200,{'actor':actor,'csrf':session['csrf']},cookie=f'stage_l_session={token}; HttpOnly; SameSite=Strict; Path=/')
            need(session is not None and hmac.compare_digest(self.headers.get('X-CSRF-Token',''),session['csrf']),'Session/CSRF validation failed')
            if path=='/api/logout':
                self.server.sessions.pop(key,None);return self.respond(200,{'ok':True},cookie='stage_l_session=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/')
            if path.startswith('/api/action/'):
                b,action=unquote(path[len('/api/action/'):]).split('/')
                return self.respond(200,self.server.store.apply(b,session['actor'],action,body))
            return self.respond(404,{'error':'Not found'})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:self.respond(409 if 'Version conflict' in str(e) else 400,{'error':str(e)})

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['serve','add-user','verify','export','backup']:
        p=sub.add_parser(command);p.add_argument('--workspace',type=Path,required=True)
        if command=='serve':p.add_argument('--port',type=int,default=8765);p.add_argument('--web',type=Path,default=Path(__file__).parent/'stage_l_reference_web_v1')
        if command=='add-user':p.add_argument('--id',required=True);p.add_argument('--role',choices=['author','checker','coordinator'],required=True)
        if command in ['export','backup']:p.add_argument('--destination',type=Path,required=True)
        if command=='export':p.add_argument('--final',action='store_true')
    args=parser.parse_args();store=Store(args.workspace)
    if args.command=='add-user':
        password=getpass.getpass('Choose password (12+ characters): ');need(password==getpass.getpass('Repeat password: '),'Passwords differ')
        store.add_user(args.id,args.role,password);print('Account registered. Coordinator must separately verify the real human identity and independence.')
    elif args.command=='serve':
        store.verify_sources(True)
        print('Source-only workbench: http://127.0.0.1:'+str(args.port)+' — Ctrl+C to stop. No model outputs or metrics are served.',flush=True)
        Server(('127.0.0.1',args.port),store,args.web).serve_forever()
    elif args.command=='verify':print(json.dumps({'sources':store.verify_sources(True),'workflow':store.audit()},indent=2))
    elif args.command=='export':print(json.dumps(store.export(args.destination,args.final),indent=2))
    elif args.command=='backup':
        need(not args.destination.exists(),'Backup file must be new');args.destination.parent.mkdir(parents=True,exist_ok=True)
        with store.connect() as source,closing(sqlite3.connect(args.destination)) as target:source.backup(target)
        print('Consistent SQLite backup SHA256: '+file_hash(args.destination))

if __name__=='__main__':main()
