"""A02-only portable continuation; reuses the original rubric, UI and score validator."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import webbrowser

from stage_n_human_evaluation_v1 import Study, Handler, Server, CRITERIA
from stage_l_reference_app_v1 import need, safe_under, now
from s1000_final_eval_core_v1 import read, digest, file_hash


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


class PortableStudy(Study):
    def __init__(self, root):
        self.root = Path(root).resolve(); self.preparation_login = False
        self.backup_lock = threading.Lock()
        manifest = read(self.root/'MANIFEST.json')
        for rel, h in manifest['files'].items():
            need(file_hash(safe_under(self.root, rel)) == h, 'Package file changed: '+rel)
        self.binding = read(self.root/'DATA/BINDING.json')
        self.rubric = read(self.root/'DATA/RUBRIC.json')
        need([c['id'] for c in self.rubric['criteria']] == CRITERIA, 'Rubric mismatch')
        self.sources = read(self.root/'DATA/SOURCES.json')
        self.key = read(self.root/'DATA/ITEMS.json')
        self.items = {x['alias']: x for x in self.key}
        need(len(self.items) == self.binding['expected_records'], 'Assignment count mismatch')
        need(sorted(x['position'] for x in self.key) == list(range(1, len(self.key)+1)), 'Item order mismatch')
        need(all(x['annotator'] == 'A02' and 'condition' not in x for x in self.key), 'A02-only blinded data required')
        self.config = {'expected_records': len(self.key), 'synthetic': self.binding['synthetic']}
        self.db = self.root/'STATE/ratings.sqlite3'; self.db.parent.mkdir(exist_ok=True)
        with self.connect() as c:
            c.executescript('''
                CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,role TEXT NOT NULL,salt TEXT NOT NULL,password TEXT NOT NULL,created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS participants(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ratings(alias TEXT PRIMARY KEY,rater TEXT NOT NULL,version INTEGER NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_json TEXT NOT NULL,event_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS portable_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            ''')
            c.execute('BEGIN IMMEDIATE')
            package_id = file_hash(self.root/'DATA/BINDING.json')
            old = c.execute("SELECT value FROM portable_meta WHERE key='package_id'").fetchone()
            if old:
                need(old[0] == package_id, 'This STATE belongs to another package; do not mix folders')
            else:
                need(not c.execute('SELECT alias FROM ratings LIMIT 1').fetchone(), 'Unbound nonempty state refused')
                auth = read(self.root/'DATA/ACCOUNT.json')
                need(auth['id'] == 'A02' and auth['role'] == 'rater', 'Only A02 login allowed')
                c.execute('INSERT INTO users VALUES(?,?,?,?,?)', tuple(auth[k] for k in ['id','role','salt','password','created']))
                seed = read(self.root/'DATA/BASELINE.json')
                for row in seed['ratings']:
                    need(row['rater'] == 'A02' and row['alias'] in self.items, 'Foreign baseline rating')
                    c.execute('INSERT INTO ratings VALUES(?,?,?,?)', (row['alias'], row['rater'], row['version'], json.dumps(row['payload'])))
                for event in seed['events']:
                    c.execute('INSERT INTO events(event_json,event_hash) VALUES(?,?)', (json.dumps(event), digest(event)))
                c.execute('INSERT INTO portable_meta VALUES(?,?)', ('package_id', package_id))
        need(self.audit()['status'] == 'PASS', 'Stored journal failed integrity check')

    def active(self): return True

    def require_eligible(self, actor):
        need(actor == {'id':'A02','role':'rater'}, 'A02 session required')

    def public_item(self, actor, alias):
        result = super().public_item(actor, alias)
        for photo in result['photos']:
            photo['url'] = f'/api/photo/{alias}/{photo["photo_id"]}'
        return result

    def return_export(self, final=False):
        # One consistent read transaction for both materialized rows and journal.
        with self.connect() as c:
            c.execute('BEGIN')
            rows = [{'alias':r['alias'], 'rater':r['rater'], 'version':r['version'], 'payload':json.loads(r['payload'])}
                    for r in c.execute('SELECT * FROM ratings ORDER BY alias')]
            events = [json.loads(r['event_json']) for r in c.execute('SELECT * FROM events ORDER BY seq')]
        if final:
            need(len(rows) == len(self.items) and all(r['payload']['status'] == 'SUBMITTED' for r in rows), 'Complete all 120 items before final export')
        need(all(r['rater'] == 'A02' for r in rows), 'Foreign rater records refused')
        result = {'format':'stage_n_A02_portable_return_v1','rater':'A02','final':final,
            'binding_sha256':file_hash(self.root/'DATA/BINDING.json'), 'binding':self.binding,
            'baseline_sha256':file_hash(self.root/'DATA/BASELINE.json'),
            'exported_at_utc':now(), 'records':rows, 'events':events,
            'timestamp_limitation':'Portable timestamps are recorded by the participant computer; its clock and human identity are not independently authenticated.'}
        result['content_sha256'] = digest(result)
        return result

    def checkpoint(self):
        value = self.return_export()
        atomic_json(self.root/'RESULTS/A02_PROGRESS_BACKUP.json', value)
        if len(value['records']) == len(self.items) and all(r['payload']['status'] == 'SUBMITTED' for r in value['records']):
            value = self.return_export(final=True)
            atomic_json(self.root/'RESULTS/A02_RESULTS_FINAL.json', value)
        return value

    def apply(self, actor, alias, action, body):
        # The normal validator, transaction, version conflict and submitted lock are unchanged.
        result = super().apply(actor, alias, action, body)
        with self.backup_lock:
            try: self.checkpoint()
            except OSError as error:
                # SQLite was already committed. Keep the success truthful; do not invite re-submission.
                print('Rating committed; JSON backup needs disk attention: '+str(error), flush=True)
        return result


class PortableHandler(Handler):
    def context(self):
        need(self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'], 'Invalid Host')
        match = re.search(r'(?:^|;\s*)stage_n_A02_portable=([a-f0-9]+)(?:;|$)', self.headers.get('Cookie',''))
        key = match[1] if match else ''; session = self.server.sessions.get(key)
        if session and session['expires'] < time.time(): session = None
        return key, session

    def respond(self, code, value, content_type='application/json; charset=utf-8', cookie=None):
        if cookie: cookie = cookie.replace('stage_n_session=', 'stage_n_A02_portable=')
        return super().respond(code, value, content_type, cookie=cookie)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path not in ['/api/session','/api/export']:
            return super().do_GET()
        try:
            _, session = self.context(); need(session is not None, 'Login required')
            self.server.store.require_eligible(session['actor'])
            if path == '/api/session':
                return self.respond(200, {'actor':session['actor'],'csrf':session['csrf'],'participant':None,
                    'rubric':self.server.store.rubric,'live_scoring_released':True,'direct_rating_access':True})
            final = urlsplit(self.path).query == 'final=1'
            value = self.server.store.return_export(final)
            if final: atomic_json(self.server.store.root/'RESULTS/A02_RESULTS_FINAL.json', value)
            return self.respond(200, value)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self.respond(400, {'error':str(error)})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ['/api/login','/api/logout'] and not path.startswith('/api/rating/'):
            return self.respond(404, {'error':'A02 rating routes only'})
        return super().do_POST()


class PortableServer(Server):
    def __init__(self, address, study, web):
        from http.server import ThreadingHTTPServer
        self.store = study; self.web = Path(web); self.sessions = {}; self.lock = threading.Lock(); self.failures = {}
        study.backup_lock = threading.Lock()
        ThreadingHTTPServer.__init__(self, address, PortableHandler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify', action='store_true'); p.add_argument('--no-browser', action='store_true')
    p.add_argument('--port', type=int)
    args = p.parse_args(); study = PortableStudy(args.root)
    if args.verify:
        print(json.dumps({'status':'PASS','items':len(study.items),'businesses':len(study.sources),'photos':sum(len(s['photos']) for s in study.sources.values()),'audit':study.audit()}, indent=2)); return
    server = None
    for port in ([args.port] if args.port is not None else [8766,8768,8769,8770]):
        try: server = PortableServer(('127.0.0.1',port),study,args.root/'WEB'); break
        except OSError as error:
            if getattr(error,'winerror',None) != 10048 and error.errno not in [48,98,10048]: raise
    need(server is not None, 'Local study ports are occupied; close the other portable copy and try again')
    address = f'http://127.0.0.1:{server.server_port}/'
    print('A02 portable study: '+address+'\nSign in with the existing A02 account. Keep this window open.\nScores are saved in STATE; final file: RESULTS/A02_RESULTS_FINAL.json', flush=True)
    study.checkpoint()
    if not args.no_browser: threading.Timer(0.8,lambda:webbrowser.open(address)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
