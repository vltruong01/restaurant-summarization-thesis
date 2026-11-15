"""Local blinded two-rater study. Production scoring stays closed until release."""
import argparse
import copy
from contextlib import contextmanager, closing
import getpass
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from urllib.parse import unquote, urlsplit

from human_http_common_v1 import Handler as ReferenceHandler, Server as ReferenceServer
from human_http_common_v1 import need, now, new_json, json_bytes, safe_under
from human_app_core_v1 import read, digest, file_hash

RATERS = ['A01', 'A02']
CRITERIA = ['review_faithfulness', 'aspect_coverage', 'coherence', 'non_redundancy', 'usefulness', 'visual_grounding_and_contribution']


def live_release_valid(root):
    root=Path(root);p=root/'COLLECTION_OPEN.json'
    if not p.exists():return False
    r=read(p)
    return (r.get('operator_started_session') is True and r.get('config_sha256')==file_hash(root/'CONFIG.json')
        and r.get('protocol_sha256')==file_hash(root/'TWO_RATER_PROTOCOL_AMENDMENT.md') and bool(r.get('opened_at_utc')))

def direct_rater_access_valid(root,identity):
    return identity in RATERS and (read(Path(root)/'CONFIG.json').get('synthetic') is True or live_release_valid(root))


class Study:
    def __init__(self, workspace, preparation_login=False, preview=False, state_dir=None):
        self.preparation_login = preparation_login
        self.preview = preview
        self.root = Path(workspace).resolve(); self.config = read(self.root/'CONFIG.json')
        for name, expected in self.config.get('file_bindings', {}).items():
            need(file_hash(safe_under(self.root, name)) == expected, 'Frozen workspace file changed: '+name)
        self.rubric = read(self.root/'RUBRIC.json')
        need([r['id'] for r in self.rubric['criteria']] == CRITERIA, 'Rubric changed')
        self.key = read(self.root/'BLINDED_ITEMS.json')
        self.items = {r['alias']: r for r in self.key}
        need(len(self.items) == len(self.key) == self.config['expected_records'], 'Item coverage')
        need(set(r['annotator'] for r in self.key) == set(RATERS), 'Exactly two rater assignments required')
        self.sources = {}
        for b, binding in self.config['sources'].items():
            s = read(safe_under(self.root, binding['file']))
            need(digest(s) == binding['canonical_sha256'] and s['business_id'] == b, 'Source mismatch')
            need(s['selected_review_ids'] == [r['review_id'] for r in s['reviews']], 'Review order changed')
            self.sources[b] = s
        for item in self.key:
            need(item['annotator'] in RATERS and item['business_id'] in self.sources and item['condition'] in ['C1', 'C2', 'C3'], 'Private mapping invalid')
            need(hashlib.sha256(item['review_text'].encode()).hexdigest() == item['review_text_sha256'], 'Review output changed')
            need(isinstance(item['visual_text'],str), 'Both output components are applicable in this study')
            if item['visual_text'] is not None:
                need(hashlib.sha256(item['visual_text'].encode()).hexdigest() == item['visual_text_sha256'], 'Visual output changed')
        for rater in RATERS:
            assigned = [r for r in self.key if r['annotator'] == rater]
            expected = {(b, c) for b in self.sources for c in ['C1', 'C2', 'C3']}
            need(len(assigned) == len(expected) and {(r['business_id'], r['condition']) for r in assigned} == expected, 'Balanced rater coverage required')
            need(sorted(int(r['position']) for r in assigned) == list(range(1, len(assigned)+1)), 'Invalid presentation positions')
        if self.config.get('synthetic'):
            need(all(b.startswith('SYNTHETIC_') for b in self.sources), 'Synthetic unlock cannot contain genuine businesses')
        else:
            need(len(self.sources) == 15 and self.config['expected_records'] == 90, 'Frozen TripAdvisor15/90 design required')
        state = Path(state_dir).resolve() if state_dir else self.root/'PRIVATE_STATE'
        state.mkdir(parents=True, exist_ok=True)
        self.db = state/'ratings.sqlite3'
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, role TEXT NOT NULL, salt TEXT NOT NULL, password TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS participants(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ratings(alias TEXT PRIMARY KEY, rater TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_json TEXT NOT NULL,event_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS participant_declaration_history(seq INTEGER PRIMARY KEY AUTOINCREMENT,participant_id TEXT NOT NULL,event_json TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=15); c.row_factory = sqlite3.Row
        c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA synchronous=FULL')
        try:
            with c: yield c
        finally: c.close()

    def active(self):
        if self.config.get('synthetic'): return True
        return live_release_valid(self.root)

    def add_user(self, identity, role, password):
        need((role == 'rater' and identity in RATERS) or (role == 'coordinator' and identity == 'COORDINATOR'), 'Only planned participant/coordinator IDs allowed')
        need(len(password) >= 12, 'Use at least 12 password characters')
        with self.connect() as c:
            if c.execute('SELECT id FROM users WHERE id=?', (identity,)).fetchone(): return False
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 200000).hex()
            c.execute('INSERT INTO users VALUES(?,?,?,?,?)', (identity, role, salt, hashed, now()))
        return True

    def login(self, identity, password):
        with self.connect() as c: row = c.execute('SELECT * FROM users WHERE id=?', (identity,)).fetchone()
        salt = bytes.fromhex(row['salt']) if row else b'unknown-user-salt'
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200000).hex()
        need(row is not None and hmac.compare_digest(row['password'], hashed), 'Invalid credentials')
        need(row['role'] != 'rater' or self.active() or self.preparation_login or self.preview, 'This new session is not open for collection; use read-only preview until coordinator review')
        return {'id': row['id'], 'role': row['role']}

    def participant(self, identity, conn=None):
        if conn is None:
            with self.connect() as c: return self.participant(identity, c)
        row = conn.execute('SELECT payload FROM participants WHERE id=?', (identity,)).fetchone()
        return json.loads(row['payload']) if row else None

    def declarations(self, actor, body):
        need(not self.preview, 'Read-only preview: declarations and scoring are disabled')
        need(actor['role'] == 'rater' and self.active(), 'Released rater session required')
        need(body.get('consent') is True and body.get('task_familiarity') is True, 'Consent and task-familiarity confirmation required')
        need(body.get('prior_output_or_score_exposure') in ['no', 'yes', 'unsure'], 'Exposure declaration required')
        need(body.get('stage_l_involvement') in ['none', 'author', 'checker', 'both', 'unsure'], 'Stage L involvement declaration required')
        need(isinstance(body.get('conflict_of_interest'), str) and body['conflict_of_interest'].strip(), 'Declare conflict details or explicitly state none')
        payload = {k: body[k] for k in ['consent', 'task_familiarity', 'prior_output_or_score_exposure', 'stage_l_involvement', 'conflict_of_interest']}
        payload.update(declared_at_utc=now(), eligibility='PENDING_COORDINATOR_REVIEW', eligibility_reason='')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            need(not c.execute('SELECT alias FROM ratings WHERE rater=? LIMIT 1', (actor['id'],)).fetchone(), 'Declarations locked after rating work begins; contact coordinator')
            previous = self.participant(actor['id'], c)
            if previous is not None:
                history = {'actor_id': actor['id'], 'recorded_at_utc': now(), 'action': 'participant_updated_own_declaration', 'previous_payload': previous, 'replacement_payload': payload}
                c.execute('INSERT INTO participant_declaration_history(participant_id,event_json) VALUES(?,?)', (actor['id'], json.dumps(history)))
            c.execute('INSERT OR REPLACE INTO participants VALUES(?,?)', (actor['id'], json.dumps(payload)))
        return payload

    def declarations_editable(self, identity):
        if self.preview or identity not in RATERS: return False
        with self.connect() as c:
            return not c.execute('SELECT alias FROM ratings WHERE rater=? LIMIT 1', (identity,)).fetchone()

    def declaration_history(self):
        with self.connect() as c:
            return [{'sequence': r['seq'], 'participant_id': r['participant_id'], **json.loads(r['event_json'])} for r in c.execute('SELECT * FROM participant_declaration_history ORDER BY seq')]

    def eligibility(self, actor, identity, decision, reason):
        need(not self.preview, 'Read-only preview: eligibility decisions are disabled')
        need(actor['role'] == 'coordinator' and identity in RATERS, 'Coordinator required')
        need(decision in ['ELIGIBLE', 'INELIGIBLE'] and isinstance(reason, str) and reason.strip(), 'Document eligibility decision and limitations')
        with self.connect() as c:
            p = self.participant(identity, c); need(p is not None, 'Participant declarations missing')
            p.update(eligibility=decision, eligibility_reason=reason, coordinator_id=actor['id'], reviewed_at_utc=now())
            c.execute('UPDATE participants SET payload=? WHERE id=?', (json.dumps(p), identity))
        return p

    def require_eligible(self, actor):
        need(actor['role'] == 'rater' and self.active(), 'Live scoring not released')
        if direct_rater_access_valid(self.root, actor['id']): return
        p = self.participant(actor['id'])
        need(p and p['eligibility'] == 'ELIGIBLE', 'Participant declarations and coordinator eligibility review required')

    def require_viewer(self, actor):
        if self.preview:
            need(actor['role'] == 'rater', 'Assigned rater preview account required')
        else:
            self.require_eligible(actor)

    def owned(self, actor, alias):
        need(actor['role'] == 'rater' and alias in self.items and self.items[alias]['annotator'] == actor['id'], 'Item not available for this rater')
        return self.items[alias]

    def rating(self, actor, alias, conn=None):
        item = self.owned(actor, alias)
        if conn is None:
            with self.connect() as c: return self.rating(actor, alias, c)
        row = conn.execute('SELECT version,payload FROM ratings WHERE alias=? AND rater=?', (alias, actor['id'])).fetchone()
        if row: return json.loads(row['payload']), row['version']
        return {'scores': {k: 'N/A' if k == CRITERIA[-1] and item['visual_text'] is None else None for k in CRITERIA}, 'evidence_notes': '', 'serious_grounding_issue': False, 'issue_span': '', 'evidence_source_ids': [], 'issue_certainty': 'none', 'status': 'NOT_STARTED', 'submitted_at_utc': ''}, 0

    def apply(self, actor, alias, action, body):
        need(not self.preview, 'Read-only preview: rating writes are disabled')
        self.require_eligible(actor); item = self.owned(actor, alias)
        need(action in ['save', 'submit'], 'Invalid action')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE'); old, version = self.rating(actor, alias, c)
            need(type(body.get('expected_version')) is int and body['expected_version'] == version, 'Version conflict: reload; browser recovery retained')
            need(old['status'] != 'SUBMITTED', 'Submitted rating is locked; duplicate submission refused')
            draft = body.get('draft', {}); scores = draft.get('scores', {})
            need(set(scores) == set(CRITERIA), 'All six score fields required')
            for criterion, value in scores.items():
                if criterion == CRITERIA[-1] and item['visual_text'] is None:
                    need(value == 'N/A', 'Inapplicable Visual score must be N/A')
                else:
                    need((type(value) is int and 1 <= value <= 5) or (action == 'save' and value is None), 'Applicable scores must be integer1-5; no missing submission')
            notes = draft.get('evidence_notes', ''); span = draft.get('issue_span', ''); source_ids = draft.get('evidence_source_ids', [])
            serious = draft.get('serious_grounding_issue', False); certainty = draft.get('issue_certainty', 'none')
            need(isinstance(notes, str) and len(notes) <= 20000 and isinstance(span, str), 'Invalid notes')
            need(type(serious) is bool and certainty in ['none', 'uncertain', 'confirmed'], 'Invalid issue declaration')
            s = self.sources[item['business_id']]; allowed = set(s['selected_review_ids']) | {p['photo_id'] for p in s['photos']}
            need(isinstance(source_ids, list) and all(isinstance(x, str) and x in allowed for x in source_ids), 'Evidence IDs must belong to the recipient business')
            if serious and action == 'submit':
                need(notes.strip() and span and span in item['review_text']+'\n'+(item['visual_text'] or '') and source_ids and certainty != 'none', 'For a serious grounding issue provide exact output span, recipient evidence IDs, explanation and certainty')
            payload = {'scores': copy.deepcopy(scores), 'evidence_notes': notes, 'serious_grounding_issue': serious,
                'issue_span': span, 'evidence_source_ids': source_ids, 'issue_certainty': certainty,
                'status': 'SUBMITTED' if action == 'submit' else 'DRAFT', 'submitted_at_utc': now() if action == 'submit' else ''}
            previous = c.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            event = {'alias': alias, 'rater': actor['id'], 'version': version+1, 'action': action,
                'recorded_at_utc': now(), 'payload': payload, 'prior_state_sha256': digest(old), 'previous_event_sha256': previous[0] if previous else None}
            c.execute('INSERT OR REPLACE INTO ratings VALUES(?,?,?,?)', (alias, actor['id'], version+1, json.dumps(payload)))
            c.execute('INSERT INTO events(event_json,event_hash) VALUES(?,?)', (json.dumps(event), digest(event)))
        return {'rating': payload, 'version': version+1}

    def list_items(self, actor):
        need(actor['role'] == 'rater', 'Rater account required')
        result = []
        for item in sorted((x for x in self.key if x['annotator'] == actor['id']), key=lambda x: x['position']):
            rating, version = self.rating(actor, item['alias'])
            result.append({'alias': item['alias'], 'position': item['position'], 'business_name': self.sources[item['business_id']]['business_metadata'].get('name', ''), 'status': rating['status'], 'version': version})
        return result

    def public_item(self, actor, alias):
        self.require_viewer(actor); item = self.owned(actor, alias); s = self.sources[item['business_id']]
        rating, version = self.rating(actor, alias)
        body = item['visual_text']
        empty = body is not None and not (body[len('[VISUAL_EVIDENCE]\n'):] if body.startswith('[VISUAL_EVIDENCE]\n') else body).strip()
        return {'alias': alias, 'position': item['position'], 'business_id': item['business_id'],
            'business_name': s['business_metadata'].get('name', ''), 'context': s['business_metadata'],
            'reviews': s['reviews'], 'photos': [{'photo_id': p['photo_id'], 'url': f'/api/photo/{alias}/{p["photo_id"]}'} for p in s['photos']],
            'review_text': item['review_text'], 'visual_text': body, 'visual_applicable': body is not None,
            'applicable_visual_empty': empty, 'rating': rating, 'version': version, 'read_only': self.preview}

    def photo_path(self, actor, alias, photo_id):
        self.require_viewer(actor); item = self.owned(actor, alias)
        photo = next((p for p in self.sources[item['business_id']]['photos'] if p['photo_id'] == photo_id), None)
        need(photo is not None, 'Photo not in recipient evidence')
        path = safe_under(self.root, 'PHOTOS/'+photo_id+'.jpg')
        need(path.stat().st_size == photo['bytes'] and file_hash(path) == photo['sha256'] == photo['original_source_sha256'], 'Original photo unavailable or changed')
        return path

    def export(self, actor, final=False):
        need(actor['role'] == 'coordinator', 'Coordinator-only export')
        rows = []
        with self.connect() as c:
            for row in c.execute('SELECT * FROM ratings ORDER BY rater,alias'):
                item = self.items[row['alias']]; p = json.loads(row['payload'])
                rows.append({'alias': row['alias'], 'annotator': row['rater'], 'position': item['position'],
                    'business_id': item['business_id'], 'condition': item['condition'], **p['scores'],
                    'evidence_notes': p['evidence_notes'], 'issue_span': p['issue_span'],
                    'evidence_source_ids': p['evidence_source_ids'], 'issue_certainty': p['issue_certainty'],
                    'serious_grounding_issue': p['serious_grounding_issue'],
                    'status': p['status'], 'submitted_at_utc': p['submitted_at_utc'], 'version': row['version']})
        if final:
            need(not self.preview, 'Final export is unavailable in read-only preview')
            need(len(rows) == self.config['expected_records'] and all(r['status'] == 'SUBMITTED' for r in rows), 'Complete90 genuine submitted records required')
            need(self.audit()['status'] == 'PASS', 'Journal integrity failure')
            for identity in RATERS:
                p = self.participant(identity)
                need(direct_rater_access_valid(self.root, identity) or (p and p['eligibility'] == 'ELIGIBLE'), 'Eligible declarations or explicit researcher access authorization required')
        return {'final': final, 'records': rows, 'expected_records': self.config['expected_records'], 'synthetic': self.config.get('synthetic', False),
                'config_sha256': file_hash(self.root/'CONFIG.json'), 'amendment_sha256': file_hash(self.root/'TWO_RATER_PROTOCOL_AMENDMENT.md'), 'study_id':self.config['study_id'], 'blinded_items_sha256':file_hash(self.root/'BLINDED_ITEMS.json'),
                'exported_at_utc': now(), 'audit': self.audit(), 'participant_declarations': {r: self.participant(r) for r in RATERS},
                'release_provenance': read(self.root/'COLLECTION_OPEN.json') if (self.root/'COLLECTION_OPEN.json').is_file() else None,
            'direct_access_authorization': None,
            'reference_free': True, 'study_format': self.config.get('format'),
            'cohort_freeze_sha256': self.config.get('cohort_freeze_sha256'),
            'preview_mode': self.preview, 'participant_declaration_history': self.declaration_history()}

    def rater_export(self,actor,final=False):
        need(actor['role']=='rater','Rater account required')
        all_data=self.export({'id':'COORDINATOR','role':'coordinator'},False)
        rows=[r for r in all_data['records'] if r['annotator']==actor['id']]
        expected=sum(r['annotator']==actor['id'] for r in self.key)
        if final:
            need(not self.preview and self.active(),'Final export requires the collection session')
            need(len(rows)==expected and all(r['status']=='SUBMITTED' for r in rows),'Complete all assigned items before final export')
            need(self.audit()['status']=='PASS','Journal integrity failure')
        return {k:v for k,v in all_data.items() if k not in ['records','participant_declarations','participant_declaration_history','audit']}|{'records':rows,'records_sha256':digest(rows),'rater':actor['id'],'expected_records':expected,'final':final,'export_scope':'single_rater_blinded','record_count':len(rows),'submitted_count':sum(r['status']=='SUBMITTED' for r in rows)}

    def audit(self):
        problems = []; previous = None; versions = {}; states = {}; counts = {'DRAFT': 0, 'SUBMITTED': 0}
        with self.connect() as c:
            for row in c.execute('SELECT * FROM events ORDER BY seq'):
                e = json.loads(row['event_json']); a = e['alias']
                if digest(e) != row['event_hash'] or e['previous_event_sha256'] != previous: problems.append('Event hash chain')
                if e['version'] != versions.get(a, 0)+1 or e['rater'] != self.items[a]['annotator']: problems.append('Version/rater binding')
                if a in states and (e['prior_state_sha256'] != digest(states[a]) or states[a]['status'] == 'SUBMITTED'): problems.append('Invalid rating transition')
                previous = row['event_hash']; versions[a] = e['version']; states[a] = e['payload']
            rows = c.execute('SELECT * FROM ratings').fetchall()
            if set(states) != {r['alias'] for r in rows}: problems.append('State/journal item set')
            for row in rows:
                p = json.loads(row['payload']); counts[p['status']] += 1
                if p != states.get(row['alias']) or row['version'] != versions.get(row['alias']): problems.append('State/journal mismatch')
            users = c.execute('SELECT id,role FROM users').fetchall()
        return {'status': 'PASS' if not problems else 'FAIL', 'errors': problems, 'stored_rating_records': len(rows), 'counts': counts,
            'expected_records': self.config['expected_records'], 'live_scoring_released': self.active(), 'preview_mode': self.preview,
            'account_ids_and_roles': [dict(x) for x in users], 'event_chain_head_sha256': previous,
            'participant_identity_and_blindness_independently_verified': False}


class Handler(ReferenceHandler):
    def context(self):
        import re
        need(self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'], 'Invalid Host')
        match = re.search(r'(?:^|;\s*)stage_p_session=([a-f0-9]+)(?:;|$)', self.headers.get('Cookie', ''))
        key = match[1] if match else ''; session = self.server.sessions.get(key)
        if session and session['expires'] < time.time(): session = None
        return key, session

    def do_GET(self):
        try:
            _, session = self.context(); path = unquote(urlsplit(self.path).path); study = self.server.store
            if path in ['/', '/app.js', '/app.css']:
                name = {'/': 'index.html', '/app.js': 'app.js', '/app.css': 'app.css'}[path]
                return self.respond(200, (self.server.web/name).read_bytes(), {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8', '/app.css': 'text/css; charset=utf-8'}[path])
            if path == '/api/status': return self.respond(200, {'released': study.active(), 'preview': study.preview, 'synthetic': study.config.get('synthetic', False), 'planned_raters': 2, 'expected_records': study.config['expected_records'], 'businesses': len(study.sources)})
            need(session is not None, 'Login required'); actor = session['actor']
            if path == '/api/session': return self.respond(200, {'actor': actor, 'csrf': session['csrf'], 'participant': study.participant(actor['id']), 'rubric': study.rubric, 'preview_mode': study.preview, 'workspace_sha256': file_hash(study.root/'CONFIG.json'), 'study_id':study.config['study_id'], 'live_scoring_released': study.active(), 'direct_rating_access': actor['role']=='rater' and direct_rater_access_valid(study.root,actor['id']), 'declarations_editable': actor['role']=='rater' and study.declarations_editable(actor['id'])})
            if actor['role'] == 'rater': need(study.active() or study.preview, 'Collection is not open')
            if path == '/api/items': return self.respond(200, {'items': study.list_items(actor)})
            if path.startswith('/api/item/'): return self.respond(200, study.public_item(actor, path[len('/api/item/'):]))
            if path.startswith('/api/photo/'):
                alias, photo = path[len('/api/photo/'):].split('/')
                return self.respond(200, study.photo_path(actor, alias, photo).read_bytes(), 'image/jpeg')
            if path == '/api/export': return self.respond(200, study.rater_export(actor, final=urlsplit(self.path).query=='final=1'))
            if path.startswith('/api/coordinator/'):
                need(actor['role'] == 'coordinator', 'Coordinator-only route')
                if path == '/api/coordinator/mapping': return self.respond(200, [{k: r[k] for k in ['alias', 'annotator', 'position', 'business_id', 'condition']} for r in study.key])
                if path == '/api/coordinator/export': return self.respond(200, study.export(actor, final=urlsplit(self.path).query == 'final=1'))
                if path == '/api/coordinator/participants': return self.respond(200, {identity: study.participant(identity) for identity in RATERS})
            return self.respond(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error: self.respond(400, {'error': str(error)})

    def do_POST(self):
        try:
            key, session = self.context(); study = self.server.store
            need(self.headers.get('Origin') in [f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'], 'Same-origin required')
            need(self.headers.get('Content-Type', '').split(';')[0] == 'application/json', 'JSON required')
            n = int(self.headers.get('Content-Length', '0')); need(0 < n <= 200000, 'Request size')
            body = json.loads(self.rfile.read(n)); path = unquote(urlsplit(self.path).path)
            if path == '/api/login':
                failures = self.server.failures.setdefault(self.client_address[0], [])
                failures[:] = [x for x in failures if time.time()-x < 60]; need(len(failures) < 10, 'Too many login attempts')
                try: actor = study.login(body.get('identity', ''), body.get('password', ''))
                except ValueError: failures.append(time.time()); raise
                token = secrets.token_hex(32); session = {'actor': actor, 'csrf': secrets.token_hex(32), 'expires': time.time()+8*3600}
                self.server.sessions[token] = session
                return self.respond(200, {'actor': actor, 'csrf': session['csrf'], 'participant': study.participant(actor['id']), 'rubric': study.rubric}, cookie=f'stage_p_session={token}; HttpOnly; SameSite=Strict; Path=/')
            need(session is not None and hmac.compare_digest(self.headers.get('X-CSRF-Token', ''), session['csrf']), 'Session/CSRF validation failed')
            actor = session['actor']
            if path == '/api/logout':
                self.server.sessions.pop(key, None)
                return self.respond(200, {'ok': True}, cookie='stage_p_session=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/')
            if path == '/api/declarations': return self.respond(200, study.declarations(actor, body))
            if path == '/api/coordinator/eligibility': return self.respond(200, study.eligibility(actor, body['rater_id'], body['decision'], body['reason']))
            if path.startswith('/api/rating/'):
                alias, action = path[len('/api/rating/'):].split('/')
                return self.respond(200, study.apply(actor, alias, action, body))
            return self.respond(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.respond(409 if 'Version conflict' in str(error) else 400, {'error': str(error)})


class Server(ReferenceServer):
    def __init__(self, address, study, web):
        # Reuse the Stage L HTTP/security/session infrastructure with only N routes.
        self.store = study; self.web = Path(web); self.sessions = {}; self.lock = threading.Lock(); self.failures = {}
        from http.server import ThreadingHTTPServer
        ThreadingHTTPServer.__init__(self, address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve', 'add-user', 'setup-accounts', 'verify', 'export', 'backup'])
    parser.add_argument('--workspace', type=Path, required=True); parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--preparation-login', action='store_true', help='Allow authenticated rubric-only access while TripAdvisor outputs and scoring remain locked')
    parser.add_argument('--preview', action='store_true', help='Authenticated read-only source/output review; all study writes disabled')
    parser.add_argument('--state-dir', type=Path, help='Separate runtime state directory (use a temporary directory for software tests)')
    parser.add_argument('--id'); parser.add_argument('--role', choices=['rater', 'coordinator']); parser.add_argument('--destination', type=Path); parser.add_argument('--final', action='store_true')
    args = parser.parse_args(); study = Study(args.workspace, preparation_login=args.preparation_login, preview=args.preview, state_dir=args.state_dir)
    if args.command == 'serve':
        need(args.preview or study.active(), 'Live release is absent. Use --preview for read-only review; do not create a release without researcher approval.')
        print('Local study at http://127.0.0.1:'+str(args.port)+'; preview='+str(study.preview)+'; live release='+str(study.active()), flush=True)
        Server(('127.0.0.1', args.port), study, Path(__file__).parent/'stage_p_human_web_v1').serve_forever()
    elif args.command == 'add-user':
        first = getpass.getpass('Choose password (12+ characters): '); need(first == getpass.getpass('Repeat password: '), 'Passwords differ')
        print('Registered.' if study.add_user(args.id, args.role, first) else 'Account exists; password and records unchanged.')
    elif args.command == 'setup-accounts':
        words = 'Maple River Lantern Meadow Cedar Harbor Willow Cloud Garden Pebble Forest Sunlight Ocean Clover Silver Autumn Sparrow Summit Orchard Velvet Coral Amber Breeze Morning Acorn Birch Brook Cabin Cactus Cherry Comet Copper Cotton Daisy Dune Elm Fern Finch Flint Grove Hazel Indigo Island Jade Lake Laurel Leaf Lemon Lilac Lotus Lunar Mango Olive Opal Orchid Otter Peach Pearl Pine Plum Poppy Raven Reed Robin Sage Sand Shell Spruce Stone Sunset Swan Tulip Valley Violet Water Wheat Wind Wren'.split()
        generated = set()
        for identity, role in [('COORDINATOR', 'coordinator'), ('A01', 'rater'), ('A02', 'rater')]:
            password = '-'.join(secrets.choice(words) for _ in range(5))+'-'+str(secrets.randbelow(90)+10)
            while password in generated:
                password = '-'.join(secrets.choice(words) for _ in range(5))+'-'+str(secrets.randbelow(90)+10)
            generated.add(password)
            if study.add_user(identity, role, password):
                print(identity+' | '+password, flush=True)
            else:
                print(identity+' | existing account preserved; password is unchanged', flush=True)
        print('Record new passwords privately from this terminal. They are not written to reports or packages. Setup does not release scoring.', flush=True)
    elif args.command == 'verify': print(json.dumps(study.audit(), indent=2))
    elif args.command == 'backup':
        need(args.destination and not args.destination.exists(), 'New destination required'); args.destination.parent.mkdir(parents=True, exist_ok=True)
        with study.connect() as c, closing(sqlite3.connect(args.destination)) as target: c.backup(target)
        print('Private backup SHA256: '+file_hash(args.destination))
    else:
        need(args.destination, 'Destination required')
        actor = study.login('COORDINATOR', getpass.getpass('Coordinator password: '))
        payload = study.export(actor, final=args.final); new_json(args.destination, payload)
        print('Coordinator-only export saved; '+str(len(payload['records']))+' actual stored records.')


if __name__ == '__main__': main()
