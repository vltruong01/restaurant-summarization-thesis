"""Authorized one-time local release. Passwords exist only in memory/terminal."""
from pathlib import Path
import hashlib
import http.cookiejar
import json
import secrets
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / 'outputs/stage_p/three_system_human_evaluation_live_20261106_v1'
sys.path.insert(0, str(LIVE / 'APP'))
from stage_p_human_evaluation_v1 import Study
from human_app_core_v1 import file_hash
from human_http_common_v1 import now

workspace = LIVE / 'WORKSPACE'
marker = workspace / 'COLLECTION_OPEN.json'
assert not marker.exists(), 'Already released; do not reprovision or overwrite.'
study = Study(workspace, preview=False, state_dir=LIVE/'STATE')
with study.connect() as db:
    assert db.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0, 'Existing accounts preserved.'
    assert db.execute('SELECT COUNT(*) FROM ratings').fetchone()[0] == 0
words = 'River Maple Meadow Cedar Willow Ocean Clover Autumn Sparrow Orchard Amber Breeze Brook Cherry Daisy Finch Hazel Laurel Lemon Lotus Olive Orchid Otter Peach Pearl Pine Poppy Robin Sage Sunset Tulip Valley Violet Wren'.split()
credentials = {}
for identity, role in [('A01', 'rater'), ('A02', 'rater'), ('COORDINATOR', 'coordinator')]:
    while True:
        password = '-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        if password not in credentials.values(): break
    assert study.add_user(identity, role, password)
    credentials[identity] = password
with marker.open('x', encoding='utf8') as f:
    json.dump(dict(operator_started_session=True, opened_at_utc=now(),
        config_sha256=file_hash(workspace/'CONFIG.json'),
        protocol_sha256=file_hash(workspace/'TWO_RATER_PROTOCOL_AMENDMENT.md'),
        authority='Researcher explicitly requested the actual website for two evaluators in the current conversation. Authorized local release only; no human declarations, eligibility approvals or ratings are inferred.'), f, indent=2)
launch = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(LIVE/'START_STABLE_EVALUATION.ps1')], capture_output=True, text=True, check=True)
print(launch.stdout.strip())
base = 'http://127.0.0.1:8768'
status = json.load(urllib.request.urlopen(base+'/api/status'))
assert status == dict(released=True, preview=False, synthetic=False, planned_raters=2, expected_records=90, businesses=15)
checks = {}
for identity in ['A01', 'A02']:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    request = urllib.request.Request(base+'/api/login', data=json.dumps(dict(identity=identity,password=credentials[identity])).encode(), headers={'Content-Type':'application/json','Origin':base})
    assert json.load(opener.open(request))['actor'] == dict(id=identity, role='rater')
    session = json.load(opener.open(base+'/api/session'))
    assert session['direct_rating_access'] and not session['preview_mode'] and session['participant'] is None
    items = json.load(opener.open(base+'/api/items'))['items']
    assert len(items) == 45 and len({x['alias'] for x in items}) == 45
    assert all(x['status'] != 'SUBMITTED' and x['version'] == 0 for x in items)
    photos_checked = 0
    for item in items:
        data = json.load(opener.open(base+'/api/item/'+item['alias']))
        assert data['review_text'].strip() and data['visual_text'].strip() and data['visual_applicable'] and not data['read_only']
        assert data['reviews'] and data['photos']
        if photos_checked == 0:
            photo = data['photos'][0]
            raw = opener.open(base+photo['url']).read()
            assert hashlib.sha256(raw).hexdigest() == file_hash(workspace/'PHOTOS'/(photo['photo_id']+'.jpg'))
            photos_checked += 1
    exported = json.load(opener.open(base+'/api/export'))
    assert exported['record_count'] == exported['submitted_count'] == 0
    checks[identity] = dict(login='PASS', direct_access='PASS', readable_assignments=45, source_and_output_components='PASS', photo_http_sha256='PASS', submitted=0)
    csrf = session['csrf']
    opener.open(urllib.request.Request(base+'/api/logout',data=b'{}',headers={'Content-Type':'application/json','Origin':base,'X-CSRF-Token':csrf})).read()
with study.connect() as db:
    counts = {table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['users','participants','ratings','events','participant_declaration_history']}
assert counts == dict(users=3,participants=0,ratings=0,events=0,participant_declaration_history=0)
assert study.audit()['status'] == 'PASS'
protected = {
    'outputs/stage_o/human_collection_complete_20261106_v1/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json':'d89c2fc9ed7c228139a1594c9ff7b1f7b2ffda2dfbe18bff435a7ccb6a13df7f',
    'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/STAGE_P_THREE_SYSTEM_HUMAN_EVAL_READY.zip':'afea0830e245f4c99f947215bcd785f3c7fb889a636724fb44465c74d46f1488',
    'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/STAGE_P_BLINDED_EVALUATOR_WINDOWS.zip':'0fa783544f165909589ce20291b3f5403378b08852ee4a582dbbc51d1925bdc4',
}
for relative, expected in protected.items(): assert file_hash(REPO/relative) == expected, relative
receipt = dict(status='PASS', checked_at_utc=now(), url=base+'/', study_id=study.config['study_id'], status_response=status, per_rater=checks, database_counts=counts, rating_writes_performed=0, declarations_created=0, protected_artifacts_sha256=protected, passwords_in_receipt=False, prepared_packages_unchanged=True)
(LIVE/'LIVE_READINESS.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print('PASS: both actual accounts, all 90 assignments, source/photo access; zero ratings or declarations.')
print('PRIVATE CREDENTIALS - record privately; not saved as plaintext:')
for identity,password in credentials.items(): print(identity+' | '+password)
