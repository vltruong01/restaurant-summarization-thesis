"""Actual Stage O release readiness: accounts and read-only checks, zero ratings.

Never calls declaration, eligibility, rating-save or rating-submit endpoints.
Passwords exist only in memory and terminal output, never reports/commands/ZIPs.
"""
from pathlib import Path
import collections,contextlib,hashlib,http.cookiejar,json,os,secrets,socket,sqlite3,subprocess,sys,tempfile,threading,time,urllib.error,urllib.request

ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP'
W=APP/'WORKSPACE';D=ROOT/'outputs/stage_o/human_evaluation_release_20261105_v1'
sys.path.insert(0,str(APP/'scripts'))
from stage_o_human_evaluation_v1 import Study,Server,CRITERIA,live_release_valid

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
class Client:
    def __init__(self,origin):self.origin=origin;self.csrf='';self.opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    def call(self,path,body=None,raw=False):
        headers={'Origin':self.origin,'Content-Type':'application/json','X-CSRF-Token':self.csrf}
        q=urllib.request.Request(self.origin+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        with self.opener.open(q,timeout=20) as f:r=f.read()
        return r if raw else json.loads(r)
    def login(self,who,pw):
        x=self.call('/api/login',dict(identity=who,password=pw));self.csrf=x['csrf'];return x

def main():
    assert live_release_valid(W),'Actual release missing or invalid'
    config=read(W/'CONFIG.json');assert config['expected_records']==90 and config['conditions']==['RO','U','F']
    before=read(D/'FROZEN_WORKSPACE_BEFORE.json')['files'];errors=[]
    for r in before:
        p=W/r['file']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:errors.append(r['file'])
    assert not errors,errors
    study=Study(W)
    with study.connect() as c:
        initial=dict(ratings=c.execute('SELECT COUNT(*) FROM ratings').fetchone()[0],events=c.execute('SELECT COUNT(*) FROM events').fetchone()[0],participants=c.execute('SELECT COUNT(*) FROM participants').fetchone()[0])
    assert initial['ratings']==initial['events']==0,'Existing ratings: inspect and preserve; do not reinitialize'
    credentials={};created=[];words='Maple River Meadow Cedar Willow Cloud Garden Forest Ocean Clover Silver Autumn Summit Orchard Coral Amber Breeze Morning Birch Brook Cherry Comet Copper Daisy Fern Grove Hazel Island Jade Lake Laurel Lemon Lilac Lotus Mango Olive Opal Orchid Otter Peach Pearl Pine Plum Poppy Raven Robin Sage Shell Spruce Stone Sunset Swan Tulip Valley Violet Water Wind'.split()
    for who,role in [('COORDINATOR','coordinator'),('A01','rater'),('A02','rater')]:
        pw='-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        while pw in credentials.values():pw='-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        if study.add_user(who,role,pw):credentials[who]=pw;created.append(who)
    assert set(credentials)=={'COORDINATOR','A01','A02'},'Existing accounts preserved; no password reset. Supply existing credentials separately for any further login check.'
    # Bind the production server only to localhost, with no credentials in args/logs.
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8767))!=0,'Port8767 occupied; do not stop another service automatically'
    args=[str(ROOT/'.venv-caption-pilot/Scripts/python.exe'),'-X','utf8','-B',str(APP/'scripts/stage_o_human_evaluation_v1.py'),'serve','--workspace',str(W),'--port','8767']
    out=(D/'SERVER_STDOUT.log').open('ab');err=(D/'SERVER_STDERR.log').open('ab')
    try:
        proc=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS,close_fds=True)
    finally:out.close();err.close()
    origin='http://127.0.0.1:8767';live=None
    for _ in range(100):
        try:
            with urllib.request.urlopen(origin+'/api/status',timeout=1) as f:live=json.load(f)
            break
        except (OSError,urllib.error.URLError):assert proc.poll() is None,'Server exited; inspect nonsecret logs';time.sleep(.1)
    assert live and live['released'] is True and live['preview'] is False and live['expected_records']==90 and live['synthetic'] is False
    for who,pw in credentials.items():
        c=Client(origin);session=c.login(who,pw);assert session['actor']['id']==who
        role='coordinator' if who=='COORDINATOR' else 'rater';assert session['actor']['role']==role
        if role=='rater':
            items=c.call('/api/items')['items'];assert len(items)==45 and all(i['status']=='NOT_STARTED' for i in items)
            s=c.call('/api/session');assert s['participant'] is None and s['live_scoring_released'] and s['preview_mode'] is False
            for path in ['/api/item/'+items[0]['alias'],'/api/coordinator/mapping','/api/coordinator/export']:
                try:c.call(path);raise AssertionError('Unapproved eligibility/private access was granted')
                except urllib.error.HTTPError as e:assert e.code==400
        else:
            x=c.call('/api/coordinator/export');assert x['records']==[] and x['expected_records']==90 and x['audit']['status']=='PASS'
            assert c.call('/api/coordinator/participants')=={'A01':None,'A02':None}
            try:c.call('/api/coordinator/export?final=1');raise AssertionError('Incomplete final export allowed')
            except urllib.error.HTTPError as e:assert e.code==400
        c.call('/api/logout',{})
    # Source/output readiness uses read-only preview against the SAME real state.
    # No fake declarations/eligibility or scores are inserted to unlock testing.
    preview=Study(W,preview=True);server=Server(('127.0.0.1',0),preview,APP/'scripts/stage_o_human_web_v1')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();item_checks=0;photo_checks=0
    try:
        for who in ['A01','A02']:
            c=Client('http://127.0.0.1:'+str(server.server_port));c.login(who,credentials[who]);seen=set()
            for item in [i for i in study.key if i['annotator']==who]:
                x=c.call('/api/item/'+item['alias']);s=study.sources[item['business_id']]
                assert x['review_text']==item['review_text'] and x['visual_text']==item['visual_text'] and x['reviews']==s['reviews']
                assert [p['photo_id'] for p in x['photos']]==[p['photo_id'] for p in s['photos']]
                assert not set(x)&{'condition','model','metrics','reference','annotator'}
                assert x['visual_applicable']==(item['condition']!='RO')
                assert all(v is None or v=='N/A' for v in x['rating']['scores'].values());item_checks+=1
                if who=='A01' and item['business_id'] not in seen:
                    seen.add(item['business_id'])
                    for i in [0,len(x['photos'])-1]:
                        raw=c.call(x['photos'][i]['url'],raw=True);assert hashlib.sha256(raw).hexdigest()==s['photos'][i]['sha256'];photo_checks+=1
            c.call('/api/logout',{})
    finally:server.shutdown();server.server_close();thread.join()
    with study.connect() as c:
        after=dict(ratings=c.execute('SELECT COUNT(*) FROM ratings').fetchone()[0],events=c.execute('SELECT COUNT(*) FROM events').fetchone()[0],participants=c.execute('SELECT COUNT(*) FROM participants').fetchone()[0])
        users=[dict(x) for x in c.execute('SELECT id,role FROM users ORDER BY id')]
        for who,pw in credentials.items():assert c.execute('SELECT password FROM users WHERE id=?',(who,)).fetchone()[0]!=pw
    assert after==initial and item_checks==90 and photo_checks==30
    assert all(sha(W/r['file'])==r['sha256'] for r in before),'Frozen research data changed during final checks'
    report=dict(status='PASS_STAGE_O_RELEASED_AWAITING_REAL_PARTICIPANT_DECLARATIONS',URL=origin,server_PID=proc.pid,workspace=str(W.relative_to(ROOT)),
        original_frozen_files_rehashed=len(before),frozen_changes=0,restaurants=15,outputs=45,assignments=90,items_per_rater=45,photos=5143,
        account_roles=users,new_accounts=created,live_release_valid=True,live_login_checks=3,live_accounts_have_45_blank_items=True,
        read_only_actual_item_API_checks=item_checks,photo_HTTP_hash_checks=photo_checks,all_photo_files_rehashed=True,
        before=initial,after=after,artificial_ratings_generated_or_submitted=0,declarations_or_eligibility_fabricated=0,
        write_endpoints_called=['login','logout'],final_export_with_incomplete_records='CORRECTLY_BLOCKED',coordinator_partial_export='PASS_EMPTY_REAL_STATE',
        submission_flow='Existing software validation retained; no save/submit invoked during this readiness task',
        eligibility_workflow='Raters make their own truthful declarations; coordinator reviews actual declarations before scoring',
        credentials_written_to_disk=False,bind_address='127.0.0.1',Stage_P_Qwen='DEFERRED_NOT_RUNNING')
    put(D/'FINAL_READINESS.json',report);put(D/'SERVER_PROCESS.json',dict(PID=proc.pid,url=origin,workspace=str(W.relative_to(ROOT)),command_without_credentials=args))
    print(json.dumps(report,indent=2),flush=True)
    print('\nPRIVATE LOGIN CREDENTIALS - terminal only; give each rater only their own line:',flush=True)
    for who in ['A01','A02','COORDINATOR']:print(who+' | '+credentials[who],flush=True)
    print('No rating, declaration or eligibility decision has been created. Server remains running at '+origin,flush=True)
    credentials.clear()

if __name__=='__main__':main()
