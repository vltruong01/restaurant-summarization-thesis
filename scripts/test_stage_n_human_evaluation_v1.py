"""Bounded synthetic tests only; never writes to a real study workspace."""
import argparse
import copy
import hashlib
import http.cookiejar
import json
from pathlib import Path
import secrets
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from stage_n_human_evaluation_v1 import Study, Server, RATERS, CRITERIA, direct_rater_access_valid
from analyze_stage_n_two_raters_v1 import analyze, kappa_quadratic
from s1000_final_eval_core_v1 import digest, file_hash

HERE=Path(__file__).resolve().parent

def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf8')

def fixture(root,n=1):
    """Every business/evidence/output here is visibly synthetic, no real data read."""
    root.mkdir(parents=True,exist_ok=True);(root/'PHOTOS').mkdir(exist_ok=True)
    sources={};items=[]
    for i in range(n):
        b=f'SYNTHETIC_BUSINESS_{i:03d}';rid=f'SYNTHETIC_REVIEW_{i:03d}';pid=f'SYNTHETIC_PHOTO_{i:03d}'
        from PIL import Image
        img=root/'PHOTOS'/(pid+'.jpg');Image.new('RGB',(80,60),(28,112,102)).save(img,'JPEG')
        h=file_hash(img);s={'business_id':b,'split':'SYNTHETIC_NOT_TEST','business_metadata':{'name':'Synthetic cafe '+str(i)},'selected_review_ids':[rid],
            'reviews':[{'business_id':b,'review_id':rid,'text':'SYNTHETIC source only: a visitor mentions a quiet room.','stars':4,'date':'2000-01-01'}],
            'photos':[{'photo_id':pid,'bytes':img.stat().st_size,'sha256':h,'original_source_sha256':h}]}
        put(root/'SOURCES'/(b+'.json'),s);sources[b]={'file':'SOURCES/'+b+'.json','canonical_sha256':digest(s)}
    for rater in RATERS:
        pairs=[(b,c) for b in sources for c in ['RO','U','F','S']]
        pairs.sort(key=lambda t:digest(['SYNTHETIC',rater,*t]))
        for pos,(b,c) in enumerate(pairs,1):
            rv='[REVIEW_CONSENSUS]\nSYNTHETIC output: visitors mention a quiet room.'
            vi=None if c=='RO' else '[VISUAL_EVIDENCE]\n' if c=='F' else '[VISUAL_EVIDENCE]\nSYNTHETIC output: a green rectangle.'
            items.append({'alias':digest(['SYNTHETIC',rater,b,c])[:20],'annotator':rater,'position':pos,'business_id':b,'condition':c,'review_text':rv,
                'review_text_sha256':hashlib.sha256(rv.encode()).hexdigest(),'visual_text':vi,'visual_text_sha256':hashlib.sha256(vi.encode()).hexdigest() if vi is not None else None})
    put(root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',items)
    put(root/'RUBRIC.json',{'criteria':[{'id':c,'name':c.replace('_',' ').title(),'scope':'SYNTHETIC interface test; not the research rubric','anchors':{'1':'Synthetic low anchor','3':'Synthetic middle anchor','5':'Synthetic high anchor'}} for c in CRITERIA]})
    (root/'TWO_RATER_PROTOCOL_AMENDMENT.md').write_text('SYNTHETIC TEST ONLY',encoding='utf8')
    put(root/'CONFIG.json',{'synthetic':True,'expected_records':len(items),'sources':sources,'file_bindings':{n:file_hash(root/n) for n in ['RUBRIC.json','COORDINATOR_ONLY/PRIVATE_ITEMS.json']}})
    return Study(root)

DECL={'consent':True,'task_familiarity':True,'prior_output_or_score_exposure':'no','stage_l_involvement':'none','conflict_of_interest':'Synthetic technical fixture; not a real participant.'}
COORD={'id':'COORDINATOR','role':'coordinator'}

def enroll(s):
    for r in RATERS:s.declarations({'id':r,'role':'rater'},DECL);s.eligibility(COORD,r,'ELIGIBLE','Synthetic test only')

def body(item,value=3,version=0):
    return {'expected_version':version,'draft':{'scores':{c:'N/A' if item['condition']=='RO' and c==CRITERIA[-1] else value for c in CRITERIA},'evidence_notes':'','issue_span':'','evidence_source_ids':[],'issue_certainty':'none','serious_grounding_issue':False}}

class TechnicalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage_n_SYNTHETIC_');self.root=Path(self.temp.name);self.s=fixture(self.root/'fixture');self.a={'id':'A01','role':'rater'}
        self.item=next(x for x in self.s.key if x['annotator']=='A01' and x['condition']=='F')
    def tearDown(self):self.temp.cleanup()
    def test_no_prefill_or_real_database(self):
        d,v=self.s.rating(self.a,self.item['alias']);self.assertEqual(v,0);self.assertTrue(all(x is None for x in d['scores'].values()));self.assertEqual(self.s.audit()['stored_rating_records'],0)
    def test_release_guard_and_invalid_synthetic_unlock(self):
        c=json.loads((self.s.root/'CONFIG.json').read_text());c['synthetic']=False;put(self.s.root/'CONFIG.json',c);real=Study(self.s.root)
        self.assertFalse(real.active());pw=secrets.token_urlsafe(18);real.add_user('A01','rater',pw)
        with self.assertRaises(ValueError):real.login('A01',pw)
        with self.assertRaises(ValueError):real.apply(self.a,self.item['alias'],'save',body(self.item))
        release={'supervisor_approved':True,'researcher_live_scoring_approved':True,'approved_at_utc':'SYNTHETIC','rater_ids':RATERS,'config_sha256':file_hash(self.s.root/'CONFIG.json'),'amendment_sha256':file_hash(self.s.root/'TWO_RATER_PROTOCOL_AMENDMENT.md')};put(self.s.root/'LIVE_RELEASE.json',release);self.assertTrue(real.active())
        release['config_sha256']='wrong';put(self.s.root/'LIVE_RELEASE.json',release);self.assertFalse(real.active())
    def test_accounts_hashed_idempotent(self):
        password=secrets.token_urlsafe(18);self.assertTrue(self.s.add_user('A01','rater',password));self.assertFalse(self.s.add_user('A01','rater',secrets.token_urlsafe(18)));self.assertEqual(self.s.login('A01',password),self.a)
        self.assertNotIn(password,self.s.db.read_bytes().decode('latin1'))
        with self.assertRaises(ValueError):self.s.add_user('A03','rater',password)
    def test_researcher_release_preserves_unverified_supervisor_provenance(self):
        config=json.loads((self.s.root/'CONFIG.json').read_text());config['synthetic']=False;put(self.s.root/'CONFIG.json',config)
        s=Study(self.s.root)
        record=s.root/'COORDINATOR_ONLY/SYNTHETIC_AUTHORIZATION.md';record.write_text('SYNTHETIC researcher release only',encoding='utf8')
        release={'supervisor_approved':False,'supervisor_approval_status':'UNVERIFIED','researcher_live_scoring_approved':True,'release_authority':'researcher_direct_instruction',
            'researcher_requested_opening_with_supervisor_unverified':True,'approved_at_utc':'SYNTHETIC','rater_ids':RATERS,'config_sha256':file_hash(s.root/'CONFIG.json'),
            'amendment_sha256':file_hash(s.root/'TWO_RATER_PROTOCOL_AMENDMENT.md'),'authorization_record':'COORDINATOR_ONLY/SYNTHETIC_AUTHORIZATION.md','authorization_record_sha256':file_hash(record)}
        put(s.root/'LIVE_RELEASE.json',release);self.assertTrue(s.active());self.assertFalse(release['supervisor_approved'])
        self.assertEqual(s.export(COORD)['release_provenance']['supervisor_approval_status'],'UNVERIFIED')
        with self.assertRaises(ValueError):s.public_item(self.a,self.item['alias'])
        self.assertEqual(s.audit()['stored_rating_records'],0)
        record.write_text('changed record',encoding='utf8');self.assertFalse(s.active())
        put(s.root/'LIVE_RELEASE.json',release|{'researcher_requested_opening_with_supervisor_unverified':False});self.assertFalse(s.active())
    def test_preparation_login_allows_only_authenticated_rubric(self):
        config=json.loads((self.s.root/'CONFIG.json').read_text());config['synthetic']=False;put(self.s.root/'CONFIG.json',config)
        s=Study(self.s.root,preparation_login=True);password=secrets.token_urlsafe(18);s.add_user('A01','rater',password)
        self.assertEqual(s.login('A01',password),self.a);self.assertFalse(s.active())
        with self.assertRaises(ValueError):s.declarations(self.a,DECL)
        with self.assertRaises(ValueError):s.public_item(self.a,self.item['alias'])
        with self.assertRaises(ValueError):s.apply(self.a,self.item['alias'],'save',body(self.item))
        self.assertEqual(s.audit()['stored_rating_records'],0);self.assertIsNone(s.participant('A01'))
        server=Server(('127.0.0.1',0),s,HERE/'stage_n_human_web_v1');t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        origin='http://127.0.0.1:'+str(server.server_port);opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        try:
            req=urllib.request.Request(origin+'/api/login',data=json.dumps({'identity':'A01','password':password}).encode(),headers={'Origin':origin,'Content-Type':'application/json'})
            with opener.open(req) as r:self.assertEqual(json.loads(r.read())['actor'],self.a)
            with opener.open(origin+'/api/session') as r:
                sess=json.loads(r.read());self.assertFalse(sess['live_scoring_released']);self.assertEqual(len(sess['rubric']['criteria']),6)
            for route in ['/api/items','/api/item/'+self.item['alias'],'/api/coordinator/mapping']:
                with self.assertRaises(urllib.error.HTTPError):opener.open(origin+route)
        finally:server.shutdown();server.server_close();t.join()
    def test_direct_access_without_fabricated_declarations_preserves_isolation(self):
        config=json.loads((self.s.root/'CONFIG.json').read_text());config['synthetic']=False;put(self.s.root/'CONFIG.json',config)
        s=Study(self.s.root);put(s.root/'LIVE_RELEASE.json',{'supervisor_approved':True,'researcher_live_scoring_approved':True,'approved_at_utc':'SYNTHETIC',
            'rater_ids':RATERS,'config_sha256':file_hash(s.root/'CONFIG.json'),'amendment_sha256':file_hash(s.root/'TWO_RATER_PROTOCOL_AMENDMENT.md')})
        auth=s.root/'COORDINATOR_ONLY/SYNTHETIC_DIRECT_AUTH.json';put(auth,{'source':'SYNTHETIC researcher direct access test, not a real declaration'})
        access={'authority':'researcher_direct_instruction','rater_ids':RATERS,'declarations_and_coordinator_gate_waived':True,
            'config_sha256':file_hash(s.root/'CONFIG.json'),'amendment_sha256':file_hash(s.root/'TWO_RATER_PROTOCOL_AMENDMENT.md'),
            'live_release_sha256':file_hash(s.root/'LIVE_RELEASE.json'),'authorization_record':'COORDINATOR_ONLY/SYNTHETIC_DIRECT_AUTH.json','authorization_record_sha256':file_hash(auth)}
        put(s.root/'DIRECT_RATER_ACCESS.json',access)
        self.assertTrue(direct_rater_access_valid(s.root,'A01'));self.assertFalse(direct_rater_access_valid(s.root,'A03'))
        self.assertIsNone(s.participant('A01'));self.assertTrue(s.public_item(self.a,self.item['alias']))
        foreign=next(r for r in s.key if r['annotator']=='A02')
        with self.assertRaises(ValueError):s.public_item(self.a,foreign['alias'])
        with self.assertRaises(ValueError):s.export(self.a)
        for item in s.key:s.apply({'id':item['annotator'],'role':'rater'},item['alias'],'submit',body(item))
        exported=s.export(COORD,True);self.assertEqual(exported['direct_access_authorization'],access)
        self.assertEqual(exported['participant_declarations'],{'A01':None,'A02':None})
        auth.write_text('changed authorization',encoding='utf8');self.assertFalse(direct_rater_access_valid(s.root,'A01'))
    def test_eligibility_required(self):
        with self.assertRaises(ValueError):self.s.public_item(self.a,self.item['alias'])
        self.s.declarations(self.a,DECL)
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',body(self.item))
        self.s.eligibility(COORD,'A01','ELIGIBLE','Synthetic only');self.assertTrue(self.s.public_item(self.a,self.item['alias']))
    def test_cross_rater_isolation_and_mapping_export_denied(self):
        enroll(self.s);foreign=next(x for x in self.s.key if x['annotator']=='A02')
        for f in [lambda:self.s.public_item(self.a,foreign['alias']),lambda:self.s.apply(self.a,foreign['alias'],'save',body(foreign)),lambda:self.s.export(self.a)]:
            with self.assertRaises(ValueError):f()
        pub=self.s.public_item(self.a,self.item['alias']);self.assertFalse(set(pub)&{'condition','checkpoint_sha256','donor_id','reference','metrics'})
    def test_ro_na_and_empty_visual(self):
        enroll(self.s);ro=next(x for x in self.s.key if x['annotator']=='A01' and x['condition']=='RO');self.assertEqual(self.s.rating(self.a,ro['alias'])[0]['scores'][CRITERIA[-1]],'N/A')
        bad=body(ro);bad['draft']['scores'][CRITERIA[-1]]=1
        with self.assertRaises(ValueError):self.s.apply(self.a,ro['alias'],'submit',bad)
        self.assertTrue(self.s.public_item(self.a,self.item['alias'])['applicable_visual_empty']);good=body(self.item);self.s.apply(self.a,self.item['alias'],'submit',good)
    def test_save_reopen_resume_and_duplicate_submit(self):
        enroll(self.s);draft=body(self.item,None);self.s.apply(self.a,self.item['alias'],'save',draft);s=Study(self.s.root);self.assertEqual(s.rating(self.a,self.item['alias'])[1],1)
        s.apply(self.a,self.item['alias'],'submit',body(self.item,3,1))
        with self.assertRaises(ValueError):s.apply(self.a,self.item['alias'],'submit',body(self.item,3,2))
        self.assertEqual(s.audit()['status'],'PASS')
    def test_missing_invalid_boolean_scores_and_stale_version(self):
        enroll(self.s)
        for value in [None,0,6,True,'3']:
            with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',body(self.item,value))
        self.s.apply(self.a,self.item['alias'],'save',body(self.item))
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'save',body(self.item))
    def test_two_simultaneous_writes_only_one_commits(self):
        enroll(self.s);results=[]
        def work():
            try:self.s.apply(self.a,self.item['alias'],'save',body(self.item));results.append('saved')
            except ValueError:results.append('conflict')
        ts=[threading.Thread(target=work) for i in range(2)]
        for t in ts:t.start()
        for t in ts:t.join()
        self.assertEqual(sorted(results),['conflict','saved']);self.assertEqual(self.s.audit()['stored_rating_records'],1)
    def test_serious_claim_evidence_validation(self):
        enroll(self.s);d=body(self.item);d['draft']['serious_grounding_issue']=True
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',d)
        d['draft'].update(issue_span='quiet room',evidence_notes='SYNTHETIC test explanation',evidence_source_ids=self.s.sources[self.item['business_id']]['selected_review_ids'],issue_certainty='uncertain');self.s.apply(self.a,self.item['alias'],'submit',d)
    def test_recipient_photo_hash_and_wrong_id(self):
        enroll(self.s);pid=self.s.sources[self.item['business_id']]['photos'][0]['photo_id'];self.assertTrue(self.s.photo_path(self.a,self.item['alias'],pid).is_file())
        with self.assertRaises(ValueError):self.s.photo_path(self.a,self.item['alias'],'SYNTHETIC_WRONG_DONOR')
        (self.s.root/'PHOTOS'/(pid+'.jpg')).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.s.photo_path(self.a,self.item['alias'],pid)
    def test_final_export_and_private_condition_decoding(self):
        enroll(self.s)
        with self.assertRaises(ValueError):self.s.export(COORD,True)
        for item in self.s.key:self.s.apply({'id':item['annotator'],'role':'rater'},item['alias'],'submit',body(item))
        e=self.s.export(COORD,True);self.assertEqual(len(e['records']),8);self.assertTrue(e['synthetic']);self.assertEqual({r['condition'] for r in e['records']},{'RO','U','F','S'})
    def test_binding_tamper_denied(self):
        p=self.s.root/'COORDINATOR_ONLY/PRIVATE_ITEMS.json';data=json.loads(p.read_text());data[0]['position']=999;put(p,data)
        with self.assertRaises(ValueError):Study(self.s.root)
    def test_event_tamper_detected(self):
        enroll(self.s);self.s.apply(self.a,self.item['alias'],'save',body(self.item))
        with self.s.connect() as c:c.execute("UPDATE events SET event_hash='bad'")
        self.assertEqual(self.s.audit()['status'],'FAIL')
    def test_qwk_known_and_undefined_cases(self):
        self.assertAlmostEqual(kappa_quadratic([1,2,3,4,5],[1,2,3,4,5])['value'],1)
        self.assertAlmostEqual(kappa_quadratic([1,2,3,4,5],[5,4,3,2,1])['value'],-1)
        # Observed squared disagreement=2/5; expected=4.8, hence 1-(.4/4.8).
        self.assertAlmostEqual(kappa_quadratic([1,2,3,4,5],[1,1,2,4,5])['value'],11/12)
        self.assertIsNone(kappa_quadratic([3]*30,[3]*30)['value'])
        self.assertEqual(kappa_quadratic([2]*30,[4]*30)['value'],0)
    def test_full_240_synthetic_analysis_not_human_results(self):
        s=fixture(self.root/'full_panel',30);rows=[]
        for i,k in enumerate(s.key):
            rows.append({f:k[f] for f in ['alias','annotator','business_id','condition']}|{c:'N/A' if c==CRITERIA[-1] and k['condition']=='RO' else (i%5)+1 for c in CRITERIA}|{'status':'SUBMITTED','submitted_at_utc':'SYNTHETIC_ONLY','evidence_notes':''})
        data={'records':rows,'synthetic':True};a=analyze(data,s.key,B=100,P=100)
        self.assertEqual(a['counts']['structural_na_scores'],60);self.assertEqual(len(a['paired_comparisons']),17);self.assertEqual(len(a['quadratic_weighted_cohen_kappa']),23)
        self.assertTrue(all(r['n_businesses']==30 and r['n']==30 for r in a['paired_comparisons']))
        bad=copy.deepcopy(data);bad['records'].pop()
        with self.assertRaises(ValueError):analyze(bad,s.key,B=10,P=10)
    def test_http_security_login_save_and_resume(self):
        enroll(self.s);password=secrets.token_urlsafe(18);self.s.add_user('A01','rater',password)
        server=Server(('127.0.0.1',0),self.s,HERE/'stage_n_human_web_v1');t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        origin='http://127.0.0.1:'+str(server.server_port);opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def request(path,body=None,csrf='',headers=None):
            h={'Origin':origin,'Content-Type':'application/json','X-CSRF-Token':csrf}|(headers or {})
            return opener.open(urllib.request.Request(origin+path,data=json.dumps(body).encode() if body is not None else None,headers=h))
        try:
            with request('/') as r:self.assertIn("default-src 'self'",r.headers['Content-Security-Policy']);self.assertIn(b'Sign in',r.read())
            with self.assertRaises(urllib.error.HTTPError):request('/api/item/'+self.item['alias'])
            with request('/api/login',{'identity':'A01','password':password}) as r:
                self.assertIn('HttpOnly',r.headers['Set-Cookie']);self.assertIn('SameSite=Strict',r.headers['Set-Cookie']);sess=json.loads(r.read())
            with self.assertRaises(urllib.error.HTTPError):request('/api/coordinator/mapping')
            with self.assertRaises(urllib.error.HTTPError):request('/api/rating/'+self.item['alias']+'/save',body(self.item))
            with self.assertRaises(urllib.error.HTTPError):request('/api/status',headers={'Host':'evil.example'})
            with self.assertRaises(urllib.error.HTTPError):request('/api/rating/'+self.item['alias']+'/save',body(self.item),sess['csrf'],{'Origin':'http://evil.example'})
            with request('/api/rating/'+self.item['alias']+'/save',body(self.item),sess['csrf']) as r:self.assertEqual(json.loads(r.read())['version'],1)
            with request('/api/item/'+self.item['alias']) as r:self.assertEqual(json.loads(r.read())['rating']['status'],'DRAFT')
            server.sessions[next(iter(server.sessions))]['expires']=0
            with self.assertRaises(urllib.error.HTTPError):request('/api/session')
        finally:server.shutdown();server.server_close();t.join()

if __name__=='__main__':unittest.main(verbosity=2)
