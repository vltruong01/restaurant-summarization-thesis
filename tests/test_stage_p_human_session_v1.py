"""All scoring/authentication tests use invented SYNTHETIC data, never research items."""
import unittest,tempfile,sys,json,hashlib,threading,http.cookiejar,urllib.request,urllib.error,subprocess,copy,sqlite3
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT if (ROOT/'EVALUATOR_APP').is_dir() else ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/PACKAGE';APP=PACKAGE/'EVALUATOR_APP';sys.path.insert(0,str(APP/'APP'));sys.path.insert(0,str(PACKAGE/'STATISTICS'))
from stage_p_human_evaluation_v1 import Study,Server,CRITERIA
from human_app_core_v1 import digest,file_hash
from validate_stage_p_human_exports_v1 import validate

def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def fixture(base):
    w=base/'EVALUATOR_APP/WORKSPACE';w.mkdir(parents=True);(w/'RUBRIC.json').write_bytes((APP/'WORKSPACE/RUBRIC.json').read_bytes());(w/'TWO_RATER_PROTOCOL_AMENDMENT.md').write_text('SYNTHETIC SOFTWARE TEST ONLY',encoding='utf8');sources={};items=[];key=[]
    for k in range(3):
        b='SYNTHETIC_RESTAURANT_'+str(k);photos=[];reviews=[dict(review_id=b+'_R1',stars=3,date='2000-01-01',text='Invented software-test review. Not research evidence.')]
        for j in range(50):
            pid=b+'_PHOTO_'+str(j);p=w/'PHOTOS'/(pid+'.jpg');p.parent.mkdir(exist_ok=True);Image.new('RGB',(96,64),(k*70,j*4,100)).save(p);h=file_hash(p);photos.append(dict(photo_id=pid,bytes=p.stat().st_size,sha256=h,original_source_sha256=h))
        source=dict(business_id=b,business_metadata=dict(name=b),selected_review_ids=[reviews[0]['review_id']],reviews=reviews,photos=photos);put(w/'SOURCES'/(b+'.json'),source);sources[b]=dict(file='SOURCES/'+b+'.json',canonical_sha256=digest(source))
        for r in ['A01','A02']:
            for c in range(3):
                alias=hashlib.sha256((r+b+str(c)).encode()).hexdigest()[:24];review='[REVIEW_CONSENSUS] Invented fixture text '+str(c)+'.';visual='[VISUAL_EVIDENCE] Synthetic colored rectangles.';x=dict(alias=alias,annotator=r,position=k*3+c+1,business_id=b,condition='C'+str(c+1),review_text=review,review_text_sha256=hashlib.sha256(review.encode()).hexdigest(),visual_text=visual,visual_text_sha256=hashlib.sha256(visual.encode()).hexdigest());items.append(x);key.append(dict(alias=alias,annotator=r,position=x['position'],business_id=b,neutral_slot=x['condition'],within_restaurant_position=c+1,system=['LED F original','Qwen-Caption','Qwen-Image'][c],review_text_sha256=x['review_text_sha256'],visual_text_sha256=x['visual_text_sha256']))
    put(w/'BLINDED_ITEMS.json',items);cfg=dict(format='stage_p_three_system_blinded_human_v1',study_id='SYNTHETIC_SOFTWARE_TEST_SESSION',expected_records=18,conditions=['C1','C2','C3'],raters=['A01','A02'],businesses=3,sources=sources,synthetic=True,file_bindings={n:file_hash(w/n) for n in ['BLINDED_ITEMS.json','RUBRIC.json','TWO_RATER_PROTOCOL_AMENDMENT.md']});put(w/'CONFIG.json',cfg);put(base/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json',key)
    plan=dict(seed=20261106,bootstrap_draws=20000,restaurant_order=list(sources),raters=['A01','A02'],systems=['LED F original','Qwen-Caption','Qwen-Image'],Review_only_criteria=CRITERIA[:2]+['usefulness'],contrasts=[dict(left='LED F original',right='Qwen-Caption',family='PRIMARY'),dict(left='Qwen-Caption',right='Qwen-Image',family='PRIMARY'),dict(left='LED F original',right='Qwen-Image',family='ADDITIONAL_EXPLORATORY')]);put(base/'COORDINATOR_ONLY/ANALYSIS_PLAN.json',plan);return w

class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='STAGE_P_SYNTHETIC_ONLY_');cls.base=Path(cls.temp.name);cls.w=fixture(cls.base)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        self.state=Path(tempfile.mkdtemp(dir=self.base));self.s=Study(self.w,state_dir=self.state)
        for who,role in [('A01','rater'),('A02','rater'),('COORDINATOR','coordinator')]:self.s.add_user(who,role,'Fixture-River-123')
        self.a=self.s.login('A01','Fixture-River-123');self.b=self.s.login('A02','Fixture-River-123');self.item=self.s.list_items(self.a)[0]
    def draft(self,value=3):return dict(expected_version=0,draft=dict(scores={c:value for c in CRITERIA},evidence_notes='',issue_span='',evidence_source_ids=[],issue_certainty='none',serious_grounding_issue=False))
    def test_01_login_and_idempotent_accounts(self):
        self.assertFalse(self.s.add_user('A01','rater','Another-Pass-123'));self.assertEqual(self.s.login('A01','Fixture-River-123'),self.a)
        with self.assertRaises(ValueError):self.s.login('A01','wrong')
        with self.s.connect() as c:self.assertNotIn('Fixture-River-123',json.dumps([dict(x) for x in c.execute('SELECT * FROM users')]))
    def test_02_ownership_and_blinding(self):
        self.assertEqual(len(self.s.list_items(self.a)),9);x=self.s.public_item(self.a,self.item['alias']);self.assertNotIn('condition',x);self.assertNotIn('system',x)
        with self.assertRaises(ValueError):self.s.public_item(self.b,self.item['alias'])
    def test_03_sources_and_photo_bytes(self):
        x=self.s.public_item(self.a,self.item['alias']);self.assertEqual(len(x['photos']),50);self.assertTrue(x['visual_applicable'])
        for photo in [x['photos'][0],x['photos'][-1]]:self.assertTrue(self.s.photo_path(self.a,x['alias'],photo['photo_id']).is_file())
        with self.assertRaises(ValueError):self.s.photo_path(self.a,x['alias'],'../bad')
    def test_04_partial_save_resume_and_version(self):
        body=self.draft(None);body['draft']['scores'][CRITERIA[0]]=4;v=self.s.apply(self.a,self.item['alias'],'save',body);self.assertEqual(v['version'],1)
        reopened=Study(self.w,state_dir=self.state);self.assertEqual(reopened.rating(self.a,self.item['alias'])[0]['scores'][CRITERIA[0]],4)
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'save',self.draft())
    def test_05_submit_lock_and_audit(self):
        self.s.apply(self.a,self.item['alias'],'submit',self.draft());self.assertEqual(self.s.audit()['status'],'PASS')
        body=self.draft();body['expected_version']=1
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',body)
    def test_06_all_six_scores_and_no_NA(self):
        for value in [None,'N/A',True,0,6]:
            with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',self.draft(value))
    def test_07_preview_cannot_write(self):
        preview=Study(self.w,preview=True,state_dir=self.state)
        with self.assertRaises(ValueError):preview.apply(self.a,self.item['alias'],'save',self.draft())
        with self.assertRaises(ValueError):preview.rater_export(self.a,True)
    def test_08_serious_issue_requires_exact_evidence(self):
        d=self.draft();d['draft']['serious_grounding_issue']=True
        with self.assertRaises(ValueError):self.s.apply(self.a,self.item['alias'],'submit',d)
        d['draft'].update(issue_span='Invented fixture text',evidence_notes='Synthetic test note.',evidence_source_ids=['SYNTHETIC_RESTAURANT_0_R1'],issue_certainty='uncertain');self.s.apply(self.a,self.item['alias'],'submit',d)
    def test_09_independent_exports_and_final_rejection(self):
        self.s.apply(self.a,self.item['alias'],'submit',self.draft());self.assertEqual(self.s.rater_export(self.b)['records'],[])
        with self.assertRaises(ValueError):self.s.rater_export(self.a,True)
    def test_10_database_journal_detects_tamper(self):
        self.s.apply(self.a,self.item['alias'],'submit',self.draft())
        with self.s.connect() as c:c.execute('UPDATE ratings SET version=99')
        self.assertEqual(self.s.audit()['status'],'FAIL')
    def test_11_HTTP_auth_host_origin_csrf_and_export(self):
        server=Server(('127.0.0.1',0),self.s,APP/'WEB');t=threading.Thread(target=server.serve_forever,daemon=True);t.start();origin='http://127.0.0.1:'+str(server.server_port);client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def post(path,body,csrf=None,site=None):
            headers={'Content-Type':'application/json','Origin':site or origin}
            if csrf:headers['X-CSRF-Token']=csrf
            return client.open(urllib.request.Request(origin+path,data=json.dumps(body).encode(),headers=headers))
        try:
            with self.assertRaises(urllib.error.HTTPError):client.open(origin+'/api/items')
            with post('/api/login',dict(identity='A01',password='Fixture-River-123')) as f:session=json.load(f)
            with client.open(origin+'/api/items') as f:items=json.load(f)['items'];self.assertEqual(len(items),9)
            with self.assertRaises(urllib.error.HTTPError):post('/api/rating/'+items[0]['alias']+'/save',self.draft())
            with self.assertRaises(urllib.error.HTTPError):post('/api/rating/'+items[0]['alias']+'/save',self.draft(),session['csrf'],'http://evil.invalid')
            with post('/api/rating/'+items[0]['alias']+'/submit',self.draft(),session['csrf']) as f:self.assertEqual(json.load(f)['rating']['status'],'SUBMITTED')
            with client.open(origin+'/api/export') as f:self.assertEqual(json.load(f)['submitted_count'],1)
            with self.assertRaises(urllib.error.HTTPError):client.open(origin+'/api/coordinator/mapping')
        finally:server.shutdown();server.server_close();t.join()
    def test_12_complete_fixture_export_validation_and_statistics(self):
        from analyze_stage_p_human_ratings_v1 import analyze
        for i,item in enumerate(self.s.key):
            actor=self.a if item['annotator']=='A01' else self.b;self.s.apply(actor,item['alias'],'submit',self.draft(2+i%4))
        paths=[]
        for a in [self.a,self.b]:p=self.state/(a['id']+'.json');put(p,self.s.rater_export(a,True));paths.append(p)
        rows,v=validate(self.base,paths,True);self.assertEqual(len(rows),18);self.assertEqual(v['old_Stage_O_records_reused'],0)
        out=self.state/'stats';result=analyze(self.base,paths,out,True);self.assertEqual(result['comparisons'],18);self.assertEqual(result['agreement_cells'],18)
        for x in json.loads((out/'PAIRED_COMPARISONS.json').read_text()):self.assertGreaterEqual(x['holm_p_within_family'],x['raw_exact_signflip_p'])
        d=json.loads(paths[0].read_text());d['study_id']='OLD_SESSION';put(paths[0],d)
        with self.assertRaises(AssertionError):validate(self.base,paths,True)
    def test_13_weighted_kappa_known_cases(self):
        from analyze_stage_p_human_ratings_v1 import qwk
        self.assertEqual(qwk([1,2,3,4,5],[1,2,3,4,5]),1);self.assertEqual(qwk([3,3],[3,3]),None);self.assertLess(qwk([1,2,3,4,5],[5,4,3,2,1]),0)
    def test_14_restaurant_cluster_keeps_opposing_raters_together(self):
        from analyze_stage_p_human_ratings_v1 import analyze
        for item in self.s.key:
            actor=self.a if item['annotator']=='A01' else self.b;slot=int(item['condition'][1:])-1
            score=([4,3,2] if actor==self.a else [2,3,4])[slot]
            self.s.apply(actor,item['alias'],'submit',self.draft(score))
        paths=[]
        for a in [self.a,self.b]:p=self.state/(a['id']+'.json');put(p,self.s.rater_export(a,True));paths.append(p)
        out=self.state/'cluster_stats';analyze(self.base,paths,out,True)
        for row in json.loads((out/'PAIRED_COMPARISONS.json').read_text()):
            self.assertEqual(row['mean_difference'],0);self.assertEqual(row['pointwise_CI95_low'],0);self.assertEqual(row['pointwise_CI95_high'],0);self.assertEqual(row['raw_exact_signflip_p'],1)

if __name__=='__main__':unittest.main(verbosity=2)
