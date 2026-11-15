"""Synthetic-only workflow tests. Never opens or writes genuine TEST drafts."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import http.cookiejar
from PIL import Image
from stage_l_reference_app_v1 import Store,Server,new_json,digest,file_hash,read,reference_content

AUTHOR={'id':'demo_author','role':'author'}
CHECKER={'id':'demo_checker','role':'checker'}
AUTHOR_ATTEST={k:True for k in ['human_written','sources_checked','no_predictions_or_teacher']}
CHECKER_ATTEST={k:True for k in ['independent_check','sources_checked','no_predictions_or_teacher','coverage_and_component_separation_checked']}

def fixture(root,n=1):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);package=root/'SOURCE_PACKAGE';package.mkdir()
    image=root/'demo.jpg';Image.new('RGB',(32,24),(100,120,140)).save(image)
    rows=[]
    for i in range(n):
        b=f'DEMO_ONLY_{i:03d}';rid=f'demo_review_{i}';pid=f'demo_photo_{i}'
        text='Demo reviewers liked the soup but reported slow service.'
        s={'business_id':b,'split':'test','business_metadata':{'name':f'Synthetic Demo {i}'},'selected_review_ids':[rid],
            'selected_review_block':text,'review_block_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
            'reviews':[{'business_id':b,'review_id':rid,'text':text,'stars':3,'date':'2020-01-01'}],
            'photos':[{'photo_id':pid,'path':'demo.jpg','bytes':image.stat().st_size,'sha256':file_hash(image),'original_source_sha256':file_hash(image),'original_source_path':'DEMO_ONLY'}],
            'photo_scope':'SYNTHETIC ONLY','contains_predictions':False,'contains_teacher_targets':False,'captions_provided':False}
        new_json(package/'SOURCES'/f'{b}.json',s);rows.append({'business_id':b,'source_file':f'SOURCES/{b}.json','sha256':digest(s),'reviews':1,'photos':1})
        r={'business_id':b,'source_bundle_sha256':digest(s),'review_consensus':'','visual_evidence':'','review_claims':[],'visual_claims':[],'status':'NOT_STARTED',
            'author':{'human_id':'','at_utc':'','human_written_attestation':False,'predictions_seen':False},
            'checker':{'human_id':'','at_utc':'','independent_check_attestation':False,'predictions_seen':False,'reviewed_content_sha256':''},'history':[],'notes':''}
        new_json(package/'TEST100_REFERENCE_TEMPLATE'/f'{b}.json',r)
    new_json(package/'ANNOTATION_PACKAGE_MANIFEST.json',{'businesses':n,'selected_reviews':n,'original_photos':n,'frozen_sources':rows})
    new_json(root/'CONFIG.json',{'source_manifest_sha256':file_hash(package/'ANNOTATION_PACKAGE_MANIFEST.json'),'expected_businesses':n,'expected_reviews':n,'expected_photos':n,'project_root':str(root),'portable_photo_root':None})
    (root/'HUMAN_REFERENCE_ANNOTATION_GUIDELINES.md').write_text('SYNTHETIC DEMO ONLY',encoding='utf8')
    return Store(root)

def draft_for(store,b):
    i=int(b.rsplit('_',1)[1]);return {'review_consensus':f'Demo {i}: reviewers liked the soup but reported slow service.',
        'visual_evidence':f'Demo {i}: a synthetic image contains blue-gray pixels.',
        'review_claims':[{'output_span':'reviewers liked the soup','evidence':[{'review_id':f'demo_review_{i}','quote':'liked the soup'}]}],
        'visual_claims':[{'output_span':'blue-gray pixels','evidence':[{'photo_id':f'demo_photo_{i}','direct_image_viewed':True}]}], 'notes':'SYNTHETIC TECHNICAL TEST. NOT A HUMAN ANNOTATION.'}

class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=fixture(Path(self.temp.name)/'demo');self.b=next(iter(self.store.frame));self.draft=draft_for(self.store,self.b)
    def tearDown(self):self.temp.cleanup()
    def action(self,actor,action,v,draft=None,**extra):
        return self.store.apply(self.b,actor,action,{'expected_version':v,'draft':draft or self.draft,**extra})
    def test_source_and_photo_loading(self):
        r=self.store.verify_sources(True);self.assertTrue(r['all_photo_sha256_and_image_decode_checked'])
        with self.assertRaises(ValueError):self.store.photo_path(self.b,'foreign_photo')
    def test_draft_restart_and_revision_approval_export(self):
        self.action(AUTHOR,'save',0);reopened=Store(self.store.root);self.assertEqual(reopened.get(self.b)[1],1)
        self.action(AUTHOR,'submit',1,attestations=AUTHOR_ATTEST)
        self.action(CHECKER,'request_revision',2,comment='SYNTHETIC: clarify minority feedback')
        self.assertEqual(self.store.get(self.b)[0]['status'],'NEEDS_REVISION')
        self.action(AUTHOR,'save',3);self.action(AUTHOR,'submit',4,attestations=AUTHOR_ATTEST)
        result=self.action(CHECKER,'approve',5,attestations=CHECKER_ATTEST,comment='SYNTHETIC ONLY')
        self.assertFalse(self.store.errors(result['reference'],True));self.assertEqual(result['version'],6)
        self.assertEqual(self.store.audit()['counts']['APPROVED'],1)
        self.store.export(Path(self.temp.name)/'inspection');self.assertFalse(read(Path(self.temp.name)/'inspection/EXPORT_MANIFEST.json')['final_frozen'])
        self.assertEqual(len(read(Path(self.temp.name)/'inspection/ANNOTATIONS'/f'{self.b}.json')['history']),6)
    def test_conflict_and_approval_lock(self):
        self.action(AUTHOR,'save',0)
        with self.assertRaisesRegex(ValueError,'Version conflict'):self.action(AUTHOR,'save',0)
        self.action(AUTHOR,'submit',1,attestations=AUTHOR_ATTEST);self.action(CHECKER,'approve',2,attestations=CHECKER_ATTEST)
        with self.assertRaises(ValueError):self.action(AUTHOR,'save',3)
    def test_roles_identity_and_attestations(self):
        with self.assertRaises(ValueError):self.action(CHECKER,'save',0)
        with self.assertRaises(ValueError):self.action(AUTHOR,'submit',0)
        self.action(AUTHOR,'submit',0,attestations=AUTHOR_ATTEST)
        with self.assertRaises(ValueError):self.action({'id':'demo_author','role':'checker'},'approve',1,attestations=CHECKER_ATTEST)
        with self.assertRaises(ValueError):self.action(CHECKER,'approve',1)
        with self.assertRaises(ValueError):self.action(AUTHOR,'approve',1,attestations=CHECKER_ATTEST)
    def test_evidence_bindings_invalid_marker_missing_fields(self):
        for mutate in [lambda d:d['review_claims'][0]['evidence'][0].update(review_id='foreign'),
                       lambda d:d['review_claims'][0]['evidence'][0].update(quote='not present'),
                       lambda d:d['visual_claims'][0]['evidence'][0].update(photo_id='foreign'),
                       lambda d:d.update(review_consensus='[VISUAL_EVIDENCE] invalid'),
                       lambda d:d.update(visual_evidence='')]:
            d=copy.deepcopy(self.draft);mutate(d)
            with self.assertRaises(ValueError):self.action(AUTHOR,'submit',0,draft=d,attestations=AUTHOR_ATTEST)
        self.assertEqual(self.store.get(self.b)[1],0)
    def test_photo_attestation_enforced_at_submission(self):
        d=copy.deepcopy(self.draft);d['visual_claims'][0]['evidence'][0]['direct_image_viewed']=False
        with self.assertRaises(ValueError):self.action(AUTHOR,'submit',0,draft=d,attestations=AUTHOR_ATTEST)
    def test_missing_claims_and_bad_shapes_are_rejected(self):
        d=copy.deepcopy(self.draft);d['review_claims']=[]
        with self.assertRaises(ValueError):self.action(AUTHOR,'submit',0,draft=d,attestations=AUTHOR_ATTEST)
        d['review_claims']=[None]
        with self.assertRaises(ValueError):self.action(AUTHOR,'save',0,draft=d)
    def test_tainted_template_rejected(self):
        p=self.store.package/'TEST100_REFERENCE_TEMPLATE'/f'{self.b}.json';r=read(p);r['review_consensus']='must not be prefilled';p.write_text(json.dumps(r),encoding='utf8')
        with self.assertRaises(ValueError):self.store.get(self.b)
    def test_duplicate_references_detected(self):
        store=fixture(Path(self.temp.name)/'duplicate_demo',2)
        for b in store.frame:
            d=draft_for(store,b);d['review_consensus']='Demo reviewers liked the soup but reported slow service.';d['visual_evidence']='A synthetic image contains blue-gray pixels.'
            store.apply(b,AUTHOR,'submit',{'expected_version':0,'draft':d,'attestations':AUTHOR_ATTEST})
            store.apply(b,CHECKER,'approve',{'expected_version':1,'attestations':CHECKER_ATTEST})
        self.assertEqual(len(store.audit()['duplicate_reference_pairs']),1)
    def test_backup_reopens_identical_draft(self):
        self.action(AUTHOR,'save',0)
        import sqlite3
        from contextlib import closing
        destination=Path(self.temp.name)/'backup.sqlite3'
        with self.store.connect() as source,closing(sqlite3.connect(destination)) as target:source.backup(target)
        with closing(sqlite3.connect(destination)) as c:
            self.assertEqual(json.loads(c.execute('SELECT payload FROM refs').fetchone()[0])['review_consensus'],self.draft['review_consensus'])
    def test_journal_corruption_detected(self):
        self.action(AUTHOR,'save',0)
        with self.store.connect() as c:c.execute("UPDATE refs SET version=999")
        self.assertEqual(self.store.audit()['status'],'NOT_READY')
    def test_rejection_preserves_history(self):
        self.action(AUTHOR,'submit',0,attestations=AUTHOR_ATTEST);self.action(CHECKER,'reject',1,comment='SYNTHETIC rejection')
        r,v=self.store.get(self.b);self.assertEqual(r['status'],'NEEDS_REVISION');self.assertEqual(r['history'][-1]['event'],'reject')
    def test_final_refused_when_incomplete(self):
        dest=Path(self.temp.name)/'must_not_exist'
        with self.assertRaises(ValueError):self.store.export(dest,True)
        self.assertFalse(dest.exists())
    def test_source_tamper_rejected(self):
        p=self.store.package/'SOURCES'/f'{self.b}.json';s=read(p);s['reviews'][0]['text']='tampered';p.write_text(json.dumps(s),encoding='utf8')
        with self.assertRaises(ValueError):Store(self.store.root)
    def test_http_auth_csrf_source_isolation(self):
        self.store.add_user(AUTHOR['id'],'author','synthetic-only-password')
        server=Server(('127.0.0.1',0),self.store,Path(__file__).parent/'stage_l_reference_web_v1');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        origin=f'http://127.0.0.1:{server.server_port}';jar=http.cookiejar.CookieJar();client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        def req(path,body=None,csrf=None):
            headers={'Origin':origin,'Content-Type':'application/json'}
            if csrf:headers['X-CSRF-Token']=csrf
            return client.open(urllib.request.Request(origin+path,data=json.dumps(body).encode() if body is not None else None,headers=headers))
        try:
            with self.assertRaises(urllib.error.HTTPError):req('/api/cases')
            session=json.load(req('/api/login',{'identity':AUTHOR['id'],'password':'synthetic-only-password'}));self.assertEqual(session['actor'],AUTHOR)
            self.assertEqual(len(json.load(req('/api/cases'))['cases']),1)
            self.assertEqual(req('/api/photo/'+self.b+'/demo_photo_0').headers['Content-Type'],'image/jpeg')
            for path in ['/CONFIG.json','/PRIVATE_STATE/references.sqlite3','/../outputs/stage_k/raw_predictions.json','/api/photo/'+self.b+'/foreign']:
                with self.assertRaises(urllib.error.HTTPError):req(path)
            with self.assertRaises(urllib.error.HTTPError):req('/api/action/'+self.b+'/save',{'expected_version':0,'draft':self.draft})
            result=json.load(req('/api/action/'+self.b+'/save',{'expected_version':0,'draft':self.draft},session['csrf']));self.assertEqual(result['version'],1)
        finally:server.shutdown();server.server_close();thread.join()

class FullSyntheticFreeze(unittest.TestCase):
    def test_100_synthetic_approvals_existing_stage_m_contract(self):
        with tempfile.TemporaryDirectory() as td:
            store=fixture(Path(td)/'demo100',100)
            for b in store.frame:
                store.apply(b,AUTHOR,'submit',{'expected_version':0,'draft':draft_for(store,b),'attestations':AUTHOR_ATTEST})
                store.apply(b,CHECKER,'approve',{'expected_version':1,'attestations':CHECKER_ATTEST})
            export=Path(td)/'SYNTHETIC_ONLY_FINAL';store.export(export,True)
            freeze=read(export/'REFERENCE_FREEZE.json');self.assertEqual(freeze['businesses'],100)
            from validate_s1000_human_references_v1 import validate_all
            _,report=validate_all(store.package,export/'ANNOTATIONS',True);self.assertFalse(report['errors'])
            self.assertEqual(len(read(export/'REFERENCES_INSPECTION.json')),100)
            for rel,sha in read(export/'EXPORT_MANIFEST.json')['files'].items():self.assertEqual(file_hash(export/rel),sha)

if __name__=='__main__':unittest.main(verbosity=2)
