"""Portable continuation/import tests use fictional sources and disposable databases only."""
import copy
import http.cookiejar
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from test_stage_n_human_evaluation_v1 import fixture, enroll, body
from stage_n_a02_portable_v1 import PortableStudy, PortableServer
from import_stage_n_a02_portable_v1 import process
from s1000_final_eval_core_v1 import read, digest, file_hash


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage_n_A02_SYNTHETIC_')
        base=Path(self.temp.name);self.original=fixture(base/'original',n=30);enroll(self.original)
        self.a={'id':'A02','role':'rater'}
        self.original.add_user('A02','rater','Synthetic-Only-Password')
        own=[x for x in self.original.key if x['annotator']=='A02']
        a01=next(x for x in self.original.key if x['annotator']=='A01')
        self.original.apply({'id':'A01','role':'rater'},a01['alias'],'submit',body(a01))
        self.original.apply(self.a,own[0]['alias'],'submit',body(own[0]))
        self.original.apply(self.a,own[1]['alias'],'save',body(own[1]))
        self.root=base/'portable';(self.root/'DATA').mkdir(parents=True)
        shutil.copytree(self.original.root/'PHOTOS',self.root/'PHOTOS')
        shutil.copy2(self.original.root/'RUBRIC.json',self.root/'DATA/RUBRIC.json')
        fields=['alias','annotator','position','business_id','review_text','review_text_sha256','visual_text','visual_text_sha256']
        put(self.root/'DATA/ITEMS.json',[{k:x[k] for k in fields} for x in own])
        put(self.root/'DATA/SOURCES.json',self.original.sources)
        with self.original.connect() as c:
            rows=[{'alias':r['alias'],'rater':r['rater'],'version':r['version'],'payload':json.loads(r['payload'])} for r in c.execute("SELECT * FROM ratings WHERE rater='A02' ORDER BY alias")]
            events=[json.loads(r['event_json']) for r in c.execute('SELECT event_json FROM events ORDER BY seq') if json.loads(r['event_json'])['rater']=='A02']
            auth=dict(c.execute("SELECT * FROM users WHERE id='A02'").fetchone())
        seed=[];previous=None
        for e in events:
            e={**e,'previous_event_sha256':previous,'baseline_source_event_sha256':digest(e)};seed.append(e);previous=digest(e)
        put(self.root/'DATA/BASELINE.json',{'ratings':rows,'events':seed,'original_a02_states_sha256':digest(rows)})
        put(self.root/'DATA/ACCOUNT.json',auth)
        put(self.original.root/'COHORT30.json',{'business_ids':list(self.original.sources)})
        for name in ['LIVE_RELEASE.json','DIRECT_RATER_ACCESS.json']:put(self.original.root/name,{'synthetic_test_only':True})
        b={'expected_records':120,'synthetic':True,'baseline_sha256':file_hash(self.root/'DATA/BASELINE.json'),
            'blinded_items_sha256':file_hash(self.root/'DATA/ITEMS.json'),
            'source_canonical_sha256':{x:digest(s) for x,s in self.original.sources.items()}}
        for rel,key in [('CONFIG.json','original_config_sha256'),('COORDINATOR_ONLY/PRIVATE_ITEMS.json','original_assignment_key_sha256'),('RUBRIC.json','original_rubric_sha256'),('COHORT30.json','original_cohort_sha256'),('LIVE_RELEASE.json','original_release_sha256'),('DIRECT_RATER_ACCESS.json','original_direct_access_sha256')]:b[key]=file_hash(self.original.root/rel)
        put(self.root/'DATA/BINDING.json',b)
        put(self.root/'MANIFEST.json',{'files':{p.relative_to(self.root).as_posix():file_hash(p) for p in self.root.rglob('*') if p.is_file()}})
        self.s=PortableStudy(self.root)

    def tearDown(self):self.temp.cleanup()

    def finish(self):
        for item in self.s.key:
            old,v=self.s.rating(self.a,item['alias'])
            if old['status']=='SUBMITTED':continue
            original=self.original.items[item['alias']]
            self.s.apply(self.a,item['alias'],'submit',body(original,version=v))
        p=self.root/'RESULTS/A02_RESULTS_FINAL.json';self.assertTrue(p.is_file());return p

    def test_baseline_resume_lock_ownership_and_no_prefill(self):
        self.assertEqual(self.s.audit()['stored_rating_records'],2)
        self.assertEqual(len(self.s.list_items(self.a)),120)
        self.assertTrue(self.s.login('A02','Synthetic-Only-Password'))
        with self.assertRaises(ValueError):self.s.login('A01','Synthetic-Only-Password')
        locked=next(x for x in self.s.key if self.s.rating(self.a,x['alias'])[0]['status']=='SUBMITTED')
        with self.assertRaises(ValueError):self.s.apply(self.a,locked['alias'],'save',body(self.original.items[locked['alias']],version=1))
        unstarted=next(x for x in self.s.key if self.s.rating(self.a,x['alias'])[1]==0)
        self.assertTrue(all(v is None or v=='N/A' for v in self.s.rating(self.a,unstarted['alias'])[0]['scores'].values()))
        foreign=next(x for x in self.original.key if x['annotator']=='A01')
        with self.assertRaises(ValueError):self.s.public_item(self.a,foreign['alias'])
        again=PortableStudy(self.root);self.assertEqual(again.audit()['stored_rating_records'],2)
        with self.assertRaises(ValueError):self.s.return_export(final=True)

    def test_full_return_dry_run_import_idempotence_and_A01_preservation(self):
        path=self.finish()
        before=self.original.export({'id':'COORDINATOR','role':'coordinator'})['records']
        report=process(self.original.root,self.root,path)
        self.assertEqual(report['status'],'VALIDATED_NOT_IMPORTED')
        self.assertEqual(self.original.audit()['stored_rating_records'],3)
        result=process(self.original.root,self.root,path,apply=True)
        self.assertEqual(result['status'],'IMPORTED');self.assertTrue(result['A01_unchanged'])
        self.assertEqual(self.original.audit()['status'],'PASS')
        after=self.original.export({'id':'COORDINATOR','role':'coordinator'})['records']
        self.assertEqual([x for x in before if x['annotator']=='A01'],[x for x in after if x['annotator']=='A01'])
        self.assertEqual(len([x for x in after if x['annotator']=='A02']),120)
        self.assertEqual(process(self.original.root,self.root,path,apply=True)['status'],'ALREADY_IMPORTED')
        from check_stage_n_rater_completion_v1 import blank
        self.assertEqual(blank(self.original.key[0])['status'],'NOT_STARTED')

    def test_tamper_wrong_reference_and_parallel_local_changes_rejected(self):
        path=self.finish();value=read(path)
        first=value['records'][0];first['payload']['scores']['coherence']=5
        bad=self.root/'tampered.json';put(bad,value)
        with self.assertRaisesRegex(ValueError,'checksum'):process(self.original.root,self.root,bad)
        value['content_sha256']=digest({k:v for k,v in value.items() if k!='content_sha256'});put(bad,value)
        with self.assertRaisesRegex(ValueError,'journal mismatch'):process(self.original.root,self.root,bad)
        item=next(x for x in self.original.key if x['annotator']=='A02' and self.original.rating(self.a,x['alias'])[1]==0)
        self.original.apply(self.a,item['alias'],'save',body(item))
        with self.assertRaisesRegex(ValueError,'changed on the original'):process(self.original.root,self.root,path,apply=True)

    def test_http_exact_sources_photo_export_and_csrf(self):
        web=Path(__file__).parent/'stage_n_human_web_v1'
        server=PortableServer(('127.0.0.1',0),self.s,web)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}';jar=http.cookiejar.CookieJar();client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        def request(path,body=None,token=None):
            headers={}
            if body is not None:headers.update({'Content-Type':'application/json','Origin':base,'X-CSRF-Token':token or ''})
            req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
            with client.open(req) as r:return r.read()
        try:
            login=json.loads(request('/api/login',{'identity':'A02','password':'Synthetic-Only-Password'}))
            self.assertTrue(any(c.name=='stage_n_A02_portable' for c in jar))
            session=json.loads(request('/api/session'));self.assertTrue(session['direct_rating_access'])
            listing=json.loads(request('/api/items'))['items'];self.assertEqual(len(listing),120)
            item=json.loads(request('/api/item/'+listing[0]['alias']));self.assertNotIn('condition',item)
            self.assertEqual(request(item['photos'][0]['url']), (self.root/'PHOTOS'/(item['photos'][0]['photo_id']+'.jpg')).read_bytes())
            self.assertEqual(json.loads(request('/api/export'))['rater'],'A02')
            with self.assertRaises(urllib.error.HTTPError):request('/api/export?final=1')
            unstarted=next(x for x in listing if x['status']=='NOT_STARTED')
            with self.assertRaises(urllib.error.HTTPError):request('/api/rating/'+unstarted['alias']+'/save',body(self.original.items[unstarted['alias']]))
            response=json.loads(request('/api/rating/'+unstarted['alias']+'/save',body(self.original.items[unstarted['alias']]),login['csrf']))
            self.assertEqual(response['rating']['status'],'DRAFT')
            self.assertTrue((self.root/'RESULTS/A02_PROGRESS_BACKUP.json').is_file())
            with self.assertRaises(urllib.error.HTTPError):request('/api/coordinator/mapping')
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main(verbosity=2)
