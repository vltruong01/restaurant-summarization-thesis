"""Disposable software checks; never creates records inside delivered app."""
import argparse,copy,hashlib,http.cookiejar,json,sys,tempfile,threading,urllib.request,urllib.error
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--app',type=Path,required=True);args=parser.parse_args();app=args.app
    sys.path.insert(0,str(app/'APP'))
    from stage_p_a02_portable_revision_v1 import RevisionStudy,Server
    from human_app_core_v1 import file_hash
    actor={'id':'A02','role':'rater'}
    password='Synthetic-Test-Only-2849'
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC_A02_STATE_') as state:
        study=RevisionStudy(app/'WORKSPACE',Path(state))
        assert study.add_user('A02','rater',password)
        assert not study.add_user('A02','rater','Do-Not-Overwrite-555')
        assert study.login('A02',password)==actor
        try:study.login('A01',password);raise AssertionError('A01 login allowed')
        except ValueError:pass
        assert study.audit()['stored_rating_records']==0
        server=Server(('127.0.0.1',0),study,app/'WEB');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def get(path):return json.load(opener.open(base+path))
        def post(path,body,csrf=None):
            headers={'Content-Type':'application/json','Origin':base}
            if csrf:headers['X-CSRF-Token']=csrf
            return json.load(opener.open(urllib.request.Request(base+path,data=json.dumps(body).encode(),headers=headers)))
        try:
            assert urllib.request.urlopen(base+'/').status==200
            csrf=post('/api/login',{'identity':'A02','password':password})['csrf']
            assert get('/api/session')['direct_rating_access']
            items=get('/api/items')['items'];assert len(items)==45
            seen=set();photos=0
            for item in items:
                d=get('/api/item/'+item['alias']);assert d['review_text'].strip() and d['visual_text'].strip() and d['visual_applicable'] and d['reviews'] and d['photos']
                assert d['rating']['status']=='BASELINE' and d['version']==0 and all(type(x) is int for x in d['rating']['scores'].values())
                if d['business_id'] not in seen:
                    seen.add(d['business_id']);photo=d['photos'][0];raw=opener.open(base+photo['url']).read()
                    assert hashlib.sha256(raw).hexdigest()==file_hash(app/'WORKSPACE/PHOTOS'/(photo['photo_id']+'.jpg'));photos+=1
            inherited=get('/api/export?final=1')
            assert inherited['records']==study.baseline['records'] and inherited['portable_revision']['newly_submitted_count']==0
            assert inherited['portable_revision']['inherited_original_count']==45
            alias=items[0]['alias'];d=get('/api/item/'+alias);draft=copy.deepcopy(d['rating']);draft.pop('status');draft.pop('submitted_at_utc')
            draft['evidence_notes']+='\nSYNTHETIC SOFTWARE TEST ONLY - disposable state, never delivered.'
            result=post('/api/rating/'+alias+'/save',dict(expected_version=0,draft=draft),csrf);assert result['version']==1
            try:get('/api/export?final=1');raise AssertionError('Draft exported as final')
            except urllib.error.HTTPError as error:assert error.code==400
            result=post('/api/rating/'+alias+'/submit',dict(expected_version=1,draft=draft),csrf);assert result['version']==2
            exported=get('/api/export?final=1')
            assert len(exported['records'])==45 and exported['portable_revision']['newly_submitted_count']==1 and exported['portable_revision']['inherited_original_count']==44
            assert len(exported['portable_revision']['changes'])==1 and 'evidence_notes' in exported['portable_revision']['changes'][0]['fields']
            for row in exported['records']:
                if row['alias']!=alias:assert row==study.original[row['alias']]
            try:post('/api/rating/'+alias+'/save',dict(expected_version=2,draft=draft),csrf);raise AssertionError('Locked edit allowed')
            except urllib.error.HTTPError as error:assert error.code==400
            other=next(x['alias'] for x in study.key if x['annotator']=='A01')
            try:get('/api/item/'+other);raise AssertionError('A01 item visible')
            except urllib.error.HTTPError as error:assert error.code==400
            try:get('/api/coordinator/mapping');raise AssertionError('Coordinator route visible')
            except urllib.error.HTTPError as error:assert error.code==400
            resumed=RevisionStudy(app/'WORKSPACE',Path(state));assert resumed.audit()['status']=='PASS'
            assert resumed.rating(actor,alias)[1]==2 and resumed.rater_export(actor,True)['records']==exported['records']
        finally:server.shutdown();server.server_close();thread.join()
    assert not (app/'STATE').exists()
    print(json.dumps(dict(status='PASS',login='PASS',readable_A02_items=45,review_and_visual_prefilled='PASS',photo_http_sha256_restaurants=photos,
        baseline_unchanged_export='PASS',disposable_save_submit_persistence='PASS',draft_final_export_guard='PASS',unchanged_44_records_retained='PASS',
        revision_diff_and_journal='PASS',A01_and_coordinator_access_blocked='PASS',test_state_delivered=False)))

if __name__=='__main__':main()
