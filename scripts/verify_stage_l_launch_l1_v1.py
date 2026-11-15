"""Read-only Stage L.1 source/HTTP audit; never authors or approves TEST records.

The real HTTP server remains login-protected. A separate in-process test server
uses an ephemeral technical read-only session to exercise the exact source routes
without creating a participant account or modifying any TEST reference.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import secrets
import threading
import urllib.request
import urllib.error
from PIL import Image
from stage_l_reference_app_v1 import Store,Server,new_json,file_hash,digest,now

def csv_file(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8765');a=p.parse_args()
    if a.output.exists():raise ValueError('Audit output directory must be new')
    a.output.mkdir(parents=True);store=Store(a.workspace);before=store.audit()
    binding=file_hash(store.package/'ANNOTATION_PACKAGE_MANIFEST.json')
    source_files={row['source_file']:file_hash(store.package/row['source_file']) for row in store.manifest['frozen_sources']}
    with store.connect() as c:before_counts={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['users','refs','events']}
    code_before={name:file_hash(Path(__file__).parent/name) for name in ['stage_l_reference_app_v1.py','stage_l_reference_web_v1/index.html','stage_l_reference_web_v1/app.js','stage_l_reference_web_v1/app.css']}
    live={}
    for path in ['/','/app.js','/app.css','/api/cases']:
        try:
            with urllib.request.urlopen(a.base_url+path,timeout=15) as r:
                raw=r.read();live[path]={'status_code':r.status,'content_type':r.headers['Content-Type'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        except urllib.error.HTTPError as e:live[path]={'status_code':e.code,'body':e.read().decode('utf8')}
    if not (all(live[path]['status_code']==200 for path in ['/','/app.js','/app.css']) and 'Login required' in live['/api/cases'].get('body','')):
        raise ValueError('Actual application HTTP startup/authentication check failed')
    new_json(a.output/'ACTUAL_LOCAL_SERVER_HTTP.json',{'url':a.base_url,'verified_at_utc':now(),'routes':live,'browser_rendering_verified':False})
    # This server is isolated from the actual production listener and accepts
    # only GET requests in this audit. No technical account is inserted in SQLite.
    server=Server(('127.0.0.1',0),store,Path(__file__).parent/'stage_l_reference_web_v1')
    token=secrets.token_hex(32)
    server.sessions[token]={'actor':{'id':'TECHNICAL_READONLY_NOT_A_PARTICIPANT','role':'coordinator'},'csrf':secrets.token_hex(32),'expires':dt.datetime.now().timestamp()+3600}
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def get(path):
        req=urllib.request.Request(base+path,headers={'Cookie':'stage_l_session='+token})
        with urllib.request.urlopen(req,timeout=20) as r:return r.read(),r.headers
    photo_rows=[];missing=[];case_rows=[];source_rows=[]
    try:
        case_json,_=get('/api/cases');listing=json.loads(case_json)
        ids=[r['business_id'] for r in store.manifest['frozen_sources']]
        if [r['business_id'] for r in listing['cases']]!=ids or len(set(ids))!=100:raise ValueError('HTTP case list differs from frozen manifest')
        for position,b in enumerate(ids,1):
            source=store.source(b);raw,_=get('/api/case/'+b);case=json.loads(raw);served=case['source']
            if served['reviews']!=source['reviews'] or served['selected_review_ids']!=source['selected_review_ids']:raise ValueError('HTTP review IDs/text changed')
            if [x['photo_id'] for x in served['photos']]!=[x['photo_id'] for x in source['photos']]:raise ValueError('HTTP photo IDs/order changed')
            if any(served[k] for k in ['contains_predictions','contains_teacher_targets','captions_provided']):raise ValueError('Non-source evidence exposed')
            ref,version=store.get(b);batch='PILOT_001_002' if position<=2 else f'BATCH_{3+((position-3)//10)*10:03d}_{min(12+((position-3)//10)*10,100):03d}'
            case_rows.append({'manifest_position':position,'batch':batch,'business_id':b,'name':source['business_metadata'].get('name',''),
                'reviews':len(source['reviews']),'photos':len(source['photos']),'status':ref['status'],'version':version,'author_id':ref['author']['human_id'],'checker_id':ref['checker']['human_id']})
            source_rows.append({'business_id':b,'source_file':store.frame[b]['source_file'],'source_file_sha256':source_files[store.frame[b]['source_file']],
                'frozen_canonical_sha256':store.frame[b]['sha256'],'review_ids':source['selected_review_ids'],'photo_ids':[x['photo_id'] for x in source['photos']]})
            for photo in source['photos']:
                try:
                    actual=store.photo_path(b,photo['photo_id']);data,headers=get('/api/photo/'+b+'/'+photo['photo_id'])
                    if hashlib.sha256(data).hexdigest()!=photo['sha256'] or headers['Content-Type']!='image/jpeg':raise ValueError('HTTP photo bytes/type differ')
                    with Image.open(io.BytesIO(data)) as image:image.load();width,height=image.size
                    photo_rows.append({'business_id':b,'photo_id':photo['photo_id'],'manifest_expected_path':photo['path'],'resolved_served_path':str(actual),
                        'bytes':len(data),'expected_original_sha256':photo['original_source_sha256'],'served_sha256':hashlib.sha256(data).hexdigest(),
                        'width':width,'height':height,'status':'PASS_FILE_SHA_HTTP_BYTES_AND_FULL_DECODE'})
                except (ValueError,FileNotFoundError,urllib.error.HTTPError,OSError) as e:
                    missing.append({'business_id':b,'photo_id':photo['photo_id'],'expected_path':photo['path'],'error':str(e),
                        'recovery':'Restore exact photo ID from the pinned original Yelp archive recorded in SOURCE_PACKAGE/ANNOTATION_PACKAGE_MANIFEST.json, verify original_source_sha256, then re-audit. Do not change manifests or substitute an image.'})
        denied={}
        for path in ['/CONFIG.json','/PRIVATE_STATE/references.sqlite3','/outputs/stage_k/','/api/photo/'+ids[0]+'/NOT_A_PHOTO']:
            try:get(path);denied[path]=False
            except urllib.error.HTTPError as e:denied[path]=e.code in [400,404]
        if not all(denied.values()):raise ValueError('Source-only routing failed')
    finally:server.shutdown();server.server_close();thread.join()
    after=store.audit()
    with store.connect() as c:after_counts={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['users','refs','events']}
    unchanged=(before==after and before_counts==after_counts and binding==file_hash(store.package/'ANNOTATION_PACKAGE_MANIFEST.json')
        and all(file_hash(store.package/path)==sha for path,sha in source_files.items())
        and all(file_hash(Path(__file__).parent/path)==sha for path,sha in code_before.items()))
    if not unchanged:raise ValueError('Reference state, frozen source or implementation changed during read-only audit')
    csv_file(a.output/'PHOTO_AVAILABILITY_1136.csv',photo_rows)
    csv_file(a.output/'ACTUAL_PROGRESS_AND_BATCH_ORDER.csv',case_rows)
    new_json(a.output/'FIRST_TWO_FROZEN_BINDINGS.json',{'selection_rule':'First two entries in frozen annotation manifest; no model output/performance argument was used',
        'source_manifest_sha256':binding,'cases':[{**case_rows[i],**source_rows[i]} for i in range(2)]})
    new_json(a.output/'MISSING_OR_BAD_PHOTOS.json',missing)
    result={'status':'PASS_SOURCE_AND_HTTP_BACKEND' if not missing else 'ATTENTION_PHOTO_FAILURES','verified_at_utc':now(),'businesses':len(case_rows),
        'selected_reviews':sum(x['reviews'] for x in case_rows),'indexed_original_photos':sum(x['photos'] for x in case_rows),'photos_file_hash_http_full_decode_pass':len(photo_rows),
        'missing_or_bad_photos':len(missing),'actual_progress':after,'real_database_counts_before':before_counts,'real_database_counts_after':after_counts,
        'frozen_sources_and_reference_state_unchanged':unchanged,'source_manifest_sha256':binding,'implementation_sha256_unchanged':code_before,
        'actual_listener':a.base_url,'actual_listener_auth_check':'Requires human login; no real human account was created',
        'source_route_test':'Separate ephemeral server; GET-only technical session, no participant registration and no TEST save/submit/approve',
        'browser_visual_testing':'UNVERIFIED: no connected browsers; Windows native pipe unavailable (os error 2)',
        'not_certified':['Browser rendering/zoom/typing','Human authorship or factual verification','Authentic two-reference completion','Stage M scores'],
        'route_denials':denied}
    new_json(a.output/'STAGE_L1_VERIFICATION_RECEIPT.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
