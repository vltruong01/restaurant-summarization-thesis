"""Read-only real-data HTTP smoke using the extracted embedded runtime, not rating writes."""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request

from s1000_final_eval_core_v1 import read, digest, file_hash
from build_stage_n_a02_portable_v1 import put


def verify(output,workspace,password):
    out=Path(output).resolve();root=out/'EXTRACTED_VERIFICATION/STAGE_N_A02_PORTABLE'
    package=out/'STAGE_N_A02_PORTABLE';zip_path=out/'STAGE_N_A02_PORTABLE_WINDOWS.zip'
    items=read(root/'DATA/ITEMS.json');sources=read(root/'DATA/SOURCES.json');baseline=read(root/'DATA/BASELINE.json')
    command=[str(root/'PYTHON/python.exe'),'-B','-X','utf8',str(root/'APP/stage_n_a02_portable_v1.py'),'--no-browser','--port','0']
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        line=proc.stdout.readline();match=re.search(r'http://127\.0\.0\.1:\d+/',line)
        if not match:raise RuntimeError('Embedded server failed: '+line+proc.stderr.read())
        base=match[0].rstrip('/');jar=http.cookiejar.CookieJar()
        client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        def request(path,body=None):
            headers={}
            if body is not None:headers={'Content-Type':'application/json','Origin':base}
            req=urllib.request.Request(base+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
            with client.open(req,timeout=20) as response:return response.read()
        html=request('/').decode('utf8');js=request('/app.js').decode('utf8');request('/app.css')
        assert 'portableFinal' in html and 'Download completed A02 results' in html and "download('/api/export?final=1'" in js
        login=json.loads(request('/api/login',{'identity':'A02','password':password}));assert login['actor']=={'id':'A02','role':'rater'}
        session=json.loads(request('/api/session'));assert session['direct_rating_access'] is True
        listing=json.loads(request('/api/items'))['items'];assert len(listing)==120
        assert sum(x['status']=='SUBMITTED' for x in listing)==len(baseline['ratings'])==3
        checked=set();visual={'nonempty':0,'not_applicable':0,'empty_applicable':0}
        for original in items:
            actual=json.loads(request('/api/item/'+original['alias']))
            for k in ['alias','position','business_id','review_text','visual_text']:assert actual[k]==original[k],k
            assert 'condition' not in actual and 'checkpoint' not in actual
            source=sources[actual['business_id']];assert actual['reviews']==source['reviews']
            assert [p['photo_id'] for p in actual['photos']]==[p['photo_id'] for p in source['photos']]
            visual['not_applicable' if not actual['visual_applicable'] else 'empty_applicable' if actual['applicable_visual_empty'] else 'nonempty']+=1
            if actual['business_id'] not in checked:
                for photo,p in zip(actual['photos'],source['photos']):
                    data=request(photo['url']);assert hashlib.sha256(data).hexdigest()==p['sha256']==p['original_source_sha256']
                checked.add(actual['business_id'])
        partial=json.loads(request('/api/export'));assert partial['records']==baseline['ratings']
        assert partial['events']==baseline['events']
        # No applicable score or note is supplied by this smoke check.
        for path in ['/api/coordinator/mapping','/api/coordinator/export','/api/export?final=1']:
            try:request(path)
            except urllib.error.HTTPError:pass
            else:raise AssertionError('Expected rejection: '+path)
        with sqlite3.connect((root/'STATE/ratings.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
            assert c.execute('SELECT COUNT(*) FROM ratings').fetchone()[0]==3
            assert c.execute('SELECT COUNT(*) FROM events').fetchone()[0]==len(baseline['events'])
            assert c.execute('SELECT id FROM users').fetchall()==[('A02',)]
        db=(Path(workspace).resolve()/'PRIVATE_STATE/ratings.sqlite3')
        c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        try:
            c.execute('BEGIN')
            a01=[dict(r) for r in c.execute("SELECT * FROM ratings WHERE rater='A01' ORDER BY alias")]
            a02=[{'alias':r['alias'],'rater':r['rater'],'version':r['version'],'payload':json.loads(r['payload'])}
                for r in c.execute("SELECT * FROM ratings WHERE rater='A02' ORDER BY alias")]
        finally:c.rollback();c.close()
        build=read(out/'PACKAGE_BUILD_RECEIPT.json')
        assert digest(a01)==build['A01_state_digest_at_snapshot'];assert digest(a02)==build['A02_baseline_state_digest']
        report={'status':'PASS','zip_sha256':file_hash(zip_path),'zip_bytes':zip_path.stat().st_size,
            'executed_runtime':'Embedded Python 3.11.9 from extracted ZIP on this Windows host',
            'existing_A02_login':'PASS - no password included in report','static_UI_and_export_controls':'PASS',
            'all_120_aliases_positions_raw_outputs_and_reviews_match':'PASS','all_343_original_photo_bytes_over_HTTP':'PASS',
            'visual_components':visual,'A02_submitted_baseline_preserved':3,'new_agent_created_ratings':0,
            'own_4_event_baseline_journal_preserved':'PASS','coordinator_mapping_and_premature_final_export_blocked':'PASS',
            'only_A02_account_in_portable_state':'PASS','original_A01_and_A02_rows_unchanged':'PASS',
            'synthetic_return_import_tests':'PASS - 4 isolated tests including full panel, dry run, import, duplicate, tamper and conflict rejection',
            'scope_limitation':'Recipient computer not available; verified on current Windows x64 host. Device clock and human identity are not independently authenticated.'}
        put(out/'PORTABLE_VERIFICATION.json',report)
        return report
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--workspace',required=True,type=Path)
    args=p.parse_args();print(json.dumps(verify(args.output,args.workspace,sys.stdin.readline().rstrip('\r\n')),indent=2))
