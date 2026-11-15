"""Headless Chromium software E2E; all writes are SYNTHETIC, isolated from research."""
from pathlib import Path
import sys,json,tempfile,shutil,hashlib,subprocess,socket,time,urllib.request,threading
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT if (ROOT/'EVALUATOR_APP').is_dir() else ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1/PACKAGE';D=ROOT/'RETEST_RESULTS' if PACKAGE==ROOT else PACKAGE.parent;APP=PACKAGE/'EVALUATOR_APP';sys.path.insert(0,str(APP/'APP'));sys.path.insert(0,str(Path(__file__).parent))
from stage_p_human_evaluation_v1 import Study,CRITERIA
from human_app_core_v1 import digest,file_hash
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def run():
    out=D/'VALIDATION';out.mkdir(exist_ok=True);checks=[];errors=[]
    with tempfile.TemporaryDirectory(prefix='P_BROWSER_SYNTHETIC_ONLY_') as td,sync_playwright() as pw:
        temp=Path(td)
        from test_stage_p_human_session_v1 import fixture
        w=fixture(temp/'INPUTS')
        cfg=json.loads((w/'CONFIG.json').read_text());bid=next(iter(cfg['sources']));sp=w/cfg['sources'][bid]['file'];source=json.loads(sp.read_text());sample=w/'PHOTOS'/(source['photos'][0]['photo_id']+'.jpg')
        for i in range(50,2040):
            pid=bid+'_PHOTO_'+str(i);f=w/'PHOTOS'/(pid+'.jpg');shutil.copy2(sample,f);h=file_hash(f);source['photos'].append(dict(photo_id=pid,bytes=f.stat().st_size,sha256=h,original_source_sha256=h))
        put(sp,source);cfg['sources'][bid]['canonical_sha256']=digest(source);put(w/'CONFIG.json',cfg)
        # The launcher creates a local operator-start marker for this fixture only.
        if (w/'COLLECTION_OPEN.json').exists():(w/'COLLECTION_OPEN.json').unlink()
        state=temp/'STATE';s=Study(w,state_dir=state)
        for r,role in [('A01','rater'),('A02','rater'),('COORDINATOR','coordinator')]:s.add_user(r,role,'Fixture-River-123')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        origin='http://127.0.0.1:'+str(port);args=[str(APP/'PYTHON/python.exe'),'-u','-B','-X','utf8',str(APP/'APP/launch_stage_p_human_v1.py'),'--open-session','--no-browser','--port',str(port),'--workspace',str(w),'--state-dir',str(state)]
        def start():
            p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8')
            for _ in range(100):
                try:
                    with urllib.request.urlopen(origin+'/api/status',timeout=.5) as f:d=json.load(f)
                    assert d['synthetic'] is True;return p
                except OSError:assert p.poll() is None;time.sleep(.05)
            raise AssertionError('Fixture server start timeout')
        proc=start();browser=pw.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu']);ctx=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True);page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('dialog',lambda d:d.accept())
        def login(who):
            page.goto(origin);page.fill('#identity',who);page.fill('#password','Fixture-River-123');page.click('#loginForm button');expect(page.locator('#work')).to_be_visible();expect(page.locator('#environment')).to_contain_text('SYNTHETIC')
        def open_item(item):page.select_option('#items',item['alias']);expect(page.locator('#itemId')).to_contain_text(item['alias'])
        try:
            page.goto(origin);page.fill('#identity','A01');page.fill('#password','Incorrect-Fixture-Password');page.click('#loginForm button');expect(page.locator('#message')).to_contain_text('Invalid credentials');checks.append('Wrong credentials rejected in browser')
            login('A01');expect(page.locator('#progress')).to_have_text('Submitted 0 / 9');item=s.list_items({'id':'A01','role':'rater'})[0];open_item(item)
            expect(page.locator('#reviewOutput')).to_contain_text('Invented fixture');expect(page.locator('#visualOutput')).to_contain_text('Synthetic colored');expect(page.locator('#score_'+CRITERIA[-1])).to_be_enabled();expect(page.locator('#score_'+CRITERIA[0])).to_have_value('');checks.append('Authentication, blank six-score form, distinct Review/Visual and direct item access')
            for c in CRITERIA:page.select_option('#score_'+c,'3')
            page.fill('#notes','DISPOSABLE SOFTWARE TEST, NOT HUMAN RESEARCH.');page.click('#save');expect(page.locator('#saveStatus')).to_contain_text('Draft saved on server')
            proc.terminate();proc.communicate(timeout=10);proc=start();login('A01');open_item(item);expect(page.locator('#score_'+CRITERIA[0])).to_have_value('3');expect(page.locator('#notes')).to_have_value('DISPOSABLE SOFTWARE TEST, NOT HUMAN RESEARCH.');checks.append('Bundled Python server restart, fresh browser login and draft persistence')
            page.click('#reviewCount');page.locator('#photoRange').scroll_into_view_if_needed();expect(page.locator('#photos a')).to_have_count(24);expect(page.locator('#photoRange')).to_contain_text('of 2040');page.click('#photoNext');expect(page.locator('#photoPage')).to_have_value('2');page.fill('#photoPage','85');page.click('#photoGo');expect(page.locator('#photoRange')).to_contain_text('2017');last=source['photos'][-1]['photo_id'];page.fill('#photoIdJump',last);page.click('#photoFind');expect(page.locator('#photoDialogTitle')).to_have_text('Photo 2040 of 2040');expect(page.locator('#photoLargeNext')).to_be_disabled();page.keyboard.press('ArrowLeft');expect(page.locator('#photoDialogTitle')).to_have_text('Photo 2039 of 2040');page.keyboard.press('Escape');expect(page.locator('#photoDialog')).not_to_be_visible();checks.append('2040-photo synthetic stress gallery: bounded24 thumbnails,page85,exact ID jump,zoom and keyboard navigation')
            page.locator('#photoRange').scroll_into_view_if_needed();page.screenshot(path=str(out/'SYNTHETIC_GALLERY_DESKTOP.png'));page.set_viewport_size({'width':390,'height':844});page.locator('#photoRange').scroll_into_view_if_needed();assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(out/'SYNTHETIC_GALLERY_MOBILE.png'));page.set_viewport_size({'width':1440,'height':1000});checks.append('Desktop/mobile render without horizontal overflow')
            page.click('#submit');expect(page.locator('#saveStatus')).to_have_text('Submitted and locked.');expect(page.locator('#submit')).to_be_disabled();checks.append('Submit locks the item and updates progress')
            for rater in ['A01','A02']:
                if rater=='A02':page.click('#logout');expect(page.locator('#login')).to_be_visible();login(rater);expect(page.locator('#progress')).to_have_text('Submitted 0 / 9')
                for current in s.list_items({'id':rater,'role':'rater'}):
                    open_item(current)
                    if current['status']=='SUBMITTED':continue
                    for c in CRITERIA:page.select_option('#score_'+c,'3')
                    page.click('#submit');expect(page.locator('#saveStatus')).to_have_text('Submitted and locked.')
                expect(page.locator('#progress')).to_have_text('Submitted 9 / 9')
                with page.expect_download() as event:page.click('#ownFinal')
                assert event.value.suggested_filename=='NEW_SESSION_'+rater+'_RESULTS_FINAL.json'
                dest=temp/(rater+'.json');event.value.save_as(dest);export=json.loads(dest.read_text());assert export['synthetic'] is True and export['final'] is True and len(export['records'])==9 and {r['annotator'] for r in export['records']}=={rater};assert export['records_sha256']==digest(export['records'])
            checks.append('Both fixture evaluators finish independently; separate final JSON downloads contain only their own9 synthetic records')
            assert not errors,errors;assert s.audit()['status']=='PASS';assert s.audit()['counts']['SUBMITTED']==18
            report=dict(status='PASS',browser=browser.version,bundled_runtime='Official Python3.11.9 Windows x64, no system installation',checks=checks,synthetic_fixture_records=18,research_rating_records_created=0,production_state_created=False,production_collection_open=False,javascript_errors=errors,largest_gallery_stress_photos=2040,default_thumbnail_DOM_bound=24,synthetic_export_files_in_delivery=False)
            put(out/'BROWSER_E2E_RESULTS.json',report);print(json.dumps(report,indent=2))
        finally:ctx.close();browser.close();proc.terminate();proc.communicate(timeout=10)
if __name__=='__main__':run()
