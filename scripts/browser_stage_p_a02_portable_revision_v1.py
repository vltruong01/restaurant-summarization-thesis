"""Browser checks with disposable state, not the participant's actual records."""
from pathlib import Path
import json,sys,tempfile,threading
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'outputs/stage_p/a02_portable_revision_20261106_v1/STAGE_P_A02_REVISION_APP'
sys.path.insert(0,str(APP/'APP'))
from stage_p_a02_portable_revision_v1 import RevisionStudy,Server

def main():
    errors=[]
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC_A02_BROWSER_') as td:
        study=RevisionStudy(APP/'WORKSPACE',Path(td)/'STATE')
        study.add_user('A02','rater','Synthetic-Browser-Only-5643')
        server=Server(('127.0.0.1',0),study,APP/'WEB');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with sync_playwright() as pw:
                browser=pw.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu'])
                context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
                photo_responses=[]
                page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('dialog',lambda d:d.accept())
                page.on('response',lambda r:photo_responses.append(r.status) if '/api/photo/' in r.url else None)
                page.goto('http://127.0.0.1:'+str(server.server_port)+'/')
                page.fill('#password','Synthetic-Browser-Only-5643');page.locator('#loginForm button').click()
                expect(page.locator('#items option')).to_have_count(45)
                page.click('#open');expect(page.locator('#reviewOutput')).not_to_be_empty();expect(page.locator('#visualOutput')).not_to_be_empty()
                assert page.locator('#reviews article').count()>0
                assert page.locator('#photos img').count()>0
                page.locator('#photos img').first.scroll_into_view_if_needed()
                page.wait_for_load_state('networkidle')
                page.locator('#photos img').first.screenshot(path=str(Path(td)/'SOURCE_THUMBNAIL.png'))
                assert photo_responses and all(x==200 for x in photo_responses)
                assert all(page.locator('#score_'+c).input_value() in ['1','2','3','4','5'] for c in study.rubric['criteria'] for c in [c['id']])
                page.locator('#photos a').first.click();expect(page.locator('#photoDialog')).to_be_visible()
                page.wait_for_load_state('networkidle')
                page.locator('#photoLarge').screenshot(path=str(Path(td)/'SOURCE_LARGE.png'))
                page.click('#photoClose')
                page.fill('#notes','SYNTHETIC BROWSER TEST ONLY - never delivered as a human record.')
                page.uncheck('#serious');page.click('#save');expect(page.locator('#saveStatus')).to_have_text('Draft saved on server. You may resume later.')
                page.reload();expect(page.locator('#items option')).to_have_count(45);page.click('#open')
                expect(page.locator('#notes')).to_have_value('SYNTHETIC BROWSER TEST ONLY - never delivered as a human record.')
                page.click('#submit');expect(page.locator('#saveStatus')).to_have_text('Submitted and locked.')
                with page.expect_download() as event:page.click('#ownFinal')
                download=event.value;assert download.suggested_filename=='A02_REVISED_RESULTS_FINAL.json'
                path=Path(td)/'SYNTHETIC_EXPORT.json';download.save_as(path);d=json.loads(path.read_text(encoding='utf8'))
                assert len(d['records'])==45 and d['portable_revision']['inherited_original_count']==44 and len(d['portable_revision']['changes'])==1
                assert not errors,errors
                context.close();browser.close()
        finally:server.shutdown();server.server_close();thread.join()
    assert not (APP/'STATE').exists()
    report=dict(status='PASS',items=45,review_and_visual_visible=True,source_reviews_and_photo_pixels_visible=True,prefilled_six_scores=True,save_reload_submit_and_JSON_download=True,unchanged_originals_retained=44,javascript_errors=errors,research_records_written=0,disposable_test_state_in_delivery=False)
    (APP.parent/'BROWSER_VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
