"""Freeze a NEW90-record session, with no old ratings or model identities in rater data."""
from pathlib import Path
import json,hashlib,shutil,secrets,random,itertools,collections,zipfile,datetime
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1';P=D/'PACKAGE';APP=P/'EVALUATOR_APP';O=ROOT/'outputs/stage_o/tripadvisor_external_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not P.exists();D.mkdir(exist_ok=True);P.mkdir();w=APP/'WORKSPACE';co=P/'COORDINATOR_ONLY';co.mkdir();w.mkdir(parents=True)
    old=O/'HUMAN_APP';cohort=read(O/'COHORT15_FREEZE.json');ids=[r['business_id'] for r in cohort['restaurants']]
    frozen=read(ROOT/'outputs/stage_p/three_system_comparison_20261106_v1/REVIEW_PACKAGE/THREE_SYSTEM_PREDICTIONS_45.json');assert len(frozen)==45
    by={(r['business_id'],r['system']):r for r in frozen};systems=['LED F original','Qwen-Caption','Qwen-Image'];assert set(k[0] for k in by)==set(ids)
    native={(x['business_id'],x['task']):x for x in read(ROOT/'outputs/stage_p/three_system_comparison_20261106_v1/NATIVE/INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')}
    for b in ids:
        assert by[b,'LED F original']['review_text']==native[b,'review']['text'] and by[b,'LED F original']['visual_text']==native[b,'visual']['text']
        assert by[b,'Qwen-Caption']['review_text']==by[b,'Qwen-Image']['review_text']
    protected=[]
    for f in [ROOT/'outputs/stage_o/human_collection_complete_20261106_v1/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json',ROOT/'outputs/stage_o/human_collection_complete_20261106_v1/PRIVATE_BACKUP/ratings.sqlite3',O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json',ROOT/'outputs/stage_p/three_system_comparison_20261106_v1/REVIEW_PACKAGE/THREE_SYSTEM_PREDICTIONS_45.json']:
        protected.append(dict(file=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)))
    for f in (old/'WORKSPACE').rglob('*'):
        if f.is_file() and f.suffix not in ['.sqlite3-wal','.sqlite3-shm']:protected.append(dict(file=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)))
    put(D/'PROTECTED_PRIOR_ARTIFACTS.json',protected)
    sources={};photos={}
    (w/'SOURCES').mkdir();(w/'PHOTOS').mkdir()
    original_cfg=read(old/'WORKSPACE/CONFIG.json')
    for b in ids:
        src=old/'WORKSPACE/SOURCES'/f'{b}.json';shutil.copy2(src,w/'SOURCES'/src.name);s=read(src);sources[b]=dict(file='SOURCES/'+src.name,canonical_sha256=hashlib.sha256(canon(s)).hexdigest());assert sources[b]==original_cfg['sources'][b]
        for ph in s['photos']:
            if ph['photo_id'] not in photos:
                f=old/'WORKSPACE/PHOTOS'/(ph['photo_id']+'.jpg');assert sha(f)==ph['sha256'];shutil.copy2(f,w/'PHOTOS'/f.name);photos[ph['photo_id']]=ph
    assert len(photos)==5143
    shutil.copy2(old/'WORKSPACE/RUBRIC.json',w/'RUBRIC.json')
    study_id='tripadvisor15_three_system_new_session_v1_'+secrets.token_hex(8);items=[];key=[];seeds={};balance={}
    for rater in ['A01','A02']:
        seed=secrets.token_hex(32);seeds[rater]=seed;rng=random.Random(int(seed,16));order=ids.copy();rng.shuffle(order);schedule=[]
        for cycle in range(5):
            labels=systems.copy();rng.shuffle(labels);shifts=[0,1,2];rng.shuffle(shifts);columns=[0,1,2];rng.shuffle(columns)
            schedule.extend([[labels[(j+shift)%3] for j in columns] for shift in shifts])
        rng.shuffle(schedule);counts=collections.Counter()
        for block,(b,models) in enumerate(zip(order,schedule)):
            for slot,system in enumerate(models,1):
                output=by[b,system];alias=hashlib.sha256((seed+'|'+b+'|'+str(slot)).encode()).hexdigest()[:24];pos=block*3+slot
                item=dict(alias=alias,annotator=rater,position=pos,business_id=b,condition='C'+str(slot),review_text=output['review_text'],review_text_sha256=output['review_text_sha256'],visual_text=output['visual_text'],visual_text_sha256=output['visual_text_sha256']);items.append(item)
                key.append(dict(alias=alias,annotator=rater,position=pos,business_id=b,neutral_slot='C'+str(slot),within_restaurant_position=slot,system=system,review_text_sha256=item['review_text_sha256'],visual_text_sha256=item['visual_text_sha256']));counts[system,slot]+=1
        assert all(counts[s,p]==5 for s in systems for p in [1,2,3]);balance[rater]=[dict(system=s,position=p,count=counts[s,p]) for s in systems for p in [1,2,3]]
    assert len(items)==90 and len({i['alias'] for i in items})==90
    put(w/'BLINDED_ITEMS.json',items);put(co/'ASSIGNMENT_KEY_90.json',key);put(co/'RANDOMIZATION_SEEDS.json',seeds);put(co/'POSITION_BALANCE.json',balance);put(co/'FROZEN_OUTPUTS_45.json',frozen)
    protocol='''# New independent reference-free session

15 frozen restaurants,three anonymous outputs each,two separate evaluators,45 items each,90 new records. All outputs contain Review Consensus and Visual Evidence. All six existing criteria/anchors remain byte-identical in RUBRIC.json. Read only the displayed selected reviews and original photo evidence for the current restaurant. Score Review Faithfulness,Aspect Coverage and Usefulness against reviews; Coherence and Non-Redundancy across both components; Visual Grounding and Contribution against original photographs. Keep the same1-5 scale;2 and4 are intermediate. Never substitute old scores. Work independently without exchanging judgments. Do not penalize an item merely because another item has identical text. Source galleries contain all original photos and are paginated; the first page is not the complete source set. Comments may be blank unless reporting a serious defect,which requires exact output span,valid source IDs,explanation and certainty. Save draft to pause; submit locks a completed item. Existing prior session records remain separate. No reference summaries,previous scores,resource figures or diagnostics are supplied. New-session account actions and timestamps do not authenticate human identity or source inspection independently.
'''
    (w/'TWO_RATER_PROTOCOL_AMENDMENT.md').write_text(protocol,encoding='utf8')
    config=dict(format='stage_p_three_system_blinded_human_v1',study_id=study_id,expected_records=90,conditions=['C1','C2','C3'],raters=['A01','A02'],businesses=15,sources=sources,synthetic=False,cohort_freeze_sha256=sha(O/'COHORT15_FREEZE.json'),file_bindings={n:sha(w/n) for n in ['BLINDED_ITEMS.json','RUBRIC.json','TWO_RATER_PROTOCOL_AMENDMENT.md']});put(w/'CONFIG.json',config)
    put(co/'BINDINGS.json',dict(study_id=study_id,config_sha256=sha(w/'CONFIG.json'),assignment_key_sha256=sha(co/'ASSIGNMENT_KEY_90.json'),blinded_items_sha256=sha(w/'BLINDED_ITEMS.json'),rubric_sha256=sha(w/'RUBRIC.json'),rubric_exact_Stage_O=True,old_rating_records_reused=0,new_expected_records=90,source_photos=5143,selected_reviews=sum(len(read(w/v['file'])['reviews']) for v in sources.values()),benchmark_LED_outputs_used=False,shared_Qwen_Review_businesses=15,scope='New concurrently collected three-system ratings; original Stage O remains separate.'))
    appsrc=APP/'APP';appsrc.mkdir();web=APP/'WEB';web.mkdir()
    for name in ['human_http_common_v1.py','human_app_core_v1.py','stage_p_human_evaluation_v1.py','launch_stage_p_human_v1.py']:shutil.copy2(ROOT/'scripts'/name,appsrc/name)
    for name in ['app.js','app.css','index.html']:shutil.copy2(old/'scripts/stage_o_human_web_v1'/name,web/name)
    html=(web/'index.html').read_text(encoding='utf8').replace('Stage O','New session').replace('STAGE O','INDEPENDENT SESSION').replace('Stage L reference involvement','Prior reference involvement')
    html=html.replace('<nav><h2 id="progress">','<div class="actions"><button id="ownBackup">Download my progress</button><button id="ownFinal">Download my completed results</button></div><nav><h2 id="progress">')
    (web/'index.html').write_text(html,encoding='utf8')
    js=(web/'app.js').read_text(encoding='utf8').replace('stage_o','stage_p').replace('Stage O live study released','New independent session open')
    js+='\n$("ownBackup").onclick=()=>download("/api/export",session.actor.id+"_PROGRESS.json").catch(error);\n$("ownFinal").onclick=()=>download("/api/export?final=1",session.actor.id+"_RESULTS_FINAL.json").catch(error);\n'
    (web/'app.js').write_text(js,encoding='utf8')
    runtime=ROOT/'outputs/stage_n/A02_portable_20261102_v1/python-3.11.9-embed-amd64.zip'
    assert hashlib.md5(runtime.read_bytes()).hexdigest()=='6d9aa08531d48fcc261ba667e2df17c4'
    with zipfile.ZipFile(runtime) as z:assert z.testzip() is None;z.extractall(APP/'PYTHON')
    (APP/'PYTHON/python311._pth').write_text('python311.zip\n.\n..\\APP\n',encoding='ascii')
    put(APP/'RUNTIME_PROVENANCE.json',dict(source='Existing verified official Windows x64 embeddable Python3.11.9 archive',source_url='https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip',upstream_MD5='6d9aa08531d48fcc261ba667e2df17c4',upstream_SHA256=sha(runtime),license='PYTHON/LICENSE.txt',change='Relative APP directory in python311._pth; no system Python,pip,GPU or internet required.'))
    for name,args in [('START_PREVIEW.cmd',''),('START_COLLECTION.cmd','--open-session')]:
        (APP/name).write_bytes(('@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n"%~dp0PYTHON\\python.exe" -B -X utf8 "%~dp0APP\\launch_stage_p_human_v1.py" '+args+'\r\nif errorlevel 1 pause\r\n').encode('ascii'))
    put(D/'PREPARATION_RECEIPT.json',dict(status='FROZEN_PENDING_TECHNICAL_TESTS',study_id=study_id,items=90,outputs=45,restaurants=15,raters=2,source_photos=len(photos),review_rubric_sha256=sha(w/'RUBRIC.json'),new_ratings_created=0,collection_open=False,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Prepared NEW study90 assignments,balanced5/system/slot/rater; no ratings or collection marker.')
if __name__=='__main__':main()
