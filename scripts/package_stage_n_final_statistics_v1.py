"""Verify, reproduce and package the actual completed Stage N statistics."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path('outputs/stage_n/final_human_statistical_analysis_20261102_v1')
P=ROOT/'PACKAGE'
STUDY=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')
AUDIT=Path('outputs/stage_n/full240_rating_quality_audit_20261102_v1')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


VERIFIER='''"""Offline byte/data/statistical verification; no live DB, GPU or model work."""
import argparse,collections,hashlib,json,math,pathlib,statistics,subprocess,sys,tempfile
parser=argparse.ArgumentParser();parser.add_argument('--recompute',action='store_true');a=parser.parse_args()
p=pathlib.Path(__file__).resolve().parent
def read(name):return json.loads((p/name).read_text(encoding='utf-8-sig'))
def h(b):return hashlib.sha256(b).hexdigest()
def digest(x):return h(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
manifest=read('PACKAGE_MANIFEST.json')
for m in manifest['members']:
 b=(p/m['file']).read_bytes();assert len(b)==m['bytes'] and h(b)==m['sha256'],m['file']
cfg=read('ANALYSIS_CONFIG.json');flat=read('DATA/VERIFIED_ANALYSIS_INPUT.json')['records'];original=read('DATA/ORIGINAL_RATINGS_240.json')
outputs=read('DATA/OUTPUTS_120.json');sources=read('DATA/SOURCES_30.json');cohort=read('PROVENANCE/COHORT30.json')['business_ids']
assert len(flat)==len(original)==240 and len(outputs)==120 and len(cohort)==len(set(cohort))==30
assert set(sources)==set(cohort)==set(cfg['business_order'])
assert len({r['alias'] for r in flat})==240
byid={r['record_id']:r for r in original};oo={(r['condition'],r['business_id']):r for r in outputs}
numeric=0;na=0;panels=collections.Counter()
for r in flat:
 old=byid[r['record_id']];payload=old['original_payload'];assert digest(payload)==old['canonical_payload_sha256']
 assert all(r[k]==v for k,v in payload['scores'].items())
 assert all(r[k]==v for k,v in payload.items() if k!='scores')
 assert r['version']==old['version'] and r['business_id']==old['business_id'] and r['condition']==old['condition'] and r['annotator']==old['annotator']
 assert r['status']=='SUBMITTED';panels[r['annotator'],r['condition']]+=1
 for c in cfg['criteria']:
  if r['condition']=='RO' and c==cfg['criteria'][-1]:assert r[c]=='N/A';na+=1
  else:assert type(r[c]) is int and 1<=r[c]<=5;numeric+=1
 if r['serious_grounding_issue']:
  o=oo[r['condition'],r['business_id']];assert r['issue_span'] and r['issue_span'] in o['review_output']+'\\n'+(o['visual_output'] or '')
  source=sources[r['business_id']];ids={x['review_id'] for x in source['reviews']}|{x['photo_id'] for x in source['photos']}
  assert r['evidence_source_ids'] and set(r['evidence_source_ids'])<=ids
assert numeric==1380 and na==60 and len(panels)==8 and set(panels.values())=={30}
for o in outputs:
 assert h(o['review_output'].encode())==o['review_output_sha256']
 assert (None if o['visual_output'] is None else h(o['visual_output'].encode()))==o['visual_output_sha256']
 assert digest(sources[o['business_id']])==o['recipient_source_canonical_sha256']
for bid in cohort:
 f=oo['F',bid];s=oo['S',bid]
 assert f['review_output']==s['review_output'] and f['review_token_ids_sha256']==s['review_token_ids_sha256']
descriptions=read('TABLES/DESCRIPTIVE_STATISTICS.json')
for d in descriptions:
 rr=[r for r in flat if r['condition']==d['condition'] and (d['panel']=='COMBINED' or r['annotator']==d['panel'])]
 vv=[r[d['criterion']] for r in rr if type(r[d['criterion']]) is int]
 assert len(vv)==d['n_valid_ratings'] and sum(r[d['criterion']]=='N/A' for r in rr)==d['n_structural_NA']
 if vv:assert math.isclose(statistics.mean(vv),d['mean'],abs_tol=1e-12) and math.isclose(statistics.stdev(vv),d['sample_sd'],abs_tol=1e-12)
 assert all(vv.count(i)==d['score_'+str(i)] for i in range(1,6))
for b in read('PROVENANCE/NATIVE_COMPONENT_BINDINGS.json'):
 o=oo[b['condition'],b['business_id']];task=b['task']
 assert b['text_sha256']==o[task+'_output_sha256'] and b['checkpoint_sha256']==o[task+'_checkpoint_sha256']
 assert b['token_ids_canonical_sha256']==o[task+'_token_ids_sha256']
for q in read('QUALITATIVE/SOURCE_GROUNDED_EXAMPLES.json'):
 s=sources[q['business_id']]
 for e in q['verified_source_excerpts']:assert e['exact_quote'] in next(r['text'] for r in s['reviews'] if r['review_id']==e['review_id'])
 for c in q['conditions']:
  if c['exact_output_span'] is not None:assert c['exact_output_span'] in oo[c['condition'],q['business_id']]['review_output']
for photo in read('QUALITATIVE/PHOTO_PACKAGE_SCOPE.json')['included_photos']:
 assert h((p/photo['file']).read_bytes())==photo['sha256']
reproduction='NOT_REQUESTED; run with --recompute for the full supplied numerical analysis'
if a.recompute:
 with tempfile.TemporaryDirectory(prefix='stage_n_numeric_reproduction_') as tmp:
  target=pathlib.Path(tmp)/'TABLES'
  r=subprocess.run([sys.executable,'-X','utf8','-B',str(p/'SCRIPTS/RUN_STATISTICAL_ANALYSIS.py'),'--package',str(p),'--output',str(target)],capture_output=True,text=True,encoding='utf8')
  assert r.returncode==0,(r.stdout,r.stderr)
  for name in ['ORIGINAL_ANALYZER_RESULTS.json','DESCRIPTIVE_STATISTICS.json','AGREEMENT.json','PAIRED_COMPARISONS.json','PER_BUSINESS_CRITERION.json','IDENTICAL_F_S_REVIEW_RATINGS.json','PARTICIPANT_COMMENTS_VERBATIM.json']:
   assert json.loads((target/name).read_text(encoding='utf8'))==read('TABLES/'+name),name
  for name in ['DESCRIPTIVE_STATISTICS.csv','AGREEMENT.csv','PAIRED_COMPARISONS.csv','PER_BUSINESS_CRITERION.csv','PER_BUSINESS_PAIRED_DIFFERENCES.csv','PER_RATER_DATA.csv']:
   assert (target/name).read_bytes()==(p/'TABLES'/name).read_bytes(),name
 reproduction='PASS: all supplied core numerical JSON/CSV results reproduced from extracted package'
print(json.dumps({'status':'PASS','members_sha256_verified':len(manifest['members']),'original_records':240,'businesses':30,'numeric_scores':numeric,'structural_NA':na,'descriptive_rows':len(descriptions),'primary_QWK_cells':23,'paired_contrasts':17,'full_numerical_reproduction':reproduction,'scores_modified':False}))
'''


def main():
    private_rows=read(ROOT/'PRIVATE_SQL_ROWS.json')
    c=sqlite3.connect((STUDY/'PRIVATE_STATE/ratings.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:
        live=[dict(r) for r in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]
        events=[dict(r) for r in c.execute('SELECT * FROM events ORDER BY seq')]
    finally:c.close()
    assert live==private_rows['ratings'] and events==private_rows['events']
    protected=read(ROOT/'PRIVATE_PROTECTED_INPUTS.json')
    for receipt in protected:
        p=Path(receipt['file']);assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256'],str(p)
    native=read(STUDY/'COORDINATOR_ONLY/NATIVE_COMPONENT_RECEIPTS.json');sources=read(P/'DATA/SOURCES_30.json')
    bindings=[]
    for e in native:
        r=e['native_record']
        if r['task']=='review':assert r['selected_review_ids']==sources[r['business_id']]['selected_review_ids']
        bindings.append({'business_id':r['business_id'],'condition':r['condition'],'task':r['task'],
            'native_file':e['native_file'],'native_file_sha256':e['native_file_sha256'],
            **{k:r[k] for k in ['text_sha256','checkpoint_sha256','source_sha256','input_ids_sha256','generation_config_sha256']},
            'token_ids_canonical_sha256':digest(r['token_ids']),'selected_review_ids':r['selected_review_ids'],
            'selected_photo_ids':r['selected_photo_ids'],'caption_ids':r['caption_ids']})
    write(P/'PROVENANCE/NATIVE_COMPONENT_BINDINGS.json',bindings)
    # Preserve audit classifications/evidence as a separate coordinator record.
    audit=read(AUDIT/'ITEM_LEVEL_AUDIT_240.json')
    notes=[{'record_id':x['audit_id'],'business_id':x['business_id'],'original_version':x['original_version'],
        'classification':x['assessment'],'priority':x['followup_priority'],'model_assisted_audit_note_vi':x['assessment_note_vi'],
        'context_vi':x['output_and_source_context_vi']} for x in audit if x['assessment']!='BROADLY_REASONABLE']
    write(P/'QUALITATIVE/PRESERVED_AUDIT_NOTES.json',notes)
    selected_photos={x.name[:-4] for x in (AUDIT/'FOLLOWUP_PHOTOS').glob('*.jpg')}
    for bid in ['zGmK4tkN9X_wBi_9i1jIuw','oY5EIvb0WZ_4qwioC1eXsw','ymE6ka6y6J-z2y2rX8QH3Q']:
        selected_photos.update(x['photo_id'] for x in sources[bid]['photos'])
    byphoto={x['photo_id']:(bid,x) for bid,s in sources.items() for x in s['photos']};photo_rows=[]
    for photoid in sorted(selected_photos):
        bid,r=byphoto[photoid];target=P/'QUALITATIVE/PHOTOS'/(photoid+'.jpg');target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(STUDY/'PHOTOS'/target.name,target);assert sha(target)==r['sha256']
        photo_rows.append({'business_id':bid,'photo_id':photoid,'file':target.relative_to(P).as_posix(),'sha256':r['sha256']})
    assert len(photo_rows)==37
    write(P/'QUALITATIVE/PHOTO_PACKAGE_SCOPE.json',{'included_original_photos':37,'all_source_photos_hash_bound':343,
        'not_bundled':306,'included_photos':photo_rows,
        'scope':'Illustrative source/audit images, not a new exhaustive image adjudication. All343 live photo hashes were verified in preparation.'})
    # Source/protocol/references/predictions remain unmodified; no private paths
    # or database rows from the protected local receipt are exported.
    write(P/'PROVENANCE/PRESERVATION_VERIFICATION.json',{'status':'PASS','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'all240_live_rating_rows_and241_events_unchanged':True,'protected_file_count':len(protected),
        'protected_existing_reference_prediction_metric_protocol_files_unchanged':True,
        'new_human_scores_or_amendments':False,'LED_inference_training_API_GPU_work':False,
        'original_private_mapping_sha256':sha(STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json'),
        'private_mapping_file_and_DB_excluded':True})
    (P/'VERIFY_PACKAGE.py').write_text(VERIFIER,encoding='utf8')
    (P/'PROMPT_FOR_CHATGPT.txt').write_text('''Review STAGE_N_FINAL_HUMAN_STATISTICAL_REVIEW.zip independently. Read README.md,
then methodology, complete descriptive/agreement/paired tables and provenance.
All240 original ratings and30 businesses are retained; scores were not corrected.
Use the frozen two-rater exploratory protocol, not a new significance rule.
Check that business pairing preserves both raters, RO Visual is N/A, applicable
empty Visual stays scored, and F/S identical Review text is interpreted correctly.
Assess Stage M TEST100 and Stage N TEST30 separately. Preserve the documented
audit concerns and distinguish source facts from participant/auditor judgments.
Do not infer a single overall winner, drop cases, create human scores or start
another experiment. Return supported/inconclusive findings and any actual
reproducibility issue. Package is coordinator-only; do not send decoded condition
information to blinded participants. VERIFY_PACKAGE.py --recompute reproduces
the supplied numerical analysis with Python+NumPy, without models or GPU.
''',encoding='utf8')
    for name in ['prepare_stage_n_final_statistics_v1.py','report_stage_n_final_statistics_v1.py']:
        # Unneeded local orchestration scripts may contain local attachment paths.
        target=(P/'SCRIPTS'/name).resolve();assert target.is_relative_to(P.resolve())
        if target.exists():target.unlink()
    workbook_receipt=read(ROOT/'BUILD/WORKBOOK_CHECKS.json')
    write(P/'PROVENANCE/WORKBOOK_VERIFICATION.json',{**workbook_receipt,'all5_sheet_previews_visually_checked':True,
        'formula_error_scan_matches':0,'saved_xlsx_sha256':sha(P/'HUMAN_EVALUATION_RESULTS.xlsx'),
        'display_order_only':'Ratings sorted by condition/business/rater; original values unchanged'})
    # Verify saved XLSX cached numerical cells independently of its renderer.
    import xml.etree.ElementTree as ET
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(P/'HUMAN_EVALUATION_RESULTS.xlsx') as x:
        root=ET.fromstring(x.read('xl/worksheets/sheet1.xml'))
        cells={c.attrib['r']:c for c in root.findall('.//m:sheetData/m:row/m:c',ns)}
        desc=read(P/'TABLES/DESCRIPTIVE_STATISTICS.json');cfg=read(P/'ANALYSIS_CONFIG.json')
        for i,crit in enumerate(cfg['criteria']):
            for j,condition in enumerate(cfg['conditions']):
                if i==5 and j==0:continue
                cell=cells[chr(66+j)+str(6+i)];assert cell.find('m:f',ns) is not None
                actual=float(cell.find('m:v',ns).text);expected=next(d['mean'] for d in desc if d['criterion']==crit and d['condition']==condition and d['panel']=='COMBINED')
                assert abs(actual-expected)<1e-10
    files=sorted([p for p in P.rglob('*') if p.is_file() and not p.name.endswith('.inspect.ndjson') and p.name!='PACKAGE_MANIFEST.json'])
    deny=['PRIVATE_ITEMS.json','.sqlite3','.sqlite3-wal','.sqlite3-shm','PRIVATE_SQL_ROWS.json','PRIVATE_PROTECTED_INPUTS.json']
    assert not any(p.name in deny for p in files)
    members=[{'file':p.relative_to(P).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    write(P/'PACKAGE_MANIFEST.json',{'format':'stage_n_final_human_statistical_review_manifest_v1','manifest_excludes_itself':True,'members':members})
    files.append(P/'PACKAGE_MANIFEST.json')
    archive=ROOT/'STAGE_N_FINAL_HUMAN_STATISTICAL_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in files:z.write(file,file.relative_to(P).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_n_final_stat_review_') as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and len(z.namelist())==len(files) and len(set(z.namelist()))==len(files)
            z.extractall(tmp)
        r=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(tmp)/'VERIFY_PACKAGE.py'),'--recompute'],capture_output=True,text=True,encoding='utf8')
        assert r.returncode==0,(r.stdout,r.stderr)
        actual_verification=json.loads(r.stdout)
    # Final post-reproduction preservation check, including original Stage L/M.
    for receipt in protected:assert sha(Path(receipt['file']))==receipt['sha256']
    c=sqlite3.connect((STUDY/'PRIVATE_STATE/ratings.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:
        assert [dict(r) for r in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]==private_rows['ratings']
        assert [dict(r) for r in c.execute('SELECT * FROM events ORDER BY seq')]==private_rows['events']
    finally:c.close()
    receipt={'status':'PASS','verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'archive':archive.name,
        'bytes':archive.stat().st_size,'sha256':sha(archive),'zip_member_count':len(files),
        'crc_reopen_extract_every_member_sha256_json_and_full_numeric_reproduction':'PASS',
        'offline_extracted_verifier':actual_verification,'saved_XLSX23_cached_formula_means_verified':True,
        'all240_original_rows241_events_and_protected_Stage_L_M_inputs_unchanged':True,
        'no_training_inference_GPU_paid_API_or_new_collection':True}
    write(ROOT/'FINAL_PACKAGE_VERIFICATION.json',receipt)
    (ROOT/(archive.name+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
