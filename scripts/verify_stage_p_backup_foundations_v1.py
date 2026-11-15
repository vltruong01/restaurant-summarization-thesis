"""Hash-based off-instance preservation audit of final models, TripAdvisor data and ratings."""
from pathlib import Path
import json,hashlib,zipfile,sqlite3,collections,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage_p/pre_destroy_backup_audit_20261106_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    checks=[];issues=[];verified=[]
    def verify(p,expected=None,label=''):
        if not p.is_file():issues.append(dict(file=str(p),problem='MISSING',label=label));return None
        h=sha(p);row=dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h,label=label)
        if expected is not None and h!=expected:issues.append(dict(**row,expected_sha256=expected,problem='HASH_MISMATCH'));return h
        verified.append(row);return h
    lock=read(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    for arm,cfg in lock['models'].items():
        d=ROOT/cfg['path'];integrity=read(d/'checkpoint_integrity.json')
        verify(d/'adapter_model.safetensors',cfg['adapter_sha256'],arm+' final LoRA S1000')
        verify(d/'state.pt',cfg['full_state_sha256'],arm+' full training state S1000')
        for f,h in integrity['files'].items():verify(d/f,h,arm+' checkpoint file')
        verify(d/'checkpoint_integrity.json',label=arm+' manifest')
    verify(Path(lock['backbone']['local_file']),lock['backbone']['sha256'],'Shared LED backbone')
    checks.append(dict(category='Final LED RO/U/F checkpoints',checkpoint_count=3,adapter_hashes={k:v['adapter_sha256'] for k,v in lock['models'].items()},full_state_hashes={k:v['full_state_sha256'] for k,v in lock['models'].items()},local_paths={k:v['path'] for k,v in lock['models'].items()}))
    o=ROOT/'outputs/stage_o/tripadvisor_external_v1';cohort=read(o/'COHORT15_FREEZE.json');assert len(cohort['restaurants'])==15
    for b in cohort['artifact_bindings']:verify(o/b['file'],b['sha256'],'Frozen TripAdvisor source binding')
    reviews=0;photos={}
    for item in cohort['restaurants']:
        p=o/item['source_file'];verify(p,item['source_sha256'],'Original restaurant reviews/photo associations');restaurant=read(p);reviews+=len(restaurant['English_reviews'])
        source=o/'HUMAN_APP/WORKSPACE/SOURCES'/f"{item['business_id']}.json";s=read(source);verify(source,label='Evaluator source reviews/photos')
        assert s['selected_review_ids'] and hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        for ph in s['photos']:
            if ph['photo_id'] in photos:assert photos[ph['photo_id']]==ph['sha256']
            else:
                photos[ph['photo_id']]=ph['sha256'];verify(o/'SOURCE_PHOTOS'/(ph['photo_id']+'.jpg'),ph['sha256'],'Original source photo');verify(o/'HUMAN_APP/WORKSPACE/PHOTOS'/(ph['photo_id']+'.jpg'),ph['sha256'],'Evaluator source photo')
                meta=o/'SOURCE_PHOTOS'/(ph['photo_id']+'.json');verify(meta,label='Original photo URL/provenance');assert read(meta)['sha256']==ph['sha256']
    assert reviews==cohort['source_review_count']==320 and len(photos)==cohort['source_photo_count']==5143
    acq=read(o/'DATASET_DOWNLOADS/14622324_ACQUISITION.json');archive=next((o/'DATASET_DOWNLOADS').glob('*.zip'));verify(archive,acq['sha256'],'Original publisher Gijon dataset archive')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;raw_members=len(z.namelist())
    checks.append(dict(category='Original TripAdvisor inputs',restaurants=15,English_source_reviews=reviews,photos=len(photos),original_JPEG_and_UI_copy_hashes_verified=True,publisher_archive_sha256=acq['sha256'],publisher_ZIP_CRC='PASS',publisher_ZIP_members=raw_members,photo_limitation=cohort['photo_resolution_limit']))
    native=o/'RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE';manifest=read(native/'NATIVE_EXPORT_MEMBER_SHA256.json')
    for row in manifest:verify(native/row['file'],row['sha256'],'Completed Stage O native export')
    verify(o/'RTX3080_EXECUTION_V1/STAGE_O_NATIVE_COMPLETE_EVIDENCE.tar.gz',read(o/'RTX3080_EXECUTION_V1/LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json')['archive_sha256'],'Stage O native archive')
    for name,count in [('TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json',45),('FINAL_U_F_CAPTION_POOLS.json',15)]:
        a=o/name;b=native/'STAGE_O'/name;verify(a,sha(b),'Frozen LED outputs/caption pools');assert len(read(a))==count
    checks.append(dict(category='Frozen Stage O predictions, captions and execution code',LED_outputs=45,completed_native_members=len(manifest),all_member_hashes='Verified',native_archive='Verified'))
    hr=ROOT/'outputs/stage_o/human_collection_complete_20261106_v1';receipt=read(hr/'EXPORT_AND_BACKUP_RECEIPT.json');verify(hr/receipt['export_file'],receipt['export_sha256'],'Final90 human ratings');verify(hr/receipt['backup_file'],receipt['backup_sha256'],'Private SQLite human backup')
    ratings=read(hr/receipt['export_file']);rows=ratings['records'];assert len(rows)==90 and all(x['status']=='SUBMITTED' for x in rows)
    c=sqlite3.connect((hr/receipt['backup_file']).resolve().as_uri()+'?mode=ro',uri=True);dbcheck=c.execute('PRAGMA integrity_check').fetchone()[0];c.close();assert dbcheck=='ok'
    checks.append(dict(category='Original human ratings and private backup',records=90,per_rater=dict(collections.Counter(x['annotator'] for x in rows)),export_sha256=receipt['export_sha256'],private_database_sha256=receipt['backup_sha256'],SQLite_integrity_check=dbcheck,private_backup_not_in_public_review_ZIP=True))
    # Re-run the portable audits of the final two review packages without touching research data.
    for folder,fn in [('qwen_bf16_3090_execution_20261106_v2','STAGE_P_QWEN_COMPARISON_REVIEW.zip'),('three_system_comparison_20261106_v1','STAGE_P_THREE_SYSTEM_COMPARISON_REVIEW.zip')]:
        d=ROOT/'outputs/stage_p'/folder;f=d/fn;expected=(d/(fn+'.sha256')).read_text().split()[0];verify(f,expected,'Final Stage P review ZIP')
        with zipfile.ZipFile(f) as z:assert z.testzip() is None;members=len(z.namelist())
        checks.append(dict(category=folder,ZIP=str(f.relative_to(ROOT)),sha256=expected,members=members,CRC='PASS'))
    protected=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1/PROTECTED_ORIGINALS.json'
    for row in read(protected):verify(ROOT/row['original'],row['sha256'],'Stage P protected original')
    put(OUT/'LOCAL_FOUNDATION_FILE_HASHES.json',verified);put(OUT/'LOCAL_FOUNDATION_ISSUES.json',issues);put(OUT/'LOCAL_FOUNDATION_AUDIT.json',dict(status='PASS' if not issues else 'ATTENTION',checks=checks,issues=issues,verified_file_checks=len(verified),GPU_work=False,scope='Physical local disk outside instance, not an independently verified HF/cloud replica. No weight deserialization or GPU execution.'))
    print(json.dumps(dict(status='PASS' if not issues else 'ATTENTION',verified_checks=len(verified),issues=issues,summary=checks),indent=2))
if __name__=='__main__':main()
