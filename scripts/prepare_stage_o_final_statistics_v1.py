"""Read-only preparation of the final Stage O statistical review inputs."""
import hashlib,json,shutil,datetime as dt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_o/final_human_statistical_analysis_20261106_v1'
P=OUT/'PACKAGE'
H=ROOT/'outputs/stage_o/completed_human_evaluation_handoff_20261106_v1/PACKAGE'
W=ROOT/'outputs/stage_o/tripadvisor_external_v1/HUMAN_APP/WORKSPACE'
N=ROOT/'outputs/stage_n/final_human_statistical_analysis_20261102_v1/PACKAGE'
O=ROOT/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE/STAGE_O'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    assert not P.exists(),'Use a new preparation destination'
    protected=[]
    def copy(src,dest):
        dest=P/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        protected.append(dict(original_path=src.relative_to(ROOT).as_posix(),package_path=dest.relative_to(P).as_posix(),bytes=src.stat().st_size,sha256=sha(src)))
    for f in H.rglob('*'):
        if f.is_file():copy(f,Path('COLLECTION')/f.relative_to(H))
    export=read(H/'STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json')
    assert sha(H/'STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json')=='d89c2fc9ed7c228139a1594c9ff7b1f7b2ffda2dfbe18bff435a7ccb6a13df7f'
    cfg=read(W/'CONFIG.json');items=read(W/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')
    for rel,h in cfg['file_bindings'].items():assert sha(W/rel)==h
    assert sha(W/'CONFIG.json')==export['config_sha256']
    copy(W/'COORDINATOR_ONLY/PRIVATE_ITEMS.json','DATA/ORIGINAL_PRIVATE_ITEMS_90.json')
    photos=[];review_count=0
    for b,binding in cfg['sources'].items():
        src=W/binding['file'];s=read(src);assert digest(s)==binding['canonical_sha256']
        assert b==s['business_id'] and s['split']=='external_tripadvisor'
        assert [r['review_id'] for r in s['reviews']]==s['selected_review_ids']
        for r in s['reviews']:assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        review_count+=len(s['reviews']);copy(src,Path('DATA/SOURCES')/src.name)
        for ph in s['photos']:
            f=W/'PHOTOS'/(ph['photo_id']+'.jpg');h=sha(f)
            assert h==ph['sha256']==ph['original_source_sha256'] and f.stat().st_size==ph['bytes']
            photos.append(dict(business_id=b,photo_id=ph['photo_id'],sha256=h,bytes=f.stat().st_size,original_path=f.relative_to(ROOT).as_posix()))
    predictions=read(O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json');assert len(predictions)==45
    lookup={(r['business_id'],r['condition']):r for r in predictions};assert len(lookup)==45
    native=[]
    for c in ['RO','U','F']:
        f=O/f'LED_OUTPUTS/{c}/raw_predictions.json';native.extend(read(f));copy(f,f'DATA/NATIVE/{c}_raw_predictions.json')
    nk={(r['business_id'],r['condition'],r['task']):r for r in native};assert len(nk)==75
    for i in items:
        a=lookup[i['business_id'],i['condition']]
        for comp,key in [('review','review_text'),('visual','visual_text')]:
            text=i[key]
            if i['condition']=='RO' and comp=='visual':assert text is None;continue
            h=hashlib.sha256(text.encode()).hexdigest()
            r=nk[i['business_id'],i['condition'],comp]
            assert text==a['component_texts'][comp]==r['text'] and h==i[key+'_sha256']==a['component_hashes'][comp]==r['text_sha256']
    copy(O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json','DATA/ORIGINAL_OUTPUTS_45.json')
    for name in ['COHORT15_FREEZE.json','FINAL_CONDITION_INPUTS_45.json','LED_INFERENCE_CALL_PLAN_75.json']:
        copy(O/name,'DATA/NATIVE/'+name)
    for rel in ['TABLES/DESCRIPTIVE_STATISTICS.json','TABLES/PAIRED_COMPARISONS.json','TABLES/AGREEMENT.json','QUALITATIVE_FINDINGS_AND_LIMITATIONS.md','QUALITATIVE/PRESERVED_AUDIT_NOTES.json','ANALYSIS_CONFIG.json','PACKAGE_MANIFEST.json']:
        copy(N/rel,'YELP_TEST30/'+rel)
    config=dict(format='stage_o_final_statistics_v1',business_order=sorted(cfg['sources']),conditions=['RO','U','F'],raters=['A01','A02'],
        criteria=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution'],
        seed=20261028,bootstrap_draws=20000,signflip_draws=50000,n_businesses=15,
        inference='EXPLORATORY; pointwise percentile intervals; raw two-sided Monte Carlo sign-flip p with +1 correction; no multiplicity adjustment',
        primary_plan='F-U all six criteria and F-RO Review-only criteria; frozen two-rater amendment',
        supplementary='F-RO coherence/non-redundancy requested as applicable comparisons; component-budget confounded, not explicitly prespecified',
        aggregation='Both raters averaged within restaurant; restaurant resampling retains both raters/all conditions',
        no_composite=True,no_exclusions=True,created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    put(P/'ANALYSIS_CONFIG.json',config)
    put(P/'PROVENANCE/PROTECTED_INPUT_FILES.json',protected)
    put(P/'PROVENANCE/PHOTO_INTEGRITY_5143.json',photos)
    put(P/'PROVENANCE/INPUT_VERIFICATION.json',dict(status='PASS',records=90,unique_outputs=45,native_components=75,businesses=15,
        reviews=review_count,photos=len(photos),all_photo_bytes_hashed=True,source_canonical_bindings=True,
        original_text_to_native_component_binding=True,ratings_sha256=sha(H/'STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json'),
        checkpoint_weights_rehashed=False,model_provenance='Existing native checkpoint SHA receipts; no checkpoint files modified or newly loaded',
        identity_or_source_inspection_independently_authenticated=False))
    for name in ['analyze_stage_o_final_statistics_v1.py','report_stage_o_final_statistics_v1.py','verify_stage_o_final_statistics_v1.py']:
        src=ROOT/'scripts'/name
        if src.exists():shutil.copy2(src,P/'SCRIPTS'/name) if (P/'SCRIPTS').exists() else None
    print(json.dumps(dict(package=str(P),reviews=review_count,photos=len(photos),status='PASS')))
if __name__=='__main__':main()
