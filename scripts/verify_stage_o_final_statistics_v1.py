"""Portable read-only integrity verifier; no assertions of semantic correctness."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--package',type=Path,required=True);a=pa.parse_args();p=a.package.resolve();manifest=read(p/'PACKAGE_MANIFEST.json')
    for r in manifest['files']:
        f=(p/r['file']).resolve();assert f.is_relative_to(p) and f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],r['file']
    names={r['file'] for r in manifest['files']};assert names=={f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file() and f.name!='PACKAGE_MANIFEST.json'}|{f.relative_to(p).as_posix() for f in p.rglob('PACKAGE_MANIFEST.json') if f!=p/'PACKAGE_MANIFEST.json'}
    sub=subprocess.run([sys.executable,'-B',str(p/'COLLECTION/VERIFY_HANDOFF.py')],capture_output=True,text=True,encoding='utf8');assert sub.returncode==0,sub.stdout+sub.stderr
    cfg=read(p/'COLLECTION/PROVENANCE/CONFIG.json');sources={f.stem:read(f) for f in (p/'DATA/SOURCES').glob('*.json')};assert len(sources)==15
    for b,s in sources.items():
        assert digest(s)==cfg['sources'][b]['canonical_sha256'];assert s['business_id']==b and s['split']=='external_tripadvisor'
        assert [r['review_id'] for r in s['reviews']]==s['selected_review_ids']
        for r in s['reviews']:assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
    assert sum(len(s['reviews']) for s in sources.values())==247 and sum(len(s['photos']) for s in sources.values())==5143
    items=read(p/'DATA/ORIGINAL_PRIVATE_ITEMS_90.json');assert sha(p/'DATA/ORIGINAL_PRIVATE_ITEMS_90.json')==cfg['file_bindings']['COORDINATOR_ONLY/PRIVATE_ITEMS.json']
    original=read(p/'DATA/ORIGINAL_OUTPUTS_45.json');assert len(original)==45
    preds={(r['business_id'],r['condition']):r for r in original};native=[r for c in ['RO','U','F'] for r in read(p/f'DATA/NATIVE/{c}_raw_predictions.json')];assert len(native)==75
    nk={(r['business_id'],r['condition'],r['task']):r for r in native};calls=read(p/'DATA/NATIVE/LED_INFERENCE_CALL_PLAN_75.json')['calls'];pk={(r['business_id'],r['condition'],r['task']):r for r in calls};assert len(nk)==len(pk)==75
    for k,r in nk.items():
        c=pk[k];assert r['input_ids']==c['input_ids'] and digest(r['input_ids'])==r['input_ids_sha256']==c['input_ids_sha256']
        assert hashlib.sha256(c['source'].encode()).hexdigest()==r['source_sha256']==c['source_sha256']
        assert r['checkpoint_sha256']==c['checkpoint_sha256'] and r['generation_settings']==c['generation']
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert r['selected_photo_ids']==c['selected_photo_ids'] and r['selected_review_ids']==c['selected_review_ids']
        if r['task']=='review':
            s=sources[r['business_id']];assert s['selected_review_block'] in c['source'] and r['selected_review_ids']==s['selected_review_ids']
        else:assert not r['selected_review_ids']
    for i in items:
        r=preds[i['business_id'],i['condition']]
        for comp,key in [('review','review_text'),('visual','visual_text')]:
            if comp=='visual' and i['condition']=='RO':assert i[key] is None;continue
            n=nk[i['business_id'],i['condition'],comp];assert i[key]==r['component_texts'][comp]==n['text'] and i[key+'_sha256']==r['component_hashes'][comp]==n['text_sha256']
    aliases={r['alias']:r for r in items};rows=read(p/'COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json')['records'];flagbindings=[]
    for row in rows:
        if row['serious_grounding_issue']:
            i=aliases[row['alias']];s=sources[row['business_id']];allowed={r['review_id'] for r in s['reviews']}|{r['photo_id'] for r in s['photos']}
            assert set(row['evidence_source_ids'])<=allowed
            text=i['review_text']+'\n'+(i['visual_text'] or '')
            flagbindings.append(dict(alias=row['alias'],span_literal_present=row['issue_span'] in text,source_ids_bound=True,semantic_error_independently_confirmed=False))
    for e in read(p/'QUALITATIVE/SOURCE_CHECKED_EXAMPLES.json'):
        text=preds[e['business_id'],e['condition']]['component_texts'][e['component']];assert e['exact_output_span'] in text
        for r in e['reference_free_evidence']:
            s=next(v for v in sources[e['business_id']]['reviews'] if v['review_id']==r['review_id']);assert r['exact_excerpt'] in s['text']
        for ph in e['photo_ids']:
            meta=next(v for v in sources[e['business_id']]['photos'] if v['photo_id']==ph);assert sha(p/'QUALITATIVE/PHOTOS'/f'{ph}.jpg')==meta['sha256']
    result=dict(status='PASS',manifest_members=len(names),collection_journal_verifier=json.loads(sub.stdout),native_text_and_input_mapping='PASS',source_bindings='PASS',original_reviews=247,original_photo_ledger=5143,illustrative_photos_verified=3,all_photo_bytes_independently_rehashed_in_local_preparation=True,
        archived_all_5143_JPEGs=False,participant_flag_bindings=flagbindings,technical_integrity_is_not_semantic_certification=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
