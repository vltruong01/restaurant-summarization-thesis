"""Offline stdlib verifier for the evaluator-only package; never creates state."""
import argparse,json,hashlib,collections
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(app):
    manifest=read(app/'IMMUTABLE_MANIFEST.json')
    for row in manifest['files']:
        p=app/row['file'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['file']
    w=app/'WORKSPACE';c=read(w/'CONFIG.json');items=read(w/'BLINDED_ITEMS.json');assert c['synthetic'] is False and c['expected_records']==90 and len(items)==90 and len(c['sources'])==15 and len({r['alias'] for r in items})==90
    assert collections.Counter(r['annotator'] for r in items)==dict(A01=45,A02=45)
    for rater in ['A01','A02']:
        rows=[r for r in items if r['annotator']==rater];assert sorted(r['position'] for r in rows)==list(range(1,46));assert {(r['business_id'],r['condition']) for r in rows}=={(b,c) for b in c['sources'] for c in ['C1','C2','C3']}
    total=0;reviews=0;photos={}
    for b,binding in c['sources'].items():
        s=read(w/binding['file']);assert s['business_id']==b and s['selected_review_ids']==[r['review_id'] for r in s['reviews']];reviews+=len(s['reviews'])
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        for p in s['photos']:
            photos[p['photo_id']]=p['sha256'];f=w/'PHOTOS'/(p['photo_id']+'.jpg');assert sha(f)==p['sha256']==p['original_source_sha256']
    assert len(photos)==5143
    for r in items:
        assert hashlib.sha256(r['review_text'].encode()).hexdigest()==r['review_text_sha256'] and hashlib.sha256(r['visual_text'].encode()).hexdigest()==r['visual_text_sha256']
        assert r['visual_text'] is not None
        assert 'system' not in r and 'model' not in r
    assert not (app/'COORDINATOR_ONLY').exists()
    return dict(status='PASS',study_id=c['study_id'],assignments=90,per_evaluator=45,restaurants=15,selected_reviews=reviews,original_photos=5143,immutable_files_checked=len(manifest['files']),visual_applicable_assignments=90,ratings_created=0,model_identity_mapping_in_evaluator_data=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--app',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(json.dumps(verify(a.app.resolve()),indent=2))
