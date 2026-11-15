"""Read-only package verification; never opens or creates a rating database."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from s1000_final_eval_core_v1 import read, file_hash, digest

def verify(root):
    manifest=read(root/'PACKAGE_MANIFEST.json')
    for r in manifest['files']:
        p=(root/r['path']).resolve();assert p.is_relative_to(root.resolve())
        assert p.is_file() and p.stat().st_size==r['bytes'] and file_hash(p)==r['sha256'],r['path']
    ws=root/'WORKSPACE';c=read(ws/'CONFIG.json');assert not c['synthetic'] and c['expected_records']==240
    for name,h in c['file_bindings'].items():assert file_hash(ws/name)==h,name
    cohort=read(ws/'COHORT30.json')['business_ids'];assert len(cohort)==len(set(cohort))==30 and set(cohort)==set(c['sources'])
    hist=ws/'COORDINATOR_ONLY/HISTORICAL'
    test=read(hist/'COHORT100_FREEZE.json')['business_ids']
    assert cohort==sorted(sorted(test,key=lambda b:(digest([20261028,'HUMAN_PANEL',b]),b))[:30])
    source_manifest={r['business_id']:r for r in read(ws/'COORDINATOR_ONLY/SOURCE_PACKAGE_MANIFEST.json')['frozen_sources']}
    for b,v in c['sources'].items():
        s=read(ws/v['file']);assert digest(s)==v['canonical_sha256'] and file_hash(ws/v['file'])==v['file_sha256']
        assert s['business_id']==b and s['split']=='test' and s['selected_review_ids']==[r['review_id'] for r in s['reviews']]
        assert digest(s)==source_manifest[b]['sha256']
        for photo in s['photos']:
            p=ws/'PHOTOS'/(photo['photo_id']+'.jpg');assert p.stat().st_size==photo['bytes'] and file_hash(p)==photo['sha256']==photo['original_source_sha256']
    key=read(ws/'COORDINATOR_ONLY/PRIVATE_ITEMS.json');assert len(key)==len({r['alias'] for r in key})==240
    legacy=list(csv.DictReader((hist/'PRIVATE_HUMAN_DESIGN_KEY.csv').open(encoding='utf8')))
    assert [{f:str(r[f]) for f in ['alias','annotator','position','business_id','condition']} for r in key]==[r for r in legacy if r['annotator'] in ['A01','A02']]
    for rater in ['A01','A02']:
        rows=sorted((r for r in key if r['annotator']==rater),key=lambda r:r['position'])
        order=sorted([(b,c) for b in cohort for c in ['RO','U','F','S']],key=lambda v:digest([42,'presentation',rater,*v]))
        assert [(r['business_id'],r['condition']) for r in rows]==order
        assert all(r['alias']==digest([42,'blind',rater,i,r['business_id'],r['condition']])[:20] for i,r in enumerate(rows,1))
    native={(r['native_record']['condition'],r['native_record']['business_id'],r['native_record']['task']):r['native_record'] for r in read(ws/'COORDINATOR_ONLY/NATIVE_COMPONENT_RECEIPTS.json')};assert len(native)==210
    for r in key:
        for component in ['review','visual']:
            text=r[component+'_text']
            if text is None:assert component=='visual' and r['condition']=='RO';continue
            n=native[r['condition'],r['business_id'],component]
            assert text==n['text'] and hashlib.sha256(text.encode()).hexdigest()==r[component+'_text_sha256']==n['text_sha256']
    blank=list(csv.DictReader((ws/'BLANK_RATING_EXPORT_240.csv').open(encoding='utf8')));assert len(blank)==240
    criteria=read(ws/'RUBRIC.json')['criteria'];assert len(criteria)==6
    assert sum(r[c['id']]=='N/A' for r in blank for c in criteria)==60
    assert sum(r[c['id']]=='' for r in blank for c in criteria)==1380
    assert all(r['status']=='NOT_STARTED' and not r['submitted_at_utc'] for r in blank)
    result={'status':'PASS','manifest_files_hashed':len(manifest['files']),'businesses':30,'assignments':240,'native_components':210,'blank_numeric_cells':1380,'structural_na':60,'no_database_created_by_verifier':True}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
