"""Validate only the NEW session exports; reject old-session, incomplete or mismatched files."""
from pathlib import Path
import argparse,json,hashlib,datetime
CRITERIA=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(package,files,allow_fixture=False):
    app=package/'EVALUATOR_APP';w=app/'WORKSPACE';cfg=read(w/'CONFIG.json');key=read(package/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json');items={x['alias']:x for x in read(w/'BLINDED_ITEMS.json')};mapping={x['alias']:x for x in key};records=[];raters=[]
    assert cfg['synthetic'] is False or (allow_fixture and all(b.startswith('SYNTHETIC_') for b in cfg['sources'])),'Non-production fixture flag is not permitted for real analysis'
    if not cfg['synthetic']:
        binding=read(package/'COORDINATOR_ONLY/BINDINGS.json');assert binding['study_id']==cfg['study_id'] and binding['config_sha256']==sha(w/'CONFIG.json')
        assert binding['assignment_key_sha256']==sha(package/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json') and binding['blinded_items_sha256']==sha(w/'BLINDED_ITEMS.json')
        assert binding['analysis_plan_sha256']==sha(package/'COORDINATOR_ONLY/ANALYSIS_PLAN.json')
    for f in files:
        d=read(f);r=d.get('rater');assert r in cfg['raters'] and r not in raters,'Exactly one export per rater';raters.append(r)
        assert d['study_id']==cfg['study_id'] and d['study_format']==cfg['format'],'Wrong study/session'
        assert d['config_sha256']==sha(w/'CONFIG.json') and d['blinded_items_sha256']==sha(w/'BLINDED_ITEMS.json')
        assert d['amendment_sha256']==sha(w/'TWO_RATER_PROTOCOL_AMENDMENT.md') and d['synthetic']==cfg['synthetic']
        rows=d['records'];expected={x['alias'] for x in key if x['annotator']==r};assert d['final'] is True and len(rows)==len(expected) and {x['alias'] for x in rows}==expected,'Incomplete or duplicate export'
        assert d['records_sha256']==digest(rows),'Recordset hash changed'
        for row in rows:
            m=mapping[row['alias']];i=items[row['alias']]
            assert row['annotator']==r==m['annotator'] and row['business_id']==m['business_id'] and row['position']==m['position'] and row['condition']==m['neutral_slot'],'Assignment binding changed'
            assert row['status']=='SUBMITTED' and row['version']>=1 and row['submitted_at_utc'] and datetime.datetime.fromisoformat(row['submitted_at_utc'])
            assert all(type(row[c]) is int and 1<=row[c]<=5 for c in CRITERIA),'All six criteria applicable; integer1-5 required'
            source=read(w/cfg['sources'][row['business_id']]['file']);allowed=set(source['selected_review_ids'])|{p['photo_id'] for p in source['photos']};assert set(row['evidence_source_ids'])<=allowed
            if row['serious_grounding_issue']:
                assert row['issue_span'] in i['review_text']+'\n'+i['visual_text'] and row['issue_span'] and row['evidence_source_ids'] and row['evidence_notes'].strip() and row['issue_certainty'] in ['uncertain','confirmed']
            records.append({**row,'system':m['system']})
    assert set(raters)==set(cfg['raters']) and len(records)==cfg['expected_records'],'Both complete evaluator exports required'
    assert len({x['alias'] for x in records})==len(records)
    return records,dict(status='PASS',study_id=cfg['study_id'],records=len(records),raters=raters,restaurants=len(cfg['sources']),old_Stage_O_records_reused=0,scope='Structural/content-hash matching; not independent authentication of participant identity or source inspection.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--exports',nargs=2,type=Path,required=True);a=p.parse_args();_,r=validate(a.package,a.exports);print(json.dumps(r,indent=2))
