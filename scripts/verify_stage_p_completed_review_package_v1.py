"""Offline, read-only verification of the completed Stage P review handoff."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(root):
    root=root.resolve()
    manifest=read(root/'PACKAGE_MANIFEST.json')
    expected={r['file'] for r in manifest['files']}|{'PACKAGE_MANIFEST.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==expected, 'Missing or unexpected package files'
    for row in manifest['files']:
        path=root/row['file']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['file']
    sys.path.insert(0,str(root/'STATISTICS'))
    from validate_stage_p_human_exports_v1 import validate, CRITERIA
    from stage_p_revision_provenance_v1 import valid_import_event
    package=root/'FROZEN_STUDY'
    records,validation=validate(package,[root/'RATINGS/A01_FINAL_45.json',root/'RATINGS/A02_FINAL_45.json'])
    combined=read(root/'RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')
    assert combined['final'] and combined['audit']['status']=='PASS' and combined['study_id']==validation['study_id']
    assert combined['config_sha256']==sha(package/'EVALUATOR_APP/WORKSPACE/CONFIG.json')
    exported={r['alias']:r for r in combined['records']}
    assert len(exported)==len(combined['records'])==90
    for row in records:
        assert {k:v for k,v in row.items() if k!='system'}==exported[row['alias']]
    events=read(root/'RATINGS/RATING_EVENT_JOURNAL.json')
    previous=None;import_cache={};verified_imports=0
    states={}
    versions={}
    items={r['alias']:r for r in read(package/'EVALUATOR_APP/WORKSPACE/BLINDED_ITEMS.json')}
    for event in events:
        alias=event['alias']
        assert event['previous_event_sha256']==previous
        assert event['version']==versions.get(alias,0)+1 and event['rater']==items[alias]['annotator']
        if alias in states:
            assert event['prior_state_sha256']==digest(states[alias])
            if states[alias]['status']=='SUBMITTED':
                assert valid_import_event(event,states[alias],root/'EVIDENCE/REVISION_IMPORTS',import_cache),'Submitted-state change is not a valid source-bound import'
                verified_imports+=1
        previous=digest(event)
        versions[alias]=event['version']
        states[alias]=event['payload']
    assert previous==combined['audit']['event_chain_head_sha256'] and set(states)==set(exported)
    for alias,payload in states.items():
        row=exported[alias]
        assert row['version']==versions[alias] and row['status']=='SUBMITTED'
        assert all(row[c]==payload['scores'][c] for c in CRITERIA)
        assert all(row[k]==payload[k] for k in payload if k!='scores')
    w=package/'EVALUATOR_APP/WORKSPACE'
    cfg=read(w/'CONFIG.json')
    for name,expected_sha in cfg['file_bindings'].items(): assert sha(w/name)==expected_sha
    sources={}
    for business,binding in cfg['sources'].items():
        source=read(w/binding['file'])
        assert digest(source)==binding['canonical_sha256'] and source['business_id']==business
        assert source['selected_review_ids']==[r['review_id'] for r in source['reviews']]
        assert hashlib.sha256(source['selected_review_block'].encode()).hexdigest()==source['review_block_sha256']
        sources[business]=source
    key=read(package/'COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json')
    outputs=read(package/'COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json')
    outputs_by_pair={(r['business_id'],r['system']):r for r in outputs}
    assert len(outputs_by_pair)==len(outputs)==45
    for mapping in key:
        item=items[mapping['alias']]
        output=outputs_by_pair[mapping['business_id'],mapping['system']]
        for component in ['review','visual']:
            assert item[component+'_text']==output[component+'_text']
            assert hashlib.sha256(item[component+'_text'].encode()).hexdigest()==item[component+'_text_sha256']==mapping[component+'_text_sha256']==output[component+'_text_sha256']
    for rater in ['A01','A02']:
        own=[r for r in key if r['annotator']==rater]
        assert len(own)==45
        for system in {r['system'] for r in own}:
            assert Counter(r['within_restaurant_position'] for r in own if r['system']==system)=={1:5,2:5,3:5}
    assert all(outputs_by_pair[b,'Qwen-Caption']['review_text']==outputs_by_pair[b,'Qwen-Image']['review_text'] for b in sources)
    report=read(root/'EVIDENCE/COMPLETION_VERIFICATION.json')
    assert report['status']=='PASS' and report['records']==90 and report['numeric_scores']==540
    assert report['live_database_logical_sha256_before']==report['live_database_logical_sha256_after']
    copied=read(root/'EVIDENCE/COPY_PROVENANCE.json')['copied_files']
    exported_binding=next(r for r in copied if r['package_file']=='RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')
    assert sha(root/'RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')==exported_binding['sha256']
    assert not any(p.suffix.lower() in {'.sqlite3','.db','.pem','.key','.jpg','.jpeg'} for p in root.rglob('*') if p.is_file())
    return dict(status='PASS',study_id=validation['study_id'],restaurants=15,records=90,numeric_scores=540,
        per_rater=dict(Counter(r['annotator'] for r in records)),journal_events=len(events),selected_source_reviews=sum(len(s['reviews']) for s in sources.values()),
        photo_metadata_records=sum(len(s['photos']) for s in sources.values()),frozen_outputs=45,shared_Qwen_Review_pairs=15,
        grounding_flags=dict(Counter(r['annotator'] for r in records if r['serious_grounding_issue'])),
        manifest_members_checked=len(manifest['files']),score_changes_by_packaging=0,verified_administrative_import_events=verified_imports,statistical_analysis_performed=False,
        photo_pixels_included=False,photo_semantic_verification_performed=False,
        scope='Package bytes, record completeness, source/output/mapping bindings and journal verified; no independent participant authentication or semantic confirmation of flags.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    print(json.dumps(verify(args.package),indent=2))
