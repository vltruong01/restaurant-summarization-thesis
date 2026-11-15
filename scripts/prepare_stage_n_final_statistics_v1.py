"""Freeze verified analysis inputs/configuration before calculating group results."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import zipfile

from s1000_final_eval_core_v1 import read, digest, file_hash

ROOT=Path('outputs/stage_n/final_human_statistical_analysis_20261102_v1')
PACKAGE=ROOT/'PACKAGE'
STUDY=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')
AUDIT=Path('outputs/stage_n/full240_rating_quality_audit_20261102_v1')
HANDOFF=Path('outputs/stage_n/completed_human_evaluation_handoff_20261102_v1')
M=Path('outputs/stage_m/final_test100_evaluation_review_20261101_v1/PACKAGE')
CRITERIA=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']


def write(path,x):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def load_db(path):
    c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:
        c.execute('BEGIN')
        assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        records=[dict(r) for r in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]
        events=[dict(r) for r in c.execute('SELECT * FROM events ORDER BY seq')]
        participants=[dict(r) for r in c.execute('SELECT * FROM participants ORDER BY id')]
        roles=[dict(r) for r in c.execute('SELECT id,role FROM users ORDER BY id')]
        return records,events,participants,roles
    finally:c.close()


def main():
    assert not ROOT.exists(),'Use a fresh analysis directory; do not overwrite frozen settings.'
    cfg=read(STUDY/'CONFIG.json');cohort=read(STUDY/'COHORT30.json');private=read(STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')
    keys={k['alias']:k for k in private};ids=sorted(cohort['business_ids'])
    assert len(ids)==len(set(ids))==30 and len(keys)==240 and cfg['expected_records']==240
    for name,h in cfg['file_bindings'].items():assert file_hash(STUDY/name)==h
    assert file_hash(STUDY/'CONFIG.json')=='93fa3e7539f4a02f98d26a8c27cfe41c261f5f2e8637d1b75af5df7db580c387'
    records,events,participants,roles=load_db(STUDY/'PRIVATE_STATE/ratings.sqlite3')
    old,old_events,_,_=load_db(AUDIT/'PRIVATE_SNAPSHOT.sqlite3')
    assert records==old and events==old_events and len(records)==240 and len(events)==241
    frozen=read(HANDOFF/'RATINGS_240.json');frozen_by={r['alias']:r for r in frozen}
    receipt=read(HANDOFF/'HANDOFF_VERIFICATION.json');archive=HANDOFF/receipt['archive']
    assert file_hash(archive)==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in read(HANDOFF/'PACKAGE_MANIFEST.json')['members']:
            b=z.read(member['file']);assert len(b)==member['bytes'] and hashlib.sha256(b).hexdigest()==member['sha256']
        assert json.loads(z.read('RATINGS_240.json'))==frozen
    states={};versions={};chain=None
    for row in events:
        e=json.loads(row['event_json']);k=keys[e['alias']]
        blank={'scores':{c:'N/A' if c==CRITERIA[-1] and k['visual_text'] is None else None for c in CRITERIA},
            'evidence_notes':'','serious_grounding_issue':False,'issue_span':'','evidence_source_ids':[],
            'issue_certainty':'none','status':'NOT_STARTED','submitted_at_utc':''}
        before=states.get(e['alias'],blank)
        assert row['event_hash']==digest(e) and e['previous_event_sha256']==chain
        assert e['version']==versions.get(e['alias'],0)+1 and e['rater']==k['annotator']
        assert e['prior_state_sha256']==digest(before) and before['status']!='SUBMITTED'
        states[e['alias']]=e['payload'];versions[e['alias']]=e['version'];chain=row['event_hash']
    sources={bid:read(STUDY/cfg['sources'][bid]['file']) for bid in ids}
    for bid,s in sources.items():
        assert s['split'].upper()=='TEST' and digest(s)==cfg['sources'][bid]['canonical_sha256']
        assert file_hash(STUDY/cfg['sources'][bid]['file'])==cfg['sources'][bid]['file_sha256']
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        for p in s['photos']:
            path=STUDY/'PHOTOS'/(p['photo_id']+'.jpg')
            assert file_hash(path)==p['sha256']==p['original_source_sha256'] and path.stat().st_size==p['bytes']
    native=read(STUDY/'COORDINATOR_ONLY/NATIVE_COMPONENT_RECEIPTS.json');native_keys={}
    raw_files={x['native_file']:x['native_file_sha256'] for x in native}
    for name,h in raw_files.items():assert Path(name).exists() and file_hash(Path(name))==h
    for entry in native:
        r=entry['native_record'];native_keys[r['business_id'],r['condition'],r['task']]=r
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
    assert len(native_keys)==210
    outputs=[]
    for bid in ids:
        for condition in ['RO','U','F','S']:
            k=next(k for k in private if k['annotator']=='A01' and k['business_id']==bid and k['condition']==condition)
            other=next(a for a in private if a['annotator']=='A02' and a['business_id']==bid and a['condition']==condition)
            assert k['review_text']==other['review_text'] and k['visual_text']==other['visual_text']
            for task,field in [('review','review_text'),('visual','visual_text')]:
                if condition=='RO' and task=='visual':assert k[field] is None;continue
                nr=native_keys[bid,condition,task]
                assert nr['text']==k[field] and nr['text_sha256']==k[field+'_sha256']
            rr=native_keys[bid,condition,'review'];vr=native_keys.get((bid,condition,'visual'))
            outputs.append({'business_id':bid,'business_name':sources[bid]['business_metadata'].get('name',bid),'condition':condition,
                'review_output':k['review_text'],'visual_output':k['visual_text'],
                'review_output_sha256':k['review_text_sha256'],'visual_output_sha256':k['visual_text_sha256'],
                'review_checkpoint_sha256':rr['checkpoint_sha256'],'visual_checkpoint_sha256':None if vr is None else vr['checkpoint_sha256'],
                'review_input_ids_sha256':rr['input_ids_sha256'],'visual_input_ids_sha256':None if vr is None else vr['input_ids_sha256'],
                'caption_ids':[] if vr is None else vr.get('caption_ids',[]),'model_visible_photo_ids':[] if vr is None else vr.get('selected_photo_ids',[]),
                'caption_count':None if vr is None else len(vr.get('caption_ids',[])),
                'no_visual_intervention':False if vr is None else vr.get('no_visual_intervention'),
                'shuffled_donor_id':None if vr is None else vr.get('shuffled_donor_id'),
                'review_token_ids_sha256':digest(rr['token_ids']),'visual_token_ids_sha256':None if vr is None else digest(vr['token_ids']),
                'recipient_source_canonical_sha256':digest(sources[bid])})
    for bid in ids:
        f=native_keys[bid,'F','review'];s=native_keys[bid,'S','review']
        assert f['text']==s['text'] and f['token_ids']==s['token_ids']
    flat=[];original=[];counts=Counter();numeric=0;na=0
    for row in records:
        p=json.loads(row['payload']);k=keys[row['alias']];hand=frozen_by[row['alias']]
        assert p==states[row['alias']] and row['version']==versions[row['alias']]
        assert p==hand['rating'] and row['version']==hand['version']
        assert row['rater']==k['annotator']==hand['rater'] and k['position']==hand['position']
        assert p['status']=='SUBMITTED' and p['submitted_at_utc']
        for criterion in CRITERIA:
            v=p['scores'][criterion]
            if k['condition']=='RO' and criterion==CRITERIA[-1]:assert v=='N/A';na+=1
            else:assert type(v) is int and 1<=v<=5;numeric+=1
        s=sources[k['business_id']];valid_ids={r['review_id'] for r in s['reviews']}|{r['photo_id'] for r in s['photos']}
        if p['serious_grounding_issue']:
            assert p['issue_span'] in k['review_text']+'\n'+(k['visual_text'] or '')
            assert p['issue_span'] and p['evidence_source_ids'] and set(p['evidence_source_ids'])<=valid_ids
        rid=f"{row['rater']}-{k['position']:03d}"
        flat.append({'record_id':rid,'alias':row['alias'],'annotator':row['rater'],'position':k['position'],
            'business_id':k['business_id'],'condition':k['condition'],**p['scores'],
            **{x:p[x] for x in p if x!='scores'},'version':row['version']})
        original.append({'record_id':rid,'business_id':k['business_id'],'condition':k['condition'],'annotator':row['rater'],
            'position':k['position'],'version':row['version'],'original_payload':p,'canonical_payload_sha256':digest(p)})
        counts[row['rater'],k['condition']]+=1
    assert numeric==1380 and na==60 and len(counts)==8 and set(counts.values())=={30}
    protected=[STUDY/'CONFIG.json',STUDY/'COHORT30.json',STUDY/'RUBRIC.json',STUDY/'TWO_RATER_PROTOCOL_AMENDMENT.md',
        STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',STUDY/'LIVE_RELEASE.json',STUDY/'DIRECT_RATER_ACCESS.json',
        archive,AUDIT/'RESEARCHER_ACCEPTANCE_AS_RECORDED.json']
    protected += list((M/'RESULTS_ORIGINAL').glob('*'))
    protected += [p for p in Path('outputs/stage_l/approved_test100_final_export_v1').rglob('*') if p.is_file()]
    protected += [Path(name) for name in raw_files]
    protected_receipts=[{'file':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in protected]
    settings={'format':'stage_n_final_statistical_analysis_config_v1','frozen_before_first_group_calculation_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'conditions':['RO','U','F','S'],'raters':['A01','A02'],'criteria':CRITERIA,'business_order':ids,
        'primary_businesses':30,'primary_records':240,'excluded_records':[],
        'within_business_aggregation':'arithmetic mean of both unchanged raters; no consensus/imputation',
        'descriptive_sd':'sample SD ddof=1; pooled scores descriptive, not60 independent businesses',
        'quantiles':'numpy.quantile method=linear; IQR=Q75-Q25',
        'agreement':'fixed1..5 quadratic-weighted Cohen kappa by condition/criterion, n30; pooled supplementary',
        'agreement_difference':'A01 minus A02 on -4..4; exact and within-one percentages; no unplanned reliability CI',
        'ordinal_alpha':'NOT_RUN: explicitly not implemented/used in frozen two-rater amendment',
        'comparisons':['F-RO','F-U','F-S'],'valid_comparisons':17,'seed':20261028,
        'rng':'numpy default_rng PCG64','bootstrap_draws':20000,'signflip_draws':50000,
        'bootstrap_unit':'business; retain both raters and all conditions/criteria together using common resample indices',
        'ci':'unadjusted pointwise percentile95%; no population rater resampling',
        'inference':'EXPLORATORY; two-sided Monte Carlo sign-flip with+1 correction; no confirmatory significance claims',
        'multiple_comparisons':'No adjustment prespecified for human comparisons; raw p-values only, no adjusted winner claim',
        'effect':'unstandardized mean paired difference in1-5 score points',
        'sensitivity':'None planned or run; all30 retained including audit concerns and empty components',
        'no_overall_composite_score':True,'F_S_identical_Review':'Report actual repeated ratings; not separate Review generation effect',
        'method_extension':'Reporting clarification only: add SD/linear IQR/exact agreement/difference frequencies. No inferential amendment.',
        'protocol_sha256':file_hash(Path('outputs/stage_n/test30_two_rater_preparation_v1/STATISTICAL_ANALYSIS_PLAN.md')),
        'amendment_sha256':file_hash(STUDY/'TWO_RATER_PROTOCOL_AMENDMENT.md'),
        'protocol_decided_without_new_aggregate_results':True}
    write(PACKAGE/'ANALYSIS_CONFIG.json',settings)
    write(PACKAGE/'DATA/VERIFIED_ANALYSIS_INPUT.json',{'format':'readonly_derived_native_analyzer_input_v1','records':flat})
    write(PACKAGE/'DATA/ORIGINAL_RATINGS_240.json',original)
    write(PACKAGE/'DATA/OUTPUTS_120.json',outputs)
    write(PACKAGE/'DATA/SOURCES_30.json',sources)
    write(ROOT/'PRIVATE_PROTECTED_INPUTS.json',protected_receipts)
    write(ROOT/'PRIVATE_SQL_ROWS.json',{'ratings':records,'events':events})
    write(PACKAGE/'PROVENANCE/INPUT_VERIFICATION.json',{'status':'PASS','records':240,'businesses':30,'numeric':numeric,'structural_NA':na,
        'submitted_per_rater_condition':{r+'_'+c:n for (r,c),n in counts.items()},'event_count':241,'journal_head_sha256':chain,
        'all_rows_match_accepted_handoff_and_audit_snapshot':True,'all_sources_and343_photo_hashes_verified':True,
        'all210_native_components_match_original_text_hashes_and_raw_files':True,'F_S_Review_text_and_token_identity_all30':True,
        'private_mapping_file_sha256':file_hash(STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json'),
        'mapping_source':'Original frozen coordinator mapping; no identity inferred from output',
        'private_key_and_database_not_in_package':True,'recorded_account_roles':roles,
        'participant_declarations_count':len(participants),'actual_identity_source_inspection_independently_authenticated':False,
        'recorded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    for path,name in [(STUDY/'COHORT30.json','COHORT30.json'),(STUDY/'RUBRIC.json','RUBRIC.json'),
        (STUDY/'TWO_RATER_PROTOCOL_AMENDMENT.md','TWO_RATER_PROTOCOL_AMENDMENT.md'),
        (Path('outputs/stage_n/test30_two_rater_preparation_v1/STATISTICAL_ANALYSIS_PLAN.md'),'ORIGINAL_STATISTICAL_ANALYSIS_PLAN.md'),
        (STUDY/'LIVE_RELEASE.json','LIVE_RELEASE.json'),(STUDY/'DIRECT_RATER_ACCESS.json','DIRECT_RATER_ACCESS.json'),
        (AUDIT/'RESEARCHER_ACCEPTANCE_AS_RECORDED.json','RESEARCHER_ACCEPTANCE_AS_RECORDED.json'),
        (AUDIT/'AUDIT_SUMMARY.json','MODEL_ASSISTED_AUDIT_SUMMARY.json')]:
        (PACKAGE/'PROVENANCE'/name).write_bytes(path.read_bytes())
    for name in ['AGGREGATE_METRICS.csv','PAIRED_COMPARISONS.json','COMPLETION.json']:
        dest=PACKAGE/'STAGE_M_ORIGINAL'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((M/'RESULTS_ORIGINAL'/name).read_bytes())
    for name in ['analyze_stage_n_two_raters_v1.py','s1000_final_eval_core_v1.py','stage_n_human_evaluation_v1.py']:
        dest=PACKAGE/'SCRIPTS_ORIGINAL'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((Path('scripts')/name).read_bytes())
    print(json.dumps({'input_verification':'PASS','config_frozen_before_analysis':True,'records':240,'businesses':30,'native_components':210,'group_metrics_not_computed_yet':True,'package':str(PACKAGE)},indent=2))


if __name__=='__main__':main()
