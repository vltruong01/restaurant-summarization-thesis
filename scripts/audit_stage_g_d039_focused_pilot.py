"""Independent offline integrity/comparison audit of the bounded D-039 pilot; no API/key access."""
import copy
import json
import statistics
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1_recovery import payload, response_text, validate_output
from stage_g_canonical_review_ids import resolve_review
from stage_g_visual_target_serialization import visual_claims

NEW = ROOT / 'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'
EXE = NEW / 'execution_v1'
PARENTS = {'original_v3':ROOT/'outputs/stage_g/teacher_pilot_v3_canonical_review_ids_v1/execution_v1',
           'D038':ROOT/'outputs/stage_g/teacher_pilot_v4_1_grounding_d038_v1/execution_v2_quote_recovery',
           'D039':EXE}


def percentile(values, q):
    if not values:return None
    a=sorted(values);p=(len(a)-1)*q;i=int(p)
    return a[i] if i==len(a)-1 else a[i]*(i+1-p)+a[i+1]*(p-i)


def stats(values):
    return {'n':len(values),'min':min(values) if values else None,
            'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None,
            'p95':percentile(values,.95),'max':max(values) if values else None}


def collect(execution, jobs):
    visual=[]; reviews=[]; failed=[]; targets=[]
    for job in jobs:
        cp=read(execution/'checkpoints'/(job['job_id']+'.json'))
        if 'output' not in cp:
            failed.append({'business_id':job['business_id'],'job_id':job['job_id'],'failure':cp})
            continue
        if job['task']=='review_consensus':
            for index,claim in enumerate(cp['resolved_output']['claims']):
                reviews.append(dict(claim,business_id=job['business_id'],split=job['split'],claim_index=index))
        else:
            for cap,photo,a in zip(job['evidence']['captions'],job['images'],cp['output']['assessments']):
                for index,claim in enumerate(a['claims']):
                    visual.append({'business_id':job['business_id'],'split':job['split'],'condition':job['condition'],
                        'photo_id':cap['photo_id'],'photo_sha256':photo['raw_sha256'],
                        'photo_marker':cap['photo_marker'],'caption_id':cap['caption_id'],'caption':cap['caption'],
                        'claim_index':index,**copy.deepcopy(claim)})
    for bid in dict.fromkeys(j['business_id'] for j in jobs):
        p=execution/'provisional_targets'/(bid+'.json')
        if p.exists():targets.append(read(p))
    return {'visual_claims':visual,'review_claims':reviews,'failed_tasks':failed,'target_sets':targets}


def consistency(rows):
    groups={}
    for row in rows:
        key=(row['business_id'],row['photo_id'],row['photo_sha256'],row['caption'],row['text'])
        groups.setdefault(key,[]).append(row)
    shared=[]
    for key,g in groups.items():
        if {r['condition'] for r in g}=={'unfiltered','filtered'}:
            shared.append({'business_id':key[0],'photo_id':key[1],'caption':key[3],'exact_claim':key[4],
                'verdicts':sorted({r['verdict'] for r in g}),'conflict':len({r['verdict'] for r in g})>1,'rows':g})
    return {'exact_matching_rule':'business/photo/hash/exact_caption/exact_claim',
            'matched_groups':len(shared),'conflict_groups':sum(g['conflict'] for g in shared),
            'shared_groups':shared,'conflicts':[g for g in shared if g['conflict']],
            'paraphrases_not_assumed_identical':True,'labels_reconciled':False}


def run():
    cfg=read(ROOT/'configs/stage_g_teacher_pilot_v5_focused_d039.json');pre=read(NEW/'preflight.json')
    completion=read(EXE/'completion.json');jobs=read(NEW/'pilot_jobs.json')
    plan={p['job_id']:p for p in read(NEW/'execution_plan.json')}
    assert digest(canonical(jobs))==pre['jobs_sha256']
    for relative,expected in {**pre['input_sha256'],**pre['preserved_original_sha256']}.items():
        assert file_sha(ROOT/relative)==expected,'Protected source/output changed: '+relative
    schemas=read(ROOT/cfg['schema']);artifacts={};slots=[];spans=0;request_checks=0
    native=[];imported=[];outputs={}
    for job in jobs:
        p=EXE/'checkpoints'/(job['job_id']+'.json');cp=read(p)
        artifacts[p.relative_to(ROOT).as_posix()]=file_sha(p)
        prompt=(ROOT/plan[job['job_id']]['prompt_path']).read_text(encoding='utf-8');schema=schemas[job['task']]
        assert cp['binding']['job_sha256']==digest(canonical(job))
        assert cp['binding']['config_sha256']==file_sha(ROOT/'configs/stage_g_teacher_pilot_v5_focused_d039.json')
        assert cp['binding']['prompt_sha256']==digest(prompt.encode('utf-8'))
        assert cp['binding']['schema_sha256']==digest(canonical(schema))
        raw_path=EXE/'raw_responses'/(job['job_id']+'.json');body=payload(job,cfg,prompt,schema)
        if raw_path.exists():
            raw=read(raw_path);artifacts[raw_path.relative_to(ROOT).as_posix()]=file_sha(raw_path)
            assert raw.get('model')==cfg['requested_model'] and raw.get('status')=='completed'
            if cp.get('imported_completed_response'):
                assert cp['import_lineage']['identical_request_sha256']==digest(canonical(body))
                source=ROOT/cp['import_lineage']['checkpoint']
                assert file_sha(source)==cp['import_lineage']['source_checkpoint_sha256']
                assert read(source)['output']==cp['output']
                imported.append(raw)
            else:
                intent=EXE/'request_intents'/(job['job_id']+'.json')
                assert read(intent)['request_sha256']==digest(canonical(body))
                native.append(raw)
            request_checks+=1
        if 'output' not in cp:
            reproduced=False
            if raw_path.exists():
                try:validate_output(job,json.loads(response_text(read(raw_path))),schema)
                except ValueError:reproduced=True
            slots.append({'job_id':job['job_id'],'status':'FAILED_PRESERVED','failure_reproduced':reproduced,
                          'diagnostic':cp.get('local_validation_error',cp.get('error_type'))})
            continue
        assert cp['validation']==validate_output(job,cp['output'],schema)
        if cp.get('api_called'):
            assert file_sha(raw_path)==cp['raw_response_sha256']
            assert json.loads(response_text(read(raw_path)))==cp['output']
        if job['task']=='review_consensus':
            resolved,provenance=resolve_review(job,cp['output'])
            assert resolved==cp['resolved_output'] and provenance==cp['evidence_resolution']
            spans+=len(provenance)
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
        slots.append({'job_id':job['job_id'],'status':'STRUCTURAL_AND_SOURCE_BINDING_PASS',
                      'mode':plan[job['job_id']]['mode'],'factual_human_verified':False})
    manifest=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json');folder=ROOT/manifest['relative_directory']
    for name,expected in manifest['sha256'].items():assert file_sha(folder/name)==expected
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'));tokenizer.no_truncation();tokenizer.no_padding()
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(str(folder),local_files_only=True)
    lengths={c:[] for c in ['reviews_only','unfiltered','filtered']};checks=0;empty=[];valid=[]
    for business in read(NEW/'pilot_businesses.json'):
        bid=business['business_id'];p=EXE/'provisional_targets'/(bid+'.json');o=outputs.get(bid,{})
        if not p.exists():
            assert set(o)!={'shared','unfiltered','filtered'};continue
        assert set(o)=={'shared','unfiltered','filtered'}
        record=read(p);assert record['human_verified'] is False and record['evidence_records']==o
        consensus='[REVIEW_CONSENSUS]\n'+' '.join(c['text'] for c in o['shared']['claims'])
        assert record['targets']['reviews_only']==consensus
        for c in ['unfiltered','filtered']:
            groups=visual_claims(o[c]['assessments']);assert groups==record['visual_target_claims_with_all_evidence'][c]
            expected=consensus+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(x['text'] for x in groups)
            assert record['targets'][c]==expected
            if not o[c]['assessments']:
                assert expected.endswith('[VISUAL_EVIDENCE]\n');empty.append({'business_id':bid,'condition':c})
        for c,text in record['targets'].items():
            ids=tokenizer.encode(text,add_special_tokens=True).ids
            assert slow.encode(text,add_special_tokens=True,truncation=False)==ids
            lengths[c].append(len(ids));checks+=1
        valid.append(bid);artifacts[p.relative_to(ROOT).as_posix()]=file_sha(p)
    assert len(valid)==completion['businesses_completed'] and len(slots)==27 and len(native)<=19
    assert len(imported)==7 and len({r['id'] for r in native+imported})==len(native+imported)
    report={'status':'INDEPENDENT_STRUCTURAL_INTEGRITY_PASS_WITH_REPORTED_FAILURES' if completion['failed_tasks'] else 'INDEPENDENT_STRUCTURAL_INTEGRITY_PASS',
        'businesses_attempted':9,'businesses_complete':len(valid),'target_texts':checks,'task_slots':slots,
        'fresh_completed_API_responses':len(native),'imported_review_responses':len(imported),
        'request_SHA_checks':request_checks,'exact_source_span_checks':spans,'slow_native_LED_target_ID_checks':checks,
        'target_tokens':{k:stats(v) for k,v in lengths.items()},'over512_diagnostic_only':{k:sum(x>512 for x in v) for k,v in lengths.items()},
        'zero_caption_cases':empty,'original_files_preserved':len(pre['preserved_original_sha256']),
        'source_reviews_captions_order_images_unchanged':True,'schema_model_generation_settings_unchanged':True,
        'native_usage':{'input_tokens':sum(r.get('usage',{}).get('input_tokens',0) for r in native),
            'output_tokens':sum(r.get('usage',{}).get('output_tokens',0) for r in native),
            'reasoning_tokens':sum(r.get('usage',{}).get('output_tokens_details',{}).get('reasoning_tokens',0) for r in native)},
        'human_verified_businesses':0,'factual_grounding_certified_by_structural_audit':False,
        'no_manual_target_edit':True,'no_stage_F_change':True,'no_test_teacher':True,'production_frozen':False,
        'artifact_sha256':artifacts}
    immutable(NEW/'independent_integrity_report.json',report)
    versions={name:collect(execution,jobs) for name,execution in PARENTS.items()}
    comparison={'status':'BOUNDED_THREE_VERSION_MACHINE_COMPARISON_NOT_HUMAN_GROUND_TRUTH',
        'cohort_business_ids':[b['business_id'] for b in read(NEW/'pilot_businesses.json')],
        'versions':versions,'exact_cross_condition':{k:consistency(v['visual_claims']) for k,v in versions.items()},
        'counts':{k:{'target_sets':len(v['target_sets']),'review_claims':len(v['review_claims']),
            'visual_claims':len(v['visual_claims']),'verdicts':dict(Counter(r['verdict'] for r in v['visual_claims']))} for k,v in versions.items()},
        'warning':'Changing claim granularity and targeted selection preclude error-rate/causal-prompt claims; semantic paraphrases require separate AI/user review.'}
    immutable(NEW/'machine_comparison.json',comparison)
    print(json.dumps({k:v for k,v in report.items() if k not in ['task_slots','artifact_sha256']},indent=2))


if __name__=='__main__':run()
