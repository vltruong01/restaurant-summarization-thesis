"""Read-only source/schema/target and cross-condition D039/D040 audit; no API/key."""
import json
import copy
from pathlib import Path
from collections import Counter
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1_recovery import payload, response_text, validate_output
from stage_g_canonical_review_ids import resolve_review
from stage_g_visual_target_serialization import visual_claims
from audit_stage_g_d039_focused_pilot import consistency, stats
from tokenizers import Tokenizer

NEW=ROOT/'outputs/stage_g/teacher_pilot_v6_focused_d040_v1'
EXE=NEW/'execution_v1'
OLD=ROOT/'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'


def collect(execution,jobs):
    visual=[];reviews=[];failed=[];targets=[]
    for job in jobs:
        path=execution/'checkpoints'/(job['job_id']+'.json')
        if not path.exists():
            failed.append({'job_id':job['job_id'],'business_id':job['business_id'],'status':'NOT_ATTEMPTED'});continue
        cp=read(path)
        if 'output' not in cp:
            failed.append({'business_id':job['business_id'],'job_id':job['job_id'],'failure':cp});continue
        if job['task']=='review_consensus':
            for index,claim in enumerate(cp['resolved_output']['claims']):
                reviews.append(dict(claim,business_id=job['business_id'],split=job['split'],claim_index=index))
        else:
            for cap,photo,a in zip(job['evidence']['captions'],job['images'],cp['output']['assessments']):
                for index,claim in enumerate(a['claims']):
                    visual.append({'business_id':job['business_id'],'split':job['split'],'condition':job['condition'],
                        'photo_id':cap['photo_id'],'photo_sha256':photo['raw_sha256'],'photo_marker':cap['photo_marker'],
                        'caption_id':cap['caption_id'],'caption':cap['caption'],'claim_index':index,**copy.deepcopy(claim)})
    for bid in dict.fromkeys(j['business_id'] for j in jobs):
        p=execution/'provisional_targets'/(bid+'.json')
        if p.exists():targets.append(read(p))
    return {'visual_claims':visual,'review_claims':reviews,'failed_tasks':failed,'target_sets':targets}


def run():
    cfgpath=ROOT/'configs/stage_g_teacher_pilot_v6_focused_d040.json'
    cfg=read(cfgpath);pre=read(NEW/'preflight.json');completion=read(EXE/'completion.json')
    jobs=read(NEW/'pilot_jobs.json');plan={p['job_id']:p for p in read(NEW/'execution_plan.json')}
    assert digest(canonical(jobs))==pre['jobs_sha256']
    for rel,h in {**pre['input_sha256'],**pre['preserved_original_sha256']}.items():
        assert file_sha(ROOT/rel)==h,'Protected source changed: '+rel
    parent_cfg=read(ROOT/cfg['parent_pilot_config'])
    for key in ['requested_model','snapshot','generation_proposal','schema','source_pointer','tasks',
                'evidence_representation','target_serialization','citation_metadata_policy']:
        assert cfg[key]==parent_cfg[key]
    parent_jobs={j['job_id']:j for j in read(OLD/'pilot_jobs.json')}
    schemas=read(ROOT/cfg['schema']);outputs={};slots=[];spans=0;requests=0;native=[];imports=[];artifacts={}
    for job in jobs:
        prior=parent_jobs[job['job_id']]
        assert {k:v for k,v in job.items() if k!='prompt_sha256'}=={k:v for k,v in prior.items() if k!='prompt_sha256'}
        path=EXE/'checkpoints'/(job['job_id']+'.json')
        if not path.exists():
            slots.append({'job_id':job['job_id'],'status':'NOT_ATTEMPTED'});continue
        cp=read(path);artifacts[path.relative_to(ROOT).as_posix()]=file_sha(path)
        prompt=(ROOT/plan[job['job_id']]['prompt_path']).read_text(encoding='utf-8');schema=schemas[job['task']]
        for k,v in {'job_sha256':digest(canonical(job)),'config_sha256':file_sha(cfgpath),
            'prompt_sha256':digest(prompt.encode('utf-8')),'schema_sha256':digest(canonical(schema))}.items():assert cp['binding'][k]==v
        rawpath=EXE/'raw_responses'/path.name;body=payload(job,cfg,prompt,schema)
        if rawpath.exists():
            raw=read(rawpath);artifacts[rawpath.relative_to(ROOT).as_posix()]=file_sha(rawpath)
            assert raw['model']==cfg['requested_model'] and raw['status']=='completed'
            if cp.get('imported_completed_response'):
                lineage=cp['import_lineage'];assert lineage['identical_request_sha256']==digest(canonical(body))
                source=ROOT/lineage['checkpoint'];assert file_sha(source)==lineage['source_checkpoint_sha256']
                original=read(source);assert original['output']==cp['output']
                assert original['resolved_output']==cp['resolved_output']
                imports.append(raw)
            else:
                assert read(EXE/'request_intents'/path.name)['request_sha256']==digest(canonical(body))
                native.append(raw)
            requests+=1
        if 'output' not in cp:
            reproduced=False
            if rawpath.exists():
                try:validate_output(job,json.loads(response_text(read(rawpath))),schema)
                except ValueError:reproduced=True
            slots.append({'job_id':job['job_id'],'status':'FAILED_PRESERVED','schema_or_quote_failure_reproduced':reproduced,
                'diagnostic':cp.get('local_validation_error',cp.get('error_type')),'http_status':cp.get('http_status')});continue
        assert cp['validation']==validate_output(job,cp['output'],schema)
        if cp.get('api_called'):
            assert file_sha(rawpath)==cp['raw_response_sha256']
            assert json.loads(response_text(read(rawpath)))==cp['output']
        if job['task']=='review_consensus':
            resolved,provenance=resolve_review(job,cp['output']);assert resolved==cp['resolved_output']
            assert provenance==cp['evidence_resolution'];spans+=len(provenance)
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
        slots.append({'job_id':job['job_id'],'status':'STRUCTURAL_AND_SOURCE_PASS',
            'mode':plan[job['job_id']]['mode'],'factual_human_verified':False})
    m=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json');directory=ROOT/m['relative_directory']
    for n,h in m['sha256'].items():assert file_sha(directory/n)==h
    native_tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'));native_tokenizer.no_truncation();native_tokenizer.no_padding()
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(str(directory),local_files_only=True)
    lengths={k:[] for k in ['reviews_only','unfiltered','filtered']};checks=0;valid=[];empty=[]
    for b in read(NEW/'pilot_businesses.json'):
        bid=b['business_id'];path=EXE/'provisional_targets'/(bid+'.json');o=outputs.get(bid,{})
        if not path.exists():assert set(o)!={'shared','unfiltered','filtered'};continue
        record=read(path);assert set(o)=={'shared','unfiltered','filtered'} and record['human_verified'] is False
        assert record['evidence_records']==o
        review='[REVIEW_CONSENSUS]\n'+' '.join(c['text'] for c in o['shared']['claims'])
        assert record['targets']['reviews_only']==review
        for condition in ['unfiltered','filtered']:
            groups=visual_claims(o[condition]['assessments']);assert groups==record['visual_target_claims_with_all_evidence'][condition]
            expected=review+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(c['text'] for c in groups)
            assert expected==record['targets'][condition]
            if not o[condition]['assessments']:
                assert expected.endswith('[VISUAL_EVIDENCE]\n');empty.append({'business_id':bid,'condition':condition})
        for condition,text in record['targets'].items():
            ids=native_tokenizer.encode(text,add_special_tokens=True).ids
            assert ids==slow.encode(text,add_special_tokens=True,truncation=False)
            lengths[condition].append(len(ids));checks+=1
        valid.append(bid);artifacts[path.relative_to(ROOT).as_posix()]=file_sha(path)
    assert len(valid)==completion['businesses_completed'] and len(native)<=16
    usage={'input_tokens':sum(r.get('usage',{}).get('input_tokens',0) for r in native),
        'output_tokens':sum(r.get('usage',{}).get('output_tokens',0) for r in native),
        'reasoning_tokens':sum(r.get('usage',{}).get('output_tokens_details',{}).get('reasoning_tokens',0) for r in native)}
    report={'status':'AUTOMATED_INTEGRITY_PASS_WITH_PRESERVED_FAILURES' if completion['failed_tasks'] else 'AUTOMATED_INTEGRITY_PASS',
        'businesses_planned':8,'businesses_complete':len(valid),'target_texts':checks,'task_slots':slots,
        'fresh_completed_API_responses':len(native),'imported_review_responses':len(imports),
        'request_SHA_checks':requests,'exact_source_span_checks':spans,'slow_native_LED_target_ID_checks':checks,
        'target_tokens':{k:stats(v) for k,v in lengths.items()},
        'over512_diagnostic_only':{k:sum(t>512 for t in v) for k,v in lengths.items()},
        'zero_caption_cases':empty,'preserved_existing_files':len(pre['preserved_original_sha256']),
        'reviews_captions_photos_order_unchanged':True,'schema_model_generation_settings_unchanged':True,
        'native_usage':usage,'factual_grounding_certified_by_automated_validation':False,
        'human_verified_businesses':0,'no_manual_target_edit':True,'production_frozen':False,
        'no_Stage_F_change_or_test_teacher_or_LED_training':True,'artifact_sha256':artifacts}
    immutable(NEW/'independent_integrity_report.json',report)
    versions={'D039':collect(OLD/'execution_v1',jobs),'D040':collect(EXE,jobs)}
    machine={'status':'D039_D040_BOUNDED_MACHINE_COMPARISON_NOT_HUMAN_GROUND_TRUTH',
        'cohort_business_ids':[b['business_id'] for b in read(NEW/'pilot_businesses.json')],
        'versions':versions,'exact_cross_condition':{k:consistency(v['visual_claims']) for k,v in versions.items()},
        'counts':{k:{'target_sets':len(v['target_sets']),'review_claims':len(v['review_claims']),
            'visual_claims':len(v['visual_claims']),'verdicts':dict(Counter(r['verdict'] for r in v['visual_claims']))} for k,v in versions.items()},
        'warning':'Targeted cohort and changing claim decomposition are not population error rates or causal proof; no verdict reconciliation.'}
    immutable(NEW/'machine_comparison.json',machine)
    print(json.dumps({k:v for k,v in report.items() if k not in ['task_slots','artifact_sha256']},indent=2))


if __name__=='__main__':run()
