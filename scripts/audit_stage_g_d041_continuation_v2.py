# Original audit SHA256: 57d1a4d4219a553bdd1bde9d6a93a8d12893ad3bc3ca2d7f930217875c092ade
"""Offline D041 input/result checks and technical flags; never asserts factual accuracy."""
import argparse
import gzip
import json
import difflib
from collections import Counter, defaultdict
from tokenizers import Tokenizer
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from prepare_stage_g_extended_pilot_d041 import CONFIG, choose, rank
from run_stage_g_d041_continuation_v2 import FOLDER
from run_stage_g_d041_continuation_v2 import check_inputs, build_target
from run_stage_g_teacher_pilot_v4_1_recovery import payload, validate_output, response_text
from stage_g_canonical_review_ids import resolve_review, citation_diagnostics
from audit_stage_g_d039_focused_pilot import stats, consistency


def tokenizer_pair():
    manifest=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    directory=ROOT/manifest['relative_directory']
    for name,h in manifest['sha256'].items():assert file_sha(directory/name)==h
    native=Tokenizer.from_file(str(directory/'tokenizer.json'));native.no_truncation();native.no_padding()
    from transformers import LEDTokenizer
    slow=LEDTokenizer.from_pretrained(str(directory),local_files_only=True)
    return native,slow


def preflight():
    cfg,pre,jobs=check_inputs();businesses=read(FOLDER/'pilot_businesses.json')
    sampling=read(FOLDER/'sampling_manifest.json');manual=read(FOLDER/'manual_review_25_manifest.json')
    chosen={b['business_id'] for b in businesses};excluded=set(sampling['excluded_prior_business_ids'])
    assert len(chosen)==100 and not chosen&excluded and all(b['split'] in ['train','validation'] for b in businesses)
    assert Counter(b['split'] for b in businesses)=={'train':80,'validation':20}
    regenerated,strata=choose(read(FOLDER/'sampling_candidate_features.json'))
    assert [(b['business_id'],b['split']) for b in regenerated]==[(b['business_id'],b['split']) for b in businesses]
    assert strata==sampling['strata']
    expected=sorted(businesses,key=lambda b:rank('manual-random',b['business_id']))[:25]
    assert [b['business_id'] for b in manual['businesses']]==[b['business_id'] for b in expected]
    assert len(jobs)==300 and len({j['job_id'] for j in jobs})==300
    assert sum(not j.get('skip_api_empty_captions') for j in jobs)==pre['max_new_API_calls']
    # Reconstruct all 300 source token IDs with native and slow implementations.
    native,slow=tokenizer_pair();lengths=defaultdict(list);checks=0
    for source in read(FOLDER/'frozen_source_witness.json'):
        assert source['business_id'] in chosen and source['split'] in ['train','validation']
        ids=native.encode(source['source'],add_special_tokens=True).ids
        assert ids==slow.encode(source['source'],add_special_tokens=True,truncation=False)
        assert len(ids)==source['tokens']<=5000
        lengths[source['condition']].append(len(ids));checks+=1
    assert checks==300
    zero=[b['business_id'] for b in businesses if b['zero_filtered_edge_case']]
    for b in businesses:
        these=[j for j in jobs if j['business_id']==b['business_id']]
        assert len(these)==3 and {j['condition'] for j in these}=={'shared','unfiltered','filtered'}
        assert all(j['review_block_sha256']==b['review_block_sha256'] and j['split']==b['split'] for j in these)
        review=next(j for j in these if j['condition']=='shared')
        assert [r['review_id'] for r in review['evidence']['reviews']]==b['selected_review_ids']
        for cond in ['unfiltered','filtered']:
            job=next(j for j in these if j['condition']==cond)
            assert len(job['evidence']['captions'])==len(job['images'])<=5
            assert not job.get('skip_api_empty_captions') or not job['evidence']['captions']
            assert [c['photo_id'] for c in job['evidence']['captions']]==[p['photo_id'] for p in job['images']]
    # Spending guard must reject missing authorization before reading a key/calling API.
    from run_stage_g_extended_pilot_d041 import approved_authorization
    blocked=False
    try:approved_authorization(FOLDER/'_missing_authorization_guard_check.json',pre)
    except ValueError:blocked=True
    assert blocked
    report={'status':'D041_OFFLINE_VALIDATION_PASS_NO_API_CALLS_AWAITING_AUTHORIZATION',
        'businesses':100,'businesses_by_split':dict(Counter(b['split'] for b in businesses)),
        'excluded_prior_businesses':len(excluded),'prior_overlap':0,'held_out_payloads':0,
        'preselected_random_human_review_businesses':25,'manual25_selection_reproduced':True,'cohort100_selection_reproduced':True,
        'task_slots':300,'maximum_API_calls':pre['max_new_API_calls'],'zero_filtered_business_ids':zero,
        'slow_native_source_token_ID_checks':checks,'source_tokens':{k:stats(v) for k,v in lengths.items()},
        'missing_authorization_blocks_before_credentials_or_API':True,
        'all_D040_critical_fields_hashes_and_historical_inputs_preserved':True,
        'API_calls':0,'new_teacher_targets':0,'human_annotations_created':0,'factual_grounding_certified':False}
    immutable(FOLDER/'offline_validation_report.json',report)
    print(json.dumps(report,indent=2));return report


def results():
    cfg,pre,jobs=check_inputs();out=FOLDER/'execution_v1'
    if not (out/'completion.json').exists():raise ValueError('No completed API execution to audit; no outputs simulated')
    native,slow=tokenizer_pair();schemas=read(ROOT/cfg['schema']);businesses=read(FOLDER/'pilot_businesses.json')
    slots=[];raws=[];visual=[];reviews=[];flags=[];outputs={};target_lengths=defaultdict(list);artifact_hashes={}
    span_checks=0;request_checks=0;target_checks=0;exact_component_sentences=0
    for job in jobs:
        path=out/'checkpoints'/(job['job_id']+'.json');rawpath=out/'raw_responses'/path.name
        request=out/'raw_requests'/(job['job_id']+'.json.gz')
        if not path.exists():
            slots.append({'job_id':job['job_id'],'business_id':job['business_id'],'status':'NOT_ATTEMPTED_OR_UNRESOLVED'})
            continue
        cp=read(path);artifact_hashes[path.relative_to(ROOT).as_posix()]=file_sha(path)
        prompt=(ROOT/cfg['prompts'][job['task']]).read_text(encoding='utf-8');schema=schemas[job['task']]
        assert cp['binding']['job_sha256']==digest(canonical(job)) and cp['binding']['config_sha256']==file_sha(CONFIG)
        assert cp['binding']['prompt_sha256']==job['prompt_sha256'] and cp['binding']['schema_sha256']==job['schema_sha256']
        if rawpath.exists():
            raw=read(rawpath);raws.append(raw);artifact_hashes[rawpath.relative_to(ROOT).as_posix()]=file_sha(rawpath)
        if request.exists():
            exact=gzip.decompress(request.read_bytes());body=payload(job,cfg,prompt,schema)
            assert exact==canonical(body)
            assert read(out/'request_intents'/path.name)['request_sha256']==digest(exact)
            assert 'Authorization' not in json.loads(exact)
            request_checks+=1
        if cp['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            reproduced=False
            if rawpath.exists():
                try:validate_output(job,json.loads(response_text(raw)),schema)
                except (ValueError,TypeError,KeyError,json.JSONDecodeError):reproduced=True
            slots.append({'job_id':job['job_id'],'business_id':job['business_id'],'status':'FAILED_PRESERVED',
                'validation_failure_reproduced':reproduced,'failure':cp})
            flags.append({'business_id':job['business_id'],'flag':'technical_failed_component','job_id':job['job_id'],'human_verified':False})
            continue
        assert cp['validation']==validate_output(job,cp['output'],schema)
        if cp.get('api_called'):
            assert request.exists() and file_sha(rawpath)==cp['raw_response_sha256']
            assert raw['model']==cfg['requested_model'] and raw['status']=='completed'
            assert json.loads(response_text(raw))==cp['output']
        else:
            assert job.get('skip_api_empty_captions') and cp['output']['assessments']==[]
        if job['task']=='review_consensus':
            resolved,provenance=resolve_review(job,cp['output'])
            assert resolved==cp['resolved_output'] and provenance==cp['evidence_resolution']
            assert [(c['text'],c['polarity']) for c in resolved['claims']]==[(c['text'],c['polarity']) for c in cp['output']['claims']]
            span_checks+=len(provenance)
            for i,claim in enumerate(resolved['claims']):reviews.append(dict(claim,business_id=job['business_id'],claim_index=i))
            for d in citation_diagnostics(job,cp['output']):flags.append(dict(d,business_id=job['business_id'],flag='redundant_reported_review_ID_mismatch'))
            if not resolved['claims']:flags.append({'business_id':job['business_id'],'flag':'empty_review_consensus','human_verified':False})
        else:
            for cap,photo,a in zip(job['evidence']['captions'],job['images'],cp['output']['assessments']):
                assert (a['caption_id'],a['photo_id'])==(cap['caption_id'],cap['photo_id'])
                for i,claim in enumerate(a['claims']):visual.append(dict(claim,business_id=job['business_id'],condition=job['condition'],
                    caption_id=cap['caption_id'],photo_id=cap['photo_id'],photo_marker=cap['photo_marker'],caption=cap['caption'],
                    photo_sha256=photo['raw_sha256'],claim_index=i))
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
        slots.append({'job_id':job['job_id'],'business_id':job['business_id'],'status':'STRUCTURAL_SOURCE_PASS_NOT_FACTUAL_CERTIFICATE'})
    valid=[];zero=[]
    for b in businesses:
        target=build_target(b,outputs.get(b['business_id'],{}));path=out/'provisional_targets'/(b['business_id']+'.json')
        if target is None:
            assert not path.exists();continue
        assert path.exists() and read(path)==target
        artifact_hashes[path.relative_to(ROOT).as_posix()]=file_sha(path)
        for cond,text in target['targets'].items():
            ids=native.encode(text,add_special_tokens=True).ids
            assert ids==slow.encode(text,add_special_tokens=True,truncation=False)
            target_lengths[cond].append(len(ids));target_checks+=1
            if len(ids)>512:flags.append({'business_id':b['business_id'],'condition':cond,'flag':'target_over512_diagnostic_not_rejection','tokens':len(ids),'human_verified':False})
        if b['zero_filtered_edge_case']:
            assert target['targets']['filtered']==target['targets']['reviews_only']+'\n\n[VISUAL_EVIDENCE]\n'
            zero.append(b['business_id'])
        exact_component_sentences+=len(target['evidence_records']['shared']['claims'])+sum(len(x) for x in target['visual_target_claims_with_all_evidence'].values())
        valid.append(b['business_id'])
    # Exact identity audit has no authority to reconcile teacher judgments.
    exact=consistency(visual)
    photo_groups=defaultdict(list)
    for claim in visual:photo_groups[(claim['business_id'],claim['photo_sha256'],claim['caption'])].append(claim)
    # Lexical similarity is a reporting heuristic only, never a semantic correctness verdict.
    for claims in photo_groups.values():
        left=[c for c in claims if c['condition']=='unfiltered'];right=[c for c in claims if c['condition']=='filtered']
        for a in left:
            for b in right:
                if a['verdict']==b['verdict']:continue
                similarity=difflib.SequenceMatcher(None,a['text'].lower(),b['text'].lower()).ratio()
                if similarity>=0.72:
                    flags.append({'business_id':a['business_id'],'flag':'lexically_similar_cross_condition_verdict_difference',
                        'photo_id':a['photo_id'],'similarity':similarity,'claim_a':a,'claim_b':b,
                        'not_equivalence_proof':True,'human_verified':False})
    claims_by_condition={cond:[c for c in visual if c['condition']==cond] for cond in ['unfiltered','filtered']}
    completion=read(out/'completion.json')
    assert len(valid)==completion['businesses_completed'] and target_checks==completion['target_texts']
    assert len(list((out/'request_intents').glob('*.json')))<=pre['max_new_API_calls']
    manual_ids={x['business_id'] for x in read(FOLDER/'manual_review_25_manifest.json')['businesses']}
    repeated=Counter(tuple(x['targets'].values()) for p in (out/'provisional_targets').glob('*.json') for x in [read(p)])
    report={'status':'D041_AUTOMATED_VALIDATION_PASS_WITH_PRESERVED_FAILURES' if len(valid)<100 else 'D041_AUTOMATED_VALIDATION_PASS_NOT_FACTUAL_CERTIFICATE',
        'businesses_planned':100,'businesses_complete':len(valid),'target_texts':target_checks,'task_slots':slots,
        'raw_API_responses':len(raws),'model_metadata':sorted({str(r.get('model')) for r in raws}),
        'response_ids':[r.get('id') for r in raws],'immutable_snapshot_verified':False,
        'exact_API_request_checks':request_checks,'exact_review_source_span_checks':span_checks,
        'native_slow_LED_target_ID_checks':target_checks,'teacher_API_sentence_origin_checks':exact_component_sentences,
        'review_claim_count':len(reviews),'visual_verdicts':{c:dict(Counter(x['verdict'] for x in values)) for c,values in claims_by_condition.items()},
        'target_tokens':{c:stats(values) for c,values in target_lengths.items()},
        'target_512_limit_diagnostic_only':True,'zero_caption_businesses_complete':zero,
        'exact_cross_condition':exact,'repeated_identical_three_target_sets':sum(n-1 for n in repeated.values() if n>1),
        'technical_flags':len(flags),'automatic_retries':0,'human_verified_businesses':0,'factual_accuracy_estimate':None,
        'random_manual_sample_count':25,'manual_random_sample_flag_overlap':sorted({x['business_id'] for x in flags}&manual_ids),
        'flags_do_not_replace_random_manual25_sample':True,'source_manifests_and_historical_files_unchanged':True,
        'API_usage_runtime_cost':completion,'production_protocol_frozen':False,'artifact_sha256':artifact_hashes,
        'warning':'Schema/source identity checks do not establish semantic entailment or independent human factual accuracy.'}
    immutable(FOLDER/'validation_report.json',report)
    immutable(FOLDER/'technical_flags.json',{'role':'technical_diagnostics_not_human_ground_truth','flags':flags,
        'additional_flagged_businesses_outside_random25':sorted({x['business_id'] for x in flags}-manual_ids)})
    immutable(FOLDER/'evidence_records_for_review.json',{'review_claims':reviews,'visual_claims':visual})
    print(json.dumps({k:report[k] for k in ['status','businesses_complete','target_texts','raw_API_responses','technical_flags']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-results',action='store_true')
    args=parser.parse_args();results() if args.audit_results else preflight()
