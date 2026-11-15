"""Offline production/source/length audit. Structural PASS is not factual ground truth."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from stage_g_production_d042 import (
    ROOT, CONFIG, FOLDER, OUT, PILOT_OUT, PILOT_CONFIG, read, rows, file_sha, immutable,
    atomic, check_inputs, location, verify_checkpoint, save_completed_business, build_target,
    citation_diagnostics, now,
)
from audit_stage_g_d041_continuation_v2 import tokenizer_pair
from audit_stage_g_d039_focused_pilot import stats, percentile, consistency


def distribution(values):
    return dict(stats(values),p90=percentile(values,.90),p99=percentile(values,.99),
                over512=sum(v>512 for v in values),over640=sum(v>640 for v in values),
                over768=sum(v>768 for v in values),over1024=sum(v>1024 for v in values),
                histogram={str(k):v for k,v in sorted(Counter(values).items())})


def run(smoke=False):
    cfg,jobs,businesses,imports=check_inputs()
    if smoke:
        ids=set(read(FOLDER/'small_batch_manifest.json')['business_ids'])
        businesses=[b for b in businesses if b['business_id'] in ids]
        jobs=[j for j in jobs if j['business_id'] in ids]
        directory=FOLDER/'small_batch_audit'
    else: directory=FOLDER/'final_validation'
    directory.mkdir(parents=True,exist_ok=True)
    native,slow=tokenizer_pair(); by_business=defaultdict(list);components={};visual=[];flags=[]
    span_checks=0;request_checks=0;api_sentences=0;hashes={};usage=Counter();models=set();response_ids=[]
    for j in jobs:
        by_business[j['business_id']].append(j)
        origin=location(j,imports);cp=verify_checkpoint(j,cfg,origin,PILOT_CONFIG if j['job_id'] in imports else CONFIG)
        p=origin/'checkpoints'/(j['job_id']+'.json');hashes[p.relative_to(ROOT).as_posix()]=file_sha(p)
        components.setdefault(j['business_id'],{})[j['condition']]=cp.get('resolved_output',cp['output'])
        if cp.get('api_called'):
            request_checks+=1
            raw=read(origin/'raw_responses'/p.name);models.add(raw['model']);response_ids.append(raw['id'])
            for path in [origin/'raw_responses'/p.name,origin/'request_intents'/p.name,origin/'raw_requests'/(j['job_id']+'.json.gz')]:
                hashes[path.relative_to(ROOT).as_posix()]=file_sha(path)
            u=raw.get('usage') or {}
            if u:
                usage['responses_with_usage']+=1
                usage['input_tokens']+=u.get('input_tokens',0);usage['output_tokens']+=u.get('output_tokens',0)
                usage['reasoning_tokens']+=u.get('output_tokens_details',{}).get('reasoning_tokens',0)
        if j['task']=='review_consensus':
            span_checks+=len(cp['evidence_resolution'])
            flags += [dict(d,business_id=j['business_id'],flag='auxiliary_reported_review_ID_disagreement',human_verified=False)
                      for d in citation_diagnostics(j,cp['output'])]
            if not cp['resolved_output']['claims']: flags.append({'business_id':j['business_id'],'flag':'empty_review_consensus','human_verified':False})
        else:
            for cap,photo,a in zip(j['evidence']['captions'],j['images'],cp['output']['assessments']):
                for i,c in enumerate(a['claims']):
                    visual.append(dict(c,business_id=j['business_id'],condition=j['condition'],
                           photo_id=cap['photo_id'],photo_sha256=photo['raw_sha256'],caption=cap['caption'],
                           caption_id=cap['caption_id'],photo_marker=cap['photo_marker'],claim_index=i))
    source_pointer=read(ROOT/cfg['source_pointer']);source_root=ROOT/source_pointer['production_directory']
    sources={}
    for split in ('train','validation'):
        for c in ('reviews_only','unfiltered','filtered'):
            sources[split,c]={r['business_id']:r for r in rows(source_root/f'manifests/{split}/{c}_sources.jsonl')}
    records=[];lengths=defaultdict(list);source_lengths=defaultdict(list);length_rows=[];over512=[];zero=[]
    for b in businesses:
        target=build_target(b,components[b['business_id']]); assert target
        path=save_completed_business(b,by_business,imports);assert read(path)==target
        hashes[path.relative_to(ROOT).as_posix()]=file_sha(path)
        review=target['targets']['reviews_only']
        for c,text in target['targets'].items():
            assert c=='reviews_only' or text.startswith(review+'\n\n[VISUAL_EVIDENCE]\n')
            ids=native.encode(text,add_special_tokens=True).ids
            assert ids==slow.encode(text,add_special_tokens=True,truncation=False)
            s=sources[b['split'],c][b['business_id']]
            assert s['selected_review_ids']==b['selected_review_ids'] and s['review_block_sha256']==b['review_block_sha256']
            source_ids=native.encode(s['source'],add_special_tokens=True).ids
            assert source_ids==slow.encode(s['source'],add_special_tokens=True,truncation=False)
            assert len(source_ids)==s['tokens']<=5000
            lengths[c].append(len(ids));source_lengths[c].append(len(source_ids))
            row={'business_id':b['business_id'],'split':b['split'],'condition':c,'target_tokens':len(ids),'source_tokens':len(source_ids)}
            length_rows.append(row)
            if len(ids)>512:over512.append(row)
            record={**row,'source':s['source'],'target':text,'source_sha256':s['source_sha256'],
                    'selected_review_ids':s['selected_review_ids'],'selected_photo_ids':s['selected_photo_ids'],
                    'target_artifact':path.relative_to(ROOT).as_posix(), 'target_artifact_sha256':file_sha(path),
                    'evidence_records':target['evidence_records']['shared'] if c=='reviews_only' else target['evidence_records'][c],
                    'shared_review_evidence':target['evidence_records']['shared'],
                    'visual_claims_with_evidence':target['visual_target_claims_with_all_evidence'].get(c,[]),
                    'target_content_origin':'teacher_API_only_frozen_serialization','independent_human_verified':False,
                    'decision':'D-042','imported_D041_without_regeneration':by_business[b['business_id']][0]['job_id'] in imports}
            records.append(record)
        if b['zero_filtered_edge_case']:
            assert target['targets']['filtered']==review+'\n\n[VISUAL_EVIDENCE]\n';zero.append(b['business_id'])
        api_sentences+=len(target['evidence_records']['shared']['claims'])+sum(len(v) for v in target['visual_target_claims_with_all_evidence'].values())
    assert len(records)==3*len(businesses) and len({(r['business_id'],r['condition']) for r in records})==len(records)
    assert len(response_ids)==len(set(response_ids))
    cross=consistency(visual)
    report={'status':'D042_SMALL_BATCH_AUTOMATED_PASS' if smoke else 'STAGE_G_PRODUCTION_AUTOMATED_VALIDATION_PASS_NOT_FACTUAL_CERTIFICATE',
            'businesses_complete':len(businesses),'by_split':dict(Counter(b['split'] for b in businesses)),
            'condition_aligned_targets':len(records),'checkpoints_verified':len(jobs),'exact_API_request_checks':request_checks,
            'exact_review_span_checks':span_checks,'API_sentence_origin_checks':api_sentences,
            'native_slow_LED_target_ID_checks':len(records),'native_slow_LED_source_ID_checks':len(records),
            'review_ID_text_order_shared':True,'same_review_consensus_across_conditions':True,
            'no_unverified_assessments_serialized_by_provenance':True,'source_over5000':0,
            'zero_filtered_business_ids':zero,'target_tokens':{c:distribution(v) for c,v in lengths.items()},
            'source_tokens':{c:distribution(v) for c,v in source_lengths.items()},'targets_over512':len(over512),
            'teacher_reported_visual_verdicts':dict(Counter(c['verdict'] for c in visual)),
            'exact_cross_condition_matched_groups':cross['matched_groups'],'exact_cross_condition_conflicts':cross['conflict_groups'],
            'auxiliary_or_empty_review_flags':len(flags),'model_aliases':sorted(models),'immutable_model_snapshot_verified':False,
            'usage':dict(usage),'no_content_truncated_or_manually_rewritten':True,'decoder_limit':None,
            'human_annotation_status_preserved':True,'independent_factual_quality_certified':False,'human_final_records_created':0,
            'held_out_targets':0,'LED_training_started':False,'artifact_sha256':hashes}
    immutable(directory/'validation_report.json',report)
    immutable(directory/'target_length_cases_over512.json',over512)
    immutable(directory/'technical_flags.json',flags)
    immutable(directory/'cross_condition_conflicts.json',cross['conflicts'])
    with (directory/'target_lengths.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(length_rows[0]));writer.writeheader();writer.writerows(length_rows)
    if smoke:
        immutable(FOLDER/'small_batch_validation.json',{'status':report['status'],'businesses':len(businesses),
                  'targets':len(records),'API_model_unchanged':True,'schema_evidence_and_empty_behavior_checked':True,
                  'validation_report':(directory/'validation_report.json').relative_to(ROOT).as_posix(),
                  'validation_report_sha256':file_sha(directory/'validation_report.json')})
    else:
        assert len(businesses)==2200 and len(records)==6600
        dataset=FOLDER/'dataset';dataset.mkdir(exist_ok=True)
        for split,expected in [('train',2000),('validation',200)]:
            for c in ('reviews_only','unfiltered','filtered'):
                selected=[r for r in records if r['split']==split and r['condition']==c];assert len(selected)==expected
                p=dataset/f'{split}_{c}.jsonl'
                content=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in selected)
                if p.exists():assert p.read_text(encoding='utf-8')==content
                else:p.write_text(content,encoding='utf-8')
                hashes[p.relative_to(ROOT).as_posix()]=file_sha(p)
        checklist={'teacher_generation_complete':True,'all6600_targets_source_schema_valid':True,'held_out_targets_absent':True,
                   'source_condition_alignment_valid':True,'known_visual_grounding_limitations_accepted':True,
                   'independent_human25_annotation_completed':False,'25_materials_preserved':True,
                   'decoder_label_generation_length_policy_approved':False,'LED_training_implementation_validated':False,
                   'LED_training_authorized':False,'next_action':'review_full_target_distribution_and_approve_student_length_policy_before_training'}
        immutable(FOLDER/'training_readiness_checklist.json',checklist)
        immutable(FOLDER/'completion_report.json',{'status':'STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS',
                  'validation_report':(directory/'validation_report.json').relative_to(ROOT).as_posix(),
                  'businesses':2200,'targets':6600,'reused_businesses':100,'new_businesses':2100,
                  'dataset_sha256':{k:v for k,v in hashes.items() if '/dataset/' in k},'target_tokens':report['target_tokens'],
                  'decoder_limit':None,'training_started':False,'training_readiness':checklist,'completed_utc':now()})
        text='# Stage G production completion\n\n'+json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},ensure_ascii=False,indent=2)+'\n\n'
        text+='Structural/source validation is separate from factual quality. Known ingredient/object/count errors and cross-condition verification disagreements remain possible. Captions may be simple. Provider exposes an alias, not an attested immutable backend snapshot. The existing random25 material/status is unchanged.\n\nNo decoder cap or LED training has been adopted. Review the full target length distribution and every >512 case before deciding student preprocessing/generation limits.\n'
        (FOLDER/'COMPLETION_REPORT.md').write_text(text,encoding='utf-8')
        state=read(OUT/'execution_progress.json');state.update(status='STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS',phase='complete_stop_before_training',updated_utc=now(),current_jobs=[])
        atomic(OUT/'execution_progress.json',state)
        from run_stage_g_production_d042 import update_docs
        update_docs(state)
        atomic(ROOT/'outputs/stage_g/production_manifest_pointer.json',{'status':'STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS',
              'production_directory':FOLDER.relative_to(ROOT).as_posix(),'completion_report':(FOLDER/'completion_report.json').relative_to(ROOT).as_posix(),
              'completion_report_sha256':file_sha(FOLDER/'completion_report.json'),'teacher_targets_for_test':False,'decoder_limit':None,'training_started':False})
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifact_sha256','target_tokens','source_tokens')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(args.smoke)
