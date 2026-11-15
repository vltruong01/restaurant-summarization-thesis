"""100-business fixed-D040 API pilot; explicit paid authorization, no automatic retry."""
import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from prepare_stage_g_extended_pilot_d041 import CONFIG, FOLDER, PARENT, CRITICAL_KEYS
from stage_g_teacher_credentials import credentials, BASE_URL, MODEL, PROVIDER
from run_stage_g_teacher_pilot_v4_1_recovery import job_result, payload, response_text, now
from stage_g_visual_target_serialization import visual_claims


def check_inputs():
    cfg=read(CONFIG);pre=read(FOLDER/'preflight.json');parent=read(PARENT)
    for k in CRITICAL_KEYS:
        if cfg[k]!=parent[k]:raise ValueError('D040 fixed field changed: '+k)
    for rel,h in {**pre['input_sha256'],**pre['preserved_original_sha256']}.items():
        if file_sha(ROOT/rel)!=h:raise ValueError('Protected input changed: '+rel)
    for rel,h in pre['frozen_output_sha256'].items():
        if file_sha(FOLDER/rel)!=h:raise ValueError('Frozen sampling/preflight input changed: '+rel)
    for rel,h in pre['protocol_snapshot_sha256'].items():
        if file_sha(FOLDER/'protocol_snapshot'/rel)!=h:raise ValueError('Protocol snapshot changed: '+rel)
    jobs=read(FOLDER/'pilot_jobs.json')
    if digest(canonical(jobs))!=pre['jobs_sha256']:raise ValueError('Jobs changed')
    return cfg,pre,jobs


def approved_authorization(path, pre):
    """Offline guard; authorization must be created only after an explicit user approval."""
    if not path.is_file():raise ValueError('Paid execution not authorized. No API call or credential access performed.')
    auth=read(path)
    for key,value in {'approved':True,'approved_by':'user','business_count':100,'production_authorized':False}.items():
        if auth.get(key)!=value:raise ValueError('Invalid bounded authorization: '+key)
    if auth.get('sampling_manifest_sha256')!=file_sha(FOLDER/'sampling_manifest.json'):
        raise ValueError('Authorization does not bind this predeclared cohort')
    if auth.get('config_sha256')!=file_sha(CONFIG):raise ValueError('Authorization config mismatch')
    if auth.get('maximum_calls')!=pre['max_new_API_calls']:raise ValueError('Authorization call ceiling mismatch')
    if not auth.get('provider_tariff_confirmed') or not auth.get('account_usage_policy_confirmed'):
        raise ValueError('Provider tariff/account usage policy not confirmed')
    for key in ['input_price_per_million','output_price_per_million','currency_limit','input_token_reservation_per_call']:
        value=auth.get(key)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or value<=0:
            raise ValueError('Missing positive cost authorization field: '+key)
    if not auth.get('currency'):raise ValueError('Currency unit missing')
    if not auth.get('reservation_is_not_a_provider_input_token_hard_cap_acknowledged'):
        raise ValueError('Per-call input reservation limit must be acknowledged')
    return auth


def usage_so_far(out):
    records=[read(p) for p in (out/'raw_responses').glob('*.json')]
    return {'input_tokens':sum((r.get('usage') or {}).get('input_tokens',0) for r in records),
        'output_tokens':sum((r.get('usage') or {}).get('output_tokens',0) for r in records),
        'reasoning_tokens':sum((r.get('usage') or {}).get('output_tokens_details',{}).get('reasoning_tokens',0) for r in records),
        'responses_with_usage':sum(bool(r.get('usage')) for r in records)}


def cost(usage,auth):
    return (usage['input_tokens']*auth['input_price_per_million']+usage['output_tokens']*auth['output_price_per_million'])/1000000


def build_target(business,content):
    """Only frozen concatenation/deduplication of verbatim API sentences; no model writing."""
    if set(content)!={'shared','unfiltered','filtered'}:return None
    review='[REVIEW_CONSENSUS]\n'+' '.join(c['text'] for c in content['shared']['claims'])
    targets={'reviews_only':review};visual={}
    for cond in ['unfiltered','filtered']:
        visual[cond]=visual_claims(content[cond]['assessments'])
        targets[cond]=review+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(c['text'] for c in visual[cond])
    return {'status':'PROVISIONAL_FIXED_PROTOCOL_PILOT_TARGETS_NOT_HUMAN_VERIFIED_NOT_PRODUCTION',
        'business_id':business['business_id'],'split':business['split'],'targets':targets,'evidence_records':content,
        'human_verified':False,'review_quote_origin':'code_resolved_from_source_span_IDs',
        'visual_target_claims_with_all_evidence':visual,'empty_review_consensus_flag':not content['shared']['claims'],
        'review_IDs_text_order_shared':True,'zero_filtered_edge_case':business['zero_filtered_edge_case'],
        'review_component_source':'fresh_v7_API','target_content_origin':'teacher_API_only_frozen_serialization'}


def execute(authorization_path):
    cfg,pre,jobs=check_inputs()
    out=FOLDER/'execution_v1'
    if (out/'completion.json').exists():
        print(json.dumps(read(out/'completion.json'),indent=2));return
    auth=approved_authorization(authorization_path,pre)  # Must precede credential resolution.
    base,provider,key=credentials()
    assert base==BASE_URL and provider==PROVIDER and cfg['requested_model']==MODEL
    manifest={'scope':'D041_100_BUSINESSES_ONLY_NOT_PRODUCTION','created_utc':now(),
        'config_sha256':file_sha(CONFIG),'sampling_manifest_sha256':file_sha(FOLDER/'sampling_manifest.json'),
        'authorization_sha256':file_sha(authorization_path),'requested_model':cfg['requested_model'],
        'snapshot':cfg['snapshot'],'provider':provider,'generation_settings':cfg['generation_proposal'],
        'maximum_calls':pre['max_new_API_calls'],'automatic_retries':0,'Codex_configuration_or_authentication_access':False}
    if not (out/'run_manifest.json').exists():immutable(out/'run_manifest.json',manifest)
    else:
        existing=read(out/'run_manifest.json')
        for k in ['authorization_sha256','config_sha256','sampling_manifest_sha256','requested_model','maximum_calls']:
            assert existing[k]==manifest[k]
    immutable(out/'paid_authorization_without_secrets.json',auth)
    schemas=read(ROOT/cfg['schema']);prompts={k:(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    started=time.monotonic();failures=[];outputs={};stop_reason=None
    for job in jobs:
        path=out/'checkpoints'/(job['job_id']+'.json');intent=out/'request_intents'/path.name
        if path.exists() and read(path)['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
            failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':read(path)});continue
        if intent.exists() and not path.exists():
            failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':{'status':'UNRESOLVED_REQUEST_NO_RESEND'}})
            stop_reason='Unresolved prior request may have incurred cost; review before further spending';break
        is_new=not path.exists() and not job.get('skip_api_empty_captions')
        if is_new:
            assert len(list((out/'request_intents').glob('*.json')))<pre['max_new_API_calls']
            current=usage_so_far(out)
            expected_reservation=(auth['input_token_reservation_per_call']*auth['input_price_per_million']+
                cfg['generation_proposal']['max_output_tokens']*auth['output_price_per_million'])/1000000
            if cost(current,auth)+expected_reservation>auth['currency_limit']:
                stop_reason='Known usage plus approved next-call reservation exceeds currency limit';break
            body=payload(job,cfg,prompts[job['task']],schemas[job['task']])
            raw_request=canonical(body)
            # Exact compressed body; Authorization header/key are never written.
            request_path=out/'raw_requests'/(job['job_id']+'.json.gz')
            compressed=gzip.compress(raw_request,mtime=0)
            request_path.parent.mkdir(parents=True,exist_ok=True)
            if request_path.exists():assert gzip.decompress(request_path.read_bytes())==raw_request
            else:request_path.write_bytes(compressed)
        try:
            cp=job_result(job,cfg,prompts[job['task']],schemas[job['task']],base,provider,key,out,CONFIG)
        except (RuntimeError,ValueError):
            failed=read(path) if path.exists() else {'status':'UNRESOLVED_REQUEST_NO_RESEND'}
            failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':failed})
            if failed.get('http_status') in [401,403,429] or not (out/'raw_responses'/path.name).exists():
                stop_reason='Account/rate/network failure; preserved and stopped without retry';break
            raw=read(out/'raw_responses'/path.name)
            if raw.get('model')!=MODEL or not raw.get('usage'):
                stop_reason='Provider model/usage unavailable; review provenance before more calls';break
            continue
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
        if cp.get('api_called') and not cp.get('usage'):
            stop_reason='Missing usage accounting; no further paid calls';break
        if is_new and cost(usage_so_far(out),auth)>auth['currency_limit']:
            stop_reason='Actual cost estimate exceeded authorization; stop, do not claim hard billing-cap guarantee';break
    completed=[]
    for b in read(FOLDER/'pilot_businesses.json'):
        result=build_target(b,outputs.get(b['business_id'],{}))
        if result is not None:
            immutable(out/'provisional_targets'/(b['business_id']+'.json'),result);completed.append(b)
    usage=usage_so_far(out)
    report={'status':'D041_STOPPED_FOR_REVIEW' if stop_reason else ('D041_ATTEMPTED_WITH_FAILURES' if failures else 'D041_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE'),
        'businesses_planned':100,'businesses_completed':len(completed),'target_texts':3*len(completed),
        'businesses_by_split':dict(Counter(b['split'] for b in completed)),
        'new_API_request_intents':len(list((out/'request_intents').glob('*.json'))),
        'empty_tasks_planned':pre['empty_tasks'],'usage':usage,'currency':auth['currency'],
        'cost_estimated_from_confirmed_tariff':cost(usage,auth),'provider_invoice_cost':None,
        'runtime_seconds_this_execution':round(time.monotonic()-started,3),'failed_tasks':failures,'stop_reason':stop_reason,
        'automatic_retries':0,'human_verified_businesses':0,'production_protocol_frozen':False,
        'held_out_teacher_targets':0,'remaining_2100_generation_started':False,'LED_training_started':False}
    immutable(out/'completion.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-pilot',action='store_true')
    parser.add_argument('--authorization',type=Path,default=FOLDER/'paid_authorization.json')
    args=parser.parse_args()
    if args.execute_pilot:execute(args.authorization)
    else:
        cfg,pre,jobs=check_inputs()
        print(json.dumps({'status':pre['status'],'model':cfg['requested_model'],'maximum_calls':pre['max_new_API_calls'],
            'API_calls':0,'new_teacher_targets':0,'paid_authorization_present':args.authorization.is_file()},indent=2))
