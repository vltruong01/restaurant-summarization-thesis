"""D-040: eight frozen businesses, <=16 fresh calls, independent teacher credential."""
import argparse
import copy
import json
from pathlib import Path
from collections import Counter
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1_recovery import (
    job_result, payload, validate_output, response_text, visual_claims, now,
)
from stage_g_teacher_credentials import credentials, BASE_URL, MODEL

PARENT = ROOT / 'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'
SOURCE = PARENT / 'execution_v1'
CONFIG = ROOT / 'configs/stage_g_teacher_pilot_v6_focused_d040.json'
SELECTED = ['bLJP0fVgscuUQy0hE_CRXA', 'UCzFaBbPDKtHTWsBE5LGtQ',
    'FFV8_b-ec5Z0E3xgvkVxKw', 'PRrWtQ3rdINngwESZNefVQ',
    '8xTHtLoNIwdpf0FEvIpQIw', 'nv59KN19fEQax9RL0lUdkw',
    'jMStvE-tQzSpRCAO0nAE6g', 'EzJI5x25U5GgbYBWhu5ZQQ']
REVIEW_RERUN = {'UCzFaBbPDKtHTWsBE5LGtQ'}


def prepare():
    parent_cfg = read(ROOT / 'configs/stage_g_teacher_pilot_v5_focused_d039.json')
    cfg = copy.deepcopy(parent_cfg)
    cfg.update(status='USER_APPROVED_D040_BOUNDED_PILOT_ONLY_NOT_PRODUCTION',
        protocol_version='stage_g_d040_visible_identity_count_and_category_scope_prompt',
        output_directory='outputs/stage_g/teacher_pilot_v6_focused_d040_v1', decision='D-040',
        parent_pilot_config='configs/stage_g_teacher_pilot_v5_focused_d039.json',
        revision_reason='Minimal prompt refinement for visible ingredient/count basis and review category scope; schema and retry unchanged.')
    cfg['prompts'] = {'review_consensus': 'prompts/stage_g_review_consensus_v7.txt',
                      'visual_evidence': 'prompts/stage_g_visual_evidence_v4.txt'}
    cfg['pilot'] = {'business_ids': SELECTED,
        'selection': 'four_affected_businesses_plus_hotdog_pizza_hamburger_zero_filtered_regression_controls',
        'review_rerun_business_ids': sorted(REVIEW_RERUN), 'max_new_API_calls': 16,
        'human_inspection': 'researcher_review_before_production', 'not_population_error_rate': True}
    immutable(CONFIG, cfg)
    folder = ROOT / cfg['output_directory']
    parents = read(PARENT / 'pilot_jobs.json')
    parent_plan = {p['job_id']: p for p in read(PARENT / 'execution_plan.json')}
    businesses = read(PARENT / 'pilot_businesses.json')
    protected = dict(read(PARENT / 'preflight.json')['preserved_original_sha256'])
    for path in PARENT.rglob('*'):
        if path.is_file(): protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    for relative in [parent_cfg['parent_pilot_config'], 'configs/stage_g_teacher_pilot_v5_focused_d039.json',
        *parent_cfg['prompts'].values(), 'scripts/run_stage_g_d039_focused_pilot.py',
        'scripts/audit_stage_g_d039_focused_pilot.py', 'scripts/run_stage_g_teacher_pilot_v4_1_recovery.py',
        cfg['schema'], cfg['source_pointer']]:
        protected[relative] = file_sha(ROOT / relative)
    # Preserve existing portable discussion artifacts as well as scientific pilots.
    for name in ['chatgpt_stage_g_pilot_review_bundle.zip', 'chatgpt_stage_g_d038_comparison_bundle.zip',
                 'chatgpt_stage_g_d039_comparison_bundle.zip']:
        path = ROOT / 'outputs/stage_g' / name
        if path.exists(): protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    jobs, plan = [], []
    for bid in SELECTED:
        selected = [j for j in parents if j['business_id'] == bid]
        assert len(selected) == 3 and {j['split'] for j in selected} <= {'train','validation'}
        for old in selected:
            job = copy.deepcopy(old)
            reuse = job['task'] == 'review_consensus' and bid not in REVIEW_RERUN
            actual_old_prompt = parent_plan[job['job_id']]['prompt_path']
            prompt_path = actual_old_prompt if reuse else cfg['prompts'][job['task']]
            job['prompt_sha256'] = file_sha(ROOT / prompt_path)
            assert {k:v for k,v in job.items() if k!='prompt_sha256'} == {k:v for k,v in old.items() if k!='prompt_sha256'}
            for photo in job['images']:
                assert file_sha(ROOT / photo['relative_path']) == photo['raw_sha256']
                protected[photo['relative_path']] = photo['raw_sha256']
            jobs.append(job)
            plan.append({'job_id':job['job_id'], 'prompt_path':prompt_path,
                'source_prompt_path':actual_old_prompt,
                'mode':'import_unchanged_D039_review' if reuse else
                    ('deterministic_empty' if job.get('skip_api_empty_captions') else 'new_call_once')})
    assert Counter(p['mode'] for p in plan) == {'new_call_once':16,'import_unchanged_D039_review':7,'deterministic_empty':1}
    selected_businesses = [next(b for b in businesses if b['business_id']==bid) for bid in SELECTED]
    immutable(folder / 'pilot_jobs.json', jobs)
    immutable(folder / 'pilot_businesses.json', selected_businesses)
    immutable(folder / 'execution_plan.json', plan)
    inputs = [CONFIG.relative_to(ROOT).as_posix(), *cfg['prompts'].values(), cfg['schema'],
        'scripts/run_stage_g_d040_focused_pilot.py', 'scripts/stage_g_teacher_credentials.py',
        'scripts/run_stage_g_teacher_pilot_v4_1_recovery.py', 'scripts/stage_g_canonical_review_ids.py',
        'scripts/stage_g_visual_target_serialization.py']
    observed = []
    for job in jobs:
        if next(p for p in plan if p['job_id']==job['job_id'])['mode'] == 'new_call_once':
            cp = read(SOURCE / 'checkpoints' / (job['job_id']+'.json'))
            observed.append(cp.get('usage') or {})
    estimate = {'basis':'provider-reported D039 usage for these same tasks; prompt sizes changed slightly',
        'comparable_prior_input_tokens':sum(x.get('input_tokens',0) for x in observed),
        'comparable_prior_output_tokens':sum(x.get('output_tokens',0) for x in observed),
        'fresh_call_ceiling':16, 'max_output_tokens_per_call':8192,
        'total_output_token_ceiling':16*8192,
        'gateway_currency_cost':None, 'not_an_invoice':True,
        'spending_authorization':'Previously authorized bounded teacher pilots plus current explicit focused rerun instruction; no production spending.'}
    immutable(folder / 'pilot_usage_estimate.json', estimate)
    immutable(folder / 'preflight.json', {'status':'D040_OFFLINE_PREFLIGHT_PASS', 'businesses':8,
        'task_slots':24, 'max_new_API_calls':16, 'review_imports':7, 'empty_tasks':1,
        'jobs_sha256':digest(canonical(jobs)),
        'input_sha256':{p:file_sha(ROOT/p) for p in inputs}, 'preserved_original_sha256':protected,
        'same_reviews_captions_order_and_original_photos':True,
        'same_model_generation_schema_validators_and_Stage_F':True,
        'teacher_runtime':'independent_teacher_key_only_no_Codex_config_or_login_access',
        'cross_condition_policy':'audit_exact_photo_hash_caption_claim_matches; never force verdicts or rewrite outputs',
        'automatic_retries':0, 'paid_call_authorization':estimate['spending_authorization'],
        'no_test_or_full_production_or_LED_training':True})
    for p,h in protected.items(): assert file_sha(ROOT/p)==h
    print(json.dumps({'preflight':'PASS','businesses':8,'max_new_calls':16,'review_imports':7,'empty_tasks':1,'usage_estimate':estimate}),flush=True)
    return cfg, folder


def run():
    cfg = read(CONFIG); folder = ROOT / cfg['output_directory']; pre = read(folder/'preflight.json')
    for relative, expected in {**pre['input_sha256'], **pre['preserved_original_sha256']}.items():
        if file_sha(ROOT / relative) != expected: raise ValueError('Protected input changed: '+relative)
    out = folder / 'execution_v1'
    if (out/'completion.json').exists():
        print(json.dumps(read(out/'completion.json'),indent=2)); return
    assert cfg['requested_model']==MODEL
    base, provider, key = credentials()
    assert base==BASE_URL
    jobs=read(folder/'pilot_jobs.json'); plan={p['job_id']:p for p in read(folder/'execution_plan.json')}
    assert digest(canonical(jobs))==pre['jobs_sha256']
    schemas=read(ROOT/cfg['schema']); parent_cfg=read(ROOT/cfg['parent_pilot_config'])
    parent_jobs={j['job_id']:j for j in read(PARENT/'pilot_jobs.json')}
    manifest={'status':'D040_BOUNDED_ONLY','config_sha256':file_sha(CONFIG),
        'runner_sha256':file_sha(Path(__file__)), 'credential_resolver_sha256':file_sha(ROOT/'scripts/stage_g_teacher_credentials.py'),
        'provider':provider,'requested_model':cfg['requested_model'],
        'generation_settings':cfg['generation_proposal'], 'max_new_API_calls':16,
        'credential_source':'teacher-only runtime; secret excluded', 'production_authorized':False,'user_labels_changed':False}
    immutable(out/'run_manifest.json',manifest)
    failures=[]; outputs={}; fresh_attempts=0
    for job in jobs:
        spec=plan[job['job_id']]; prompt=(ROOT/spec['prompt_path']).read_text(encoding='utf-8'); schema=schemas[job['task']]
        path=out/'checkpoints'/(job['job_id']+'.json')
        if spec['mode']=='import_unchanged_D039_review':
            original=read(SOURCE/'checkpoints'/path.name); raw=SOURCE/'raw_responses'/path.name
            assert original['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
            assert file_sha(raw)==original['raw_response_sha256']
            assert json.loads(response_text(read(raw)))==original['output']
            old_prompt=(ROOT/spec['source_prompt_path']).read_text(encoding='utf-8')
            body=payload(job,cfg,prompt,schema)
            assert canonical(body)==canonical(payload(parent_jobs[job['job_id']],parent_cfg,old_prompt,schema))
            validate_output(job,original['output'],schema)
            cp=copy.deepcopy(original)
            cp['binding'].update(job_sha256=digest(canonical(job)),config_sha256=file_sha(CONFIG),
                prompt_sha256=digest(prompt.encode('utf-8')),schema_sha256=digest(canonical(schema)))
            cp['imported_completed_response']=True
            cp['import_lineage']={'checkpoint':(SOURCE/'checkpoints'/path.name).relative_to(ROOT).as_posix(),
                'source_checkpoint_sha256':file_sha(SOURCE/'checkpoints'/path.name),
                'identical_request_sha256':digest(canonical(body)),
                'actual_source_prompt_path':spec['source_prompt_path'],'review_prompt_v7_not_applied':True}
            rp=out/'raw_responses'/path.name;rp.parent.mkdir(parents=True,exist_ok=True)
            if rp.exists():assert file_sha(rp)==file_sha(raw)
            else:rp.write_bytes(raw.read_bytes())
            immutable(path,cp)
            print('Imported unchanged review: '+job['job_id'],flush=True)
        else:
            if path.exists() and read(path)['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':read(path)});continue
            if spec['mode']=='new_call_once' and not path.exists():
                # Total intents, including a prior interrupted execution, enforce the ceiling.
                assert len(list((out/'request_intents').glob('*.json')))<16
                fresh_attempts+=1
            try:
                cp=job_result(job,cfg,prompt,schema,base,provider,key,out,CONFIG)
            except (RuntimeError,ValueError):
                failed=read(path) if path.exists() else {'status':'UNRESOLVED_REQUEST_NO_RESEND'}
                failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':failed})
                print('Failed task preserved; no retry: '+job['job_id'],flush=True)
                if failed.get('http_status') in [401,403,429]:break
                continue
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
    completed=[]
    for business in read(folder/'pilot_businesses.json'):
        bid=business['business_id'];content=outputs.get(bid,{})
        if set(content)!={'shared','unfiltered','filtered'}:continue
        consensus=' '.join(c['text'] for c in content['shared']['claims'])
        targets={'reviews_only':'[REVIEW_CONSENSUS]\n'+consensus};visual={}
        for condition in ['unfiltered','filtered']:
            visual[condition]=visual_claims(content[condition]['assessments'])
            targets[condition]=targets['reviews_only']+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(c['text'] for c in visual[condition])
        immutable(out/'provisional_targets'/(bid+'.json'),{
            'status':'PROVISIONAL_PILOT_TARGETS_NOT_HUMAN_VERIFIED_NOT_PRODUCTION',
            'business_id':bid,'split':business['split'],'targets':targets,'evidence_records':content,
            'human_verified':False,'review_quote_origin':'code_resolved_from_source_span_IDs',
            'visual_target_claims_with_all_evidence':visual,'empty_review_consensus_flag':not consensus,
            'review_IDs_text_order_shared':True,'zero_filtered_edge_case':business['zero_filtered_edge_case'],
            'review_component_source':'fresh_v7' if bid in REVIEW_RERUN else 'unchanged_D039_import_actual_prompt_retained'})
        completed.append(business)
    intents=list((out/'request_intents').glob('*.json'))
    report={'status':'D040_ATTEMPTED_WITH_FAILURES' if failures else 'D040_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE',
        'businesses_planned':8,'businesses_completed':len(completed),'target_texts':len(completed)*3,
        'new_API_request_intents':len(intents),'fresh_attempts_this_execution':fresh_attempts,
        'review_imports_planned':7,'empty_tasks_planned':1,'failed_tasks':failures,
        'valid_businesses_by_split':dict(Counter(b['split'] for b in completed)),
        'human_verified_businesses':0,'production_method_frozen':False,'held_out_teacher_targets':0,'LED_training_started':False}
    immutable(out/'completion.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute-pilot',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.execute_pilot:run()
    if not (args.prepare or args.execute_pilot):parser.error('Explicit --prepare or --execute-pilot required')
