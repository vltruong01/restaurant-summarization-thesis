"""Explicit bounded D-039 pilot: 19 fresh calls, seven unchanged review imports, no production."""
import argparse
import copy
import json
from pathlib import Path
from collections import Counter
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical, immutable
from run_stage_g_teacher_pilot_v4_1_recovery import (
    credentials, job_result, payload, validate_output, response_text, visual_claims, now,
)

PARENT = ROOT / 'outputs/stage_g/teacher_pilot_v4_1_grounding_d038_v1'
SOURCE = PARENT / 'execution_v2_quote_recovery'
CONFIG = ROOT / 'configs/stage_g_teacher_pilot_v5_focused_d039.json'
SELECTED = ['8xTHtLoNIwdpf0FEvIpQIw', 'bLJP0fVgscuUQy0hE_CRXA',
    'UCzFaBbPDKtHTWsBE5LGtQ', 'PRrWtQ3rdINngwESZNefVQ', 'FFV8_b-ec5Z0E3xgvkVxKw',
    'pWT3WAUvww4iqF-v2htBXA', 'nv59KN19fEQax9RL0lUdkw', 'jMStvE-tQzSpRCAO0nAE6g',
    'EzJI5x25U5GgbYBWhu5ZQQ']
REVIEW_RERUN = {'PRrWtQ3rdINngwESZNefVQ', 'EzJI5x25U5GgbYBWhu5ZQQ'}


def prepare():
    parent_cfg = read(ROOT / 'configs/stage_g_teacher_pilot_v4_1_grounding.json')
    cfg = copy.deepcopy(parent_cfg)
    cfg.update(status='USER_APPROVED_D039_FOCUSED_PILOT_ONLY_NOT_PRODUCTION',
        protocol_version='stage_g_d039_focused_identity_grounding_and_literal_quote_prompt',
        output_directory='outputs/stage_g/teacher_pilot_v5_focused_d039_v1', decision='D-039',
        parent_pilot_config='configs/stage_g_teacher_pilot_v4_1_grounding.json',
        revision_reason='Prompt-only visual identity/nonpropagation/exact-quote and review category/contrast refinement; schema unchanged.')
    cfg['prompts'] = {'review_consensus': 'prompts/stage_g_review_consensus_v6.txt',
                      'visual_evidence': 'prompts/stage_g_visual_evidence_v3.txt'}
    cfg['pilot'] = {'business_ids': SELECTED, 'selection': 'user_identified_six_problematic_plus_two_ordinary_description_controls_plus_one_zero_filtered_control',
        'review_rerun_business_ids': sorted(REVIEW_RERUN), 'max_new_API_calls': 19,
        'human_inspection': 'bounded_pilot_requires_researcher_review', 'not_population_error_rate': True}
    immutable(CONFIG, cfg)
    folder = ROOT / cfg['output_directory']
    parent_jobs = read(PARENT / 'pilot_jobs.json')
    businesses = read(PARENT / 'pilot_businesses.json')
    jobs = []
    plan = []
    protected = {}
    # Preserve all historical pilot scientific artifacts, not just selected successes.
    roots = [ROOT / 'outputs/stage_g/teacher_pilot_v3_canonical_review_ids_v1',
             ROOT / 'outputs/stage_g/teacher_pilot_v4_grounding_d038_v1', PARENT]
    for r in roots:
        for path in r.rglob('*'):
            if path.is_file(): protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    for path in [ROOT / 'configs/stage_g_teacher_pilot_v4_1_grounding.json',
                 ROOT / 'prompts/stage_g_review_consensus_v5.txt', ROOT / 'prompts/stage_g_visual_evidence_v2.txt',
                 ROOT / cfg['schema'], ROOT / cfg['source_pointer']]:
        protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    bindings = read(PARENT / 'preparation_report.json')['input_sha256']
    for relative, expected in bindings.items():
        assert file_sha(ROOT / relative) == expected
        protected[relative] = expected
    for bid in SELECTED:
        selected = [j for j in parent_jobs if j['business_id'] == bid]
        assert len(selected) == 3 and {j['split'] for j in selected} <= {'train', 'validation'}
        for source_job in selected:
            job = copy.deepcopy(source_job)
            reuse = job['task'] == 'review_consensus' and bid not in REVIEW_RERUN
            prompt_path = parent_cfg['prompts'][job['task']] if reuse else cfg['prompts'][job['task']]
            job['prompt_sha256'] = file_sha(ROOT / prompt_path)
            assert {k:v for k,v in job.items() if k!='prompt_sha256'} == {k:v for k,v in source_job.items() if k!='prompt_sha256'}
            for image in job['images']:
                assert file_sha(ROOT / image['relative_path']) == image['raw_sha256']
                protected[image['relative_path']] = image['raw_sha256']
            jobs.append(job)
            plan.append({'job_id': job['job_id'], 'prompt_path': prompt_path,
                'mode': 'import_unchanged_D038_review' if reuse else ('deterministic_empty' if job.get('skip_api_empty_captions') else 'new_call_once')})
    selected_businesses = [next(b for b in businesses if b['business_id'] == bid) for bid in SELECTED]
    assert sum(p['mode']=='new_call_once' for p in plan) == 19
    assert Counter(p['mode'] for p in plan) == {'new_call_once':19, 'import_unchanged_D038_review':7, 'deterministic_empty':1}
    immutable(folder / 'pilot_jobs.json', jobs)
    immutable(folder / 'pilot_businesses.json', selected_businesses)
    immutable(folder / 'execution_plan.json', plan)
    input_sha = {p: file_sha(ROOT / p) for p in [CONFIG.relative_to(ROOT).as_posix(), *cfg['prompts'].values(), cfg['schema'],
        'scripts/run_stage_g_d039_focused_pilot.py', 'scripts/run_stage_g_teacher_pilot_v4_1_recovery.py',
        'scripts/stage_g_canonical_review_ids.py', 'scripts/stage_g_visual_target_serialization.py']}
    immutable(folder / 'preflight.json', {'status': 'FOCUSED_PILOT_PREFLIGHT_PASS',
        'businesses':9, 'task_slots':27, 'max_new_API_calls':19, 'review_imports':7, 'empty_tasks':1,
        'generation_model_schema_settings_unchanged':True, 'no_held_out_test':True,
        'jobs_sha256':digest(canonical(jobs)), 'input_sha256':input_sha, 'preserved_original_sha256':protected,
        'spending_authorization':'Existing user-authorized paid bounded Stage G pilot calls plus explicit current focused-rerun instruction; no full production authorization',
        'cost_policy':'No gateway dollar invoice inferred from official model prices; at most19freshcalls, zeroautomaticretries',
        'prompt_refinement_only':True, 'schema_and_validators_unchanged':True})
    print(json.dumps({'preflight':'PASS','businesses':9,'max_new_calls':19,'review_imports':7,'empty_tasks':1}),flush=True)
    return cfg, folder


def run():
    cfg = read(CONFIG); folder = ROOT / cfg['output_directory']; pre = read(folder / 'preflight.json')
    for relative, expected in {**pre['input_sha256'], **pre['preserved_original_sha256']}.items():
        if file_sha(ROOT / relative) != expected: raise ValueError('Protected input changed: ' + relative)
    jobs = read(folder / 'pilot_jobs.json'); plan = {p['job_id']:p for p in read(folder / 'execution_plan.json')}
    assert digest(canonical(jobs)) == pre['jobs_sha256']
    out = folder / 'execution_v1'; base, provider, key = credentials()
    schemas = read(ROOT / cfg['schema'])
    parent_cfg = read(ROOT / 'configs/stage_g_teacher_pilot_v4_1_grounding.json')
    parent_jobs = {j['job_id']:j for j in read(PARENT / 'pilot_jobs.json')}
    immutable(out / 'run_manifest.json', {'status':'D039_BOUNDED_ONLY', 'started_utc':now(),
        'config_sha256':file_sha(CONFIG),'runner_sha256':file_sha(Path(__file__)), 'provider':provider,
        'requested_model':cfg['requested_model'], 'generation_settings':cfg['generation_proposal'],
        'max_new_API_calls':19,'schema_changed':False,'validators_changed':False,
        'production_authorized':False, 'user_labels_changed':False})
    failures=[]; new_calls=0; outputs={}; results=[]
    for job in jobs:
        spec = plan[job['job_id']]; prompt = (ROOT / spec['prompt_path']).read_text(encoding='utf-8'); schema=schemas[job['task']]
        if spec['mode']=='import_unchanged_D038_review':
            original=read(SOURCE / 'checkpoints' / (job['job_id']+'.json'))
            assert original['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
            raw=SOURCE / 'raw_responses' / (job['job_id']+'.json')
            assert file_sha(raw)==original['raw_response_sha256']
            assert json.loads(response_text(read(raw)))==original['output']
            old_prompt=(ROOT / parent_cfg['prompts'][job['task']]).read_text(encoding='utf-8')
            old_body=payload(parent_jobs[job['job_id']],parent_cfg,old_prompt,schema)
            new_body=payload(job,cfg,prompt,schema)
            assert canonical(old_body)==canonical(new_body)
            validate_output(job,original['output'],schema)
            cp=copy.deepcopy(original)
            cp['binding'].update(job_sha256=digest(canonical(job)), config_sha256=file_sha(CONFIG),
                prompt_sha256=digest(prompt.encode('utf-8')), schema_sha256=digest(canonical(schema)))
            cp['imported_completed_response']=True
            cp['import_lineage']={'checkpoint':(SOURCE/'checkpoints'/(job['job_id']+'.json')).relative_to(ROOT).as_posix(),
                'source_checkpoint_sha256':file_sha(SOURCE/'checkpoints'/(job['job_id']+'.json')),
                'identical_request_sha256':digest(canonical(new_body)), 'review_prompt_v6_not_applied':True}
            rp=out/'raw_responses'/raw.name;rp.parent.mkdir(parents=True,exist_ok=True)
            if rp.exists():assert file_sha(rp)==file_sha(raw)
            else:rp.write_bytes(raw.read_bytes())
            immutable(out/'checkpoints'/(job['job_id']+'.json'),cp)
            print('Imported unchanged review: '+job['job_id'],flush=True)
        else:
            path=out/'checkpoints'/(job['job_id']+'.json')
            if path.exists() and read(path)['status']!='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW':
                failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':read(path)});continue
            if spec['mode']=='new_call_once' and not path.exists():
                new_calls+=1
                assert new_calls<=19
            try:
                cp=job_result(job,cfg,prompt,schema,base,provider,key,out,CONFIG)
            except RuntimeError:
                failures.append({'job_id':job['job_id'],'business_id':job['business_id'],'failure':read(path)})
                print('Failed task preserved; no retry: '+job['job_id'],flush=True)
                continue
        results.append(cp)
        outputs.setdefault(job['business_id'],{})[job['condition']]=cp.get('resolved_output',cp['output'])
    completed=[]
    for business in read(folder/'pilot_businesses.json'):
        bid=business['business_id']; content=outputs.get(bid,{})
        if set(content)!={'shared','unfiltered','filtered'}:continue
        consensus=' '.join(c['text'] for c in content['shared']['claims'])
        targets={'reviews_only':'[REVIEW_CONSENSUS]\n'+consensus}; visual={}
        for condition in ['unfiltered','filtered']:
            visual[condition]=visual_claims(content[condition]['assessments'])
            targets[condition]=targets['reviews_only']+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(c['text'] for c in visual[condition])
        immutable(out/'provisional_targets'/(bid+'.json'),{'status':'PROVISIONAL_PILOT_TARGETS_NOT_HUMAN_VERIFIED_NOT_PRODUCTION',
            'business_id':bid,'split':business['split'],'targets':targets,'evidence_records':content,
            'human_verified':False,'review_quote_origin':'code_resolved_from_source_span_IDs',
            'visual_target_claims_with_all_evidence':visual, 'empty_review_consensus_flag':not consensus,
            'review_IDs_text_order_shared':True,'zero_filtered_edge_case':business['zero_filtered_edge_case'],
            'review_component_source':'fresh_v6' if bid in REVIEW_RERUN else 'unchanged_D038_v5_import'})
        completed.append(business)
    report={'status':'BOUNDED_D039_ATTEMPTED_WITH_FAILURES' if failures else 'BOUNDED_D039_STRUCTURAL_PASS_NOT_FACTUAL_CERTIFICATE',
        'businesses_attempted':9,'businesses_completed':len(completed),'target_texts':len(completed)*3,
        'new_API_calls':new_calls,'imported_review_components':7,'empty_tasks':1,'failed_tasks':failures,
        'valid_businesses_by_split':dict(Counter(b['split'] for b in completed)), 'task_slots':27,
        'human_verified_businesses':0,'production_method_frozen':False,'held_out_teacher_targets':0,'LED_training_started':False}
    immutable(out/'completion.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute-pilot',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.execute_pilot:run()
    if not (args.prepare or args.execute_pilot):parser.error('No API calls without explicit --execute-pilot')
