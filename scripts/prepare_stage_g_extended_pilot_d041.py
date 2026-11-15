"""Freeze 100 previously unseen train/validation inputs offline; no API or key access."""
import copy
import math
from collections import Counter, defaultdict
from pathlib import Path
from tokenizers import Tokenizer
from prepare_stage_g_teacher_pilot import ROOT, read, rows, file_sha, digest, canonical, immutable, split_reviews
from audit_stage_g_d039_focused_pilot import stats

CONFIG = ROOT / 'configs/stage_g_teacher_extended_pilot_d041.json'
FOLDER = ROOT / 'outputs/stage_g/teacher_extended_pilot_d041_v1'
PARENT = ROOT / 'configs/stage_g_teacher_pilot_v6_focused_d040.json'
SEED = 42
SPLIT_QUOTAS = {'train': 80, 'validation': 20}
CRITICAL_KEYS = ['requested_model', 'snapshot', 'generation_proposal', 'tasks', 'prompts',
    'schema', 'source_pointer', 'target_length_policy', 'production_retry_policy',
    'production_model_version_policy', 'evidence_representation', 'target_serialization',
    'citation_metadata_policy']
CORE_PATHS = ['scripts/prepare_stage_g_teacher_pilot.py', 'scripts/run_stage_g_teacher_pilot_v4_1_recovery.py',
    'scripts/stage_g_canonical_review_ids.py', 'scripts/stage_g_evidence_id_protocol.py',
    'scripts/prepare_stage_g_review_evidence_ids.py', 'scripts/stage_g_visual_target_serialization.py',
    'scripts/stage_g_teacher_credentials.py', 'schemas/stage_g_teacher_pilot_v1.json']


def rank(tag, bid):
    return digest(f'{SEED}|D041|{tag}|{bid}'.encode('utf-8'))


def exclusions():
    ids = set()
    sources = {}
    for pattern in ['*/pilot_businesses.json', '*/pilot_jobs.json']:
        for path in sorted((ROOT/'outputs/stage_g').glob(pattern)):
            if path.is_relative_to(FOLDER):
                continue
            data = read(path)
            ids.update(x['business_id'] for x in data)
            sources[path.relative_to(ROOT).as_posix()] = file_sha(path)
    for path in sorted((ROOT/'configs').glob('stage_g_teacher_pilot*.json')):
        ids.update(read(path).get('pilot', {}).get('business_ids', []))
        sources[path.relative_to(ROOT).as_posix()] = file_sha(path)
    return ids, sources


def allocate(cells, quota):
    """Minimum one per nonempty cell, then proportional largest-remainder allocation."""
    keys = sorted(cells)
    if quota < len(keys):
        raise ValueError('Quota cannot cover all nonempty strata; review design before changing it')
    allocation = {k: 1 for k in keys}
    capacity = {k: len(cells[k])-1 for k in keys}
    remaining = quota-len(keys)
    total = sum(capacity.values())
    if remaining > total:
        raise ValueError('Insufficient unseen businesses')
    if remaining:
        fractions = {k: remaining*capacity[k]/total for k in keys}
        for k in keys:
            allocation[k] += math.floor(fractions[k])
        residual = quota-sum(allocation.values())
        for k in sorted(keys, key=lambda k: (-(fractions[k]-math.floor(fractions[k])), k))[:residual]:
            allocation[k] += 1
    assert sum(allocation.values()) == quota
    return allocation


def choose(pool):
    chosen = []
    strata = []
    for split, quota in SPLIT_QUOTAS.items():
        eligible = sorted((x for x in pool if x['split']==split), key=lambda x: (x['review_tokens'], x['business_id']))
        for i, item in enumerate(eligible):
            item['review_length_quartile'] = min(4, 1+(4*i)//len(eligible))
        zero = sorted((x for x in eligible if x['filtered_caption_count']==0), key=lambda x: rank('edge', x['business_id']))
        if len(zero) >= quota:
            raise ValueError('Zero-caption edge inclusion exceeds split quota; researcher review required')
        cells = defaultdict(list)
        for x in eligible:
            n=x['filtered_caption_count']
            if n:
                group='1-2' if n<=2 else ('3-4' if n<=4 else '5')
                cells[f"Q{x['review_length_quartile']}|F{group}"].append(x)
        allocation=allocate(cells, quota-len(zero))
        selected=list(zero)
        for cell, count in allocation.items():
            selected.extend(sorted(cells[cell], key=lambda x: rank(f'{split}|{cell}', x['business_id']))[:count])
            strata.append({'split':split,'stratum':cell,'available':len(cells[cell]),'selected':count})
        strata.append({'split':split,'stratum':'all_remaining_zero_filtered','available':len(zero),'selected':len(zero)})
        chosen.extend(selected)
    return sorted(chosen, key=lambda x: rank('execution-order', x['business_id'])), strata


def historical_usage(jobs):
    bank=defaultdict(list)
    observations=[]
    seen=set()
    for folder in ['teacher_pilot_v5_focused_d039_v1','teacher_pilot_v6_focused_d040_v1']:
        directory=ROOT/'outputs/stage_g'/folder
        oldjobs={j['job_id']:j for j in read(directory/'pilot_jobs.json')}
        for path in sorted((directory/'execution_v1/checkpoints').glob('*.json')):
            cp=read(path)
            if not cp.get('api_called') or cp.get('imported_completed_response') or not cp.get('usage'):
                continue
            if cp.get('response_id') in seen:
                continue
            seen.add(cp.get('response_id'))
            task=oldjobs[path.stem]['task']; bank[task].append(cp)
            observations.append({'checkpoint':path.relative_to(ROOT).as_posix(),'sha256':file_sha(path),
                'task':task,'reported_model':cp.get('reported_model'),'usage':cp['usage'],'seconds':cp.get('seconds')})
    expected=defaultdict(float);by_task={}
    for task in ['review_consensus','visual_evidence']:
        count=sum(j['task']==task and not j.get('skip_api_empty_captions') for j in jobs)
        old=bank[task]
        means={k:sum(x['usage'].get(k,0) for x in old)/len(old) for k in ['input_tokens','output_tokens']}
        seconds=sum(x.get('seconds',0) for x in old)/len(old)
        for k in means:expected[k]+=count*means[k]
        expected['seconds']+=count*seconds
        by_task[task]={'planned_calls':count,'prior_observations':len(old),'mean_reported_usage':means,'mean_seconds':seconds}
    calls=sum(not j.get('skip_api_empty_captions') for j in jobs)
    return {'status':'OFFLINE_USAGE_ESTIMATE_NOT_INVOICE_PAID_EXECUTION_NOT_AUTHORIZED',
        'basis':'Fresh D039/D040 provider-reported usage by task; not exact GPT tokenizer/image billing prediction',
        'prior_observations':observations,'by_task':by_task,
        'expected_input_tokens':round(expected['input_tokens']), 'expected_output_tokens':round(expected['output_tokens']),
        'expected_serial_runtime_seconds':round(expected['seconds']),
        'maximum_calls':calls,'max_output_tokens_per_call':8192,'total_output_token_ceiling':calls*8192,
        'input_token_hard_ceiling':None,'cached_input_discount_assumed':False,
        'currency_cost':None,'gateway_price_verified':False,'account_balance_verified':False,
        'cost_formula':'(actual_input_tokens*input_price_per_million + actual_output_tokens*output_price_per_million)/1000000; apply gateway multiplier only if verified',
        'limitations':['D040 fresh review-v7 observations limited to one; D039 review estimates use v6',
            'Unseen source-span counts, photo billing, latency and reasoning vary; averages are estimates, not bounds',
            'No public accessible provider tariff or invoice cost field verified; no model-price guess',
            'Protocol has no automatic retries and only one initial request per nonempty component',
            'A currency spending ceiling needs explicit verified tariff and authorization before execution'],
        'API_calls':0,'new_teacher_targets':0}


def prepare():
    parent=read(PARENT)
    cfg=copy.deepcopy(parent)
    cfg.update(status='D041_FIXED_PROTOCOL_PILOT_APPROVED_SCOPE_PAID_EXECUTION_PENDING',
        decision='D-041',parent_pilot_config=PARENT.relative_to(ROOT).as_posix(),
        output_directory=FOLDER.relative_to(ROOT).as_posix(),
        revision_reason='Extended fixed-D040 evaluation on 100 unseen businesses; no prompt/schema/model optimization')
    cfg['pilot']={'seed':SEED,'businesses_per_split':SPLIT_QUOTAS,'business_count':100,
        'manual_review_count':25,'selection':'split_review_length_quartile_filtered_count_stratification_plus_all_remaining_zero_filtered',
        'exclude_all_prior_Stage_G_pilots':True,'imports_from_previous_targets':False,
        'manual_review_seed_tag':'42|D041|manual-random|business_id',
        'manual_review_selection':'smallest 25 SHA256 ranks independent of teacher outputs',
        'paid_execution_requires_separate_authorization':True}
    for key in CRITICAL_KEYS:
        assert cfg[key]==parent[key],key
    immutable(CONFIG,cfg)
    bindings={}
    def bound(path,expected=None):
        h=file_sha(path)
        if expected is not None and h!=expected:raise ValueError('Frozen input changed: '+path.name)
        bindings[path.relative_to(ROOT).as_posix()]=h
        return path
    pointer=read(bound(ROOT/cfg['source_pointer']))
    assert pointer['K']==5 and pointer['budgets']=={'full_source':5000,'review':4000,'caption':750}
    completion=read(bound(ROOT/pointer['completion_report'],pointer['completion_report_sha256']))
    production=ROOT/pointer['production_directory']
    def stage_f(relative):return bound(production/relative,completion['artifact_sha256'][relative])
    reviews={x['business_id']:x for x in rows(stage_f('manifests/shared_review_inputs.jsonl')) if x['split'] in SPLIT_QUOTAS}
    assert len(reviews)==2200
    caps={};sources={}
    for split,n in [('train',2000),('validation',200)]:
        for cond in ['unfiltered','filtered']:
            records=list(rows(stage_f(f'manifests/{split}/{cond}_captions.jsonl')))
            assert len(records)==n and all(x['split']==split for x in records)
            caps[split,cond]={x['business_id']:x for x in records}
        for cond in ['reviews_only','unfiltered','filtered']:
            records=list(rows(stage_f(f'manifests/{split}/{cond}_sources.jsonl')))
            assert len(records)==n and all(x['split']==split for x in records)
            sources[split,cond]={x['business_id']:x for x in records}
    manifest=read(bound(ROOT/'assets/tokenizers/led-base-16384.manifest.json'))
    token_dir=ROOT/manifest['relative_directory']
    for name,h in manifest['sha256'].items():bound(token_dir/name,h)
    tokenizer=Tokenizer.from_file(str(token_dir/'tokenizer.json'));tokenizer.no_padding();tokenizer.no_truncation()
    exclude,exclude_sources=exclusions()
    for rel,h in exclude_sources.items():bound(ROOT/rel,h)
    pool=[]
    for bid,x in sorted(reviews.items()):
        if bid in exclude:continue
        split=x['split'];tokens=len(tokenizer.encode(x['review_block'],add_special_tokens=False).ids)
        assert tokens<=4000
        pool.append({'business_id':bid,'split':split,'review_tokens':tokens,
            'selected_review_count':len(x['selected_review_ids']),
            'unfiltered_caption_count':caps[split,'unfiltered'][bid]['retained_caption_count'],
            'filtered_caption_count':caps[split,'filtered'][bid]['retained_caption_count'],
            'zero_filtered_edge_case':caps[split,'filtered'][bid]['retained_caption_count']==0})
    immutable(FOLDER/'sampling_candidate_features.json',pool)
    chosen,strata=choose(pool)
    assert len(chosen)==len({x['business_id'] for x in chosen})==100
    assert not set(x['business_id'] for x in chosen)&exclude
    manual=sorted(chosen,key=lambda x:rank('manual-random',x['business_id']))[:25]
    manual_ids={x['business_id'] for x in manual}
    inventory={x['photo_id']:x for x in rows(stage_f('manifests/complete_caption_pool_inventory.jsonl')) if x['split'] in SPLIT_QUOTAS}
    schemas=read(bound(ROOT/cfg['schema']))
    prompts={k:bound(ROOT/v).read_text(encoding='utf-8') for k,v in cfg['prompts'].items()}
    jobs=[];image_hashes={};source_witness=[]
    for item in chosen:
        bid,split=item['business_id'],item['split'];shared=reviews[bid]
        item['manual_review_preselected']=bid in manual_ids
        item['selected_review_ids']=shared['selected_review_ids'];item['review_block_sha256']=shared['review_block_sha256']
        item['review_text_sha256_in_order']=shared['review_text_sha256_in_order']
        item['source_sha256']={};item['caption_condition_sha256']={}
        extracted=split_reviews(shared)
        for cond in ['reviews_only','unfiltered','filtered']:
            source=sources[split,cond][bid]
            assert source['selected_review_ids']==shared['selected_review_ids'] and source['review_block_sha256']==shared['review_block_sha256']
            assert digest(source['source'].encode('utf-8'))==source['source_sha256']
            ids=tokenizer.encode(source['source'],add_special_tokens=True).ids
            assert len(ids)==source['tokens']<=5000
            item['source_sha256'][cond]=source['source_sha256']
            source_witness.append(source)
        jobs.append({'job_id':f'{split}_{bid}_review_consensus','business_id':bid,'split':split,
            'task':'review_consensus','condition':'shared','evidence':{'business_id':bid,'reviews':extracted,'exact_review_block':shared['review_block']},
            'review_block_sha256':shared['review_block_sha256'],'images':[],
            'prompt_sha256':digest(prompts['review_consensus'].encode()),'schema_sha256':digest(canonical(schemas['review_consensus']))})
        for cond in ['unfiltered','filtered']:
            record=caps[split,cond][bid]
            assert record['selected_photo_ids']==[x['photo_id'] for x in record['captions']]
            assert 0<=len(record['captions'])<=5
            item['caption_condition_sha256'][cond]=digest(canonical(record))
            cc=[];images=[]
            for i,cap in enumerate(record['captions'],1):
                pid=cap['photo_id'];inv=inventory[pid]
                assert (inv['business_id'],inv['split'],inv['normalized_caption'])==(bid,split,cap['normalized_caption'])
                checkpoint=read(bound(ROOT/inv['caption_record_path'],inv['caption_record_sha256']))
                assert (checkpoint['business_id'],checkpoint['split'],checkpoint['photo_id'],checkpoint['normalized_caption'])==(bid,split,pid,cap['normalized_caption'])
                path=(ROOT/checkpoint['image_path']).resolve();assert path.is_relative_to(ROOT)
                bound(path,checkpoint['raw_sha256']);image_hashes[pid]=checkpoint['raw_sha256']
                cc.append({'caption_id':f'IMG_CAP_{i}','photo_id':pid,'photo_marker':f'PHOTO_{i}','caption':cap['normalized_caption']})
                images.append({'photo_id':pid,'photo_marker':f'PHOTO_{i}','relative_path':path.relative_to(ROOT).as_posix(),'raw_sha256':checkpoint['raw_sha256']})
            jobs.append({'job_id':f'{split}_{bid}_{cond}_visual_evidence','business_id':bid,'split':split,'task':'visual_evidence','condition':cond,
                'evidence':{'business_id':bid,'condition':cond,'captions':cc},'images':images,'review_block_sha256':shared['review_block_sha256'],
                'skip_api_empty_captions':not cc,'prompt_sha256':digest(prompts['visual_evidence'].encode()),'schema_sha256':digest(canonical(schemas['visual_evidence']))})
    assert len(jobs)==300
    for p in [CONFIG,PARENT,Path(__file__),*[ROOT/x for x in CORE_PATHS]]:bound(p)
    for p in (ROOT/'scripts').glob('*stage_g*extended*d041*.py'):bound(p)
    historical={}
    for rel in read(ROOT/'outputs/stage_g/teacher_pilot_v6_focused_d040_v1/preflight.json')['preserved_original_sha256']:
        historical[rel]=file_sha(ROOT/rel)
    for d in (ROOT/'outputs/stage_g').glob('*'):
        if d==FOLDER:continue
        for p in ([d] if d.is_file() else d.rglob('*')):
            if p.is_file():historical[p.relative_to(ROOT).as_posix()]=file_sha(p)
    snapshot=FOLDER/'protocol_snapshot'
    snapshot_files={}
    for relative in [PARENT.relative_to(ROOT).as_posix(),*cfg['prompts'].values(),cfg['schema'],*CORE_PATHS]:
        path=snapshot/relative;path.parent.mkdir(parents=True,exist_ok=True);data=(ROOT/relative).read_bytes()
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        snapshot_files[relative]=digest(data)
    immutable(snapshot/'manifest.json',snapshot_files)
    immutable(FOLDER/'pilot_businesses.json',chosen)
    immutable(FOLDER/'manual_review_25_manifest.json',{'seed':SEED,'method':'SHA256 rank 42|D041|manual-random|business_id; choose first25 before any generation',
        'businesses':[{'business_id':x['business_id'],'split':x['split']} for x in manual],
        'selection_independent_of_teacher_outputs':True,'additional_flagged_cases_kept_separate':True,
        'human_verified':False})
    immutable(FOLDER/'pilot_jobs.json',jobs)
    immutable(FOLDER/'frozen_source_witness.json',source_witness)
    sampling={'seed':SEED,'unseen_candidate_count':len(pool),'excluded_prior_business_ids':sorted(exclude),'exclusion_source_sha256':exclude_sources,
        'split_quotas':SPLIT_QUOTAS,'strata':strata,'rank_function':'sha256 UTF8 42|D041|<tag>|business_id',
        'quartiles':'Within-split unseen pool sorted by (exact LED review-block tokens, business_id); Q=min(4,1+floor(4*rank/N))',
        'allocation':'All remaining zero-filtered included first; nonzero split/quartile/F1-2,F3-4,F5 cells minimum1 then proportional largest remainder',
        'deliberate_oversampling':['Validation 20% versus original200/2200 (9.09%)',
            'All remaining zero-Filtered businesses included; not a purely representative random sample',
            'Minimum1 per nonempty stratum ensures length/count coverage and can oversample small cells'],
        'selection_features_used':['split','review_token_length','filtered_caption_count','zero_filtered_visual_availability'],
        'unfiltered_caption_count_monitored_not_stratified':True,'teacher_outputs_used_for_selection':False,
        'cohort_stats':{'review_tokens':stats([x['review_tokens'] for x in chosen]),
            'filtered_caption_counts':dict(Counter(x['filtered_caption_count'] for x in chosen)),
            'unfiltered_caption_counts':dict(Counter(x['unfiltered_caption_count'] for x in chosen))},
        'manual25_business_ids':[x['business_id'] for x in manual],'API_calls':0}
    immutable(FOLDER/'sampling_manifest.json',sampling)
    estimate=historical_usage(jobs);immutable(FOLDER/'pilot_usage_estimate.json',estimate)
    frozen_files=['pilot_businesses.json','manual_review_25_manifest.json','pilot_jobs.json','frozen_source_witness.json','sampling_manifest.json','sampling_candidate_features.json','pilot_usage_estimate.json']
    pre={'status':'D041_OFFLINE_PREPARATION_PASS_AWAITING_PRICE_AND_PAID_AUTHORIZATION',
        'businesses':100,'businesses_by_split':dict(Counter(x['split'] for x in chosen)),
        'manual_review_count':25,'task_slots':300,'max_new_API_calls':estimate['maximum_calls'],
        'empty_tasks':sum(j.get('skip_api_empty_captions',False) for j in jobs),
        'unique_original_photos_verified':len(image_hashes),'selected_reviews':sum(x['selected_review_count'] for x in chosen),
        'excluded_prior_business_count':len(exclude),'no_overlap_prior_or_held_out':True,
        'jobs_sha256':digest(canonical(jobs)),'input_sha256':bindings,'preserved_original_sha256':historical,
        'frozen_output_sha256':{name:file_sha(FOLDER/name) for name in frozen_files},
        'protocol_snapshot_sha256':snapshot_files,'D040_critical_fields_unchanged':CRITICAL_KEYS,
        'API_calls':0,'new_teacher_targets':0,'human_annotations_created':0,'paid_execution_authorized':False,
        'full_generation_authorized':False,'model_snapshot_attested':False,
        'remaining_blockers':['Provider tariff/account usage policy and currency spending ceiling unverified',
            'Explicit paid authorization required under user request section8 before any API call'],
        'target_512_tokens_diagnostic_only_not_frozen_cap':True}
    immutable(FOLDER/'preflight.json',pre)
    for rel,h in {**bindings,**historical}.items():assert file_sha(ROOT/rel)==h
    print({k:pre[k] for k in ['status','businesses_by_split','max_new_API_calls','empty_tasks','unique_original_photos_verified','selected_reviews','API_calls']})
    return pre


if __name__=='__main__':prepare()
