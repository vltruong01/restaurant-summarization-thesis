"""Offline, snapshot-based random quality review. Never calls API or production helpers that write."""
import argparse
import copy
import html
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, digest, canonical
from prepare_stage_g_review_evidence_ids import evidence_spans
import stage_g_production_d042 as frozen
import stage_g_business_id_guard_v1 as guard

BASE = ROOT / 'outputs/stage_g/teacher_production_d042_v1'
LIVE = BASE / 'guarded_production6_retry_v1'
FOLDER = ROOT / 'outputs/stage_g/production_quality_audit100_v1'
BUNDLE = FOLDER / 'stage_g_production_quality_audit_bundle'
ZIP = FOLDER / 'chatgpt_stage_g_production_quality_audit100.zip'
SEED = 2026101202
SIZE = 100
PASS = 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def draw(pool, seed=SEED):
    """Simple random sample without replacement; every eligible business equally likely."""
    assert len(pool) >= SIZE
    assert len({b['business_id'] for b in pool}) == len(pool)
    ordered = sorted(pool, key=lambda b: b['business_id'])
    selected = random.Random(seed).sample(ordered, SIZE)
    return selected, dict(Counter(b['split'] for b in selected))


def exclusions():
    excluded = defaultdict(list)
    sources = {}
    first100 = ROOT / 'outputs/stage_g/teacher_extended_pilot_d041_v1/pilot_businesses.json'
    previous50 = ROOT / 'outputs/stage_g/production_quality_audit50_v1/sampling_manifest.json'
    initial = read(first100)
    previous = read(previous50)['businesses']
    assert len(initial) == len({b['business_id'] for b in initial}) == 100
    assert len(previous) == len({b['business_id'] for b in previous}) == 50
    for path, businesses in ((first100, initial), (previous50, previous)):
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = file_sha(path)
        for business in businesses:
            excluded[business['business_id']].append(relative)
    return dict(excluded), sources


def checkpoint_sources(jobs, selected_ids):
    reuse = read(LIVE / 'reuse_index.json')
    candidates = defaultdict(list)
    for directory in sorted((LIVE / 'execution_v1/attempts').glob('attempt_*')):
        for path in (directory / 'checkpoints').glob('*.json'):
            candidates[path.stem].append(path)
    result = {}
    for job in jobs:
        if job['business_id'] not in selected_ids:
            continue
        jid = job['job_id']
        if jid in reuse:
            entry = reuse[jid]
            path = ROOT / entry['directory'] / 'checkpoints' / (jid + '.json')
            config_path = ROOT / entry['config_path']
        else:
            matches = [p for p in candidates[jid] if read(p)['status'] == PASS]
            assert len(matches) == 1, 'Selected completed business has missing/duplicate accepted checkpoint: ' + jid
            path = matches[0]
            config_path = ROOT / 'configs/stage_g_production6_retry_v1.json'
        cp = read(path)
        assert cp['status'] == PASS
        result[jid] = {'checkpoint_path': path.relative_to(ROOT).as_posix(), 'checkpoint_sha256': file_sha(path),
                       'config_path': config_path.relative_to(ROOT).as_posix(), 'config_sha256': file_sha(config_path)}
    assert len(result) == SIZE * 3
    return result


def prepare():
    assert not (FOLDER / 'sampling_manifest.json').exists(), 'Frozen sample already exists; use --export'
    started = now()
    pointer = read(ROOT / 'outputs/stage_g/active_production_execution_pointer.json')
    state = read(ROOT / pointer['progress_path'])
    targets = read(LIVE / 'reuse_targets.json')
    listed = sorted((LIVE / 'execution_v1/targets').glob('*.json'))
    for path in listed:
        assert path.stem not in targets
        targets[path.stem] = path.relative_to(ROOT).as_posix()
    excluded, exclusion_sources = exclusions()
    pool = []
    known = {b['business_id']: b for b in read(BASE / 'production_businesses.json')}
    for bid, rel in sorted(targets.items()):
        if bid in excluded:
            continue
        business = known[bid]
        assert business['split'] in ('train', 'validation')
        target = read(ROOT / rel)
        assert target['business_id'] == bid and target['split'] == business['split']
        assert set(target['targets']) == {'reviews_only', 'unfiltered', 'filtered'}
        pool.append({'business_id': bid, 'split': business['split'], 'target_source': rel, 'target_sha256': file_sha(ROOT / rel)})
    selected, allocation = draw(pool)
    assert not {b['business_id'] for b in selected} & set(excluded)
    jobs = read(BASE / 'production_jobs.json')
    bindings = checkpoint_sources(jobs, {b['business_id'] for b in selected})
    manifest = {'schema_version': 'stage_g_production_quality_audit100_v1', 'export_only': True, 'seed': SEED,
                'selection_method': 'simple random sampling without replacement from the complete eligible snapshot; every business has equal inclusion probability; no split quotas or quality stratification',
                'selection_not_based_on_target_content_scores_warnings_or_quality': True,
                'snapshot_started_utc': started, 'snapshot_completed_utc': now(), 'completed_target_files_at_listing': len(targets),
                'production_status_snapshot': state, 'eligible_pool_size': len(pool),
                'eligible_pool_split_counts': dict(Counter(b['split'] for b in pool)),
                'selected_split_counts': allocation, 'businesses': selected, 'excluded_business_ids_with_sources': excluded,
                'exclusion_source_sha256': exclusion_sources, 'checkpoint_bindings': bindings,
                'production_jobs_sha256': file_sha(BASE / 'production_jobs.json'),
                'API_calls': 0, 'production_lock_acquired': False, 'production_interrupted': False,
                'no_resampling_after_quality_inspection': True,
                'excluded_D041_businesses': 100, 'excluded_previous_audit50_businesses': 50,
                'uniform_inclusion_probability_per_eligible_business': SIZE / len(pool),
                'exporter_script_sha256': file_sha(Path(__file__)),
                'previous_audit50_zip_sha256': file_sha(ROOT / 'outputs/stage_g/production_quality_audit50_v1/chatgpt_stage_g_production_quality_audit50.zip')}
    save(FOLDER / 'candidate_pool.json', pool)
    save(FOLDER / 'sampling_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('eligible_pool_size', 'eligible_pool_split_counts', 'selected_split_counts', 'seed')}), flush=True)


RUBRIC = '''# Independent Stage G production quality audit

Review the100 sampled businesses without treating teacher judgments as human truth.
Actually read selected reviews and open mapped original photos. If an item cannot
be opened, mark NOT REVIEWED. Report exact business/condition/claim/REV/IMG_CAP/PHOTO
IDs and quotations. Do not rewrite outputs, adjust settings or fabricate negatives.

REVIEW_CONSENSUS: only selected reviews, never images/captions/external knowledge.
Check entailment, accurate food/object categories, prevalence/frequency wording,
positive and negative opinions, minority contrasting views, and meaningful omissions.
Valid IDs and exact quotes establish mapping, not semantic entailment.

VISUAL_EVIDENCE: a claim must follow the exact normalized caption supplied to LED
AND its corresponding original photo. Raw BLIP-2 captions are also preserved.
Do not substitute photo-only facts absent from captions. Ordinary plausible visual
descriptions and correct simple content are acceptable; low detail is not error.
Check unsupported identities/ingredients/counts/actions/relations without requiring
unnecessarily precise taxonomy. Use uncertain for genuine ambiguity. Review both
verified claims included in targets and rejected/uncertain claims excluded from them
for excessive strictness. Identical claim-caption-photo evidence with contradictory
verdicts is a review flag, not automatic proof of factual error. Do not force agreement.

Zero-Filtered-caption businesses retain the shared review consensus and exact empty
visual header. No fabricated or substitute evidence. Shared reviews stay identical.

Separate factual errors, unsupported claims, uncertainty, correct-but-simple content,
style issues, over-strict judgments and lost contrast. Automated integrity PASS is
not independent factual validation. ChatGPT is an optional second opinion; only the
researcher's explicit confirmation is a human final decision. This export creates
no human labels and does not modify scientific outputs. Additional technical flags
are stored separately and never influence the frozen random100 selection.

The sample covers only completed train/validation production at the recorded snapshot,
not the entire2200 cohort or held-out test. Sampling is uniform without replacement across the eligible pool; split counts and
inclusion probability are reported, and unavailable splits are not represented. Do not claim this
sample demonstrates quality of unsampled, incomplete or held-out businesses.
'''


def render(cases):
    esc = lambda v: html.escape(str(v))
    sections = []
    for case in cases:
        bid = case['business_id']
        parts = [f'<article id="b-{esc(bid)}"><h2>{esc(case["metadata"].get("name",bid))}</h2><p>{esc(bid)} · {esc(case["split"])}</p>']
        for condition, text in case['final_targets'].items():
            parts.append(f'<details><summary>Target: {esc(condition)}</summary><pre>{esc(text)}</pre></details>')
        parts.append('<h3>Review consensus evidence</h3>')
        for claim in case['teacher_evidence_records']['shared']['claims']:
            parts.append('<details><summary>'+esc(claim['text'])+'</summary><pre>'+esc(json.dumps(claim['support'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<h3>Complete selected reviews</h3>')
        for review in case['reviews']:
            parts.append('<details><summary>'+esc(review['review_marker']+' / '+review['review_id'])+' · '+esc(review['stars'])+'★</summary><pre>'+esc(review['text'])+'</pre></details>')
        for condition in ('unfiltered', 'filtered'):
            parts.append('<h3>'+esc(condition)+' captions/photos/teacher verdicts</h3>')
            component = case['components'][condition]
            if not component['captions']:
                parts.append('<p>Zero captions: empty visual evidence, unchanged.</p>')
            for cap in component['captions']:
                parts.append('<section class="photo"><a href="'+esc(cap['image_file'])+'" target="_blank"><img src="'+esc(cap['image_file'])+'" alt="'+esc(cap['photo_id'])+'"></a><div><code>'+esc(cap['caption_id']+' / '+cap['photo_marker']+' / '+cap['photo_id'])+'</code><p>LED caption: '+esc(cap['caption'])+'</p><details><summary>Raw BLIP-2 caption</summary><pre>'+esc(cap['raw_caption'])+'</pre></details>')
                assessment = next(a for a in component['teacher_output']['assessments'] if a['caption_id'] == cap['caption_id'])
                for claim in assessment['claims']:
                    parts.append('<div class="claim"><b>'+esc(claim['verdict'])+'</b><p>'+esc(claim['text'])+'</p><p>Caption quote: '+esc(claim['caption_quote'])+'</p><small>'+esc(claim['explanation'])+'</small></div>')
                parts.append('</div></section>')
        parts.append('<details><summary>Validation warnings</summary><pre>'+esc(json.dumps(case['validation_warnings'],ensure_ascii=False,indent=2))+'</pre></details></article>')
        sections.append(''.join(parts))
    nav = ''.join('<a href="#b-'+esc(c['business_id'])+'">'+esc(c['business_id'])+'</a> ' for c in cases)
    return '<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage G production quality audit100</title><style>body{font:16px system-ui;line-height:1.5;background:#f5f6f8;color:#182533}main{max-width:1150px;margin:auto;padding:24px}article{background:white;padding:24px;margin:22px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px system-ui}details{padding:8px;border-bottom:1px solid #ddd}summary{cursor:pointer}.photo{display:grid;grid-template-columns:40% 1fr;gap:20px;margin:24px 0}.photo img{width:100%;max-height:550px;object-fit:contain}.claim{background:#f1f5fa;padding:12px;margin:10px 0}nav{font-size:12px;overflow-wrap:anywhere}@media(max-width:750px){.photo{grid-template-columns:1fr}}</style><main><h1>Stage G production quality audit · 100 businesses</h1><p>Existing API outputs only. Teacher verdicts are not human labels. Read SUMMARY.md and rubric.md. Click images for unchanged originals.</p><nav>'+nav+'</nav>'+''.join(sections)+'</main></html>'


def export():
    manifest = read(FOLDER / 'sampling_manifest.json')
    selected = manifest['businesses']
    pool = read(FOLDER / 'candidate_pool.json')
    assert draw(pool)[0] == selected
    assert len(selected) == len({b['business_id'] for b in selected}) == SIZE
    assert file_sha(Path(__file__)) == manifest['exporter_script_sha256']
    for relative, sha in manifest['exclusion_source_sha256'].items():
        assert file_sha(ROOT / relative) == sha, 'Exclusion manifest changed: ' + relative
    assert file_sha(BASE / 'production_jobs.json') == manifest['production_jobs_sha256']
    assert not ZIP.exists(), 'Preserve existing completed ZIP'
    jobs = read(BASE / 'production_jobs.json')
    grouped = defaultdict(dict)
    for job in jobs:
        grouped[job['business_id']][job['condition']] = job
    photos_needed = {c['photo_id'] for b in selected for j in grouped[b['business_id']].values()
                     for c in j['evidence'].get('captions', [])}
    stage_f = ROOT / read(ROOT / 'outputs/stage_f/production_manifest_pointer.json')['production_directory']
    inventory = {}
    with (stage_f / 'manifests/complete_caption_pool_inventory.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['photo_id'] in photos_needed:
                inventory[row['photo_id']] = row
    assert set(inventory) == photos_needed
    metadata = {}
    with (stage_f / 'manifests/shared_review_inputs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['business_id'] in {b['business_id'] for b in selected}:
                metadata[row['business_id']] = row
    protected = {}
    exported = {}
    def original(path):
        data = path.read_bytes()
        protected[path.relative_to(ROOT).as_posix()] = digest(data)
        return data
    def put(name, data):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        if name in exported:
            assert exported[name] == digest(data)
            return
        path = BUNDLE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, 'Partial export differs; preserve and investigate: ' + name
        else:
            with path.open('xb') as stream:
                stream.write(data)
        exported[name] = digest(data)
    def putjson(name, value):
        put(name, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
    cases = []
    claims = []
    flags = []
    counts = Counter()
    for order, sampled in enumerate(selected, 1):
        bid = sampled['business_id']
        assert sampled['split'] in ('train','validation') and bid not in manifest['excluded_business_ids_with_sources']
        target_bytes = original(ROOT / sampled['target_source'])
        assert digest(target_bytes) == sampled['target_sha256']
        target = json.loads(target_bytes)
        put('artifacts/student_targets/'+bid+'.json', target_bytes)
        js = grouped[bid]
        review_job = js['shared']
        assert metadata[bid]['review_block'] == review_job['evidence']['exact_review_block']
        case = {'case_id': bid, 'sample_order': order, 'business_id': bid, 'split': sampled['split'],
                'metadata': metadata[bid]['metadata'], 'status': 'ACCEPTED_PRODUCTION_OUTPUTS_NOT_INDEPENDENTLY_HUMAN_VERIFIED',
                'human_verified': target.get('human_verified',False), 'reviews': copy.deepcopy(review_job['evidence']['reviews']),
                'exact_selected_review_block': review_job['evidence']['exact_review_block'],
                'final_targets': copy.deepcopy(target['targets']), 'teacher_evidence_records': copy.deepcopy(target['evidence_records']),
                'shared_REVIEW_CONSENSUS': target['targets']['reviews_only'].removeprefix('[REVIEW_CONSENSUS]\n'),
                'VISUAL_EVIDENCE': {c: target['targets'][c].split('\n\n[VISUAL_EVIDENCE]\n',1)[1] for c in ('unfiltered','filtered')},
                'visual_target_claims_with_all_evidence': copy.deepcopy(target['visual_target_claims_with_all_evidence']),
                'original_target_file': 'artifacts/student_targets/'+bid+'.json', 'components': {}, 'validation_warnings': []}
        for review in case['reviews']:
            review['source_spans'] = evidence_spans(review)
        cps = {}
        visual_rows = {}
        for condition, job in js.items():
            binding = manifest['checkpoint_bindings'][job['job_id']]
            path = ROOT / binding['checkpoint_path']
            assert file_sha(path) == binding['checkpoint_sha256']
            cfg_path = ROOT / binding['config_path']
            assert file_sha(cfg_path) == binding['config_sha256']
            cfg = read(cfg_path)
            cp = read(path)
            schema = read(ROOT / cfg['schema'])[job['task']]
            if cp['binding']['schema_sha256'] == digest(canonical(guard.schema_for_job(job, schema))):
                guard.verify_checkpoint(job,cfg,path.parent.parent,cfg_path)
            else:
                frozen.verify_checkpoint(job,cfg,path.parent.parent,cfg_path)
            assert cp.get('resolved_output',cp['output']) == target['evidence_records'][condition]
            cps[condition] = cp
            artifact = 'artifacts/checkpoints/'+job['job_id']+'.json'
            put(artifact, original(path))
            put('protocol/configs/'+cfg_path.name, original(cfg_path))
            component = {'job_id': job['job_id'], 'captions': [], 'teacher_output': copy.deepcopy(cp['output']),
                         'resolved_output': copy.deepcopy(cp.get('resolved_output',cp['output'])),
                         'validation': copy.deepcopy(cp['validation']), 'checkpoint_file': artifact,
                         'response_id': cp.get('response_id'), 'reported_model': cp.get('reported_model'),
                         'usage': cp.get('usage'), 'api_called': cp.get('api_called'), 'raw_response_file': None,
                         'evidence_resolution': cp.get('evidence_resolution',[]),
                         'citation_metadata_diagnostics': cp.get('citation_metadata_diagnostics',[])}
            raw = path.parent.parent / 'raw_responses' / path.name
            if cp.get('api_called'):
                raw_bytes = original(raw)
                assert digest(raw_bytes) == cp['raw_response_sha256']
                raw_name = 'raw_api_responses/'+raw.name
                put(raw_name, raw_bytes)
                component['raw_response_file'] = raw_name
            for warning in cp.get('citation_metadata_diagnostics',[]):
                case['validation_warnings'].append({'type': 'teacher_reported_auxiliary_review_ID_mismatch',
                                                     'condition': condition, 'details': warning})
            for cap, photo in zip(job['evidence'].get('captions',[]),job['images']):
                assert cap['photo_id'] == photo['photo_id']
                inv = inventory[cap['photo_id']]
                assert inv['business_id'] == bid and inv['split'] == sampled['split']
                assert inv['normalized_caption'] == cap['caption'] and inv['basic_eligible']
                data = original(ROOT / photo['relative_path'])
                assert digest(data) == photo['raw_sha256']
                image_file = 'images/'+cap['photo_id']+'.jpg'
                put(image_file, data)
                exported_cap = dict(cap, image_file=image_file, original_image_sha256=photo['raw_sha256'],
                                    raw_caption=inv['raw_caption'], normalized_caption=inv['normalized_caption'],
                                    BLIP2_provenance={k:inv[k] for k in ('model_id','resolved_commit','generation_settings','chosen_attempt','retry_attempts','basic_quality_policy_sha256')})
                component['captions'].append(exported_cap)
                counts['caption_occurrences'] += 1
            case['components'][condition] = component
        assert frozen.build_target({'business_id':bid,'split':sampled['split'],
                     'zero_filtered_edge_case':target['zero_filtered_edge_case']},
                     {c:cp.get('resolved_output',cp['output']) for c,cp in cps.items()}) == target
        review_map = {r['review_marker']:r for r in case['reviews']}
        review_claim_ids = []
        for ci, claim in enumerate(cps['shared']['resolved_output']['claims']):
            cid = bid+'/shared/claim_'+str(ci+1)
            review_claim_ids.append(cid)
            mappings = []
            for resolution in cps['shared']['evidence_resolution']:
                if resolution['claim_index'] != ci:
                    continue
                r = review_map[resolution['review_marker']]
                quote = r['text'][resolution['start_character']:resolution['end_character']]
                assert r['review_id'] == resolution['review_id'] and digest(quote.encode()) == resolution['quote_sha256']
                mappings.append(dict(resolution, exact_quote=quote))
            assert mappings
            claims.append({'claim_id':cid,'business_id':bid,'condition':'shared','kind':'review_consensus',
                           'claim_index':ci,'text':claim['text'],'polarity':claim['polarity'],
                           'included_in_target_conditions':['reviews_only','unfiltered','filtered'],
                           'review_evidence':mappings,'caption_evidence':[],'photo_evidence':[],
                           'case_file':'businesses/'+bid+'.json','human_verified':False})
        match = defaultdict(list)
        for condition in ('unfiltered','filtered'):
            cap_map = {c['caption_id']:c for c in case['components'][condition]['captions']}
            for assessment in cps[condition]['output']['assessments']:
                cap = cap_map[assessment['caption_id']]
                assert cap['photo_id'] == assessment['photo_id']
                for ci, claim in enumerate(assessment['claims']):
                    cid = bid+'/'+condition+'/'+cap['caption_id']+'/claim_'+str(ci+1)
                    row = {'claim_id':cid,'business_id':bid,'condition':condition,'kind':'visual_evidence',
                           'claim_index_within_caption':ci,'text':claim['text'],'teacher_verdict':claim['verdict'],
                           'teacher_explanation':claim['explanation'],'caption_id':cap['caption_id'],
                           'caption_quote':claim['caption_quote'],'normalized_caption':cap['caption'],
                           'raw_caption':cap['raw_caption'],'photo_id':cap['photo_id'],'photo_marker':cap['photo_marker'],
                           'image_file':cap['image_file'],'image_sha256':cap['original_image_sha256'],
                           'included_in_target':claim['verdict']=='verified','review_evidence':[],
                           'case_file':'businesses/'+bid+'.json','human_verified':False}
                    claims.append(row)
                    visual_rows[(condition,cap['caption_id'],ci)] = cid
                    key = (cap['photo_id'],cap['caption'],' '.join(claim['text'].lower().split()))
                    match[key].append(row)
                    counts['visual_verdict_'+claim['verdict']] += 1
            if not cap_map:
                assert case['VISUAL_EVIDENCE'][condition] == ''
                counts['zero_'+condition] += 1
        for key, rows in match.items():
            if len({r['condition'] for r in rows}) > 1 and len({r['teacher_verdict'] for r in rows}) > 1:
                warning = {'type':'exact_evidence_cross_condition_verdict_conflict','business_id':bid,
                           'photo_id':key[0],'normalized_caption':key[1],
                           'claims':[{'claim_id':r['claim_id'],'condition':r['condition'],'text':r['text'],'verdict':r['teacher_verdict']} for r in rows],
                           'independent_factual_error_confirmed':False}
                flags.append(warning)
                case['validation_warnings'].append(warning)
        case['target_claim_index'] = {}
        for condition in ('reviews_only','unfiltered','filtered'):
            visual = []
            for target_claim in target['visual_target_claims_with_all_evidence'].get(condition,[]):
                ids = [visual_rows[(condition,s['caption_id'],s['raw_claim_index'])] for s in target_claim['support']]
                visual.append({'text':target_claim['text'],'supporting_claim_ids':ids,'support':target_claim['support']})
            case['target_claim_index'][condition] = {'review_claim_ids':review_claim_ids,'visual_claims':visual}
        counts['selected_reviews'] += len(case['reviews'])
        counts['review_claims'] += len(review_claim_ids)
        cases.append(case)
        putjson('businesses/'+bid+'.json',case)
        print(json.dumps({'exported_businesses':len(cases),'planned':SIZE}),flush=True)
    putjson('cases.json',{'export_only':True,'schema_version':'stage_g_production_quality_audit100_v1','businesses':cases})
    putjson('claim_evidence_index.json',{'schema_version':'stage_g_production_claim_index_v1','claims':claims,
                    'condition_target_index':{c['business_id']:c['target_claim_index'] for c in cases}})
    putjson('additional_suspicious_cases.json',{'selection_unchanged':True,'additional_businesses_added':0,
                    'scope':'separate automated technical flags for sampled cases; not human factual labels',
                    'flags':flags})
    put('sampling_manifest.json',original(FOLDER/'sampling_manifest.json'))
    put('candidate_pool.json',original(FOLDER/'candidate_pool.json'))
    for rel in ['prompts/stage_g_review_consensus_v7.txt','prompts/stage_g_visual_evidence_v4.txt',
                'schemas/stage_g_teacher_pilot_v3.json','specs/007_teacher_target_construction.md']:
        put('protocol/'+Path(rel).name,original(ROOT/rel))
    summary = {'status':'OFFLINE_EXPORT_INTEGRITY_PASS_NOT_FACTUAL_CERTIFICATION','sample_businesses':SIZE,
               'final_student_targets':SIZE*3,'seed':SEED,'split_counts':manifest['selected_split_counts'],
               'eligible_pool_split_counts':manifest['eligible_pool_split_counts'],'eligible_pool_size':manifest['eligible_pool_size'],
               'completed_target_files_at_snapshot':manifest['completed_target_files_at_listing'],
               'original_unique_photos':len(photos_needed),'counts':dict(counts),
               'technical_conflict_groups':len(flags),'businesses_with_warnings':sum(bool(c['validation_warnings']) for c in cases),
               'review_ID_warning_entries':sum(w['type']=='teacher_reported_auxiliary_review_ID_mismatch' for c in cases for w in c['validation_warnings']),
               'D041_overlap':0,'previous_audit50_overlap':0,'held_out_businesses':0,'API_calls':0,'BLIP2_calls':0,
               'existing_targets_or_labels_changed':False,'production_interrupted':False,
               'image_resize_or_reencoding':False,'teacher_verdicts_not_human_ground_truth':True,
               'uniform_inclusion_probability_per_eligible_business':SIZE/manifest['eligible_pool_size'],
               'sample_limitations':['snapshot of completed businesses only; generation order can bias the available pool',
                                     ('no eligible validation businesses remain after excluding D041 and the previous audit50 sample; this sample cannot assess validation quality'
                                      if not manifest['eligible_pool_split_counts'].get('validation',0)
                                      else 'split allocation and inclusion probabilities must be considered when interpreting aggregate quality'),
                                     'structural evidence checks do not establish semantic or factual accuracy']}
    putjson('summary_report.json',summary)
    summary_text = '# Stage G production quality audit100\n\n'+'\n'.join([
        f'- Snapshot: {summary["completed_target_files_at_snapshot"]} complete target files; eligible completed pool after D041/prior-audit50 exclusions {summary["eligible_pool_size"]}.',
        f'- Random sample:100 businesses, split {summary["split_counts"]}, seed {SEED};300 unchanged student targets.',
        f'- Eligible pool split: {summary["eligible_pool_split_counts"]}. Simple random sampling without quotas; split counts are reported as drawn, with no substitutions.',
        f'- Complete selected reviews:{counts["selected_reviews"]}; original photos:{len(photos_needed)}; caption occurrences:{counts["caption_occurrences"]}.',
        f'- Review claims:{counts["review_claims"]}; visual verdict counts:{dict((k,v) for k,v in counts.items() if k.startswith("visual_verdict_"))}.',
        f'- Zero Filtered-caption cases:{counts["zero_filtered"]}; exact cross-condition verdict conflicts:{len(flags)}; auxiliary review-ID warnings:{summary["review_ID_warning_entries"]}.',
        '- Every photo is copied unchanged; exact source texts/IDs/order and API output strings are preserved. No target generation, paid API or BLIP-2 calls.',
        '- Sampling was frozen before warning/quality checks. No case substitution; additional technical flags are separate.',
        '- This is an independent review package, not a completed factual audit. Teacher verdicts are model judgments; human annotation status is unchanged.',
        '- Legacy target-file status wording is preserved verbatim; D042 production approval and accepted checkpoint bindings establish current provenance.',
        '- Results concern the completed snapshot, not unsampled/incomplete businesses or held-out test. See rubric.md.',
        '- Split limitation: '+summary['sample_limitations'][1]+'.'])+'\n'
    put('SUMMARY.md',summary_text.encode())
    put('rubric.md',RUBRIC.encode())
    put('README.md',('''# Stage G Production Quality Audit Bundle

Extract the whole ZIP and open index.html to inspect the100 sampled businesses.
Read SUMMARY.md, rubric.md and sampling_manifest.json first. cases.json contains
complete selected reviews, raw/normalized BLIP-2 captions, teacher assessments,
all three condition-aligned targets and exact evidence. claim_evidence_index.json
connects every claim and final target statement to its review spans or caption/photo.
Original images are at images/<photo_id>.jpg; per-business JSONs support small batches.
Raw responses, accepted checkpoints and original target artifacts are preserved.

Suggested ChatGPT request: Independently review these100 businesses using rubric.md.
Actually inspect the original images and complete selected reviews, in batches of5
if needed. State which IDs were inspected and mark inaccessible items NOT REVIEWED.
Report exact claim/REV/IMG_CAP/PHOTO IDs, quotations, strengths and concerns. Separate
factual errors, uncertainty, overgeneralization, lost contrast and excessive strictness
from correct simple content. Do not infer correctness from teacher verdicts or claim
that ChatGPT review is human verification. Do not rewrite or change existing targets.
Return findings for researcher discussion in Vietnamese. Additional technical flags
are separate from the frozen random sample and are not proven factual errors.
''').encode())
    put('index.html',render(cases).encode())
    putjson('review_response_template.json',{'human_verified':False,'businesses':[{'business_id':c['business_id'],
                    'inspected_photo_ids':[],'read_review_ids':[],'observations':[],'human_final_decision':None} for c in cases]})
    # Screen portable structured material without reading credentials. Opaque provider
    # ciphertext is preserved in raw API files and omitted only from the screening view.
    secret_pattern = re.compile(r'sk-[A-Za-z0-9_-]{20,}')
    for name in exported:
        if name.endswith('.jpg'):
            continue
        text = (BUNDLE/name).read_text(encoding='utf-8')
        if name.startswith('raw_api_responses/'):
            screening = json.loads(text)
            for item in screening.get('output',[]):
                if item.get('type') == 'reasoning' and isinstance(item.get('encrypted_content'),str):
                    assert re.fullmatch(r'[A-Za-z0-9_+/=-]+',item['encrypted_content'])
                    item['encrypted_content'] = '[opaque provider ciphertext; original file unchanged]'
            text = json.dumps(screening,ensure_ascii=False)
        assert not secret_pattern.search(text), 'Potential credential-like text in export: '+name
        # A drive letter must start a path token; the trailing s in https://
        # is not a drive. Keep provider URLs and original API provenance intact.
        assert not re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[/\\]',text), 'Absolute Windows path in portable export: '+name
    for relative, sha in protected.items():
        assert file_sha(ROOT/relative) == sha, 'Selected immutable evidence changed during export: '+relative
    assert file_sha(ROOT / 'outputs/stage_g/production_quality_audit50_v1/chatgpt_stage_g_production_quality_audit50.zip') == manifest['previous_audit50_zip_sha256']
    putjson('export_manifest.json',{'export_only':True,'source_artifact_sha256':protected,
                    'file_sha256':dict(exported),'API_calls':0,'production_lock_acquired':False})
    with zipfile.ZipFile(ZIP,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(exported):
            archive.write(BUNDLE/name,BUNDLE.name+'/'+name)
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for name, sha in exported.items():
            assert digest(archive.read(BUNDLE.name+'/'+name)) == sha
        restored = json.loads(archive.read(BUNDLE.name+'/cases.json'))['businesses']
        assert restored == cases
        for case in restored:
            for condition in ('unfiltered','filtered'):
                for cap in case['components'][condition]['captions']:
                    assert digest(archive.read(BUNDLE.name+'/'+cap['image_file'])) == cap['original_image_sha256']
            assert read(ROOT / next(b['target_source'] for b in selected if b['business_id']==case['business_id']))['targets'] == case['final_targets']
    report = dict(summary,zip_path=ZIP.relative_to(ROOT).as_posix(),zip_bytes=ZIP.stat().st_size,
                  zip_sha256=file_sha(ZIP),viewer_path=(BUNDLE/'index.html').relative_to(ROOT).as_posix(),
                  export_source_checks=len(protected),exported_files=len(exported),completed_utc=now())
    save(FOLDER/'export_report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--export',action='store_true')
    args = parser.parse_args()
    assert args.prepare != args.export
    prepare() if args.prepare else export()
