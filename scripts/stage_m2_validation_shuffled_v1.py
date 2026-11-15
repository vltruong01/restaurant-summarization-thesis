"""Supplementary VALIDATION-only shuffled output reassignment and CPU metrics.

No LED generation, training, network, human annotation, or TEST scoring entrypoint.
The Stage K source-only assignment function is adapted only at its split guard.
"""
import argparse
import ast
import contextlib
import csv
import datetime
import difflib
import inspect
import json
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import s1000_final_eval_core_v1 as core
from s1000_final_eval_core_v1 import read, write, write_csv, file_hash, text_hash, digest, require, isolated, paired
from stage_m1_existing_teacher_metrics_v1 import ROOT, PREP, DATA, HIST, jsonl, normalized_predictions, target_parts
from stage_m_evaluation_v1 import METRICS, metric_runtime, score_components, strip_header, descriptive_score
from stage_m1_pooled_bertscore_v1 import ExactPairCache

M1 = ROOT / 'outputs/stage_m1/existing_teacher_validation200_v1'
OUT = ROOT / 'outputs/stage_m2/validation200_shuffled_teacher_v1'
TITLE = 'VALIDATION200 — Supplementary Shuffled Negative-Control / Teacher-Target Agreement'
SCHEMA = ['business_id', 'split', 'source', 'source_sha256', 'selected_review_ids', 'selected_photo_ids']


def adapted_mapping_function():
    """Keep Stage K numeric/category/tie costs exactly; replace split literals only."""
    original = inspect.getsource(core.shuffled_mapping)
    tree = ast.parse(original)
    class ValidationOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'test':
                return ast.copy_location(ast.Constant('validation'), node)
            if node.value == 'TEST source cohort only':
                return ast.copy_location(ast.Constant('VALIDATION source cohort only'), node)
            return node
    tree = ValidationOnly().visit(tree)
    tree.body[0].name = 'validation_shuffled_mapping'
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + '\n'
    namespace = dict(vars(core))
    exec(compile(tree, '<Stage K split-only VALIDATION adaptation>', 'exec'), namespace)
    return namespace['validation_shuffled_mapping'], original, code


def input_paths():
    paths = [DATA/f'validation_{c}.jsonl' for c in ['reviews_only', 'unfiltered', 'filtered']]
    paths += [M1/n for n in ['ALL_VALIDATION200_PER_BUSINESS_METRICS.csv', 'METRIC_COMPUTATION_COMPLETE.json',
        'PREDICTION_ALIGNMENT_AUDIT.json', 'TARGET_AND_SPLIT_AUDIT.json', 'ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json',
        'ACTUAL_MODEL_ADAPTER_BACKBONE_REHASH.json', 'COHORT200.json']]
    paths += [ROOT/r['prediction_file'] for r in read(M1/'PREDICTION_ALIGNMENT_AUDIT.json')]
    paths += [PREP/n for n in ['SHUFFLED_POLICY.json', 'MODEL_AND_DECODER_LOCK.json', 'METRIC_LOCK.json']]
    paths += [HIST/n for n in ['RUN_DECODE.py', 'source_separated_contract.py', 'INPUTS/PRIMARY_CASES.json', 'PAYLOAD_MANIFEST.json']]
    paths += [ROOT/'scripts'/n for n in ['s1000_final_eval_core_v1.py', 'stage_m_evaluation_v1.py',
        'stage_m1_existing_teacher_metrics_v1.py', 'stage_m1_pooled_bertscore_v1.py', 'score_s1000_automatic_prepared_v1.py']]
    paths += [ROOT/'data/processed/yelp_2300_v1/business_split.csv', ROOT/'assets/tokenizers/led-base-16384/tokenizer.json']
    return paths


def protected_test_hashes():
    # Hash only frozen files; no TEST output/source parsing, new mapping, or scoring.
    paths = [ROOT/f'outputs/stage_k/s1000_test100_phase1_v1/EXECUTION/{c}/raw_predictions.json' for c in ['RO','U','F','S']]
    paths += [PREP/n for n in ['TEST100_SHUFFLED_MAPPING.csv','TEST100_SHUFFLED_INPUT_MANIFEST.json','TEST100_INFERENCE_PLAN.json','METRIC_LOCK.json']]
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}


def current_data():
    audit = read(M1/'TARGET_AND_SPLIT_AUDIT.json')
    data = {}
    for c in ['reviews_only','unfiltered','filtered']:
        path = DATA/f'validation_{c}.jsonl'
        binding = next(x for x in audit['dataset_inventory'] if x['split']=='validation' and x['condition']==c)
        require(file_hash(path)==binding['sha256'], 'Canonical validation targets changed')
        rr = jsonl(path)
        require(len(rr)==len({r['business_id'] for r in rr})==200, 'Validation duplicates/missing')
        data['validation',c] = {r['business_id']:r for r in rr}
    raw,calls,refs,ids,proof = normalized_predictions(data)
    require(proof==read(M1/'PREDICTION_ALIGNMENT_AUDIT.json'), 'M1 prediction binding changed')
    require(ids==read(M1/'COHORT200.json')['business_ids'], 'Frozen cohort changed')
    require(file_hash(M1/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv')==read(M1/'METRIC_COMPUTATION_COMPLETE.json')['metrics_sha256'], 'M1 scores changed')
    return data,raw,calls,refs,ids,proof


def prepare():
    from tokenizers import Tokenizer
    import scipy
    require(not OUT.exists(), 'Refuse overwrite prior mapping/evaluation')
    OUT.mkdir(parents=True)
    data,raw,calls,refs,ids,proof = current_data()
    splits = {r['business_id']:r['split'] for r in csv.DictReader((ROOT/'data/processed/yelp_2300_v1/business_split.csv').open(encoding='utf8'))}
    require(all(splits[b]=='validation' for b in ids), 'Cross-split source')
    rows = [{k:data['validation','filtered'][b][k] for k in SCHEMA} for b in ids]
    write(OUT/'SOURCE_ONLY_MAPPING_INPUTS.json', rows)
    policy = read(PREP/'SHUFFLED_POLICY.json')
    require(policy['seed']==42 and scipy.__version__=='1.15.2', 'Established assignment runtime changed')
    tok = Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    function,original,derived = adapted_mapping_function()
    (OUT/'ORIGINAL_STAGE_K_MAPPING_FUNCTION.py.txt').write_text(original,encoding='utf8')
    (OUT/'VALIDATION_MAPPING_FUNCTION.py.txt').write_text(derived,encoding='utf8')
    write(OUT/'MAPPING_POLICY.json', {'original_policy':policy,'adaptation':'Only function name and test-to-validation split guard/record literal; no costs, seeds, feasibility, category, zero-caption, or one-to-one changes.',
        'policy_version_retained_for_identical_cost_hashing':True,'source_policy_file_sha256':file_hash(PREP/'SHUFFLED_POLICY.json'),
        'original_function_sha256':text_hash(original),'adapted_function_sha256':text_hash(derived),'scipy':scipy.__version__,
        'mapping_source_only':True,'single_mapping_no_seed_selection':True,'mapping_generated_after_F_predictions_existed':True,
        'mapping_frozen_before_S_scoring':True,'historical_policy_status':'Historical preparation approval text is preserved; present researcher Stage M.2 request authorizes this VALIDATION application.'})
    print('Constructing single source-only validation mapping, Stage K costs/seed42 unchanged',flush=True)
    mapping = function(rows,tok,policy)
    mapping['TEST_inference_performed'] = False
    mapping['VALIDATION_model_inference_performed'] = False
    write(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json',mapping)
    write(OUT/'MAPPING_FREEZE_BEFORE_SCORING.json', {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'mapping_sha256':file_hash(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json'),'source_only_inputs_sha256':file_hash(OUT/'SOURCE_ONLY_MAPPING_INPUTS.json'),
        'seed':42,'score_inspection_used_to_select_mapping':False,'model_inference':False})
    finish_preparation()


def finish_preparation():
    """Continue evidence serialization from the already frozen single mapping."""
    from tokenizers import Tokenizer
    require(not (OUT/'EVALUATION_PLAN_BEFORE_SCORING.json').exists(),'Preparation already complete')
    data,raw,calls,refs,ids,proof=current_data()
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    freeze=read(OUT/'MAPPING_FREEZE_BEFORE_SCORING.json')
    require(file_hash(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')==freeze['mapping_sha256'] and file_hash(OUT/'SOURCE_ONLY_MAPPING_INPUTS.json')==freeze['source_only_inputs_sha256'],'Frozen mapping/source inputs changed')
    require(read(OUT/'SOURCE_ONLY_MAPPING_INPUTS.json')==[{k:data['validation','filtered'][b][k] for k in SCHEMA} for b in ids],'Frozen source rows changed')
    mapping=read(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')
    # Direct verification of the actual frozen native input and generation loop.
    cases = {r['business_id']:r for r in read(HIST/'INPUTS/PRIMARY_CASES.json')}
    require(set(cases)==set(ids), 'Native validation input cohort differs')
    for b in ids:
        visual = isolated(data['validation','filtered'][b]['source'])[1]
        call = cases[b]['calls']['visual']
        require(call['source']==visual and call['input_ids']==tok.encode(visual).ids, 'Actual native Visual input contains extra information')
        require(call['source_sha256']==text_hash(visual), 'Actual native source hash differs')
        require('[BUSINESS_INFORMATION]' not in visual and '[REV_' not in visual, 'Review/metadata leakage to Visual')
    adapter = next(r for r in read(M1/'ACTUAL_MODEL_ADAPTER_BACKBONE_REHASH.json')['adapters'] if r['model']=='F')
    require(file_hash(ROOT/adapter['file'])==adapter['sha256']==next(p for p in proof if p['model']=='F')['checkpoint_sha256'], 'F-S1000 adapter changed')
    equivalence={'status':'PASS_STRUCTURAL_AND_ACTUAL_TOKEN_INPUT_INVARIANT',
        'businesses_verified':200,'visual_input':'[TASK_VISUAL_EVIDENCE] + [VISUAL_CONTEXT] + own ordered caption block only',
        'business_metadata_or_review_in_Visual_input':False,'native_input_tokens_hash_verified_all200':True,
        'native_generation_loop':'RUN_DECODE.py: fully eval model; batch1; labels removed; only input IDs, masks and shared generation_config passed to generate; BOS-only global attention; no previous output/cache passed between calls.',
        'decoder':'Original deterministic beam4, no sampling, length penalty1, max768; unchanged adapter/backbone per original restoration receipts.',
        'same_donor_caption_text_implies_same_token_ids_padding_BOS_mask_and_generation_arguments':True,
        'numerical_qualification':'Functional equivalence at frozen deterministic input/model/decoder level, not a new empirical cross-call GPU reproducibility test. Exact historical donor token IDs/text are reused; no fresh LED generation.',
        'F_adapter_independently_rehashed_now':adapter,'backbone_integrity':'M1 independently rehashed original backbone; native generation receipt verified unchanged backbone. M2 does not load or modify LED weights.',
        'hash_bound_files':[p.relative_to(ROOT).as_posix() for p in [HIST/'RUN_DECODE.py',HIST/'source_separated_contract.py',HIST/'INPUTS/PRIMARY_CASES.json']]}
    eqpath=OUT/'OUTPUT_REASSIGNMENT_EQUIVALENCE.json'
    if eqpath.exists():require(read(eqpath)==equivalence,'Prior equivalence receipt changed')
    else:write(eqpath,equivalence)
    by = {(r['business_id'],r['task']):r for r in raw['F']}
    derived_records=[];ledger=[]
    for m in mapping['mappings']:
        b=m['recipient_id'];donor=m['donor_id'] or b
        r,v=by[b,'review'],by[donor,'visual']
        require(text_hash(calls['F']['F',donor,'visual']['source'])==m['visual_task_source_sha256'], 'Donor caption source differs')
        # Keep full original pass records, including their original business IDs.
        derived_records.append({'business_id':b,'condition':'S','split':'validation','review_original_F_pass':r,'visual_original_F_donor_pass':v,
            'donor_id':m['donor_id'],'independent_generation':False,'combined_text':r['text']+'\n\n'+v['text'],
            'recipient_reference_business_id':b,'mapping_sha256':file_hash(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')})
        ledger.append({'business_id':b,'donor_id':m['donor_id'],'split':'validation','caption_count':m['caption_count'],'matching_tier':m['matching_tier'],
            'shared_categories':json.dumps(m['shared_categories'],ensure_ascii=False),'genuinely_shuffled':m['donor_changed'] and m['visual_source_changed'],
            'review_identical_to_F':r['text']==by[b,'review']['text'],'review_output_sha256':r['text_sha256'],
            'F_visual_output_sha256':by[b,'visual']['text_sha256'],'S_visual_output_sha256':v['text_sha256'],
            'S_token_ids_sha256':digest(v['token_ids']),'donor_token_ids_sha256':digest(by[donor,'visual']['token_ids']),
            'F_visual_source_sha256':by[b,'visual']['source_sha256'],'S_visual_source_sha256':m['visual_task_source_sha256'],
            'donor_visual_source_sha256':v['source_sha256'],'reference_business_id':b,'F_reference_visual_sha256':text_hash(refs['F'][b]['visual_evidence']),
            'S_reference_visual_sha256':text_hash(refs['F'][b]['visual_evidence']),'output_text_changed':v['text']!=by[b,'visual']['text'],
            'checkpoint_sha256':v['checkpoint_sha256'],'recipient_photo_ids':json.dumps(m['recipient_F_photo_ids']),
            'donor_photo_ids':json.dumps(m['donor_photo_ids']),'visual_task_tokens':m['visual_task_tokens'],'full_source_tokens':m['full_source_tokens'],
            'caption_increment_tokens':m['caption_increment_tokens']})
    positive=[x for x in ledger if x['caption_count']>0]
    require(len(positive)==len({x['donor_id'] for x in positive}), 'Donor reuse')
    require({x['donor_id'] for x in positive}=={x['business_id'] for x in positive}, 'Not a permutation')
    require(all(x['donor_id']!=x['business_id'] and x['genuinely_shuffled'] for x in positive), 'Inactive/self positive mapping')
    require(all(x['F_reference_visual_sha256']==x['S_reference_visual_sha256'] and x['S_visual_source_sha256']==x['donor_visual_source_sha256'] for x in ledger), 'Wrong reference/source binding')
    write_csv(OUT/'VALIDATION200_SHUFFLED_MAPPING.csv',ledger)
    write(OUT/'VALIDATION200_SHUFFLED_DERIVED_PREDICTIONS.json',derived_records)
    write(OUT/'BUSINESS_LEVEL_INTEGRITY_AND_LEAKAGE_CHECKS.json', {'status':'PASS_ACTUAL_ALL200','businesses':200,
        'genuinely_shuffled':len(positive),'zero_caption_unchanged':200-len(positive),'text_changed':sum(x['output_text_changed'] for x in ledger),
        'matching_tiers':dict(Counter(x['matching_tier'] for x in ledger)),'caption_count_groups':mapping['groups'],
        'no_self_positive':True,'one_to_one_positive':True,'all_donors_VALIDATION':True,'all200_review_verbatim_F':True,
        'all200_visual_original_donor_text_and_token_ids':True,'all200_reference_recipient_F_not_donor_or_U':True,
        'no_TEST_records_used_as_donors_or_references':True,'mapping_uses_only_source_metadata':True,
        'zero_policy':'Unchanged header-only empty Visual, donor null, intervention N/A; retained, all six empty-side scores zero.'})
    write(OUT/'PROTECTED_TEST_HASHES_BEFORE.json',protected_test_hashes())
    bindings=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in input_paths()]
    write_csv(OUT/'REPRODUCIBILITY_INPUT_MANIFEST.csv',bindings)
    write(OUT/'EVALUATION_PLAN_BEFORE_SCORING.json',{'title':TITLE,'businesses':200,'mapping_seed':42,'mapping_file_sha256':file_hash(OUT/'VALIDATION200_SHUFFLED_MAPPING.csv'),
        'F_scores':'Reuse actual verified M1 CPU6 values; independently replay all200 full-text ROUGE. S new CPU6 BERTScore against same recipient F targets.',
        'BERTScore':read(M1/'ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json'),'bootstrap_seed':20261028,'bootstrap_replicates':20000,
        'primary_CI':'Paired business percentile CI conditional on this one fixed permutation. Shared without-replacement donors mean observations are dependent; CI does not model assignment uncertainty.',
        'sensitivity_CI':'Resample entire permutation cycles with replacement, ratio of summed differences to summed cycle sizes; descriptive sensitivity, not independent randomization inference.',
        'all200_including_zero':True,'no_p_values_or_winner_selection':True,'new_training_LED_Teacher_or_human_work':False})
    print('PASS mapping frozen and actual200 donor/source/token/reference bindings',flush=True)


def check_inputs():
    for r in csv.DictReader((OUT/'REPRODUCIBILITY_INPUT_MANIFEST.csv').open(encoding='utf8')):
        require(file_hash(ROOT/r['file'])==r['sha256'], 'Changed input '+r['file'])
    freeze=read(OUT/'MAPPING_FREEZE_BEFORE_SCORING.json')
    require(file_hash(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')==freeze['mapping_sha256'], 'Frozen mapping changed')


def score():
    require(not (OUT/'METRIC_COMPUTATION_COMPLETE.json').exists(), 'Refuse repeat finished metrics')
    check_inputs()
    data,raw,calls,refs,ids,proof=current_data()
    sealed=read(HIST/'PAYLOAD_MANIFEST.json')['files']
    for name in ['RUN_DECODE.py','source_separated_contract.py','INPUTS/PRIMARY_CASES.json']:
        require(file_hash(HIST/name)==sealed[name]['sha256'],'Native generation implementation/input differs from original seal')
    generation=read(HIST/'EFFECTIVE_GENERATION_CONFIG.json')
    require(all(generation[k]==v for k,v in {'num_beams':4,'do_sample':False,'max_length':768,'length_penalty':1.0,'eos_token_id':2,'forced_eos_token_id':2}.items()),'Frozen decoder protocol differs')
    reuse=read(HIST/'REUSE_LEDGER.json')
    reused=[r for r in reuse['entries'] if r['model']=='V1_1000' and r['canonical']]
    write(OUT/'SEALED_IMPLEMENTATION_AND_DECODER_RECHECK.json',{'status':'PASS','sealed_native_implementation_and_actual_input_files_verified':3,
        'generation_config_sha256':file_hash(HIST/'EFFECTIVE_GENERATION_CONFIG.json'),
        'beam':4,'sampling':False,'length_penalty':1.0,'max_length':768,'eos':2,'forced_eos':2,
        'historical_exact_reuse_ledger_sha256':file_hash(HIST/'REUSE_LEDGER.json'),'prior_reuse_ledger_F_records':len(reused),
        'reuse_qualification':'Existing historical ledger attests exact checkpoint/backbone/source/decoder/runtime matches; M2 independently rechecks all final200 source/input/output-token/checkpoint identities. Earlier GPU calls are not rerun.'})
    by={(r['business_id'],r['task']):r for r in raw['F']}
    mapping=read(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')['mappings']
    sraw=[];scalls={}
    for m in mapping:
        b=m['recipient_id'];donor=m['donor_id'] or b;v=by[donor,'visual']
        sraw.append({**v,'condition':'S','business_id':b,'source_sha256':m['visual_task_source_sha256'],
            'input_ids_sha256':m['visual_task_input_ids_sha256'],'original_donor_business_id':donor,'independent_generation':False})
        scalls['S',b,'visual']={'source':m['visual_task_source']}
    scorer,runtime=metric_runtime(PREP,ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large')
    import torch
    torch.set_num_threads(6)
    runtime.update(torch_CPU_threads=6,BERTScorer_nthreads=1,execution_amendment='Inherited Stage M.1 VALIDATION CPU6 execution; unchanged TEST lock',
        F_scores='Unmodified M1 final CPU6 rows',S_scores='This actual original BERTScorer forward; grouping roundoff may differ at FP32 sub-micro scale')
    write(OUT/'ACTUAL_METRIC_RUNTIME.json',runtime)
    pairs=list(dict.fromkeys((strip_header(r['text'],'visual'),refs['F'][r['business_id']]['visual_evidence']) for r in sraw
        if strip_header(r['text'],'visual').strip() and refs['F'][r['business_id']]['visual_evidence'].strip()))
    print(f'Actual S Visual BERTScore: {len(pairs)} nonempty pairs, CPU FP32 threads6, batch8',flush=True)
    start=time.perf_counter()
    with (OUT/'BERTSCORE_PROGRESS.log').open('x',encoding='utf8',buffering=1) as log,contextlib.redirect_stderr(log):
        p,r,f=scorer.score([x[0] for x in pairs],[x[1] for x in pairs],batch_size=8,verbose=True)
    cache=ExactPairCache(scorer._tokenizer,{pair:(float(p[i]),float(r[i]),float(f[i])) for i,pair in enumerate(pairs)})
    scores,receipt=score_components(sraw,scalls,refs['F'],cache,'teacher')
    prior=list(csv.DictReader((M1/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv').open(encoding='utf8')))
    frows={r['business_id']:r for r in prior if r['condition']=='F' and r['task']=='visual'}
    smap={r['business_id']:r for r in scores};mmap={r['recipient_id']:r for r in mapping};wide=[]
    for b in ids:
        row=smap[b];old=frows[b];m=mmap[b];donor=m['donor_id'] or b
        reference=refs['F'][b]['visual_evidence']
        require(old['component_reference_sha256']==text_hash(reference), 'F reference differs')
        require(old['output_sha256']==by[b,'visual']['text_sha256'], 'Cached F output differs')
        replay=descriptive_score(strip_header(by[b,'visual']['text'],'visual'),reference,calls['F']['F',b,'visual']['source'])
        require(all(abs(float(old[k])-replay[k])<1e-12 for k in METRICS[:3]), 'Actual F ROUGE replay differs')
        row.update(split='validation',donor_id=m['donor_id'],genuinely_shuffled=m['donor_changed'] and m['visual_source_changed'],
            component_reference_sha256=text_hash(reference),reference_condition='filtered',execution_torch_CPU_threads=6,
            output_origin='ORIGINAL_F_DONOR_VERBATIM_NO_INFERENCE',donor_output_sha256=by[donor,'visual']['text_sha256'])
        wide.append({'business_id':b,'donor_id':m['donor_id'],'split':'validation','reference_type':'COMMON_RECIPIENT_F_SPECIFIC_TEACHER_VISUAL',
            'caption_count':m['caption_count'],'genuinely_shuffled':row['genuinely_shuffled'],'F_output_sha256':old['output_sha256'],'S_output_sha256':row['output_sha256'],
            'F_reference_sha256':text_hash(reference),'S_reference_sha256':text_hash(reference),'empty_candidate_F':old['empty_candidate'],
            'empty_candidate_S':row['empty_candidate'],'empty_reference':row['empty_reference'],
            **{prefix+k:(float(old[k]) if prefix=='F_' else row[k] if prefix=='S_' else float(old[k])-row[k]) for k in METRICS for prefix in ['F_','S_','F_minus_S_']}})
    require(len(scores)==len(wide)==200,'Silent exclusion')
    write_csv(OUT/'S_VISUAL_COMPONENT_METRICS.csv',scores)
    write_csv(OUT/'VALIDATION200_SHUFFLED_VISUAL_METRICS.csv',wide)
    write(OUT/'METRIC_COMPUTATION_COMPLETE.json',{'status':'PASS_ACTUAL200_S_VISUAL_AND_VERIFIED200_F_CACHED','businesses':200,
        'S_nonempty_pairs_forwarded':len(pairs),'F_ROUGE_replayed':200,'F_scores_reused':200,'F_S_common_reference_pairs':200,
        'score_cache':receipt,'seconds_after_model_load':time.perf_counter()-start,'metrics_sha256':file_hash(OUT/'VALIDATION200_SHUFFLED_VISUAL_METRICS.csv'),
        'TEST_scored':0,'LED_generation_calls':0,'Teacher_calls':0,'training_updates':0,'human_work':False})
    print('COMPLETE actual200 S Visual metrics; F common-reference pairing verified',flush=True)


def permutation_cycles(mapping):
    donor={r['recipient_id']:r['donor_id'] or r['recipient_id'] for r in mapping}
    cycles=[];seen=set()
    for b in sorted(donor):
        if b in seen:continue
        chain=[];v=b
        while v not in seen:
            seen.add(v);chain.append(v);v=donor[v]
        require(v==b,'Not disjoint permutation cycles')
        cycles.append(chain)
    return cycles


def cycle_bootstrap(differences,cycles,seed=20261028,B=20000):
    import numpy as np
    sums=np.array([sum(differences[b] for b in cycle) for cycle in cycles]);sizes=np.array([len(cycle) for cycle in cycles])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(cycles),(B,len(cycles)))
    draws=sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def finalize():
    require(not (OUT/'STAGE_M2_VALIDATION200_SHUFFLED_REVIEW.zip').exists(), 'Refuse overwrite final ZIP')
    require(read(OUT/'METRIC_COMPUTATION_COMPLETE.json')['businesses']==200,'Complete actual scores required')
    check_inputs()
    derived=read(OUT/'VALIDATION200_SHUFFLED_DERIVED_PREDICTIONS.json')
    original={r['business_id']:r for r in read(HIST/'MERGED/V1_1000/predictions.json')}
    mm={r['recipient_id']:r for r in read(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')['mappings']}
    require(len(derived)==len({r['business_id'] for r in derived})==200,'Derived outputs duplicated/missing')
    for r in derived:
        b=r['business_id'];donor=mm[b]['donor_id'] or b
        for saved,actual in [(r['review_original_F_pass'],original[b]['passes']['review']),(r['visual_original_F_donor_pass'],original[donor]['passes']['visual'])]:
            require(all(saved[k]==actual[k] for k in ['business_id','text','text_sha256','token_ids','source_sha256','input_ids_sha256','checkpoint_sha256']), 'Derived pass changed original content/provenance')
        require(r['combined_text']==original[b]['passes']['review']['text']+'\n\n'+original[donor]['passes']['visual']['text'],'Combined prediction rewritten')
        require(r['recipient_reference_business_id']==b and r['donor_id']==mm[b]['donor_id'],'Derived business/reference/donor mismatch')
    write(OUT/'DERIVED_OUTPUT_INDEPENDENT_REPLAY_CHECK.json',{'status':'PASS_ACTUAL_ALL200','Review_original_F_passes_exact':200,'Visual_original_donor_passes_exact':200,
        'token_ID_text_source_checkpoint_identity_unchanged':True,'combined_output_verbatim':True,'recipient_reference_identity_separate_from_original_donor_identity':True})
    m1rows=list(csv.DictReader((M1/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv').open(encoding='utf8')))
    reused_review=[{**r,'condition':'S','measurement_origin':'EXACT_UNCHANGED_F_REVIEW_METRICS_REUSE_NOT_NEW_FORWARD'} for r in m1rows if r['condition']=='F' and r['task']=='review']
    require(len(reused_review)==200 and all(r['output_sha256']==original[r['business_id']]['passes']['review']['text_sha256'] for r in reused_review), 'S Review cached score/output binding mismatch')
    write_csv(OUT/'S_REVIEW_REUSED_METRICS.csv',reused_review)
    rows=list(csv.DictReader((OUT/'VALIDATION200_SHUFFLED_VISUAL_METRICS.csv').open(encoding='utf8')))
    mapping=read(OUT/'VALIDATION200_SHUFFLED_INPUT_MANIFEST.json')['mappings'];cycles=permutation_cycles(mapping)
    integrity=read(OUT/'BUSINESS_LEVEL_INTEGRITY_AND_LEAKAGE_CHECKS.json');comparisons=[]
    for metric in METRICS:
        a=[float(r['F_'+metric]) for r in rows];s=[float(r['S_'+metric]) for r in rows]
        result=paired(a,s,seed=20261028,B=20000,P=1)
        result.pop('signflip_two_sided_p');result.pop('signflip_replicates')
        differences={r['business_id']:float(r['F_minus_S_'+metric]) for r in rows}
        comparisons.append({'metric':metric,'F_mean':sum(a)/200,'S_mean':sum(s)/200,**result,
            'CI_scope':'Descriptive business-paired, conditional on one fixed dependent permutation; not assignment-level inference',
            'cycle_cluster_CI95_sensitivity':cycle_bootstrap(differences,cycles),'permutation_cycles':len(cycles),
            'genuinely_shuffled':integrity['genuinely_shuffled'],'unchanged_zero':integrity['zero_caption_unchanged']})
    write(OUT/'PAIRED_BOOTSTRAP_COMPARISONS.json',comparisons);write_csv(OUT/'PAIRED_BOOTSTRAP_COMPARISONS.csv',comparisons)
    write(OUT/'PERMUTATION_CYCLE_DEPENDENCE.json',{'cycles':cycles,'cycle_sizes':list(map(len,cycles)),
        'business_bootstrap_limitation':'Donor outputs are not independent draws: all positive outputs occur once under one constrained matching. Ordinary paired bootstrap conditions on this mapping and ignores assignment variability.',
        'cycle_sensitivity_method':'Sample whole disjoint cycles (including unchanged zero singleton) with replacement, same number of cycles, then divide summed business differences by summed sampled business counts. Seed20261028, B20000.',
        'cycle_sensitivity_limit':'Retains within-cycle reuse/dependence but few cycles and globally constrained matching limit inference. Cycles need not be statistically independent. Neither interval is an exact randomization test or robust population-level causal CI.'})
    aggregates=list(csv.DictReader((M1/'AGGREGATE_VALIDATION200_METRICS.csv').open(encoding='utf8')))
    table=[]
    for model in ['RO','U','F','S']:
        for task in ['review','visual']:
            if model=='RO' and task=='visual':continue
            if model=='S' and task=='visual':
                values={r['metric']:r['S_mean'] for r in comparisons}
            else:
                prior=next(r for r in aggregates if r['model']==('F' if model=='S' else model) and r['task']==task)
                values={k:float(prior[k]) for k in METRICS}
            table.append({'model':model,'task':task,'n':200,'reference':'Common identical Review' if task=='review' else 'U-specific Visual (not direct F/S comparator)' if model=='U' else 'Same recipient F-specific Visual',
                'measurement_origin':'Exact F Review reuse by construction' if model=='S' and task=='review' else 'Actual new S metrics, no LED generation' if model=='S' else 'Unchanged verified M1 actual scores',**values})
    write_csv(OUT/'FINAL_RO_U_F_S_COMPARISON.csv',table)
    md=[f'# {TITLE}\n','All means use all200 VALIDATION businesses; scores0–1. F=selected V1-S1000, not Final V2. RO Visual=N/A.\n',
        '| Model | Component | n | Reference | R1 | R2 | RL | BERT P | BERT R | BERT F1 |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in table:md.append('| '+' | '.join([r['model'],r['task'],'200',r['reference']]+[f'{r[k]:.4f}' for k in METRICS])+' |')
    md += ['\nS Review is F verbatim, not a new independent measurement. U/F Visual use different condition-specific targets; no U–F superiority claim.\n',
        '## Paired Visual F minus S — same recipient F Teacher reference\n',
        '| Metric | F mean | S mean | F−S | Conditional business-paired95% CI | Whole-cycle sensitivity95% CI | n |',
        '|---|---:|---:|---:|---|---|---:|']
    for r in comparisons:md.append(f"| {r['metric']} | {r['F_mean']:.6f} | {r['S_mean']:.6f} | {r['mean_difference']:.6f} | [{r['CI95'][0]:.6f}, {r['CI95'][1]:.6f}] | [{r['cycle_cluster_CI95_sensitivity'][0]:.6f}, {r['cycle_cluster_CI95_sensitivity'][1]:.6f}] | 200 |")
    bf=next(r for r in comparisons if r['metric']=='bertscore_f1')
    md += [f"\n## Assignment and provenance\n\nOne mapping only: original Stage K seed42 and numeric cost version retained, scipy1.15.2, same exact positive caption-count strata, globally minimized allowlisted category fallbacks followed by seeded tie cost. No self donors; no donor reuse; no metric/target/output information in matching. {integrity['genuinely_shuffled']} genuine source interventions, {integrity['zero_caption_unchanged']} unchanged zero-caption case(s), {integrity['text_changed']} changed Visual strings. Category tiers: {json.dumps(integrity['matching_tiers'])}. All200 Review strings/token IDs remain F; Visual text/token IDs exactly match each original F donor. Mapping was made after existing F predictions, but frozen once before any S scoring; no mapping search or seed tuning.\n",
        'Actual200 native Visual input strings/token IDs contain only task marker + caption block. Fully eval, batch1, BOS-only global attention, shared deterministic generation settings; the generation API receives no recipient identity, reviews, metadata, previous decoder output or persistent per-business cache. Functional input/model/decoder equivalence supports exact output reassignment. No new GPU reproducibility claim or inference was performed. The full original donor pass keeps its original business identity; derived recipient identity and reference are explicit separate fields.\n',
        '## Metrics and empty cases\n\nOriginal Stage M implementation reused: rouge-score0.1.2 stemmer=True, full component; BERTScore0.3.13 roberta-large frozen revision/weights, layer17, slow tokenizer, CPU FP32 six Torch threads / tokenization nthreads1, batch8, no IDF/rescaling,512-token prefix. F/RO/U actual M1 scores are unchanged. All200 F full-text ROUGE values independently replayed within1e-12. S actual BERTScore was computed now, caching exact candidate+recipient reference pairs. FP32 grouping may yield sub-micro roundoff versus the earlier F pool. No reference or prediction rewriting. Either empty side gives six zeros, including both-empty; all200 retained. Zero-caption case has no artificial positive score. Per-row prefix/cap flags remain available.\n',
        f"## Dependence and interpretation\n\n20,000 paired percentile bootstrap replicates, established seed20261028. The requested business-paired intervals are conditional descriptive intervals for this fixed assignment. Donors are sampled without replacement and are not200 independent outputs. There are {len(cycles)} permutation cycles (sizes {list(map(len,cycles))}); resampling intact cycles is reported as an additional descriptive sensitivity, preserving within-cycle dependence. Global matching dependence, small cycle counts and no repeated randomized assignments prevent either interval from being an exact randomization or population causal interval. No p-values or significance/model-selection claims.\n\nObserved BERTScore F1 F−S={bf['mean_difference']:.6f}. This measures how correctly assigned outputs agree with same-source Teacher references relative to mismatched outputs. Teacher F Visual references were constructed from Filtered BLIP-2 captions, so reference/source alignment naturally favors F. It does NOT independently demonstrate grounding to original Yelp photo pixels, human factual accuracy, superior information coverage, or causal review–caption integration. The architecture separates tasks; Review is unchanged by design. This is supplementary validation analysis, not primary TEST performance or independent human evaluation.\n",
        '## Scope and stop\n\nTEST100 F–S Teacher-reference ROUGE/BERTScore: PENDING — no matching legitimate common TEST references. TEST mapping/predictions/preparation hashes remain unchanged. No TEST target substitution, new inference, training, Teacher generation, paid API, GPU rental, human-reference writing/approval or human evaluation. Selected F-S1000 unchanged. Stop after verified bundle delivery.\n']
    (OUT/'VALIDATION200_F_VS_S_COMPARISON.md').write_text('\n'.join(md),encoding='utf8')
    check_inputs();after=protected_test_hashes()
    require(after==read(OUT/'PROTECTED_TEST_HASHES_BEFORE.json'),'Protected TEST files changed')
    write(OUT/'FINAL_SCOPE_AND_INPUT_RECHECK.json',{'status':'PASS','all_reproducibility_inputs_unchanged':True,'protected_TEST_before_after_equal':True,
        'protected_TEST_sha256':after,'new_LED_generation_calls':0,'Teacher_calls':0,'new_training_updates':0,'paid_calls':0,'human_state_opened_or_modified':False})
    docs=OUT/'DOCUMENTATION_UPDATES';docs.mkdir();docreceipts=[]
    addition=f'''### Stage M.2 VALIDATION200 Shuffled negative-control COMPLETE — 2026-10-29

Actual supplementary same-F-Teacher-reference F–S metrics computed on all200 businesses. Exactly{integrity['genuinely_shuffled']} one-to-one caption/output interventions, {integrity['zero_caption_unchanged']} unchanged zero-caption; original Stage K seed42/cost/category/count rules, split-only validation adaptation. Actual native inputs verify caption-only Visual; donor text/token IDs reused verbatim, Review S=F. No LED generation/training/API/human work. BERT F1 F={bf['F_mean']:.6f}, S={bf['S_mean']:.6f}, F−S={bf['mean_difference']:.6f}, descriptive fixed-mapping paired95% CI [{bf['CI95'][0]:.6f},{bf['CI95'][1]:.6f}],20,000 replicates/seed20261028. Without-replacement dependence and whole-cycle bootstrap sensitivity explicitly reported. These are Teacher/source-alignment diagnostics, not photo-pixel grounding/human quality/causal fusion proof. U Visual uses different targets; no direct U–F claim. TEST100 Teacher-reference metrics remain PENDING. F-S1000 selection unchanged; stop after delivery.

Files: `outputs/stage_m2/validation200_shuffled_teacher_v1/VALIDATION200_F_VS_S_COMPARISON.md` and `STAGE_M2_VALIDATION200_SHUFFLED_REVIEW.zip`.

'''
    for name in ['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/'thesis_sdd_docs'/name;before=p.read_bytes();text=before.decode('utf-8-sig');(docs/(name+'.before')).write_bytes(before)
        updated=addition+text;p.write_text(updated,encoding='utf8')
        (docs/(name+'.diff')).write_text(''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile=name+'.before',tofile=name)),encoding='utf8')
        shutil.copy2(p,docs/name);docreceipts.append({'file':p.relative_to(ROOT).as_posix(),'before_sha256':file_hash(docs/(name+'.before')),'after_sha256':file_hash(p)})
    write(docs/'CHANGE_RECEIPT.json',docreceipts)
    (OUT/'README.md').write_text(f'''# {TITLE}

Read VALIDATION200_F_VS_S_COMPARISON.md. The wide Visual CSV has200 paired recipient rows; S_VISUAL_COMPONENT_METRICS.csv is the SAME S measurements, not extra cases. FINAL_RO_U_F_S_COMPARISON.csv reuses unchanged M1 means and F Review for S. Do not append duplicate representations.

Inputs are byte-for-byte frozen validation data/predictions. ID/source/token/output/checkpoint/reference hashes, source-only mapping, exact donor provenance, numeric runtime and bootstrap dependence limits are provided. ID-matched recipient F references are used for both F and S. The derived predictions keep original donor pass identity. No fabricated independent inference.

Reproduce: use original workspace paths and pinned Stage M environment; run prepare, score, finalize once in a new output directory. Existing finished outputs are never overwritten. See REPRODUCIBILITY_INPUT_MANIFEST.csv and scripts/. Large backbone/BERT binaries are excluded; their existing exact hashes/runtime pins are recorded.

No new training, LED/Teacher generation, human work or TEST evaluation. TEST100 reference metrics remain PENDING. Package contains no credentials or human state.
''',encoding='utf8')
    # Compact, portable review bundle with exact original validation evidence.
    stage=OUT/'REVIEW_PACKAGE';stage.mkdir()
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,stage/p.name)
    shutil.copytree(docs,stage/'DOCUMENTATION_UPDATES')
    (stage/'INPUTS').mkdir();(stage/'scripts').mkdir()
    for p in [DATA/'validation_filtered.jsonl',HIST/'MERGED/V1_1000/predictions.json']:
        shutil.copy2(p,stage/'INPUTS'/('F_predictions.json' if p.name=='predictions.json' else p.name))
    for name in ['ALL_VALIDATION200_PER_BUSINESS_METRICS.csv','AGGREGATE_VALIDATION200_METRICS.csv','ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json','PREDICTION_ALIGNMENT_AUDIT.json','TARGET_AND_SPLIT_AUDIT.json']:
        shutil.copy2(M1/name,stage/'INPUTS'/('M1_'+name))
    for name in ['stage_m2_validation_shuffled_v1.py','test_stage_m2_validation_shuffled_v1.py','s1000_final_eval_core_v1.py','stage_m_evaluation_v1.py','stage_m1_existing_teacher_metrics_v1.py','stage_m1_pooled_bertscore_v1.py','score_s1000_automatic_prepared_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,stage/'scripts'/name)
    payload=[{'file':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    write_csv(stage/'PACKAGE_MANIFEST.csv',payload)
    archive=OUT/'STAGE_M2_VALIDATION200_SHUFFLED_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory(prefix='m2_verify_') as temp:
        require(z.testzip() is None,'ZIP CRC failed');z.extractall(temp)
        require(len(z.namelist())==len(payload)+1 and len(set(z.namelist()))==len(z.namelist()),'ZIP missing/duplicate files')
        parsed=Counter()
        for row in payload:
            p=Path(temp)/row['file'];require(p.stat().st_size==row['bytes'] and file_hash(p)==row['sha256'],'Extracted member hash/size differs')
            if p.suffix=='.json':read(p);parsed['json']+=1
            elif p.suffix=='.jsonl':jsonl(p);parsed['jsonl']+=1
            elif p.suffix=='.csv':list(csv.DictReader(p.open(encoding='utf-8-sig')));parsed['csv']+=1
            elif p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'));parsed['python']+=1
        manifest_copy=Path(temp)/'PACKAGE_MANIFEST.csv'
        require(manifest_copy.stat().st_size==(stage/'PACKAGE_MANIFEST.csv').stat().st_size and file_hash(manifest_copy)==file_hash(stage/'PACKAGE_MANIFEST.csv'), 'Extracted manifest SHA256/size differs')
        manifest_rows=list(csv.DictReader(manifest_copy.open(encoding='utf8')))
        require(len(manifest_rows)==len(payload) and {r['file'] for r in manifest_rows}=={r['file'] for r in payload}, 'Extracted manifest contents differ')
        exported=list(csv.DictReader((Path(temp)/'VALIDATION200_SHUFFLED_VISUAL_METRICS.csv').open(encoding='utf8')))
        require(len(exported)==len({r['business_id'] for r in exported})==200,'Archive cohort malformed')
    receipt={'status':'PASS_REOPEN_CRC_EXTRACT_EVERY_MEMBER_SHA256_SIZE_PARSE','payload_files':len(payload),'ZIP_members':len(payload)+1,
        'parsed_files':dict(parsed),'businesses':200,'ZIP_bytes':archive.stat().st_size,'ZIP_sha256':file_hash(archive),'no_duplicate_members':True,
        'manifest_member_sha256':file_hash(stage/'PACKAGE_MANIFEST.csv'),'manifest_self_hash_external_to_manifest':True}
    write(OUT/'PACKAGE_INTEGRITY_REPORT.json',receipt)
    (OUT/(archive.name+'.sha256')).write_text(receipt['ZIP_sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps({'comparisons':comparisons,'package':receipt},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['prepare','finish_preparation','score','finalize']);args=parser.parse_args();globals()[args.command]()
