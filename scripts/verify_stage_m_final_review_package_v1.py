"""Offline integrity/arithmetic audit only; no metric/model/statistical reruns."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
METRICS = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
CONTRASTS = [('review', 'F-RO'), ('review', 'F-U'), ('visual', 'F-U'), ('visual', 'F-S')]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def audit(root, package_manifest=True):
    root = Path(root).resolve(); checks = {}; failures = []
    def check(name, passed, detail):
        checks[name] = {'status': 'PASS' if passed else 'FAIL', 'detail': detail}
        if not passed: failures.append(name)
    original = root/'RESULTS_ORIGINAL'
    done = read(original/'COMPLETION.json')
    ids = read(root/'PROTOCOL/COHORT100_FREEZE.json')['business_ids']
    lock = read(root/'PROTOCOL/METRIC_LOCK.json')
    model_lock = read(root/'PROTOCOL/MODEL_AND_DECODER_LOCK.json')
    check('TEST100_membership_binding', len(ids) == len(set(ids)) == 100 and sha(root/'PROTOCOL/COHORT100_FREEZE.json') == done['prediction_integrity']['preparation_bindings']['COHORT100_FREEZE.json'], {'businesses': len(ids)})
    check('completed_metric_file_and_evaluator_hashes', done['status'] == 'COMPLETED_MEASURED_TEST100' and sha(original/'PER_BUSINESS_COMPONENT_METRICS.csv') == done['metric_file_sha256'] and sha(root/'ORIGINAL_SCRIPTS/stage_m_evaluation_v1.py') == done['evaluator_sha256'], 'Unchanged recorded completion receipt; no scoring invoked')
    check('metric_lock_unchanged', sha(root/'PROTOCOL/METRIC_LOCK.json') == done['runtime']['metric_lock_sha256'], {'seed': lock['statistics']['seed'], 'bootstrap_replicates': lock['statistics']['bootstrap_replicates']})
    refs = read(root/'FINAL_REFERENCES/APPROVED_REFERENCES.json')
    freeze = read(root/'FINAL_REFERENCES/REFERENCE_FREEZE.json')
    byref = {r['business_id']: r for r in refs}
    ref_content = lambda r: {k: r[k] for k in ['business_id', 'source_bundle_sha256', 'review_consensus', 'visual_evidence', 'review_claims', 'visual_claims']}
    ref_hash = {b: digest(ref_content(r)) for b, r in byref.items()}
    check('same_final_100_approved_references', len(refs) == len(byref) == 100 and set(byref) == set(ids) and all(r['status'] == 'APPROVED' and r['checker']['reviewed_content_sha256'] == ref_hash[b] for b, r in byref.items()) and sha(root/'FINAL_REFERENCES/APPROVED_REFERENCES.json') == freeze['reference_file_sha256'] and freeze['content_hashes'] == ref_hash, 'Same recipient reference for every applicable model')
    check('Stage_M_reference_input_hashes', all((root/'FINAL_REFERENCES'/name).is_file() and sha(root/'FINAL_REFERENCES'/name) == value for name, value in done['reference_files'].items()), {'json_files': len(done['reference_files'])})
    plan = read(root/'PROTOCOL/TEST100_INFERENCE_PLAN.json')
    calls = {(r['condition'], r['business_id'], r['task']): r for r in plan['calls']}
    native = {}; raw_ok = len(calls) == 700
    for condition in ['RO', 'U', 'F', 'S']:
        folder = root/'FROZEN_PREDICTIONS'/condition
        receipt = read(folder/'COMPLETION.json'); raw = read(folder/'raw_predictions.json')
        raw_ok &= receipt['raw_predictions_sha256'] == sha(folder/'raw_predictions.json') and receipt['plan_sha256'] == sha(root/'PROTOCOL/TEST100_INFERENCE_PLAN.json')
        raw_ok &= len(raw) == (100 if condition == 'RO' else 200)
        raw_ok &= sha(folder/'COMPLETION.json') == done['prediction_integrity']['condition_bindings'][condition]['completion_sha256']
        raw_ok &= sha(folder/'raw_predictions.json') == done['prediction_integrity']['condition_bindings'][condition]['raw_predictions_sha256']
        for r in raw:
            key = (condition, r['business_id'], r['task'])
            raw_ok &= key not in native and key in calls
            c = calls[key]; native[key] = r
            raw_ok &= hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
            raw_ok &= r['input_ids'] == c['input_ids'] and digest(r['input_ids']) == r['input_ids_sha256']
            raw_ok &= all(r[k] == c[k] for k in ['source_sha256', 'checkpoint_sha256', 'input_ids_sha256'])
            raw_ok &= hashlib.sha256(c['source'].encode()).hexdigest() == c['source_sha256']
            raw_ok &= r['generation_settings'] == model_lock['generation'] and r['generation_config_sha256'] == digest(model_lock['generation'])
    check('original_native_outputs_receipts_inputs_decoder_preserved', bool(raw_ok) and len(native) == 700, {'native_components': len(native), 'token_text_decoding': 'Existing completion integrity proof retained; no new tokenizer/model execution'})
    mapping = read(root/'PROTOCOL/TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings']
    reuse_ok = True
    for b in ids:
        f = native['F', b, 'review']; s = native['S', b, 'review']
        reuse_ok &= f['text'] == s['text'] and f['token_ids'] == s['token_ids']
    for item in mapping:
        b = item['recipient_id']; donor = item['donor_id'] or b
        f = native['F', donor, 'visual']; s = native['S', b, 'visual']
        reuse_ok &= f['text'] == s['text'] and f['token_ids'] == s['token_ids']
    check('F_S_structural_reuse_and_visual_donor_mapping', bool(reuse_ok), {'review_reused_pairs': 100, 'visual_F_donor_pairs': len(mapping)})
    rows = csv_rows(original/'PER_BUSINESS_COMPONENT_METRICS.csv')
    by = {(r['condition'], r['business_id'], r['task']): r for r in rows}
    check('all_700_unique_metric_rows_no_RO_visual', len(rows) == len(by) == 700 and set(by) == set(native) and not any(k[0] == 'RO' and k[2] == 'visual' for k in by), {'review_rows': sum(r['task'] == 'review' for r in rows), 'visual_rows': sum(r['task'] == 'visual' for r in rows)})
    row_bindings = True; finite = True; empty_policy = True
    for key, r in by.items():
        raw = native[key]
        row_bindings &= r['reference_business_id'] == r['business_id'] and r['reference_sha256'] == ref_hash[r['business_id']]
        row_bindings &= all(r[k] == raw[k] for k in ['checkpoint_sha256', 'source_sha256']) and r['output_sha256'] == raw['text_sha256']
        finite &= all(math.isfinite(float(r[m])) for m in METRICS)
        if r['empty_candidate'] == 'True' or r['empty_reference'] == 'True':
            empty_policy &= all(float(r[m]) == 0 for m in METRICS)
    check('every_score_binds_original_output_and_recipient_reference', bool(row_bindings), {'rows': len(rows)})
    check('finite_scores_and_original_empty_output_policy', bool(finite and empty_policy), {'empty_candidate_rows': sum(r['empty_candidate'] == 'True' for r in rows), 'empty_rows_not_removed': True})
    aggregate = csv_rows(original/'AGGREGATE_METRICS.csv'); agg_ok = len(aggregate) == 7
    for r in aggregate:
        group = [x for x in rows if x['condition'] == r['condition'] and x['task'] == r['task']]
        agg_ok &= len(group) == int(r['n']) == 100 and {x['business_id'] for x in group} == set(ids)
        agg_ok &= all(abs(sum(float(x[m]) for x in group)/100-float(r[m])) < 1e-12 for m in METRICS)
    check('stored_aggregate_arithmetic_no_business_exclusions', bool(agg_ok), {'aggregate_groups': len(aggregate), 'n_per_applicable_group': 100})
    pairs = read(original/'PAIRED_COMPARISONS.json'); pair_ok = len(pairs) == 24
    pair_ok &= {(r['task'], r['contrast'], r['metric']) for r in pairs} == {(t, c, m) for t, c in CONTRASTS for m in METRICS}
    for p in pairs:
        other = p['contrast'].split('-')[1]
        value = sum(float(by['F', b, p['task']][p['metric']])-float(by[other, b, p['task']][p['metric']]) for b in ids)/100
        pair_ok &= abs(value-p['mean_difference']) < 1e-12 and p['n'] == 100
        pair_ok &= p['seed'] == lock['statistics']['seed'] and p['bootstrap_replicates'] == lock['statistics']['bootstrap_replicates']
        pair_ok &= len(p['CI95']) == 2 and all(math.isfinite(x) for x in p['CI95']) and p['CI95'][0] <= p['CI95'][1]
    primary = [p for p in pairs if p['metric'] == 'bertscore_f1']
    pair_ok &= len(primary) == 4 and all(p['signflip_replicates'] == 50000 and 'holm_p_four_prespecified_tests' in p for p in primary)
    pair_ok &= all('signflip_two_sided_p' not in p for p in pairs if p['metric'] != 'bertscore_f1')
    check('stored_paired_differences_and_prespecified_test_metadata', bool(pair_ok), {'stored_CIs_and_p_values_not_recomputed': True, 'paired_contrasts': len(pairs), 'primary_test_family': 4})
    equal = all(by['F', b, 'review'][m] == by['S', b, 'review'][m] for b in ids for m in METRICS)
    check('F_S_Review_scores_identical_no_independent_test', equal, {'zero_paired_differences': 600})
    cases_ok = True
    for b in ids:
        case = read(root/'BUSINESS_CASES'/f'{b}.json')
        cases_ok &= case['business_id'] == b and case['reference']['content_sha256'] == ref_hash[b]
        cases_ok &= case['reference']['review_consensus'] == byref[b]['review_consensus'] and case['reference']['visual_evidence'] == byref[b]['visual_evidence']
        for condition in ['RO', 'U', 'F', 'S']:
            for task in (['review'] if condition == 'RO' else ['review', 'visual']):
                item = case['models'][condition][task]; key = condition, b, task
                cases_ok &= item['raw_output_text'] == native[key]['text'] and item['stored_metric_row'] == by[key]
        cases_ok &= case['models']['RO']['visual'] is None
    check('all_100_readable_business_joins_exact', bool(cases_ok), {'business_files': len(list((root/'BUSINESS_CASES').glob('*.json')))})
    if package_manifest:
        manifest = read(root/'PACKAGE_MANIFEST.json')['files']
        check('package_member_SHA256_and_sizes', all((root/name).is_file() and (root/name).stat().st_size == bind['bytes'] and sha(root/name) == bind['sha256'] for name, bind in manifest.items()), {'files_checked': len(manifest)})
    return {'status': 'PASS_REPORTING_INTEGRITY' if not failures else 'FAIL', 'failed_checks': failures, 'checks': checks,
        'execution_scope': 'Hash, identifier and arithmetic checks on existing scores; no ROUGE/BERTScore/embedding, bootstrap, sign-flip, generation or training calls',
        'not_reestablished': ['Human identity/authorship authenticity; see Stage L provenance', 'New independent numerical BERTScore reproduction (not requested)', 'Semantic factuality or photo grounding', 'Population causal validity of the fixed shuffled mapping']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    result = audit(p.parse_args().package)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['failed_checks'] else 0)
