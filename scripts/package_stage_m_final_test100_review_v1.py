"""Report/package already completed Stage M results; never calls a scorer."""
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
from s1000_final_eval_core_v1 import read, write, file_hash, require, digest, reference_content

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_m/final_test100_evaluation_review_20261101_v1'
PACKAGE = OUT/'PACKAGE'
RESULTS = ROOT/'outputs/stage_m/human_primary_scored_v1'
EVIDENCE = ROOT/'outputs/stage_m/human_primary_execution_20261101_v1'
PREP = ROOT/'outputs/final_evaluation/s1000_preparation_v1'
EXPORT = ROOT/'outputs/stage_l/approved_test100_final_export_v1'
METRICS = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False); PACKAGE.mkdir()
    originals = {}; copy_map = {}
    def add(path, member):
        path = Path(path); target = PACKAGE/member
        require(not target.exists(), 'Duplicate destination: '+str(member))
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        require(file_hash(target) == file_hash(path), 'Copy mismatch')
        originals[path.relative_to(ROOT).as_posix()] = file_hash(path)
        copy_map[target.relative_to(PACKAGE).as_posix()] = path.relative_to(ROOT).as_posix()
    for path in RESULTS.iterdir():
        if path.is_file(): add(path, Path('RESULTS_ORIGINAL')/path.name)
    for path in EXPORT.rglob('*'):
        if path.is_file(): add(path, Path('FINAL_REFERENCES')/path.relative_to(EXPORT))
    source = ROOT/'outputs/stage_l/test100_human_references_v1/SOURCE_PACKAGE'
    for path in source.rglob('*'):
        if path.is_file() and (path.name == 'ANNOTATION_PACKAGE_MANIFEST.json' or 'SOURCES' in path.relative_to(source).parts):
            add(path, Path('SOURCE_EVIDENCE')/path.relative_to(source))
    for name in ['COHORT100_FREEZE.json', 'MODEL_AND_DECODER_LOCK.json', 'METRIC_LOCK.json', 'AUTOMATIC_EVALUATION_PROTOCOL.md', 'TEST100_INFERENCE_PLAN.json', 'TEST100_SHUFFLED_INPUT_MANIFEST.json', 'TEST100_SHUFFLED_MAPPING.csv', 'SHUFFLED_POLICY.json', 'requirements-metrics.txt']:
        add(PREP/name, Path('PROTOCOL')/name)
    for condition in ['RO', 'U', 'F', 'S']:
        for name in ['raw_predictions.json', 'COMPLETION.json']:
            add(ROOT/'outputs/stage_k/s1000_test100_phase1_v1/EXECUTION'/condition/name, Path('FROZEN_PREDICTIONS')/condition/name)
    add(ROOT/'outputs/stage_m/s1000_final_automatic_v1/REFERENCE_FREE_REPLAY/TEST100_RO_U_F_S_PREDICTIONS.jsonl', Path('FROZEN_PREDICTIONS/ORIGINAL_400_BUSINESS_RECORDS.jsonl'))
    for name in ['EXECUTION_RELEASE.json', 'PROCESS_RECEIPT.json', 'PROCESS_COMPLETION.json', 'PRE_SCORING_INPUT_VERIFICATION.json', 'PROTECTED_BEFORE_SCORING.json', 'REGRESSION_TEST_RECEIPT.json']:
        add(EVIDENCE/name, Path('REPRODUCIBILITY')/name)
    for name in ['s1000_final_eval_core_v1.py', 'stage_m_evaluation_v1.py', 'score_s1000_automatic_prepared_v1.py', 'stage_k_bert_loader_compat_v1.py', 'validate_s1000_human_references_v1.py', 'test_stage_m_evaluation_v1.py']:
        add(ROOT/'scripts'/name, Path('ORIGINAL_SCRIPTS')/name)
    add(PREP/'METRIC_ASSETS/bert_score-0.3.13-py3-none-any.whl', Path('METRIC_ASSETS/bert_score-0.3.13-py3-none-any.whl'))
    model = ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large'
    lock = read(PREP/'METRIC_LOCK.json')
    for name in lock['bertscore']['small_file_sha256']:
        add(model/name, Path('METRIC_ASSETS/roberta-large')/name)
    for path in [ROOT/'outputs/stage_l/final_reference_independent_review_20261101_v1/PACKAGE/AI_ASSISTANCE_AND_LIMITATIONS.md', ROOT/'outputs/stage_l/final_reference_independent_review_20261101_v1/PACKAGE/AUDIT/READ_ONLY_REFERENCE_AUDIT.json']:
        add(path, Path('REFERENCE_PROVENANCE')/path.name)
    add(ROOT/'scripts/verify_stage_m_final_review_package_v1.py', Path('VERIFY_STAGE_M_PACKAGE.py'))
    # Preserve rubric/sample hashes without distributing separate human-study
    # materials or starting that study.
    for name in ['HUMAN_EVALUATION_PROTOCOL.md', 'HUMAN_EVALUATION_COHORT30.json']:
        originals[(PREP/name).relative_to(ROOT).as_posix()] = file_hash(PREP/name)
    # Existing binaries are hashed read-only, never copied or executed.
    binary_checks = []
    model_lock = read(PREP/'MODEL_AND_DECODER_LOCK.json')
    candidates = [(condition, ROOT/row['path']/'adapter_model.safetensors', row['adapter_sha256']) for condition, row in model_lock['models'].items()]
    candidates += [('LED_BACKBONE', Path(model_lock['backbone']['local_file']), model_lock['backbone']['sha256']), ('BERTSCORE_WEIGHTS', model/'pytorch_model.bin', lock['bertscore']['weights_sha256'])]
    for label, path, expected in candidates:
        if path.is_file():
            actual = file_hash(path); require(actual == expected, 'Existing model file hash differs: '+label)
            originals[path.relative_to(ROOT).as_posix()] = actual
            binary_checks.append({'role': label, 'workspace_path': path.relative_to(ROOT).as_posix(), 'sha256': actual, 'status': 'PASS_READ_ONLY_HASH_MATCH', 'included_in_ZIP': False})
        else:
            binary_checks.append({'role': label, 'recorded_sha256': expected, 'status': 'NOT_RECHECKED_FILE_NOT_LOCAL; recorded native bindings retained', 'included_in_ZIP': False})
    write(PACKAGE/'REPRODUCIBILITY/MODEL_FILE_READ_ONLY_CHECK.json', binary_checks)
    rows = csv_rows(RESULTS/'PER_BUSINESS_COMPONENT_METRICS.csv'); agg = csv_rows(RESULTS/'AGGREGATE_METRICS.csv')
    by = {(r['condition'], r['business_id'], r['task']): r for r in rows}
    native = {}
    for condition in ['RO', 'U', 'F', 'S']:
        for index, r in enumerate(read(PACKAGE/'FROZEN_PREDICTIONS'/condition/'raw_predictions.json')):
            native[condition, r['business_id'], r['task']] = (index, r)
    refs = {r['business_id']: r for r in read(EXPORT/'APPROVED_REFERENCES.json')}
    ids = read(PREP/'COHORT100_FREEZE.json')['business_ids']; source_manifest = read(source/'ANNOTATION_PACKAGE_MANIFEST.json')
    name_by_id = {item['business_id']: read(source/item['source_file'])['business_metadata'].get('name', '') for item in source_manifest['frozen_sources']}
    for index, b in enumerate(ids, 1):
        reference = refs[b]
        case = {'business_order': index, 'business_id': b, 'business_name': name_by_id[b],
            'reference': {'file': f'FINAL_REFERENCES/ANNOTATIONS/{b}.json', 'version': reference['history'][-1]['version'], 'content_sha256': digest(reference_content(reference)), 'source_bundle_sha256': reference['source_bundle_sha256'], 'review_consensus': reference['review_consensus'], 'visual_evidence': reference['visual_evidence']}, 'models': {}}
        for condition in ['RO', 'U', 'F', 'S']:
            case['models'][condition] = {'visual': None} if condition == 'RO' else {}
            for task in (['review'] if condition == 'RO' else ['review', 'visual']):
                row_index, raw = native[condition, b, task]
                expected = '[REVIEW_CONSENSUS]\n' if task == 'review' else '[VISUAL_EVIDENCE]\n'
                case['models'][condition][task] = {'raw_output_text': raw['text'], 'output_sha256': raw['text_sha256'], 'native_file': f'FROZEN_PREDICTIONS/{condition}/raw_predictions.json', 'native_zero_based_array_index': row_index, 'candidate_after_exact_expected_header_removal': raw['text'][len(expected):] if raw['text'].startswith(expected) else raw['text'], 'stored_metric_row': by[condition, b, task]}
        write(PACKAGE/'BUSINESS_CASES'/f'{b}.json', case)
    # New tables below contain arithmetic projections of existing scores only.
    # No ROUGE, BERTScore, bootstrap or sign-flip routine is called.
    paired = read(RESULTS/'PAIRED_COMPARISONS.json')
    differences = []
    for task, other in [('review', 'RO'), ('review', 'U'), ('review', 'S'), ('visual', 'U'), ('visual', 'S')]:
        for b in ids:
            for metric in METRICS:
                f = float(by['F', b, task][metric]); c = float(by[other, b, task][metric])
                differences.append({'business_id': b, 'task': task, 'contrast': 'F-'+other, 'metric': metric, 'score_F': f, 'score_comparator': c, 'paired_difference': f-c, 'interpretation': 'structural exact reuse; no independent test' if task == 'review' and other == 'S' else 'arithmetic projection of existing stored scores'})
    csv_write(PACKAGE/'SUMMARY/DERIVED_BUSINESS_PAIRED_DIFFERENCES.csv', differences)
    for task in ['review', 'visual']:
        title = 'REVIEW_CONSENSUS' if task == 'review' else 'VISUAL_EVIDENCE'
        text = [f'# [{title}]\n', 'Unweighted business macro means, all100 TEST businesses per applicable condition, scale0-1. RO Visual is N/A.\n', '| Condition | n | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERT P | BERT R | BERT F1 |', '|---|---|---|---|---|---|---|---|']
        for r in agg:
            if r['task'] == task: text.append('| '+' | '.join([r['condition'], r['n']]+[f'{float(r[m]):.6f}' for m in METRICS])+' |')
        path = PACKAGE/'SUMMARY'/f'{title}_TABLE.md'; path.parent.mkdir(exist_ok=True); path.write_text('\n'.join(text)+'\n', encoding='utf8')
    primary = [r for r in paired if r['metric'] == 'bertscore_f1']
    lines = ['# Prespecified paired statistical analysis\n', 'F minus comparator; n=100 paired businesses. Stored original results, without resampling/rerunning tests. Percentile95% CIs are unadjusted intervals; Holm adjusts the four BERT F1 test p-values, not these intervals.\n', '| Component | Contrast | Difference | 95% CI | Raw sign-flip p | Holm-adjusted p |', '|---|---|---|---|---|---|']
    for r in primary:
        lines.append(f"| {r['task']} | {r['contrast']} | {r['mean_difference']:.6f} | [{r['CI95'][0]:.6f}, {r['CI95'][1]:.6f}] | {r['signflip_two_sided_p']:.8f} | {r['holm_p_four_prespecified_tests']:.8f} |")
    lines += ['\nAll24 saved contrasts (ROUGE1/2/L and BERT P/R/F1) and their95% paired intervals are unchanged in RESULTS_ORIGINAL/PAIRED_COMPARISONS.json and .csv. Only four BERT F1 tests were prespecified; no new p-values were added for ROUGE or BERT P/R.\n',
        'F-S Review: identical raw text/token IDs and all six scores for every business. Its100 paired differences are all zero. No independent bootstrap interval or hypothesis test is reported; this is a structural identity, not evidence of equivalence between independently generated systems. F-RO Visual: not applicable because RO has no Visual component.\n',
        'The F-RO Review percentile interval narrowly excludes zero, while its unadjusted sign-flip p is0.058319 and its Holm p is0.174957. The procedures are different; the prespecified family-wise test does not establish a difference at0.05. Do not choose whichever procedure favors a desired conclusion.\n',
        'Shuffled is a fixed without-replacement permutation and several components are reused. The business bootstrap/sign-flip outputs retain the prespecified assumptions and must not be interpreted as strong population-level causal evidence.\n']
    (PACKAGE/'SUMMARY/PAIRED_STATISTICAL_READING.md').write_text('\n'.join(lines), encoding='utf8')
    interpretation = '''# Interpretation against the research questions

These conclusions concern agreement with the accepted common TEST100 references. They are not a new model ranking or checkpoint selection, and they do not establish factuality by themselves.

| Question addressed by this automatic evaluation | What the completed results support | What remains unsupported or inconclusive |
|---|---|---|
| Does F improve Review agreement over RO? | F has a slightly lower BERT F1 mean (0.857170 vs0.859479); F-RO=-0.002308. | The four-test Holm-adjusted p=0.174957 does not establish a difference. There is no evidence of F superiority and no equivalence conclusion. |
| Does F improve Review agreement over U? | The means are close (F0.857170, U0.858716); F-U=-0.001546. | CI[-0.004965,0.000745] includes zero; Holm p=0.813384. F superiority is not supported. Review inputs are source-separated, so this is not a direct test of visual conditioning. |
| Does filtering improve Visual agreement over U? | U has higher full100 point means for all three ROUGE F1 metrics and BERT F1 (U0.852471, F0.837365). | BERT F-U CI[-0.041334,0.002773] includes zero; Holm p=0.813384. These results do not establish a reliable BERTScore advantage for either pipeline. Stored ROUGE intervals are descriptive, not new confirmatory tests. |
| Does correct caption/business alignment matter versus Shuffled? | F Visual exceeds S: BERT F1 difference+0.007852, CI[0.005865,0.009825], Holm p approximately0.000080; F also has higher ROUGE means. | This supports an alignment-sensitive reference-agreement signal under the fixed negative control, not proof of original-photo grounding or a population-level causal filtering effect. |
| Does filtering reduce redundancy while retaining information? | F Visual is shorter (51.68 vs67.74 words) with lower mean repeated-trigram fraction (0.060546 vs0.122986) than U. | Lower repetition/fewer claims do not prove better factuality or adequate scene coverage. The common reference includes original-photo information that may be absent from filtered captions. |
| Are conclusions robust to empty/generation-flag cases? | Both F/S empty Visual outputs remain in the primary100. Prespecified98-case sensitivities and two F Review flag sensitivities are already provided. | On the Visual98 subset F-U BERT F1 is+0.001923 with CI[-0.000023,0.003890]. This small, inconclusive secondary difference cannot replace the primary100 or establish F superiority. |
| Do human-reference and Teacher-reference TEST findings agree? | This package reports the completed accepted-reference track only. | No matching completed TEST100 Teacher references are available; no cross-track TEST robustness claim is possible. Historical VALIDATION Teacher scores are not TEST results. |

The central supported result is sensitivity of Visual reference agreement to the correct business-caption pairing (F versus S), with the stated permutation/dependence caveat. The automatic metrics do not establish an overall advantage of F over RO/U or settle factual accuracy, important opinion coverage or human preference. The validation-selected F checkpoint stays fixed; no TEST-driven reselection or optimization was performed.

The frozen models have differing ancestry, supervision and training budgets. A caption-trained initializer must not be called caption-naive RO. U/F are pipeline comparisons, not a clean isolated causal effect of caption filtering. Known original-image reuse affects17 TEST businesses; all100 remain in the main analysis. Reference assistance/provenance and the fixed512-token BERT prefix remain disclosed. Separate human model-output evaluation has not been initiated by this reporting task.
'''
    (PACKAGE/'SUMMARY/INTERPRETATION_AND_RESEARCH_QUESTIONS.md').write_text(interpretation, encoding='utf8')
    methodology = '''# Methodology and reproducibility

- Cohort: exactly100 frozen TEST businesses;700 component score rows from400 condition/business records. Review: RO/U/F/S each100. Visual: U/F/S each100. RO Visual is N/A, with no zero imputation or penalty.
- Reference matching: every model uses the same final approved recipient-business reference for the corresponding component. Shuffled is scored against recipient references, never donor references. Reference versions, content hashes and exact exported files are included. F/S Review is exact reuse; S Visual reuses its frozen F donor output. Reuse is not independent generation.
- ROUGE: rouge-score0.1.2, RougeScorer ROUGE1/2/L with Porter stemming, full component text; per-business P/R/F1 retained and F1 macro means reported. Only the exact expected leading component marker is removed. Wrong headings/content are retained, not repaired. No rougeLsum or outcome-based preprocessing change.
- BERTScore: bert-score0.3.13, FacebookAI/roberta-large revision722cf37b1afa9454edce342e7895e588b6ff1d59, layer17, slow tokenizer, IDF false, no baseline rescaling, CPU FP32, batch8, one thread. Exact package/config/tokenizer/weight hashes remain in METRIC_LOCK and COMPLETION. Standard fixed512-token prefix including special tokens; one F Review output and its S reuse are prefix-truncated, no reference component is truncated. Full raw outputs and full-text ROUGE remain unchanged.
- Empty policy: if either scored component is empty, all six primary similarity scores are zero (including both-empty); rows remain in the main cohort. F and S each have two empty Visual outputs; U has none. No empty reference component is present. Exact source reasons/flags are in raw data and sensitivity files.
- Aggregation: unweighted macro mean by business for each condition/component. There is no arbitrary Review/Visual weighted overall score. F/S duplicated outputs are not independent evidence of better quality.
- Statistics: paired business-index bootstrap,20,000 replicates, numpy default_rng PCG64 seed20261028,95% percentile CIs. Four prespecified BERT F1 contrasts use50,000 paired sign flips, two-sided p=(extreme+1)/(P+1), Holm family-wise adjustment at0.05. Other stored intervals are descriptive; p-values were not invented for secondary metrics. Fixed shuffled pairing and cross-split/repeated image content weaken independent-business/exchangeability assumptions.
- Sensitivities: original full100 results are primary. Existing changed-caption/intervention98 and pre-existing generation-flag exclusions are secondary and explicitly labelled. This reporting task created no new subset and excluded no TEST business from the main results.
- Scope/limits: reference similarity is not verified factuality, visual grounding, completeness or human preference. All-source photo references can expose caption bottleneck omissions. Known17-business image reuse, adapter ancestry/training differences and source-separated Review input design limit causal claims. Stage L acceptance does not erase the documented AI-assisted feedback/evidence/minor editorial provenance; private identity/authorship authentication was not established by local hashes.
- Historical artifacts: frozen protocol/metric-lock software-status fields describe the preparation-time state. The original actual COMPLETION and runtime receipts document the later completed execution. The snapshot named PROTECTED_BEFORE_SCORING was taken after process start and is treated as an in-run observation; the release aggregate/freeze bindings predate launch. These existing records are preserved, not corrected retrospectively.
- Reporting-only verification: no scorer, embedding model, ROUGE computation, bootstrap/sign-flip rerun, generation or training was invoked. The portable audit checks hashes, IDs, unchanged stored values, native/source bindings and arithmetic means/differences. Exact original results are in RESULTS_ORIGINAL. New joined cases/differences are clearly derived conveniences.

## Reproduction boundaries

Run the included portable verifier with Python3.11+: python -B VERIFY_STAGE_M_PACKAGE.py --package .

It uses only the standard library and performs no model or metric execution, network access, downloads or file edits. Existing original scorer/loader/validator/test source and the pinned BERTScore wheel/small tokenizer assets are supplied for methodological inspection. Large LED/BERT model binaries are excluded; their immutable identities/hashes are preserved. Numerical BERTScore reproduction would require those external pinned assets and a separate deliberate execution, not this reporting package. Original12 synthetic regression-test results are preserved; tests were not rerun here.
'''
    (PACKAGE/'SUMMARY/METHODOLOGY.md').write_text(methodology, encoding='utf8')
    (PACKAGE/'README.md').write_text('''# Stage M final TEST100 evaluation review

This is a self-contained review of ALREADY COMPLETED results. No training, LED inference, metric computation, resampling or new research stage was run while preparing it. Stage L's final freeze was accepted for this review as reported by the researcher; original assistance/authenticity limitations are retained.

## Reading order

1. SUMMARY/REVIEW_CONSENSUS_TABLE.md and VISUAL_EVIDENCE_TABLE.md: all100-business macro means.
2. SUMMARY/PAIRED_STATISTICAL_READING.md and INTERPRETATION_AND_RESEARCH_QUESTIONS.md: supported versus inconclusive findings, including the F-S Review identity and RO Visual N/A.
3. SUMMARY/METHODOLOGY.md and RESULTS_ORIGINAL/SENSITIVITY_ANALYSIS.json: frozen metric/statistical rules and labelled secondary results.
4. BUSINESS_CASES/<business_id>.json: all100 businesses, unchanged reference text/version/hash, original raw outputs and original stored score rows for every applicable model/component. RO visual is null. Derived display removes only an exact expected heading, never repairs text.
5. RESULTS_ORIGINAL/: byte-for-byte original final tables,700-row CSV,24 paired contrasts, completion and sensitivity/integrity records. FROZEN_PREDICTIONS/ contains native original text/token IDs/inputs/receipts and the existing400-record view. FINAL_REFERENCES/ is the exact export used by Stage M. SOURCE_EVIDENCE/ holds selected reviews and photo identifiers/hashes; JPEGs are excluded.
6. PROTOCOL/, ORIGINAL_SCRIPTS/, METRIC_ASSETS/, REPRODUCIBILITY/ and AUDIT/: unchanged settings/code/runtime evidence and the new reporting-only audit. Large checkpoints/scorer weights, private DB/credentials, private participant details and separate Human Evaluation materials are excluded. Do not send this scored/model-identified package to blinded raters.

To verify after extraction, run: python -B VERIFY_STAGE_M_PACKAGE.py --package .

The verifier uses only Python's standard library, checks hashes/IDs/arithmetic and never evaluates models or reruns metrics/statistical tests. PACKAGE_MANIFEST.json hashes every payload file except itself; the archive is separately SHA256-hashed and verified after extraction. Full numerical scorer reproduction is not bundled with large model weights and is not automatically initiated.

Main interpretation: F has higher Visual agreement than Shuffled under the prespecified analysis; automatic metrics do not establish F superiority over RO/U. All100 remain primary, including empty/flagged outputs. No Human Evaluation, Qwen/cross-dataset comparison, model selection or further experiment is initiated. Stop for independent ChatGPT review.
''', encoding='utf8')
    spec = importlib.util.spec_from_file_location('stage_m_reporting_audit', PACKAGE/'VERIFY_STAGE_M_PACKAGE.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    audit = verifier.audit(PACKAGE, package_manifest=False)
    require(not audit['failed_checks'], 'Reporting audit failed: '+str(audit['failed_checks']))
    write(PACKAGE/'AUDIT/REPORTING_ONLY_AUDIT.json', audit)
    require(all(file_hash(ROOT/name) == value for name, value in originals.items()), 'Original artifact changed')
    write(PACKAGE/'AUDIT/ORIGINAL_PRESERVATION_AND_COPY_MAP.json', {'original_file_sha256': originals, 'byte_identical_copy_mapping': copy_map, 'all_original_files_unchanged_after_reporting': True, 'new_training_runs': 0, 'new_inference_runs': 0, 'new_evaluation_runs': 0, 'new_bootstrap_or_signflip_runs': 0, 'reference_or_approval_edits': 0, 'human_evaluation_started': False})
    files = {p.relative_to(PACKAGE).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
    write(PACKAGE/'PACKAGE_MANIFEST.json', {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'files': files, 'manifest_self_hash': 'Archive checksum sidecar'})
    archive = OUT/'STAGE_M_FINAL_TEST100_EVALUATION_REVIEW.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(PACKAGE).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_m_final_review_') as tmp:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist(); require(len(names) == len(set(names)) and set(names) == set(files) | {'PACKAGE_MANIFEST.json'}, 'ZIP member set')
            require(z.testzip() is None, 'ZIP CRC failure'); z.extractall(tmp)
        for name, bind in files.items():
            p = Path(tmp)/name
            require(p.stat().st_size == bind['bytes'] and file_hash(p) == bind['sha256'], 'Extracted member hash differs')
        require(file_hash(Path(tmp)/'PACKAGE_MANIFEST.json') == file_hash(PACKAGE/'PACKAGE_MANIFEST.json'), 'Manifest hash differs')
        run = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(tmp)/'VERIFY_STAGE_M_PACKAGE.py'), '--package', tmp], capture_output=True, text=True, encoding='utf8')
        (OUT/'EXTRACTED_PORTABLE_AUDIT.json').write_text(run.stdout, encoding='utf8')
        (OUT/'EXTRACTED_PORTABLE_AUDIT_STDERR.log').write_text(run.stderr, encoding='utf8')
        require(run.returncode == 0, 'Extracted portable audit failed: '+run.stderr)
        require(not json.loads(run.stdout)['failed_checks'], 'Extracted portable audit flags')
    require(all(file_hash(ROOT/name) == value for name, value in originals.items()), 'Original artifact changed during ZIP verification')
    checksum = file_hash(archive)
    (OUT/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n', encoding='ascii')
    write(OUT/'DELIVERY_VERIFICATION.json', {'status': 'PASS', 'members': len(files)+1, 'archive_bytes': archive.stat().st_size, 'archive_sha256': checksum, 'all_members_extracted_and_sha256_verified': True, 'portable_audit_exit_code': run.returncode, 'businesses': 100, 'score_rows': 700, 'native_raw_component_records': 700, 'business_case_files': 100, 'original_files_preserved': len(originals), 'new_evaluation_training_or_inference_runs': 0, 'new_human_evaluation_or_other_research_stages': 0})
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': checksum, 'members': len(files)+1, 'audit': audit['status']}, ensure_ascii=False))


if __name__ == '__main__': main()
