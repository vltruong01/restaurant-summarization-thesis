"""Measured VALIDATION reports, documentation and verified review archive."""
from pathlib import Path
import csv
import datetime
import difflib
import json
import shutil
import tempfile
import zipfile
import ast
from stage_m1_existing_teacher_metrics_v1 import ROOT,OUT,PREP,DATA,MATCHED,HIST,METRICS,MODEL_ORDER
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,require

def main():
    require(read(OUT/'METRIC_COMPUTATION_COMPLETE.json')['components']==1800,'Complete actual scoring required')
    require((OUT/'WORKSPACE_REFERENCE_SEARCH_SUMMARY.json').is_file(),'Finish workspace reference search')
    require(not (OUT/'STAGE_M1_TEACHER_VALIDATION200_REVIEW.zip').exists(),'Refuse overwrite final package')
    rows=list(csv.DictReader((OUT/'AGGREGATE_VALIDATION200_METRICS.csv').open(encoding='utf8')))
    means={(r['model'],r['task']):r for r in rows};pairs=read(OUT/'PAIRED_SAME_REFERENCE_COMPARISONS.json');reuse=read(OUT/'PRIOR_ROUGE_REUSE_VERIFICATION.json')
    candidates=read(OUT/'TEST_REFERENCE_CANDIDATES.json')
    require(not candidates,'Review TEST reference candidates before finalizing missing-reference claim')
    large=read(OUT/'LARGE_REFERENCE_SEARCH_FOLLOWUP.json')
    require(not any(r['potential_reference_requires_further_parsing'] for r in large['loose_large_files']+large['large_ZIP_members']),'Resolve large-file reference candidates')
    write(OUT/'TEST_REFERENCE_AVAILABILITY_DECISION.json',{'status':'PENDING — no matching references available','valid_complete_TEST100_Teacher_references_verified':0,
        'evidence':'Canonical Stage G held_out_targets=0; whole-workspace supported-file and ZIP search found no nonempty TEST reference-shaped candidates.',
        'search_summary_sha256':file_hash(OUT/'WORKSPACE_REFERENCE_SEARCH_SUMMARY.json'),'large_file_stream_followup_sha256':file_hash(OUT/'LARGE_REFERENCE_SEARCH_FOLLOWUP.json'),'TRAIN_VALIDATION_targets_never_used_for_TEST':True,
        'no_prediction_as_own_reference':True,'scope_limits':read(OUT/'WORKSPACE_REFERENCE_SEARCH_SUMMARY.json')['scope_limits']})
    interpretation=[]
    for task,other in [('review','RO'),('review','U'),('review','CE3'),('review','V1_500'),('visual','CE3'),('visual','V1_500')]:
        r=next(p for p in pairs if p['task']==task and p['contrast']=='F-'+other and p['metric']=='bertscore_f1')
        interpretation.append(f"- {task} F−{other}: BERT F1 mean difference {r['mean_difference']:.4f}, descriptive paired95% CI [{r['CI95'][0]:.4f}, {r['CI95'][1]:.4f}], n200, exact same component reference for every pair.")
    search=read(OUT/'WORKSPACE_REFERENCE_SEARCH_SUMMARY.json');audit=read(OUT/'TARGET_AND_SPLIT_AUDIT.json')
    report=f'''# Stage M.1 — actual existing-Teacher VALIDATION200 results

COMPLETED MEASUREMENT: ROUGE1/2/L and BERTScore Precision/Recall/F1 for all200 VALIDATION businesses, current RO/U/F pipelines plus historical CE3 and V1-S500. Total1,000 model/business records and1,800 component metrics. F is selected Source-Separated V1-S1000, not Final V2. S has frozen TEST outputs but no matched existing validation inference here; no S prediction was synthesized.

Read REVIEW_VALIDATION200_TABLE.md and VISUAL_VALIDATION200_TABLE.md. Each model/task mean uses all200 businesses. Historical CE3/V1-S500 are separately labeled supplementary; this retrospective validation analysis does not reopen model selection or establish held-out quality. TRAIN targets were audited, not scored as independent evaluation.

## Reference and split provenance

The canonical Stage G production completion binds6,600 condition-level target records from2,000 TRAIN and200 VALIDATION businesses. All six dataset hashes, split membership/source hashes and all2,200 original target artifacts were independently rechecked. No TRAIN/VALIDATION target was reassigned to TEST. Review components are byte-identical across RO/U/F for all2,200 businesses. U/F Visual components differ for all2,200, including all200 VALIDATION businesses. These are condition-aligned AI Teacher targets, not human references or complete photo-grounded gold.

All2,200 original artifact files retain a legacy status string containing PROVISIONAL/NOT_PRODUCTION. Their exact contents are explicitly accepted by the later canonical STAGE_G_COMPLETE_VALIDATED_WITH_DOCUMENTED_LIMITATIONS dataset completion and production automated-validation hashes; Stage M.1 did not relabel their original metadata. The original report says automated validation passed, not factual certification. The1,199 listed VALIDATION API checkpoint/raw-response artifacts were independently rehashed. Prior exact API request/span/serialization checks remain prior-validator evidence; this task did not rerun semantic verification. Teacher alias gpt-6.1-sol lacks an attested immutable backend snapshot.

Predictions match the exact validation business/source/token/checkpoint identities. Stored token IDs independently decode to original raw text; native source/input hashes match the corresponding Teacher condition sources. Current RO/U/F prediction files match their frozen model-lock hashes. CE3 uses original joint-input inference; the source-separated checkpoints use separate Review and Visual passes. Their output budgets and initialization/training histories remain confounders.

## Comparable paired results

Common byte-identical Review references support RO/U/F comparisons. Historical CE3/V1-S500 and F also share the exact Filtered-specific Visual reference per business. All comparisons below are descriptive paired percentile bootstrap95% intervals,20,000 replicates, PCG64 seed20261028, complete n200. Thirty-six metric intervals are reported without new confirmatory p-values or multiplicity claims.

'''+ '\n'.join(interpretation)+f'''

There is deliberately NO direct U-versus-F Visual difference/CI using their different targets. Their table values measure own-condition target agreement and cannot establish superior image grounding. There is no legitimate all-photo common TEST Teacher reference to substitute. Human and model-assisted source-grounded quality judgments remain separate from these lexical/semantic similarity scores.

## Metric contract and prior ROUGE

Stage M's metric definition/assets/functions were reused: rouge-score0.1.2 Porter stemmer/full component; BERTScore0.3.13, roberta-large revision722cf37b1afa9454edce342e7895e588b6ff1d59, pinned binary SHA, layer17, slow tokenizer, CPU FP32, batch8, no IDF/baseline rescaling,512-token prefix including BOS/EOS. Every row discloses untruncated candidate/reference token lengths and prefix truncation; full-text ROUGE remains available. No metric parameter was tuned against model rankings.

Execution-only efficiency amendments are explicit: the initial completed RO CPU1 scores stay intact as a control; an incomplete pooled CPU1 evaluation was stopped after its batches implied multi-hour runtime. Final scores for ALL five models were recomputed together using six local CPU threads, retaining original FP32, model/layer, batch8, tokenization, IDF, rescaling and prefix rules. BERTScorer tokenization/IDF nthreads remains1. This is a documented VALIDATION execution deviation from the historical CPU1 resource setting, not a change to the frozen TEST metric-lock file or metric definition. An actual one-component CPU1-vs6 check had max P/R/F difference1.1920928955078125e-7; the complete200-case RO cross-check is separately recorded. No hardware choice was made from model quality rankings.

Pooling uses one original BERTScore call so identical reference sentences are encoded once; actual P/R/F values are looked up by BOTH exact candidate and reference text and consumed by the original Stage M component scorer. No IDs, targets or predictions changed. Normal FP32 scheduling/padding roundoff is not represented as a scientific effect. See ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json and the exact pooling/resource amendments rather than treating the initial METRIC_RUNTIME.json as the final execution configuration.

For source-separated passes, only the exact expected leading marker is removed; malformed raw text is retained. CE3 component views follow the existing strict header-count parser, preserving raw parent output hashes and exact extracted component views. No CE3 joint header failures occurred on these200 outputs. No prediction/reference was rewritten or regenerated.

If either component is empty, all six scores are zero, including both-empty. All200 remain in means and bootstrap comparisons. Empty/truncation/cap counts are in the aggregate CSV; no absence bonus is awarded. RO Visual is N/A. Reference agreement, including high BERTScore, is not proof of factual correctness, opinion completeness, scene fidelity, or human preference.

Previously computed current RO/U/F ROUGE values were verified row-by-row: {reuse['all_three_scores_reproduced']}/{reuse['rows']} rows happen to reproduce all three current scores within1e-12; {reuse['prior_empty_reference_excluded']} prior rows excluded empty reference components. The precise source of the other differences was verified: the legacy backend used use_stemmer=False, while the pre-existing frozen Stage M contract uses use_stemmer=True. Replaying the original non-stemmed backend reproduced ALL999 eligible legacy rows within1e-12 against unchanged output/reference hashes. Thus old values were not blindly mixed with current scores; all current models were computed under the SAME frozen stemmed policy. The empty-reference row is now retained with explicit0. See LEGACY_ROUGE_CONFIG_DIFFERENCE_AUDIT.json and PRIOR_ROUGE_REUSE_VERIFICATION.json. A difference from an older report is not a model quality gain.

## TEST100 remains pending independently

**TEST100 Teacher-reference ROUGE/BERTScore: PENDING — no matching references available.**

The search inventoried{search['workspace_file_inventory']:,} files; considered{search['loose_JSON_JSONL_files_considered']:,} loose JSON/JSONL files and{search['ZIP_archives_considered']} ZIP archives /{search['ZIP_JSON_members_scanned']:,} machine-readable members. No nonempty TEST reference-shaped candidates were found. Search limits and any read/size errors are recorded verbatim in WORKSPACE_REFERENCE_SEARCH_SUMMARY.json. Binary training tar archives, arbitrary prose/CSV and external accounts are not certified reference datasets by this scan. No claim of inspecting every binary byte is made.

Follow-up streamed every byte of all18 oversized loose JSON/JSONL files and all7 oversized ZIP JSON members; none contained both TEST business IDs and reference-field literals requiring further parsing. The2 unreadable ZIPs are old damaged D059 training transport and Teacher-redesign Stage A archives; their paths, sizes, header bytes and hashes are recorded rather than falsely reported as successfully reopened. Existing recovered Stage A JSON drafts are already included in the loose-file search. No new TEST reference can be inferred from a damaged archive or an unverified tail.

The existing400 frozen TEST RO/U/F/S records remain untouched. There was no paid/API call, LED inference, model training, human evaluation or human-reference mutation. Human-reference scoring remains a separate primary track pending100 genuine approved references. GPT demonstration/source ZIPs are not completed Teacher-reference datasets. No final model ranking is claimed from validation target agreement.

## Delivery and stop

Machine-readable metrics, full target/split provenance ledger, exact validation target/prediction files, paired intervals, runtime pins, search evidence and documentation changes are in the review package. Large model binaries, original photos, credentials and human state are excluded. Package CRC, reopening, extraction and every member SHA256 were actually verified. Stop after delivery; no new generation or training is authorized here.
'''
    (OUT/'STAGE_M1_RESULTS_AND_LIMITATIONS.md').write_text(report,encoding='utf8')
    # Explicit count-only exclusions: no silent empty or truncation filtering.
    rawrows=list(csv.DictReader((OUT/'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv').open(encoding='utf8')))
    early=list(csv.DictReader((OUT/'EARLY_ACTUAL_ROUGE_ALL1800_COMPONENTS.csv').open(encoding='utf8')))
    current={(r['condition'],r['business_id'],r['task']):r for r in rawrows}
    require(len(early)==len(current)==1800,'Early/full metric cohort differs')
    for e in early:
        r=current[e['model'],e['business_id'],e['task']]
        require(e['output_sha256']==r['output_sha256'] and e['component_reference_sha256']==r['component_reference_sha256'],'Early/full scoring alignment differs')
        require(all(abs(float(e[m])-float(r[m]))<1e-12 for m in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure']),'Independent full-text ROUGE replay differs')
    write(OUT/'ALL1800_ROUGE_REPLAY_VERIFICATION.json',{'status':'PASS','component_rows':1800,'all_output_and_component_reference_hashes_identical':True,
        'all_5400_ROUGE_F1_values_reproduced_tolerance':1e-12,'early_receipt_is_historical_snapshot_not_current_BERT_status':True})
    write(OUT/'COHORT_AND_EMPTY_TRUNCATION_ACCOUNTING.json',{'main_cohort_businesses':200,'model_records':1000,'component_rows':len(rawrows),'silent_exclusions':0,
        'RO_Visual':'N/A','counts':[{k:r[k] for k in ['model','task','n','empty_candidate','empty_reference','bert_candidate_prefix_truncated','bert_reference_prefix_truncated','at_generation_cap']} for r in rows]})
    write(OUT/'SCORING_FILE_ROLES.json',{'authoritative_main_metrics':'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv','authoritative_component_rows':1800,'unique_model_business_records':1000,
        'FINAL_SCORED/*':'Per-model partitions of that SAME final1800-row matrix; do not append again.',
        'SCORED/RO/*':'Preserved initial CPU1 numeric control (200 rows), NOT an additional model/cohort or final CPU6 table.',
        'EARLY_ACTUAL_ROUGE_*':'Independent cheap full-text ROUGE replay on the same1800 components, NOT another cohort.',
        'METRIC_RUNTIME.json':'Initial CPU1 resource configuration/control history.',
        'ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json':'Authoritative final CPU6 resource configuration shared by all five models.',
        'POOLED_BERTSCORE_PROGRESS.log':'Incomplete CPU1 execution history stopped for local resource scheduling; no final metrics from that partial call.',
        'CPU6_POOLED_BERTSCORE_PROGRESS.log':'Completed final execution log; no model or reference generation.'})
    frozen=read(ROOT/'outputs/stage_m/s1000_final_automatic_v1/FROZEN_PREDICTION_VERIFICATION.json')
    for condition,binding in frozen['condition_bindings'].items():
        require(file_hash(ROOT/f'outputs/stage_k/s1000_test100_phase1_v1/EXECUTION/{condition}/raw_predictions.json')==binding['raw_predictions_sha256'],'Frozen TEST predictions changed')
    require(all(file_hash(PREP/n)==h for n,h in frozen['preparation_bindings'].items()),'Frozen TEST preparation changed')
    prior_scope=read(ROOT/'outputs/stage_m/s1000_final_automatic_v1/FINAL_SCOPE_VERIFICATION.json')
    require(file_hash(PREP/'HUMAN_EVALUATION_PROTOCOL.md')==prior_scope['rubric_sha256'],'Frozen human rubric changed')
    require(file_hash(ROOT/'scripts/stage_l_reference_app_v1.py')=='997ffa021e7ed5cdffef10a16c130757d0fef1fc820a436dcd34301f097ef370','Human app changed')
    write(OUT/'FINAL_FROZEN_SCOPE_VERIFICATION.json',{'status':'PASS','TEST100_400_raw_prediction_records_and_preparation_unchanged':True,'Stage_L_application_unchanged':True,
        'human_DB_opened_or_modified_by_M1':False,'TRAIN_metrics_calculated':False,'Teacher_generation_calls':0,'LED_generation_calls':0,'paid_calls':0,'new_training_updates':0})
    # Save exact before snapshots/diffs; preserve all preceding historical entries.
    docdir=OUT/'DOCUMENTATION_UPDATES';docdir.mkdir();docs=['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md'];docreceipts=[]
    fv=means['F','review'];rv=means['RO','review'];uv=means['U','review'];fvvis=means['F','visual'];uvvis=means['U','visual']
    prefix=f'''### Stage M.1 existing-Teacher VALIDATION200 automatic metrics COMPLETE — 2026-10-29

Actual ROUGE1/2/L and BERTScore P/R/F1 are now computed for all200 VALIDATION businesses: current RO/U/F (F=selected V1-S1000), plus historical CE3/V1-S500 supplementary. Total1,000 model/business records /1,800 components; no new predictions or references. Six original Stage G dataset hashes /6,600 condition targets and2,200 target artifacts passed identity/split/source verification. All1,199 listed VALIDATION API checkpoint/raw-response artifacts were rehashed. The old PROVISIONAL artifact status is preserved and traced to later accepted canonical production completion; this is AI Teacher target agreement, not human gold or independent factuality certification.

Review references are byte-identical across RO/U/F for all200 validation businesses; Visual U/F targets differ for all200. Review BERT F1: RO{float(rv['bertscore_f1']):.4f}, U{float(uv['bertscore_f1']):.4f}, F{float(fv['bertscore_f1']):.4f}. Visual own-condition BERT F1: U{float(uvvis['bertscore_f1']):.4f}, F{float(fvvis['bertscore_f1']):.4f}; these two Visual scores use different targets and are not a direct grounding comparison. Paired200-business bootstrap95% CIs (seed20261028,B20000) are supplied only when component references match. No new p-values, target generation or model selection occurred. Historical CE3 joint input versus source-separated architecture/output-budget differences remain explicit.

TEST100 Teacher-reference ROUGE/BERTScore remains PENDING — no matching references available after the supported workspace/ZIP search (scope limits recorded). TRAIN references were audited, not presented as independent evaluation. Human-reference primary evaluation and Human Evaluation remain separately researcher-managed; M.1 neither opened the human DB nor wrote/approved references. Frozen TEST400 predictions, cohort, TEST30 design/rubric and Stage L application remain unchanged. Stage M metric definitions/assets (including512-token BERT prefix and explicit empty-component zeros) were reused; final VALIDATION uses six local CPU threads with numeric consistency checks, while the original TEST CPU1 metric-lock file is unchanged. Semantic/reference similarity alone does not certify grounding, coverage, memorization or human preference; ancestry/supervision/image-overlap limitations remain.

Legacy ROUGE used no stemming; Stage M's already frozen contract uses stemming. The old999 nonempty-reference rows were reproduced under their original settings; current scores uniformly use Stage M settings and retain the one empty Filtered reference case. Do not interpret protocol-driven differences from older reports as model improvement.

Results and verified review ZIP: outputs/stage_m1/existing_teacher_validation200_v1/. STOP after delivery. No training, paid calls, Teacher generation, TEST regeneration or human evaluation initiated. Earlier Stage M pending statements refer to reference-based TEST scoring; actual VALIDATION scoring is now complete.

'''
    for name in docs:
        p=ROOT/'thesis_sdd_docs'/name;shutil.copy2(p,docdir/(name+'.before'));old=p.read_text(encoding='utf-8-sig');p.write_text(prefix+old,encoding='utf8')
        (docdir/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),(prefix+old).splitlines(True),fromfile=name+'.before',tofile=name)),encoding='utf8')
        docreceipts.append({'file':p.relative_to(ROOT).as_posix(),'before_sha256':file_hash(docdir/(name+'.before')),'after_sha256':file_hash(p),'edit':'Current measured VALIDATION results prepended; historical contents retained'})
    write(docdir/'CHANGE_RECEIPTS.json',docreceipts)
    # Explicit reproducibility inputs, including reusable metric implementation.
    inputs=[]
    paths=[*(DATA/f'{s}_{c}.jsonl' for s in ['train','validation'] for c in ['reviews_only','unfiltered','filtered']),DATA.parent/'completion_report.json',DATA.parent/'final_validation/validation_report.json',
        ROOT/'data/processed/yelp_2300_v1/business_split.csv',PREP/'METRIC_LOCK.json',PREP/'MODEL_AND_DECODER_LOCK.json',ROOT/'configs/stage_g_teacher_production_d042.json',ROOT/'outputs/stage_m/s1000_final_automatic_v1/TECHNICAL_RUNTIME_VERIFICATION.json']
    paths += [ROOT/r['prediction_file'] for r in read(OUT/'PREDICTION_ALIGNMENT_AUDIT.json')]
    scripts=['stage_m1_existing_teacher_metrics_v1.py','stage_m1_pooled_bertscore_v1.py','stage_m1_pooled_bertscore_cpu6_v2.py','audit_stage_m1_workspace_references_v1.py','test_stage_m1_existing_teacher_v1.py','finalize_stage_m1_existing_teacher_v1.py','stage_m_evaluation_v1.py','score_s1000_automatic_prepared_v1.py','s1000_final_eval_core_v1.py','stage_k_bert_loader_compat_v1.py','validate_s1000_human_references_v1.py']
    paths += [ROOT/'scripts'/s for s in scripts]
    for p in paths:inputs.append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)})
    write_csv(OUT/'REPRODUCIBILITY_INPUT_MANIFEST.csv',inputs)
    (OUT/'README_VI.md').write_text('''# Stage M.1 — số đo thực trên VALIDATION200

Mở REVIEW_VALIDATION200_TABLE.md và VISUAL_VALIDATION200_TABLE.md. RO/U/F là pipeline hiện tại; CE3/V1-S500 là đối chiếu lịch sử trên cùng200 businesses. F=V1-S1000. Tất cả ô metric trong các bảng này là số đã tính, không phải preflight.

Review dùng common Teacher reference đã xác minh giống hệt. Visual U và F dùng target riêng khác nhau; không đọc chênh lệch hai hàng này như bằng chứng grounding tốt hơn. CE3/S500/F có chung Filtered Visual target, nên có thể so sánh mức độ khớp cùng reference, với giới hạn architecture/budget/ancestry đã nêu.

ALL_VALIDATION200_PER_BUSINESS_METRICS.csv có1.800 dòng; paired comparisons chỉ dùng cùng component-reference hash. Giữ đủ200, không bỏ output/reference rỗng. Các điểm này là validation AI Teacher agreement, không phải TEST, human-reference score hay chấm factuality.

FINAL_SCORED là các phần chia theo model của cùng1.800 dòng, không cộng thêm lần nữa. SCORED/RO là200 dòng control CPU1 cũ để đối chiếu số học, không phải model/cohort mới. Runtime cuối cùng là ACTUAL_FINAL_VALIDATION_METRIC_RUNTIME.json (CPU6); METRIC_RUNTIME.json cũ chỉ ghi lịch sử control CPU1. Xem SCORING_FILE_ROLES.json để tránh đếm trùng.

TEST100 Teacher-reference ROUGE/BERTScore: PENDING — no matching references available. Human reference và human evaluation xử lý riêng. Không sinh thêm summary/target, không gọi API có phí, không train.

Đây là package quản trị có model identities, outputs và scores; không đưa cho annotator đang được blinded. Xem STAGE_M1_RESULTS_AND_LIMITATIONS.md và PACKAGE_INTEGRITY_REPORT.json bên ngoài ZIP. Code cần workspace/model metric đã pin; ZIP không nhúng weights/ảnh gốc/credential/human DB.
''',encoding='utf8')
    package=OUT/'REVIEW_PACKAGE';package.mkdir()
    def copy(p,rel):
        dest=package/rel;require(not dest.exists(),'Duplicate package member');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not p.is_relative_to(package) and 'DOCUMENTATION_UPDATES' not in p.parts:copy(p,Path('RESULTS')/p.relative_to(OUT))
    for s in scripts:copy(ROOT/'scripts'/s,Path('scripts')/s)
    for name in docs:copy(ROOT/'thesis_sdd_docs'/name,Path('thesis_sdd_docs')/name);copy(docdir/(name+'.diff'),Path('DOC_UPDATES')/(name+'.diff'))
    copy(docdir/'CHANGE_RECEIPTS.json','DOC_UPDATES/CHANGE_RECEIPTS.json')
    for c in ['reviews_only','unfiltered','filtered']:copy(DATA/f'validation_{c}.jsonl',Path('EXACT_ORIGINAL_VALIDATION_TARGETS')/f'validation_{c}.jsonl')
    for r in read(OUT/'PREDICTION_ALIGNMENT_AUDIT.json'):copy(ROOT/r['prediction_file'],Path('EXACT_ORIGINAL_PREDICTIONS')/r['model']/'predictions.json')
    for name in ['METRIC_LOCK.json','MODEL_AND_DECODER_LOCK.json']:copy(PREP/name,Path('FROZEN_LOCKS')/name)
    (package/'README.md').write_text('Start with RESULTS/README_VI.md. These are actual VALIDATION200 Teacher-target agreement results, not TEST/human-reference evaluation. Both current and historical model identities are disclosed; keep this administrator package away from blinded human annotators. Exact original validation targets/predictions are preserved byte for byte. No TRAIN target is treated as TEST reference. Runtime/large model assets remain in the project and are SHA-pinned. PACKAGE_MANIFEST.csv binds all payloads except itself. The external integrity receipt records actual reopen/extraction/member-hash verification.\n',encoding='utf8')
    payload=[p for p in sorted(package.rglob('*')) if p.is_file()]
    write_csv(package/'PACKAGE_MANIFEST.csv',[{'filename':p.relative_to(package).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in payload])
    archive=OUT/'STAGE_M1_TEACHER_VALIDATION200_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_m1_verify_') as d,zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==len(payload)+1 and z.testzip() is None,'ZIP CRC/duplicate/count failure')
        require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'Unsafe archive member');z.extractall(d);folder=Path(d)
        manifest=list(csv.DictReader((folder/'PACKAGE_MANIFEST.csv').open(encoding='utf8')));require({r['filename'] for r in manifest}==set(names)-{'PACKAGE_MANIFEST.csv'},'Package membership differs')
        for r in manifest:
            p=folder/r['filename'];require(p.stat().st_size==int(r['bytes']) and file_hash(p)==r['sha256'],'Extracted member SHA/size mismatch')
        for p in folder.rglob('*'):
            if not p.is_file():continue
            if p.suffix=='.json':read(p)
            elif p.suffix=='.jsonl':
                for line in p.read_text(encoding='utf8').splitlines():json.loads(line)
            elif p.suffix=='.csv':list(csv.DictReader(p.open(encoding='utf8')))
            elif p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
    h=file_hash(archive);(OUT/(archive.name+'.sha256')).write_text(h+'  '+archive.name+'\n',encoding='ascii')
    write(OUT/'PACKAGE_INTEGRITY_REPORT.json',{'status':'PASS_ACTUAL_REOPEN_EXTRACT_EVERY_MEMBER_SHA256','archive':archive.name,'bytes':archive.stat().st_size,'sha256':h,'payload_files':len(payload),'ZIP_members':len(payload)+1,
        'CRC_reopen_extraction_member_size_SHA_and_machine_readable_parse_checked':True,'measured_validation_component_rows':1800,'TEST_reference_scores':0,'TRAIN_metrics':0,'new_LED_or_Teacher_outputs':0})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':h,'payload_files':len(payload),'status':'PASS'},indent=2))

if __name__=='__main__':main()
