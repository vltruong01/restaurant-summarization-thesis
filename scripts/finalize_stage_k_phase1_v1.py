"""Package actual Stage K inference/reference-free results; no reference-based TEST score."""
from pathlib import Path
import json,csv,collections,statistics,shutil,zipfile,tempfile,datetime,difflib,sys
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_k/s1000_test100_phase1_v1';P=R/'outputs/final_evaluation/s1000_preparation_v1';D=O/'ANALYSIS';E=O/'EVIDENCE'

def prose(name,s):
    p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(s.strip()+'\n')
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);require(file_hash(p)==file_hash(q),'Copy checksum failed')
def table(h,rows):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)

def finalize():
    require(not (O/'PACKAGE_INTEGRITY_REPORT.json').exists(),'Preserve completed Stage K package')
    receipt=read(D/'GENERATION_INTEGRITY_RECEIPT.json');require(receipt['condition_records']==400 and receipt['logical_components']==700,'Inference incomplete')
    agg=read(D/'REFERENCE_FREE_AGGREGATES.json');pair=read(D/'REFERENCE_FREE_PAIRED_ANALYSIS.json');runtime=read(E/'METRIC_SYNTHETIC_RUNTIME_VERIFICATION.json')
    require(runtime['TEST_businesses_scored']==0 and runtime['status'].startswith('PASS_ACTUAL_PINNED'),'Metric test status')
    # Verify every frozen prep file remains unchanged, especially the complete annotation workspace/key/cohort.
    z=P/'S1000_FINAL_EVALUATION_PREPARATION_REVIEW.zip'
    require(file_hash(z)=='fa42d176c46fd2fa3211d8d88ed9f15eceaa8046fe5110447955c34df64d395e','Preparation archive changed')
    with zipfile.ZipFile(z) as arc:
        mf=list(csv.DictReader(arc.read('PACKAGE_MANIFEST.csv').decode('utf8').splitlines()))
        for f in mf:
            require(file_hash(P/'REVIEW_PACKAGE'/f['filename'])==f['sha256'],'Frozen preparation package payload changed: '+f['filename'])
            if not f['filename'].startswith(('UPDATED_DOCUMENTATION/','ADMIN_ONLY/')):
                require(file_hash(P/f['filename'])==f['sha256'],'Functional preparation file changed: '+f['filename'])
            elif f['filename'].startswith('ADMIN_ONLY/'):
                require(file_hash(P/Path(f['filename']).name)==f['sha256'],'Human design/key changed')
    write(E/'FINAL_PREPARATION_AND_ANNOTATION_PRESERVATION.json',dict(status='PASS_ALL_FROZEN_PREPARATION_FILES_UNCHANGED',payload_files=len(mf),annotation_templates_unchanged=True,human_cohort30_unchanged=True,
        Shuffled_mapping_unchanged=True,metric_lock_unchanged=True,model_and_decoder_lock_unchanged=True,preparation_archive_sha256=file_hash(z),human_reference_payloads_filled=0,human_approvals_created=0))
    locks=read(P/'MODEL_AND_DECODER_LOCK.json')
    for a,m in locks['models'].items():require(file_hash(R/m['path']/'adapter_model.safetensors')==m['adapter_sha256'],'Frozen adapter changed')
    body=[]
    for a in ['RO','U','F','S']:
        rr=receipt['input_receipts'][a];native=read(O/'EXECUTION'/a/'COMPLETION.json')
        body.append([a,100,100 if a=='RO' else 200,rr['generation_calls'],f"{rr['seconds']:.2f}",f"{rr['peak_allocated_bytes']/2**30:.3f}"])
    executions=table(['Condition','Businesses','Logical components','New calls','Wall seconds','Peak allocated GiB'],body)
    prose('TEST100_FINAL_INFERENCE_REPORT.md',f'''# Stage K Phase1 final TEST100 inference

All400 actual condition-business records across the complete frozen100-business TEST cohort are generated and preserved. RO/U/F retain their exact S1000 checkpoints; S uses F and the unchanged frozen donor mapping. There are700 logical components,598 new generation calls and102 explicitly reused components (100 F Review and2 unchanged empty F Visual outputs for S). No S model was trained.

{executions}

Execution used the existing local NVIDIA GeForce RTX3050 6GB Laptop GPU. There was no new cloud rental, provider lifecycle operation or purchase. Local power/hardware costs are not measured as provider charges; cloud rental cost for this stage is zero new rentals. Actual per-condition runtime, package versions, GPU/memory and raw call timings remain in native receipts. Conditions execute sequentially; S reused components have zero additional generation time, not an invented independent runtime. CUDA/driver/platform and the local-vs-prior3090 environment are recorded; deterministic settings do not guarantee cross-hardware bitwise identity.

The latest human Stage K request explicitly authorizes inference/reference-free diagnostics BEFORE independent references. It supersedes the earlier preparation's references-before-inference gate only for Phase1. The original preparation, annotation workspace and metric/cohort/mapping locks are unchanged. A separate Stage K runner records this operational scope amendment and additional provenance, without changing model weights, source inputs, task structure, decoding or precision.

Each task preserves raw text and generation IDs, actual input IDs, review/photo/caption marker IDs, source/token/checkpoint/decoder hashes, full generation settings, task/condition, donor ID, EOS/cap flags and status. Components concatenate verbatim; no quality-driven regeneration, post-edit, reranking, source reselection or selective exclusion occurred. Exact raw and assembled outputs are in TEST100_RO_U_F_S_PREDICTIONS.jsonl. RO Visual is N/A. The two zero-F-caption cases retain original empty Visual task behavior, and their exact F outputs are reused for S; any model-generated claims from empty caption input are retained and flagged, never filled manually or represented as supported evidence.

Complete TEST100 remains official. Known original-photo/caption reuse (17 affected businesses; U66/F44 selected duplicate photos, U59/F42 own-TRAIN caption/support overlap) is a methodological limitation, not proven memorization or score inflation. Source-separated Review is caption-free; S Review equality is structural, not learned robustness to a joint distractor. No model is selected or tuned using these TEST outputs. STOP after Phase1; independent references and official reference metrics remain pending.
''')
    integrity_counts={r['condition']+'_'+r['task']:{k:r.get(k) for k in ['businesses','empty','malformed','at_max_length','forced_EOS_at_cap_possible','empty_caption_nonempty_output']} for r in agg}
    prose('TEST100_GENERATION_INTEGRITY_REPORT.md',f'''# Final generation integrity

PASS:100 unique TEST business IDs and exactly400 unique condition-business records.700 logical component records comprise100 RO Review,200 U,200 F and200 S. All required raw outputs, source/task/adapter/decoder hashes and actual input IDs match the frozen plan. Independently decoding all raw generation IDs with the pinned tokenizer reproduces the exact saved text. Raw text SHA, sequence limits/decoder-start and native completion receipts are verified. All100 F/S Review outputs are byte/token-identical;2 unchanged empty Visual outputs are reused identically.98 S Visual inputs actually change. Donor mapping and seed were not recreated or modified.

No silent parser repair occurs: wrong/multiple/missing headings, foreign component markers, empty bodies, cap/EOS and incomplete punctuation are descriptive flags. At max_length768, final EOS may be forced and is not proof of naturally complete content. Raw defective or low-quality outputs, if any, remain in the dataset. Numeric counts below are from the actual diagnostic tables, not semantic judgments:

```json
{json.dumps(integrity_counts,indent=2)}
```

Native restoration receipts verify456 adapter tensors, FP32 parameters and unchanged original LED-large backbone bytes with FP16 autocast. No optimizer/backward/training occurred. There were no quality-based retries; native sequence completion records technical provenance. Old prepare-stage metadata is not falsely labeled a completed GPU restore; the new RESTORATION receipts are the actual GPU-loaded execution evidence. Original model files and all361 preparation payloads were rehashed unchanged, including100 blank reference templates and the frozen30-case human cohort. No human references/approvals or ratings were created. TEST predictions and diagnostics are stored only in the separate Stage K tree.
''')
    metrics=[]
    for r in agg:
        if r['status'].startswith('N/A'):metrics.append([r['condition'],r['task'],'N/A','N/A','N/A','N/A','N/A','N/A']);continue
        metrics.append([r['condition'],r['task'],r['businesses'],f"{r['words']['mean']:.3f}",f"{r['raw_generated_tokens']['mean']:.3f}",f"{r['repeated_3gram_fraction']['mean']:.6f}",f"{r['duplicate_sentence_fraction']['mean']:.6f}",f"{r['source_lexical_precision_proxy']['mean']:.6f}"])
    summary=table(['Condition','Task','n','Mean words','Mean raw tokens','Repeated trigrams','Duplicate sentences','Input word overlap proxy'],metrics)
    paired=[]
    for r in pair['paired']:
        lo,hi=r['CI95'];paired.append([r['contrast'],r['metric'],r['n'],f"{r['mean_difference']:.6f}",f"[{lo:.6f}, {hi:.6f}]"])
    paired_table=table(['Contrast','Metric','Paired businesses','Mean difference','Paired percentile95% CI'],paired)
    distributions={r['condition']:r['caption_count_distribution'] for r in agg if r['task']=='visual' and r['status']=='ACTUAL_DESCRIPTIVE_METRICS'}
    prose('TEST100_REFERENCE_FREE_COMPARISON.md',f'''# Reference-free TEST100 descriptive comparison

These are actual output-format, length, repetition and lexical diagnostics. They do not use human references, Teacher targets or an LLM judge. Lower repetition or higher word overlap is NOT verified factuality, visual grounding, sentiment balance, usefulness or overall model quality. No winning checkpoint is selected.

{summary}

Words use the frozen NFKC/casefold regex. Raw token counts include decoder-start/EOS; content-token counts excluding IDs0/1/2 are separately retained. Trigram repetition is1-unique/total trigrams, and full-sentence repetition normalizes regex sentence units; repeated full-sentence occurrences/types are preserved in the CSV. Only the exact leading expected header is excluded from body metrics. Malformed source text is not repaired. EOS after the decoder-start token is distinguished from the initial decoder-start ID2. A cap-terminated sequence with EOS may have forced termination. Completeness flags describe surface shape/termination, not semantic coverage.

Visual caption-count distributions (the frozen input, not output-derived selection):
```json
{json.dumps(distributions,indent=2)}
```

U-F redundancy analysis includes all100 and the prospectively input-defined same-count subset of27 businesses. This controls caption NUMBER only, not caption length, content, model ancestry or supervision; it cannot isolate a causal filtering effect. F-S reports all100 and the98 actual input-intervention businesses;2 empty cases are no-intervention and retained. RO Visual is N/A, not zero or a missing-Visual penalty. F/S Review equality is structural exact reuse.

{paired_table}

These intervals use paired business-index bootstrap20,000 replicates, numpy PCG64 seed20261028 and95% percentile intervals. Contrasts are labeled explicitly: U-minus-F or F-minus-S. They are descriptive; no confirmatory factuality claim or model-selection test follows. Known chain/photo reuse can weaken business-independence assumptions. Input source word overlap is computed against caption BODY text without task/caption marker tokens. For S, this is similarity to deliberately wrong DONOR captions, not grounding in recipient photos. Photo faithfulness requires later independent reference/human evidence; it is not inferred here.

Detailed per-business results and exact input-defined subset IDs are in the CSV and REFERENCE_FREE_PAIRED_ANALYSIS.json. No TEST83 replacement or outcome-based case deletion exists. Independent human reference and output-evaluation phases remain separate and unexecuted.
''')
    prose('REFERENCE_BASED_EVALUATION_RUNTIME_READINESS.md',f'''# Reference-based metric runtime readiness

Actual synthetic CPU checks PASS for ROUGE1/2/L, paired CI and BERTScore Precision/Recall/F1 using the exact pinned model. No official TEST ROUGE or BERTScore has been computed. The technical test uses only the explicitly listed artificial sentences in METRIC_SYNTHETIC_RUNTIME_VERIFICATION.json; its numbers are synthetic diagnostics, not thesis TEST results.

BERTScore0.3.13, FacebookAI/roberta-large revision722cf37b1afa9454edce342e7895e588b6ff1d59, SHA25636a10a8b694fadf9bf4f9049d14e257e88be45313ae02d882af9e60f39b8b2e8, layer17, slow tokenizer, no IDF, no baseline rescaling, CPU FP32/batch8/one thread are unchanged. The actual1,425,941,629-byte scorer binary and tokenizer/config have been downloaded and hash-verified locally under METRIC_RUNTIME; they are excluded from the review ZIP. The pinned wheel is reused from the original preparation without modifying the user's Python environment. Synthetic setup/forward/checks took{runtime['seconds']:.3f} seconds, with process peak working set{runtime['process_peak_working_set_bytes']/2**30:.3f} GiB; this is NOT a full700-component benchmark.

The initial synthetic preflight failed because torch2.1 could not deserialize the legacy RoBERTa storage onto `meta`. The exact same SHA-pinned file succeeds with weights_only=True CPU deserialization. A narrow context-managed loader adapter changes only that storage placement during the meta key probe for this exact file. It does not set weights_only=False, switch checkpoints/precision/tokenizer/layers or modify scientific scoring settings. Initial failure, exact CPU probe and successful synthetic retry receipts are preserved. The future official scoring wrapper includes this operational compatibility adapter while keeping human-approval validation mandatory.

The library's512-token prefix behavior was tested with802 synthetic input tokens yielding512 including BOS/EOS. Every future TEST BERTScore row must expose original token counts/truncation flags and aggregate affected counts/fractions. Full generation stays at768 per task; ROUGE and length/repetition diagnostics use full text. Prefix similarity is not full-document semantic fidelity or photo grounding. ROUGE0.1.2 uses the frozen Porter-stemmed component preprocessing; paired business bootstrap/sign-flip/Holm procedures remain the existing METRIC_LOCK. No settings were adjusted using TEST quality outcomes.

Official scoring remains BLOCKED until all100 common human-written references are completed, checked by a different human, approved, hash-frozen and explicitly released. No Teacher/pseudo/model-output substitute is permitted. The original annotation package remains unchanged with100 blank templates. Human scoring must follow official automatic completion, with staffing decided separately by the researcher. The frozen30-business cohort/rubric are preserved.

Future scorer entry point: run_stage_k_reference_metrics_after_human_approval_v1.py. It delegates to the frozen scorer/validator, requires an explicit --execute-approved release,100 real approved per-business reference JSONs, all4 native result folders and the verified local scorer directory. It is not executed in Phase1.
''')
    prose('TEST100_AUTOMATIC_EVALUATION_STATUS.md','''# Stage K automatic evaluation status

Completed: exact frozen setup verification, actual400-record TEST inference, raw700-component integrity checks, reference-free descriptive metrics and paired input-defined comparisons, synthetic reference-metric runtime validation, preservation checks and review packaging. No training or paid GPU rental occurred.

Pending: independent human drafting and second-human checking of all100 references; no approvals are claimed. Then validate/freeze real references and obtain the next-stage release before official TEST ROUGE/BERTScore/paired reference-based evaluation. Human output scoring follows official Automatic Evaluation; the existing30-business sample and rubric remain frozen and annotator staffing is separately decided by the researcher. Reference writers/checkers must never receive this prediction ZIP, its outputs or diagnostics; give them only the unchanged reference source workspace and original photos.

Stage K outputs are stored under outputs/stage_k/s1000_test100_phase1_v1, separate from outputs/final_evaluation/s1000_preparation_v1/TEST100_REFERENCE_ANNOTATION_PACKAGE. Exact original preparation/mapping/checkpoint/input/metric hashes remain unchanged. No human-reference text was filled and no human evaluation ran. Complete TEST100 is retained, including2 no-F-caption cases and17 known photo-overlap businesses. No TEST83 replacement, model selection, tuning, external baseline/generalization run or future training follows this package.

Remaining thesis milestones: human reference construction/approval, official reference metrics, separately released human output study, externally approved baseline work, results/discussion/limitations/computational accounting and thesis defense writing. STOP after this Phase1 delivery for researcher approval.
''')
    # Current researcher-facing documentation, exact history retained in this new stage namespace.
    documents=['THESIS_PROGRESS.md','LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','CODEX_THESIS_IMPLEMENTATION_PLAN.md','CHATGPT_NEW_CONVERSATION_HANDOFF.md'];changes=[]
    note='''## Stage K Phase1 completed — 2026-10-29

The latest researcher authorization permits TEST inference and reference-free diagnostics before human references. All400 RO/U/F/S condition records are generated for complete TEST100:700 logical components,598 new calls,100 F Review and2 empty F Visual components explicitly reused for S. The frozen mapping, sources, checkpoints, precision and decoder are unchanged.98 S Visual cases have real interventions;2 empty cases remain no-intervention. Actual local RTX3050 6GB execution used no new paid rental. Raw tokens, input/adapter/source/decoder hashes and native runtime/memory receipts are preserved under `outputs/stage_k/s1000_test100_phase1_v1/`.

Reference-free length/repetition/format/EOS and lexical-caption proxies are complete; they are not factuality/image-grounding or final quality scores. Input-defined paired U/F same-caption-count analysis has27 businesses, and F/S reports98 active plus full100. Exact common human-reference templates remain blank/unchanged and separate from predictions; no human approvals or ratings were fabricated. Actual synthetic-only ROUGE/BERTScore/paired-runtime checks pass, including the unchanged512-token BERTScore-prefix limitation and a recorded legacy-binary CPU-loading compatibility fix. Official TEST ROUGE/BERTScore remains pending100 real independently checked/approved references.

Next order: independent human reference drafting/checking/freezing -> separately released official reference-based Automatic Evaluation -> Human Evaluation on the already frozen30 businesses (staffing decided separately) -> results/discussion. Do not expose this prediction ZIP or automatic diagnostics to reference writers/checkers. No model selection/tuning, training, external baseline, TEST83 replacement or human scoring was performed. Original750/S1000 histories, source-separated Review conditioning, different U/F supervision/output budgets and known visual-evidence reuse (17businesses; U66/F44 selected duplicate photos, U59/F42 own-TRAIN caption/support) remain material limitations. No memorization or score inflation is proven. STOP for researcher review.

'''
    replacements={
        'Approve protocols -> human reference construction/check/freeze -> finite inference and Automatic Evaluation -> separate30-business/three-rater human output evaluation -> results/discussion.':'Stage K inference/reference-free analysis is complete; next human reference construction/check/freeze -> official reference-based Automatic Evaluation -> separately staffed human output evaluation -> results/discussion.',
        'Final predictions, independent human references, automatic scores and human output scoring have NOT been executed by this preparation.':'Stage K final predictions and reference-free diagnostics are now complete; independent references, official reference-based scores and human output scoring remain pending.',
        'No model outputs exist from this preparation.':'The original preparation contained no outputs; completed Stage K predictions are kept in the separate execution tree.',
        'Do not reopen training, generate TEST outputs/scores, write AI human references or start human scoring from this handoff.':'Do not reopen training, regenerate TEST outputs for quality, compute official reference scores before approved references, write AI human references or start human scoring from this handoff.'}
    for name in documents:
        p=R/'thesis_sdd_docs'/name;before=p.read_bytes();backup=O/'DOCUMENTATION_BEFORE'/name;backup.parent.mkdir(exist_ok=True);backup.write_bytes(before)
        text=before.decode('utf8')
        for a,b in replacements.items():text=text.replace(a,b)
        first=text.find('\n\n');after=text[:first+2]+note+text[first+2:]
        p.write_text(after,encoding='utf8');diff=''.join(difflib.unified_diff(before.decode('utf8').splitlines(True),after.splitlines(True),fromfile=name+' before StageK',tofile=name+' current'))
        dp=O/'DOCUMENTATION_DIFFS'/f'{name}.patch';dp.parent.mkdir(exist_ok=True);dp.write_text(diff,encoding='utf8')
        changes.append(dict(path='thesis_sdd_docs/'+name,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=file_hash(p),snapshot=backup.relative_to(O).as_posix()))
    decisions=R/'DECISIONS.md';before=decisions.read_bytes();(O/'DOCUMENTATION_BEFORE/DECISIONS.md').write_bytes(before)
    with decisions.open('ab') as f:f.write(('\n\n## Stage K Phase1 — 2026-10-29\n\n'+note.split('\n\n',1)[1]).encode('utf8'))
    changes.append(dict(path='DECISIONS.md',before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=file_hash(decisions),snapshot='DOCUMENTATION_BEFORE/DECISIONS.md',old_bytes_preserved_as_prefix=True))
    write(E/'DOCUMENTATION_CHANGE_MANIFEST.json',changes)
    prose('THESIS_SDD_DOCS_UPDATE_SUMMARY.md','''# Stage K documentation update

Four existing thesis_sdd_docs entry points now record actual completed inference/reference-free analysis, provenance, the amended evaluation order and pending human references/reference-based metrics/human scoring. Root DECISIONS received an appended record, preserving previous bytes. Exact before snapshots/diffs/hashes remain locally in the Stage K namespace. Original proposal, experiment logs, model/source/mapping/preparation files and annotation workspace remain unchanged. Notes retain material adapter-ancestry, source-separated task, caption-pipeline/budget, mimicry-screening and image-reuse limitations without reintroducing discarded training experiments. No human staffing or approval is invented.
''')
    prose('REPRODUCIBLE_STAGE_K_COMMANDS.md','''# Reproducible execution commands and boundaries

Actual commands/runner hashes and native receipts are preserved in EVIDENCE and EXECUTION metadata. Use the verified existing workspace and pinned package/runtime. The Stage K runner binds the precise original plan/model lock and human request SHA. No mapping or inputs are rebuilt. For a fresh authorized environment, select ONE visible suitable GPU, run F/RO/U once sequentially, then S with --F-results; never overwrite existing outputs or rerun for quality. The already-completed local run must be reused, not repeated.

```powershell
.venv-caption-pilot/Scripts/python.exe -u -X utf8 -B scripts/run_stage_k_s1000_phase1_v1.py --workspace C:/thesis_proposal/yelp_dataset --preparation outputs/final_evaluation/s1000_preparation_v1 --condition F --execute --approval-message outputs/stage_k/s1000_test100_phase1_v1/EVIDENCE/RESEARCHER_STAGE_K_AUTHORIZATION.txt --output NEW_AUTHORIZED_OUTPUT/F
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_k_reference_free_analysis_v1.py --preparation outputs/final_evaluation/s1000_preparation_v1 --execution outputs/stage_k/s1000_test100_phase1_v1/EXECUTION --output NEW_DIAGNOSTICS_DIRECTORY
```

Reference-free analysis reads no reference payloads. The official reference scorer remains approval-validated and is not executed now. Its future wrapper uses the exact pinned scorer wheel/model and narrow legacy CPU-load adapter; no token/model/metric switch. Prediction/source artifacts are administrator/researcher material and must not be sent to reference authors/checkers. Stop after this delivery.
''')
    prose('README.md','''# Stage K Phase1 review package

Read TEST100_FINAL_INFERENCE_REPORT.md, GENERATION_INTEGRITY_REPORT and REFERENCE_FREE_COMPARISON first. Predictions are actual400-record JSONL with700 nested logical task outputs/raw tokens and explicit reuse. Metrics are reference-free, not ROUGE/BERTScore TEST scores. Frozen source/input token plan and mappings allow independent provenance review. Raw native completion/restoration/call metadata, technical metric-only synthetic checks and scripts accompany the reports. Model binaries, bulk raw Yelp/image datasets and scorer weights are excluded.

This package contains TEST outputs and diagnostics. DO NOT GIVE IT TO HUMAN REFERENCE AUTHORS OR CHECKERS. Their existing source-only annotation workspace is separate and remains unchanged. Official reference-based scores are pending100 real independently checked/approved references. Human scoring is not started. No model selection, quality retry, training, automatic rental, external baseline or TEST83 replacement occurred. All100 businesses remain.

PACKAGE_MANIFEST.csv binds every payload; the ZIP is reopened, CRC-checked, freshly extracted and verified member-by-member. Check the separate checksum/integrity report. STOP for researcher approval before the next stage.
''')
    Q=O/'REVIEW_PACKAGE';Q.mkdir()
    for p in O.glob('*.md'):cp(p,Q/p.name)
    for p in D.iterdir():
        if p.is_file():cp(p,Q/p.name)
    for p in E.iterdir():
        if p.is_file():cp(p,Q/'EVIDENCE'/p.name)
    for a in ['RO','U','F','S']:
        for p in (O/'EXECUTION'/a).rglob('*'):
            if p.is_file() and p.name not in ['predictions.json','raw_predictions.json'] and 'RAW' not in p.parts:cp(p,Q/'EXECUTION_METADATA'/a/p.relative_to(O/'EXECUTION'/a))
    for name in ['COHORT100_FREEZE.json','MODEL_AND_DECODER_LOCK.json','INFERENCE_RUNTIME_LOCK.json','METRIC_LOCK.json','SHUFFLED_POLICY.json','TEST100_SHUFFLED_MAPPING.csv','TEST100_SHUFFLED_INPUT_MANIFEST.json','TEST100_INFERENCE_PLAN.json','HUMAN_EVALUATION_COHORT30.json']:
        cp(P/name,Q/'FROZEN_PREPARATION'/name)
    for name in ['s1000_final_eval_core_v1.py','stage_k_reference_free_analysis_v1.py','run_stage_k_s1000_phase1_v1.py','run_stage_k_local_sequence_v1.py','finish_stage_k_after_local_sequence_v1.py','stage_k_bert_loader_compat_v1.py','verify_stage_k_metric_runtime_v1.py','test_stage_k_reference_free_v1.py','run_stage_k_reference_metrics_after_human_approval_v1.py','score_s1000_automatic_prepared_v1.py','validate_s1000_human_references_v1.py','finalize_stage_k_phase1_v1.py']:
        cp(R/'scripts'/name,Q/'SCRIPTS'/name)
    for c in changes:
        if c['path'].startswith('thesis_sdd_docs/'):cp(R/c['path'],Q/'UPDATED_DOCUMENTATION'/c['path'])
    # Bind complete prediction/reference-free table frame without introducing reference scores.
    mf=[dict(filename=p.relative_to(Q).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(Q.rglob('*')) if p.is_file()]
    write_csv(Q/'PACKAGE_MANIFEST.csv',mf);Z=O/'STAGE_K_TEST100_PHASE1_REVIEW.zip'
    with zipfile.ZipFile(Z,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as arc:
        for p in sorted(Q.rglob('*')):
            if p.is_file():arc.write(p,p.relative_to(Q).as_posix())
    with zipfile.ZipFile(Z) as arc:
        require(arc.testzip() is None and len(arc.namelist())==len(set(arc.namelist()))==len(mf)+1,'ZIP CRC/member inventory')
        with tempfile.TemporaryDirectory(prefix='stage_k_zipcheck_',dir=O) as tmp:
            t=Path(tmp);arc.extractall(t)
            for f in mf:require((t/f['filename']).stat().st_size==f['bytes'] and file_hash(t/f['filename'])==f['sha256'],'Member SHA mismatch')
            pp=[json.loads(l) for l in (t/'TEST100_RO_U_F_S_PREDICTIONS.jsonl').read_text(encoding='utf8').splitlines()]
            require(len(pp)==400 and len({(r['condition'],r['business_id']) for r in pp})==400 and len({r['business_id'] for r in pp})==100,'Reopened prediction frame')
            mm=list(csv.DictReader((t/'TEST100_REFERENCE_FREE_METRICS.csv').open(encoding='utf8')));require(len(mm)==700,'Reopened metric table')
            for p in t.rglob('*.json'):read(p)
    final=dict(status='PASS_ACTUAL_STAGE_K_PHASE1_AND_ZIP_INTEGRITY',archive=Z.name,bytes=Z.stat().st_size,sha256=file_hash(Z),payload_files=len(mf),zip_members=len(mf)+1,
        reopened_extracted_every_member_SHA256_and_parsed=True,actual_businesses=100,actual_condition_records=400,actual_logical_components=700,new_generation_calls=598,reused_components=102,
        reference_free_metric_rows=700,official_TEST_ROUGE_or_BERTScore=0,human_reference_fields_changed=0,human_approvals_created=0,human_evaluation_started=False,
        new_GPU_rentals=0,suitable_local_GPU_used=True,preparation_payload_files_unchanged=361,documentation_files_updated=5,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(O/'PACKAGE_INTEGRITY_REPORT.json',final);(O/(Z.name+'.sha256')).write_text(final['sha256']+'  '+Z.name+'\n',encoding='utf8');print(json.dumps(final,indent=2))

if __name__=='__main__':finalize()
