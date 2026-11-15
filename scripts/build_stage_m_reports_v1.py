"""Read-only evidence collation. Writes new Stage M reports, never annotations."""
from pathlib import Path
from collections import Counter
import csv
import datetime
import json
import shutil
import sqlite3
from s1000_final_eval_core_v1 import read,write,write_csv,file_hash,require,paired

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_m/s1000_final_automatic_v1'
PREP=ROOT/'outputs/final_evaluation/s1000_preparation_v1'
K=ROOT/'outputs/stage_k/s1000_test100_phase1_v1'
L=ROOT/'outputs/stage_l/test100_human_references_v1'

def human_status():
    with sqlite3.connect((L/'PRIVATE_STATE/references.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        entries=[json.loads(r[0]) for r in c.execute('SELECT payload FROM refs')]
        events=c.execute('SELECT count(*) FROM events').fetchone()[0]
    counts=Counter(r['status'] for r in entries);counts['NOT_STARTED']+=100-len(entries)
    return {'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stored_records':len(entries),'event_records':events,
      'status_counts':{k:counts[k] for k in ['NOT_STARTED','DRAFT','SUBMITTED','NEEDS_REVISION','APPROVED']},
      'approved':counts['APPROVED'],'primary_ready':counts['APPROVED']==100,'access':'SQLite mode=ro; no export, mutation or account access',
      'human_evaluation_ratings':'Researcher-managed separately; completion not independently verified'}

def main():
    require(not (OUT/'STAGE_M_STATUS.json').exists(),'Refuse overwrite finalized Stage M reports')
    require(all(r['byte_identical'] for r in read(OUT/'REFERENCE_FREE_REPLAY_VERIFICATION.json')),'Stage K replay differs')
    require(read(OUT/'TECHNICAL_RUNTIME_VERIFICATION.json')['status'].startswith('PASS_'),'CPU technical test required')
    status=human_status();write(OUT/'HUMAN_REFERENCE_STATUS.json',status)
    release=Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/eda9a50f-b6df-40d4-b09a-3b7b47d088a0/Pasted text.txt')
    shutil.copyfile(release,OUT/'RESEARCHER_STAGE_M_REQUEST.txt')
    lock=read(PREP/'MODEL_AND_DECODER_LOCK.json');bindings=[]
    for condition,m in lock['models'].items():
        directory=ROOT/m['path'];adapter=directory/'adapter_model.safetensors'
        require(file_hash(adapter)==m['adapter_sha256'],'Protected adapter changed')
        bindings.append({'condition':condition,'path':str(adapter.relative_to(ROOT)),'sha256':file_hash(adapter),'status':'PASS_ACTUAL_LOCAL_ADAPTER_SHA256'})
    b=lock['backbone'];require(file_hash(b['local_file'])==b['sha256'],'LED backbone changed')
    bindings.append({'condition':'ALL','path':str(Path(b['local_file']).relative_to(ROOT)),'sha256':b['sha256'],'status':'PASS_ACTUAL_LOCAL_BACKBONE_SHA256'})
    write(OUT/'MODEL_FILE_REVERIFICATION.json',bindings)
    metrics=list(csv.DictReader((OUT/'REFERENCE_FREE_REPLAY/TEST100_REFERENCE_FREE_METRICS.csv').open(encoding='utf8')))
    require(len(metrics)==700,'700 diagnostic rows required');ids=read(PREP/'COHORT100_FREEZE.json')['business_ids']
    by={(r['condition'],r['business_id'],r['task']):r for r in metrics}
    extra=[]
    for task,other in [('review','RO'),('review','U'),('visual','U'),('visual','S')]:
        for metric in ['words','repeated_3gram_fraction','duplicate_sentence_fraction']:
            a=[float(by['F',b,task][metric]) for b in ids];v=[float(by[other,b,task][metric]) for b in ids]
            stat=paired(a,v,B=20000,P=1);stat.pop('signflip_two_sided_p');stat.pop('signflip_replicates')
            extra.append({'task':task,'contrast':'F-'+other,'metric':metric,'mean_F':sum(a)/100,'mean_comparator':sum(v)/100,'interpretation':'Measured reference-free descriptive CI only; not grounding or significance',**stat})
    write(OUT/'REFERENCE_FREE_PRIMARY_CONTRASTS.json',extra)
    for task in ['review','visual']:
        rows=[];md=[f'# Measured reference-free {task} diagnostics\n\nMacro means on all frozen TEST100 businesses. No reference or factuality assessment.\n','| Condition | n | Words | Trigram repetition | Duplicate sentence fraction | Empty | Malformed | At cap |','|---|---|---|---|---|---|---|---|']
        tex=['\\begin{tabular}{lrrrrrrr}','Condition & n & Words & Trigram repeat & Dup. sentence & Empty & Malformed & Cap \\\\','\\hline']
        for c in (['RO','U','F','S'] if task=='review' else ['U','F','S']):
            rr=[by[c,b,task] for b in ids];row={'condition':c,'task':task,'n':100,**{m:sum(float(r[m]) for r in rr)/100 for m in ['words','repeated_3gram_fraction','duplicate_sentence_fraction']},
                **{m:sum(r[m]=='True' for r in rr) for m in ['empty','malformed','at_max_length']},'status':'MEASURED_REFERENCE_FREE'};rows.append(row)
            cells=[c,'100',f"{row['words']:.2f}",f"{row['repeated_3gram_fraction']:.6f}",f"{row['duplicate_sentence_fraction']:.6f}",str(row['empty']),str(row['malformed']),str(row['at_max_length'])]
            md.append('| '+' | '.join(cells)+' |');tex.append(' & '.join(cells)+' \\\\')
        tex.append('\\end{tabular}');write_csv(OUT/f'REFERENCE_FREE_{task.upper()}.csv',rows)
        (OUT/f'REFERENCE_FREE_{task.upper()}.md').write_text('\n'.join(md)+'\n',encoding='utf8');(OUT/f'REFERENCE_FREE_{task.upper()}.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    for track,label in [('HUMAN_PRIMARY','PENDING_100_APPROVED_HUMAN_REFERENCES'),('TEACHER_SUPPLEMENTARY','BLOCKED_NO_GENERATED_TEACHER_REFERENCES_OR_PAID_RELEASE')]:
        directory=OUT/track;directory.mkdir();rows=[]
        for r in metrics:rows.append({k:r[k] for k in ['condition','business_id','task','output_sha256','source_sha256','checkpoint_sha256']}|{'reference_type':track,'status':label,**{m:None for m in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']}})
        write_csv(directory/'PER_BUSINESS_PENDING_MATRIX.csv',rows)
        md=[f'# {track}: NOT SCORED\n\n{label}. Blank cells mean unavailable, not zero. Full100 required. RO Visual N/A.\n','| Condition | Task | Planned n | ROUGE-1/2/L | BERT P/R/F1 |','|---|---|---|---|---|']
        tex=['\\begin{tabular}{llrll}','Condition & Task & Planned n & ROUGE-1/2/L & BERT P/R/F1 \\\\','\\hline']
        for c,t in [('RO','review'),('U','review'),('F','review'),('S','review'),('U','visual'),('F','visual'),('S','visual')]:
            md.append(f'| {c} | {t} | 100 | Pending | Pending |');tex.append(f'{c} & {t} & 100 & Pending & Pending \\\\')
        tex.append('\\end{tabular}');(directory/'TABLES_PENDING.md').write_text('\n'.join(md)+'\n',encoding='utf8');(directory/'TABLES_PENDING.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    historical=ROOT/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset';inventory=[]
    for p in sorted(historical.glob('*.jsonl')):
        rr=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()];inventory.append({'file':str(p.relative_to(ROOT)),'sha256':file_hash(p),'rows':len(rr),'splits':dict(Counter(r['split'] for r in rr)),'has_targets':all('target' in r for r in rr)})
    write(OUT/'TEACHER_REFERENCE_INVENTORY.json',{'inspected_historical_dataset_files':inventory,'historical_TEST_target_rows':sum(r['splits'].get('test',0) for r in inventory),
        'source_only_example_packages_are_references':False,'compatible_complete_TEST100_teacher_reference_found_in_inspected_locations':False,
        'scope':'Canonical Teacher dataset plus Stage L source-only example packages and Stage M preparation. Not an assertion about every external file/account.',
        'historical_held_out_test_authorized':read(ROOT/'configs/stage_g_teacher_production_d042.json')['production_scope']['held_out_test']})
    plan=read(OUT/'TEACHER_PREPARATION/TEACHER_PLAN.json')
    write(OUT/'STAGE_M_STATUS.json',{'status':'FEASIBLE_WORK_COMPLETE_REFERENCE_SCORING_PENDING','human_approved':status['approved'],'human_required':100,'human_reference_scoring':'PENDING',
        'teacher_reference_scoring':'BLOCKED_NO_REFERENCES_DEDICATED_CREDENTIAL_AND_PAID_RELEASE','teacher_api_calls':0,'new_LED_inference_calls':0,'new_training_updates':0,
        'reference_free':'MEASURED_AND_BYTE_IDENTICALLY_REPLAYED','metric_runtime':'ACTUAL_CPU_SYNTHETIC_AND_VALIDATION_PASS','frozen_condition_records':400,'logical_components':700,
        'teacher_resource_estimate':{k:plan[k] for k in ['task_requests','original_photos','image_bytes','review_text_characters','output_token_upper_bound','cost_estimate']},
        'stop':'No paid Teacher calls, decoding, training, model selection or modification of human evaluation.'})
    report=f'''# Stage M — automatic evaluation status

Completed: frozen TEST100 / 400 condition records / 700 logical components verified against source, token, decoder and checkpoint bindings. All three local adapters and the LED-large backbone were rehashed successfully. The five Stage K diagnostic artifacts were recomputed using the original implementation and match byte for byte. No new LED generation or training occurred.

## Track A: primary human references — PENDING

Stage L read-only snapshot: {status['approved']}/100 approved, {status['stored_records']} stored records, {status['event_records']} revision events. No human references were written, exported, edited or approved by Stage M. Official scoring requires all100 approved references. Blank pending tables are not scores. Primary references must be genuinely human-written and independently checked; model-written examples are not substitutes. Local account/attestation checks cannot independently authenticate real people. The researcher manages Human Evaluation separately; rating completion is not independently verified here.

## Track B: AI Teacher supplementary — BLOCKED

No compatible completed TEST100 Teacher reference dataset was found in the inspected canonical locations. Historical Teacher data have TRAIN/VALIDATION targets, not TEST. Stage L example ZIPs contain source inputs, not generated references. Dedicated Teacher key and dedicated key file are absent at the snapshot; no paid release exists. No API call was made.

The ready source-only plan has 200 requests (100 Review +100 Visual), 2,817 reviews / 1,251,649 review-text characters, 1,136 original-byte photos /41,469,768 bytes. Maximum requested output budget is 1,638,400 tokens including any provider accounting; this is a ceiling, not predicted usage. Concurrency1, zero automatic retries. USD cannot be defensibly estimated without current provider image/reasoning rates. The alias gpt-6.1-sol has no attested immutable backend snapshot. A future run must record the returned model and raw usage; actual vision access remains untested.

Historical training Visual targets were condition-specific caption+photo-supported. TEST Teacher Visual references instead inspect all original photos without captions. Review and Visual remain isolated; exact citation IDs/quotes are mechanically checked. New soft 80–130 Review /40–80 Visual word guides follow the researcher's reference-example prompt; they are not metric or decoder changes. Full text is preserved. This adapted methodology is supplementary and cannot replace primary human evaluation.

## Measured reference-free findings

Review mean words: RO87.09, U84.90, F/S86.72. Mean trigram repetition: RO0.031442, U0.034570, F/S0.039527. Review differences can reflect initialization and task/training differences; the source-separated Review pass receives no captions.

Visual mean words: U67.74, F/S51.68. Trigram repetition: U0.122986, F/S0.060546. U-minus-F mean repetition difference0.062440, paired95% CI[0.044258,0.082059]. Same-caption-count27-business descriptive sensitivity:0.038697, CI[0.019983,0.058818]. Lower repetition and shorter text do not prove better factuality, scene coverage or usefulness.

F/S Review is identical for100/100 businesses. S Visual is an exact permutation of F donor outputs for100/100; global length/repetition equality is therefore structural, not recipient grounding. Future S scores always use the recipient business reference. RO Visual is N/A.

Two distinct F Review cases remain frozen and are reused by S: oP7An13EMBE_IZSqLzQ_bQ contains a foreign Visual marker (130 tokens); ymE6ka6y6J-z2y2rX8QH3Q reaches768 tokens with trigram repetition0.968504. These are2 independent F generations /4 logical F+S records. They were neither repaired nor regenerated. F/S each retain2 empty Visual outputs for zero F-caption businesses; U remains scored on those businesses. EOS at the768 cap may be forced, not natural completion.

## Frozen scoring policy and interpretation

Full100 is primary; no silent omissions. Each reference track separately reports ROUGE1/2/L and BERTScore P/R/F1. ROUGE uses full component text and Porter stemming; only the exact leading expected marker is removed. BERTScore uses pinned roberta-large revision722cf37b1afa9454edce342e7895e588b6ff1d59, layer17, slow tokenizer, FP32 CPU, no IDF/rescaling and512-token prefix. Every row reports original token length and truncation. If either component is empty, all six scores are zero, including both-empty, so absence earns no automatic reward. Nonempty approved references are required by the primary validator.

Paired bootstrap: business unit,20,000 replicates, PCG64 seed20261028, percentile95% CI. F-RO/F-U Review and F-U/F-S Visual are fixed contrasts. Four prespecified BERT F1 sign-flip tests use50,000 replicates and Holm adjustment. Other metric CIs and newly explicit generation-flag sensitivities are descriptive, not confirmatory. Full100 remains reported alongside98-business active Visual /preexisting-flag Review sensitivities. No task-weighted composite is computed.

Reference agreement is not verified factuality. Pipeline ancestry/supervision and original-photo overlap remain limitations:17 TEST businesses have confirmed partial image reuse; no TEST83 replacement is made. Human-versus-Teacher agreement, F-versus-U grounding/coverage, and any final quality ranking remain unresolved until actual reference scores and separate grounded human judgments exist.

## Verification and stop

11 synthetic regression tests passed, including a100-business synthetic Stage L approval/export integration, primary-track rejection of AI provenance, Shuffled recipient alignment, empty policies, flags and paired deterministic statistics. Actual pinned CPU forwards passed on synthetic text and one existing VALIDATION business (two components); these technical numbers are not TEST results or checkpoint comparisons. See separate technical receipt and tests. Teacher request serialization was checked without network; only sampled request serialization is claimed, while all100 source bundles and1,136 photo SHA256 bindings were checked.

STOP: feasible infrastructure and reference-free analysis complete. No paid generation, GPU work, human annotation mutation or final model selection is authorized by this package.
'''
    (OUT/'STAGE_M_REPORT.md').write_text(report,encoding='utf8')
    (OUT/'README_VI.md').write_text('''# Stage M\n\nĐã hoàn tất kiểm tra400 predictions/700 thành phần và tái tính diagnostics trùng byte với Stage K. Không chạy lại inference/training.\n\n- HUMAN_PRIMARY: bảng chờ, cần đủ100 reference do người viết và checker khác duyệt.\n- TEACHER_SUPPLEMENTARY: bảng chờ, chưa có reference; cần credential Teacher riêng và phê duyệt chi phí trước200 request.\n- REFERENCE_FREE_REVIEW/VISUAL: số đo thực, không phải chấm factuality.\n- TECHNICAL_RUNTIME_VERIFICATION: kiểm thử CPU bằng dữ liệu giả lập/VALIDATION, không phải kết quả TEST.\n- TEACHER_PREPARATION: chỉ nguồn và kế hoạch; không chứa đáp án đã sinh.\n\nKhông đưa package quản trị này cho người đang viết/check reference hoặc người chấm model-blinded. Model identities, predictions và diagnostics nằm trong package. App human localhost không được thay đổi.\n\nXem RUN_WHEN_REFERENCES_READY.md cho lệnh chạy sau khi có đầy đủ đầu vào. Stage M dừng ở đây.\n''',encoding='utf8')
    (OUT/'RUN_WHEN_REFERENCES_READY.md').write_text('''# Ready-to-run commands (from project root, PowerShell)

These commands use existing inference artifacts. Outputs must be new directories; no overwrite. Do not execute primary scoring before100 genuine human-written references have independent checker approval. The Stage L final export command itself refuses partial approval. Never import GPT examples as human work.

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_l_reference_app_v1.py export --workspace outputs/stage_l/test100_human_references_v1 --destination outputs/stage_l/approved_test100_final_export_v1 --final
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_m_evaluation_v1.py --preparation outputs/final_evaluation/s1000_preparation_v1 --stage-k outputs/stage_k/s1000_test100_phase1_v1 --track human --references outputs/stage_l/approved_test100_final_export_v1 --bert-model outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large --output outputs/stage_m/human_primary_scored_v1 --researcher-release outputs/stage_m/s1000_final_automatic_v1/RESEARCHER_STAGE_M_REQUEST.txt
```

The primary validator rejects partial exports before loading BERTScore. Only Stage L-approved reference content enters this track. Scoring produces700 component rows,7 condition/task means,24 paired contrasts, descriptive sensitivities and Markdown/LaTeX/CSV/JSON reports.

Teacher generation remains unauthorized to spend. A future JSON paid release must explicitly contain paid_teacher_generation_approved=true, teacher_plan_sha256=<exact TEACHER_PLAN.json file hash>, maximum_requests=200, maximum_output_tokens_per_request=8192 and expires_utc=<valid UTC ISO timestamp>. It must be researcher-approved; this document is NOT such approval. Use only the dedicated Teacher credential mechanism, never Codex credentials. The endpoint/model are historical aliases; record provider-returned identity and usage. No automatic retries/resume.

After explicit spending approval ONLY:

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_m_teacher_references_v1.py --preparation outputs/final_evaluation/s1000_preparation_v1 --output outputs/stage_m/teacher_generation_v1 --execute --plan outputs/stage_m/s1000_final_automatic_v1/TEACHER_PREPARATION/TEACHER_PLAN.json --paid-release <researcher-approved-release.json>
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/stage_m_evaluation_v1.py --preparation outputs/final_evaluation/s1000_preparation_v1 --stage-k outputs/stage_k/s1000_test100_phase1_v1 --track teacher --references outputs/stage_m/teacher_generation_v1 --bert-model outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large --output outputs/stage_m/teacher_supplementary_scored_v1 --researcher-release outputs/stage_m/s1000_final_automatic_v1/RESEARCHER_STAGE_M_REQUEST.txt
```

Raw streamed responses, citations, request/image hashes and usage must all validate. Teacher statements are AI-generated supplementary references, not human-grounded accuracy. Stop on failure; preserve raw evidence. Do not rerun all200 calls because a partial batch failed.
''',encoding='utf8')
    # Immutable source provenance, including original photo bytes; no credentials/DB.
    files={PREP/name:'FROZEN_PREPARATION' for name in ['COHORT100_FREEZE.json','MODEL_AND_DECODER_LOCK.json','METRIC_LOCK.json','TEST100_SHUFFLED_INPUT_MANIFEST.json','TEST100_SHUFFLED_MAPPING.csv','HUMAN_EVALUATION_COHORT30.json']}
    package=PREP/'TEST100_REFERENCE_ANNOTATION_PACKAGE';files[package/'ANNOTATION_PACKAGE_MANIFEST.json']='SOURCE_MANIFEST'
    for r in read(package/'ANNOTATION_PACKAGE_MANIFEST.json')['frozen_sources']:
        p=package/r['source_file'];files[p]='SELECTED_REVIEW_ORIGINAL_PHOTO_BUNDLE'
        for ph in read(p)['photos']:files[ROOT/ph['path']]='ORIGINAL_BYTE_PHOTO'
    for name in ['stage_l_reference_app_v1.py','stage_m_evaluation_v1.py','stage_m_teacher_references_v1.py','test_stage_m_evaluation_v1.py','verify_stage_m_metric_runtime_v1.py','build_stage_m_reports_v1.py','stage_k_reference_free_analysis_v1.py','s1000_final_eval_core_v1.py','score_s1000_automatic_prepared_v1.py','validate_s1000_human_references_v1.py','stage_k_bert_loader_compat_v1.py','stage_g_teacher_credentials.py']:files[ROOT/'scripts'/name]='IMPLEMENTATION'
    for v in [ROOT/'configs/stage_g_teacher_production_d042.json',ROOT/'prompts/stage_g_review_consensus_v7.txt',ROOT/'prompts/stage_g_visual_evidence_v4.txt',ROOT/'schemas/stage_g_teacher_pilot_v3.json',Path('C:/thesis_proposal/thesis_proposal_v16.docx')]:files[v]='METHODOLOGY'
    write_csv(OUT/'INPUT_REPRODUCIBILITY_MANIFEST.csv',[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'role':role,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p,role in sorted(files.items(),key=lambda x:str(x[0]))])
    shutil.copyfile(L/'EVIDENCE/PROPOSAL_V16_REFERENCE_PROTOCOL_EXCERPTS.json',OUT/'PROPOSAL_V16_PROTOCOL_EXCERPTS.json')
    print('Stage M measured diagnostics and separate pending reference reports complete')

if __name__=='__main__':main()
