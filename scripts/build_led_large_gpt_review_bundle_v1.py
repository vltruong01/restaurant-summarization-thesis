"""Build a review-only GPT handoff from the preserved completed Vast run."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'outputs/stage_i/led_large_vast_d059_v1_quality_review'
AUDIT = ROOT/'outputs/stage_i/led_large_vast_d059_v1_download_audit'
FROZEN = ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'
NAME = 'LED_LARGE_VAST_D059_F_GPT_REVIEW_V1'
DEST = ROOT/'outputs/cloud_handoff'/NAME

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def put(rel, value):
    p = DEST/rel
    p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value, bytes):
        p.write_bytes(value)
    elif isinstance(value,str):
        p.write_text(value,encoding='utf-8')
    else:
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    DEST.mkdir(exist_ok=False)
    shutil.copytree(PREP/'portable_review',DEST/'original_review')
    for name in ('QUALITY_REVIEW_VI.md','DESCRIPTIVE_DIAGNOSTICS.json'):
        shutil.copyfile(PREP/name,DEST/name)
    shutil.copyfile(AUDIT/'DOWNLOAD_PRESERVATION_AUDIT.json',DEST/'DOWNLOAD_PRESERVATION_AUDIT.json')
    input_zip = AUDIT/'downloaded_archives/F_large_d059_v1.zip'
    prefix = 'F_large_d059_v1/'
    with zipfile.ZipFile(input_zip) as z:
        included = []
        for info in z.infolist():
            if not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            # Preserve actual readable run records; binary full states remain in the audited full-run ZIP.
            if rel.startswith(('quality_review/','evaluate/epoch')) or not rel.endswith(('.json','.jsonl')):
                continue
            if rel.endswith(('gradient_checks.jsonl','MEMORY_SAMPLES.json')):
                continue
            put('run_evidence/'+rel,z.read(info))
            included.append(rel)
        state = json.loads(z.read(prefix+'train/checkpoint-1250/trainer_state.json'))
        eval_metrics = [r for r in state['log_history'] if 'eval_loss' in r]
        put('TRAINING_VALIDATION_METRICS.json',dict(training=json.loads(z.read(prefix+'train/COMPLETION.json')),
            validation_by_epoch=eval_metrics,source='Unmodified saved checkpoint-1250 trainer_state.log_history'))
        per_tensor = {};samples=[];last=None;count=0
        with z.open(prefix+'train/gradient_checks.jsonl') as f:
            for line in f:
                row=json.loads(line);count+=1;last=row
                if row['step'] in (1,2,3,750,1250):samples.append(row)
                assert len(row['gradients'])==456
                for name,value in row['gradients'].items():
                    t=per_tensor.setdefault(name,dict(observations=0,finite=0,nonzero=0,min_norm=None,max_norm=0.))
                    t['observations']+=1;t['finite']+=int(value['finite']);t['nonzero']+=int(value['nonzero'])
                    t['min_norm']=value['norm'] if t['min_norm'] is None else min(t['min_norm'],value['norm'])
                    t['max_norm']=max(t['max_norm'],value['norm'])
        assert count==1250 and last['step']==1250
        put('GRADIENT_LOG_SUMMARY.json',dict(source='train/gradient_checks.jsonl',source_sha256=hashlib.sha256(z.read(prefix+'train/gradient_checks.jsonl')).hexdigest(),
            rows=count,tensors_per_row=456,per_tensor=per_tensor,
            limitation='Aggregates of logged finite gradients after recovery/clipping; not a proof of healthy generation. Raw overflow attempts have separate records.'))
        put('GRADIENT_SELECTED_STEPS.json',dict(source='Selected rows only, not complete gradient log',rows=samples))
        put('RUN_EVIDENCE_INDEX.json',dict(included_unmodified_run_files=included,full_run_archive_sha256=sha(input_zip),
            omitted='Full model/adapter/optimizer/scaler/RNG binaries, raw gradient dump binaries, full gradient JSONL and memory samples retained locally in full run archive.'))
    with zipfile.ZipFile(AUDIT/'downloaded_archives/launcher_logs.zip') as z:
        for info in z.infolist():
            if not info.is_dir():put(info.filename,z.read(info))
    technical = [
        'scripts/run_stage_i_led_large_vast_d059_v1.py','scripts/stage_i_led_large_vast_d059_v1.py',
        'scripts/stage_i_led_large_full_train_v5.py','scripts/stage_i_led_large_runtime_v2.py',
        'scripts/led_large_dynamic_amp_v2.py','scripts/led_large_gradient_evidence_v3.py',
        'scripts/stage_hi_led_data.py','scripts/stage_i_paired_pilot.py','scripts/stage_i_d058.py',
        'scripts/run_stage_i_d050_feasibility_v1.py','scripts/stage_i_d053_evaluation.py',
        'configs/stage_i_led_large_vast_d059_v1.json','configs/stage_i_d059_expanded_lora_v2.json',
        'VAST_EXECUTION_BINDINGS.json','D059_METHOD_COMPARISON.json',
        'outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
        'outputs/stage_i/led_large_preparation_v1/META_LORA_INVENTORY.json',
        'outputs/stage_i/led_large_preparation_v1/TOKENIZER_DATA_COMPATIBILITY.json',
        'outputs/stage_i/led_large_preparation_v1/remote_metadata/official/config.json',
        'outputs/stage_i/led_large_preparation_v1/remote_metadata/official/README.md',
        'outputs/stage_i/led_large_preparation_v1/REMOTE_PROVENANCE.json',
        'thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md',
    ]
    for rel in technical:
        put('frozen_technical/'+rel,(FROZEN/rel).read_bytes())
    tokenroot=FROZEN/'assets/tokenizers/led-base-16384'
    for p in tokenroot.iterdir():
        if p.is_file():put('frozen_technical/assets/tokenizers/led-base-16384/'+p.name,p.read_bytes())
    runner=DEST/'frozen_technical/scripts/run_stage_i_led_large_vast_d059_v1.py'
    assert sha(runner)=='baac81156359bfdd603468b4710abdaadf0030110ed8455ac180a4cd9def7915'
    cases=json.loads((DEST/'original_review/SOURCES_AND_TEACHER_TARGETS.json').read_text(encoding='utf-8'))
    refs=json.loads((DEST/'original_review/APPROVED_REFERENCES.json').read_text(encoding='utf-8'))
    files=[('Large epoch3','predictions/epoch3.json'),('Large epoch5','predictions/epoch5.json'),
           ('D059 Expanded epoch3','benchmarks/D059_Expanded_epoch3.json'),
           ('D059 Expanded epoch5','benchmarks/D059_Expanded_epoch5.json'),('D058 B epoch5','benchmarks/D058_B_epoch5.json')]
    models={name:json.loads((DEST/'original_review'/rel).read_text(encoding='utf-8')) for name,rel in files}
    index=['# Case-by-case reading index','', 'All 24 fixed DEVELOPMENT cases, same order in every model. Source/Teacher/reference records are unmodified in original_review. These readable views are derived, without assigning labels.','']
    for i,c in enumerate(cases):
        bid=c['business_id'];name=f'{i+1:02d}_{bid}.md'
        lines=[f'# Case {i+1}: {bid} ({c["subset"]})','',f'Source photo/caption count: {len(c["selected_photo_ids"])}','',
            '## Frozen review and caption source','',c['source'],'','## Approved scoring-eligible reference units','']
        units=[u for u in refs['units'] if u['business_id']==bid and u.get('scoring_eligible') is True and u['support_status']=='supported' and u['importance'] in ('core','supporting')]
        lines += [f'- {u["unit_id"]} [{u["importance"]}; {u["domain"]}]: {u["text"]}' for u in units]
        lines += ['', 'Full reference quotes/photo support are in original_review/APPROVED_REFERENCES.json.','',
                  '## Frozen Teacher target (training supervision; not independent truth)','',c['target'],'',
                  '## Historical F saved output','',str(c.get('baseline_prediction','Unavailable'))]
        for label,rows in models.items():
            r=rows[i];assert r['business_id']==bid
            lines += ['',f'## {label}','',r['text'] or '**EMPTY DECODED TEXT**']
        put('cases/'+name,'\n'.join(lines)+'\n')
        index.append(f'- [{i+1:02d} {bid} — {c["subset"]}](cases/{name})')
    put('CASE_INDEX.md','\n'.join(index)+'\n')
    put('PRIMARY_SOURCE_LEADS.md', '''# External diagnostic leads (checked 2026-10-21)

These are historical first-party repository issue reports, not a reproduced diagnosis of this run.

- https://github.com/huggingface/transformers/issues/18190 — LED-large fine-tuning reported repeated BOS and empty decoded text while base remained usable. Discussion considers global attention and BOS-related initialization; commenters report differing outcomes. Older runtime, different dataset and optimization. Similar symptoms do not prove the same cause or validate any workaround here.
- https://github.com/huggingface/transformers/issues/15559 — BART-large generation behavior and BOS handling were discussed. This is a diagnostic lead, not evidence that the installed Transformers 4.51.3 has that historical defect.
- https://huggingface.co/allenai/led-large-16384/blob/cd59d11c3528415b7dda4dfc95cc8f138aceda2e/README.md — pinned official initialization provenance. Local copy is included under frozen_technical.
- https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/led — check against actual pinned source, not assumptions from a newer release.

No internet-suggested fix, weight perturbation, special-token suppression, decoding change or new training was applied. The root cause remains UNCONFIRMED.
''')
    put('README_FIRST.md', '''# LED-large Vast F: independent GPT review package

REVIEW ONLY. Completed F run; no new execution, retry, training, decoding variation, Teacher generation or held-out access is requested or authorized by this package. Historical authorization records describe the finished run only. No automatic RO/U.

Read in order:
1. GPT_REVIEW_REQUEST.md
2. QUALITY_REVIEW_VI.md (prior AI-assisted observations; independently challenge them)
3. DESCRIPTIVE_DIAGNOSTICS.json and TRAINING_VALIDATION_METRICS.json
4. CASE_INDEX.md and cases/ — 24 side-by-side source/output views. Start with cases 01,05,18,21,23,24, then inspect the rest before any all-case quality claim.
5. original_review/RESULTS_REVIEW.html — portable browser view with photos and full source-linked references.
6. frozen_technical/ and run_evidence/ for static implementation/integrity review; launcher_logs/ for full launcher output.

Original raw JSON, 48 token sequences and decoded outputs, saved base benchmarks, sources/captions/photos, Teacher targets and approved references are preserved in original_review/. Its original SHA256_MANIFEST is unchanged. New diagnostics are derived and marked as such. Human output annotation templates remain blank.

Important observed distinction: execution completed successfully, but generation has severe BOS loops, empty outputs and lexical/semantic repetition. Root cause is unconfirmed. The provisional reviewer recommends not adopting these outputs or authorizing matched RO/U on this evidence. Researcher decision remains pending.

Full-state 5 epoch checkpoints and original 450,767,164-byte full-run ZIP are already hash-verified and backed up locally. This lightweight review ZIP intentionally excludes those binaries, full raw gradient history and model weights. Receipts and selected/aggregate gradient evidence are included. It supports review, not standalone forward-pass reproduction.

Historical LED-base matched RO/U/F remain the established baseline. D059 Expanded is a practical unmatched quality benchmark; no causal architecture-size claim. Fixed cases are validation/development, not held-out. D051 strict replay FAIL and earlier D058/D059/D060/Unlikelihood evidence remain unchanged.
''')
    put('GPT_REVIEW_REQUEST.md', '''# Independent review request — respond in Vietnamese

Please review the ACTUAL completed LED-large Filtered run before recommending any further action. Do not execute supplied scripts. Treat the included AI review as provisional and independently verify its claims against raw outputs and evidence.

## Protocol and outcome
Official allenai/led-large-16384, pinned revision cd59d11c3528415b7dda4dfc95cc8f138aceda2e; frozen backbone, fresh all-linear LoRA r8/alpha16, 228 modules/4,915,200 trainable parameters. Frozen Yelp F2000train/200val, five epochs/1250 successful optimizer updates, warmup63, unchanged positive CE/targets/sources; beam4/max_length768, exactly24 cases at epochs3/5. Same fixed random20 and targeted4.

Gate passed 3 successful updates with 2 bounded overflow skips, final scale16384. Fresh training separately started65536; completed1250 updates with5 overflow skips, final scale2048. No evidence that scale alone caused poor output. Full checkpoint hashes and 48 outputs were verified after download. Training loss is teacher-forced, not proof of healthy free-running generation.

Initial software scan: epoch3 empty6/20random+2/4targeted; epoch5 empty8/20random+1/4targeted. Every empty output has token sequence [2]+[0]*766+[2]. Exact-repeat flags epoch3 9/20+2/4, epoch5 7/20+1/4. Do not credit an empty output as successful repetition avoidance. Most nonempty outputs also have long initial BOS runs and malformed markers. D059epoch3 has no empty outputs, exact-repeat3/20+0/4 and correct headers24/24; it still has known content/grounding defects.

## Questions
1. Independently assess harmful exact/semantic repetition, Core business-specific coverage, opinion balance, factual/caption/photo grounding, visual usefulness, zero-caption behavior and usable completion. Separate random20/targeted4, and empty/nonempty denominators. Cite business IDs, checkpoint/epoch, exact text and source/reference IDs. Do not present Teacher targets as independent factual truth.
2. Compare large epoch3/5 with saved D059 Expanded epoch3 as the principal practical benchmark, plus D059epoch5/D058B/historicalF. State capacity, initialization, hardware and training-history confounds; no isolated causal size claim.
3. Audit STATIC code and saved records for possible label right-shift/BOS/EOS/padding mismatch, collator/global-attention errors, adapter save/load/activation differences, train/eval modes, cache behavior and precision differences. Distinguish confirmed defect from plausible concern and missing evidence. Does the supplied evidence identify a code bug, a learned generation collapse, or neither conclusively?
4. Consider the historical LED-large BOS issues in PRIMARY_SOURCE_LEADS.md cautiously. Do not infer that an old version's cause or workaround applies unchanged here. Do not recommend an unverified BOS deletion, embedding perturbation, decoding patch or model switch as a proven repair.
5. Explain why finite loss/gradients and valid hashes can coexist with failed generation. Rank root-cause hypotheses by evidence and give disconfirming checks. Loss-scale backoff is not automatically a quality failure or a successful fix.
6. Recommend the smallest READ-ONLY/OFFLINE diagnostic next step using saved artifacts. If any proposed check needs a model forward pass/GPU/new predictions, separate it and state that it requires a new researcher scope. Do not silently expand the generation budget or propose training/search execution.
7. Give a decision recommendation for CURRENT outputs (researcher confirmation pending), and explain what is required before any matched RO/U decision. Do not infer inherent LED-large incapability from this run or replace only F in the main matched study.

Do not invent all-case Core scores or human labels. State exactly which cases/photos you inspected and what cannot be verified from this lightweight archive. No new experiment is authorized; the researcher will decide after this review.
''')
    put('PACKAGE_CONTEXT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-10-21',client_timezone='Asia/Saigon',
        purpose='Researcher-requested GPT report review before any next action',model_execution_performed_by_builder=False,
        original_review_archive_sha256='1d00c6b446a6c9fb47faf78d20e9faf79564cac1bb2bb3a030ea738b1afc3261',
        human_quality_decision='PENDING',human_output_labels_created=0,new_training_authorized=False,new_generation_authorized=False))
    put('SHA256_MANIFEST.json',dict(files={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}))
    archive=DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
        for rel,digest in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
        old=json.loads(z.read('original_review/SHA256_MANIFEST.json'))['files']
        for rel,digest in old.items():assert hashlib.sha256(z.read('original_review/'+rel)).hexdigest()==digest,rel
        for ep in (3,5):assert len(json.loads(z.read(f'original_review/predictions/epoch{ep}.json')))==24
        for name in z.namelist():
            if name.endswith('RESULTS_REVIEW.html'):
                import re
                text=z.read(name).decode('utf-8')
                for target in re.findall(r'<img src="([^"]+)"',text):assert 'original_review/'+target in z.namelist()
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=len(manifest)+1,
                 all_member_sha256_and_crc='PASS',original_review_manifest='PASS',predictions=48,case_views=24,photo_links='PASS')
    (archive.with_suffix('.zip.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf-8')
    (PREP/'GPT_REVIEW_BUNDLE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
