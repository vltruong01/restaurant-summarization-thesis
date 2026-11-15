"""Offline evidence/report packaging after the one approved run; never retries GPU work.

--watch waits for orchestration exit markers. It does not launch or modify training.
Quality judgments remain explicitly unverified until researcher annotation.
"""
import argparse,collections,datetime,html,json,random,shutil,statistics,time,zipfile
from stage_i_d060 import *

AUDIT=ROOT/'outputs/stage_i/d060_execution_v1'
REPORT=ROOT/'outputs/stage_i/d060_result_report_v1'

def timestamp():return datetime.now(timezone.utc).isoformat()
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def terminal():
    return any((AUDIT/name).exists() for name in ['ORCHESTRATION_COMPLETION.json','STOP_NO_RETRY.json','ORCHESTRATION_FAILURE_STOP.json'])

def finish():
    cfg,m=preflight(allow_existing_run=True);run=ROOT/cfg['run_root'];REPORT.mkdir(exist_ok=False)
    approval=read(AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json')
    require(approval['GPU_execution_approved'] and approval['config_sha256']==sha(CONFIG),'Execution authorization/binding differs')
    # Offline reporting remains permitted after expiry/failure; it never authorizes GPU work.
    complete=(AUDIT/'ORCHESTRATION_COMPLETION.json').exists()
    fail=next((read(AUDIT/n) for n in ['STOP_NO_RETRY.json','ORCHESTRATION_FAILURE_STOP.json'] if (AUDIT/n).exists()),None)
    inputs=[json.loads(s) for s in (BASE_PREP/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    source_root=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1/source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
    cases=read(source_root/'source_cases.json');refs=read(ROOT/cfg['final_references'])
    require([c['business_id'] for c in cases]==[c['business_id'] for c in inputs],'Frozen evidence cases changed')
    from stage_i_d053_evaluation import repetition_diagnostics,structural_diagnostics,token_termination,human_output_annotation_template
    benchmarks=read(ROOT/'outputs/stage_i/d059_result_report_v2/COMPARISON_RECORDS.json')
    require(len(benchmarks)==96,'Missing frozen benchmark outputs')
    rows=[];shells=[];blind=[];key=[];rng=random.Random(20261016)
    for case,c in zip(cases,inputs):
        bid=case['business_id'];units=[u for u in refs['units'] if u['business_id']==bid]
        predictions=[dict(p) for p in benchmarks if p['business_id']==bid]
        require(len(predictions)==4 and {p['arm'] for p in predictions}=={'historical_full_F','D058_B_full_epoch5','Expanded_full_epoch3','Expanded_full_epoch5'},'Wrong benchmark arm inventory')
        require(all(p['source_sha256']==c['source_sha256'] and p['subset']==case['subset'] for p in predictions),'Saved benchmark input mismatch')
        for epoch in [1,3]:
            p=run/'evaluate'/f'epoch{epoch}'/(bid+'.json')
            if p.exists():
                pred=read(p);require(pred['source_sha256']==c['source_sha256'] and pred['subset']==case['subset'],'Prediction/input mapping differs')
                require(pred['reference_sha256']==sha(ROOT/cfg['final_references']),'Prediction reference binding differs')
                predictions.append(pred)
        for p in predictions:
            automatic={'repetition':repetition_diagnostics(p['text'],p['token_ids'],read(BASE_PREP/'expected_generation_config.json')),
                'structure':structural_diagnostics(p['text'],len(case['captions'])),
                'termination':token_termination(p['token_ids'],read(BASE_PREP/'expected_generation_config.json'))}
            rows.append({'business_id':bid,'subset':case['subset'],'arm':p['arm'],'text':p['text'],'token_ids':p['token_ids'],
                'source_sha256':c['source_sha256'],'automatic':automatic,'human_verified':False,'human_quality_status':'PENDING_RESEARCHER_REVIEW',
                'source_grounding':None,'Core_coverage':None,'opinion_balance':None,'semantic_redundancy':None,'useful_visual_information':None})
            shells.append(human_output_annotation_template(p,units))
        shuffled=list(predictions);rng.shuffle(shuffled);blind_outputs=[]
        for i,p in enumerate(shuffled):
            alias='Summary '+chr(65+i);blind_outputs.append({'alias':alias,'text':p['text'],'human_verified':False,
                'usability':None,'semantic_repetition':None,'grounding':None,'Core_information':None,'opinion_contrasts':None,
                'visual_evidence':None,'semantic_completion':None,'notes':'','evidence_references':[]})
            key.append({'business_id':bid,'alias':alias,'arm':p['arm']})
        blind.append({'business_id':bid,'subset':case['subset'],'display_name':case['display_name'],'source':case['source'],
            'reviews':case['reviews'],'captions':case['captions'],'approved_reference_units':units,'outputs':blind_outputs})
    newrows=[r for r in rows if r['arm'].startswith('RO_to_F_epoch')]
    if complete:require(len(newrows)==48,'Completed run missing48 outputs')
    summary=[]
    for arm in ['historical_full_F','D058_B_full_epoch5','Expanded_full_epoch3','Expanded_full_epoch5','RO_to_F_epoch1','RO_to_F_epoch3']:
        for subset in ['random','targeted']:
            group=[r for r in rows if r['arm']==arm and r['subset']==subset]
            if not group:continue
            rep=[x['automatic']['repetition'] for x in group];st=[x['automatic']['structure'] for x in group];te=[x['automatic']['termination'] for x in group]
            summary.append({'arm':arm,'subset':subset,'n':len(group),
                'exact_repeated_sentence_outputs':sum(bool(x.get('exact_repeated_sentence')) for x in rep),
                'adjacent_word_repetition_outputs':sum(bool(x.get('consecutive_repeated_word')) for x in rep),
                'structure_compliant':sum(bool(x.get('section_headers_compliant')) for x in st),
                'cap_hits':sum(x['cap_reached'] for x in te),'EOS_at_cap_may_be_forced':sum(x['eos_at_cap_may_be_forced'] for x in te),
                'mean_tokens_including_decoder_start':statistics.mean(len(x['token_ids']) for x in group),
                'grounding_Core_semantic_quality_status':'NOT_YET_RESEARCHER_VERIFIED'})
    write(REPORT/'COMPARISON_RECORDS.json',rows);write(REPORT/'SOFTWARE_METRICS_BY_SUBSET.json',summary)
    write(REPORT/'BLINDED_REVIEW_CASES.json',blind);write(REPORT/'BLINDING_KEY_KEEP_SEPARATE.json',key)
    write(REPORT/'OUTPUT_ANNOTATION_TEMPLATE_UNVERIFIED.json',{'human_annotations':shells,'status':'UNVERIFIED_OUTPUT_JUDGMENTS_ONLY_SOURCE_REFERENCES_ALREADY_FROZEN'})
    write(REPORT/'FINAL_REFERENCES_UNCHANGED.json',refs);write(REPORT/'EVIDENCE_CASES.json',cases)
    write(REPORT/'TEACHER_COMPARISON_INPUTS.json',inputs)
    for c in cases:
        for cap in c['captions']:
            p=(source_root/cap['image_file']).resolve();require(p.is_relative_to(source_root.resolve()),'Unsafe image path')
            copy(p,REPORT/cap['image_file'])
    report={'status':'TRAIN_AND48_OUTPUTS_COMPLETE_QUALITY_REVIEW_PENDING' if complete else 'EXECUTION_STOPPED_NO_RETRY',
        'UTC':timestamp(),'decision':'D-060','initialization':'exact historical trained RO; fresh Stage2 optimizer/scheduler/scaler/RNG','experiment_completed':complete,'new_predictions':len(newrows),'historical_baseline_reused':24,'D058_epoch5_reused':24,'D059_expanded_epochs3_and5_reused':48,'failure':fail,
        'random_cases':20,'targeted_cases':4,'D051_strict_replay':'FAIL_UNCHANGED','human_quality_judgments_verified':0,
        'GO_NO_GO':'PENDING_RESEARCHER_CONFIRMATION' if complete else 'TECHNICALLY_INCOMPLETE_STOP',
        'no_automatic_future_experiment':True,'Teacher_calls':0,'held_out_access':False,'bindings_verified':len(m['files']),
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER),'execution_authorized_at_utc':approval['execution_authorized_at_utc'],'viability_deadline_utc':approval['viability_deadline_utc'],'original_D059_deadline_utc':approval['original_D059_deadline_utc'],
        'blinding_limit':'Known prior validation examples may be recognized. Keep key and unblinded tables separate until output review.',
        'quality_provenance':'Software metrics only here; substantive evidence-linked AI inspection is still pending, and blank judgments are not AI or human labels. Source refs preserve prior researcher approval.'}
    write(REPORT/'D060_COMPLETION_AND_REVIEW_STATUS.json',report)
    md=['# D-060 execution and quality-review report','',f"Status: **{report['status']}**.",'',
        'One approved RO-initialized Filtered continuation; no silent retry/fallback. Original config/preflight/runner bindings verified. Existing Stage F/G/H data, D053 references and historical checkpoints unchanged. D051 strict replay FAIL preserved.',
        '',f"New outputs: {len(newrows)}/48. Historical full-F outputs reused:24; D058epoch5 outputs reused:24; D059Expanded epochs3/5 reused:48. Random20 and targeted4 are separate below. Historical F is a practical benchmark, not a matched causal architecture control.",'',
        '## Software results','', '| Model | Subset | n | Repeated sentences | Adjacent repeats | Structure | Cap hits | EOS at cap possibly forced | Mean tokens |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in summary:md.append(f"| {x['arm']} | {x['subset']} | {x['n']} | {x['exact_repeated_sentence_outputs']} | {x['adjacent_word_repetition_outputs']} | {x['structure_compliant']} | {x['cap_hits']} | {x['EOS_at_cap_may_be_forced']} | {x['mean_tokens_including_decoder_start']:.1f} |")
    md += ['', '## Quality and independent review','',
        'Grounding, semantic repetition, Core coverage, opinion balance, useful visual information and semantic completeness require evidence-linked inspection. They are **not assessed as human-confirmed here**. All corresponding fields remain null/unverified. Teacher targets are comparisons, not independent ground truth. Inspect BLINDED_REVIEW.html with full reviews, captions, photos and already approved Core/Supporting references. Reuse source annotations; annotate only new outputs.', '',
        'An EOS at the generation cap may be forced and is not proof of natural completion. Correct headers, reduced repetition or lower loss alone cannot yield GO. Apply the approved multidomain rubric separately to random20 and targeted4; supporting omissions are not automatic errors. Zero-caption case must have no visual assertion anywhere.', '',
        'No automatic GO or new training. Researcher must confirm substantial improvement. Mixed/unconvincing completed quality or 72-hour viability deadline -> stop LED optimization and return to original matched RO/U/F. Researcher-confirmed GO permits proposing matched Stage2 RO/U from the same RO initialization, requiring separate authorization within the same72-hour window. No new diagnostics or decoder search is launched.', '',
        '## Execution evidence','',
        'The portable bundle includes exact predictions, sources/teacher targets, approved references, images, automatic metrics, unverified output annotation shells, configs, authorization, logs and checkpoint receipts. Full optimizer/RNG/checkpoint binaries remain in the original run directory; their hashes/receipt paths are preserved. Nothing is overwritten.', '',
        'Inspect epoch checkpoint trainer_state.json and raw logs for training loss, validation loss, LR, update counts, elapsed time and memory. Numerical training success and human summary-quality acceptance are separate.']
    if fail:md+=['','## Failure preserved; no automatic continuation','', '```json',json.dumps(fail,indent=2),'```']
    metrics=training_metrics(run)
    write(REPORT/'TRAINING_AND_VALIDATION_METRICS.json',metrics)
    md += ['', '## Actual training and validation metrics', '', '```json', json.dumps(metrics,indent=2), '```', '', '## Continuation confounds', '', 'Stage1 RO5epochs/1250updates plus Stage2 F3epochs/750updates differs from five F-only epochs of the saved benchmarks. Architecture/parameter counts and initialization also differ. This is a practical quality comparison, not an isolated causal staging result. Selected24 validation cases were repeatedly examined. No held-out quality claim.']
    write(REPORT/'D060_REPORT.md','\n'.join(md));make_html(blind)
    write(REPORT/'README_FIRST.md','''# D060 review

Read D060_REPORT.md. For practical blinding inspect BLINDED_REVIEW.html before
unblinded tables or BLINDING_KEY_KEEP_SEPARATE.json. Every review case includes
the original source, captions/photos and frozen D053 references. There are no new
human quality labels; output-review fields are blank. Do not redo source labels.
Full factual/semantic judgments and final GO/NO-GO remain for the researcher.
No further experiment is automatically authorized or launched.
''')
    write(REPORT/'CHATGPT_REVIEW_REQUEST.md','''# Independent D060 review

Check the single-run authorization and unchanged three hashes, training scope,
epoch1/3 selection, original beam4 generation, split/input/target integrity and
stop-loss policy. Review raw outputs against permitted review and caption/photo
evidence plus frozen D053 references. Report semantic repetition, unsupported
claims/relations/opinion exaggeration, important Core/opinion omissions, useful
visual content, actual semantic completion and zero-caption compliance.
Report random20 separately from targeted4. Historical F is an unmatched practical
benchmark. Distinguish software metrics, AI suggestions and researcher-confirmed
labels. Lower loss/headers alone are insufficient; no automatic GO or more LED
diagnostics. Return evidence-linked findings, then researcher decides GO/NO-GO.
''')
    update_docs(report)
    bundle=ROOT/'outputs/stage_i/stage_i_d060_review_bundle_v1';bundle.mkdir(exist_ok=False)
    for p in REPORT.rglob('*'):
        if p.is_file():copy(p,bundle/p.relative_to(REPORT))
    for folder,dest in [(AUDIT,'execution'),(PREP,'preparation')]:
        for p in folder.rglob('*'):
            if p.is_file():copy(p,bundle/dest/p.relative_to(folder))
    checkpoint_receipts=[]
    if run.exists():
        for p in run.rglob('*'):
            if p.is_file():
                if p.suffix in ['.pt','.pth','.bin','.safetensors']:
                    checkpoint_receipts.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'included_in_zip':False})
                else:copy(p,bundle/'run_evidence'/p.relative_to(run))
    write(bundle/'CHECKPOINT_BINARY_HASHES.json',checkpoint_receipts)
    copy(CONFIG,bundle/'configs'/CONFIG.name)
    for name in ['run_stage_i_d060_v1.py','stage_i_d060.py','execute_stage_i_d060_authorized_v1.py','finish_stage_i_d060_when_ready_v1.py','stage_i_d058.py','stage_i_d053_evaluation.py','stage_i_decoding_diagnostics.py']:
        copy(ROOT/'scripts'/name,bundle/'code'/name)
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        copy(ROOT/rel,bundle/'project_snapshot'/rel)
    manifest={p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    write(bundle/'SHA256_MANIFEST.json',manifest)
    z=bundle.with_suffix('.zip')
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.write(p,bundle.name+'/'+p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(z) as archive:
        for rel,h in manifest.items():
            import hashlib
            require(hashlib.sha256(archive.read(bundle.name+'/'+rel)).hexdigest()==h,'ZIP bytes mismatch')
    write(REPORT/'DELIVERY_RECEIPT.json',{'zip':str(z),'bytes':z.stat().st_size,'sha256':sha(z),'archive_file_hashes_verified':len(manifest),
        'model_or_GPU_calls_by_finisher':0,'status':report['status'],'quality_judgments':'PENDING_RESEARCHER_REVIEW'})
    z.with_suffix('.zip.sha256').write_text(sha(z)+'  '+z.name+'\n',encoding='ascii')
    print(json.dumps({'status':report['status'],'review_zip':str(z),'new_predictions':len(newrows),'no_further_execution':True}),flush=True)

def training_metrics(run):
    train=run/'train';result={'completion':read(train/'COMPLETION.json') if (train/'COMPLETION.json').exists() else None,
        'first_update_headroom':read(train/'FIRST_UPDATE_HEADROOM.json') if (train/'FIRST_UPDATE_HEADROOM.json').exists() else None,
        'initial_model_bindings':read(train/'INITIAL_MODEL_BINDINGS.json') if (train/'INITIAL_MODEL_BINDINGS.json').exists() else None,
        'checkpoint_receipts':[],'validation_losses':[],'training_log':[],'pre_optimizer_gradient_records':0,
        'D051_strict_replay':'FAIL_UNCHANGED','not_a_checkpoint_replay_experiment':True}
    checkpoints=[]
    for step in [250,500,750]:
        p=train/f'checkpoint-{step}'/'checkpoint_integrity.json'
        if p.exists():
            proof=read(p)
            require(proof['config_sha256']==sha(CONFIG),'Checkpoint config binding mismatch')
            for name,h in proof['files'].items():require(sha(p.parent/name)==h,'New checkpoint corruption: '+name)
            result['checkpoint_receipts'].append({'step':step,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'receipt':proof})
            checkpoints.append(p.parent)
    if checkpoints:
        state=read(checkpoints[-1]/'trainer_state.json')
        result['saved_global_step']=state['global_step'];result['saved_epoch']=state['epoch']
        result['validation_losses']=[x for x in state['log_history'] if 'eval_loss' in x]
        result['training_log']=[x for x in state['log_history'] if 'loss' in x]
    p=train/'gradient_checks.jsonl'
    if p.exists():
        records=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        result['pre_optimizer_gradient_records']=len(records)
        result['gradient_tensors_per_record']=sorted({len(x['branches']) for x in records})
        result['all_recorded_gradients_finite']=all(g['finite'] for r in records for g in r['branches'].values())
        result['all_recorded_gradients_nonzero']=all(g['nonzero'] for r in records for g in r['branches'].values())
        result['gradient_norm_range']=[min(g['norm'] for r in records for g in r['branches'].values()),max(g['norm'] for r in records for g in r['branches'].values())] if records else None
    if result['completion']:
        require(result['completion']['steps']==750 and len(checkpoints)==3,'Incomplete approved training evidence')
        require(result['pre_optimizer_gradient_records']==750 and result['gradient_tensors_per_record']==[48],'Gradient evidence incomplete')
    return result

def make_html(cases):
    sections=[]
    for i,c in enumerate(cases):
        cards=[]
        for j,o in enumerate(c['outputs']):
            k=f'{i}_{j}'
            cards.append('<article><h3>'+o['alias']+'</h3><pre>'+html.escape(o['text'])+'</pre><label>Usability <select data-key="'+k+'" data-field="usability"><option value="">Unreviewed</option><option>usable</option><option>minor_repair</option><option>unusable</option><option>uncertain</option></select></label><textarea data-key="'+k+'" data-field="notes" placeholder="Grounding / semantic repetition / Core / opinion contrasts / visuals / completion; cite REV/IMG_CAP/unit IDs. AI advice must be identified separately."></textarea><label><input type="checkbox" data-key="'+k+'" data-field="researcher_confirmed"> Researcher explicitly confirms this output assessment after evidence review</label></article>')
        imgs=''.join('<figure><img loading="lazy" src="'+html.escape(x['image_file'],quote=True)+'"><figcaption>'+html.escape(x['caption_id']+' / '+x['photo_id']+': '+x['text'])+'</figcaption></figure>' for x in c['captions'])
        units='\n'.join(u['unit_id']+' ['+u['importance']+'] '+u['text'] for u in c['approved_reference_units'])
        sections.append('<section><h2>'+html.escape(c['display_name']+' / '+c['business_id']+' / '+c['subset'])+'</h2><details><summary>Complete source evidence — inspect before judging outputs</summary><pre>'+html.escape(c['source'])+'</pre>'+imgs+'</details><details><summary>Existing researcher-approved Core / Supporting units</summary><pre>'+html.escape(units)+'</pre></details><div class="cards">'+''.join(cards)+'</div></section>')
    js='''const storeKey='D060_output_review_v1';let state=JSON.parse(localStorage.getItem(storeKey)||'{}');document.querySelectorAll('[data-key]').forEach(e=>{let k=e.dataset.key,f=e.dataset.field;if(state[k]?.[f]!==undefined){if(e.type==='checkbox')e.checked=state[k][f];else e.value=state[k][f];}e.addEventListener('change',()=>{state[k]??={};state[k][f]=e.type==='checkbox'?e.checked:e.value;localStorage.setItem(storeKey,JSON.stringify(state));});});function exportReview(){let x={stage:'D060',status:'output_review_progress_only',annotation_source:'researcher_interface',source_reference_annotations_unchanged:true,assessments:state};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(x,null,2)],{type:'application/json'}));a.download='D060_output_review_progress.json';a.click();}'''
    write(REPORT/'BLINDED_REVIEW.html','<!doctype html><meta charset="utf-8"><title>D060 blinded output review</title><style>body{font:16px system-ui;margin:24px;background:#f4f5f7;color:#17212b}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}article,section{background:white;padding:16px;margin:18px 0;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui}textarea{width:100%;min-height:110px;margin-top:12px}img{max-width:700px;max-height:500px}label{display:block;margin:10px 0}button{padding:12px;position:sticky;top:8px}@media(max-width:1000px){.cards{grid-template-columns:1fr}}</style><h1>D060: output review, not source reannotation</h1><p>Model identities hidden per case. Prior familiarity limits blinding. Inspect all sources, approved references and relevant photos; do not count source units as new claims needing relabeling. No quality labels are prefilled. A checkbox requires your explicit confirmation; exported case-level notes do not pretend to be a complete claim-level factual metric.</p><button onclick="exportReview()">Export review progress</button>'+''.join(sections)+'<script>'+js+'</script>')

def update_docs(report):
    entry=f'''## D-060 — Execution finished; independent output-quality review pending

Approved hash-bound single-run execution status: **{report['status']}**.
New epoch1/3 outputs:{report['new_predictions']}/48; saved historicalF24,D058epoch5 24,D059Expanded48 reused.
Evidence/software report:`outputs/stage_i/d060_result_report_v1/D060_REPORT.md`.
Blinded review:`outputs/stage_i/d060_result_report_v1/BLINDED_REVIEW.html`;
review ZIP:`outputs/stage_i/stage_i_d060_review_bundle_v1.zip`.
Source references remain frozen; output semantic/grounding/Core/opinion/visual
judgments are unverified, not fabricated human labels. Software results alone
do not authorize GO. Random20 and targeted4 reported separately. D051 strict
replay FAIL unchanged. No retries, Teacher/test/data changes or next experiment.
Stop for researcher-confirmed substantial usable-quality decision; mixed/poor
quality means return to original matched RO/U/F, not open another optimization series.

'''
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/rel;old=p.read_text(encoding='utf-8');write(REPORT/'documentation_before'/p.name,old)
        head,_,body=old.partition('\n');p.write_text(head+'\n\n'+entry+body.lstrip('\n'),encoding='utf-8',newline='\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    require(not REPORT.exists(),'Result report already exists; no automatic report overwrite')
    if args.watch:
        write(AUDIT/'OFFLINE_FINISHER_WATCH_STARTED.json',{'UTC':timestamp(),'GPU_calls':0,'scope':'Wait, summarize existing artifacts, prepare review evidence, stop'})
        while not terminal():time.sleep(30)
    require(terminal(),'Approved run has not finished/stopped')
    try:finish()
    except Exception as exc:
        import traceback
        write(AUDIT/'OFFLINE_FINISHER_FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc(),'no_GPU_or_retry':True})
        raise

if __name__=='__main__':main()
