"""Additive CAD closeout/readiness audit; no model, network or held-out reads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/main_rouf_shuffled_readiness_v2'
OLD=R/'outputs/stage_i/main_rouf_shuffled_readiness_v1'
B=R/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
L=R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2'
CAD=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1'
observed={}


def guard(p):
    parts=p.as_posix().replace('\\','/').lower().split('/')
    if p.suffix.lower() in ('.json','.jsonl','.npz','.parquet','.csv','.jpg','.png'):
        assert 'test' not in parts and 'heldout' not in parts and 'held_out' not in parts, 'Forbidden held-out read: '+str(p)
        assert not (p.name.lower().startswith('test_') and p.suffix.lower() in ('.jsonl','.npz','.parquet','.csv'))
    return p
def key(p):
    try:return p.relative_to(R).as_posix()
    except ValueError:return str(p)
def record(p):
    p=guard(p);k=key(p)
    if k not in observed:
        with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
        observed[k]=dict(bytes=p.stat().st_size,sha256=d)
    return observed[k]
def read(p):
    record(p);return json.loads(p.read_text(encoding='utf-8'))
def write(name,value):
    with (P/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def check_manifest(root,entries):
    for e in entries:
        x=record(root/e['path']);assert x['sha256']==e['sha256'],e['path']
        if 'bytes' in e:assert x['bytes']==e['bytes'],e['path']
    return dict(status='PASS',files=len(entries))
def zip_check(path,count):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        files=json.loads(z.read('SHA256_MANIFEST.json'))['files'];assert len(files)==count
        assert set(z.namelist())==set(files)|{'SHA256_MANIFEST.json'}
        for n,d in files.items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
    return dict(path=key(path),entries_verified=count,**record(path))


def main():
    utc=datetime.now(timezone.utc).isoformat()
    assert not (P/'CAD_CLOSURE_RECEIPT.json').exists(),'New namespace/version required'
    # Recheck prior verified inputs, allowing only code named test_* (not test payloads).
    old_input=read(OLD/'INPUT_SHA256_MANIFEST.json')['files']
    prior=[]
    for name,expected in old_input.items():
        path=Path(name) if Path(name).is_absolute() else R/name
        if path.is_file():
            actual=record(path);assert actual==expected,name
            prior.append(dict(path=name,status='PASS'))
        else:prior.append(dict(path=name,status='MISSING'))
    write('PREVIOUS_INPUT_REVERIFICATION.json',dict(entries=len(prior),
        verified=sum(x['status']=='PASS' for x in prior),missing=[x['path'] for x in prior if x['status']=='MISSING']))
    original=zip_check(R/'outputs/cloud_handoff/LED_LARGE_CAD_V1_VAST_CHATGPT_REVIEW.zip',213)
    corrected=zip_check(R/'outputs/cloud_handoff/LED_LARGE_CAD_V1_COMPLETE_REVIEW_CORRECTED_SUBMISSION_V1.zip',221)
    export=read(CAD/'downloaded/ARTIFACT_MANIFEST.json')['files']
    for n,d in export.items():assert record(CAD/'downloaded'/n)==d,n
    assert len(export)==146
    # Fresh source-only CAD identity check; researcher verified evidence offsets are attributed, not fabricated.
    sources=read(R/'outputs/stage_i/led_large_cad_v1/TRAIN_SOURCE_ONLY_CASES.json')
    pairs=read(CAD/'downloaded/run/PAIRED_OUTPUTS.json')
    assert len(sources)==len(pairs)==12
    for source,pair in zip(sources,pairs):
        assert source['business_id']==pair['business_id'] and source['split']=='train'
        assert hashlib.sha256(source['source'].encode()).hexdigest()==source['source_sha256']==pair['source_sha256']
        assert pair['vanilla']['text']==pair['alpha0']['text']
        assert pair['vanilla']['raw_token_ids']==pair['alpha0']['raw_token_ids']
    independent=P/'INDEPENDENT_RESEARCHER_REVIEW.md'
    closure=dict(recorded_utc=utc,authority='Latest independent researcher review of corrected complete CAD results',
        technical_decision='PASS',scientific_decision='NO_GO_FOR_PRIMARY_ADOPTION_FIXED_CAD_ALPHA_0_3',
        study='12 previously examined TRAIN cases;9 captioned/3zero-caption;not independent generalization',
        interpretation='Technically successful but scientifically unsuccessful exploratory decoding pilot for this fixed configuration; no universal CAD ineffectiveness claim',
        independent_review=dict(path=key(independent),**record(independent)),
        reviewer_reported_verification=dict(original_manifest=213,execution_export=146,source_bindings=12,
            quoted_evidence_offsets=54,vanilla_alpha0_token_text_identities=12),
        this_audit_verification=dict(original_archive=original,corrected_archive=corrected,
            execution_files=146,source_bindings=12,alpha0_identities=12),
        completed_businesses=12,completed_generation_calls=37,new_generation_calls=0,
        CAD_tuning_authorized=False,CAD_rerun_authorized=False,CAD_combination_authorized=False,
        promote_to_RO_U_F_Shuffled=False,new_decoding_method_authorized=False,
        Boundary_status='PAUSED_NO_FURTHER_GPU_DIAGNOSTICS_OR_TRAINING',
        GPU_execution_authorized=False,held_out_access_authorized=False,
        preserve_all_original_artifacts=True,
        supersedes_current_status='outputs/stage_i/led_large_cad_v1_workflow_correction_v1/CURRENT_WORKFLOW.json',
        historical_closure_and_wrong_folder_correction_preserved=True)
    write('CAD_CLOSURE_RECEIPT.json',closure)
    backup=read(B/'BACKUP_MANIFEST.json')
    base_proof=check_manifest(B,backup['files']);base={};configurations=[];valsets=[]
    import numpy as np  # Only immutable TRAIN/validation ID arrays, no model imports.
    h=read(R/'outputs/stage_h/led_dataset_v1/manifest.json');trainsets=[];frozen=[]
    for name,d in h['datasets'].items():
        assert not name.startswith('test_')
        assert record(R/d['path'])['sha256']==d['sha256']
        assert record(R/d['stage_g_jsonl'])['sha256']==d['stage_g_jsonl_sha256']
        frozen.append(dict(dataset=name,rows=d['rows'],source_pack_sha256=d['sha256'],original_target_file_sha256=d['stage_g_jsonl_sha256']))
    for condition in ('reviews_only','unfiltered','filtered'):
        out=B/'outputs/stage_i/runs_v3'/condition
        done=read(out/'completion_report.json');state=read(out/'trainer_state.json')
        runtime=read(out/'runtime_initial.json');binding=read(out/'run_binding.json')
        selected=out/'best_adapter/adapter_model.safetensors'
        assert record(selected)['sha256']==record(out/'checkpoint-1250/adapter_model.safetensors')['sha256']
        assert done['test_used'] is False and state['global_step']==1250 and state['epoch']==5
        predictions_path=out/'validation_predictions.jsonl';record(predictions_path)
        predictions=[json.loads(s) for s in predictions_path.read_text(encoding='utf-8').splitlines()]
        with np.load(R/h['datasets']['train_'+condition]['path'],allow_pickle=False) as d:train_ids=set(d['business_ids'].tolist())
        with np.load(R/h['datasets']['validation_'+condition]['path'],allow_pickle=False) as d:val_ids=set(d['business_ids'].tolist())
        assert len(train_ids)==2000 and len(val_ids)==200 and not train_ids&val_ids
        assert len(predictions)==200 and {x['business_id'] for x in predictions}==val_ids
        assert all(x['condition']==condition and x['no_prediction_rewrite'] for x in predictions)
        cp=[]
        for step in (1000,1250):
            folder=out/f'checkpoint-{step}'
            full=('adapter_model.safetensors','adapter_config.json','training_args.bin','optimizer.pt',
                'scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json')
            assert all((folder/n).is_file() for n in full)
            cp.append(dict(step=step,full_state_files=[key(folder/n) for n in full]))
        base[condition]=dict(epochs=5,updates=1250,selected_step=1250,selected_adapter=key(selected),
            selected_adapter_sha256=record(selected)['sha256'],selection_policy='Best validation loss;1250 selected for all',
            trainable_parameters=done['trainable_parameters'],settings_signature=binding['settings_signature'],
            retained_checkpoints=cp,validation_predictions=200,
            metrics=dict(train_loss=done['metrics']['train_loss'],best_validation_CE=done['best_eval_loss'],
                train_runtime_seconds=done['metrics']['train_runtime'],format_warnings=done['validation_generation_format_warnings'],
                cap_warnings=done['validation_generation_cap_warnings']))
        configurations.append(runtime['config']);valsets.append(val_ids);trainsets.append(train_ids)
    assert configurations[0]==configurations[1]==configurations[2]
    assert valsets[0]==valsets[1]==valsets[2] and trainsets[0]==trainsets[1]==trainsets[2]
    done=read(L/'train/COMPLETION.json');assert done['epochs']==5 and done['steps']==1250
    checkpoints=[]
    for step in (250,500,750,1000,1250):
        cp=L/f'train/checkpoint-{step}';proof=read(cp/'checkpoint_integrity.json');assert proof['global_step']==step
        verified=check_manifest(cp,[dict(path=n,sha256=d) for n,d in proof['files'].items()])
        checkpoints.append(dict(step=step,path=key(cp),**verified))
    assert sum(x['files'] for x in checkpoints)==50
    adapter=L/'train/checkpoint-750/adapter_model.safetensors'
    assert record(adapter)['sha256']=='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'
    econfig=read(R/'configs/stage_i_d053_evaluation_v1.json')
    expected=set(econfig['random_business_ids']+econfig['targeted_business_ids']);assert len(expected)==24 and expected<=valsets[0]
    outputs={}
    for epoch in (3,5):
        path=L/f'evaluate/epoch{epoch}/predictions.json';rows=read(path)
        assert len(rows)==24 and {x['business_id'] for x in rows}==expected
        outputs[str(epoch)]=dict(path=key(path),count=len(rows),**record(path))
    labels=read(R/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json')
    for v in labels['views'].values():assert record(R/v['path'])['sha256']==v['sha256']
    assert set(labels['views'])=={'train_filtered','validation_filtered'}
    large=dict(epochs=5,updates=1250,checkpoints=checkpoints,selected_F_candidate=dict(step=750,
        path=key(adapter),sha256=record(adapter)['sha256'],status='Preserved current F candidate; confirm common family-selection contract'),
        trainable_parameters=4915200,development_predictions=outputs,unique_validation_businesses=24,
        train_runtime_seconds=done['metrics']['train_runtime'],evaluation=read(L/'evaluate/COMPLETION.json'),
        complete_matched_RO_U='NOT_FOUND_IN_CANONICAL_INVENTORY',
        RO_U_materialized_corrected_views='MISSING',RO_U_condition_runner_bindings='MISSING')
    write('CHECKPOINT_INVENTORY.json',dict(base=base,base_backup_reverified=base_proof,
        base_runtime_config_exact=True,base_validation_businesses_exact=True,large=large,
        frozen_train_validation=frozen,hash_integrity_is_not_runtime_restore_test=True))
    REF=R/'outputs/stage_i/d053_final_references_v1'
    refproof=check_manifest(REF,read(REF/'REFERENCE_FREEZE_MANIFEST.json')['files'])
    ref=read(REF/'D053_FINAL_REFERENCE_READINESS.json')
    ratings=read(R/'outputs/stage_i/led_large_v2_blinded_quality_review_v1/BLINDED_REVIEW/BLANK_RESEARCHER_ANNOTATIONS.json')['ratings']
    metric=dict(base_native_metrics={k:v['metrics'] for k,v in base.items()},
        large_train_metrics=done['metrics'],loss_is_not_scientific_quality=True,
        source_references=dict(integrity=refproof,businesses=ref['businesses'],core=ref['core'],
            supporting=ref['supporting'],excluded=ref['excluded'],unresolved=ref['unresolved_researcher_decisions'],
            scope='24 Filtered validation businesses, not held-out references'),
        saved_development_rating_slots=len(ratings),researcher_confirmed_slots=sum(x.get('researcher_confirmed') is True for x in ratings),
        approved_development_metric_contract=econfig['metric_contract'],development_bootstrap=econfig['bootstrap'],
        final_test_metrics='PENDING_NO_TEST_OUTCOMES_OPENED',test_reference_status='NOT_FOUND_IN_AUDITED_CANONICAL_NAMESPACES',
        formal_study_plan=dict(businesses=30,conditions=4,annotators=3,output_rating_sets=360,
            saved_final_results='NOT_FOUND_IN_AUDITED_CANONICAL_NAMESPACES'),
        ROUGE_BERTScore='Planned in thesis;approved pinned final scorer/reference binding unresolved;not computed here')
    write('METRICS_AND_REFERENCES.json',metric)
    pointer=read(R/'outputs/stage_f/production_manifest_pointer.json')
    assert pointer['shuffled_assignment_pending'] is True
    # Directory/path inventory only: never open test payloads or held-out membership CSVs.
    listed=subprocess.run(['rg','--files','outputs/stage_i','outputs/stage_h','outputs/stage_f'],
        cwd=R,capture_output=True,text=True,encoding='utf-8',check=True).stdout.splitlines()
    normalized=[s.replace('\\','/') for s in listed]
    excluded=('main_rouf_shuffled_readiness','/downloaded/locked_payload/','/contrastive_closure/')
    shuffled=[s for s in normalized if not any(t in s for t in excluded) and 'shuffl' in s.lower() and
        (Path(s).suffix.lower() in ('.npz','.jsonl','.parquet') or
        (Path(s).suffix.lower()=='.json' and any(x in Path(s).name.lower() for x in ('assignment','mapping','prediction','source','completion'))))]
    candidates=[s for s in normalized if 'large' in s.lower() and Path(s).name in ('COMPLETION.json','completion_report.json')
        and any(p in s.lower().split('/') for p in ('reviews_only','unfiltered','ro','u'))]
    final_namespaces=[p.name for p in (R/'outputs/stage_i').iterdir() if p.is_dir() and any(x in p.name.lower()
        for x in ('final_test','human_study','human_evaluation','heldout','held_out','shuffled','large_ro','large_unfiltered'))]
    write('SEARCH_SCOPE_AND_BLOCKERS.json',dict(inventory_method='rg --files; filenames only;no held-out reads',
        matched_large_RO_U_completion_candidates=candidates,shuffled_data_candidates=shuffled,
        final_result_namespace_candidates=final_namespaces,search_does_not_cover_browser_or_unimported_external_files=True,
        test_caption_paths=[dict(path=v,exists=(R/v).is_file(),content_opened=False) for v in pointer['split_manifests']['test'].values()],
        test_StageH_packs={c:(R/f'outputs/stage_h/led_dataset_v1/test_{c}.npz').is_file() for c in base},
        shuffled_assignment='PENDING_CANONICAL_POINTER',held_out_reads=0,
        blockers=['Select/confirm matched family and common checkpoint policy','Large RO/U condition bindings/views and separate GPU approval',
            'Actual Shuffled sampler/mapping/constraints binding','Target-free test sources and approved source-access scope',
            'Independent test references and pinned final scorers','Formal30 sample/blinded3-annotator ratings/agreement',
            'Real calendar fit with protected evaluation/writing/defense reserve']))
    assert not candidates and not shuffled,'New candidate artifacts found; inspect metadata before declaring missing'
    for path in [R/'configs/project.yaml',R/'configs/image_overlap_evaluation_policy.yaml',
        R/'configs/led_source_template_d032.json',R/'configs/stage_i_led_large_decoder_policy_v2.json',
        R/'specs/009_lora_led_training_preparation.md',R/'specs/009m_d059_expanded_lora_viability.md',
        R/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        record(path)
    # Pause/closure is recorded additively; no historical workflow receipt is overwritten.
    historical=[OLD/'RESEARCHER_CAD_CLOSURE.json',
        R/'outputs/stage_i/led_large_cad_v1_workflow_correction_v1/CURRENT_WORKFLOW.json',
        R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_V1_OFFLINE_REVIEW.zip',
        R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1_OFFLINE_REVIEW.zip',
        R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1_PRE_GPU_STOP_REVIEW.zip']
    preserved={key(p):record(p) for p in historical}
    write('CURRENT_WORKFLOW.json',dict(recorded_utc=utc,CAD='CLOSED_TECHNICAL_PASS_SCIENTIFIC_NO_GO_FIXED_ALPHA_0_3',
        CAD_tuning_or_rerun_authorized=False,Boundary='PAUSED',Contrastive='REMAINS_CLOSED',
        priority='Complete controlled matched-family comparisons,final evaluation,human study and thesis;no open-ended F optimization',
        intended_large_family='F available;RO/U incomplete;separate completion approval required',
        existing_base_family='Complete historical matched baseline;fallback choice needs researcher decision',
        GPU_execution_authorized=False,held_out_access_authorized=False,instance_lifecycle_actions_authorized=False,
        authoritative_closure=key(P/'CAD_CLOSURE_RECEIPT.json'),historical_receipts_overwritten=False))
    write('PRESERVATION_AND_SCOPE.json',dict(original_F_adapter=dict(path=key(adapter),**record(adapter)),
        old_workflows_and_Boundary=preserved,original_CAD_archive=original,corrected_CAD_archive=corrected,
        model_loads=0,GPU_operations=0,generations=0,optimizer_updates=0,Teacher_calls=0,network_calls=0,
        held_out_payload_reads=0,frozen_scientific_artifact_writes=0,new_evaluation_metrics_computed=0,
        file_hash_checks_are_not_new_GPU_or_runtime_tests=True))
    # Make the current workflow discoverable, preserving every previous progress-log byte.
    progress=R/'thesis_sdd_docs/THESIS_PROGRESS.md';old_bytes=progress.read_bytes()
    prefix=('## Corrected CAD independent review finalized; controlled-family completion priority (2026-10-24)\n\n'
        'Latest researcher review of the correct complete CAD RTX3090 package:Technical PASS,Scientific NO-GO for fixed alpha0.3 primary adoption. '
        'Preserve all12TRAIN pairs,37calls and limitations;no CAD rerun/tuning/combination. Boundary PAUSED;no GPU or held-out authorization. '
        'Read-only V2 readiness rechecks historical matched base RO/U/F,large F five checkpoints,source/reference bindings and missing controls. '
        'Intended large-family path needs matched RO/U only,then Shuffled via F and approved final evaluation;do not mix model families. '
        'Existing base is retained as an established baseline/fallback,not automatically substituted for the large-family goal. '
        'Stop optimization and prioritize references,final400outputs and planned30-business/3-annotator study. '
        'Current authority and blockers:outputs/stage_i/main_rouf_shuffled_readiness_v2/CAD_CLOSURE_RECEIPT.json and READINESS_AND_COMPLETION_PLAN.md. '
        'All older notes below remain historical.\n\n').encode()
    progress.write_bytes(prefix+old_bytes)
    assert progress.read_bytes()[len(prefix):]==old_bytes
    write('PROGRESS_NOTE_PRESERVATION.json',dict(path=key(progress),previous_body_sha256=hashlib.sha256(old_bytes).hexdigest(),
        current_sha256=hashlib.sha256(prefix+old_bytes).hexdigest(),previous_body_preserved_exact=True,scientific_artifact=False))
    write('INPUT_SHA256_MANIFEST.json',dict(files=observed))
    corrected_path=R/corrected['path']
    archive=R/'outputs/cloud_handoff/CAD_V1_CLOSURE_AND_PRIMARY_READINESS_V2.zip'
    assert not archive.exists()
    members=[x for x in P.iterdir() if x.is_file()]+[Path(__file__)]
    payload={key(p):p.read_bytes() for p in members}
    payload['READ_THIS_FIRST.md']=(P/'STATUS_VI.md').read_bytes()
    payload['historical_CAD_review/LED_LARGE_CAD_V1_COMPLETE_REVIEW_CORRECTED_SUBMISSION_V1.zip']=corrected_path.read_bytes()
    proof={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(payload.items()):z.writestr(n,b)
        z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=proof),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==d for n,d in proof.items())
    for n,e in preserved.items():
        with (R/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==e['sha256']
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    with archive.with_suffix('.zip.sha256').open('x',encoding='ascii') as f:f.write(digest+'  '+archive.name+'\n')
    write('DELIVERY_RECEIPT.json',dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,
        manifest_entries=len(proof),verified_input_files=len(observed),CAD='Technical PASS/Scientific NO-GO',
        Boundary='PAUSED',GPU_execution_authorized=False,held_out_payload_reads=0))
    print(json.dumps(json.loads((P/'DELIVERY_RECEIPT.json').read_text(encoding='utf-8')),indent=2))


if __name__=='__main__':main()
