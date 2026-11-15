"""Read-only scientific-artifact audit. No ML imports, network, or test payload reads."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import zipfile

R = Path(__file__).resolve().parents[1]
O = R / 'outputs/stage_i/main_rouf_shuffled_readiness_v1'
B = R / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
L = R / 'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2'
REF = R / 'outputs/stage_i/d053_final_references_v1'
observed = {}


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rel(p):
    try:
        return p.relative_to(R).as_posix()
    except ValueError:
        return str(p)


def record(p):
    key = rel(p)
    if key not in observed:
        observed[key] = dict(bytes=p.stat().st_size, sha256=sha(p))
    return observed[key]


def read(p):
    record(p)
    return json.loads(p.read_text(encoding='utf-8'))


def write(name, value):
    with (O / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def manifest_check(root, entries):
    results = []
    for entry in entries:
        p = root / entry['path']
        actual = record(p)
        ok = actual['sha256'] == entry['sha256']
        if 'bytes' in entry:
            ok = ok and actual['bytes'] == entry['bytes']
        results.append(dict(path=rel(p), matches=ok))
    assert all(x['matches'] for x in results), results
    return dict(status='PASS', files=len(results), entries=results)


def main():
    O.mkdir(exist_ok=False)
    utc = datetime.now(timezone.utc).isoformat()
    closure = dict(recorded_utc=utc, source='Current explicit independent researcher review',
        decision='FINAL_RESEARCHER_NO_GO_CAD_V1_ALPHA_0_3',
        CAD_tuning_authorized=False, Contrastive_reopening_authorized=False,
        Original_CE_epoch3_preserved=True, historical_reports_overwritten=False,
        limitations='Bounded, previously examined TRAIN sample; do not infer universal CAD/Contrastive ineffectiveness',
        GPU_execution_authorized=False, current_instance_policy='KEEP; no lifecycle operations')
    cad_zip = R/'outputs/cloud_handoff/LED_LARGE_CAD_V1_VAST_CHATGPT_REVIEW.zip'
    with zipfile.ZipFile(cad_zip) as z:
        entries = json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in entries.items())
    closure['sealed_CAD_review'] = dict(path=rel(cad_zip), **record(cad_zip), manifest_entries=len(entries))
    write('RESEARCHER_CAD_CLOSURE.json', closure)

    backup = read(B/'BACKUP_MANIFEST.json')
    base_integrity = manifest_check(B, backup['files'])
    H = read(R/'outputs/stage_h/led_dataset_v1/manifest.json')
    frozen = []
    for k, v in H['datasets'].items():
        for key, expected in [(v['path'], v['sha256']), (v['stage_g_jsonl'], v['stage_g_jsonl_sha256'])]:
            assert record(R/key)['sha256'] == expected, key
        frozen.append(dict(dataset=k, rows=v['rows'], npz_sha256=v['sha256'], target_jsonl_sha256=v['stage_g_jsonl_sha256']))
    base = {}
    configs = []
    ids = []
    for condition in ('reviews_only', 'unfiltered', 'filtered'):
        p = B/'outputs/stage_i/runs_v3'/condition
        done, state, runtime, binding = [read(p/n) for n in ('completion_report.json','trainer_state.json','runtime_initial.json','run_binding.json')]
        pred = p/'validation_predictions.jsonl'
        record(pred)
        rows = [json.loads(s) for s in pred.read_text(encoding='utf-8').splitlines()]
        bid = {x['business_id'] for x in rows}
        assert len(rows) == len(bid) == 200
        assert all(x['condition'] == condition and x['token_ids'] and x['no_prediction_rewrite'] for x in rows)
        selected = p/'best_adapter/adapter_model.safetensors'
        cp = p/'checkpoint-1250/adapter_model.safetensors'
        assert record(selected)['sha256'] == record(cp)['sha256']
        full_state = ['adapter_model.safetensors','adapter_config.json','training_args.bin','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json']
        present = {n:(p/'checkpoint-1250'/n).is_file() for n in full_state}
        assert all(present.values())
        assert state['global_step'] == 1250 and state['epoch'] == 5
        assert done['test_used'] is False
        base[condition] = dict(path=rel(p), completed_epochs=state['epoch'], updates=state['global_step'],
            selected_checkpoint_step=1250, checkpoint_policy='best validation loss; epoch5 selected in all three',
            settings_signature=binding['settings_signature'], runner_sha256=binding['runner_sha256'],
            selected_adapter=rel(selected), selected_adapter_sha256=record(selected)['sha256'],
            retained_checkpoint_steps=[1000,1250], full_state_components=present,
            validation_predictions=200, predictions_sha256=record(pred)['sha256'], test_used=False,
            trainable_parameters=done['trainable_parameters'])
        configs.append(runtime['config']); ids.append(bid)
    assert configs[0] == configs[1] == configs[2]
    assert ids[0] == ids[1] == ids[2]
    assert len({v['settings_signature'] for v in base.values()}) == 1
    base_integrity['matched_config_exact'] = True
    base_integrity['same_200_validation_businesses'] = True
    base_integrity['backbone_revision'] = configs[0]['revision']
    base_integrity['generation'] = configs[0]['generation']
    base_cache = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub/models--allenai--led-base-16384/snapshots'/configs[0]['revision']/'pytorch_model.bin'
    base_integrity['local_pinned_backbone'] = dict(path=str(base_cache), exists=base_cache.is_file())
    if base_cache.is_file():
        base_integrity['local_pinned_backbone'].update(record(base_cache))
    write('BASE_MATCHED_FAMILY_AUDIT.json', dict(backup=base_integrity, arms=base, frozen_training_and_validation=frozen))

    done = read(L/'train/COMPLETION.json')
    checkpoints = []
    for step in (250,500,750,1000,1250):
        cp = L/f'train/checkpoint-{step}'
        proof = read(cp/'checkpoint_integrity.json')
        assert proof['global_step'] == step
        result = manifest_check(cp, [dict(path=n,sha256=h) for n,h in proof['files'].items()])
        checkpoints.append(dict(step=step, **result))
    assert record(L/'train/checkpoint-750/adapter_model.safetensors')['sha256'] == '507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'
    evals = {}
    eval_config = read(R/'configs/stage_i_d053_evaluation_v1.json')
    expected_ids = set(eval_config['random_business_ids'] + eval_config['targeted_business_ids'])
    for epoch in (3,5):
        p = L/f'evaluate/epoch{epoch}/predictions.json'
        rows = read(p)
        assert len(rows) == 24 and {r['business_id'] for r in rows} == expected_ids
        evals[str(epoch)] = dict(outputs=len(rows), path=rel(p), sha256=record(p)['sha256'])
    views = read(R/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json')
    label_audit = read(R/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_AUDIT.json')
    for v in views['views'].values():
        assert record(R/v['path'])['sha256'] == v['sha256']
    large = dict(train_epochs=done['epochs'], updates=done['steps'], checkpoints=checkpoints, validation=evals,
        checkpoint_choice='Original CE epoch3 provisional Filtered; not a complete matched family',
        RO_training='NOT_FOUND_IN_CANONICAL_RUN_INVENTORY', U_training='NOT_FOUND_IN_CANONICAL_RUN_INVENTORY',
        Shuffled_training='NOT_REQUIRED; must reuse selected F adapter',
        corrected_label_policy=views['policy'], materialized_views=list(views['views']),
        all_condition_label_audit_rows=label_audit['rows_total'],
        RO_U_condition_specific_runner_and_bindings='NOT_PREPARED; current V2 execution path is Filtered-specific',
        no_promotion_without_separate_family_authorization=True)
    write('LARGE_FILTERED_ONLY_AUDIT.json', large)

    rf = read(REF/'REFERENCE_FREEZE_MANIFEST.json')
    ref_check = manifest_check(REF, rf['files'])
    readiness = read(REF/'D053_FINAL_REFERENCE_READINESS.json')
    blank_path = R/'outputs/stage_i/led_large_v2_blinded_quality_review_v1/BLINDED_REVIEW/BLANK_RESEARCHER_ANNOTATIONS.json'
    blanks = read(blank_path)['ratings']
    labels = sum(x.get('researcher_confirmed') is True for x in blanks)
    refs = dict(integrity=ref_check, finalized_validation_businesses=readiness['businesses'],
        core=readiness['core'], supporting=readiness['supporting'], excluded=readiness['excluded'],
        unresolved_source_reference_decisions=readiness['unresolved_researcher_decisions'],
        scope='Fixed24 Filtered development cases; repeated exposure; not test ground truth',
        existing_large_blinded_ratings=len(blanks), researcher_confirmed_in_saved_rating_file=labels,
        distinction='Approved source references do not imply completed student-output annotation',
        RO_U_visual_reference_coverage='Do not transfer Filtered visual denominators blindly; verify evidence domain per condition',
        test_human_references='NOT_FOUND in audited canonical reference/evaluation namespaces',
        formal_30_business_study='No finalized sample/rating/results artifact found in audited canonical namespaces',
        teacher_test_targets='NOT_EXPECTED_AND_MUST_NOT_BE_CREATED')
    write('REFERENCES_AND_EVALUATION_AUDIT.json', refs)

    # Names and existence only. Do not open any held-out source/caption/photo payload.
    pointer = read(R/'outputs/stage_f/production_manifest_pointer.json')
    test_metadata = [dict(path=v, exists=(R/v).is_file(), content_opened=False)
                     for v in pointer['split_manifests']['test'].values()]
    top_names = sorted(p.name for p in (R/'outputs/stage_i').iterdir() if p.is_dir())
    filenames = []
    scan_roots = [R/'configs', R/'scripts', R/'specs', R/'thesis_sdd_docs']
    for p in scan_roots:
        filenames.extend(rel(x) for x in p.iterdir() if x.is_file() and any(s in x.name.lower() for s in ('shuff','human','evaluat','test')))
    namespace_hits = [n for n in top_names if any(s in n.lower() for s in ('shuff','human','heldout','held_out','final_eval','large_ro','large_u','large_reviews_only','large_unfiltered'))]
    write('SEARCH_SCOPE_AND_MISSING_ARTIFACTS.json', dict(audited_utc=utc,
        stage_i_top_level_names=top_names, possible_final_evaluation_namespaces=namespace_hits,
        tool_or_protocol_names=filenames, test_source_metadata_only=test_metadata,
        Stage_H_test_packs_exist={c:(R/f'outputs/stage_h/led_dataset_v1/test_{c}.npz').exists() for c in base},
        actual_shuffled_assignment='NOT_FOUND; StageF canonical pointer still explicitly pending',
        actual_shuffled_predictions='NOT_FOUND_IN_AUDITED_NAMESPACES',
        test_predictions_and_results='NOT_FOUND_IN_AUDITED_NAMESPACES',
        missing_is_not_proof_of_absence_in_browser_external_storage_or_unimported_downloads=True,
        remote_host_not_accessed=True, held_out_file_contents_accessed=False))
    write('AUDIT_SCOPE_AND_SAFETY.json', dict(created_utc=utc, GPU_operations=0, model_loads=0,
        network_calls=0, optimizer_steps=0, generations=0, Teacher_calls=0,
        held_out_payload_reads=0, frozen_artifact_writes=0, lifecycle_actions=0,
        historical_byte_manifests_verified=True, hash_checks_are_not_runtime_restore_tests=True,
        current_authorization='Offline readiness audit only; no execution approval inferred'))
    write('INPUT_SHA256_MANIFEST.json', dict(files=observed))
    print(json.dumps(dict(status='OFFLINE_AUDIT_COMPLETE', base_conditions=len(base), base_predictions=600,
        large_F_predictions=48, reference_core=refs['core'], reference_supporting=refs['supporting'],
        source_files_hashed=len(observed), output=rel(O)), indent=2))


if __name__ == '__main__':
    main()
