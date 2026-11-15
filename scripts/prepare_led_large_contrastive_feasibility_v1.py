"""Package an offline methodology proposal; no model loading or execution path."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_i/led_large_contrastive_feasibility_v1'
PORTABLE = OUT/'portable_review'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(name, obj):
    p = PORTABLE/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def copy(rel, name):
    p = PORTABLE/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT/rel).read_bytes())


def main():
    assert not OUT.exists(), 'New versioned proposal only; do not overwrite a previous package'
    now = datetime.now(timezone.utc).isoformat()
    PORTABLE.mkdir(parents=True)
    nbest_zip = ROOT/'outputs/cloud_handoff/LED_LARGE_NBEST_V2_PHASE_A_CHATGPT_REVIEW.zip'
    with zipfile.ZipFile(nbest_zip) as z:
        manifest = json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(manifest) == 171
        for rel, expected in manifest.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == expected, rel

    cp = ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
    proof = read(cp/'checkpoint_integrity.json')
    for name, expected in proof['files'].items():
        assert Path(name).name == name and sha(cp/name) == expected, name
    assert proof['global_step'] == 750
    assert proof['files']['adapter_model.safetensors'] == '507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'

    parent_config = read(ROOT/'configs/stage_i_led_large_nbest_v2.json')
    protected = {ROOT/rel: expected for rel, expected in parent_config['parent_bindings'].items()}
    stage_h = read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    for key in ('train_reviews_only', 'train_unfiltered', 'train_filtered'):
        meta = stage_h['datasets'][key]
        protected[ROOT/meta['path']] = meta['sha256']
        protected[ROOT/meta['stage_g_jsonl']] = meta['stage_g_jsonl_sha256']
    label_views = read(ROOT/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json')
    corrected = label_views['views']['train_filtered']
    protected[ROOT/corrected['path']] = corrected['sha256']
    for p, expected in protected.items():
        assert sha(p) == expected, str(p)

    cases = read(ROOT/'outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json')
    original_sources = ROOT/'outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json'
    fit, probe = [], []
    for zero, take in ((True, 2), (False, 6)):
        population = sorted([x for x in cases if (not x['captions']) == zero],
            key=lambda x: hashlib.sha256(('LED_CONTRASTIVE_FEASIBILITY_V1:'+x['business_id']).encode()).hexdigest())
        fit += [x['business_id'] for x in population[:take]]
        probe += [x['business_id'] for x in population[take:]]
    assert len(fit) == 8 and len(probe) == 4 and not set(fit)&set(probe)
    source_copy = []
    index = []
    for case in cases:
        assert case['split'] == 'train' and case['condition'] == 'filtered'
        assert hashlib.sha256(case['source'].encode('utf-8')).hexdigest() == case['source_sha256']
        source_copy.append({k: case[k] for k in ('business_id','split','condition','source','source_sha256','captions')})
        index.append(dict(business_id=case['business_id'], role='fit' if case['business_id'] in fit else 'probe',
            caption_count=len(case['captions']), source_sha256=case['source_sha256'],
            original_packet=str(original_sources.relative_to(ROOT)).replace('\\','/'),
            target_admissibility='NOT_AUDITED_NO_POSITIVE_OR_NEGATIVE_CREATED',
            pair_ids=[], human_adjudication=None))
    save('SOURCE_ONLY_TRAIN12.json', source_copy)
    save('TRAIN_ONLY_PILOT_MANIFEST.json', dict(selection='SHA256(LED_CONTRASTIVE_FEASIBILITY_V1:+business_id), stratified first2 zero + first6 captioned fit',
        fit=fit,probe=probe,cases=index,probe_is_unseen=False,validation_businesses_used=0,
        new_negative_examples=0, scientific_quality_GO=False))

    execution = ROOT/'outputs/stage_i/led_large_nbest_v2_execution_v1/portable_review/execution/CANDIDATES.json'
    outputs = read(execution)
    assert len(outputs) == 12 and {x['business_id'] for x in outputs} == {x['business_id'] for x in cases}
    # Preserve all 60 existing sequences verbatim, not just the preferred beams.
    assert all(len(x['candidates']) == 4 for x in outputs)
    copy(str(execution.relative_to(ROOT)), 'evidence/EXISTING_NBEST_60_SEQUENCES.json')
    copy(str((cp/'checkpoint_integrity.json').relative_to(ROOT)), 'evidence/CE3_CHECKPOINT_RECEIPT.json')
    copy('outputs/stage_i/led_large_nbest_v2_execution_v1/TECHNICAL_AGGREGATES.json', 'evidence/NBEST_TECHNICAL_AGGREGATES.json')
    copy('outputs/stage_i/led_large_nbest_v2_execution_v1/TRAIN_SOURCE_GROUNDED_PROVISIONAL_REVIEW.json', 'evidence/ORIGINAL_MODEL_ASSISTED_REVIEW_UNCHANGED.json')
    copy('thesis_sdd_docs/LED_LARGE_CONTRASTIVE_FEASIBILITY_V1.md','METHODOLOGY_AND_NUMERICAL_FEASIBILITY.md')
    copy('scripts/led_large_contrastive_objective_spec_v1.py','code/led_large_contrastive_objective_spec_v1.py')
    copy('tests/test_led_large_contrastive_objective_spec_v1.py','code/test_led_large_contrastive_objective_spec_v1.py')
    copy('scripts/prepare_led_large_contrastive_feasibility_v1.py','code/prepare_led_large_contrastive_feasibility_v1.py')

    contract = dict(experiment='LED_LARGE_CONTRASTIVE_FEASIBILITY_V1', status='OFFLINE_PROPOSAL_NOT_EXECUTION_READY',
        GPU_execution_approved=False,training_authorized=False,inference_authorized=False,
        negative_corpus_creation_authorized=False,Teacher_calls=False,held_out_access=False,
        objective=dict(name='CE_PLUS_SOURCE_VERIFIED_PAIRWISE_SOFTPLUS',pair_weight=.1,margin=.1,temperature=1.,
            reduction='mean valid tokens per sequence, then mean of8 examples',include_EOS=True,include_headers=True,ignore=-100,
            CPU_reference_only=True, production_LED_runner_exists=False),
        initialization=dict(backbone=parent_config['backbone_id'],revision=parent_config['backbone_revision'],
            backbone_sha256=parent_config['backbone_sha256'],adapter_sha256=parent_config['adapter_sha256'],
            adapter_checkpoint=750,new_branches_from_same_CE3=True,fresh_optimizer_scheduler_scaler=True,original_CE3_immutable=True),
        pair_policy=dict(max_proposed_edits=24,accepted_pairs_required=16,fit_businesses=8,probe_businesses=4,
            min_pairs_per_fit_business=1,max_pairs_per_fit_business=3,positive='unchanged original Teacher F target only if source-verified',
            original_targets_overwritten=False,negative_examples_created=0,independent_adjudication_required=True,
            minimum_categories=dict(core_omission=4,same_aspect_disagreement_loss=4,grounding_error=4),
            larger_negative_corpus=False,automatic_beam_rejection=False),
        proposed_continuation=dict(arms=['CE_control','CE_plus_pair'],LR=2e-5,AdamW_betas=[.9,.999],epsilon=1e-8,
            weight_decay=.01,max_grad_norm=1.,physical_batch=1,accumulation=8,warmup_updates=1,scheduler='linear',
            seed=42,pilot_updates_per_arm=16,gate_updates_per_arm=3,gate_max_microbatches_per_arm=88,
            pilot_max_microbatches_per_arm=192,total_successful_update_cap=38,save_pilot_steps=[8,16],
            generation_calls_cap=16,returned_sequences_cap=16,generation=read(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json')['generation'],
            no_validation_inference=True,no_automatic_follow_on=True),
        numerical=dict(strategy='RNG-replayed two scouts then sequential positive/negative gradient contributions',
            logits_reduction='FP32 chosen logit minus logsumexp',scout_replay_atol=1e-6,scout_replay_rtol=1e-6,
            initial_scale=65536,growth_factor=2.,backoff_factor=.5,growth_interval=2000,max_backoffs_per_job=8,
            min_scale=256,headroom_required=.15,all_trainable_gradient_tensors=456,
            frozen_backbone=True,LED_CUDA_replay_proven=False,exact_CUDA_replay_claim=False),
        authorization=dict(current_scope='methodology and offline mathematical reference only',
            next_step='researcher review of methodology, then source-pair qualification and implementation review',
            researcher_GPU_approval_timestamp=None,execution_deadline_utc=None,
            future_max_window_hours=48,future_GPU_active_time_cap_hours=1,protected_thesis_reserve_needs_fresh_confirmation=True),
        blocked_until=['16 independently accepted pairs with8 admissible immutable positives',
            'implemented and reviewed pinned LED sequential-gradient/RNG/AMP accumulation/recovery tests',
            'new exact code/config/preflight authorization hashes and explicit GPU approval'],
        matched_family=dict(primary_filtered='CE DecoderPolicyV2 epoch3 provisional',
            main_thesis_primary='historical matched LED-base RO/U/F',RO_U_training_authorized=False,
            shuffled_reuses_exact_matched_F=True,no_F_only_substitution=True))
    save('PROPOSED_CONTRACT_NOT_GPU_APPROVAL.json',contract)
    save('PAIR_ADJUDICATION_TEMPLATE.json',dict(status='EMPTY_TEMPLATE_NOT_GENERATED_TARGETS',
        business_id=None,original_target_sha256=None,corrected_label_ids_sha256=None,positive_text=None,negative_text=None,
        minimal_diff=None,defect_category=None,negative_is_incomplete_vs_false=None,evidence_spans=[],
        all_positive_claims_supported=None,opposing_same_aspect_evidence=[],recurrence_evidence=[],
        visual_caption_relationship_check=None,zero_caption_check=None,edit_token_count=None,
        length_style_shortcut_review=None,author_kind=None,independent_researcher_acceptance=None,
        fit_only=True,researcher_confirmed=False))
    save('CPU_MATH_TEST_EVIDENCE.json',dict(command='.venv-caption-pilot/Scripts/python.exe -m unittest discover -s tests -p test_led_large_contrastive_objective_spec_v1.py -v',
        observed_exit_code=0,tests_passed=11,tests_failed=0,reported_test_seconds=2.131,
        evidence_source='Actual local tool execution in this preparation turn; summary of completed test run',
        environment=dict(python='3.11.7',torch='2.1.0+cu118',numpy='1.26.4',transformers='4.51.3',peft='0.15.2',accelerate='1.6.0',tokenizers='0.21.1'),
        CUDA_VISIBLE_DEVICES='',model_loads=0,GPU_operations=0,optimizer_updates=0,
        production_LED_RNG_replay_tested=False,GPU_AMP_or_memory_tested=False,
        bindings={n:sha(PORTABLE/n) for n in ('code/led_large_contrastive_objective_spec_v1.py','code/test_led_large_contrastive_objective_spec_v1.py')}))

    save('NBEST_PHASE_A_REVIEW_DECISION.json',dict(recorded_utc=now,
        provenance='Independent source-grounded review supplied by user in current instruction',
        technical='SUCCESS',semantic_diversity_gate='NOT_ESTABLISHED',quality_headroom_gate='NOT_ESTABLISHED',
        phase_B='DO_NOT_PROCEED',phase_C='NOT_AUTHORIZED',reranker_validated=False,
        provisional_Filtered_checkpoint='Original CE DecoderPolicyV2 epoch3',
        existing_assistant_assessments_remain_provisional=True,per_case_human_labels_imported=False,
        independent_findings=dict(manifest_pass=171,identity=12,valid_N4=48,identical_review_businesses=9,
            clear_material_profile_businesses=5,borderline_businesses=2,
            clear_grounding_repair='Mesa c2',plausible_incomplete_repair='Habit c3',
            unsupported_zero_caption_candidates=11,total_zero_caption_candidates=12,
            required_four_quality_headroom_businesses_established=False),
        permitted_action='One concise offline source-verified contrastive feasibility proposal only'))
    ce_complete = ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/COMPLETION.json'
    save('PRESERVATION_AND_RESOURCE_EVIDENCE.json',dict(created_utc=now,
        Nbest_zip=dict(path=str(nbest_zip.relative_to(ROOT)),sha256=sha(nbest_zip),manifest_verified=171),
        checkpoint750=dict(path=str(cp.relative_to(ROOT)),receipt_sha256=sha(cp/'checkpoint_integrity.json'),full_state_files_verified=len(proof['files'])),
        TRAIN_frozen_files_verified={str(p.relative_to(ROOT)).replace('\\','/'):h for p,h in protected.items()},
        source_pack_sha256=sha(original_sources),all12_source_hashes_verified=True,
        CE_training_completion_path=str(ce_complete.relative_to(ROOT)),CE_training_completion_sha256=sha(ce_complete),
        CE_training_resource_record=read(ce_complete),
        historical_V1_evidence='outputs/stage_i/led_large_nbest_v1_execution_v1/backups/',
        old_files_overwritten=0,checkpoint_weights_modified=False,Teacher_targets_modified=False,
        full_parent_workspace_revalidated_now=False,full_parent_GPU_preflight_required_later=True,
        new_generation_calls=0,new_training_updates=0,new_negative_examples=0))
    # Exact quotes are found in existing source fields, never synthesized examples.
    by_id = {x['business_id']:x for x in cases}
    notes = []
    for bid, domain, marker, quote in [
        ('EzJI5x25U5GgbYBWhu5ZQQ','review','REV_2','The tacos were really good.'),
        ('EzJI5x25U5GgbYBWhu5ZQQ','review','REV_3',"It was very tasteless and salsa didn't help much."),
        ('5uhG34VRsD0A6lOY8F41Dg','caption','IMG_CAP_1','a tray of chicken tenders, fries, and a sandwich on a table.'),
        ('5uhG34VRsD0A6lOY8F41Dg','caption','IMG_CAP_2','a sign for a burger and sandwich restaurant in front of a building.'),
        ('wjZwXKDiFBQXlO8-TjQTXA','caption','IMG_CAP_3','a woman is making a drink at a bar.')]:
        case=by_id[bid]; text=case['reviews' if domain=='review' else 'captions'][marker]
        start=text.index(quote)
        notes.append(dict(business_id=bid,domain=domain,marker=marker,quote=quote,start_in_marker=start,
                          end_in_marker=start+len(quote),source_sha256=case['source_sha256']))
    save('TRAIN_EVIDENCE_ANCHORS.json',dict(status='EXISTING_SOURCE_QUOTES_ONLY_NOT_NEW_TARGETS_OR_ACCEPTED_PAIRS',anchors=notes,
        zero_caption_cases=[x['business_id'] for x in cases if not x['captions']],
        caveat='These anchors motivate categories. They do not establish admissibility of the unchanged full Teacher positive or researcher approval.'))
    (PORTABLE/'README_FIRST.md').write_text('''# Offline review only — contrastive feasibility V1

N-best V2 Phase A is technically successful, but its semantic and quality-headroom gates are not established. The current beam4 reranking branch stops. CE3 remains the provisional Filtered checkpoint.

Read METHODOLOGY_AND_NUMERICAL_FEASIBILITY.md for the single proposed objective, source-verified pair policy, matched continuation control, memory-safe recomputation design and bounded TRAIN-only pilot. PROPOSED_CONTRACT_NOT_GPU_APPROVAL.json locks proposed constants and budgets; it is not an authorization receipt.

There are no new positives, negatives, targets, model weights, production runner, GPU operations or optimizer updates in this package. Eleven synthetic CPU tests passed. Full LED/RNG/AMP replay and memory feasibility are unproven. The eight fit/four probe TRAIN split is deterministic; all cases were already in CE training and previously reviewed.

Approval requested now: methodology and bounded pair-qualification/implementation preparation. GPU execution requires separately reviewed pairs, production code, new execution hashes and fresh explicit authorization. No validation inference, reranker phase B/C, large corpus, Teacher changes, RO/U or held-out access. Existing raw outputs are preserved as evidence, not new human labels.
''',encoding='utf-8')
    # Bind the complete compact proposal, including the evidence copies.
    files={p.relative_to(PORTABLE).as_posix():sha(p) for p in sorted(PORTABLE.rglob('*')) if p.is_file()}
    save('SHA256_MANIFEST.json',dict(created_utc=now,files=files))
    for rel,digest in files.items():assert sha(PORTABLE/rel)==digest
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_FEASIBILITY_V1_OFFLINE_REVIEW.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(PORTABLE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PORTABLE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,digest in files.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    receipt=dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
        manifest_entries=len(files),CPU_math_tests=11,GPU_operations=0,training_updates=0,negative_examples_created=0,
        status='DELIVERED_OFFLINE_STOP_FOR_RESEARCHER_REVIEW')
    (OUT/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
