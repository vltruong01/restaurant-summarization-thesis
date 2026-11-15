"""Offline delivery builder. No test payload, CUDA, model or inference calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
import led_large_rou_gate_only_v2 as g

R, P = g.ROOT, g.ROOT/g.PREP
OLD = R/'outputs/stage_i/led_large_matched_family_v1'
CODE = ['scripts/led_large_rou_gate_only_v2.py', 'scripts/run_led_large_rou_gate_only_v2.py',
        'scripts/led_large_shuffled_production_v2.py', 'scripts/led_large_final_analysis_v2.py',
        'scripts/prepare_led_large_matched_completion_v2.py', 'tests/test_led_large_matched_completion_v2.py']


def draft(path, data):
    g.require(not (P/'PORTABLE_MANIFEST.json').exists(), 'Sealed V2 package cannot be overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prepare():
    P.mkdir(parents=True, exist_ok=True)
    g.require(not (P/'ACCEPTED_V1_CHAIN.json').exists(), 'Preparation already materialized')
    old_manifest = g.read(OLD/'PORTABLE_MANIFEST.json')
    for rel, digest in old_manifest['files'].items():
        g.require(g.sha(R/rel) == digest, 'Accepted V1 changed: '+rel)
    archive = R/'outputs/cloud_handoff/LED_LARGE_MATCHED_FAMILY_V1_OFFLINE_REVIEW.zip'
    g.require(g.sha(archive) == '65e050e7a67acec304d55be01e9bb852ceeb6a808efdc0ba3f3c041ecaf8611a', 'Wrong accepted archive')
    old_bindings = g.read(OLD/'EXECUTION_BINDINGS.json')
    draft(P/'ACCEPTED_V1_CHAIN.json', dict(archive=archive.relative_to(R).as_posix(), archive_sha256=g.sha(archive),
          portable_manifest_sha256=g.sha(OLD/'PORTABLE_MANIFEST.json'), portable_files_verified=len(old_manifest['files']),
          preflight_manifest_sha256=g.sha(OLD/'PREFLIGHT_MANIFEST.json'), preflight_files=49,
          accepted_condition_bindings=old_bindings, original_files_overwritten=False))
    # Review recorded now; no researcher execution timestamp is invented.
    draft(P/'FAMILY_LOCK_RECEIPT.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
          authority='Latest explicit researcher acceptance of matched-family preparation', researcher_decision_timestamp=None,
          primary_family='LED-large RO/U/F/Shuffled', selected_epoch=3, selected_step=750,
          F_development_informed=True, RO_U_opportunistic_selection_forbidden=True,
          initial_RO_U='Independent original pretrained backbone + fresh Expanded-LoRA; never F adapter initialization',
          Shuffled='Exact selected F adapter, no fourth training', native_CE_and_BOS_free_policy_unchanged=True,
          GPU_feasibility_approved=False, training_authorized=False, held_out_access=False,
          baseline_F_adapter_sha256='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'))
    base = R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train'
    selected, all_proofs = {}, []
    for step in (250,500,750,1000,1250):
        folder = base/f'checkpoint-{step}'
        proof = g.read(folder/'checkpoint_integrity.json')
        for name, digest in proof['files'].items():
            g.require(Path(name).name == name and g.sha(folder/name) == digest, 'F checkpoint modified')
            if step == 750: selected[(folder/name).relative_to(R).as_posix()] = digest
        if step == 750:
            selected[(folder/'checkpoint_integrity.json').relative_to(R).as_posix()] = g.sha(folder/'checkpoint_integrity.json')
        all_proofs.append(dict(step=step, components=len(proof['files']), status='HASH_PASS',
                               receipt_sha256=g.sha(folder/'checkpoint_integrity.json')))
    draft(P/'HISTORICAL_CHECKPOINT_FILES.json', dict(files=selected,
          scope='Selected immutable F750 full state plus integrity receipt; hashes only, no state loaded'))
    draft(P/'LOCAL_PRESERVATION.json', dict(F_checkpoints=all_proofs, verified_components=50,
          accepted_V1_files_verified=82, CAD='CLOSED', Boundary='PAUSED', Contrastive='CLOSED',
          matched_LED_base='Preserved; see accepted V1 checkpoint inventory', model_weights_changed=False))
    # One fixed source-metadata policy, awaiting explicit scientific adjudication.
    allowlist = ['American (Traditional)','American (New)','Italian','Mexican','Tex-Mex','New Mexican Cuisine',
                 'Japanese','Sushi Bars','Ramen','Chinese','Szechuan','Dim Sum','Taiwanese','Thai','Vietnamese',
                 'Korean','Asian Fusion','Indian','Pakistani','Malaysian','Filipino','Mediterranean','Greek',
                 'Middle Eastern','Lebanese','Turkish','French','Spanish','Basque','German','British','Irish',
                 'Modern European','Caribbean','Cuban','Latin American','Brazilian','Hawaiian','Canadian (New)',
                 'Cajun/Creole','Southern','Soul Food','Barbeque','Seafood','Steakhouses','Pizza','Burgers',
                 'Sandwiches','Delis','Diners','Breakfast & Brunch','Chicken Wings','Chicken Shop','Tacos','Poke',
                 'Cheesesteaks','Hot Dogs','Fish & Chips','Tapas/Small Plates','Tapas Bars','Noodles','Soup',
                 'Vegetarian','Vegan','Bagels','Bakeries','Creperies','Waffles','Hot Pot','Kebab','Falafel']
    policy = dict(version='SHUFFLED_PRODUCTION_POLICY_V2_PROPOSED', researcher_approved=False,
          protocol_source='CODEX_THESIS_IMPLEMENTATION_PLAN.md: Shuffled-caption rules', seed=42,
          cuisine_subcategory_allowlist=sorted(allowlist), category_normalization='NFKC + strip + casefold; exact tag intersection',
          category_source='Frozen model-visible categories= metadata; no review/caption/outcome analysis',
          similar_cuisine_inference='NONE; exact allowlisted cuisine/subcategory overlap only',
          donor_reuse='ALLOWED', caption_count='EXACT_RECIPIENT_F_COUNT', same_split=True, no_self_match=True,
          first_tier='Any budget-feasible candidate sharing at least one allowlisted category',
          fallback='ANY_BUDGET_FEASIBLE_SAME_SPLIT_SAME_COUNT',
          deterministic_rank='minimum SHA256(canonical JSON [42,policy_version,split,recipient_id,donor_id]); tie donor_id',
          no_eligible_donor='BLOCK_ENTIRE_MAPPING', zero_captions='donor=None; exact original empty visual context',
          budgets=dict(review_block=4000, full_source=5000, caption_increment=750, maximum_captions=5),
          caption_increment='len(full Shuffled source tokens)-len(original recipient RO source tokens), both with BOS/EOS',
          donor_caption_editing=False, policy_changes_after_test_access=False, mapping_uses_targets_or_predictions=False)
    draft(P/'SHUFFLED_POLICY_PROPOSED.json', policy)
    # Existing, immutable, source-only TRAIN/validation fixture, copied byte for byte.
    source = OLD/'SHUFFLED_SOURCE_FIXTURES.json'
    target = P/'SHUFFLED_SOURCE_FIXTURES.json'
    target.write_bytes(source.read_bytes())
    from tokenizers import Tokenizer
    from led_large_shuffled_production_v2 import build_mapping
    tokenizer = Tokenizer.from_file(str(R/'assets/tokenizers/led-base-16384/tokenizer.json'))
    result = build_mapping(g.read(target)['rows'], tokenizer, policy)
    draft(P/'SHUFFLED_FIXTURE_RESULTS.json', result)
    # Pin the agreement implementation without installing into the training env.
    url = 'https://pypi.org/pypi/krippendorff/0.8.1/json'
    with urllib.request.urlopen(url, timeout=30) as response: metadata = json.load(response)
    wheel = next(x for x in metadata['urls'] if x['filename'].endswith('.whl'))
    with urllib.request.urlopen(wheel['url'], timeout=30) as response: data = response.read()
    g.require(hashlib.sha256(data).hexdigest() == wheel['digests']['sha256'], 'Agreement wheel hash mismatch')
    vendor = P/'analysis_source_pin'; vendor.mkdir()
    (vendor/wheel['filename']).write_bytes(data)
    agreement = dict(distribution='krippendorff', version='0.8.1', filename=wheel['filename'],
          sha256=wheel['digests']['sha256'], url=wheel['url'], source=url,
          API="alpha(reliability_data=three_raters_by_units, value_domain=np.arange(1,6), level_of_measurement='ordinal')")
    draft(P/'ANALYSIS_LOCK.json', dict(version='FINAL_ANALYSIS_LOCK_V2_PROPOSED_FOR_RESEARCHER_REVIEW',
          review_metrics_primary=True, visual_metrics_supplementary=True, RO_visual='N/A',
          metric_lock_path=(OLD/'METRIC_LOCK.json').relative_to(R).as_posix(), metric_lock_sha256=g.sha(OLD/'METRIC_LOCK.json'),
          reference_binding=None, references_status='UNVERIFIED_RESEARCHER_UNCERTAIN_NO_APPROVED_100_BINDING',
          numpy='1.26.4', resampling='NumPy Generator PCG64 seed42,5000 paired business bootstrap samples',
          percentile_method='np.quantile(method=linear), .025/.975', primary_contrast='F-U',
          secondary_contrasts=['F-Shuffled','U-RO','F-RO'], p_values=False,
          multiplicity='Pointwise95% CIs only; no family-wise significance claim',
          main_businesses=100, secondary_expected_businesses=83, secondary_policy='D-022 unchanged; membership unopened',
          human_businesses=30, annotators=3, output_rating_sets=360, original_rubric_dimensions=6,
          human_resampling='Mean3raters per business/condition, then paired business bootstrap; preserve rater cluster',
          visual_text_pairing='Use pre-scoring source/reference-applicable intersection; retain100-frame N/A reasons and denominators',
          visual_human='All30 U/F/Shuffled, including zero-caption abstention assessment; RO structurally N/A',
          agreement=agreement, N_A='Structural N/A never zero; no incomplete ratings or silent pairwise dropping',
          constant_agreement='N/A/UNDEFINED_NO_VARIATION, not perfect alpha',
          actual_metrics_computed=False, GPU_operations=0))
    from led_large_final_analysis_v2 import separate_annotation_packets
    ids = g.read(OLD/'SAMPLING_FIXTURE.json')['ids']
    private, packets = separate_annotation_packets(ids, b'SYNTHETIC_FIXTURE_KEY_NOT_FOR_REAL_ANNOTATION'*2)
    draft(P/'PRIVATE_ADMIN_FIXTURE'/'CONDITION_KEY.json', private)
    for who, packet in packets.items(): draft(P/'ANNOTATOR_VISIBLE_FIXTURES'/(who+'.json'), packet)
    draft(P/'HUMAN_WORKFLOW_COUNTS.json', dict(businesses=30, conditions=4, annotators=3,
          rating_sets=360, per_annotator=120, scales=6, verified_annotator_enrollments=0,
          actual_enrollment_status='NOT_ESTABLISHED', actual_ratings_in_this_package=0, fixture_only=True, private_keys_separate=True,
          source_and_summary_payloads_populated=False, researcher_reference_availability_reply='Uncertain; needs checking'))
    draft(P/'TEST_REFERENCE_INDEX_PENDING.json', dict(schema='METADATA_ONLY_INDEPENDENT_TEST_REFERENCE_INDEX_V2',
          expected_businesses=100, verified_covered_businesses=None, verified_approved_businesses=None,
          unresolved=None, reference_payload_sha256=None, source_manifest_sha256=None,
          independent_approval_receipt_sha256=None, approved_by_researcher=False,
          prepared_independently_of_model_predictions=None, payload_opened=False,
          state='PENDING_NOT_EVIDENCE_OF_EXISTENCE_OR_APPROVAL'))
    for condition, short in [('reviews_only','RO'),('unfiltered','U')]:
        draft(P/(condition+'_GATE_CONFIG.json'), dict(experiment='LED_LARGE_'+short+'_FEASIBILITY_ONLY_V2',
              condition=condition, accepted_V1_binding=old_bindings[condition],
              output_namespace='runs/'+short+'_large_gate_only_v2',
              selected_checkpoint_step=750, training_updates_authorized=0,
              locked_gate=dict(native_CE_updates=3, nominal_microbatches=24, maximum_microbatches=88,
                  maximum_overflow_skips=8, minimum_AMP_scale=256, TRAIN_generation_calls=2,
                  minimum_headroom=.15, full_state_restore=True, fresh_initialization=True)))
    print('V2 offline proposal materialized; no GPU or test payload access')


def bindings():
    files = CODE + [x.relative_to(R).as_posix() for x in P.rglob('*') if x.is_file() and x.suffix in ('.md','.whl')]
    names = ['ACCEPTED_V1_CHAIN.json','FAMILY_LOCK_RECEIPT.json','HISTORICAL_CHECKPOINT_FILES.json',
             'SHUFFLED_POLICY_PROPOSED.json','SHUFFLED_SOURCE_FIXTURES.json','ANALYSIS_LOCK.json',
             'reviews_only_GATE_CONFIG.json','unfiltered_GATE_CONFIG.json']
    files += [(P/n).relative_to(R).as_posix() for n in names]
    files += ['outputs/stage_i/led_large_matched_family_v1/PREFLIGHT_MANIFEST.json',
              'outputs/stage_i/led_large_matched_family_v1/METRIC_LOCK.json']
    draft(P/'PREFLIGHT_MANIFEST.json', dict(files={rel:g.sha(R/rel) for rel in sorted(set(files))}))
    for condition, short in [('reviews_only','RO'),('unfiltered','U')]:
        template = dict(experiment=g.read(g.config(condition))['experiment'], condition=condition,
              binding=g.binding(condition), scopes=['feasibility'], approval_source='PENDING_RESEARCHER',
              researcher_explicit_authorization=None, GPU_feasibility_approved=False, training_authorized=False,
              authorized_at_utc=None, deadline_utc=None, researcher_absolute_stop_utc=None,
              defense_reserve_at_least_14_days_confirmed=False, held_out_access=False, Teacher_calls=False,
              automatic_train=False, automatic_retry=False, fresh_only=True,
              max_successful_gate_updates=3, max_generation_calls=2, max_actual_microbatches=88,
              max_training_updates=0, evaluation_outputs_max=0, early_sanity_outputs_max=0)
        draft(P/'PENDING_APPROVALS'/(short+'_GATE_ONLY_PENDING.json'), template)
        draft(P/(short+'_GPU_AUTHORIZATION_READINESS.json'), dict(status='OFFLINE_ACCEPTED_FUTURE_HOST_AND_GPU_UNVERIFIED',
              condition=condition, binding=g.binding(condition), accepted_training_binding=g.binding(condition)['accepted_V1_binding'],
              output_namespace=g.read(g.config(condition))['output_namespace'],
              namespace_exists_at_preparation=(R/g.read(g.config(condition))['output_namespace']).exists(),
              selected_epoch=3, selected_step=750, preparation_accepted_by_researcher=True,
              GPU_execution_authorized=False, training_authorized=False, actual_authorization_UTC=None,
              host_CPU_tests='MUST_RERUN_26_ACCEPTED_PLUS35_INHERITED_PLUS_NEW_V2',
              host_full_parent_model_tokenizer_selected_F_checkpoint='MUST_REVERIFY',
              actual_GPU_feasibility='NOT_RUN', measured_RO_U_headroom=None,
              future_scope='Gate-only3discardedupdates/2TRAINtraces; STOP, no auto training',
              training_handoff='Separate approval and explicit accepted-gate bridge required; V1 train CLI not auto-compatible'))
    print('Gate-only bindings prepared')


def metadata_audit():
    """Filename/stat inventory only. Never opens candidate payloads or archives."""
    import re
    roots=[R/'outputs'/n for n in ('stage_m','stage_n','stage_o','test_references','human_evaluation','evaluation')]
    roots += [R/'outputs',Path.home()/'Downloads',R/'thesis_sdd_docs']
    candidates=[];counts=[];seen=set()
    query=re.compile(r'(test|held.?out).*(reference|annotation|rating)|(reference|annotation|rating).*(test|held.?out)',re.I)
    for root in roots:
        visited=0;truncated=False
        if not root.is_dir():
            counts.append(dict(root=str(root),exists=False));continue
        for directory,folders,files in os.walk(root,followlinks=False):
            folders[:]=[n for n in folders if n not in ('.git','node_modules','__pycache__','assets')
                        and not n.startswith('.venv') and not Path(directory,n).is_symlink()
                        and Path(directory,n).resolve()!=P.resolve()]
            for name in files:
                visited+=1
                if visited>200000:truncated=True;break
                path=Path(directory)/name
                collective=('reference' in name.lower() and any(x in name.lower() for x in ('manifest','approved','final','human')))
                if path.is_symlink() or not (query.search(name) or collective) or str(path) in seen:continue
                seen.add(str(path))
                stat=path.stat();lower=str(path).lower()
                classification='POSSIBLE_ARTIFACT_UNOPENED'
                if 'd053' in lower or 'validation' in lower:classification='KNOWN_VALIDATION_CONTEXT_NOT_TEST_REFERENCE'
                elif path.suffix.lower() in ('.py','.pyc','.log','.ipynb') or 'tests' in lower:
                    classification='CODE_OR_UNIT_TEST_EVIDENCE_NOT_REFERENCE_APPROVAL'
                candidates.append(dict(path=str(path),bytes=stat.st_size,
                    modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
                    classification=classification,contents_opened=False,archive_members_opened=False))
            if truncated:break
        counts.append(dict(root=str(root),exists=True,filenames_examined=visited,truncated=truncated))
    draft(P/'REFERENCE_METADATA_AUDIT.json',dict(status='APPROVED_100_TEST_REFERENCE_BINDING_NOT_ESTABLISHED',
        researcher_reply='Uncertain; needs checking',scope=counts,candidates=candidates,
        verified_reference_businesses=None,payloads_opened=0,archive_payloads_opened=0,
        limitation='Filename/stat evidence only; no content/approval verification, no external/cloud search; not a claim of nonexistence'))
    print('Reference metadata only:',sum(x.get('filenames_examined',0) for x in counts),'filenames;',len(candidates),'candidates; payloads opened0')


def seal():
    g.verify_files()
    tests = g.read(P/'CPU_TEST_RESULTS.json')
    g.require(tests['status']=='PASS', 'Current CPU evidence required')
    for condition in g.CONDITIONS:
        report=g.read(P/(condition+'_CPU_PREFLIGHT.json'))
        g.require(report['binding']==g.binding(condition) and report['status']=='CPU_PREFLIGHT_PASS_NOT_GPU_PASS', 'Stale CPU proof')
    files = sorted(set(CODE+[p.relative_to(R).as_posix() for p in P.rglob('*') if p.is_file()]))
    # Carry tiny accepted locks, not another copy of the73MB accepted archive.
    files += ['outputs/stage_i/led_large_matched_family_v1/EXECUTION_BINDINGS.json',
              'outputs/stage_i/led_large_matched_family_v1/METRIC_LOCK.json',
              'outputs/stage_i/led_large_matched_family_v1/PREFLIGHT_MANIFEST.json']
    files=sorted(set(files));manifest=dict(files={rel:g.sha(R/rel) for rel in files},
          dependency='Accepted V1 overlay/full parent/selected F750 checkpoint remain separate and immutable')
    g.write(P/'PORTABLE_MANIFEST.json', manifest)
    path=R/'outputs/cloud_handoff/LED_LARGE_MATCHED_COMPLETION_V2_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in files+[(P/'PORTABLE_MANIFEST.json').relative_to(R).as_posix()]:z.write(R/rel,rel)
    with zipfile.ZipFile(path) as z:
        g.require(z.testzip() is None,'ZIP CRC failure')
        for rel,digest in manifest['files'].items():g.require(hashlib.sha256(z.read(rel)).hexdigest()==digest,'Portable hash mismatch')
    Path(str(path)+'.sha256').write_text(g.sha(path)+'  '+path.name+'\n')
    print(json.dumps(dict(zip=str(path),bytes=path.stat().st_size,manifest_entries=len(files),sha256=g.sha(path))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['prepare','bindings','metadata','seal'])
    action=parser.parse_args().mode
    {'prepare':prepare,'bindings':bindings,'metadata':metadata_audit,'seal':seal}[action]()
