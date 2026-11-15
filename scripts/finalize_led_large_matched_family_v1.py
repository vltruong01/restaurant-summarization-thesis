"""Prepare bindings or seal the offline review ZIP; does not execute experiments."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import zipfile
import led_large_matched_family_v1 as m

R = m.ROOT
P = R / m.PREP
CODE = ['scripts/led_large_matched_family_v1.py', 'scripts/run_led_large_matched_family_v1.py',
        'scripts/led_large_shuffled_fixture_v1.py', 'scripts/led_large_final_evaluation_spec_v1.py',
        'scripts/prepare_led_large_matched_family_v1.py', 'scripts/finalize_led_large_matched_family_v1.py',
        'scripts/check_led_large_matched_routing_cpu_v1.py',
        'tests/test_led_large_matched_family_v1.py']


def save_draft(path, data):
    """Only new package outputs may change before the portable manifest is sealed."""
    path = Path(path)
    m.require(not (P / 'PORTABLE_MANIFEST.json').exists(), 'Sealed package is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def bindings():
    files = list(CODE)
    files += [m.paths(c).relative_to(R).as_posix() for c in m.CONDITIONS]
    files += list(m.read(P / 'ORIGINAL_SOURCE_FILES.json')['files'])
    files += [(P / n).relative_to(R).as_posix() for n in ('LABEL_VIEWS.json', 'PARENT_REQUIRED_FILES.json',
               'ORIGINAL_SOURCE_FILES.json', 'METRIC_LOCK.json', 'METRIC_UPSTREAM_METADATA.json',
               'SHUFFLED_SOURCE_FIXTURES.json', 'SHUFFLED_FIXTURE_RESULTS.json',
               'SAMPLING_FIXTURE.json', 'ANNOTATION_BLANK_FIXTURE.json',
               'README_VI.md', 'MATCHED_FAMILY_SPEC.md', 'EXECUTION_REVIEW.md',
               'SHUFFLED_PROTOCOL.md', 'FINAL_EVALUATION_READINESS.md', 'CRITICAL_PATH.md',
               'REQUEST_GPU_AUTHORIZATION.md')]
    files += [v['path'] for v in m.read(P / 'LABEL_VIEWS.json')['views'].values()]
    files += [p.relative_to(R).as_posix() for p in (P / 'metric_source_pins').iterdir() if p.is_file()]
    for c in m.CONDITIONS:
        files += [(P / (c + s)).relative_to(R).as_posix() for s in
                  ('_validation_cases.jsonl', '_FEASIBILITY_CASES.json')]
    save_draft(P / 'PREFLIGHT_MANIFEST.json', dict(version='MATCHED_FAMILY_V1', files={f: m.sha(R/f) for f in sorted(set(files))}))
    b = {}
    for c, short in [('reviews_only', 'RO'), ('unfiltered', 'U')]:
        b[c] = m.binding(c)
        approval = dict(experiment=m.read(m.paths(c))['experiment'], condition=c, binding=b[c],
                        GPU_execution_approved=False, training_authorized=False,
                        approval_source='PENDING_RESEARCHER', researcher_explicit_authorization=None,
                        execution_authorized_at_utc=None, execution_deadline_utc=None, researcher_absolute_stop_utc=None,
                        defense_reserve_at_least_14_days_confirmed=False, scopes=['feasibility', 'train'],
                        fresh_only=True, held_out_access=False, Teacher_calls=False, resume_authorized=False,
                        conditional_train_after_feasibility_pass=True, early_sanity_generation_approved=True,
                        max_training_updates=1250, feasibility_updates_max=3, feasibility_microbatches_max=88,
                        all_generation_calls_max=6, early_sanity_outputs_max=4, technical_generation_calls_max=2,
                        evaluation_outputs_max=0, selected_checkpoint_step=750,
                        note='UNSIGNED TEMPLATE. No authorization is inferred from this package.')
        save_draft(R / 'approvals' / f'{short}_PENDING.json', approval)
    save_draft(P / 'EXECUTION_BINDINGS.json', b)
    print('Draft bindings prepared:', len(files), 'manifest entries')


def seal():
    m.verify_overlay()
    report = m.read(P / 'CPU_TEST_RESULTS.json')
    m.require(report['status'] == 'PASS' and report['new_tests']['returncode'] == 0
              and report['inherited_tests']['returncode'] == 0, 'All offline tests required')
    for c in m.CONDITIONS:
        proof = m.read(P / f'{c}_CPU_PREFLIGHT.json')
        m.require(proof['status'] == 'CPU_PREFLIGHT_PASS' and proof['binding'] == m.binding(c), 'CPU preflight missing/stale')
    # Preserve old scope/status evidence by reference and reverify its supplied audit.
    prior = R / 'outputs/stage_i/main_rouf_shuffled_readiness_v2'
    preserved = {}
    for name in ('CAD_CLOSURE_RECEIPT.json', 'CHECKPOINT_INVENTORY.json', 'METRICS_AND_REFERENCES.json',
                 'CURRENT_WORKFLOW.json', 'INDEPENDENT_RESEARCHER_REVIEW.md'):
        src = prior / name
        dst = P / 'prior_readiness' / name
        dst.parent.mkdir(exist_ok=True)
        with dst.open('xb') as f: f.write(src.read_bytes())
        preserved[src.relative_to(R).as_posix()] = m.sha(src)
    # Metadata-only filename availability audit. No candidate contents are opened.
    roots = [R/'outputs/stage_i', R/'outputs/evaluation', R/'outputs/stage_o']
    candidates = []
    for directory in roots:
        if not directory.is_dir(): continue
        for f in directory.rglob('*'):
            if not f.is_file() or f.is_relative_to(P): continue
            name = f.name.lower()
            if ('test' in name or 'heldout' in name or 'held_out' in name) and ('reference' in name or 'annotation' in name or 'human' in name):
                stat = f.stat()
                candidates.append(dict(path=f.relative_to(R).as_posix(), bytes=stat.st_size, content_opened=False))
    save_draft(P / 'REFERENCE_METADATA_AUDIT.json', dict(status='NO_APPROVED_HELD_OUT_REFERENCE_BINDING_ESTABLISHED',
               searched_roots=[str(x.relative_to(R)) for x in roots], filename_candidates=candidates,
               payloads_opened=0, limits='Only these local output roots and filename metadata; external/browser artifacts not ruled out',
               D053_scope='validation24_only_not_test', reference_binding=None))
    save_draft(P / 'PRESERVATION_RECEIPT.json', dict(prior_readiness_hashes=preserved,
               held_out_payloads_opened=0, model_forward_calls=0, generation_calls=0, optimizer_updates=0,
               CAD='CLOSED_TECHNICAL_PASS_SCIENTIFIC_NO_GO', Boundary='PAUSED', Contrastive='CLOSED_NO_GO',
               F='PRESERVED_ORIGINAL_CE_EPOCH3_AND_ALL_CHECKPOINTS', base_family='PRESERVED_MATCHED_FALLBACK',
               GPU_execution_approved=False, source_artifacts_overwritten=False))
    save_draft(P / 'DELIVERY_STATUS.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               offline_RO_U='PASS_PENDING_SEPARATE_RESEARCHER_GPU_APPROVAL',
               Shuffled='FIXTURE_PASS_PRODUCTION_BLOCKED_EXACT_CUISINE_FALLBACK_POLICY',
               final_evaluation='BLOCKED_APPROVED_TEST_REFERENCES_AND_REAL_HUMAN_STUDY_AND_FINAL_ACCESS_AUTHORIZATION',
               selected_epoch=3, selected_step=750, F_development_informed_selection_disclosed=True,
               test_predictions=0, GPU_operations=0, report_scope='OFFLINE_PREPARATION_ONLY'))
    files = [p.relative_to(R).as_posix() for p in P.rglob('*') if p.is_file()]
    files += list(m.read(P/'PREFLIGHT_MANIFEST.json')['files'])
    files += ['approvals/RO_PENDING.json', 'approvals/U_PENDING.json']
    # Original methods excerpt permits independent review of the precise recovered rules.
    origin = R/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md'
    lines = origin.read_text(encoding='utf-8').splitlines()
    save_draft(P/'RECOVERED_SHUFFLED_PROTOCOL_SOURCE.json', dict(source=origin.relative_to(R).as_posix(),
               complete_document_sha256=m.sha(origin), first_line=1022, last_line=1050, excerpt=lines[1021:1050]))
    files.append((P/'RECOVERED_SHUFFLED_PROTOCOL_SOURCE.json').relative_to(R).as_posix())
    files = sorted(set(files))
    manifest = dict(scope='SELF_CONTAINED_OVERLAY_WITH_EXPLICIT_FULL_PARENT_DEPENDENCY', files={f:m.sha(R/f) for f in files})
    m.write(P/'PORTABLE_MANIFEST.json', manifest)
    files.append((P/'PORTABLE_MANIFEST.json').relative_to(R).as_posix())
    zip_path = R/'outputs/cloud_handoff/LED_LARGE_MATCHED_FAMILY_V1_OFFLINE_REVIEW.zip'
    m.require(not zip_path.exists(), 'Do not overwrite an existing delivery')
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files: z.write(R/rel, rel)
    with zipfile.ZipFile(zip_path) as z:
        m.require(z.testzip() is None, 'ZIP CRC error')
        for rel, digest in manifest['files'].items():
            m.require(hashlib.sha256(z.read(rel)).hexdigest() == digest, 'Portable SHA mismatch: '+rel)
    digest = m.sha(zip_path)
    with Path(str(zip_path)+'.sha256').open('x') as f: f.write(digest+'  '+zip_path.name+'\n')
    print(json.dumps(dict(zip=str(zip_path), bytes=zip_path.stat().st_size, manifest_files=len(manifest['files']), sha256=digest)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['bindings', 'seal'])
    a = p.parse_args()
    bindings() if a.mode == 'bindings' else seal()
