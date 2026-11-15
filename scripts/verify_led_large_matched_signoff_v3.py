"""Structural, integrity and reporting checks only; no model or held-out payload."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from prepare_led_large_matched_signoff_v3 import OUT, OLD, ARCHIVE, ROOT, fixture_statistics, sha


def main():
    def read(p):
        return json.loads(p.read_text(encoding='utf-8'))

    before = read(OUT / 'MANIFEST.json')['files']
    for rel, digest in before.items():
        assert sha(OUT / rel) == digest, rel
    checks = ['Initial portable SHA256 manifest PASS']
    for short in ('RO', 'U'):
        approval = OUT / f'{short}_GATE_ONLY_SIGNOFF/APPROVAL_PENDING.json'
        assert approval.read_bytes() == (OLD / f'PENDING_APPROVALS/{short}_GATE_ONLY_PENDING.json').read_bytes()
        a = read(approval)
        assert a['authorized_at_utc'] is None and a['deadline_utc'] is None
        assert a['GPU_feasibility_approved'] is False and a['training_authorized'] is False
        assert a['max_successful_gate_updates'] == 3 and a['max_generation_calls'] == 2
        assert a['max_training_updates'] == 0 and a['automatic_train'] is False
        assert not (ROOT / read(OLD / f'{short}_GPU_AUTHORIZATION_READINESS.json')['output_namespace']).exists()
    checks.append('Both unsigned approvals byte-identical to accepted V2;3updates/2TRAINtraces/0training; fresh local namespaces')
    assert (OUT / 'SHUFFLED_POLICY_PROPOSED_UNCHANGED.json').read_bytes() == (OLD / 'SHUFFLED_POLICY_PROPOSED.json').read_bytes()
    assert read(OUT / 'SHUFFLED_POLICY_ADJUDICATION_PENDING.json')['researcher_explicit_acceptance'] is None
    checks.append('Exact Shuffled policy preserved; researcher acceptance remains pending')
    # Independently hand-calculated assignment proportions:2/3 and1/3.
    rows = [{'caption_count': 1, 'tier': 'EXACT_SHARED_ALLOWED_CATEGORY', 'split': 'train',
             'donor_id': 'A', 'recipient_tags': ['Thai'], 'donor_tags': ['Thai'], 'eligible_count': 1},
            {'caption_count': 1, 'tier': 'DECLARED_SAME_COUNT_FALLBACK', 'split': 'train',
             'donor_id': 'A', 'recipient_tags': [], 'donor_tags': ['Thai'], 'eligible_count': 1},
            {'caption_count': 1, 'tier': 'DECLARED_SAME_COUNT_FALLBACK', 'split': 'train',
             'donor_id': 'B', 'recipient_tags': [], 'donor_tags': [], 'eligible_count': 1},
            {'caption_count': 0, 'split': 'train'}]
    s = fixture_statistics(rows)
    assert s['category_match_rate'] == 1 / 3 and s['fallback_rate'] == 2 / 3
    assert abs(s['donor_assignment_HHI'] - 5 / 9) < 1e-12
    assert s['max_donor_assignment_share'] == 2 / 3 and s['zero_caption_unchanged'] == 1
    assert fixture_statistics([])['donor_assignment_HHI'] is None
    assert fixture_statistics([{'caption_count': 0, 'split': 'train'}])['category_match_rate'] is None
    checks.append('Assignment concentration/rate denominators verified against hand-calculated reuse and zero-caption cases; empty denominators N/A')
    audit = read(OUT / 'REFERENCE_METADATA_INVENTORY.json')
    assert not audit['errors'] and all(not s['truncated'] for s in audit['scope'])
    assert all(not x['contents_opened'] and not x['archive_members_opened'] for x in audit['candidates'])
    index = read(OUT / 'REFERENCE_COVERAGE_APPROVAL_INDEX.json')
    assert index['verified_unique_test_coverage'] is None and index['missing_test_count'] is None
    h = read(OUT / 'HUMAN_COORDINATION_PENDING.json')
    assert len(h['annotators']) == 3 and all(not x['is_confirmed_real_annotator'] for x in h['annotators'])
    assert h['rating_sets_total'] == 30 * 4 * 3 == 360
    assert h['expected_numeric_fields'] == 360 * 6 - 30 * 3 == 2070
    checks.append('Untruncated metadata scope, unknown coverage, unconfirmed real raters and360sets/2070numeric fields preserved')
    report = {'status': 'OFFLINE_STRUCTURAL_AND_REPORTING_CHECKS_PASS', 'checks': checks,
              'accepted_scientific_code_changed': False, 'new_pinned_host_CPU_suites_run': False,
              'note': 'These checks are not the90-test host preflight, semantic reference verification or GPU feasibility.',
              'GPU_operations': 0, 'held_out_payloads_opened': 0}
    (OUT / 'SUPPLEMENT_VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(Path(__file__), OUT / 'VALIDATION_SOURCE.py')
    entries = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and p.name != 'MANIFEST.json'}
    (OUT / 'MANIFEST.json').write_text(json.dumps({'path_base': 'archive_root', 'files': entries}, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for rel, digest in entries.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == digest
        assert hashlib.sha256(z.read('ACCEPTED_V2.zip')).hexdigest() == '06e0bdb1da82941c9a4638ad4db03b43c528e37bb7d372efe5f5491c6b8cf73b'
    ARCHIVE.with_suffix('.zip.sha256').write_text(sha(ARCHIVE) + '  ' + ARCHIVE.name + '\n', encoding='ascii')
    print(json.dumps({'checks': checks, 'manifest_entries': len(entries), 'ZIP_bytes': ARCHIVE.stat().st_size,
                      'ZIP_sha256': sha(ARCHIVE)}, indent=2))


if __name__ == '__main__':
    main()
