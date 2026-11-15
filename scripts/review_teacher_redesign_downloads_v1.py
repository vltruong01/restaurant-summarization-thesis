"""Read-only scientific inputs; import/download audit and CPU token counts only."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import hashlib
import json
import shutil
import statistics
import struct
import zipfile
import zlib

from teacher_redesign_token_count_v1 import load_tokenizer, target_counts, validate_proposal

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = Path('C:/Users/VUONGLOCTRUONG/Downloads')
OUT = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_result_review_v1'
PREP = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_v1/bundle'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member: ' + name)
    return p


def recover_complete_local_entries(path, dest):
    """Recover only CRC-verified complete members; never claim a repaired ZIP."""
    raw = path.read_bytes()
    offset, members, failure = 0, {}, None
    while offset + 30 <= len(raw):
        sig, version, flags, method, mtime, mdate, crc, size, usize, nlen, xlen = struct.unpack_from('<4s5H3I2H', raw, offset)
        if sig != b'PK\x03\x04':
            failure = {'offset': offset, 'reason': 'No next complete local header'}
            break
        name = raw[offset + 30:offset + 30 + nlen].decode('utf-8')
        safe_name(name)
        start, end = offset + 30 + nlen + xlen, offset + 30 + nlen + xlen + size
        if flags & 8 or end > len(raw) or usize > 100_000_000:
            failure = {'member': name, 'reason': 'Incomplete/unsupported local entry; not recovered'}
            break
        try:
            content = zlib.decompress(raw[start:end], -15) if method == 8 else raw[start:end] if method == 0 else None
            if content is None or len(content) != usize or zlib.crc32(content) & 0xffffffff != crc:
                raise ValueError('Size/CRC mismatch')
        except (ValueError, zlib.error) as error:
            failure = {'member': name, 'reason': str(error)}
            break
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        members[name] = hashlib.sha256(content).hexdigest()
        offset = end
    return {'whole_zip_valid': False, 'complete_CRC_verified_members': members,
            'stopped_at': failure, 'unrecovered_tail_not_certified': True}


def main():
    assert not OUT.exists(), 'Do not overwrite an existing audit'
    OUT.mkdir(parents=True)
    incoming = OUT / 'incoming'
    incoming.mkdir()
    inputs = {}
    for name in ('STAGE_A_INDEPENDENT_DRAFTS.zip', 'STAGE_B_FIRST_PASS_COMPARISON_32.zip'):
        path = DOWNLOAD / name
        inputs[name] = {'bytes': path.stat().st_size, 'sha256': sha(path), 'whole_zip_valid': zipfile.is_zipfile(path)}
        shutil.copyfile(path, incoming / name)
        assert sha(incoming / name) == inputs[name]['sha256']
    bdir = OUT / 'stage_b_verified'
    bdir.mkdir()
    with zipfile.ZipFile(incoming / 'STAGE_B_FIRST_PASS_COMPARISON_32.zip') as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        for info in z.infolist():
            safe_name(info.filename)
            assert (info.external_attr >> 16) & 0o170000 != 0o120000
            target = bdir / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    manifest_b = read(bdir / 'SHA256_OUTPUT_MANIFEST.json')['files']
    assert all(sha(bdir / name) == value for name, value in manifest_b.items())
    inputs['STAGE_B_FIRST_PASS_COMPARISON_32.zip'].update(CRC='PASS', manifest_files_verified=len(manifest_b))
    adir = OUT / 'stage_a_complete_entries_only'
    adir.mkdir()
    recovery = recover_complete_local_entries(incoming / 'STAGE_A_INDEPENDENT_DRAFTS.zip', adir)
    manifest_a = read(adir / 'SHA256_OUTPUT_MANIFEST.json')['files']
    checked_a = {name: value for name, value in recovery['complete_CRC_verified_members'].items() if name in manifest_a}
    assert all(value == manifest_a[name] for name, value in checked_a.items())
    recovery['manifest_bound_recovered_members_verified'] = len(checked_a)
    save(OUT / 'STAGE_A_PARTIAL_ARCHIVE_RECOVERY.json', recovery)
    a_file = adir / 'drafts/proposed_targets.jsonl'
    b_a_file = bdir / 'locked_inputs/proposed_targets_STAGE_A_UNCHANGED.jsonl'
    assert a_file.read_bytes() == b_a_file.read_bytes()
    assert sha(a_file) == 'a98f2ae80b4ae5155fe3621567154b2757c0976adbeffe8cfe596e36f4d7742e'
    assert (adir / 'SOURCE_EVIDENCE.jsonl').read_bytes() == (PREP / 'SOURCE_EVIDENCE.jsonl').read_bytes()
    original_file = bdir / 'locked_inputs/ORIGINAL_TARGETS_UNCHANGED.jsonl'
    assert original_file.read_bytes() == (PREP / 'stage_b_comparison/ORIGINAL_TARGETS.jsonl').read_bytes()
    assert (bdir / 'locked_inputs/ORIGINAL_EXPORTED_TOKEN_COUNTS_NOT_REMEASURED.json').read_bytes() == (PREP / 'stage_b_comparison/ORIGINAL_TARGET_TOKEN_COUNTS.json').read_bytes()
    source = [json.loads(line) for line in (PREP / 'SOURCE_EVIDENCE.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = [r['business_id'] for r in source]
    proposals = [json.loads(line) for line in a_file.read_text(encoding='utf-8').splitlines()]
    originals = [json.loads(line) for line in original_file.read_text(encoding='utf-8').splitlines()]
    assert len(proposals) == 32 and len({r['business_id'] for r in proposals}) == 32
    assert {r['business_id'] for r in proposals} == set(expected) == {r['business_id'] for r in originals}
    s_map = {r['business_id']: r for r in source}
    o_map = {r['business_id']: r for r in originals}
    p_map = {r['business_id']: r for r in proposals}
    mechanical = []
    for row in proposals:
        try:
            result = validate_proposal(row, s_map[row['business_id']])
        except (ValueError, KeyError) as error:
            result = {'status': 'FAIL_REQUIRES_REVIEW_NO_AUTOMATIC_EDIT', 'error': str(error)}
        mechanical.append({'business_id': row['business_id'], **result})
    # Independent local JSON schema validation, if available; no installs/network.
    try:
        import jsonschema
        schema = read(PREP / 'TARGET_OUTPUT_SCHEMA.json')
        schema_errors = [{ 'business_id': r['business_id'], 'path': list(e.path), 'error': e.message}
                         for r in proposals for e in jsonschema.Draft202012Validator(schema).iter_errors(r)]
        schema_status = 'PASS' if not schema_errors else 'FAIL'
    except ImportError:
        schema_status, schema_errors = 'UNAVAILABLE_NO_ESTIMATE', []
    tok = load_tokenizer(PREP / 'tokenizer')
    measurements = []
    for bid in expected:
        for condition in ('reviews_only', 'unfiltered', 'filtered'):
            old = target_counts(tok, o_map[bid]['targets'][condition], condition)
            new = target_counts(tok, p_map[bid]['targets'][condition]['text'], condition)
            assert old == o_map[bid]['target_token_counts'][condition]
            measurements.append({'business_id': bid, 'stratum': s_map[bid]['selection_category'],
                'condition': condition, 'old': old, 'new': new,
                'delta_content_EOS_tokens_new_minus_old': new['corrected_training_content_EOS_label_tokens'] - old['corrected_training_content_EOS_label_tokens'],
                'measurement_origin': 'LOCAL_PINNED_TOKENIZER_NO_MODEL', 'quality_improvement_implied': False})
    save(OUT / 'LOCAL_EXACT_LED_TOKEN_COUNTS_96.json', measurements)
    summary = []
    for stratum in ('random', 'zero_filtered', 'review_disagreement', 'varied_visual', 'information_dense', 'all32_descriptive_only'):
        for condition in ('reviews_only', 'unfiltered', 'filtered'):
            group = [m for m in measurements if m['condition'] == condition and
                     (stratum == 'all32_descriptive_only' or m['stratum'] == stratum)]
            summary.append({'stratum': stratum, 'condition': condition, 'businesses': len(group),
                'old_mean_content_EOS_tokens': statistics.mean(m['old']['corrected_training_content_EOS_label_tokens'] for m in group),
                'new_mean_content_EOS_tokens': statistics.mean(m['new']['corrected_training_content_EOS_label_tokens'] for m in group),
                'new_min': min(m['new']['corrected_training_content_EOS_label_tokens'] for m in group),
                'new_max': max(m['new']['corrected_training_content_EOS_label_tokens'] for m in group),
                'new_review_body_mean_no_special': statistics.mean(m['new']['review_body_standalone_no_special'] for m in group),
                'new_visual_body_mean_no_special': None if condition == 'reviews_only' else statistics.mean(m['new']['visual_body_standalone_no_special'] for m in group)})
    save(OUT / 'LOCAL_TOKEN_COUNT_SUMMARY.json', summary)
    comparisons = [json.loads(line) for line in (bdir / 'COMPARISONS_96.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(comparisons) == len({(r['business_id'], r['condition']) for r in comparisons}) == 96
    for c in comparisons:
        bid, condition = c['business_id'], c['condition']
        assert c['old_text'] == o_map[bid]['targets'][condition]
        assert c['new_text'] == p_map[bid]['targets'][condition]['text']
        assert c['researcher_confirmed'] is False
    business = read(bdir / 'BUSINESS_COMPARISONS_32.json')
    assert {r['business_id'] for r in business} == set(expected) and len(business) == 32
    counts = {stratum: dict(Counter(r['preference'] for r in business if r['stratum'] == stratum))
              for stratum in ('random', 'zero_filtered', 'review_disagreement', 'varied_visual', 'information_dense')}
    assert all(sha(DOWNLOAD / name) == info['sha256'] for name, info in inputs.items())
    receipt = {'status': 'DOWNLOADS_AUDITED_ORIGINALS_UNCHANGED_QUALITY_REVIEW_PENDING',
        'archives': inputs, 'stage_A_whole_ZIP_damaged': True,
        'stage_A_actual_draft_JSONL_recovered_complete': True,
        'stage_A_draft_matches_Stage_B_locked_copy': True, 'draft_sha256': sha(a_file),
        'source_copy_matches_frozen_export': True, 'original_targets_copy_matches_frozen_export': True,
        'businesses': 32, 'condition_comparisons': 96,
        'mechanical_proposal_status_counts': dict(Counter(r['status'] for r in mechanical)),
        'schema_validation': schema_status, 'schema_errors': schema_errors,
        'mechanical_findings': mechanical, 'exact_old_new_token_pairs_measured_locally': 96,
        'old_exported_token_counts_reproduced': 96,
        'reported_assistant_preferences_NOT_researcher_labels': counts,
        'no_drafts_edited': True, 'no_training_or_model_API_calls': True,
        'scientific_GO_NO_GO': 'NOT_ASSIGNED'}
    save(OUT / 'DOWNLOAD_REVIEW_RECEIPT.json', receipt)
    report = '# Kiểm tra file tải về — Teacher redesign A/B\n\n'
    report += '**Đã có đủ 32 bản A và 96 cặp so sánh B; chưa phải target được researcher phê duyệt.**\n\n'
    report += 'ZIP A không có central directory hợp lệ và có một entry tokenizer không giải nén hoàn chỉnh. '
    report += f'Đã giữ nguyên ZIP lỗi, khôi phục {len(recovery["complete_CRC_verified_members"])} entry hoàn chỉnh có CRC đúng. '
    report += 'Bản drafts/proposed_targets.jsonl đủ 32 ca và trùng byte-for-byte với bản A khóa trong ZIP B. '
    report += 'Không sửa ZIP A hay giả định các entry chưa khôi phục là đầy đủ.\n\n'
    report += f'ZIP B: CRC PASS, {len(manifest_b)} SHA256 manifest PASS. Nguồn và target cũ trong các bản tải về khớp bản export đóng băng. '
    report += f'Đối chiếu 96 cặp old/new khớp văn bản gốc. Mechanical checks: {receipt["mechanical_proposal_status_counts"]}; JSON schema: {schema_status}.\n\n'
    report += 'Đã bổ sung phép đo CPU với tokenizer pinned 0.21.1: 96 cặp old/new, tái lập đúng cả 96 số token cũ. '
    report += 'Kết quả đo lưu riêng, không sửa báo cáo GPT Work vốn ghi thiếu thư viện. Token giảm không chứng minh chất lượng tăng.\n\n'
    report += '|Nhóm / điều kiện|n|Token cũ TB|Token A TB|A min–max|\n|---|---:|---:|---:|---:|\n'
    for r in summary:
        report += f'|{r["stratum"]} / {r["condition"]}|{r["businesses"]}|{r["old_mean_content_EOS_tokens"]:.1f}|{r["new_mean_content_EOS_tokens"]:.1f}|{r["new_min"]}–{r["new_max"]}|\n'
    report += '\nToken bảng là corrected content/EOS labels, gồm headers; breakdown phần thân có trong JSON.\n\n'
    report += 'Báo cáo B là tự đánh giá không blind của chính trợ lý đã viết A. Core/Supporting là panel hậu nghiệm chưa được duyệt; '
    report += '32 doanh nghiệp là đơn vị so sánh, không coi 96 nhánh RO/U/F là mẫu độc lập. '
    report += 'Random24: mới 7, cũ 2, hòa 14, chưa kết luận 1; hai ca zero-F đều ưu tiên cũ về nội dung dù visual cả hai bên rỗng đúng. '
    report += 'Các kết quả này chưa đủ để thay toàn bộ Teacher targets. Giữ nguyên A/old/B và chờ researcher review; không train hoặc tự sửa A.\n'
    (OUT / 'DOWNLOAD_REVIEW_VI.md').write_text(report, encoding='utf-8')
    print(json.dumps({'receipt': str(OUT / 'DOWNLOAD_REVIEW_RECEIPT.json'),
          'report': str(OUT / 'DOWNLOAD_REVIEW_VI.md'),
          'mechanical': receipt['mechanical_proposal_status_counts'], 'schema': schema_status,
          'token_summary_all32': summary[-3:], 'recovered_A_entries': len(recovery['complete_CRC_verified_members']),
          'reported_preferences': counts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
