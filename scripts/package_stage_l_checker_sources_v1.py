"""Package existing prose, exact sources, individual photos and separate evidence suggestions."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from stage_l_reference_app_v1 import Store, digest, file_hash, need, new_json, now, safe_under


def main():
    workspace = Path('outputs/stage_l/test100_human_references_v1')
    dest = Path('outputs/stage_l/test100_checker_reference_package_20261031_v1')
    dest.mkdir(parents=True, exist_ok=False)
    content = dest/'PACKAGE'; content.mkdir()
    store = Store(workspace)
    with store.connect() as conn:
        conn.execute('BEGIN')
        rows_before = [tuple(x) for x in conn.execute('SELECT * FROM refs ORDER BY business')]
        events_before = [tuple(x) for x in conn.execute('SELECT * FROM events ORDER BY seq')]
        users_before = [tuple(x) for x in conn.execute('SELECT * FROM users ORDER BY id')]
        snapshot = {bid: store.get(bid, conn) for bid in store.frame}
    ledger = []; readable = []; photos = []; reviews = 0
    for index, (bid, (r, version)) in enumerate(snapshot.items(), 1):
        source = store.source(bid)
        need(not store.errors(r), 'Invalid existing record: ' + bid)
        relative = f'BUSINESSES/{index:03d}_{bid}'
        folder = content/relative; (folder/'PHOTOS').mkdir(parents=True)
        heading = f"BUSINESS_ID: {bid}\nBUSINESS_NAME: {source['business_metadata'].get('name', '')}\nSTATUS: {r['status']}\nVERSION: {version}\n"
        text = heading + '\n[REVIEW_CONSENSUS]\n' + r['review_consensus'] + '\n\n[VISUAL_EVIDENCE]\n' + r['visual_evidence'] + '\n'
        (folder/'REFERENCE_REVIEW_VISUAL.txt').write_text(text, encoding='utf8')
        readable.append(f'BUSINESS {index:03d}/100\n' + text)
        blocks = [heading, 'EXACT SELECTED REVIEWS — complete original text; no Teacher or model answers.\n']
        for review in source['reviews']:
            blocks.append(f"REVIEW_ID: {review['review_id']}\nSTARS: {review.get('stars', '')}\nDATE: {review.get('date', '')}\n{review['text']}\n")
        (folder/'SELECTED_REVIEWS.txt').write_text('\n'.join(blocks), encoding='utf8')
        frozen = safe_under(store.package, store.frame[bid]['source_file'])
        shutil.copy2(frozen, folder/'FROZEN_SOURCE.json')
        need(digest(json.loads((folder/'FROZEN_SOURCE.json').read_text(encoding='utf8'))) == store.frame[bid]['sha256'], 'Frozen source changed')
        suggestions = [heading, 'EXISTING AI-ASSISTED EVIDENCE SUGGESTIONS — optional examples only.\nLiteral matching and IDs do not establish semantic support or full coverage. No checker approval is implied. Read the original sources and decide yourself.\n']
        for component in ['review', 'visual']:
            suggestions.append(component.upper() + ' LINKS\n')
            for claim in r[component+'_claims']:
                suggestions.append('Output span: ' + claim['output_span'])
                for evidence in claim['evidence']:
                    if component == 'review':
                        suggestions.append('  Review ID: ' + evidence['review_id'] + '\n  Exact quotation: ' + evidence['quote'])
                    else:
                        suggestions.append('  Photo file: PHOTOS/' + evidence['photo_id'] + '.jpg')
                    if evidence.get('note'):
                        suggestions.append('  Existing link note: ' + evidence['note'])
                suggestions.append('')
        suggestions.append('EXISTING PROVENANCE / UNLINKED CANDIDATES (may refer to earlier wording):\n' + r['notes'])
        (folder/'AI_EVIDENCE_SUGGESTIONS.txt').write_text('\n'.join(suggestions), encoding='utf8')
        item = {'order': index, 'business_id': bid, 'name': source['business_metadata'].get('name', ''), 'folder': relative, 'version': version, 'status': r['status'], 'source_bundle_sha256': store.frame[bid]['sha256'], 'review_sha256': hashlib.sha256(r['review_consensus'].encode()).hexdigest(), 'visual_sha256': hashlib.sha256(r['visual_evidence'].encode()).hexdigest(), 'selected_review_ids': source['selected_review_ids'], 'photo_ids': [p['photo_id'] for p in source['photos']], 'reviews': len(source['reviews']), 'photos': len(source['photos'])}
        ledger.append(item); reviews += len(source['reviews'])
        for photo in source['photos']:
            original = store.photo_path(bid, photo['photo_id'])
            target = folder/'PHOTOS'/(photo['photo_id']+'.jpg')
            shutil.copy2(original, target)
            need(file_hash(target) == photo['original_source_sha256'], 'Photo bytes differ')
            photos.append({'business_id': bid, 'photo_id': photo['photo_id'], 'path': target.relative_to(content).as_posix(), 'sha256': photo['sha256']})
        if not source['photos']:
            (folder/'PHOTOS'/'NO_PHOTOS.txt').write_text('No original photos supplied for this business.\n', encoding='utf8')
    need(len(ledger) == len({x['business_id'] for x in ledger}) == 100, 'Business total')
    need(reviews == store.config['expected_reviews'] and len(photos) == store.config['expected_photos'], 'Source total')
    (content/'REFERENCES_100_REVIEW_VISUAL.txt').write_text('\n\n'.join(readable), encoding='utf8')
    new_json(content/'BUSINESS_LEDGER.json', ledger)
    new_json(content/'PHOTO_LEDGER.json', photos)
    with (content/'BUSINESS_INDEX.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['order', 'business_id', 'name', 'folder', 'status', 'version', 'reviews', 'photos'])
        writer.writeheader(); writer.writerows({k: row[k] for k in writer.fieldnames} for row in ledger)
    (content/'README_VI.md').write_text('''# Gói tham khảo cho checker — 100 business

1. Mở BUSINESS_INDEX.csv để tìm nhà hàng theo số thứ tự hoặc tên.
2. REFERENCES_100_REVIEW_VISUAL.txt chứa toàn bộ 100 bản Review/Visual hiện tại, giữ nguyên câu chữ. Mỗi thư mục business cũng có bản riêng REFERENCE_REVIEW_VISUAL.txt.
3. SELECTED_REVIEWS.txt chứa đầy đủ selected reviews, kèm review IDs. PHOTOS/ chứa từng ảnh JPG gốc riêng biệt, tên file là photo ID. Không có ảnh ghép hoặc ảnh bị thu nhỏ. FROZEN_SOURCE.json giữ nguyên dữ liệu nguồn và các binding để đối chiếu.
4. AI_EVIDENCE_SUGGESTIONS.txt được để riêng: ví dụ các liên kết bằng chứng đã có, bao gồm nguồn gốc và điểm chưa chắc. Không phải đáp án chuẩn, không chứng minh mọi câu đã đúng. Có thể tự đọc nguồn trước, rồi mới mở file này tham khảo cách nối câu với review/photo ID.
5. Khi kiểm: chú ý câu không có nguồn, sai phủ định/đối tượng, biến một người thành số đông, thiếu ý kiến tiêu cực quan trọng, ghép chi tiết các ảnh thành một cảnh. Không cần sửa chỉ vì khác văn phong.
6. Nếu cần sửa, ghi business ID, nguyên câu cần xem lại, review/photo ID và lý do ngắn. Nếu ổn, checker thực hiện xác nhận và duyệt đúng phiên bản trên ứng dụng. Đọc ZIP không tự tạo phê duyệt.

Đây là ảnh chụp dữ liệu của ứng dụng tại lúc đóng gói. 100 bài đang SUBMITTED, chưa phải bộ reference đã được checker duyệt. Việc người tham gia thực tế dùng hay không dùng hỗ trợ AI cần mô tả đúng thực tế; tên tài khoản hoặc gói file không chứng minh tính độc lập. Các gợi ý bằng chứng trong gói đã có nguồn gốc AI hỗ trợ.

Gói không chứa tài khoản/mật khẩu, database, model predictions, Teacher answers hoặc điểm đánh giá. Không chạy lại AI, tạo summary hay thay đổi nội dung trong lần đóng gói này.
''', encoding='utf8')
    manifest = {p.relative_to(content).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in sorted(content.rglob('*')) if p.is_file()}
    new_json(content/'PACKAGE_MANIFEST.json', {'created_at_utc': now(), 'businesses': 100, 'selected_reviews': reviews, 'individual_original_photos': len(photos), 'source_manifest_sha256': store.config['source_manifest_sha256'], 'snapshot_type': 'submitted_not_checker_approved', 'files': manifest})
    archive = dest/'TEST100_CHECKER_REFERENCE_WITH_INDIVIDUAL_PHOTOS.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(content.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(content).as_posix())
    with tempfile.TemporaryDirectory(prefix='checker_package_verify_') as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            names = z.namelist(); need(len(names) == len(set(names)), 'Duplicate archive member')
            need(z.testzip() is None, 'ZIP CRC failed')
            need(set(names) == set(manifest) | {'PACKAGE_MANIFEST.json'}, 'ZIP member set')
            z.extractall(extracted)
        for name, binding in manifest.items():
            p = extracted/name
            need(p.stat().st_size == binding['bytes'] and file_hash(p) == binding['sha256'], 'Extracted member mismatch')
        need(file_hash(extracted/'PACKAGE_MANIFEST.json') == file_hash(content/'PACKAGE_MANIFEST.json'), 'Manifest member mismatch')
        need(len(list(extracted.rglob('*.jpg'))) == 1136, 'Photo file count')
        for item in ledger:
            b = item['business_id']; r = snapshot[b][0]; folder = extracted/item['folder']
            src = json.loads((folder/'FROZEN_SOURCE.json').read_text(encoding='utf8'))
            need(src == store.source(b), 'Source content changed')
            text = (folder/'REFERENCE_REVIEW_VISUAL.txt').read_text(encoding='utf8')
            review, visual = text.split('\n[REVIEW_CONSENSUS]\n', 1)[1].split('\n\n[VISUAL_EVIDENCE]\n', 1)
            need(review == r['review_consensus'] and visual == r['visual_evidence']+'\n', 'Prose changed')
        need(json.loads((extracted/'BUSINESS_LEDGER.json').read_text(encoding='utf8')) == ledger, 'Ledger changed')
    with store.connect() as conn:
        need(rows_before == [tuple(x) for x in conn.execute('SELECT * FROM refs ORDER BY business')], 'App refs changed during packaging')
        need(events_before == [tuple(x) for x in conn.execute('SELECT * FROM events ORDER BY seq')], 'App events changed during packaging')
        need(users_before == [tuple(x) for x in conn.execute('SELECT * FROM users ORDER BY id')], 'Accounts changed')
    checksum = file_hash(archive)
    (dest/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n', encoding='ascii')
    report = {'status': 'PASS', 'businesses': 100, 'selected_reviews': reviews, 'individual_original_photos': len(photos), 'unchanged_component_texts': 200, 'zip_reopened_crc_extracted_parsed': True, 'all_member_sha256_verified': True, 'members': len(manifest)+1, 'application_refs_events_users_unchanged': True, 'no_checker_approvals_created': True, 'archive_bytes': archive.stat().st_size, 'archive_sha256': checksum}
    new_json(dest/'PACKAGE_VERIFICATION.json', report)
    print(json.dumps({'zip': str(archive), **report}, ensure_ascii=False))


if __name__ == '__main__':
    main()
