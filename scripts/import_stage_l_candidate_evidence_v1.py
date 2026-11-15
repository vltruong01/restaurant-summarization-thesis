"""Import structurally aligned candidate links only. No semantic/human certification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from stage_l_reference_app_v1 import Store, digest, file_hash, now


def select_candidates(candidate, reference, source):
    result = {}; summary = {'unlinked_review_spans': [], 'unlinked_visual_spans': []}
    counts = {'review_links': 0, 'visual_links': 0, 'span_mismatches': 0, 'bad_links': 0}
    reviews = {r['review_id']: r['text'] for r in source['reviews']}
    photos = {p['photo_id'] for p in source['photos']}
    for part, field in [('review', 'review_consensus'), ('visual', 'visual_evidence')]:
        selected = []
        for claim in candidate.get(part + '_claims', []):
            span = claim.get('output_span', '')
            if not span or span not in reference[field]:
                counts['span_mismatches'] += 1
                summary['unlinked_' + part + '_spans'].append('[Câu chữ khác bản hiện tại] ' + span)
                continue
            valid = []
            for link in claim.get('evidence', []):
                if part == 'review':
                    rid = link.get('review_id'); quote = link.get('quote', '')
                    ok = rid in reviews and bool(quote) and quote in reviews[rid]
                else:
                    ok = link.get('photo_id') in photos
                if not ok:
                    counts['bad_links'] += 1
                    continue
                item = copy.deepcopy(link)
                if part == 'visual':
                    item['direct_image_viewed'] = False
                valid.append(item); counts[part + '_links'] += 1
            if valid:
                selected.append({'output_span': span, 'evidence': valid})
            else:
                summary['unlinked_' + part + '_spans'].append(span)
        result[part + '_claims'] = selected
    return result, summary, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--commit', action='store_true')
    args = p.parse_args(); store = Store(args.workspace)
    supplied = json.loads(args.candidates.read_text(encoding='utf-8-sig'))
    rows = supplied['businesses']; by_id = {r['business_id']: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(store.frame):
        raise ValueError('Exact unique frozen cohort required')
    out = args.destination; out.mkdir(parents=True, exist_ok=True)
    frozen_input = out/'CANDIDATE_EVIDENCE_ORIGINAL.json'
    if frozen_input.exists() and file_hash(frozen_input) != file_hash(args.candidates):
        raise ValueError('Preserve existing candidate intake; different input')
    if not frozen_input.exists(): shutil.copyfile(args.candidates, frozen_input)
    before = {}; prepared = []; report = []
    input_hash = file_hash(frozen_input)
    for bid in store.frame:
        r, version = store.get(bid); before[bid] = {k: r[k] for k in ['review_consensus', 'visual_evidence']}
        if r['status'] != 'DRAFT' or r['author']['human_id'] != 'author01' or r['review_claims'] or r['visual_claims']:
            report.append({'business_id': bid, 'status': 'PRESERVED_EXISTING_WORK', 'version': version})
            continue
        links, summary, counts = select_candidates(by_id[bid], r, store.source(bid))
        notes = (r['notes'] + '\n\nAI-assisted candidate evidence imported on researcher instruction. '
                 'Only IDs, literal quotations and exact output spans were mechanically aligned; '
                 'semantic support, consensus scope and human source inspection remain unconfirmed. '
                 'Input SHA256: ' + input_hash + '.\nEVIDENCE_SUGGESTIONS_SUMMARY: ' +
                 json.dumps(summary, ensure_ascii=False, separators=(',', ':')))
        draft = {**before[bid], **links, 'notes': notes}
        prepared.append((bid, version, draft))
        report.append({'business_id': bid, 'status': 'PREPARED', 'version_before': version,
                       **counts, 'unlinked_review_groups': len(summary['unlinked_review_spans']),
                       'unlinked_visual_groups': len(summary['unlinked_visual_spans'])})
    receipt = {'candidate_file_sha256': input_hash, 'all_businesses': len(rows), 'records': report,
               'semantic_support_verified': False, 'created_at_utc': now(), 'committed': False}
    if args.commit:
        backup = store.root/'PRIVATE_STATE/BACKUPS'/('BEFORE_CANDIDATE_EVIDENCE_' + input_hash[:16] + '.sqlite3')
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            with store.connect() as src, sqlite3.connect(backup) as dst: src.backup(dst)
        with store.connect() as c:
            users = [tuple(r) for r in c.execute('SELECT * FROM users ORDER BY id')]
        for bid, version, draft in prepared:
            result = store.apply(bid, {'id': 'author01', 'role': 'author'}, 'save', {
                'expected_version': version, 'draft': draft,
                'comment': 'Researcher-authorized administrative candidate-evidence import. No human confirmations, submission or approval.'})
            r = result['reference']
            assert r['status'] == 'DRAFT' and not r['author']['human_written_attestation']
            assert not r['checker']['independent_check_attestation']
            assert all(e['direct_image_viewed'] is False for cl in r['visual_claims'] for e in cl['evidence'])
        for bid, text in before.items():
            r, _ = store.get(bid)
            assert all(r[k] == v for k, v in text.items()), 'Prose changed during evidence intake'
        with store.connect() as c:
            assert [tuple(r) for r in c.execute('SELECT * FROM users ORDER BY id')] == users
        audit = store.audit(); assert not audit['journal_errors']
        receipt.update(committed=True, imported=len(prepared), all_prose_preserved=True,
                       backup_path=str(backup), backup_sha256=file_hash(backup), counts=audit['counts'],
                       journal_errors=audit['journal_errors'], accounts_unchanged=True,
                       note='Pending direct-image confirmations are expected in unsubmitted drafts; these are not approved references.')
    target = out/('IMPORT_RECEIPT_' + digest(receipt)[:12] + '.json')
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'receipt': str(target), 'prepared_or_imported': len(prepared), 'committed': args.commit,
                      'review_links': sum(r.get('review_links', 0) for r in report),
                      'visual_links': sum(r.get('visual_links', 0) for r in report),
                      'unaligned_spans_not_imported': sum(r.get('span_mismatches', 0) for r in report),
                      'counts': receipt.get('counts')}, ensure_ascii=False))


if __name__ == '__main__': main()
