"""Import explicit offline checker decisions through the existing versioned workflow.

Preserves participant declarations as supplied, without independently authenticating
identity, source inspection or the absence of AI assistance. Never edits references.
"""
import argparse
from collections import Counter
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3

from stage_l_reference_app_v1 import (Store, CHECK_ATTESTATIONS, digest, file_hash,
    reference_content, new_json, need, now)

ACTIONS = {'APPROVE': 'approve', 'REQUEST_REVISION': 'request_revision', 'REJECT': 'reject'}


def validate(store, doc):
    need(doc.get('format') == 'stage_l_offline_checker_review_v1', 'Unexpected JSON format')
    identity = doc.get('checker_id')
    with store.connect() as c:
        user = c.execute('SELECT role FROM users WHERE id=?', (identity,)).fetchone()
    need(user is not None and user['role'] == 'checker', 'Registered checker required')
    rows = doc.get('reviews')
    need(isinstance(rows, list) and len(rows) == len(store.frame), 'Complete cohort required')
    need(all(isinstance(x, dict) for x in rows), 'Malformed business record')
    ids = [r.get('business_id') for r in rows]
    need(len(set(ids)) == len(ids) and set(ids) == set(store.frame), 'Duplicate, missing or foreign business')
    need(ids == list(store.frame), 'Frozen business order changed')
    declarations = doc.get('checker_declarations', {})
    if any(r.get('decision') == 'APPROVE' for r in rows):
        need(all(declarations.get(k) is True for k in CHECK_ATTESTATIONS), 'All checker declarations required for approvals')
    plan = []
    for order, row in enumerate(rows, 1):
        b = row['business_id']; r, version = store.get(b)
        need(row.get('order') == order, 'Order number mismatch')
        need(row.get('business_name') == store.source(b)['business_metadata'].get('name', ''), 'Business name mismatch')
        need(row.get('status_at_export') == 'SUBMITTED', 'Expected submitted export')
        need(type(row.get('reference_version')) is int and row['reference_version'] == version, 'Stale reference version: ' + b)
        need(row.get('reference_content_sha256') == digest(reference_content(r)), 'Stale content hash: ' + b)
        decision = row.get('decision'); comment = row.get('comment')
        need(decision in set(ACTIONS) | {'PENDING'}, 'Unknown decision')
        need(isinstance(comment, str) and len(comment) <= 20000, 'Invalid checker comment')
        if decision == 'PENDING':
            continue
        need(r['status'] == 'SUBMITTED', 'Reference no longer submitted: ' + b)
        need(identity != r['author']['human_id'], 'Author and checker must differ')
        if decision != 'APPROVE':
            need(comment.strip(), 'Revision/rejection reason required')
        else:
            probe = copy.deepcopy(r)
            probe['status'] = 'APPROVED'
            probe['checker'] = {'human_id': identity, 'at_utc': now(),
                'independent_check_attestation': True, 'predictions_seen': False,
                'reviewed_content_sha256': digest(reference_content(r))}
            need(not store.errors(probe, True), 'Approval integrity blocked: ' + b + ' ' + str(store.errors(probe, True)))
        plan.append({'business_id': b, 'version': version, 'action': ACTIONS[decision], 'comment': comment})
    need(not store.audit()['journal_errors'], 'Journal invalid before import')
    return plan


def execute(store, source, destination):
    source = Path(source)
    raw = source.read_bytes(); source_sha = file_hash(source)
    doc = json.loads(raw.decode('utf-8-sig'))
    plan = validate(store, doc)
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    (out/'ORIGINAL_CHECKER_DECISIONS.json').write_bytes(raw)
    before = {b: store.get(b) for b in store.frame}
    new_json(out/'PRE_IMPORT_BINDINGS.json', {b: {'version': v, 'state_sha256': digest(r), 'content_sha256': digest(reference_content(r))} for b, (r, v) in before.items()})
    backup = store.db.parent/'BACKUPS'/('BEFORE_OFFLINE_CHECKER_' + source_sha[:16] + '.sqlite3')
    backup.parent.mkdir(exist_ok=True); need(not backup.exists(), 'Backup already exists; inspect previous receipt before retrying')
    with store.connect() as c:
        accounts = [tuple(x) for x in c.execute('SELECT * FROM users ORDER BY id')]
        with closing(sqlite3.connect(backup)) as target:
            c.backup(target)
    receipt = {'started_at_utc': now(), 'original_file_sha256': source_sha,
        'source_manifest_sha256': store.config['source_manifest_sha256'],
        'private_backup_path': str(backup), 'private_backup_sha256': file_hash(backup),
        'checker_id': doc['checker_id'], 'supplied_declarations': doc['checker_declarations'],
        'identity_and_source_inspection_independently_authenticated': False,
        'provenance': 'Researcher supplied this file as the completed human checker decisions. Codex administratively records the supplied decisions and declarations; this is not a participant web login or a new AI qualitative review.',
        'applied': [], 'errors': [], 'final_reference_freeze_performed': False}
    actor = {'id': doc['checker_id'], 'role': 'checker',
        'recorded_by': 'Codex administrative offline JSON import', 'offline_decisions_sha256': source_sha}
    try:
        for item in plan:
            result = store.apply(item['business_id'], actor, item['action'], {
                'expected_version': item['version'], 'comment': item['comment'],
                'attestations': doc['checker_declarations'] if item['action'] == 'approve' else {}})
            receipt['applied'].append({**item, 'new_version': result['version'], 'new_status': result['reference']['status']})
    except Exception as error:
        receipt['errors'].append(str(error))
    new_json(out/'IMPORT_RECEIPT.json', receipt)
    for b, (previous, _) in before.items():
        current, _ = store.get(b)
        need(reference_content(current) == reference_content(previous), 'Reference prose/evidence changed')
        need(current['notes'] == previous['notes'] and current['author'] == previous['author'], 'Author or provenance notes changed')
    with store.connect() as c:
        need(accounts == [tuple(x) for x in c.execute('SELECT * FROM users ORDER BY id')], 'Accounts changed')
    audit = store.export(out/'CURRENT_WORKFLOW_EXPORT', final=False)
    receipt.update(completed_at_utc=now(), audit=audit,
        decision_counts=dict(Counter(r['decision'] for r in doc['reviews'])),
        reference_content_unchanged=True, accounts_unchanged=True, author_records_unchanged=True)
    # Replace only this newly-created running receipt with its completed form.
    (out/'IMPORT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    need(not receipt['errors'], 'Import interrupted; see saved receipt. No automatic retry.')
    need(not audit['journal_errors'] and not audit['reference_errors'], 'Post-import workflow audit failed')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination')
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args(); store = Store(args.workspace)
    if not args.commit:
        plan = validate(store, json.loads(Path(args.source).read_text(encoding='utf-8-sig')))
        print(json.dumps({'status': 'PASS', 'planned_actions': dict(Counter(x['action'] for x in plan)), 'read_only': True}))
    else:
        need(args.destination, '--destination required')
        receipt = execute(store, args.source, args.destination)
        print(json.dumps({'applied': len(receipt['applied']), 'counts': receipt['audit']['counts'], 'content_unchanged': receipt['reference_content_unchanged'], 'journal_errors': receipt['audit']['journal_errors']}, ensure_ascii=False))
