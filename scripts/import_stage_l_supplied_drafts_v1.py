"""Preserve supplied prose and import unapproved drafts through the existing journal.

No submissions, evidence links, attestations or approvals are manufactured.
Existing different records are refused; exact repeated imports are no-ops.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

from stage_l_reference_app_v1 import Store, digest, file_hash, now


ID_LINE = re.compile(r'^BUSINESS_ID:[ \t]*(\S+)[ \t]*$', re.M)
LABEL = re.compile(r'^(?:Batch\s+\d{3}\s*[–—-]\s*\d{3}|\d{3}\s*[–—-]\s+.+)[ \t]*$', re.M)
SECTIONS = re.compile(r'\s*\[REVIEW_CONSENSUS\]\s*\n(.*?)\n\[VISUAL_EVIDENCE\]\s*\n(.*?)\s*\Z', re.S)
FIELDS = ('review_consensus', 'visual_evidence')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_text(text):
    # Normalize file line endings only; preserve prose, punctuation and Unicode.
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')
    matches = list(ID_LINE.finditer(text))
    require(matches, 'No business records')
    require(not LABEL.sub('', text[:matches[0].start()]).strip(), 'Unrecognized preamble')
    records = {}
    for i, match in enumerate(matches):
        block = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        labels = list(LABEL.finditer(block))
        if labels:
            tail = block[labels[0].start():]
            require(not LABEL.sub('', tail).strip(), 'Unexpected text after case/batch label')
            block = block[:labels[0].start()]
        sections = SECTIONS.fullmatch(block)
        require(sections is not None, 'Missing/invalid sections: ' + match[1])
        values = [s.strip() for s in sections.groups()]
        require(all(values), 'Empty component: ' + match[1])
        require(all(not re.search(r'\[(?:REVIEW_CONSENSUS|VISUAL_EVIDENCE)\]', s) for s in values), 'Repeated section')
        require(match[1] not in records, 'Duplicate business: ' + match[1])
        records[match[1]] = dict(zip(FIELDS, values))
    return records


def assemble(paths, ordered_ids, patch_ids):
    sets = [parse_text(p.read_text(encoding='utf-8-sig')) for p in paths]
    require(list(sets[0]) == ordered_ids[:50], 'First 50 cohort/order mismatch')
    require(list(sets[1]) == ordered_ids[50:], 'Last 50 cohort/order mismatch')
    require(set(sets[2]) == set(patch_ids), 'Seven-replacement identity mismatch')
    merged = {**sets[0], **sets[1]}
    merged.update(sets[2])
    require(set(merged) == set(ordered_ids) and len(merged) == 100, 'Complete unique cohort required')
    origin = {bid: (2 if bid in sets[2] else 0 if bid in sets[0] else 1) for bid in ordered_ids}
    return merged, origin


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def content_equal(reference, record):
    return all(reference[k] == record[k] for k in FIELDS)


def import_records(store, records, author, manifest_sha256):
    """Validate the whole cohort before writes; preserve each existing save transaction."""
    with store.connect() as conn:
        account = conn.execute('SELECT id,role FROM users WHERE id=?', (author,)).fetchone()
    require(account is not None and account['role'] == 'author', 'Existing author account required')
    pending = []
    skipped = []
    for record in records:
        bid = record['business_id']
        require(record['source_bundle_sha256'] == store.frame[bid]['sha256'], 'Source binding mismatch')
        current, version = store.get(bid)
        if version:
            require(content_equal(current, record), 'Existing different annotation; refusing overwrite: ' + bid)
            skipped.append(bid)
        else:
            require(current['status'] == 'NOT_STARTED', 'Unexpected initial state')
            pending.append((record, version))
    require(not store.audit()['journal_errors'], 'Existing journal corruption')
    imported = []
    for record, version in pending:
        provenance = record['provenance']
        note = ('Administrative import of researcher-supplied draft; prose copied verbatim. '
                'Imported by Codex on researcher instruction, assigned to ' + author + '. '
                'Not a manual UI save or certification of independent human authorship. '
                'Source file: ' + provenance['submission_file'] + '; SHA256: ' + provenance['submission_sha256'] + '. '
                'Intake manifest SHA256: ' + manifest_sha256 + '. '
                'Evidence links, author confirmations, submission and independent checker approval remain pending.')
        draft = {k: record[k] for k in FIELDS}
        draft.update(review_claims=[], visual_claims=[], notes=note)
        result = store.apply(record['business_id'], {'id': author, 'role': 'author'}, 'save', {
            'expected_version': version, 'draft': draft,
            'comment': 'Researcher-authorized administrative import of supplied prose; no human attestations or approval.',
        })
        r = result['reference']
        require(content_equal(r, record) and r['status'] == 'DRAFT', 'Save did not preserve draft')
        require(not r['author']['human_written_attestation'] and not r['checker']['independent_check_attestation'], 'Unexpected attestation')
        imported.append(record['business_id'])
    return {'imported': len(imported), 'existing_identical_skipped': len(skipped), 'imported_business_ids': imported}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--first50', required=True, type=Path)
    parser.add_argument('--last50', required=True, type=Path)
    parser.add_argument('--replacements', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--author', default='author01')
    parser.add_argument('--commit-drafts', action='store_true')
    args = parser.parse_args()
    require((args.workspace/'PRIVATE_STATE/references.sqlite3').is_file(), 'Existing annotation DB required')
    store = Store(args.workspace)
    ids = list(store.frame)
    patches = [ids[i - 1] for i in (7, 9, 14, 18, 56, 71, 72)]
    paths = [args.first50, args.last50, args.replacements]
    merged, origins = assemble(paths, ids, patches)
    names = ['01_FIRST_50.txt', '02_LAST_50.txt', '03_SEVEN_CORRECTIONS.txt']
    receipts = [{'file': 'ORIGINAL_SUBMISSIONS/' + name, 'sha256': file_hash(path),
                 'original_path': str(path.resolve())} for path, name in zip(paths, names)]
    records = []
    for index, bid in enumerate(ids, 1):
        source = receipts[origins[bid]]
        records.append({'index': index, 'business_id': bid, 'source_bundle_sha256': store.frame[bid]['sha256'],
                        **merged[bid], 'status': 'DRAFT', 'review_claims': [], 'visual_claims': [],
                        'provenance': {'submission_file': source['file'], 'submission_sha256': source['sha256'],
                                       'author_claim': 'Researcher states these are human-written drafts.',
                                       'model_assisted_feedback': True, 'independently_approved': False}})
    dest = args.destination
    if dest.exists():
        require((dest/'DRAFTS_100.jsonl').is_file(), 'Existing incomplete destination; preserve and inspect')
        saved = [json.loads(line) for line in (dest/'DRAFTS_100.jsonl').read_text(encoding='utf-8').splitlines()]
        require(saved == records, 'Existing package differs; refusing overwrite')
        manifest = json.loads((dest/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
        require(all(file_hash(dest/p) == h for p, h in manifest['files'].items()), 'Package member changed')
    else:
        (dest/'ORIGINAL_SUBMISSIONS').mkdir(parents=True)
        (dest/'BATCHES').mkdir()
        for path, name in zip(paths, names):
            shutil.copyfile(path, dest/'ORIGINAL_SUBMISSIONS'/name)
        lines = [json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in records]
        (dest/'DRAFTS_100.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        batches = []
        for start in range(0, 100, 5):
            text = f'Batch {start + 1:03d}–{start + 5:03d}\n\n' + '\n\n'.join(
                f'BUSINESS_ID: {r["business_id"]}\n[REVIEW_CONSENSUS]\n{r["review_consensus"]}\n\n'
                f'[VISUAL_EVIDENCE]\n{r["visual_evidence"]}' for r in records[start:start + 5]) + '\n'
            (dest/'BATCHES'/f'BATCH_{start + 1:03d}_{start + 5:03d}.txt').write_text(text, encoding='utf-8')
            batches.append(text)
        (dest/'REFERENCES_100_READABLE.txt').write_text('\n'.join(batches), encoding='utf-8')
        reread = parse_text((dest/'REFERENCES_100_READABLE.txt').read_text(encoding='utf-8'))
        require(reread == merged, 'Readable export does not preserve supplied prose')
        report = {'records': 100, 'unique_businesses': len(set(ids)), 'complete_components': 200,
                  'replacement_business_ids': patches, 'exact_component_text_preserved': True,
                  'original_submission_bytes_preserved': all(file_hash(dest/r['file']) == r['sha256'] for r in receipts),
                  'matches_frozen_cohort_and_order': True, 'source_manifest_sha256': store.config['source_manifest_sha256'],
                  'review_word_range': [min(len(r['review_consensus'].split()) for r in records), max(len(r['review_consensus'].split()) for r in records)],
                  'visual_word_range': [min(len(r['visual_evidence'].split()) for r in records), max(len(r['visual_evidence'].split()) for r in records)],
                  'semantic_factuality_certified': False, 'submission_and_approval': 'PENDING_REAL_PARTICIPANTS',
                  'created_at_utc': now(), 'submission_files': receipts}
        json_write(dest/'INTEGRITY_REPORT.json', report)
        json_write(dest/'PACKAGE_MANIFEST.json', {'status': 'UNAPPROVED_DRAFT_INTAKE',
                   'files': {str(p.relative_to(dest)).replace('\\', '/'): file_hash(p) for p in sorted(dest.rglob('*')) if p.is_file()}})
    print(json.dumps({'package': str(dest), 'dataset_sha256': file_hash(dest/'DRAFTS_100.jsonl'),
                      'manifest_sha256': file_hash(dest/'PACKAGE_MANIFEST.json'), 'records': 100}))
    if not args.commit_drafts:
        return
    # A private consistent backup includes credential hashes and must stay out of review archives.
    backup = store.root/'PRIVATE_STATE/BACKUPS'/('BEFORE_USER_DRAFT_IMPORT_' + digest(records)[:16] + '.sqlite3')
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        with store.connect() as source, sqlite3.connect(backup) as target:
            source.backup(target)
    with store.connect() as conn:
        users_before = [tuple(r) for r in conn.execute('SELECT * FROM users ORDER BY id')]
    result = import_records(store, records, args.author, file_hash(dest/'PACKAGE_MANIFEST.json'))
    audit = store.audit()
    require(not audit['journal_errors'] and not audit['duplicate_reference_pairs'], 'Post-import integrity failure')
    for record in records:
        require(content_equal(store.get(record['business_id'])[0], record), 'Post-import prose mismatch')
    with store.connect() as conn:
        require([tuple(r) for r in conn.execute('SELECT * FROM users ORDER BY id')] == users_before, 'Accounts changed')
    result.update(checked_at_utc=now(), backup_path=str(backup), backup_sha256=file_hash(backup),
                  audit=audit, accounts_unchanged=True, all_200_components_exact=True,
                  no_attestations_or_submissions_created=True)
    # Retain every execution receipt rather than replacing a prior receipt on repeat runs.
    receipt = dest/('IMPORT_RECEIPT_' + str(result['imported']) + '_' + digest(result)[:12] + '.json')
    json_write(receipt, result)
    print(json.dumps({'receipt': str(receipt), 'imported': result['imported'], 'skipped': result['existing_identical_skipped'],
                      'counts': audit['counts'], 'journal_errors': audit['journal_errors']}))


if __name__ == '__main__':
    main()
