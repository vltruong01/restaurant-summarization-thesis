"""Administrative submission on an explicit reported author confirmation, never approval."""
import argparse
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from stage_l_reference_app_v1 import Store, ATTESTATIONS, digest, file_hash, new_json, now, need

MUNGO = '-78Va3llKcPbL6SW4is8lw'
COMPONENTS = ['review_consensus', 'visual_evidence']


def prepare(store, bid, actor_id, confirmation):
    r, version = store.get(bid)
    if r['status'] != 'DRAFT':
        return None
    need(r['author']['human_id'] == actor_id, 'Different assigned author')
    draft = {k: copy.deepcopy(r[k]) for k in COMPONENTS + ['review_claims', 'visual_claims', 'notes']}
    additions = []
    # The current Mungo revision replaced all old output spans. Preserve prose and
    # attach two narrow existing-source pointers, not an exhaustive claim audit.
    if bid == MUNGO and not draft['review_claims'] and not draft['visual_claims']:
        span = 'missing takeout sides'
        rid = 'RBqFxdDkL67eY28m9sEt_Q'
        quote = 'For example our side dishes, we only received one.'
        need(span in draft['review_consensus'], 'Current Mungo Review revision differs')
        need(any(x['review_id'] == rid and quote in x['text'] for x in store.source(bid)['reviews']), 'Source quote mismatch')
        photo = 'LqMIkE4iV1egsOM4VP6QIw'
        visual_span = 'a layered cake slice'
        need(visual_span in draft['visual_evidence'], 'Current Mungo Visual revision differs')
        need((bid, photo) in store.photo_index, 'Photo binding mismatch')
        note = 'Administratively linked to the current exact span; limited source pointer, not exhaustive semantic or coverage certification. Checker must review the full text and sources.'
        draft['review_claims'] = [{'output_span': span, 'evidence': [{'review_id': rid, 'quote': quote, 'note': note}]}]
        draft['visual_claims'] = [{'output_span': visual_span, 'evidence': [{'photo_id': photo, 'direct_image_viewed': False, 'note': note}]}]
        additions = ['Review: ' + span, 'Visual: ' + visual_span]
    need(all(draft[k].strip() for k in COMPONENTS), 'Empty component')
    need(draft['review_claims'] and draft['visual_claims'], 'Missing component evidence')
    for claim in draft['visual_claims']:
        for link in claim['evidence']:
            link['direct_image_viewed'] = True
    draft['notes'] += '\n\nAdministrative batch submission on researcher-reported author confirmation. Confirmation receipt SHA256: ' + digest(confirmation) + '. Author/source/blindness declarations are recorded from that report, not independently observed by Codex. Existing AI-assisted evidence provenance remains; checker approval and complete source/coverage review are still pending.'
    probe = copy.deepcopy(r); probe.update(draft)
    need(not store.errors(probe), 'Submission invalid: ' + str(store.errors(probe)))
    return {'business_id': bid, 'expected_version': version, 'draft': draft, 'source_pointers_added': additions}


def execute(store, confirmation, destination):
    need(confirmation['author_id'] == 'author01', 'Unexpected author')
    need(confirmation['scope'] == 'all_100_current_drafts', 'Unexpected confirmation scope')
    need(confirmation['author_attestations_reported'] == {k: True for k in ATTESTATIONS}, 'Explicit reported declarations required')
    need(confirmation['statement_verbatim'].strip(), 'Confirmation source required')
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    new_json(out/'REPORTED_AUTHOR_CONFIRMATION.json', confirmation)
    before = {b: store.get(b) for b in store.frame}
    prepared = [p for b in store.frame if (p := prepare(store, b, confirmation['author_id'], confirmation))]
    need(not store.audit()['journal_errors'], 'Journal invalid before submission')
    new_json(out/'PRE_SUBMISSION_BINDINGS.json', {b: {'version': v, 'reference_sha256': digest(r), 'prose_sha256': {k: digest(r[k]) for k in COMPONENTS}} for b, (r, v) in before.items()})
    backup = store.db.parent/'BACKUPS'/('BEFORE_BATCH_SUBMISSION_' + out.name + '.sqlite3')
    backup.parent.mkdir(exist_ok=True)
    need(not backup.exists(), 'Backup already exists')
    with store.connect() as src:
        users = [tuple(x) for x in src.execute('SELECT * FROM users ORDER BY id')]
        with closing(sqlite3.connect(backup)) as target:
            src.backup(target)
    receipt = {'started_at_utc': now(), 'confirmation_sha256': digest(confirmation), 'backup_path': str(backup), 'backup_sha256': file_hash(backup), 'submitted': [], 'errors': [], 'checker_actions_performed': False, 'independently_authenticated_human_attestations': False}
    actor = {'id': 'author01', 'role': 'author', 'recorded_by': 'Codex administrative batch', 'confirmation_source': 'researcher-reported author confirmation', 'confirmation_sha256': digest(confirmation)}
    try:
        for p in prepared:
            result = store.apply(p['business_id'], actor, 'submit', {'expected_version': p['expected_version'], 'draft': p['draft'], 'attestations': confirmation['author_attestations_reported'], 'comment': 'Researcher explicitly requested batch submission and reported all author declarations accepted; recorded administratively, not a manual participant login or independent authorship/inspection verification. Confirmation SHA256: ' + digest(confirmation)})
            receipt['submitted'].append({'business_id': p['business_id'], 'version': result['version'], 'source_pointers_added': p['source_pointers_added']})
    except Exception as error:
        receipt['errors'].append(str(error))
    for b, (r, _) in before.items():
        current, _ = store.get(b)
        need(all(current[k] == r[k] for k in COMPONENTS), 'Prose changed')
        need(current['checker'] == r['checker'], 'Checker record changed')
    with store.connect() as c:
        need(users == [tuple(x) for x in c.execute('SELECT * FROM users ORDER BY id')], 'Accounts changed')
    audit = store.export(out/'CURRENT_WORKFLOW_EXPORT', final=False)
    receipt.update(completed_at_utc=now(), audit=audit, all_prose_preserved=True, accounts_unchanged=True, checker_records_unchanged=True)
    new_json(out/'BATCH_SUBMISSION_RECEIPT.json', receipt)
    need(not receipt['errors'], 'Batch interrupted; see receipt, no retry performed')
    need(not audit['journal_errors'] and not audit['reference_errors'], 'Post-submission audit failed')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--confirmation', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    need(args.commit, 'Explicit --commit required')
    result = execute(Store(args.workspace), json.loads(Path(args.confirmation).read_text(encoding='utf8')), args.destination)
    print(json.dumps({'submitted_now': len(result['submitted']), 'counts': result['audit']['counts'], 'journal_errors': result['audit']['journal_errors'], 'prose_preserved': result['all_prose_preserved']}, ensure_ascii=False))
