"""Pure-stdlib D-053 source-reference annotation validation.

This checks evidence mappings and explicit provenance, not whether a researcher's
semantic judgment is true. It never labels, rewrites, or accepts a claim.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

LABELS = ('supported', 'unsupported', 'contradicted', 'uncertain', 'needs_revision')
IMPORTANCE = ('core', 'supporting')
INFORMATIVENESS = ('useful_specific', 'useful_general', 'low_information')
REVISION_DISPOSITIONS = (None, 'pending_unresolved', 'resolved_with_confirmed_units', 'excluded_from_reference')
BINDING_KEYS = ('case_inputs_sha256', 'protocol_sha256', 'candidate_sha256',
                'source_cases_sha256', 'evaluation_config_sha256')
PROVENANCE_NOTICE = ('Explicit confirmation fields record researcher decisions; '
                     'software cannot independently establish the truth of a human-verification flag.')


def _nonblank(x):
    return isinstance(x, str) and bool(x.strip())


def _timestamp(x):
    if not _nonblank(x):
        return False
    try:
        value = dt.datetime.fromisoformat(x.replace('Z', '+00:00'))
        return value.tzinfo is not None
    except ValueError:
        return False


def make_initial_state(cases, candidates, bindings, bundle_id='D053_REFERENCE_VERIFICATION_V1', reviewer_id=''):
    """Create an entirely unverified working sheet; no candidate is accepted."""
    return {
        'schema_version': 1, 'stage': 'D-053', 'bundle_id': bundle_id,
        'bindings': copy.deepcopy(bindings), 'reviewer_id': reviewer_id,
        'candidate_annotations': {c['unit_id']: {
            'support_label': None, 'human_verified': False, 'confirmed_by': None,
            'confirmed_at': None, 'importance': None, 'informativeness': None,
            'reviewer_notes': '', 'corrected_claim_text': '', 'final_unit_ids': [],
            'revision_disposition': None,
        } for c in candidates},
        'units': [],
        'business_reviews': {c['business_id']: {
            'human_verified': False, 'positive_negative_and_specificity_checked': False,
            'confirmed_by': None, 'confirmed_at': None, 'notes': '',
            'missing_information_notes': '',
        } for c in cases},
        'researcher_confirmed_complete': False, 'history': [],
    }


def _validate_annotations(state, cases, candidates, bindings, final=False):
    """Validate a draft or final sheet without changing it.

    Drafts may be incomplete. Malformed/foreign evidence, binding mismatches and
    false-shaped confirmation values are errors even in drafts. Final readiness
    is always computed; final=True promotes incompleteness to errors.
    """
    errors, pending, warnings = [], [], [PROVENANCE_NOTICE]

    def check(ok, message):
        if not ok:
            errors.append(message)

    def need(ok, message):
        if not ok:
            pending.append(message)

    if not isinstance(state, dict):
        return {'valid': False, 'ready_for_freeze': False, 'errors': ['State must be an object'],
                'warnings': warnings, 'pending': [], 'counts': {}}
    check(type(state.get('schema_version')) is int and state.get('schema_version') == 1, 'schema_version must be integer 1')
    check(state.get('stage') == 'D-053', 'stage must be D-053')
    check(_nonblank(state.get('bundle_id')), 'bundle_id is missing')
    check(isinstance(state.get('reviewer_id'), str), 'reviewer_id must be a string')
    need(_nonblank(state.get('reviewer_id')), 'Researcher identity is not set')
    check(isinstance(state.get('bindings'), dict) and state.get('bindings') == bindings, 'Frozen input/config bindings mismatch')
    for key in BINDING_KEYS:
        check(isinstance(bindings.get(key), str) and re.fullmatch('[0-9a-f]{64}', bindings.get(key, '')) is not None,
              'Invalid expected SHA-256 binding: ' + key)
    check(type(state.get('researcher_confirmed_complete')) is bool, 'researcher_confirmed_complete must be boolean')
    need(state.get('researcher_confirmed_complete') is True, 'Global researcher confirmation is pending')
    check(isinstance(state.get('history'), list), 'history must be an array')
    confirmation_history = {}
    for event in state.get('history', []) if isinstance(state.get('history'), list) else []:
        if not isinstance(event, dict):
            errors.append('history entries must be objects')
            continue
        if event.get('kind') in ('candidate_confirmed', 'unit_confirmed', 'business_confirmed'):
            check(_nonblank(event.get('entity_id')), 'Confirmation history entity_id missing')
            check(_nonblank(event.get('confirmed_by')) and _timestamp(event.get('confirmed_at')),
                  'Confirmation history provenance malformed')
            check(isinstance(event.get('snapshot'), dict), 'Confirmation history snapshot missing')
            confirmation_history[(event.get('kind'), event.get('entity_id'))] = event

    def confirmation(kind, entity, record):
        event = confirmation_history.get((kind, entity))
        check(isinstance(event, dict) and event.get('snapshot') == record
              and event.get('confirmed_by') == record.get('confirmed_by')
              and event.get('confirmed_at') == record.get('confirmed_at'),
              entity + ': verified record changed without matching explicit confirmation snapshot')

    bycase = {c['business_id']: c for c in cases}
    bycandidate = {c['unit_id']: c for c in candidates}
    check(len(bycase) == len(cases), 'Source package contains duplicate business IDs')
    check(len(bycandidate) == len(candidates), 'Source package contains duplicate candidate IDs')
    for bid, case in bycase.items():
        check(case.get('split') == 'validation' and case.get('condition') == 'filtered', bid + ': only the frozen validation Filtered sources are allowed')
        if case.get('source_sha256') is not None:
            check(isinstance(case.get('source'), str) and hashlib.sha256(case['source'].encode('utf-8')).hexdigest() == case['source_sha256'],
                  bid + ': source text/hash mismatch')
        reviews, captions = case.get('reviews', []), case.get('captions', [])
        check(len({r['marker'] for r in reviews}) == len(reviews), bid + ': duplicate REV markers')
        check(len({c['caption_id'] for c in captions}) == len(captions), bid + ': duplicate caption markers')
        if 'selected_review_ids' in case:
            check([r['review_id'] for r in reviews] == case['selected_review_ids'], bid + ': review IDs/order mismatch')
        if 'selected_photo_ids' in case:
            check([c['photo_id'] for c in captions] == case['selected_photo_ids'], bid + ': photo IDs/order mismatch')
    for cid, candidate in bycandidate.items():
        check(candidate.get('business_id') in bycase, 'Unknown source candidate business: ' + cid)
        check(candidate.get('domain') in ('review', 'visual'), 'Unknown source candidate domain: ' + cid)

    annotations = state.get('candidate_annotations')
    businesses = state.get('business_reviews')
    units = state.get('units')
    check(isinstance(annotations, dict), 'candidate_annotations must be an object')
    check(isinstance(businesses, dict), 'business_reviews must be an object')
    check(isinstance(units, list), 'units must be an array')
    if not isinstance(annotations, dict):
        annotations = {}
    if not isinstance(businesses, dict):
        businesses = {}
    if not isinstance(units, list):
        units = []
    check(set(annotations) == set(bycandidate), 'Candidate ID set differs from the frozen queue')
    check(set(businesses) == set(bycase), 'Business ID set differs from the frozen 24-case package')

    byunit, unit_ready = {}, {}
    for index, unit in enumerate(units):
        where = 'unit[' + str(index) + ']'
        if not isinstance(unit, dict):
            errors.append(where + ' must be an object')
            continue
        uid = unit.get('unit_id')
        check(_nonblank(uid), where + ' needs a unique unit_id')
        if not _nonblank(uid):
            uid = where
        check(uid not in byunit, 'Duplicate unit ID: ' + uid)
        byunit[uid] = unit
        before = len(pending)
        errors_before = len(errors)
        bid, domain = unit.get('business_id'), unit.get('domain')
        check(bid in bycase, uid + ': unknown business')
        check(domain in ('review', 'visual'), uid + ': invalid evidence domain')
        if bid not in bycase or domain not in ('review', 'visual'):
            continue
        need(_nonblank(unit.get('text')), uid + ': empty claim text')
        check(unit.get('support_status') in (None, 'supported'), uid + ': only separately supported units may enter units')
        need(unit.get('support_status') == 'supported', uid + ': supported decision pending')
        for flag in ('human_verified', 'atomicity_confirmed', 'evidence_confirmed'):
            check(type(unit.get(flag, False)) is bool, uid + ': ' + flag + ' must be boolean')
            need(unit.get(flag) is True, uid + ': ' + flag + ' is pending')
        if unit.get('human_verified') is True:
            confirmation('unit_confirmed', uid, unit)
        need(_nonblank(unit.get('confirmed_by')), uid + ': confirmer missing')
        need(_timestamp(unit.get('confirmed_at')), uid + ': timezone-aware confirmation timestamp missing')
        check(unit.get('importance') in (None,) + IMPORTANCE, uid + ': invalid importance')
        need(unit.get('importance') in IMPORTANCE, uid + ': importance pending')
        check(unit.get('informativeness') in (None,) + INFORMATIVENESS, uid + ': invalid informativeness')
        need(unit.get('informativeness') in INFORMATIVENESS, uid + ': informativeness pending')
        need(_nonblank(unit.get('aspect')), uid + ': aspect pending')
        check(unit.get('restaurant_specific') is None or type(unit.get('restaurant_specific')) is bool,
              uid + ': restaurant_specific must be boolean or null')
        need(type(unit.get('restaurant_specific')) is bool, uid + ': specificity pending')
        if domain == 'review':
            check(unit.get('polarity') in (None, 'positive', 'negative', 'mixed', 'neutral'), uid + ': invalid review polarity')
            need(unit.get('polarity') in ('positive', 'negative', 'mixed', 'neutral'), uid + ': review polarity pending')
        else:
            check(unit.get('polarity') is None, uid + ': visual units must not carry review-opinion polarity')
            check(type(unit.get('visual_photo_confirmed', False)) is bool, uid + ': visual_photo_confirmed must be boolean')
            need(unit.get('visual_photo_confirmed') is True, uid + ': photo corroboration pending')
            check(bool(bycase[bid].get('captions')), uid + ': visual unit forbidden for zero-caption business')
        provenance = unit.get('provenance')
        need(isinstance(provenance, dict), uid + ': provenance pending')
        parent = unit.get('parent_candidate_id')
        if isinstance(provenance, dict):
            check(provenance.get('origin') in ('candidate_split', 'researcher_added'), uid + ': invalid unit origin')
            if parent is not None and provenance.get('parent_candidate_id') != parent:
                errors.append(uid + ': conflicting parent candidate IDs')
            parent = provenance.get('parent_candidate_id')
            if provenance.get('origin') == 'candidate_split':
                check(parent in bycandidate, uid + ': missing/unknown parent candidate')
            if provenance.get('origin') == 'researcher_added':
                check(parent is None, uid + ': researcher-added unit cannot impersonate a candidate split')
        if parent is not None:
            check(parent in bycandidate, uid + ': unknown parent candidate')
            if parent in bycandidate:
                check(bycandidate[parent]['business_id'] == bid and bycandidate[parent]['domain'] == domain,
                      uid + ': candidate business/domain mismatch')
        evidence = unit.get('evidence')
        check(evidence is None or isinstance(evidence, list), uid + ': evidence must be an array')
        need(isinstance(evidence, list) and bool(evidence), uid + ': evidence pending')
        if isinstance(evidence, list):
            mapping = {r['marker']: r for r in bycase[bid]['reviews']} if domain == 'review' else {
                c['caption_id']: c for c in bycase[bid]['captions']}
            seen_evidence = set()
            for e in evidence:
                if not isinstance(e, dict):
                    errors.append(uid + ': evidence entry must be an object')
                    continue
                marker = e.get('marker')
                check(marker in mapping, uid + ': wrong evidence domain or unknown marker')
                quote = e.get('quote')
                check(_nonblank(quote), uid + ': evidence quote must not be blank')
                if marker not in mapping:
                    continue
                original = mapping[marker]
                idfield = 'review_id' if domain == 'review' else 'photo_id'
                foreign = 'photo_id' if domain == 'review' else 'review_id'
                check(e.get(idfield) == original[idfield], uid + ': original evidence ID mismatch')
                check(foreign not in e or e[foreign] is None, uid + ': cross-domain evidence ID')
                check(_nonblank(quote) and quote in original['text'], uid + ': quote is not an exact substring of permitted evidence')
                signature = (marker, e.get(idfield), quote)
                check(signature not in seen_evidence, uid + ': duplicate evidence entry')
                seen_evidence.add(signature)
        unit_ready[uid] = before == len(pending) and errors_before == len(errors)

    verified_candidates, label_counts = 0, Counter()
    linked = Counter()
    for cid, annotation in annotations.items():
        if cid not in bycandidate:
            continue
        if not isinstance(annotation, dict):
            errors.append(cid + ': annotation must be an object')
            continue
        label = annotation.get('support_label')
        check(label in (None,) + LABELS, cid + ': invalid support label')
        check(type(annotation.get('human_verified')) is bool, cid + ': human_verified must be boolean')
        check(annotation.get('importance') in (None,) + IMPORTANCE, cid + ': invalid importance')
        check(annotation.get('informativeness') in (None,) + INFORMATIVENESS, cid + ': invalid informativeness')
        for key in ('reviewer_notes', 'corrected_claim_text'):
            check(isinstance(annotation.get(key), str), cid + ': ' + key + ' must be a string')
        disposition = annotation.get('revision_disposition')
        check(disposition in REVISION_DISPOSITIONS, cid + ': invalid revision_disposition')
        ids = annotation.get('final_unit_ids')
        check(isinstance(ids, list) and all(_nonblank(u) for u in ids), cid + ': final_unit_ids must be an ID array')
        if not isinstance(ids, list):
            ids = []
        check(len(set(ids)) == len(ids), cid + ': duplicate final_unit_ids')
        for uid in ids:
            check(uid in byunit, cid + ': unknown final unit ID ' + str(uid))
            if uid in byunit:
                u = byunit[uid]
                parent = u.get('provenance', {}).get('parent_candidate_id') if isinstance(u.get('provenance'), dict) else None
                check(parent == cid, cid + ': final unit lacks matching parent provenance')
                check(u.get('business_id') == bycandidate[cid]['business_id'] and u.get('domain') == bycandidate[cid]['domain'],
                      cid + ': final unit business/domain mismatch')
                need(unit_ready.get(uid, False), cid + ': linked final unit is not independently confirmed')
                linked[uid] += 1
        confirmed = annotation.get('human_verified') is True
        if confirmed:
            verified_candidates += 1
            check(label in LABELS, cid + ': confirmed candidate needs a label')
            check(_nonblank(annotation.get('confirmed_by')), cid + ': confirmed candidate needs confirmer')
            check(_timestamp(annotation.get('confirmed_at')), cid + ': confirmed candidate needs timezone-aware timestamp')
            confirmation('candidate_confirmed', cid, annotation)
        need(confirmed, cid + ': researcher decision pending')
        # Candidate wording may contain multiple propositions. Descriptive
        # priorities here remain optional; confirmed atomic units carry the
        # required importance/informativeness used by evaluation denominators.
        if label in LABELS:
            label_counts[label] += 1
        if label == 'supported':
            need(bool(ids), cid + ': supported candidate needs separately confirmed atomic unit(s)')
        elif label == 'needs_revision':
            need(_nonblank(annotation.get('corrected_claim_text')), cid + ': corrected wording pending')
            need(disposition in ('resolved_with_confirmed_units', 'excluded_from_reference'), cid + ': revision remains unresolved')
            if disposition == 'resolved_with_confirmed_units':
                need(bool(ids), cid + ': resolved revision needs separately confirmed atomic unit(s)')
            elif disposition == 'excluded_from_reference':
                need(_nonblank(annotation.get('reviewer_notes')), cid + ': revision exclusion reason pending')
                check(not ids, cid + ': excluded revision must not link final reference units')
        else:
            message = cid + ': unsupported/contradicted/uncertain/pending candidate must not populate final units'
            if confirmed:
                check(not ids, message)
            else:
                # Preserve separately confirmed units/history while a researcher
                # edits a previous decision. This transition is a valid draft,
                # never an admissible reference freeze or confirmed exclusion.
                need(not ids, message + ' (unconfirmed transition pending)')
            if label in ('unsupported', 'contradicted', 'uncertain'):
                need(_nonblank(annotation.get('reviewer_notes')), cid + ': exclusion/uncertainty reason pending')
        if label != 'needs_revision':
            check(disposition is None, cid + ': revision disposition requires needs_revision label')
    for uid, u in byunit.items():
        p = u.get('provenance')
        if isinstance(p, dict) and p.get('origin') == 'candidate_split':
            check(linked[uid] == 1, uid + ': candidate-derived unit must be linked exactly once')

    verified_businesses = 0
    for bid, record in businesses.items():
        if bid not in bycase:
            continue
        if not isinstance(record, dict):
            errors.append(bid + ': business review must be an object')
            continue
        for flag in ('human_verified', 'positive_negative_and_specificity_checked'):
            check(type(record.get(flag)) is bool, bid + ': ' + flag + ' must be boolean')
            need(record.get(flag) is True, bid + ': ' + flag + ' pending')
        for key in ('notes', 'missing_information_notes'):
            check(isinstance(record.get(key), str), bid + ': ' + key + ' must be a string')
        if record.get('human_verified') is True:
            verified_businesses += 1
            check(_nonblank(record.get('confirmed_by')) and _timestamp(record.get('confirmed_at')), bid + ': business confirmer/time missing')
            confirmation('business_confirmed', bid, record)
        need(any(u.get('business_id') == bid and u.get('domain') == 'review' and unit_ready.get(uid) for uid, u in byunit.items()),
             bid + ': at least one confirmed review reference unit required')
    ready = not errors and not pending
    final_errors = errors + pending if final else errors
    return {'valid': not final_errors, 'ready_for_freeze': ready, 'errors': final_errors,
            'warnings': warnings, 'pending': pending if not final else [],
            'counts': {'businesses': len(bycase), 'candidates': len(bycandidate),
                       'verified_candidates': verified_candidates,
                       'pending_candidates': len(bycandidate) - verified_candidates,
                       'candidate_labels': dict(label_counts), 'units': len(units),
                       'confirmed_units': sum(unit_ready.values()),
                       'verified_businesses': verified_businesses}}


def validate_annotations(state, cases, candidates, bindings, final=False):
    """Reject malformed JSON-shaped input cleanly as well as incomplete sheets."""
    try:
        return _validate_annotations(state, cases, candidates, bindings, final)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return {'valid': False, 'ready_for_freeze': False,
                'errors': ['Malformed annotation/evidence structure: ' + str(exc)],
                'warnings': [PROVENANCE_NOTICE], 'pending': [], 'counts': {}}


def reference_freeze(state, cases, candidates, bindings):
    """Return a D-049-compatible immutable export only after explicit completion."""
    result = validate_annotations(state, cases, candidates, bindings, final=True)
    if not result['valid']:
        raise ValueError('Reference freeze blocked: ' + '; '.join(result['errors']))
    frozen = copy.deepcopy(state)
    frozen.update({key: bindings[key] for key in BINDING_KEYS})
    frozen['status'] = 'RESEARCHER_CONFIRMED_SOURCE_REFERENCES_D053'
    frozen['provenance_notice'] = PROVENANCE_NOTICE
    frozen['exclusion_log'] = [dict(candidate_id=c['unit_id'], business_id=c['business_id'], domain=c['domain'],
        original_text=c['text'], **copy.deepcopy(state['candidate_annotations'][c['unit_id']]))
        for c in candidates if state['candidate_annotations'][c['unit_id']]['support_label'] != 'supported']
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, help='D053 self-contained package directory')
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--output', type=Path, help='Write validation report (not annotations)')
    parser.add_argument('--export-reference', '--freeze-out', dest='freeze_out', type=Path)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    if args.package:
        args.cases = args.cases or args.package / 'source_cases.json'
        args.candidates = args.candidates or args.package / 'candidates_UNVERIFIED.jsonl'
        args.bindings = args.bindings or args.package / 'bindings.json'
    if not all((args.cases, args.candidates, args.bindings)):
        parser.error('Use --package, or supply --cases, --candidates and --bindings')
    state, cases, binding_document = read(args.annotations), read(args.cases), read(args.bindings)
    bindings = binding_document.get('bindings', binding_document)
    expected_bundle_id = binding_document.get('bundle_id')
    package_paths = [(args.cases, 'source_cases_sha256'), (args.candidates, 'candidate_sha256')]
    if args.package:
        package_paths.extend(((args.package / 'D049_FROZEN_EVALUATION_PROTOCOL.md', 'protocol_sha256'),
                              (args.package / 'FROZEN_EVALUATION_CONFIG.json', 'evaluation_config_sha256')))
    for path, key in package_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != bindings.get(key):
            raise SystemExit('Package source file hash mismatch: ' + str(path))
    if expected_bundle_id and hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest() != expected_bundle_id:
        raise SystemExit('Package bundle_id does not match canonical binding digest')
    if expected_bundle_id and state.get('bundle_id') != expected_bundle_id:
        raise SystemExit('Annotation bundle_id differs from package bindings')
    raw = args.candidates.read_text(encoding='utf-8-sig')
    candidates = json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(x) for x in raw.splitlines() if x.strip()]
    result = validate_annotations(state, cases, candidates, bindings, args.final or bool(args.freeze_out))
    if args.freeze_out:
        if args.freeze_out.exists():
            raise SystemExit('Refusing to overwrite an existing reference freeze')
        value = reference_freeze(state, cases, candidates, bindings)
        args.freeze_out.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(serialized)
    print(serialized, end='')
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
