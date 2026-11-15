"""Shuffled constraint engine. Production cuisine eligibility is NOT supplied.

No labels/targets/photos or held-out records are consumed. Fixture edges are
explicit test data, never a scientific donor-selection heuristic.
"""
import hashlib
import json
import random


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def partition(source):
    marker = '[VISUAL_CONTEXT]'
    require(source.count(marker) == 1, 'Exactly one visual-context boundary required')
    head, captions = source.split(marker)
    return head + marker, captions


def build_mapping(rows, approved_edges, tokenizer, seed=42, *, fixture_only=True, policy_approved=False):
    require(fixture_only or policy_approved, 'Production BLOCKED: cuisine/fallback eligibility not approved')
    require(rows and all(r['split'] in ('train', 'validation') for r in rows),
            'This offline implementation accepts TRAIN/validation fixtures only')
    table = {r['business_id']: r for r in rows}
    require(len(table) == len(rows), 'Duplicate business identity')
    rng = random.Random(seed)
    result = []
    for bid in sorted(table):
        r = table[bid]
        head, own = partition(r['source'])
        count = len(r['selected_photo_ids'])
        require(own.count('[IMG_CAP_') == count, 'Recipient caption count inconsistent')
        if count == 0:
            require(not own.strip(), 'Zero-caption source contains visual text')
            result.append(dict(recipient_id=bid, donor_id=None, split=r['split'], caption_count=0,
                               source=r['source'], source_sha256=digest(r['source']),
                               review_prefix_sha256=digest(head), donor_caption_sha256=digest(own),
                               input_ids=tokenizer.encode(r['source']).ids))
            continue
        require(bid in approved_edges, 'Missing explicit eligibility decision')
        options = sorted(approved_edges[bid])
        require(options, 'No approved donor; no automatic fallback')
        # The engine never silently prunes invalid edges or invents cuisine matches.
        for donor in options:
            require(donor in table and donor != bid, 'Self/unknown donor')
            d = table[donor]
            require(d['split'] == r['split'], 'Cross-split donor forbidden')
            require(len(d['selected_photo_ids']) == count, 'Caption-count mismatch')
            _, body = partition(d['source'])
            require(body.count('[IMG_CAP_') == count, 'Donor caption count inconsistent')
            source = head + body
            ids = tokenizer.encode(source).ids
            # Same marginal caption-token accounting as frozen packing: difference
            # between full source and its review/scaffolding prefix; no truncation.
            increment = len(ids) - len(tokenizer.encode(head).ids)
            require(0 < increment <= 750 and len(ids) <= 5000, 'Token-budget violation; stop, no fallback')
        donor = rng.choice(options)
        _, body = partition(table[donor]['source'])
        source = head + body
        require(partition(source)[0] == head, 'Recipient metadata/reviews changed')
        result.append(dict(recipient_id=bid, donor_id=donor, split=r['split'], caption_count=count,
                           source=source, source_sha256=digest(source), review_prefix_sha256=digest(head),
                           donor_caption_sha256=digest(body), donor_source_sha256=digest(table[donor]['source']),
                           recipient_photo_ids=r['selected_photo_ids'], donor_photo_ids=table[donor]['selected_photo_ids'],
                           input_ids=tokenizer.encode(source).ids))
    return dict(status='FIXTURE_ONLY_NOT_APPROVED_PRODUCTION_MAPPING', seed=seed,
                eligibility_sha256=digest(json.dumps(approved_edges, sort_keys=True)), mappings=result,
                one_to_one_required=False, one_to_one_note='Not specified in recovered protocol; not claimed here',
                production_policy_approved=policy_approved)
