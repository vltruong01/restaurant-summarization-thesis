"""Proposed source-only Shuffled mapping. No file reader, model or test access.

Policy is pending researcher adjudication. The production algorithm is exercised
only on synthetic/TRAIN/validation inputs during preparation.
"""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone, timedelta


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha_obj(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalize(value):
    return unicodedata.normalize('NFKC', value).strip().casefold()


def parts(row):
    require(row['source'].count('[VISUAL_CONTEXT]') == 1, 'Ambiguous visual boundary')
    head, body = row['source'].split('[VISUAL_CONTEXT]')
    require(head.endswith('\n\n'), 'Frozen Stage H source boundary differs')
    return head + '[VISUAL_CONTEXT]', body


def tags(row, policy):
    # Exact metadata category strings only. No review/caption semantic classifier.
    metadata = row['source'].split('[REVIEWS]')[0]
    lines = [x[len('categories='):] for x in metadata.splitlines() if x.startswith('categories=')]
    require(len(lines) == 1, 'Missing/ambiguous category metadata')
    allowed = {normalize(x) for x in policy['cuisine_subcategory_allowlist']}
    return sorted({normalize(x) for x in lines[0].split(',')} & allowed)


def authorize_test_rows(rows, policy, authorization, now=None):
    """Future source-access gate; never manufactures researcher approval."""
    a = authorization or {}
    require(a.get('policy_researcher_approved') is True and a.get('test_source_mapping_access_approved') is True,
            'Held-out mapping needs separate explicit policy and source-access approval')
    require(a.get('approval_source') == 'researcher_message' and a.get('researcher_explicit_authorization'),
            'Explicit researcher approval required')
    require(a.get('policy_sha256') == sha_obj(policy) and a.get('source_rows_sha256') == sha_obj(rows),
            'Mapping authorization does not bind exact policy/source rows')
    require(a.get('inference_authorized') is False, 'Mapping receipt cannot authorize inference')
    start = datetime.fromisoformat(a['authorized_at_utc'])
    end = datetime.fromisoformat(a['deadline_utc'])
    require(all(t.tzinfo and t.utcoffset().total_seconds() == 0 for t in (start, end)), 'UTC clocks required')
    require(start <= (now or datetime.now(timezone.utc)) < end <= start + timedelta(hours=48), 'Expired mapping scope')


def build_mapping(rows, tokenizer, policy, *, test_authorization=None):
    """Allow donor reuse; deterministic hash rank within predeclared tiers.

    A failed mapping returns no partial production mapping. Eligibility rejection
    counts and source bindings accompany each successful recipient assignment.
    """
    required = {'business_id', 'split', 'source', 'source_sha256', 'selected_review_ids', 'selected_photo_ids'}
    require(rows and all(set(r) == required for r in rows), 'Source-only schema; no targets/references/outputs')
    require(policy['version'] == 'SHUFFLED_PRODUCTION_POLICY_V2_PROPOSED' and policy['seed'] == 42
            and policy['donor_reuse'] == 'ALLOWED' and policy['caption_count'] == 'EXACT_RECIPIENT_F_COUNT'
            and policy['fallback'] == 'ANY_BUDGET_FEASIBLE_SAME_SPLIT_SAME_COUNT'
            and policy['no_eligible_donor'] == 'BLOCK_ENTIRE_MAPPING', 'Unreviewed policy variant')
    require(all(r['split'] in ('train', 'validation', 'test') for r in rows), 'Unknown split')
    if any(r['split'] == 'test' for r in rows):
        require(all(r['split'] == 'test' for r in rows), 'Held-out mapping must be isolated from other splits')
        authorize_test_rows(rows, policy, test_authorization)
    table = {r['business_id']: r for r in rows}
    require(len(table) == len(rows), 'Duplicate business identities')
    encoded, prepared, base_lengths, review_lengths = {}, {}, {}, {}
    for bid, r in table.items():
        require(sha_text(r['source']) == r['source_sha256'], 'Source hash mismatch')
        require(len(r['selected_photo_ids']) == len(set(r['selected_photo_ids'])) <= 5, 'Caption identity/count error')
        head, body = parts(r)
        base = head.removesuffix('\n\n[VISUAL_CONTEXT]')
        require(base.count('[REVIEWS]\n') == 1 and 0 < len(r['selected_review_ids']) <= 30
                and len(set(r['selected_review_ids'])) == len(r['selected_review_ids']), 'Invalid frozen reviews')
        review_block = base.split('[REVIEWS]\n', 1)[1]
        require(re.findall(r'\[REV_(\d+)\]', review_block) ==
                [str(i) for i in range(1, len(r['selected_review_ids'])+1)], 'Frozen review marker/ID count mismatch')
        review_tokens = len(tokenizer.encode(review_block).ids)
        require(review_tokens <= 4000, 'Frozen review block exceeds 4000')
        count = len(r['selected_photo_ids'])
        numbers = re.findall(r'\[IMG_CAP_(\d+)\]', body)
        require(numbers == [str(i) for i in range(1, count+1)], 'Missing/duplicate/out-of-order caption marker')
        require(count > 0 or not body.strip(), 'Zero-caption payload contains visual text')
        ids = tokenizer.encode(r['source']).ids
        require(ids[0] == 0 and ids[-1] == 2 and len(ids) <= 5000, 'Source exceeds frozen token budget')
        base_lengths[bid] = len(tokenizer.encode(base).ids)
        review_lengths[bid] = review_tokens
        require(0 <= len(ids) - base_lengths[bid] <= 750, 'Original caption allowance exceeded')
        prepared[bid] = (head, body, count, tags(r, policy))
        encoded[bid] = ids
    policy_hash = sha_obj(policy)
    result = []
    for bid in sorted(table):
        recipient = table[bid]
        head, body, count, recipient_tags = prepared[bid]
        rejects = dict(self_match=0, different_split=0, different_count=0, token_budget=0)
        if count == 0:
            result.append(dict(recipient_id=bid, donor_id=None, split=recipient['split'], tier='ZERO_CAPTION_UNCHANGED',
                               caption_count=0, recipient_source_sha256=recipient['source_sha256'],
                               shuffled_source_sha256=recipient['source_sha256'], source=recipient['source'],
                               input_ids=encoded[bid], input_ids_sha256=sha_obj(encoded[bid]),
                               caption_increment_tokens=len(encoded[bid])-base_lengths[bid], review_block_tokens=review_lengths[bid],
                               review_prefix_sha256=sha_text(head), selected_review_ids=recipient['selected_review_ids'],
                               donor_photo_ids=[], recipient_tags=recipient_tags))
            continue
        candidates, same_cuisine = [], []
        full_tokens = {}
        for donor_id in sorted(table):
            donor = table[donor_id]
            if donor_id == bid:
                rejects['self_match'] += 1; continue
            if donor['split'] != recipient['split']:
                rejects['different_split'] += 1; continue
            _, donor_body, donor_count, donor_tags = prepared[donor_id]
            if donor_count != count:
                rejects['different_count'] += 1; continue
            source = head + donor_body
            ids = tokenizer.encode(source).ids
            increment = len(ids) - base_lengths[bid]
            if not (0 < increment <= 750 and len(ids) <= 5000):
                rejects['token_budget'] += 1; continue
            full_tokens[donor_id] = ids
            candidates.append(donor_id)
            if set(recipient_tags) & set(donor_tags):
                same_cuisine.append(donor_id)
        require(candidates, 'BLOCK_NO_ELIGIBLE_DONOR: ' + bid)
        pool = same_cuisine or candidates
        rank = lambda donor: (sha_obj([policy['seed'], policy['version'], recipient['split'], bid, donor]), donor)
        donor_id = min(pool, key=rank)
        donor = table[donor_id]
        _, donor_body, _, donor_tags = prepared[donor_id]
        source = head + donor_body
        require(parts(dict(source=source))[0] == head, 'Recipient review prefix changed')
        result.append(dict(recipient_id=bid, donor_id=donor_id, split=recipient['split'],
                           tier='EXACT_SHARED_ALLOWED_CATEGORY' if same_cuisine else 'DECLARED_SAME_COUNT_FALLBACK',
                           caption_count=count, source=source, shuffled_source_sha256=sha_text(source),
                           input_ids=full_tokens[donor_id], input_ids_sha256=sha_obj(full_tokens[donor_id]),
                           caption_increment_tokens=len(full_tokens[donor_id])-base_lengths[bid],
                           review_block_tokens=review_lengths[bid],
                           recipient_source_sha256=recipient['source_sha256'], donor_source_sha256=donor['source_sha256'],
                           review_prefix_sha256=sha_text(head), donor_caption_sha256=sha_text(donor_body),
                           selected_review_ids=recipient['selected_review_ids'], donor_photo_ids=donor['selected_photo_ids'],
                           recipient_tags=recipient_tags, donor_tags=donor_tags,
                           eligible_count=len(candidates), same_category_count=len(same_cuisine),
                           rejected_edges=rejects, chosen_rank_sha256=rank(donor_id)[0]))
    from collections import Counter
    reuse = Counter(r['donor_id'] for r in result if r['donor_id'] is not None)
    return dict(status='MAPPING_COMPLETE_PENDING_RESEARCHER_POLICY_APPROVAL' if test_authorization is None
                else 'SOURCE_MAPPING_ONLY_NO_INFERENCE', policy_sha256=policy_hash,
                source_rows_sha256=sha_obj(rows), mappings=result, donor_reuse_counts=dict(sorted(reuse.items())),
                one_to_one=False, policy_approved_by_this_function=False, selection_uses_targets=False)
