"""Target-free packing and annotation validators, tested on synthetic/TRAIN fixtures.

No held-out payload reader or model-generation entry point exists in this module.
Scorer implementations are pinned in METRIC_LOCK.json; actual metric execution
and reference binding remain blocked until separate researcher authorization.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict

CONDITIONS = ('RO', 'U', 'F', 'Shuffled')
CRITERIA = ('review_faithfulness', 'aspect_coverage', 'coherence',
            'non_redundancy', 'usefulness', 'visual_grounding_and_contribution')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def target_free_record(record):
    required = {'business_id', 'split', 'condition', 'input_ids', 'attention_mask',
                'global_attention_mask', 'source_sha256', 'selected_review_ids', 'selected_photo_ids'}
    require(set(record) == required, 'Target-free schema forbids labels, target text, references or extra payload')
    require(record['condition'] in CONDITIONS, 'Unknown condition')
    ids, mask, glob = (record[k] for k in ('input_ids', 'attention_mask', 'global_attention_mask'))
    require(len(ids) == len(mask) == len(glob) and len(ids) <= 5120, 'Shape overflow')
    require(all(x in (0, 1) for x in mask + glob), 'Non-binary mask')
    valid = [i for i, x in enumerate(mask) if x]
    require(valid and len(valid) <= 5000 and valid == list(range(len(valid)))
            and ids[0] == 0 and ids[valid[-1]] == 2, 'Truncated/malformed source')
    require(all(ids[i] == 1 for i in range(len(valid), len(ids))), 'Padding mismatch')
    require(glob == [1] + [0] * (len(glob) - 1), 'BOS-only global policy required')
    require(record['condition'] != 'RO' or not record['selected_photo_ids'], 'RO caption leak')
    return True


def sample_fixture(records, size=30, seed=42):
    """Proposed fixed proportional 2x3 design; no outcome information accepted.

    Quantile ties use business ID. Photo groups are zero/1-2/3-5. Allocations use
    largest remainders. This is a prospective plan, pending researcher acceptance.
    """
    require(all(set(r) == {'business_id', 'review_tokens', 'usable_photo_count', 'split'} for r in records),
            'Sampling receives source metadata only; no outputs, labels or ratings')
    require(all(r['split'] in ('train', 'validation') for r in records), 'Only non-held-out fixtures authorized now')
    require(size <= len(records) and len({r['business_id'] for r in records}) == len(records), 'Bad sample frame')
    ordered = sorted(records, key=lambda r: (r['review_tokens'], r['business_id']))
    cells = defaultdict(list)
    for rank, r in enumerate(ordered):
        require(0 <= r['usable_photo_count'] <= 5, 'Photo budget')
        photo = 0 if r['usable_photo_count'] == 0 else 1 if r['usable_photo_count'] <= 2 else 2
        cells[(min(1, rank * 2 // len(records)), photo)].append(r['business_id'])
    exact = {k: size * len(v) / len(records) for k, v in cells.items()}
    allocation = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(cells, key=lambda k: (-(exact[k] - allocation[k]), k))[:size - sum(allocation.values())]:
        allocation[k] += 1
    rng = random.Random(seed)
    ids = []
    for k in sorted(cells):
        ids.extend(rng.sample(sorted(cells[k]), allocation[k]))
    return dict(ids=sorted(ids), allocations={str(k): v for k, v in allocation.items()}, seed=seed,
                status='NON_HELD_OUT_FIXTURE_PROPOSED_SAMPLER')


def annotation_fixture(businesses, seed=42, caption_counts=None):
    require(len(businesses) == 30 and len(set(businesses)) == 30, 'Exactly 30 unique businesses required')
    rng = random.Random(seed)
    key, tasks = {}, []
    for annotator in ('A01', 'A02', 'A03'):
        order = [(b, c) for b in businesses for c in CONDITIONS]
        rng.shuffle(order)
        for i, (bid, condition) in enumerate(order):
            alias = hashlib.sha256(f'{seed}:{annotator}:{i}:{bid}:{condition}'.encode()).hexdigest()[:16]
            count = 0 if condition == 'RO' else (caption_counts or {}).get((bid, condition), 1)
            require(type(count) is int and 0 <= count <= 5, 'Invalid condition caption count')
            key[alias] = dict(business_id=bid, condition=condition, annotator=annotator, caption_count=count)
            tasks.append(dict(alias=alias, annotator=annotator, position=i + 1,
                              scores={k: 'N/A' if condition == 'RO' and k.startswith('visual_') else None for k in CRITERIA},
                              diagnostic_flags=dict(core_omission=None, lost_disagreement=None,
                                                    unsupported_visual_claim=None, wrong_visual_relation=None),
                              visual_contribution_applicability='N/A_NO_CAPTIONS' if count == 0 else 'APPLICABLE',
                              evidence_notes='', researcher_confirmed=False))
    require(len(tasks) == 360 and all(v == 120 for v in Counter(t['annotator'] for t in tasks).values()), 'Incomplete design')
    return dict(tasks=tasks, PRIVATE_CONDITION_KEY=key, status='BLANK_FIXTURE_NOT_HUMAN_EVIDENCE')


def validate_ratings(tasks, key, require_complete=True):
    require(len(tasks) == 360 and len({t['alias'] for t in tasks}) == 360, 'Missing/duplicate output-rating sets')
    for t in tasks:
        require(t['alias'] in key and t['annotator'] == key[t['alias']]['annotator'], 'Blinding binding mismatch')
        require(set(t['scores']) == set(CRITERIA), 'Rubric drift')
        for criterion, value in t['scores'].items():
            info = key[t['alias']]
            na = info['condition'] == 'RO' and criterion.startswith('visual_')
            valid = value == 'N/A' if na else (type(value) is int and 1 <= value <= 5)
            if not na and not require_complete and value is None:
                valid = True
            require(valid, 'Invalid or missing human rating')
    return True


def reference_contract(manifest):
    require(manifest.get('split') == 'test' and manifest.get('independent_human_approved') is True,
            'Independent held-out human references required')
    require(manifest.get('business_count') == 100 and manifest.get('unresolved') == 0,
            'Incomplete test-reference coverage/adjudication')
    require(manifest.get('reference_sha256') and manifest.get('approval_receipt_sha256'), 'Unbound references')
    require(manifest.get('origin') not in ('D053_validation', 'Teacher', 'model_assisted_only'), 'Invalid final reference substitute')
    return True
