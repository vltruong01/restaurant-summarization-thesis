"""D-028 deterministic verification selection, provenance and balance-only expansion."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

CHOICES = {'consistency': ['consistent', 'inconsistent', 'uncertain'],
           'relevance': ['relevant', 'irrelevant', 'uncertain'],
           'redundancy': ['same_information', 'complementary_information', 'uncertain'],
           'broad_topic': ['same_broad_visual_topic', 'different_topic', 'uncertain'],
           'image_duplicate': ['same_visual_content', 'distinct_content', 'uncertain']}
KINDS = {'image_caption': ['consistency', 'relevance'],
         'caption_pair': ['redundancy', 'broad_topic', 'image_duplicate']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def rank(case_id):
    return hashlib.sha256(('42|D028_verification|' + case_id).encode()).hexdigest()


def quantile_bins(cases, scores, key):
    ordered = sorted(cases, key=lambda c: (scores[c['case_id']][key], c['case_id']))
    return {c['case_id']: min(4, 5 * i // len(ordered)) for i, c in enumerate(ordered)}


def verification_orders(dataset, private, photo_plan):
    cases = dataset['cases']; orders = {}
    for kind in KINDS:
        rows = [c for c in cases if c['kind'] == kind]
        by_business = defaultdict(list)
        for c in rows: by_business[c['business_id']].append(c)
        selected = []; used = set()
        def add(c):
            if c['case_id'] not in used:
                selected.append(c['case_id']); used.add(c['case_id'])
        # Coverage prefixes are selected without reading AI/human labels.
        for pass_index in range(1 if kind == 'image_caption' else 2):
            for bid in sorted(by_business, key=rank):
                ordered = sorted(by_business[bid], key=lambda c: rank(c['case_id']))
                if len(ordered) > pass_index: add(ordered[pass_index])
        if kind == 'image_caption':
            for c in sorted(rows, key=lambda c: rank(c['case_id'])):
                if any(photo_plan[p]['label'] == 'menu' for p in c['photo_ids']): add(c)
            q1 = quantile_bins(rows, private, 'clip_score')
            q2 = quantile_bins(rows, private, 'relevance_margin')
        else:
            q1, q2 = {}, {}
            for items in by_business.values():
                q1.update(quantile_bins(items, private, 'semantic_cosine'))
                q2.update(quantile_bins(items, private, 'phash_hamming'))
        cells = defaultdict(lambda: defaultdict(list))
        for c in rows:
            if c['case_id'] not in used:
                cells[q1[c['case_id']], q2[c['case_id']]][c['business_id']].append(c)
        for businesses in cells.values():
            for items in businesses.values(): items.sort(key=lambda c: rank(c['case_id']))
        queues = {}
        for cell, businesses in cells.items():
            queues[cell] = []
            while any(businesses.values()):
                for bid in sorted(businesses, key=rank):
                    if businesses[bid]: queues[cell].append(businesses[bid].pop(0))
        while any(queues.values()):
            for cell in sorted(queues):
                if queues[cell]: add(queues[cell].pop(0))
        assert len(selected) == len(rows) == len(set(selected))
        orders[kind] = selected
    return orders


def active_ids(order, selection):
    return (order['orders']['image_caption'][:selection['image_cases']] +
            order['orders']['caption_pair'][:selection['pair_cases']])


def validate_export(dataset, order, selection, ai, labels):
    assert dataset['dataset_sha256'] == canonical_sha({k:v for k,v in dataset.items() if k!='dataset_sha256'})
    assert order['manifest_sha256'] == canonical_sha({k:v for k,v in order.items() if k!='manifest_sha256'})
    assert ai['dataset_sha256'] == dataset['dataset_sha256']
    assert ai['verification_order_sha256'] == order['manifest_sha256']
    assert labels['schema_version'] == 2 and labels['workflow'] == 'AI_assisted_human_verification'
    assert labels['dataset_kind'] == dataset['dataset_kind'] == 'official_calibration'
    assert labels['dataset_sha256'] == dataset['dataset_sha256']
    assert labels['verification_order_sha256'] == order['manifest_sha256']
    assert labels['ai_artifact_sha256'] == canonical_sha(ai)
    assert labels['human_confirmed'] is True and labels['annotator'].strip()
    assert labels['active_case_ids'] == active_ids(order, selection), 'Stale selection: reopen current assisted UI'
    expected = {c['case_id']: c for c in dataset['cases']}
    actual = {c['case_id']: c for c in labels['cases']}
    suggestions = {c['case_id']: c for c in ai['cases']}
    assert len(actual) == len(labels['cases']) and set(actual) == set(expected) == set(suggestions)
    selected = set(labels['active_case_ids'])
    for cid, row in actual.items():
        keys = set(expected[cid]['labels'])
        assert row['ai_suggested_label'] == suggestions[cid]['ai_suggested_label']
        assert set(row['human_final_label']) == keys
        assert type(row['human_verified']) is bool
        for name, value in row['human_final_label'].items():
            assert value == '' or value in CHOICES[name]
        if row['human_verified']:
            assert cid in selected, 'Verified outside predeclared active selection'
            assert all(row['human_final_label'].values()) and row['verified_at']
            assert row['verification_action'] in ['accepted_ai', 'corrected_ai', 'manual']
            if row['verification_action']=='accepted_ai':
                assert row['human_final_label']==row['ai_suggested_label']
            assert not (row['human_final_label'].get('redundancy') == 'same_information' and
                        row['human_final_label'].get('broad_topic') == 'different_topic')
    return actual


def balance_and_expansion(dataset, order, selection, actual, cfg):
    expected = {c['case_id']: c for c in dataset['cases']}
    active = active_ids(order, selection)
    missing = [cid for cid in active if not actual[cid]['human_verified']]
    counts = {k: Counter(positive=0, negative=0, uncertain=0) for k in CHOICES}
    for cid in active:
        row = actual[cid]
        if not row['human_verified']: continue
        for key in expected[cid]['labels']:
            label = row['human_final_label'][key]
            category = 'uncertain' if label == 'uncertain' else 'positive' if label == CHOICES[key][0] else 'negative'
            counts[key][category] += 1
    result = {'counts': {k: dict(c) for k, c in counts.items()},
              'pending_case_ids': missing, 'verified_cases': len(active) - len(missing),
              'active_cases': len(active), 'expanded': False, 'selection': dict(selection)}
    if missing:
        result['status'] = 'WAITING_ACTIVE_HUMAN_VERIFICATION'; return result
    deficient = [k for k, c in counts.items() if min(c['positive'], c['negative']) < cfg['minimum_each_binary_class']]
    result['deficient_criteria'] = deficient
    if not deficient:
        result['status'] = 'BALANCED_VERIFIED_LABELS_READY_FOR_THRESHOLD_SEARCH'; return result
    exhausted = []
    for kind, key, increment in [('image_caption', 'image_cases', 'image_expansion_increment'),
                                  ('caption_pair', 'pair_cases', 'pair_expansion_increment')]:
        if not any(k in deficient for k in KINDS[kind]): continue
        current = selection[key]; total = len(order['orders'][kind])
        if current == total: exhausted.extend(k for k in deficient if k in KINDS[kind])
        else:
            result['selection'][key] = min(total, current + cfg[increment]); result['expanded'] = True
    result['exhausted_criteria'] = exhausted
    result['status'] = ('CLASS_BALANCE_BLOCKER_QUEUE_EXHAUSTED' if exhausted else
                        'EXPANDED_FOR_VERIFIED_CLASS_BALANCE_ONLY')
    # A terminal exhausted criterion blocks the whole freeze. Do not request needless new labels.
    if exhausted: result['selection'] = dict(selection); result['expanded'] = False
    return result
