"""One fixed encoder-mask intervention. No model loading or source/label changes."""
import hashlib
import json
import re

POLICY = 'BOS_PLUS_FIRST_BPE_OF_EACH_EXISTING_REVIEW_CAPTION_MARKER_V1'
ARMS = ('bos_only', 'boundary')
MAX_GLOBAL = 36  # BOS + <=30 reviews + <=5 captions; never a top-k selection.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode('ascii')).hexdigest()


def anchors_from_encoding(source, ids, offsets, review_count, caption_count):
    """Use offsets from the EXACT complete frozen encoding, not marker re-encoding."""
    require(0 < len(ids) <= 5000 and ids[0] == 0 and ids[-1] == 2, 'Incomplete/overflow source')
    require(len(ids) == len(offsets), 'Offsets/IDs mismatch')
    require(1 <= review_count <= 30 and 0 <= caption_count <= 5, 'Counts outside frozen envelope')
    require(source.count('\n\n[REVIEWS]\n') == source.count('\n\n[VISUAL_CONTEXT]') == 1,
            'Ambiguous source section boundaries')
    review_start = source.index('\n\n[REVIEWS]\n')
    visual_start = source.index('\n\n[VISUAL_CONTEXT]')
    found = []
    for kind, pattern, expected in (
        ('review', r'(?m)^\[REV_(\d+)\](?= stars=[1-5] date=)', review_count),
        ('caption', r'(?m)^\[IMG_CAP_(\d+)\](?=\n|$)', caption_count),
    ):
        matches = list(re.finditer(pattern, source))
        require([int(m.group(1)) for m in matches] == list(range(1, expected+1)),
                'Missing, duplicate, embedded or non-contiguous '+kind+' marker')
        for m in matches:
            require((review_start < m.start() < visual_start) if kind == 'review' else m.start() > visual_start,
                    'Marker in wrong evidence section')
            token_indices = [i for i,(a,b) in enumerate(offsets) if b > a and b > m.start() and a < m.end()]
            require(bool(token_indices), 'Marker missing/truncated in token encoding')
            first, last = token_indices[0], token_indices[-1]
            require(offsets[first][0] <= m.start() < offsets[first][1]
                    and offsets[last][0] < m.end() <= offsets[last][1], 'Partial/truncated marker')
            require(token_indices == list(range(first,last+1)), 'Non-contiguous marker BPEs')
            require(all(ids[i] not in (0,1,2) for i in token_indices), 'Special/PAD token in marker')
            found.append(dict(kind=kind, marker=m.group(0), char_start=m.start(), char_end=m.end(),
                              token_start=first, token_end_exclusive=last+1, anchor=first))
    found.sort(key=lambda m:m['char_start'])
    positions = [0] + [m['anchor'] for m in found]
    require(len(positions) == len(set(positions)) == 1+review_count+caption_count <= MAX_GLOBAL,
            'Anchor collision/count/envelope mismatch')
    return dict(input_ids_sha256=ids_sha(ids), source_length=len(ids), review_count=review_count,
                caption_count=caption_count, positions=positions, markers=found)


class MaskIndex:
    def __init__(self, rows):
        self.by_hash = {}
        for row in rows:
            key = row['input_ids_sha256']
            if key in self.by_hash:
                require(self.by_hash[key] == row, 'Ambiguous source-token identity')
            self.by_hash[key] = row
        require(bool(self.by_hash), 'Empty mask index')

    def lookup(self, ids):
        key = ids_sha(ids)
        require(key in self.by_hash, 'Unknown/modified/truncated source; no mask fallback')
        row = self.by_hash[key]
        require(len(ids) == row['source_length'] and ids[0] == 0 and ids[-1] == 2, 'Source identity differs')
        positions = row['positions']
        require(positions[0] == 0 and len(positions) == len(set(positions))
                == 1+row['review_count']+row['caption_count'] <= MAX_GLOBAL, 'Invalid sealed anchor count')
        require(all(type(i) is int and 0 <= i < len(ids)-1 for i in positions), 'Anchor on padding/EOS/out of range')
        for m in row['markers']:
            require(m['anchor'] == m['token_start'] in positions and ids[m['anchor']] not in (0,1,2),
                    'Marker anchor inconsistent')
        return row


class MaskCollator:
    """Wrap the trusted BOS-free collator; change only global_attention_mask."""
    def __init__(self, original_factory, index, arm, stress=False, observer=None):
        require(arm in ARMS, 'Unknown arm')
        self.original = original_factory(stress=stress)
        self.index, self.arm, self.observer = index, arm, observer

    def __call__(self, features):
        rows = [self.index.lookup(f['input_ids']) for f in features]
        batch = self.original(features)
        if self.arm == 'boundary':
            mask = batch['global_attention_mask'].clone()
            mask.zero_()
            for i,row in enumerate(rows):
                mask[i,row['positions']] = 1
            batch['global_attention_mask'] = mask
        mask = batch['global_attention_mask']
        require(not (mask.bool() & ~batch['attention_mask'].bool()).any().item(), 'Global attention on padding')
        expected = [1 if self.arm == 'bos_only' else len(r['positions']) for r in rows]
        require(mask.sum(1).tolist() == expected, 'Mask count differs')
        if self.observer is not None:
            for row in rows:
                self.observer(dict(input_ids_sha256=row['input_ids_sha256'], arm=self.arm,
                    positions=[0] if self.arm == 'bos_only' else row['positions'],
                    padded_source=batch['input_ids'].shape[1], padded_labels=batch['labels'].shape[1]))
        return batch


def install_collator(label_module, index, arm, observer=None):
    """One shared injection point covers old train/gate/eval AND epoch1 sanity."""
    from contextlib import contextmanager
    from unittest.mock import patch
    original = label_module.CorrectedLabelCollator
    @contextmanager
    def installed():
        def factory(stress=False):
            return MaskCollator(original, index, arm, stress, observer)
        with patch.object(label_module, 'CorrectedLabelCollator', factory):
            yield
    return installed()
