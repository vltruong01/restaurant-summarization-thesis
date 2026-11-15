"""Portable, lossless Stage H reader and LED collator; no model/API calls."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_bundle(config, root=ROOT):
    path = root / config['stage_h_manifest']
    if config.get('stage_h_manifest_sha256'):
        require(sha(path) == config['stage_h_manifest_sha256'], 'Reviewed Stage H manifest changed')
    manifest = read(path)
    require(manifest['status'] == 'STAGE_H_DATA_VALIDATED_DECODER_POLICY_PENDING', 'Stage H data not accepted')
    require(sha(root / manifest['validation_report']) == manifest['validation_report_sha256'], 'Stage H report changed')
    tm = read(root / config['tokenizer_manifest'])
    require(tm['revision'] == manifest['tokenizer_revision'] == config['revision'], 'Tokenizer/model revision mismatch')
    for name, digest in tm['sha256'].items():
        require(sha(root / tm['relative_directory'] / name) == digest, 'Tokenizer asset changed')
    mm = read(root / config['model_config_manifest'])
    require(mm['revision'] == config['revision'], 'Backbone config revision mismatch')
    require(sha(root / mm['config_file']) == mm['config_sha256'], 'Backbone config changed')
    mc = read(root / mm['config_file'])
    require(config['source_limit'] == manifest['budgets']['source'] == 5000, 'Source budget changed')
    require(config['generation']['max_length'] == config['label_limit'] <= mc['max_decoder_position_embeddings'], 'Decoder capacity/limit mismatch')
    require(mc['attention_window'] == [1024]*6, 'Frozen encoder window changed')
    require(mc['bos_token_id'] == 0 and mc['eos_token_id'] == mc['decoder_start_token_id'] == 2 and mc['pad_token_id'] == 1, 'Special tokens changed')
    for info in manifest['datasets'].values():
        require(sha(root / info['path']) == info['sha256'], 'Packed dataset changed')
    return manifest, tm, mc


class PackedLEDDataset:
    def __init__(self, info, source_limit, label_limit, root=ROOT):
        require(sha(root / info['path']) == info['sha256'], 'Packed dataset hash mismatch')
        with np.load(root / info['path'], allow_pickle=False) as pack:
            self.pack = {k: pack[k].copy() for k in pack.files}
        self.business_ids = self.pack['business_ids'].tolist()
        n = len(self.business_ids)
        require(n == info['rows'] and n == len(set(self.business_ids)), 'Packed row/ID mismatch')
        for key, off, limit in [('input_ids', 'input_offsets', source_limit), ('labels', 'label_offsets', label_limit)]:
            o, ids = self.pack[off], self.pack[key]
            require(len(o) == n+1 and o[0] == 0 and o[-1] == len(ids) and np.all(np.diff(o) > 0), 'Invalid ragged offsets')
            require(np.max(np.diff(o)) <= limit, f'{key} exceeds approved/candidate limit; truncation forbidden')
            require(np.all(ids >= 0) and np.all(ids < 50265), 'Token ID outside frozen vocabulary')
            require(np.all(ids[o[:-1]] == 0) and np.all(ids[o[1:]-1] == 2), 'Missing complete BOS/EOS')

    def __len__(self):
        return len(self.business_ids)

    def __getitem__(self, index):
        io, lo = self.pack['input_offsets'], self.pack['label_offsets']
        return {'input_ids': self.pack['input_ids'][io[index]:io[index+1]].astype(np.int64).tolist(),
                'labels': self.pack['labels'][lo[index]:lo[index+1]].astype(np.int64).tolist()}


class LEDCompleteCollator:
    def __init__(self, source_limit=5000, label_limit=768, window=1024, pad_source_to=None, pad_label_to=None):
        self.source_limit, self.label_limit, self.window = source_limit, label_limit, window
        self.pad_source_to, self.pad_label_to = pad_source_to, pad_label_to

    def __call__(self, features):
        import torch
        require(features, 'Empty batch')
        for f in features:
            require(0 < len(f['input_ids']) <= self.source_limit, 'Source overflow; no truncation permitted')
            require(0 < len(f['labels']) <= self.label_limit, 'Label overflow; no truncation permitted')
            require(f['input_ids'][0] == f['labels'][0] == 0 and f['input_ids'][-1] == f['labels'][-1] == 2, 'Incomplete BOS/EOS')
        source_max = max(len(f['input_ids']) for f in features)
        label_max = max(len(f['labels']) for f in features)
        # Padding is not counted as content and may exceed the logical source cap.
        sl = ((source_max + self.window-1)//self.window)*self.window
        ll = min(self.label_limit, ((label_max+7)//8)*8)
        if self.pad_source_to is not None:
            require(self.pad_source_to >= sl and self.pad_source_to % self.window == 0, 'Invalid stress source shape')
            sl = self.pad_source_to
        if self.pad_label_to is not None:
            require(label_max <= self.pad_label_to <= self.label_limit, 'Invalid stress label shape')
            ll = self.pad_label_to
        ids = torch.full((len(features), sl), 1, dtype=torch.long)
        attention = torch.zeros_like(ids); global_attention = torch.zeros_like(ids)
        labels = torch.full((len(features), ll), -100, dtype=torch.long)
        for i, f in enumerate(features):
            n, m = len(f['input_ids']), len(f['labels'])
            ids[i, :n] = torch.tensor(f['input_ids'], dtype=torch.long)
            attention[i, :n] = 1; global_attention[i, 0] = 1
            labels[i, :m] = torch.tensor(f['labels'], dtype=torch.long)
        require(torch.equal(global_attention.sum(1), torch.ones(len(features), dtype=torch.long)), 'Global attention count wrong')
        require(not torch.any(global_attention.bool() & ~attention.bool()), 'Global attention on padding')
        return dict(input_ids=ids, attention_mask=attention, global_attention_mask=global_attention, labels=labels)


def settings_signature(config):
    # Authorization switches do not change scientific settings or dataset identity.
    value = {k: v for k, v in config.items() if k not in ('approval', 'status', 'output_root')
             and not k.startswith('approved_feasibility_report')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def execution_gate(config, mode):
    require(mode in ('feasibility', 'train'), 'Unknown execution scope')
    require(config['approval']['length_and_training_settings'], 'Researcher has not approved the proposed decoder/optimization settings')
    require(config['approval']['gpu_feasibility'], 'GPU feasibility execution is not authorized')
    require(config.get('stage_h_manifest_sha256'), 'Approved dataset manifest checksum is required')
    if mode == 'train':
        require(config['approval']['full_training'], 'Full LED training is not authorized')
        require(config.get('approved_feasibility_report') and config.get('approved_feasibility_report_sha256'), 'Reviewed successful GPU feasibility report required')
        p = ROOT / config['approved_feasibility_report']
        require(sha(p) == config['approved_feasibility_report_sha256'], 'Approved feasibility report changed')
        r = read(p)
        require(r['status'] == 'GPU_FEASIBILITY_PASS_PENDING_RESEARCHER_REVIEW', 'GPU feasibility has not passed')
        require(r['settings_signature'] == settings_signature({k: v for k, v in config.items() if not k.startswith('approved_feasibility_report')}), 'GPU report/config settings differ')


def generation_format(text, condition, allow_empty_visual=False):
    has_review = text.startswith('[REVIEW_CONSENSUS]\n') and text.count('[REVIEW_CONSENSUS]') == 1
    if condition == 'reviews_only':
        return bool(has_review and '[VISUAL_EVIDENCE]' not in text and text[len('[REVIEW_CONSENSUS]\n'):].strip())
    marker = '\n\n[VISUAL_EVIDENCE]\n'
    if not (has_review and text.count('[VISUAL_EVIDENCE]') == 1 and marker in text):
        return False
    review, visual = text.split(marker)
    return bool(review[len('[REVIEW_CONSENSUS]\n'):].strip() and (visual.strip() or allow_empty_visual))
