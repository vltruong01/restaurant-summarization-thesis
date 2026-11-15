"""Offline-safe pseudo-repetition UL primitives. No model import or execution.

This is a custom, teacher-derived negative-prefix objective, NOT a reproduction
of DITTO or Welleck's sampled sequence-UL procedure. Original positives are intact.
"""
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage_i_unlikelihood_pilot_v1.json'
PREP = ROOT / 'outputs/stage_i/unlikelihood_preparation_v1'
RUNNER = ROOT / 'scripts/run_stage_i_unlikelihood_pilot_v1.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write((value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                     indent=2, allow_nan=False)) + '\n')


def stable_key(seed, business_id, purpose):
    return hashlib.sha256(f'{seed}|{purpose}|{business_id}'.encode()).hexdigest()


def body_spans(text):
    """Only bodies of canonical sections, never markers or their newline."""
    markers = list(re.finditer(r'(?m)^\[(?:REVIEW_CONSENSUS|VISUAL_EVIDENCE)\]\n', text))
    require(len(markers) == 2 and markers[0].start() == 0, 'Noncanonical F target')
    require(markers[0].group().startswith('[REVIEW') and
            markers[1].group().startswith('[VISUAL'), 'Wrong section order')
    return [(m.end(), markers[i+1].start() if i+1 < len(markers) else len(text))
            for i, m in enumerate(markers)]


def candidates(text, kind):
    require(kind in ('sentence', 'word'), 'Unknown negative kind')
    found = []
    for lo, hi in body_spans(text):
        body = text[lo:hi]
        pattern = r'[^.!?\n]+[.!?](?=\s|$)' if kind == 'sentence' else r"\b[A-Za-z][A-Za-z'-]{2,}\b"
        for match in re.finditer(pattern, body):
            a, b = lo + match.start(), lo + match.end()
            while a < b and text[a].isspace():
                a += 1
            if kind == 'sentence' and len(re.findall(r'\w+', text[a:b])) < 4:
                continue
            # Do not label an already valid exact continuation as a negative.
            if text[b:].lstrip().startswith(text[a:b]):
                continue
            found.append((a, b))
    return found


def build_negative(text, tokenizer, business_id, kind, limit=768, seed=42):
    """Repeat one train-derived unit, keep only prefix through injected copy.

    Positive labels remain full length. No suffix is silently cut from a positive.
    Character/token alignment is checked; conflicts at the original branching
    prefix are unpenalized. Overflow/ineligible views are explicitly skipped.
    """
    original = tokenizer.encode(text).ids
    options = candidates(text, kind)
    options.sort(key=lambda x: stable_key(seed, business_id, f'{kind}|{x[0]}|{x[1]}'))
    reasons = {}
    for a, b in options:
        negative_text = text[:b] + ' ' + text[a:b]
        start = b + 1
        encoded = tokenizer.encode(negative_text)
        if len(encoded.ids) > limit:
            reasons['overflow'] = reasons.get('overflow', 0) + 1
            continue
        positions = [i for i, (x, y) in enumerate(encoded.offsets)
                     if x >= start and y > x and y <= len(negative_text) and encoded.ids[i] not in (0, 1, 2)]
        if not positions:
            reasons['no_complete_tokens'] = reasons.get('no_complete_tokens', 0) + 1
            continue
        branch = positions[0]
        if encoded.ids[:branch] != original[:branch]:
            reasons['prefix_retokenized'] = reasons.get('prefix_retokenized', 0) + 1
            continue
        mask = [False] * len(encoded.ids)
        for i in positions:
            # If the two prefixes are identical, never penalize the true label.
            if encoded.ids[:i] == original[:i] and i < len(original) and encoded.ids[i] == original[i]:
                continue
            mask[i] = True
        if not any(mask):
            reasons['all_gold_conflicts'] = reasons.get('all_gold_conflicts', 0) + 1
            continue
        return {'eligible': True, 'kind': kind, 'negative_text': negative_text,
                'negative_ids': encoded.ids, 'penalty_mask': mask,
                'unit_char_span': [a, b], 'inserted_char_span': [start, len(negative_text)],
                'positive_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'negative_sha256': hashlib.sha256(negative_text.encode()).hexdigest(),
                'branch_token_position': branch, 'penalized_tokens': sum(mask),
                'rejected_candidates': reasons}
    return {'eligible': False, 'kind': kind, 'reason': 'NO_SAFE_BOUNDED_NEGATIVE',
            'rejected_candidates': reasons, 'positive_sha256': hashlib.sha256(text.encode()).hexdigest()}


def unlikelihood_loss(logits, negative_ids, penalty_mask):
    """Mean -log(1-p(negative token)) on inserted-copy positions only.

    logits[:,t] predicts negative_ids[:,t] with explicit right-shifted decoder
    input. FP32 stable log1mexp; no clamping/tolerance relaxation hides infinity.
    Returns finite differentiable zero for an empty mask.
    """
    import torch
    require(logits.ndim == 3 and logits.shape[:2] == negative_ids.shape == penalty_mask.shape,
            'UL alignment/shape mismatch')
    require(penalty_mask.dtype == torch.bool, 'UL mask must be boolean')
    require(torch.isfinite(logits).all().item(), 'Nonfinite UL logits')
    require(not ((negative_ids < 0) | (negative_ids >= logits.shape[-1])).any().item(), 'Invalid UL token ID')
    require(not (penalty_mask & ((negative_ids == 0) | (negative_ids == 1) | (negative_ids == 2))).any().item(),
            'UL on BOS/pad/EOS forbidden')
    if not penalty_mask.any().item():
        return logits.float().reshape(-1)[0] * 0.0
    selected = logits.float()[penalty_mask]
    ids = negative_ids[penalty_mask]
    logp = selected.log_softmax(-1).gather(1, ids[:, None]).squeeze(1)
    # Split before evaluating either branch: torch.where would evaluate an
    # unstable inactive branch too, which can poison backward with NaNs.
    lower = logp < -math.log(2)
    log_complement = torch.empty_like(logp)
    log_complement[lower] = torch.log1p(-torch.exp(logp[lower]))
    log_complement[~lower] = torch.log(-torch.expm1(logp[~lower]))
    loss = -log_complement.mean()
    require(torch.isfinite(loss).item(), 'UL saturated/nonfinite; stop, no clamp')
    return loss


def paired_objective(model, inputs, weight, training, recorder=None):
    """CE validation; equal negative forwards in BOTH training arms."""
    import torch
    require(weight in (0.0, 0.1), 'Unreviewed loss weight')
    positive = {k: v for k, v in inputs.items() if not k.startswith('negative_')}
    outputs = model(**positive, use_cache=False)
    ce = outputs.loss
    require(torch.isfinite(ce).item(), 'Nonfinite positive CE')
    ul = ce.new_zeros((), dtype=torch.float32)
    if training and inputs['negative_penalty_mask'].any().item():
        ids = inputs['negative_ids']
        decoder = torch.full_like(ids, 1)
        decoder[:, 0] = 2
        decoder[:, 1:] = ids[:, :-1]
        attention = torch.ones_like(decoder)
        # Physical batch1 is frozen, so negative rows are unpadded.
        require(ids.shape[0] == 1 and not (ids == 1).any().item(), 'Negative padding/scope mismatch')
        negative = model(input_ids=positive['input_ids'], attention_mask=positive['attention_mask'],
                         global_attention_mask=positive['global_attention_mask'],
                         decoder_input_ids=decoder, decoder_attention_mask=attention,
                         use_cache=False)
        ul = unlikelihood_loss(negative.logits, ids, inputs['negative_penalty_mask'])
    total = ce + weight * ul
    require(torch.isfinite(total).item(), 'Nonfinite combined objective')
    if recorder is not None:
        recorder({'training': bool(training), 'CE': float(ce.detach()), 'UL': float(ul.detach()),
                  'weight': weight if training else 0.0, 'total': float(total.detach()),
                  'penalized_tokens': int(inputs['negative_penalty_mask'].sum()) if training else 0})
    return total, outputs


def authorization_gate(path, config, manifest, mode, arm, now=None):
    """No executable GPU permission is created by offline preparation."""
    a = read(path)
    require(arm in ('control_ce', 'treatment_ul') and mode in ('feasibility', 'train', 'evaluate'), 'Wrong scope')
    require(a.get('experiment') == config['experiment'] and a.get('GPU_execution_approved') is True
            and a.get('approval_source') == 'researcher_message' and a.get('researcher_explicit_authorization'),
            'Explicit new researcher GPU authorization required')
    for key, p in [('config_sha256', CONFIG), ('preflight_manifest_sha256', PREP/'PREFLIGHT_MANIFEST.json'),
                   ('runner_sha256', RUNNER)]:
        require(a.get(key) == sha(p), 'Authorization binding mismatch: ' + key)
    require(a.get('approved_arms') == ['control_ce', 'treatment_ul'] and mode in a.get('approved_modes', []), 'Mode not approved')
    require(a.get('max_training_updates_total') == 256 and a.get('max_gate_updates_total') == 6
            and a.get('max_generation_outputs_total') == 48, 'Budget differs')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False, 'Forbidden scope')
    for key in ('execution_authorized_at_utc', 'new_execution_deadline_utc'):
        require(a.get(key), 'New explicit timing decision required')
    start = datetime.fromisoformat(a['execution_authorized_at_utc'])
    end = datetime.fromisoformat(a['new_execution_deadline_utc'])
    current = now or datetime.now(timezone.utc)
    require(start.tzinfo is not None and end.tzinfo is not None, 'Explicit UTC timestamps required')
    require(start.utcoffset().total_seconds() == end.utcoffset().total_seconds() == 0 and start < end
            and start <= current < end, 'Execution time outside new approved window')
    require(end <= start + timedelta(hours=48), 'Proposed48hour execution/review ceiling exceeded')
    require(a.get('prior_stop_loss_override_explicit') is True, 'Historical final-attempt limits require explicit amendment')
    return a
