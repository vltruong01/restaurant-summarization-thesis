"""CPU-only mathematical reference. No LED model, optimizer or GPU runner.

This validates the proposed loss and its detached-coefficient gradient identity.
It does NOT establish LED/dropout/AMP replay equivalence on the pinned GPU host.
"""
import torch
import torch.nn.functional as F

PAIR_WEIGHT = 0.1
MARGIN = 0.1
TEMPERATURE = 1.0


def sequence_score(logits, labels):
    if logits.device.type != 'cpu' or labels.device.type != 'cpu':
        raise ValueError('Offline mathematical reference is CPU-only')
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError('Expected [tokens,vocabulary] and [tokens]')
    valid = labels != -100
    if not valid.any():
        raise ValueError('All-ignored sequence')
    if not torch.isfinite(logits[valid]).all():
        raise ValueError('Nonfinite valid logits')
    work = logits if logits.dtype == torch.float64 else logits.float()
    tokens = labels[valid]
    if (tokens < 0).any() or (tokens >= logits.shape[-1]).any():
        raise ValueError('Invalid target token')
    chosen = work[valid].gather(1, tokens[:, None]).squeeze(1)
    return (chosen - torch.logsumexp(work[valid], dim=-1)).mean()


def objective(positive_score, negative_score, weight=PAIR_WEIGHT):
    if weight not in (0.0, PAIR_WEIGHT):
        raise ValueError('Fixed proposal weight; no search')
    ce = -positive_score
    if weight == 0.0:
        return ce
    return ce + weight * F.softplus((MARGIN - positive_score + negative_score) / TEMPERATURE)


def replay_coefficient(positive_score, negative_score):
    z = (MARGIN - positive_score.detach() + negative_score.detach()) / TEMPERATURE
    return (PAIR_WEIGHT / TEMPERATURE) * torch.sigmoid(z)


def assert_replayed_score(observed, expected):
    """Proposed FP32 scalar tolerance; full LED gradient tests remain required."""
    if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
        raise ValueError('Nonfinite replay score')
    if not torch.allclose(observed, expected, rtol=1e-6, atol=1e-6):
        raise ValueError('Scout/replayed score differs; stop, do not approximate silently')
