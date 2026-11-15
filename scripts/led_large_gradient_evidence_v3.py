"""Read-only gradient evidence. No model execution or AMP/optimizer policy changes."""
import math


def finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def inspect_gradients(model, detailed=False):
    import torch
    records = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            records[name] = dict(status='missing', elements=param.numel())
            continue
        g = param.grad.detach()
        finite = torch.isfinite(g)
        count = int(finite.sum().item())
        item = dict(status='finite' if count == g.numel() else 'nonfinite',
                    elements=g.numel(), finite=count,
                    nan=int(torch.isnan(g).sum().item()),
                    positive_inf=int(torch.isposinf(g).sum().item()),
                    negative_inf=int(torch.isneginf(g).sum().item()), dtype=str(g.dtype))
        if detailed:
            values = g[finite].double()
            item.update(finite_nonzero=int(torch.count_nonzero(values).item()),
                        finite_norm=finite_value(values.norm().item()),
                        finite_max_abs=finite_value(values.abs().max().item()) if values.numel() else None)
        records[name] = item
    return dict(trainable_tensors=len(records),
                missing=[n for n, v in records.items() if v['status'] == 'missing'],
                nonfinite=[n for n, v in records.items() if v['status'] == 'nonfinite'],
                tensors=records)


def snapshot_raw_gradients(model, path):
    """Preserve exact observed gradients for offline inspection; do not edit gradients."""
    import torch
    with path.open('xb') as f:
        torch.save({n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None}, f)


def snapshot_rng(torch):
    import hashlib
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()
                      for s in torch.cuda.get_rng_state_all()])
