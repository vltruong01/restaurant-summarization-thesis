"""Exploratory V3: standard model CE in both arms, no manual logits reduction.

Historical V2.1 gradient equivalence remains FAIL/UNRESOLVED. This module does
not certify equivalence or reuse the sequential/custom-autograd implementation.
"""
import torch

LAMBDA=.10
MARGIN=.10
TEMPERATURE=1.


def native_ce(model,batch,*,kind,observe=None):
    if observe is not None:observe(kind,batch)
    ce=model(**batch,use_cache=False).loss
    if ce.ndim!=0 or not torch.isfinite(ce).item():
        raise ValueError('Nonfinite/non-scalar native CE: '+kind)
    return ce


def objective(model,positive,negative=None,*,arm,observe=None):
    if arm not in ('ce','pair'):raise ValueError('Unapproved arm')
    ce=native_ce(model,positive,kind='positive',observe=observe)
    if arm=='ce':return ce,dict(CE=float(ce.detach()),negative_CE=None,contrastive=None)
    if negative is None:raise ValueError('Locked negative required')
    for key in ('input_ids','attention_mask','global_attention_mask'):
        if not torch.equal(positive[key],negative[key]):raise ValueError('Negative source differs')
    neg=native_ce(model,negative,kind='negative',observe=observe)
    # s(y)=-native mean valid-token CE; headers and EOS included, padding ignored.
    pair=torch.nn.functional.softplus((MARGIN+ce-neg)/TEMPERATURE)
    total=ce+LAMBDA*pair
    if not torch.isfinite(total).item():raise ValueError('Nonfinite joint objective')
    return total,dict(CE=float(ce.detach()),negative_CE=float(neg.detach()),
                     contrastive=float(pair.detach()))
