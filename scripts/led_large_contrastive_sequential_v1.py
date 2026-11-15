"""Candidate production objective; explicit RNG replay, no optimizer ownership.

The same scalar loss can be scaled by native GradScaler and accumulated normally.
Full pretrained LED/CUDA/FP16 qualification is a separate mandatory precondition.
"""
from contextlib import contextmanager
import copy
import hashlib
import pickle
import random
import numpy as np
import torch

WEIGHT = .10
MARGIN = .10
TEMPERATURE = 1.
ATOL = RTOL = 1e-6

def capture_rng(device):
    return dict(python=copy.deepcopy(random.getstate()), numpy=copy.deepcopy(np.random.get_state()),
                cpu=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None)

def restore_rng(state, device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device)

def fingerprints(state):
    def h(x):return hashlib.sha256(x).hexdigest()
    return dict(python=h(pickle.dumps(state['python'],protocol=4)),numpy=h(pickle.dumps(state['numpy'],protocol=4)),
                cpu=h(state['cpu'].cpu().numpy().tobytes()),
                cuda=h(state['cuda'].cpu().numpy().tobytes()) if state['cuda'] is not None else None)

def same_rng(a,b):return fingerprints(a)==fingerprints(b)

def _notify(recorder, event):
    if recorder is not None:recorder(event)

def _mode_signature(model):
    active=getattr(model,'active_adapters',None)
    if callable(active):active=active()
    return dict(modes=tuple((n,m.training) for n,m in model.named_modules()),
                use_cache=getattr(getattr(model,'config',None),'use_cache',None),active=copy.deepcopy(active))

def _check_batch(positive,negative):
    keys={'input_ids','attention_mask','global_attention_mask','labels'}
    if set(positive)!=keys or set(negative)!=keys:raise ValueError('Unexpected model fields')
    for k in keys-{'labels'}:
        if not torch.equal(positive[k],negative[k]):raise ValueError('Positive/negative source or mask changed')
    if positive['input_ids'].shape[0]!=1:raise ValueError('Physical batch must be one')
    for batch in (positive,negative):
        y=batch['labels'];valid=y[y!=-100]
        if (valid.numel()<2 or valid[0].item() in (0,1,2) or valid[-1].item()!=2 or
                (valid==0).any() or (valid==1).any() or (valid==2).sum().item()!=1):
            raise ValueError('BOS-free content/EOS label policy differs')
        if (batch['global_attention_mask'].sum(1)!=1).any():raise ValueError('Global attention differs')

@contextmanager
def _autocast(device,enabled,dtype):
    if device.type=='cuda':
        # Prevent no_grad scouts from populating a cast cache reused by grad recomputation.
        with torch.autocast('cuda',enabled=enabled,dtype=dtype,cache_enabled=False):yield
    else:
        yield

def _score(model,batch):
    # Model performs shift_tokens_right once. No manual decoder_input_ids.
    output=model(**batch,use_cache=False)
    score=-output.loss.float()
    if score.ndim!=0 or not torch.isfinite(score).item():raise ValueError('Nonfinite/non-scalar forward CE')
    return score

def _check_score(score,expected):
    if not torch.isfinite(score).item() or not torch.allclose(score.detach(),expected,atol=ATOL,rtol=RTOL):
        raise ValueError('Scout/replay score mismatch; no approximate/stale coefficient fallback')

class _SequentialPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx,model,positive,negative,recorder,*params):
        _check_batch(positive,negative)
        if not model.training:raise ValueError('Pair objective requires original training/dropout mode')
        ctx.model,ctx.positive,ctx.negative,ctx.recorder=model,positive,negative,recorder
        ctx.device=params[0].device
        if any(p.device!=ctx.device for p in params):raise ValueError('Single-device parameters required')
        ctx.amp_enabled=torch.is_autocast_enabled() if ctx.device.type=='cuda' else False
        ctx.amp_dtype=torch.get_autocast_gpu_dtype() if ctx.device.type=='cuda' else torch.float32
        ctx.mode=_mode_signature(model)
        ctx.before_pos=capture_rng(ctx.device)
        try:
            with _autocast(ctx.device,ctx.amp_enabled,ctx.amp_dtype):sp=_score(model,positive)
            ctx.before_neg=capture_rng(ctx.device)
            with _autocast(ctx.device,ctx.amp_enabled,ctx.amp_dtype):sn=_score(model,negative)
            ctx.after_pair=capture_rng(ctx.device)
            ctx.sp,ctx.sn=sp.detach(),sn.detach()
            z=(MARGIN-ctx.sp+ctx.sn)/TEMPERATURE
            ctx.coefficient=(WEIGHT/TEMPERATURE)*torch.sigmoid(z)
            value=-ctx.sp+WEIGHT*torch.nn.functional.softplus(z)
            if _mode_signature(model)!=ctx.mode:raise ValueError('Model/cache/adapter mode changed')
            ctx.save_for_backward(*params)
            _notify(recorder,dict(event='scout',positive_score=float(sp),negative_score=float(sn),true_loss=float(value),
                coefficient=float(ctx.coefficient),before_positive=fingerprints(ctx.before_pos),
                before_negative=fingerprints(ctx.before_neg),after_pair=fingerprints(ctx.after_pair)))
            return value
        except Exception as error:
            _notify(recorder,dict(event='scout_failure',error=repr(error),rng=fingerprints(capture_rng(ctx.device))))
            raise

    @staticmethod
    def backward(ctx, upstream):
        params=ctx.saved_tensors
        entry=capture_rng(ctx.device)
        try:
            if _mode_signature(ctx.model)!=ctx.mode:raise ValueError('Model state differs before replay')
            restore_rng(ctx.before_pos,ctx.device)
            with torch.enable_grad():
                with _autocast(ctx.device,ctx.amp_enabled,ctx.amp_dtype):sp=_score(ctx.model,ctx.positive)
                _check_score(sp,ctx.sp)
                gp=torch.autograd.grad(-upstream*(1+ctx.coefficient)*sp,params,allow_unused=False)
            if not same_rng(capture_rng(ctx.device),ctx.before_neg):raise ValueError('Positive replay RNG differs')
            # The first activation graph is freed by autograd.grad before the next forward.
            del sp
            restore_rng(ctx.before_neg,ctx.device)
            with torch.enable_grad():
                with _autocast(ctx.device,ctx.amp_enabled,ctx.amp_dtype):sn=_score(ctx.model,ctx.negative)
                _check_score(sn,ctx.sn)
                gn=torch.autograd.grad(upstream*ctx.coefficient*sn,params,allow_unused=False)
            if not same_rng(capture_rng(ctx.device),ctx.after_pair):raise ValueError('Negative replay RNG differs')
            del sn
            if _mode_signature(ctx.model)!=ctx.mode:raise ValueError('Model state changed during replay')
            gradients=tuple(a+b for a,b in zip(gp,gn))
            # Finite gradients are checked at the full effective-batch boundary by native AMP;
            # checking scaled individual components here would reinstate the old diagnostic defect.
            _notify(ctx.recorder,dict(event='replay_gradients_returned',gradient_tensors=len(gradients),
                all_replayed_scores_match=True,replayed_end_rng=fingerprints(capture_rng(ctx.device))))
        except Exception as error:
            _notify(ctx.recorder,dict(event='replay_failure',error=repr(error),rng=fingerprints(capture_rng(ctx.device))))
            raise
        finally:
            restore_rng(entry,ctx.device)
            if not same_rng(capture_rng(ctx.device),entry):raise ValueError('Backward-entry RNG restoration failed')
            _notify(ctx.recorder,dict(event='backward_rng_restored',fingerprints=fingerprints(entry)))
        return (None,None,None,None,*gradients)

def loss(model,positive,negative=None,*,weight=WEIGHT,recorder=None):
    if weight==0.:
        # Exact existing CE delegation. No scouts, negative forward or alternate reduction.
        result=model(**positive,use_cache=False).loss
        if not torch.isfinite(result).item():raise ValueError('Nonfinite CE-control forward loss')
        return result
    if weight!=WEIGHT:raise ValueError('Fixed pair weight; no hyperparameter search')
    params=tuple(p for p in model.parameters() if p.requires_grad)
    if not params:raise ValueError('No trainable parameters')
    return _SequentialPair.apply(model,positive,negative,recorder,*params)
