"""Inference-only CAD at raw LED logits, before native beam log_softmax.

One single-business beam4 call. Both KV caches follow the SAME beam ancestry.
No Transformers file, checkpoint, generation config or parameter is modified.
"""
from functools import wraps
import hashlib
import math

SCAFFOLD = '[BUSINESS_INFORMATION]\n\n[REVIEWS]\n\n[VISUAL_CONTEXT]\n'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contextless_source(source):
    markers = ('[BUSINESS_INFORMATION]', '[REVIEWS]', '[VISUAL_CONTEXT]')
    require(all(source.count(x) == 1 for x in markers), 'Ambiguous original scaffolding')
    require(source.startswith(markers[0]) and source.index(markers[1]) < source.index(markers[2]),
            'Unexpected source format')
    # Metadata (including business name/categories) is factual context too.
    # No invented "no photos" instruction and no original factual fields survive.
    return SCAFFOLD


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def mix_logits(conditional, contextless, alpha):
    import torch
    require(math.isfinite(alpha) and alpha in (0., .3), 'Only locked alpha0 / alpha0.3')
    require(conditional.shape == contextless.shape, 'Branch vocabulary/shape mismatch')
    require(bool(torch.isfinite(conditional).all()) and bool(torch.isfinite(contextless).all()),
            'Nonfinite raw branch logits')
    if alpha == 0:
        return conditional  # Bit-exact identity, no multiply/subtract rounding.
    result = (1. + alpha) * conditional.float() - alpha * contextless.float()
    require(bool(torch.isfinite(result).all()), 'Nonfinite CAD logits')
    return result


class CADSession:
    """Temporary hooks on the actual LED model with active LoRA modules.

    Encoder for scaffold runs once; its cache is independent from full-source
    encoder/cache. Native GenerationMixin owns beam search and constraints.
    _reorder_cache reorders BOTH caches with its exact selected beam indices.
    The full-prefix equality check prevents scoring a different beam's history.
    """
    def __init__(self, model, contextless_batch, alpha, beams=4, on_step=None):
        self.model = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.batch, self.alpha, self.beams = contextless_batch, alpha, beams
        self.on_step = on_step or (lambda row: None)
        self.prefix = self.previous = self.past = None
        self.inside = False
        self.steps = self.reorders = 0
        self.rows = []
        self.handles = []

    def __enter__(self):
        import torch
        from transformers.modeling_outputs import BaseModelOutput
        m = self.model
        require(not torch.is_grad_enabled(), 'CAD requires no_grad/inference_mode')
        require(all(not x.training for x in m.modules()), 'All modules must be in eval mode')
        require(m.config.model_type == 'led' and self.beams == 4, 'Only single-business LED beam4')
        require(self.batch['input_ids'].shape[0] == 1, 'One business per call')
        require(set(self.batch) == {'input_ids', 'attention_mask', 'global_attention_mask'}, 'Scaffold batch drift')
        enc = m.get_encoder()(**self.batch, return_dict=True)
        self.encoder = BaseModelOutput(last_hidden_state=enc.last_hidden_state.repeat_interleave(self.beams, 0))
        self.mask = self.batch['attention_mask'].repeat_interleave(self.beams, 0)
        self.prepare_original = m.prepare_inputs_for_generation
        self.reorder_original = m._reorder_cache
        # Restore instance attributes exactly, including absence of an override.
        self.old_attrs = {k: (k in m.__dict__, m.__dict__.get(k))
                          for k in ('prepare_inputs_for_generation', '_reorder_cache')}

        @wraps(self.prepare_original)
        def prepare(input_ids, *args, **kwargs):
            self.prefix = input_ids.detach().clone()
            return self.prepare_original(input_ids, *args, **kwargs)

        @wraps(self.reorder_original)
        def reorder(past_key_values, beam_idx):
            require(self.past is not None and self.previous is not None, 'Missing contextless cache')
            require(beam_idx.numel() == self.beams and bool(((beam_idx >= 0) & (beam_idx < self.beams)).all()),
                    'Invalid beam ancestry')
            self.past = self.reorder_original(self.past, beam_idx)
            self.previous = self.previous.index_select(0, beam_idx.to(self.previous.device))
            self.reorders += 1
            return self.reorder_original(past_key_values, beam_idx)

        m.prepare_inputs_for_generation = prepare
        m._reorder_cache = reorder
        self.handles.append(m.register_forward_hook(self._hook, with_kwargs=True))
        return self

    def _hook(self, module, args, kwargs, output):
        import torch
        if self.inside:
            return None
        require(not args and kwargs.get('labels') is None, 'No training / unexpected positional forward')
        require(all(not x.training for x in module.modules()), 'Eval mode drift')
        p = self.prefix
        require(p is not None and p.shape[0] == self.beams and bool((p[:, 0] == 2).all()),
                'Missing/wrong full decoder prefix')
        if self.previous is None:
            require(p.shape[1] == 1 and self.past is None, 'Unexpected first prefix')
        else:
            require(p.shape[1] == self.previous.shape[1] + 1 and torch.equal(p[:, :-1], self.previous),
                    'Conditional/contextless beam prefixes diverged')
            require(self.past[0][0].shape[2] == p.shape[1] - 1, 'Contextless KV length drift')
        tail = p if self.past is None else p[:, -1:]
        require(torch.equal(kwargs['decoder_input_ids'], tail), 'Native conditional cache prefix differs')
        require(output.past_key_values is not None, 'Conditional cache disabled')
        self.inside = True
        try:
            other = module(attention_mask=self.mask, encoder_outputs=self.encoder,
                           decoder_input_ids=tail, past_key_values=self.past,
                           use_cache=True, return_dict=True)
        finally:
            self.inside = False
        require(other.past_key_values is not None, 'Contextless cache disabled')
        conditional = output.logits[:, -1:, :]
        contextless = other.logits[:, -1:, :]
        combined = mix_logits(conditional, contextless, self.alpha)
        row = dict(step=self.steps, prefix_length=p.shape[1], prefix_sha256=tensor_sha(p),
                   conditional_contextless_prefix_exact=True, branches_finite=True,
                   conditional_min=float(conditional.min()), conditional_max=float(conditional.max()),
                   contextless_min=float(contextless.min()), contextless_max=float(contextless.max()),
                   combined_min=float(combined.min()), combined_max=float(combined.max()),
                   alpha=self.alpha, beam_reorders=self.reorders,
                   alpha0_raw_logits_bit_exact=bool(torch.equal(combined, conditional)) if self.alpha == 0 else None)
        self.rows.append(row)
        self.on_step(row)  # Durable receipt can be flushed before subsequent guards.
        self.steps += 1
        self.past, self.previous = other.past_key_values, p.detach().clone()
        if self.alpha != 0:
            output.logits = combined
        return output

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for name, (present, value) in self.old_attrs.items():
            if present:
                setattr(self.model, name, value)
            else:
                delattr(self.model, name)
        self.past = self.encoder = self.mask = self.previous = self.prefix = None
        return False
