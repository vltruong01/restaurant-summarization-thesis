"""Bounded native-GradScaler backoff; replay the same accumulation window, not extra updates.

Single-device only. No monkey-patching of PyTorch/Accelerate sources or scaler internals.
"""
from pathlib import Path

POLICY = dict(max_overflow_skips=8, minimum_scale=256.0, accumulation=8,
              retry_same_window=True, restore_window_RNG=True,
              gate_microbatches_max=88, train_microbatches_max=10064)


def capture_rng():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    import random
    import numpy as np
    import torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])


class Recovery:
    def __init__(self, folder, max_updates, check, write, append):
        if max_updates not in (3,1250):raise ValueError('Unexpected successful-update budget')
        self.folder=Path(folder);self.maximum=max_updates;self.check=check;self.write=write;self.append=append
        self.successful_updates=0;self.overflow_skips=0;self.microbatches=0
        self.window=[];self.rng=None;self.pending=False;self.replaying=False

    def begin(self):
        if self.window or self.pending:raise ValueError('Previous window not committed')
        self.rng=capture_rng()

    def microbatch(self):
        self.microbatches+=1
        maximum=POLICY['gate_microbatches_max'] if self.maximum==3 else POLICY['train_microbatches_max']
        if self.microbatches>maximum:raise ValueError('Dynamic AMP microbatch ceiling exceeded')

    def finish(self, model, optimizer, scaler, unscale, native_skip, repeat, max_norm=1., norm_type=2):
        """Return clipped finite grads. Caller performs exactly ONE successful optimizer update."""
        import torch
        from led_large_gradient_evidence_v3 import inspect_gradients, snapshot_raw_gradients
        from run_stage_i_d050_feasibility_v1 import fingerprint
        params=[p for p in model.parameters() if p.requires_grad]
        if len(params)!=456:raise ValueError('Expected all456 trainable adapter tensors')
        while True:
            self.check();unscale()
            evidence=inspect_gradients(model)
            if evidence['missing']:raise ValueError('Missing adapter gradient; not recoverable overflow')
            scale=float(scaler.get_scale())
            if not evidence['nonfinite']:
                norm=torch.nn.utils.clip_grad_norm_(params,max_norm,norm_type=norm_type,error_if_nonfinite=True)
                if not torch.isfinite(norm).item():raise ValueError('Nonfinite clipping norm')
                self.pending=True
                return norm
            i=self.overflow_skips+1
            self.write(self.folder/f'AMP_overflow_{i:02d}.json',dict(
                successful_updates=self.successful_updates,scale_before=scale,microbatches=self.microbatches,
                gradients=evidence,policy=POLICY))
            snapshot_raw_gradients(model,self.folder/f'AMP_overflow_{i:02d}_gradients.pt')
            if self.overflow_skips>=POLICY['max_overflow_skips'] or scale*.5<POLICY['minimum_scale']:
                raise ValueError('Bounded AMP overflow/scale limit reached; STOP, no fallback')
            before=[p.detach().cpu().clone() for p in params]
            opt_before=fingerprint(optimizer.state_dict())
            # GradScaler sees the inf flags from unscale_ and skips AdamW; update() halves the scale.
            native_skip()
            after=float(scaler.get_scale())
            if after!=scale*.5:raise ValueError('Native scaler did not halve scale after overflow')
            if fingerprint(optimizer.state_dict())!=opt_before:raise ValueError('Overflow changed optimizer state')
            if not all(torch.equal(p.detach().cpu(),b) for p,b in zip(params,before)):
                raise ValueError('Overflow changed adapter weights')
            self.overflow_skips+=1
            self.append(self.folder/'AMP_DYNAMIC.jsonl',dict(event='native_skip_backoff',
                scale_before=scale,scale_after=after,overflow_skips=self.overflow_skips,
                successful_updates=self.successful_updates,optimizer_unchanged=True,weights_unchanged=True,
                microbatches_before_retry=self.microbatches))
            optimizer.zero_grad(set_to_none=True)
            restore_rng(self.rng)
            self.replaying=True
            try:repeat()
            finally:self.replaying=False

    def commit(self, scaler):
        if not self.pending:raise ValueError('No finite gradient window ready for successful update')
        self.successful_updates+=1
        if self.successful_updates>self.maximum:raise ValueError('Successful update ceiling exceeded')
        self.append(self.folder/'AMP_DYNAMIC.jsonl',dict(event='successful_update',**self.state(scaler)))
        self.window=[];self.rng=None;self.pending=False

    def state(self, scaler):
        return dict(successful_updates=self.successful_updates,overflow_skips=self.overflow_skips,
                    actual_forward_backward_microbatches=self.microbatches,
                    scale=float(scaler.get_scale()),policy=POLICY)


def install_trainer(trainer, folder, check, write, append):
    """Install a local public clip hook; original Trainer train loop/scheduler stay intact."""
    from transformers import Seq2SeqTrainer
    recovery=Recovery(folder,1250,check,write,append)
    trainer.amp_recovery=recovery

    def clip(parameters, max_norm, norm_type=2):
        # Trainer's public clip hook is reached after all8 original backwards,
        # before on_pre_optimizer_step, AdamW, scheduler or global_step advancement.
        if len(recovery.window)!=8:raise ValueError('Expected exactly8 buffered microbatches')
        def repeat():
            for inputs in recovery.window:
                recovery.microbatch()
                Seq2SeqTrainer.training_step(trainer,trainer.model,inputs)
        def skip():
            trainer.optimizer.step() # AcceleratedOptimizer calls native scaler.step/update.
            if not trainer.accelerator.optimizer_step_was_skipped:
                raise ValueError('Overflow did not produce native skipped optimizer update')
        return recovery.finish(trainer.model,trainer.optimizer,trainer.accelerator.scaler,
            lambda:trainer.accelerator.unscale_gradients(trainer.optimizer),skip,repeat,max_norm,norm_type)
    trainer.accelerator.clip_grad_norm_=clip
    return recovery
