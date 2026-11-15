# D-051 — Bounded B replay investigation

**2026-10-16. Researcher approved one B-only diagnostic execution; no pilot training.**

Preserve D-049/D-050 configuration, code, inputs and results unchanged. A versioned D-051 runner adds observation and writes all evidence before the replay verdict. Use the same pinned base, 48 modules/589,824 parameters, seed42, three fixed real train controls and order, physical batch1/accumulation8, FP32 weights/FP16 AMP, checkpointing, AdamW/scheduler/clipping, 5,120/768 padded masks and step-2 checkpoint save/restore. Three distinct updates plus one replay of step3: four physical calls/32 microbatches. B only; no automatic retry or additional diagnostic run.

Persist original/replayed step-3 entry/exit adapter tensors, optimizer/scheduler/scaler/RNG, pre/post-clipping gradient tensors and norms; each microbatch's full-logit fingerprint, loss, input/RNG fingerprints, accumulated-gradient fingerprints and memory. Snapshot tensors with independent CPU clones to prevent state_dict aliasing. Capture final peak allocated/reserved memory in finally even when execution fails. Retain exact checkpoint functions; change only isolated destinations/bindings and diagnostics.

The scientific acceptance test remains torch.allclose(original,replayed,rtol=1e-5,atol=1e-6), same orientation/dtype. Count failing elements using identical torch.isclose semantics. Compute errors in FP64: max absolute, RMS, relative L2 and max symmetric relative error with floor1e-12; separately count near-zero elements. Report per tensor/module/branch plus aggregate element denominators and parameter-update differences. Do not loosen tolerance based on results.

Localize by comparing entry states, matching input/RNG, complete forward-logit fingerprints/loss, accumulated backward gradients, unscaled/preclip and clipped gradients, then Adam moments/updates. Fingerprints establish equality/difference but not error magnitude; saved tensors permit quantitative comparisons. A backward-localized mismatch includes checkpoint recomputation and does not by itself prove a particular CUDA kernel.

Pre-execution interpretation: PASS means every adapter passes the original tolerance. CONDITIONAL is a researcher-facing readiness assessment when strict replay still fails but saved evidence quantifies small deviations and explains remaining reproducibility risks; it never converts the strict gate to PASS. FAIL means incorrect restoration, nonfinite/meaningful unexplained divergence or insufficient evidence to justify readiness. No new numeric smallness cutoff, relaxed tolerance or automatic pilot authorization. Report single-replay limitations and distinguish short-run numerical sensitivity from long-run quality/divergence.

No deterministic-policy/kernel/precision/architecture/hyperparameter/data change, held-out use, Teacher requests, decoding experiment, accepted checkpoint overwrite or pilot training. If a correction is needed, propose it and stop for approval. Deliver report, full numeric comparisons, runtime evidence, current readiness and self-contained review ZIP/SHA manifest. Stop after D-051.

## Completed outcome

One B-only execution completed with full evidence. Strict gate FAIL:18/589,824 elements,7 tensors,max3.7234e-6,RMS1.5267e-8. Entry1,151 leaves/full8 forward pairs/RNG exact; first divergence after encoder backward microbatch1. Whole-update relative L2 difference0.03515%, worst module0.12902%; CONDITIONAL readiness assessment only. No tolerance/policy change, retrospective PASS, pilot execution or extra run. Researcher must review/decide the unresolved gate before training approval. See D051 report, saved tensors and numeric comparisons.
