# Stage I paired architecture pilot preparation

**ID:**009d / D-049. **Date:**2026-10-16. **Status:** Offline preparation only; execution unapproved.

The researcher authorizes finalizing the evaluation protocol and preparing a paired Filtered-only pilot, not running it. Preserve all accepted source/target data, captions, thresholds, splits, weights and historical outputs. No held-out use, new model inference/training, unrelated decoding trial or rental/cost study.

## Scope and definitions

Reuse the exact512 train IDs and20 random+4 targeted validation cases from the D-048 manifest. Both arms train fresh from the pinned base at batch1/accum8 for2 epochs/128 steps; full200 validation supplies loss. Future generation is24 Filtered cases per arm,48 total. The historical full F model is not a matched architecture baseline.

A adapts24 decoder self/cross Q/V projections/294,912 parameters. B retains those and adds12 encoder local Q/V plus12 encoder global Q/V:48 modules/589,824 parameters. Verify complete module sets, tensor shapes, frozen base and active adapters on installed LED/PEFT using meta tensors. Independently initialize standard Kaiming A/zero B with a deterministic seed derived from common42 and canonical module name; decoder tensors must match across arms. Two fixed epoch orders use42+epoch and are checked through Accelerate on CPU. The extra scope changes capacity, optimizer geometry and encoder activation memory; it is not an isolated location-only test.

The active `STAGE_I_PILOT_EVALUATION_PROTOCOL.md` binds evidence references, atomic annotation rules, exact/lexical/semantic repetition, specific information/aspect/contrast coverage, source grounding, zero-caption and literal structure/EOS behavior. Teacher claims are unverified comparison candidates, not human truth. Reference-unit/claim labels and AI suggestions remain separate. Supersede D-048's proposed numerical quality-win thresholds before any new output; use predefined meaningful/mixed/regressed/inconclusive criteria, sample denominators and paired business uncertainty. No automatic quality winner.

## Execution guards and future checks

Default runner mode is offline. Feasibility/train/generate require --execute plus external explicit approval bound to config/preparation/protocol/code hashes. Runtime pins and immutable inputs must match. Training additionally requires both reviewed feasibility reports with matching code/preparation bindings. Generation requires both matching completed pilot trainings plus a researcher-confirmed source-unit freeze; unit review may accompany separately approved training but must precede all new outputs.

Future feasibility per arm is3 optimizer updates at accumulation8 plus one replay of update3 after restoring step2:4 physical updates/32 microbatches, padded5120/768 actual-content masks, no generation. Verify local/global encoder and self/cross decoder gradients, initial zero-A and zero-LR behavior, save/load optimizer/scheduler/scaler/RNG and numerical replay, actual memory/headroom; stop without fallback. It does not demonstrate fresh-process recovery or certify the unexecuted full Trainer loop. Feasibility weights are discarded. Pilot training and final24-case generation are separate approved modes; no automatic transition.

## Offline acceptance

Required deliverables: protocol, paired plan, new guarded runner/helpers/config, offline tests and logs, readiness report and self-contained review ZIP with SHA256 manifest. Verify same sampling/data, scope/counts/freezing, named initialization, epoch sampler, masks/labels, schedule, approval denials, reference-domain/quote provenance, structure and paired uncertainty. CPU/meta/synthetic-tensor checks do not count as GPU gradient/model numerical evidence. Preserve earlier artifacts and record D-049/progress. Stop after review preparation; wait for approval.
