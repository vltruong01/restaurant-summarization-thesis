# D-050 — Bounded GPU feasibility of paired LoRA architectures

**2026-10-16. Researcher-approved execution scope: feasibility only.**

Execute the unchanged D-049 feasibility routine for decoder-only A and encoder+decoder B on the available local GPU. Retain the exact prepared config, datasets, adapter initialization, attention masks, precision and optimizer/scheduler. Use the three frozen real train controls (longest source, longest target, first sampled zero-Filtered-caption business), padded to 5,120/768 with complete content and correct masks.

Per arm, run three optimizer updates with eight microbatches per update, save at step 2, restore and replay step 3 once. Thus there are three distinct optimizer steps and four physical update calls / 32 microbatches. No generation or pilot training. A/B use independent fresh pretrained loads and matching common decoder initialization. No simultaneous models compete for GPU memory.

The D-050 observation wrapper adds durable logs, frozen/trainable parameter hashes, optimizer-update records, exact checkpoint-state fingerprints, mask/special-token assertions and device telemetry. It delegates the actual forward/backward, clipping, update, scheduler, checkpoint and replay math to the unmodified D-049 runner. It must not change loss, data order, parameters, lengths, precision, LoRA scope or optimization settings.

Record actual attached modules/counts, finite/nonzero gradients by encoder local/global and decoder branch, losses, scaler/optimizer/scheduler behavior, allocated/reserved/device memory, and exact/tolerance-based replay. Respect initial LoRA B=0, initial scheduled LR=0 and delayed A-gradient activation. Replay tolerance remains rtol=1e-5 / atol=1e-6; disclose exact equality separately. Same-process replay is not fresh-process recovery.

Stop the affected run on failure without fallback; preserve every attempt. Frozen backbone tensors must remain byte-identical and gradient-free. Verify original imported datasets/checkpoints and D-049 preparation hashes before and after execution. Do not access held-out data or change accepted targets, filtering or historical artifacts.

Deliver the D-050 comparison/report, logs/config snapshots, updated readiness and self-contained ChatGPT ZIP/SHA manifest. Even two feasibility passes do not authorize the 512-business pilot or establish quality improvement. Stop after D-050; training and generation require later approval.

## Completed execution outcome

A passes; B completes all32 microbatches with correct encoder/local/global/decoder gradients and frozen backbone, but replay fails unchanged tolerance. Model/optimizer/scheduler/scaler/RNG restore is exact; differences are encoder-localized. B numerical losses and allocator maxima are missing due the prepared post-assert serialization path. Preserve all evidence and mark overall paired-pilot readiness blocked. Further bounded replay instrumentation is proposed, not executed or approved. No pilot training/generation follows D-050. See STAGE_I_D050_GPU_FEASIBILITY_REPORT.md under outputs/stage_i/d050_gpu_feasibility_v1.
