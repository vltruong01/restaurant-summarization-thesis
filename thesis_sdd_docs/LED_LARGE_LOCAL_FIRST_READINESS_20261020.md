# LED-large: local-first readiness and one Filtered proposal

2026-10-20 ICT. Current instruction: **OFFLINE preparation only**, local first.
This updates readiness, not scientific settings. Reuse the original
[full proposal](LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md) and prepared
data/runner; do not restart preparation or Unlikelihood. The preceding explicit
v3 gate approval is preserved as history, not erased or treated as full-training
approval. No model forward/backward/generation occurred in this local review.

## Initialization decision

Choose **one initialization: official `allenai/led-large-16384`**, revision
`cd59d11c3528415b7dda4dfc95cc8f138aceda2e`. Expected binary1,839,633,783bytes,
SHA256 `b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081`.
The local binary has not been downloaded/verified. The earlier Colab v2 loader
recorded that binary/tensor-inventory check passing before its gradient failure.
The official card describes BART-large initialization extended for long context;
it is not an already Yelp-fine-tuned summarizer.
[Pinned official card](https://huggingface.co/allenai/led-large-16384/blob/cd59d11c3528415b7dda4dfc95cc8f138aceda2e/README.md).

Amazon candidate assessed, not selected: `vltruong01/amazon-all-beauty-led-large-summarization`,
revision `4fdc41ed9bd4fbebe4fd6de131af1f27181f51ea`. Saved author artifacts report
full review-summarization fine-tuning,1,567/196/196examples,3epochs,lr1e-5,
source4096/target192 and synthetic references. Task transfer is plausible, not
demonstrated for our Yelp caption-plus-review targets. Short text-only targets
could bias against our visual section; that is a hypothesis, not measured evidence.
Exact upstream backbone revision, selected training step, teacher version and full
training environment are not established. Do not run a second initialization.
[Pinned author artifacts](https://huggingface.co/vltruong01/amazon-all-beauty-led-large-summarization/tree/4fdc41ed9bd4fbebe4fd6de131af1f27181f51ea).

Both configs:12encoder/12decoder layers,hidden1024,FFN4096,16heads,
encoder16384/decoder1024 positions,vocabulary50265. Rechecked17 cached metadata
hashes offline against their recorded pins; this is not a fresh Hub claim or
independent confirmation of the author's training history. Original tokenizer
compatibility evidence remains preserved: original Yelp token IDs/merges retained,
no truncation; Amazon's saved192-token truncation incident is not hidden.

## Verified local readiness and environment choice

| Item | Local observation | Implication |
|---|---|---|
| GPU | RTX3050 Laptop,6,144MiB; driver596.36 | Candidate only; fit not tested |
| Host RAM |25,463,480,320bytes ≈23.7GiB | Below original32GB planning preference; not an empirical failure |
| Disk free |102,701,346,816bytes ≈95.6GiB | Above original40GB planning allowance at inspection |
| Core dependencies | All seven pins match | CPU/offline compatibility check only |
| Official large weights | Absent locally | Download/hash verification needed before model execution |
| Preparation regression |18/18PASS | No model/GPU execution |

Use the existing expanded scope:r8/alpha16/dropout0,**228 projections,
456LoRA tensors,4,915,200trainable parameters**,459,801,600frozen backbone parameters.
Inventory includes encoder local/globalQKV,attention outputs,decoder self/crossQKVO
and encoder/decoderFFN layers; excludes LM head and tied/shared embeddings.
Actual installed META/PEFT attachment and serialization evidence is retained. D059
is an engineering reference, not proof the scope/lr is optimal for large.

FP32 backbone storage alone≈1.71GiB; adapter/gradient/Adam storage≈75MiB. Activations,
Longformer workspaces,decoder logits and beam caches cannot be inferred by multiplying
the base model's observed peak by its parameter ratio.15%headroom on6GiB leaves
roughly5.1GiB for conservative measured occupancy. Actual local fit/throughput remain
unknown; no GPU feasibility result is claimed from this inventory.

**Execution order proposed:** local integrity/model-file preflight → separately
authorized bounded local gate → if memory/hardware cannot satisfy unchanged settings,
retain that failure and prepare the same gate on Colab T4. A numerical gradient
failure is not automatically a hardware-capacity failure: do not silently rerun it
on Colab as a cure. Colab v2 already stopped at its first gradient check, with no
completed optimizer update; this was not an OOM or a quality NO-GO. The v3 diagnostic
logs now prepared distinguish missing gradients from NaN/Inf. Its GPU execution is
not performed during this offline review. No precision/batch/token reduction fallback.

## One meaningful Filtered viability run, after feasibility and authorization

Fresh official-large backbone/adapters; original F2,000train/200validation,5epochs/
1,250updates,warmup63. CE only;AdamW2e-4,WD.01,batch1/accum8,FP32parameters+FP16AMP,
seed42/frozen order,non-reentrant checkpointing,linear scheduler. Save all5epochs;
no best-loss reload/early stopping. Original5000source/4000review/750caption budgets,
K5,768labels,BOS-global mask and label ignore=-100 remain unchanged.

Feasibility ceiling3updates/24microbatches plus<=2train-input memory traces;
>=15%measured headroom; actual gradients/AMP/frozen backbone/masks/checkpoint
restoration required. Gate adapters are discarded before one fresh training run.
No gate replay or automatic numerical retry. A gate-only approval does not authorize
this full-data run under changed runner bindings.

Evaluate ONLY epoch3and5:24fixed validation cases each,20random+4targeted reported
separately,48outputs. Original beam4/do_sampleFalse/max_length768,no-repeat0.
Compare historicalF,D058B and D059Expanded as practical, non-causal benchmarks;
use D053 approved Core/Supporting/Exclude units and actual reviews/caption/photo
evidence. No held-out contents or new Teacher/BLIP2/annotation generation.

GO requires researcher-confirmed substantive gains in business-specific Core
coverage,opinion synthesis,grounded useful visuals/coherence and reduced harmful
semantic repetition,without severe factual/opinion/completion regressions. Headers,
loss,length or fewer exact repeats alone do not establish GO. Check unchanged9train/
1validation zero-caption cases. Separate software metrics,AI-assisted assessment and
researcher labels. Mixed/inconclusive/poor quality or missed time limit is not adoption.

If F is accepted, obtain separate approval for matched RO/U using the SAME backbone,
LoRA,initialization,optimizer/schedule and common epoch3or5 selection. Shuffled uses
the exact selected F checkpoint. Historical matched LED-baseRO/U/F remain primary
until the complete matched large family is accepted; no primary comparison mixes
largeF with old baseRO/U.

## Timeline and stop conditions

No historical clock is reset. Existing v2/v3 diagnostic deadline remains
**2026-10-21T17:11:48+00:00 =2026-10-22 00:11:48ICT**. Original proposed large-family
freeze remains2026-10-26T00:00:00Z. Exact defense date is still not recorded; the
earlier “more than three weeks” estimate is not a confirmed date. Confirm that the
approved schedule preserves>=14days for held-out evaluation,Shuffled,30-business
human study,writing/defense before any fresh full-training approval. Do not extend
deadlines to accommodate local slowness. Use measured gate timings to reject an
unfinishable schedule, not unmeasured promises of local/Colab speed.

Unlikelihood remains CLOSED_NUMERICAL_FEASIBILITY_FAILURE,quality not evaluated;
no patch/retry. D051 strict replay FAIL and D058–D060/UL/Colab-v2 failures,all targets,
datasets,references and checkpoints preserved. Current deliverable is OFFLINE
readiness only; no GPU/model execution or automatic training follows this document.

Evidence: `outputs/stage_i/led_large_local_first_review_20261020/`.
