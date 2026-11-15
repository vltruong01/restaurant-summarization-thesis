# D-059 — One final Expanded-LoRA LED viability attempt

Status: **OFFLINE PREPARATION ONLY; GPU EXECUTION NOT AUTHORIZED**. Researcher selected this strategy after D-058. This supersedes the recommendation to finish immediately with existing models; it does not rewrite D-058 as GO or NO-GO. Preserve D-051 strict replay FAIL and every accepted dataset/checkpoint.

## Exact architecture and invariant settings

Pinned `allenai/led-base-16384`, revision `38335783885b338d93791936c54bb4be46bebed9`, 161,844,480 frozen backbone parameters. PEFT 0.15.2 `target_modules="all-linear"` must resolve to the following exact inventory in Transformers 4.51.3. A mismatch stops execution; no naming assumptions borrowed from other model families.

| Module family (i = 0…5) | Count | LoRA parameters |
| --- | ---: | ---: |
| `led.encoder.layers.i.self_attn.longformer_self_attn.{query,key,value}` | 18 | 221,184 |
| Same prefix, `{query_global,key_global,value_global}` | 18 | 221,184 |
| `led.encoder.layers.i.self_attn.output` | 6 | 73,728 |
| `led.decoder.layers.i.{self_attn,encoder_attn}.{q_proj,k_proj,v_proj,out_proj}` | 48 | 589,824 |
| `led.{encoder,decoder}.layers.i.{fc1,fc2}` | 24 | 737,280 |
| **Total** | **114** | **1,843,200 (228 tensors)** |

Rank8/alpha16/dropout0/bias none, scaling2, FP32 parameter storage, FP16 autocast, nonreentrant gradient checkpointing. No quantization. PEFT excludes the tied `lm_head`; shared/token/position embeddings, normalization and all original biases/weights stay frozen. Attention output projections are intentional targets; the vocabulary output head is excluded.

Offline meta construction uses only config, no pretrained tensors or model forwards. CPU-only adapter initialization and safetensors roundtrip validate exact keys/shapes/config. CPU initialization retains D-058's seed42 + canonical-name Kaiming-A/zero-B scheme. Explicitly disabling unused PEFT bitsandbytes 4/8-bit dispatchers avoids an optional-backend import failure and enforces ordinary torch.nn.Linear dispatch; this applies consistently in preparation and runtime. It changes no weights, precision or LoRA formula. Preserve the initial environment failure note.

Reuse D-058 accepted 2,000 F training / 200 validation examples, orders, optimizer AdamW2e-4/wd.01, betas(.9,.999), eps1e-8, physical batch1, accumulation8, clip1, fresh linear1,250/warmup63. Five epochs/1,250 updates maximum, save all five snapshots at250/500/750/1,000/1,250, no loss early-stop or best-loss reload. Fresh run only; never resume trained D-058 or feasibility adapters.

Unchanged source5,000/reviews4,000/captions750, labels768, K5; unchanged data, tokens, review/caption order, loss masking and BOS/global-attention behavior. Generation is the exact original beam4, `max_length=768` (including decoder-start), do_sampleFalse, no_repeat_ngram_size0, length_penalty1, early_stoppingTrue, forcedEOS2, use_cacheTrue. Evaluate only epoch3 and epoch5, 24 fixed validation cases each (20random/4targeted), total48. No other decoding experiment.

## Minimal feasibility gate and resource budget

After explicit hash-bound GPU authorization, rerun preflight in that environment. One feasibility invocation: exactly3 optimizer updates/24 accumulated microbatches, no replay update or generation. Use deterministic real training cases selected by maximum source length, maximum target length and first sorted zero-caption ID, cycling them without changing tokens. Pad to5,120/768 to exercise tensor dimensions; masked padding does not establish memory for 768 supervised tokens. Check actual module attachment, every adapter gradient, frozen-base gradients, finite loss, optimizer/scheduler steps, AMP updates, device memory, and complete adapter/optimizer/scheduler/scaler/RNG serialization and restoration. At zero-B initialization, first-step A gradients can be zero; require nonzero B and later A gradients. Initial warmup LR is zero as in D-058. No fourth replay update or repeat feasibility is hidden in the gate; D-051's update-replay failure remains unresolved history.

Measured headroom must be at least15% using peak Torch reservation plus sampled non-Torch device use. OOM, nonfinite/skipped update, integrity, deadline or scope mismatch stops with preserved evidence; no automatic fallback or retry. A future authorization may cover feasibility plus conditional F training/evaluation together: continue only after PASS under that same approval/config/runner/preflight/deadline. If only feasibility was authorized, stop there; a later training approval binds its report SHA. No extra approval checkpoint is imposed when conditional training was already explicitly authorized. Changing GPU/environment requires explicit revalidation, not a silent continuation. Checkpoint roundtrip is not claimed as strict stochastic update replay.

Planning, not a measurement: 1,843,200 adapters are3.125× D-058's589,824, but additional FP32 adapter weights/gradients/Adam moments are only about19.13MiB. Activation lifetimes, added paths and workspaces can dominate. D-058 used about2.65GiB device memory; plan roughly3–5GiB Torch reservation with possible additional device overhead at worst padded shapes. Local6GB is a candidate for the gate, not certified. A12–24GB GPU offers more margin if separately approved; no rental is performed. Host RAM16GB minimum/32GB comfortable, disk20GB free for this small-adapter run plus environment/cache; do not duplicate photos/datasets. Estimated one F run5–8h plus validation review/export versus observed D-0584h56; no direct module-count→runtime scaling or guarantee. Replace this estimate with measured feasibility microbatch throughput before full-run approval. Three matched runs reserve roughly15–24h of training plus evaluation/review; human turnaround is the principal seven-day scheduling risk.

## Scope, deadlines and authorization

The seven-day clock begins at the future explicit **execution authorization timestamp**, not at preparation, first optimizer step, or GO. Record aware UTC start/deadline exactly seven days apart. Enforce at each training loss computation, optimizer callback, feasibility microbatch and generation case. No future RO/U approval resets that clock. Expiration stops new work and preserves partial evidence; no claim of a completed matched study. Operations already in flight may take until the next safe check. Reserve day7 for completion and documentation rather than new searches.

Proposed authorization scope: one3-update gate, conditional on PASS one fresh F run≤1,250 updates and48 epoch3/5 outputs. Zero Teacher calls, held-out access or historical overwrite. Current authorization template is false. RO/U training is not part of the initial executable scope; after researcher-confirmed GO it requires explicit approval under the same architecture/protocol and original deadline. Maximum eventual three-condition training ceiling3,750 updates plus the3 gate updates, no sweeps or extra architecture trials.

Commands (only after actual approval):

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\run_stage_i_d059_v1.py feasibility --execute --approval <approval.json>
# Only after gate PASS and if the same approval explicitly permits conditional train/evaluate:
.\.venv-caption-pilot\Scripts\python.exe scripts\run_stage_i_d059_v1.py train --execute --approval <approval.json>
.\.venv-caption-pilot\Scripts\python.exe scripts\run_stage_i_d059_v1.py evaluate --execute --approval <approval.json>
```

All execution scopes must be explicitly present in approval. No unattended RO/U launch or automated quality GO. Local paths in integrity manifests bind this runtime; relocation needs equivalent-version/source/hash verification and a transparent authorization rebind. Do not edit a frozen manifest to make it pass.

## Prespecified quality decision

Use D-053 approved references and D-058 rubric unchanged. Benchmarks are saved historical F and D-058 epoch5 (24 each), reused without generation. Differences are practical comparisons, not isolated causal module effects. Report random20 and targeted4 separately. Core is important content, not mandatory verbatim reproduction; Supporting omissions are not automatic errors; excluded/uncertain units do not enter denominators.

For each business/epoch record output spans and exact REV/IMG_CAP/PHOTO/reference-unit bindings, supported/unsupported claims, covered Core units, specific useful details, preserved/lost opinion contrasts, lexical and semantic repetition, grounded visual usefulness, headers, natural versus forced EOS, censoring and semantic completeness. Keep software metrics, AI-assisted assessments and explicit researcher confirmation in separate fields. Do not fill labels or change references during prep. Report per-business paired improvements/regressions, random-sample uncertainty and n, not only averages. Inspect severe groundedness/contrast regressions individually.

GO requires researcher-confirmed substantial improvement in usable, informative, grounded summaries on random20, with no compensating severe factual/opinion/targeted-zero regressions. No automatic GO based on loss, length, headers or repetition alone. Mixed, inconclusive, poor substantive quality, or expiry ends LED optimization and returns to original matched RO/U/F checkpoints. Rare zero-caption behavior has only9train/1validation businesses; no augmentation, oversampling, fabricated evidence, Teacher rewrite or silent output repair is approved. Explicitly inspect the fixed zero-caption validation case; one success does not establish population reliability.

If GO, freeze one common checkpoint-selection policy before any subsequent RO/U output review; apply it identically to all conditions. Final F may be the rescue run only if it exactly matches that selected common protocol. Do not choose different output-quality policies per condition or present Expanded-F versus old decoder-only RO/U as the main comparison. D-058 remains supplementary. No rank/LR/staged/decode trials are authorized.

## Parallel offline thesis-completion plan

No held-out contents are opened now. Before any final test execution, lock the architecture/checkpoint policy, generation settings, metric/reporting plan and test-source integrity gate. Reuse accepted test input manifests only at the separately authorized evaluation stage; do not create test Teacher targets. Infer the three matched conditions, then the existing Shuffled evaluation control using the F model and frozen donor/count/token constraints. Audit donor mappings, shared reviews, overflow and reproducibility before generating; never tune from test outcomes. Test reference availability/approval is a prerequisite for reference-based metrics, not permission to invent references.

Prepare the existing representative30-business human evaluation as a separate final study: fixed declared sample plan, blinded output aliases, original source evidence/photos, rubric, reviewer consent/instructions, independent ratings and documented adjudication/provenance. D-053 validation24 and optional ChatGPT checks do not replace the formal30. Plan fields/report shells now only; no test IDs/content, human labels, selected output examples or references are fabricated. Compare paired conditions with uncertainty and preserve failures rather than silently dropping them.

Schedule: day1 gate/F start; days2–3 finish F and independent quality decision; days3–5 conditional matched RO/U only if GO and projected completion fits; days6–7 export/quality verification/method lock. Protect the separate final evaluation/human-review/Results/Discussion/defense time. If review or resources make this impossible, stop the improvement route on time and use matched historical models.
