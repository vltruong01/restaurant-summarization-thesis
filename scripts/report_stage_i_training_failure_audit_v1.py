"""Write offline audit/proposal from verified artifacts. No model execution or training."""
import json
from pathlib import Path
import statistics

from stage_i_decoding_diagnostics import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/training_failure_audit_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    for name in ['STAGE_I_TRAINING_FAILURE_AUDIT.md','STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md','proposed_training_config.json','OFFLINE_EXECUTION_NOTE.json']:
        if (OUT/name).exists():raise FileExistsError('Preserve existing report; choose a new version')
    arch=read(OUT/'architecture_audit.json');opt=read(OUT/'optimization_audit.json')
    data=read(OUT/'dataset_target_audit.json');pilot=read(OUT/'proposed_pilot_manifest.json')
    saved=read(OUT/'saved_prediction_audit.json');integrity=read(OUT/'AUDIT_INTEGRITY.json')
    config=read(ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2/configs/stage_i_lora_led_training_approved_v2.json')
    b=read(ROOT/'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1/paired_summary.json')
    audit='''# Stage I Training Failure Audit

Date: 2026-10-16, Asia/Saigon. D-048. **OFFLINE AUDIT COMPLETE; no new training or inference.**

## Main finding and evidence boundary

No critical implementation fault was found in the inspected code, accepted input/label bindings, saved optimizer/checkpoint states or inference settings that establishes the cause of the poor summaries. Training did occur. The strongest design concern is **decoder-only adaptation with a frozen encoder**, but its causal contribution remains untested. Adding encoder adapters is a hypothesis-driven architecture experiment, not correction of a proven broken training run.

Verified facts below come from the actual imported `run_stage_i_lora_led_cloud_v2.py`, saved arguments/configs and checkpoint states; installed runtime code matches the archived package pins. The model architecture was inspected on **meta tensors only**: no pretrained model was loaded, no forward/backward/generate call was made. Optimizer/tensor inspection was CPU-only. All275 original archive-manifest files retain their SHA256 values. No test content, Teacher requests, data/weight edits or further decoding experiment.

## 1. Actual modules and frozen weights

| Component | Exact suffix / scope | Current adapted projections | Trainable parameters |
| --- | --- | ---: | ---: |
| Decoder self attention | `led.decoder.layers.{0..5}.self_attn.{q_proj,v_proj}` | 12 | 147,456 |
| Decoder cross attention | `led.decoder.layers.{0..5}.encoder_attn.{q_proj,v_proj}` | 12 | 147,456 |
| Encoder local attention | `led.encoder.layers.{0..5}.self_attn.longformer_self_attn.{query,value}` | 0 | 0 |
| Encoder global attention | same prefix, `{query_global,value_global}` | 0 | 0 |
| Total | r8 / alpha16 / dropout0 / bias none | **24** | **294,912** |

There are48 trainable A/B tensors, approximately0.18189% of the161,844,480-parameter base plus adapters. All base weights are frozen, including encoder, decoder base matrices, shared embeddings and tied LM head. No feed-forward, normalization, key or output-projection adapters are present.

PEFT's configured list matches module names/suffixes, not the conceptual phrase 'query/value everywhere'. LED encoder projections are named `query/key/value` and `query_global/key_global/value_global`; only decoder projections use `q_proj/v_proj`. The actual runner explicitly asserts24 decoder matches and zero encoder adapters. Thus this scope was known/enforced, not an unexpected silent PEFT failure. See `architecture_audit.json`, all matched/trainable names, actual adapter configs and [pinned PEFT matching code](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/tuners/tuners_utils.py).

A frozen encoder still processes every unmasked source token; decoder cross attention can attend to all encoder positions. BOS-only global attention does **not** mean the decoder reads only BOS. Freezing limits learned adaptation, not source visibility. Frozen weights and LoRA branches are distinct, as in the [original LoRA method](https://arxiv.org/abs/2106.09685).

## 2. Input, teacher forcing, masks and generation audit

| Check | Verified result | Interpretation |
| --- | --- | --- |
| Dataset identity/alignment | Same six accepted Stage H NPZ / Stage G JSONL hashes; 2,000 train +200 validation per condition, unique aligned IDs; full shared selected review block/IDs and teacher consensus agree | No supported input-target shuffle, split or wrong-condition mapping bug |
| Source and labels | Logical source <=5000; complete BOS0/EOS2 arrays; labels <=768; longest existing target656 | No silent truncation in reader/collator or generation cap causing target-label loss |
| Encoder padding | Right padding ID1 to1024 multiple; logical5000 may pad to5120; padding attention0 | Padding shape is not extra content or a source-budget violation |
| Global attention | Exactly first real BOS has global mask1; other source tokens local; six attention windows1024 | Matches the established LED summarization setup; no missing caption tokens |
| Labels | Only right-padding positions become -100; actual labels remain intact | CrossEntropyLoss ignores -100; real BOS/EOS participate in labels |
| Decoder shift | LED internally shifts labels right, replacing masked positions with pad; decoder-start2 then original BOS0; predictions correspond to unshifted labels | No double shift or one-token alignment defect found |
| Decoder attention mask | Not separately provided; right-padded teacher-forcing with causal attention and ignored pad losses | Pads follow all supervised tokens and cannot influence earlier causal positions; not evidence of a padding-label bug |
| Training versus inference | Training eval/loss uses teacher forcing/cache false; final generation uses eval/no-grad, FP16 autocast/cache true; same pinned base/accepted adapter/tokenizer | Intended operational distinction; free-running behavior is not guaranteed by teacher-forced loss |
| Decoder settings | Baseline beam4/no-repeat0, A beam1/no-repeat0, B beam4/no-repeat4; all deterministic/no-sampling/max_length768 including decoder-start | B's only intentional baseline change is no-repeat4; no max_new_tokens769 cap |

Current rechecks cover every packed row's complete special-token endpoints and length bounds, exact frozen file hashes, cross-condition review text/ID/consensus equality, and20 CPU collator/decoder-shift boundary examples. Earlier accepted H/cloud audits independently checked complete raw source/target token arrays and all600 baseline prediction mappings. The latest A/B binding checks preserve the exact72 full inputs. No new model inference was needed. The complete current checks and per-row source/target hashes are in `dataset_target_audit.json` and `input_target_bindings.jsonl`.

The pinned [LED implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/led/modeling_led.py) implements the decoder shift and ignored-label loss; [LED documentation](https://huggingface.co/docs/transformers/v4.15.0/model_doc/led) describes BOS global attention for summarization. Installed exact function excerpts/hashes are supplied for inspection. BOS-only attention is a defensible fixed choice, not proof that encoder adaptation is sufficient.

## 3. Optimizer, accumulation, scheduler and selection

| Condition | Actual steps / epochs | Validation loss epochs1 ->5 | Adam states at step1250 | Best checkpoint |
| --- | --- | --- | ---: | --- |
'''
    for c,o in opt.items():
        ev=[r for r in o['trainer_state']['log_history'] if 'eval_loss'in r]
        audit+=f'| {c} | 1250 /5 | {ev[0]["eval_loss"]:.4f} -> {ev[-1]["eval_loss"]:.4f} | {o["optimizer_tensor_state_count"]} | 1250 |\n'
    audit+='''
Physical batch1, accumulation8, one process gives effective8 businesses per update.2,000/8=250 updates/epoch; five epochs=1,250. All48 Adam tensor states have step1250, finite/nonzero first moments; all48 adapter tensors changed from1000 to1250. Best adapters are byte/tensor consistent with1250, not an untrained/default adapter.

AdamW initial LR2e-4, decay0.01, betas0.9/0.999, eps1e-8, gradient clipping1.0; linear schedule with ceil(1250*0.05)=63 warmup steps. All125 logged LR values per run match the LR used for the preceding update; final stored LR is0, scheduler last_epoch1250. The slight pre-update versus post-scheduler logging offset is not an LR bug. AMP scale65536, growth tracker1250 and all48 Adam steps1250 support successful optimizer updates; these records are not a replay of every historical microbatch gradient.

The custom loss passes labels directly to LED and disables num_items_in_batch forwarding (`model_accepts_loss_kwargs=False`). Installed Transformers4.51.3 divides the mean microbatch loss by8; its Accelerator is created with default accumulation1, so the inspected path does not double-divide by8. Exact `Trainer.training_step`, accelerator construction and `Accelerator.backward` are preserved. Loss averages each business's supervised-token mean; it is not a pooled-token global loss. This is a weighting choice, not a discovered scaling fault. See [pinned Trainer implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py).

Evaluation/save occurred at250/500/750/1000/1250, all five losses improve, so patience2 did not trigger early stopping. Selection by lowest validation teacher-forced loss chose1250 in all conditions. No in-training free-running generation metric selected checkpoints (`predict_with_generate=False`). This is the accepted selection rule, with a known quality gap: falling loss does not certify factual free-running output. Losses across RO/U/F have different target content and are not factual-quality rankings.

The cloud runner differs from its original by only `label_names=['labels']`, fixing PEFT signature visibility for eval_loss. That fix is present in all successful runs; preserved earlier failed attempts must not be mistaken for the successful training implementation. Future encoder/decoder scope would require changing the current24-module assertions and fresh optimizer geometry; it cannot be activated just by passing the old runner a larger module list.

## 4. GPU memory: plausible, not evidence of skipped learning

Frozen base weights occupy roughly617.4MiB in FP32; current trainable tensors/gradients/Adam moments are only a few MiB. Batch1, mixed precision, non-reentrant gradient checkpointing, and no encoder parameter gradients reduce memory. Checkpointing conserves saved activations; it does not remove the forward reading of long sources. Exact historical graph activation lifetimes were not captured and are not inferred solely from this footprint.

Archived training peaks: maximum Torch allocated1.4351GiB, reserved2.2734GiB, sampled whole GPU2712MiB. Sampling every~10s may miss short device peaks. The optimizer/tensor-change evidence contradicts 'training never updated'. GPU compute utilization and memory occupancy are different measurements; low occupancy is neither proof of good summaries nor a reason to enlarge the model to fill VRAM.

Encoder LoRA would introduce backward paths through long encoder activations. Doubling the small adapter parameter count does not predict the activation-memory increase. **New-scope feasibility is unmeasured**; the current decoder-only footprint does not establish that the local6GB device fits it.

## 5. Target patterns versus student behavior

| Split / condition | Targets | Exact repeated-sentence outputs | Adjacent-word outputs | Mean adjacent-pair fraction | Mean target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for k,v in data['teacher_target_statistics'].items():
        audit+=f'| {k} | {v["targets"]} | {v["exact_repeated_sentence_outputs"]} | {v["adjacent_word_repeated_outputs"]} | {v["mean_adjacent_word_pair_fraction"]:.6f} | {v["target_tokens"]["mean"]:.2f} |\n'
    audit+='''
There are **0/6600 teacher targets with exact sentence repetition** under the same diagnostic. Some adjacent repeats exist in teacher text (including possible legitimate lexical/entity patterns); they are not automatically factual errors. Mean teacher adjacent-pair fractions are around0.0002-0.0003 across the complete splits, while random20 student means are around0.016-0.025. Student pathological repetition is not explained as literal copying of many repeated target sentences.

Targets do share conventional discourse. In2,000 training review consensuses, 'food opinions are mixed' occurs189 times(9.45%), 'service experiences vary'358(17.9%), and 'value opinions vary'147(7.35%). In random20 Baseline review outputs, 'service experiences vary' appears12/20 RO,17/20 U,14/20 F, versus3/20 of their teacher references. B still has11/20,17/20,14/20. This descriptive phrase concentration supports generic-template imitation as a hypothesis; a phrase is not itself bad or proof of a causal training mechanism. Templates are present, but the model overuses them and drops the source-specific content around them.

Training target means244.50/338.85/317.39 tokens (RO/U/F) versus validation192.35/286.15/266.08 show a descriptive length shift; every label fits768. Adding visual sections increases label length, while validation loss averages all supervised tokens rather than explicitly rewarding claim-level evidence alignment. The exported evidence IDs and photo verification records are auxiliary provenance, not an extra supervised alignment loss in the final plain-text student targets. This may weaken source/claim binding, but was the accepted design and is not automatically defective.

There are **9 zero-Filtered-caption training businesses out of2000 (0.45%)**, and1/200 validation. All ten teacher visual sections are correctly empty. The student fails the one validation zero case in Baseline/A/B. Scarce exposure is a plausible contributor; one validation failure is not an estimated population failure rate. Do not alter class mix, targets or source serialization as an automatic fix.

## 6. Paired saved-validation findings

Use the frozen seed20261015 random20 plus four predeclared targeted cases, no overlap; no new sample chosen on favorable results. Automated results cover72 case-condition pairs per setting; targeted cases have separate denominators. Teacher-target agreement is not independent factual truth. Detailed unchanged sources/teacher/evidence and all three predictions accompany this audit.

| Group | Condition | Repeated sentences Baseline/A/B | Adjacent-word repeats Baseline/A/B | Correct headers Baseline/A/B |
| --- | --- | --- | --- | --- |
'''
    idx={(r['subset'],r['condition'],r['experiment']):r for r in b['rows']}
    for group in ['random','targeted']:
        for c in ['reviews_only','unfiltered','filtered']:
            rows=[idx[group,c,x] for x in ['baseline','greedy_A','beam_ngram_B']]
            cols=[' / '.join(str(r[k]) for r in rows) for k in ['exact_repeated_sentence_count','consecutive_repeated_word_count','section_headers_compliant_count']]
            audit+=f'| {group}{rows[0]["outputs"]} | {c} | '+' | '.join(cols)+' |\n'
    audit+='''
**Repetition:** A changes beam search to greedy yet U/F sentence repetition becomes20/20; therefore the observed failure is not specific to beam4 alone. B eliminates exact BPE4 and exact sentences in this sample but leaves adjacent lists and paraphrased repetitions. This supports a learned free-running/decoding interaction, not proof that any specific training parameter is the cause. No new inference was performed to test teacher-forced versus sampled token distributions.

**Informativeness/contrast:** `HhD8DLES5ZeULKDXmfkOOg`/RO omits sports/TV/beer detail in REV_4/REV_23 despite complete source text and informative teacher consensus. `gpSv4OvISIeyBAqcNN5bqw`/A RO incorrectly places buffalo chicken pasta under pizza despite REV_10; B removes that category error but retreats to generic pizza/service lists. `ZhA88ILcgEP9DNwZDx7f0A`/B RO groups beer/food under music; REV_6/REV_10/REV_22 distinguish music, food, drinks, positive/negative opinions. These show information/category binding problems beyond exact repetition.

**Visual grounding:** `thlyktt2c7v1DyOpSVSibw`/B U says 'A dog is with a spoon in it.' while IMG_CAP_4/PHOTO:NJ_yv-odfL62gOwdKF2aRg describes a spoon in a bowl. `b1pdnbYvBdtTxJOOom5WmA`/B U merges IMG_CAP_1 sweet-potato fries with IMG_CAP_2/4 rice/meat as the same bowl. `p7-1xG891NdGrcYBMOvxpw`/B U puts a fish on a wooden bar, mixing IMG_CAP_3 wall-mounted fish and IMG_CAP_2 bar/bottles. `gpSv4OvISIeyBAqcNN5bqw`/B U adds a pizza identity absent from all supplied captions but present in reviews. `j9fzpALNCgvDwCXmA5Vcjw`/A F adds review-only bergamot; B drops it but substitutes an unsupported biscuit identity. `j3kGicU3mPG_ciIKG1Vuyg`/F generates bagel/tray/cream-cheese claims in all three settings despite no caption/photo evidence. Those are unsupported by permitted visual text, not claims that the original photos were independently judged false.

**Real local gains:** B/F at `HhD8DLES5ZeULKDXmfkOOg` restores shrimp/grits plus people-at-bar/beer dispenser; `aZN5R7Xfea0GkDBwBZOQrQ`/F retains short supported fish/chips/table/wine content; `noUma64stIOSYAQ1GIIFjw`/F produces two simple caption-supported statements. Correct ordinary descriptions are valid. These gains weaken an absolute claim that the trained student cannot use any caption detail, while not establishing overall reliability.

**Header constraint artifact:** three targeted B cases emit `[VISUAL__EVIDENCE]`. Canonical `[VISUAL_EVIDENCE]` begins token IDs10975/23111/17710/1215, already emitted by a wrong first `[VISUAL_CONTEXT]`/`[VISUAL_CONSENSUS]`. The no-repeat4 rule blocks that canonical route; the double-underscore token changes it. This is verified token arithmetic on saved outputs, not proof of the original header-error cause or impossibility of other tokenizations. Decoding constraints can introduce artifacts; stop tuning them here.

Full exact claims, REV text/IDs, IMG_CAP/PHOTO mappings, Baseline/A/B sections and22 AI notes are included, along with the18 earlier A notes. They are illustrative AI assessments, not researcher-confirmed human labels or a population factual-error score. ROUGE reflects teacher agreement only; BERTScore remains unavailable/unpinned and no score is fabricated. Existing human worksheet fields remain unchanged/unfilled.

## 7. Bugs and concerns: severity matters

| Finding | Status / severity | Proposed response |
| --- | --- | --- |
| Wrong business/condition, source truncation, bad BOS/shift, labels trained on padding, skipped optimizer or wrong checkpoint | Not found in inspected evidence | No retrospective output/weight repair justified |
| Decoder-only scope | Verified design limitation, not a surprise implementation mismatch | Propose a controlled encoder+decoder scope experiment; do not assume success |
| Original zero-caption format validator only allowed an empty visual section rather than requiring it | Confirmed **diagnostic validation weakness**, already surfaced by later audit; it does not cause the student claim | Use the existing strict diagnostic `structure` check before future acceptance; preserve historical flags/outputs |
| Selection solely by teacher-forced validation loss | Accepted design concern | Keep checkpoint rule fixed for scope comparison; assess free-running grounding independently |
| New-scope runner cannot retain hardcoded24-module assertions / old optimizer resume | Future implementation hazard | Explicit48-module/96-tensor assertions, fresh scope-bound optimizer/checkpoints, approval gate |
| Header/BPE constraint interaction | Confirmed B-specific artifact after an existing wrong header | No output repair or further decoding trial; use frozen original decoding in any approved scope pilot |

No critical implementation error was established that requires correcting training before a scope proposal. This is not proof that every historical runtime detail or all teacher claims are flawless; no new numerical model replay is authorized.

## 8. Evidence-backed hypotheses

| Hypothesis | Evidence supporting it | Evidence weakening / limiting it | Status / discriminating test |
| --- | --- | --- | --- |
| Frozen encoder insufficiently adapts long review/caption representation | Zero encoder adapters; recurrent omissions and cross-caption relationship errors despite intact inputs | Pretrained encoder still reads all tokens; decoder cross attention adapts; some caption-specific B outputs are correct | Plausible; matched encoder+decoder scope pilot, not established cause |
| Too little trainable capacity / location of capacity | Only294,912 trainable parameters in decoder attention, no FFN/head updates | Parameter count alone is not a quality predictor; adding encoder doubles count and changes location together | Untested; scope+capacity bundle, not isolated encoder-location proof |
| Generic teacher-style imitation / underlearning details | Student common-phrase concentration exceeds teacher sample, review-specific facts omitted, validation loss still improves at epoch5 | Targets contain specifics; all6600 have no exact repeated sentences; template phrases can be appropriate | Plausible mode/generalization failure, not proof targets require rewriting |
| Teacher/data limitations | Known accepted Stage G grounding/informativeness limitations; simple captions; only9 zero-F train cases | Sources/targets contain useful facts the student omits; student invents errors absent from captions/targets | Real inherited limitations plus student-added errors; no blanket causal attribution |
| Optimization horizon/LR insufficient | Best loss at last epoch, short low-rank tuning may be insufficient | Correct nonzero updates/schedule, all48 tensors change; no matched LR/epoch ablation; losses do not diagnose semantics | Keep settings fixed first; do not automatically add epochs/rank/LR search |
| Ordinary teacher-forcing/free-running exposure gap and weak evidence objective | Training loss improves while generated behavior remains poor; targets do not supervise explicit claim-to-ID alignment | No new likelihood/attention/gradient experiment; many causes could yield same pattern | Hypothesis; saved text alone cannot identify mechanism |
| Only decoding is broken | Beam4 and greedy both fail; B removes strings but invents new objects/relationships | B gives some useful local gains | Pure decoding-only explanation weakened; no further decoding search |

## Boundary

Proceed only to **proposal review**. `STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md` specifies a small matched pilot and its limits. No training/inference, dataset/Teacher changes, test use, automatic production change or further decoding experiment has been performed or approved.
'''
    (OUT/'STAGE_I_TRAINING_FAILURE_AUDIT.md').write_text(audit,encoding='utf-8')
    proposal=f'''# Stage I Training Improvement Proposal

Date:2026-10-16. D-048. **PROPOSED ONLY - NOT AUTHORIZED OR EXECUTED.**

## Decision requested after offline audit

The audit found no critical implementation fault explaining the failures. The smallest useful next architecture investigation is a **paired Filtered-only pilot**: current decoder-only LoRA versus encoder+decoder Q/V LoRA. Freeze all accepted RO/U/F data, teacher targets, caption selection, budgets and split identities. Do not launch full RO/U/F retraining yet. Encoder adapters might help representation/binding, but may also overfit, add errors or fail to improve quality.

“Validation-only experiment” means **training on train businesses only, selection/evaluation on existing validation only**. It must never mean backpropagation on the200 validation businesses. No test input/reference/metric is used.

## 1. Exact architecture and counts, verified offline

| Component | A: current decoder-only | B: proposed encoder+decoder |
| --- | --- | --- |
| Decoder, every layer0..5 | `self_attn.q_proj/v_proj` and `encoder_attn.q_proj/v_proj` | Identical24 projections |
| Encoder, every layer0..5 | None | `self_attn.longformer_self_attn.query/value/query_global/value_global` (24 additional projections) |
| Trainable tensors | 48 A/B matrices | 96 A/B matrices |
| Trainable parameters | **294,912** | **589,824** |
| Frozen | Entire base, embeddings, tied head, FFN/norm/key/output projections | Same base weights remain frozen; only additional encoder LoRA branches train |

Both use **r8, alpha16 (scale2), dropout0, bias none, SEQ_2_SEQ_LM**. Each768x768 adapter has8*(768+768)=12,288 parameters. Proposed48 modules total589,824. Both encoder local and global Q/V branches are included; adapting only local query/value would be a different, unapproved experiment.

Exact proposed PEFT full-match regex (not a conceptual suffix guess):

```text
{arch['proposed_not_approved']['target_modules']}
```

`architecture_audit.json` contains **every exact intended name**, shape, count and trainable tensor from a meta-only PEFT injection. Future implementation must assert set equality,24 encoder+24 decoder modules,96 tensors/589,824 trainable parameters, and no other trainable base weights. The existing runner hardcodes24 decoder-only modules and must not be reused unchanged or silently relaxed.

This changes both adapter location and trainable count. It tests an **encoder+decoder scope/capacity package**, not an isolated proof that encoder location alone caused improvement. A parameter-count-matched third arm would be a separate experiment and is not proposed for execution now.

## 2. Minimal paired design and existing checkpoint reference

- **Primary practical reference:** reuse existing full-trained F best checkpoint1250 and its exact24 saved Baseline predictions; no regeneration. RO/U/F full checkpoints and A/B decoding results remain unchanged context.
- **Matched causal screen:** propose two fresh F-only pilot arms, A decoder-only and B encoder+decoder. Start both from the same pinned pretrained LED base, never by extending the already-trained A adapter or resuming its optimizer into B.
- **Train subset:** the proposed manifest fixes512 unique train businesses using seed20261016. Proportional largest-remainder sampling over Filtered caption-count bins0/1-2/3-5 and full-source-length quartiles, then original row order. No saved-output/label-quality selection. This realizes{len(pilot['selected_zero_filtered_train_ids'])} zero-F train cases without adding/duplicating them. Both arms use exactly the same512 inputs/targets.
- **Training duration:**2 epochs per arm,1,024 microbatches and128 optimizer updates per arm at batch1/accum8. Total two fits/256 optimizer updates. Epoch eval/save at64 and128; choose the lower loss under the existing rule. Do not extend an arm or try more seeds/epochs after seeing a preferred result.
- **Validation:** all200 existing F validation businesses for loss, without training on them. Free-running generation only for the same fixed20 random+4 targeted F inputs per arm (48 future outputs total, not72 or an RO/U/F retraining). Random/targeted results stay separate.
- **Decoder:** original baseline beam4, do_samplefalse, no_repeat_ngram_size0, max_length768 including decoder-start, length_penalty1, early_stoppingtrue, forced EOS2/cachetrue. No new decoding setting; do not use B's ngram constraint to mask training problems.

The old full F run saw2,000 businesses for5 epochs,1,250 updates; a512/2-epoch candidate is **not compute/data matched** to it. Existing full F is the main practical reference, while only fresh A-pilot versus B-pilot can support a controlled scope comparison. Comparing new B directly with old A alone would confound data exposure/horizon and adapter scope. The subset/duration reduction and its shorter scheduler are explicit common pilot proposals, not changes to the accepted full training run.

## 3. Common settings and unavoidable differences

Keep pinned `allenai/led-base-16384` revision38335783885b338d93791936c54bb4be46bebed9, tokenizer/special tokens, exact D-032 template, K5, source/review/caption5000/4000/750 and target768 unchanged. Preserve complete reviews/captions/targets and zero-caption representation. Use FP32 base/adapters with FP16 autocast/AMP, no quantization/offload, non-reentrant checkpointing, batch1/accum8, AdamW2e-4, decay0.01, betas0.9/0.999, eps1e-8, max grad norm1, linear scheduler/warmup ratio0.05, zero label smoothing and validation-loss selection in both arms. Same software pins, seed/data_seed42 and same hardware for both arms if executed.

Warmup is ceil128*0.05=**7** steps, versus63 in the completed full run, because the pilot has fewer updates. Do not claim the reduced run has the identical LR trajectory to the full experiment; it is identical **between pilot arms**. Epoch count, update count, sample population and evaluation opportunity differences versus the old full model are explicit. Eval/save policies are epoch-based; patience2 remains but two epochs provide little early-stop information.

Encoder additions change initialization RNG consumption. A common seed alone does not guarantee identical decoder LoRA initialization. Before future training, create/persist a shared fresh decoder-adapter initialization and verify its hash/tensor equality in both arms, with normal LoRA A initialization and zero B matrices. Initialize new encoder adapters with a separately recorded fixed seed43; reset training/data RNG42 after initialization and record sampler order per epoch. Both initially represent the same base function when B matrices are zero. Do not copy trained decoder weights from the existing full reference into just one pilot arm.

Record model dropout and base config, model.train/eval transitions, token losses, finite gradients, optimizer steps, actual LR, global attention, all intended module gradient/update evidence and scope-bound checkpoint/resume state. Save distinct adapter/optimizer/scaler/RNG checkpoints in new namespaces. Never resume across scopes. Future feasibility must confirm decoder-input/label shift and successful save/load before complete pilot execution.

## 4. Feasibility gate after separate execution approval

No GPU check was executed for the proposed architecture. The existing decoder-only memory figure cannot establish encoder+decoder training feasibility. The extra294,912 adapter parameters add only~4.5MiB for FP32 parameters/gradients/Adam moments, but long-encoder activation gradients can add far more. No new-scope memory/runtime guarantee is made.

On the approved available local/Colab device, first run only an expressly approved feasibility phase for each scope: actual long train source/target cases, approved masks, and worst-case padded5120-source/768-label shapes using masked padding rather than invented text. Check finite loss, successful scaled backward/optimizer steps, actual CUDA allocated/reserved and sampled device memory; require at least15% measured device headroom for a feasibility PASS. Stop on OOM, mismatch, loss/gradient failure or insufficient headroom. Do not change batch, dtype, length, scope, attention windows or accumulation automatically. Device/profile changes need review before the pilot proceeds.

LoRA B matrices initialize to zero, so A-matrix gradients may be zero on the first step; that is expected. Require finite gradients on all intended adapter branches, nonzero B-gradient/update evidence initially where active, and finite/nonzero A/B updates after multiple successful steps. Explicitly inspect encoder local/global and decoder self/cross branches; an early all-zero A-gradient check alone must not produce a false failure. No base weights may update.

Generation uses the exact frozen beam4 settings; report format/EOS/768-cap behavior and checkpoint recovery on the approved fixed controls. Feasibility is numerical compatibility, not factual-quality acceptance. The production subset/both pilot arms must not start automatically after a feasibility run; review measured compatibility first unless the researcher explicitly authorizes both phases together.

## 5. Predeclared evaluation

Compare A-full saved reference, fresh A-pilot and fresh B-pilot on the same24 F sources, but use fresh matched A/B to interpret scope. Generate only once from each loss-selected pilot checkpoint. Do not select a decoding option or checkpoint because its sampled text looks best. No decoder/threshold search.

**Automatic measures:** same exact sentence and adjacent-word definitions; repeated BPE/lexical2/3/4grams, distinct-n with length caveat; output length/EOS/cap; exact required headers and **required empty visual** for zero-F; full/section ROUGE as teacher agreement. BERTScore remains uncomputed until a scorer/backbone is separately specified; do not invent a score or change a metric model after results.

**Information/grounding review:** use a fixed source-content-unit sheet derived from the existing teacher REV/IMG_CAP claims as candidates, but confirm units against the selected reviews/captions. Teacher claims are not independent factual truth. Before pilot outputs, freeze the reference units, evidence IDs, polarity/contrast and canonical visual grouping for the24 cases. Do not rewrite Teacher labels or student targets. Keep verification status/provenance separate; uncertain units are not converted to negative or quietly removed differently across arms. If no source-verified unit sheet is completed, information/grounding gates are **not evaluable**; do not substitute ROUGE.

For each output, mark distinct supported units recovered, important positive/negative aspect/contrast preservation, redundant paraphrases and unsupported/uncertain review or visual claims. An ordinary simple caption-compatible statement is valid, while a plausible photo-only/review-only detail is not caption support. Audit counts, actions, ingredients, identities and cross-photo object relations. Zero-F must produce no visual assertion. Report empty/very short visual sections and available supported units so silence cannot game precision. No original-photo factual judgment is claimed without a separate authorized review.

Use all20 random F cases for the primary descriptive quality comparison, targeted4 only for stress/failure examples. For human adjudication, present anonymized arm IDs in a fixed shuffled order(seed20261016) with full sources/evidence, retain AI suggestions separately and let the researcher confirm final decisions. No AI suggestion becomes a human-final label automatically. Keep counts/uncertainty and all exact excerpts; do not fabricate a population factual-error rate or independent human review. Paired resampling intervals, if reported, are descriptive with n20 and do not establish generalization.

## 6. Proposed go/no-go criteria, to approve before training

These are **new proposed pilot acceptance criteria**, not previously frozen scientific parameters. Apply all, without post-hoc relaxation:

1. **Technical gate:** exact module/count/data/decoder/gradient/checkpoint integrity; no truncation/fallback/nonfinite loss, failed intended branch or scope-resume mix. No training on validation.
2. **Format/safety gate:** every random20 output has valid exact headers and EOS before768; targeted header compliance must not be worse than matched A-control. The targeted zero-F case must be empty-visual compliant1/1; no unsupported visual content is excused by missing/malformed headers.
3. **Repetition gate:** at least30% relative reduction versus matched A-control in macro adjacent-word-pair fraction on random20, with no increase in exact repeated-sentence outputs. If control has zero adjacent repetition, require it remain zero and mark reduction not applicable; do not change the metric. Semantic paraphrase redundancy is independently reported.
4. **Information gate:** source-verified macro distinct useful-unit coverage improves by at least10 percentage points versus matched A-control, and important review contrast retention does not decrease. Review and visual coverage are separately reported; a gain confined to generic plate/table phrases is not sufficient.
5. **Grounding gate:** unsupported-claim fraction must not exceed matched A-control on random20, with separate review/visual denominators, uncertainty and omitted/empty outputs reported. No new severe invented ingredient/object/action/category error may appear in the targeted controls. Precision cannot pass by omitting all useful visual evidence; information/zero-F gates still apply.
6. **Practical reference check:** report all the above against existing full F too. A B-pilot that beats a weak short A-control but remains materially worse than full F in support or useful information is **inconclusive for full retraining**, not a success. Require no more than2 percentage points absolute loss of source-verified information coverage/support precision versus full F, while at least one substantive measure improves. Small/uncertain denominators mean inconclusive, not an automatic pass.

ROUGE and validation loss are supporting measures and cannot override factual/coverage failure. A successful small pilot justifies **proposing** full common-scope RO/U/F retraining for approval; it does not authorize it. Failure or mixed/uncertain results require review of the audit, objective/data supervision and undertraining risk, without automatic rank/epoch/LR/decoder changes. A negative512/2-epoch result does not prove encoder LoRA cannot work; the reduced horizon is a limitation, not permission to extend the run.

## Risks and scientific value

Benefits: tests a specific architectural limitation while retaining frozen source/target evidence and controlling pilot exposure. Risks: small sample/one seed and repeated validation inspection; added capacity versus location confounding; shorter schedule versus old full reference; inherited Teacher/caption weaknesses; sparse zero-F supervision; more encoder memory and possible global-branch gradient issues. More parameters might memorize templates or unsupported associations instead of improving evidence use.

No split, threshold, K, source/target budget, caption/teacher content or existing checkpoint is modified. Proposed sampling/config/module maps and all scope/fairness limits are recorded now. **Stop after offline review bundle; explicit approval is required before any future feasibility, new inference or training.**
'''
    (OUT/'STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md').write_text(proposal,encoding='utf-8')
    proposed={'status':'PROPOSED_ONLY_DO_NOT_EXECUTE','execution_authorized':False,'new_training_authorized':False,'new_inference_authorized':False,
              'backbone':{k:config[k] for k in ['model_id','revision','source_limit','label_limit']},
              'reference':'Existing F checkpoint1250/saved predictions; no regeneration; full run not compute-matched to pilot',
              'arms':{'A_decoder_only':{**config['lora'],'expected_modules':24,'expected_trainable_parameters':294912},
                      'B_encoder_decoder':{**config['lora'],'target_modules':arch['proposed_not_approved']['target_modules'],'expected_modules':48,'expected_trainable_parameters':589824}},
              'shared_optimization':{**config['optimization'],'num_train_epochs':2},'generation':config['generation'],
              'train_manifest':'proposed_pilot_manifest.json','validation_gradient_updates':False,
              'new_decoder_experiments':False,'held_out_use':False,'automatic_scope_fallback':False,
              'requires_new_reviewed_runner_not_original_24_module_assertion':True,
              'common_decoder_initialization_hash_match_required':True,'data_seed':42,'initial_encoder_adapter_seed':43,
              'memory_feasibility_tested':False,'warmup_steps_for_128_updates':7}
    write_json(OUT/'proposed_training_config.json',proposed)
    write_json(OUT/'OFFLINE_EXECUTION_NOTE.json',{'audit_artifacts_complete':True,'no_model_forward_backward_generate_train_executed':True,
        'PowerShell_log_note':'Optional Triton import emitted Failed to find CUDA warning on stderr; PowerShell5 formatted it as NativeCommandError and surfaced shell exit1. The auditor emitted its complete success JSON and all final integrity artifacts; no traceback/failing assertion in the Python audit. This meta/CPU audit does not require CUDA. Original log is preserved; no GPU test is claimed.',
        'proposal_is_not_execution_approval':True})
    print(json.dumps({'status':'AUDIT_AND_PROPOSAL_REPORTS_WRITTEN','current_params':294912,'proposed_params':589824,'pilot_executed':False}))


if __name__=='__main__':main()
