# Post-D060: paired pseudo-repetition Unlikelihood preparation v1

Status: offline implementation authorized by researcher message "ok hãy triển khai"
following the explicit proposal to prepare offline and return before GPU.
No GPU/training permission, new deadline, or independent hash approval is inferred.
Historical final-attempt limits remain recorded; renewed execution needs an explicit
exception and a new UTC end time. D051 strict replay FAIL is unchanged.

## Hypothesis and intervention

MLE-only continuation versus MLE + 0.1 * mean negative-prefix token Unlikelihood.
This is an exploratory CUSTOM adaptation inspired by Welleck et al., not an exact
paper reproduction. DITTO motivates sentence-copy negatives but its objective is
not implemented. No semantic-paraphrase UL, grounding classifier training,
quantization, architecture change, reward model, or NLI/MMR filtering is included.

L_UL = mean_{inserted-copy token positions} -log(1 - p_theta(token | source, negative_prefix)).
FP32 stable log1mexp; no probability clamp hides numerical failure. Positive CE
uses every original label token incl. BOS/EOS, ignore_index=-100 only on padding.
Validation loss is positive CE only, common across arms. Per-negative-token UL
is averaged inside each example; ineligible examples contribute exactly zero.
The 0.1 weight is a proposed fixed engineering choice, not literature-optimal.
Relative CE/UL scales must be logged, not tuned after observing validation output.

## Training data and negative views

Hash-rank seed42 selects512 of2000 F TRAIN businesses, no outcome/quality/stratified
selection. Same order/positive labels for both arms, no zero-caption oversampling.
Alternate sentence/word views by sample rank. Select complete body sentences or
words deterministically by hash; repeat exactly one unit after its occurrence.
Keep the negative prefix through the inserted copy, with explicit EOS unpenalized;
this is a synthetic view, never a replacement or truncation of the full positive.
Repeated token positions align via tokenizer character offsets. Marker, pad,
BOS/EOS and same-prefix correct gold continuation are not penalized. Overflow,
changed token prefixes, missing/unsafe units are recorded, never clipped or filled.
No generated validation errors, D053 reference text or Teacher-verifier labels are
training negatives. Names/ordinary repetition are not globally banned. The objective
can still overpenalize legitimate constructions; that is a proposed-method risk.

## Matched experiment (PROPOSED, not executed)

Both arms start from exact same historical full-F decoder-only LoRA adapter:
24 Q/V modules,294912 trainable parameters,rank8 alpha16 dropout0, pinned LED-base.
No inherited optimizer/scheduler/scaler/RNG. Fresh seed42, AdamW2e-4 wd.01,
physical batch1/accumulation8, fp16 AMP, frozen backbone, original attention1024,
full source5000/labels768. Same2 epochs128updates, fresh linear schedule/warmup7.
Save steps64/128, no early stop or best-loss reloading. The shorter horizon,
initialization and auxiliary objective are proposed amendments, not D060 settings.

Both arms execute the same eligible negative forwards, including control(weight0),
to match data/RNG/forward exposure. CE and negative graphs coexist; memory/time
cannot be inferred by simply reusing old allocator readings. Two real-training
forwards per eligible microbatch; no generation during training.

Bounded gate3updates/24microbatches PER ARM (6updates total), no generation/replay.
Discard all gate adapters and optimizer state before two fresh pilot runs.
Verify real attachment/gradients, frozen tensor hashes, masks, finite loss/AMP,
15% measured memory headroom, checkpoint serialization/restoration. Restoration
is not strict replay and cannot supersede D051 FAIL. Stop on failure, no fallback.

Only final epoch2:24 outputs per arm =48 total, original beam4/max_length768.
No repetition guard, hard NLI filter, new decoding or checkpoint search.

## Evaluation/acceptance

Existing20random+4targeted are DEVELOPMENT cases already repeatedly inspected.
Report separately, pair business comparisons, annotate blind A/B where practical,
paired uncertainty intervals and exact denominators, no statistical generalization.
Automated: exact sentence/n-gram repetition, proper punctuation-aware word stutter,
list duplication, EOS/forced-cap/completeness and header/zero-caption compliance.
Human/AI-assisted separately: semantic repetition, source-grounded facts, Core
coverage vs approved D053 units, business specificity and positive/negative opinion
coverage. Supporting omission is not an automatic factual error; Teacher is not
independent truth. No NLI-derived factual accuracy or automatic human labels.

PROMISING only if reviewer-confirmed reduction of harmful sentence/semantic loops
is accompanied by maintained or improved Core/detail/opinion coverage and grounding,
with no compensating severe regressions in completion, zero-caption or factuality.
MIXED/INCONCLUSIVE/POOR means no adoption. Lower loss/repetition/length alone never
passes. There is no software-only GO. No fixed percentage threshold retrofitted.
Any independent confirmation must preselect validation IDs after exposure audit;
no unseen status is asserted now, and no held-out100 is opened or used for tuning.
No automatic RO/U training or main-model replacement; matched future methodology
would need separate approval across all conditions.
