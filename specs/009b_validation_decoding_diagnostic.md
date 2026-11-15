# Stage I specification: controlled validation decoding diagnostic

**Spec ID:** 009b / D-046  
**Date:** 2026-10-15; execution result updated2026-10-16  
**Status:** A explicitly approved/completed on local RTX3050; no production adoption recommended. D-047 B completed/evaluated; no production decoder adoption. Retraining remains unapproved.

**Execution addendum (2026-10-16):** Researcher approved A, then explicitly substituted local RTX3050 Laptop6GB after Vast access failed. The local v2 runner changes operational hardware admission/A-only bindings/metadata; original code/preparation stay preserved, the generate-call AST is identical and all scientific generation settings remain fixed apart from num_beams1. All72 outputs completed once with exact frozen bindings and matching actual GenerationConfigs. Random20 repeated-sentence counts beam4->A: RO5->2/U19->20/F15->20; zero-caption F0/1->0/1. Greedy is not recommended for production adoption. Runtime507.77s including setup,439.13s summed generation; sampled whole-GPU peak3460MiB; no OOM/retries, new paid-action charge0. Baseline RTX3090 versus A RTX3050 is disclosed; no isolated hardware-independent speed/bitwise claim. See `../outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1/DIAGNOSTIC_REPORT.md` and progress. Stop after A/evaluation. Earlier preparation/future-approval wording below is historical; it does not grant B, retraining or test access.

**D-047 B result (2026-10-16):**72/72 completed once on local RTX3050; exact same frozen inputs/checkpoints and actual configs PASS. Length explicitly clarified to max_length768; only no_repeat_ngram0->4 changes from beam4. Exact sentence/BPE4 repetition disappears but adjacent-word/semantic repetition, unsupported visual/review claims and zero-caption failure persist. Three wrong-header outputs add a double underscore to VISUAL__EVIDENCE, with canonical four-token header-prefix conflict documented offline. No B production adoption recommended; stop after evaluation/review bundle, with no retraining or new experiment. Full results in outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1 and latest progress.

## Historical authorization - 2026-10-16: D-047 Diagnostic B only, length clarified

The researcher reviewed A and explicitly approved B on the identical 20 random + 4 targeted validation businesses / 72 RO/U/F cases, existing best adapters and frozen inputs. After a preflight caught conflicting length wording, the researcher explicitly chose **max_length=768 including decoder-start**, preserving Baseline/A semantics. B changes only **no_repeat_ngram_size 0 -> 4**, with num_beams4, do_samplefalse and every other baseline generation setting unchanged. The historical max_new_tokens=768 request is superseded by this explicit clarification, not silently adopted.

Offline checkpoint/dataset/source/tokenizer/code bindings, 72 complete source/label roundtrips, Baseline/A ID alignment and global-BOS collator checks PASS. Reconstructed B defaults match the pinned preparation. A B-only local RTX3050 6GB runner preserves the original A/baseline files; generate-call AST is unchanged. Runtime must verify actual complete GenerationConfigs and loaded adapter tensors, record memory/time, and stop on OOM/mismatch without fallback or parameter changes. Previous A memory does not prove beam4 feasibility.

Output namespace: `outputs/stage_i/decoding_beam_ngram_B_local_v1/`; evidence/control: `outputs/stage_i/decoding_B_preflight_v1/` and `decoding_B_execution_control_v1/`. No generation results exist at this authorization checkpoint. Evaluate Baseline/A/B separately for random/targeted groups, with n-gram/lexical metrics, formatting/EOS, information and evidence-grounding review. Lower repetition alone is not success. BERTScore remains unavailable/unpinned and must not be fabricated. Stop after B/evaluation and a portable review bundle. No training, checkpoint/data/teacher edits, Teacher requests, held-out access or further experiments. No production decoder is frozen. Earlier statements that B is unapproved are historical.

## Scope and frozen inputs

Investigate the existing three trained checkpoints before proposing retraining. Preserve accepted RO/U/F weights, Stage F/G/H content, K5, source/review/caption budgets5000/4000/750, training splits, tokenizer/backbone revision, generation limit768 and all training settings. No Teacher requests, paid execution, held-out access, new model inference or output repair during this preparation. Existing baseline text evaluation and CPU tokenizer/collator checks are allowed; these are not generation.

Use the imported runs_v3 best adapters, verified equal to each condition's checkpoint1250 and selected by validation loss. Explicitly load the original pinned base revision38335783885b338d93791936c54bb4be46bebed9; do not rely on the adapter's null revision field to resolve a current backbone. No optimizer or Trainer is needed for inference.

## Sampling before new generation

Sample20 from the full200 validation business IDs using Python random.Random(20261015).sample(sorted(ids),20). Add the four previously identified warning businesses in the diagnostic config. Record any overlap and de-duplicate only execution, not group membership. The realized v1 draw has no overlap:20 random+4 targeted=24 unique businesses,72 condition-aligned cases. Use identical IDs across RO/U/F. Random and targeted results must have separate denominators; do not interpret targeted rates as prevalence. The random set retains the earlier audit's ten seed-prefix cases by deterministic sampling, without replacing IDs after viewing new results.

## Only the requested configurations

| Experiment | num_beams | do_sample | no_repeat_ngram_size | Execution |
| --- | ---: | --- | ---: | --- |
| Baseline | 4 | false | 0 | Reuse exact saved outputs; do not regenerate |
| Diagnostic A | 1 | false | 0 | Proposed first; one override from baseline |
| Diagnostic B | 4 | false | 4 | Separate proposal, no automatic execution after A |

Other explicit settings remain max_length768 (includes decoder start), length_penalty1, early_stoppingtrue, forced_eos_token_id2, use_cachetrue, FP16 autocast over the same FP32 base/adapter weights, eval mode, batch1, seed42 and identical complete input IDs/global attention. In greedy mode beam-specific early_stopping/length_penalty are not active; keep their supplied values and report the warning rather than changing a second parameter. BOS0, PAD1, decoder-start2, EOS2; do not strip or add content, force minimum lengths, resize embeddings or introduce additional generation settings.

The original run saved explicit generation kwargs, but not the complete model.generation_config. Offline preparation records a clearly labeled reconstruction from the pinned backbone config and installed Transformers4.51.3 defaults; this is not a recovered observed runtime snapshot. At approved execution, resolve the same pinned backbone, log its entire effective GenerationConfig and fail on an unexpected generation-default difference instead of silently adopting it. Generation must use an explicitly pinned backbone and exactly loaded adapter tensors, independently checked before and after inference.

## Reproducible assessment

- Exact repeated sentences: same accepted audit rule, section markers removed, split on sentence-final .!? plus whitespace, lowercase/collapse whitespace; any sentence occurring twice flags the output. Report sentence instances/fractions separately.
- Consecutive repeated words: remove section markers, lowercase, tokenize `[a-z0-9]+(?:'[a-z]+)?`, compare adjacent lexical tokens. Punctuation is ignored, so `bland, bland` counts. Report output-level rate and adjacent-pair count/fraction. Apply the same rule to teacher targets. This is a diagnostic definition, not a semantic error label.
- Length: min/mean/median/p90/p95/max, including decoder-start. Exclude initial token2 from EOS detection. Report EOS-before-cap separately from cap-reaching EOS, because forced_eos_token_id2 can close an incomplete768-token loop.
- Structure: exact required review/visual headers, order and nonempty review; RO has no visual section. U/F retain the visual section; zero-caption F requires it to be empty. Do not rewrite outputs to enforce this.
- Review coverage: full selected-review source, exact teacher review claims/REV evidence and student review text in human-review worksheets and viewer. ROUGE review recall is a lexical teacher-agreement proxy, not proof of important-opinion coverage, faithful categories or preserved contrast. Keep human coverage/contrast/support fields blank until reviewed.
- Visual support: list every generated visual sentence with the original caption/photo mapping, any exact teacher-claim match, and a lexical navigation candidate. These are not automatic entailment/factual verification. Cases with no caption evidence are explicitly flagged. Human caption-support and factual judgments remain separate and unfilled. Ordinary simple but supported descriptions are acceptable. No population factual-error rate is fabricated from lexical similarity.
- ROUGE1/2/L precision/recall/F1 against complete teacher targets and sections, CPU-only rouge-score, no stemming. Macro means, empty-reference sections N/A; label all scores teacher-target agreement. No BERTScore model is installed/pinned: report not computed; no automatic scorer download or inference.
- Keep formatting, repetition, grounding and teacher agreement separate; no composite quality score. Use the same cases and definitions for all conditions/settings. Absence of A/B results must be explicit, never encoded as a zero error rate.

## Execution gates and preservation

Preparation produces subset/data/checkpoint/code bindings and baseline-only reports in a new namespace. A separate future approval JSON must bind the config and subset hashes, identify the approved experiment, confirm cost review and permit any backbone download. The runner requires both that file and --execute before CUDA/model loading. B needs its own explicit approval after A review; no automatic A-to-B transition. Approval to train in the old cloud archive is not approval for this new inference diagnostic.

Approved inference later runs conditions serially, uses per-case resumable files with immutable bindings, records generation config/device/precision/time/memory/errors and preserves the original checkpoints/predictions. Any config/data/hash mismatch or unexpected generation default is a blocker, not permission to change settings. Resume only matching successful cases; failures remain recorded. Never summarize a partial matrix as a complete comparison.

## Acceptance and next decision

Offline: reproduce baseline600 sentence counts31/195/166, verify random/targeted manifests, all72 source/label/baseline token bindings, same shared reviews across conditions, CPU global-BOS/padding/label shift checks, exact A/B parameter diffs, adapter hashes/scope and approval gates. Prepare cost/time assumptions grounded in prior per-case generation times and current primary-source pricing; do not claim an unmeasured greedy speedup.

Execute A first only after researcher approval. Judge coherence, source-specific information and evidence faithfulness alongside repetition. If A is insufficient, return evidence and the separately prepared B proposal for review. Decoder-only adaptation limits, generic imitation, decoding effects and configuration mismatch remain hypotheses; no retraining/module/epoch/budget change is approved by this spec.
