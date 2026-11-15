# LED-large decoder-label policy: offline readiness — 22/10/2026 ICT

**OFFLINE READY FOR RESEARCHER REVIEW. EXECUTION NOT APPROVED.**

One scientific correction is proposed: remove only tokenizer-added initial BOS0 from the training/validation label view, retaining all content tokens and finalEOS2. The pinned checkpoint is official allenai/led-large-16384 revision cd59d11c3528415b7dda4dfc95cc8f138aceda2e. No decoding, architecture, LoRA rank, optimizer, numerical policy, source or Teacher change is proposed.

## A–C. Verified pinned runtime and label semantics

Transformers4.51.3 LED shifts labels once when decoder inputs are absent. CE compares each logit position with the unshifted label; it does not shift the targets a second time. Padding remains ignored as-100. Pure runtime helper tests confirm generation without a decoder prompt begins with decoder-start2 only; forced_bos_token_id remains null. Cache is disabled for labeled training/validation and enabled by the unchanged generation config for eval-mode decoding. Pinned LED/generation/Trainer/Accelerator Python source hashes are bound and checked by the new preflight.

All6 frozen train/validation datasets were audited locally: RO/U/F each2000train+200validation =6600 labels. Every old sequence equals `[BOS]+tokenizer.encode(Teacher_text,add_special_tokens=False)+[EOS]`. Each has exactly one BOS and one terminalEOS, no interior BOS/EOS/PAD/MASK, zero unknown tokens. Removing BOS yields exactly the unchanged content encoding plusEOS and decodes byte-for-text identically to the original Teacher string. No first word, header fragment, punctuation, newline or EOS is removed. Original source IDs are retained from the byte-verified original NPZ; no source reconstruction/compression is introduced.

The initial BOS is therefore a tokenizer wrapper rather than business evidence. Dropping it from the LABEL VIEW is semantically justified for the proposed decoder policy. It is not a claim that the old StageH data are corrupted or that BOS removal is proven to fix all generation quality. Source/encoder BOS0 and its global attention stay intact. The historical StageH status remains `STAGE_H_DATA_VALIDATED_DECODER_POLICY_PENDING`; this new proposed policy has its own receipt and is not retrospectively marked approved.

## D–E. Exact, non-destructive correction

Two new F-only label NPZs contain business IDs, label offsets and corrected labels, without any rewritten input IDs. The versioned dataset wrapper pairs them with original sources and verifies every row equals old_labels[1:]. The generic transformation is applicable consistently to RO/U later, but those arms are not prepared for execution or authorized here.

The new collator reconstructs `[0]+view_labels` only as an internal input to the original trusted padding routine, then returns its label tensor shifted left by exactly one and pads the vacated final slot with-100. Thus source tensors, attention/global masks, source padding and even original label padding SHAPES remain identical. Model loss targets contain `[content...,EOS]`; model-auto-shift yields `[EOS,first_content,...]`. It does not merely mask a BOS target while leaving BOS in the decoder prefix.

`EXACT_PROPOSED_PATCH.diff` shows the complete old-to-versioned helper/runner/reporter/config diff and new label/sanity modules. No old code or scientific file is overwritten. The new fixed run namespace is `runs/F_large_decoder_policy_v1`; resume or import of old large/base/gate adapter state is forbidden. Parent files and new overlay files are separately bound.

19 focused tests PASS, including raw-label rejection cases, full2200F-view loading, unchanged source/masks/padding, exact pinned shift, synthetic CE alignment and ignored padding, generation-start helper/configuration, scope/hash/deadline rejection, old approval rejection, early-sanity failure propagation and mocked mode/cache/RNG restoration. Tests use tensors/helpers/mocks; no pretrained model forward, generation, backward or optimizer runs. Full portable-root CPU preflight PASS, including all9961 parent payload bindings and18 new manifest entries. Final scientific integrity recheck PASS on all6 original NPZ/JSONL datasets and tokenizer assets.

## F. Competing defects and limits

- Adapter loading: old actual run/earlier CPU evidence verifies saved tensor equality; all5 large adapters are finite/nonzero. Original loader requires228modules/456tensors/4915200trainable parameters, active LoRA scaling2 and frozen backbone/hash integrity. No evidence of an entirely untrained or skipped adapter was found.
- Generation/mode: actual old eval config matches the pinned expected config; no forced BOS, token suppression or new repetition constraints. Runner sets eval mode and omits labels/decoder prompts from generate inputs. PEFT's LoRA path delegates labels to the base seq2seq model, not a prompt-learning branch.
- Masks/shift: encoder global BOS anchor is supplied and verified. No double label shift or PAD-vs-content error was identified. The prior tests correctly checked implementation of the old policy but did not establish checkpoint appropriateness.
- Cache: static behavior is consistent with labeled/no-cache and eval/generation/cache. No new cache-on/off model parity experiment was performed; a runtime-specific defect is not ruled out solely by static inspection.
- AMP: unchanged65536 initial scale with bounded native backoff. Historical D059 saved scalers also remained65536. Prior FP32CPU prefix probes reproduce the large-specific initial-position anomaly. These facts weaken a scale-only explanation; they do not prove every numerical path is ideal.
- Learning capacity/hyperparameters and lexical/semantic repetition remain possible additional issues. They are not searched or modified here.

## G. Evidence calibration

CONFIRMED: exact old label/decoder prefix, all6600 text/token identities, old48 raw BOS-loop prefixes, null forced BOS, versioned view/collator alignment, preserved frozen artifacts and passed offline checks.

STRONGLY SUPPORTED HYPOTHESIS: interaction of this LED-large checkpoint with the old EOS/BOS prefix contributes to collapse. Earlier FP32CPU probes on2synthetic sources found near-identical first-two-position logits for pretrained/epoch3/5 large, while D059baseepoch3 distinguished them. This is prior diagnostic evidence, not a new model call in this preparation. Historical maintainer/user discussion of a related workaround is a supporting lead: https://github.com/huggingface/transformers/issues/18190 .

UNPROVEN: causal recovery under the corrected view; full real-case FP16 behavior; elimination of phrase/sentence/semantic repetition; improved Core coverage, opinion balance, factual/visual grounding and usability. No scientific GO, researcher labels or GPU-readiness PASS is inferred from offline tests or loss.

## H. ONE proposed bounded fresh F verification

Future approval must bind this exact new config/runner/preflight and the decoder-policy amendment, carry actual approval UTC, a maximum48hour execution-and-review window plus researcher absolute stop, and confirm final-phase reserve. No historical deadline is changed. Actual defense date is not supplied here, so reserve cannot be independently certified from this package.

1. Verify all parent/new hashes, pinned runtime source/dependency versions and healthy idle RTX3090/24GB. New approval is required; the embedded template is FALSE and old execution approvals are rejected.
2. Run one fresh corrected-label feasibility gate:3successful updates/24nominal, at most88actual microbatches under the unchanged8-overflow bound;2existing TRAIN-input beam4 memory traces, no replay updates. Require all original gradient/AMP/frozen-backbone/save-restore checks and >=15% measured headroom. Gate states are discarded.
3. Start ONE fresh F2000/200 run from pinned original pretrained weights and fresh named-seeded LoRA. Same228modules/4915200params, AdamW2e-4, batch1/accum8, seed42/order, FP32storage/FP16AMP, r8/alpha16/dropout0, linear schedule/warmup63, maximum5epochs/1250successful updates. No inherited optimizer/scaler/RNG/adapters. Save all5epoch checkpoints.
4. After epoch1 checkpoint250, generate exactly4predeclared development sanity outputs: first2random cases plus zero-caption targeted case and KFC targeted case. No case replacement. Require nonempty text, no BOS after decoder start, >=16ordinary tokens and naturalEOS before768cap on all4. Raw outputs are kept. Preserve/verify adapter state and restore all module modes, cache setting and RNG before continuing. Failure stops this run without retry/fallback/setting changes. It is a technical generation-verification failure, not proof of inherent large incapability or a substantive quality NO-GO after250updates.
5. Only after technical PASS continue the SAME fresh run to1250. Generate the existing24random20/targeted4 cases at epoch3 and5:48final outputs. No epoch1 quality checkpoint selection. Total maximum54generation calls:2gate technical+4early sanity+48final. This transparently adds4early validation outputs to the prior48-output final review scope and requires explicit approval; it is not hidden inside the old budget.

## I. Quality decision and matched-family rule

Early sanity is not quality GO. Review actual final outputs against D059Expandedepoch3 as the principal practical benchmark, also historicalF/D058B/D059epoch5. Keep random20 and targeted4 separate; do not credit empty output as reduced repetition. Require meaningful business-specific Core coverage, balanced opinion synthesis, grounded useful visual details, zero-caption compliance, semantic repetition reduction and usable completion without severe regressions. Corrected-label loss has a changed token denominator/alignment; raw loss values alone are not directly interchangeable with the old-label run.

AI assessments remain provisional. Only researcher-confirmed substantive success could support separately approved matched corrected-policy largeRO/U/F with identical initialization/architecture/optimization and a single common epoch3or5 selection rule. Shuffled reuses the exact matchedF checkpoint. Do not replace onlyF in the main caption-effect comparison. Historical matched LED-baseRO/U/F remain primary until a complete matched large family is accepted. Preserve D051FAIL, all previous runs and stopped Unlikelihood evidence. No further optimization experiment is initiated by this proposal.

## Exact bindings

```json
{
  "config_sha256": "89202ab27ec6adecba3ee293cb1c8151b3a5935a20ff0c31da12c9bc784a3d20",
  "runner_sha256": "6762bed2bf4f44ecb2077d87f5687e3ab8193bce421b291dceaaf48c80298f8f",
  "preflight_manifest_sha256": "cfa5b638cc7cae4fa5013f4bf220e4a79318d287d80e24d06983f6b4d944946c",
  "helper_sha256": "390ee6dbcb7a5bd85ca4ff11def1cb1e74e7b87e34cb75e2db33b943b57aad28",
  "label_module_sha256": "413a408ed384bdd5d40ef21ea7c99891a2ee42e88554a1f6e543d1ad6f535b96"
}
```

The approval template is deliberately NOT approved and contains no fabricated approval timestamp. This package requests review and approval of the scope, not retrospective authorization.
