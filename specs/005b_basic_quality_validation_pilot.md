# E1 - Adopted minimal Basic Quality and actual24-photo validation

2026-10-07, D-026. User explicitly approves implementing the proposed draft, collecting EOS/token-cap evidence and rerunning the same24-photo pilot in a new directory. Basic policy and fixed conditional retry are approved for this validation; production pool/K/source allocation/manifest acceptance remains pending.

Use `configs/basic_caption_quality.yaml` and `configs/blip2_caption_basic_quality_pilot.yaml`. E1 backend is `scripts/blip2_caption_basic_quality_pilot.py`, separate from immutable historical E0. Preserve the same24 original train/validation IDs/order, eight businesses, checkpointCOCO commitcda95b9319b722f79c9451a3d8ff92eea02048dc, unprompted initial generation, beam3/do_sample=False, max_new_tokens64, batch1 and original GPU+CPU offload/precision profile. No test tuning, new model, fine-tuning, quantization or source increase. Fresh preflight before actual model load; stop on load/OOM/unsupported execution, save partial checkpoints.

Store decoded `raw_caption` exactly, with whitespace untouched. Derived `normalized_caption` only collapses whitespace and adds a period when missing (accounting for closing quotes). Preserve lexical tokens/case/content. Missing punctuation alone never fails or triggers retry. Normalization is not grammar correction or visual-grounding validation.

Minimal failures: empty/nonlexical/under5 words; adjacent repeated4-8-word block; existing declared six-entry exact generic list; evidence of token cap with no generated EOS. Three-word repetition remains diagnostic. Nonlexical detection does not detect all semantic nonsense. Generic-but-not-exact-match descriptions and possible ingredient/OCR/grounding errors are retained with diagnostics; no per-photo manual exclusion, CLIP/relevance/semantic redundancy/diversity at Basic Quality. Stage F keeps substantive filtering separate.

Generate at most one retry only if initial basic failure, using fixed `Describe this image in one complete sentence:` and the same beam settings. No per-photo prompts. Choose first basic-eligible attempt; if none, retain raw/normalized failure evidence and exclude only from the basic-eligible caption view, with no cohort/photo deletion or fabricated fallback.

Persist complete generated token IDs, prompt/BOS prefix IDs, suffix IDs, EOS/pad IDs, first generated EOS index, effective suffix length, cap status and termination reason. Exclude prefix before checking EOS (OPT BOS may equal EOS); ignore padding after first EOS; EOS at token cap is not truncation. Decode from the stored suffix and validate independently. Record all attempts/flags/prompt/time/peak VRAM/runtime/processor/device-map identity. New namespace/resume requires matching config/code/source/plan/runtime/map hashes.

Measure three diagnostic source views: all raw, all normalized, basic-eligible normalized. Encode whole unchanged review+metadata+caption sources without truncation/padding, compare exploratory5000/750. Neither all-normalized nor basic-eligible view is the final filtered experimental condition; K/Stage F still pending. Exact source identity and213 full original pilot reviews are independently checked with slow LEDTokenizer.

Acceptance: actual24-photo hashes/IDs/splits/settings, raw token decodes, EOS/cap, normalization/selection, all24 full source examples, fresh timing/VRAM, all original image-caption pairs visually inspected. Record raw-vs-normalized quality and compare legacy output without modifying it. Distinguish minimal-rule eligibility from semantic correctness, formal human evaluation, production precision/A2 and production pool acceptance.

Notebook00 is the primary review interface; ordinary Run All reads cache. Explicit RUN_CAPTION_PILOT routes to E1, local verified cache only. Retain E0 and draft tables/galleries as historical evidence. Update both progress logs with actual results and remaining decisions; do not freeze2300 manifest, generate GPT targets, train LED or rent compute in this pilot.

## Actual validation tooling

Active independent checker: scripts/validate_basic_quality_caption_outputs.py. The initial identity-bound checker scripts/validate_basic_quality_caption_pilot.py is retained as historical code: its tokenizer-EOS assumption failed on this checkpoint. Language model text_config uses EOS50118(newline), not tokenizerEOS2; the corrected checker reads pinned model config. Failed checks/logs preserved; generator/outputs/identity unaffected. No additional model inference was needed for this correction.
