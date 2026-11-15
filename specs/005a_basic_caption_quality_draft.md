# Basic caption quality - minimal normalization draft

Status: **DRAFT, NOT ADOPTED**. User asks to consider the proposed distinction between raw captions, cosmetic normalization and basic failures. This document and its cached24-photo replay are a concrete proposal, not a change to the accepted E0 reports or production protocol.

## Data and normalization

Keep each model attempt's `raw_caption` exactly as stored, with photo/business IDs, generation settings and identity. Create a separate `normalized_caption`: collapse whitespace and append a terminal period only when needed, accounting for closing quotes/brackets. Empty/nonlexical output does not become a period. No case/word/content/grammar/OCR corrections. Missing punctuation is a diagnostic, never a basic failure. A caption fragment can be usable without being rewritten into a grammatical full sentence.

Normalization is applied consistently before downstream tokenization in both Unfiltered and Filtered conditions if adopted. Re-encode full sources after normalization; do not assume token cost is unchanged.

## Proposed reproducible basic failures

- Empty output, no alphanumeric/lexical content, or fewer than5 word tokens. This narrow nonlexical rule does **not** detect every semantically meaningless string.
- Adjacent repetition of the same4-8-word token block. Three-word repetitions are diagnostic only, because recursive descriptions can be legitimate (e.g. a screenshot of a photo).
- Exact case-insensitive match to the existing declared generic list, after whitespace/outer punctuation normalization: `a plate of food`, `a plate of food on a table`, `a picture of food`, `an image of food`, `a restaurant`, `a table with food`. No new generic regex or ad hoc list additions after individual inspection. Longer simple captions remain eligible under this draft, even if information content is limited.
- Evidence of truncation: generation token cap reached AND the complete generated sequence contains no EOS. Missing punctuation alone is insufficient. Existing pilot checkpoints lack stop/EOS metadata: mark truncation unknown, do not fabricate either failure or verified completion. Before any new pilot/full generation, instrument stop metadata separately without changing model/generation settings.

No automatic grammar repair or ingredient/OCR correction. Grounding errors and generic-but-not-exact-match captions remain diagnostics at Basic Quality; do not selectively remove them by manual judgment from Unfiltered. Substantive image-text consistency, relevance, redundancy and diversity belong to Stage F.

## Retry and experimental boundary

Keep the main unprompted COCO model/beam3 setting. At most one retry, triggered only by declared basic failures, with one protocol fixed using train/validation before production. The existing complete-sentence prompt can be evaluated as a candidate; it is not newly validated or frozen by this replay. Do not retry punctuation alone, vary prompts per photo, or use test to pick rules/prompts.

Offline replay first evaluates the existing initial attempt; only if it fails does it inspect the already-recorded fixed-prompt retry. This is historical-output reuse, **zero new model calls**, not a new prompt experiment or runtime measurement. Both attempts failing leaves the photo's raw records and failure reasons available; no business/cohort/image-pool deletion or fabricated fallback caption.

The same basic eligibility and normalized text must feed both caption conditions if adopted. No CLIP, relevance, semantic deduplication/diversity or individual quality judgments in this baseline step. Final K, caption/source budgets, full2300 manifest, teacher/train and A2 stay pending.

Rules were developed with access to these24 train/validation outputs. Replay results are development diagnostics, not an independent quality benchmark or formal human evaluation. Approve/revise this exact draft and the stop-metadata/retry plan before adopting a new production policy.

Reference: [implementation plan, Basic Quality and Stage F](../thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md), [accepted E0 pilot](005_blip2_caption_feasibility_pilot.md). Machine-readable rules: `configs/basic_caption_quality_draft.yaml`. Outputs: `outputs/stage_e/basic_caption_quality_draft_v1/`.

## Adoption after this historical draft

2026-10-07 D-026: user explicitly approves implementation and actual24-photo validation. This file/replay remains the historical draft. Active policy: configs/basic_caption_quality.yaml; active validation protocol: specs/005b_basic_quality_validation_pilot.md. Stop metadata and actual retry behavior are validated in the new E1 directory; final production pool/K/budgets remain pending.
