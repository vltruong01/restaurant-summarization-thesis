# Stage I - D-053 source-based reference verification preparation

Status: researcher-approved offline preparation; human verification and future
Pilot A/B generation remain pending. The D-052 trained weights are unchanged.

## Frozen source scope

Reuse exactly the D-049 validation cases in their original order: 20 randomly
sampled businesses and four separately reported targeted cases. Condition is
Filtered only. Preserve all 226 candidate IDs/text/evidence records byte-for-byte
(136 review candidates, 90 visual candidates), all selected reviews and supplied
captions, original photo mappings and existing split membership. Candidate claims
are AI/Teacher-derived suggestions, not independent human facts.

The preparation checks exact source hashes, REV/review IDs, IMG_CAP/PHOTO IDs,
quotes and photo bytes. These structural checks do not verify claim semantics.
No accepted source, teacher target, filtering rule or checkpoint may be changed.

## Researcher annotation workflow

The source-only HTML presents complete source context and five candidate labels:
supported, unsupported, contradicted, uncertain and needs_revision. Labels begin
empty. Review and visual evidence domains remain distinct. The user supplies
notes, optional candidate importance/informativeness, corrected drafts and
missing-important-information notes. Candidate source text remains unchanged.

Supported final reference units are separate, atomic records with exact evidence,
required importance/informativeness/aspect/specificity, review polarity where
applicable, explicit atomicity/support confirmation and researcher provenance.
Splitting, correcting or adding facts is a researcher action, never an automated
semantic rewrite. Corrected text alone is not a verified reference. Uncertain and
excluded candidates remain in the audit; no forced binary/class-balance rule.
Needs-revision candidates require separately confirmed corrected/split units or
explicit exclusion with a reason before final freeze.

The researcher reviews positive/negative aspects, contrasts, specificity and
important omissions and confirms completeness for every business. Each business
requires at least one confirmed review unit under D-049. Zero-caption sources
cannot have visual reference units. Candidate decisions, units and business
completeness have confirmation snapshots; edits invalidate dependent confirmation.
Progress import/export preserves explicit provenance and immutable bindings.
Software cannot prove that a human-confirmation flag is truthful.

## Explicit D-053 photo clarification

The D-049 protocol remains byte-identical, including its historical sentence
placing photo verification outside that original protocol. The D-053 researcher
instruction now additionally requires original-photo corroboration for visual
reference units. The fact must still be recoverable from the exact supplied
caption. Photos cannot add ingredients, counts, actions or relationships absent
from that caption. No image scorer, threshold or automatic factual judge is added.
Ordinary correct but simple descriptions remain valid; low information density
is assessed separately from factual support.

## Evaluation implementation prepared before predictions

Preserve the D-049 multidomain criteria: accepted exact-sentence/adjacent-word
repetition, lexical and generated-BPE n-grams, human proposition redundancy,
verified source-unit coverage/omissions, review and caption grounding, contrasts,
cross-caption errors, zero-caption behavior, section structure and EOS/cap
semantics. Unverified/uncertain labels are separate and N/A where necessary.
No composite score or automatic architecture winner is produced.

Random-business paired intervals use the existing NumPy default_rng seed
20261016, 5,000 paired business bootstrap draws and 95% percentile intervals.
Targeted cases remain individual stress diagnostics with no pooled prevalence or
inferential interval. Teacher-target agreement is secondary, not factual truth.
No new embedding/scorer or BERTScore configuration is chosen.

The future comparison accepts exactly 48 actual bound outputs from the fresh
D-052 A/B best checkpoints, the same 24 sources and complete frozen generation
configuration: max_length 768, beam 4, do_sample false, no-repeat setting 0, and
all remaining accepted settings unchanged. This preparation cannot generate
outputs. Source annotations must validate and freeze before separately approved
generation. Historical full F is not the matched comparator.

## Acceptance and limits

Deliver the guide, HTML/photos, schema/validator, frozen evaluation config,
readiness/evidence reports, offline tests and portable ChatGPT review ZIP with
README_FIRST.md, CHATGPT_REVIEW_REQUEST.md and SHA-256 manifest.

Initial readiness must truthfully report 0 human-confirmed candidate decisions,
0 final units, 0 confirmed complete businesses and no new student summaries.
Browser visual QA and DOM/state behavioral tests must be reported separately.

No training, GPU/model inference, Teacher requests, held-out access, full RO/U/F
retraining, accepted-data edit or D-052 checkpoint overwrite is authorized.
Stop when preparation is ready for researcher review.


## D-053 standalone review bundle correction v2 (2026-10-16)

The independent review requested packaging fixes before researcher annotation.
Corrected export: `outputs/stage_i/stage_i_d053_reference_verification_chatgpt_bundle_v2.zip`.
Active corrected worksheet: `outputs/stage_i/stage_i_d053_reference_verification_chatgpt_bundle_v2/SOURCE_REFERENCE_ANNOTATION.html`.
The old v1 worksheet/package and all prior history remain preserved; use v2 for the next interface review.

The ZIP corrects schemas/, includes the original paired-pilot bootstrap helper,
and supplies `python code/check_stage_i_d053_bundle_readiness_v2.py --readiness`.
This portable preparation checker verifies14 supplied frozen bindings and both
adapter hashes without loading models. Historical target/output-containing case
inputs and the blind-order key stay separate; both are explicitly NOT_REVERIFIED
in the standalone export. Original full-project evaluation readiness is pending,
not bypassed. The original frozen config/evaluator/hashes remain unchanged.

The release-candidate ZIP passed all83 original offline tests (22 annotation,
34 evaluation,27 UI; zero skips) after extraction outside the project, plus30
supplemental UI and8 readiness checks. Final ZIP test receipts are in
`outputs/stage_i/d053_standalone_bundle_correction_v2/final_zip_verification/`.
UIv2 also fixes candidate-confirmation controls to permit already-approved optional
priorities; mandatory final-unit metadata and confirmation safeguards are unchanged.
The embedded source/initial annotation payload is identical to v1.

Practical Chrome/Edge smoke remains NOT_EXECUTED_POLICY_BLOCKED: Windows computer
use could not reliably determine the current browser URL and stopped. No bypass
was attempted. Synthetic Node/DOM checks are not practical browser QA, so this
acceptance item remains pending. See the correction report and manual smoke guide.

The researcher must inspect every full selected review, caption and original
photo, including source material absent from the Teacher candidates, and add
important omissions as separately confirmed atomic units with evidence. Notes
alone do not add a reference unit. Retained scope:24 validation businesses,
652 reviews,90 photos,226 unverified candidates,1,407 exact quote checks; one
zero-caption case. No real labels/units were created:0 confirmed candidates,
0 reference units,0 completed businesses. No predictions, training, Teacher/API
requests, dataset edits or checkpoint changes. Next: review corrected interface,
complete practical smoke, then researcher verifies one or two businesses; no
automatic progression to full annotation or48-output inference.



## D-053 researcher-approved handoff imported; final reference freeze BLOCKED (2026-10-17)

Input: D053_CODEX_FINAL_REFERENCE_HANDOFF.zip, SHA-256
`ea1a73662411964411e738baca8d5a756325a32a94587ba88e7ae2dd5d903d0b`.
The direct researcher conversation approves203 Supported candidate decisions,
21 exact corrected texts subject to original evidence rules,and two Uncertain
exclusions for all24 fixed validation businesses. That chat approval is recorded
separately; the archived edited JSON remains byte-identical with all226 original
human_verified flags false. Imported candidate confirmations are explicit chat
approval materialization, not fabricated UI clicks or prior inspection timestamps.

All202 nested source-bundle files match hashes. The24 cases retain652 full
reviews,90 photos/captions,226 original candidate IDs and1,407 exact evidence
bindings. No original review/caption/source/order/checkpoint/protocol was changed.
Both uncertain cases have zero unit links and remain outside scoring, including
the one with nonblank corrected text.

Finalization is NOT ready.15 of21 approved corrections add/replace a detail not
established by the exact LED-visible caption; photo-compatible wording alone is
insufficient. All21 approved texts remain unchanged. Separate caption-safe
alternatives are proposals only and await researcher adjudication. Six corrections
have no new caption fact identified. Five potential important source omissions
are shown with exact REV quotes (noise/conversation,happy-hour refusal,pickup
errors,cold-brew availability and a non-chicken dish concern); none was auto-added.

All226 submitted importance fields are null. Candidate priorities are optional,
but final units require core/supporting,aspect,specificity,review polarity and
separate unit-level confirmation under the frozen protocol.321 exact clause
spans are construction proposals,not completed atomic references or a scoring
denominator. Their parent context,residual lists and evidence attribution remain
pending semantic atomization; no human unit metadata/confirmation is invented.
Final business completeness remains pending while these issues resolve. Original
strict final acceptance and reference_freeze correctly BLOCK; no final reference
freeze has been emitted and no acceptance criterion was relaxed.

Artifacts: outputs/stage_i/d053_researcher_reference_finalization_v1/.
Portable review snapshot: outputs/stage_i/stage_i_d053_researcher_reference_resolution_bundle_v1.zip.
Open FINALIZATION_REVIEW.html inside that bundle to inspect exact corrections,
photos,omission examples and all original source material. The package is an
immutable BLOCKED resolution handoff,not accepted final references. Final ZIP
software-test receipts are stored in FINAL_ZIP_TESTS/ beside the workspace report.

This records approval provenance and execution status; it does not amend D049/
D053 evaluation criteria. No Teacher request,new Pilot A/B prediction,training,
held-out access,method retuning or checkpoint change. Stop for researcher
resolution before true atomic-reference freeze;48-output inference still requires
later independent review and explicit authorization.

