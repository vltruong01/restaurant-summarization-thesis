# Stage F annotation roles and provenance

The user is the primary human annotator. The current workflow is:

`AI pre-label -> user independently inspects image + caption -> user chooses/edits label -> optional ChatGPT comparison -> user confirms final label`

"Independently inspects" means that the user makes their own image/caption judgment. This model-assisted workflow does not imply a blinded independent annotation experiment. The final 30-business human evaluation remains separate and unchanged.

## Separate roles

- `ai_suggested_label`: initial AI pre-annotation, preserved as a suggestion.
- `chatgpt_second_reviews`: optional second opinions, reasons and the associated review-file SHA; these are comparison/adjudication aids, not human ground truth.
- `human_final_label`: the user's explicit final decision. It is usable for calibration only when `human_verified=true` and the existing protocol gates pass.

Opening the form, displaying a suggestion or saving a draft does not confirm a case. A named user must explicitly confirm each case. Editing a label invalidates its prior confirmation. The user can agree with a suggestion or choose a different label.

When ChatGPT comparison is used, the intended provenance wording is:

> ChatGPT review was used only as an optional comparison/adjudication aid after the user's own inspection. The final annotation remained a user-confirmed human decision.

## UI and future exports

The second-review panel remains optional. Its confirmation button states that the user confirms a final label after comparison; it does not state that ChatGPT labeled the case. Future confirmations record `verification_source=user_confirmed_after_chatgpt_comparison` when that route is explicitly used, with `human_confirmation_provenance.final_label_author=human_user`, the decision role and the second-review SHA. Other confirmation routes also describe the user's own inspection/selection.

These are self-reported user confirmations, not a machine attestation of inspection. If another comparison route was not explicitly recorded, metadata says `not_recorded` rather than claiming the user did not consult an aid.

Existing `verification_action` codes (`accepted_ai`, `corrected_ai`, `manual`) remain compatible with the existing importer. They describe the user's verification route relative to the initial suggestion; they do not make AI the final annotator. Legacy `verification_source=chatgpt_second_review` tags mean the user confirmed after consulting that panel, not that ChatGPT authored the human label. Historical records are not rewritten.

## Unchanged research constraints

This clarification changes UI wording and future provenance descriptions only. It does not change any prior human labels, initial AI suggestions, second-review records, captions, scores, queues, roles, threshold/K/budget values, source datasets or scientific outputs. Threshold fitting continues to use only explicitly verified human-final labels. No new threshold fitting or K selection is performed for this clarification.
