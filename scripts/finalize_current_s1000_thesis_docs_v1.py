"""Finalize concise existing documentation and exact paths; preserve original bytes in snapshots."""
from pathlib import Path
import json,hashlib,difflib,re
R=Path(__file__).resolve().parents[1];O=R/'outputs/pre_evaluation/s1000_readiness_v1'
e=json.loads((O/'EVIDENCE/MODEL_FREEZE_AUDIT.json').read_text())
models='\n'.join(f"- **{a}-S1000:** `{v['adapter_sha256']}`" for a,v in e['models'].items())
texts={
'thesis_sdd_docs/THESIS_PROGRESS.md':'''# Thesis progress — final method and pre-evaluation

Updated: 2026-10-28. **Source-Separated LED-large S1000 is selected and frozen.** Training is complete. Final TEST evaluation, independent references and human evaluation have not been executed. The original logs and exact earlier documentation remain preserved separately; this narrative focuses on the accepted method and remaining work.

## Completed methodology and artifacts

- Frozen Yelp cohort: 2,300 restaurant businesses, split into TRAIN2000, VALIDATION200 and TEST100 with no business overlap. Preserve the previously approved main TEST100 and secondary 83-business exact-image-disjoint analysis; business disjointness alone does not imply image independence.
- Representative review selection: accepted v4.2 length-balanced 4000-token scenario, at most 30 whole reviews, star/time stratification and original `(date, review_id)` order. RO/U/F/S share review content, ordering and allocation. No TEST tuning.
- BLIP-2 COCO caption production completed: 30,027 caption records. Unfiltered uses Basic-valid captions and fixed-seed ranking. Filtered uses the accepted exact-pixel duplicate collapse, relevance, semantic redundancy and diversity ranking. CLIP is soft-only; pHash is diagnostic-only. Maximum K=5 was frozen from validation calibration/K study. Budgets remain full source 5000, review block 4000 and incremental visual context 750; no silent truncation or reselection.
- Original source-grounded, condition-aligned Teacher targets completed for TRAIN/VALIDATION only. Review supervision is shared; U/F Visual supervision follows its own caption inputs and evidence. Provenance and support links are retained; Teacher targets are imperfect supervision, not factual gold. No TEST Teacher targets.
- Official LED-large with one shared Expanded-LoRA adapter: rank 8, alpha 16, dropout 0, 456 tensors and 4,915,200 trainable parameters. Review-only and caption-only tasks are explicitly isolated, with BOS-only global attention. Native CE, FP16, AdamW, LR `2e-5`, warmup 25, linear horizon 1000, physical batch 1; original label text/EOS preserved.
- Final RO-S1000/U-S1000/F-S1000 completed 1,000 new successful updates after their own RO750/U750/CE3-750 parents. Each retains 4,000 new Review presentations; U/F additionally have 4,000 Visual presentations. RO uses four actual Review microbatches divided by 8; U/F use four Review plus four Visual divided by 8. Initialization histories, caption exposure, auxiliary gradients, normalization, different actual work and material recovery provenance remain explicitly documented in the freeze audit.
- VALIDATION200 is complete: 600 assembled outputs and 1,000 raw task generations with token IDs and exact source/model bindings. Full states, logs, AMP/numerical evidence and archive-member/original-timestamp checks are preserved off-instance. F predictions were reused unchanged. Instance 58067102 was destroyed on the explicit human closure request; provider confirmation is saved. No replacement GPU is allocated by this workflow.

## Current evidence and limitations

All 200 validation businesses have descriptive diagnostics. The bounded 24-case primary panel plus one zero-F-caption supplement has model-assisted source-grounded analysis, with prior F annotations reused transparently. Local repetitions, genericness, useful details, important omissions and attribution errors remain. This is not independent human evaluation, exhaustive 200-case semantic adjudication or proof of unseen TEST superiority.

All Review calls are caption-free. Their differences reflect adapters and training histories, not direct caption conditioning during Review inference. U/F caption sets and Visual labels differ, so comparisons are of condition pipelines. Caption fidelity is not proof of accuracy against the original photo. S changes only Visual input and reuses F; no separate S model is trained.

## Pre-evaluation preparation completed

The final models/backbone/tokenizer and stored output hashes have been reverified. The read-only TEST100 audit verifies 2,817 selected reviews, 500 U captions, 341 F captions, all 1,136 original photo files and 660 selected photo/caption provenance bindings. Two zero-F-caption businesses are retained. No TEST reference payloads, predictions or quality metrics were accessed.

Shuffled specification and CPU implementation are prepared, with eight focused tests. Essential donor/reuse/fallback/zero policies await approval; the actual TEST mapping is uncommitted. TRAIN/VALIDATION-only mimicry screening separates writing templates, exact/near copying, source support and semantic errors. External-data feasibility was investigated without downloading datasets or executing models.

## Remaining work

1. Approve the four-condition protocol and exact Shuffled policy, including F/S identical-Review reuse. Freeze actual donor mappings before outputs.
2. Independently write, verify and freeze common TEST100 references: Review from selected reviews; Visual from original photos. Authenticate authorship, blinding and evidence. Development references and Teacher targets are not TEST gold.
3. Freeze metrics/scorer versions, paired business-level statistics, primary contrasts/multiplicity and main100/secondary83 reporting. Obtain separately approved compute.
4. Run finite final RO/U/F/S TEST100 inference with unchanged settings and complete raw tokens/provenance. Do not tune or reselect from TEST.
5. Complete the planned blinded 30-business, three-rater human study with original photos, rubric, agreement and uncertainty reporting. Confirm the actual membership manifest before use.
6. Review external baselines and small external generalization separately, subject to licensing, data and compute approval.
7. Write results, discussion, limitations and computational analysis; prepare the thesis defense.

Current package: `outputs/pre_evaluation/s1000_readiness_v1/`. Decisions remain in root `DECISIONS.md`; roadmap and method record remain in their existing files below. Stop for researcher/ChatGPT review. No new training, TEST inference, reference creation, human evaluation or paid resources are authorized by this update.
''',
'thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md':'''# Final accepted LED-large Source-Separated methodology

Updated: 2026-10-28. This existing filename remains the methodology entrypoint. Earlier proposal text is preserved in the pre-evaluation documentation snapshot. S1000 is selected; this record proposes no new training experiment.

## Frozen model identities

'''+models+'''

Each model uses the original pinned LED-large backbone and tokenizer, with one shared Expanded-LoRA adapter (r=8, alpha=16, dropout=0, 228 modules/456 tensors). Exact backbone, adapter, full-state, tokenizer and output hashes are in `outputs/pre_evaluation/s1000_readiness_v1/FINAL_MODEL_FREEZE_AUDIT.md` and its structured evidence. Old container base paths in adapter configs are provenance, not portable resolvers; restore with the explicit verified local backbone.

RO starts from own RO750, U from own U750, F from caption-trained CE3-750. The selected S1000 phase adds 1,000 successful updates with fresh optimization state, not a 250-update continuation or an uninterrupted 1750-step optimizer. Each new phase has 4,000 Review presentations; U/F have 4,000 Visual presentations. RO's four Review microbatches each divide by 8, versus four Review plus four Visual each divided by 8 for U/F. Equal Review normalization does not equal total compute, auxiliary gradients or clipping trajectories. Retained recovery state, actual discarded work and reproducibility limits remain in scientific evidence.

## Actual training and inference

Review input is `[TASK_REVIEW_CONSENSUS]` followed by original business metadata and selected reviews, with no captions. Visual input is `[TASK_VISUAL_EVIDENCE]` followed by `[VISUAL_CONTEXT]` and own selected captions, with no reviews or business metadata. U/F share one adapter across these two tasks; RO has no Visual task or Visual score. Original component labels remove initial target BOS only and preserve EOS. Output components concatenate verbatim.

Native CE, FP16 autocast with FP32 parameters/Adam moments, AdamW, LR `2e-5`, warmup 25, linear horizon 1000, physical batch 1 and seed 42 remain fixed. Source/review/incremental-caption budgets are 5000/4000/750 and K<=5. Full source and labels are not silently truncated.

DecoderPolicyV2 remains beam size 4, length penalty 1.0, max_length 768 **per task**, sampling disabled, original early stopping/cache/EOS/forced-EOS behavior, decoder-start 2, and global attention at BOS position 0 only. No added reranking or repetition penalties. U/F have two call budgets; RO has one.

## Differences from proposal v16

The proposal illustrates a joint review-plus-caption source. The accepted implementation uses explicit Review-only and Visual-only calls. Its provisional 256–512 output-length and LR `2e-4` table values are not actual final settings: the completed system uses 768 per call and the accepted `2e-5` new-phase schedule with condition-specific original750 initialization. Proposal text suggesting whole-caption removal to fit budgets was replaced by the accepted record-and-stop overflow policy; frozen inputs are not silently changed. The proposal document itself is unchanged.

Shuffled preserves evaluation business A and its reviews, replaces only the Visual input with filtered captions from another TEST business B, and reuses F-S1000. Exact allowlist, donor reuse versus permutation, fallback and zero/no-donor behavior remain approval decisions. F/S Review invariance is structural, not evidence of robustness to wrong captions in a joint Review decoder. U/F comparisons remain pipeline comparisons because caption sets, inherited adapters and Visual supervision differ.

Final TEST references must be common and human-authored: Review from selected reviews, Visual from original photos rather than any particular caption subset. Teacher targets are not unquestionable ground truth. The planned blinded 30-business study uses reviews/photos/outputs; it has not been executed. Preserve the previously approved main100 and exact-image-disjoint83 secondary analysis. Pre-evaluation integrity does not authorize final inference or quality scoring.
''',
'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md':'''# Thesis implementation roadmap — evaluation and completion

Updated: 2026-10-28. S1000 is selected and training is complete. Current status is maintained in `THESIS_PROGRESS.md`; the existing method record is `LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md`; the decision log is root `DECISIONS.md`. Original histories remain preserved, without reopening optimization.

| Workstream | Actual state | Next prerequisite |
|---|---|---|
| Yelp cohort/splits/reviews | Complete and frozen, 2000/200/100 | No retuning |
| BLIP-2, U/F, K=5, budgets | Complete and frozen | Preserve caption/photo provenance |
| TRAIN/VALIDATION Teacher supervision | Complete, condition-aligned and imperfect | No TEST Teacher labels |
| RO/U/F Source-Separated S1000 | Complete, selected, backed up off-instance | No retraining/reselection |
| VALIDATION200 | 600 outputs/1000 sequences, diagnostics plus bounded25 analysis | No independent full200 semantic claim |
| TEST100 source readiness | PASS, 1136 photo hashes, two zero-F cases | Reference/protocol approval |
| Shuffled negative control | CPU implementation/eight tests | Approve policy, then freeze actual mapping |
| Teacher mimicry | TRAIN/VALIDATION screening and source examples complete | Treat as risk evidence, not causal proof |
| Common human TEST100 references | Coverage/authorship not authenticated in scoped metadata | Independent authoring/verification/freeze |
| Metrics/statistics | Final protocol pending; pinned BERTScore scorer not established | Freeze versions/contrasts/paired tests |
| Final TEST inference | NOT RUN | Explicit release and available compute |
| Human 30-business study | NOT RUN; frozen membership not authenticated | IDs/rubric/blinding/raters/agreement plan |
| External baselines/generalization | Feasibility only | Licensing/data/protocol/compute approval |
| Results and defense | Accepted method/status updated | Evaluate, analyze, write limitations |

Sequence: researcher approval -> independent TEST references and evaluation/human protocol freeze -> deterministic donor mapping and actual source checks -> finite authorized TEST inference -> paired automatic analysis and blinded human study -> separately approved external pilot/baselines -> results/discussion/defense. Prepare references before predictions where feasible to protect independence.

Retain original outputs and technical failures; do not selectively regenerate. Report main100 and the already defined image-disjoint83 secondary set without choosing subsets from outcomes. The three frozen pipelines have different caption histories/Visual supervision and total work; disclose these limits. Source-separated S changes only Visual and does not train a fourth model.

Current authority covers CPU audits, source/token preparation and documentation only. Instance 58067102 was provider-confirmed deleted on the previous explicit human request. No replacement rental, TEST execution, reference authoring or human assessment is authorized now. Review `outputs/pre_evaluation/s1000_readiness_v1/PRE_EVALUATION_DECISIONS_AND_BLOCKERS.md`, then stop for researcher/ChatGPT decisions.
''',
'thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md':'''# Current researcher / ChatGPT handoff

Updated: 2026-10-28. The final development configuration is **LED-large Source-Separated S1000**. RO/U/F are trained and frozen. S reuses F and is never separately trained. Historical records do not grant new execution permission.

Read existing `THESIS_PROGRESS.md`, `LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md` and `CODEX_THESIS_IMPLEMENTATION_PLAN.md`. Current reports/evidence are in `outputs/pre_evaluation/s1000_readiness_v1/`; the review ZIP contains six reports, evidence, scripts/tests and updated documentation, without checkpoint binaries, bulk Yelp data or TEST references.

'''+models+'''

Full states, logs and VALIDATION outputs are preserved under `outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1/` and the original F artifact root. Restore with the explicit pinned local backbone, not historical container base paths. Instance 58067102 has provider-confirmed DELETE success; no replacement GPU was allocated by this workflow.

The read-only TEST100 input audit passes: 100 business-disjoint IDs, 2817 whole ordered reviews, 500 U/341 F captions, two zero-F cases, all1136 original photo hashes. No TEST predictions, metrics or reference payloads were accessed. Preserve prior main100 plus image-disjoint83 secondary reporting. Common TEST reference authorship/coverage and the frozen30-case human-study membership are not authenticated in scoped metadata; do not claim they exist or are absent everywhere.

Review inputs are identical and caption-free; Shuffled changes only Visual. Donor reuse versus permutation, cuisine allowlist/fallback, zero/no-donor handling and F Review reuse await approval. The actual TEST donor mapping is uncommitted. Teacher mimicry flags are descriptive; bounded source examples are model-assisted, not independent human judgments. No claim of best unseen TEST performance follows from development selection.

Next, researcher/ChatGPT reviews `PRE_EVALUATION_DECISIONS_AND_BLOCKERS.md`, approves exact matching/reference/evaluation policies and later separately releases compute/inference. No retraining, Teacher revisions, TEST reference creation, human ratings, external-model experiment or automatic rental is authorized by this package. Stop after preparation.
'''}
for name,text in texts.items():(R/name).write_text(text,encoding='utf8')
before=O/'DOCUMENTATION_BEFORE'
entry='''## PRE-EVAL-20261028 — freeze selected S1000 and prepare final evaluation

The researcher selected Source-Separated LED-large S1000 with condition-specific RO750/U750/CE3-750 initialization and 1000 new successful updates. Final RO/U/F adapters, full states and VALIDATION200 outputs are preserved off-instance and reverified. S reuses F and changes only Visual; no fourth model is trained. TEST100 source-only integrity passes; no TEST reference, prediction or quality assessment was performed. Preserve D-022 main100 plus the existing image-disjoint83 secondary policy.

The exact Shuffled allowlist, donor reuse/fallback and zero/no-donor rules remain proposed, not approved; actual TEST assignment is uncommitted. Independent common TEST100 references, the30-case human-study membership and final metrics/statistical protocol require authentication/approval. TRAIN/VALIDATION mimicry screening is descriptive, not causal proof or an automatic semantic judge. No checkpoint reselection, retraining, TEST/external execution or rental is authorized. Instance58067102 was provider-confirmed deleted on the human closure request. Current six reports/evidence: `outputs/pre_evaluation/s1000_readiness_v1/`.

'''
(R/'DECISIONS.md').write_bytes(entry.encode('utf8')+(before/'DECISIONS.md').read_bytes())
assert (R/'DECISIONS.md').read_bytes().endswith((before/'DECISIONS.md').read_bytes())
summary='''# Existing documentation updated

Four existing files under `thesis_sdd_docs` were updated in place:

| File | Change |
|---|---|
| `THESIS_PROGRESS.md` | Accepted milestones, selected S1000, evidence limits, current state and remaining work |
| `CODEX_THESIS_IMPLEMENTATION_PLAN.md` | Evaluation-first roadmap and approval sequence |
| `CHATGPT_NEW_CONVERSATION_HANDOFF.md` | Exact artifact paths/model identities, current authority and stopping point |
| `LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md` | Actual final methodology, provenance and differences from proposal v16 |

Root `THESIS_PROGRESS.md` points to the authoritative existing progress file. Root `DECISIONS.md` adds one dated entry and preserves every previous byte after it. Root `README.md` provides a current-status pointer. No duplicate progress/decision log was created under `thesis_sdd_docs`.

Exact originals and unified diffs remain in local `DOCUMENTATION_BEFORE/` and `DOCUMENTATION_DIFFS/`; before/after hashes are included in the review. Original experiment logs, model weights, targets and splits remain unchanged. No historical account of unsuccessful techniques was added to the current thesis narrative. Material provenance and scientific limits remain explicit.
'''
(O/'THESIS_SDD_DOCS_UPDATE_SUMMARY.md').write_text(summary,encoding='utf8')
# Repair only whitespace inserted into literal identifier/path portions of new report prose.
for p in O.glob('*.md'):
 s=p.read_text(encoding='utf8');s=re.sub(r'(?<=[_/])([sSvV])\s+(\d+)',r'\1\2',s)
 p.write_text(s,encoding='utf8')
changes=json.loads((O/'EVIDENCE/DOCUMENTATION_CHANGE_MANIFEST.json').read_text())
for c in changes:
 p=R/c['path'];c['after_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 text=''.join(difflib.unified_diff((O/c['original_snapshot']).read_text(encoding='utf8').splitlines(True),p.read_text(encoding='utf8').splitlines(True),fromfile=c['path']+' (before)',tofile=c['path']+' (after)'))
 (O/c['diff']).write_text(text,encoding='utf8')
(O/'EVIDENCE/DOCUMENTATION_CHANGE_MANIFEST.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf8')
print('Four existing thesis records finalized; exact paths/model identities and original decision-log tail verified')
