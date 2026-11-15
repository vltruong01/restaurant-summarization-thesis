# Local dataset preparation and Vast.ai handoff

**Current execution guidance (2026-10-09, D-033):** K5 and5000/4000/750source template are frozen. Local fullproduction job explicitly stopped by the user with2709/30027captioned photos retained;27318pending. For a LinuxRTX3090Codex session, use `thesis_sdd_docs/STAGE_F_VAST_CODEX_HANDOFF.md`. It records current decisions, transfer requirements and the required separate cloudruntime/remaining-photo namespace. The historicalE0pilot packages and D027validation-only scope below are provenance, not the current production recipe. Cloudadapter/required-file package preparation is now authorized and implemented; see cloud/stage_f_d033/README.md and outputs/cloud_handoff/stage_f_d033_v1/package_report.json for final integrity evidence. No instance rented, upload performed or remoteproduction resumed.

Last updated: 2026-10-06
Status: Planned execution workflow; no GPU instance has been rented or data uploaded.
Authority: user instruction recorded in `DECISIONS.md`, D-001.

## Immediate E0 handoff — compact pilot, not the final training package

D-025: user cannot wait for local15.5GBweights transfer. Local controller stopped,~3GBverifiedchunks retained,0model loads/captions. LocalRTX3050feasibility remains unverified. Agent prepared `outputs/stage_e/blip2_caption_feasibility_pilot_v1/cloud_handoff_20261006/e0_cloud_pilot.zip` **2.08MB**, independently validated24originalphotos/8businessrecords/213complete originalselectedreviews and8exact4000-tokenreview blocks. Parent IDs/order/checkpoint/beam3/FP16/prompt/sourcebudgets unchanged. This package is an explicitly marked sourceSLICE for E0, not the2300businessmanifest/fullStageA2/trainingdata.

Choose a singleCUDA GPU with engineering target>=16GiBVRAM/availableRAM>=16GiB/disk>=60GiB and good host download speed; actual price/network/compatibility must be reviewed per offer. Prefer an alreadycachedcompatiblePyTorch template/runtime. Account/budget/offer permission pending; no instance rented, data uploaded or credit charged. Server will still download pretrainedweights and speed is not guaranteed. Use [Vast.ai quickstart](https://docs.vast.ai/guides/get-started/quickstart) for current account/SSH/template/network filtering; avoid assuming current prices from old examples.

Upload/extract the small ZIP only after host approval, enter `e0_cloud_pilot` and run `bash setup.sh`, then `bash run_one_photo.sh`. Do NOT run backendprepare on the subset. Firstphoto writes actualCUDA/load/device-map/time/VRAM/caption reports then releases model. Return the results for image-caption review; only afterward `bash resume_24_after_review.sh` runs remaining photos/report/independentvalidation. Cloud outputs have a separate runtimeidentity/directory and do not establish local6GiBfeasibility. Keep finalselector/allocations/captionprotocol/K/teacher/LED/A2 pending. No need to transfer fullrawYelp/corpus/prior19GBexperimentoutputs for this small pilot.

Local hardware update 2026-10-05: user reports a 6 GB RTX 3050; `nvidia-smi` confirms **NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB VRAM, driver 596.36**. Use local GPU for suitable model stages when actual VRAM/runtime feasibility is established; the existing tokenizer and rating/time subset search execute on CPU and do not directly use CUDA. This inventory is not a passed BLIP-2/LED GPU feasibility gate and does not change the planned source length, model precision/architecture or Vast.ai A2 requirements. No model/GPU job or rental launched for the expanded C2 diagnostics.

## 1. Objective

Prepare the 2,300-business source dataset locally, then execute computationally intensive experiments on a rented Vast.ai GPU without changing the research design. This handoff checklist supplements the master plan and does not replace stage specifications.

## 2. Local milestone before GPU rental

Required artifacts, validated under the corresponding stage specs:

- Immutable cohort and business-level split: 2,000 train, 200 validation, 100 test.
- Eligibility audit, including the original 10,197-business count and any effect of English-language preprocessing.
- Pre-sampling English gate (D-006): >=10 detected-English reviews and >=5 local associated photos. Transfer its config/decision/evidence reports with the cohort. The completed gate confirms all 10,197 baseline businesses qualify and the post-gate seed-42 draw matches v1; it does not replace complete English review classification/selection in Stage C.
- Validated C1 evidence: exhaustive labels/token metadata for 601,493 cohort reviews and complete English pool at `data/processed/yelp_2300_english_v1/` (600,862 en reviews; original IDs/text/split unchanged). Transfer the diagnostic config, identity and acceptance reports when packaging. This pool references photos from v1 and is not a standalone GPU package; C2 must still freeze representative <=30-review inputs and budgets.
- Selected English-language review text with review IDs, business IDs, dates, stars, final chronological order, and structural markers or their reproducible construction rules.
- Review-selection diagnostics covering rating proportions, temporal coverage, and the fixed review-token allocation.
- Current C2 complete-review budget pilot: spec `003c_budget_constrained_pilot.md`, notebook 05 and versioned outputs `outputs/stage_c/review_selection_budget_pilot_v3/`. This explores 3,500/4,000/4,500 review-block tokens on 56 train/validation businesses; it does not provide a frozen all-cohort input manifest or final review/metadata/caption allocation. Transfer pilot provenance as supporting evidence once validated; finish content review/final C2 before claiming the source package is ready.
- Subsequent expanded C2 diagnostics under spec 003d are independently validated on all 2,000 train + 200 validation businesses, 575,749 English originals and 19,800 runs, with notebook 06 8/8 cells PASS. Same exploratory budgets retain mean 28.27/28.43/28.51 complete reviews, no <10/<5 cases; final source allocation and all-2,300 manifest remain pending. Preserve `outputs/stage_c/review_selection_train_validation_v3/` report/validation/identity as supporting evidence. Manual content/aspect review and possible short-review bias must be reviewed before freezing shared inputs; these diagnostics do not establish BLIP-2/LED GPU feasibility or a ready training package.
- Latest spec003e length/content audit confirms systematic short-review shift across2,200 train/validation, with median selected/source mean ratios~0.383 at all budgets and199 exactly30 inputs under1500 tokens per budget. Notebook07 7/7 and independent diagnostics PASS; five low-token examples admit longer same-star/time-bin alternatives well under3500. Counterfactuals are not training evidence. Preserve `outputs/stage_c/review_selection_length_audit_v1/` as supporting evidence; review candidate construction/token tie-breaking and run a new pilot before freezing selected inputs or claiming a training-ready package. Source/cohort/split/model/evaluation design unchanged.
- Subsequent v4 pilot under003f is validated on80 train/validation businesses (40,407 originals/960 runs), reducing median length shift to ratios0.808/0.855/0.876 but retaining count/rating/time tradeoffs. Preserve `outputs/stage_c/review_selection_length_balanced_pilot_v1/` comparisons/identity/validation; v4 not frozen or expanded to2300. The user-facing maintained notebook is now00, including all past/current/planned dataset/filtering stages and review viewer; transfer it and helper dependencies in any future source package. Final manifest/allocation and GPU feasibility/A2 remain pending.
- LED tokenizer identity/revision and token diagnostics; metadata, markers, and captions must fit the documented total source budget without displacing the frozen reviews.
- Business metadata used by the model, plus provenance for category and photo-label fields used in filtering.
- Original photos associated with the selected cohort and a portable photo manifest containing at least photo ID, business ID, split, and relative file path.
- Stage configs, local dependency lock, validation reports, progress/decision logs, and SHA-256 checksums for transferred artifacts.

Image readability checks and duplicate indexing may be performed locally after Stage D has a valid spec. Save statuses and duplicate groups separately from the original photo pool. Duplicate exclusion belongs to the filtered condition; preserve the eligible pool needed by the unfiltered condition.

The minimum of 5 photos is an eligibility criterion. Do not keep only 5 photos per business as a storage shortcut, choose K early, or discard images required for later filtering or independent human references.

This milestone is a source dataset, not a train-ready supervised LED dataset. BLIP-2 captions, validation-selected caption sets, teacher targets, and the three condition-specific LED datasets are later outputs of Stages E–H.

## 3. Portable package

Use one package root with configurable input/output roots and paths relative to that root. Scripts must not require `C:\thesis_proposal` or a specific Linux mount path. Keep raw files read-only.

Include code, configs, specs, dependency locks, manifests, selected review text, required business/photo metadata, diagnostics, and the original cohort images. Keep train/validation/test membership explicit in all manifests.

Do not transfer the complete raw review/user/check-in/tip files merely to rebuild already-frozen inputs. Include additional raw data only if a later stage has a documented dependency on it. Preserve the full raw source locally for provenance and reproducibility.

Record:

- Package version, file list, byte size, checksum algorithm, and per-file checksums.
- Exact split/review manifest checksums and the scripts/configs that produced them.
- Expected split counts, selected-review totals, photo totals, and missing/unreadable-file statuses.
- Original dataset provenance and associated usage-agreement documents.
- Remote package root and durable result destination once chosen.

Held-out test data remains final-evaluation data in every environment. GPU feasibility and configuration selection use synthetic inputs or train/validation examples. Do not generate GPT pseudo-reference targets for held-out test businesses or use their references/results for tuning.

## 4. Cloud preparation and A2 acceptance

Before a production run:

1. Record GPU model/VRAM, OS, driver, PyTorch CUDA runtime, Python, library versions, container/runtime identity, and dependency installation procedure.
2. Verify every transferred artifact against the package checksums. Confirm 2,000/200/100 split counts and zero overlap.
3. Verify selected review IDs, text, order, and tokenization against the locked local artifacts. Do not resample or split again.
4. Test the planned `Salesforce/blip2-opt-2.7b-coco` checkpoint with beam size 3 on a small permitted sample.
5. Test `allenai/led-base-16384` with the planned LoRA adapters and intended source length, including a forward/backward step. Log peak VRAM and runtime.
6. Verify checkpoint saving, interrupted-run recovery, and result synchronization to the chosen durable destination.
7. Mark A2 validated only after its spec acceptance criteria pass; then continue the relevant stage specs.

Hardware must support the intended methodology. If checks fail, document alternative resource arrangements before changing source length, model, LoRA design, or experimental controls. No specific GPU model, price, or budget is selected by this document.

## 5. Experiment order on rented compute

- Complete Stage D if it was not validated locally.
- Stage E: pilot caption generation/manual inspection, then full caption generation.
- Stage F: caption scores, validation-only K/threshold selection, and frozen unfiltered/filtered caption sets.
- Stage G: fixed teacher/prompt/settings, condition-aligned targets, evidence records, automatic validation, and manual inspection. API execution location is independent of GPU rental.
- Stage H: build three deterministic train/validation LED datasets.
- Stage I: sanity experiment, then the three LoRA-LED training runs with controlled optimization/decoding settings.
- Later stages: independent human references, four-condition evaluation, representative 30-business human evaluation, external generalization, and planned direct multimodal comparisons according to the master plan.

The external generalization pilot runs after a stable filtered checkpoint. External dataset choice must follow the existing approval and compatibility requirements. Representative human-evaluation sampling must be specified and checked against the 100-business test distribution without selecting businesses based on favorable predictions.

## 6. Checkpoint and artifact retention

Use stable business/photo IDs and resumable stage outputs. Record processed, failed, and pending items so interrupted jobs resume without silently mixing different configs or model versions.

Before stopping/releasing an instance, synchronize and verify:

- Generated captions, scores, selection manifests, and validation reports.
- Approved teacher targets and evidence records when generated there.
- Model datasets, configs, dependency/runtime records, logs, and progress/decision updates.
- Best model adapters/checkpoints and the tokenizer/config needed to reload them.
- Training resume state if a run is incomplete.
- Predictions, metrics, timing, and peak-memory measurements.

Do not treat instance storage as the only surviving copy. Measure and report preprocessing/captioning costs separately from LED training/inference and direct multimodal processing costs so resource comparisons remain interpretable.

## 7. Open execution choices

TBD: GPU instance/VRAM, rental budget/duration, runtime image, cloud dependency versions, storage capacity, transfer mechanism, durable destination, and job scheduling.

Research TBDs remain governed by their stage specs: global seed/config, exact token allocations, K/filtering search spaces and validation criteria, teacher/version/prompt/settings, manual target-inspection size, external dataset, and exact representative sampling procedure. Cloud migration does not resolve these choices automatically.

## Current Basic Quality protocol - D-026, 2026-10-07

Active local validation uses configs/basic_caption_quality.yaml and configs/blip2_caption_basic_quality_pilot.yaml (spec005b): raw text retained, separate cosmetic normalization, missing punctuation not a failure, fixed once-only retry on declared basic failures, full EOS/token-cap evidence, same COCO/beam3. Historical E0 config/backend/captions and the earlier E0 cloud ZIP retain their original punctuation-fail policy; that ZIP is not the current E1/production protocol. Rebuild any future handoff from the active version after final inputs/K/source allocation are approved; do not silently deploy the historical package as the new pipeline. No account, rental, upload, teacher or training is authorized by this local pilot alone.

## Stage F D-027 transfer boundary

Current active definitions/models/roles live in configs/stage_f_validation_protocol.yaml and outputs/stage_f/validation_calibration_v1. Only200validationbusinesses/2504photos, rolecal40=601/Kstudy160=1903, originalSHA andmodelpins. Local fullpool worker is resumable and bounded, no newtrain/test/teacher/train. If user selectsVast, prepare a validation-only active-version package; rent/upload only under explicit account/budget authorization. Do not deploy historicalE0ZIP or mixlocalpartialrecords witha changedcloudruntime/device-map identity. Preserve raw/normalized/token/EOS and source-role/modelconfig fingerprints, verifyproduction precision/A2 separately. Currenthost preference does notauthorizepaidinstance.
