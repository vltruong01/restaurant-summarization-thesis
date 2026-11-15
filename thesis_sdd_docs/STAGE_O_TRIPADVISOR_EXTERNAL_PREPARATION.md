> Current app status (5 November 2026): native inference is complete and the Human Evaluation app now passes backend, browser and portable extraction tests. Review-only package: `outputs/stage_o/human_evaluation_ready_20261105_v1/`. No collection/release/Qwen. See STAGE_O_HUMAN_EVALUATION_READY.md for actual validation scopes. Earlier pending app-status notes below are historical.

# Stage O — TripAdvisor external evaluation preparation

## Allocated RTX3080 execution update — 4 November 2026

The researcher rented an instance and explicitly requested connection/continuation at `root@1.208.108.242:55308`. Verified identity: instance 59069440, container 77eb3c4ad31e, GPU UUID GPU-953a5550-f5a3-b664-9017-0acd8461f37c, RTX3080 10GB, approximately 30GiB cgroup RAM. A separate Python 3.11.7 environment restores the original pinned versions. Public BLIP/CLIP/SBERT assets were restored at the frozen revisions, with publisher LFS hashes; converted CLIP tensors and the original converted-file hash match. LED backbone and all RO/U/F adapter hashes match. No training, Teacher, Qwen or human-rating release is authorized/performed.

Native caption generation started and its first actual photo checkpoints have been verified through read-only monitoring. This is an **active run, not completed inference**. It retains all 5,143 source images, frozen Basic rules (one native retry), FP16 GPU/CPU offload, original thresholds/K5 and decoder settings. A previously successful 4GiB GPU-weight offload budget keeps activation headroom on this 10GB GPU; cgroup-aware CPU allocation is used. The historical pre-download 40GiB disk check initially rejected launch before any image forward; its receipt is preserved. After all assets were restored/verified, the check reserves 8GiB additional disk instead. No scientific settings or frozen source files were edited.

Remote supervisor program `stage_o_inference` has `autorestart=false`. It proceeds sequentially from captions to frozen scoring/construction and three LED arms (75 native components /45 assembled outputs), retaining raw tokens and failures. A separate hidden local watcher polls only this instance, downloads terminal evidence, checks archive/every-member SHA-256 and validates completed outputs before producing the updated review ZIP and preparing real-output blinded assignments. It does not restart failed model work, change instance lifecycle, collect ratings or start Qwen. Local status/logs and the eventually verified deliverable live under `outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/`. The prior preparation ZIP remains historical. Completion must be checked against native receipts **and verified local download**, never inferred from this update or SSH state.

The researcher requested that cost not be reported as part of methodology. Runtime/model/source provenance remains recorded. The earlier resource-blocked section below describes the pre-allocation state and is preserved as history.

Status on 4 November 2026 (Asia/Saigon): **cohort and CPU review preprocessing complete; caption/scorer dependencies blocked; stop before model execution and human collection.**

Authoritative workspace: `outputs/stage_o/tripadvisor_external_v1/`.
Review package: `STAGE_O_TRIPADVISOR_EVALUATION_REVIEW.zip` in that workspace; its SHA-256 and actual archive-validation receipt are adjacent.

## Completed and frozen

- Inspected both original Zenodo records; selected v2 (14622324), smallest Gijon-area archive. Publisher MD5 and actual ZIP CRC verified. V1 was inspected at metadata/archive-listing level, not claimed fully audited.
- Of 1,832 publisher-en reviews, 1,828 passed the existing seeded langdetect 1.0.9 rule. Forty-nine restaurants were structurally eligible. Deterministic SHA-256 candidate ordering was saved before photo eligibility checks or any outputs. Seventeen candidates were checked; two had all photo URLs unavailable; the first 15 qualifying restaurants were frozen.
- Sources: 320 qualifying English reviews and 5,143 accessible individual photographs. Images preserve the publisher-linked CDN response bytes, with IDs, URL/review/item associations, SHA-256 and dimensions. Camera-original/full-resolution status is not attested. Some renderings are low resolution (106–550 px wide, 82–413 px high).
- Existing v4.2 review selection and pinned tokenizer were reused on CPU: 247 whole selected reviews across 15 businesses; all remain within 30 reviews, 4,000 Review tokens and 5,000 task-source tokens. Schema conversion preserves original values; day dates get an explicitly documented formatting-only midnight suffix; ratings divide exactly by ten.
- No normalized name+city match with frozen Yelp; no raw-image byte or native normalized-pixel match found; no exact normalized-pixel identity across selected restaurants. This does not certify absence of renamed venues, chains, near-duplicates or public-pretraining exposure.
- All three accepted RO/U/F-S1000 adapters and the original LED backbone were actually rehashed and match the final lock.
- Frozen common review/photo manifest prepared for future Qwen work. Captions are explicitly pending; Qwen configuration and inference are unapproved.
- Stage N code copied and adapted for three conditions, two evaluators, 45 outputs each / 90 ratings, with a separate namespace and port. Code is syntax-checked; missing-output preparation is refused. No live workspace, accounts, simulated ratings or live release were created. Full Stage O app runtime tests await real outputs.

## Actual blocker and execution boundary

BLIP-2's two pinned weight shards are absent (index total 15,495,870,464 bytes). Pinned CLIP and SBERT weights are also absent. Available RAM at the recorded audit was about 4.92 GiB, below the existing offload profile's 8 GiB minimum. No model download, GPU model load, caption, scoring, inference or rental was initiated.

The RTX3050 is not diagnosed as faulty: existing Stage K receipts show frozen LED beam4 inference ran on it at approximately 2.9–3.2 GB peak allocation. A local route requires restoring the pinned assets and freeing RAM; a separately approved 24GB GPU could accelerate captioning. `EXECUTION_PLAN_AND_COST.md` gives both routes and a quoted-price-based estimate, not an authorization or reserved instance.

## Still pending

- BLIP-2 captions for all 5,143 photos before K selection; Basic QC and native single retry only.
- Frozen relevance/redundancy/grouping/CLIP-soft-ranking/diversity/K5 construction, then 750 incremental-caption / 5,000 total-source token verification.
- 75 native LED component calls and 45 RO/U/F assembled output records. No generation failures have yet occurred because no model calls were started.
- Independent package review, then real reference-free human evaluation: 0/90 ratings currently.
- Qwen approval, model configuration and benchmarking: not started.

The 5,143-photo common evidence pool includes one 2,040-photo venue, so real annotation workload is material. No silent photo cap or post-output cohort replacement is introduced. Changes to evidence scope require explicit review.

This small city-specific external study is exploratory. It is not a replacement for Yelp TEST100/TEST30 and cannot conclusively rule out GPT Teacher mimicry. Existing Yelp datasets, models, references, predictions and rating results remain unchanged.


### 4 November 2026: conditional instance destruction authorized

The researcher explicitly requested destruction after the current job finishes. A separate local watcher is now waiting for the verified successful 75-component/45-output closeout and complete local package. It will reverify every backup member, remaining non-cache artifacts, supplementary runtime logs, terminal/idle state, container/GPU identity and provider instance 59069440 on machine 139250 before one authenticated DELETE. Provider request/result receipts are required; no DELETE has yet been sent. The active scientific runner and frozen authorization/configuration remain unchanged. Failure, missing backup, unexpected artifacts or identity/API blockers leave the instance intact for review. The local computer must remain on for this deferred watcher. Details: outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/INSTANCE_DESTROY_README.md.


## Completed native inference and verified instance closeout ? 5 November 2026

All 5,143 native BLIP records and 75 LED components / 45 RO/U/F outputs are complete for the frozen 15 businesses. Download archive and all 5,348 listed members passed SHA-256 checks; source/model/input/output/token integrity checks passed. No cap or format flags were recorded. The complete review ZIP contains 15716 files, SHA-256 `268e6fc033b94d81a35da612f727840fd5004b652964b26188c3ee73bd37c9ce`. Ninety real-output blinded assignments are prepared, with zero ratings and no live release. Full app HTTP tests remain pending release/account setup.

Instance 59069440 / machine 139250 was destroyed through one authenticated in-container provider DELETE, HTTP 200 success=true, at 04:51:31 Asia/Saigon on 5 November. Postquery returned 401, so independent postquery absence was not established; the direct provider DELETE response is the receipt. Scientific artifacts are unchanged. Local-origin authentication and other closeout blockers were resolved and preserved in lifecycle history. See outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/FINAL_CLOSEOUT_RECEIPT.json and FINAL_CLOSEOUT_REPORT.md. STOP before human collection/Qwen; no training or quality ranking was performed.
