# Stage F: local stop -> Vast.ai Codex handoff

Prepared 2026-10-10; source-state snapshot stopped2026-10-09. This is execution guidance, not a new scientific decision, paid rental authorization, a finished transfer package, or permission to restart the stopped local job.

**Implementation update:** The cloud adapter and required-file packager are now implemented: `scripts/stage_f_cloud_d033.py`, `configs/stage_f_cloud_d033.yaml`, `cloud/stage_f_d033/README.md` and `NEXT_CODEX_PROMPT.md`. They already define the separate27318-photo cloudnamespace and reuse2709originalrecords. Use those entrypoints; do not recreate the adapter or run localresume. Current package build/report: `outputs/cloud_handoff/stage_f_d033_v1/`. Transfer integrity is validated by package_report.json; actual Linux3090runtime/smoke remains pending. Localjob remains stopped. This is execution preparation only, with unchanged methodology.

## Current authoritative state

- D-033 approved: maximum K=5, selected by the validation-only smallest-feasible95%-of-best rule. K5relative coverage97.7479%; original K3/5/8/10study remains unchanged.
- Source5000/review4000/entire caption allowance750; exact D-032source template unchanged. Whole reviews/captions only; stop on overflow without silent changes.
- BLIP2Salesforce/blip2-opt-2.7b-coco, revisioncda95b9319b722f79c9451a3d8ff92eea02048dc; beam3, batch1, maxnew64, no sampling, same initial/retry prompt and D-026Basic Quality. Keep all raw/normalized/EOS/retry/model provenance.
- Filtered: exact normalized-pixel collapse, pHashdiagnostic-only, CLIPsoftquality ranking only, relevance0.08/redundancy0.59/coarser grouping0.46, unchanged farthest-first.
- Unfiltered: readable image + D026Basic only, existing seed42ordering, up-to5; no substantive filters or image deduplication.
- Immutable cohort:2000train/200validation/100test. Test input processing allowed with frozen settings; no test tuning, human test references or testteacher pseudo-targets.
- User explicitly stopped local generator/supervisor.193newtrain captions retained, all Basicpass, plus12compatible priortrain and2504validation captions:2709/30027photos. Remaining27318photos need generation. Final production manifests are not complete.
- No GPTteacher, LEDtraining, newannotation, thresholds/K/grid changes or shuffled assignment in this handoff.

Read before doing work:

1. `thesis_sdd_docs/THESIS_PROGRESS.md`
2. `DECISIONS.md`, D026/D030/D031/D032/D033
3. `specs/006b_stage_f_validation_execution.md`
4. `configs/stage_f_validation_protocol.yaml`
5. `outputs/stage_f/methodology_amendment_d033/frozen_production_method.json`
6. `outputs/stage_f/production_caption_construction_d033_v1/user_stop_report.json`
7. `outputs/stage_f/production_caption_construction_d033_v1/README.md`

## Transfer plan

Use `/workspace/yelp_dataset` as the remote project root, preserving relative paths. Transfer code/config/spec/docs/notebook, the filtered2300cohort metadata and original associated photos, the complete English review pools required by fixedreview preprocessing, pinned LEDtokenizer assets, accepted review-reference evidence, validation captions/scores/frozen thresholds, D030-D033provenance and the stopped193production caption checkpoints.

Do not simply drag the entire workspace: localWindows`.venv*`cannot serve as a Linuxenvironment, and caches/unrelated outputs may be large. Produce a required-file manifest/SHA256list before packaging. Include every file referenced by compatibility/protection manifests, or explicitly document missing optional historical-only evidence; do not silently weaken a production integrity check to bypass missing inputs.

Weights may be copied in a verified portable layout or downloaded remotely at the exact pinned revisions. Preserve the converted CLIPsafetensors identity and SBERTweight identity from existing model-file reports. Do not replace a pinned model/library with latest or use a historicalE0pilot ZIP as this D033production package.

Use directSSH/SFTP/SCP or rsync to copy the prepared package. Keep the local copy/checkpoints intact. Verify all package checksums after extraction. This file is a handoff note; no upload archive or cloud-ready adapter is claimed created yet.

## Critical cloud adapter requirement

Current `generate_stage_f_production_d033.py` is the LOCALadapter: it expects a27511-photo generation plan and a Windows/localRTX3050preflight profile. Original D026`generate_locked` binds checkpoints to runtime/model/plan/device-map identities. A3090Linuxruntime cannot simply resume that same directory, reuse the Windowsvenv, or overwrite its environment/model-load reports.

Create a new provenance-bound cloud output namespace/config/remaining-photo plan while preserving all scientific generator settings. Keep the local/checkpoint namespaces read-only. Extend the compatiblecaption source registry to reuse193localtrain +12priortrain +2504validation records by photoID and storedSHA; generate only27318missing photoIDs. Preserve every record's original runtime provenance; no relabeling or regeneration of completed caption records is required merely because hardware changes.

Retain the original D026generator. Update only operational adapters, dynamic expected-count checks, namespace/registry wiring, Linuxenvironment/preflight records and integrity receipts with an explicit migration delta. Never bypass an assertion, change a protection manifest or substitute a new model/prompt/budget just to make the cloud run pass. Preserve previous config/adapter versions before changes.

On one3090, verify actualCUDAload/VRAM/device map with frozen settings. Benchmark a deterministic pending TRAIN-only sample, e.g.100photos; this is hardware feasibility/timing, not caption quality/K/threshold tuning. Record new runtime/library versions and check compatibility with the pinned local stack. Do not start LEDtraining or claim a completed globalA2training gate from BLIP2success.

After cloud feasibility passes and the researcher instructs production continuation, run the remaining frozen pool, pinned scoring, sharedreview input preparation and final Unfiltered/Filtered construction. Preserve accepted2200train/validation review references; apply the same frozen fullv3/v4/v4.1/v4.2candidate-discovery path to100test inputs without tuning or human-reference use.

Validate all2300businesses, splitwise counts/QC/removals/zeros, whole-caption/source token budgets, identical reviewIDs/text/order across instantiated conditions, unchanged historical validation evidence, and checksums. Any unexpected runtime/scoring failure or overflow stops for a report without a scientific method change. Stop after final StageFmanifest validation, before StageGteacher or LEDtraining. Sync and verify cloudoutputs back to a durable/local destination.

## Prompt to paste into the new remote Codex session

Read thesis_sdd_docs/STAGE_F_VAST_CODEX_HANDOFF.md and its authoritative files. This is a new LinuxRTX3090execution environment for the approved D033StageFmethod. Inspect the transferred file/checksum manifest and report missing inputs. Prepare a new cloudruntime/remaining-photo namespace, preserving local records and all frozen science. Reuse2709completed captions by verifiedphotoID/provenance, not a27511-photo local resume profile. Generate only27318pending images, without selecting fivephotos before generation. Keep originalD026generator/checkpoint/beam3/batch1/prompts/QC, D030CLIPsoft-only, D031exact-only/pHashdiagnostic, thresholds0.08/0.59/0.46and D0325000/4000/750template. First run a deterministic TRAIN-only hardware smoke/benchmark and return runtime/timing/VRAM results. Do not run fullproduction until instructed after that check. Record execution changes in THESIS_PROGRESS.md without changing scientific decisions or labels. Final objective after continuation approval: validated2300business Unfiltered/Filtered manifests and completionreport; stop before GPTteacher/LEDtraining. No test tuning, no human testreferences or testteacher targets.

## Installing/signing in to Codex on the remote server

OpenAI official CLI supports Linux, inspecting/editing files and running shell commands from a project directory. If Node/npm is available:

```bash
npm install -g @openai/codex@latest
codex --version
codex login --device-auth
cd /workspace/yelp_dataset
codex
```

Complete the device login in your own browser/account; device-code login may need enabling in ChatGPTsecurity/workspace permissions. New CLI sessions do not automatically inherit this local conversation: use the handoff prompt/files. A terminal multiplexer such as tmux keeps the remote terminal session available after disconnecting. Codex manages scripts; BLIP2uses the rentedGPU.

Sources: [OpenAI CodexCLI](https://learn.chatgpt.com/docs/codex/cli), [headless authentication](https://learn.chatgpt.com/docs/auth), [VastSSH](https://docs.vast.ai/guides/instances/connect/ssh), [Vastdata movement](https://docs.vast.ai/guides/instances/storage/data-movement).
