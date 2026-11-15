# Stage F D-033: RTX3090 portable continuation package

This package is READY FOR TRANSFER, not a completed GPU feasibility check. Local jobs are stopped; no instance was rented, no upload performed, no cloud model job run by packaging.

## What is preserved

-2300business cohort/splits:2000train/200validation/100test;30027readable eligible original photos.
-2709existing captions retained by originalphotoID/path/SHA/runtime provenance:205train,2504validation. They are not regenerated.
-27318pending photos:26182train/1136test, complete eligible pool before Kselection.
-Final K5; exact5000source/4000review/750caption template; frozen thresholds0.08/0.59/0.46; CLIPsoft/pHashdiagnostic/exactduplicatesFilteredonly.
-Original D026generator, COCOcommitcda95b9319b722f79c9451a3d8ff92eea02048dc, beam3/batch1/maxnew64/prompts/Basic Quality/retry protocol.
-Original validation scores, converted CLIPweights/SBERTweights, tokenizer and all3025protected historical files, with byte checksums.

The15.5GBBLIPcheckpoint is not inside the ZIP. Setup downloads only its exact pinned files, verified against the exported originalSHA256manifest. No rawfullYelp, Windowsvenv, local credentials or unrelated storage artifacts are included.

## Machine selection

Use a singleRTX3090 24GB, LinuxPython3.11/CUDA-compatible driver, recommended>=32GBsystemRAM and>=100GBdisk. Network/download speed and DirectSSH accessibility matter for data transfer. This is an engineering target; actual GPU feasibility remains to be checked.

## Upload from Windows

Use DirectSSHconnection details from the Vastinstance Connectpanel and your SSHkey. ReplacePORT/IPwith the actual values. This is an example, not an already configured connection:

```powershell
scp -P PORT "C:\thesis_proposal\yelp_dataset\outputs\cloud_handoff\stage_f_d033_v1\stage_f_d033_vast_bundle.zip" root@IP:/workspace/
ssh -p PORT root@IP
```

SFTP/rsync can also transfer the package. Keep the local archive/checkpoints. Verify the completeZIPagainst the SHA256in package_report.json; data should be copied back before releasing rented storage.

## Extract and verify on Linux

```bash
cd /workspace
python3 -m zipfile -e stage_f_d033_vast_bundle.zip /workspace
cd /workspace/yelp_dataset
python3 scripts/verify_stage_f_cloud_package_d033.py
```

The standalone verifier needs only Python's standard library. It checks all transferred files before dependency/model setup. Run commands from the extracted project root. Do not run the localRTX3050adapter on Linux or overwrite its preflight/device-map/checkpoint files.

The primary notebook is included with its cached history/context. This compact continuation package supports the StageFCLIentry points; earlier raw/StageCaudit RunAllcells may require large historical artifacts retained locally and are not used to rebuild inputs on cloud.

## Codex on the rented machine

If Node/npm is already installed:

```bash
npm install -g @openai/codex@latest
codex login --device-auth
cd /workspace/yelp_dataset
codex
```

Complete login in your own browser. The remote Codexsession starts fresh: paste the instructions in `cloud/stage_f_d033/NEXT_CODEX_PROMPT.md`. Existing localchat memory is not assumed. A tmuxsession keeps the remote terminal available after disconnecting.

## Setup ->100train-photo smoke -> explicit production command

```bash
bash cloud/stage_f_d033/setup.sh
bash cloud/stage_f_d033/run_smoke.sh
```

Setup creates a new isolatedLinuxvenv with the same pinned scientificlibraries, downloads/verifies the originalBLIPcheckpoint and runs preflight. Smoke generates only the first100deterministic pending TRAINphotos from the completeplan. Its checkpoints count toward production and are retained. Batch1/beam3/precision/prompts/Basic Quality stay unchanged. No test quality tuning or K/coverage study is run.

Review `outputs/stage_f/production_caption_construction_d033_cloud_v1/benchmark_report.json` for seconds/photo, estimatedremainingcaption time, CUDAdevice-map and VRAM. Actual3090speed/cost cannot be claimed until this check runs.

When ready to continue production:

```bash
bash cloud/stage_f_d033/run_production.sh
```

It requires a passing smoke and executes generation ->original pinnedscoring ->original fixedreview input preparation ->original Unfiltered/Filtered construction/source validation, sequentially. It creates no GPTteacher targets or LEDtraining job. Resume by rerunning the same command in the same runtime/profile; completedcompatible captions/scores are reused. Any unexpectederror/overlength/sourceoverflow stops without changing method.

## New cloud namespace and final outputs

`outputs/stage_f/production_caption_construction_d033_cloud_v1/` contains:

-`generation_remaining/caption_checkpoints/`: only newcloudgeneration, distinct runtime from local.
-`reused_caption_sources.json`: original2709caption locations/checksums, unchanged runtimeprovenance.
-`worker_progress.json`: running/error/complete phase and counts.
-`scores/`, `review_inputs/`, `manifests/`: resumable production intermediates/results.
-`completion_report.json` / `.md`: appear only after all2300business finalmanifests validate, including6900complete source token-IDchecks, unchanged priorvalidationK5results/2200acceptedreview references and protectedhistory.

No production completion is claimed by package validation. Shuffledassignment remains later. Stop beforeStageG/GPTteacher/LEDtraining. Record newruntime and progress in THESIS_PROGRESS.md, preserving all methodology and labels.

Sources: [Vastdata movement](https://docs.vast.ai/guides/instances/storage/data-movement), [VastSSH](https://docs.vast.ai/guides/instances/connect/ssh), [OpenAI CodexCLI](https://learn.chatgpt.com/docs/codex/cli), [headlesslogin](https://learn.chatgpt.com/docs/auth).
