# Stage H/I transfer bundle

Prepared locally on 2026-10-15; handoff revision v2 adds an explicit Codex execution guide. This is the portable train/validation dataset and offline-prepared LED training implementation. The common 768-token decoder limit and draft training settings remain proposals. GPU feasibility and full training are not authorized by creating or transferring this archive; all config approval flags remain false.

**For Codex on Vast.ai:** Read `cloud/stage_hi/CODEX_HANDOFF.md` first. It defines the reading order, exact setup/preflight commands, logging, later approval-scoped GPU commands, failure handling, checkpoint recovery and required return artifacts. Follow the latest researcher authorization; the default handoff stops after offline preflight.

## Archive layout

The revised ZIP contains one directory, `stage_hi_v2/`. Paths below are relative to that directory. It includes the six packed RO/U/F train/validation datasets, pinned tokenizer and backbone configuration, training/preflight code, requirements, reports, specifications, decisions and the current progress log. Original Stage G evidence stays on the local machine; portable data integrity is bound by the Stage H manifest. The previous v1 archive is preserved separately.

There are no Teacher API credentials, Codex settings, original photos, pretrained model weights or held-out test inputs/targets in this package. Dataset inventory counts in the documentation are retained.

## After transfer is approved

Upload `stage_hi_vast_bundle.zip` and its external `.sha256` file to the same folder. Verify before extracting:

```bash
sha256sum -c stage_hi_vast_bundle.zip.sha256
unzip stage_hi_vast_bundle.zip
cd stage_hi_v2
sha256sum -c cloud/stage_hi/transfer.sha256
```

Read `cloud/stage_hi/README.md` for the pinned environment setup, GPU requirements, offline preflight and later approval-gated feasibility procedure. The first computational step after environment setup is:

```bash
python scripts/run_stage_i_lora_led.py --mode offline-preflight --report-directory outputs/stage_i/cloud_preflight_v1
```

This preflight does not load pretrained LED weights or execute model forward/backward. Preserve its report alongside the transferred local preparation reports. Do not change Stage F/G inputs, targets, split membership, budgets, K or proposed settings to bypass a failed check.

Actual GPU forward/backward, memory, checkpoint recovery and generation must be tested after researcher approval. Review those results before separately approving full RO/U/F training. Sync checkpoints and logs to durable storage before releasing a rented instance.
