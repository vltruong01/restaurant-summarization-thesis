# LED-large decoder policy V2: minimal early-sanity revision

**OFFLINE READY. Corrected labels accepted by the researcher. GPU execution remains unapproved.**

This supersedes only the V1 epoch-1 technical screen and adds state-continuity evidence. The accepted V1 label module, two F label views (2,200 rows), all source IDs/masks/padded shapes and original scientific artifacts remain byte-identical. The V1 archive and its five bindings are preserved. No additional model investigation or hyperparameter search was performed.

## Predeclared technical screen

| Observation after decoder-start EOS2 | Result |
|---|---|
| At least **5 consecutive BOS0** at any position | Hard FAIL: sustained BOS loop |
| Decoded text empty or whitespace only | Hard FAIL: blank output |
| Fewer than **16 ordinary tokens**, **and at least 90% special tokens** among tokens after decoder start | Hard FAIL: near-empty special-token-dominated output |
| Wrong decoder-start ID, missing start, or invalid token ID | Hard FAIL: protocol collapse/integrity error |
| BOS without a run of 5, including a single BOS followed by useful text | Warning; no failure by itself |
| Generation cap reached, including forced EOS at the cap | Warning; no failure by itself |
| No terminal EOS but no other collapse | Warning |

Special IDs are 0,1,2,3,50264, matching the unchanged tokenizer. Fractions exclude the initial decoder-start token and include terminal EOS when present. The near-empty rule is conjunctive: fewer than16 content tokens alone does not fail a readable short output. Raw BOS count, longest BOS run, content/special counts, fraction, terminal EOS, cap flag, warnings and hard-fail reasons are recorded per case. These transparent thresholds identify technical collapse; they do not certify semantic quality.

The same four predeclared cases and order are retained at step250/epoch1. Classification does not short-circuit their loop: all four raw token sequences, decoded texts, source/checkpoint hashes and diagnostics are saved before aggregate technical PASS/FAIL. Warnings permit continuation. A runtime/IO/deadline failure still stops immediately; missing predictions are never fabricated and no automatic retry is introduced. Neither technical PASS nor low loss is substantive quality GO.

## Inference continuity

The existing callback now passes the live Trainer into the versioned sanity function. A narrow guard saves RNG (Python/NumPy/CPU/CUDA), each individual module's train/eval flag, use_cache and exact adapter tensor values. In finally it restores RNG, individual modes and use_cache, then checks exact equality.

Read-only fingerprints before/after also check optimizer state/parameter groups and object identity, scheduler, GradScaler, AMP recovery counters, Trainer state/control, skipped-step flag, parameter identities/requires_grad, existing gradients and adapter activation/merge/disable flags. No optimizer, scheduler or scaler step/reset/restore is performed by sanity. Unexpected mutation fails continuity. STATE_CONTINUITY.json preserves both sides and checks, including when inference raises; the original exception remains primary. Runtime equality checks will be executed on the real epoch1 run; offline fixtures cannot establish actual GPU restoration in advance.

**35/35 focused tests PASS**, including the eleven unchanged accepted label/shift/source tests, single/four/five BOS boundaries, blanks, near-empty special dominance, cap warnings with/without EOS, four raw records on collapse, warning continuation, fake inference exception, exact RNG/mode/cache continuity, synthetic CUDA RNG round-trip, and mutation detection for optimizer/scheduler/scaler/counters/gradients/adapters. No real model forward/generation/backward/update occurred. Full pinned-runtime portable-root CPU preflight PASS: 9,961 original payload files, the18 accepted V1 entries and 13 V2 manifest entries are checked through the bound chain. Frozen train/validation/Teacher/tokenizer integrity PASS.

## Exact patch and bindings

EXACT_PROPOSED_PATCH.diff contains V1-to-V2 helper/runner/reporter/launcher/config/sanity/test changes. New execution namespace: runs/F_large_decoder_policy_v2. The accepted label module is reused unchanged; only technical criteria, continuity verification and versioned execution paths differ.

```json
{
  "config_sha256": "d079f41aa5ae5b1a680da5174eb3ef83235cb9b82e54c96b3d5d487215a168f4",
  "runner_sha256": "bea436309fadb632a04db82549ad63139f31bd77365e95271b77c1ff68470a61",
  "preflight_manifest_sha256": "782de3373434ff01b934818cd143e0d6c0be32fc0658e924c19ba2711401a3d3",
  "helper_sha256": "a37f7ee37047df01051d9e13cd45ab4ee0d189f3167f2b4167d384ae581a4777",
  "early_sanity_module_sha256": "10975bf25d559c248320e7070ea45387bce6cf2f8dde48dc02f7974e64b39306",
  "accepted_label_module_sha256": "413a408ed384bdd5d40ef21ea7c99891a2ee42e88554a1f6e543d1ad6f535b96"
}
```

## Minimal future execution authorization

Approve these exact V2 config/runner/preflight hashes for **one fresh Filtered run** on RTX3090, from the same pinned official LED-large backbone and fresh LoRA. Reject old adapters, interrupted Colab states and previous gate states as initialization. Preserve the existing fresh feasibility gate:3 successful updates,24 nominal microbatches (88 maximum with the already bounded AMP policy),2 technical memory generation traces, at least15% headroom and all original integrity/numerical checks. Gate states are discarded before fresh training.

Only after gate PASS: unchanged F2,000train/200validation,5epochs/1,250successful updates, same228 LoRA modules/456 tensors, seed42, order, AdamW settings, accumulation8, FP16/FP32 storage and dynamic65536 policy. Same five epoch checkpoints. At epoch1 run exactly4 technical sanity cases under the above criteria; then exactly24 outputs from each of epoch3/epoch5, same20random+4targeted. Beam4/max_length768 and all generation settings stay unchanged. Total maximum remains **54 calls =2gate+4sanity+48final**.

Approval must contain actual UTC start and a stop no later than48h afterward under the accepted absolute-stop/reserve fields. No historical deadline is reset by this preparation; no approval timestamp is fabricated. AUTHORIZATION_TEMPLATE_NOT_APPROVED.json remains false. The accepted scientific label policy does not itself authorize GPU execution.

After completion or expiry, stop for independent substantive review against D-059 Expanded-LoRA epoch3. Core information, opinion balance, factual/visual grounding, useful visual description, harmful semantic repetition, zero-caption compliance and usable completion still determine quality. Existing matched LED-base RO/U/F remain primary. No automatic RO/U, Teacher regeneration, held-out access, decoder/architecture/optimization change, retry or follow-on experiment is authorized. D051 strict replay FAIL and all stopped Unlikelihood/D058/D059/D060 evidence stay preserved.

Remaining uncertainty: the label correction's causal effectiveness and substantive output quality are still unproven. This revision addresses overstrict technical rejection, not those scientific questions. No other preparation is needed before the researcher decides on these execution hashes.
