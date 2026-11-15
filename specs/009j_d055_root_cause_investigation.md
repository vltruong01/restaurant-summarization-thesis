# D-055 — Offline training/generation forensics

Status2026-10-18: offline investigation complete; causal attribution provisional.
Authoritative outputs: outputs/stage_i/d055_root_cause_v2.

Authorization: researcher request in attached Pasted text, preserved with SHA256
and explicit scope. No training, model forward/generation or GPU use is authorized.
No held-out data, Teacher calls, accepted-data/reference/checkpoint changes.

The audit must use actual pinned LED/Trainer/Accelerate/PEFT implementation and
hash-bound completed run artifacts. Check full accepted6,600 targets, source/label
alignment, BOS/EOS/shift/masks, module and gradient scope, actual optimizer/LR
updates, selected/loaded adapters, precision/cache and effective decoding. Compare
exactly the same24 saved historical/A/B outputs with unchanged D053 diagnostics,
separating random20 from targeted4. Repetition/structure are not factual quality.
No source-only marker, output normalization or target repair is introduced.

Observed historical21/24 structure versus pilot0/24 must be interpreted alongside
historical10,000 versus pilot1,024 presentations, different data diversity/order,
initialization, schedule and hardware. No causal architecture or duration claim.

Report H1–H7 with evidence, counterevidence, unknowns and minimal discriminator.
Preserve D051 strict replay failure and all previous results. Proposed≤3-case GPU
diagnostic is a separate unapproved protocol. Do not alter numerical thresholds,
length semantics, decoder policy or scientific parameters to claim a pass.

Acceptance: exact accepted-data hashes/content, source-based implementation
evidence, reproducible72-output comparison, no scientific mutation, transparent
quality-assessment provenance, explicit residual uncertainty and one next action.
CPU-only diagnostic test success is software acceptance, not a confirmed root cause.
