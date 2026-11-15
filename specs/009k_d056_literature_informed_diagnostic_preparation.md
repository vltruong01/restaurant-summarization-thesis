# D-056 — Literature-informed minimal diagnostic preparation

Status2026-10-18: offline preparation authorized; GPU execution not authorized.
Depends on completed D055; preserves D051 strict replay FAIL and all prior evidence.
Namespace: outputs/stage_i/d056_diagnostic_preparation_v1.

Scope: inspect original four named papers; distinguish published methods/results
from hypotheses about this LED. No copied hyperparameters, target regeneration,
automatic longer training or architecture replacement.

Keep three fixed validation cases/four read-only states and ceilings24 TF /
144 cache forwards /36 traces,64 new tokens per trace. Baseline generation,
accepted source/target/adapter/tokenizer/reference bindings and numerical screens
remain fixed. No test data, Teacher calls, model forward/backward/generation or
training during preparation. No model instantiation required for offline checks.

Disclosed protocol refinement: matched cache control uses p11/12 with explicit
p-token uncached, p−1 cache setup and last-token incremental calls. This replaces
unexecuted D055 three-probe/full-TF-reference construction to avoid a physical
prefix-length comparison confound within the same144-forward ceiling. Position3
remains observed under the planned TF/trace measurements. No old artifact changes.

Exact position table, visual-boundary observations from existing TF budget,
precision screens/raw arrays, PEFT/LED forwarding, beam ancestry/score semantics,
EOS-versus-censoring and censored beam non-equivalence must be documented and
tested offline where possible. Model/runtime compatibility is not claimed without
the later authorized test. Static/hash/CPU tests are separated from factual quality.

Acceptance: original hashes/shapes verified; package pins correct; regression and
negative controls pass; future runner denies missing/stale approval before heavy
imports; config/protocol and interpretation tree frozen for review. Update living
decision/progress logs, preserve historical snapshots. Stop and request separate
explicit authorization; no downstream scientific execution is implied.
