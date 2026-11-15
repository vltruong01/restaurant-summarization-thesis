# D-056-v2 — Final offline amendment before GPU authorization

Date: 2026-10-18. Status: offline preparation authorized; GPU execution NOT authorized.
Parent: `009k_d056_literature_informed_diagnostic_preparation.md`, retained unchanged.
Namespace: `outputs/stage_i/d056_diagnostic_preparation_v2`.

The researcher's independent review accepts the literature-informed rationale and
diagnostic design. Replace only the cache-probe pair p=11/12 with p=3/11, because
D-055 already observed wrong marker choice at p=3 and failed newline transition
at p=11. Each comparison still uses three explicit uncached/setup/incremental
forwards, with identical encoder tensor, masks, precision and effective adapter.
No additional observations were obtained to choose these probes. p=12 and the
review-to-visual boundary remain in the original teacher-forced measurements.

Keep the three cases, four states, complete sources/targets, weights, numerical
screens and ceilings 24 teacher-forced / 144 cache-control / 36 traces <=64 new
tokens. No dataset or decoding-policy change. The accepted interpretation tree
and literature rationale remain unchanged. D-051 strict replay FAIL is preserved.

Require: historical D-052/D-054 tensor-fingerprint convention compatibility;
separate exact-ID/value top-10 raw and processed rankings; scored selected beam
ancestry and sequence-score reconstruction; real EOS versus custom censor versus
finalization distinctions; offline source/mask/adapter binding checks. Validate
the actual collator's Python body with an explicitly disclosed NumPy shim and
independent arrays. This does not substitute for actual Torch/LED/PEFT execution.

Preserve all v1 files, old ZIP and old failures under a parent preservation hash
manifest. Bind new code, config, alignment, implementation, sources and adapters
in a new preflight manifest. Missing/false/stale/wrong-version approval must stop
the future runner before model imports. The v2 authorization template stays false.

Acceptance: original plus amendment offline/negative-control tests pass; all frozen
hashes match; no model loading, GPU calls, forward/backward/generation, training,
Teacher calls or held-out access. Report limits honestly: backbone tensor hashes
and GPU/cache compatibility can only be checked during separately authorized
execution. Stop after preparation and request explicit hash-bound approval.
