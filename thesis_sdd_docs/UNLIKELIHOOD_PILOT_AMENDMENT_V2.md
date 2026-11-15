# Sentence-only Unlikelihood pilot v2 — final offline review

One minimal amendment requested by independent review,2026-10-19.
V1 remains intact. No GPU operation or model execution is approved/performed.

**Changed:** all512fixed train examples now use sentence-copy negatives, eliminating
the256word-stutter negatives. Prior256sentence views remain identical. All original
positive labels/sources,512IDs/order and scientific settings are preserved.

**Gate additions:** measure actual UL gradient signal against CE on all48LoRA tensors,
not merely nonzero scalar loss. Probe the same forward graphs, AMP scale/unscale,
two extra autograd traversals per microbatch, no optimizer/replay/generation updates.
Save per-tensor norms, direction/cosine, zero/missing counts, weighted UL/CE loss
and gradient ratios. Report weak/zero signal; keep weight0.1 unchanged.
Ratios are pre-clipping objective diagnostics, not the fraction of an AdamW update.
Measure actual memory;15%headroom and stop-on-failure rules remain unchanged.

**Quality benchmark:** paired CE-versus-UL remains the controlled comparison.
Saved D059 Expanded-LoRA epoch3 is a separate practical quality benchmark, with
same24sources verified but architecture/training/initialization confounds disclosed.
No new benchmark generation. compare_stage_i_unlikelihood_v2.py assembles72records
(48future outputs +24saved benchmark outputs) only when both arms exist.

**Scope unchanged:** gate3updates/24microbatches per arm; then two fresh128update/
twoepoch runs,512train/200validation, AdamW2e-4, warmup7, original historical F
adapter, no inherited optimizer, final epoch2evaluation only,48beam4 outputs,
max_length768 including decoder-start. No model/parameter/decoder search.

**Success:** meaningful reduction in harmful repetition, maintained/improved Core
coverage and opinion synthesis, grounded useful visuals, zero-caption behavior,
factual correctness and complete summaries. No compensating severe regressions.
Researcher decides; automatic metrics and AI assessments are separate. Existing24
are development cases, not independent validation. If mixed/inconclusive/poor,
stop; do not initiate another model experiment.

**Authorization still required:** new config/preflight/runner hashes, an explicit
exception to prior final-attempt restrictions, UTC approval timestamp and end time
at most48hours later. Historical D060deadline2026-10-22T07:11:22.846767+00:00 and
D059deadline2026-10-25T15:10:48.941817+00:00 are unchanged; D051strict replayFAIL
preserved. This offline revision does not start/extend either clock.

Offline commands from workspace root in the pinned environment:

    python -m unittest discover -s tests -p "test_stage_i_unlikelihood*.py" -v
    python scripts/run_stage_i_unlikelihood_pilot_v2.py preflight

The review ZIP contains the tokenizer and frozen train/source/evidence copies for
offline inspection and tests, not complete weights/full datasets for GPU portability.
No accepted reviews,captions,targets,splits,references or checkpoints are modified.
