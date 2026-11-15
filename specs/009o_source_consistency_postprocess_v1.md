# Post-D060: Source-consistency derived-output experiment

Status:executed CPU-only; mechanicalPASS; AI assessment NOT_RECOMMENDED_AS_RESCUE.
No researcher formal GO/NO-GO, official methodology change or GPU approval inferred.

Researcher requested technical research/application after guard remained inadequate.
One fixed preset processes24existing F outputs with sentence NLI, exact source
bindings, MiniLM/MMR and mutual entailment. Retained sentences are unchanged raw
spans. Visual evidence comparisons use single captions only; zero-caption visual
is empty. Serializer changes are explicit, never attributed to model learning.

Inference-only input whitelist excludes Teacher targets/verdicts/approved references.
No missing content is filled. No token truncation, hidden fallback, threshold
tuning, held-out access, training, LED generation or Teacher call. Preserve raw
and derived outputs separately; score predictions are fallible and not human labels.

12offline policy testsPASS plus91guardregressions. All24processed,66raw sentence
spans/68sourcebindingsPASS;17empty review sections,23withfewerthan3review sentences.
High-score donut relationship/count error remains. Reject quality-success claim;
softwarechecks and deterministic zero/header invariants do not establish usefulness.

Main code:scripts/stage_i_source_consistency_v1.py and
scripts/run_stage_i_source_consistency_v1.py. Config:
configs/stage_i_source_consistency_postprocess_v1.json.
Research rationale:thesis_sdd_docs/SOURCE_AWARE_OUTPUT_RESEARCH_20261019.md.
Outputs:outputs/stage_i/source_consistency_postprocess_v1.
Review:outputs/stage_i/source_consistency_report_v1_1.

No automatic additional variant, new loss/training or rollout across conditions.
OriginalD060/D059deadlines and D051strict replayFAIL unchanged. Further scientific
method changes need a concrete reviewed protocol, not silent adoption of this trial.
