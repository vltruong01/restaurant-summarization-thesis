# Thesis implementation reading guide

Start with the [executed notebook](../professor_review/THESIS_RESEARCH_OVERVIEW.ipynb). The following are original implementation files, not newly invented pseudocode. Their original paths and bytes are preserved. Historical execution receipts and original data/model assets are required for full model reproduction; the bundled notebook analyses need only small local exports.

| Thesis step | Main files |
|---|---|
| Dataset and TRAIN2000 / VALIDATION200 / TEST100 | [`build_yelp_cohort.py`](../scripts/build_yelp_cohort.py), [`english_eligibility.py`](../scripts/english_eligibility.py), [`cohort.yaml`](../configs/cohort.yaml) |
| Representative review selection and token budget | [`budget_aware_review_selection_v4_2.py`](../scripts/budget_aware_review_selection_v4_2.py), [`source_budget_audit_template.py`](../scripts/source_budget_audit_template.py) |
| BLIP-2 caption generation | [`generate_stage_f_production_d033.py`](../scripts/generate_stage_f_production_d033.py), [`generate_stage_f_validation_captions.py`](../scripts/generate_stage_f_validation_captions.py) |
| Caption scoring, relevance, redundancy, diversity and K≤5 | [`stage_f_score_validation.py`](../scripts/stage_f_score_validation.py), [`stage_f_math.py`](../scripts/stage_f_math.py), [`apply_stage_f_frozen_filters.py`](../scripts/apply_stage_f_frozen_filters.py), [`construct_stage_f_production_d033.py`](../scripts/construct_stage_f_production_d033.py), [`stage_f_validation_protocol.yaml`](../configs/stage_f_validation_protocol.yaml) |
| Teacher supervision and evidence IDs | [`stage_g_production_d042.py`](../scripts/stage_g_production_d042.py), [`stage_g_canonical_review_ids.py`](../scripts/stage_g_canonical_review_ids.py), [`stage_g_teacher_production_d042.json`](../configs/stage_g_teacher_production_d042.json) |
| LED data collation and final training contract | [`stage_hi_led_data.py`](../scripts/stage_hi_led_data.py), [exact native training-step excerpt](../professor_review/code/native_training_step.txt), [accepted model settings and hashes](../professor_review/data/model_and_method.json) |
| Frozen RO/U/F/Shuffled TEST100 inference | [`run_s1000_test_inference_prepared_v1.py`](../scripts/run_s1000_test_inference_prepared_v1.py), [`s1000_final_eval_core_v1.py`](../scripts/s1000_final_eval_core_v1.py) |
| ROUGE / BERTScore and reference alignment | [`stage_m_evaluation_v1.py`](../scripts/stage_m_evaluation_v1.py), [`score_s1000_automatic_prepared_v1.py`](../scripts/score_s1000_automatic_prepared_v1.py) |
| Yelp human-rating statistics | [`analyze_stage_n_final_statistics_v1.py`](../scripts/analyze_stage_n_final_statistics_v1.py), [`analyze_stage_n_two_raters_v1.py`](../scripts/analyze_stage_n_two_raters_v1.py) |
| TripAdvisor pipeline and statistics | [`stage_o_frozen_pipeline_v1.py`](../scripts/stage_o_frozen_pipeline_v1.py), [`run_stage_o_allocated_inference_v1.py`](../scripts/run_stage_o_allocated_inference_v1.py), [`analyze_stage_o_final_statistics_v1.py`](../scripts/analyze_stage_o_final_statistics_v1.py) |
| Qwen-Caption / Qwen-Image | [`stage_p_qwen_bf16_3090_v2.py`](../scripts/stage_p_qwen_bf16_3090_v2.py), [`run_stage_p_qwen_image_v1.py`](../scripts/run_stage_p_qwen_image_v1.py) |
| Resource benchmark and fair efficiency accounting | [`benchmark_stage_p_led_3090_v1.py`](../scripts/benchmark_stage_p_led_3090_v1.py), [`stage_p_efficiency_audit_core_v1.py`](../scripts/stage_p_efficiency_audit_core_v1.py) |
| Cross-stage statistical verification | [`verify_integrated_research_statistics_v1.py`](../scripts/verify_integrated_research_statistics_v1.py), [portable notebook verification](../professor_review/verify_package.py) |

## Which settings are authoritative?

The selected Filtered model is **Source-Separated V1-S1000**. The notebook's frozen `data/model_and_method.json`, evaluation locks and original completion evidence identify the actual models, initialization, source budgets and inference settings. The native trainer's complete archived `RUN_TRAIN.py` is not included in this focused public tree; its original hash and exact excerpt are in `data/code_provenance.json` and `code/native_training_step.txt`. The excerpt alone is not a complete runnable trainer. Full training reproduction requires access to the preserved private archive.

Original configuration/helper names may include “pilot”, “validation” or old preparation-status fields. They remain only when used by retained production functions or to bind inherited contracts; they are not alternative selected models or evidence that completed studies are still pending. In particular, do not mistake an older schema/configuration for the final S1000 training settings. The `stage_l_reference_app_v1.py` and `stage_n_human_evaluation_v1.py` modules remain as imports used by analysis/validation; this repository does not distribute a live annotation service or participant database.

## Reproduction scope

- **Bundled and supported:** read saved notebook/HTML outputs; verify package hashes, split/budget counts, rating mappings, aggregate means and existing statistical analysis; optionally replay CPU summary analyses using the notebook's lockfile.
- **Requires private archived inputs/runtime:** full captioning, target generation, training, metric-model scoring, source-grounded reinspection and frozen inference. Some original scripts retain archived execution gates and paths; those are provenance constraints, not a standalone setup recipe.
- **Not performed during this cleanup:** new experiments, score changes, inference, training, reference/rating modification or model downloads.

The final prose reports are selected extracts of the original integrated research package. Links to `SOURCES/`, `VERIFIED_TABLES/`, original `outputs/` paths or original manifests in those historical texts refer to private archived packages. For self-contained public numerical replay use `professor_review/`; no incomplete historical package is presented as directly runnable here.
