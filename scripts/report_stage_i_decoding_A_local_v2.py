"""Write evidence-linked analysis of completed A results; never edits model outputs."""
from __future__ import annotations

import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from stage_i_decoding_diagnostics import sections, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1"
RUN = ROOT / "outputs/stage_i/decoding_greedy_A_local_v2"
PREP = ROOT / "outputs/stage_i/decoding_diagnostic_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    paired = read(OUT / "paired_summary.json")
    integrity = read(OUT / "execution_integrity_and_runtime.json")
    comparisons = read(OUT / "comparison_cases.json")
    original = [json.loads(line) for line in (PREP / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    bykey = {(c["business_id"], c["condition"]): c for c in comparisons}
    ref = {(c["business_id"], c["condition"]): c for c in original}
    notes = []

    def add(bid, condition, classification, quote, explanation, caption_ids=(), review_markers=(), baseline_quote=None):
        c = bykey[(bid, condition)]
        if quote:
            assert quote in c["diagnostic"], (bid, quote)
        if baseline_quote:
            assert baseline_quote in c["baseline"], (bid, baseline_quote)
        rv = []
        for marker in review_markers:
            match = re.search(r"\[" + re.escape(marker) + r"\] (.*?)(?=\n\n\[REV_\d+\]|\Z)", c["source"], flags=re.S)
            assert match, marker
            text = match.group(1).split("\n", 1)[1]
            rv.append({"review_marker": marker,
                       "review_id": ref[(bid, condition)]["selected_review_ids"][int(marker.split("_")[1]) - 1],
                       "review_text": text})
        caps = [x for x in c["captions"] if not caption_ids or x["caption_id"] in caption_ids]
        notes.append({"business_id": bid, "condition": condition, "subset": c["subset"],
                      "classification": classification, "greedy_exact_excerpt": quote,
                      "baseline_exact_excerpt": baseline_quote, "explanation": explanation,
                      "caption_photo_evidence": caps, "review_evidence": rv,
                      "photo_verification": "NOT_INDEPENDENTLY_PERFORMED", "human_verified": False,
                      "provenance": "Codex AI qualitative assessment of unchanged sources and generated outputs"})

    add("j3kGicU3mPG_ciIKG1Vuyg", "filtered", "ZERO_CAPTION_VISUAL_GROUNDING_FAILURE_PERSISTS",
        "The bagel is on a plate.",
        "No Filtered captions or selected photos exist and the teacher visual section is empty. Both baseline and A nevertheless produce visual assertions. A removes neither the evidence violation nor repetition; plausibility from reviews is not visual support.",
        baseline_quote="There are bagels.")
    add("OpnIXsrqaK-e8Wz5gco7Vw", "unfiltered", "EOS_AND_SECTION_RECOVERY_WITH_SEMANTIC_FAILURE",
        "A large, round, round table is surrounded by tables.",
        "Baseline loops to768 before the visual section. A ends at344 and restores the visual section, but changes the caption's large swimming pool surrounded by chairs/umbrellas into a large round table surrounded by tables/chairs. Cap/format improve while evidence grounding remains poor.",
        caption_ids=("IMG_CAP_1",), baseline_quote="grouper sandwich, grouper sandwich")
    add("OpnIXsrqaK-e8Wz5gco7Vw", "filtered", "EOS_AND_SECTION_RECOVERY_WITH_UNSUPPORTED_VISUAL_RELATIONS",
        "A large plate of food is topped with hot drinks.",
        "Baseline reaches768 and omits visual evidence. A ends at246 with a visual section, but none of the five captions support hot drinks topping a plate or the later straw/wine claims. This is not a successful factual repair.",
        baseline_quote="grouper sandwich, grouper sandwich")
    add("7u40cyMxw4iGosMdkesoEQ", "filtered", "PERSISTENT_WRONG_HEADER_AND_NEW_MISSING_VISUAL_SECTION",
        "[VISUAL_CONTEXT]",
        "Baseline has the wrong review header but includes VISUAL_EVIDENCE. A retains the wrong header, emits a577-token repetitive review, ends before the cap and never emits VISUAL_EVIDENCE. A few scene-like sentences occur in the unsectioned tail; a missing parsed visual section is not proof of no visual assertions.",
        baseline_quote="[VISUAL_CONTEXT]")
    add("gcQSAQIx-KQOdf5E8HF6Lw", "filtered", "WRONG_REVIEW_HEADER_PERSISTS",
        "[VISUAL_CONSENSUS]",
        "The incorrect review header persists in A; visual repetition also persists. Do not count this as repaired merely because EOS occurs before the cap.",
        baseline_quote="[VISUAL_CONSENSUS]")
    add("gpSv4OvISIeyBAqcNN5bqw", "reviews_only", "NEW_REVIEW_FOOD_CATEGORY_ERROR",
        "Reviewers frequently praise the pizza, including the buffalo chicken pasta, the maple bourbon pork pizza, and the maple bourbon pork pizza.",
        "REV_10 explicitly distinguishes pizza from buffalo chicken pasta; REV_3/29 praise maple bourbon pork pizza. A retains food names but incorrectly groups pasta under pizza and repeats the pizza item. Baseline's generic pizza statement lacks this specific category error. More named details do not automatically mean better grounding.",
        review_markers=("REV_3", "REV_10", "REV_29"),
        baseline_quote="some reviewers praise the pizza")
    add("HhD8DLES5ZeULKDXmfkOOg", "reviews_only", "REVIEW_SPECIFICITY_AND_CONTRAST_NOT_RECOVERED",
        "Food opinions are mixed:",
        "Both baseline and A remain generic. The source's sports-bar/beer-selection details (REV_23 and self-serve beer taps REV_10) and specific food/value contrasts in the teacher are not restored. A also puts 'soggy' into an atmosphere/seating list although REV_26 describes soggy breading. This is a qualitative omission/role-binding concern, not a keyword-derived factual score.",
        review_markers=("REV_10", "REV_23", "REV_26"))
    add("HhD8DLES5ZeULKDXmfkOOg", "unfiltered", "CORRECT_SIMPLE_CONTENT_WITH_REPETITION_AND_INFORMATION_LOSS",
        "A plate of food is on a wooden table.",
        "This ordinary description is compatible with captions1/4 and is not a factual error merely for being simple. A still repeats it, omitting the supplied chicken sandwich/waffle fries, dessert, shrimp/grits and sharing details. Baseline has the same narrow repetitive focus.",
        baseline_quote="A plate of food is on a wooden table.")
    add("HhD8DLES5ZeULKDXmfkOOg", "filtered", "MIXED_SUPPORTED_SCENE_WITH_NEW_UNSUPPORTED_RELATION",
        "A wooden table is on a table.",
        "A introduces people-at-bar content supported by caption2 but loses baseline's explicit shrimp/grits identity from caption5 and adds an unsupported nested-table relation. Broader scene coverage trades off against specificity/coherence; this is not unqualified improvement.")
    add("noUma64stIOSYAQ1GIIFjw", "unfiltered", "SMALL_CAPTION_SUPPORTED_IMPROVEMENT_WITH_PERSISTENT_REPETITION",
        "A fork is sticking into it.",
        "A adds a concrete fork-in-food detail supported by caption5, absent from the baseline's repetitive plate/table focus. This is a real local improvement, but A still repeats the same plate/fork statements and does not cover most distinct evidence.",
        caption_ids=("IMG_CAP_5",), baseline_quote="A plate of food is on a table.")
    add("-Wd5KqFszREg0XdXfSv5tg", "unfiltered", "UNSUPPORTED_ACTION_PERSISTS_WITH_SUPPORTED_SIMPLE_CLAIMS",
        "A woman is holding a cup of coffee.",
        "Caption1 says the woman sits at a table with coffee, not that she holds it. Caption2 supports the man pouring coffee, and the mask description is supported. Separate unsupported action from the valid ordinary descriptions; do not claim the original photo disproves holding because no photo was independently inspected.",
        caption_ids=("IMG_CAP_1", "IMG_CAP_2"), baseline_quote="A woman is holding a cup of coffee.")
    add("-Wd5KqFszREg0XdXfSv5tg", "filtered", "UNSUPPORTED_SUBJECT_ACTION_BINDING",
        "A woman is pouring coffee into a cup.",
        "The pouring caption2 identifies a man; the separate woman caption3 describes sitting with coffee. A transfers the pouring action to the woman without caption support. The man-pouring and surgical-mask statements remain supported.",
        caption_ids=("IMG_CAP_2", "IMG_CAP_3"))
    add("JcdOQvvx3BqZ5Ga2_9n_Bg", "filtered", "ONE_ILL_FORMED_RELATION_REMOVED_BUT_UNSUPPORTED_SCENE_PERSISTS",
        "The restaurant has a bar.",
        "A removes the baseline's self-referential liquid-in-red-liquid sentence but preserves unsupported bar/red-liquid scene binding and repetition. The captions describe food, diners, wine-pouring and a bowl, not a bar. Do not confuse removal of one nonsense sentence with restored grounding or caption coverage.",
        baseline_quote="The liquid is in a red liquid.")
    add("JcdOQvvx3BqZ5Ga2_9n_Bg", "unfiltered", "UNSUPPORTED_COLOR_AND_SPATIAL_RELATIONS",
        "The table is on a red surface.",
        "The captions mention a white plate and red wine, not a white table/red supporting surface or nested tables. A produces unsupported entity/attribute relations while repeating generic plate/table content. These are caption-grounding findings, not independent photo-factual labels.")
    c = ref[("j9fzpALNCgvDwCXmA5Vcjw", "filtered")]
    marker = next(m for m, body in re.findall(r"\[(REV_\d+)\] (.*?)(?=\n\n\[REV_\d+\]|\Z)", c["source"], flags=re.S) if "bergamot" in body.lower())
    add("j9fzpALNCgvDwCXmA5Vcjw", "filtered", "UNSUPPORTED_INGREDIENT_WITH_REVIEW_ONLY_TERM",
        "The bowl is topped with bergamot.",
        "No supplied caption identifies milk or bergamot. Bergamot occurs in a selected review about a Grey Skies espresso drink, not a visual caption or a bowl topping. The output is consistent with cross-section leakage but the decoding test does not prove an internal mechanism.",
        review_markers=(marker,))
    add("qnprJxrjwuEivitA8V2aEQ", "filtered", "METADATA_FACT_IN_VISUAL_SECTION_WITHOUT_CAPTION_SUPPORT",
        "The restaurant is in Philadelphia.",
        "city=Philadelphia is true in the supplied business metadata, but none of the captions state it. A puts this location fact in VISUAL_EVIDENCE and adds unsupported residential-neighborhood/sidewalk/table-cover details. This is the wrong evidence basis, not proof that the location itself is false.")
    assert "city=Philadelphia" in bykey[("qnprJxrjwuEivitA8V2aEQ", "filtered")]["source"].split("[REVIEWS]")[0]
    add("p7-1xG891NdGrcYBMOvxpw", "unfiltered", "UNSUPPORTED_PERSON_OBJECT_ACTION",
        "A woman is holding a fish.",
        "Caption3 describes a woman sitting at a bar with a fish on the wall. It does not support holding the fish; A changes the object-person relationship. Several chicken/broccoli/rice statements are separately supported by caption4.",
        caption_ids=("IMG_CAP_3", "IMG_CAP_4"))
    add("3v0tSwHXR9LhpsoGXLgsTA", "filtered", "UNSUPPORTED_SPECIFIC_FOOD_IDENTITY",
        "The photo shows a close-up of a seafood sandwich.",
        "Caption2 describes a bag of seafood with potatoes and corn. The other captions are restaurant scenes. No caption establishes a sandwich or bib; A's specific food identity and relations lack supplied visual support.",
        caption_ids=("IMG_CAP_2",))
    write_json(OUT / "AI_QUALITATIVE_REVIEW.json", {"status": "AI_EVIDENCE_LINKED_DIAGNOSTIC_ASSESSMENT_NOT_HUMAN_GROUND_TRUTH",
               "scope": "All48 A visual-section unique-sentence sets inspected against caption pools; illustrative comparisons and selected review claims detailed below. No calibrated population factual-error rate or independent photo verification.",
               "notes": notes, "human_labels_changed": False, "teacher_or_student_content_edited": False})

    def generate_calls(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"]
    old = ROOT / "scripts/run_stage_i_decoding_diagnostic.py"
    local = ROOT / "scripts/run_stage_i_decoding_diagnostic_local_v2.py"
    assert generate_calls(old) == generate_calls(local)
    write_json(OUT / "runtime_code_check.json", {"generate_call_AST_identical_to_reviewed_runner": True,
               "local_runner_changes": "Explicitly approved6GB local admission, A-only authorization and runtime binding/metadata",
               "scientific_generation_overrides": {"num_beams": 1}, "B_executed": False})
    r = paired["runtime"]
    start = datetime.fromisoformat(r["start_utc"]).astimezone(ZoneInfo("Asia/Saigon"))
    end = datetime.fromisoformat(r["end_utc"]).astimezone(ZoneInfo("Asia/Saigon"))
    lines = ["# Stage I Diagnostic A: completed local greedy-decoding comparison", "",
             "**2026-10-16 | Execution/integrity PASS; greedy is not recommended for production adoption. Stop before B/retraining.**", "",
             "Exactly72 new outputs were generated once on the frozen24-business validation subset. All source/target IDs, checkpoint and code/tokenizer bindings match; actual loaded generation configs match the prepared reconstruction for all3 conditions. The only decoding parameter change is num_beams4->1. No accepted data/weights/teacher targets or original predictions were edited; no test access, Teacher calls, B or training.", "",
             "## Random20: saved beam4 -> greedy", "",
             "| Condition | Repeated sentences | Consecutive repeated words | Header compliance | Mean tokens | At768 limit |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in paired["rows"]:
        if row["subset"] == "random":
            lines.append(f"| {row['condition']} | {row['baseline_exact_repeated_sentence_count']}/20 -> {row['greedy_exact_repeated_sentence_count']}/20 | {row['baseline_consecutive_repeated_word_count']}/20 -> {row['greedy_consecutive_repeated_word_count']}/20 | {row['baseline_section_headers_compliant_count']}/20 -> {row['greedy_section_headers_compliant_count']}/20 | {row['baseline_length_mean']:.1f} -> {row['greedy_length_mean']:.1f} | 0 -> 0 |")
    lines += ["", "Exact-sentence repetition uses the unchanged accepted sentence rule; adjacent-word diagnostics ignore case/punctuation. These are distinct textual symptoms, not human factual-error rates. A reduces some RO sentence duplication but leaves85% of random RO outputs with consecutive repeated words; all20 U and all20 F outputs contain repeated sentences. Longer U/F outputs do not demonstrate more information.", "",
              "## Targeted4: separate denominators", "",
              "| Condition | Repeated sentences | Consecutive repeated words | Header compliance | At768 limit |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in paired["rows"]:
        if row["subset"] == "targeted":
            lines.append(f"| {row['condition']} | {row['baseline_exact_repeated_sentence_count']}/4 -> {row['greedy_exact_repeated_sentence_count']}/4 | {row['baseline_consecutive_repeated_word_count']}/4 -> {row['greedy_consecutive_repeated_word_count']}/4 | {row['baseline_section_headers_compliant_count']}/4 -> {row['greedy_section_headers_compliant_count']}/4 | {row['baseline_cap_reached_count']} -> {row['greedy_cap_reached_count']} |")
    lines += ["", "- **Zero-caption F remains0/1 compliant:** `j3kGicU3mPG_ciIKG1Vuyg` generates 'The bagel is on a plate.'/'The plate is on a table.' despite empty captions/photos and an empty teacher visual section. Review plausibility is not visual evidence.",
              "- **Severe cap-loop case partially recovers:** `OpnIXsrqaK-e8Wz5gco7Vw` drops from768 tokens without visuals to344U/246F with visual headers. However, U invents a round-table scene from a pool caption, and F says a plate is topped with hot drinks/straw. Format/termination recovery is not factual repair.",
              "- **Formatting failures remain:** `7u40cyMxw4iGosMdkesoEQ` retains the wrong review header in U/F; its F output now has no VISUAL_EVIDENCE header at all despite ending at577 tokens. `gcQSAQIx-KQOdf5E8HF6Lw` retains VISUAL_CONSENSUS in F. Scene-like text in an unsectioned tail is not treated as safely absent visual content.", "",
              "## Lengths and EOS", "",
              "All72 A sequences start with decoder-start2/BOS0 and terminate with EOS before768; none reaches the cap. Initial decoder-start2 is excluded from EOS tests. This is generation termination, not a semantic completeness guarantee. No content was silently truncated or repaired.", "",
              "| Group | Condition | Baseline median/p95/max | Greedy median/p95/max |", "| --- | --- | ---: | ---: |"]
    for row in paired["rows"]:
        lines.append(f"| {row['subset']} | {row['condition']} | {row['baseline_length_median']:.1f}/{row['baseline_length_p95']:.1f}/{row['baseline_length_max']} | {row['greedy_length_median']:.1f}/{row['greedy_length_p95']:.1f}/{row['greedy_length_max']} |")
    lines += ["", "## Teacher-target agreement, not factual correctness", "",
              "CPU rouge-score0.1.2, unchanged no-stemming macro scoring. Scores use complete original teacher references. BERTScore remains not computed; no new scorer/model download was introduced.", "",
              "| Group | Condition | ROUGE1 F1 beam->A | ROUGE2 F1 beam->A | ROUGE-L F1 beam->A | Review ROUGE-L recall beam->A |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in paired["rows"]:
        cells = []
        for metric in ("teacher_agreement_full_rouge1_fmeasure", "teacher_agreement_full_rouge2_fmeasure", "teacher_agreement_full_rougeL_fmeasure", "teacher_agreement_review_rougeL_recall"):
            cells.append(f"{row['baseline_' + metric]:.4f}->{row['greedy_' + metric]:.4f}")
        lines.append("| " + " | ".join([row["subset"], row["condition"], *cells]) + " |")
    lines += ["", "Teacher agreement is not independent grounding, and full targets differ across conditions. Small increases/decreases on this subset are descriptive, not statistically established superiority. Review recall is a lexical proxy; inspect important opinions, categories and contrasts directly.", "",
              "## Evidence-linked qualitative assessment", "",
              "AI inspection is separated from automated validation and human-final labels. Correct simple descriptions are retained as correct where caption-compatible. No original photos were independently inspected. The following examples are illustrative, not a new prevalence sample or calibrated factual-error count; full exact evidence/photos IDs and unmodified comparison texts are in AI_QUALITATIVE_REVIEW.json and COMPARISON.html.", "",
              "| Case / condition | Observation | Evidence / interpretation |", "| --- | --- | --- |",
              "| `HhD8DLES5ZeULKDXmfkOOg` / U | 'A plate of food is on a wooden table.' remains repeated. | Supported by IMG_CAP_1/4; simple is acceptable. Chicken sandwich/fries/dessert/shrimp/grits distinctions are not recovered. |",
              "| `noUma64stIOSYAQ1GIIFjw` / U | New 'A fork is sticking into it.' | Supported by IMG_CAP_5/PHOTO CRAQin8QklxNcLSl24xCEw; a local improvement, with persistent plate/fork repetition. |",
              "| `gpSv4OvISIeyBAqcNN5bqw` / RO | New pizza list includes buffalo chicken pasta. | REV_10 explicitly distinguishes pizza from that pasta; REV_3/29 support the actual pizza item. Added specificity introduces a food-category error. |",
              "| `HhD8DLES5ZeULKDXmfkOOg` / RO | Generic food/service text still omits sports-bar/beer details and specific contrasts. | REV_10/23 describe beer taps/selection; REV_26 sogginess concerns food, not seating. Some positive/negative template wording survives, but source-specific coverage is weak. |",
              "| `-Wd5KqFszREg0XdXfSv5tg` / U/F | Woman holding/pouring coffee without caption support. | U IMG_CAP_1 says sitting with coffee, not holding; F IMG_CAP_2 pouring subject is a man, while IMG_CAP_3 is a separate woman. Ordinary mask/man-pouring statements are supported. |",
              "| `JcdOQvvx3BqZ5Ga2_9n_Bg` / F | Liquid-in-liquid nonsense disappears, but unsupported bar/red-liquid scene binding remains. | IMG_CAP_1-5 describe food, diners and wine-pouring; removing one bad sentence does not establish grounding. |",
              "| `j9fzpALNCgvDwCXmA5Vcjw` / F | 'The bowl is topped with bergamot.' | No caption names the ingredient. A selected review names bergamot in an espresso drink; this is consistent with cross-section leakage, not proof of its internal cause. |",
              "| `qnprJxrjwuEivitA8V2aEQ` / F | Philadelphia appears under visual evidence. | True in city metadata, absent from the caption pool. Wrong evidence basis is distinguished from an independently false location claim. |",
              "| `p7-1xG891NdGrcYBMOvxpw` / U | 'A woman is holding a fish.' | IMG_CAP_3 places the fish on a wall; unsupported person-object relationship. |", "",
              "Generated visual claims are therefore a mixture of supported ordinary content, lost distinct information, unsupported identities/ingredients/actions and malformed relations. Quantitative human factual-error or semantic-coverage rates are not claimed. The complete review-coverage and visual-support worksheets retain blank human fields for optional researcher inspection.", "",
              "## Execution, resource use and cost", "",
              f"Local time Asia/Saigon: **{start:%Y-%m-%d %H:%M:%S} to {end:%H:%M:%S}**. Total measured pipeline **{r['pipeline_seconds_including_download_import_load_monitor_tail']:.2f}s (8m27.8s)**, including pinned617.69MiB backbone download/import/loading and monitor tail. Summed generation **{r['greedy_generation_seconds_sum']:.2f}s (7m19.1s)**: RO{r['greedy_generation_seconds_by_condition']['reviews_only']:.2f}s, U{r['greedy_generation_seconds_by_condition']['unfiltered']:.2f}s, F{r['greedy_generation_seconds_by_condition']['filtered']:.2f}s.", "",
              f"RTX3050 Laptop6GB: Torch allocated peak{r['torch_peak_allocated_GiB']:.4f}GiB, reserved{r['torch_peak_reserved_GiB']:.4f}GiB; whole-GPU sampled peak{r['whole_GPU_sampled_peak_MiB']:.0f}MiB ({r['whole_GPU_sampled_peak_MiB']/1024:.4f}GiB), including other desktop use.100 approximately5-second samples; short device peaks may be missed. No OOM, no fallback, no failed/repeated generation attempts.", "",
              "**New rental/Teacher/API charges: $0** (local execution, no Vast rental/upload). Electricity/incremental energy tariff was not measured, so no fabricated all-in cash cost. Existing sampled baseline generation totals505.80s on RTX3090; A totals439.13s on RTX3050. Different hardware prevents an isolated decoding speed comparison or bitwise numerical-control claim. The researcher explicitly approved this local substitution; software/precision/input/checkpoint and explicit generation settings remain matched.", "",
              "Original reviewed runner/preparation are unchanged. Local v2 adds6GB hardware admission, A-only authorization and provenance; the model.generate call AST is identical. The actual full GenerationConfig matches the prepared reconstruction in all3 conditions. All48 adapter tensors were checked equal to saved values before/after inference. Pinned model weights, FP32 base/LoRA, FP16 autocast, BOS-only global attention and use_cachetrue are retained. Nonfatal cache/deprecation/optional-Triton warnings are logged; actual CUDA inference succeeded without CPU fallback.", "",
              "## Assessment and next step", "",
              "**Greedy does not meaningfully resolve generation quality and should not be adopted as the production decoder from this result.** It helps some RO sentence duplication and two cap-loop failures, but U/F sentence repetition worsens, word repetition remains high, evidence violations persist and some new category/action/ingredient errors appear. The original beam4 baseline is preserved as a comparison, not declared good enough for final evaluation.", "",
              "The persistence under greedy weakens a beam4-only explanation; generic imitation/insufficient task learning, cross-section evidence binding and limited decoder-only adaptation remain hypotheses. No specific training or LoRA-scope cause is established. No proven source/token/checkpoint/config mismatch was found in this run.", "",
              "**Recommended next step: propose the already prepared Diagnostic B on the same subset, subject to separate approval.** Its no-repeat4 constraint may reduce duplicate strings by construction and can force different unsupported details; assess specificity, contrast, coherent relations and zero-caption behavior alongside repetition. Do not accept B just for a lower repetition count. If it only hides repetition while semantic failures persist, review training limitations before proposing any retraining/module change. No B, retraining, full-validation sweep or held-out evaluation is executed or authorized by this report.", "",
              "Artifacts: paired_summary.json, baseline_greedy_comparison.csv, summary.json/per_case_metrics.csv, execution_integrity_and_runtime.json, AI_QUALITATIVE_REVIEW.json, COMPARISON.html and the unchanged per-condition predictions in the sibling run directory. Human labels and previous25/30-business evaluation status remain unchanged. **Stop after A/evaluation for researcher review.**", ""]
    report = OUT / "DIAGNOSTIC_REPORT.md"
    with report.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"status": "DIAGNOSTIC_A_COMPLETE_NO_PRODUCTION_ADOPTION", "report": str(report),
                      "AI_illustrative_notes": len(notes), "next_step": "RESEARCHER_REVIEW_BEFORE_ANY_B_OR_RETRAINING"}, indent=2))


if __name__ == "__main__":
    main()
