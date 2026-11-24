# Restaurant Summarization — Master's Thesis

Implementation, frozen results and an executed research overview for supervisor review.

## Read the research

The [TripAdvisor15 interactive results viewer](https://vltruong01.github.io/restaurant-summarization-thesis/tripadvisor15/) shows aggregate ratings, restaurant-level comparisons, resource measurements and model provenance. This complete demonstration includes the selected TripAdvisor reviews, 73 original photos with BLIP-2 captions, all three systems’ original Review Consensus and Visual Evidence, and original pseudonymous A01/A02 scores.

Start with [Thesis_Summary.pdf](thesis/Thesis_Summary.pdf) for a short overview of the research questions, method, results and limitations.

For the full thesis, read [thesis_ff.pdf](thesis/thesis_ff.pdf). It provides the detailed methodology, experiments, results and discussion.

For the defense presentation, download [thesis_slides.pptx](thesis/thesis_slides.pptx) and open it in PowerPoint.

## Notebook and code

**[THESIS_RESEARCH_OVERVIEW.ipynb](professor_review/THESIS_RESEARCH_OVERVIEW.ipynb)** contains the research design, actual code excerpts, result tables and 11 saved figures. All 25 code cells already have outputs; no GPU or dataset download is needed to read it.

- [Offline HTML version](professor_review/THESIS_RESEARCH_OVERVIEW.html) — download and open in a browser.
- [Notebook verification and optional CPU replay](professor_review/README.md).
- [Code guide: method → implementation](docs/CODE_GUIDE.md).
- [Research provenance and assistance disclosures](docs/RESEARCH_PROVENANCE.md).

## Thesis materials

| Location | Purpose |
|---|---|
| [thesis/](thesis/) | Full thesis, research summary and defense slides |
| [professor_review/](professor_review/) | Executed overview, lightweight frozen metrics/ratings, model settings, examples and reproducible numerical analysis |
| [scripts/](scripts/) | Selected implementation of dataset preparation, review selection, captioning/filtering, supervision, inference and analysis, with required local helper modules |
| [configs/](configs/), [prompts/](prompts/), [schemas/](schemas/) | Supporting original method contracts, Teacher prompts and data schemas |
| [requirements/](requirements/) | Original stage-specific dependency specifications; the notebook has its own small CPU lockfile |
| [Final results and discussion](research_reports/final_research_analysis_20261106_v1/PACKAGE/FINAL_RESULTS_ANALYSIS.md) | Cross-stage findings, statistical interpretation and limitations |
| [Thesis figures](research_reports/chapter4_figures_20261112_v2/) | Final PNG/PDF figures and matplotlib source |

SDD, internal decision diaries, abandoned experiment entrypoints, cloud-management, backup/recovery and handoff tooling are omitted from the current tree. Some historically named helpers remain because the final implementation imports them. This is a focused review edition, not the complete private execution archive.

## Findings and boundaries

F substantially exceeds Shuffled on Visual grounding and reduces some repetition, but does not consistently outperform U. Qwen-Caption receives higher Stage P ratings than LED on five criteria. LED's measured latency and allocator-memory advantage uses **prepared inputs only**; it does not establish end-to-end efficiency superiority.

VALIDATION200 Teacher-target agreement, TEST100 human-reference similarity and source-grounded human ratings are separate evidence tracks. Yelp TEST30 has 240 records, TripAdvisor Stage O has 90, and Stage P uses the 90 official original V1 records. These dependent cohorts are not pooled. AI assistance, recorded-approval limitations and attributed rater flags remain disclosed in the notebook and final reports.

## Verify without installing ML libraries

```text
python -B professor_review/verify_package.py
```

Python 3.11+ suffices. Optional CPU notebook replay uses the separate environment described in its guide. Historical model scripts require separately authorized source data, weights and archived runtime inputs; do not run them just to view results.

## Data access

Original Yelp review bodies/photos, model weights and private restoration records are not distributed here. The TripAdvisor15 viewer includes its selected reviews, 73 frozen photo files and captions, original model outputs and pseudonymous criterion scores at the researcher’s request. Public source IDs/hashes bind the evidence without redistributing it. The complete original code/history and research archives remain private; the owner's [Hugging Face archive](https://huggingface.co/datasets/vltruong01/multimodal_summarization_thesis) requires separate access. See [public release scope](PUBLIC_RELEASE_SCOPE.md) and [dataset agreement](docs/Dataset_User_Agreement.pdf).
