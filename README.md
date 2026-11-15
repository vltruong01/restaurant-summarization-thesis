# Restaurant Summarization — Thesis Code and Results

**Master's thesis · public supervisor-review edition · 15 November 2026**

## Start here

### [Open the research overview notebook](professor_review/THESIS_RESEARCH_OVERVIEW.ipynb)

The notebook displays actual implementation excerpts, completed results,25 executed code cells and11 embedded figures. It covers Yelp preparation, review budgets, BLIP-2/CLIP/SBERT caption filtering (K≤5), Teacher supervision, LED-large LoRA, RO/U/F/Shuffled, TEST100 automatic evaluation, TEST30 ratings, TripAdvisor generalization, Qwen baselines and fair efficiency accounting.

- [Notebook reading and execution guide](professor_review/README.md)
- [HTML reading copy](professor_review/THESIS_RESEARCH_OVERVIEW.html): download the raw HTML for offline viewing; GitHub displays its source.
- [Public-edition validation](professor_review/VALIDATION_REPORT.json)
- [Public release scope and source-data boundaries](PUBLIC_RELEASE_SCOPE.md)

## Complete implementation

| Directory | Contents |
|---|---|
| [scripts/](scripts/) | All936 original experimental scripts |
| [configs/](configs/) |111 experimental configurations |
| [tests/](tests/) |57 implementation test files |
| [notebooks/](notebooks/) | Development notebooks, source cells retained; historical saved outputs withheld |
| [prompts/](prompts/), [schemas/](schemas/) | Prompt and data-format specifications |
| [specs/](specs/), [thesis_sdd_docs/](thesis_sdd_docs/) | Methodology and research records |
| [research_reports/](research_reports/) | Completed findings, discussion and thesis figures |
| [professor_review/](professor_review/) | Executed overview and lightweight reproducible numerical analyses |

## Findings and interpretation

F substantially exceeds Shuffled on Visual grounding and reduces some repetition, but does not consistently outperform U. In Stage P, Qwen-Caption receives higher human ratings than LED on five criteria. LED has lower measured prepared-input generation latency and PyTorch allocator-memory peaks. Complete end-to-end efficiency was not measured; BLIP-2 preparation, missing selection timing, different photo workloads, reused components and output-length differences limit that comparison.

VALIDATION200 Teacher-target similarity, TEST100 human-reference similarity and source-grounded ratings are separate evidence tracks. Stage N contains240 recorded ratings on Yelp TEST30; Stage O90 on TripAdvisor15; Stage P90 official original V1 records on the same TripAdvisor cohort. Do not pool these dependent cohorts. Paired units, confidence intervals, exploratory tests and multiplicity policies are reported explicitly.

Recorded approvals/evaluator accounts do not independently authenticate identity, unaided authorship or source inspection. AI feedback/editorial assistance and known audit concerns remain disclosed. No scores, generated outputs or research conclusions were changed for publication.

## Lightweight verification

```text
cd professor_review
python -B verify_package.py
```

Python3.11+ is sufficient for this integrity/arithmetic check. Optional isolated CPU replay instructions are in the [guide](professor_review/README.md#optional-local-verification-and-replay). Viewing or replaying summary analyses needs no GPU, paid API, large model download or full Yelp dataset.

## Data and access

Original Yelp/TripAdvisor review text, photo files and private recovery catalogs are intentionally excluded. This fresh public snapshot does not expose the private repository's earlier commits. Generated outputs, figures, numerical results and pseudonymous research ratings are included. Dataset permissions remain separate from code access; see [release scope](PUBLIC_RELEASE_SCOPE.md) and the preserved [Dataset User Agreement](docs/Dataset_User_Agreement.pdf).

Large restricted research assets remain in the owner's [private Hugging Face archive](https://huggingface.co/datasets/vltruong01/multimodal_summarization_thesis). They require separate permission and are unnecessary for reading this overview. Historical research reports may reference their private full packages; the public numerical replay uses `professor_review/`.
