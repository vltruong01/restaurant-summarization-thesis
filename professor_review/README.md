# Thesis research overview — public source edition

Prepared 15 November 2026 from completed research frozen through 6 November 2026.

## Start here

1. Extract the ZIP completely.
2. Open **THESIS_RESEARCH_OVERVIEW.html** in a browser for immediate offline reading. No installation, account, GPU or model download is needed. Expand the generated-output examples and source-ID records as desired.
3. Alternatively, open **THESIS_RESEARCH_OVERVIEW.ipynb** in Jupyter or VS Code. All 25 code cells have saved outputs, including 11 embedded figures. Trust the notebook only if you trust this package's source; HTML is the simplest reading copy.

The notebook covers Yelp preparation, review selection and token budgets, BLIP-2 captions, CLIP/SBERT filtering, K=5, Teacher targets, LED LoRA training, model selection, RO/U/F/Shuffled, validation, final TEST100 automatic evaluation, TEST30 ratings, TripAdvisor Stage O, original Stage P V1 Qwen comparison, and efficiency boundaries. It shows exact implementation excerpts, complete illustrative outputs and frozen-result tables.

## Evidence and conclusions

- VALIDATION200 Teacher-target similarity, TEST100 human-reference similarity and source-grounded ratings are separate evidence tracks.
- Stage N: 30 Yelp businesses, four conditions, two accounts, 240 records. Stage O: 15 TripAdvisor restaurants, RO/U/F, 90 records. Stage P: the same 15 restaurants, LED F/Qwen-Caption/Qwen-Image, 90 original V1 records. Do not pool them as independent restaurants.
- Filtering reduces some repetition and F substantially exceeds Shuffled on Visual grounding; F does not consistently outperform U. Qwen-Caption receives higher Stage P ratings than LED on five criteria. Pointwise intervals, multiplicity policy and inconclusive findings are retained.
- LED's measured generation time/memory advantage uses prepared inputs. BLIP-2/scoring preparation, missing final-selection timing, different photo workloads, reused Qwen Review and allocator-memory definitions prevent an end-to-end superiority claim.
- References have disclosed AI feedback/editorial assistance. Recorded approval and evaluator accounts do not independently authenticate identity, unaided authorship or source inspection. Stage O retains a historical test-context declaration, later identified by the researcher as a test; subsequent human participation is researcher-reported. Known audit concerns are preserved rather than silently corrected. Flags are rater reports, not an independently adjudicated factual-error census.

## Optional local verification and replay

Python 3.11 or later is sufficient for the standard-library integrity/arithmetic checker:

```text
python -B verify_package.py
```

It verifies delivery hashes, data alignment and frozen summaries without downloading packages or models. Verify the delivered snapshot before editing or resaving it.

To re-execute the lightweight analyses, create a separate environment. These are CPU analysis/Jupyter packages, not a training runtime:

```text
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -r requirements.lock.txt
.venv\Scripts\python -B run_notebook.py --output ../REPRODUCED_NOTEBOOK
```

On macOS/Linux use `.venv/bin/python`. Internet is needed only to install those packages initially. The analysis itself uses bundled files offline. `run_notebook.py` writes the reproduced notebook/HTML outside this delivery folder, preserving its hash-verified snapshot. The reproduced pair should remain alongside the original package; its notebook uses the original bundled inputs when replayed through this script. For interactive Run All, launch Jupyter in the extracted package directory. Resaving a notebook changes its delivery hash even when numerical results agree.

The replay recomputes arithmetic, agreement, existing paired point estimates and the frozen Stage P bootstrap/exact-test policy. It does **not** run ROUGE/BERTScore models, generate captions/summaries, train, call paid APIs or collect ratings. N/O intervals are existing exports with verified paired means, not new statistical policies.

## Directory structure

- `THESIS_RESEARCH_OVERVIEW.ipynb` / `.html`: main review artifacts.
- `data/`: lightweight original Yelp exports and final O/P tables.
- `code/`: exact implementation excerpts; displayed as source, not run as models.
- `evidence/`: unchanged original O/P ratings, output/mapping bindings, specification/alignment receipts, generated external examples, efficiency evidence and original evidence-hash bindings.
- `verify_package.py`, `verify_extension.py`: offline integrity/arithmetic checks.
- `run_notebook.py`, `requirements.lock.txt`: optional isolated analysis replay.
- `VALIDATION_REPORT.json`: actual execution, portability and preservation checks.
- `PACKAGE_MANIFEST.csv`: byte sizes and SHA-256 of every other member. The manifest excludes itself to avoid a recursive hash.

Historical evidence documents refer to source paths/evidence IDs within their original full research archives. They are preserved provenance snapshots, not commands to execute or a claim that all those large assets are included here. Executable notebook paths are local and bundled. The original Qwen planning specification retains stale pending/fallback fields; the notebook distinguishes them from completed output/runtime evidence. No fallback model was used.

## Full implementation and archived assets

- Full source/configurations: https://github.com/vltruong01/restaurant-summarization-thesis
- Archived research assets: https://huggingface.co/datasets/vltruong01/multimodal_summarization_thesis

The code repository is public. Original source data and archived assets remain restricted. This package contains no credentials. Full training/inference reproduction requires separately restored data, weights and pinned runtimes. Viewing and summary analyses do not. Original research records and the earlier notebook edition are preserved unchanged.


## Public-edition source boundaries

Original Yelp and TripAdvisor review text/photos and private restoration catalogs are not included. All generated example outputs, numerical ratings/metrics, statistical policies and limitations remain. Source review IDs and original evidence hashes identify restricted records; they do not grant redistribution rights. Historical manifests describe their original private packages, not a claim that every source artifact is public. See `PUBLIC_RELEASE_SCOPE.md`.
