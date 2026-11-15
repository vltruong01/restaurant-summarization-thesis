# Final Research Results, Cross-Stage Analysis and Discussion

Researcher-review package. Begin with `FINAL_RESULTS_ANALYSIS.md`, then the original-RQ answers and academic discussion draft. The current Word report is unchanged. This is not a new experiment or a final thesis submission.

## Contents

- Six requested Markdown reports at the root: integrated results, research questions, quality/efficiency, limitations, discussion draft and evidence traceability.
- `VERIFIED_TABLES/`: independent statistics, complete Stage P distributions/per-rater results and all18 contrasts; original frozen-script replay; machine-readable verification receipt.
- `SOURCES/`: byte-preserved selected original M/N/O evidence, the compact recovered original P V1 evidence, efficiency-audit reports, original-RQ/style extracts and accepted training provenance. Original research protocols are historical evidence, not instructions to run new experiments.
- `SCRIPTS/`: CPU-only reproduction and package checks. `requirements.txt` pins the numerical/plotting environment.
- `FIGURES/`: separate-stage descriptive means and Stage P paired intervals. No pooled score or composite ranking.
- `SOURCE_INVENTORY.json`, `SOURCE_PRESERVATION.json`, `PACKAGE_MANIFEST.json` and `VALIDATION_REPORT.json`: exact bindings and actual checks.

## Reproduce using only small included evidence

Python3.11 and NumPy1.26.4 are sufficient for statistical verification. No GPU, paid API, model download or full dataset is required. From this extracted directory:

```text
python -B SCRIPTS/verify_review_package.py --package .
python -B SCRIPTS/verify_integrated_research_statistics_v1.py --package . --output ../fresh_statistics_reproduction
```

The second command requires a new output directory and writes only derived files there. It reruns arithmetic/bootstrap/sign-flip calculations and the included frozen P analyzer, not models, reference creation or similarity scoring. The original archives are not needed for this portable replay; their original hashes/check receipts remain available. Rebuilding prose uses the repository-local report builder and is not required to read or validate the package.

The statistical reproduction also invokes `SOURCES/P/VERIFY_PACKAGE.py --package SOURCES/P` to recheck the original collection, assignment bindings and journal. The root verifier checks the entire integrated package. Do not launch historical application or packaging scripts for review.

## Scope and key conclusions

Official Stage P is recovered original V1, archive SHA-256 `e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5`. The external ChatGPT analysis ZIP is not claimed to have been inspected. Its absence was resolved by the researcher's instruction to reproduce from V1. Later rating revisions are excluded.

Alignment matters: F exceeds Shuffled on Yelp Visual outcomes. Filtering has some repetition benefits but does not show consistent superiority over U. Stage P Qwen-Caption exceeds LED on five criteria after its frozen correction; equal Visual scores do not imply overall equivalence. Qwen-Image's primary Visual advantage over Caption is inconclusive after Holm. LED's smaller recorded prepared-input generation footprint does not establish complete-pipeline superiority.

All stages remain separate. The package preserves original ratings and known provenance limitations. It does not independently authenticate participants or adjudicate every flagged claim. Stop for researcher review; no checkpoint reselection, further model execution or human collection is authorized by this deliverable.
