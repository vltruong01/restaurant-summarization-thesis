# D-053 future paired evaluation procedure

This is offline preparation. No Pilot A/B summaries have been generated or evaluated. The script never loads a model, calls an API or runs GPU inference. Generating48 summaries requires separate researcher approval after source-reference verification.

Run CPU readiness from the project root:

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\stage_i_d053_evaluation.py --readiness
.\.venv-caption-pilot\Scripts\python.exe scripts\test_stage_i_d053_evaluation.py
```

`--write-future-shell NEW.json` creates48 empty annotation shells. It contains no prediction text, spans, support labels or fake reference facts. Its blank coverage lists are populated only from the later frozen supported reference units, never from the226 unverified candidates.

After source verification and separately approved generation, preserve one JSONL file of exactly48 actual output records. Each record must include:

```json
{
  "arm": "A_decoder_only or B_encoder_decoder",
  "business_id": "unchanged validation business ID",
  "condition": "filtered",
  "subset": "random or targeted as frozen",
  "source_sha256": "exact source hash from source_cases.json",
  "adapter_sha256": "exact selected D052 checkpoint128 adapter hash",
  "generation_config_sha256": "SHA256 of the unchanged expected_generation_config.json file",
  "reference_freeze_sha256": "SHA256 of the final source-reference freeze available before generation",
  "text": "raw decoded output, including malformed sections if present",
  "token_ids": [2, 0, 42, 2]
}
```

The illustrative token list above is schema documentation, not a student prediction. The actual generation runner must store exact generated IDs, verify its complete effective GenerationConfig against the frozen JSON, and preserve its approval/execution logs. Both arms must use the identical source token arrays and generation settings. `max_length=768` includes decoder start; it is not `max_new_tokens=768`.

Actual D052 selected adapter hashes:

- A: `245fedd5d5959530f77472b059e18ba9bcdaa89b867b8c84918509cb28f479d9`
- B: `21dee7c5d5fe270553b2732f0ee61b636c7bcf20ee87a3b2df0e0b2274e959f5`

After those files exist, the following CPU-only operation checks all source-reference, configuration, adapter, case and output bindings before producing metrics:

```powershell
.\.venv-caption-pilot\Scripts\python.exe scripts\stage_i_d053_evaluation.py --compare-existing --reference-annotations CONFIRMED_ANNOTATIONS.json --reference-freeze CONFIRMED_REFERENCE_FREEZE.json --predictions ACTUAL_PAIRED48.jsonl --output NEW_COMPARISON.json
```

The operation is not authorized to create missing predictions. It refuses incomplete source references, duplicate/missing outputs or mismatched hashes. Output files are created exclusively and never overwrite previous work.

The initial report includes empty `pending_output_annotation_templates`, one for each output not yet annotated. Save their list as a separate JSON file, inspect outputs in the frozen D049 anonymized display order where feasible, then annotate every factual/opinion proposition in the entire raw output. Do not repair malformed sections before annotation. Set exact Python Unicode code-point start/end offsets and retain the verbatim span. The supplied schema in `stage_i_d053_evaluation_v1.json` defines claim fields, human support labels, severity, evidence, semantic group and notes. Supported visual claims must be recoverable from the corresponding input caption and corroborated by the photo; no photo-only or invented cross-caption relationship counts as support.

Assign every frozen source unit `covered`, `partly_covered`, `omitted` or `uncertain`; covered/partial units require links to output claim IDs. Review meaningful positive/negative contrasts, generic content, low-information content, incomplete clauses, incoherence, contradictions and visual assertions anywhere. Confirm the whole-output completeness flags only after reviewing the full raw output. These software flags document researcher provenance; they are not independently machine-confirmed human truth.

Rerun the same command with `--output-annotations CONFIRMED_OUTPUT_ANNOTATIONS.json --output ANOTHER_NEW_COMPARISON.json`. The annotations file is a list of the48 per-output annotation objects, not the surrounding shell metadata object. Incomplete/unverified human judgments produce `N/A`, never zero errors or perfect support. Uncertain claims remain separate.

Repetition uses unchanged accepted exact-sentence and adjacent-word rules plus normalized lexical and stored generated BPE2/3/4-grams. Duplicate occurrences are `sum(count-1)` over all overlapping opportunities; zero opportunities are `N/A`. BPE measures exclude known special tokens and never approximate generated token IDs by re-tokenizing text. Semantic redundancy is researcher-confirmed proposition grouping, without an embedding model or new threshold.

Source coverage uses fully covered units as its primary numerator; partial coverage receives0.5 only in the explicitly separate sensitivity summary. Grounding reports each non-uncertain class and uncertain cases, domain-specific counts, business-macro support precision, uncertainty sensitivity and descriptive pooled claim counts. Empty output does not earn perfect precision. Structure checks allow an empty visual section even when input captions exist; zero-caption factual compliance also requires the researcher to confirm there is no visual assertion elsewhere.

The20 random businesses use paired business resampling with the original D049 algorithm: NumPy1.26.4 `default_rng(20261016)`,5000 paired resamples,95% percentile interval. The same business remains paired across arms. The four targeted businesses are reported individually with no inferential CI or pooled prevalence. Report denominators, missing judgments and observed regressions. There is no composite quality score or automatic winner. Lower repetition, lower validation loss, higher distinct-n or closer teacher agreement alone cannot establish improved factual quality.

No new ROUGE implementation or BERTScore model is introduced. ROUGE may only use the existing approved secondary teacher-agreement implementation; it is not factual ground truth. Factual/source coverage judgments require confirmed human annotations.
