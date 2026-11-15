# D-054 paired pilot generation and quality evaluation

Status: researcher-authorized bounded execution,2026-10-17. Supersedes only historical statements that D-054 authorization is pending; no D049/D053 scientific definitions are changed.

Completion update (report finalized2026-10-18):all48 authorized outputs were generated and evaluated. Both model/checkpoint/reference/input integrity checks passed; all post-inference backbone/adapter tensors remained unchanged. Random20 structure compliance is0/20 for each arm; targeted zero-caption handling also fails the required output representation. Quality interpretation remains researcher-review pending; no full retraining or additional decoding is authorized. See outputs/stage_i/d054_paired_generation_v1/STAGE_I_D054_PAIRED_GENERATION_REPORT.md.

Current authorization and request hashes are preserved in `outputs/stage_i/d054_paired_generation_v1/RESEARCHER_AUTHORIZATION.json` and `RESEARCHER_REQUEST.txt`.

Generate exactly48 outputs: D052 matched decoder-only A and encoder+decoder B,24 Filtered validation inputs each,20 frozen random plus4 separately reported targeted businesses. D053 source references were independently accepted before generation. Dispatch explicitly to the accepted strict v2 reference validator; never bypass or weaken the unchanged original validator. Verify all source/adapter/checkpoint/backbone/config bindings before inference; stop on unexpected mismatch or OOM without fallback.

Use the unchanged effective generation configuration: beam4,do_sample=false,no_repeat_ngram_size=0,max_length768 (includes decoder start), and all remaining frozen fields. Original FP32 weights,FP16 CUDA autocast,batch1,global attention at token0,1024 padding multiple and untruncated source content are preserved. No extra generation, decoding search, model-weight update, training or held-out access.

Reuse deterministic D049/D053 repetition, structure, EOS and paired bootstrap definitions:20 business pairs,seed20261016,5000 resamples,95% percentile intervals. Targeted4 get individual diagnostic results,not pooled prevalence/inferential CI. Preserve raw output including malformed marker loops and unfinished text; the 768 generation cap is not silently treated as early EOS.

Use576 accepted references:261 Core315 Supporting;88 exclusions and2 uncertain parent candidates are audit-only. Core does not mandate verbatim inclusion of every atom. Supporting omission alone is not an error. Complete original reviews remain permitted review evidence; captions alone constrain visual content and photos only corroborate them. No photo-only claim support. Generic compatible descriptions are not factual errors.

Distinguish deterministic metrics,Codex model-assisted source-based assessments,and researcher-confirmed output evaluations. Source-reference approval does not approve student outputs. Human output metrics remain N/A until genuine complete confirmation. Store exact output spans,reference IDs,REV/caption/photo IDs and original quotations for claims and omissions. Unparseable outputs are not assigned perfect precision. No new embedding/scorer/threshold is chosen from results. Use the frozen blinded left/right mapping in the offline interface; arm key and unblinded reports remain separate.

Report dimensions separately and explain one-seed/two-epoch/512-train/24-validation limitations. Architecture location and parameter count both differ. Lower loss or repetition alone cannot establish an architectural winner. Deliver raw48 outputs,automatic and source-based assessment,per-business comparisons,report,blinded HTML and self-contained SHA-bound review ZIP; stop for researcher review. Full retraining,teacher regeneration,reference modification and further decoding experiments are not authorized.
