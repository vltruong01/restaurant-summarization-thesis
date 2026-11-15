# Comparisons with explicit boundaries

## A. Generation only with prepared inputs

| System | Full-cohort generation seconds | Calls | Generation allocated peak GiB | Definition |
|---|---:|---:|---:|---|
| LED F-S1000 | 97.931341 | 30 per repeat,2 measured repeats | 2.696645 | Mean after 2 warmups |
| Qwen-Caption | 186.843842 | 30,one run | 18.535985 | Cold pilot included |
| Qwen-Image NEW Visual only | 242.703157 | 15 | 17.756796 | Incremental image generation,Review reused |
| Qwen-Image constructed Review+Visual | 379.696967 | 15 earlier Review+15 new Visual | 18.535985 | Arithmetic across sessions,not fresh combined timing |

Descriptive ratios:Qwen-Caption/LED=1.9079;constructed Qwen-Image/LED=3.8772. These are descriptive timings on the same GPU, not controlled architecture effects or E2E ratios. Different output lengths/tokenizers,warmup,beam4 versus greedy,BF16 versus FP32+FP16,threads1 versus image32,modalities/framework versions and wrapper checks remain confounded. LED3140 versus Qwen-Caption3651 versus constructed image3980 native output tokens are not equal work.

## B. Complete source-to-summary pipeline

Measured shared parts M=BLIP48,642.959s + scorer 178.529s =48,821.488s. BLIP26.295s load and scorer loading are already included;no double addition. Selection/review preprocessing/source collection/cold setup/remaining request CPU costs are missing.

| System | Recorded shared parts M + generation (seconds) | Complete E2E wall time |
|---|---:|---|
| LED F | 48919.419341 | NOT MEASURED |
| Qwen-Caption | 49008.331842 | NOT MEASURED |
| Qwen-Image with actual inherited Filtered selection | 49201.184967 | NOT MEASURED |
| Hypothetical caption-independent image selection | NOT MEASURED | NOT MEASURED |

These sums are accounting illustrations from separately measured historical RTX3080/CPU prep and RTX3090 generation. They are NOT observed job times,not monetary cost,not a matched end-to-end benchmark and not a rigorous simultaneous-wall lower bound (scheduling/overlap differ). Missing terms are never treated as zero. Separate model loads can be reported but adding them still cannot fill missing pipeline steps or transform these sums into E2E measurements.

All three actual variants inherit the same selected-photo provenance. Image has no caption text at inference,but removing shared BLIP/filter costs while keeping the same caption-selected 73-photo set would describe an oracle/precomputed-selection boundary,not raw-photo end-to-end execution. The5,143-to73 scope mismatch is disclosed;neither per-photo extrapolation nor an unrun direct-selection replacement is made.

Measured original Stage O LED F loop was145.008172s on RTX3080;using it instead yields another historical component subtotal,not a controlled Qwen comparison. A same-device native all-GPU BLIP runtime is NOT MEASURED;we do not assume that13.51h CPU-offloaded execution represents an optimized3090 pipeline.

## C. Precomputed/reused captions and selected photos

Actual Stage P reused the prepared Stage O captions/selection;no new BLIP generation was performed for these inference runs. At that operational boundary,the observed summary-generation differences are relevant. Precomputed work has a prior cost even when marginal regeneration is zero.

For a hypothetical N requests over this SAME15-restaurant cohort with unchanged sources and cached caption scores/selection,known measured components per request are M/N + G(system). Full cost would be `(M + unknown_once)/N + G + unknown_request(system)`. N is a reuse scenario,not observed repeated-service data. The table omits unknown terms explicitly and cannot be called total latency:

| Hypothetical full-cohort requests N | M/N seconds | LED known components/request | Qwen-Caption known components/request | Qwen-Image inherited-selection known components/request |
|---:|---:|---:|---:|---:|
| 1 | 48821.488 | 48919.419 | 49008.332 | 49201.185 |
| 10 | 4882.149 | 4980.080 | 5068.993 | 5261.846 |
| 100 | 488.215 | 586.146 | 675.059 | 867.912 |
| 1000 | 48.821 | 146.753 | 235.665 | 428.518 |

This is arithmetic amortization,not newly measured throughput. It assumes representative same workload repeats, cache validity, no source changes, sequential fresh summary generation and no fresh model load per request. Only captions/selection are cached; final summaries are not assumed to be cached. Reusing complete deterministic summary outputs would be a different, unmeasured service boundary. No cache-hit overhead,concurrency/load testing or amortization break-even is measured. An infinite-cache/sunk-preparation boundary approaches generation-only costs plus unknown request overhead,not complete pipeline latency.

If Qwen-Caption Review already exists,adding the actual image variant costs242.703157s new generation,not379.696967s. Its complete recorded Visual job elapsed248.988514s includes5.847594s model loading,0.330128s CPU image preprocessing,0.018900s transfer and0.002985s postprocessing plus other overhead. Existing Review/selection reuse must be named rather than recharged as a fresh measured job or silently assumed free from source.

## Claims

Supported:smaller recorded prepared-input generation times and allocator peaks for this LED implementation;resource tradeoffs under cached-source preparation;explicit descriptive known-component accounting.

Not supported:LED is faster end-to-end;LED's full pipeline requires only2.697GiB;Qwen-Image's actual image selection is caption-free;an exactly matched5,143-photo E2E advantage;energy/dollar efficiency;quality superiority from resource numbers. No confidence/significance claim is based on two LED repetitions versus one Qwen run.
