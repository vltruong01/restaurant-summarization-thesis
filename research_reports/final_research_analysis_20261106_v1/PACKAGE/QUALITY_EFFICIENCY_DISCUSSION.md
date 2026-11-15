# Quality and Computational Efficiency

## 1. What was actually measured

The existing read-only efficiency audit is the resource evidence for this discussion. All values refer to the full 15-restaurant cohort unless explicitly marked otherwise. GiB means 2^30 bytes. Allocated and reserved memory are PyTorch allocator measurements; neither is total device usage nor minimum required VRAM. [E15]

| Generation boundary | Time (s) | Calls | Peak allocated / reserved (GiB) | Main limitation |
| --- | ---: | --- | ---: | --- |
| LED F, prepared inputs | 97.931 | 30 per repeat; two measured repeats | 2.697 / 3.045 | Mean after two warmups; 3,140 native output tokens |
| Qwen-Caption | 186.844 | 30; one run | 18.536 / 18.717 | Cold pilot included; 3,651 native tokens |
| Qwen-Image, new Visual only | 242.703 | 15 | 17.757 / 17.820 | 73 selected photos; 1,342 new native tokens |
| Qwen-Image, constructed Review + Visual | 379.697 | 15 earlier Review + 15 new Visual | 18.536 / 18.717 across component stages | Arithmetic across sessions; 3,980 native tokens; not a new combined timed job |

The ratio of observed generation time to LED is 1.908 for Qwen-Caption and 3.877 for the constructed Qwen-Image figure. LED repeat times are 97.435 and 98.428 seconds (sample SD 0.702); the Qwen measurements have no equivalent replication estimate. No inference-time significance test is justified. The actual new Image extension required 242.703 seconds of generation when its Review component already existed.

All three generation measurements use the same RTX 3090, but different work: LED uses FP32 parameters with FP16 autocast and beam search of size four; Qwen uses BF16 and greedy decoding with thinking disabled. Native token counts are tokenizer-specific. Framework versions, input modalities, CPU threading, output lengths and warmup boundaries differ. The LED resource benchmark reproduces 22/30 accepted original output components exactly, while its two repeats agree on 30/30. Frozen original outputs, not benchmark replacements, were evaluated by people. Thus the measured resource/quality association is approximate at the implementation level, not a per-output deterministic replay claim. Model loading was separately recorded (LED 9.279 s, Qwen-Caption 11.533 s, Image Visual 5.848 s) and is excluded from the generation table. [E15]

## 2. Pipeline accounting and missing measurements

| Existing preparation stage | Observed work | Recorded time | Boundary |
| --- | --- | ---: | --- |
| BLIP-2 | 5,143 photos; 5,150 attempts including seven retries | 48,642.959 s (13.51 h) | RTX 3080 with CPU/GPU offload; includes 26.295 s loading |
| CLIP/SBERT caption scoring | Scores/embedding export | 178.529 s | Includes loading, hashing and image I/O; not final selection |
| Final relevance/redundancy/diversity/K selection | Frozen selection exists | NOT MEASURED | Cannot substitute scorer time |
| Review selection, raw source acquisition, complete request overhead | Source records exist | NOT MEASURED as a complete comparable interval | Cannot set to zero |
| Complete source-to-summary wall time | Each whole system | NOT MEASURED | No matched end-to-end conclusion |

The two recorded preparation components sum to 48,821.488 seconds, but adding that number to generation yields historical component accounting, not an observed end-to-end run. Preparation occurred on a different device and software path; missing terms, loading boundaries and possible overlap remain. Dividing BLIP time by 5,143 and multiplying by 73 would fabricate an unobserved comparison and is not performed.

LED F and Qwen-Caption consume frozen Filtered caption text. Qwen-Image consumes the corresponding 73 selected photographs directly, but the actual photograph selection was derived from the BLIP/filter pipeline. It is caption-free at the generation input, not independent of caption-based selection end to end. Removing BLIP/filter costs while retaining those 73 selected photographs assumes precomputed selection. An alternative direct-image selector was not benchmarked. Processing 5,143 source images and 73 selected images is not an equal-photo-workload comparison. [E15]

Sequential recorded stage maxima also differ from inference-only peaks. BLIP's allocated peak of 4.168 GiB exceeds LED generation's 2.697 GiB; the recorded maximum across LED's measured sequential stages is therefore at least 4.168 GiB. This does not certify a 4.168-GiB deployment: device overhead, unmeasured stages and offloaded CPU RAM also matter. Qwen's combined component peak remains 18.536 GiB. Sampled whole-device memory is separately labeled in the original audit and is not mixed with allocator peaks.

## 3. Cached captions and selection

Stage P actually reused Stage O captions and selection. Prepared-input generation figures are useful for that operational boundary, provided prior preparation is disclosed. For hypothetical N repeated full-cohort summarization requests with unchanged sources, an accounting model is `(48,821.488 + unknown_once)/N + G_system + unknown_request`. This is amortization, not measured throughput or total latency. It assumes reusable captions/selection, no source refresh, and no cache of final summaries. No service-load test, energy measurement, cost comparison or break-even point is available. [E15]

## 4. Quality accompanying the resource difference

Qwen-Caption exceeds LED on faithfulness (+1.033), coverage (+1.533), coherence (+0.933), non-redundancy (+1.633) and usefulness (+1.467) in the official P collection. All five differences survive the primary Holm family. Visual scores are identical cell by cell, not evidence of overall quality equivalence. Review outputs are longer for Qwen (132.87 versus 81.07 words), a material scope confounder. The comparison is between a large pretrained instruction-following model and a smaller task-tuned LED, so architecture, parameter count, pretraining, prompts and generation budgets are not isolated. [E11–E14]

The Image variant improves mean Visual scores by 0.633 over Caption, but the primary Holm-adjusted p=0.147827 is inconclusive. The additional Image–LED comparison has adjusted p=0.021118 in its separate six-test family. Both findings must be retained. Direct pixels may recover caption omissions and avoid caption errors, but this sample does not establish a general modality effect or a statistically robust primary Image-over-Caption improvement.

## 5. Defensible thesis and deployment claims

Supported wording is: “For these prepared 15-restaurant inputs, the tested LED implementation had lower measured generation time and allocated-memory peaks than Qwen, while Qwen-Caption received higher human scores on five criteria. Complete end-to-end efficiency was not measured.”

The experiments demonstrate practical LoRA adaptation on a consumer GPU and a potentially useful prepared-input option when memory or latency is constrained. They do not justify claims that LED is faster end to end, has quality comparable to Qwen overall, requires only its inference allocator peak to deploy, or is more energy/cost efficient. A future deployment decision should explicitly choose an acceptable factuality/coverage standard and measure the complete intended service boundary. That prospective work has not been executed here.
