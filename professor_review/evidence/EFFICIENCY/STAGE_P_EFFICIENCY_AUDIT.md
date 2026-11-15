# Stage P computational efficiency audit

Audit date:6 November 2026 (Asia/Saigon). Read-only analysis of existing Stage O/P computational receipts. No GPU inference,model downloads,network/paid calls,caption regeneration,quality evaluation or human-rating statistical analysis was performed. Human-rating revisions are not inputs and no official results are changed.

## Defensible conclusion

With prepared inputs and the recorded policies,on the same RTX3090,LED F-S1000 had lower synchronized generation time and lower inference allocator peaks than the two Qwen configurations. This is a descriptive implementation/workload comparison,not an isolated architecture effect or an end-to-end speed result. A complete from-source latency or minimum GPU-capacity advantage is NOT ESTABLISHED.

The cost before summary generation is substantial:BLIP-2's5,143-photo stage took 48,642.959s (13.51193h) and CLIP/SBERT scoring took 178.529s. Their 48,821.488s subtotal is historical RTX3080/CPU accounting,not full preprocessing and not a current3090 benchmark. The final filtering/diversity/K selection and review/source acquisition costs are NOT MEASURED.

Crucially,the frozen extension config states that ALL THREE systems inherit the same Filtered photo selection. Exact73-photo IDs/order match original Stage O F,LED benchmark,Qwen-Caption and Qwen-Image across all 15 restaurants. Qwen-Image receives pixels,not captions,at generation,but its chosen image set was obtained through the caption-based selection pipeline. A claim that this actual image experiment incurred zero BLIP/filter preparation would omit upstream selection work. An independently selected,caption-free direct-image pipeline has NOT been measured.

## Evidence and methodology

Native receipts and runner code are copied byte-for-byte in EVIDENCE/. SOURCE_MANIFEST.json maps each to its original repository path/hash. Existing native archive checkpoint members are preserved byte-for-byte in EVIDENCE/BLIP_CHECKPOINTS.zip,with5,143 hashes in BLIP_ARCHIVE_MEMBER_HASHES.json. The audit aggregates every saved attempt,not a sample. Final condition rows are an explicitly labeled condition==F projection from an existing45-row manifest. No source data is edited.

VERIFY_EFFICIENCY_AUDIT.py uses only Python standard-library reads,hash checks,JSON/CSV parsing and arithmetic. It recomputes all headline timings,peaks,photo bindings and CSV values. Optional `--repo <repository-root>` rehashes all recorded original source files. Existing model runner scripts are evidence only;the verifier never imports or executes them.

## Timing boundaries and recovered observations

| Observation | Seconds | What the measurement includes |
|---|---:|---|
| BLIP stage,5,143 photos | 48642.959 | Processor/model load,5,150 attempts,7 retries,per-photo processing/hash/checkpoint work up to last progress receipt |
| BLIP model load | 26.295 | Already inside BLIP elapsed;do not add again |
| All saved BLIP attempt timers | 48605.321 | CPU processor,GPU work/transfer,token copy/decode;not pure GPU kernels |
| CLIP/SBERT scoring,5,138 eligible captions | 178.529 | Load and weight checks,image/text scoring and embedding export;NOT final filtering/K selection |
| LED repeat1 / repeat2 | 97.434623 / 98.428058 | Each30 calls,after 2 warm-up calls;CUDA synchronized around generate |
| LED mean (sample SD,n=2) | 97.931341 (0.702465) | Average full15-restaurant generation pass,not60-call total |
| Qwen-Caption Review / Visual | 136.993810 / 49.850033 | 15 calls each;first 2 cold pilot calls included,not independently repeated |
| Qwen-Caption total | 186.843842 | 30 model.generate calls only |
| Qwen-Image NEW Visual only | 242.703157 | 15 calls with73 selected photos;no new Review calls |
| Qwen-Image constructed Review+Visual | 379.696967 | Earlier Review136.993810s plus new Visual242.703157s,separate sessions |

The three proposed figures97.931s,186.844s and379.697s are verified at3 decimal places. The third is an arithmetic standalone workload construction,not a fresh combined wall-clock observation. Actual incremental Stage P image work reused the existing Review outputs and generated only Visual. The observed image job elapsed248.988514s includes its separate load and other work;it is not379.697s.

BLIP elapsed includes 26.295s load already. Its saved5,150 attempt timers sum48,605.321s;the residual includes loading and surrounding image/checkpoint/validation work. Scoring178.529s also includes loading;there is no independent scorer-load timer. Threshold/redundancy/diversity/grouping/K selection runs afterwards in stage_o_frozen_pipeline_v1.py without a timer. File modification times are not used to invent durations.

Original accepted Stage O LED F completion recorded145.008172s on RTX3080,with different loop boundaries and Torch2.1.0+cu118. The97.931s RTX3090 benchmark uses the same weights/decoder/source but Torch2.9.1+cu128. It matches only22/30 original accepted output components (9 Review,13 Visual);its two repetitions match each other30/30. Original accepted summaries are preserved. Benchmark speed cannot be asserted to be a timed replay producing all identical evaluated outputs.

## Hardware,configuration and output accounting

| Stage/system | Hardware and precision | Decoder/output workload |
|---|---|---|
| BLIP-2 OPT2.7b COCO | RTX3080 plus CPU offload;FP16 vision/language,FP32 Q-former | Batch1,beam3,max_new_tokens64,5,150 attempts for5,143 photos,7 retries |
| CLIP ViT-B/32 + MPNet SBERT | RTX3080,FP32,TF32 disabled | 5,138 scored,5 basic failures,all 5,143 input records |
| LED-large F-S1000 | Same RTX3090 as Qwen,FP32 parameters + FP16 autocast,CPU threads1 | Batch1,beam4,max_length768,length_penalty1.0,2 warmups,2 measured30-call repeats;3,140 generated tokens/repeat |
| Qwen3.5-9B Caption | RTX3090,BF16,no quantization,thinking disabled | Batch1,greedy,Review max_new_tokens512 / Visual256;3,651 tokens total |
| Qwen3.5-9B Image Visual | Same RTX3090,BF16,official processor defaults,CPU threads32 | Greedy,max_new_tokens256;73 selected low-resolution source JPEGs,1,342 Visual tokens;Review2,638 tokens reused |

Qwen is the official Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a,not task-fine-tuned. Qwen uses Transformers5.22.1/Torch2.9.1+cu128,LED Transformers4.51.3/same Torch. Native fallback kernels are preserved in logs;this is not a comparison of optimized serving engines. LED includes finite-logit checks inside its generation wrapper. Modalities,attention/cache behavior,beam count,warm-up policy,CPU threads,input/output tokenization and output length differ. Tokens/s are model-native descriptive accounting,not cross-tokenizer equivalent work. No summary-quality conclusion follows.

BLIP processes5,143 source photos while Qwen-Image generation reads73 selected photos. The stage workloads are deliberately different. We do not scale measured5,143-photo BLIP time to73 photos or extrapolate Qwen to5,143 photos as an observed result. The frozen73 image set remains caption-selected even though its generation pass is direct-image.

## Memory definitions and sequential stages

| Scope | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| BLIP all saved attempts | 4.167538 | 4.292969 |
| Caption scorer | 0.986125 | 1.082031 |
| LED summary generation | 2.696645 | 3.044922 |
| Qwen-Caption summary generation | 18.535985 | 18.716797 |
| Qwen-Image NEW Visual only | 17.756796 | 17.820312 |
| Qwen-Image conceptual Review+Visual | 18.535985 | 18.716797 |

For sequentially unloaded stages,use maximum,not addition. Recorded stage-wise allocated maxima are LED-pipeline4.167538GiB,Qwen-Caption18.535985GiB,Qwen-Image including reused Review18.535985GiB. These are mixed-hardware observed allocator figures,not minimum required GPU capacities. The BLIP CPU offload trades latency and unmeasured peak CPU RAM for GPU memory;its 4 GiB weight budget is a configuration,not total peak consumption.

Separately,nvidia-smi whole-device sampled peaks are LED3,442MiB,Qwen-Caption19,490MiB,Qwen-Image Visual18,572MiB. Sampling target0.25s can miss transients,includes context/driver and spans job phases. Do not compare these directly with PyTorch allocated/reserved. Original RTX3080 receipt reports9.643616GiB capacity;RTX3090 reports25,298,141,184bytes capacity. Available/free/cgroup limits are capacity observations,not measured peak RAM requirements. Complete CPU/GPU memory capacity requirements are NOT MEASURED.

## Loading and setup are not generation

LED load9.278910s,Qwen-Caption11.532909s,Qwen-Image Visual5.847594s are separate. Qwen-Caption recorded258.961020s download,25.665555s weight verification and483.086792s whole-job elapsed. These actual cold-start costs are not assumed for every cached request. Stage P extension CPU setup receipt61.016495s excludes prior Qwen download and Stage O environment preparation;complete GPU provisioning/setup time is NOT MEASURED. No complete comparable monetary/energy accounting exists. We do not infer rental cost from VRAM or timer sums.

See EFFICIENCY_COMPARISON.md for boundary-specific claims,MISSING_MEASUREMENTS.md for gaps,and EFFICIENCY_MEASUREMENTS.csv for exact values,units,source paths,hashes and definitions.

## Recommended thesis wording

“On the recorded RTX3090 implementation with prepared inputs,LED F-S1000 required 97.931s on average across two full15-restaurant repetitions (SD 0.702s),compared with 186.844s for the single Qwen-Caption run. The constructed Qwen-Image workload was 379.697s,combining an earlier Review pass with a separately measured image Visual pass. LED's measured generation allocator peak was 2.697GiB versus 18.536GiB for Qwen-Caption. These generation-stage measurements exclude upstream captioning/selection and do not establish end-to-end latency,minimum hardware requirements or superior summary quality.”

Add:“Historical BLIP captioning plus caption scoring required48,821.488s for5,143 source photos before final filtering/selection. All systems inherited the Filtered73-photo selection,including the direct-image baseline. Complete end-to-end timing was not measured;cross-stage hardware/offload and unmeasured preprocessing prevent a controlled end-to-end claim.”
