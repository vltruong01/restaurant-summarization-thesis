# Missing measurements and consequences

| Missing or unmatched evidence | Status | Consequence |
|---|---|---|
| Source reviews/photos download,cohort eligibility/image-integrity preparation and representative-review selection wall times | NOT MEASURED | Source-to-summary totals unavailable for all systems |
| Final relevance/semantic redundancy/grouping/diversity/K selection and source/token assembly duration after scoring | NOT MEASURED |178.529s scorer report cannot be labeled full filtering time |
| Separate CLIP/SBERT load duration | NOT MEASURED | Loading is included in scorer elapsed but cannot be independently broken out |
| BLIP/CLIP/SBERT full cold download/hash/environment/GPU provisioning times | NOT MEASURED | Existing weight/download receipts confirm bytes,not complete comparable setup duration |
| BLIP peak CPU RSS/RAM under CPU offload and whole-device sampling | NOT MEASURED | GPU allocator max is known,but minimum full-system memory not established;budgets/cgroup limits are not usage |
| LED/Qwen-Caption request CPU preprocessing,tokenization,transfers,decode/file work on a complete consistent boundary | NOT MEASURED | Synchronized generation timers are not user-visible end-to-end latency |
| Qwen-Image full source processing + image subset selection without caption-derived ranking | NOT MEASURED | Actual frozen image experiment inherits caption selection;cannot claim a caption-independent fresh pipeline |
| One same-GPU/same-source-count fresh pipeline measurement for each system | NOT MEASURED | Cannot infer an exactly matched complete-pipeline speed ratio from historical mixed-hardware component sums |
| Fresh standalone combined Qwen-Image Review+Visual session | NOT MEASURED |379.697s is a constructed sum across sessions;actual new Visual work242.703s |
| Repeated Qwen timing,matched warmup/threads/decoder/length and identical benchmark/evaluated LED outputs | NOT MEASURED / NOT MATCHED | Comparisons are descriptive;no causal architecture or performance-significance claim |
| Whole-pipeline minimum VRAM,driver/workspace/RAM/power/energy and provider dollar accounting | NOT MEASURED | No exact hardware-capacity,energy or monetary-efficiency claim |
| Multi-request cache/load/concurrency behavior | NOT MEASURED | N-request amortization is hypothetical arithmetic,not observed serving throughput |

## Can a fair comparison be made now?

Generation-only:yes,as a clearly labeled descriptive comparison of these recorded implementations/workloads,with the listed configuration,modality,warmup and output-length confounders. A controlled architecture-only comparison is not established.

Actual sequential stage allocator maxima:yes,reported as maxima over measured phases rather than a sum or minimum required GPU capacity. LED's pipeline has a recorded BLIP max 4.168GiB versus 2.697GiB at the summary stage. CPU-offload RAM and driver/context costs remain unmeasured.

Full raw-source pipeline:no complete comparable measurement exists. Historical known components may be accounted transparently with gaps,but not reported as measured end-to-end totals. The source scope is5,143 BLIP photos versus73 Qwen-Image generation photos,and all inherit caption-selected image IDs.

Cached captions/selection:the existing generation numbers apply to that prepared-input boundary. Reuse scenarios can be shown with formulas,unknown overheads and assumptions. They do not establish end-to-end superiority or equal quality.

No missing measurement is repaired by rerunning a model in this audit. No new experiment or Stage P human-rating statistical analysis is started. Future controlled measurement would require separate authorization.
