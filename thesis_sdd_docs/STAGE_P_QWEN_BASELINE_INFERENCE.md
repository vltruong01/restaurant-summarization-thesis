# Stage P — actual Qwen3.5-9B baseline inference

Completed 6 November 2026 (Asia/Saigon); returned for independent review. Human evaluation has not been initiated or modified.

Official model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. All four weight-shard byte sizes and SHA-256 hashes were verified on the execution host against the frozen official metadata. BF16 parameters, batch size1, SDPA, greedy non-thinking (`enable_thinking=False`), existing frozen prompts and512/256 Review/Visual new-token caps. No quantization, fallback model, project fine-tuning, new reference generation, paid API or new GPU rental.

The supplied RTX3090 instance59313861/container239abaa00671/GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759 at77.48.24.250:48324 was verified before execution. The largest actual Review prompt (3,583 tokens, TA14622324_1001674) and its Visual pass completed as the compatibility pilot; these two outputs were reused, not regenerated. The remaining14 restaurants then completed. Native completion receipt:15 restaurants/30 calls; process exit0 with autorestart disabled.

All30 Qwen source strings match the native LED F evidence exactly:247 selected reviews and73 filtered-caption presentations, with reviews/captions separated and no image pixels,LED outputs,ratings or Teacher answers supplied to Qwen. Outputs and generation token IDs remain unedited; both components are combined verbatim. No frozen Stage O artifacts changed.

| Actual resource measure | Value |
|---|---:|
| Peak torch allocated memory |18.536 GiB|
| Peak torch reserved memory |18.717 GiB|
| Sampled whole-device peak |19,490 MiB /19.033 GiB|
| Model generation time,30 calls |186.844 seconds|
| Generated token IDs,including stop/special tokens |3,651|
| Aggregate tokens/s,including prefill |19.540|
| Weight download |258.961 seconds|
| Weight SHA verification |25.666 seconds|
| Model load |11.533 seconds|

Shared BLIP-2 caption generation/filtering is excluded from model-only time. Native LED F per-call resource context is preserved separately (RTX3080,FP32 parameters/FP16 autocast,four-beam decoder). Hardware,tokenizer,decoder and precision differences preclude controlled architecture-efficiency attribution. Optional optimized causal-convolution/gated-delta packages were absent; the actual native reference PyTorch fallback warnings are disclosed. No post-output kernel tuning occurred. Upstream Qwen training provenance is not independently audited; the absence of this project's Teacher supervision does not prove absence of any synthetic data upstream.

Validation: eight CPU contracts PASS; real one-restaurant GPU compatibility PASS; all30 output-token sequences replay to their stored raw/text outputs on CPU;0 empty/missing-header/thinking/cap flags and30 stop-token endings. Original remote archive and every member were hash-verified locally; final ZIP86 members reopened,extracted and every member verified. Known credential-marker scan PASS; no private database,credentials,session tokens,model weights or large photo archive in the review ZIP. Technical integrity does not certify semantic quality or select a winner.

An initial UUID representation check stopped before any weights were downloaded/model loaded. PyTorch omits the NVIDIA `GPU-` prefix. Strict normalization matched the identical underlying UUID; the original logs and fix receipt remain preserved. It was not an OOM or output-generation retry, and no model/source/prompt/numerical policy changed.

Deliverable: `outputs/stage_p/qwen_bf16_3090_execution_20261106_v2/STAGE_P_QWEN_COMPARISON_REVIEW.zip` (3,952,252 bytes).

SHA-256: `48953369c3604b5545f0f3c3b2d725b45474a2bb16b604586422f948c3804cad`.

The worker has exited. The newly supplied instance remains intact; no lifecycle action was taken. Stop for researcher/ChatGPT review before any human-evaluation integration or further inference.
