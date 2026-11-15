"""Author a separate audit from existing files; no training/inference/network calls."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,shutil,subprocess,sys,tarfile,tempfile,zipfile
from stage_p_efficiency_audit_core_v1 import compute,verify,sha

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/stage_p/fair_efficiency_audit_20261106_v1'
PACKAGE=DEST/'PACKAGE'
O=ROOT/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE'
P=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1'
Q=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2/REVIEW_PACKAGE'
sources=[]

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def copy(src,name):
    dest=PACKAGE/'EVIDENCE'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    assert sha(src)==sha(dest)
    sources.append(dict(original_file=src.relative_to(ROOT).as_posix(),package_file=dest.relative_to(PACKAGE).as_posix(),sha256=sha(src),bytes=src.stat().st_size))

def main():
    assert not DEST.exists(),'Original audit must not be overwritten'
    (PACKAGE/'EVIDENCE').mkdir(parents=True)
    # Read hashes only; human ratings are not inputs to this computational audit.
    protected=[]
    for relative in ['outputs/stage_p/three_system_human_evaluation_live_20261106_v1/STATE/ratings.sqlite3','outputs/stage_p/completed_human_evaluation_handoff_20261106_v3/STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip','outputs/stage_p/v1_original_recovered_20261106_v1/STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip']:
        p=ROOT/relative;protected.append(dict(file=relative,sha256=sha(p)))
    stageo=O/'STAGE_O'
    for src,name in [(stageo/'PREPARED_CAPTION_JOB/generation_report.json','BLIP_GENERATION_REPORT.json'),(stageo/'PREPARED_CAPTION_JOB/model_load_report.json','BLIP_LOAD_REPORT.json'),(stageo/'PREPARED_CAPTION_JOB/environment_report.json','BLIP_ENVIRONMENT.json'),(stageo/'CAPTION_SCORES/scoring_report.json','CAPTION_SCORING_REPORT.json'),(stageo/'LED_OUTPUTS/F/COMPLETION.json','ORIGINAL_LED_F_COMPLETION.json')]:copy(src,name)
    for name in ['generate_stage_f_validation_captions.py','stage_f_score_validation.py','stage_o_frozen_pipeline_v1.py','apply_stage_f_frozen_filters.py','run_stage_o_allocated_inference_v1.py']:copy(O/'CODE'/name,'CODE/'+name)
    for src,name in [(P/'NATIVE/LED_BENCHMARK/BENCHMARK_RECORDS_62.json','LED_RECORDS_62.json'),(P/'NATIVE/LED_BENCHMARK/COMPLETION.json','LED_COMPLETION.json'),(P/'NATIVE/LED_BENCHMARK/RESTORATION.json','LED_RESTORATION.json'),(P/'NATIVE/LED_BENCHMARK/DEVICE_MEMORY_SAMPLES.json','LED_DEVICE_MEMORY_SAMPLES.json'),(P/'NATIVE/QWEN_IMAGE/COMPLETION.json','QIMAGE_COMPLETION.json'),(P/'NATIVE/QWEN_IMAGE/RESTORATION.json','QIMAGE_RESTORATION.json'),(P/'NATIVE/QWEN_IMAGE/DEVICE_MEMORY_SAMPLES.json','QIMAGE_DEVICE_MEMORY_SAMPLES.json'),(P/'NATIVE/EXPERIMENT_CONFIG.json','EXPERIMENT_CONFIG.json'),(P/'NATIVE/ASSET_SETUP_COMPLETE.json','ASSET_SETUP_COMPLETE.json'),(P/'NATIVE/IDENTITY.json','STAGE_P_GPU_IDENTITY.json'),(P/'NATIVE/PHOTO_PROVENANCE_73.json','PHOTO_PROVENANCE_73.json'),(P/'REPORTS/RESOURCE_SUMMARY.json','PRIOR_RESOURCE_SUMMARY.json'),(P/'REPORTS/LED_EQUIVALENCE_SUMMARY.json','LED_EQUIVALENCE_SUMMARY.json')]:copy(src,name)
    for name in ['benchmark_stage_p_led_3090_v1.py','run_stage_p_qwen_image_v1.py','stage_p_resource_common_v1.py']:copy(P/'NATIVE'/name,'CODE/'+name)
    for src,name in [(Q/'NATIVE/EXECUTION/COMPLETION_RECEIPT.json','QCAP_COMPLETION.json'),(Q/'NATIVE/EXECUTION/HARDWARE_AND_EXECUTION_RECEIPT.json','QCAP_HARDWARE.json'),(Q/'NATIVE/EXECUTION/DEVICE_MEMORY_SAMPLES.json','QCAP_DEVICE_MEMORY_SAMPLES.json'),(Q/'NATIVE/RUN_QWEN_V3.py','CODE/RUN_QWEN_V3.py')]:copy(src,name)
    for file in sorted((Q/'NATIVE/EXECUTION/CALLS').glob('*.json')):copy(file,'QCAP_CALLS/'+file.name)
    for file in sorted((P/'NATIVE/QWEN_IMAGE/CALLS').glob('*.json')):copy(file,'QIMAGE_CALLS/'+file.name)
    condition_source=stageo/'FINAL_CONDITION_INPUTS_45.json';raw=json.loads(condition_source.read_text(encoding='utf8'));selected=[r for r in raw if r['condition']=='F'];assert len(selected)==15
    # A projection is explicitly labeled; the original source file hash is bound.
    save(PACKAGE/'EVIDENCE/ORIGINAL_F_CONDITION_INPUTS.json',selected)
    sources.append(dict(original_file=condition_source.relative_to(ROOT).as_posix(),package_file='EVIDENCE/ORIGINAL_F_CONDITION_INPUTS.json',sha256=sha(condition_source),bytes=condition_source.stat().st_size,transformation='condition==F exact rows; source container not copied'))
    archive=ROOT/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/STAGE_O_NATIVE_COMPLETE_EVIDENCE.tar.gz'
    sources.append(dict(original_file=archive.relative_to(ROOT).as_posix(),package_file='EVIDENCE/BLIP_CHECKPOINTS.zip',sha256=sha(archive),bytes=archive.stat().st_size,transformation='Exact bytes of all native caption checkpoint JSON members repacked; no image pixels'))
    nested=PACKAGE/'EVIDENCE/BLIP_CHECKPOINTS.zip';members=[]
    with zipfile.ZipFile(nested,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z,tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if '/PREPARED_CAPTION_JOB/caption_checkpoints/' in member.name and member.name.endswith('.json'):
                raw=tar.extractfile(member).read();name=Path(member.name).name;z.writestr(name,raw)
                members.append(dict(original_tar_member=member.name,zip_member=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    assert len(members)==5143
    save(PACKAGE/'EVIDENCE/BLIP_ARCHIVE_MEMBER_HASHES.json',dict(source_archive_sha256=sha(archive),members=members))
    save(PACKAGE/'SOURCE_MANIFEST.json',dict(files=sources))
    shutil.copy2(ROOT/'scripts/stage_p_efficiency_audit_core_v1.py',PACKAGE/'stage_p_efficiency_audit_core_v1.py')
    shutil.copy2(Path(__file__),PACKAGE/'build_stage_p_efficiency_audit_v1.py')
    (PACKAGE/'VERIFY_EFFICIENCY_AUDIT.py').write_text('''"""Read-only verification; standard library only. No model imports or writes."""\nimport sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport argparse,json\nfrom stage_p_efficiency_audit_core_v1 import verify\np=argparse.ArgumentParser();p.add_argument('--repo',type=Path);a=p.parse_args()\nprint(json.dumps(verify(Path(__file__).resolve().parent,a.repo),indent=2))\n''',encoding='utf8')
    facts,measurements,native=compute(PACKAGE)
    save(PACKAGE/'VERIFIED_FACTS.json',facts)
    with (PACKAGE/'EFFICIENCY_MEASUREMENTS.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(measurements[0]));writer.writeheader();writer.writerows(measurements)
    l=facts['led_generation_seconds'];q=facts['qcap_generation_seconds'];i=facts['qimage_constructed_generation_seconds'];prep=facts['historical_measured_preparation_subtotal_seconds']
    (PACKAGE/'STAGE_P_EFFICIENCY_AUDIT.md').write_text(f'''# Stage P computational efficiency audit

Audit date:6 November2026 (Asia/Saigon). Read-only analysis of existing Stage O/P computational receipts. No GPU inference,model downloads,network/paid calls,caption regeneration,quality evaluation or human-rating statistical analysis was performed. Human-rating revisions are not inputs and no official results are changed.

## Defensible conclusion

With prepared inputs and the recorded policies,on the same RTX3090,LED F-S1000 had lower synchronized generation time and lower inference allocator peaks than the two Qwen configurations. This is a descriptive implementation/workload comparison,not an isolated architecture effect or an end-to-end speed result. A complete from-source latency or minimum GPU-capacity advantage is NOT ESTABLISHED.

The cost before summary generation is substantial:BLIP-2's5,143-photo stage took{facts['blip_elapsed_seconds']:,.3f}s ({facts['blip_elapsed_seconds']/3600:.5f}h) and CLIP/SBERT scoring took{facts['scoring_elapsed_seconds']:.3f}s. Their{prep:,.3f}s subtotal is historical RTX3080/CPU accounting,not full preprocessing and not a current3090 benchmark. The final filtering/diversity/K selection and review/source acquisition costs are NOT MEASURED.

Crucially,the frozen extension config states that ALL THREE systems inherit the same Filtered photo selection. Exact73-photo IDs/order match original Stage O F,LED benchmark,Qwen-Caption and Qwen-Image across all15 restaurants. Qwen-Image receives pixels,not captions,at generation,but its chosen image set was obtained through the caption-based selection pipeline. A claim that this actual image experiment incurred zero BLIP/filter preparation would omit upstream selection work. An independently selected,caption-free direct-image pipeline has NOT been measured.

## Evidence and methodology

Native receipts and runner code are copied byte-for-byte in EVIDENCE/. SOURCE_MANIFEST.json maps each to its original repository path/hash. Existing native archive checkpoint members are preserved byte-for-byte in EVIDENCE/BLIP_CHECKPOINTS.zip,with5,143 hashes in BLIP_ARCHIVE_MEMBER_HASHES.json. The audit aggregates every saved attempt,not a sample. Final condition rows are an explicitly labeled condition==F projection from an existing45-row manifest. No source data is edited.

VERIFY_EFFICIENCY_AUDIT.py uses only Python standard-library reads,hash checks,JSON/CSV parsing and arithmetic. It recomputes all headline timings,peaks,photo bindings and CSV values. Optional `--repo <repository-root>` rehashes all recorded original source files. Existing model runner scripts are evidence only;the verifier never imports or executes them.

## Timing boundaries and recovered observations

| Observation | Seconds | What the measurement includes |
|---|---:|---|
| BLIP stage,5,143 photos | {facts['blip_elapsed_seconds']:.3f} | Processor/model load,5,150 attempts,7 retries,per-photo processing/hash/checkpoint work up to last progress receipt |
| BLIP model load | {facts['blip_load_seconds']:.3f} | Already inside BLIP elapsed;do not add again |
| All saved BLIP attempt timers | {facts['blip_attempt_seconds_sum']:.3f} | CPU processor,GPU work/transfer,token copy/decode;not pure GPU kernels |
| CLIP/SBERT scoring,5,138 eligible captions | {facts['scoring_elapsed_seconds']:.3f} | Load and weight checks,image/text scoring and embedding export;NOT final filtering/K selection |
| LED repeat1 / repeat2 | {facts['led_repeat_seconds'][0]:.6f} / {facts['led_repeat_seconds'][1]:.6f} | Each30 calls,after2 warm-up calls;CUDA synchronized around generate |
| LED mean (sample SD,n=2) | {l:.6f} ({facts['led_sample_sd_seconds']:.6f}) | Average full15-restaurant generation pass,not60-call total |
| Qwen-Caption Review / Visual | {facts['qcap_review_seconds']:.6f} / {facts['qcap_visual_seconds']:.6f} | 15 calls each;first2 cold pilot calls included,not independently repeated |
| Qwen-Caption total | {q:.6f} | 30 model.generate calls only |
| Qwen-Image NEW Visual only | {facts['qimage_visual_seconds']:.6f} | 15 calls with73 selected photos;no new Review calls |
| Qwen-Image constructed Review+Visual | {i:.6f} | Earlier Review136.993810s plus new Visual242.703157s,separate sessions |

The three proposed figures97.931s,186.844s and379.697s are verified at3 decimal places. The third is an arithmetic standalone workload construction,not a fresh combined wall-clock observation. Actual incremental Stage P image work reused the existing Review outputs and generated only Visual. The observed image job elapsed248.988514s includes its separate load and other work;it is not379.697s.

BLIP elapsed includes26.295s load already. Its saved5,150 attempt timers sum48,605.321s;the residual includes loading and surrounding image/checkpoint/validation work. Scoring178.529s also includes loading;there is no independent scorer-load timer. Threshold/redundancy/diversity/grouping/K selection runs afterwards in stage_o_frozen_pipeline_v1.py without a timer. File modification times are not used to invent durations.

Original accepted Stage O LED F completion recorded145.008172s on RTX3080,with different loop boundaries and Torch2.1.0+cu118. The97.931s RTX3090 benchmark uses the same weights/decoder/source but Torch2.9.1+cu128. It matches only22/30 original accepted output components (9 Review,13 Visual);its two repetitions match each other30/30. Original accepted summaries are preserved. Benchmark speed cannot be asserted to be a timed replay producing all identical evaluated outputs.

## Hardware,configuration and output accounting

| Stage/system | Hardware and precision | Decoder/output workload |
|---|---|---|
| BLIP-2 OPT2.7b COCO | RTX3080 plus CPU offload;FP16 vision/language,FP32 Q-former | Batch1,beam3,max_new_tokens64,5,150 attempts for5,143 photos,7 retries |
| CLIP ViT-B/32 + MPNet SBERT | RTX3080,FP32,TF32 disabled | 5,138 scored,5 basic failures,all5,143 input records |
| LED-large F-S1000 | Same RTX3090 as Qwen,FP32 parameters + FP16 autocast,CPU threads1 | Batch1,beam4,max_length768,length_penalty1.0,2 warmups,2 measured30-call repeats;3,140 generated tokens/repeat |
| Qwen3.5-9B Caption | RTX3090,BF16,no quantization,thinking disabled | Batch1,greedy,Review max_new_tokens512 / Visual256;3,651 tokens total |
| Qwen3.5-9B Image Visual | Same RTX3090,BF16,official processor defaults,CPU threads32 | Greedy,max_new_tokens256;73 selected low-resolution source JPEGs,1,342 Visual tokens;Review2,638 tokens reused |

Qwen is the official Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a,not task-fine-tuned. Qwen uses Transformers5.22.1/Torch2.9.1+cu128,LED Transformers4.51.3/same Torch. Native fallback kernels are preserved in logs;this is not a comparison of optimized serving engines. LED includes finite-logit checks inside its generation wrapper. Modalities,attention/cache behavior,beam count,warm-up policy,CPU threads,input/output tokenization and output length differ. Tokens/s are model-native descriptive accounting,not cross-tokenizer equivalent work. No summary-quality conclusion follows.

BLIP processes5,143 source photos while Qwen-Image generation reads73 selected photos. The stage workloads are deliberately different. We do not scale measured5,143-photo BLIP time to73 photos or extrapolate Qwen to5,143 photos as an observed result. The frozen73 image set remains caption-selected even though its generation pass is direct-image.

## Memory definitions and sequential stages

| Scope | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| BLIP all saved attempts | {facts['blip_peak_allocated_GiB']:.6f} | {facts['blip_peak_reserved_GiB']:.6f} |
| Caption scorer | {json.loads((PACKAGE/'EVIDENCE/CAPTION_SCORING_REPORT.json').read_text())['peak_GPU_allocated_GiB']:.6f} | {json.loads((PACKAGE/'EVIDENCE/CAPTION_SCORING_REPORT.json').read_text())['peak_GPU_reserved_GiB']:.6f} |
| LED summary generation | {facts['led_peak_allocated_GiB']:.6f} | {facts['led_peak_reserved_GiB']:.6f} |
| Qwen-Caption summary generation | {facts['qcap_peak_allocated_GiB']:.6f} | {facts['qcap_peak_reserved_GiB']:.6f} |
| Qwen-Image NEW Visual only | {facts['qimage_visual_peak_allocated_GiB']:.6f} | {facts['qimage_visual_peak_reserved_GiB']:.6f} |
| Qwen-Image conceptual Review+Visual | {facts['qimage_constructed_peak_allocated_GiB']:.6f} | {facts['qcap_peak_reserved_GiB']:.6f} |

For sequentially unloaded stages,use maximum,not addition. Recorded stage-wise allocated maxima are LED-pipeline{facts['recorded_sequential_LED_stage_max_GiB']:.6f}GiB,Qwen-Caption{facts['recorded_sequential_QCAP_stage_max_GiB']:.6f}GiB,Qwen-Image including reused Review{facts['recorded_sequential_QIMAGE_stage_max_GiB']:.6f}GiB. These are mixed-hardware observed allocator figures,not minimum required GPU capacities. The BLIP CPU offload trades latency and unmeasured peak CPU RAM for GPU memory;its4GiB weight budget is a configuration,not total peak consumption.

Separately,nvidia-smi whole-device sampled peaks are LED3,442MiB,Qwen-Caption19,490MiB,Qwen-Image Visual18,572MiB. Sampling target0.25s can miss transients,includes context/driver and spans job phases. Do not compare these directly with PyTorch allocated/reserved. Original RTX3080 receipt reports9.643616GiB capacity;RTX3090 reports25,298,141,184bytes capacity. Available/free/cgroup limits are capacity observations,not measured peak RAM requirements. Complete CPU/GPU memory capacity requirements are NOT MEASURED.

## Loading and setup are not generation

LED load9.278910s,Qwen-Caption11.532909s,Qwen-Image Visual5.847594s are separate. Qwen-Caption recorded258.961020s download,25.665555s weight verification and483.086792s whole-job elapsed. These actual cold-start costs are not assumed for every cached request. Stage P extension CPU setup receipt61.016495s excludes prior Qwen download and Stage O environment preparation;complete GPU provisioning/setup time is NOT MEASURED. No complete comparable monetary/energy accounting exists. We do not infer rental cost from VRAM or timer sums.

See EFFICIENCY_COMPARISON.md for boundary-specific claims,MISSING_MEASUREMENTS.md for gaps,and EFFICIENCY_MEASUREMENTS.csv for exact values,units,source paths,hashes and definitions.

## Recommended thesis wording

“On the recorded RTX3090 implementation with prepared inputs,LED F-S1000 required97.931s on average across two full15-restaurant repetitions (SD0.702s),compared with186.844s for the single Qwen-Caption run. The constructed Qwen-Image workload was379.697s,combining an earlier Review pass with a separately measured image Visual pass. LED's measured generation allocator peak was2.697GiB versus18.536GiB for Qwen-Caption. These generation-stage measurements exclude upstream captioning/selection and do not establish end-to-end latency,minimum hardware requirements or superior summary quality.”

Add:“Historical BLIP captioning plus caption scoring required48,821.488s for5,143 source photos before final filtering/selection. All systems inherited the Filtered73-photo selection,including the direct-image baseline. Complete end-to-end timing was not measured;cross-stage hardware/offload and unmeasured preprocessing prevent a controlled end-to-end claim.”
''',encoding='utf8')
    table='\n'.join(f'| {r["hypothetical_full_cohort_requests"]} | {r["measured_prep_components_per_request"]:.3f} | {r["LED_known_components_per_request"]:.3f} | {r["QCAP_known_components_per_request"]:.3f} | {r["QIMAGE_inherited_selection_known_components_per_request"]:.3f} |' for r in facts['hypothetical_amortization'])
    (PACKAGE/'EFFICIENCY_COMPARISON.md').write_text(f'''# Comparisons with explicit boundaries

## A. Generation only with prepared inputs

| System | Full-cohort generation seconds | Calls | Generation allocated peak GiB | Definition |
|---|---:|---:|---:|---|
| LED F-S1000 | {l:.6f} | 30 per repeat,2 measured repeats | {facts['led_peak_allocated_GiB']:.6f} | Mean after2 warmups |
| Qwen-Caption | {q:.6f} | 30,one run | {facts['qcap_peak_allocated_GiB']:.6f} | Cold pilot included |
| Qwen-Image NEW Visual only | {facts['qimage_visual_seconds']:.6f} | 15 | {facts['qimage_visual_peak_allocated_GiB']:.6f} | Incremental image generation,Review reused |
| Qwen-Image constructed Review+Visual | {i:.6f} | 15 earlier Review+15 new Visual | {facts['qimage_constructed_peak_allocated_GiB']:.6f} | Arithmetic across sessions,not fresh combined timing |

Descriptive ratios:Qwen-Caption/LED={facts['qcap_over_led_generation_ratio']:.4f};constructed Qwen-Image/LED={facts['qimage_constructed_over_led_generation_ratio']:.4f}. These are not hardware-controlled architecture speedups or E2E ratios. Different output lengths/tokenizers,warmup,beam4 versus greedy,BF16 versus FP32+FP16,threads1 versus image32,modalities/framework versions and wrapper checks remain confounded. LED3140 versus Qwen-Caption3651 versus constructed image3980 native output tokens are not equal work.

## B. Complete source-to-summary pipeline

Measured shared parts M=BLIP48,642.959s + scorer178.529s =48,821.488s. BLIP26.295s load and scorer loading are already included;no double addition. Selection/review preprocessing/source collection/cold setup/remaining request CPU costs are missing.

| System | Recorded shared parts M + generation (seconds) | Complete E2E wall time |
|---|---:|---|
| LED F | {facts['LED_historical_measured_component_subtotal_seconds']:.6f} | NOT MEASURED |
| Qwen-Caption | {facts['QCAP_historical_measured_component_subtotal_seconds']:.6f} | NOT MEASURED |
| Qwen-Image with actual inherited Filtered selection | {facts['QIMAGE_historical_measured_component_subtotal_seconds']:.6f} | NOT MEASURED |
| Hypothetical caption-independent image selection | NOT MEASURED | NOT MEASURED |

These sums are accounting illustrations from separately measured historical RTX3080/CPU prep and RTX3090 generation. They are NOT observed job times,not monetary cost,not a matched end-to-end benchmark and not a rigorous simultaneous-wall lower bound (scheduling/overlap differ). Missing terms are never treated as zero. Separate model loads can be reported but adding them still cannot fill missing pipeline steps or transform these sums into E2E measurements.

All three actual variants inherit the same selected-photo provenance. Image has no caption text at inference,but removing shared BLIP/filter costs while keeping the same caption-selected73-photo set would describe an oracle/precomputed-selection boundary,not raw-photo end-to-end execution. The5,143-to73 scope mismatch is disclosed;neither per-photo extrapolation nor an unrun direct-selection replacement is made.

Measured original Stage O LED F loop was145.008172s on RTX3080;using it instead yields another historical component subtotal,not a controlled Qwen comparison. A same-device native all-GPU BLIP runtime is NOT MEASURED;we do not assume that13.51h CPU-offloaded execution represents an optimized3090 pipeline.

## C. Precomputed/reused captions and selected photos

Actual Stage P reused the prepared Stage O captions/selection;no new BLIP generation was performed for these inference runs. At that operational boundary,the observed summary-generation differences are relevant. Precomputed work has a prior cost even when marginal regeneration is zero.

For a hypothetical N requests over this SAME15-restaurant cohort with unchanged sources and cached caption scores/selection,known measured components per request are M/N + G(system). Full cost would be `(M + unknown_once)/N + G + unknown_request(system)`. N is a reuse scenario,not observed repeated-service data. The table omits unknown terms explicitly and cannot be called total latency:

| Hypothetical full-cohort requests N | M/N seconds | LED known components/request | Qwen-Caption known components/request | Qwen-Image inherited-selection known components/request |
|---:|---:|---:|---:|---:|
{table}

This is arithmetic amortization,not newly measured throughput. It assumes representative same workload repeats,cache validity,no source changes,sequential summary calls and no fresh model load per request. No cache-hit overhead,concurrency/load testing or amortization break-even is measured. An infinite-cache/sunk-preparation boundary approaches generation-only costs plus unknown request overhead,not complete pipeline latency.

If Qwen-Caption Review already exists,adding the actual image variant costs242.703157s new generation,not379.696967s. Its complete recorded Visual job elapsed248.988514s includes5.847594s model loading,0.330128s CPU image preprocessing,0.018900s transfer and0.002985s postprocessing plus other overhead. Existing Review/selection reuse must be named rather than recharged as a fresh measured job or silently assumed free from source.

## Claims

Supported:smaller recorded prepared-input generation times and allocator peaks for this LED implementation;resource tradeoffs under cached-source preparation;explicit descriptive known-component accounting.

Not supported:LED is faster end-to-end;LED's full pipeline requires only2.697GiB;Qwen-Image's actual image selection is caption-free;an exactly matched5,143-photo E2E advantage;energy/dollar efficiency;quality superiority from resource numbers. No confidence/significance claim is based on two LED repetitions versus one Qwen run.
''',encoding='utf8')
    (PACKAGE/'MISSING_MEASUREMENTS.md').write_text('''# Missing measurements and consequences

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

Actual sequential stage allocator maxima:yes,reported as maxima over measured phases rather than a sum or minimum required GPU capacity. LED's pipeline has a recorded BLIP max4.168GiB versus2.697GiB at the summary stage. CPU-offload RAM and driver/context costs remain unmeasured.

Full raw-source pipeline:no complete comparable measurement exists. Historical known components may be accounted transparently with gaps,but not reported as measured end-to-end totals. The source scope is5,143 BLIP photos versus73 Qwen-Image generation photos,and all inherit caption-selected image IDs.

Cached captions/selection:the existing generation numbers apply to that prepared-input boundary. Reuse scenarios can be shown with formulas,unknown overheads and assumptions. They do not establish end-to-end superiority or equal quality.

No missing measurement is repaired by rerunning a model in this audit. No new experiment or Stage P human-rating statistical analysis is started. Future controlled measurement would require separate authorization.
''',encoding='utf8')
    (PACKAGE/'README.md').write_text('''# Stage P fair efficiency audit

Start with STAGE_P_EFFICIENCY_AUDIT.md,then EFFICIENCY_COMPARISON.md and MISSING_MEASUREMENTS.md. EFFICIENCY_MEASUREMENTS.csv is a plain long-form machine-readable measurement table:exact numeric values/units,MEASURED versus DERIVED_FROM_MEASUREMENTS versus NOT MEASURED,hardware,precision,scope and source/hash definitions. Missing numeric cells are blank with an explicit NOT MEASURED status,never0.

Run `python -B VERIFY_EFFICIENCY_AUDIT.py` after extraction (Python3.11+,standard library only). Optionally add `--repo C:\\thesis_proposal\\yelp_dataset` to compare original repository source hashes. The verifier writes no files,imports no ML packages,does not use GPU/network and recomputes all headline numbers and CSV rows. EVIDENCE/CODE scripts are historical measurement definitions,NOT commands to run. No model weights,image pixels,human ratings/credentials or statistical results are included. Nested BLIP_CHECKPOINTS.zip preserves5,143 original native JSON checkpoints;the verifier reads it without extraction. No large model/dataset download is required.

SOURCE_MANIFEST.json records exact original source-file hashes. SOURCE_PRESERVATION.json reports audit-input and protected rating/archive hashes before/after;it contains hashes only,not rating data. Compact historical logs are provided as evidence. The report does not import exploratory rating revisions into official results. Outer ZIP integrity/extraction checks are in the separate DELIVERY_VALIDATION.json receipt.
''',encoding='utf8')
    (PACKAGE/'PROMPT_FOR_CHATGPT.txt').write_text('''Please independently review this Stage P read-only computational-efficiency audit before human-rating statistical analysis. Verify the underlying timers and all5143 native BLIP checkpoint records with the included standard-library verifier. Distinguish generation-only observations,historical measured-component sums,unknown complete-pipeline times and explicitly hypothetical caption-cache amortization. Check that BLIP/scorer loads are not double counted and that Qwen-Image379.697s is constructed from reused Review plus new Visual. All three actual variants inherit the caption-based Filtered73-photo selection;do not treat direct-image generation as caption-independent image selection. Judge the defensibility of proposed thesis claims,not which model is superior. No new GPU experiment,quality evaluation,training or rating analysis is authorized by this package.\n''',encoding='utf8')
    for r in sources:assert sha(ROOT/r['original_file'])==r['sha256'],r['original_file']
    for r in protected:assert sha(ROOT/r['file'])==r['sha256'],r['file']
    save(PACKAGE/'SOURCE_PRESERVATION.json',dict(status='PASS',original_source_files_checked=len(sources),original_source_hashes_unchanged=True,protected_rating_and_archive_hashes=protected,protected_hashes_unchanged=True,ratings_read_for_analysis=False,no_original_docs_or_logs_modified=True,GPU_calls=0,network_calls=0,human_rating_analysis=False,checked_at_utc=datetime.now(timezone.utc).isoformat()))
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    save(PACKAGE/'PACKAGE_MANIFEST.json',dict(files=[dict(file=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    validation=verify(PACKAGE,ROOT)
    output=DEST/'STAGE_P_FAIR_COMPUTATIONAL_EFFICIENCY_AUDIT.zip'
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(PACKAGE).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_p_efficiency_extract_') as td:
        extracted=Path(td)
        with zipfile.ZipFile(output) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
            for p in files:assert hashlib.sha256(z.read(p.relative_to(PACKAGE).as_posix())).hexdigest()==sha(p)
            z.extractall(extracted)
        result=subprocess.run([sys.executable,'-B',str(extracted/'VERIFY_EFFICIENCY_AUDIT.py')],capture_output=True,text=True,encoding='utf8',check=True)
        extracted_result=json.loads(result.stdout);assert extracted_result['status']=='PASS'
    output.with_suffix('.zip.sha256').write_text(sha(output)+'  '+output.name+'\n',encoding='ascii')
    receipt=dict(status='PASS',archive=output.name,bytes=output.stat().st_size,sha256=sha(output),members=len(files),zip_crc='PASS',every_member_sha256='PASS',repository_validation=validation,extracted_validation=extracted_result,original_sources_and_protected_ratings_unchanged=True,original_logs_or_settings_modified=False,human_statistics=False,GPU_calls=0,network_calls=0)
    save(DEST/'DELIVERY_VALIDATION.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
