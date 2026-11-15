"""Export only terminal native evidence; no cache/weights/env/private app data."""
import datetime as dt,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];L=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2';P=L/'REVIEW_PACKAGE';REMOTE='/workspace/STAGE_P_QWEN35_9B_20261106_V2';HOST='root@77.48.24.250';PORT='48324'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def run(args,**kwargs):
    r=subprocess.run(args,capture_output=True,text=True,encoding='utf8',**kwargs);assert r.returncode==0,r.stdout+r.stderr;return r.stdout
def main():
    code=f'''import os,zipfile,json,hashlib,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert os.environ.get('CONTAINER_ID')=='59313861'
assert (r/'INFERENCE_JOB_EXIT_RECEIPT.json').exists(),'Model job not terminal: do not export yet'
assert not (r/'STAGE_P_NATIVE_EXPORT.zip').exists(),'Export already exists; do not overwrite'
allowed=[f for f in r.iterdir() if f.is_file() and f.suffix in ['.json','.jsonl','.py','.log','.txt']]
allowed += [f for name in ['EXECUTION','MODEL_PROVENANCE'] for f in (r/name).rglob('*') if f.is_file()]
files=[dict(file=f.relative_to(r).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(allowed)]
(r/'EXPORT_MEMBERS.json').write_text(json.dumps(files,indent=2)+'\\n');allowed.append(r/'EXPORT_MEMBERS.json')
target=r/'STAGE_P_NATIVE_EXPORT.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(allowed):z.write(f,f.relative_to(r).as_posix())
print(json.dumps(dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,members=len(allowed))))
'''
    receipt=json.loads(run(['ssh','-p',PORT,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],input=code));put(L/'REMOTE_EXPORT_RECEIPT.json',receipt)
    archive=L/'STAGE_P_NATIVE_EXPORT.zip';assert not archive.exists();run(['scp','-P',PORT,'-o','BatchMode=yes',HOST+':'+REMOTE+'/STAGE_P_NATIVE_EXPORT.zip',str(archive)]);assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    assert not P.exists();native=P/'NATIVE';native.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==receipt['members']
        for n in z.namelist():assert (native/n).resolve().is_relative_to(native.resolve())
        z.extractall(native)
    for r in read(native/'EXPORT_MEMBERS.json'):assert sha(native/r['file'])==r['sha256'] and (native/r['file']).stat().st_size==r['bytes']
    put(P/'VALIDATION/VERIFIED_DOWNLOAD.json',dict(status='PASS',**receipt,every_exported_member_hashed=True,original_remote_archive_verified=True))
    for n in ['INPUT_ALIGNMENT.json','CPU_CONTRACT_TESTS.json','OFFICIAL_MODEL_REVISION_CHECK.json','LED_MODEL_ONLY_RESOURCE_CONTEXT.json']:
        shutil.copy2(L/n,P/n)
    scripts=P/'SCRIPTS';scripts.mkdir()
    for n in ['stage_p_qwen_bf16_3090_v2.py','verify_stage_p_qwen_execution_v2.py','test_stage_p_qwen_bf16_3090_v2.py','stage_p_remote_control_v2.py','collect_stage_p_qwen_review_v2.py']:
        shutil.copy2(ROOT/'scripts'/n,scripts/n)
    spec=read(native/'FROZEN_SPEC.json');(scripts/'requirements-inference.txt').write_text('\n'.join(k+'=='+v for k,v in spec['software'].items())+'\nJinja2==3.1.6\n',encoding='utf8')
    complete=(native/'EXECUTION/COMPLETION_RECEIPT.json').exists()
    if complete:
        calls=[read(f) for f in (native/'EXECUTION/CALLS').glob('*.json')];rec=read(native/'EXECUTION/COMPLETION_RECEIPT.json');samples=read(native/'EXECUTION/DEVICE_MEMORY_SAMPLES.json')
        resource=dict(restaurants=15,calls=30,model_id=rec['model']['model_id'],model_revision=rec['model']['revision'],GPU='RTX3090 24GB',precision='BF16',batch_size=1,thinking=False,
            model_generation_seconds=rec['model_generation_seconds'],output_tokens=rec['output_tokens'],aggregate_tokens_per_second_including_prefill=rec['model_only_tokens_per_second'],download_seconds=rec['download_seconds'],weight_verification_seconds=rec['weight_verification_seconds'],load_seconds=rec['load_seconds'],
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rec['VRAM_phase_peaks']),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rec['VRAM_phase_peaks']),sampled_peak_whole_device_MiB=samples['peak_device_used_MiB'],sampler_errors=len(samples['errors']),shared_caption_preparation_excluded=True,
            tokens_include_special_and_stop_tokens=True,CPU_decode_and_file_writes_excluded_from_model_generation_seconds=True,pilot_cold_calls_included=True,per_call=[{k:r[k] for k in ['call_id','task','phase','input_tokens','output_tokens','generation_seconds','tokens_per_second_including_prefill','VRAM','flags']} for r in calls])
        put(P/'RESOURCE_MEASUREMENTS.json',resource)
        flags=[dict(call_id=r['call_id'],**r['flags']) for r in calls];put(P/'OUTPUT_FLAGS.json',flags)
        assembled=read(native/'EXECUTION/QWEN_OUTPUTS_15.json')
        (P/'QWEN_OUTPUTS_15_READABLE.txt').write_text('\n\n'.join('BUSINESS_ID: '+r['business_id']+'\n'+r['combined_text'] for r in assembled)+'\n',encoding='utf8')
        measured=['# Actual model-only resource measurements','','All times are measured on the supplied RTX3090. Native token counts include stop/special tokens; throughput includes prefill and autoregressive decoding. Download,weight hashing,loading and shared caption preparation are excluded from the per-call generation timers.','','| Call | Input tokens | Output tokens | Generation seconds | Tokens/s | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
        for r in sorted(calls,key=lambda r:r['call_id']):measured.append(f"| {r['call_id']} | {r['input_tokens']} | {r['output_tokens']} | {r['generation_seconds']:.3f} | {r['tokens_per_second_including_prefill']:.3f} | {r['VRAM']['peak_allocated_bytes']/1024**3:.3f} |")
        measured+=['','The first two calls are a cold compatibility pilot and remain included. Other calls are not selectively warmed up or regenerated. torch allocated/reserved peaks measure this process; sampled device-used memory includes driver/context and can miss transient peaks. Native logs preserve kernel fallbacks and errors. Original LED F timers are in LED_MODEL_ONLY_RESOURCE_CONTEXT.json; the different GPU,decoder,precision policy and tokenizer prevent controlled efficiency attribution.']
        (P/'RESOURCE_MEASUREMENTS.md').write_text('\n'.join(measured)+'\n',encoding='utf8')
        peak=resource['peak_allocated_bytes']/1024**3;reserved=resource['peak_reserved_bytes']/1024**3
        lines=['# Stage P Qwen3.5-9B baseline inference review','',f"Completed15 restaurants/30 isolated calls on RTX3090 24GB, BF16,batch1,enable_thinking=False. Revision `{rec['model']['revision']}`. The largest actual Review prompt (3,583 tokens; TA14622324_1001674) was tested first with its Visual pass. The same two pilot outputs are retained in the final15; none regenerated.",'',f"Measured torch peak allocated {peak:.3f} GiB; peak reserved {reserved:.3f} GiB. Sampled whole-device peak {resource['sampled_peak_whole_device_MiB']} MiB (sampling can miss transients and includes CUDA context). Model generation {rec['model_generation_seconds']:.3f} s, {rec['output_tokens']} generated token IDs, aggregate {rec['model_only_tokens_per_second']:.3f} tokens/s including prefill; special/stop tokens count. Download,hash verification,model load and wall runtime are separately recorded.",'','## Evidence and outputs','NATIVE/INFERENCE_INPUTS.jsonl preserves exact LED F evidence bytes:247 selected reviews,73 filtered-caption presentations,15 restaurants. Review uses reviews/metadata only; Visual uses filtered captions/metadata only. No original image pixels,LED predictions,ratings,reference summaries or Teacher targets entered model requests. NATIVE/EXECUTION/CALLS has original token IDs,prompt/input/source hashes,raw decoded outputs,per-call timing/VRAM and flags. QWEN_OUTPUTS_15.json assembles both components verbatim. OUTPUT_FLAGS.json retains caps,heading and thinking flags; no repair,reranking or selective regeneration.','', '## Fairness and resources','Qwen is an external generally pretrained/post-trained instruction model without this project\'s fine-tuning or GPT Teacher supervision; LED is task-fine-tuned. The comparison does not isolate architecture,size or pretraining. Fixed Qwen greedy decoding and512/256 new-token caps differ from LED4-beam768-total-token policies and tokenizers. Prompt length targets are fixed80–130 Review and40–80 Visual words,not posthoc output edits. LED_MODEL_ONLY_RESOURCE_CONTEXT.json preserves original30 F component timers/token counts on RTX3080; cross-GPU/software/decoder/tokenizer ratios are not controlled efficiency claims. Shared BLIP-2 caption generation/filtering is excluded from both model-only summaries; Qwen loading/downloading is reported separately.','', '## Validation and use','Run `python SCRIPTS/verify_stage_p_qwen_execution_v2.py --package .` for stdlib integrity/bindings; add `--decode-tokens` with the pinned tokenizer-compatible transformers installed for independent token-to-text replay. No GPU/model weights are needed to inspect this package. Inference runner and settings are preserved; rerunning requires separately authorized hardware identity and downloading~19.3GB pinned weights, not bundled here. Validation reports distinguish actual GPU checks from prior CPU contracts.','', 'Human evaluation was not initiated or modified. Existing Stage O ratings and frozen LED outputs/captions remain unchanged. Review this package before any later research stage. The instance is left intact; the current request does not authorize destroying this newly supplied instance.']
    else:
        failure=read(native/'EXECUTION/FAILURE_RECEIPT.json') if (native/'EXECUTION/FAILURE_RECEIPT.json').exists() else read(native/'SETUP_FAILURE_RECEIPT.json')
        put(P/'TERMINAL_FAILURE.json',failure);lines=['# Stage P: stopped before completed inference','','The run did not complete. Original logs,partial outputs and actual failure evidence are preserved. See TERMINAL_FAILURE.json and NATIVE. No automatic retry,model fallback,quantization,new GPU rental or human evaluation. An A100 estimate,if relevant,is unbenchmarked and requires separate researcher provision/authorization.']
    (P/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Rehash original protected inputs; no stage-O application or rating database is copied.
    protected=[]
    for rel in ['outputs/stage_p/qwen_comparison_v1/FROZEN_SPEC.json','outputs/stage_p/qwen_comparison_v1/INFERENCE_INPUTS.jsonl','outputs/stage_p/qwen_comparison_v1/PROMPTS.json','outputs/stage_o/human_collection_complete_20261106_v1/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json','outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE/STAGE_O/TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json']:
        f=ROOT/rel;protected.append(dict(file=rel,sha256=sha(f)))
    assert protected[0]['sha256']==read(native/'EXECUTION_PLAN.json')['spec_sha256'];assert protected[1]['sha256']==spec['input_sha256'];assert protected[2]['sha256']==spec['prompts_sha256'];assert protected[3]['sha256']=='d89c2fc9ed7c228139a1594c9ff7b1f7b2ffda2dfbe18bff435a7ccb6a13df7f'
    before={r['file']:r['sha256'] for r in read(L/'PROTECTED_ORIGINALS_BEFORE_COLLECTION.json')}
    assert all(before[r['file']]==r['sha256'] for r in protected)
    put(P/'VALIDATION/PROTECTED_ORIGINAL_HASHES.json',protected)
    with (P/'README.md').open('a',encoding='utf8') as f:f.write('\n## Preserved operational provenance\nAn initial wrapper stopped before any weights were downloaded/model loaded because PyTorch reports a UUID without NVIDIA\'s GPU- prefix. Strict UUID normalization fixed the representation check after matching the same device; this was not an OOM or a generated-output retry. Original setup/exit logs,UUID_FORMAT_FIX_RECEIPT and the effective RUN_QWEN_V3.py are preserved. No source,prompt,decoder,model or precision setting changed. The old preparation FROZEN_SPEC retains historical fallback metadata, but this execution explicitly permits only Qwen3.5-9B; no fallback ran.\n\nThe software uses SDPA and native reference PyTorch implementations for causal convolution/gated delta kernels because optional causal_conv1d/flash-linear-attention packages were not installed. Warnings are preserved; performance should not be presented as an optimized Qwen limit or a controlled architecture comparison. No performance-driven kernel/configuration change was made after seeing outputs. No project-specific Qwen fine-tuning or use of our GPT Teacher targets occurred; the full provenance of upstream Qwen pretraining/post-training is not independently audited here.\n')
    def manifest():put(P/'PACKAGE_MANIFEST.json',dict(files=[dict(file=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f!=P/'PACKAGE_MANIFEST.json']))
    manifest();verify=scripts/'verify_stage_p_qwen_execution_v2.py'
    output=run([str(ROOT/'.venv-stage-p-preflight/Scripts/python.exe'),'-X','utf8','-B',str(verify),'--package',str(P),'--decode-tokens'])
    put(P/'VALIDATION/NATIVE_RECORD_VERIFICATION.json',json.loads(output));manifest()
    artifact=L/'STAGE_P_QWEN_COMPARISON_REVIEW.zip'
    with zipfile.ZipFile(artifact,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(P).as_posix())
    extracted=L/'FINAL_EXTRACTED';assert not extracted.exists()
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None;z.extractall(extracted)
        for f in P.rglob('*'):
            if f.is_file():assert sha(f)==sha(extracted/f.relative_to(P))
    verification=json.loads(run([str(ROOT/'.venv-stage-p-preflight/Scripts/python.exe'),'-X','utf8','-B',str(extracted/'SCRIPTS/verify_stage_p_qwen_execution_v2.py'),'--package',str(extracted),'--decode-tokens']))
    put(L/'EXTRACTED_ARCHIVE_VERIFICATION.json',verification);h=sha(artifact);(L/(artifact.name+'.sha256')).write_text(h+'  '+artifact.name+'\n',encoding='ascii')
    receipt=dict(status='PASS_COMPLETE_REVIEW' if complete else 'STOPPED_FAILURE_EVIDENCE_PRESERVED',artifact=artifact.name,bytes=artifact.stat().st_size,sha256=h,archive_reopened_extracted_all_members_hashed=True,verification=verification,at_utc=dt.datetime.now(dt.timezone.utc).isoformat());put(L/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
