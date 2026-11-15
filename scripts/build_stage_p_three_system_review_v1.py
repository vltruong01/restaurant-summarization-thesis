"""Build and verify the finite review package from completed evidence, no inference."""
import sys,json,hashlib,shutil,zipfile,subprocess,re,datetime
from pathlib import Path
from analyze_stage_p_three_system_v1 import analyze,plot,read,put,md
from audit_stage_p_three_system_package_v1 import audit
ROOT=Path(__file__).resolve().parents[1];L=ROOT/'outputs/stage_p/three_system_comparison_20261106_v1';P=L/'REVIEW_PACKAGE'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not P.exists();P.mkdir();shutil.copytree(L/'NATIVE',P/'NATIVE')
    for name in ['METHODOLOGY.md','PROPOSED_HUMAN_COMPARISON.md','PRELIMINARY_PHOTO_OBSERVATIONS.json']:
        shutil.copy2(L/name,P/name)
    shutil.copytree(L/'VISUAL_REVIEW_CONTACT_SHEETS',P/'PHOTO_PREVIEWS')
    code=P/'CODE';code.mkdir()
    for name in ['analyze_stage_p_three_system_v1.py','audit_stage_p_three_system_package_v1.py','benchmark_stage_p_led_3090_v1.py','run_stage_p_qwen_image_v1.py','stage_p_resource_common_v1.py','verify_stage_p_extension_cpu_v1.py','stage_p_extension_job_v1.py','stage_p_extension_setup_v1.py','prepare_stage_p_three_system_v1.py','collect_stage_p_extension_v1.py','build_stage_p_three_system_review_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,code/name)
    # Small provenance receipts from the already completed caption run. No model weights.
    old=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2/REVIEW_PACKAGE/NATIVE';dest=P/'QWEN_CAPTION_PRIOR_RECEIPTS';dest.mkdir()
    for name in ['OFFICIAL_MODEL_REVISION_CHECK.json','ENVIRONMENT_FREEZE.txt','REMOTE_IDENTITY.json','EXECUTION/WEIGHT_VERIFICATION.json','EXECUTION/MODEL_LOAD_REPORT.json','EXECUTION/COMPLETION_RECEIPT.json','RUN_QWEN_V3.py']:
        f=old/name
        if f.exists():shutil.copy2(f,dest/f.name)
    prov=P/'PROVENANCE';prov.mkdir();shutil.copy2(L/'PROTECTED_ORIGINALS.json',prov/'PROTECTED_ORIGINALS.json');shutil.copy2(L/'NATIVE_LOCAL_DOWNLOAD_VERIFICATION.json',prov/'NATIVE_LOCAL_DOWNLOAD_VERIFICATION.json');shutil.copy2(L/'PAYLOAD/PROVENANCE/PHOTO_EXIF_AUDIT.json',prov/'PHOTO_EXIF_AUDIT.json');shutil.copy2(L/'POST_EXECUTION_IDENTITY_AND_WEIGHTS.json',prov/'POST_EXECUTION_IDENTITY_AND_WEIGHTS.json')
    protected=read(L/'PROTECTED_ORIGINALS.json');unchanged=[]
    for row in protected:
        f=ROOT/row['original'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],row['original'];unchanged.append(dict(file=row['original'],sha256=row['sha256'],status='PASS_UNCHANGED'))
    put(prov/'PROTECTED_ORIGINALS_RECHECK.json',dict(status='PASS',files=len(unchanged),checks=unchanged,scope='The129 bound original files include frozen LED backbone/adapter, original LED/Qwen outputs, caption pools, Stage O final90-rating export, source records and73 selected original JPEGs. This is a file-integrity check, not independent participant/authorship authentication.'))
    n=P/'NATIVE';orig={(x['business_id'],x['task']):x for x in read(n/'INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')};qc={x['business_id']:x for x in read(n/'INPUTS/QWEN_CAPTION_OUTPUTS_15.json')};qi={x['business_id']:x for x in read(n/'QWEN_IMAGE/QWEN_IMAGE_OUTPUTS_15.json')};ip={x['business_id']:x for x in read(n/'IMAGE_PLAN_15.json')};assembled=[];readable=[]
    for bid in qc:
        readable.append('BUSINESS_ID: '+bid)
        for system,a in [('LED F original',dict(review_text=orig[bid,'review']['text'],visual_text=orig[bid,'visual']['text'])),('Qwen-Caption',qc[bid]),('Qwen-Image',qi[bid])]:
            row=dict(business_id=bid,system=system,review_text=a['review_text'],visual_text=a['visual_text'],selected_photo_ids=ip[bid]['selected_photo_ids'],photos=['NATIVE/'+x['file'] for x in ip[bid]['photos']],review_text_sha256=hashlib.sha256(a['review_text'].encode()).hexdigest(),visual_text_sha256=hashlib.sha256(a['visual_text'].encode()).hexdigest());assembled.append(row);readable.extend(['\nSYSTEM: '+system,a['review_text'],'',a['visual_text'],''])
    put(P/'THREE_SYSTEM_PREDICTIONS_45.json',assembled);(P/'ALL15_OUTPUTS_READABLE.txt').write_text('\n'.join(readable),encoding='utf8')
    count=analyze(P,P/'REPORTS');plot(P,P/'REPORTS/FIGURES');summary=read(P/'REPORTS/RESOURCE_SUMMARY.json')
    # Include all15 exact preliminary source spans, separate from numerical diagnostics.
    obs=read(P/'PRELIMINARY_PHOTO_OBSERVATIONS.json');lines=['# Preliminary source-photo review','',obs['assessment_type'],'',obs['inspection_scope'],'','No independent confirmed factual-error count or human quality ranking is supplied.','']
    for x in obs['observations']:lines.extend(['## '+x['business_id'],'','Output span: '+x['output_span'],'','Sources: '+', '.join(x['photo_ids']),'',x['assessment'],''])
    (P/'REPORTS/PRELIMINARY_PHOTO_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')
    # Keep memory definitions and full-session peaks explicit, including native sampler limitations.
    lm=read(n/'LED_BENCHMARK/DEVICE_MEMORY_SAMPLES.json');im=read(n/'QWEN_IMAGE/DEVICE_MEMORY_SAMPLES.json');qres=read(n/'INPUTS/QWEN_CAPTION_RESOURCES.json');lcomp=read(n/'LED_BENCHMARK/COMPLETION.json');icomp=read(n/'QWEN_IMAGE/COMPLETION.json')
    memory=[dict(session='LED F (includes warmup and2 repetitions)',peak_allocated_GiB=max(x['peak_allocated_bytes'] for x in lcomp['VRAM_phase_peaks'])/1024**3,peak_reserved_GiB=max(x['peak_reserved_bytes'] for x in lcomp['VRAM_phase_peaks'])/1024**3,sampled_whole_device_MiB=lm['peak_device_used_MiB']),dict(session='Qwen-Caption',peak_allocated_GiB=qres['peak_allocated_bytes']/1024**3,peak_reserved_GiB=qres['peak_reserved_bytes']/1024**3,sampled_whole_device_MiB=qres['sampled_peak_whole_device_MiB']),dict(session='Qwen-Image Visual',peak_allocated_GiB=max(x['peak_allocated_bytes'] for x in icomp['VRAM_phase_peaks'])/1024**3,peak_reserved_GiB=max(x['peak_reserved_bytes'] for x in icomp['VRAM_phase_peaks'])/1024**3,sampled_whole_device_MiB=im['peak_device_used_MiB'])]
    put(P/'REPORTS/SESSION_MEMORY.json',memory);(P/'REPORTS/SESSION_MEMORY.md').write_text('# Full-session memory\n\n'+md(memory)+'\n\nAllocator allocated/reserved and sampled whole-device measurements are different definitions; never add them. Sampling every approximately0.25 seconds may miss transients. Native receipts include separate load peaks. No OOM occurred.\n',encoding='utf8')
    pipeline=[dict(system=s,historical_shared_BLIP_seconds=48642.959,historical_shared_scoring_seconds=178.529,current_model_generation_seconds=t,new_caption_preparation_seconds=0,full_pipeline_wallclock='NOT_MEASURED',selection_provenance='Frozen BLIP/filter selected photos inherited') for s,t in [('LED F (repeat mean)',summary['LED_variability']['total_generation_seconds_mean']),('Qwen-Caption',186.84384227218106),('Qwen-Image (constructed)',379.6969671170227)]]
    put(P/'REPORTS/PIPELINE_ACCOUNTING.json',pipeline);(P/'REPORTS/PIPELINE_ACCOUNTING.md').write_text('# Pipeline component accounting\n\n'+md(pipeline)+'\n\nThe historical columns describe the same one-time shared assets, not three incurred preparations. They are not additive same-hardware end-to-end observations. BLIP time already includes load, pre/postprocessing and QC/writes. Data acquisition, selection and other CPU stages are incompletely timed; no fabricated full-pipeline wallclock or currency estimate is supplied. For this extension, no new BLIP/scoring work was run. Image inference bypasses caption text but does not remove historical selection dependence. Image full-system generation is constructed from two sessions.\n',encoding='utf8')
    b=summary['LED_variability'];readme=f'''# Stage P three-system comparison — independent review

Completed6 November2026 (Asia/Saigon). All15 Qwen-Image Visual passes completed on the supplied RTX3090 after a successful five-photo pilot. No A100, model/precision substitution, training or human-rating collection occurred. The instance remains intact; workers exited.

## Observed resources

{md(summary['comparison'])}

LED generation mean **{b['total_generation_seconds_mean']:.3f}s**, sample SD **{b['sample_SD_seconds']:.3f}s**, two warmed30-call repetitions. Qwen-Caption30 calls **186.844s**; Qwen-Image15 new Visual calls **242.703s**. Combined Image system **379.697s** is constructed using earlier Review calls, not a fresh full-system benchmark. Model loading, whole-device memory and CPU overhead are separate in REPORTS. Timing includes prefill, excludes CPU preparation/decoding; comparisons have tokenizer, output-length, beam/greedy and runtime limits.

## Important findings and limits

- LED reruns match each other30/30, but match only22/30 original Stage O component texts/tokens per repetition (9 Review,13 Visual). Every difference is preserved. Hardware and Torch changed; no unique cause established. Original LED predictions and90 human ratings were preserved.
- Qwen-Image uses the exact73 downloaded JPEGs selected for F, including low-resolution renditions, in the same photo order. Review reused verbatim15/15. CPU pixel/grid/input-ID and output-token replays passed. All15 new outputs have valid headings/EOS and no empty, thinking or length-cap flags. This does not certify factual quality.
- The direct-image run excludes caption text, but inherits BLIP/filter-based photo selection. Historical shared preprocessing subtotal48,821.488s is mixed RTX3080/CPU component accounting, not a current3090 end-to-end measurement or monetary comparison.
- Model-assisted preliminary source-photo observations cover all15 restaurants and are explicitly uncertain where needed. No Qwen human scores or winning model are invented. The later human proposal adds60 records and reuses30 original LED F ratings, with separate-session and prior-exposure limitations.

## Files

- `THREE_SYSTEM_PREDICTIONS_45.json` and `ALL15_OUTPUTS_READABLE.txt`: original accepted LED F, Qwen-Caption and Qwen-Image texts. LED benchmark reruns are stored separately.
- `REPORTS/`: resource tables per call/restaurant, two-repeat variability, all LED differences, diagnostics, preliminary observations and figure.
- `NATIVE/`: exact configurations, generation token IDs, input sources/token IDs, image grid/pixel hashes,73 JPEGs, CPU replay results, native logs/receipts, small tokenizer/processor files. Large model weights/venvs are excluded; exact hashes and pinned revisions are retained.
- `QWEN_CAPTION_PRIOR_RECEIPTS/` and `PROVENANCE/`: earlier model provenance, verified native download and recheck of129 original bound files. No passwords, session tokens, private databases or human personal identity documents.
- `METHODOLOGY.md`, `PROPOSED_HUMAN_COMPARISON.md`, `VALIDATION_REPORT.md`, `THESIS_STATUS/` and `CODE/` explain methods, stop boundary and reproducibility.

## Lightweight reproduction

Unzip into a writable directory. Python3.11+ standard library suffices:

```text
python CODE/audit_stage_p_three_system_package_v1.py --root .
```

It hashes every member, checks bindings/assembly and regenerates numerical CSV/JSON/Markdown to a temporary directory for byte comparison. No GPU/model/download is required. To regenerate the figure, use Python plus matplotlib and run `python CODE/analyze_stage_p_three_system_v1.py --root . --out RECOMPUTED --plots` (the figure reads the bundled frozen resource summary). Do not modify REPORTS when verifying the package manifest.

CPU tokenizer/pixel replay already ran on the GPU host with CUDA disabled; actual receipts are included. Optional independent replay requires pinned model-free tokenizer/processor dependencies (Torch/torchvision/Transformers/Pillow), using bundled small assets. The historical native replay script uses the recorded remote snapshot path; adapt that path to NATIVE/PROCESSOR_FILES for relocation. This does not need model weights. GPU execution scripts are provenance for authorized research reproduction, not part of lightweight viewing/audit; do not launch them as a review step.

Package manifest excludes itself to avoid recursive hashing. External ZIP checksum and actual reopen/extract/audit receipt accompany the ZIP. STOP for researcher/ChatGPT review; no human evaluation launched.
'''
    (P/'README.md').write_text(readme,encoding='utf8')
    status=P/'THESIS_STATUS';status.mkdir()
    for name in ['STAGE_P_THREE_SYSTEM_EXTENSION.md']:
        shutil.copy2(ROOT/'thesis_sdd_docs'/name,status/name)
    result=audit(P,False);put(P/'VALIDATION_RESULTS.json',result)
    val='# Validation results\n\n'+md(result['checks'])+'\n\nAdditional checks:129 original protected files rehashed unchanged; both native GPU jobs exit0; first five-photo pilot PASS; all264 native scientific-member hashes verified locally. Two CPU replay receipts cover62 LED calls and15 image calls. All45 assembled records preserve source texts. No semantic quality certification or authentication of human participants is claimed.\n\nOriginal-output equivalence is **PARTIAL**,22/30 per repetition; it is not an all-match PASS. No OOM; no A100 requirement arose. Frozen scientific outputs were not replaced. Archive reopen/extract/every-member and extracted-script audit results are in the external delivery receipt to avoid a circular package hash.\n'
    (P/'VALIDATION_REPORT.md').write_text(val,encoding='utf8')
    # Credential-pattern scan is scoped to actual secrets, not harmless variable names/provenance strings.
    patterns=[rb'hf_[A-Za-z0-9]{24,}',rb'sk-[A-Za-z0-9_-]{24,}',rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',rb'Bearer [A-Za-z0-9._-]{24,}']
    for f in P.rglob('*'):
        if f.is_file() and f.suffix not in ['.jpg','.png']:
            b=f.read_bytes()
            assert not any(re.search(p,b) for p in patterns),'Potential secret: '+str(f)
        assert not (f.is_file() and f.suffix in ['.safetensors','.bin','.sqlite','.db']),f
    put(P/'PRIVACY_CHECK.json',dict(status='PASS_PATTERN_AND_FILETYPE_SCAN',patterns='HF/API/bearer/private-key signatures',excluded='Weights, private DBs, session stores and account credentials',limitation='Pattern scan is not a formal proof against every possible encoded secret.'))
    manifest=[dict(file=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file()];put(P/'PACKAGE_MANIFEST.json',manifest)
    result=audit(P,True);zipfile_path=L/'STAGE_P_THREE_SYSTEM_COMPARISON_REVIEW.zip'
    with zipfile.ZipFile(zipfile_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(P).as_posix())
    extracted=L/'DELIVERY_EXTRACTED';assert not extracted.exists()
    with zipfile.ZipFile(zipfile_path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
        for name in z.namelist():assert (extracted/name).resolve().is_relative_to(extracted.resolve())
        z.extractall(extracted)
    # Actually execute the copied audit from the extracted archive; no imports from repository scripts.
    p=subprocess.run([sys.executable,'-B',str(extracted/'CODE/audit_stage_p_three_system_package_v1.py'),'--root',str(extracted)],capture_output=True,text=True,encoding='utf8');assert p.returncode==0,p.stdout+p.stderr
    for row in protected:
        f=ROOT/row['original'];assert sha(f)==row['sha256']
    receipt=dict(status='PASS_REOPEN_EXTRACT_PARSE_AND_PORTABLE_AUDIT',ZIP=zipfile_path.name,bytes=zipfile_path.stat().st_size,sha256=sha(zipfile_path),members=len(manifest)+1,extracted_audit=json.loads(p.stdout),original_files_rechecked_after_packaging=len(protected),no_new_ratings=True,instance_destroyed=False)
    put(L/'DELIVERY_VALIDATION.json',receipt);(L/(zipfile_path.name+'.sha256')).write_text(receipt['sha256']+'  '+zipfile_path.name+'\n',encoding='utf8');print(json.dumps({k:v for k,v in receipt.items() if k!='extracted_audit'},indent=2))
if __name__=='__main__':main()
