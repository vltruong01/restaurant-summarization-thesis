"""Finalize validation/readability of the already downloaded Stage P review ZIP."""
import hashlib,json,re,shutil,subprocess,sys,zipfile,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];L=ROOT/'outputs/stage_p/qwen_bf16_3090_execution_20261106_v2';P=L/'REVIEW_PACKAGE'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    r=read(P/'RESOURCE_MEASUREMENTS.json');f=read(P/'OUTPUT_FLAGS.json');v=read(P/'VALIDATION/NATIVE_RECORD_VERIFICATION.json');needles=[r'\bhf_[A-Za-z0-9]{20,}\b',r'\bsk-[A-Za-z0-9_-]{20,}\b',r'BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',r'"password_hash"\s*:',r'"session_token"\s*:']
    assert v['status']=='PASS_COMPLETE_NATIVE_BINDINGS' and v['token_decode_verified']
    for x in f:assert not any(x[k] for k in ['empty','heading_missing','unexpected_thinking','hit_token_cap']) and x['stop_token_seen']
    for path in P.rglob('*'):
        if not path.is_file():continue
        assert path.suffix.lower() not in ['.sqlite3','.db','.sqlite','.safetensors','.pem','.key'] and path.name!='.env'
        if path.suffix.lower() in ['.py','.json','.jsonl','.log','.txt','.md']:
            assert not any(re.search(n,path.read_text(encoding='utf-8-sig')) for n in needles),path.name
    put(P/'VALIDATION/PRIVACY_CHECK.json',dict(status='PASS',private_credentials_API_keys_sessions_rating_DB_model_weights_included=False,known_secret_pattern_scan='PASS',scope='Known credential markers and forbidden files; public source texts and required pseudonymous/source bindings retained'))
    report=f'''# Stage P actual execution validation

| Check | Result | Evidence |
|---|---|---|
| Official pinned model metadata | PASS | OFFICIAL_MODEL_REVISION_CHECK.json; four shard sizes/LFS hashes match revision c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Actual remote weight byte/hash checks | PASS | NATIVE/EXECUTION/WEIGHT_VERIFICATION.json; weights not copied into this ZIP |
| Exact LED F evidence and isolation | PASS | INPUT_ALIGNMENT.json; all30 source strings and evidence IDs;247 review/73 filtered-caption presentations |
| Prior CPU contracts | PASS | Eight tests; no model generation or human-rating fixtures |
| One-restaurant actual GPU pilot | PASS | NATIVE/EXECUTION/COMPATIBILITY_PASS.json; largest Review prompt,3,583 tokens; both components retained |
| Full actual GPU inference | PASS | Native completion and exit0;15 restaurants,30 isolated calls, BF16,batch1,thinking disabled |
| Output preservation | PASS | Every token ID/raw text stored; exact CPU tokenizer replay for all30 calls; verbatim assembly |
| Termination/format diagnostics | PASS | 30 stop-token endings;0 empty,missing-heading,thinking or cap flags; no selective regeneration |
| Resource measurements | PASS | {r['model_generation_seconds']:.3f}s model generation,{r['output_tokens']} token IDs,{r['aggregate_tokens_per_second_including_prefill']:.3f}tokens/s; peak allocated{r['peak_allocated_bytes']/1024**3:.3f}GiB,reserved{r['peak_reserved_bytes']/1024**3:.3f}GiB; device sampled peak{r['sampled_peak_whole_device_MiB']/1024:.3f}GiB;0 sampler errors |
| Local native download | PASS | Original remote archive SHA and every exported member verified; VALIDATION/VERIFIED_DOWNLOAD.json |
| Protected existing artifacts | PASS | Existing spec,prompts,inputs,Stage O frozen45 outputs and final90-rating export hashes unchanged |
| Privacy | PASS | Known credential marker scan; no private DB,API keys,sessions,weight files or JPEG archive included |

Timing separates258.961s download,25.666s weight hashing,11.533s model loading from186.844s generation. Warm/model-call timing excludes shared BLIP-2 generation/filtering and text decoding/file export. Cold pilot calls are included in the30-call total; no extra benchmark generations were run. Whole-device sampling can miss transients; allocator peaks are measured separately. Native logs disclose slower reference PyTorch gated-delta/causal-convolution kernels. LED F used a different GPU,precision policy and four-beam decoder, so these measurements do not isolate architecture efficiency.

An initial UUID-format check failed before weights were downloaded or the model loaded. Its log remains preserved; strict normalization matched the same exact UUID and enabled the first model attempt. This was not an OOM or an output retry. All actual model calls completed on the supplied RTX3090; no A100,quantization,other model,new rental or model-policy change was used.

Technical integrity does not certify factual quality. No human evaluation was initiated, no scores or existing model outputs changed, and no final quality winner is selected. The source-only Qwen outputs are returned for independent review. The current GPU instance is intact; its inference worker has exited.
'''
    (P/'VALIDATION_REPORT.md').write_text(report,encoding='utf8')
    shutil.copy2(Path(__file__),P/'SCRIPTS'/Path(__file__).name)
    put(P/'PACKAGE_MANIFEST.json',dict(files=[dict(file=x.relative_to(P).as_posix(),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(P.rglob('*')) if x.is_file() and x!=P/'PACKAGE_MANIFEST.json']))
    artifact=L/'STAGE_P_QWEN_COMPARISON_REVIEW.zip'
    with zipfile.ZipFile(artifact,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for x in sorted(P.rglob('*')):
            if x.is_file():z.write(x,x.relative_to(P).as_posix())
    target=L/'VERIFIED_FINAL_EXTRACTION';assert not target.exists()
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));members=len(z.namelist());z.extractall(target)
        for x in P.rglob('*'):
            if x.is_file():assert sha(x)==sha(target/x.relative_to(P))
    command=[str(ROOT/'.venv-stage-p-preflight/Scripts/python.exe'),'-X','utf8','-B',str(target/'SCRIPTS/verify_stage_p_qwen_execution_v2.py'),'--package',str(target),'--decode-tokens']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf8');assert result.returncode==0,result.stdout+result.stderr;check=json.loads(result.stdout)
    h=sha(artifact);(L/(artifact.name+'.sha256')).write_text(h+'  '+artifact.name+'\n',encoding='ascii');receipt=dict(status='PASS_COMPLETE_REVIEW',artifact=artifact.name,sha256=h,bytes=artifact.stat().st_size,members=members,archive_reopened_extracted_every_member_hashed=True,all_30_outputs_token_replay_verified=True,validation=check,resources={k:r[k] for k in ['peak_allocated_bytes','peak_reserved_bytes','sampled_peak_whole_device_MiB','model_generation_seconds','output_tokens','aggregate_tokens_per_second_including_prefill']},human_evaluation_started=False,instance_destroyed=False,at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    put(L/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
