"""Build V2 offline review only. Never SSHs, runs target-model inference or trains."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/stage_i/led_large_nbest_reranking_v2'
V1=ROOT/'outputs/stage_i/led_large_nbest_reranking_v1'
FAIL=ROOT/'outputs/stage_i/led_large_nbest_v1_execution_v1'
WS=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
CP=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def doc(name,text):(P/name).write_text(text.strip()+'\n',encoding='utf-8')
def verify_zip(p,manifest):
    with zipfile.ZipFile(p) as z:
        files=json.loads(z.read(manifest))['files']
        for name,digest in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    return len(files)

def main():
    P.mkdir(exist_ok=True)
    v1zip=ROOT/'outputs/cloud_handoff/LED_LARGE_NBEST_RERANKING_V1_OFFLINE_REVIEW.zip'
    failzip=ROOT/'outputs/cloud_handoff/LED_LARGE_NBEST_PHASE_A_FAILURE_REVIEW.zip'
    assert sha(v1zip)=='40038f252656051b3d112c85621d424e82b38dab4a1375b441170092c28f75d9'
    assert sha(failzip)=='a886b4876dd5e163204300b8ceaf3eb8ab822b7d4b4b8ab1e58b0101cc89914a'
    counts=dict(V1_portable_entries=verify_zip(v1zip,'outputs/stage_i/led_large_nbest_reranking_v1/SHA256_MANIFEST.json'),
                V1_failure_portable_entries=verify_zip(failzip,'SHA256_MANIFEST.json'))
    for rel,digest in read(V1/'PREFLIGHT_MANIFEST.json')['files'].items():assert sha(ROOT/rel)==digest,rel
    cfg=read(ROOT/'configs/stage_i_led_large_nbest_v2.json')
    for rel,digest in cfg['parent_bindings'].items():assert sha(WS/rel)==digest,rel
    proof=read(CP/'checkpoint_integrity.json')
    assert proof==read(V1/'CHECKPOINT_BINDING.json')
    assert sha(CP/'checkpoint_integrity.json')==read(V1/'OFFLINE_INTEGRITY.json')['checkpoint_receipt_sha256']
    for name,digest in proof['files'].items():assert sha(CP/name)==digest,name
    # Exact source/generation files copied, never regenerated.
    for name in ('TRAIN_SOURCE_ONLY_CASES.json','TRAIN_SAMPLE_MANIFEST.json','CHECKPOINT_BINDING.json',
                 'OFFLINE_INTEGRITY.json','ORIGINAL_GENERATION_CONFIG.json','NBEST_GENERATION_CONFIG.json'):
        shutil.copyfile(V1/name,P/name)
        assert sha(P/name)==sha(V1/name)
    historical=FAIL/'closed_evidence/CE_V2_WORKSPACE/runs/F_CE3_nbest_training_v1/raw/EzJI5x25U5GgbYBWhu5ZQQ_single.json'
    shutil.copyfile(historical,P/'HISTORICAL_V1_N1_UNCHANGED.json')
    old=read(historical)[0];cases=read(P/'TRAIN_SOURCE_ONLY_CASES.json')
    assert cases[0]['business_id']=='EzJI5x25U5GgbYBWhu5ZQQ' and not cases[0]['captions']
    assert old['source_sha256']==cases[0]['source_sha256']
    save('HISTORICAL_ZERO_CAPTION_OBSERVATION.json',dict(
        business_id=cases[0]['business_id'],split='train',Filtered_caption_count=0,
        original_N1_sha256=sha(historical),source_sha256=cases[0]['source_sha256'],
        visual_claims=['The photo shows a frozen taco.','The taco is next to a bell.','The taco is next to a strip of chicken.'],
        finding='UNSUPPORTED_VISUAL_ASSERTIONS_IN_ZERO_CAPTION_N1',
        provenance='Explicit researcher observation; original raw text and token IDs preserved byte-for-byte',
        technical_stop_cause=False,alternative_candidate_quality='UNASSESSED_NO_N4_IN_V1'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    python=ROOT/'.venv-caption-pilot/Scripts/python.exe'
    test_counts={}
    for version in ('v2','v1'):
        result=subprocess.run([str(python),'-m','unittest','discover','-s','tests','-p',f'test_led_large_nbest_{version}.py','-v'],
                              cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        doc('CPU_TESTS_'+version.upper()+'.txt',result.stdout+'\n'+result.stderr)
        assert result.returncode==0,result.stderr
        test_counts[version]=int(re.search(r'Ran (\d+) tests',result.stderr).group(1))
    program="import sys;from pathlib import Path;w=Path('outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE').resolve();sys.path.insert(0,str(w/'scripts'));import stage_i_led_large_decoder_policy_v2 as p;p.preflight();print('FULL_PARENT_CPU_PREFLIGHT_PASS_NO_PRETRAINED_MODEL_FORWARD')"
    result=subprocess.run([str(python),'-c',program],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    doc('PARENT_CPU_PREFLIGHT.txt',result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
    runtime_code="import transformers,torch,importlib.metadata,json;from pathlib import Path;from transformers.models.led import modeling_led;import hashlib;print(json.dumps(dict(runtime={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','numpy','tokenizers','safetensors','huggingface-hub']},encoder_source_sha256=hashlib.sha256(Path(modeling_led.__file__).read_bytes()).hexdigest())))"
    runtime=subprocess.run([str(python),'-c',runtime_code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    runtime=json.loads(runtime.stdout)
    assert runtime['encoder_source_sha256']==cfg['rng_guard']['encoder_source_sha256']
    save('CPU_VALIDATION.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),tests_passed=test_counts,
        total_tests_passed=sum(test_counts.values()),**runtime,full_parent_CPU_preflight='PASS',
        GPU_inference_calls=0,GPU_training_updates=0,pretrained_model_loads=0,
        CPU_tiny_synthetic_encoder_forward_tests=1,CPU_synthetic_generate_stubs_only=True,
        CUDA_change_tests='Synthetic CPU ByteTensor device states; actual GPU transition remains pending',
        held_out_access=False,Teacher_calls=0,target_or_source_modifications=0))
    save('INTEGRITY_RECEIPT.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),**counts,
        V1_preflight_entries_verified=27,V1_offline_zip_sha256=sha(v1zip),V1_failure_zip_sha256=sha(failzip),
        checkpoint_components_verified=len(proof['files']),checkpoint_receipt_sha256=sha(CP/'checkpoint_integrity.json'),
        adapter_sha256=sha(CP/'adapter_model.safetensors'),original_backbone_sha256=cfg['backbone_sha256'],
        parent_manifest_entries_verified=len(read(WS/'VAST_SHA256_MANIFEST.json')['files']),
        TRAIN_cases=12,zero_caption_cases=3,nonzero_caption_cases=9,
        TRAIN_case_bytes_unchanged=sha(P/'TRAIN_SOURCE_ONLY_CASES.json')==sha(V1/'TRAIN_SOURCE_ONLY_CASES.json'),
        historical_N1_bytes_unchanged=sha(P/'HISTORICAL_V1_N1_UNCHANGED.json')==sha(historical),
        original_generation_sha256=sha(P/'ORIGINAL_GENERATION_CONFIG.json'),N4_generation_sha256=sha(P/'NBEST_GENERATION_CONFIG.json'),
        historical_failed_calls=1,historical_returned_sequences=1,new_GPU_authorization=False))
    diff=''.join(difflib.unified_diff((ROOT/'scripts/run_stage_i_led_large_nbest_v1.py').read_text().splitlines(True),
                                    (ROOT/'scripts/run_stage_i_led_large_nbest_v2.py').read_text().splitlines(True),
                                    fromfile='V1_preserved/run_stage_i_led_large_nbest_v1.py',tofile='V2/run_stage_i_led_large_nbest_v2.py'))
    doc('RUNNER_V1_TO_V2.patch',diff)
    doc('README_FIRST.md', '''
# LED-large N-best V2 — OFFLINE REVIEW ONLY

This is one minimal RNG-guard correction to the failed TRAIN Phase A pilot.
Start with MINIMAL_PATCH_AND_UNCERTAINTY.md, CPU_VALIDATION.json,
INTEGRITY_RECEIPT.json and EXECUTION_AUTHORIZATION_REQUEST.md.

No GPU inference, training, validation generation, source edits, target edits,
decoding search or verifier optimization was performed in this revision.
Historical V1 and its single N1 remain unchanged and separately accounted for.

The review archive mirrors repository paths. `reference/` contains exact copies
of both independently reviewed V1 ZIPs. Inspect them without changing originals.
To reproduce tests, use the accepted full CE DecoderPolicyV2 parent workspace
and preserved V1 payload alongside this additive V2 overlay, with pinned runtime.
The compact archive intentionally does not duplicate the 1.84 GB backbone.
For an empty review folder, extract the nested V1 offline ZIP into the same root
before inspecting/running V2. CPU tests additionally require the preserved parent
scripts at outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/scripts.
Do not extract or run any historical launcher to start training.

The new CLI permits preflight and training_pilot only. Phase B/C are unavailable.
The pending receipt has GPU approval false and no authorization clock/deadline.
No old approval can authorize this V2 code. Stop for independent review.
''')
    doc('MINIMAL_PATCH_AND_UNCERTAINTY.md', '''
# Minimal revision

V1 wrongly required complete RNG invariance across LED eval generation. The pinned
LEDEncoder.forward draws one CPU scalar per encoder layer, even in eval, before
the self.training LayerDrop condition. The twelve-layer tiny CPU encoder test
executes the actual Transformers 4.51.3 implementation and confirms the transition.
The failed V1 run did not save component snapshots, so only-CPU change in that
historical run is not retrospectively established.

V2 reuses the parent capture_rng/restore_rng and value-based fingerprint functions.
It predicts CPU state using an independent CPU Generator, seeded with the captured
CPU state, followed by exactly twelve separate float32 scalar torch.rand draws.
This calculation cannot consume the global CPU stream. The actual post-generate
CPU fingerprint must match that state; Python, NumPy and each CUDA device must
match their pre-call fingerprints. Zero, thirteen or any other CPU advancement,
unexpected other-component changes, or inexact restoration are hard failures.

After each generate attempt the complete snapshot is restored and captured again.
Actual pre-restore and post-restore fingerprints are recorded per component.
Failed CUDA capture/restoration is explicitly unavailable/failure, never unchanged.
Source hash of modeling_led.py, twelve encoder layers, default float32/CPU draw
assumptions, all-module eval mode, original deterministic beam4 config and active
CE epoch-3 adapter are checked. Transformers is never patched.

The new narrow helper flushes STARTED evidence, then all available raw token IDs,
decoded text, elapsed time, memory samples/summary and RNG diagnostics to durable
files before collected guard errors may abort. Subsequent memory and N1/N4 identity
failures retain both call receipts. Global MemoryMeter samples are flushed on
failure too. A generation/device failure cannot provide token IDs that were never
returned; such absence is explicit. The deadline watchdog can still interrupt an
in-flight kernel, so partial evidence cannot be guaranteed after a hard device/OS
termination. No exception permits retry.

The actual model.generate expression, no_grad/FP16 autocast, frozen inputs,
tokenizer, adapter, backbone and generation configs are unchanged. The CLI and
new namespace are limited to TRAIN Phase A. No scorer/verifier work is added.
The old N1 zero-caption visual assertions remain documented semantic errors,
separate from the technical RNG stop. No candidate is edited or rewritten.

Remaining uncertainty: full pretrained GPU N1/N4 behavior, per-device CUDA RNG
transition, measured inference headroom, candidate semantic diversity and oracle
headroom have NOT been validated by this CPU revision. Require fresh GPU approval.
''')
    doc('EXPECTED_GPU_COST.md', '''
# Planning estimate, not a new benchmark

One RTX 3090 24 GB; exactly 24 calls (12 N1 + 12 N4), 60 returned sequences.
Both modes retain four beams. Returning four finished hypotheses does not launch
four additional beam searches. No model training or optimizer is constructed.

Allow roughly 15–45 minutes (0.25–0.75 GPU-hours) for the bounded inference pass
and model/integrity overhead on an already prepared host. This is a conservative
planning allowance, not measured V2 throughput or a completion guarantee.
The V1 single-call run lasted about 33 seconds including load/preflight overhead;
its isolated generation timing was NOT saved and cannot support a precise rate.
Fresh setup previously took about 26 minutes, dominated by the 2.2 GiB torch
wheel download. Keep installation/transfer time separate from inference timing.

GPU charges = the actual provider hourly rate times elapsed billed hours; no live
rate is assumed. Storage/network charges are separate. The prior CE training
receipt had 17,684,430,848 conservative peak bytes and 30.10% headroom on 24 GB,
which is only a historical engineering reference, not N-best inference evidence.
No V1 inference peak was flushed before failure. Preserve the >=15% measured
headroom hard stop and the new durable per-call measurements; no OOM fallback.
The new authorization must enforce its actual UTC window, at most 48 hours.
''')
    doc('COMMANDS_AFTER_SEPARATE_APPROVAL.md', '''
# Commands only — no GPU execution authorized now

Apply this additive overlay to the full verified CE DecoderPolicyV2/V1 workspace.
Preserve runs/F_CE3_nbest_training_v1 and all old receipts. Use new namespace
runs/F_CE3_nbest_training_v2; refuse if it already exists. Reverify the full parent,
V1/V2 hashes, checkpoint-750 contents and runtime on the execution host first.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p test_led_large_nbest_v2.py -v
.venv/bin/python scripts/run_stage_i_led_large_nbest_v2.py preflight
# Only after a fresh explicit hash-bound V2 researcher approval receipt:
.venv/bin/python -u scripts/run_stage_i_led_large_nbest_v2.py training_pilot \
  --workspace /workspace/CE_V2_WORKSPACE \
  --checkpoint /workspace/CE_V2_SAVED_RUN/train/checkpoint-750 \
  --output /workspace/CE_V2_WORKSPACE/runs/F_CE3_nbest_training_v2 \
  --approval /workspace/NBEST_V2_RESEARCHER_APPROVAL.json --execute
```

The first command is CPU-only synthetic regression; it does not load the pretrained
LED or generate. No old tmux/supervisor wrapper may relaunch V1 or authorize V2.
Any future service must have autostart=false, autorestart=false, startretries=0.
Stop on any guard/integrity/deadline failure. Export receipts and stop for review.
''')
    code=['configs/stage_i_led_large_nbest_v2.json','scripts/run_stage_i_led_large_nbest_v2.py',
          'scripts/led_large_nbest_rng_v2.py','tests/test_led_large_nbest_v2.py','scripts/package_led_large_nbest_v2_review.py']
    excluded={'PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',
              'EXECUTION_AUTHORIZATION_REQUEST.md','SHA256_MANIFEST.json','DELIVERY_RECEIPT.json','PAYLOAD_PREFLIGHT.txt'}
    files=[ROOT/x for x in code]+[x for x in P.rglob('*') if x.is_file() and x.name not in excluded]
    save('PREFLIGHT_MANIFEST.json',dict(files={x.relative_to(ROOT).as_posix():sha(x) for x in sorted(files)},
                                     scope='V2 additive Phase A correction; complete preserved V1 and CE parent verified separately'))
    bindings=dict(config_sha256=sha(ROOT/code[0]),runner_sha256=sha(ROOT/code[1]),preflight_sha256=sha(P/'PREFLIGHT_MANIFEST.json'))
    save('EXECUTION_BINDINGS.json',dict(experiment=cfg['experiment'],authorization_bindings=bindings,
        RNG_helper_sha256=sha(ROOT/code[2]),adapter_sha256=cfg['adapter_sha256'],GPU_inference_approved=False,
        fresh_output_namespace='runs/F_CE3_nbest_training_v2',historical_failed_calls=1))
    template=dict(experiment=cfg['experiment'],GPU_inference_approved=False,researcher_explicit_authorization=False,
        bindings=bindings,phases=['training_pilot'],authorized_at_utc=None,deadline_utc=None,
        generation_calls_max=24,returned_sequences_max=60,training_authorized=False,held_out_access=False,
        Teacher_calls=False,automatic_retry=False)
    save('AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',template)
    doc('EXECUTION_AUTHORIZATION_REQUEST.md',f'''
# Exact request — NOT APPROVED

Request one new LED_LARGE_NBEST_RERANKING_V2 TRAIN Phase A inference attempt only.

- Config SHA-256: `{bindings['config_sha256']}`
- Runner SHA-256: `{bindings['runner_sha256']}`
- Preflight SHA-256: `{bindings['preflight_sha256']}`
- RNG helper SHA-256: `{sha(ROOT/code[2])}`
- Unchanged CE epoch-3 adapter: `{cfg['adapter_sha256']}`
- New namespace: `runs/F_CE3_nbest_training_v2` (must not exist).

Upon separate explicit approval, record actual UTC receipt time and a hard deadline
no later than 48 hours afterward. Do not copy V1's clock or approval. Verify the
full pinned host workspace/runtime and checkpoint, then generate N1/N4 for exactly
the SAME 12 TRAIN businesses (3 zero-caption, 9 captioned), 24 calls/60 sequences.
Require expected CPU12 advancement, unchanged other RNG components, complete exact
restoration, all-module eval, technical sanity, >=15% headroom, and N1/N4 rank-0
content/text identity (trailing PAD differences reported separately).

The historical V1 call/sequence remain separate. A complete new attempt would bring
the historical cumulative total to 25 calls/61 sequences, not erase or resume V1.
No automatic retry, validation inference, reranker optimization, training, Teacher
change, decoding change, Contrastive Fine-tuning or held-out access is requested.
Export the 60 sequences, source-only packets and all technical receipts; stop for
independent TRAIN diversity/oracle review. The 6/12 and 4/12 scientific gates remain
pending; the scorer is not validated. Current GPU approval remains FALSE.
''')
    result=subprocess.run([str(python),'scripts/run_stage_i_led_large_nbest_v2.py','preflight'],cwd=ROOT,env=env,
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    doc('PAYLOAD_PREFLIGHT.txt',result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
    files += [P/n for n in ('PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',
                            'EXECUTION_AUTHORIZATION_REQUEST.md','PAYLOAD_PREFLIGHT.txt')]
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_NBEST_RERANKING_V2_OFFLINE_REVIEW.zip'
    members={x.relative_to(ROOT).as_posix():x for x in files}
    members['reference/'+v1zip.name]=v1zip
    members['reference/'+failzip.name]=failzip
    portable={name:sha(path) for name,path in sorted(members.items())}
    save('SHA256_MANIFEST.json',dict(files=portable))
    members[P.relative_to(ROOT).as_posix()+'/SHA256_MANIFEST.json']=P/'SHA256_MANIFEST.json'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(members.items()):z.write(path,name)
    with zipfile.ZipFile(archive) as z:
        for name,digest in portable.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    receipt=dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,portable_entries_verified=len(portable),
                 new_CPU_tests=test_counts['v2'],inherited_CPU_tests=test_counts['v1'],bindings=bindings,
                 RNG_helper_sha256=sha(ROOT/code[2]),GPU_inference_calls=0,GPU_execution_authorized=False)
    save('DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
