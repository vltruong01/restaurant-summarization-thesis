"""Prepare/close offline artifacts; never imports model runtime or runs inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/stage_i/led_large_cad_v1'
WS=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
CP=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
OLD=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1'
NB=ROOT/'outputs/stage_i/led_large_nbest_reranking_v2'
PYTHON=ROOT/'.venv-caption-pilot/Scripts/python.exe'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def doc(name,text):(P/name).write_text(text.strip()+'\n',encoding='utf-8')


def main():
    assert not (P/'PREFLIGHT_MANIFEST.json').exists(), 'Preparation already sealed'
    P.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc).isoformat()
    archive=OLD/'LED_LARGE_CF_NATIVE_CE_V3_EXECUTION_EVIDENCE.tar.gz'
    assert sha(archive)=='9d0805d40b538730cd50077641e698dd03aa50274cacbe4bb129c2ae6ca80e1b'
    cfzip=ROOT/'outputs/cloud_handoff/LED_LARGE_NATIVE_CE_CONTRASTIVE_V3_CHATGPT_REVIEW.zip'
    assert sha(cfzip)=='d6783f6b8d9eb806ad90af8fb366219b21e43e36cf4041d87a881f7291bc6d73'
    historical=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_GPU_DIAGNOSTIC_REVIEW.zip'
    assert sha(historical)=='2fc650edf9720b46f8fd696981c93b690746a7a271cae616180c08ec8070c053'
    manifest=read(OLD/'downloaded/ARTIFACT_MANIFEST.json')
    # Support explicit manifest layout, no assumptions about original receipt status.
    files=manifest['files']
    for rel,entry in files.items():
        expected=entry if isinstance(entry,str) else entry['sha256']
        assert sha(OLD/'downloaded'/rel)==expected,rel
    proof=read(CP/'checkpoint_integrity.json')
    for name,h in proof['files'].items():assert sha(CP/name)==h,name
    assert sha(CP/'adapter_model.safetensors')=='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'
    close=dict(decision='FINAL_RESEARCHER_NO_GO_STOP_CONTRASTIVE_DEVELOPMENT',recorded_utc=now,
        decision_source='current researcher message: decided to STOP following latest Codex NO-GO',
        actual_updates=dict(CE=16,CE_plus_contrastive=16),probe_generation_outputs=12,probe_businesses=4,
        historical_lambda0_equivalence='FAIL_UNRESOLVED:106/456 tensors,63611 elements outside locked tolerance',
        findings=['Terra Mar: mixed, restores disagreement vs CE16 but unsupported claims persist',
                  'Minerva: loss of specificity/value; no substantive Core gain',
                  'Kobe: unsupported steaks-on-cherry relation',
                  'Whistlers: local review gains, shared visual merge/redundancy regression'],
        limitations=['8 fit businesses,15 pairs,4 previously examined TRAIN probes; not generalization',
            'post-adjudication/post-failed-diagnostic amendment,not preregistered',
            'negative wording/length shortcuts,unequal per-pair exposure,realized dropout streams differ',
            'native joint finite training does not resolve historical gradient-equivalence FAIL'],
        conclusion_scope='Stop this bounded branch; not evidence that all Contrastive approaches are scientifically ineffective',
        additional_contrastive_work_authorized=False,primary_checkpoint='Original LED-large CE epoch3 step750',
        archive_files_reverified=len(files),immutable_archives={str(x.relative_to(ROOT)):dict(sha256=sha(x),bytes=x.stat().st_size)
            for x in (archive,cfzip,historical)},
        original_checkpoint_files={str((CP/n).relative_to(ROOT)):h for n,h in proof['files'].items()},
        main_RO_U_F_Shuffled_protocol_unchanged=True)
    write('CONTRASTIVE_CLOSURE_RECEIPT.json',close)
    for n in ['EXECUTION_REPORT.md','QUALITY_COMPARISON.md','SCIENTIFIC_RECOMMENDATION.json','EXPERIMENT_RECEIPT.json',
              'ALL_PROBE_OUTPUTS.md','PROBE_SOURCE_EVIDENCE_QUOTES.json','INDEPENDENT_CPU_EXECUTION_AUDIT.json']:
        (P/'contrastive_closure').mkdir(exist_ok=True);shutil.copyfile(OLD/n,P/'contrastive_closure'/n)
    for n in ['TRAIN_SOURCE_ONLY_CASES.json','TRAIN_SAMPLE_MANIFEST.json','ORIGINAL_GENERATION_CONFIG.json','CHECKPOINT_BINDING.json']:
        shutil.copyfile(NB/n,P/n)
    cfg=read(ROOT/'configs/stage_i_led_large_nbest_v2.json')
    pc=read(WS/'configs/stage_i_led_large_decoder_policy_v2.json')
    modeldir=WS/pc['local_model_directory'];tokdir=WS/pc['source_tokenizer']
    model_paths=set(x for x in modeldir.iterdir() if x.is_file()) | set(x for x in tokdir.iterdir() if x.is_file())
    c=dict(experiment='LED_LARGE_CAD_V1',alpha=.3,formula='(1+alpha)*conditional_raw_logits-alpha*contextless_raw_logits',
        parent_code_bindings=cfg['parent_bindings'],runtime_pins=dict(cfg['runtime_pins'],**{'huggingface-hub':'0.36.2'}),
        model_files={str(x.relative_to(WS)).replace('\\','/'):sha(x) for x in sorted(model_paths)},
        backbone_revision=cfg['backbone_revision'],adapter_sha256=cfg['adapter_sha256'],
        checkpoint_receipt_sha256=sha(CP/'checkpoint_integrity.json'),
        contextless_source='[BUSINESS_INFORMATION]\n\n[REVIEWS]\n\n[VISUAL_CONTEXT]\n',
        generation_config_sha256=sha(P/'ORIGINAL_GENERATION_CONFIG.json'),
        train_cases_sha256=sha(P/'TRAIN_SOURCE_ONLY_CASES.json'),
        hardware_policy='one CUDA GPU; local RTX3050 6GB explicitly permitted, only if measured headroom>=15%',
        precision='unchanged FP32 storage, FP16 autocast, TF32 false; no quantization/offload/fallback',
        max_generation_calls=37,scientific_outputs=24,alpha0_verification_outputs=12,repeat_determinism_outputs=1,
        max_runtime_seconds=3600,memory_headroom_min=.15,
        training_updates=0,optimizer_construction=False,validation_access=False,held_out_access=False,
        no_extra_contrastive=True,no_automatic_retry=True,zero_caption_safeguard=False)
    write('CONFIG.json',c)
    doc('METHODOLOGY.md',r'''
# CAD V1: one fixed-alpha TRAIN-only exploratory pilot

Reference: Shi et al., NAACL 2024, https://aclanthology.org/2024.naacl-short.69/
Official code: https://github.com/xhan77/context-aware-decoding
The paper includes encoder-decoder FLAN-T5; it does not establish effectiveness
for our LED/LoRA model, Yelp input format, alpha0.3 or beam4. We adopt the specified
formula and fixed researcher alpha, without borrowing its reported quality gains.

## Exact LED implementation

The full branch receives the unchanged saved input IDs, tokenizer, padding to1024,
attention mask and BOS-only global attention. Contextless removes ALL factual
business metadata, review bodies/ratings/dates and captions, retaining precisely:
```
[BUSINESS_INFORMATION]

[REVIEWS]

[VISUAL_CONTEXT]
```
The final newline is included. It is tokenized with the SAME tokenizer/BOS/EOS,
padded with PAD1 and masked identically. No no-photo instruction or sentinel fact
is added. Original source and IDs are preserved; only the second branch differs.

One model/active adapter serves both branches. The contextless encoder runs once.
Both decoders maintain independent self/cross KV caches; the native selected beam
indices reorder BOTH caches. Before every score, check the previous reordered
prefix equals the current full prefix except its one new token. Compare actual
conditional tail input to the contextless tail. Start is EOS2 in both branches.
No labels/teacher forcing are introduced during inference.

Intercept actual LED forward output BEFORE native beam log_softmax. In FP32,
z_CAD=1.3*z_full-0.3*z_scaffold. Native beam4 then normalizes this mixed distribution,
applies original processors/forced EOS/length penalty and performs beam selection.
A post-log_softmax processor without re-normalization would have different beam
path scores; we explicitly avoid it. All generation parameters remain unchanged.
Alpha0 still executes both branches and returns ORIGINAL raw logits bit-exactly.
All12 alpha0 complete sequences/text must equal their native vanilla outputs.

Temporary instance prepare/cache wrappers and forward hooks are removed exactly
on exit/failure. No installed Transformers code or parameter tensor is patched.
Python/NumPy/CUDA RNG must remain unchanged; pinned LED consumes12 CPU encoder
draws per branch (12 vanilla,24 CAD/alpha0). Capture component states, verify the
exact expected transition, restore complete snapshot after each call. All modules
stay eval; config/use_cache, active adapters and frozen weights remain unchanged.

## Fixed budget and stop conditions

Exactly the original12 N-best TRAIN businesses (9 captioned,3 zero-caption).
Per case: native vanilla, alpha0 technical identity, alpha0.3 CAD. Repeat CAD only
on the first predeclared case to verify determinism. Maximum37 calls,24 scientific
outputs; no more businesses, alpha sweep, sampling or retries. First run local
GPU only if the tests and resource gate pass. 1h runtime cap. Any integrity,
finite-logit, prefix/cache, RNG, identity, technical-collapse or>=15%headroom failure
stops the run; no quantization, offload, different precision or lower beam fallback.
These are technical limits, not a scientific quality gate. Isolated BOS/cap warnings
are retained; sustained BOS/blank generation is a technical stop.

## Resource expectation, not measured feasibility

Single shared ~1.84GB FP32 backbone plus LoRA; autocast temporaries and TWO decoder
KV caches increase memory. For12 decoder layers,width1024,beam4,FP16 caches,
cross-cache storage is 2*K/V*12*4*1024*2bytes*(padded_source+1024).
At source5120 this is ~1.125GiB; two self caches at767 tokens add ~0.281GiB.
This excludes FP32 fallback tensors, encoder workspaces, allocator and display.
6GB local VRAM may fail the15%headroom gate. No memory-feasibility claim until
measured. Cost is about62 decoder branch-equivalents vs12 vanilla calls, not a
promised wall time. Same machine/timing per method are recorded.

## Scientific decision, locked before output

Review all12 pairs against ONLY model-visible selected reviews and Filtered
captions. Report Core/business-specific coverage, same-aspect disagreement,
factual/visual support, semantic repetition, zero-caption failures, completeness,
format and regressions. Link claims to source marker and exact quotation/offset;
quotation match alone is not entailment. Preserve raw texts/tokens even on failure.
Zero-caption visual assertions are unsupported even if plausible from reviews.
Do not use Teacher targets, D053 references, photos, validation or held-out data.
No visual safeguard is applied. Any later safeguard requires separate paired
evaluation equally on raw vanilla and CAD, never post-hoc credit for only CAD.

GO means source-grounded, meaningful net improvement across several cases without
serious new grounding/omission regressions, supported by independent researcher
review; technical recovery alone is insufficient. INCONCLUSIVE means insufficient
or mixed evidence/technical resource stop. NO-GO means no meaningful net benefit
or important regressions. No forced win quota, length preference, ROUGE-only
decision, or automatic checkpoint promotion. Report all local tradeoffs. These12
previously examined TRAIN cases are not an independent generalization benchmark.
Any later main-study CAD must use a matched policy for RO/U/F/Shuffled; CAD gains
cannot be credited to caption filtering. Current primary protocol stays unchanged.
''')
    doc('CONTRASTIVE_FINAL_NO_GO.md','''
# Contrastive branch CLOSED by researcher

The latest actual pilot completed CE16 and Pair16 and12 outputs on four TRAIN
probes. Terra Mar/Whistlers show local mixed benefits; Minerva loses useful
specificity and Kobe introduces unsupported steaks/cherry relations. No clear
overall source-grounded improvement justifies further tuning. Historical V2.1
gradient-equivalence FAIL remains unresolved. This is a final stop for THIS
bounded branch, not a general scientific claim against Contrastive learning.

All113 exported files, the220MB full execution/checkpoint/log archive, portable
review ZIP and historical diagnostic ZIP were hash-reverified. New closure
receipt records original archive hashes; original execution reports, raw
gradients, generated outputs and checkpoint receipts remain unchanged.
Original CE epoch3 remains the Filtered checkpoint. No more Contrastive GPU
diagnostics, training, negatives, targets, tuning or checkpoint promotion.
''')
    # Reproducible CPU tests; no pretrained model load or target-model generation.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    r=subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p','test_led_large_cad_v1.py','-v'],
        cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    doc('CPU_TESTS.txt',r.stdout+'\n'+r.stderr);assert r.returncode==0,r.stderr
    program=r'''import json,sys,hashlib,torch
from pathlib import Path
from transformers import LEDTokenizerFast
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from led_large_cad_v1 import contextless_source
p=root/'outputs/stage_i/led_large_cad_v1';ws=root/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
pc=json.loads((ws/'configs/stage_i_led_large_decoder_policy_v2.json').read_text(encoding='utf-8'))
t=LEDTokenizerFast.from_pretrained(ws/pc['source_tokenizer'],local_files_only=True)
cases=json.loads((p/'TRAIN_SOURCE_ONLY_CASES.json').read_text(encoding='utf-8'));rows=[]
for c in cases:
 ids=t.encode(c['source'],add_special_tokens=True)
 assert ids==c['input_ids'],c['business_id']
 ctx=contextless_source(c['source']);v=t.encode(ctx,add_special_tokens=True)
 assert v[0]==0 and v[-1]==2
 rows.append(dict(business_id=c['business_id'],input_tokens=len(ids),source_exact=True,captions=len(c['captions']),contextless_text=ctx,contextless_ids=v))
print(json.dumps(dict(cases=rows,torch=torch.__version__,pretrained_model_loads=0,validation_or_test_read=False)))
'''
    r=subprocess.run([str(PYTHON),'-c',program],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    doc('SOURCE_AUDIT_LOG.txt',r.stdout+'\n'+r.stderr);assert r.returncode==0,r.stderr
    write('SOURCE_TOKENIZER_AUDIT.json',json.loads(r.stdout))
    bound=[ROOT/'scripts/led_large_cad_v1.py',ROOT/'scripts/run_led_large_cad_v1.py',ROOT/'tests/test_led_large_cad_v1.py']
    bound+=sorted(x for x in P.iterdir() if x.is_file() and x.name not in {'PREFLIGHT_MANIFEST.json','BINDINGS.json'})
    write('PREFLIGHT_MANIFEST.json',dict(files={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in bound}))
    write('BINDINGS.json',dict(config_sha256=sha(P/'CONFIG.json'),runner_sha256=sha(ROOT/'scripts/run_led_large_cad_v1.py'),
        cad_module_sha256=sha(ROOT/'scripts/led_large_cad_v1.py'),preflight_sha256=sha(P/'PREFLIGHT_MANIFEST.json')))
    print('PREPARED:',P)


if __name__=='__main__':main()
