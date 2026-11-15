"""Integrity closeout and additive GPT V2 handoff; no model forward or GPU work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import json,hashlib,zipfile,math
from datetime import datetime,timezone
import torch
from huggingface_hub import try_to_load_from_cache

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/stage_i/led_large_vast_bos_cause_audit_v1'
FROZEN=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    proof={}
    for step in (250,750,1250):
        folder=ROOT/f'outputs/stage_i/d059_expanded_filtered_v2/train/checkpoint-{step}'
        receipt=json.loads((folder/'checkpoint_integrity.json').read_text())
        assert sha(folder/'scaler.pt')==receipt['files']['scaler.pt']
        scaler=torch.load(folder/'scaler.pt',map_location='cpu',weights_only=True)
        assert scaler['scale']==65536 and scaler['_growth_tracker']==step
        proof[str(step)]=dict(state=scaler,scaler_sha256=sha(folder/'scaler.pt'),receipt_sha256=sha(folder/'checkpoint_integrity.json'))
    revision='38335783885b338d93791936c54bb4be46bebed9'
    basepath=Path(try_to_load_from_cache('allenai/led-base-16384','pytorch_model.bin',revision=revision))
    state=torch.load(str(basepath),map_location='cpu',weights_only=True,mmap=True)
    canonical=json.loads((ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json').read_text())['architectures'][0]['canonical_initial_backbone_hashes']
    for name,digest in canonical.items():
        # Historical parameters() fingerprints include type, dtype, shape and
        # delimiters, not a raw tensor-byte SHA. Reproduce that exact encoding.
        a=state[name].detach().cpu().contiguous().numpy()
        h=hashlib.sha256()
        h.update(b'Parameter:')
        h.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        h.update(b';')
        assert h.hexdigest()==digest,name
    proof['base_pretrained_tensor_hashes_match_historical_D052']=dict(tensors=len(canonical),status='PASS',binary_sha256=sha(basepath))
    write(DEST/'D059_SCALER_AND_INTEGRITY.json',proof)
    cpu=json.loads((DEST/'CPU_PREFIX_LOGITS.json').read_text())
    assert set(cpu['models'])=={'large_pretrained_zero_LoRA','large_epoch3','large_epoch5','D059_base_epoch3'}
    assert all(v['finite'] for v in cpu['models'].values())
    assert all(r['logits_position0_vs1_max_abs']<2e-5 for k,m in cpu['models'].items() if k.startswith('large') for r in m['rows'])
    assert all(r['logits_position0_vs1_max_abs']>20 for r in cpu['models']['D059_base_epoch3']['rows'])
    large_weights=FROZEN/'assets/models/led-large-16384/pytorch_model.bin'
    assert sha(large_weights)=='b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081'
    for name in ('STATIC_ARTIFACT_AUDIT.json','CPU_PREFIX_LOGITS.json','D059_SCALER_AND_INTEGRITY.json','CAUSE_FINDINGS_VI.md'):
        assert (DEST/name).is_file()
    write(DEST/'DIAGNOSTIC_RECEIPT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-10-22',
        actual_forward_calls=4,synthetic_sources_per_call=2,device='CPU',precision='float32',
        generate_calls=0,backward_calls=0,optimizer_updates=0,held_out_access=False,
        frozen_large_binary_unchanged=True,base_pretrained_tensors_match_historical_D052=True,
        original_actual_summary_prediction_budget_unchanged=48,
        causal_repair_validation='NOT_PERFORMED',researcher_decision='PENDING',
        diagnostic_preparation_interruptions=[
            'An over-strict CPU availability guard stopped before model work on a CUDA-capable Windows host; replaced with a CUDA-init prohibition and explicit CPU placement assertions.',
            'Base config transport path was absent after the three large CPU probes. Corrected to existing frozen local base config; ran only the base comparison afterward. No large probe repeated.'],
        evidence={name:sha(DEST/name) for name in ('STATIC_ARTIFACT_AUDIT.json','CPU_PREFIX_LOGITS.json','D059_SCALER_AND_INTEGRITY.json','CAUSE_FINDINGS_VI.md')},
        scripts={name:sha(ROOT/'scripts'/name) for name in ('audit_led_large_bos_artifacts_v1.py','diagnose_led_large_bos_cpu_v1.py','package_led_large_bos_findings_v2.py')}))
    src=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_GPT_REVIEW_V1.zip'
    out=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_GPT_REVIEW_V2.zip'
    manifest={}
    with zipfile.ZipFile(src) as old,zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def put(name,raw):
            assert name not in manifest
            z.writestr(name,raw);manifest[name]=hashlib.sha256(raw).hexdigest()
        for name in old.namelist():
            put('V1_ORIGINAL_SHA256_MANIFEST.json' if name=='SHA256_MANIFEST.json' else name,old.read(name))
        for p in sorted(DEST.iterdir()):
            if p.is_file() and p.suffix in ('.json','.md'):put('bos_diagnosis/'+p.name,p.read_bytes())
        for name in ('audit_led_large_bos_artifacts_v1.py','diagnose_led_large_bos_cpu_v1.py','package_led_large_bos_findings_v2.py'):
            put('bos_diagnosis/scripts/'+name,(ROOT/'scripts'/name).read_bytes())
        readme='''# V2: read this first — new CPU BOS evidence, 22 October 2026 ICT

This version includes the complete V1 review material byte-for-byte plus bos_diagnosis/.
The original V1 root manifest is V1_ORIGINAL_SHA256_MANIFEST.json; the new SHA256_MANIFEST.json covers this entire ZIP.
V1 context/README describe the earlier package stage. The following CPU prefix probes occurred AFTER V1 was prepared, and are the new evidence to review first.

1. Read bos_diagnosis/CAUSE_FINDINGS_VI.md and CPU_PREFIX_LOGITS.json.
2. Four FP32 CPU teacher-forced forward calls, each on the same two synthetic sources, tested official-large zero-LoRA, actual large epoch3/5, and actual D059 base epoch3. No generate/backward/optimizer, no real validation or held-out inference, no new summaries.
3. Large has nearly identical logits at initial decoder positions0/1 (max difference<0.00002); after training both prefer BOS. Base differs substantially (>27 on both probes) and moves to text. This reproduces a checkpoint/prefix-specific anomaly; it does NOT prove a particular repair will improve substantive quality. On synthetic unformatted sources base does not produce the Yelp header either; this is not a quality benchmark.
4. All2200 F labels have exactly one initial BOS, no gold BOS loops. All48 saved large outputs start with >=5 BOS; D059epoch3 starts with exactly1 in all24 outputs.
5. D059 saved scalers at250/750/1250 are ALL65536 with matching growth trackers and receipts. Large's problem cannot be attributed simply to initial65536; FP32/pretrained evidence and unresolved AMP interactions must be distinguished.
6. Review the historical issue link in the diagnostic report, specifically maintainer/user comments dated10–14October2022. Do not assume an old workaround is a proven fix here.

Independent GPT tasks: challenge the BOS mechanism with these records and frozen source; rank causes; distinguish symptom/mechanism/causal repair; explain any residual repetition/Core/grounding defects that BOS alone cannot establish. Recommend only the smallest diagnostic/protocol PREPARATION next step. Do not authorize or execute more training, decoding changes, generation, Teacher calls, held-out access or RO/U. Frozen data/targets/references and D051FAIL remain unchanged. Researcher decision is pending.

Then follow README_FIRST.md and GPT_REVIEW_REQUEST.md for source-grounded 48-output quality review. Treat all prior AI assessments as provisional. Full checkpoints are separately preserved locally; this remains a lightweight review archive, not a standalone executable training workspace.
'''
        put('V2_READ_FIRST.md',readme.encode('utf-8'))
        z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=manifest),indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
    receipt=dict(archive=str(out),bytes=out.stat().st_size,sha256=sha(out),members=len(manifest)+1,
        integrity='ALL_MEMBER_SHA256_AND_CRC_PASS',new_training=False,new_generation=False)
    write(DEST/'GPT_REVIEW_V2_RECEIPT.json',receipt)
    out.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+out.name+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
