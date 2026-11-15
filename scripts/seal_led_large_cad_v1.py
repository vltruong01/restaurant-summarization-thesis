"""Lock final tested CAD implementation and local inference receipt, no model load."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib
import json
import hashlib
import shutil
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/stage_i/led_large_cad_v1'
WS=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
E=ROOT/'outputs/stage_i/led_large_cad_v1_local_execution_v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    assert not E.exists(),'One fresh local attempt only'
    old=read(P/'BINDINGS.json');write(P/'PRE_EXECUTION_PREPARATION_BINDINGS.json',old)
    c=read(P/'CONFIG.json')
    c['parent_code_bindings'].update({str(x.relative_to(WS)).replace('\\','/'):sha(x) for x in (WS/'scripts').glob('*.py')})
    c['runtime_source_hashes']={n:sha(importlib.import_module(n).__file__) for n in
        ['transformers.models.led.modeling_led','transformers.generation.utils','peft.peft_model']}
    write(P/'CONFIG.json',c)
    before=read(ROOT/'outputs/stage_i/led_large_nbest_v2_execution_v1/downloaded/CE_V2_WORKSPACE/runs/F_CE3_nbest_training_v2/CANDIDATES.json')
    write(P/'HISTORICAL_TRAIN_N1_REFERENCE.json',[dict(business_id=x['business_id'],source_sha256=x['source_sha256'],
        checkpoint_sha256=x['checkpoint_sha256'],single=x['single']) for x in before])
    (P/'PREPARATION_NOTES.md').write_text('''# Preparation evidence and limits

CPU suite:10 tests passed,1 CUDA-specific test intentionally skipped. Then the
explicit local suite passed all11 including CUDA/FP16 reduced random LED. These
are not pretrained LED-large generation/quality evidence. Source re-tokenization
matches all12 frozen inputs with UTF-8, no normalization/truncation/source edits.

Two offline preparation errors were corrected before the only full-model attempt:
unused bitsandbytes dispatch in the toy PEFT test now uses the SAME exclusion
as the accepted parent loader; a source-audit helper now explicitly reads UTF-8
instead of Windows default cp1252. Original error logs are preserved. Neither
error occurred in historical training or changed frozen artifacts.

All parent Python dependencies and installed LED/generation/PEFT source bytes
are newly bound before inference, without reading validation/test content.
Local hardware is RTX3050 Laptop6GB, not the prior RTX3090. Historical N1 outputs
are preserved for transparency; local numerical/driver differences may change
native beam tie decisions. Same-host alpha0 MUST reproduce local vanilla exactly.
''',encoding='utf-8')
    bound=[ROOT/'scripts/led_large_cad_v1.py',ROOT/'scripts/run_led_large_cad_v1.py',ROOT/'tests/test_led_large_cad_v1.py']
    bound += sorted(x for x in P.rglob('*') if x.is_file() and x.name not in {'PREFLIGHT_MANIFEST.json','BINDINGS.json'})
    write(P/'PREFLIGHT_MANIFEST.json',dict(files={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in bound}))
    b=dict(config_sha256=sha(P/'CONFIG.json'),runner_sha256=sha(ROOT/'scripts/run_led_large_cad_v1.py'),
        cad_module_sha256=sha(ROOT/'scripts/led_large_cad_v1.py'),preflight_sha256=sha(P/'PREFLIGHT_MANIFEST.json'))
    write(P/'BINDINGS.json',b)
    E.mkdir()
    now=datetime.now(timezone.utc)
    write(E/'LOCAL_INFERENCE_AUTHORIZATION.json',dict(experiment='LED_LARGE_CAD_V1',bindings=b,
        inference_approved=True,training_authorized=False,validation_access=False,held_out_access=False,
        GPU_name='NVIDIA GeForce RTX 3050 6GB Laptop GPU',
        authorization_source='researcher CAD12TRAIN inference pilot instruction + explicit permission to use local GPU',
        authorized_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=48)).isoformat(),
        runtime_cap_seconds=3600,generation_calls_max=37,automatic_retry=False,
        no_precision_or_beam_fallback=True,weights_unchanged_required=True))
    print(json.dumps(b,indent=2))
if __name__=='__main__':main()
