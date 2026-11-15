"""Transport immutable CAD V1 and its exact inference dependencies; no data search."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,tarfile
R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/led_large_cad_v1'
E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1'
WS=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
CP=R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    E.mkdir(parents=True,exist_ok=False)
    bindings=read(P/'BINDINGS.json');manifest=read(P/'PREFLIGHT_MANIFEST.json')['files']
    for rel,h in manifest.items():assert sha(R/rel)==h,rel
    c=read(P/'CONFIG.json');files={}
    def add(path,rel):
        assert path.is_file() and not path.is_symlink(),path
        if rel in files:assert files[rel]==path
        files[rel]=path
    for rel in manifest:add(R/rel,rel)
    for name in ['PREFLIGHT_MANIFEST.json','BINDINGS.json']:add(P/name,'outputs/stage_i/led_large_cad_v1/'+name)
    # Existing parent import/loader dependencies, with original bytes/paths.
    for rel,h in c['parent_code_bindings'].items():
        assert sha(WS/rel)==h,rel;add(WS/rel,'parent/'+rel)
    for rel,h in c['model_files'].items():assert sha(WS/rel)==h;add(WS/rel,'parent/'+rel)
    for path in (WS/'configs').glob('*.json'):add(path,'parent/'+path.relative_to(WS).as_posix())
    for rel in ['outputs/stage_i/led_large_preparation_v1/remote_metadata/official/config.json',
                'outputs/stage_i/led_large_preparation_v1/META_LORA_INVENTORY.json',
                'outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
                'outputs/stage_i/led_large_colab_preparation_v2/INITIALIZATION_HASHES.json']:
        add(WS/rel,'parent/'+rel)
    for rel in ['scripts/led_large_nbest_rng_v2.py','scripts/stage_i_led_large_nbest_v1.py']:
        add(R/rel,rel)
    proof=read(CP/'checkpoint_integrity.json')
    for name,h in proof['files'].items():assert sha(CP/name)==h;add(CP/name,'checkpoint-750/'+name)
    add(CP/'checkpoint_integrity.json','checkpoint-750/checkpoint_integrity.json')
    # Parent data/targets/images/validation/test are intentionally absent.
    exported={rel:dict(sha256=sha(path),bytes=path.stat().st_size) for rel,path in files.items()}
    save(E/'TRANSFER_MANIFEST.json',dict(files=exported,validation_or_test_transferred=False,
        original_Teacher_targets_transferred=False,TRAIN_source_cases=12))
    now=datetime.now(timezone.utc)
    save(E/'VAST_CAD_RESEARCHER_AUTHORIZATION.json',dict(experiment='LED_LARGE_CAD_V1',bindings=bindings,
        inference_approved=True,training_authorized=False,validation_access=False,held_out_access=False,
        GPU_name='NVIDIA GeForce RTX 3090',authorization_source='current researcher: connect new instance and run',
        authorized_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=48)).isoformat(),
        generation_calls_max=37,automatic_retry=False,runtime_cap_seconds=3600,
        fresh_reexecution_explicitly_approved=True,local_resource_failure_preserved=True,
        no_precision_or_beam_fallback=True,weights_unchanged_required=True))
    archive=E/'CAD_V1_FRESH_HOST_PAYLOAD.tar'
    with tarfile.open(archive,'x') as t:
        for rel,path in files.items():t.add(path,arcname=rel,recursive=False)
        t.add(E/'TRANSFER_MANIFEST.json',arcname='TRANSFER_MANIFEST.json')
    save(E/'TRANSFER_RECEIPT.json',dict(files=len(files),archive_bytes=archive.stat().st_size,sha256=sha(archive),
        model_sha256=c['model_files']['assets/models/led-large-16384/pytorch_model.bin'],
        local_failure_review_zip_sha256=sha(R/'outputs/cloud_handoff/LED_LARGE_CAD_V1_LOCAL_RESOURCE_STOP_REVIEW.zip')))
    print(json.dumps(read(E/'TRANSFER_RECEIPT.json'),indent=2))
if __name__=='__main__':main()
