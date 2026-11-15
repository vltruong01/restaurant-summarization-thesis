"""Materialize a hash-bound finite Stage O inference payload; no frozen input edits."""
from pathlib import Path
import ast,datetime,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/stage_o/tripadvisor_external_v1'
E=O/'RTX3080_EXECUTION_V1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    E.mkdir(exist_ok=True)
    files=set();todo=['stage_o_frozen_pipeline_v1','generate_stage_f_validation_captions','stage_f_score_validation','apply_stage_f_frozen_filters','s1000_final_eval_core_v1']
    seen=set()
    while todo:
        mod=todo.pop();p=ROOT/'scripts'/(mod+'.py')
        if mod in seen or not p.is_file():continue
        seen.add(mod);files.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(n,ast.ImportFrom) and n.module:todo.append(n.module)
            elif isinstance(n,ast.Import):todo.extend(x.name for x in n.names)
    f=json.loads((O/'COHORT15_FREEZE.json').read_text(encoding='utf8'))
    files.add(O/'COHORT15_FREEZE.json')
    for x in f['artifact_bindings']:files.add(O/x['file'])
    for row in f['restaurants']:
        files.add(O/row['source_file'])
        for photo in json.loads((O/row['source_file']).read_text(encoding='utf8'))['photos']:files.add(O/photo['file'])
    for folder in ['PREPROCESSING','PREPARED_CAPTION_JOB']:
        files.update(p for p in (O/folder).rglob('*') if p.is_file())
    files.add(O/'SOURCE_IMAGE_DIAGNOSTICS.json')
    files.add(O/'FROZEN_MODEL_AND_DECODER_BINDINGS.json')
    lockpath=ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json'
    lock=json.loads(lockpath.read_text(encoding='utf8'));files.add(lockpath)
    for m in lock['models'].values():
        p=ROOT/m['path'];files.update(p/x for x in ['adapter_model.safetensors','adapter_config.json'])
    for p in (ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384').iterdir():
        if p.is_file():files.add(p)
    for p in (ROOT/'assets/tokenizers/led-base-16384').iterdir():
        if p.is_file():files.add(p)
    files.add(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    for filename in ['stage_f_validation_caption_pool.yaml','stage_f_validation_protocol.yaml','basic_caption_quality.yaml','review_selection_source_length_expanded.yaml','english_eligibility.yaml','led_source_template_d032.json']:
        files.add(ROOT/'configs'/filename)
    files.add(ROOT/'outputs/stage_f/validation_calibration_v1/model_files_report.json')
    for folder in ['assets/model_cache/models--Salesforce--blip2-opt-2.7b-coco/snapshots/cda95b9319b722f79c9451a3d8ff92eea02048dc',
                   'assets/model_cache/verified_conversions/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                   'assets/model_cache/stage_f_native_models/sentence-transformers--all-mpnet-base-v2/e8c3b32edf5434bc2275fc9bab85f82640a19130']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.stat().st_size<20_000_000)
    now=datetime.datetime.now(datetime.timezone.utc)
    auth=dict(scope='STAGE_O_FROZEN_CAPTION_AND_LED_INFERENCE_ONLY',execution_approved=True,
              authorization_source='Researcher message: rented RTX3080, provided ssh -p 55308 root@1.208.108.242 and requested connect and continue.',
              price_reporting='Researcher requested no cost reporting in methodology; no hourly price provided.',
              issued_at_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(hours=24)).isoformat(),
              cohort_freeze_sha256=sha(O/'COHORT15_FREEZE.json'),host='1.208.108.242',port=55308,
              instance_id='59069440',container='77eb3c4ad31e',GPU_UUID='GPU-953a5550-f5a3-b664-9017-0acd8461f37c',
              training_allowed=False,Qwen_allowed=False,Teacher_allowed=False,human_ratings_allowed=False,
              lifecycle_actions_allowed=False,automatic_rental=False,successful_output_records=45,native_LED_components=75,
              BLIP_photos=5143,precision='Frozen FP16 BLIP + GPU/CPU offload; original FP32 LED parameters / FP16 autocast',
              runtime_limit_note='Finite single-instance execution; conservative 24-hour internal maximum, no automatic renewal or instance switching.',
              operational_memory='BLIP GPU weight budget capped at the previously successful 4GiB offload profile to preserve 3080 activation headroom; unchanged weights, precision and generation. CPU budget bounded by actual cgroup available RAM.')
    ap=E/'EXECUTION_AUTHORIZATION.json';ap.write_text(json.dumps(auth,indent=2)+'\n',encoding='utf8');files.add(ap)
    manifest=[dict(file=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]
    mp=E/'PAYLOAD_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    archive=E/'STAGE_O_INPUTS_AND_LED_ASSETS.tar'
    with tarfile.open(archive,'w') as t:
        for p in sorted(files):t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        t.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    print(json.dumps(dict(files=len(manifest),bytes=archive.stat().st_size,sha256=sha(archive),path=str(archive)),indent=2))
if __name__=='__main__':main()
