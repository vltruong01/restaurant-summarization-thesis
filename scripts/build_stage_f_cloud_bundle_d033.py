"""Required-file portable ZIP with byte identity, no model jobs, upload, rental or local resume."""
from pathlib import Path
import hashlib,json,os,time,zipfile
import psutil
import blip2_caption_pilot as legacy
from stage_f_cloud_d033 import OUT as CLOUD,LOCAL,GEN,prepare
from verify_stage_f_cloud_package_d033 import safe_relative,verify

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/cloud_handoff/stage_f_d033_v1'
ARCHIVE=DEST/'stage_f_d033_vast_bundle.zip'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert not ARCHIVE.exists(),'Preserve an existing export; use a new version for a new state'
    for p in psutil.process_iter(['name','cmdline']):
        if any(a.replace('\\','/') in ['scripts/generate_stage_f_production_d033.py','scripts/run_stage_f_production_worker_d033.py'] for a in p.info['cmdline'] or []):
            raise RuntimeError('Local production must remain stopped while exporting immutable checkpoints')
    prepared=prepare();files=set()
    def add(name):
        name=safe_relative(name);p=ROOT/name;assert p.is_file(),f'Required file missing {name}';files.add(name)
    def directory(relative):
        for p in (ROOT/relative).rglob('*'):
            if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts and '.cache' not in p.parts and p.suffix not in ['.pyc','.lock','.tmp']:
                add(p.relative_to(ROOT).as_posix())
    protected=legacy.read(ROOT/'outputs/stage_f/methodology_amendment_d033/protected_before_sha256.json')
    for name,expected in protected.items():
        assert digest(ROOT/name)==expected,f'Protected history changed: {name}';add(name)
    for relative in ['scripts','configs','specs','thesis_sdd_docs','requirements','cloud/stage_f_d033','assets/tokenizers']:
        directory(relative)
    add('DECISIONS.md');add('notebooks/00_thesis_dataset_filtering_review_selection.ipynb')
    for name in ['businesses.jsonl','business_split.csv','checksums.json','dataset_manifest.json','manifest.json']:
        p=ROOT/'data/processed/yelp_2300_v1'/name
        if p.is_file():add(p.relative_to(ROOT).as_posix())
    for name in ['business_split.csv','checksums.json','dataset_manifest.json','manifest.json']:
        p=ROOT/'data/processed/yelp_2300_english_v1'/name
        if p.is_file():add(p.relative_to(ROOT).as_posix())
    for split in ['train','validation','test']:add(f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl')
    add('outputs/stage_c/led_source_token_allocation_audit_v1/selection_reference.jsonl')
    for relative in ['outputs/stage_f/methodology_amendment_d033','outputs/stage_f/production_caption_construction_d033_v1']:
        directory(relative)
    for p in (ROOT/'outputs/stage_f/methodology_amendment_d032').iterdir():
        if p.is_file():add(p.relative_to(ROOT).as_posix())
    photo_plan=legacy.read(CLOUD/'complete_photo_plan.json')
    for photo in photo_plan:
        name='data/processed/yelp_2300_v1/'+photo['relative_path'];add(name)
        assert digest(ROOT/name)==photo['raw_sha256'],photo['photo_id']
    reuse=legacy.read(CLOUD/'reused_caption_sources.json')
    for item in reuse.values():add(item['checkpoint_path'])
    model_files=legacy.read(ROOT/'outputs/stage_f/validation_calibration_v1/model_files_report.json')
    for key in ['CLIP_local_safetensors','SBERT_local_snapshot']:directory(model_files[key])
    # Full BLIP files are downloaded remotely: bind original cached bytes, do not ship15.5GBweights.
    model='Salesforce/blip2-opt-2.7b-coco';revision='cda95b9319b722f79c9451a3d8ff92eea02048dc'
    snapshot=ROOT/'assets/model_cache/models--Salesforce--blip2-opt-2.7b-coco/snapshots'/revision
    pinned={}
    for p in sorted(snapshot.iterdir()):
        if p.is_file():pinned[p.name]=digest(p)
    assert 'model.safetensors.index.json' in pinned and sum(p.stat().st_size for p in snapshot.iterdir() if p.is_file())>15_000_000_000
    legacy.save(CLOUD/'blip_weight_files.json',{'model_id':model,'revision':revision,'sha256':pinned,'download_remotely_not_in_archive':True})
    seal_names=['scripts/stage_f_cloud_d033.py','configs/stage_f_cloud_d033.yaml','requirements/stage_f_cloud_d033.in',
                'scripts/generate_stage_f_validation_captions.py','scripts/blip2_caption_pilot.py','scripts/caption_basic_quality.py',
                'scripts/construct_stage_f_production_d033.py','scripts/prepare_stage_f_production_reviews_d033.py',
                'scripts/stage_f_d033_protocol.py','scripts/stage_f_active_protocol.py','scripts/stage_f_math.py',
                'scripts/stage_f_score_validation.py','scripts/apply_stage_f_frozen_filters.py','scripts/source_budget_audit_template.py',
                'configs/stage_f_validation_protocol.yaml','configs/basic_caption_quality.yaml','configs/led_source_template_d032.json',
                'outputs/stage_f/methodology_amendment_d033/frozen_production_method.json',
                (CLOUD/'complete_photo_plan.json').relative_to(ROOT).as_posix(),
                (CLOUD/'reused_caption_sources.json').relative_to(ROOT).as_posix(),
                (CLOUD/'blip_weight_files.json').relative_to(ROOT).as_posix(),
                (GEN/'pilot_photo_plan.json').relative_to(ROOT).as_posix(),
                (GEN/'pilot_identity.json').relative_to(ROOT).as_posix()]
    legacy.save(CLOUD/'execution_seal.json',{n:digest(ROOT/n) for n in seal_names})
    directory(CLOUD.relative_to(ROOT).as_posix())
    # Include every source referenced by both execution identities/seals and the historical final-validation seal.
    for path in [LOCAL/'production_adapter_seal.json',GEN/'pilot_identity.json',CLOUD/'execution_seal.json']:
        value=legacy.read(path);value=value.get('source_sha256',value)
        for name in value:add(name)
    inventory={}
    for i,name in enumerate(sorted(files),1):
        p=ROOT/name;inventory[name]={'bytes':p.stat().st_size,'sha256':digest(p)}
        if i%5000==0:print(f'Payload inventory {i}/{len(files)}',flush=True)
    manifest={'status':'PORTABLE_D033_PACKAGE_NOT_GPU_VALIDATED','cohort_businesses':2300,'full_photo_pool':30027,
              'reused_caption_records':2709,'pending_generation':27318,'protected_history_files':len(protected),
              'package_root':'yelp_dataset','files':inventory,'BLIP_download_pinned_separately':True,
              'scientific_method_changed':False,'local_jobs_restarted':False,'cloud_runtime_validated':False,
              'teacher_calls':False,'LED_training':False,'credentials_included':False}
    manifest_path=ROOT/'cloud/stage_f_d033/package_manifest.json';legacy.save(manifest_path,manifest)
    DEST.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE.with_suffix('.zip.partial'),'w',allowZip64=True) as z:
        for i,name in enumerate(sorted(files|{'cloud/stage_f_d033/package_manifest.json'}),1):
            p=ROOT/name
            compressed=p.suffix.lower() not in ['.jpg','.jpeg','.png','.zip','.safetensors','.bin']
            z.write(p,'yelp_dataset/'+name,compress_type=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED,compresslevel=3 if compressed else None)
            if i%5000==0:print(f'ZIP packed {i}/{len(files)+1}',flush=True)
    ARCHIVE.with_suffix('.zip.partial').replace(ARCHIVE)
    integrity=verify(archive=ARCHIVE)
    for name,expected in protected.items():assert digest(ROOT/name)==expected
    report={'status':'PASS_COMPLETE_PORTABLE_PACKAGE_NO_GPU_EXECUTION','archive':ARCHIVE.relative_to(ROOT).as_posix(),
            'archive_bytes':ARCHIVE.stat().st_size,'archive_GiB':ARCHIVE.stat().st_size/2**30,'archive_sha256':digest(ARCHIVE),
            'package_manifest_sha256':digest(manifest_path),'verification':integrity,
            'reuse_by_split':prepared['reused_by_split'],'pending_by_split':prepared['pending_by_split'],
            'preserved_history_files':len(protected),'all_source_images_and_captions_verified':True,
            'frozen_scientific_core_unchanged':True,'local_jobs_still_stopped':True,'cloud_gpu_test_pending':True}
    legacy.save(DEST/'package_report.json',report)
    (DEST/'stage_f_d033_vast_bundle.zip.sha256').write_text(report['archive_sha256']+'  '+ARCHIVE.name+'\n',encoding='ascii')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
