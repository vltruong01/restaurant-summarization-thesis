"""Create a compact cloud E0 package from unchanged original24 photos/eight review sets."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
OUT=PARENT/'cloud_handoff_20261006'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    package=OUT/'e0_cloud_pilot'
    if package.exists():raise RuntimeError('Existing cloud package; preserve it and use a new version rather than overwriting.')
    package.mkdir(parents=True)
    plan=read(PARENT/'pilot_photo_plan.json')
    parent_identity=read(PARENT/'pilot_identity.json')
    assert len(plan)==24 and len({p['business_id'] for p in plan})==8
    assert {p['split'] for p in plan}=={'train','validation'}
    business_ids={p['business_id'] for p in plan}
    cfg=yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))
    cfg['output_directory']='outputs/stage_e/e0_cloud_pilot_v1'
    cfg['isolated_environment']='.venv-cloud-e0'
    result=package/cfg['output_directory']
    included=[]
    def copy(name):
        destination=package/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
        assert sha(ROOT/name)==sha(destination)
        included.append(name)
    for name in ['scripts/blip2_caption_pilot.py','scripts/validate_caption_pilot.py',
                 'scripts/render_caption_pilot_gallery.py','configs/research_pending.yaml',
                 'configs/image_overlap_evaluation_policy.yaml','requirements/caption_pilot.in']:
        copy(name)
    for path in sorted((ROOT/'assets/tokenizers/led-base-16384').iterdir()):
        if path.is_file():copy(path.relative_to(ROOT).as_posix())
    for photo in plan:
        copy(photo['image_path'])
        assert sha(package/photo['image_path'])==photo['raw_sha256']
    selections=[]
    source=ROOT/'outputs/stage_c/review_selection_source_length_expanded_v1/selection_runs.jsonl'
    selection_name=source.relative_to(ROOT).as_posix()
    target=package/selection_name
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open(encoding='utf-8') as original,target.open('w',encoding='utf-8',newline='') as subset:
        for line in original:
            row=json.loads(line)
            if row['business_id'] in business_ids and row['method']=='v4_2_length_balanced' and row['review_budget']==4000:
                selections.append(row);subset.write(line)
    assert len(selections)==8
    included.append(selection_name)
    needed={rid for s in selections for rid in s['selected_review_ids']}
    found=set()
    for split in ['train','validation']:
        name=f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl'
        target=package/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with (ROOT/name).open(encoding='utf-8') as original,target.open('w',encoding='utf-8',newline='') as subset:
            for line in original:
                row=json.loads(line)
                if row['review_id'] in needed:
                    assert row['review_id'] not in found and row['business_id'] in business_ids
                    found.add(row['review_id']);subset.write(line)
        included.append(name)
    assert found==needed
    name='data/processed/yelp_2300_v1/businesses.jsonl'
    target=package/name
    seen=set()
    with (ROOT/name).open(encoding='utf-8') as original,target.open('w',encoding='utf-8',newline='') as subset:
        for line in original:
            row=json.loads(line)
            if row['business_id'] in business_ids:seen.add(row['business_id']);subset.write(line)
    assert seen==business_ids
    included.append(name)
    config_name='configs/blip2_caption_feasibility_pilot.yaml'
    (package/config_name).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    included.append(config_name)
    save(result/'pilot_photo_plan.json',plan)
    save(result/'model_revision.json',read(PARENT/'model_revision.json'))
    included.extend([(result/'pilot_photo_plan.json').relative_to(package).as_posix(),
                     (result/'model_revision.json').relative_to(package).as_posix()])
    (package/'setup.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_TRUSTED_HOST PIP_FIND_LINKS
python -m venv --system-site-packages .venv-cloud-e0
.venv-cloud-e0/bin/python -m pip install --index-url https://pypi.org/simple -r requirements/caption_pilot.in
.venv-cloud-e0/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0 torchvision==0.16.0
.venv-cloud-e0/bin/python -c "import torch,transformers; assert torch.cuda.is_available(); assert transformers.__version__=='4.51.3'; print(torch.__version__,torch.cuda.get_device_name(0))"
''',encoding='utf-8')
    (package/'run_one_photo.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=4
.venv-cloud-e0/bin/python scripts/blip2_caption_pilot.py preflight --execution-origin agent
.venv-cloud-e0/bin/python scripts/blip2_caption_pilot.py generate --max-new-photos 1 --allow-model-download --execution-origin agent
''',encoding='utf-8')
    (package/'resume_24_after_review.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=4
.venv-cloud-e0/bin/python scripts/blip2_caption_pilot.py preflight --execution-origin agent
.venv-cloud-e0/bin/python scripts/blip2_caption_pilot.py generate --allow-model-download --execution-origin agent
.venv-cloud-e0/bin/python scripts/blip2_caption_pilot.py report --execution-origin agent
.venv-cloud-e0/bin/python scripts/validate_caption_pilot.py
''',encoding='utf-8')
    (package/'README.md').write_text('''# E0 cloud pilot only — not a training dataset

This package contains the unchanged24train/validation photos,8accepted4000-token review sets and original complete selected review lines. It is a SMALL SOURCE SLICE with full parent provenance, not the2300-business cohort/final manifest. No test photos, teacher targets or LED training.

Model Salesforce/blip2-opt-2.7b-coco; pinned commit in model_revision.json; beam3, FP16 with retainedFP32 modules, no quantization, identical caption prompts/surface diagnostics. Source budgets/K/final selector remain provisional. Running on cloud does not establish RTX3050 local feasibility.

Use a CUDA-enabled Linux PyTorch environment, preferably the existingTorch2.1.0/CUDA11.8 runtime to avoid anotherTorch download. A singleGPU with>=16GiBVRAM,>=16GiBavailableRAM and>=60GiBdisk is an engineering target, not a validated production/A2 profile. Check host download bandwidth and actual total price before renting; no instance was rented or data uploaded by package creation.

Extract e0_cloud_pilot.zip and enter e0_cloud_pilot, then:

```bash
bash setup.sh
bash run_one_photo.sh
```

Do NOT run backend `prepare`: the source is intentionally sliced and the fixed parent photo plan/child identity were prepared locally. The first command runs a resource preflight; the second downloads pretrained weights ON THE SERVER and generates ONE new photo, releases model and stops pending picture-caption/VRAM/time review. Server download speed is not guaranteed by packaging.

After inspecting the first actual output, resume the SAME24-photo plan:

```bash
bash resume_24_after_review.sh
```

Return outputs/stage_e/e0_cloud_pilot_v1 and environment_freeze.txt for local notebook/quality review. Do not mix these cloud runtime identities/device maps with local checkpoints. CanonicalYelp datasets/splits stay unchanged. Capacity reports preserve complete reviews and actual raw caption strings. A passed24-output validator is not productioncaptionquality/A2 or formalhuman evaluation.
''',encoding='utf-8')
    included.extend(['setup.sh','run_one_photo.sh','resume_24_after_review.sh','README.md'])
    manifest={'scope':'cloud_E0_source_slice_only_not_final_dataset',
              'businesses':8,'photos':24,'selected_complete_reviews':len(needed),'test_photos':0,
              'parent_identity_sha256':parent_identity['identity_sha256'],
              'parent_source_sha256':parent_identity['source_sha256'],
              'packaged_file_sha256':{name:sha(package/name) for name in included},
              'photo_plan_same_as_parent':True,'source_texts_unedited':True,
              'original_model_generation_parameters_preserved':True,'rental_upload_training_started':False}
    save(package/'cloud_manifest.json',manifest)
    identity={'config_sha256':sha(package/config_name),
              'source_sha256':{**manifest['packaged_file_sha256'],'cloud_manifest.json':sha(package/'cloud_manifest.json')},
              'plan_sha256':canonical(plan),'image_overlap_policy':parent_identity['image_overlap_policy'],
              'parent_identity_sha256':parent_identity['identity_sha256'],
              'execution':'cloud_E0_SOURCE_SLICE_not_full_cohort',
              'source_hash_scope':'packaged_subset_and_exact_parent_provenance'}
    identity['identity_sha256']=canonical(identity)
    save(result/'pilot_identity.json',identity)
    archive=OUT/'e0_cloud_pilot.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(package.rglob('*')):
            if path.is_file():stream.write(path,'e0_cloud_pilot/'+path.relative_to(package).as_posix())
    result_report={'status':'cloud_package_prepared_not_gpu_executed','archive':archive.relative_to(ROOT).as_posix(),
                   'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
                   'businesses':8,'photos':24,'complete_reviews':len(needed),
                   'first_run_limit':1,'parent_plan_unchanged':True,'source_texts_unedited':True,
                   'rental_or_upload_started':False,'actual_model_loads':0,'actual_captions':0,
                   'full_training_manifest':False,'cloud_runtime_identity':identity['identity_sha256']}
    save(OUT/'package_preparation_report.json',result_report)
    print(json.dumps(result_report,indent=2))


if __name__=='__main__':main()
