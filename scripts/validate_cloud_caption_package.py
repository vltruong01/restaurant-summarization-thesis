"""Check the compact cloud package against parent source records without model execution."""
from pathlib import Path
import hashlib
import json
import zipfile
import yaml
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1'
OUT=PARENT/'cloud_handoff_20261006'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    package=OUT/'e0_cloud_pilot'
    cfg=yaml.safe_load((package/'configs/blip2_caption_feasibility_pilot.yaml').read_text())
    original_cfg=yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text())
    expected_cfg=dict(original_cfg)
    expected_cfg.update(output_directory='outputs/stage_e/e0_cloud_pilot_v1',isolated_environment='.venv-cloud-e0')
    assert cfg==expected_cfg
    child=package/cfg['output_directory']
    manifest=read(package/'cloud_manifest.json')
    identity=read(child/'pilot_identity.json')
    parent_identity=read(PARENT/'pilot_identity.json')
    assert manifest['parent_identity_sha256']==parent_identity['identity_sha256']
    assert manifest['parent_source_sha256']==parent_identity['source_sha256']
    for name,value in manifest['packaged_file_sha256'].items():assert sha(package/name)==value,name
    for name,value in identity['source_sha256'].items():assert sha(package/name)==value,name
    plan=read(child/'pilot_photo_plan.json')
    assert plan==read(PARENT/'pilot_photo_plan.json')
    assert len(plan)==24 and len({p['business_id'] for p in plan})==8
    assert all(p['split'] in ['train','validation'] for p in plan)
    for photo in plan:assert sha(package/photo['image_path'])==sha(ROOT/photo['image_path'])==photo['raw_sha256']
    selections=[json.loads(line) for line in (package/'outputs/stage_c/review_selection_source_length_expanded_v1/selection_runs.jsonl').read_text().splitlines()]
    assert len(selections)==8
    needed={rid for s in selections for rid in s['selected_review_ids']}
    subset={}
    for split in ['train','validation']:
        name=f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl'
        for line in (package/name).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            assert row['review_id'] not in subset
            subset[row['review_id']]=row
        with (ROOT/name).open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['review_id'] in needed:
                    assert subset[row['review_id']]==row
    assert set(subset)==needed and len(needed)==manifest['selected_complete_reviews']
    tokenizer=Tokenizer.from_file(str(package/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    for choice in selections:
        selected=[subset[rid] for rid in choice['selected_review_ids']]
        assert all(r['business_id']==choice['business_id'] for r in selected)
        assert selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
        block='\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i,r in enumerate(selected,1))
        assert len(tokenizer.encode(block,add_special_tokens=True).ids)==choice['tokens_after_budget_enforcement']
        assert choice['review_budget']==4000 and choice['method']=='v4_2_length_balanced'
    for path in (package/'scripts').glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    with zipfile.ZipFile(OUT/'e0_cloud_pilot.zip') as archive:
        assert archive.testzip() is None
        for path in package.rglob('*'):
            if path.is_file():
                key='e0_cloud_pilot/'+path.relative_to(package).as_posix()
                assert hashlib.sha256(archive.read(key)).hexdigest()==sha(path)
    result={'status':'PASS','scope':'cloud_pilot_package_data_integrity_only_not_GPU_execution',
            'businesses':8,'photos':24,'original_complete_reviews_equal':len(needed),
            'review_token_counts_equal':8,'generation_and_research_config_unchanged':True,
            'parent_plan_ids_order_equal':True,'archive_contents_verified':True,
            'actual_model_loads':0,'captions_generated':0,'rental_upload_started':False,
            'source_dataset_not_modified':True,'full_training_manifest':False}
    (OUT/'package_validation_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
