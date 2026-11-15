"""Build a self-contained offline pilot viewer from the validated export; no API calls."""
from pathlib import Path
import base64
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/stage_g/chatgpt_stage_g_pilot_review_bundle.zip'
OUT = ROOT / 'outputs/stage_g/pilot_review_ui_v1'
ASSETS = ROOT / 'scripts/stage_g_review_ui'
PREFIX = 'chatgpt_stage_g_pilot_review_bundle/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    export_report = json.loads(SOURCE.with_name('chatgpt_stage_g_pilot_review_bundle_report.json').read_text(encoding='utf-8'))
    assert sha(SOURCE) == export_report['zip_sha256']
    with zipfile.ZipFile(SOURCE) as archive:
        manifest = json.loads(archive.read(PREFIX + 'export_manifest.json'))
        for relative, expected in manifest['file_sha256'].items():
            assert hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() == expected, relative
        data = json.loads(archive.read(PREFIX + 'cases.json'))
        audit = json.loads(archive.read(PREFIX + 'provenance/pilot_integrity_and_tokens_report.json'))
        images = {}
        for case in data['businesses']:
            for condition in case['caption_conditions'].values():
                for caption in condition['captions']:
                    if caption['photo_id'] not in images:
                        original = archive.read(PREFIX + caption['image_file'])
                        assert hashlib.sha256(original).hexdigest() == caption['original_image_sha256']
                        images[caption['photo_id']] = 'data:image/jpeg;base64,' + base64.b64encode(original).decode('ascii')
    wanted = {b['business_id'] for b in data['businesses']}
    metadata = {}
    metadata_source_sha = {}
    # Read only the permitted train/validation source manifests, never the held-out test.
    for split in ['train', 'validation']:
        path = ROOT / 'outputs/stage_f/production_caption_construction_d033_colab_v1/manifests' / split / 'reviews_only_sources.jsonl'
        metadata_source_sha[path.relative_to(ROOT).as_posix()] = sha(path)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['business_id'] not in wanted:
                    continue
                head = row['source'].split('\n\n[REVIEWS]', 1)[0]
                fields = dict(l.split('=', 1) for l in head.splitlines() if '=' in l)
                assert all(k in fields for k in ['name', 'city', 'state'])
                metadata[row['business_id']] = fields
    assert set(metadata) == wanted and len(images) == 128
    data.update({'metadata': metadata, 'images': images,
                 'target_lengths': {r['business_id']: r for r in audit['business_details']},
                 'source_zip_sha256': export_report['zip_sha256']})
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (ASSETS / 'index.html').read_text(encoding='utf-8')
    page = template.replace('__STYLES__', (ASSETS / 'styles.css').read_text(encoding='utf-8'))
    page = page.replace('__APP__', (ASSETS / 'app.js').read_text(encoding='utf-8'))
    page = page.replace('__DATA__', encoded)
    assert not any(marker in page for marker in ['__DATA__', '__STYLES__', '__APP__'])
    # Validate exact embedded data, images and scientific artifacts before saving a viewer.
    start = page.index('<script id="review-data" type="application/json">') + len('<script id="review-data" type="application/json">')
    end = page.index('</script>', start)
    restored = json.loads(page[start:end])
    assert restored == data
    for photo_id, url in restored['images'].items():
        assert url.startswith('data:image/jpeg;base64,')
        assert hashlib.sha256(base64.b64decode(url.split(',',1)[1])).hexdigest() == manifest['file_sha256']['images/' + photo_id + '.jpg']
    for relative, expected in audit['artifact_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(SOURCE) == export_report['zip_sha256']
    output = OUT / 'dist/index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    report = {'status':'OFFLINE_VIEWER_BUILD_INTEGRITY_PASS', 'businesses':20, 'condition_targets':60,
        'selected_reviews':557, 'original_images_embedded':128, 'images_resized':False,
        'all_embedded_case_data_semantically_identical':True, 'existing_scientific_artifacts_unchanged':True,
        'source_zip_unchanged':True, 'source_zip_sha256':export_report['zip_sha256'],
        'metadata_source_sha256':metadata_source_sha, 'teacher_API_calls':0,
        'label_or_method_changes':False, 'notes_storage':'browser_local_only_not_scientific_annotations',
        'standalone_no_server_or_external_assets_required':True,
        'html_path':output.relative_to(ROOT).as_posix(), 'html_bytes':output.stat().st_size,
        'html_sha256':sha(output)}
    (OUT / 'build_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
