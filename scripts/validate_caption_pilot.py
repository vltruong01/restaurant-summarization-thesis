"""Validate the small E0 output against original photos/reviews and slow LED tokenization."""
from pathlib import Path
import csv
import hashlib
import json
import re
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))
    out = ROOT/cfg['output_directory']
    generation = read(out/'generation_report.json')
    assert generation['status'] == 'raw_pilot_complete_pending_review'
    identity = read(out/'pilot_identity.json')
    for name, expected in identity['source_sha256'].items():
        assert digest(ROOT/name) == expected, name
    plan = read(out/'pilot_photo_plan.json')
    assert len(plan) == 24 and len({p['business_id'] for p in plan}) == 8
    assert {p['split'] for p in plan} == {'train', 'validation'}
    assert len({p['photo_id'] for p in plan}) == 24
    revision = read(out/'model_revision.json')['resolved_commit']
    load = read(out/'model_load_report.json')
    assert 'disk' not in load['device_map'].values()
    assert load['runtime']['quantization'] == 'none'
    assert load['runtime']['weight_dtype'] == 'float16'
    assert load['runtime']['model_id'] == cfg['model_id']
    assert load['runtime']['resolved_commit'] == revision
    assert cfg['generation']['num_beams'] == 3
    assert cfg['final_manifest'] is False and cfg['teacher_or_LED_training'] is False
    records = []
    for photo in plan:
        assert digest(ROOT/photo['image_path']) == photo['raw_sha256']
        record = read(out/'caption_checkpoints'/(photo['photo_id']+'.json'))
        for key in ['photo_id', 'business_id', 'split', 'raw_sha256', 'image_path']:
            assert record[key] == photo[key], (photo['photo_id'], key)
        assert record['resolved_commit'] == revision
        assert record['runtime_sha256'] == load['runtime_sha256']
        assert record['device_map_sha256'] == load['device_map_sha256']
        assert record['generation_settings'] == cfg['generation']
        assert record['final_caption_set'] is False and record['teacher_target'] is False
        attempts = record['attempts']
        assert 1 <= len(attempts) <= 2
        assert attempts[0]['prompt'] == cfg['generation']['initial_prompt']
        if len(attempts) == 2:
            assert not attempts[0]['surface_eligible']
            assert attempts[1]['prompt'] == cfg['generation']['retry_prompt']
        eligibility = []
        for attempt in attempts:
            text = attempt['raw_caption']
            assert isinstance(text, str) and text == text.strip()
            assert attempt['seconds'] >= 0 and attempt['generated_suffix_tokens'] >= 0
            count = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))
            normal = ' '.join(text.lower().split()).strip(' .!?\n\t')
            flags = []
            if count < cfg['surface_diagnostics']['minimum_words']:
                flags.append('under5_words')
            if not text.endswith(tuple(cfg['surface_diagnostics']['terminal_punctuation'])):
                flags.append('no_terminal_punctuation')
            if normal in cfg['surface_diagnostics']['exact_generic_phrases']:
                flags.append('exact_generic_phrase')
            assert attempt['word_count'] == count
            assert attempt['surface_flags'] == flags
            assert attempt['surface_eligible'] == (not flags)
            eligibility.append(not flags)
        chosen = next((i for i, ok in enumerate(eligibility, 1) if ok), 1)
        assert record['chosen_attempt'] == chosen
        assert record['raw_caption'] == attempts[chosen-1]['raw_caption']
        assert record['surface_eligible'] == eligibility[chosen-1]
        records.append(record)
    business_ids = {p['business_id'] for p in plan}
    sources = read(out/'complete_provisional_source_examples.json')
    assert len(sources) == 16
    assert len({(s['business_id'], s['condition']) for s in sources}) == 16
    needed = {rid for s in sources for rid in s['selected_review_ids']}
    originals = {}
    for split in ['train', 'validation']:
        with (ROOT/f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                review = json.loads(line)
                if review['review_id'] in needed:
                    assert review['review_id'] not in originals
                    originals[review['review_id']] = review
    assert set(originals) == needed
    businesses = {}
    with (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            business = json.loads(line)
            if business['business_id'] in business_ids:
                businesses[business['business_id']] = business
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',
                                              use_fast=False, local_files_only=True)
    assert not tokenizer.is_fast
    for source in sources:
        bid = source['business_id']
        selected = [originals[rid] for rid in source['selected_review_ids']]
        assert 1 <= len(selected) <= 30 and len({r['review_id'] for r in selected}) == len(selected)
        assert all(r['business_id'] == bid for r in selected)
        assert selected == sorted(selected, key=lambda r: (r['date'], r['review_id']))
        block = '\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}"
                            for i, r in enumerate(selected, 1))
        metadata = '[BUSINESS]\n'+'\n'.join(f"{key}={businesses[bid].get(key) or ''}"
                                           for key in ['name', 'categories', 'city', 'state'])
        base = metadata+'\n\n[REVIEWS]\n'+block
        captions = [r for r in records if r['business_id'] == bid]
        if source['condition'].startswith('surface_eligible'):
            captions = [r for r in captions if r['surface_eligible']]
        caption_block = '\n'.join(f"[IMG_CAP_{i}] {r['raw_caption']}" for i, r in enumerate(captions, 1))
        full = base+('\n\n[CAPTIONS]\n'+caption_block if captions else '')
        assert full == source['complete_provisional_source']
        assert source['caption_photo_ids'] == [r['photo_id'] for r in captions]
        assert source['retained_reviews'] == len(selected) and source['captions'] == len(captions)
        assert source['caption_set_final'] is False and source['teacher_or_training_input'] is False
        base_tokens = len(tokenizer.encode(base, add_special_tokens=True, truncation=False))
        full_tokens = len(tokenizer.encode(full, add_special_tokens=True, truncation=False))
        assert source['text_only_base_tokens'] == base_tokens
        assert source['actual_full_source_tokens'] == full_tokens
        assert source['actual_incremental_caption_block_tokens'] == full_tokens-base_tokens
        assert source['within_provisional_caption750'] == (full_tokens-base_tokens <= 750)
        assert source['within_diagnostic_total5000'] == (full_tokens <= 5000)
    stats = rows(out/'raw_caption_statistics.csv')
    assert len(stats) == 24
    by_id = {r['photo_id']: r for r in records}
    for row in stats:
        record = by_id[row['photo_id']]
        assert row['raw_caption'] == record['raw_caption']
        assert int(row['LED_body_tokens']) == len(tokenizer.encode(record['raw_caption'], add_special_tokens=False))
    report = {'status': 'PASS', 'scope': 'actual_E0_outputs_and_independent_source_token_validation',
              'actual_caption_outputs_validated': 24, 'original_photos_hash_checked': 24,
              'complete_original_reviews_checked': len(needed), 'source_examples_checked': 16,
              'independent_tokenizer': type(tokenizer).__name__+' use_fast=False',
              'surface_eligible': sum(r['surface_eligible'] for r in records),
              'manual_picture_caption_review': 'pending_separate_assistant_inspection_not_formal_human_evaluation',
              'production_caption_quality_or_A2_passed': False,
              'source_hashes_unchanged': True, 'research_parameters_changed': False,
              'elapsed_seconds': round(time.perf_counter()-started, 3)}
    (out/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
