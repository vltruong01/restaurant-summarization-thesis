"""Independent ID/hash, token termination, raw decode and full-source validation."""
from pathlib import Path
import hashlib
import json
import re
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT/'configs/blip2_caption_basic_quality_pilot.yaml').read_text())
    policy = yaml.safe_load((ROOT/cfg['basic_quality_policy']).read_text())
    out = ROOT/cfg['output_directory']; parent = ROOT/cfg['parent_pilot_directory']
    assert read(out/'generation_report.json')['completed'] == 24
    identity = read(out/'pilot_identity.json')
    for name, expected in identity['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    plan = read(out/'pilot_photo_plan.json')
    assert plan == read(parent/'pilot_photo_plan.json')
    load = read(out/'model_load_report.json')
    assert 'disk' not in load['device_map'].values()
    assert load['runtime']['generation'] == cfg['generation'] and cfg['generation']['num_beams'] == 3
    assert load['runtime']['resolved_commit'] == cfg['requested_revision']
    caption_tok = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['requested_revision'],
                       cache_dir=ROOT/cfg['model_cache'], local_files_only=True, trust_remote_code=False)
    snapshot = ROOT/cfg['model_cache']/('models--'+cfg['model_id'].replace('/', '--'))/'snapshots'/cfg['requested_revision']
    pinned_config = read(snapshot/'config.json')
    model_eos = pinned_config['text_config']['eos_token_id']
    model_eos_ids = model_eos if isinstance(model_eos, list) else [model_eos]
    model_pad = pinned_config['text_config']['pad_token_id']
    # GenerationConfig comes from the pinned language-model config, not tokenizer EOS.
    word = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
    records = []
    for p in plan:
        assert sha(ROOT/p['image_path']) == p['raw_sha256']
        r = read(out/'caption_checkpoints'/f"{p['photo_id']}.json")
        for key in ['photo_id', 'business_id', 'split', 'raw_sha256', 'image_path']:
            assert r[key] == p[key]
        assert r['runtime_sha256'] == load['runtime_sha256']
        assert r['device_map_sha256'] == load['device_map_sha256']
        assert r['basic_quality_policy_sha256'] == identity['basic_policy_sha256']
        assert r['generation_settings'] == cfg['generation']
        assert not r['final_caption_set'] and not r['teacher_target']
        assert 1 <= len(r['attempts']) <= 2
        eligible = []
        for i, a in enumerate(r['attempts']):
            assert a['prompt'] == (None if i == 0 else policy['retry']['prompt'])
            seq, prefix = a['generated_sequence_ids'], a['prefix_tokens']
            expected_prefix = caption_tok(a['prompt'])['input_ids'] if a['prompt'] else [caption_tok.bos_token_id]
            assert expected_prefix == a['input_prefix_ids'] == seq[:prefix]
            assert a['eos_token_ids'] == sorted(model_eos_ids)
            assert a['pad_token_id'] == model_pad
            suffix = seq[prefix:]
            assert suffix == a['generated_suffix_ids']
            first_eos = next((j for j, t in enumerate(suffix) if t in a['eos_token_ids']), None)
            assert a['eos_seen'] == (first_eos is not None)
            assert a['first_generated_eos_index'] == first_eos
            effective = len(suffix) if first_eos is None else first_eos+1
            assert effective == a['effective_generated_tokens']
            assert a['token_cap_reached'] == (effective >= cfg['generation']['max_new_tokens'])
            assert caption_tok.decode(suffix, skip_special_tokens=True) == a['raw_caption']
            assert word.findall(a['raw_caption']) == word.findall(a['normalized_caption'])
            assert len(word.findall(a['raw_caption'])) == a['word_count']
            text = ' '.join(a['raw_caption'].split())
            end = len(text)
            while end and text[end-1] in '\"\'\u201d\u2019)]}':
                end -= 1
            normalized = text if not word.search(text) or text[:end].endswith(('.', '!', '?')) else text[:end]+'.'+text[end:]
            assert normalized == a['normalized_caption']
            tokens = [w.lower() for w in word.findall(a['raw_caption'])]
            flags = []
            if not a['raw_caption'].strip(): flags.append('empty')
            elif not tokens: flags.append('nonlexical')
            if len(tokens) < 5: flags.append('under5_words')
            repeated = any(tokens[j:j+n] == tokens[j+n:j+2*n]
                           for n in range(4, 9) for j in range(len(tokens)-2*n+1))
            if repeated: flags.append('adjacent_repeated_4_to_8_word_block')
            key = ' '.join(a['raw_caption'].lower().split()).strip(' .!?\"\'\u201d\u2019')
            if key in policy['generic']['phrases']: flags.append('exact_generic_phrase')
            if a['token_cap_reached'] and not a['eos_seen']: flags.append('evidenced_truncation')
            assert flags == a['basic_quality_flags'] and bool(flags) == a['basic_quality_fail']
            assert a['surface_ok'] == (not flags)
            eligible.append(not flags)
        assert len(r['attempts']) == (2 if not eligible[0] else 1)
        chosen = next((i for i, ok in enumerate(eligible, 1) if ok), 1)
        assert r['chosen_attempt'] == chosen and r['basic_eligible'] == eligible[chosen-1]
        assert r['raw_caption'] == r['attempts'][chosen-1]['raw_caption']
        assert r['normalized_caption'] == r['attempts'][chosen-1]['normalized_caption']
        records.append(r)
    sources = read(out/'complete_provisional_source_examples.json')
    assert len(sources) == 24
    needed = {rid for s in sources for rid in s['selected_review_ids']}
    originals = {}
    for split in ['train', 'validation']:
        with (ROOT/f'data/processed/yelp_2300_english_v1/reviews/{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                review = json.loads(line)
                if review['review_id'] in needed:
                    originals[review['review_id']] = review
    assert set(originals) == needed
    businesses = {b['business_id']: b for b in map(json.loads, (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').read_text(encoding='utf-8').splitlines())}
    led = AutoTokenizer.from_pretrained(ROOT/'assets/tokenizers/led-base-16384', use_fast=False, local_files_only=True)
    for s in sources:
        selected = [originals[rid] for rid in s['selected_review_ids']]
        assert all(r['business_id'] == s['business_id'] for r in selected)
        assert selected == sorted(selected, key=lambda r: (r['date'], r['review_id']))
        reviews = '\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i, r in enumerate(selected, 1))
        metadata = '[BUSINESS]\n'+'\n'.join(f"{k}={businesses[s['business_id']].get(k) or ''}" for k in ['name', 'categories', 'city', 'state'])
        base = metadata+'\n\n[REVIEWS]\n'+reviews
        captions = [r for r in records if r['business_id'] == s['business_id']]
        if s['condition'].startswith('basic_eligible'): captions = [r for r in captions if r['basic_eligible']]
        block = '\n'.join(f"[IMG_CAP_{i}] {r[s['caption_text_field']]}" for i, r in enumerate(captions, 1))
        full = base+('\n\n[CAPTIONS]\n'+block if captions else '')
        assert full == s['complete_provisional_source']
        assert s['caption_photo_ids'] == [r['photo_id'] for r in captions]
        assert len(led.encode(full, truncation=False)) == s['actual_full_source_tokens']
        assert len(led.encode(base, truncation=False)) == s['text_only_base_tokens']
        assert len(led.encode(full, truncation=False))-len(led.encode(base, truncation=False)) == s['actual_incremental_caption_block_tokens']
    result = {'status': 'PASS', 'actual_photos_hash_checked': 24, 'actual_attempts_checked': sum(len(r['attempts']) for r in records),
              'raw_token_decodes_normalization_and_EOS_checked': True, 'original_complete_reviews_checked': len(needed),
              'independent_slow_LED_sources_checked': 24, 'source_hashes_unchanged': True,
              'basic_quality_policy_adopted_for_pilot': True, 'production_quality_or_A2_passed': False,
              'formal_human_evaluation': False, 'generation_EOS_from_pinned_text_config': model_eos_ids,
              'tokenizer_EOS_is_not_generation_EOS': caption_tok.eos_token_id != model_eos_ids[0],
              'pinned_model_config_sha256': sha(snapshot/'config.json'),
              'validator_sha256': sha(Path(__file__)),
              'validator_v1_failed_assertion_retained': 'maintenance_20261007/validator_v1_failure.log',
              'elapsed_seconds': round(time.perf_counter()-started, 3)}
    (out/'validation_report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
