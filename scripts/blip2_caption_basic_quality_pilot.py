"""E1 fixed24-photo pilot with approved minimal Basic Quality and EOS evidence."""
from pathlib import Path
import argparse
import csv
import gc
import json
import time
import traceback

import yaml
import blip2_caption_pilot as legacy
from caption_basic_quality import evaluate, termination_metadata

ROOT = legacy.ROOT
CONFIG = ROOT/'configs/blip2_caption_basic_quality_pilot.yaml'
ORIGIN = 'user'
read, save, sha, canon_hash = legacy.read, legacy.save, legacy.sha, legacy.canon_hash


def prepare(cfg, out):
    parent = ROOT/cfg['parent_pilot_directory']
    parent_identity = read(parent/'pilot_identity.json')
    legacy.check_sources(parent_identity)
    plan = read(parent/'pilot_photo_plan.json')
    assert len(plan) == 24 and {p['split'] for p in plan} == {'train', 'validation'}
    assert cfg['requested_revision'] == read(parent/'model_revision.json')['resolved_commit']
    assert cfg['model_id'] == 'Salesforce/blip2-opt-2.7b-coco'
    assert cfg['generation']['num_beams'] == 3 and cfg['generation']['initial_prompt'] is None
    policy = yaml.safe_load((ROOT/cfg['basic_quality_policy']).read_text(encoding='utf-8'))
    assert policy['status'] == 'adopted_for_basic_quality_pilot'
    assert cfg['generation']['retry_prompt'] == policy['retry']['prompt']
    names = [CONFIG.relative_to(ROOT).as_posix(), cfg['basic_quality_policy'],
             'scripts/caption_basic_quality.py', 'scripts/blip2_caption_basic_quality_pilot.py',
             'scripts/validate_basic_quality_caption_pilot.py']
    sources = dict(parent_identity['source_sha256'])
    sources.update({name: sha(ROOT/name) for name in names})
    identity = {'config_sha256': sha(CONFIG), 'source_sha256': sources,
                'parent_identity_sha256': parent_identity['identity_sha256'],
                'plan_sha256': canon_hash(plan), 'basic_policy_sha256': sha(ROOT/cfg['basic_quality_policy'])}
    identity['identity_sha256'] = canon_hash(identity)
    if (out/'pilot_identity.json').exists():
        assert read(out/'pilot_identity.json') == identity, 'Use a new directory after identity changes.'
        assert read(out/'pilot_photo_plan.json') == plan
    else:
        save(out/'pilot_identity.json', identity)
        save(out/'pilot_photo_plan.json', plan)
        save(out/'model_revision.json', read(parent/'model_revision.json'))
    result = {'status': 'pilot_prepared', 'photos': 24, 'businesses': 8,
              'same_original_photo_plan': True, 'source_changed': False,
              'identity_sha256': identity['identity_sha256'], 'policy_adopted_for_pilot': True,
              'execution_origin': ORIGIN, 'final_manifest': False}
    save(out/'pilot_preparation_report.json', result)
    return result


def generate(cfg, out):
    from filelock import FileLock
    with FileLock(out/'caption_model_execution.lock', timeout=0):
        return generate_locked(cfg, out)


def generate_locked(cfg, out):
    import torch
    from PIL import Image, ImageOps
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    identity = read(out/'pilot_identity.json')
    legacy.check_sources(identity)
    env = read(out/'environment_report.json')
    assert env['status'] == 'preflight_passed_not_model_load_passed'
    policy = yaml.safe_load((ROOT/cfg['basic_quality_policy']).read_text(encoding='utf-8'))
    plan = read(out/'pilot_photo_plan.json')
    revision = read(out/'model_revision.json')['resolved_commit']
    runtime = {'model_id': cfg['model_id'], 'resolved_commit': revision,
               'plan_identity': identity['identity_sha256'], 'generation': cfg['generation'],
               'basic_quality': policy, 'weight_dtype': 'float16', 'quantization': 'none',
               'versions': env['versions'], 'gpu_name': env['gpu_name'], 'torch_cuda': env['torch_cuda']}
    runtime_sha = canon_hash(runtime)
    checkpoints = out/'caption_checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records = {}
    for photo in plan:
        path = checkpoints/f"{photo['photo_id']}.json"
        if path.exists():
            r = read(path)
            assert r['runtime_sha256'] == runtime_sha and r['raw_sha256'] == photo['raw_sha256']
            records[photo['photo_id']] = r
    previous = read(out/'generation_report.json') if (out/'generation_report.json').exists() else None
    if len(records) == 24 and previous and previous['status'] == 'raw_pilot_complete_pending_review':
        return previous
    memory = {0: int(env['GPU_weight_budget_bytes']), 'cpu': int(env['CPU_weight_budget_bytes'])}
    model = None
    started = time.perf_counter()
    try:
        print('Loading pinned COCO checkpoint from cache; fixed beam3 GPU+CPU offload.', flush=True)
        processor = Blip2Processor.from_pretrained(cfg['model_id'], revision=revision,
                    cache_dir=ROOT/cfg['model_cache'], local_files_only=True, trust_remote_code=False)
        model = Blip2ForConditionalGeneration.from_pretrained(cfg['model_id'], revision=revision,
                    cache_dir=ROOT/cfg['model_cache'], local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, torch_dtype=torch.float16, device_map='auto',
                    max_memory=memory, low_cpu_mem_usage=True, offload_state_dict=True)
        model.eval()
        device_map = {k: str(v) for k, v in model.hf_device_map.items()}
        assert 'disk' not in device_map.values()
        map_sha = canon_hash(device_map)
        if previous and records:
            assert previous['device_map_sha256'] == map_sha, 'Do not mix partial runtime maps.'
        save(out/'model_load_report.json', {'status': 'loaded_not_quality_verified', 'runtime': runtime,
             'runtime_sha256': runtime_sha, 'device_map': device_map, 'device_map_sha256': map_sha,
             'model_load_seconds': round(time.perf_counter()-started, 3),
             'GPU_allocated_GiB': torch.cuda.memory_allocated()/2**30,
             'processor_query_tokens': getattr(processor, 'num_query_tokens', None),
             'model_image_token_index': getattr(model.config, 'image_token_index', None),
             'module_weight_dtypes': {name: str(next(module.parameters()).dtype) for name, module in
                    [('vision', model.vision_model), ('qformer', model.qformer), ('language', model.language_model)]}})
        eos = model.language_model.generation_config.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        assert all(isinstance(i, int) for i in eos_ids)
        pad = model.language_model.generation_config.pad_token_id

        def attempt(rgb, prompt):
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            inputs = processor(images=rgb, return_tensors='pt', **({'text': prompt} if prompt else {}))
            inputs = {k: v.to('cuda:0', dtype=torch.float16) if v.is_floating_point() else v.to('cuda:0')
                      for k, v in inputs.items()}
            prefix = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 1+(
                     model.config.num_query_tokens if getattr(model.config, 'image_token_index', None) is not None else 0)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                generated = model.generate(**inputs, num_beams=3, do_sample=False,
                                            max_new_tokens=cfg['generation']['max_new_tokens'])
            torch.cuda.synchronize()
            tokens = generated[0].tolist()
            expected_prefix = inputs['input_ids'][0].tolist() if 'input_ids' in inputs else [model.config.text_config.bos_token_id]
            assert tokens[:prefix] == expected_prefix, 'Generated prefix must match prompt/BOS.'
            evidence = termination_metadata(tokens, prefix, eos_ids, pad, cfg['generation']['max_new_tokens'])
            raw = processor.tokenizer.decode(evidence['generated_suffix_ids'], skip_special_tokens=True)
            result = {'prompt': prompt, 'raw_caption': raw, 'input_prefix_ids': expected_prefix, **evidence,
                      'seconds': round(time.perf_counter()-tick, 3),
                      'peak_GPU_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                      'peak_GPU_reserved_GiB': torch.cuda.max_memory_reserved()/2**30}
            result.update(evaluate(result, policy))
            return result

        for number, photo in enumerate(plan, 1):
            if photo['photo_id'] in records:
                continue
            assert sha(ROOT/photo['image_path']) == photo['raw_sha256']
            with Image.open(ROOT/photo['image_path']) as original:
                original.load()
                rgb = ImageOps.exif_transpose(original).convert('RGB')
                attempts = [attempt(rgb, cfg['generation']['initial_prompt'])]
                if attempts[0]['basic_quality_fail']:
                    attempts.append(attempt(rgb, cfg['generation']['retry_prompt']))
            chosen = next((i for i, a in enumerate(attempts) if not a['basic_quality_fail']), 0)
            a = attempts[chosen]
            r = {**photo, 'model_id': cfg['model_id'], 'resolved_commit': revision,
                 'runtime_sha256': runtime_sha, 'device_map_sha256': map_sha,
                 'basic_quality_policy_sha256': identity['basic_policy_sha256'],
                 'generation_settings': cfg['generation'], 'attempts': attempts, 'chosen_attempt': chosen+1,
                 'raw_caption': a['raw_caption'], 'normalized_caption': a['normalized_caption'],
                 'basic_eligible': not a['basic_quality_fail'], 'surface_ok': a['surface_ok'],
                 'final_caption_set': False, 'teacher_target': False}
            save(checkpoints/f"{photo['photo_id']}.json", r)
            records[photo['photo_id']] = r
            result = {'status': 'partial_raw_pilot', 'completed': len(records), 'planned': 24,
                      'runtime_sha256': runtime_sha, 'device_map_sha256': map_sha,
                      'elapsed_seconds': round(time.perf_counter()-started, 3), 'execution_origin': ORIGIN}
            save(out/'generation_report.json', result)
            print(f"{number}/24 {photo['name']}: {r['normalized_caption']} | basic_ok={r['basic_eligible']}", flush=True)
        result.update(status='raw_pilot_complete_pending_review', completed=24,
                      basic_eligible=sum(r['basic_eligible'] for r in records.values()),
                      full_generation_complete=True, production_quality_accepted=False)
        save(out/'generation_report.json', result)
        legacy.check_sources(identity)
        return result
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def report(cfg, out):
    from tokenizers import Tokenizer
    assert read(out/'generation_report.json')['status'] == 'raw_pilot_complete_pending_review'
    identity = read(out/'pilot_identity.json')
    legacy.check_sources(identity)
    parent = ROOT/cfg['parent_pilot_directory']
    records = [read(out/'caption_checkpoints'/f"{p['photo_id']}.json") for p in read(out/'pilot_photo_plan.json')]
    bases = [s for s in read(parent/'complete_provisional_source_examples.json')
             if s['condition'] == 'all_raw_pilot_captions_capacity_only']
    tok = Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tok.no_truncation(); tok.no_padding()
    sources = []
    for old in bases:
        base, _ = old['complete_provisional_source'].rsplit('\n\n[CAPTIONS]\n', 1)
        current = [r for r in records if r['business_id'] == old['business_id']]
        for condition, field, captions in [
                ('all_raw_capacity_only', 'raw_caption', current),
                ('all_normalized_capacity_only', 'normalized_caption', current),
                ('basic_eligible_normalized_NOT_final_caption_set', 'normalized_caption', [r for r in current if r['basic_eligible']])]:
            block = '\n'.join(f"[IMG_CAP_{i}] {r[field]}" for i, r in enumerate(captions, 1))
            full = base+('\n\n[CAPTIONS]\n'+block if captions else '')
            total = len(tok.encode(full).ids)
            delta = total-len(tok.encode(base).ids)
            sources.append({'business_id': old['business_id'], 'name': old['name'], 'split': old['split'],
                 'condition': condition, 'caption_text_field': field, 'captions': len(captions),
                 'selected_review_ids': old['selected_review_ids'], 'retained_reviews': old['retained_reviews'],
                 'caption_photo_ids': [r['photo_id'] for r in captions], 'review_IDs_unchanged': True,
                 'text_only_base_tokens': len(tok.encode(base).ids), 'actual_full_source_tokens': total,
                 'actual_incremental_caption_block_tokens': delta, 'within_diagnostic5000': total <= 5000,
                 'within_provisional_caption750': delta <= 750, 'complete_provisional_source': full,
                 'caption_set_final': False, 'teacher_or_training_input': False})
    stats = []
    for r in records:
        a = r['attempts'][r['chosen_attempt']-1]
        stats.append({k: r[k] for k in ['business_id', 'name', 'split', 'photo_id', 'image_path', 'raw_caption', 'normalized_caption', 'basic_eligible']} |
             {'word_count': a['word_count'], 'attempts': len(r['attempts']), 'chosen_attempt': r['chosen_attempt'],
              'basic_quality_flags': '|'.join(a['basic_quality_flags']),
              'raw_LED_body_tokens': len(tok.encode(r['raw_caption'], add_special_tokens=False).ids),
              'normalized_LED_body_tokens': len(tok.encode(r['normalized_caption'], add_special_tokens=False).ids),
              'eos_seen': a['eos_seen'], 'token_cap_reached': a['token_cap_reached'],
              'termination_reason': a['termination_reason'], 'seconds_all_attempts': sum(a['seconds'] for a in r['attempts']),
              'peak_GPU_allocated_GiB': max(a['peak_GPU_allocated_GiB'] for a in r['attempts'])})
    with (out/'caption_statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
    save(out/'complete_provisional_source_examples.json', sources)
    save(out/'source_token_ledger.json', [{k:v for k,v in s.items() if k not in ['complete_provisional_source', 'selected_review_ids']} for s in sources])
    attempts = [a for r in records for a in r['attempts']]
    normalized = [s for s in sources if s['condition'] == 'basic_eligible_normalized_NOT_final_caption_set']
    result = {'status': 'actual_generation_complete_pending_validation_and_review', 'photos': 24, 'businesses': 8,
              'attempts': len(attempts), 'retry_photos': sum(len(r['attempts']) == 2 for r in records),
              'initial_basic_eligible': sum(not r['attempts'][0]['basic_quality_fail'] for r in records),
              'final_basic_eligible': sum(r['basic_eligible'] for r in records),
              'empty_attempts': sum(not a['raw_caption'].strip() for a in attempts),
              'eos_attempts': sum(a['eos_seen'] for a in attempts),
              'cap_without_EOS_attempts': sum(a['token_cap_reached'] and not a['eos_seen'] for a in attempts),
              'all_attempt_seconds': sum(a['seconds'] for a in attempts),
              'peak_GPU_allocated_GiB': max(a['peak_GPU_allocated_GiB'] for a in attempts),
              'peak_GPU_reserved_GiB': max(a['peak_GPU_reserved_GiB'] for a in attempts),
              'normalized_source_min': min(s['actual_full_source_tokens'] for s in normalized),
              'normalized_source_max': max(s['actual_full_source_tokens'] for s in normalized),
              'source5000_overflows': sum(not s['within_diagnostic5000'] for s in sources),
              'caption750_overflows': sum(not s['within_provisional_caption750'] for s in sources),
              'source_examples': len(sources), 'original_raw_text_preserved': True,
              'production_quality_A2_accepted': False, 'final_manifest_K_source_allocation': False,
              'teacher_training_rental_started': False, 'execution_origin': ORIGIN}
    save(out/'pilot_report.json', result)
    return result


def main():
    global ORIGIN
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['prepare', 'preflight', 'generate', 'report'])
    parser.add_argument('--execution-origin', choices=['agent', 'user'], default='user')
    args = parser.parse_args(); ORIGIN = args.execution_origin; legacy.ORIGIN = ORIGIN
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); out = ROOT/cfg['output_directory']; out.mkdir(parents=True, exist_ok=True)
    try:
        result = prepare(cfg, out) if args.step == 'prepare' else legacy.preflight(cfg, out) if args.step == 'preflight' else generate(cfg, out) if args.step == 'generate' else report(cfg, out)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        save(out/f'{args.step}_failure.json', {'status': 'blocked', 'error': str(error),
             'traceback': traceback.format_exc(), 'research_settings_automatically_changed': False})
        raise


if __name__ == '__main__':
    main()
