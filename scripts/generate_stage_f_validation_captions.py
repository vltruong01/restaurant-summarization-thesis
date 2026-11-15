"""Full validation caption generation; same E1 algorithm, new2504-photo identity."""
from pathlib import Path
import argparse,gc,json,time,traceback
import yaml
import blip2_caption_pilot as legacy
from caption_basic_quality import evaluate,termination_metadata
ROOT=legacy.ROOT
CONFIG=ROOT/'configs/stage_f_validation_caption_pool.yaml'
ORIGIN='agent'
read,save,sha,canon_hash=legacy.read,legacy.save,legacy.sha,legacy.canon_hash

def generate_locked(cfg, out, max_new_photos=None):
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
    if len(records) == len(plan) and previous and previous['status'] == 'raw_pilot_complete_pending_review':
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

        initial_count = len(records)
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
            result = {'status': 'partial_raw_pilot', 'completed': len(records), 'planned': len(plan),
                      'runtime_sha256': runtime_sha, 'device_map_sha256': map_sha,
                      'elapsed_seconds': round(time.perf_counter()-started, 3), 'execution_origin': ORIGIN}
            save(out/'generation_report.json', result)
            print(f"{number}/{len(plan)} {photo['name']}: {r['normalized_caption']} | basic_ok={r['basic_eligible']}", flush=True)
            if max_new_photos and len(records)-initial_count >= max_new_photos and len(records)<len(plan):
                result.update(status='partial_validation_caption_pool', new_photos_this_invocation=len(records)-initial_count)
                save(out/'generation_report.json', result)
                legacy.check_sources(identity)
                return result
        result.update(status='raw_pilot_complete_pending_review', completed=len(plan),
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

def prepare(cfg,out):
    plan=read(out/'validation_photo_plan.json')
    assert len(plan)==2504 and {p['split'] for p in plan}=={'validation'}
    parent=read(ROOT/'outputs/stage_e/blip2_caption_basic_quality_pilot_v2/pilot_identity.json')
    legacy.check_sources(parent)
    names=['configs/stage_f_validation_caption_pool.yaml','configs/stage_f_validation_protocol.yaml','scripts/generate_stage_f_validation_captions.py','scripts/caption_basic_quality.py','outputs/stage_f/validation_calibration_v1/business_roles.json','outputs/stage_f/validation_calibration_v1/validation_photo_plan.json','data/processed/yelp_2300_v1/photo_metadata/validation.jsonl']
    sources=dict(parent['source_sha256']);sources.update({name:sha(ROOT/name) for name in names})
    identity={'config_sha256':sha(CONFIG),'source_sha256':sources,'plan_sha256':canon_hash(plan),'basic_policy_sha256':sha(ROOT/cfg['basic_quality_policy']),'parent_identity_sha256':parent['identity_sha256']}
    identity['identity_sha256']=canon_hash(identity)
    if (out/'pilot_identity.json').exists():assert read(out/'pilot_identity.json')==identity
    else:save(out/'pilot_identity.json',identity);save(out/'pilot_photo_plan.json',plan)
    save(out/'model_revision.json',read(ROOT/'outputs/stage_e/blip2_caption_basic_quality_pilot_v2/model_revision.json'))
    return {'status':'full_validation_plan_prepared','photos':2504,'businesses':200,'test_used':False,'final_caption_set':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','preflight','generate']);parser.add_argument('--max-new-photos',type=int);args=parser.parse_args()
    if args.max_new_photos is not None and args.max_new_photos<1:parser.error('--max-new-photos must be positive')
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];legacy.ORIGIN='agent'
    try:
        if args.step=='prepare':result=prepare(cfg,out)
        elif args.step=='preflight':result=legacy.preflight(cfg,out)
        else:
            from filelock import FileLock
            with FileLock(out/'caption_model_execution.lock',timeout=0):result=generate_locked(cfg,out,args.max_new_photos)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        save(out/(args.step+'_failure.json'),{'status':'blocked','error':str(error),'traceback':traceback.format_exc(),'no_automatic_method_or_budget_change':True});raise
if __name__=='__main__':main()
