"""Estimate E0 placement using meta tensors; never download weights or infer captions."""
from pathlib import Path
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_e/blip2_caption_feasibility_pilot_v1/resource_audit_20261006'


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '20'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '20'
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'checking', 'model_weights_downloaded': False,
              'model_weights_loaded': False, 'caption_generation_started': False,
              'research_parameters_changed': False, 'method': 'config_only_meta_tensor_placement'}
    try:
        import psutil
        import torch
        import yaml
        from accelerate import init_empty_weights, infer_auto_device_map
        from accelerate.utils import compute_module_sizes
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import Blip2Config, Blip2ForConditionalGeneration

        cfg = yaml.safe_load((ROOT / 'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))
        info = HfApi().model_info(cfg['model_id'], revision=cfg['requested_revision'], timeout=20)
        revision = info.sha
        assert revision
        print('Resolved config-only metadata:', revision, flush=True)
        config_path = hf_hub_download(cfg['model_id'], 'config.json', revision=revision,
                                      cache_dir=OUT / 'config_only_cache')
        config = Blip2Config.from_dict(json.loads(Path(config_path).read_text(encoding='utf-8')))
        with init_empty_weights(include_buffers=True):
            model = Blip2ForConditionalGeneration(config)
            model.tie_weights()
        assert all(p.device.type == 'meta' for p in model.parameters())
        special = {name: torch.float32 for name, _ in model.named_parameters()
                   if any(part in name.split('.') for part in model._keep_in_fp32_modules)}
        sizes = compute_module_sizes(model, dtype=torch.float16, special_dtypes=special)
        free, total = torch.cuda.mem_get_info(0)
        available = psutil.virtual_memory().available
        memory = {0: min(free, total) - int(cfg['device']['gpu_safety_reserve_GiB'] * 2**30),
                  'cpu': int(max(0, available - cfg['device']['cpu_safety_reserve_GiB'] * 2**30))}
        device_map = infer_auto_device_map(model, max_memory=memory,
                    no_split_module_classes=model._no_split_modules,
                    dtype=torch.float16, special_dtypes=special)
        # One later probe is sufficient: use the declared RAM gate, not a changed reserve.
        gate_memory = dict(memory)
        gate_memory['cpu'] = int((cfg['device']['minimum_available_RAM_GiB'] - cfg['device']['cpu_safety_reserve_GiB']) * 2**30)
        gate_map = infer_auto_device_map(model, max_memory=gate_memory,
                    no_split_module_classes=model._no_split_modules,
                    dtype=torch.float16, special_dtypes=special)
        report.update(status='estimated_not_actual_model_load', model_id=cfg['model_id'],
            resolved_config_commit=revision, available_RAM_GiB=available/2**30,
            GPU_free_GiB=free/2**30, GPU_total_GiB=total/2**30,
            effective_weight_GiB=sizes['']/2**30,
            parameter_count=sum(p.numel() for p in model.parameters()),
            fp32_retained_modules=model._keep_in_fp32_modules,
            component_weight_GiB={name: sizes[name]/2**30 for name in
                                   ['vision_model', 'qformer', 'language_model', 'query_tokens', 'language_projection']},
            weight_budget_bytes={str(k): v for k, v in memory.items()},
            current_estimated_device_map={k: str(v) for k, v in device_map.items()},
            current_requires_disk_offload=any(str(v)=='disk' for v in device_map.values()),
            declared_RAM_gate_GiB=cfg['device']['minimum_available_RAM_GiB'],
            gate_estimated_device_map={k: str(v) for k, v in gate_map.items()},
            gate_requires_disk_offload=any(str(v)=='disk' for v in gate_map.values()),
            placement_estimate_limitations='Does not measure generation activations, transfer time, model-load peak or successful inference. No weights downloaded.')
    except Exception as error:
        report.update(status='probe_failed_not_model_failure', error=type(error).__name__+': '+str(error))
    report['elapsed_seconds'] = round(time.perf_counter()-started, 3)
    (OUT/'model_memory_estimate.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status']=='estimated_not_actual_model_load' else 1


if __name__ == '__main__':
    raise SystemExit(main())
