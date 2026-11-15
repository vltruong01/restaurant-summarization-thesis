"""Offline preflight by default. GPU feasibility/training require reviewed approval."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

from stage_hi_led_data import (ROOT, read, sha, require, verify_bundle, PackedLEDDataset,
                               LEDCompleteCollator, execution_gate, settings_signature, generation_format)

DEFAULT = ROOT / 'configs/stage_i_lora_led_v1.json'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def versions():
    packages = ['torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'numpy', 'safetensors']
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={k: importlib.metadata.version(k) for k in packages})


def datasets(config, manifest, split):
    return {c: PackedLEDDataset(manifest['datasets'][f'{split}_{c}'], config['source_limit'], config['label_limit'])
            for c in manifest['conditions']}


def architecture(config, mc):
    import torch
    from transformers import LEDConfig, LEDForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    with torch.device('meta'):
        model = LEDForConditionalGeneration(LEDConfig(**mc))
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                 and n.rsplit('.', 1)[-1] in config['lora']['target_modules']]
        require(len(names) == 24 and all(n.startswith('led.decoder.') for n in names), 'Unexpected literal LoRA module scope')
        base_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, LoraConfig(**config['lora']))
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        require(all('lora_' in n and '.decoder.' in n for n, _ in trainable), 'Unexpected trainable base/encoder parameters')
        require(all(p.device.type == 'meta' for p in model.parameters()), 'Architecture probe allocated real model weights')
    return dict(matched_modules=names, matched_count=len(names), encoder_adapter_count=0,
                scope='decoder_self_attention_and_decoder_cross_attention_only', base_parameters=base_params,
                trainable_parameters=sum(v for _, v in trainable),
                trainable_fraction_pct=100*sum(v for _, v in trainable)/(base_params+sum(v for _, v in trainable)),
                trainable_parameter_names=[n for n, _ in trainable], weights_loaded=False, forward_backward_run=False)


def preflight(config, path, report_directory=None):
    import torch
    from transformers import LEDTokenizerFast, LEDTokenizer
    from transformers.models.led.modeling_led import shift_tokens_right
    manifest, tm, mc = verify_bundle(config)
    inventory = versions()
    expected = {'torch': '2.1.0', 'transformers': '4.51.3', 'peft': '0.15.2', 'accelerate': '1.6.0',
                'tokenizers': '0.21.1', 'numpy': '1.26.4', 'safetensors': '0.8.0'}
    require(all(inventory['packages'][k].split('+')[0] == v for k, v in expected.items()), 'Dependency drift; use prepared pins or review a new runtime')
    tokenizer = LEDTokenizerFast.from_pretrained(ROOT / tm['relative_directory'], local_files_only=True)
    slow = LEDTokenizer.from_pretrained(ROOT / tm['relative_directory'], local_files_only=True)
    collator = LEDCompleteCollator(config['source_limit'], config['label_limit'], mc['attention_window'][0])
    checks = []; fast_slow = 0
    for split in ('train', 'validation'):
        data = datasets(config, manifest, split)
        order = None
        for c, ds in data.items():
            if order is None: order = ds.business_ids
            else: require(order == ds.business_ids, 'Condition row order differs')
            sizes = np.diff(ds.pack['label_offsets'])
            indices = sorted({0, int(sizes.argmax()), int(np.diff(ds.pack['input_offsets']).argmax())})
            features = [ds[i] for i in indices]
            batch = collator(features)
            shifted = shift_tokens_right(batch['labels'], tokenizer.pad_token_id, mc['decoder_start_token_id'])
            require(torch.all(shifted[:, 0] == 2), 'Decoder-start shift wrong')
            for i, f in enumerate(features):
                n, m = len(f['input_ids']), len(f['labels'])
                require(batch['input_ids'][i, :n].tolist() == f['input_ids'] and batch['labels'][i, :m].tolist() == f['labels'], 'Collator altered content')
                require(torch.all(batch['labels'][i, m:] == -100), 'Label padding not masked')
                require(torch.all(batch['attention_mask'][i, n:] == 0) and torch.all(batch['global_attention_mask'][i, 1:] == 0), 'Attention padding/global scope wrong')
                require(shifted[i, 1:m].tolist() == f['labels'][:-1], 'Decoder teacher-forcing labels misaligned')
                # Exact raw text is checked against full JSONL in Stage H; this checks the runtime tokenizer on the same IDs.
                for ids in (f['input_ids'], f['labels']):
                    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    require(tokenizer.encode(text, add_special_tokens=True, truncation=False) == ids == slow.encode(text, add_special_tokens=True, truncation=False), 'Runtime tokenizer/ID roundtrip drift')
                    fast_slow += 1
            checks.append(dict(split=split, condition=c, sample_business_ids=[ds.business_ids[i] for i in indices],
                               collated_source_shape=list(batch['input_ids'].shape), label_shape=list(batch['labels'].shape),
                               complete_tokens_preserved=True, global_BOS_only=True, padding_labels_minus100=True))
    probe = architecture(config, mc)
    require(config['label_limit'] == config['generation']['max_length'] == 768, 'Current proposed limit changed')
    report = dict(status='STAGE_I_OFFLINE_PREFLIGHT_PASS_GPU_UNTESTED', environment=inventory,
                  settings_signature=settings_signature(config), config_sha256=sha(path), stage_h_manifest_sha256=sha(ROOT/config['stage_h_manifest']),
                  architecture=probe, collator_checks=checks, runtime_fast_slow_ID_checks=fast_slow,
                  attention_window=mc['attention_window'], logical_source_limit=5000, max_padded_encoder_shape=5120,
                  candidate_label_limit=768, candidate_generation_max_length=768, decoder_start_tokens=1,
                  decoder_position_capacity=mc['max_decoder_position_embeddings'],
                  length_policy_approved=config['approval']['length_and_training_settings'],
                  forward_backward_tested=False, pretrained_generation_tested=False, checkpoint_recovery_tested=False,
                  actual_GPU_memory_or_training_time_measured=False, GPU_rental_or_upload=False, teacher_API_calls=0,
                  training_started=False, held_out_test_content_read=False,
                  blockers=['researcher approval of common length/optimization/decoding proposals',
                            'reviewed actual 24GB GPU forward/backward, memory, generation and checkpoint feasibility',
                            'explicit full-training authorization'])
    folder = ROOT / (report_directory or 'outputs/stage_i/preparation_v1')
    write_new(folder/'offline_preflight_report.json', report)
    summary = ['# Stage I offline readiness', '', '**Offline preflight PASS; no GPU forward/backward or LED training performed.**', '',
               f"Literal q_proj/v_proj matches {probe['matched_count']} decoder attention linear modules; {probe['trainable_parameters']:,} trainable adapter parameters ({probe['trainable_fraction_pct']:.4f}%). Encoder adapters:0. Do not silently add query/value projections.", '',
               f"Runtime tokenizer and slow-tokenizer comparisons:{fast_slow}; collator ID preservation, BOS-only global attention, padding masks and decoder-start shifts PASS for all six datasets.", '',
               'Recommend reviewing the 768-token common limit: labels include BOS/EOS; generation max_length includes decoder-start. Every existing label fits. Logical input5000 pads to5120 with the unchanged1024 attention window.', '',
               'Draft optimization/decoding defaults are in configs/stage_i_lora_led_v1.json. Approval switches are false. GPU commands fail before loading weights. No scientific parameters were frozen by this report.', '',
               'Not yet validated: actual pretrained forward/backward, FP16 numerical behavior, peak VRAM, generation format/EOS/cap behavior, real checkpoint recovery, overfit sanity and training throughput. Use the prepared feasibility mode only after approval.', '',
               'Proposed first rental: one24GB RTX3090/4090-class GPU, >=32GB host RAM, >=4vCPU, >=50GB free persistent disk.24GB sufficiency is untested. A second card does not pool VRAM automatically. Target15% free VRAM headroom for training and beam4 generation.', '',
               'Time estimate method: warmed-up measured microstep time * ceil(2000/physical_batch) per epoch + measured full validation + checkpoint time; multiply by actual epochs/three runs, then add measured final validation decoding. No unmeasured completion-time guarantee.', '',
               'Stage G diagnostics/human audit limitations remain unchanged. No teacher/test calls, test tuning, upload, rental or training.']
    (folder/'TRAINING_READINESS_REPORT.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status','runtime_fast_slow_ID_checks','length_policy_approved','forward_backward_tested','training_started']}))
    print(json.dumps(probe)); return report


def init_gpu(config, manifest, tm, mc):
    import torch
    from transformers import LEDForConditionalGeneration, LEDTokenizerFast, set_seed
    from peft import LoraConfig, get_peft_model
    require(torch.cuda.is_available(), 'CUDA GPU required; CPU fallback disabled')
    vram = torch.cuda.get_device_properties(0).total_memory/1024**3
    require(vram >= config['feasibility']['minimum_vram_GiB'], 'GPU below reviewed first-test VRAM profile; no automatic fallback')
    set_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = LEDTokenizerFast.from_pretrained(ROOT/tm['relative_directory'], local_files_only=True)
    base = LEDForConditionalGeneration.from_pretrained(config['model_id'], revision=config['revision'], torch_dtype=torch.float32)
    for key in ['attention_window','max_decoder_position_embeddings','max_encoder_position_embeddings','d_model','encoder_layers','decoder_layers','decoder_start_token_id','bos_token_id','eos_token_id','pad_token_id']:
        require(getattr(base.config, key) == mc[key], f'Loaded backbone differs: {key}')
    matched = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear) and n.rsplit('.',1)[-1] in config['lora']['target_modules']]
    require(len(matched) == 24 and all(n.startswith('led.decoder.') for n in matched), 'Unexpected adapter scope')
    # Attach first; non-reentrant checkpointing avoids requiring artificial embedding gradients.
    model = get_peft_model(base, LoraConfig(**config['lora']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config['optimization']['gradient_checkpointing_kwargs'])
    model.config.use_cache = False
    model.to('cuda')
    return model, tokenizer, dict(name=torch.cuda.get_device_name(0), vram_GiB=vram, cuda=torch.version.cuda)


def generate_case(model, tokenizer, feature, collator, config, condition, business_id):
    import torch
    batch = {k:v.cuda() for k,v in collator([feature]).items() if k != 'labels'}
    model.eval(); torch.cuda.synchronize(); start = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        ids = model.generate(**batch, **config['generation'], use_cache=True)
    torch.cuda.synchronize()
    require(ids.shape[1] <= config['generation']['max_length'], 'Generation exceeded explicit limit')
    text = tokenizer.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    source_text = tokenizer.decode(feature['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    empty_visual = condition == 'filtered' and source_text.endswith('[VISUAL_CONTEXT]\n')
    return dict(business_id=business_id, condition=condition, token_ids=ids[0].cpu().tolist(), text=text,
                seconds=time.perf_counter()-start, tokens_including_decoder_start=int(ids.shape[1]),
                cap_reached=ids.shape[1] == config['generation']['max_length'],
                section_format_ok=generation_format(text,condition,allow_empty_visual=empty_visual), no_prediction_rewrite=True)


def feasibility(config, path):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    execution_gate(config, 'feasibility')
    manifest, tm, mc = verify_bundle(config)
    folder = ROOT / config['output_root'] / 'feasibility'
    folder.mkdir(parents=True, exist_ok=False)
    data = datasets(config,manifest,'train')
    # Preselect controls by length and zero-caption status, never by model results.
    sample_ids = set()
    for ds in data.values():
        sample_ids.add(ds.business_ids[int(np.diff(ds.pack['input_offsets']).argmax())])
        sample_ids.add(ds.business_ids[int(np.diff(ds.pack['label_offsets']).argmax())])
    f = data['filtered']; zero = np.flatnonzero(f.pack['caption_counts'] == 0)
    if len(zero): sample_ids.add(f.business_ids[int(zero[0])])
    sample_ids = sorted(sample_ids)
    write_new(folder/'sample_manifest.json', dict(business_ids=sample_ids, rule='maximum-source/target and first zero-Filtered train case; frozen before model execution'))
    results = []
    try:
        for condition, ds in data.items():
            model, tokenizer, device = init_gpu(config,manifest,tm,mc)
            collator = LEDCompleteCollator(config['source_limit'],config['label_limit'],1024)
            stress = LEDCompleteCollator(config['source_limit'],config['label_limit'],1024,5120,768)
            o = config['optimization']
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=o['learning_rate'],
                                         weight_decay=o['weight_decay'], betas=(o['adam_beta1'],o['adam_beta2']), eps=o['adam_epsilon'])
            scaler = torch.cuda.amp.GradScaler(enabled=o['fp16'])
            feats = [ds[ds.business_ids.index(b)] for b in sample_ids]
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            gradients = []; timed = []; losses = []
            bsize = o['physical_batch_size']
            # Stress includes masked padding, not synthetic review/teacher content.
            for step in range(config['feasibility']['overfit_steps']):
                model.train(); optimizer.zero_grad(set_to_none=True)
                use_stress = step < config['feasibility']['timed_microsteps']
                mini = [feats[(step*bsize+j) % len(feats)] for j in range(bsize)]
                batch = {k:v.cuda() for k,v in (stress if use_stress else collator)(mini).items()}
                torch.cuda.synchronize(); start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16): loss=model(**batch,use_cache=False).loss
                require(torch.isfinite(loss).item(), 'Non-finite forward loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                grad = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                require(grad and all(torch.isfinite(v).all().item() for v in grad), 'Non-finite/missing adapter gradients')
                gradients.append(any(torch.count_nonzero(v).item() > 0 for v in grad))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],o['max_grad_norm'])
                scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                if use_stress and step > 0: timed.append(time.perf_counter()-start)
                losses.append(float(loss.detach())); del loss, batch
            require(all(gradients), 'No effective LoRA gradients')
            checkpoint = folder / condition / 'checkpoint_sanity'; checkpoint.mkdir(parents=True)
            model.save_pretrained(checkpoint,safe_serialization=True)
            state = dict(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                         step=len(losses), settings_signature=settings_signature(config))
            torch.save(state,checkpoint/'optimizer_scaler_rng.pt')
            saved = load_file(str(checkpoint/'adapter_model.safetensors'))
            next(p for p in model.parameters() if p.requires_grad).data.add_(0.01)
            set_peft_model_state_dict(model,saved)
            require(all(torch.equal(v.cpu(),saved[k]) for k,v in get_peft_model_state_dict(model).items()), 'Adapter checkpoint roundtrip failed')
            loaded=torch.load(checkpoint/'optimizer_scaler_rng.pt',map_location='cuda',weights_only=False)
            optimizer.load_state_dict(loaded['optimizer']); scaler.load_state_dict(loaded['scaler'])
            torch.set_rng_state(loaded['torch_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in loaded['cuda_rng']])
            np.random.set_state(loaded['numpy_rng']); random.setstate(loaded['python_rng'])
            # Resume an actual update with restored optimizer/scaler/RNG rather than only reading files.
            model.train(); optimizer.zero_grad(set_to_none=True)
            batch={k:v.cuda() for k,v in collator(feats[:bsize]).items()}
            with torch.autocast('cuda',dtype=torch.float16): resumed_loss=model(**batch,use_cache=False).loss
            require(torch.isfinite(resumed_loss).item(),'Non-finite resumed loss')
            scaler.scale(resumed_loss).backward(); scaler.unscale_(optimizer)
            require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad and p.grad is not None),'Non-finite resumed gradient')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],o['max_grad_norm'])
            scaler.step(optimizer); scaler.update(); del resumed_loss,batch
            generated=[generate_case(model,tokenizer,feature,collator,config,condition,bid) for bid,feature in zip(sample_ids,feats)]
            allocated=torch.cuda.max_memory_allocated()/1024**3; reserved=torch.cuda.max_memory_reserved()/1024**3
            headroom_ok=reserved <= device['vram_GiB']*(1-config['feasibility']['recommended_free_headroom_fraction'])
            results.append(dict(condition=condition, device=device, physical_batch_size=bsize, stress_shapes=dict(source=5120,labels=768),
                                stress_valid_source_lengths=[len(v['input_ids']) for v in feats],
                                stress_valid_label_lengths=[len(v['labels']) for v in feats], stress_padding_masked=True,
                                finite_loss_and_nonzero_gradients=True, losses=losses, checkpoint_roundtrip=True,
                                resumed_optimizer_step=True, checkpoint_state_step=loaded['step'],
                                peak_allocated_GiB=allocated,peak_reserved_GiB=reserved,headroom_ok=headroom_ok,
                                warmed_stress_microstep_seconds=timed, generation=generated,
                                overfit_note='Constant-LR single-example diagnostic updates, not the full accumulation/scheduler experiment; format/loss trajectory require researcher review.'))
            del model,optimizer,scaler; torch.cuda.empty_cache()
        status = 'GPU_FEASIBILITY_PASS_PENDING_RESEARCHER_REVIEW' if all(v['headroom_ok'] for v in results) else 'GPU_FEASIBILITY_HEADROOM_BLOCKER'
        report=dict(status=status, settings_signature=settings_signature(config),config_sha256=sha(path),results=results,
                    generation_format_warnings=sum(not v['section_format_ok'] for r in results for v in r['generation']),
                    generation_cap_warnings=sum(v['cap_reached'] for r in results for v in r['generation']),
                    full_validation_epoch_seconds=None, full_training_time_estimate=None,
                    note='Compatibility/memory PASS is separate from format/informativeness/overfit assessment. No full training or test use.')
        write_new(folder/'report.json',report); print(json.dumps(report))
    except Exception as exc:
        write_new(folder/'failure_report.json',dict(status='GPU_FEASIBILITY_BLOCKED',error_type=type(exc).__name__,error=str(exc),
                  completed_conditions=results,automatic_fallback=False,settings_signature=settings_signature(config)))
        raise


def train(config, path, condition, resume):
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainerCallback
    execution_gate(config,'train')
    manifest,tm,mc=verify_bundle(config)
    info_train=manifest['datasets'][f'train_{condition}']; info_val=manifest['datasets'][f'validation_{condition}']
    ds=PackedLEDDataset(info_train,5000,config['label_limit']); val=PackedLEDDataset(info_val,5000,config['label_limit'])
    folder=ROOT/config['output_root']/condition
    run_binding=dict(settings_signature=settings_signature(config),stage_h_manifest_sha256=sha(ROOT/config['stage_h_manifest']),
                     condition=condition,runner_sha256=sha(Path(__file__)),data_code_sha256=sha(ROOT/'scripts/stage_hi_led_data.py'))
    if resume:
        require(read(folder/'run_binding.json') == run_binding,'Resume config/data/code differs from original run')
        r=Path(resume).resolve(); require(r.is_relative_to(folder.resolve()) and (r/'trainer_state.json').is_file(),'Invalid/cross-condition checkpoint')
    else:
        folder.mkdir(parents=True,exist_ok=False); write_new(folder/'run_binding.json',run_binding)
    model,tokenizer,device=init_gpu(config,manifest,tm,mc)
    write_path=folder/'runtime_initial.json'
    if not write_path.exists(): write_new(write_path,dict(config=config,environment=versions(),device=device))
    o=config['optimization']
    args=Seq2SeqTrainingArguments(output_dir=str(folder),learning_rate=o['learning_rate'],weight_decay=o['weight_decay'],
          per_device_train_batch_size=o['physical_batch_size'],per_device_eval_batch_size=o['validation_batch_size'],
          gradient_accumulation_steps=o['gradient_accumulation_steps'],num_train_epochs=o['num_train_epochs'],
          optim=o['optimizer'],adam_beta1=o['adam_beta1'],adam_beta2=o['adam_beta2'],adam_epsilon=o['adam_epsilon'],
          lr_scheduler_type=o['lr_scheduler_type'],warmup_ratio=o['warmup_ratio'],max_grad_norm=o['max_grad_norm'],
          fp16=o['fp16'],bf16=o['bf16'],gradient_checkpointing=o['gradient_checkpointing'],
          gradient_checkpointing_kwargs=o['gradient_checkpointing_kwargs'],eval_strategy=o['eval_strategy'],
          save_strategy=o['save_strategy'],save_total_limit=o['save_total_limit'],load_best_model_at_end=True,
          metric_for_best_model=o['metric_for_best_model'],greater_is_better=o['greater_is_better'],
          seed=config['seed'],data_seed=config['seed'],full_determinism=o['full_determinism'],
          dataloader_num_workers=0,report_to='none',logging_steps=10,remove_unused_columns=False,predict_with_generate=False,
          label_smoothing_factor=0.0)
    class FiniteLoss(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            for key in ('loss','eval_loss'):
                if logs and key in logs: require(math.isfinite(logs[key]),'Non-finite logged loss')
    class CompleteLEDTrainer(Seq2SeqTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            # LED has no num_items_in_batch kwarg. Check each actual loss before AMP/backward.
            outputs=model(**inputs,use_cache=False)
            require(torch.isfinite(outputs.loss).item(),'Non-finite actual LED loss')
            return (outputs.loss,outputs) if return_outputs else outputs.loss
    trainer=CompleteLEDTrainer(model=model,args=args,train_dataset=ds,eval_dataset=val,processing_class=tokenizer,
              data_collator=LEDCompleteCollator(5000,config['label_limit'],1024),
              callbacks=[EarlyStoppingCallback(early_stopping_patience=o['early_stopping_patience']),FiniteLoss()])
    # PEFT forwards **kwargs, but LED does not accept num_items_in_batch. Keep standard gradient-accumulated loss scaling.
    trainer.model_accepts_loss_kwargs=False
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    result=trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(folder/'best_adapter')); trainer.save_state(); tokenizer.save_pretrained(folder/'best_adapter')
    generated=[]; collator=LEDCompleteCollator(5000,config['label_limit'],1024)
    with (folder/'validation_predictions.jsonl').open('x',encoding='utf-8') as stream:
        for i,bid in enumerate(val.business_ids):
            prediction=generate_case(model,tokenizer,val[i],collator,config,condition,bid)
            stream.write(json.dumps(prediction,ensure_ascii=False)+'\n'); generated.append(prediction)
    write_new(folder/'completion_report.json',dict(status='TRAINING_COMPLETED_STOP_BEFORE_TEST',condition=condition,
              settings_signature=settings_signature(config),metrics=result.metrics,best_checkpoint=trainer.state.best_model_checkpoint,
              best_eval_loss=trainer.state.best_metric,total_seconds=time.perf_counter()-start,
              peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              validation_generation_format_warnings=sum(not v['section_format_ok'] for v in generated),
              validation_generation_cap_warnings=sum(v['cap_reached'] for v in generated),test_used=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--mode',choices=['offline-preflight','feasibility','train'],default='offline-preflight')
    parser.add_argument('--condition',choices=['reviews_only','unfiltered','filtered'])
    parser.add_argument('--execute',action='store_true',help='Required for any explicitly approved GPU execution')
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--report-directory',help='Fresh offline report directory; preserves previous preflight reports')
    args=parser.parse_args(); cfg=read(args.config)
    if args.mode=='offline-preflight': preflight(cfg,args.config,args.report_directory)
    else:
        require(args.execute,'GPU execution flag missing; default is offline preparation only')
        execution_gate(cfg,args.mode)  # Before CUDA allocation or pretrained download.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if args.mode=='feasibility': feasibility(cfg,args.config)
        else:
            require(args.condition,'Choose a training condition explicitly')
            train(cfg,args.config,args.condition,args.resume_from_checkpoint)
