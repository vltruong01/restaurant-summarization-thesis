"""Prepared complete large F runner. No execution until external hash-bound GPU approval."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time
import traceback

from stage_i_led_large_runtime_v2 import (ROOT, CONFIG, PREP, LARGE_PREP, read, write, sha, require,
    append, authorize, preflight, binding, check_deadline, require_gate, load_model, generation,
    verify_backbone, optimizer_scheduler, training_arguments, callback_class)


class MemoryMeter:
    """Measure reservation plus observed non-Torch occupancy; no unsupported VRAM estimate."""
    def __init__(self, torch):
        self.torch = torch;self.samples = [];self.failure = None
        self.stop = threading.Event();self.thread = None

    @staticmethod
    def estimate(samples, peak_reserved):
        require(bool(samples), 'No measured GPU memory samples')
        total = samples[-1]['total']
        non_torch = max(max(0, s['used']-s['reserved']) for s in samples)
        used = max(max(s['used'] for s in samples), peak_reserved+non_torch)
        return dict(total_bytes=total, conservative_peak_used_bytes=used, headroom_fraction=1-used/total)

    def sample(self):
        free,total = self.torch.cuda.mem_get_info()
        self.samples.append(dict(time=time.perf_counter(), used=total-free, total=total,
                                 reserved=self.torch.cuda.memory_reserved()))

    def __enter__(self):
        self.torch.cuda.reset_peak_memory_stats();self.sample()
        def monitor():
            while not self.stop.wait(.5):
                try:
                    self.sample()
                except Exception as e:
                    self.failure = str(e);return
        self.thread = threading.Thread(target=monitor, daemon=True);self.thread.start()
        return self

    def summary(self):
        return dict(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
                    **self.estimate(list(self.samples), self.torch.cuda.max_memory_reserved()))

    def check(self):
        require(self.failure is None, 'GPU memory measurement failed: '+str(self.failure))
        self.sample()
        require(self.summary()['headroom_fraction'] >= .15, 'Measured GPU headroom below 15%; STOP, no fallback')

    def __exit__(self, *exc):
        self.stop.set();self.thread.join(timeout=5)
        self.sample()


def data(c):
    from stage_hi_led_data import PackedLEDDataset
    manifest = read(ROOT/c['stage_h_manifest'])
    train = PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768,root=ROOT)
    val = PackedLEDDataset(manifest['datasets']['validation_filtered'],5000,768,root=ROOT)
    require(len(train) == 2000 and len(val) == 200 and not set(train.business_ids)&set(val.business_ids), 'Split mismatch')
    require(train.business_ids == read(LARGE_PREP/'EPOCH_ORDERS.json')['training_business_ids'], 'Train ordering differs')
    return train,val


def collator(stress=False):
    from stage_hi_led_data import LEDCompleteCollator
    from stage_i_paired_pilot import verify_mask_batch
    original = LEDCompleteCollator(5000,768,1024,pad_source_to=5120 if stress else None,
                                  pad_label_to=768 if stress else None)
    def checked(features):
        batch = original(features);verify_mask_batch(batch,features);return batch
    return checked


def checkpoint_verified(p):
    proof = read(p/'checkpoint_integrity.json')
    require(proof['binding'] == binding(), 'Foreign checkpoint binding')
    for name,h in proof['files'].items():
        require(Path(name).name == name and sha(p/name) == h, 'Checkpoint file changed')
    return proof


def check_gpu_same(gate, torch):
    require(torch.cuda.get_device_name() == gate['GPU']
            and torch.cuda.get_device_properties(0).total_memory == gate['GPU_total_bytes'], 'GPU differs from feasibility gate')
    require(hardware_profile(torch) == gate['hardware_profile'], 'GPU runtime/driver differs from gate; STOP')


def hardware_profile(torch):
    import subprocess
    p = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                       capture_output=True,text=True,check=True,timeout=15)
    return dict(driver=p.stdout.strip(),torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                capability=list(torch.cuda.get_device_capability()),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)


def feasible(c, a, out):
    import torch
    from peft import get_peft_model_state_dict
    from stage_i_paired_pilot import gradient_record
    from run_stage_i_d050_feasibility_v1 import states
    from run_stage_i_paired_lora_pilot_v1 import save_feas_checkpoint, restore_feas_checkpoint
    p = out/'feasibility';p.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    try:
        write(p/'AUTHORIZATION.json', a);write(p/'BINDING.json', binding())
        train,_ = data(c);ids = read(LARGE_PREP/'FEASIBILITY_CASES.json')['business_ids']
        features = [train[train.business_ids.index(b)] for b in ids]
        model,before,scope = load_model(c);write(p/'INITIAL_MODEL_BINDINGS.json', scope)
        opt,sched = optimizer_scheduler(model,c);scaler = torch.cuda.amp.GradScaler(enabled=True)
        pack = collator(stress=True);records = []
        with MemoryMeter(torch) as memory:
            model.train()
            for step in range(1,4):
                opt.zero_grad(set_to_none=True);losses = []
                for j in range(8):
                    check_deadline(a)
                    feature = features[((step-1)*8+j)%len(features)]
                    batch = {k:v.cuda() for k,v in pack([feature]).items()}
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss = model(**batch,use_cache=False).loss
                    require(torch.isfinite(loss).item(), 'Nonfinite CE loss')
                    losses.append(float(loss.detach()));scaler.scale(loss/8).backward()
                    memory.check()
                scaler.unscale_(opt);grads = gradient_record(model,expect_later_A=step>=3)
                require(len(grads) == 456, 'Missing intended large adapter gradients')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                lr = opt.param_groups[0]['lr'];scaler.step(opt);scaler.update();sched.step();torch.cuda.synchronize()
                require({int(s['step']) for s in opt.state.values()} == {step}, 'AMP skipped update')
                record = dict(step=step, losses=losses, used_lr=lr, gradients=grads)
                write(p/f'update_{step}.json',record);records.append(record);memory.check()
            opt.zero_grad(set_to_none=True)
            target = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            previous = states(opt,sched,scaler)
            save_feas_checkpoint(model,opt,sched,scaler,p/'checkpoint_step3',binding(),3)
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.zero_()
            restored = restore_feas_checkpoint(model,opt,sched,scaler,p/'checkpoint_step3',binding())
            require(restored['step'] == 3 and states(opt,sched,scaler) == previous, 'Optimizer/scaler/scheduler/RNG restore differs')
            require(all(torch.equal(v.detach().cpu(),target[k]) for k,v in get_peft_model_state_dict(model).items()), 'Adapter roundtrip differs')
            verify_backbone(model,before)
            g = generation(c);model.eval()
            write(p/'EFFECTIVE_GENERATION_CONFIG.json',g.to_dict())
            for i in range(2):
                check_deadline(a)
                batch = {k:v.cuda() for k,v in pack([features[i]]).items() if k != 'labels'}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    tokens = model.generate(**batch,generation_config=g)[0].cpu().tolist()
                write(p/f'train_memory_trace_{i+1}.json',dict(business_id=ids[i],token_ids=tokens,
                      scope='TRAIN_INPUT_TECHNICAL_MEMORY_ONLY_NOT_QUALITY_OUTPUT',generation_call=i+1))
                require(len(tokens) <= 768, 'Technical trace exceeds approved full-length memory limit')
                memory.check()
            require(all(torch.equal(v.detach().cpu(),target[k]) for k,v in get_peft_model_state_dict(model).items()), 'Generation changed adapters')
            verify_backbone(model,before);check_deadline(a)
        write(p/'MEMORY_SAMPLES.json',memory.samples)
        write(p/'REPORT.json',dict(status='PASS',binding=binding(),approval_sha256=a['_approval_sha256'],
              deadline_utc=a['execution_deadline_utc'],GPU=torch.cuda.get_device_name(),
              hardware_profile=hardware_profile(torch),
              GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory, optimizer_updates=3,
              microbatches=24,generation_calls=2,replay_updates=0,state_restore_exact=True,
              adapter_restore_exact=True,frozen_backbone_unchanged=True,seconds=time.perf_counter()-start,
              feasibility_states_discarded_before_fresh_F_train=True,D051_strict_replay='FAIL_UNCHANGED',**memory.summary()))
    except BaseException as exc:
        write(p/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def train(c, a, out, resume=None):
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
    from stage_i_paired_pilot import PairedEpochSampler
    gate = require_gate(out,a);folder = out/'train'
    if resume:
        resume = Path(resume).resolve()
        require(a.get('resume_after_interruption_approved') is True, 'Manual resume requires separate explicit scope')
        require(resume.is_relative_to(folder.resolve()) and sha(resume/'checkpoint_integrity.json') == a.get('resume_checkpoint_integrity_sha256'), 'Resume checkpoint not approved')
        checkpoint_verified(resume)
        require(not (folder/'COMPLETION.json').exists() and not (folder/'FAILURE_STOP_NO_RETRY.json').exists(), 'Completed or failed numerical run cannot silently resume')
        progress = [__import__('json').loads(s) for s in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        step = read(resume/'trainer_state.json')['global_step']
        require(step == max(x['step'] for x in progress), 'Uncheckpointed updates exist; rollback/repeated optimization NOT authorized')
        require(read(folder/'BINDING.json') == binding(), 'Original run binding differs')
        write(folder/f'RESUME_AUTHORIZATION_STEP_{step}.json',a)
    else:
        folder.mkdir(parents=True,exist_ok=False)
        write(folder/'BINDING.json',binding());write(folder/'AUTHORIZATION.json',a)
    start = time.perf_counter()
    try:
        trainset,val = data(c);model,before,scope = load_model(c);check_gpu_same(gate,torch)
        if not resume:
            write(folder/'INITIAL_MODEL_BINDINGS.json',scope)
        opt,sched = optimizer_scheduler(model,c);orders = read(LARGE_PREP/'EPOCH_ORDERS.json')['indices']
        class Trainer(Seq2SeqTrainer):
            def _get_train_sampler(self, train_dataset=None):
                return PairedEpochSampler(orders)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                check_deadline(a)
                output = model(**inputs,use_cache=False)
                require(torch.isfinite(output.loss).item(), 'Nonfinite train/validation loss')
                return (output.loss,output) if return_outputs else output.loss

            def _load_rng_state(self, checkpoint):
                # Torch2.1 weights_only=True rejects NumPy RNG pickle. Hash-verified OWN checkpoint only.
                import random
                import numpy as np
                checkpoint_verified(Path(checkpoint))
                s = torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
                require(set(s) == {'python','numpy','cpu','cuda'}, 'Single-GPU RNG state differs')
                random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'])
                require(isinstance(s['cuda'],torch.Tensor) and s['cuda'].dtype == torch.uint8,
                        'Expected pinned Trainer single-GPU CUDA RNG tensor')
                torch.cuda.set_rng_state(s['cuda'])
        with MemoryMeter(torch) as memory:
            callback = callback_class()(folder,a,memory)
            args = Seq2SeqTrainingArguments(**training_arguments(c,folder))
            trainer = Trainer(model=model,args=args,train_dataset=trainset,eval_dataset=val,
                data_collator=collator(),optimizers=(opt,sched),callbacks=[callback])
            callback.trainer = trainer;trainer.model_accepts_loss_kwargs = False;set_seed(42)
            if not resume:
                write(folder/'ACTUAL_TRAINING_ARGUMENTS.json',args.to_dict())
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            require(trainer.state.global_step == 1250 and trainer.state.epoch == 5, 'Incomplete planned five-epoch run')
            for step in (250,500,750,1000,1250):
                checkpoint_verified(folder/f'checkpoint-{step}')
            verify_backbone(model,before);memory.check();check_deadline(a)
        write(folder/'MEMORY_SAMPLES.json',memory.samples)
        write(folder/'FROZEN_BACKBONE_INTEGRITY.json',dict(unchanged=True,D051_strict_replay='FAIL_UNCHANGED'))
        write(folder/'COMPLETION.json',dict(status='TRAINING_COMPLETE',binding=binding(),steps=1250,epochs=5,
              metrics=result.metrics,seconds=time.perf_counter()-start,checkpoint_steps=[250,500,750,1000,1250],
              generation_calls=0,**memory.summary()))
    except BaseException as exc:
        write(folder/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def evaluate(c, a, out):
    import torch
    from transformers import LEDTokenizerFast
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_i_d053_evaluation import repetition_diagnostics, structural_diagnostics, token_termination
    gate = require_gate(out,a)
    complete = read(out/'train/COMPLETION.json')
    require(complete['binding'] == binding() and complete['steps'] == 1250, 'Training scope incomplete')
    require(not (out/'train/FAILURE_STOP_NO_RETRY.json').exists()
            and read(out/'train/FROZEN_BACKBONE_INTEGRITY.json')['unchanged'], 'Training not accepted technically')
    folder = out/'evaluate';folder.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        write(folder/'AUTHORIZATION.json',a);model,before,scope = load_model(c);check_gpu_same(gate,torch)
        cases = [__import__('json').loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
        require([x['business_id'] for x in cases] == read(LARGE_PREP/'EVALUATION_MANIFEST.json')['business_ids'], 'Fixed cases changed')
        tok = LEDTokenizerFast.from_pretrained(ROOT/c['source_tokenizer'],local_files_only=True)
        g = generation(c);model.eval();count = 0;pack = collator()
        write(folder/'EFFECTIVE_GENERATION_CONFIG.json',g.to_dict())
        with MemoryMeter(torch) as memory:
            for epoch,step in ((3,750),(5,1250)):
                checkpoint = out/'train'/f'checkpoint-{step}';checkpoint_verified(checkpoint)
                weights = load_file(checkpoint/'adapter_model.safetensors',device='cpu')
                require(set(weights) == set(get_peft_model_state_dict(model)), 'Wrong adapter state keys')
                set_peft_model_state_dict(model,weights)
                require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()), 'Adapter load mismatch')
                predictions = []
                for case in cases:
                    check_deadline(a);count += 1;require(count <= 48, 'Generation ceiling exceeded')
                    batch = pack([dict(input_ids=case['input_ids'],labels=case['label_ids'])])
                    batch = {k:v.cuda() for k,v in batch.items() if k != 'labels'}
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                        ids = model.generate(**batch,generation_config=g)[0].cpu().tolist()
                    require(0 < len(ids) <= 768, 'Generation length mismatch')
                    text = tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    row = dict(business_id=case['business_id'],subset=case['subset'],condition='filtered',epoch=epoch,
                        arm=f'LED_large_Expanded_epoch{epoch}',source_sha256=case['source_sha256'],token_ids=ids,text=text,
                        reference_sha256=sha(ROOT/c['final_references']),checkpoint_sha256=sha(checkpoint/'adapter_model.safetensors'),
                        automatic=dict(structure=structural_diagnostics(text,len(case['selected_photo_ids'])),
                            termination=token_termination(ids,g.to_dict()),repetition=repetition_diagnostics(text,ids,g.to_dict())),
                        researcher_output_annotation_status='NOT_REVIEWED')
                    write(folder/f'epoch{epoch}'/(case['business_id']+'.json'),row);predictions.append(row)
                    print(f'epoch{epoch}: {len(predictions)}/24 predictions saved',flush=True);memory.check()
                write(folder/f'epoch{epoch}'/'predictions.json',predictions)
                require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()), 'Generation changed adapters')
            require(count == 48, 'Missing evaluation outputs');verify_backbone(model,before);check_deadline(a)
        write(folder/'MEMORY_SAMPLES.json',memory.samples)
        write(folder/'COMPLETION.json',dict(status='48_OUTPUTS_COMPLETE_STOP_FOR_RESEARCHER_REVIEW',binding=binding(),
              generation_calls=48,seconds=time.perf_counter()-start,frozen_backbone_unchanged=True,**memory.summary()))
    except BaseException as exc:
        write(folder/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('feasibility','train','evaluate'))
    ap.add_argument('--execute',action='store_true');ap.add_argument('--approval',type=Path)
    ap.add_argument('--output',type=Path,required=True,help='Persistent fresh output folder; no existing mode overwrite')
    ap.add_argument('--resume',type=Path,help='Only explicitly approved intact epoch checkpoint; never automatic')
    args = ap.parse_args()
    require(not args.resume or args.mode == 'train', 'Resume applies only to training')
    # All authorization, file and runtime gates precede torch/CUDA/model imports.
    approval = authorize(args.approval,args.mode,args.execute);cfg = preflight()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT/'data') and out != ROOT, 'Unsafe run output location')
    check_deadline(approval)
    if args.mode == 'feasibility':
        require(not out.exists(), 'Fresh experiment output root required; no automatic repeat')
        feasible(cfg,approval,out)
    elif args.mode == 'train':
        train(cfg,approval,out,args.resume)
    else:
        evaluate(cfg,approval,out)


if __name__ == '__main__':
    main()
