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


from led_large_gradient_evidence_v3 import inspect_gradients, finite_value, snapshot_raw_gradients, snapshot_rng

PATCH_PREP = ROOT/'outputs/stage_i/led_large_gradient_gate_v3'
PATCH_CONFIG = ROOT/'configs/stage_i_led_large_gradient_gate_v3.json'
PATCH_RUNNER = ROOT/'scripts/run_stage_i_led_large_gradient_gate_v3.py'

def binding():
    return dict(config_sha256=sha(PATCH_CONFIG),runner_sha256=sha(PATCH_RUNNER),
                preflight_manifest_sha256=sha(PATCH_PREP/'PREFLIGHT_MANIFEST.json'))

def authorize_patch(path, execute):
    require(execute and path is not None,'New diagnostic approval required; GPU disabled')
    a=read(path)
    require(a.get('GPU_execution_approved') is True and a.get('researcher_explicit_authorization'), 'New explicit researcher approval required')
    require(a.get('experiment')=='LED_LARGE_GRADIENT_GATE_V3' and a.get('approval_source')=='researcher_message', 'Wrong approval scope/provenance')
    require(a.get('scopes')==['feasibility'] and a.get('training_authorized') is False, 'Gate-only scope required')
    require(all(a.get(k)==v for k,v in binding().items()),'New diagnostic approval hash mismatch')
    require(a.get('execution_deadline_utc')=='2026-10-21T17:11:48+00:00','Original deadline must remain unchanged')
    require(a.get('feasibility_updates_max')==3 and a.get('feasibility_microbatches_max')==24
            and a.get('technical_generation_calls_max')==2,'Gate ceiling changed')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False,'Additional scope forbidden')
    check_deadline(a)
    return dict(a,_approval_sha256=sha(path))

def preflight_patch():
    c=preflight()  # original v2 runtime pins and all immutable frozen input bindings
    new=read(PATCH_CONFIG)
    require(all(new[k]==v for k,v in c.items()),'Original v2 config changed')
    require(new['diagnostic_scope']['train_authorized'] is False,'Gate-only revision required')
    for rel,digest in read(PATCH_PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Diagnostic/frozen dependency changed: '+rel)
    return c



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
    diagnostics = dict(stage='before_model_load', completed_optimizer_updates=0)
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
                diagnostics.update(step=step, stage='before_accumulation', scaler_scale=scaler.get_scale())
                opt.zero_grad(set_to_none=True);losses = []
                for j in range(8):
                    check_deadline(a)
                    feature = features[((step-1)*8+j)%len(features)]
                    batch = {k:v.cuda() for k,v in pack([feature]).items()}
                    diagnostics.update(microbatch=j+1, business_id=ids[((step-1)*8+j)%len(ids)], stage='forward')
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss = model(**batch,use_cache=False).loss
                    diagnostics.update(stage='forward_complete', CE_loss=finite_value(loss.detach().item()), scaler_scale=scaler.get_scale())
                    append(p/'MICROBATCH_EVIDENCE.jsonl',dict(diagnostics))
                    require(torch.isfinite(loss).item(), 'Nonfinite CE loss')
                    losses.append(float(loss.detach()));scaler.scale(loss/8).backward()
                    diagnostics.update(stage='scaled_backward_complete')
                    evidence=inspect_gradients(model)
                    append(p/'SCALED_GRADIENT_EVIDENCE.jsonl',dict(step=step,microbatch=j+1,scaler_scale=scaler.get_scale(),**evidence))
                    memory.check()
                scaler.unscale_(opt)
                diagnostics.update(stage='after_unscale_before_original_gradient_assertion',scaler_scale=scaler.get_scale())
                evidence=inspect_gradients(model,detailed=True)
                write(p/f'UNSCALED_GRADIENTS_step{step}.json',dict(diagnostics,**evidence))
                if evidence['missing'] or evidence['nonfinite']:
                    snapshot_raw_gradients(model,p/f'RAW_UNSCALED_GRADIENTS_step{step}.pt')
                grads = gradient_record(model,expect_later_A=step>=3)
                require(len(grads) == 456, 'Missing intended large adapter gradients')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                lr = opt.param_groups[0]['lr'];scaler.step(opt);scaler.update();sched.step();torch.cuda.synchronize()
                require({int(s['step']) for s in opt.state.values()} == {step}, 'AMP skipped update')
                record = dict(step=step, losses=losses, used_lr=lr, gradients=grads)
                write(p/f'update_{step}.json',record);records.append(record);memory.check()
                diagnostics.update(completed_optimizer_updates=step,stage='update_complete')
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
        saved_traceback=traceback.format_exc()
        try:
            if 'memory' in locals():
                diagnostics['memory_summary']=memory.summary()
                write(p/'FAILURE_MEMORY_SAMPLES.json',memory.samples)
            if 'scaler' in locals():
                diagnostics['scaler_state']=scaler.state_dict()
            if 'opt' in locals():
                diagnostics['optimizer_state_steps']=sorted({int(s['step']) for s in opt.state.values()})
            if 'model' in locals():
                diagnostics['RNG_fingerprints']=snapshot_rng(torch)
            write(p/'FAILURE_DIAGNOSTICS.json',diagnostics)
        except BaseException as capture_error:
            write(p/'FAILURE_DIAGNOSTIC_CAPTURE_ERROR.json',dict(error=str(capture_error)))
        write(p/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=saved_traceback,automatic_retry=False))
        raise


def main():
    ap=argparse.ArgumentParser(description='Versioned evidence-only LED-large feasibility gate. No training retry.')
    ap.add_argument('mode',choices=('feasibility',))
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--approval',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    a=authorize_patch(args.approval,args.execute)
    c=preflight_patch()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out=args.output.resolve()
    require(out!=ROOT and not out.is_relative_to(ROOT/'data') and not out.exists(),'Fresh diagnostic output required; preserve failed v2 run')
    feasible(c,a,out)

if __name__=='__main__':main()

