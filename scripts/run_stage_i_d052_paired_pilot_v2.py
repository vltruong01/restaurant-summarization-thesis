"""D-052 v2: conditionally authorized paired F pilot using unchanged D-049 training math."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import TrainerCallback

import run_stage_i_paired_lora_pilot_v1 as prepared
from run_stage_i_d050_feasibility_v1 import parameters
from stage_i_paired_pilot import ROOT, ARMS, read, require, sha, write_new, inspect_attached, verify_mask_batch

CONFIG=ROOT/'configs/stage_i_paired_lora_pilot_v1.json'
OUT=ROOT/'outputs/stage_i/d052_paired_pilot_v1'


def input_digest(ids):
    return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    from stage_i_d052_authorization import conditional_training_gate
    config,prep,approval=conditional_training_gate(args.approval,args.arm,args.execute,Path(__file__))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    audit=OUT/args.arm
    audit.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    model_ref=[];trainer_ref=[];initial={};train_seen=[];validation_seen=[]
    feature_map={};step_records=[];error=None

    def event(name,**value):
        row={'seconds':time.perf_counter()-started,'event':name,**value}
        with (audit/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
        print(json.dumps({'arm':args.arm,**row},ensure_ascii=False),flush=True)

    def memory():
        free,total=torch.cuda.mem_get_info()
        return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'device_free':free,'device_total':total}

    original_datasets=prepared.datasets
    def audited_datasets(*call_args,**call_kwargs):
        train,val,tm,mc,cloud=original_datasets(*call_args,**call_kwargs)
        for split,dataset in [('train',train),('validation',val)]:
            for i,bid in enumerate(dataset.business_ids):
                feature=dataset[i];digest=input_digest(feature['input_ids'])
                require(digest not in feature_map,'Ambiguous source ID')
                feature_map[digest]={'business_id':bid,'split':split,'feature':feature}
        return train,val,tm,mc,cloud
    original_load=prepared.load_gpu
    def audited_load(*call_args,**call_kwargs):
        destination=ROOT/config['run_root']/args.arm/'train'
        write_new(destination/'d052_authorization_binding.json',{'decision':'D-052',
            'approval_sha256':sha(args.approval),'wrapper_sha256':sha(Path(__file__))})
        model,scope,initialization=original_load(*call_args,**call_kwargs)
        initial.update(parameters(model));model_ref.append(model)
        write_new(audit/'parameter_hashes_before.json',initial)
        if args.arm==ARMS[1]:
            previous=read(OUT/ARMS[0]/'initial_decoder_hashes.json')
            shared={k:v for k,v in initialization.items() if '.decoder.layers.' in k}
            require(previous==shared,'Paired initial decoder tensors differ')
        write_new(audit/'initial_decoder_hashes.json',{k:v for k,v in initialization.items() if '.decoder.layers.' in k})
        runtime={'scope':scope,'initialization':initialization,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'CUDA':torch.version.cuda,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'config_sha256':sha(CONFIG),'approval_sha256':sha(args.approval),'wrapper_sha256':sha(Path(__file__))}
        if args.arm==ARMS[1]:
            first_runtime=read(OUT/ARMS[0]/'runtime.json')
            for key in ['gpu','torch','CUDA','deterministic_algorithms','cudnn_deterministic','cudnn_benchmark','matmul_allow_tf32','cudnn_allow_tf32']:
                require(runtime[key]==first_runtime[key],'Paired runtime changed: '+key)
        write_new(audit/'runtime.json',runtime)
        require(not torch.are_deterministic_algorithms_enabled(),'Unexpected deterministic policy')
        # This authorization can never generate a summary, even if a caller asks.
        def denied_generation(*unused_args,**unused_kwargs):
            raise ValueError('D-052 authorizes teacher-forced training/validation only; generation denied')
        model.generate=denied_generation
        event('fresh_model_ready',modules=len(scope['modules']),trainable_parameters=scope['trainable_parameters'])
        return model,scope,initialization

    def check_batch(inputs,split):
        batch={k:v.detach().cpu() for k,v in inputs.items() if isinstance(v,torch.Tensor)}
        require(batch['input_ids'].shape[0]==1,'Batch changed')
        n=int(batch['attention_mask'][0].sum())
        ids=batch['input_ids'][0,:n].tolist();row=feature_map[input_digest(ids)]
        require(row['split']==split,'Train/validation leakage')
        verify_mask_batch(batch,[row['feature']])
        f=row['feature'];require(f['input_ids'][0]==f['labels'][0]==0 and f['input_ids'][-1]==f['labels'][-1]==2,'BOS/EOS differs')
        return row['business_id']

    # Warm the exact local import set before resolving the live lazy package.
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback,TrainerCallback,set_seed
    import importlib
    live_transformers=importlib.import_module('transformers')
    original_trainer=live_transformers.Seq2SeqTrainer
    class AuditedTrainer(original_trainer):
        def __init__(self,*call_args,**call_kwargs):
            super().__init__(*call_args,**call_kwargs)
            trainer_ref.append(self)
            require(not self.args.predict_with_generate and self.args.label_smoothing_factor==0,'Training settings changed')
            self.add_callback(AuditCallback(self))
            self._d052_audit_attached=True
            event('trainer_audit_attached',class_MRO=[c.__name__ for c in type(self).__mro__])
            write_new(audit/'actual_training_arguments.json',self.args.to_dict())
        def training_step(self,model,inputs,*call_args,**call_kwargs):
            bid=check_batch(inputs,'train')
            loss=super().training_step(model,inputs,*call_args,**call_kwargs)
            require(torch.isfinite(loss).item(),'Nonfinite returned training loss')
            train_seen.append(bid)
            with (audit/'microbatch_loss_and_order.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'physical_microbatch':len(train_seen),'business_id':bid,
                    'training_step_return_loss_after_accumulation_scaling':float(loss.detach())})+'\n')
            return loss
        def prediction_step(self,model,inputs,*call_args,**call_kwargs):
            bid=check_batch(inputs,'validation');result=super().prediction_step(model,inputs,*call_args,**call_kwargs)
            validation_seen.append(bid)
            return result

    class AuditCallback(TrainerCallback):
        def __init__(self,trainer):self.trainer=trainer;self.pending=None
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            require(self.trainer.model_accepts_loss_kwargs is False,'Accumulation loss contract changed')
            require(state.max_steps==128 and args.num_train_epochs==2,'Unexpected duration')
            inspect_attached(model,args_arm)
            event('training_started',max_steps=state.max_steps,epochs=2)
        def on_pre_optimizer_step(self,args,state,control,optimizer=None,**kwargs):
            scaler=self.trainer.accelerator.scaler
            self.pending={'step':state.global_step+1,'used_lr':optimizer.param_groups[0]['lr'],
                'scaler_before':scaler.get_scale() if scaler else None}
        def on_optimizer_step(self,args,state,control,optimizer=None,**kwargs):
            skipped=bool(self.trainer.accelerator.optimizer_step_was_skipped)
            observed=sorted({int(s['step']) for s in optimizer.state.values()})
            scaler=self.trainer.accelerator.scaler
            self.pending.update(optimizer_steps=observed,AMP_optimizer_skipped=skipped,
                scaler_after=scaler.get_scale() if scaler else None)
            # Persist the abnormal event before stopping.
            if skipped or observed!=[state.global_step+1]:
                event('optimizer_step_blocker',**self.pending)
            require(not skipped and observed==[state.global_step+1],'AMP skipped or incorrect optimizer step')
        def on_step_end(self,args,state,control,lr_scheduler=None,**kwargs):
            scheduler=getattr(lr_scheduler,'scheduler',lr_scheduler)
            self.pending.update(global_step=state.global_step,scheduler_last_epoch=scheduler.last_epoch,
                next_lr=scheduler.get_last_lr(),memory=memory())
            with (audit/'optimizer_scheduler_scaler_memory.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(self.pending)+'\n');stream.flush()
            require(scheduler.last_epoch==state.global_step,'Scheduler drift')
            step_records.append(self.pending);self.pending=None
            if state.global_step<=2 or state.global_step%8==0:
                event('optimizer_progress',step=state.global_step,physical_microbatches=len(train_seen),
                    lr=scheduler.get_last_lr(),peak_allocated_MiB=memory()['peak_allocated']/2**20)
        def on_log(self,args,state,control,logs=None,**kwargs):
            logs=logs or {}
            for key in ['loss','eval_loss','grad_norm']:
                if key in logs:require(math.isfinite(float(logs[key])),'Nonfinite logged '+key)
            event('trainer_log',step=state.global_step,logs=logs)
        def on_save(self,args,state,control,**kwargs):
            path=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json','checkpoint_integrity.json']
            require(state.global_step in [64,128],'Unexpected checkpoint')
            require(all((path/n).is_file() for n in required),'Checkpoint state incomplete')
            saved=read(path/'checkpoint_integrity.json')
            for name,digest in saved['files'].items():require(sha(path/name)==digest,'Checkpoint corruption')
            write_new(audit/f'checkpoint_{state.global_step}_audit.json',{'path':path.relative_to(ROOT).as_posix(),
                'required_files':required,'sha256':{n:sha(path/n) for n in required}})
            event('epoch_checkpoint_verified',step=state.global_step)
        def on_evaluate(self,args,state,control,metrics=None,**kwargs):
            event('validation_completed',step=state.global_step,validation_examples_seen=len(validation_seen),metrics=metrics)

    args_arm=args.arm
    prepared.datasets=audited_datasets;prepared.load_gpu=audited_load
    original_train_method=original_trainer.train
    def guarded_train(trainer,*call_args,**call_kwargs):
        require(isinstance(trainer,AuditedTrainer) and trainer in trainer_ref and
                getattr(trainer,'_d052_audit_attached',False),'D052 audit not attached; stop before training')
        return original_train_method(trainer,*call_args,**call_kwargs)
    original_trainer.train=guarded_train
    live_transformers.Seq2SeqTrainer=AuditedTrainer
    from transformers import Seq2SeqTrainer as resolved_trainer
    require(resolved_trainer is AuditedTrainer,'Live Trainer import not instrumented; stop before training')
    try:
        event('D052_conditional_train_authorized',D051_strict_replay='FAIL_UNCHANGED',scope='Fresh F-only512x2, no generation')
        prepared.train_pilot(CONFIG,config,prep,args.arm,approval,resume=None)
        expected=read(prep/'epoch_orders.json')['business_ids']
        require(train_seen==[bid for epoch in expected for bid in epoch],'Actual train order differs')
        val_ids=read(prep/'pilot_manifest.json')['validation_loss_business_ids']
        require(validation_seen==val_ids*2,'Validation order/count differs')
        require(len(step_records)==128,'Incorrect actual update count')
        write_new(audit/'ACTUAL_ORDER_VERIFICATION.json',{'train_microbatches':len(train_seen),'validation_batches':len(validation_seen),
            'train_ids_order_sha256':hashlib.sha256(json.dumps(train_seen).encode()).hexdigest(),
            'exact_epoch_plans_match':True,'validation_all200_twice':True})
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        write_new(audit/'failure.json',error);event('affected_arm_stopped',error_type=type(exc).__name__,message=str(exc))
    finally:
        if model_ref:
            after=parameters(model_ref[0]);write_new(audit/'parameter_hashes_after.json',after)
            changed=[n for n,p in initial.items() if not p['requires_grad'] and after[n]['sha256']!=p['sha256']]
            write_new(audit/'FROZEN_BACKBONE_INTEGRITY.json',{'changed_frozen_tensors':changed,
                'frozen_gradients_absent':all(p.grad is None for p in model_ref[0].parameters() if not p.requires_grad)})
            if changed:error=error or {'type':'FrozenWeightsChanged','message':'Unexpected base mutation'}
            write_new(audit/'GPU_MEMORY_FINAL.json',memory())
        write_new(audit/'observation_completion.json',{'status':'TRAINING_AND_AUDIT_PASS' if not error else 'AFFECTED_ARM_STOPPED',
            'error':error,'seconds':time.perf_counter()-started,'actual_microbatches':len(train_seen),
            'actual_optimizer_steps':len(step_records),'validation_batches':len(validation_seen),
            'generation_calls':0,'D051_replay_still_FAIL':True,'conditional_authorization_only':True})
    if error:raise RuntimeError(error['type']+': '+error['message'])


if __name__=='__main__':main()
