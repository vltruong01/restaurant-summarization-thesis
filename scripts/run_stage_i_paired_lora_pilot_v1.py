"""Paired F pilot. Default offline; all GPU actions require a hash-bound future approval."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from stage_i_paired_pilot import (ROOT,ARMS,read,sha,require,write_new,signature,checked_preparation,execution_gate,
    lora_config,inspect_attached,initialize_named_adapters,meta_probe,gradient_record,verify_mask_batch,
    BusinessSubset,PairedEpochSampler)
from stage_hi_led_data import verify_bundle,PackedLEDDataset,LEDCompleteCollator

DEFAULT=ROOT/'configs/stage_i_paired_lora_pilot_v1.json'


def datasets(config,prep):
    cloud=ROOT/config['frozen_training_root'];old=read(cloud/config['original_training_config'])
    manifest,tm,mc=verify_bundle(old,root=cloud)
    sample=read(prep/'pilot_manifest.json')
    train=PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768,root=cloud)
    val=PackedLEDDataset(manifest['datasets']['validation_filtered'],5000,768,root=cloud)
    require(val.business_ids==sample['validation_loss_business_ids'],'Validation order differs')
    require(not set(sample['training_business_ids']) & set(val.business_ids),'Training/validation overlap')
    return BusinessSubset(train,sample['training_business_ids']),val,tm,mc,cloud


def load_gpu(config,prep,arm,approval,mc,cloud):
    # This function is reachable only after execution_gate. Never call in offline preparation/tests.
    from transformers import LEDForConditionalGeneration,set_seed
    from peft import LoraConfig,get_peft_model
    require(torch.cuda.is_available(),'CUDA unavailable; no fallback')
    set_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
    base=LEDForConditionalGeneration.from_pretrained(config['model_id'],revision=config['revision'],torch_dtype=torch.float32,
        local_files_only=not approval.get('allow_backbone_download',False))
    for key in ['attention_window','max_encoder_position_embeddings','max_decoder_position_embeddings','d_model','encoder_layers','decoder_layers','bos_token_id','pad_token_id','eos_token_id','decoder_start_token_id']:
        require(getattr(base.config,key)==mc[key],'Backbone config mismatch: '+key)
    model=get_peft_model(base,LoraConfig(**lora_config(config,arm)));scope=inspect_attached(model,arm)
    init=initialize_named_adapters(model,config['seed']);expected=read(prep/'initialization_hashes.json')
    require(init=={k:expected['per_module_sha256'][k] for k in scope['modules']},'Independent adapter initialization differs')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    model.to('cuda');set_seed(config['seed'])
    return model,scope,init


def optimizer_scheduler(model,config):
    from transformers import get_linear_schedule_with_warmup
    o=config['optimization'];params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':params,'weight_decay':o['weight_decay']},{'params':[],'weight_decay':0.0}],
        lr=o['learning_rate'],betas=(o['adam_beta1'],o['adam_beta2']),eps=o['adam_epsilon'])
    return optimizer,get_linear_schedule_with_warmup(optimizer,7,128)


def binding(config_path,prep,arm):
    return {'config_sha256':sha(config_path),'preparation_manifest_sha256':sha(prep/'PREPARATION_MANIFEST.json'),
            'arm':arm,'runner_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'scripts/stage_i_paired_pilot.py')}


def fresh_output(config,arm,mode):
    path=ROOT/config['run_root']/arm/mode
    require(path.resolve().is_relative_to((ROOT/config['run_root']).resolve()),'Unsafe output path')
    path.mkdir(parents=True,exist_ok=False);return path


def save_feas_checkpoint(model,optimizer,scheduler,scaler,folder,bind,step):
    folder.mkdir(exist_ok=False)
    model.save_pretrained(folder,safe_serialization=True)
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
        'python_rng':random.getstate(),'step':step,'binding':bind},folder/'state.pt')
    write_new(folder/'integrity.json',{'binding':bind,'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})


def restore_feas_checkpoint(model,optimizer,scheduler,scaler,folder,bind):
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    manifest=read(folder/'integrity.json');require(manifest['binding']==bind,'Cross-scope resume denied')
    for f,h in manifest['files'].items():require(sha(folder/f)==h,'Checkpoint changed')
    state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);require(state['binding']==bind,'State binding differs')
    weights=load_file(folder/'adapter_model.safetensors');require(set(weights)==set(get_peft_model_state_dict(model)),'Adapter geometry differs')
    set_peft_model_state_dict(model,weights)
    require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter roundtrip differs')
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state


def feasible(config_path,config,prep,arm,approval):
    from peft import get_peft_model_state_dict
    folder=fresh_output(config,arm,'feasibility');bind=binding(config_path,prep,arm);write_new(folder/'binding.json',bind)
    try:
        train,_,_,mc,cloud=datasets(config,prep);model,scope,init=load_gpu(config,prep,arm,approval,mc,cloud)
        write_new(folder/'scope_and_initialization.json',{'scope':scope,'initialization':init})
        controls=read(prep/'feasibility_cases.json')['business_ids'];features=[train[train.business_ids.index(b)] for b in controls]
        collator=LEDCompleteCollator(5000,768,1024,pad_source_to=5120,pad_label_to=768)
        optimizer,scheduler=optimizer_scheduler(model,config);scaler=torch.cuda.amp.GradScaler(enabled=True)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();records=[];memory=[]
        def update(step):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for j in range(8):
                f=features[((step-1)*8+j)%len(features)];batch=collator([f]);verify_mask_batch(batch,[f]);batch={k:v.cuda() for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.float16):loss=model(**batch,use_cache=False).loss
                require(torch.isfinite(loss).item(),'Nonfinite feasibility loss');losses.append(float(loss.detach()))
                scaler.scale(loss/8).backward()
            scaler.unscale_(optimizer);grads=gradient_record(model,expect_later_A=step>=3)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            used_lr=optimizer.param_groups[0]['lr'];scaler.step(optimizer);scaler.update();scheduler.step();torch.cuda.synchronize()
            require({int(s['step']) for s in optimizer.state.values()}=={step},'Skipped AMP/optimizer update')
            free,total=torch.cuda.mem_get_info();memory.append({'free_bytes':free,'total_bytes':total,'used_bytes':total-free})
            return {'step':step,'losses':losses,'used_lr':used_lr,'gradients':grads}
        for step in [1,2,3]:
            records.append(update(step))
            if step==2:save_feas_checkpoint(model,optimizer,scheduler,scaler,folder/'checkpoint_step2',bind,step)
        target={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        restore_feas_checkpoint(model,optimizer,scheduler,scaler,folder/'checkpoint_step2',bind)
        replay=update(3);after=get_peft_model_state_dict(model)
        exact=all(torch.equal(target[k],after[k].detach().cpu()) for k in target)
        close=all(torch.allclose(target[k],after[k].detach().cpu(),rtol=1e-5,atol=1e-6) for k in target)
        require(close,'Checkpoint/RNG replay not numerically reproducible')
        peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
        # Conservative headroom estimate includes peak Torch reservation plus observed non-Torch usage.
        non_torch=max(0,max(m['used_bytes'] for m in memory)-torch.cuda.memory_reserved())
        estimate=peak_reserved+non_torch;total=memory[-1]['total_bytes'];headroom=1-estimate/total
        status='TECHNICAL_FEASIBILITY_PASS' if headroom>=config['feasibility']['minimum_headroom_fraction'] else 'MEMORY_HEADROOM_BLOCKER'
        write_new(folder/'report.json',{'status':status,'arm':arm,'config_sha256':sha(config_path),'binding':bind,
            'scope':scope,'optimizer_updates':3,'additional_replay_updates':1,'microbatches_total':32,'gradient_records':records,
            'replay_step3':replay,'replay_exact':exact,'replay_allclose':close,'replay_tolerance':{'rtol':1e-5,'atol':1e-6},
            'roundtrip_scope':'Same-process model/optimizer/scheduler/scaler/RNG restore and actual replay; fresh-process recovery not claimed',
            'peak_allocated_bytes':peak_alloc,'peak_reserved_bytes':peak_reserved,'memory_samples':memory,
            'conservative_reserved_plus_non_torch_estimate':estimate,'estimated_headroom_fraction':headroom,
            'seconds':time.perf_counter()-started,'generation_calls':0,'weights_discarded_not_used_for_pilot':True})
        require(status=='TECHNICAL_FEASIBILITY_PASS','Insufficient measured headroom; stop without fallback')
    except Exception as exc:
        write_new(folder/'failure.json',{'error':str(exc),'type':type(exc).__name__,'automatic_fallback':False});raise


def train_pilot(config_path,config,prep,arm,approval,resume):
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback,TrainerCallback,set_seed
    require(config['optimization']['physical_batch_size']==1 and config['optimization']['gradient_accumulation_steps']==8,'Batch changed')
    out=ROOT/config['run_root']/arm/'train';bind=binding(config_path,prep,arm)
    if resume:
        require(out.is_dir() and read(out/'binding.json')==bind and not (out/'completion.json').exists(),'Incompatible/already complete resume')
        require(not (out/'best_adapter').exists(),'Best adapter already saved; review interrupted completion without overwriting it')
        checkpoint=Path(resume).resolve();require(checkpoint.is_relative_to(out.resolve()) and (checkpoint/'checkpoint_integrity.json').is_file(),'Unsafe/unbound checkpoint')
        proof=read(checkpoint/'checkpoint_integrity.json');require(proof['binding']==bind,'Cross-scope checkpoint')
        for name,h in proof['files'].items():require(sha(checkpoint/name)==h,'Checkpoint hash mismatch')
    else:
        out=fresh_output(config,arm,'train');write_new(out/'binding.json',bind)
    try:
        train,val,_,mc,cloud=datasets(config,prep);model,scope,init=load_gpu(config,prep,arm,approval,mc,cloud)
        if not (out/'initialization.json').exists():write_new(out/'initialization.json',{'scope':scope,'hashes':init})
        plans=read(prep/'epoch_orders.json')['indices'];o=config['optimization'];optimizer,scheduler=optimizer_scheduler(model,config)
        args=Seq2SeqTrainingArguments(output_dir=str(out),num_train_epochs=2,per_device_train_batch_size=1,per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,learning_rate=o['learning_rate'],weight_decay=o['weight_decay'],optim='adamw_torch',
            adam_beta1=o['adam_beta1'],adam_beta2=o['adam_beta2'],adam_epsilon=o['adam_epsilon'],max_grad_norm=1.0,
            lr_scheduler_type='linear',warmup_ratio=.05,fp16=True,bf16=False,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
            load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
            seed=42,data_seed=42,full_determinism=False,remove_unused_columns=False,label_names=['labels'],
            predict_with_generate=False,label_smoothing_factor=0.0,report_to='none',logging_steps=8,dataloader_num_workers=0)
        class PilotTrainer(Seq2SeqTrainer):
            def _get_train_sampler(self,train_dataset=None):return PairedEpochSampler(plans)
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                outputs=model(**inputs,use_cache=False);require(torch.isfinite(outputs.loss).item(),'Nonfinite training loss')
                return (outputs.loss,outputs) if return_outputs else outputs.loss
        class Record(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
                gradients=gradient_record(model,expect_later_A=state.global_step>=2)
                with (out/'gradients.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'next_step':state.global_step+1,'gradients':gradients})+'\n')
            def on_save(self,args,state,control,**kwargs):
                p=out/f'checkpoint-{state.global_step}'
                write_new(p/'checkpoint_integrity.json',{'binding':bind,'files':{q.name:sha(q) for q in p.iterdir() if q.is_file()}})
        trainer=PilotTrainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=LEDCompleteCollator(5000,768,1024),
            optimizers=(optimizer,scheduler),callbacks=[EarlyStoppingCallback(early_stopping_patience=2),Record()])
        trainer.model_accepts_loss_kwargs=False;set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        result=trainer.train(resume_from_checkpoint=str(checkpoint) if resume else None)
        require(trainer.state.global_step==128 and trainer.state.epoch==2,'Unexpected pilot duration')
        trainer.save_model(str(out/'best_adapter'));trainer.save_state();inspect_attached(model,arm)
        write_new(out/'completion.json',{'status':'PILOT_TRAINING_COMPLETE_NO_GENERATION','binding':bind,'arm':arm,'steps':128,'epochs':2,
            'best_checkpoint':trainer.state.best_model_checkpoint,'best_eval_loss':trainer.state.best_metric,'metrics':result.metrics,
            'epoch_order_sha256':sha(prep/'epoch_orders.json'),'seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'generation_calls':0,'best_adapter_sha256':sha(out/'best_adapter/adapter_model.safetensors')})
    except Exception as exc:
        # Unique failure file; never silently rewrite prior attempts.
        import uuid
        write_new(out/f'failure_{uuid.uuid4().hex}.json',{'error':str(exc),'automatic_fallback':False});raise


def generate_pilot(config_path,config,prep,arm,approval):
    from transformers import LEDTokenizerFast
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_i_decoding_diagnostics import structure,termination
    def semantic_generation(value):return {k:v for k,v in value.items() if k not in ('transformers_version','_from_model_config','_commit_hash')}
    for other in ARMS:
        report=read(ROOT/config['run_root']/other/'train/completion.json')
        require(report['binding']==binding(config_path,prep,other) and report['steps']==128,'Both matching training runs must finish before new outputs')
    trainout=ROOT/config['run_root']/arm/'train';complete=read(trainout/'completion.json')
    require(complete['binding']==binding(config_path,prep,arm),'Training provenance differs')
    checkpoint=trainout/'best_adapter/adapter_model.safetensors';require(sha(checkpoint)==complete['best_adapter_sha256'],'Best checkpoint changed')
    out=fresh_output(config,arm,'generation')
    _,_,tm,mc,cloud=datasets(config,prep);model,_,_=load_gpu(config,prep,arm,approval,mc,cloud)
    weights=load_file(checkpoint);require(set(weights)==set(get_peft_model_state_dict(model)),'Wrong adapter geometry')
    set_peft_model_state_dict(model,weights)
    require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Loaded best adapter differs')
    effective=deepcopy(model.generation_config)
    for k,v in config['generation'].items():setattr(effective,k,v)
    effective.use_cache=True
    expected=read(prep/'expected_generation_config.json')
    require(semantic_generation(effective.to_dict())==semantic_generation(expected),'Unexpected effective generation defaults')
    write_new(out/'actual_generation_config.json',effective.to_dict())
    tokenizer=LEDTokenizerFast.from_pretrained(cloud/tm['relative_directory'],local_files_only=True)
    cases=[json.loads(l) for l in (prep/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(cases)==24,'Generation scope differs');model.eval();collator=LEDCompleteCollator(5000,768,1024);predictions=[]
    for c in cases:
        batch={k:v.cuda() for k,v in collator([{'input_ids':c['input_ids'],'labels':c['label_ids']}]).items() if k!='labels'}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,**config['generation'],use_cache=True)[0].cpu().tolist()
        require(0<len(ids)<=768,'Generation limit mismatch')
        text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        predictions.append({'business_id':c['business_id'],'condition':'filtered','subset':c['subset'],'arm':arm,'token_ids':ids,'text':text,
            'source_sha256':c['source_sha256'],**structure(text,'filtered',len(c['selected_photo_ids'])),**termination(ids)})
        write_new(out/'cases'/(hashlib_name(c['business_id'])+'.json'),predictions[-1])
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter changed during generation')
    write_new(out/'completion.json',{'status':'ALL24_PILOT_OUTPUTS_COMPLETE','binding':binding(config_path,prep,arm),'predictions_sha256':sha(out/'predictions.jsonl'),'weights_unchanged':True})


def hashlib_name(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--mode',choices=['offline','feasibility','train','generate'],default='offline');parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--approval',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--resume')
    args=parser.parse_args()
    if args.mode=='offline':
        config,prep,_=checked_preparation(args.config);_,_,_,mc,_=datasets(config,prep)
        print(json.dumps({'status':'OFFLINE_INPUT_BINDINGS_PASS','scope':[meta_probe(config,mc,a) for a in ARMS],'GPU_actions':0}));return
    config,prep,_,approval=execution_gate(args.config,args.mode,args.arm,args.approval,args.execute)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.mode=='feasibility':feasible(args.config,config,prep,args.arm,approval)
    elif args.mode=='train':train_pilot(args.config,config,prep,args.arm,approval,args.resume)
    else:generate_pilot(args.config,config,prep,args.arm,approval)


if __name__=='__main__':main()
