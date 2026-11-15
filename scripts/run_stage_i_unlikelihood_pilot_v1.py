"""Prepared paired pilot. Offline default; GPU modes require NEW hash-bound approval."""
import argparse
import copy
import importlib.metadata
import json
import os
import random
import time
import traceback
from pathlib import Path
from stage_i_unlikelihood_v1 import *


def preflight():
    cfg = read(CONFIG)
    m = read(PREP/'PREFLIGHT_MANIFEST.json')
    for rel, digest in m['files'].items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==digest, 'Frozen binding differs: '+rel)
    require({k:importlib.metadata.version(k) for k in cfg['runtime_pins']} == cfg['runtime_pins'], 'Runtime drift')
    require(cfg['arms']=={'control_ce':0.0,'treatment_ul':0.1} and cfg['training_steps_per_arm']==128,
            'Unreviewed design')
    return cfg, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight','feasibility','train','evaluate'])
    p.add_argument('--arm', choices=['control_ce','treatment_ul'])
    p.add_argument('--approval', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    cfg, manifest = preflight()
    if a.mode=='preflight':
        print(json.dumps({'status':'OFFLINE_INTEGRITY_PASS_NO_MODEL','file_bindings':len(manifest['files']),
                          'config_sha256':sha(CONFIG),'preflight_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
                          'runner_sha256':sha(RUNNER)})); return
    require(a.execute and a.approval and a.arm, 'STOP: new explicit GPU approval, arm and --execute required')
    approval = authorization_gate(a.approval, cfg, manifest, a.mode, a.arm)
    out = ROOT/cfg['run_root']/a.arm/a.mode
    require(not out.exists(), 'No overwrite, automatic retry or resume')
    root = ROOT/cfg['run_root']
    if a.mode=='train':
        for arm in cfg['arms']:
            report = read(root/arm/'feasibility/COMPLETION.json')
            require(report['status']=='FEASIBILITY_PASS' and report['config_sha256']==sha(CONFIG)
                    and report['runner_sha256']==sha(RUNNER), 'Both hash-bound gates must pass before training')
    if a.mode=='evaluate':
        require(read(root/a.arm/'train/COMPLETION.json')['status']=='TRAINING_COMPLETE', 'Training not complete')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import LEDForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed
    from transformers import get_linear_schedule_with_warmup
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from unittest.mock import patch
    from stage_hi_led_data import PackedLEDDataset, LEDCompleteCollator
    from stage_i_paired_pilot import BusinessSubset, PairedEpochSampler, inspect_attached, gradient_record, verify_mask_batch
    from run_stage_i_d050_feasibility_v1 import parameters, fingerprint
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'One CUDA GPU required; no fallback')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = None
    def write_log(name, value):
        with (out/name).open('a',encoding='utf-8') as f:
            f.write(json.dumps(value, allow_nan=False)+'\n')
    def deadline():
        authorization_gate(a.approval,cfg,manifest,a.mode,a.arm)
    def memory():
        free,total = torch.cuda.mem_get_info()
        reserve,alloc = torch.cuda.memory_reserved(),torch.cuda.memory_allocated()
        headroom = (free+reserve-alloc-torch.cuda.max_memory_reserved())/total
        return {'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
                'conservative_headroom_fraction':headroom,'total_gpu_memory':total}
    try:
        write(out/'AUTHORIZATION_CONFIG.json', {'approval':approval,'config':cfg,'arm':a.arm,'GPU':torch.cuda.get_device_name()})
        set_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        base = LEDForConditionalGeneration.from_pretrained(cfg['model_id'], revision=cfg['revision'],
                    torch_dtype=torch.float32, local_files_only=True)
        with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
            model = PeftModel.from_pretrained(base,ROOT/cfg['initial_adapter'],is_trainable=True,torch_device='cpu')
        scope = inspect_attached(model,'A_decoder_only')
        original = load_file(ROOT/cfg['initial_adapter']/'adapter_model.safetensors',device='cpu')
        actual = get_peft_model_state_dict(model)
        require(set(actual)==set(original) and all(torch.equal(actual[k],v) for k,v in original.items()), 'Initial F adapter differs')
        initial = parameters(model)
        history = read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')['architectures'][0]['canonical_initial_backbone_hashes']
        canonical = {n.removeprefix('base_model.model.').replace('.base_layer.','.'):v['sha256'] for n,v in initial.items() if not v['requires_grad']}
        require(canonical==history,'Frozen backbone differs from historical tensor fingerprints')
        write(out/'INITIAL_BINDINGS.json',{'adapter_sha256':cfg['initial_adapter_sha256'],'scope':scope,
             'frozen_backbone_canonical_match':True,'fresh_optimizer_scheduler_scaler_rng':True})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
        model.to('cuda'); torch.cuda.reset_peak_memory_stats(); set_seed(cfg['seed'])
        hm = read(ROOT/cfg['stage_h_manifest'])
        full = PackedLEDDataset(hm['datasets']['train_filtered'],5000,768)
        val = PackedLEDDataset(hm['datasets']['validation_filtered'],5000,768)
        sampling = read(PREP/'TRAINING_MANIFEST.json')
        subset = BusinessSubset(full,sampling['business_ids'])
        require(len(subset)==512 and len(val)==200 and not set(subset.business_ids)&set(val.business_ids), 'Scope leakage')
        views = [json.loads(s) for s in (PREP/'NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
        require([v['business_id'] for v in views]==subset.business_ids, 'Negative-positive alignment differs')
        class TrainingViews:
            def __len__(self): return len(subset)
            def __getitem__(self,i): return {**subset[i],'view_index':i}
        def collate(features):
            require(len(features)==1,'Physical batch size must remain1')
            batch = LEDCompleteCollator(5000,768,1024)(features)
            verify_mask_batch(batch,features)
            i = features[0].get('view_index')
            view = views[i] if i is not None else {'eligible':False}
            ids = view['negative_ids'] if view['eligible'] else [0,2]
            mask = view['penalty_mask'] if view['eligible'] else [False,False]
            batch['negative_ids']=torch.tensor([ids],dtype=torch.long)
            batch['negative_penalty_mask']=torch.tensor([mask],dtype=torch.bool)
            return batch
        if a.mode in ('feasibility','train'):
            o=cfg['optimization']; maximum=3 if a.mode=='feasibility' else 128
            optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'weight_decay':.01},
                                        {'params':[],'weight_decay':0.}],lr=.0002,betas=(.9,.999),eps=1e-8)
            scheduler=get_linear_schedule_with_warmup(optimizer,7,128)
            from stage_i_d058 import training_arguments
            args_dict=training_arguments(cfg,out)
            args_dict.update(num_train_epochs=2,max_steps=maximum,warmup_ratio=0.0,warmup_steps=7,save_total_limit=2,
                save_strategy='no' if a.mode=='feasibility' else 'epoch',eval_strategy='no' if a.mode=='feasibility' else 'epoch',
                prediction_loss_only=True)
            class Trainer(Seq2SeqTrainer):
                def _get_train_sampler(self,train_dataset=None): return PairedEpochSampler(sampling['epochs'])
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    deadline()
                    loss, outputs=paired_objective(model,inputs,cfg['arms'][a.arm],model.training,
                                      lambda r:write_log('objective.jsonl',{'update_before':self.state.global_step,**r}))
                    return (loss,outputs) if return_outputs else loss
            class Monitor(TrainerCallback):
                def on_train_begin(self,args,state,control,**kw):
                    require(state.max_steps==maximum and not optimizer.state,'Schedule/optimizer initialization differs')
                def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
                    write_log('gradients.jsonl',{'step':state.global_step+1,'gradients':gradient_record(model,expect_later_A=True)})
                def on_optimizer_step(self,args,state,control,**kw):
                    require(not trainer.accelerator.optimizer_step_was_skipped,'AMP update skipped')
                    m=memory();write_log('memory.jsonl',{'step':state.global_step+1,**m})
                    require(m['conservative_headroom_fraction']>=.15,'Headroom below15%; no fallback')
                    require({int(v['step']) for v in optimizer.state.values()}=={state.global_step+1},'Optimizer step differs')
                def on_step_end(self,args,state,control,**kw):
                    deadline(); require(state.global_step<=maximum,'Update ceiling exceeded')
                def on_save(self,args,state,control,**kw):
                    checkpoint=out/f'checkpoint-{state.global_step}'
                    needed=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json']
                    require(all((checkpoint/n).exists() for n in needed),'Incomplete checkpoint')
                    write(checkpoint/'integrity.json',{'config_sha256':sha(CONFIG),'files':{n:sha(checkpoint/n) for n in needed}})
            trainer=Trainer(model=model,args=Seq2SeqTrainingArguments(**args_dict),train_dataset=TrainingViews(),
                eval_dataset=val,data_collator=collate,optimizers=(optimizer,scheduler),callbacks=[Monitor()])
            trainer.model_accepts_loss_kwargs=False
            write(out/'ACTUAL_TRAINING_ARGUMENTS.json',trainer.args.to_dict())
            result=trainer.train(resume_from_checkpoint=None)
            require(trainer.state.global_step==maximum,'Unexpected update count')
            if a.mode=='feasibility':
                trainer._save_checkpoint(model,trial=None)
                checkpoint=out/'checkpoint-3'
                weights=load_file(checkpoint/'adapter_model.safetensors',device='cpu')
                before={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
                require(all(torch.equal(before[k],v) for k,v in weights.items()),'Gate serialization mismatch')
                saved_optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cuda',weights_only=False)
                saved_scheduler=torch.load(checkpoint/'scheduler.pt',weights_only=False)
                saved_scaler=torch.load(checkpoint/'scaler.pt',weights_only=False)
                rng_before=torch.load(checkpoint/'rng_state.pth',weights_only=False)
                require(fingerprint(optimizer.state_dict())==fingerprint(saved_optimizer),'Optimizer save mismatch')
                require(fingerprint(scheduler.state_dict())==fingerprint(saved_scheduler),'Scheduler save mismatch')
                # Restore from the new temporary gate checkpoint, no replay updates.
                set_peft_model_state_dict(model,weights)
                optimizer.load_state_dict(saved_optimizer)
                scheduler.load_state_dict(saved_scheduler)
                scaler=trainer.accelerator.scaler
                require(scaler is not None,'Expected FP16 scaler absent')
                require(fingerprint(scaler.state_dict())==fingerprint(saved_scaler),'Scaler save mismatch')
                scaler.load_state_dict(saved_scaler)
                require(scheduler.last_epoch==3 and {int(v['step']) for v in optimizer.state.values()}=={3},'Gate restored steps differ')
                trainer._load_rng_state(checkpoint)
                require(fingerprint(optimizer.state_dict())==fingerprint(saved_optimizer),'Optimizer restore mismatch')
                require(fingerprint(scheduler.state_dict())==fingerprint(saved_scheduler),'Scheduler restore mismatch')
                require(fingerprint(scaler.state_dict())==fingerprint(saved_scaler),'Scaler restore mismatch')
                rng_after={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.random.get_rng_state(),
                           'cuda':torch.cuda.random.get_rng_state()}
                require(fingerprint(rng_before)==fingerprint(rng_after),'RNG restoration mismatch')
                require(all(torch.equal(before[k],v.detach().cpu()) for k,v in get_peft_model_state_dict(model).items()),'Gate restoration changed adapters')
                write(out/'RESTORATION.json',{'status':'PASS_RESTORATION_NO_REPLAY','restored':['adapter','optimizer','scheduler','scaler','RNG'],
                    'replay_updates':0,'strict_replay_claimed':False,'D051_strict_replay':'FAIL_UNCHANGED',
                    'fingerprints':{'optimizer':fingerprint(saved_optimizer),'scheduler':fingerprint(saved_scheduler),
                                    'scaler':fingerprint(saved_scaler),'RNG_before':fingerprint(rng_before),'RNG_after':fingerprint(rng_after)}})
            status='FEASIBILITY_PASS' if a.mode=='feasibility' else 'TRAINING_COMPLETE'
            details={'steps':maximum,'metrics':result.metrics}
        else:
            checkpoint=root/a.arm/'train/checkpoint-128'
            proof=read(checkpoint/'integrity.json')
            require(proof['config_sha256']==sha(CONFIG),'Foreign checkpoint')
            for name,digest in proof['files'].items(): require(sha(checkpoint/name)==digest,'Checkpoint changed')
            weights=load_file(checkpoint/'adapter_model.safetensors',device='cpu')
            set_peft_model_state_dict(model,weights)
            require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Evaluation adapter differs')
            expected=read(ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json')
            g=copy.deepcopy(model.generation_config)
            for k,v in cfg['generation'].items(): setattr(g,k,v)
            g.use_cache=True
            semantic=lambda d:{k:v for k,v in d.items() if k not in ('transformers_version','_from_model_config','_commit_hash')}
            require(semantic(g.to_dict())==semantic(expected),'Effective generation config differs')
            write(out/'EFFECTIVE_GENERATION_CONFIG.json',g.to_dict())
            from transformers import LEDTokenizerFast
            tm=read(ROOT/hm['tokenizer_manifest'])
            tok=LEDTokenizerFast.from_pretrained(ROOT/tm['relative_directory'],local_files_only=True)
            cases=[json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
            require([c['business_id'] for c in cases]==read(PREP/'EVALUATION_MANIFEST.json')['business_ids'],'Case list differs')
            from stage_i_d053_evaluation import repetition_diagnostics,structural_diagnostics,token_termination
            predictions=[];model.eval()
            for c in cases:
                deadline()
                batch=LEDCompleteCollator(5000,768,1024)([{'input_ids':c['input_ids'],'labels':c['label_ids']}])
                model_inputs={k:v.to('cuda') for k,v in batch.items() if k!='labels'}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    tokens=model.generate(**model_inputs,generation_config=g)[0].tolist()
                text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                row={'business_id':c['business_id'],'subset':c['subset'],'arm':a.arm,'source_sha256':c['source_sha256'],
                    'token_ids':tokens,'text':text,'human_verified':False,'development_only':True,
                    'automatic':{'repetition':repetition_diagnostics(text,tokens,g.to_dict()),
                        'structure':structural_diagnostics(text,len(c['selected_photo_ids'])),
                        'termination':token_termination(tokens,g.to_dict())}}
                write(out/'cases'/(c['business_id']+'.json'),row);predictions.append(row)
            require(len(predictions)==24,'Wrong generation scope')
            require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Inference changed weights')
            write(out/'predictions.json',predictions);status='EVALUATION_COMPLETE_STOP_FOR_RESEARCHER_REVIEW'
            details={'generation_calls':24}
        final=parameters(model)
        require(all(final[n]['sha256']==v['sha256'] for n,v in initial.items() if not v['requires_grad']), 'Frozen base changed')
        write(out/'FROZEN_BACKBONE_INTEGRITY.json',{'unchanged':True,'D051_strict_replay':'FAIL_UNCHANGED'})
        write(out/'COMPLETION.json',{'status':status,'config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER),
            'seconds':time.perf_counter()-started,**details,**memory()})
    except Exception as exc:
        write(out/'FAILURE_STOP_NO_RETRY.json',{'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    main()
