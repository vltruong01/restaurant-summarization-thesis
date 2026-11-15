"""Prepared single-run D060 train/evaluate adapter. NO execution without new approval.

Reuses accepted dataset, collation, initialization, sampler, LoRA and metric code.
Exact trained RO adapter initialization; fresh Stage-2 optimizer750/warmup38. No GPU execution without separate approval.
"""
import argparse,copy,json,os,time,traceback
from stage_i_d060 import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval',type=Path)
    a=parser.parse_args();require(a.execute and a.approval,'STOP: explicit D060 approval and --execute required')
    approval=gate(a.approval,a.mode);cfg,_=preflight(allow_existing_run=a.mode=='evaluate')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainerCallback,set_seed,LEDTokenizerFast
    from transformers import get_linear_schedule_with_warmup
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_hi_led_data import PackedLEDDataset,LEDCompleteCollator
    from stage_i_paired_pilot import PairedEpochSampler,gradient_record,inspect_attached,verify_mask_batch
    from run_stage_i_d050_feasibility_v1 import parameters
    require(torch.cuda.is_available(),'CUDA unavailable; stop, no fallback')
    manifest=read(ROOT/cfg['stage_h_manifest']);train=PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768)
    val=PackedLEDDataset(manifest['datasets']['validation_filtered'],5000,768)
    tm=read(ROOT/manifest['tokenizer_manifest']);mc=read(ROOT/'assets/models/led-base-16384/config.json')
    require(len(train)==2000 and len(val)==200 and not set(train.business_ids)&set(val.business_ids),'Split scope differs')
    out=ROOT/cfg['run_root'];modeout=out/a.mode;modeout.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();model=None
    def memory():return {'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
    try:
        write(modeout/'AUTHORIZATION_AND_CONFIG.json',{'approval':approval,'config':cfg,'config_sha256':sha(CONFIG),
            'runner_sha256':sha(Path(__file__)),'GPU':torch.cuda.get_device_name(),'D051_strict_replay':'FAIL_UNCHANGED'})
        model,scope,initialization=load_ro_cpu(cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
        torch.backends.cuda.matmul.allow_tf32=False;model.to('cuda');set_seed(42)
        initial=parameters(model);historical=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')['architectures'][0]['canonical_initial_backbone_hashes']
        canonical={n.removeprefix('base_model.model.').replace('.base_layer.','.'):v['sha256'] for n,v in initial.items() if not v['requires_grad']}
        require(canonical==historical,'Pretrained backbone tensors differ')
        write(modeout/'INITIAL_MODEL_BINDINGS.json',{'scope':scope,'initialization':initialization,'pretrained_hashes_match':True})
        collator=LEDCompleteCollator(5000,768,1024);torch.cuda.reset_peak_memory_stats()
        if a.mode=='train':
            orders=read(PREP/'EPOCH_ORDERS.json');require(orders['training_business_ids']==train.business_ids,'Order input differs')
            o=cfg['optimization'];params=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW([{'params':params,'weight_decay':.01},{'params':[],'weight_decay':0.}],lr=.0002,betas=(.9,.999),eps=1e-8)
            scheduler=get_linear_schedule_with_warmup(optimizer,38,750)
            class Trainer(Seq2SeqTrainer):
                def _get_train_sampler(self,train_dataset=None):return PairedEpochSampler(orders['indices'])
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    check_deadline(approval)
                    outputs=model(**inputs,use_cache=False);require(torch.isfinite(outputs.loss).item(),'Nonfinite loss')
                    return (outputs.loss,outputs) if return_outputs else outputs.loss
            class Record(TrainerCallback):
                def on_train_begin(self,args,state,control,**kw):
                    require(state.max_steps==750,'Unexpected schedule horizon');check_deadline(approval)
                    require(not optimizer.state,'Optimizer must be fresh');torch.cuda.reset_peak_memory_stats()
                def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
                    check_deadline(approval)
                    gradients=gradient_record(model,expect_later_A=True)
                    with (modeout/'gradient_checks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':state.global_step+1,'branches':gradients})+'\n')
                def on_optimizer_step(self,args,state,control,optimizer=None,**kw):
                    require(not trainer.accelerator.optimizer_step_was_skipped,'AMP update skipped; stop')
                    free,total=torch.cuda.mem_get_info();reserved=torch.cuda.memory_reserved();allocated=torch.cuda.memory_allocated()
                    conservative=(free+reserved-allocated-torch.cuda.max_memory_reserved())/total
                    if state.global_step==0:
                        write(modeout/'FIRST_UPDATE_HEADROOM.json',{'conservative_headroom_fraction':conservative,'minimum':0.15,**memory()})
                    require(conservative>=0.15,'Insufficient measured headroom; stop without fallback')
                    require({int(v['step']) for v in optimizer.state.values()}=={state.global_step+1},'Optimizer step differs')
                def on_step_end(self,args,state,control,lr_scheduler=None,**kw):
                    require(state.global_step<=750,'Update budget exceeded')
                    if state.global_step%8==0:
                        with (modeout/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'step':state.global_step,'epoch':state.epoch,'lr':lr_scheduler.get_last_lr(),**memory()})+'\n')
                def on_save(self,args,state,control,**kw):
                    require(state.global_step in [250,500,750],'Unexpected epoch checkpoint')
                    p=modeout/f'checkpoint-{state.global_step}'
                    needed=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json']
                    require(all((p/n).exists() for n in needed),'Incomplete checkpoint')
                    write(p/'checkpoint_integrity.json',{'config_sha256':sha(CONFIG),'files':{q.name:sha(q) for q in p.iterdir() if q.is_file()}})
            args=Seq2SeqTrainingArguments(**training_arguments(cfg,modeout))
            trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=collator,
                optimizers=(optimizer,scheduler),callbacks=[Record()])
            trainer.model_accepts_loss_kwargs=False;set_seed(42)
            write(modeout/'ACTUAL_TRAINING_ARGUMENTS.json',args.to_dict())
            result=trainer.train(resume_from_checkpoint=None)
            require(trainer.state.global_step==750 and trainer.state.epoch==3,'Run did not complete planned ceiling')
            write(modeout/'COMPLETION.json',{'status':'TRAINING_COMPLETE_STOP_BEFORE_EVALUATION_COMMAND','steps':750,'epochs':3,
                'seconds':time.perf_counter()-started,'metrics':result.metrics,'quality_checkpoint_steps':[250,750],**memory()})
        else:
            finished=read(out/'train/COMPLETION.json');require(finished['steps']==750,'Full bounded run not complete')
            require(not (out/'train/FAILURE_STOP_NO_RETRY.json').exists() and read(out/'train/FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Training integrity not accepted')
            from stage_i_d053_evaluation import repetition_diagnostics,structural_diagnostics,token_termination
            tok=LEDTokenizerFast.from_pretrained(ROOT/tm['relative_directory'],local_files_only=True)
            cases=[json.loads(s) for s in (BASE_PREP/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
            require([x['business_id'] for x in cases]==read(PREP/'EVALUATION_MANIFEST.json')['business_ids'],'Evaluation cases changed')
            expected=read(BASE_PREP/'expected_generation_config.json');semantic=lambda v:{k:x for k,x in v.items() if k not in ['transformers_version','_from_model_config','_commit_hash']}
            g=copy.deepcopy(model.generation_config)
            for k,v in cfg['generation'].items():setattr(g,k,v)
            g.use_cache=True;require(semantic(g.to_dict())==semantic(expected),'Effective GenerationConfig differs')
            write(modeout/'EFFECTIVE_GENERATION_CONFIG.json',g.to_dict());model.eval();count=0
            for epoch,step in [(1,250),(3,750)]:
                checkpoint=out/'train'/f'checkpoint-{step}';proof=read(checkpoint/'checkpoint_integrity.json')
                require(proof['config_sha256']==sha(CONFIG),'Foreign checkpoint')
                for n,h in proof['files'].items():require(sha(checkpoint/n)==h,'Checkpoint corruption')
                weights=load_file(checkpoint/'adapter_model.safetensors',device='cpu');require(set(weights)==set(get_peft_model_state_dict(model)),'Wrong adapter keys')
                set_peft_model_state_dict(model,weights)
                require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter load differs')
                predictions=[]
                for c in cases:
                    check_deadline(approval)
                    f={'input_ids':c['input_ids'],'labels':c['label_ids']};batch=collator([f]);verify_mask_batch(batch,[f])
                    batch={k:v.cuda() for k,v in batch.items() if k!='labels'};count+=1;require(count<=48,'Generation budget exceeded')
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,generation_config=g)[0].cpu().tolist()
                    require(0<len(ids)<=768,'Generation length differs')
                    text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    row={'business_id':c['business_id'],'subset':c['subset'],'epoch':epoch,'condition':'filtered','arm':f'RO_to_F_epoch{epoch}',
                        'source_sha256':c['source_sha256'],'token_ids':ids,'text':text,'reference_sha256':sha(ROOT/cfg['final_references']),
                        'automatic':{'structure':structural_diagnostics(text,len(c['selected_photo_ids'])),'termination':token_termination(ids,g.to_dict()),
                            'repetition':repetition_diagnostics(text,ids,g.to_dict())},'researcher_output_annotation_status':'NOT_REVIEWED'}
                    write(modeout/f'epoch{epoch}'/(c['business_id']+'.json'),row);predictions.append(row)
                write(modeout/f'epoch{epoch}'/'predictions.json',predictions)
                require(all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items()),'Evaluation altered adapter')
            require(count==48,'Missing fixed-scope predictions')
            write(modeout/'COMPLETION.json',{'status':'ALL48_PREDICTIONS_COMPLETE_STOP_FOR_QUALITY_REVIEW','generation_calls':count,
                'seconds':time.perf_counter()-started,**memory()})
        after=parameters(model)
        require(all(after[n]['sha256']==v['sha256'] for n,v in initial.items() if not v['requires_grad']),'Frozen backbone changed')
        write(modeout/'FROZEN_BACKBONE_INTEGRITY.json',{'unchanged':True,'D051_strict_replay':'FAIL_UNCHANGED'})
    except Exception as exc:
        write(modeout/'FAILURE_STOP_NO_RETRY.json',{'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started})
        raise

if __name__=='__main__':main()
