"""Prepared D056 GPU runner. Requires NEW hash-bound researcher authorization.

Importing this file or requesting help cannot load torch/models or initialize GPU.
This script has NOT been executed on a model as part of offline preparation.
"""
import argparse,contextlib,copy,gc,hashlib,json,math,os,shutil,time,traceback
from pathlib import Path
from stage_i_d056_v2 import *

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--approval',type=Path)
    args=p.parse_args(argv)
    require(args.execute and args.approval,'STOP: separate explicit GPU approval and --execute required')
    manifest_path=OUT/'PREFLIGHT_MANIFEST.json';approval_gate(args.approval,CONFIG,manifest_path)
    cfg=read(CONFIG);check_config(cfg);verify_manifest(manifest_path)
    from validate_stage_i_d056_v2 import validate_prepared_runtime
    offline=validate_prepared_runtime(cfg,read(OUT/'cases.json'))
    require(offline['masks']==read(OUT/'RUNTIME_SOURCE_MASK_ADAPTER_BINDINGS.json')['masks'],'Offline runtime binding drift')
    destination=ROOT/cfg['future_output'];require(not destination.exists(),'Fresh diagnostic namespace required; no automatic resume/retry')
    require(shutil.disk_usage(ROOT).free>=cfg['limits']['minimum_free_disk_bytes'],'Insufficient output disk capacity')
    # Everything above is pure offline validation; heavy imports are behind approval.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import LEDForConditionalGeneration,GenerationConfig,set_seed,StoppingCriteria,StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from stage_hi_led_data import LEDCompleteCollator
    from stage_i_paired_pilot import lora_config,inspect_attached,verify_mask_batch
    require(torch.cuda.is_available(),'Local GPU unavailable; no automatic fallback')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Preserve the actual D054 cudnn TF32 state if different; config drift must stop.
    d052=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/A_decoder_only/runtime.json')
    require(d052['matmul_allow_tf32'] is False,'Unexpected previous matmul TF32 policy')
    torch.backends.cudnn.allow_tf32=d052['cudnn_allow_tf32']
    destination.mkdir(parents=True);budget=Budget(cfg['ceilings']);runtime_calls=[];phase={};started=time.perf_counter()
    inputs=read(OUT/'cases.json');align={x['business_id']:x for x in read(OUT/'TOKEN_POSITION_ALIGNMENT.json')}
    oldcfg=read(ROOT/'configs/stage_i_paired_lora_pilot_v1.json');collator=LEDCompleteCollator(5000,768,1024)
    baseline_hashes=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')['architectures'][0]['canonical_initial_backbone_hashes']
    semantic=lambda d:{k:v for k,v in d.items() if k not in ['transformers_version','_from_model_config','_commit_hash']}
    def memory():
        free,total=torch.cuda.mem_get_info()
        return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),
            'peak_reserved':torch.cuda.max_memory_reserved(),'device_free':free,'device_total':total}
    def guard_memory():
        m=memory();require(m['device_free']/m['device_total']>=cfg['limits']['minimum_GPU_headroom_fraction'],'GPU headroom below approved floor; stop, no fallback')
    def save_npz(path,**arrays):
        path.parent.mkdir(parents=True,exist_ok=True);require(not path.exists(),'Raw array file already exists')
        np.savez_compressed(path,**arrays)
    def compute_context(precision):return torch.autocast('cuda',dtype=torch.float16) if precision=='FP16_autocast' else contextlib.nullcontext()
    def call(kind,core,precision,**kwargs):
        budget.take(kind);require(not core.training,'Training mode during diagnostic');guard_memory()
        require('labels' not in kwargs and kwargs.get('decoder_attention_mask') is None,'Unexpected labels/decoder mask')
        if kind=='cache_control':
            pfx=phase['cache_prefix_length'];dec=kwargs['decoder_input_ids'];part=phase['cache_part']
            expected=z[:,:pfx] if part=='uncached' else z[:,:pfx-1] if part=='setup' else z[:,pfx-1:pfx]
            require(torch.equal(dec,expected),'Cache probe decoder-position drift')
            require(kwargs['use_cache']==(part!='uncached'),'Cache-mode mismatch')
            require((kwargs.get('past_key_values') is not None)==(part=='incremental'),'Past binding mismatch')
        with torch.no_grad(),compute_context(precision):result=core(**kwargs)
        require(torch.isfinite(result.logits).all().item(),'NaN/Inf logits')
        return result
    def hook(module,args,kw):
        dec=kw.get('decoder_input_ids');past=kw.get('past_key_values')
        expected=next(x for x in offline['masks']['cases'] if x['business_id']==phase['business_id'])
        for key in ['attention_mask','global_attention_mask']:
            values=kw.get(key);require(values is not None,'Missing source/global mask')
            rows=values.detach().cpu().numpy()
            require(rows.ndim==2 and all(array_binding(row[None,:])==expected['arrays'][key] for row in rows),'Runtime source mask binding mismatch')
        if kw.get('input_ids') is not None:
            values=kw['input_ids'].detach().cpu().numpy()
            require(all(array_binding(row[None,:])==expected['arrays']['input_ids'] for row in values),'Runtime source ID binding mismatch')
        runtime_calls.append({**phase,'decoder_shape':list(dec.shape) if dec is not None else None,
            'decoder_ids':dec.detach().cpu().tolist() if dec is not None else None,
            'past_self_length':int(past[0][0].shape[2]) if past is not None else 0,
            'use_cache':kw.get('use_cache'),'encoder_reused':kw.get('encoder_outputs') is not None,
            'effective_entry':'LEDForConditionalGeneration.forward','model_training':module.training,
            'source_attention_shape':list(kw['attention_mask'].shape) if kw.get('attention_mask') is not None else None,
            'global_attention_shape':list(kw['global_attention_mask'].shape) if kw.get('global_attention_mask') is not None else None,
            'decoder_attention_mask':None if kw.get('decoder_attention_mask') is None else 'PRESENT_UNEXPECTED',
            'frozen_source_mask_bindings_verified':True})
    def cache_shapes(past,expected_length,source_length):
        require(isinstance(past,(tuple,list)) and len(past)==6,'Unexpected LED legacy cache layout')
        shapes=[]
        for layer in past:
            require(len(layer)==4,'Expected self K/V and cross K/V')
            for j,x in enumerate(layer):
                require(list(x.shape)==[1,12,expected_length if j<2 else source_length,64],'Cache shape/prefix mismatch')
            shapes.append([{'shape':list(x.shape),'dtype':str(x.dtype)} for x in layer])
        return shapes
    class TraceCap(StoppingCriteria):
        def __init__(self,prompt):self.prompt=list(prompt);self.calls=[]
        def __call__(self,input_ids,scores,**unused):
            require(input_ids.shape[1]>=len(self.prompt),'Missing decoder prompt')
            require(all(row[:len(self.prompt)].tolist()==self.prompt for row in input_ids),'Decoder prompt prepended/changed unexpectedly')
            new=int(input_ids.shape[1]-len(self.prompt));require(new<=64,'Trace exceeds approved censoring cap')
            self.calls.append({'new_tokens':new,'candidate_rows':input_ids.shape[0],'censor':new>=64})
            return torch.full((input_ids.shape[0],),new>=64,device=input_ids.device,dtype=torch.bool)
    try:
        write(destination/'APPROVAL_AND_CONFIG.json',{'approval':read(args.approval),'config':cfg,'GPU':torch.cuda.get_device_name(),
            'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(manifest_path),'no_training':True,'D051_strict_replay':'FAIL_UNCHANGED'})
        for state in cfg['states']:
            set_seed(42);torch.cuda.reset_peak_memory_stats()
            base=LEDForConditionalGeneration.from_pretrained(cfg['local_backbone_snapshot'],torch_dtype=torch.float32,local_files_only=True)
            require({n:tensor_fingerprint(v) for n,v in base.named_parameters()}==baseline_hashes,'Pinned backbone fingerprints changed')
            if state!='pretrained_no_adapter':
                desc=cfg['adapter_states'][state];model=get_peft_model(base,LoraConfig(**lora_config(oldcfg,desc['scope'])))
                inspect_attached(model,desc['scope']);saved=load_file(ROOT/desc['path'],device='cpu')
                require(set(saved)==set(get_peft_model_state_dict(model)),'Adapter tensor keys differ')
                set_peft_model_state_dict(model,saved)
                require(all(torch.equal(v.detach().cpu(),saved[k]) for k,v in get_peft_model_state_dict(model).items()),'Loaded adapter differs')
                core=model.get_base_model();require(core is model.base_model.model,'Unexpected PEFT base forwarding path')
            else:model=core=base
            effective=copy.deepcopy(core.generation_config)
            for k,v in oldcfg['generation'].items():setattr(effective,k,v)
            effective.use_cache=True;require(semantic(effective.to_dict())==semantic(cfg['generation']),'Actual GenerationConfig mismatch')
            core.generation_config=effective;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});core.config.use_cache=False
            model.eval();before={n:tensor_fingerprint(v) for n,v in model.named_parameters()};handle=core.register_forward_pre_hook(hook,with_kwargs=True)
            write(destination/(state+'_RUNTIME_MODEL.json'),{'model_class':type(model).__name__,'core_class':type(core).__name__,
                'forward_path':'PEFT non-prompt generation delegates to the same underlying LED core used for explicit diagnostic forwards; adapters reside in core submodules',
                'actual_core_forward':core.forward.__qualname__,'model_config':core.config.to_dict(),'generation_config':effective.to_dict(),
                'AMP_wrapper_present':hasattr(model,'_original_forward'),'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_TF32':torch.backends.cudnn.allow_tf32,'no_optimizer_or_backward':True})
            model.to('cuda');guard_memory()
            for c in inputs:
                bid=c['business_id'];a=align[bid];stem=destination/state/bid;stem.mkdir(parents=True)
                feature={'input_ids':c['input_ids'],'labels':c['label_ids']};cpu=collator([feature]);verify_mask_batch(cpu,[feature])
                expected=next(x for x in offline['masks']['cases'] if x['business_id']==bid)
                require(all(array_binding(v.numpy())==expected['arrays'][k] for k,v in cpu.items()),'Actual Torch collator differs from offline mask bindings')
                batch={k:v.cuda() for k,v in cpu.items() if k!='labels'}
                require(batch['global_attention_mask'].sum().item()==1,'Global mask changed')
                z=torch.tensor([a['teacher_decoder_input_ids']],device='cuda',dtype=torch.long)
                labels=np.asarray(c['label_ids'],dtype=np.int64);references={}
                for precision in cfg['precision_modes']:
                    phase.clear();phase.update(state=state,business_id=bid,precision=precision,phase='teacher_forced')
                    full=call('teacher_forced',core,precision,**batch,decoder_input_ids=z,decoder_attention_mask=None,use_cache=False,return_dict=True)
                    raw=full.logits[0].float().cpu().numpy();require(raw.shape==(len(labels),50265),'Teacher logits shape mismatch')
                    critical=a['critical_logit_rows'];selected=raw[critical].copy();perlog=np.asarray([log_probs(x)[y] for x,y in zip(raw,labels)])
                    encoded=BaseModelOutput(last_hidden_state=full.encoder_last_hidden_state);encoder_tensor=encoded.last_hidden_state
                    encoder_hash=tensor_fingerprint(encoder_tensor)
                    save_npz(stem/(precision+'_teacher_raw.npz'),critical_rows=np.asarray(critical),logits=selected,labels=labels,
                        correct_logprobs=perlog,correct_logits=raw[np.arange(len(labels)),labels],
                        log_normalizers=raw[np.arange(len(labels)),labels].astype(np.float64)-perlog,
                        decoder_input_ids=np.asarray(a['teacher_decoder_input_ids']))
                    top=[]
                    for row in critical:
                        lp=log_probs(raw[row]);ids=np.argsort(-raw[row],kind='stable')[:10];correct=int(labels[row]);other=raw[row].copy();other[correct]=-np.inf
                        top.append({'row':row,'prefix_length':row+1,'expected_id':correct,'expected_probability':float(np.exp(lp[correct])),
                            'top_ids':ids.tolist(),'top_logits':raw[row,ids].tolist(),'top_logprobs':lp[ids].tolist(),
                            'margin_to_best_competitor':float(raw[row,correct]-other.max())})
                    write(stem/(precision+'_teacher.json'),{'mean_NLL':float(-perlog.mean()),'per_token_NLL':(-perlog).tolist(),'critical_positions':top,
                        'encoder_hash':encoder_hash,'encoder_shape':list(encoder_tensor.shape),'encoder_dtype':str(encoder_tensor.dtype),
                        'raw_logits_dtype':str(full.logits.dtype),'model_training':core.training,'precision':precision,'explicit_decoder_ids':True})
                    references[precision]={int(row):selected[i].copy() for i,row in enumerate(critical)}
                    del full,raw
                    for pfx in cfg['cache_prefix_lengths']:
                        phase.update(phase='cache_control',cache_prefix_length=pfx,cache_part='uncached');kwargs=dict(encoder_outputs=encoded,attention_mask=batch['attention_mask'],global_attention_mask=batch['global_attention_mask'],decoder_attention_mask=None,return_dict=True)
                        # 1: exact p-token prefix, NO supplied past. Primary uncached reference.
                        unc=call('cache_control',core,precision,**kwargs,decoder_input_ids=z[:,:pfx],use_cache=False)
                        refvec=unc.logits[0,-1].float().cpu().numpy();del unc
                        # 2: build cache from p-1 tokens, same actual encoder tensor/masks.
                        phase['cache_part']='setup'
                        setup=call('cache_control',core,precision,**kwargs,decoder_input_ids=z[:,:pfx-1],use_cache=True)
                        past=setup.past_key_values;past_shape=cache_shapes(past,pfx-1,batch['input_ids'].shape[1]);del setup
                        # 3: feed only final prefix token. Both paths predict labels[p-1].
                        phase['cache_part']='incremental'
                        inc=call('cache_control',core,precision,**kwargs,decoder_input_ids=z[:,pfx-1:pfx],past_key_values=past,use_cache=True)
                        actual=inc.logits[0,-1].float().cpu().numpy();new_shapes=cache_shapes(inc.past_key_values,pfx,batch['input_ids'].shape[1])
                        require(tensor_fingerprint(encoder_tensor)==encoder_hash,'Encoder representation changed between paths')
                        screen=cfg['cache_screens']['FP32' if precision=='FP32' else 'FP16'];cmp=vector_comparison(refvec,actual,int(labels[pfx-1]),**screen)
                        save_npz(stem/(precision+f'_cache_p{pfx}_raw.npz'),uncached_logits=refvec,cached_logits=actual,correct_id=np.asarray(labels[pfx-1]),prefix=z[0,:pfx].cpu().numpy())
                        write(stem/(precision+f'_cache_p{pfx}.json'),{**cmp,'prefix_length':pfx,'predicted_label_index':pfx-1,
                            'reference':'explicit p-token uncached forward; full-target TF logits are secondary only',
                            'past_before':past_shape,'past_after':new_shapes,'memory':memory()})
                        del past,inc,refvec,actual;guard_memory()
                    del encoded,encoder_tensor;torch.cuda.empty_cache()
                cross=[]
                for row in a['critical_logit_rows']:
                    cmp=vector_comparison(references['FP32'][row],references['FP16_autocast'][row],int(labels[row]),0.0001,0.0001)
                    # Cross-precision is NOT judged by cache allclose criteria.
                    cmp.pop('allclose_screen_pass');cmp['cross_precision_investigation_flag']=cmp['max_abs']>0.1 or cmp['argmax_changed']
                    cross.append({'row':row,**cmp})
                write(stem/'CROSS_PRECISION.json',{'rows':cross,'interpretation':'Screens are flags, not proof of instability; inspect correct-token margin, probability and ties.'})
                del references
                for mode in cfg['trace_modes']:
                    budget.take('traces');phase.clear();phase.update(state=state,business_id=bid,phase='trace',precision='FP16_autocast',mode=mode)
                    prompt=a['rescue_prefix_ids'] if mode=='greedy_correct_header_prefix' else [2]
                    stop=TraceCap(prompt);g=copy.deepcopy(effective);g.num_beams=4 if mode=='beam4_unprefixed' else 1
                    for k,v in cfg['trace_instrumentation'].items():setattr(g,k,v)
                    set_seed(42);guard_memory()
                    with torch.no_grad(),compute_context('FP16_autocast'):
                        generated=model.generate(**batch,decoder_input_ids=torch.tensor([prompt],device='cuda'),generation_config=g,
                            stopping_criteria=StoppingCriteriaList([stop]))
                    require(len(generated.scores)<=64,'Trace score steps exceed ceiling')
                    ids=generated.sequences[0].cpu().tolist();require(ids[:len(prompt)]==prompt,'Generation decoder-prefix drift')
                    observation=trace_observation(prompt,ids,any(x['censor'] for x in stop.calls))
                    ancestry=getattr(generated,'beam_indices',None)
                    if mode=='beam4_unprefixed':require(ancestry is not None and hasattr(generated,'sequences_scores'),'Expected pinned beam ancestry/scores missing; stop')
                    trans=core.compute_transition_scores(generated.sequences,generated.scores,ancestry,normalize_logits=False).cpu().tolist()
                    normalized_trans=core.compute_transition_scores(generated.sequences,generated.scores,ancestry,normalize_logits=True).cpu().tolist()
                    steps=[]
                    require(len(generated.logits)==len(generated.scores),'Raw/processed step count differs')
                    for j,(logits,scores) in enumerate(zip(generated.logits,generated.scores)):
                        candidates=separate_candidates(logits.float().cpu().numpy(),scores.float().cpu().numpy())
                        steps.append({'step':j,**candidates})
                    score_reconstruction=None
                    if ancestry is not None:
                        valid=int((ancestry[0]>=0).sum().item())
                        rows=ancestry[0,:valid].cpu().tolist();new=ids[len(prompt):]
                        require(len(new)==valid,'Beam ancestry/selected output mismatch')
                        direct=[float(generated.scores[j][row,new[j]].item()) for j,row in enumerate(rows)]
                        raw_chosen=[float(generated.logits[j][row,new[j]].item()) for j,row in enumerate(rows)]
                        score_reconstruction=beam_trace_audit(prompt,ids,ancestry[0].cpu().tolist(),trans[0],direct,raw_chosen,
                            float(generated.sequences_scores[0].item()),g.length_penalty,len(generated.scores),
                            any(x['censor'] for x in stop.calls),g.num_beams)
                    write(stem/(mode+'.json'),{'raw_ids':ids,'prompt_ids':prompt,**observation,'stop_checks':stop.calls,
                        'beam_indices':ancestry.cpu().tolist() if ancestry is not None else None,
                        'sequences_scores':generated.sequences_scores.cpu().tolist() if hasattr(generated,'sequences_scores') else None,
                        'selected_transition_scores':trans,'selected_normalized_transition_logprobs':normalized_trans,
                        'transition_score_semantics':'normalize_logits=False preserves processed scores (greedy logits versus beam logprobs); normalized logprobs are stored separately',
                        'sequence_score_reconstruction':score_reconstruction,'per_step_per_beam_top10':steps,'generation_config':g.to_dict(),
                        'warning':'Censored diagnostic, not a complete summary or guaranteed prefix/winner of full768 decoding. Top10 summaries do not permit full beam-search replay.'})
                    del generated;torch.cuda.empty_cache()
                del batch,z
            handle.remove();model.cpu();torch.cuda.empty_cache()
            require({n:tensor_fingerprint(v) for n,v in model.named_parameters()}==before,'Model changed during read-only run')
            write(destination/(state+'_COMPLETION.json'),{'weights_unchanged':True,'memory':memory(),'scope':'three validation cases only'})
            del core,model,base;gc.collect()
        verify_manifest(manifest_path)
        require(budget.used==cfg['ceilings'],'Unexpected missing/extra diagnostic job')
        write(destination/'COMPLETION.json',{'status':'BOUNDED_DIAGNOSTIC_COMPLETE_STOP_FOR_REVIEW','counts':budget.used,'seconds':time.perf_counter()-started,
            'training_updates':0,'Teacher_calls':0,'held_out_access':False,'runtime_forward_calls':runtime_calls,'frozen_files_unchanged':True})
    except Exception as e:
        write(destination/'FAILURE_STOP_NO_FALLBACK.json',{'error':str(e),'traceback':traceback.format_exc(),'counts':budget.used,'runtime_forward_calls':runtime_calls,
            'seconds':time.perf_counter()-started,'memory':memory() if torch.cuda.is_initialized() else None})
        raise
if __name__=='__main__':main()
