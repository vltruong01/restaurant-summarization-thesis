"""D055 offline forensics. CPU token/mask arithmetic only; no model forward or GPU.

Reproduction creates a NEW output directory, never rewrites accepted artifacts.
No pretrained model is instantiated. Pinned implementation source is inspected.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse, collections, difflib, hashlib, importlib.metadata, inspect, json, re, statistics
from pathlib import Path
import numpy as np
import torch
from transformers import LEDTokenizerFast, LEDTokenizer, Trainer
from transformers.models.led import modeling_led as led
from accelerate import Accelerator
from stage_hi_led_data import PackedLEDDataset, LEDCompleteCollator
from stage_i_d053_evaluation import repetition_diagnostics, structural_diagnostics, token_termination

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
PREP = ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
D054 = ROOT/'outputs/stage_i/d054_paired_generation_v1'
ARMS = ['historical_full_F', 'A_decoder_only', 'B_encoder_decoder']
REVIEW = '[REVIEW_CONSENSUS]'
VISUAL = '[VISUAL_EVIDENCE]'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f: f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def require(x,m):
    if not x: raise ValueError(m)
def stats(x):
    return dict(n=len(x),min=min(x),mean=statistics.mean(x),median=statistics.median(x),p95=float(np.percentile(x,95)),max=max(x)) if x else None
def compatible_config(d):
    return {k:v for k,v in d.items() if k not in ('_from_model_config','transformers_version','_commit_hash')}
def prefix_diagnostic(ids, labels, tokenizer):
    expected = [2]+labels
    n=next((i for i,(a,b) in enumerate(zip(ids,expected)) if a!=b),min(len(ids),len(expected)))
    prefix=[2]+tokenizer(REVIEW+'\n',add_special_tokens=True)['input_ids'][:-1]
    k=next((i for i,(a,b) in enumerate(zip(ids,prefix)) if a!=b),min(len(ids),len(prefix)))
    return dict(exact_teacher_common_prefix_tokens=n,first_teacher_token_mismatch=n,
        teacher_expected_id=expected[n] if n<len(expected) else None,
        actual_id=ids[n] if n<len(ids) else None,
        required_header_prefix_ids=prefix,required_header_prefix_complete=ids[:len(prefix)]==prefix,
        first_header_mismatch=k if k<len(prefix) else None,
        expected_header_token=tokenizer.convert_ids_to_tokens(prefix[k]) if k<len(prefix) else None,
        actual_header_token=tokenizer.convert_ids_to_tokens(ids[k]) if k<len(ids) and k<len(prefix) else None,
        mismatch_stage=('decoder_start' if k==0 else 'BOS' if k==1 else 'target_header_or_newline' if k<len(prefix) else 'after_complete_target_header'),
        warning='Body divergence from one teacher wording is not itself an error; header mismatch can be formatting only.')

def implementation(out):
    functions={
        'LED.shift_tokens_right':led.shift_tokens_right,
        'LED.forward':led.LEDForConditionalGeneration.forward,
        'LED.decoder_forward':led.LEDDecoder.forward,
        'LED.encoder_forward':led.LEDEncoder.forward,
        'LED.causal_mask':led._create_4d_causal_attention_mask,
        'LED.prepare_generation':led.LEDForConditionalGeneration.prepare_inputs_for_generation,
        'Trainer.training_step':Trainer.training_step,
        'Trainer.accelerator_setup':Trainer.create_accelerator_and_postprocess,
        'Trainer.optimizer_setup':Trainer.create_optimizer,
        'Accelerator.backward':Accelerator.backward,
        'Accelerator.init':Accelerator.__init__,
    }
    index={}; files={}
    for name,obj in functions.items():
        path=Path(inspect.getsourcefile(obj)); lines,first=inspect.getsourcelines(obj)
        key=path.name
        if key in files and files[key]!=path: key=digest(str(path))[:8]+'_'+key
        files[key]=path
        snippet=''.join(lines)
        (out/'implementation').mkdir(exist_ok=True)
        dest=out/'implementation'/(name.replace('.','_')+'.txt')
        dest.write_text(f'# {path}\n# starts at line {first}; SHA256 of source file {sha(path)}\n'+snippet,encoding='utf-8')
        index[name]=dict(installed_path=str(path),source_file_sha256=sha(path),start_line=first,end_line=first+len(lines)-1,excerpt=dest.relative_to(out).as_posix())
    for name,p in files.items(): (out/'implementation'/name).write_bytes(p.read_bytes())
    require('shift_tokens_right' in inspect.getsource(led.LEDForConditionalGeneration.forward),'Unexpected label implementation')
    require('CrossEntropyLoss()' in inspect.getsource(led.LEDForConditionalGeneration.forward),'Unexpected loss')
    require('self.gradient_checkpointing and self.training' in inspect.getsource(led.LEDDecoder.forward),'Cache condition changed')
    write(out/'IMPLEMENTATION_INDEX.json',index)
    return index

def mask_tests(examples):
    # Real source/target content, two uneven lengths; tensor operations, no forward.
    samples=[{'input_ids':c['input_ids'],'labels':c['label_ids']} for c in examples]
    features=[min(samples,key=lambda f:len(f['labels'])),max(samples,key=lambda f:len(f['labels']))]
    batch=LEDCompleteCollator()(features)
    shifted=led.shift_tokens_right(batch['labels'],1,2)
    for i,f in enumerate(features):
        n,m=len(f['input_ids']),len(f['labels'])
        require(batch['input_ids'][i,:n].tolist()==f['input_ids'],'Source content changed')
        require(batch['labels'][i,:m].tolist()==f['labels'],'Target content changed')
        require(torch.all(batch['labels'][i,m:]==-100).item(),'Padding must be -100')
        require(torch.all(batch['attention_mask'][i,:n]==1).item() and torch.all(batch['attention_mask'][i,n:]==0).item(),'Source mask')
        require(batch['global_attention_mask'][i].sum().item()==1 and batch['global_attention_mask'][i,0].item()==1,'Global mask')
        require(shifted[i,:m].tolist()==[2]+f['labels'][:-1],'Teacher shift misalignment')
    require(not torch.any(shifted==-100).item(),'Negative decoder IDs')
    # Four positions with/without two cached past tokens; future keys forbidden.
    causal=[]
    for past in [0,2]:
        mask=led._create_4d_causal_attention_mask((1,4),torch.float32,torch.device('cpu'),past_key_values_length=past)
        for q in range(4):
            for k in range(4+past):
                require((mask[0,0,q,k].item()==0)==(k<=q+past),'Causal leakage')
        causal.append({'past':past,'values':mask.tolist()})
    # Synthetic logits check CE default ignore_index exactly excludes -100.
    logits=torch.tensor([[1.,2.,3.],[3.,2.,1.],[.1,.2,.3]])
    labels=torch.tensor([2,0,-100])
    loss=torch.nn.CrossEntropyLoss()(logits,labels)
    require(torch.equal(loss,torch.nn.CrossEntropyLoss()(logits[:2],labels[:2])),'CE padding mismatch')
    return dict(status='PASS_CPU_TENSORS_ONLY',model_forward_calls=0,batch_shape=list(batch['input_ids'].shape),label_shape=list(batch['labels'].shape),
        shifted_first_16=shifted[0,:16].tolist(),original_first_16=batch['labels'][0,:16].tolist(),causal_masks=causal,
        source_and_label_preservation=True,padded_label_count=int((batch['labels']==-100).sum()),
        decoder_padding_note='No explicit decoder_attention_mask: right padding follows EOS, causal mask prevents it affecting earlier supervised positions. Loss ignores padded positions.',
        loss_alignment='Logit at decoder position t predicts labels[t]; decoder consumes [start]+labels[:-1]. No second loss shift.')

def targets(out,tok,slow,manifest):
    aggregate={}; warnings=[]; all_by={}; markers=collections.Counter(); source_markers=collections.Counter(); lengths=[]; checks=0; fastslow=0
    for key,info in manifest['datasets'].items():
        require(sha(ROOT/info['path'])==info['sha256'],'NPZ hash')
        require(sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256'],'JSONL hash')
        data=rows(ROOT/info['stage_g_jsonl']); pack=PackedLEDDataset(info,5000,768)
        require(len(data)==len(pack),'Row mismatch'); seen=set(); ts=[]; ss=[]; reps=[]; bad=[]; zero=[]
        # Batch fast tokenizer is exact with no truncation, padding or max_length.
        ti=tok([x['target'] for x in data],truncation=False,padding=False)['input_ids']
        si=tok([x['source'] for x in data],truncation=False,padding=False)['input_ids']
        for i,c in enumerate(data):
            bid=c['business_id'];require(bid not in seen,'Duplicate ID');seen.add(bid)
            require(pack.business_ids[i]==bid and pack[i]['labels']==ti[i] and pack[i]['input_ids']==si[i],'JSONL/NPZ/tokenizer alignment')
            require(ti[i][0]==0 and ti[i][-1]==2 and si[i][0]==0 and si[i][-1]==2,'Missing BOS/EOS')
            require(ti[i].count(0)==ti[i].count(2)==1 and si[i].count(0)==si[i].count(2)==1,'Internal special token')
            require(len(ti[i])<=768 and len(si[i])<=5000,'Silent truncation/overflow')
            if i==0 or i==len(data)-1 or len(ti[i])==max(map(len,ti)):
                require(slow(c['target'],truncation=False)['input_ids']==ti[i],'Slow tokenizer mismatch');fastslow+=1
            text=c['target']; condition=c['condition']; isvisual=condition!='reviews_only'
            if not text.startswith(REVIEW+'\n') or text.count(REVIEW)!=1 or text.count(VISUAL)!=(1 if isvisual else 0):bad.append(bid)
            if isvisual and not structural_diagnostics(text,len(c['selected_photo_ids']))['section_headers_compliant']:bad.append(bid)
            if isvisual and not c['selected_photo_ids']:zero.append({'business_id':bid,'empty_visual':text.split(VISUAL)[-1].strip()==''})
            rep=repetition_diagnostics(text)
            if rep['exact_repeated_sentence'] or rep['consecutive_repeated_word']:reps.append({'business_id':bid,'repetition':rep,'target':text})
            markers.update(re.findall(r'\[[A-Z_0-9]+\]',text));source_markers.update(re.findall(r'\[[A-Z_0-9]+\]',c['source']))
            ts.append(len(ti[i]));ss.append(len(si[i]));lengths.append(len(ti[i]));checks+=1
            all_by[(key.split('_')[0],bid,condition)]=c
        aggregate[key]=dict(rows=len(data),all_start_correct=sum(x['target'].startswith(REVIEW+'\n') for x in data),
            malformed_business_ids=sorted(set(bad)),target_tokens=stats(ts),source_tokens=stats(ss),
            repetition_flagged_targets=len(reps),zero_caption_targets=zero,ids=sorted(seen))
        warnings.extend({'dataset':key,**x} for x in reps)
        print('target audit',key,len(data),flush=True)
    shared=0
    for (split,bid,condition),c in all_by.items():
        if condition!='reviews_only':continue
        review=c['target']; rosource=c['source']
        for other in ['unfiltered','filtered']:
            oc=all_by[(split,bid,other)]
            require(oc['target'].split('\n\n'+VISUAL)[0]==review,'Shared consensus changed')
            require(oc['selected_review_ids']==c['selected_review_ids'],'Review IDs changed')
            require(oc['source'].split('\n\n[VISUAL_CONTEXT]')[0]==rosource,'Review/source prefix differs')
            shared+=1
    train=set(aggregate['train_filtered']['ids']);val=set(aggregate['validation_filtered']['ids']);require(not train&val,'Train/validation leakage')
    for split in ['train','validation']:
        require(all(set(aggregate[split+'_'+c]['ids'])==set(aggregate[split+'_filtered']['ids']) for c in ['reviews_only','unfiltered']),'Condition split mismatch')
    report=dict(status='PASS_LOSSLESS_ACCEPTED_DATA',datasets=aggregate,targets_checked=checks,tokenization_checks=checks*2,
        slow_fast_checks=fastslow,shared_consensus_and_source_checks=shared,train_validation_disjoint=True,
        starts_with_review_header=sum(x['all_start_correct'] for x in aggregate.values()),
        target_token_distribution=stats(lengths),over512=sum(x>512 for x in lengths),over640=sum(x>640 for x in lengths),
        marker_counts=dict(markers),source_marker_counts=dict(source_markers),repetition_flags=len(warnings),
        warning='Surface repetition flags are diagnostics, not factual/quality failures. Uncertain factual truth requires source/human review.')
    write(out/'TARGET_AUDIT.json',report);write(out/'TARGET_REPETITION_FLAGS.json',warnings)
    return report

def training(out,config):
    results={};historical=CLOUD/'outputs/stage_i/runs_v3/filtered'; hc=read(historical/'runtime_initial.json')['config']
    for arm in ARMS:
        hist=arm=='historical_full_F'
        folder=historical if hist else ROOT/'outputs/stage_i/paired_lora_pilot_runs_v1'/arm/'train'
        state=read(folder/'trainer_state.json'); comp=read(folder/('completion_report.json' if hist else 'completion.json'))
        adapter=folder/'best_adapter/adapter_model.safetensors'
        selected=Path(state['best_model_checkpoint'].replace('\\','/')).name
        checkpoint=folder/selected
        require(sha(adapter)==sha(checkpoint/'adapter_model.safetensors'),'Best adapter != selected checkpoint')
        cfg=hc if hist else config
        runtime=read(historical/'runtime_initial.json') if hist else read(ROOT/'outputs/stage_i/d052_paired_pilot_v1'/arm/'runtime.json')
        result=dict(steps=state['global_step'],epochs=state['epoch'],best_step=selected,best_eval_loss=state['best_metric'],
            eval_curve=[{k:v for k,v in x.items() if k in ['epoch','step','eval_loss']} for x in state['log_history'] if 'eval_loss' in x],
            training_examples=2000 if hist else 512,presentations=10000 if hist else 1024,
            adapter_sha256=sha(adapter),adapter_path=adapter.relative_to(ROOT).as_posix(),
            selected_adapter_matches_checkpoint=True,optimizer=cfg['optimization'],
            warmup_updates=63 if hist else 7,initialization='PEFT default Kaiming A / zero B' if hist else config['initialization'],
            packages=runtime['environment']['packages'] if hist else read(D054/'PREFLIGHT_REFERENCES_AND_FILES.json')['runtime'],
            hardware=runtime['device'] if hist else runtime['gpu'],
            data_order='Trainer random sampler, seed42' if hist else 'Frozen identical epoch permutations, seeds42/43',
            causal_comparator=False if hist else 'A/B matched except module scope and doubled adapter parameter count')
        if not hist:
            aud=ROOT/'outputs/stage_i/d052_paired_pilot_v1'/arm
            steps=rows(aud/'optimizer_scheduler_scaler_memory.jsonl')
            grads=rows(folder/'gradients.jsonl')
            require([x['global_step'] for x in steps]==list(range(1,129)),'Missing optimizer update')
            require(all(not x['AMP_optimizer_skipped'] and x['optimizer_steps']==[x['global_step']] and x['scheduler_last_epoch']==x['global_step'] for x in steps),'Optimizer/scheduler mismatch')
            later=[g for x in grads if x['next_step']>=3 for g in x['gradients'].values()]
            require(all(g['finite'] and g['nonzero'] for g in later),'Later gradients missing')
            result.update(module_scope=runtime['scope'],microbatches=len(rows(aud/'microbatch_loss_and_order.jsonl')),
                optimizer_updates_verified=len(steps),AMP_skipped_updates=0,gradient_records=len(grads),all_later_adapter_gradients_finite_nonzero=True,
                first_used_lr=steps[0]['used_lr'],max_used_lr=max(x['used_lr'] for x in steps),last_used_lr=steps[-1]['used_lr'],
                frozen_backbone_evidence=read(aud/'FROZEN_BACKBONE_INTEGRITY.json'),order_verification=read(aud/'ACTUAL_ORDER_VERIFICATION.json'))
        results[arm]=result
    write(out/'TRAINING_EXPOSURE_AND_PROVENANCE.json',results)
    return results,hc

def outputs(out,tok,inputs,generation,historical_config):
    saved={c['business_id']:c for c in rows(CLOUD/'outputs/stage_i/runs_v3/filtered/validation_predictions.jsonl')}
    pilot=rows(D054/'predictions.jsonl'); require(len(pilot)==48,'D054 count')
    for k,v in historical_config['generation'].items():require(generation[k]==v,'Historical decoding overrides differ')
    allrows=[]
    for c in inputs:
        h=c['baseline_prediction'];require(h==saved[c['business_id']],'Embedded historical output != saved full-F output')
        require(c['split']=='validation' and c['condition']=='filtered','Wrong scope')
        cases=[{**h,'arm':ARMS[0]}]+[p for p in pilot if p['business_id']==c['business_id']]
        require(len(cases)==3,'Missing A/B pair')
        for p in cases:
            require(tok.decode(p['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==p['text'],'Token/text mismatch')
            if p['arm']!=ARMS[0]:
                require(compatible_config(p['generation_config'])==compatible_config(generation),'Generation mismatch')
                require(p['source_sha256']==c['source_sha256'],'Source mismatch')
            text=p['text'];struct=structural_diagnostics(text,len(c['selected_photo_ids']));rep=repetition_diagnostics(text,p['token_ids'],generation)
            token=token_termination(p['token_ids'],generation)
            match=difflib.SequenceMatcher(None,c['source'],text,autojunk=False).find_longest_match()
            copy=text[match.b:match.b+match.size]
            markers=collections.Counter(re.findall(r'\[[A-Z_0-9]+\]',text))
            allrows.append(dict(business_id=c['business_id'],arm=p['arm'],subset=c['subset'],text=text,token_ids=p['token_ids'],
                source_sha256=c['source_sha256'],source=c['source'],teacher_target=c['target'],selected_photo_ids=c['selected_photo_ids'],
                structure=struct,repetition=rep,termination=token,prefix=prefix_diagnostic(p['token_ids'],c['label_ids'],tok),
                exact_bracket_markers=dict(markers),source_only_marker_occurrences=sum(v for k,v in markers.items() if k in ['[VISUAL_CONTEXT]','[REVIEWS]','[BUSINESS_INFORMATION]'] or re.fullmatch(r'\[(REV|IMG_CAP)_\d+\]',k)),
                whitespace_fraction=sum(x.isspace() for x in text)/max(1,len(text)),
                nonbreaking_space_count=text.count('\u00a0'),longest_exact_source_copy=dict(characters=match.size,text=copy,source_start=match.a,output_start=match.b,
                    interpretation='Retrieval diagnostic only. Source overlap does not establish coherent or correctly attributed factual claims.'),
                human_quality_verified=False))
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for subset in ['random','targeted','all24']:
            x=[v for v in allrows if v['arm']==arm and (subset=='all24' or v['subset']==subset)]
            mean=lambda key:statistics.mean(key(v) for v in x)
            fractions=lambda kind:[v['repetition'][kind]['4']['duplicate_fraction'] for v in x if v['repetition'][kind]['4']['duplicate_fraction'] is not None]
            summary[arm][subset]=dict(n=len(x),exact_one_each_marker=sum(v['structure']['review_marker_count']==v['structure']['visual_marker_count']==1 for v in x),
                structure_compliant=sum(v['structure']['section_headers_compliant'] for v in x),sentence_repetition=sum(v['repetition']['exact_repeated_sentence'] for v in x),
                adjacent_word_repetition=sum(v['repetition']['consecutive_repeated_word'] for v in x),
                mean_duplicate_sentence_fraction=mean(lambda v:v['repetition']['duplicate_sentence_fraction']),
                lexical4_duplicate_fraction=stats(fractions('lexical_ngrams')),bpe4_duplicate_fraction=stats(fractions('generated_bpe_ngrams')),
                cap_hits=sum(v['termination']['cap_reached'] for v in x),early_eos=sum(v['termination']['eos_before_length_cap'] for v in x),
                lengths=stats([len(v['token_ids']) for v in x]),header_prefix_complete=sum(v['prefix']['required_header_prefix_complete'] for v in x),
                header_mismatch_positions=dict(collections.Counter(str(v['prefix']['first_header_mismatch']) for v in x)),
                outputs_with_source_only_markers=sum(v['source_only_marker_occurrences']>0 for v in x),
                whitespace_fraction=stats([v['whitespace_fraction'] for v in x]),zero_caption_cases=[dict(business_id=v['business_id'],compliant=v['structure']['empty_visual_structurally_compliant']) for v in x if not v['selected_photo_ids']])
    write(out/'EXISTING_72_OUTPUTS_DIAGNOSTICS.json',allrows);write(out/'HISTORICAL_VS_PILOT_SUMMARY.json',summary)
    write(out/'HISTORICAL_COMPATIBILITY.json',dict(status='PASS_EXISTING_ARTIFACT_COMPARISON',historical_embedded_outputs_match_saved_predictions=24,
        source_token_ids_from_same_accepted_filtered_dataset=True,generation_overrides_identical=True,
        effective_config_limitation='Historical runner stores seven explicit overrides, not a contemporaneous full GenerationConfig. Same pinned model config, library version, tokenizer and code path support compatibility; historical unlogged runtime defaults cannot be independently replayed offline.',
        historical_training_config=historical_config,baseline_generation_config=generation,
        unmatched_training_exposure=True,no_quality_causality=True))
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/stage_i/d055_root_cause_v1');args=parser.parse_args();out=args.output.resolve()
    require(not out.exists(),'Use a NEW versioned output directory');out.mkdir(parents=True)
    require(not torch.cuda.is_initialized(),'CUDA already initialized: offline-only scope')
    inputs=rows(PREP/'evaluation_inputs.jsonl');require(len(inputs)==24,'Expected frozen24')
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json');config=read(ROOT/'configs/stage_i_paired_lora_pilot_v1.json')
    protected=read(D054/'PREFLIGHT_REFERENCES_AND_FILES.json')['protected'].copy()
    protected.update({info[k]:info[h] for info in manifest['datasets'].values() for k,h in [('path','sha256'),('stage_g_jsonl','stage_g_jsonl_sha256')]})
    for p in [D054/'predictions.jsonl',D054/'AUTOMATIC_EVALUATION.json',CLOUD/'outputs/stage_i/runs_v3/filtered/validation_predictions.jsonl',CLOUD/'outputs/stage_i/runs_v3/filtered/best_adapter/adapter_model.safetensors',CLOUD/'outputs/stage_i/runs_v3/filtered/trainer_state.json']:
        protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel,h in protected.items():require(sha(ROOT/rel)==h,'Protected input mismatch before audit: '+rel)
    write(out/'PROTECTED_BEFORE.json',protected)
    tokenpath=ROOT/'assets/tokenizers/led-base-16384';tok=LEDTokenizerFast.from_pretrained(tokenpath,local_files_only=True);slow=LEDTokenizer.from_pretrained(tokenpath,local_files_only=True)
    index=implementation(out);write(out/'LABEL_SHIFT_MASK_TESTS.json',mask_tests(inputs))
    vocab={}
    for marker in [REVIEW,VISUAL,'[VISUAL_CONTEXT]','[REV_1]','[REV_30]','[IMG_CAP_1]','[REVIEWS]','[BUSINESS_INFORMATION]']:
        vocab[marker]={ctx:{'ids':tok(ctx+marker,add_special_tokens=False)['input_ids'],'tokens':tok.tokenize(ctx+marker)} for ctx in ['', '\n',' ']}
    write(out/'MARKER_TOKENIZATION.json',dict(markers=vocab,added_vocabulary=tok.get_added_vocab(),all_special_ids=tok.all_special_ids,
        warning='Section/review markers are ordinary multi-token BPE strings, not registered single special tokens. This alone is not an implementation bug.'))
    targets(out,tok,slow,manifest);tr,hc=training(out,config);summary=outputs(out,tok,inputs,read(PREP/'expected_generation_config.json'),hc)
    for rel,h in protected.items():require(sha(ROOT/rel)==h,'Accepted artifact changed during offline audit')
    require(not torch.cuda.is_initialized(),'CUDA was unexpectedly initialized')
    write(out/'EXECUTION_RECEIPT.json',dict(status='OFFLINE_AUDIT_COMPLETE',protected_files=len(protected),all_unchanged=True,
        cuda_initialized=False,model_instantiations=0,model_forward_calls=0,generation_calls=0,training_updates=0,teacher_calls=0,held_out_data_accessed=False,
        script_sha256=sha(Path(__file__)),runtime={k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','peft','tokenizers','numpy']},
        read_only_scope='Six accepted train/validation datasets; frozen24 validation cases and72 existing outputs. No held-out test file opened.'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
