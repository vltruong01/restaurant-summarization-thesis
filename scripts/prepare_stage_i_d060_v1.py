"""Offline CPU-only compatibility audit. No forward, backward, generation or training."""
import copy
import json
import os
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
from stage_i_d060 import *

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    cfg=read(CONFIG);check_config(cfg)
    require(not any((PREP/name).exists() for name in ('DATA_ALIGNMENT.json','CPU_ADAPTER_COMPATIBILITY.json','OFFLINE_CHECKS.json','PREFLIGHT_MANIFEST.json')),'Preserve completed audit/preparation; no overwrite')
    PREP.mkdir(parents=True,exist_ok=True)
    # All existing D059-v2 preparation bindings plus RO provenance stay unchanged.
    protected=dict(read(ROOT/'outputs/stage_i/d059_preparation_v2/PREFLIGHT_MANIFEST.json')['files'])
    cloud=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
    accepted=read(cloud/'configs/stage_i_lora_led_training_approved_v2.json')
    protected[(cloud/'configs/stage_i_lora_led_training_approved_v2.json').relative_to(ROOT).as_posix()]=sha(cloud/'configs/stage_i_lora_led_training_approved_v2.json')
    manifest=read(ROOT/cfg['stage_h_manifest'])
    require(sha(ROOT/cfg['stage_h_manifest'])==accepted['stage_h_manifest_sha256'],'Different Stage H manifest')
    require(accepted['revision']==cfg['revision'] and accepted['generation']==cfg['generation'],'Historical model/generation differs')
    require(accepted['source_limit']==5000 and accepted['label_limit']==768,'Historical token limits differ')
    for condition in ('reviews_only','filtered'):
        folder=cloud/'outputs/stage_i/runs_v3'/condition
        runtime=read(folder/'runtime_initial.json');state=read(folder/'trainer_state.json')
        require(runtime['config']['revision']==cfg['revision'] and runtime['config']['lora']==accepted['lora'],'Runtime revision/scope differs')
        require(state['global_step']==1250 and state['best_model_checkpoint'].endswith('checkpoint-1250'),'Historical chosen checkpoint differs')
        require(sha(folder/'best_adapter/adapter_model.safetensors')==sha(folder/'checkpoint-1250/adapter_model.safetensors'),'Best adapter mismatch')
        for name in ('runtime_initial.json','trainer_state.json','run_binding.json','completion_report.json','best_adapter/adapter_config.json','best_adapter/adapter_model.safetensors','checkpoint-1250/adapter_model.safetensors','checkpoint-1250/adapter_config.json'):
            p=folder/name;protected[p.relative_to(ROOT).as_posix()]=sha(p)
    tm=read(ROOT/manifest['tokenizer_manifest'])
    require(tm['revision']==cfg['revision'],'Tokenizer revision differs')
    for name,h in tm['sha256'].items():
        p=ROOT/tm['relative_directory']/name;require(sha(p)==h,'Tokenizer bytes differ');protected[p.relative_to(ROOT).as_posix()]=h
    from tokenizers import Tokenizer
    import numpy as np
    tok=Tokenizer.from_file(str(ROOT/tm['relative_directory']/'tokenizer.json'))
    stats={};ids={};rows={};encodes=0
    for split,n in (('train',2000),('validation',200)):
        for cond in ('reviews_only','unfiltered','filtered'):
            key=f'{split}_{cond}';info=manifest['datasets'][key]
            for field,hfield in (('path','sha256'),('stage_g_jsonl','stage_g_jsonl_sha256')):
                require(sha(ROOT/info[field])==info[hfield],'Frozen input mismatch '+key)
                protected[info[field]]=info[hfield]
            rs=[json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
            with np.load(ROOT/info['path'],allow_pickle=False) as archive:
                # NpzFile indexing decompresses the full array on every access.
                # Materialize once; values and all assertions stay unchanged.
                z={k:archive[k] for k in archive.files}
                require(len(rs)==n and len(set(r['business_id'] for r in rs))==n,'Wrong/duplicate business count')
                ids[key]=[r['business_id'] for r in rs];require(z['business_ids'].tolist()==ids[key],'Packed ID mapping differs')
                for i,r in enumerate(rs):
                    require(r['split']==split and r['condition']==cond,'Wrong split/condition')
                    for text,values,offsets in [('source','input_ids','input_offsets'),('target','labels','label_offsets')]:
                        encoded=tok.encode(r[text]).ids;packed=z[values][z[offsets][i]:z[offsets][i+1]].tolist()
                        require(encoded==packed,'Silent truncation/token mismatch '+key);encodes+=1
                        require(encoded[0]==0 and encoded[-1]==2 and len(encoded)<=(5000 if text=='source' else 768),'Token budget or BOS/EOS mismatch')
                    require(hashlib.sha256(r['source'].encode()).hexdigest()==r['source_sha256'],'Source text hash differs')
                stats[key]={'businesses':n,'max_source':int(np.diff(z['input_offsets']).max()),'max_label':int(np.diff(z['label_offsets']).max()),'zero_captions':int((z['caption_counts']==0).sum())}
            rows[key]=rs
        for cond in ('unfiltered','filtered'):
            require(ids[f'{split}_reviews_only']==ids[f'{split}_{cond}'],'Cross-condition order differs')
        for ro,u,f in zip(rows[f'{split}_reviews_only'],rows[f'{split}_unfiltered'],rows[f'{split}_filtered']):
            for r in (u,f):
                require(ro['selected_review_ids']==r['selected_review_ids'],'Selected review order differs')
                require(ro['source']==r['source'].split('\n\n[VISUAL_CONTEXT]',1)[0],'Review/source text differs')
                require(ro['target'].strip()==r['target'].split('[VISUAL_EVIDENCE]',1)[0].strip(),'Review consensus not shared')
    require(not set(ids['train_filtered'])&set(ids['validation_filtered']),'Train/validation leakage')
    write(PREP/'DATA_ALIGNMENT.json',{'status':'PASS','stats':stats,'exact_source_and_target_encodes':encodes,'shared_review_ids_text_order_and_consensus':True,'no_truncation':True,'test_opened':False})
    # Actual trained tensors, not a shape-only compatibility assertion.
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from transformers import LEDForConditionalGeneration
    from safetensors import safe_open
    from stage_i_d056_v2 import tensor_fingerprint
    from stage_i_paired_pilot import inspect_attached
    deny=RuntimeError('Offline audit forbids model execution')
    require(not torch.cuda.is_initialized(),'Unexpected prior CUDA initialization')
    with patch.object(torch.cuda,'_lazy_init',side_effect=deny) as cuda_guard,patch.object(LEDForConditionalGeneration,'forward',side_effect=deny),patch.object(LEDForConditionalGeneration,'generate',side_effect=deny),patch.object(PeftModel,'forward',side_effect=deny),patch.object(PeftModel,'generate',side_effect=deny),patch.object(torch.Tensor,'backward',side_effect=deny):
        model,scope,loaded=load_ro_cpu(cfg)
        history=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')['architectures'][0]['canonical_initial_backbone_hashes']
        canonical={n.removeprefix('base_model.model.').replace('.base_layer.','.'):tensor_fingerprint(p) for n,p in model.named_parameters() if not p.requires_grad}
        require(canonical==history,'Historical pretrained tensor fingerprints differ')
        ro=ROOT/cfg['initialization']['adapter_path'];ff=cloud/'outputs/stage_i/runs_v3/filtered/best_adapter'
        tensors={}
        for role,folder in [('RO',ro),('F',ff)]:
            with safe_open(folder/'adapter_model.safetensors',framework='numpy') as z:
                tensors[role]={k:{'shape':list(z.get_tensor(k).shape),'dtype':str(z.get_tensor(k).dtype),'finite':bool(np.isfinite(z.get_tensor(k)).all())} for k in z.keys()}
        require(tensors['RO']==tensors['F'],'Historical RO/F geometry differs')
        before={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        with tempfile.TemporaryDirectory(prefix='d060_cpu_adapter_') as tmp:
            model.save_pretrained(tmp,safe_serialization=True)
            from safetensors.torch import load_file
            saved=load_file(str(Path(tmp)/'adapter_model.safetensors'),device='cpu')
            require(set(saved)==set(before) and all(torch.equal(v,saved[k]) for k,v in before.items()),'CPU adapter serialization mismatch')
        require(inspect_attached(model,'A_decoder_only')==scope,'Saving changed trainable scope')
        # Instantiate, but never step, the exact fresh Stage-2 optimizer/scheduler.
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'weight_decay':.01},{'params':[],'weight_decay':0.}],lr=.0002,betas=(.9,.999),eps=1e-8)
        from transformers import get_linear_schedule_with_warmup
        scheduler=get_linear_schedule_with_warmup(optimizer,38,750)
        require(not optimizer.state,'Unexpected inherited optimizer moments')
        cuda_guard.assert_not_called()
        require(all(torch.equal(v,get_peft_model_state_dict(model)[k]) for k,v in before.items()),'Offline audit altered adapter')
        write(PREP/'CPU_ADAPTER_COMPATIBILITY.json',{'status':'PASS_CPU_LOAD_AND_SERIALIZATION_ONLY','loaded':loaded,'scope':scope,'RO_F_tensor_inventory':tensors,'backbone_historical_fingerprints_match':True,'backbone_parameter_count':sum(p.numel() for p in model.parameters() if not p.requires_grad),'adapter_trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'all_parameters_on_cpu':True,'optimizer_initial_state_entries':len(optimizer.state),'scheduler_initial_state':scheduler.state_dict(),'forward_backward_generation_optimizer_step_calls':0,'GPU_feasibility_claimed':False,'adapter_metadata_revision':read(ro/'adapter_config.json')['revision'],'revision_provenance':'Pinned historical runtime, approved config, snapshot file hash, and exact D052 canonical backbone tensor fingerprints; null adapter revision is not treated as proof.'})
    orders=read(ROOT/'outputs/stage_i/d059_preparation_v2/EPOCH_ORDERS.json')
    require(orders['training_business_ids']==ids['train_filtered'],'Data-order input differs')
    orders['indices']=orders['indices'][:3]
    write(PREP/'EPOCH_ORDERS.json',orders)
    for name in ('EVALUATION_MANIFEST.json','FEASIBILITY_CASES.json'):
        write(PREP/name,read(ROOT/'outputs/stage_i/d059_preparation_v2'/name))
    # No historical evidence is overwritten or relabeled by this preparation.
    for rel,h in protected.items():
        p=Path(rel);p=p if p.is_absolute() else ROOT/p
        require(sha(p)==h,'Frozen history changed: '+rel)
    write(PREP/'PROTECTED_INPUTS.json',{'files':protected,'verified_unchanged':len(protected),'D051_strict_replay':'FAIL_UNCHANGED'})
    write(PREP/'OFFLINE_CHECKS.json',{'status':'PASS','recorded_utc':datetime.now(timezone.utc).isoformat(),'protected_files':len(protected),'exact_encodes':encodes,'train_validation_only':True,'new_reference_labels':0,'Teacher_calls':0,'GPU_operations':0,'new_predictions':0,'CPU_model_loads':1})
    print(json.dumps({'status':'OFFLINE_PASS','encodes':encodes,'protected_files':len(protected),'GPU_operations':0}))

if __name__=='__main__':main()
