"""D056 preflight: tokenizers/NumPy/AST/hash reads only. Never imports model code."""
import ast,copy,importlib.metadata,importlib.util,json,math,shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from stage_i_d056 import *

def main():
    require(not CONFIG.exists(),'Preserve existing preparation config')
    old=read(ROOT/'outputs/stage_i/d055_root_cause_v2/PROPOSED_GPU_DIAGNOSTIC_CONFIG.json')
    training=read(ROOT/'outputs/stage_i/d055_root_cause_v2/TRAINING_EXPOSURE_AND_PROVENANCE.json')
    config={'decision':'D-056','status':'OFFLINE_PREPARED_GPU_NOT_AUTHORIZED','cases':CASES,'states':STATES,
        'ceilings':{'teacher_forced':24,'cache_control':144,'traces':36},'trace_new_token_cap':64,
        'precision_modes':['FP32','FP16_autocast'],'weights':'FP32','seed':42,'tf32':False,'batch_size':1,
        'model_id':'allenai/led-base-16384','revision':'38335783885b338d93791936c54bb4be46bebed9',
        'source_limit':5000,'label_limit':768,'attention_window':1024,'global_attention':'source_token0_only',
        'decoder_attention_mask':None,'cache_prefix_lengths':[11,12],'cache_calls_per_comparison':3,
        'cache_design_change':'D055 proposed3/11/12 with full-TF reference; D056 uses11/12 with explicit same-prefix uncached/setup/incremental calls. Position3 remains in teacher-forced and trace observations. No GPU executed.',
        'cache_screens':old['cache_allclose_screen'],'cross_precision_screen':{'max_abs_gt':0.1,'or_argmax_change':True,'near_zero_floor':1e-8},
        'generation':read(PREP/'expected_generation_config.json'),
        'trace_modes':['greedy_unprefixed','beam4_unprefixed','greedy_correct_header_prefix'],
        'trace_instrumentation':{'return_dict_in_generate':True,'output_scores':True,'output_logits':True,'output_attentions':False,'output_hidden_states':False},
        'limits':{'minimum_free_disk_bytes':2*1024**3,'minimum_GPU_headroom_fraction':0.15,'automatic_fallback':False,'max_models_live':1},
        'adapter_states':{k:{'path':v['adapter_path'],'sha256':v['adapter_sha256'],'scope':'B_encoder_decoder' if k=='B_encoder_decoder' else 'A_decoder_only'} for k,v in training.items()},
        'runtime_pins':{'torch':'2.1.0+cu118','transformers':'4.51.3','peft':'0.15.2','accelerate':'1.6.0','numpy':'1.26.4','tokenizers':'0.21.1','safetensors':'0.8.0'},
        'D051_strict_replay_status':'FAIL_UNCHANGED','GPU_execution_approved':False,'training_authorized':False,
        'future_output':'outputs/stage_i/d056_gpu_diagnostic_v1'}
    # Bind all old accepted scientific files; living logs are snapshotted separately.
    files=read(ROOT/'outputs/stage_i/d055_root_cause_v2/PROTECTED_BEFORE.json').copy()
    for folder in ['outputs/stage_i/d055_root_cause_v2','outputs/stage_i/d051_replay_investigation_v1']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'outputs/stage_i/d053_final_references_v1').glob('*.json'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ['outputs/stage_i/stage_i_d055_chatgpt_review_bundle_v2.zip','outputs/stage_h/led_dataset_v1/manifest.json',
        'assets/tokenizers/led-base-16384.manifest.json','assets/models/led-base-16384/config.json','configs/stage_i_paired_lora_pilot_v1.json',
        'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json','outputs/stage_i/d052_paired_pilot_v1/A_decoder_only/runtime.json']:
        files[rel]=sha(ROOT/rel)
    tm=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    for rel,h in tm['sha256'].items():files[str(Path(tm['relative_directory'])/rel).replace('\\','/')]=h
    pre=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/PRE_EXECUTION_INTEGRITY.json')
    cache=Path.home()/'.cache/huggingface/hub/models--allenai--led-base-16384/snapshots'/config['revision']
    require(cache.is_dir(),'Pinned local backbone absent; no download allowed')
    for name,row in pre['cached_backbone_files'].items():files[str(cache/name)]=row['sha256']
    config['local_backbone_snapshot']=str(cache)
    for path,h in files.items():require(sha(Path(path) if Path(path).is_absolute() else ROOT/path)==h,'Pre-existing integrity mismatch: '+path)
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    allcases=rows(PREP/'evaluation_inputs.jsonl');selected=[next(c for c in allcases if c['business_id']==bid) for bid in CASES]
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json');npz=manifest['datasets']['validation_filtered']
    with np.load(ROOT/npz['path'],allow_pickle=False) as pack:
        bids=pack['business_ids'].tolist();align=[]
        for c,original in zip(selected,old['cases']):
            require(c['split']=='validation' and c['condition']=='filtered','Not frozen validation Filtered')
            i=bids.index(c['business_id']);lo=pack['label_offsets'];io=pack['input_offsets']
            require(pack['labels'][lo[i]:lo[i+1]].tolist()==c['label_ids'] and pack['input_ids'][io[i]:io[i+1]].tolist()==c['input_ids'],'NPZ/source mismatch')
            require(tok.encode(c['source']).ids==c['input_ids'] and tok.encode(c['target']).ids==c['label_ids'],'Exact tokenization mismatch')
            require(ids_sha(c['input_ids'])==original['input_ids_sha256'] and ids_sha(c['label_ids'])==original['label_ids_sha256'],'D055 frozen token hashes')
            require(c['source_sha256']==original['source_sha256'],'D055 source binding')
            align.append(alignment(c,tok))
    ref=read(ROOT/'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json')
    require(len(ref['units'])==576 and len(ref['approved_exclusion_log'])==88,'References changed')
    geometry={}
    for state,d in config['adapter_states'].items():
        h=tensor_header(ROOT/d['path']);n=sum(math.prod(x['shape']) for x in h.values())
        require(n==(589824 if state=='B_encoder_decoder' else 294912),'Adapter parameter mismatch')
        require(all(v['dtype']=='F32' for v in h.values()),'Storage dtype mismatch')
        require(all((v['shape']==[8,768] if '.lora_A.' in k else v['shape']==[768,8]) for k,v in h.items()),'Unexpected adapter tensor shape')
        geometry[state]={'parameters':n,'tensors':len(h),'tensor_shapes':{k:v['shape'] for k,v in h.items()}}
    index={}
    for package,rel,names in [('transformers','models/led/modeling_led.py',['shift_tokens_right','forward']),
        ('transformers','generation/utils.py',['_beam_search','_update_finished_beams','_prepare_decoder_input_ids_for_generation','compute_transition_scores']),
        ('peft','peft_model.py',['generate','get_base_model']),('peft','tuners/lora/model.py',['__getattr__'])]:
        path=Path(importlib.util.find_spec(package).origin).parent/rel;txt=path.read_text(encoding='utf-8');tree=ast.parse(txt)
        dest=OUT/'implementation'/package/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        index[package+'/'+rel]={'installed_path':str(path),'sha256':sha(path),'functions':{name:[{'start':n.lineno,'end':n.end_lineno} for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name] for name in names}}
        files[str(path)]=sha(path)
    versions={k:importlib.metadata.version(k) for k in config['runtime_pins']}
    require(versions==config['runtime_pins'],'Pinned local package versions differ')
    write(CONFIG,config);check_config(config)
    adapter_bindings=adapter_checkpoint_bindings(config)
    for binding in adapter_bindings.values():files.update(binding['files'])
    write(OUT/'cases.json',selected);write(OUT/'TOKEN_POSITION_ALIGNMENT.json',align)
    write(OUT/'ADAPTER_GEOMETRY.json',geometry);write(OUT/'IMPLEMENTATION_BINDINGS.json',index)
    for p in [CONFIG,ROOT/'scripts/stage_i_d056.py',ROOT/'scripts/run_stage_i_d056_gpu_diagnostic.py',ROOT/'scripts/prepare_stage_i_d056_offline.py',ROOT/'scripts/stage_hi_led_data.py',ROOT/'scripts/stage_i_paired_pilot.py']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ['cases.json','TOKEN_POSITION_ALIGNMENT.json','ADAPTER_GEOMETRY.json','IMPLEMENTATION_BINDINGS.json']:
        p=OUT/name;files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'PREFLIGHT_MANIFEST.json',{'status':'OFFLINE_HASH_AND_SHAPE_PASS_GPU_UNTESTED','files':files,'config_sha256':sha(CONFIG),
        'runtime_pins':versions,'source_target_checks':6,'adapter_states_verified':3,'pretrained_weight_file_hash_verified':True,
        'D051_strict_replay':'FAIL_UNCHANGED','model_instantiations':0,'model_forward_calls':0,'GPU_execution':False,'Teacher_calls':0,'held_out_files_opened':False})
    verify_manifest(OUT/'PREFLIGHT_MANIFEST.json')
    write(OUT/'GPU_AUTHORIZATION_TEMPLATE.json',{'decision':'D-056','scope':'bounded_read_only_GPU_diagnosis','GPU_execution_approved':False,
        'researcher_explicit_authorization':None,'approval_source':None,'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(OUT/'PREFLIGHT_MANIFEST.json'),'training_authorized':False})
    for p in [ROOT/'DECISIONS.md',ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md',ROOT/'thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        dest=OUT/'documentation_before'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    print(json.dumps({'status':'OFFLINE_PREFLIGHT_PASS_GPU_UNTESTED','protected_files':len(files),'selected_cases':len(selected),'versions':versions},indent=2))
if __name__=='__main__':main()
