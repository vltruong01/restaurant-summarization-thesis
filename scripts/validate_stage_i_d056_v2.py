"""D056-v2 CPU/AST/file-only validation. No model runtime import or weight load."""
import ast,hashlib,types,importlib.metadata,importlib.machinery
from pathlib import Path
import numpy as np
from stage_i_d056_v2 import *

class Tensor:
    """NumPy stand-in ONLY for historical serializer testing, not a model tensor."""
    def __init__(self,array):self.array=np.asarray(array)
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):self.array=np.array(self.array,copy=True,order='C');return self
    def numpy(self):return self.array

class Parameter(Tensor):pass

def historical_fingerprint_function():
    path=ROOT/'scripts/run_stage_i_d050_feasibility_v1.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
    isolated=ast.Module(body=[node],type_ignores=[])
    scope={'np':np,'torch':types.SimpleNamespace(Tensor=Tensor),'hashlib':hashlib}
    exec(compile(ast.fix_missing_locations(isolated),str(path)+'[fingerprint only]','exec'),scope)
    return scope['fingerprint'],{'path':path.relative_to(ROOT).as_posix(),'file_sha256':sha(path),
        'function_AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
        'method':'Isolated historical function AST; NumPy Tensor/Parameter stand-ins. No historical module import.'}

def fingerprint_audit(config):
    historical,source=historical_fingerprint_function();checks=[]
    for dtype in [np.float32,np.float16,np.int64]:
        for array in [np.arange(12,dtype=dtype).reshape(3,4),np.arange(12,dtype=dtype).reshape(3,4).T,np.array(0,dtype=dtype)]:
            for cls in [Tensor,Parameter]:
                expected=historical(cls(array));actual=tensor_fingerprint(cls(array));require(actual==expected,'Historical serializer mismatch')
                checks.append({'type':cls.__name__,'shape':list(array.shape),'dtype':str(array.dtype),'sha256':actual})
    matched={};files={source['path']:source['file_sha256']}
    for arm in ['A_decoder_only','B_encoder_decoder']:
        recpath=ROOT/f'outputs/stage_i/d052_paired_pilot_v1/{arm}/parameter_hashes_after.json'
        records=read(recpath);adapter=ROOT/config['adapter_states'][arm]['path'];header=tensor_header(adapter)
        with adapter.open('rb') as f:
            length=int.from_bytes(f.read(8),'little');base=8+length;n=0
            for key,desc in header.items():
                require(desc['dtype']=='F32','Unexpected adapter storage')
                start,end=desc['data_offsets'];f.seek(base+start);raw=f.read(end-start)
                require(len(raw)==end-start,'Truncated safetensors payload')
                arr=np.frombuffer(raw,dtype='<f4').reshape(desc['shape'])
                name=key.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                require(name in records and tensor_fingerprint(Parameter(arr))==records[name]['sha256'],'Saved D052 adapter tensor fingerprint differs: '+name)
                require(historical(Parameter(arr))==records[name]['sha256'],'Historical AST serializer differs on saved adapter')
                n+=1
        matched[arm]={'actual_saved_tensors_checked':n,'record':recpath.relative_to(ROOT).as_posix(),'same_recorded_Parameter_fingerprint':True}
        files[recpath.relative_to(ROOT).as_posix()]=sha(recpath)
    for rel,needle in [('scripts/run_stage_i_d054_generation_v1.py','fingerprint'),('scripts/run_stage_i_d052_paired_pilot_v2.py','parameters')]:
        tree=ast.parse((ROOT/rel).read_text(encoding='utf-8'))
        require(any(isinstance(n,ast.ImportFrom) and n.module=='run_stage_i_d050_feasibility_v1' and needle in [a.name for a in n.names] for n in ast.walk(tree)),'Historical fingerprint provenance changed')
        files[rel]=sha(ROOT/rel)
    report=read(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json')
    canonical=[a['canonical_initial_backbone_hashes'] for a in report['architectures']]
    require(canonical[0]==canonical[1],'Historical A/B backbone hash disagreement')
    return {'status':'PASS_CONVENTION_AND_SAVED_ADAPTER_PAYLOADS','historical_function':source,
            'convention':'type(value).__name__ + colon + dtype + shape + contiguous C-order bytes + semicolon; no separators between dtype/shape/bytes',
            'synthetic_fixture_comparisons':checks,'real_adapter_payload_checks':matched,'canonical_backbone_fingerprints':len(canonical[0]),
            'canonical_backbone_d052_A_equals_B':True,'bare_Parameter_and_bare_Tensor_hashes_intentionally_differ':True,
            'limitation':'Backbone file bytes verified, but backbone tensor fingerprints are not recomputed offline: no model or .bin weight loading. Future authorized runner compares every named Parameter before inference.',
            'files':files,'actual_torch_imported':False}

def function_node(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)

def implementation_audit():
    bindings=read(OUT/'IMPLEMENTATION_BINDINGS.json');item=bindings['transformers/generation/utils.py']
    path=Path(item['installed_path']);require(sha(path)==item['sha256'],'Pinned generation implementation differs')
    text=path.read_text(encoding='utf-8');beam=function_node(path,'_beam_search');finish=function_node(path,'_update_finished_beams')
    transitions=function_node(path,'compute_transition_scores');beamtext=ast.get_source_segment(text,beam)
    finishtext=ast.get_source_segment(text,finish);transitiontext=ast.get_source_segment(text,transitions)
    require('_update_finished_beams' in beamtext and 'decoder_prompt_len + max_generated_length' in beamtext,'Unexpected beam route')
    require('next_token_hits_stopping_criteria & top_num_beam_mask' in finishtext,'Unexpected finished beam predicate')
    require('cur_len + 1 - decoder_prompt_len' in finishtext,'Unexpected sequence-score length semantics')
    require('beam_scorer.finalize' not in beamtext and 'sequence_outputs' not in beamtext,'Unexpected legacy finalizer')
    require('beam_indices' in transitiontext and 'cut_idx' in transitiontext,'Unknown transition score alignment')
    runner=ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py';tree=ast.parse(runner.read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={'backward','train','step','save_pretrained','save_model','zero_grad'}
    require(not any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls),'Forbidden training operation')
    heavy=[n.lineno for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) and
           ((isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0] in ['torch','transformers','peft']) or
            (isinstance(n,ast.Import) and any(x.name.split('.')[0] in ['torch','transformers','peft'] for x in n.names)))]
    gate=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='approval_gate')
    require(heavy and min(heavy)>gate,'Model imports before approval gate')
    diagnostic_calls=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='call']
    require(len(diagnostic_calls)==4,'Unexpected explicit diagnostic call sites')
    require(not any(kw.arg=='labels' for n in diagnostic_calls for kw in n.keywords),'Labels disable cache unexpectedly')
    require(all(any(kw.arg=='decoder_input_ids' for kw in n.keywords) for n in diagnostic_calls),'Implicit decoder IDs')
    cache=[n for n in diagnostic_calls if isinstance(n.args[0],ast.Constant) and n.args[0].value=='cache_control']
    require(len(cache)==3,'Cache comparison must have exactly three explicit calls')
    slices=[ast.unparse(next(kw.value for kw in n.keywords if kw.arg=='decoder_input_ids')) for n in sorted(cache,key=lambda n:n.lineno)]
    require(slices==['z[:, :pfx]','z[:, :pfx - 1]','z[:, pfx - 1:pfx]'],'Cache decoder slice mismatch')
    return {'status':'PASS_STATIC_PREPARED_IMPLEMENTATION','generation_source':item,
            'pinned_beam_path':'vectorized GenerationMixin._beam_search; _update_finished_beams; not BeamSearchScorer.finalize',
            'EOS':'Only counted when newly emitted selected token ID2 has an ancestry/score; initial decoder-start2 is not EOS.',
            'censor':'custom stopping criterion can finalize non-EOS candidates at64; can change beam winner and normalization',
            'finalization':'pinned route crops by valid beam ancestry and does not append EOS; selected earlier EOS can coexist with later global censor',
            'length_normalization':'sum processed transition scores / emitted-token-count**length_penalty, includes selected EOS and excludes decoder prefix',
            'source_functions':{n.name:{'start':n.lineno,'end':n.end_lineno,'AST_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()} for n in [beam,finish,transitions]},
            'cache_call_decoder_slices':slices,'cache_calls_per_comparison':3,'imports_after_gate':True,
            'raw_and_processed_top10_selected_independently':True,'no_model_loaded':True}

def validate_prepared_runtime(config,cases):
    check_config(config)
    versions={k:importlib.metadata.version(k) for k in config['runtime_pins']}
    require(versions==config['runtime_pins'],'Prepared runtime package version mismatch')
    bindings=read(OUT/'IMPLEMENTATION_BINDINGS.json')
    for key,desc in bindings.items():
        # PathFinder inspects package paths without executing __init__ or the
        # model-import guard. Normal imports remain blocked by the test finder.
        package,rel=key.split('/',1);spec=importlib.machinery.PathFinder.find_spec(package)
        require(spec is not None and spec.origin,'Pinned package not resolvable')
        actual=Path(spec.origin).parent/rel
        require(actual.resolve()==Path(desc['installed_path']).resolve() and sha(actual)==desc['sha256'],'Resolved runtime source differs from pinned implementation')
    masks=offline_mask_bindings(cases);adapters=adapter_checkpoint_bindings(config)
    for state,desc in config['adapter_states'].items():
        header=tensor_header(ROOT/desc['path'])
        require(all(v['dtype']=='F32' and v['shape']==([8,768] if '.lora_A.' in k else [768,8]) for k,v in header.items()),'Adapter shape/dtype mismatch')
        require(sum(math.prod(v['shape']) for v in header.values())==(589824 if state=='B_encoder_decoder' else 294912),'Adapter parameter-count mismatch')
    old=read(ROOT/'outputs/stage_i/d055_root_cause_v2/PROPOSED_GPU_DIAGNOSTIC_CONFIG.json')
    for row,prior in zip(masks['cases'],old['cases']):
        require(row['business_id']==prior['business_id'],'Runtime case identity/order mismatch')
        for key in ['source_sha256','input_ids_sha256','label_ids_sha256']:require(row[key]==prior[key],'D055 source/token binding mismatch')
    return {'status':'PASS_OFFLINE_BINDINGS_GPU_UNTESTED','masks':masks,'adapter_bindings':adapters,'runtime_versions':versions,'resolved_implementation_paths_verified':True,
            'implementation':implementation_audit(),'remaining_runtime_checks':['actual LED/PEFT module attachment','actual backbone named Parameter hashes','Torch collation and generation mask propagation','GPU cache tensors, precision/logits and beam outputs'],
            'model_loaded':False,'files':{rel:sha(ROOT/rel) for rel in ['scripts/stage_hi_led_data.py','scripts/stage_i_paired_pilot.py','scripts/run_stage_i_d050_feasibility_v1.py']}}

if __name__=='__main__':
    cfg=read(CONFIG);report=validate_prepared_runtime(cfg,read(OUT/'cases.json'));fp=fingerprint_audit(cfg)
    print(json.dumps({'runtime':report,'fingerprints':fp},indent=2))
