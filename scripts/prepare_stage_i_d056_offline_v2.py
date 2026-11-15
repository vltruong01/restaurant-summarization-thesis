"""Build/refresh D056-v2 offline reports and bindings, preserving each earlier build."""
import argparse,importlib.metadata,sys,shutil
from tokenizers import Tokenizer
from stage_i_d056_v2 import *
from validate_stage_i_d056_v2 import validate_prepared_runtime,fingerprint_audit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    config=read(CONFIG);check_config(config)
    prior=ROOT/'outputs/stage_i/d056_diagnostic_preparation_v1'
    require(sha(ROOT/'configs/stage_i_d056_minimal_diagnostic_v1.json')==config['parent_config_sha256'],'Parent config changed')
    protected=read(OUT/'PARENT_V1_PRESERVATION_MANIFEST.json')
    for rel,h in protected.items():require(sha(ROOT/rel)==h,'Preserved original changed: '+rel)
    files=verify_manifest(prior/'PREFLIGHT_MANIFEST.json')['files'].copy();files.update(protected)
    cases=read(OUT/'cases.json');require((OUT/'cases.json').read_bytes()==(prior/'cases.json').read_bytes(),'Frozen input records changed')
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    align=[alignment(c,tok) for c in cases]
    runtime=validate_prepared_runtime(config,cases);fingerprints=fingerprint_audit(config)
    versions={k:importlib.metadata.version(k) for k in config['runtime_pins']};require(versions==config['runtime_pins'],'Runtime package drift')
    reports={'TOKEN_POSITION_ALIGNMENT.json':align,'RUNTIME_SOURCE_MASK_ADAPTER_BINDINGS.json':runtime,
             'HISTORICAL_FINGERPRINT_COMPATIBILITY.json':fingerprints}
    names=list(reports)+['PREFLIGHT_MANIFEST.json','GPU_AUTHORIZATION_TEMPLATE.json']
    existing=[n for n in names if (OUT/n).exists()]
    if existing:
        require(args.refresh,'Existing build: explicit --refresh required')
        root=OUT/'preflight_build_history';root.mkdir(exist_ok=True);dest=root/f'build_{len(list(root.iterdir()))+1:03d}'
        dest.mkdir()
        for n in existing:shutil.copyfile(OUT/n,dest/n);(OUT/n).unlink()
    for name,value in reports.items():write(OUT/name,value)
    files.update(runtime['files']);files.update(fingerprints['files'])
    for b in runtime['adapter_bindings'].values():files.update(b['files'])
    paths=[CONFIG,*[ROOT/'scripts'/s for s in ['stage_i_d056_v2.py','run_stage_i_d056_gpu_diagnostic_v2.py',
        'validate_stage_i_d056_v2.py','create_stage_i_d056_amendment_v2.py','document_stage_i_d056_v2.py','prepare_stage_i_d056_offline_v2.py',
        'check_stage_i_d056_offline_v2.py','test_stage_i_d056_preparation_v2.py','package_stage_i_d056_review_v2.py','verify_stage_i_d056_review_bundle_v2.py']],
        ROOT/'tests/test_stage_i_d056_offline_v2.py',ROOT/'specs/009k_d056_amendment_v2.md']
    paths.extend(OUT/n for n in list(reports)+['cases.json','ADAPTER_GEOMETRY.json','IMPLEMENTATION_BINDINGS.json',
        'PARENT_V1_PRESERVATION_MANIFEST.json','D056_V2_AMENDMENT.md','REVISED_MINIMAL_GPU_PROTOCOL.md',
        'PROPOSED_GPU_EXECUTION_AUTHORIZATION.md','LITERATURE_INFORMED_RATIONALE.md','INTERPRETATION_AND_NEXT_EXPERIMENT_TREE.md'])
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    write(OUT/'PREFLIGHT_MANIFEST.json',{'status':'OFFLINE_HASH_SHAPE_AND_RUNTIME_BINDING_PASS_GPU_UNTESTED',
        'amendment':'D-056-v2','parent_config_sha256':config['parent_config_sha256'],'parent_preflight_manifest_sha256':sha(prior/'PREFLIGHT_MANIFEST.json'),
        'files':files,'config_sha256':sha(CONFIG),'runtime_pins':versions,'source_target_checks':6,'adapter_states_verified':3,
        'pretrained_weight_file_hash_verified':True,'backbone_tensor_hashing_convention_verified':True,'backbone_tensor_fingerprints_recomputed':False,
        'cache_probes':[3,11],'teacher_forced_retains_p12_and_visual_boundary':True,
        'model_instantiations':0,'model_forward_calls':0,'GPU_execution':False,'Teacher_calls':0,'held_out_files_opened':False,'D051_strict_replay':'FAIL_UNCHANGED'})
    verify_manifest(OUT/'PREFLIGHT_MANIFEST.json')
    write(OUT/'GPU_AUTHORIZATION_TEMPLATE.json',{'decision':'D-056','amendment':'D-056-v2','scope':'bounded_read_only_GPU_diagnosis',
        'GPU_execution_approved':False,'researcher_explicit_authorization':None,'approval_source':None,
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(OUT/'PREFLIGHT_MANIFEST.json'),'training_authorized':False})
    require(not any(k in sys.modules for k in ['torch','transformers','peft']),'Model runtime imported during preparation')
    print(json.dumps({'status':'OFFLINE_PASS_GPU_UNTESTED','files_bound':len(files),'preserved_v1_files':len(protected),
        'cache_probes':[3,11],'ceilings':config['ceilings'],'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(OUT/'PREFLIGHT_MANIFEST.json')},indent=2))

if __name__=='__main__':main()
