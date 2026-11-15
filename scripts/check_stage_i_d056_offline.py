"""Repeatable read-only D056 offline verification; no model libraries or GPU."""
import importlib.metadata,json,sys
from tokenizers import Tokenizer
from stage_i_d056 import *
def main():
    config=read(CONFIG);check_config(config);manifest=verify_manifest(OUT/'PREFLIGHT_MANIFEST.json')
    adapter_bindings=adapter_checkpoint_bindings(config)
    versions={k:importlib.metadata.version(k) for k in config['runtime_pins']};require(versions==config['runtime_pins'],'Package versions changed')
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));cases=read(OUT/'cases.json')
    require([x['business_id'] for x in cases]==CASES,'Case order changed')
    expected=read(OUT/'TOKEN_POSITION_ALIGNMENT.json')
    require([alignment(c,tok) for c in cases]==expected,'Alignment report differs from recomputation')
    for c in cases:
        require(c['split']=='validation' and c['condition']=='filtered','Unexpected input scope')
        require(tok.encode(c['source']).ids==c['input_ids'] and tok.encode(c['target']).ids==c['label_ids'],'Source/label encodes differ')
    require(not any(k in sys.modules for k in ['torch','transformers','peft']),'Offline checker imported model runtime')
    require(not (ROOT/config['future_output']).exists(),'GPU execution artifacts unexpectedly exist')
    print(json.dumps({'status':'OFFLINE_PREFLIGHT_PASS_GPU_UNTESTED','hashes_checked':len(manifest['files']),
        'cases':3,'source_target_tokenization_checks':6,'alignment_recomputed':True,'adapter_checkpoint_bindings':adapter_bindings,'model_libraries_imported':False,
        'model_forward_calls':0,'GPU_calls':0,'training_updates':0,'Teacher_calls':0,'held_out_files_opened':False,
        'D051_strict_replay_status':'FAIL_UNCHANGED'},indent=2))
if __name__=='__main__':main()
