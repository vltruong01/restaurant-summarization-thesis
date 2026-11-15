"""Portable review-artifact verification, not a replacement for workspace preflight."""
import argparse,hashlib,json,pathlib,sys
def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=pathlib.Path);a=p.parse_args();r=a.bundle.resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(r/'SHA256_MANIFEST.json')
    for rel,h in manifest.items():
        f=(r/rel).resolve();assert f.is_relative_to(r) and f.is_file(),rel
        assert hashlib.sha256(f.read_bytes()).hexdigest()==h,rel
    from tokenizers import Tokenizer
    sys.path.insert(0,str(r/'code'));import stage_i_d056 as d
    cfg=read(r/'configs/stage_i_d056_minimal_diagnostic_v1.json');tok=Tokenizer.from_file(str(r/'evidence/tokenizer.json'))
    cases=read(r/'reports/cases.json');align=read(r/'reports/TOKEN_POSITION_ALIGNMENT.json')
    assert [c['business_id'] for c in cases]==d.CASES==cfg['cases']
    assert cfg['states']==d.STATES and cfg['ceilings']=={'teacher_forced':24,'cache_control':144,'traces':36}
    assert cfg['cache_prefix_lengths']==[11,12] and cfg['cache_calls_per_comparison']==3 and cfg['trace_new_token_cap']==64
    old=read(r/'evidence/D055_PROPOSED_GPU_DIAGNOSTIC_CONFIG.json');assert cfg['cache_screens']==old['cache_allclose_screen']
    for c,v in zip(cases,align):
        assert c['split']=='validation' and tok.encode(c['source']).ids==c['input_ids'] and tok.encode(c['target']).ids==c['label_ids']
        assert d.alignment(c,tok)==v
    approval=read(r/'reports/GPU_AUTHORIZATION_TEMPLATE.json');assert approval['GPU_execution_approved'] is False
    assert approval['config_sha256']==hashlib.sha256((r/'configs/stage_i_d056_minimal_diagnostic_v1.json').read_bytes()).hexdigest()
    assert approval['preflight_manifest_sha256']==hashlib.sha256((r/'reports/PREFLIGHT_MANIFEST.json').read_bytes()).hexdigest()
    assert cfg['D051_strict_replay_status']=='FAIL_UNCHANGED'
    assert not any(x in sys.modules for x in ['torch','transformers','peft'])
    print(json.dumps({'status':'PASS_REVIEW_ARTIFACTS_ONLY','hashes':len(manifest),'cases':3,'encodes_checked':6,'alignment_recomputed':True,
        'approval_template_false':True,'model_libraries_imported':False,'GPU_calls':0,
        'limitation':'Full281-file preflight and38-test suite require original workspace assets. This verifies the exported review evidence, not GPU feasibility or all absent model/dataset bytes.'},indent=2))
if __name__=='__main__':main()
