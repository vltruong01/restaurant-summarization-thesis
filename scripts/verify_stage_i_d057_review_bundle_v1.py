"""Portable D057 bytes/tokenization/numerical verification; no model runtime."""
import argparse,hashlib,json,pathlib,sys
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=pathlib.Path);args=parser.parse_args();root=args.bundle.resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(root/'SHA256_MANIFEST.json')
    for rel,h in manifest.items():
        p=(root/rel).resolve();assert p.is_relative_to(root) and p.is_file(),rel
        assert digest(p)==h,rel
    sys.path.insert(0,str(root/'code'));import stage_i_d056_v2 as d
    approval=read(root/'reports/RESEARCHER_GPU_AUTHORIZATION.json')
    assert approval['GPU_execution_approved'] is True and approval['approval_source']=='researcher_message'
    assert approval['training_authorized'] is False and approval['execution_attempts_authorized']==1
    assert digest(root/'configs/stage_i_d056_minimal_diagnostic_v2.json')==approval['config_sha256']
    assert digest(root/'evidence/D056_V2_PREFLIGHT_MANIFEST.json')==approval['preflight_manifest_sha256']
    assert digest(root/'code/run_stage_i_d056_gpu_diagnostic_v2.py')==approval['GPU_runner_sha256']
    cfg=read(root/'configs/stage_i_d056_minimal_diagnostic_v2.json')
    assert cfg['cache_prefix_lengths']==[3,11] and cfg['ceilings']=={'teacher_forced':24,'cache_control':144,'traces':36}
    assert cfg['D051_strict_replay_status']=='FAIL_UNCHANGED'
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(root/'evidence/tokenizer.json'))
    cases=read(root/'evidence/cases.json');align=read(root/'evidence/TOKEN_POSITION_ALIGNMENT.json')
    assert [x['business_id'] for x in cases]==d.CASES
    for case,expected in zip(cases,align):
        assert tok.encode(case['source']).ids==case['input_ids'] and tok.encode(case['target']).ids==case['label_ids']
        assert d.alignment(case,tok)==expected
    comparisons=0
    for path in (root/'raw_gpu').glob('*/*/*_cache_p*.json'):
        report=read(path)
        with np.load(path.with_name(path.stem+'_raw.npz'),allow_pickle=False) as pack:
            expected=d.vector_comparison(pack['uncached_logits'],pack['cached_logits'],int(pack['correct_id']),report['atol'],report['rtol'])
            assert all(report[k]==v for k,v in expected.items()),str(path)
        comparisons+=1
    summary=read(root/'reports/D057_SUMMARY.json')
    assert comparisons==summary['observed_artifact_counts']['cache_comparisons']
    assert not any(k in sys.modules for k in ['torch','transformers','peft'])
    print(json.dumps({'status':'PORTABLE_REVIEW_VERIFICATION_PASS','file_hashes_checked':len(manifest),
        'reviewed_three_way_hash_bindings_match':True,'exact_source_target_encodes':6,'token_alignments':3,
        'full_cache_vector_comparisons_recomputed':comparisons,'GPU_execution_was_authorized':True,
        'verification_model_or_GPU_calls':0,'original_execution_status':summary['status'],
        'limitation':'Verifies exported evidence, not absent backbone/checkpoint/dataset bytes or a new GPU run. No new factual-quality validation is claimed.'},indent=2))

if __name__=='__main__':main()
