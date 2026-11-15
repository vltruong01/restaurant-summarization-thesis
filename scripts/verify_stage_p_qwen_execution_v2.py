"""Read-only portable audit of Stage P native records and resource tables."""
import argparse,hashlib,json,sys
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tsha(s):return hashlib.sha256(s.encode()).hexdigest()
def ish(x):return tsha(json.dumps(x,separators=(',',':')))
def main():
    a=argparse.ArgumentParser();a.add_argument('--package',type=Path,required=True);a.add_argument('--decode-tokens',action='store_true');args=a.parse_args();p=args.package.resolve()
    for r in read(p/'PACKAGE_MANIFEST.json')['files']:
        f=(p/r['file']).resolve();assert f.is_relative_to(p) and f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],r['file']
    root=p/'NATIVE';spec=read(root/'FROZEN_SPEC.json');plan=read(root/'EXECUTION_PLAN.json');identity=read(root/'REMOTE_IDENTITY.json');execution=root/'EXECUTION'
    for file,h in spec['models']['Qwen3.5-9B']['tokenizer_files'].items():assert sha(root/'MODEL_PROVENANCE/TOKENIZERS/Qwen3.5-9B'/file)==h
    assert sha(root/'INFERENCE_INPUTS.jsonl')==spec['input_sha256'] and sha(root/'PROMPTS.json')==spec['prompts_sha256']
    rows=[json.loads(l) for l in (root/'INFERENCE_INPUTS.jsonl').read_text(encoding='utf8').splitlines() if l];by={r['call_id']:r for r in rows};rendered={r['call_id']:r for r in read(root/'RENDERED_CALLS_30.json')}
    assert len(by)==30 and len({r['business_id'] for r in rows})==15
    if not (execution/'COMPLETION_RECEIPT.json').exists():
        assert (execution/'FAILURE_RECEIPT.json').exists() or (root/'SETUP_FAILURE_RECEIPT.json').exists()
        print(json.dumps(dict(status='PRESERVED_TERMINAL_FAILURE_NOT_COMPLETE',complete=False,successful_outputs_not_claimed=True)));return
    receipt=read(execution/'COMPLETION_RECEIPT.json');pilot=read(execution/'COMPATIBILITY_PASS.json');outputs=read(execution/'QWEN_OUTPUTS_15.json');calls=[read(f) for f in (execution/'CALLS').glob('*.json')];ck={r['call_id']:r for r in calls}
    assert len(calls)==len(ck)==30 and set(ck)==set(by) and len(outputs)==15
    assert receipt['status']=='COMPLETE_30_NATIVE_CALLS' and receipt['components']==30 and receipt['restaurants']==15
    assert receipt['GPU_uuid']==identity['gpu_uuid'] and receipt['model']==spec['models']['Qwen3.5-9B']
    weights=read(execution/'WEIGHT_VERIFICATION.json');assert weights['status']=='PASS' and weights['model']==receipt['model']
    assert sha(execution/'QWEN_OUTPUTS_15.json')==receipt['outputs_sha256']
    assert pilot['restaurant']==plan['pilot_restaurant'] and pilot['pilot_outputs_reused']
    assert set(pilot['records'])=={r['call_id'] for r in rows if r['business_id']==pilot['restaurant']}
    expected_review_count=expected_captions=0
    for cid,r in ck.items():
        inp=by[cid];rr=rendered[cid]
        assert r['model_id']=='Qwen/Qwen3.5-9B' and r['model_revision']==spec['models']['Qwen3.5-9B']['revision']
        assert r['source_sha256']==inp['source_sha256']==tsha(inp['source'])
        assert r['input_ids']==rr['input_ids'] and ish(r['input_ids'])==r['input_ids_sha256']==rr['input_ids_sha256']
        assert tsha(r['text'])==r['text_sha256'] and r['precision']=='BF16' and r['enable_thinking'] is False
        assert r['selected_review_ids']==inp['selected_review_ids'] and r['selected_photo_ids']==inp['selected_photo_ids']
        expected_review_count+=len(inp['selected_review_ids']);expected_captions+=len(inp['selected_photo_ids'])
        assert r['input_tokens']==len(r['input_ids']) and r['output_tokens']==len(r['generation_token_ids'])
        assert r['generation_seconds']>0 and abs(r['tokens_per_second_including_prefill']-r['output_tokens']/r['generation_seconds'])<1e-12
        assert sha(execution/'CALLS'/(cid+'.json'))==receipt['native_call_hashes'][cid]
        assert r['generation_settings']['do_sample'] is False and r['generation_settings']['num_beams']==1
        assert r['generation_settings']['max_new_tokens']==spec['generation']['max_new_tokens_by_task'][r['task']]
        assert r['phase']==('pilot_generation' if cid in pilot['records'] else 'full_generation')
    assert expected_review_count==247 and expected_captions==73
    for r in outputs:
        rv=ck[r['business_id']+'__review'];vi=ck[r['business_id']+'__visual']
        assert r['review_text']==rv['text'] and r['visual_text']==vi['text'] and r['combined_text']==rv['text']+'\n\n'+vi['text']
        assert r['review_text_sha256']==rv['text_sha256'] and r['visual_text_sha256']==vi['text_sha256']
    decoded=False
    if args.decode_tokens:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(root/'MODEL_PROVENANCE/TOKENIZERS/Qwen3.5-9B',local_files_only=True,trust_remote_code=False)
        for r in calls:
            assert tok.decode(r['generation_token_ids'],skip_special_tokens=True)==r['text']
            assert tok.decode(r['generation_token_ids'],skip_special_tokens=False)==r['raw_decoded_with_special_tokens']
        decoded=True
    print(json.dumps(dict(status='PASS_COMPLETE_NATIVE_BINDINGS',restaurants=15,components=30,review_source_presentations=247,filtered_caption_presentations=73,pilot_reused_not_regenerated=True,verbatim_assembly=True,token_decode_verified=decoded,no_human_evaluation_started=True,semantic_quality_not_certified=True),indent=2))
if __name__=='__main__':main()
