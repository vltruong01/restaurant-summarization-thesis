"""Preserve2200 accepted inputs; apply the same frozen v3/v4/v4.1/v4.2 discovery to test inputs."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,os
for option in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(option,'1')
import yaml,orjson
from stage_f_verification_workflow import read,write,sha
from stage_f_d033_protocol import freeze_method
from source_budget_audit_template import review_block,text_sha,pinned_tokenizer,encoding
from token_length_cache import TokenLengthCache
from review_scoring_memoization import memoized_review_scoring

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_v1/review_inputs'
POOL=ROOT/'data/processed/yelp_2300_english_v1'
ACCEPTED=ROOT/'outputs/stage_c/led_source_token_allocation_audit_v1/selection_reference.jsonl'


def test_worker(args):
    bid,pop,info,identity=args
    from run_budget_review_pilot import run_business
    from run_length_balanced_review_pilot import worker as worker4
    from run_shared_pool_review_pilot import worker as worker41
    from run_source_length_tie_pilot import worker as worker42
    filename=hashlib.sha256(bid.encode()).hexdigest()+'.json'
    folders={v:OUT/f'test_witnesses/{v}/business_runs' for v in ['v3','v4','v41','v42']}
    for d in folders.values():d.mkdir(parents=True,exist_ok=True)
    cfg3=yaml.safe_load((ROOT/'configs/review_selection_expanded.yaml').read_text(encoding='utf-8'))
    cfg4=yaml.safe_load((ROOT/'configs/review_selection_length_balanced_pilot.yaml').read_text(encoding='utf-8'))
    cfg41=yaml.safe_load((ROOT/'configs/review_selection_shared_pool_pilot.yaml').read_text(encoding='utf-8'))
    cfg42=yaml.safe_load((ROOT/'configs/review_selection_source_length_expanded.yaml').read_text(encoding='utf-8'))
    # Only witness/cache destinations differ: every search/scoring/tie parameter is unchanged.
    cfg41['accepted_v4_directory']=folders['v4'].parent.relative_to(ROOT).as_posix()
    cfg42['accepted_v4_directory']=folders['v4'].parent.relative_to(ROOT).as_posix()
    cfg42['accepted_v4_1_directory']=folders['v41'].parent.relative_to(ROOT).as_posix()
    tok,_=pinned_tokenizer();tok=TokenLengthCache(tok)
    with memoized_review_scoring():
        for version in ['v3','v4','v41','v42']:
            path=folders[version]/filename
            if path.exists():
                payload=read(path);assert payload['production_identity']==identity
            else:
                if version=='v3':payload=run_business(bid,pop,cfg3,tok,info)
                elif version=='v4':payload=worker4((bid,pop,cfg4,info,str(folders['v3']/filename)))
                elif version=='v41':payload=worker41((bid,pop,cfg41,info,str(folders['v3']/filename)))
                else:payload=worker42((bid,pop,cfg42,info))
                payload['production_identity']=identity;write(path,payload)
    chosen=next(r for r in payload['runs'] if r['method']=='v4_2_length_balanced' and r['review_budget']==4000)
    assert chosen['selection_status']=='feasible','Frozen4000review selection infeasible; stop without changing method'
    selected=chosen['selected_review_ids'];byid={r['review_id']:r for r in pop}
    block=review_block([byid[rid] for rid in selected])
    return {'business_id':bid,'split':'test','review_budget':4000,'method':'v4_2_length_balanced',
            'selected_review_ids':selected,'review_block_sha256':text_sha(block),
            'review_tokens':len(tok.encode(block,add_special_tokens=True).ids),'production_identity':identity,
            'origin':'frozen_full_v3_v4_v41_v42_discovery_no_tuning_no_human_test_references'}


def run(workers=2):
    freeze_method();OUT.mkdir(parents=True,exist_ok=True)
    import csv
    cohort={r['business_id']:r['split'] for r in csv.DictReader((POOL/'business_split.csv').open(encoding='utf-8'))}
    assert len(cohort)==2300
    refs={}
    for line in ACCEPTED.open('rb'):
        r=orjson.loads(line)
        if r['review_budget']==4000:
            assert r['business_id'] not in refs and cohort[r['business_id']] in ['train','validation']
            refs[r['business_id']]={**r,'split':cohort[r['business_id']],'origin':'accepted_train_validation_reference_unchanged'}
    assert len(refs)==2200
    tests={bid:[] for bid,split in cohort.items() if split=='test'}
    for line in (POOL/'reviews/test.jsonl').open('rb'):
        r=orjson.loads(line);tests[r['business_id']].append(r)
    metadata={}
    for line in (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').open('rb'):
        r=orjson.loads(line)
        if r['business_id'] in tests:metadata[r['business_id']]=r
    settings=['review_selection_expanded.yaml','review_selection_length_balanced_pilot.yaml',
              'review_selection_shared_pool_pilot.yaml','review_selection_source_length_expanded.yaml']
    codes=['run_budget_review_pilot.py','run_length_balanced_review_pilot.py','run_shared_pool_review_pilot.py',
           'run_source_length_tie_pilot.py','budget_aware_review_selection.py','budget_aware_review_selection_v4.py',
           'budget_aware_review_selection_v4_1.py','budget_aware_review_selection_v4_2.py','review_selection_pilot.py',
           'token_length_cache.py','review_scoring_memoization.py','prepare_stage_f_production_reviews_d033.py']
    identity={'cohort_sha256':sha(POOL/'business_split.csv'),'test_input_review_sha256':sha(POOL/'reviews/test.jsonl'),
              'accepted_train_validation_sha256':sha(ACCEPTED),
              'config_sha256':{n:sha(ROOT/'configs'/n) for n in settings},'code_sha256':{n:sha(ROOT/'scripts'/n) for n in codes},
              'final_budget':4000,'no_search_parameters_tuned':True,'no_test_reference_or_teacher':True}
    stamp=text_sha(json.dumps(identity,sort_keys=True))
    receipt=OUT/'review_preparation_identity.json'
    if receipt.exists():assert read(receipt)==identity
    else:write(receipt,identity)
    cache=OUT/'test_final_references';cache.mkdir(exist_ok=True)
    pending=[]
    for bid,pop in sorted(tests.items()):
        assert len(pop)>=10
        path=cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json')
        if path.exists():
            r=read(path);assert r['production_identity']==stamp;refs[bid]=r
        else:
            count=len(pop);band='10-30' if count<=30 else '31-100' if count<=100 else '101-300' if count<=300 else '301+'
            info={'name':metadata[bid]['name'],'split':'test','volume_band':band,'sample_group':'production_test_fixed_inputs'}
            pending.append((bid,pop,info,stamp))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(test_worker,a):a[0] for a in pending}
        for f in as_completed(futures):
            bid=futures[f];r=f.result();assert r['review_tokens']<=4000
            write(cache/(hashlib.sha256(bid.encode()).hexdigest()+'.json'),r);refs[bid]=r
            print(f'Test input preprocessing {len(refs)-2200}/100 (no tuning)',flush=True)
    assert set(refs)==set(cohort)
    path=OUT/'shared_review_references.json'
    result=[refs[bid] for bid in sorted(refs)]
    if path.exists():assert read(path)==result
    else:write(path,result)
    write(OUT/'review_reference_report.json',{'status':'COMPLETE_FIXED_SHARED_INPUT_REFERENCES','businesses':2300,
          'accepted2200_unchanged':True,'test100_fixed_method_only':True,'test_method_tuning':False,
          'human_test_references':False,'selected_budget':4000,'reference_sha256':sha(path),'identity':stamp})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(a.workers)
