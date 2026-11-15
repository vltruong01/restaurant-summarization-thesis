"""Bounded two-arm adapter over the accepted V2 runner. Default is CPU preflight.

Original parent code stays byte-identical. All phase invocations need fresh approval.
"""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from led_large_boundary_global_v1 import ARMS, POLICY, MaskIndex, install_collator, require

R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/led_large_boundary_global_v1'
CONFIG=R/'configs/stage_i_led_large_boundary_global_v1.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')


def binding(arm):
    require(arm in ARMS,'Unknown arm')
    return dict(config_sha256=sha(CONFIG),runner_sha256=sha(Path(__file__)),
        preflight_manifest_sha256=sha(P/'PREFLIGHT_MANIFEST.json'),attention_arm=arm,
        runtime_config_sha256=sha(P/f'RUNTIME_CONFIG_{arm}.json'),
        anchor_manifest_sha256=sha(P/'ANCHOR_ROWS.jsonl'))


def check_time(a,now=None):
    values=[datetime.fromisoformat(a[k]) for k in ('execution_authorized_at_utc','execution_deadline_utc','researcher_absolute_stop_utc')]
    start,end,absolute=values
    require(all(x.tzinfo is not None and x.utcoffset().total_seconds()==0 for x in values),'Explicit actual UTC clocks required')
    require(start <= (now or datetime.now(timezone.utc)) < end <= min(start+timedelta(hours=48),absolute), 'Expired/invalid48h scope')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True,'Explicit evaluation reserve confirmation required')


def authorize(path,execute,mode):
    require(execute and path is not None,'OFFLINE ONLY: no explicit GPU approval')
    a=read(path)
    require(a.get('experiment')=='LED_LARGE_BOUNDARY_GLOBAL_V1' and a.get('GPU_execution_approved') is True,
            'Current preparation is not GPU approval')
    require(a.get('approval_source')=='researcher_message' and bool(a.get('researcher_explicit_authorization')),'Researcher receipt required')
    require(a.get('approved_arms')==list(ARMS) and mode in a.get('scopes',[]),'Wrong arm/phase scope')
    require(a.get('bindings')=={arm:binding(arm) for arm in ARMS},'Exact authorization bindings differ')
    spec=read(CONFIG)
    expected=dict(max_training_updates=1250,evaluation_outputs_max=48,early_sanity_outputs_max=4,
        early_sanity_generation_approved=True,technical_generation_calls_max=2,all_generation_calls_max=54,
        feasibility_updates_max=3,feasibility_microbatches_max=88,fresh_only=True,held_out_access=False,
        Teacher_calls=False,RO_U_training_authorized=False,automatic_retry=False,
        conditional_train_after_both_complete_gates_pass=True,
        full_cache_stress_approved=True,minimum_memory_headroom=0.15,
        amp_scaler=spec['shared']['amp_scaler'],dynamic_overflow_policy=spec['shared']['dynamic_overflow_policy'])
    require(all(a.get(k)==v for k,v in expected.items()),'Budget/numerical/forbidden scope differs')
    if mode=='train':require(a.get('training_authorized') is True,'Gate approval alone does not grant training')
    check_time(a)
    return dict(a,_approval_sha256=sha(path))


def validate_runtime_configs(spec,parent_config):
    require(sha(parent_config)==spec['parent_config_sha256'],'Original V2 config changed')
    old=read(parent_config)
    require(spec['shared']=={k:old[k] for k in spec['shared']},'Shared scientific settings changed')
    configs={}
    for arm in ARMS:
        new=read(P/f'RUNTIME_CONFIG_{arm}.json')
        expected=dict(old,experiment=spec['experiment'],run_root=spec['run_root']+'/'+arm,
            global_attention=old['global_attention'] if arm=='bos_only' else POLICY,
            vast_fresh=dict(old['vast_fresh'],experiment=spec['experiment'],output=spec['run_root']+'/'+arm))
        require(new==expected,'Unrequested arm configuration difference')
        configs[arm]=new
    return configs


def preflight(parent):
    spec=read(CONFIG)
    require(spec['preparation_only'] and not spec['GPU_execution_approved'],'Offline specification must not self-authorize')
    for rel,digest in read(P/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(R/rel)==digest,'New package binding differs: '+rel)
    require(parent.is_dir() and (parent/'VAST_SHA256_MANIFEST.json').is_file(),'Complete accepted parent workspace required')
    configs=validate_runtime_configs(spec,parent/'configs/stage_i_led_large_decoder_policy_v2.json')
    sys.path.insert(0,str(parent/'scripts'))
    import stage_i_led_large_decoder_policy_v2 as h
    import run_stage_i_led_large_decoder_policy_v2 as b
    require(h.ROOT.resolve()==parent.resolve() and b.ROOT.resolve()==parent.resolve(), 'Foreign already-imported parent modules; use fresh process')
    h.preflight()  # Full accepted V1/V2 byte chain and runtime source checks; no model loads.
    require(h.sha(h.CONFIG)==spec['parent_config_sha256'],'Unexpected parent config')
    for rel,digest in read(P/'FROZEN_INPUT_BINDINGS.json')['files'].items():
        require(sha(parent/rel)==digest,'Original train/validation source differs')
    rows=[json.loads(s) for s in (P/'ANCHOR_ROWS.jsonl').read_text(encoding='utf-8').splitlines()]
    index=MaskIndex(rows)
    require(len(rows)==2200 and read(P/'MASK_VALIDATION.json')['observed_max_globals']==36,'Mask population drift')
    return spec,configs,index,h,b


@contextmanager
def route_parent(h,b,index,arm,observer):
    """Reuse ALL native CE/AMP/checkpoint/sanity code; inject mask in its shared collator."""
    from unittest.mock import patch
    import led_large_decoder_label_view_v1 as labels
    old_read=b.read
    stress=read(P/'STRESS_CASES.json')['train_business_ids']
    def gated_read(path):
        if Path(path)==b.LARGE_PREP/'FEASIBILITY_CASES.json':return dict(business_ids=stress)
        return old_read(path)
    with ExitStack() as stack:
        stack.enter_context(install_collator(labels,index,arm,observer))
        stack.enter_context(patch.object(b,'read',gated_read))
        stack.enter_context(patch.object(h,'CONFIG',P/f'RUNTIME_CONFIG_{arm}.json'))
        for module in (h,b):
            stack.enter_context(patch.object(module,'binding',lambda:binding(arm)))
            stack.enter_context(patch.object(module,'check_deadline',check_time))
        yield


def cache_stress(c,a,out,b):
    """Fresh discarded model. Beam4 full768 cache geometry, no generate/optimizer calls."""
    import torch
    import gc
    from transformers.modeling_outputs import BaseModelOutput
    check_time(a)
    # The previous discarded gate model must not coexist with this fresh resource model.
    gc.collect();torch.cuda.empty_cache()
    train,_=b.data(c)
    bid=read(P/'STRESS_CASES.json')['train_business_ids'][0]
    feature=train[train.business_ids.index(bid)]
    model,before,scope=b.load_model(c)
    model.eval()
    pack=b.collator(stress=True)
    with b.MemoryMeter(torch) as memory,torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        batch={k:v.cuda() for k,v in pack([feature]).items() if k!='labels'}
        global_count=int(batch['global_attention_mask'].sum().item())
        require(global_count==(36 if b.binding()['attention_arm']=='boundary' else 1),'Maximum-global stress case missing')
        enc=model.get_encoder()(**batch,return_dict=True)
        repeated=BaseModelOutput(last_hidden_state=enc.last_hidden_state.repeat_interleave(4,dim=0))
        attention=batch['attention_mask'].repeat_interleave(4,dim=0)
        # Synthetic technical prefix using only TRAIN label tokens; not a summary or model supervision.
        body=feature['labels'][:-1]
        prefix=([2]+(body*((766+len(body)-1)//len(body)))[:766])
        decoder=torch.tensor([prefix]*4,device='cuda',dtype=torch.long)
        value=model(encoder_outputs=repeated,attention_mask=attention,decoder_input_ids=decoder,use_cache=True,return_dict=True)
        require(torch.isfinite(value.logits).all().item(),'Nonfinite full-cache prefill logits')
        require(value.past_key_values[0][0].shape[-2]==767,'Wrong decoder cache geometry')
        cached=value.past_key_values
        del value
        value=model(encoder_outputs=repeated,attention_mask=attention,decoder_input_ids=decoder[:,-1:],
            past_key_values=cached,use_cache=True,return_dict=True)
        require(torch.isfinite(value.logits).all().item() and value.past_key_values[0][0].shape[-2]==768,
                'Nonfinite or wrong final cache step')
        memory.check();check_time(a)
    b.verify_backbone(model,before)
    write(out/'RESOURCE_GATE.json',dict(status='PASS',binding=b.binding(),approval_sha256=a['_approval_sha256'],
        source_business_id=bid,initialization=scope['initialization'],source_shape=[1,5120],global_count=global_count,
        full_cache_batch=4,cache_tokens=768,new_generate_calls=0,optimizer_updates=0,
        fresh_model_discarded=True,**memory.summary()))


def require_both_gates(suite,a,h):
    reports=[]
    for arm in ARMS:
        out=suite/arm
        with __import__('unittest.mock',fromlist=['patch']).patch.object(h,'binding',lambda arm=arm:binding(arm)):
            r=h.require_gate(out,a)
        resource=read(out/'RESOURCE_GATE.json')
        require(resource['status']=='PASS' and resource['binding']==binding(arm)
            and resource['approval_sha256']==a['_approval_sha256'] and resource['headroom_fraction']>=.15,
            'Both complete resource gates required')
        reports.append(r)
    require(reports[0]['hardware_profile']==reports[1]['hardware_profile'],'Matched GPU runtime required')
    init=[read(suite/arm/'feasibility/INITIAL_MODEL_BINDINGS.json')['initialization'] for arm in ARMS]
    require(init[0]==init[1],'Fresh arm initialization differs')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('preflight','feasibility','train','evaluate'))
    ap.add_argument('--arm',choices=ARMS,default='bos_only')
    ap.add_argument('--parent-root',type=Path,default=R)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--approval',type=Path)
    args=ap.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # Deny unapproved GPU phases BEFORE importing the old runner or touching CUDA/model code.
    a=None if args.mode=='preflight' else authorize(args.approval,args.execute,args.mode)
    parent=args.parent_root.resolve()
    spec,configs,index,h,b=preflight(parent)
    if args.mode=='preflight':print('OFFLINE_PARENT_AND_MASK_PREFLIGHT_PASS',binding(args.arm));return
    suite=parent/spec['run_root'];out=suite/args.arm
    require(not (suite/'STOP_FAILURE.json').exists(),'Previous failure; no automatic retry/other arm continuation')
    require(not (suite/'QUALITY_REVIEW_COMPLETE.json').exists(),'Experiment already closed')
    if args.mode=='feasibility':require(not out.exists(),'Fresh arm namespace required')
    if args.mode=='feasibility' and args.arm=='boundary':
        require((suite/'bos_only/RESOURCE_GATE.json').is_file(),'Fixed BOS-first gate order')
    if args.mode=='train':
        require_both_gates(suite,a,h)
        if args.arm=='boundary':require((suite/'bos_only/train/COMPLETION.json').is_file(),'Fixed BOS-first execution order')
    if args.mode=='evaluate':
        require(all((suite/x/'train/COMPLETION.json').is_file() for x in ARMS),'No final comparison before both complete')
        if args.arm=='boundary':require((suite/'bos_only/evaluate/COMPLETION.json').is_file(),'Fixed BOS-first evaluation order')
    receipt=suite/'receipts'/f'{args.arm}_{args.mode}_START.json'
    write(receipt,dict(binding=binding(args.arm),authorization_sha256=a['_approval_sha256'],actual_utc=datetime.now(timezone.utc).isoformat()))
    log=receipt.with_name(f'{args.arm}_{args.mode}_MASKS.jsonl')
    def observe(row):
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    try:
        with route_parent(h,b,index,args.arm,observe):
            c=configs[args.arm]
            if args.mode=='feasibility':b.feasible(c,a,out);cache_stress(c,a,out,b)
            elif args.mode=='train':b.train(c,a,out)
            else:b.evaluate(c,a,out)
        write(receipt.with_name(f'{args.arm}_{args.mode}_COMPLETE.json'),dict(status='PASS',binding=binding(args.arm)))
    except BaseException as e:
        write(suite/'STOP_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':main()
