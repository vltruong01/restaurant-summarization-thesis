"""CPU-only audit of exported V3 evidence; no inference or training."""
from pathlib import Path
import collections,hashlib,json,math,os,re,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1'
DATA=OUT/'downloaded'
RUN=DATA/'runs/CF_EXPLORATORY_NATIVE_CE_V3'
sys.path.insert(0,str(ROOT/'scripts'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def metrics(ids,text):
    sentences=[re.sub(r'\s+',' ',s).strip().lower() for s in re.split(r'(?<=[.!?])\s+',text) if s.strip()]
    counts=collections.Counter(sentences);run=0;longest=0
    for token in ids[1:]:
        run=run+1 if token==0 else 0;longest=max(longest,run)
    review,sep,visual=text.partition('[VISUAL_EVIDENCE]')
    return dict(raw_token_count=len(ids),words=len(text.split()),nonempty=bool(text.strip()),
        terminal_EOS=bool(ids and ids[-1]==2),BOS_after_start=ids[1:].count(0),longest_BOS_run=longest,
        exact_duplicate_sentences=[dict(sentence=s,count=n) for s,n in counts.items() if n>1],
        review_body=review.replace('[REVIEW_CONSENSUS]','').strip(),visual_body=visual.strip() if sep else None,
        semantic_quality='MODEL_ASSISTED_REVIEW_REQUIRED_NOT_AUTOMATIC_GO')


def main():
    import torch
    from safetensors.torch import load_file
    from transformers.models.led.modeling_led import shift_tokens_right
    import run_led_large_exploratory_contrastive_v2 as old
    c=read(ROOT/'outputs/stage_i/led_large_exploratory_native_ce_v3/CONFIG.json')
    completion=read(RUN/'COMPLETION.json')
    assert completion['successful_updates']=={'ce':16,'pair':16} and completion['generation_calls']==12
    assert completion['historical_equivalence']=='FAIL_UNRESOLVED'
    assert not (RUN/'FAILURE_STOP_NO_RETRY.json').exists()
    assert read(DATA/'operations/POSTFLIGHT_PRESERVATION.json')['status']=='PASS'
    assert read(OUT/'DOWNLOAD_VERIFIED.json')['manifest_PASS']
    safety=read(RUN/'live_safety/SAFETY_CHECKS_PASS.json');assert safety['successful_optimizer_updates']==0
    expected=old.collated(c,'cpu')
    original=load_file(str(ROOT/c['adapter']),device='cpu')
    checkpoints=[];arms={};initials=[]
    for arm in ('ce','pair'):
        p=RUN/('train_'+arm);progress=rows(p/'progress.jsonl');updates=rows(p/'optimizer_updates.jsonl');grads=rows(p/'gradient_checks.jsonl')
        assert len(progress)==len(updates)==len(grads)==16
        assert [r['step'] for r in progress]==list(range(1,17))
        assert [r['successful_optimizer_updates'] for r in updates]==list(range(1,17))
        assert all(r['optimizer_state_tensors']==456 and r['optimizer_state_steps']==[r['step']] and r['scaler']['scale']==65536 and r['scaler']['_growth_tracker']==r['step'] for r in progress)
        assert all(len(r['gradients'])==456 and all(g['finite'] and math.isfinite(g['norm']) and math.isfinite(g['max_abs']) for g in r['gradients']) for r in grads)
        assert len(rows(p/'microbatches.jsonl'))==128
        prefix_rows=rows(p/'decoder_prefixes.jsonl');assert len(prefix_rows)==16
        for window,record in enumerate(prefix_rows):
            expected_prefixes=[]
            for batch in expected[window*8:(window+1)*8]:
                for kind in (('positive',) if arm=='ce' else ('positive','negative')):
                    shifted=shift_tokens_right(batch[kind]['labels'],1,2)
                    expected_prefixes.append(dict(kind=kind,shape=list(shifted.shape),sha256=hashlib.sha256(shifted.numpy().tobytes()).hexdigest(),own_prefix=True))
            assert record['receipts']==expected_prefixes
        init=read(p/'INITIALIZATION.json');initials.append(init['adapter_fingerprint'])
        assert init['source_adapter_sha256']==c['adapter_sha256'] and init['optimizer_state_empty']
        for step in (8,16):
            cp=p/f'checkpoint-{step}';proof=read(cp/'checkpoint_integrity.json')
            for name,digest in proof['files'].items():assert sha(cp/name)==digest
            opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True)
            scaler=torch.load(cp/'scaler.pt',map_location='cpu',weights_only=True)
            sched=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=True)
            assert len(opt['state'])==456 and {int(v['step']) for v in opt['state'].values()}=={step}
            assert scaler['scale']==65536 and scaler['_growth_tracker']==step and sched['last_epoch']==step
            state=load_file(str(cp/'adapter_model.safetensors'),device='cpu')
            assert set(state)==set(original) and len(state)==456
            assert all(torch.isfinite(t).all() and t.shape==original[name].shape for name,t in state.items())
            changed=sum(not torch.equal(state[n],original[n]) for n in state)
            assert changed>0
            checkpoints.append(dict(arm=arm,step=step,adapter_sha256=sha(cp/'adapter_model.safetensors'),
                receipt_sha256=sha(cp/'checkpoint_integrity.json'),optimizer_state_tensors=456,
                optimizer_step=step,scaler_scale=scaler['scale'],scheduler_epoch=sched['last_epoch'],
                changed_tensors_from_CE3=changed,full_files=proof['files']))
            del opt,state
        safe=read(RUN/'live_safety'/arm/'SAFETY.json')
        assert safe['successful_optimizer_updates']==0 and safe['optimizer_state_empty'] and safe['adapter_unchanged']
        assert len(safe['scaled_gradients'])==len(safe['unscaled_gradients'])==456
        assert all(r['finite'] for r in safe['scaled_gradients']+safe['unscaled_gradients'])
        arms[arm]=dict(successful_updates=16,positive_exposures=128,steps_with_nonzero_learning_rate=sum(r['lr_before'][0]>0 for r in progress),
            first_mean_CE=progress[0]['mean_CE'],last_mean_CE=progress[-1]['mean_CE'],first_mean_total=progress[0]['mean_total_loss'],last_mean_total=progress[-1]['mean_total_loss'],
            max_unclipped_gradient_norm=max(r['global_unclipped_gradient_norm'] for r in progress),
            minimum_measured_headroom=min(r['memory']['headroom_fraction'] for r in progress),
            all_gradients_finite=True,all_optimizer_steps_verified=True,scaler_backoffs=0,
            first_update_UTC=updates[0]['utc'],last_update_UTC=updates[-1]['utc'])
    assert initials[0]==initials[1]
    calls=rows(RUN/'generation_calls.jsonl');assert [r['call'] for r in calls]==list(range(1,13))
    sources={x['business_id']:x for x in read(OUT/'LOCKED_TRAIN_PROBE_SOURCES.json')}
    all_cases=[];transcript=['# All twelve TRAIN-probe summaries','',
        'Exploratory post-failure deviation; historical equivalence FAIL remains unresolved.','']
    for bid in c['probe']:
        source=sources[bid];name=source['source'].split('name=',1)[1].split('\n')[0]
        case=dict(business_id=bid,name=name,split='train',filtered_caption_count=len(source['captions']),source_sha256=source['source_sha256'],outputs={})
        transcript.extend(['## '+name,'','Business ID: `'+bid+'`; Filtered captions: '+str(len(source['captions'])), ''])
        for variant in ('original_ce3','ce16','pair16'):
            folder=RUN/('evaluate_'+variant);r=read(folder/'generations'/f'{bid}.json');rng=read(folder/'generations'/f'{bid}_RNG.json')
            assert r['source_sha256']==source['source_sha256'] and r['technical']['status']=='PASS_TECHNICAL_ONLY' and rng['status']=='PASS'
            m=metrics(r['token_ids'],r['text']);assert m['nonempty']
            case['outputs'][variant]=dict(text=r['text'],token_ids=r['token_ids'],seconds=r['seconds'],metrics=m,
                source_binding=r['source_sha256'],adapter_binding=read(folder/'MODEL_BINDING.json'))
            transcript.extend(['### '+variant,'',r['text'],''])
        all_cases.append(case)
    write(OUT/'INDEPENDENT_CPU_EXECUTION_AUDIT.json',dict(status='EXPORTED_RUN_CONSISTENCY_PASS_NOT_NUMERICAL_EQUIVALENCE',
        historical_equivalence='FAIL_UNRESOLVED',matched_initial_adapter_fingerprint=initials[0],arms=arms,
        checkpoints=checkpoints,live_safety_optimizer_updates=0,decoding_calls=12,source_only_train_probes=4,
        validation_or_heldout_used=False,model_inference_in_this_audit=False))
    write(OUT/'ALL_PROBE_OUTPUTS_AND_METRICS.json',all_cases)
    (OUT/'ALL_PROBE_OUTPUTS.md').write_text('\n'.join(transcript)+'\n',encoding='utf-8')
    print(json.dumps(dict(arms=arms,generation_calls=12,case_names=[x['name'] for x in all_cases]),indent=2))


if __name__=='__main__':main()
