"""Versioned post-adjudication amendment; preserves all V1 code/data and receipts."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2'
OLD=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
RUNNER=ROOT/'scripts/run_led_large_exploratory_contrastive_v2.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def replace_function(source,name,body):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+body.strip()+'\n'+''.join(lines[node.end_lineno:])

PREFLIGHT='''
def preflight():
    c=read(CONFIG)
    for rel,digest in c['files'].items():require(sha(ROOT/rel)==digest,'Bound file changed: '+rel)
    for dist,expected in c['runtime_pins'].items():require(importlib.metadata.version(dist)==expected,'Runtime differs: '+dist)
    require(c['objective']==dict(pair_weight=.1,margin=.1,temperature=1.),'Objective drift')
    require(c['pilot_updates']==16 and c['accumulation']==8,'Budget drift')
    require(c['pair_ids']==[f'P{i:02d}' for i in range(1,17) if i!=3],'Locked15 pair drift')
    require(c['excluded_pairs']==['P03','P17','P18'],'Excluded pairs drift')
    require(len(c['fit'])==8 and len(c['probe'])==4 and not set(c['fit'])&set(c['probe']),'TRAIN roster drift')
    cp=ROOT/c['adapter'];proof=read(cp.parent/'checkpoint_integrity.json')
    for name,h in proof['files'].items():require(Path(name).name==name and sha(cp.parent/name)==h,'CE3 component changed')
    require(sha(cp)==c['adapter_sha256'],'Foreign adapter')
    counts={}
    for bid in c['fit']:
        p=read(PREP/'positives'/f'{bid}.json')
        require(p['split']=='train' and p['condition']=='filtered','Foreign data')
        require(p['corrected_labels']==p['original_labels'][1:] and p['original_labels'][0]==0 and p['corrected_labels'][-1]==2,'Decoder labels differ')
    for pid in c['pair_ids']:
        p=read(PREP/'pairs'/f'{pid}.json');counts[p['category']]=counts.get(p['category'],0)+1
    require(counts==dict(core_omission=5,same_aspect_disagreement_loss=6,grounding_error=4),'Quotas drift')
    return c
'''

AUTH='''
def authorization(args,c):
    require(args.execute and args.approval,'Explicit V2 authorization required before CUDA')
    a=read(args.approval)
    require(a['experiment']=='EXPLORATORY_CONTRASTIVE_PILOT_V2_POST_ADJUDICATION','Foreign approval')
    require(a['GPU_execution_approved'] is True and a['approval_source']=='researcher_message','Approval absent')
    require(a['post_adjudication_amendment'] is True and a['not_preregistered'] is True,'Amendment disclosure absent')
    for key,path in [('config_sha256',CONFIG),('runner_sha256',Path(__file__)),('objective_sha256',ROOT/'scripts/led_large_contrastive_sequential_v1.py'),('preflight_sha256',CONFIG.parent/'PREFLIGHT_MANIFEST.json')]:
        require(a[key]==sha(path),'Approval hash differs: '+key)
    require(args.mode in a['scopes'] and args.arm in ('ce','pair'),'Unapproved mode')
    start=datetime.fromisoformat(a['authorized_utc']);end=datetime.fromisoformat(a['deadline_utc'])
    require(start.tzinfo and end.tzinfo and timedelta(0)<end-start<=timedelta(hours=48) and start<=datetime.now(timezone.utc)<end,'Expired authorization')
    require(a['updates_per_arm_max']==16 and a['generation_calls_max']==12 and a['diagnostic_attempts_max']==1,'Scope drift')
    require(args.decisions and sha(args.decisions)==a['researcher_decisions_sha256'],'Adjudication binding differs')
    d=read(args.decisions)
    require(d['provenance']=='Explicit current researcher message; historical PENDING files preserved','Foreign adjudication provenance')
    require(set(d['positives'])==set(c['fit']) and all(x['decision']=='ACCEPT' for x in d['positives'].values()),'Eight accepted positives required')
    require(set(d['pairs'])==set(c['pair_ids']) and all(x['decision']=='ACCEPT' for x in d['pairs'].values()),'Exactly15 accepted locked pairs required')
    require(d['excluded_pairs']==['P03','P17','P18'] and d['P03_decision']=='REJECT','Exclusions differ')
    for bid,v in d['positives'].items():require(v['sha256']==sha(PREP/'positives'/f'{bid}.json'),'Positive decision hash differs')
    for pid,v in d['pairs'].items():require(v['sha256']==sha(PREP/'pairs'/f'{pid}.json'),'Pair decision hash differs')
    if args.mode!='diagnose':
        require(args.numerical_receipt is not None,'Diagnostic receipt required')
        expected=ROOT/c['execution_root']/'diagnose_pair/COMPLETION.json'
        require(args.numerical_receipt.resolve()==expected.resolve() and expected.is_file(),'Foreign diagnostic path')
        nr=read(expected)
        require(nr['status']=='PASS' and nr['full_pretrained_CUDA_FP16'] is True,'Mandatory diagnostic did not pass')
        require(not (expected.parent/'FAILURE_STOP_NO_RETRY.json').exists(),'Diagnostic failed; no training')
    return a
'''

DIAGNOSE='''
def diagnose(model,c,folder,memory,check):
    import torch
    import led_large_contrastive_sequential_v1 as seq
    from run_stage_i_d050_feasibility_v1 import fingerprint
    params=tuple(p for p in model.parameters() if p.requires_grad)
    batches=collated(c,'cuda');sample=batches[1] # Frozen Tempo pair P02, source4021 -> padded4096; both own prefixes.
    positive,negative=sample['positive'],sample['negative'];device=params[0].device
    def event(stage,**fields):
        append(folder/'NUMERICAL_STAGES.jsonl',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**fields))
        print('DIAGNOSTIC_STAGE',stage,flush=True)
    def gradstats(grads):
        return [dict(index=i,shape=list(g.shape),finite=bool(torch.isfinite(g).all()),max_abs=float(torch.nan_to_num(g,nan=0.,posinf=0.,neginf=0.).abs().max())) for i,g in enumerate(grads)]
    modes=seq._mode_signature(model)
    weights={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    event('START',intended_accumulation=8,physical_batch=1,AMP_init_scale=65536,comparison_atol=1e-5,comparison_rtol=1e-4)
    initial=seq.capture_rng(device)
    with torch.autocast('cuda',dtype=torch.float16):native=model(**positive,use_cache=False).loss
    ga=torch.autograd.grad(native,params);seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):delegated=seq.loss(model,positive,negative,weight=0.)
    gb=torch.autograd.grad(delegated,params)
    equal=torch.equal(native,delegated) and all(torch.equal(x,y) for x,y in zip(ga,gb))
    event('LAMBDA0_NATIVE_CE',loss=float(native),delegated_loss=float(delegated),bit_exact=equal,gradients=gradstats(gb))
    require(equal,'Lambda0 native CE loss/gradient mismatch')
    require(all(torch.isfinite(g).all().item() for g in ga+gb),'Lambda0 nonfinite gradients')
    del native,delegated,ga,gb
    check();memory.check();initial=seq.capture_rng(device)
    with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
        sp=-model(**positive,use_cache=False).loss.float();sn=-model(**negative,use_cache=False).loss.float()
        joint=-sp+.1*torch.nn.functional.softplus(.1-sp+sn)
    joint_value=float(joint.detach());reference=torch.autograd.grad(joint,params)
    expected_rng=seq.capture_rng(device);del sp,sn,joint
    seq.restore_rng(initial,device)
    with torch.autocast('cuda',dtype=torch.float16):streamed=seq.loss(model,positive,negative,recorder=lambda x:append(folder/'rng.jsonl',x))
    streamed_value=float(streamed.detach());got=torch.autograd.grad(streamed,params);del streamed
    finite=all(torch.isfinite(x).all().item() for x in reference+got)
    comparison=[dict(index=i,max_abs_error=float((x-y).abs().max()),allclose=bool(torch.allclose(x,y,atol=1e-5,rtol=1e-4))) for i,(x,y) in enumerate(zip(reference,got))]
    rng_ok=seq.same_rng(seq.capture_rng(device),expected_rng)
    event('JOINT_VS_SEQUENTIAL',joint_loss=joint_value,sequential_loss=streamed_value,finite=finite,RNG_equal=rng_ok,per_tensor=comparison)
    require(finite and all(x['allclose'] for x in comparison),'Full LED joint/sequential gradient equivalence failed')
    require(abs(joint_value-streamed_value)<=1e-5+1e-4*abs(joint_value) and rng_ok,'Loss/RNG equivalence failed')
    del reference,got
    optimizer=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    opt_before=fingerprint(optimizer.state_dict())
    accumulation_initial=seq.capture_rng(device)
    expected=[torch.zeros_like(p) for p in params]
    # Explicit sum of eight per-example gradients is the accumulation reference.
    for i,item in enumerate(batches[:8]):
        check()
        with torch.autocast('cuda',dtype=torch.float16):loss=seq.loss(model,**item,recorder=lambda x:append(folder/'rng.jsonl',x))
        grads=torch.autograd.grad(65536*(loss/8),params)
        for x,g in zip(expected,grads):x.add_(g)
        event('MANUAL_ACCUMULATION_MICROBATCH',index=i,loss=float(loss.detach()),all_finite=all(torch.isfinite(g).all().item() for g in grads))
        del grads,loss;memory.check()
    reference_rng=seq.capture_rng(device)
    event('MANUAL_ACCUMULATION_REFERENCE',gradients=gradstats(expected))
    require(all(torch.isfinite(g).all().item() for g in expected),'FP16 scaled accumulation nonfinite; stop without retry or changing scale')
    seq.restore_rng(accumulation_initial,device)
    scaler=torch.cuda.amp.GradScaler(init_scale=65536,growth_factor=2.,backoff_factor=.5,growth_interval=2000)
    for i,item in enumerate(batches[:8]):
        check()
        with torch.autocast('cuda',dtype=torch.float16):loss=seq.loss(model,**item,recorder=lambda x:append(folder/'rng.jsonl',x))
        scaler.scale(loss/8).backward();del loss;memory.check()
    require(all(p.grad is not None for p in params),'Missing native accumulated gradients')
    accumulated_equal=all(torch.equal(p.grad,g) for p,g in zip(params,expected))
    require(accumulated_equal and seq.same_rng(seq.capture_rng(device),reference_rng),'Native8 accumulation/RNG mismatch')
    scaler.unscale_(optimizer)
    event('NATIVE_FP16_ACCUMULATION',exact_scaled_sum=accumulated_equal,gradients=gradstats([p.grad for p in params]),memory=memory.summary())
    require(all(torch.isfinite(p.grad).all().item() for p in params),'Nonfinite unscaled gradients')
    scaler.update();optimizer.zero_grad(set_to_none=True);del expected
    # Native nonfinite skip must leave AdamW state and all adapter weights unchanged.
    skip=torch.cuda.amp.GradScaler(init_scale=65536,growth_interval=2000);skip.scale(torch.ones((),device=device))
    for p in params:p.grad=torch.zeros_like(p)
    params[0].grad.fill_(float('inf'));skip.step(optimizer);skip.update();optimizer.zero_grad(set_to_none=True)
    unchanged=all(torch.equal(p.detach().cpu(),weights[n]) for n,p in model.named_parameters() if p.requires_grad)
    require(skip.get_scale()==32768 and fingerprint(optimizer.state_dict())==opt_before and unchanged,'Overflow skip/optimizer integrity failed')
    require(seq._mode_signature(model)==modes,'Dropout/cache/adapter mode changed')
    check();memory.check()
    event('ALL_MANDATORY_DIAGNOSTICS_PASS',optimizer_unchanged=True,adapter_unchanged=unchanged,memory=memory.summary())
    return dict(status='PASS',full_pretrained_CUDA_FP16=True,lambda0_exact=True,joint_vs_sequential=comparison,
        native_eight_example_accumulation_exact=True,injected_overflow_skip_verified=True,successful_optimizer_updates=0,
        generation_calls=0,all_modes_unchanged=True,memory=memory.summary())
'''

GENERATE='''
def generate_probes(model,c,folder,memory,check,trainer=None):
    import torch
    from contextlib import contextmanager
    from peft import get_peft_model_state_dict
    from led_large_decoder_early_sanity_v2 import preserve_training_state,sanity_result
    from led_large_dynamic_amp_v2 import capture_rng,restore_rng
    from led_large_nbest_rng_v2 import rng_report
    from run_stage_i_d050_feasibility_v1 import fingerprint
    from transformers import AutoTokenizer,GenerationConfig
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    @contextmanager
    def baseline_guard():
        rng=capture_rng();modes={n:m.training for n,m in model.named_modules()};cache=model.config.use_cache
        state=fingerprint(get_peft_model_state_dict(model))
        try:yield
        finally:
            for n,m in model.named_modules():m.training=modes[n]
            model.config.use_cache=cache;restore_rng(rng)
            require(fingerprint(get_peft_model_state_dict(model))==state,'Untouched CE3 adapter changed')
            require(fingerprint(capture_rng())==fingerprint(rng),'Baseline RNG restore failed')
    source={x['business_id']:x for x in read(ROOT/c['TRAIN_source_packets'])}
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
    g=GenerationConfig(**read(ROOT/c['effective_generation_config']))
    results=[];continuity={}
    try:
        with (preserve_training_state(model,trainer,continuity) if trainer is not None else baseline_guard()):
            model.eval();require(all(not m.training for m in model.modules()),'Eval mode differs')
            for bid in c['probe']:
                check();case=source[bid];require(case['split']=='train','Foreign probe split')
                batch=CorrectedLabelCollator()([dict(input_ids=case['input_ids'],labels=[5,2])])
                batch={k:v.cuda() for k,v in batch.items() if k!='labels'}
                before=capture_rng();start=time.perf_counter()
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,generation_config=g)[0].cpu().tolist()
                    after=capture_rng();text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    record=dict(business_id=bid,token_ids=ids,text=text,seconds=time.perf_counter()-start,technical=sanity_result(ids,text,g.max_length),source_sha256=case['source_sha256'])
                    write(folder/'generations'/f'{bid}.json',record)
                finally:restore_rng(before)
                receipt=rng_report(before,after,capture_rng(),torch,fingerprint)
                write(folder/'generations'/f'{bid}_RNG.json',receipt)
                require(receipt['status']=='PASS' and record['technical']['status']=='PASS_TECHNICAL_ONLY','Generation/RNG failure')
                results.append(record);memory.check()
    finally:write(folder/'STATE_CONTINUITY.json',continuity if trainer is not None else dict(baseline_adapter_and_RNG_guard=True))
    return results
'''

def main():
    assert not OUT.exists() and not RUNNER.exists(),'New version exists; preserve evidence.'
    OUT.mkdir(parents=True)
    old=read(OLD/'PRODUCTION_CONFIG_NOT_APPROVED.json')
    accepted=[f'P{i:02d}' for i in range(1,17) if i!=3]
    decisions=dict(provenance='Explicit current researcher message; historical PENDING files preserved',
        positives={bid:dict(decision='ACCEPT',sha256=sha(OLD/'positives'/f'{bid}.json')) for bid in old['fit']},
        pairs={pid:dict(decision='ACCEPT',sha256=sha(OLD/'pairs'/f'{pid}.json')) for pid in accepted},
        excluded_pairs=['P03','P17','P18'],P03_decision='REJECT',P03_rejection_reason='Not provided in current message; not invented',
        prior_protocol_required_pairs=16,adjudication_accepted_pairs=15,prior_protocol_outcome='NO_GO',
        amendment_after_adjudication=True,not_preregistered=True)
    write(OUT/'RESEARCHER_ADJUDICATION_AMENDMENT.json',decisions)
    source=(ROOT/'scripts/run_led_large_contrastive_candidate_v1.py').read_text(encoding='utf-8')
    new=source.replace('CONFIG=PREP/\'PRODUCTION_CONFIG_NOT_APPROVED.json\'',"CONFIG=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2/CONFIG.json'")
    new=new.replace('Not GPU-qualified. No GPU mode is authorized by the offline preparation receipt.',
        'Exploratory Contrastive Pilot V2 — Post-Adjudication Amendment. Conditional execution authorized separately; never preregistered.')
    for name,body in [('preflight',PREFLIGHT),('authorization',AUTH),('diagnose',DIAGNOSE)]:new=replace_function(new,name,body)
    start=new.index('    maximum=3 if args.mode==\'gate\' else 16')
    end=new.index("    data=collated(c,'cpu')",start)
    new=new[:start]+"    maximum=16\n    cap=192\n"+new[end:]
    start=new.index('    # TRAIN-only generation, using unchanged generation settings')
    end=new.index("    return dict(status='PASS',arm=args.arm",start)
    new=new[:start]+"    results=generate_probes(model,c,folder,memory,check,trainer)\n"+new[end:]
    new=new.replace("choices=['preflight','diagnose','gate','pilot']","choices=['preflight','diagnose','baseline','pilot']")
    new=new.replace("accepted_pairs=0,full_pretrained_CUDA_FP16='UNTESTED'","accepted_pairs=15,full_pretrained_CUDA_FP16='UNVERIFIED_UNTIL_CURRENT_DIAGNOSTIC'")
    new=new.replace("suite=(ROOT/'runs/CF_QUALIFICATION_V1').resolve()","suite=(ROOT/c['execution_root']).resolve()")
    new=new.replace("    if args.mode=='pilot':\n            require(args.gate_receipt and read(args.gate_receipt).get('hardware')==profile,'Gate GPU differs')\n",'')
    new=new.replace("            from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter", "        from run_stage_i_led_large_decoder_policy_v2 import MemoryMeter")
    new=new.replace("report=diagnose(model,c,folder,memory,check) if args.mode=='diagnose' else train(model,c,a,args,folder,memory,check)",
        "report=diagnose(model,c,folder,memory,check) if args.mode=='diagnose' else (dict(status='PASS',arm='untouched_ce3',generation_calls=len(generate_probes(model,c,folder,memory,check)),steps=0) if args.mode=='baseline' else train(model,c,a,args,folder,memory,check))")
    new=new.replace("if __name__=='__main__':main()",GENERATE+"\nif __name__=='__main__':main()")
    # Reuse mandatory exact optimizer-step check from successful parent callback.
    new=new.replace("            trainer.amp_recovery.commit(trainer.accelerator.scaler)","            trainer.amp_recovery.commit(trainer.accelerator.scaler)\n            require({int(v['step']) for v in trainer.optimizer.state.values()}=={state.global_step+1},'Optimizer step integrity failed')")
    ast.parse(new);RUNNER.write_text(new,encoding='utf-8')
    (OUT/'V1_TO_V2_RUNNER.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='V1 preserved',tofile='V2 post-adjudication')),encoding='utf-8')
    c=dict(old,experiment='EXPLORATORY_CONTRASTIVE_PILOT_V2_POST_ADJUDICATION',
        title='Exploratory Contrastive Pilot V2 — Post-Adjudication Amendment',pair_ids=accepted,
        excluded_pairs=['P03','P17','P18'],accepted_pairs=15,post_adjudication_amendment=True,not_preregistered=True,
        gate_updates=0,total_successful_update_ceiling=32,total_generation_calls_ceiling=12,
        execution_root='runs/CF_EXPLORATORY_V2',all_researcher_adjudications='8_POSITIVES_AND_15_PAIRS_ACCEPTED_P03_REJECTED',
        diagnostic_attempts=1,diagnostic_successful_optimizer_updates=0,diagnosis_generation_calls=0,
        GPU_execution_approved=False,training_authorized=False,
        production_scope='One diagnostic, conditional CE16 and pair16, untouched CE3 + two-arm4 TRAIN probe outputs; no extra 3-step training gates',
        checkpoint_policy='Steps8 and16 per arm; no resume, search, extra epochs or seed selection')
    c['files']=dict(old['files'])
    c['files'][RUNNER.relative_to(ROOT).as_posix()]=sha(RUNNER)
    c['files'][(OUT/'RESEARCHER_ADJUDICATION_AMENDMENT.json').relative_to(ROOT).as_posix()]=sha(OUT/'RESEARCHER_ADJUDICATION_AMENDMENT.json')
    write(OUT/'CONFIG.json',c)
    files=dict(c['files']);files[(OUT/'CONFIG.json').relative_to(ROOT).as_posix()]=sha(OUT/'CONFIG.json')
    write(OUT/'PREFLIGHT_MANIFEST.json',dict(files=files))
    now=datetime.now(timezone.utc)
    a=dict(experiment=c['experiment'],GPU_execution_approved=True,approval_source='researcher_message',
        researcher_explicit_authorization='Current message authorizes one controlled exploratory pilot subject to mandatory diagnostics.',
        post_adjudication_amendment=True,not_preregistered=True,
        authorized_utc=now.isoformat(),authorization_timestamp_semantics='UTC time this explicit approval was recorded, not an inferred message-send time',
        deadline_utc=(now+timedelta(hours=48)).isoformat(),max_suite_wall_clock_hours=1,
        scopes=['diagnose','baseline','pilot'],updates_per_arm_max=16,generation_calls_max=12,diagnostic_attempts_max=1,
        config_sha256=sha(OUT/'CONFIG.json'),runner_sha256=sha(RUNNER),objective_sha256=sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),
        preflight_sha256=sha(OUT/'PREFLIGHT_MANIFEST.json'),researcher_decisions_sha256=sha(OUT/'RESEARCHER_ADJUDICATION_AMENDMENT.json'),
        conditional_training_after_all_diagnostics_PASS=True,automatic_retry=False,
        prior_14_day_reserve_confirmation='Historical explicit user confirmation retained; not relabeled as a new confirmation',
        held_out_access=False,validation_tuning=False,Teacher_modification=False,additional_experiments=False)
    write(OUT/'RESEARCHER_AUTHORIZATION.json',a)
    (OUT/'PROTOCOL_AMENDMENT.md').write_text('''# Exploratory Contrastive Pilot V2 — Post-Adjudication Amendment

The original protocol required16 independently accepted pairs. Researcher adjudication accepted15,
rejectedP03, and accepted all8 immutable positive targets. Therefore the original protocol is NO-GO.
The researcher now explicitly amends the protocol AFTER adjudication. This is disclosed exploratory
work, NOT preregistered, and is not a replacement for primary thesis RO/U/F/Shuffled comparisons.
No rejected pair is replaced: P01-P02 and P04-P16 only (5Core omission,6disagreement loss,4grounding).
P03/P17/P18 remain excluded. The same8 fit and4 previously examined TRAIN probes remain locked.

## Objective and matched arms

s(y|x)=mean valid-token log p(y_t|x,y_<t); EOS and section headers included, padding ignored.
L=CE(positive)+0.10 softplus((0.10-s(positive)+s(negative))/1.0).
Each candidate is scored under its OWN decoder prefix, with the accepted BOS-free labels and
native single decoder-label shift. Sequential recomputation and exact dropout RNG replay are
unchanged from the reviewed objective. No WCE, DPO, auxiliary head or changed supervision.

A=CE continuation, B=CE+pair continuation. Independently reload exact Original CE3 checkpoint750
(adapter507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60), fresh identical
AdamW/scheduler/scaler, seed42, LR2e-5, betas.9/.999, eps1e-8, wd.01, warmup1 then linear16,
clip1, physical batch1, accumulation8, FP16 autocast/FP32 parameters and nonreentrant checkpointing.
Each update traverses all8 businesses in the frozen order; rotate admitted pair IDs per business
by update modulo its admitted count. Both arms see exactly128 identical positive exposures.
Additional negative forwards consume dropout RNG only in B; same initialization and seed policy
does not imply bit-identical stochastic positive masks between arms. No repeated seeds or search.
Training maximum16 successful updates/arm,192 microbatch attempts/arm including inherited bounded
same-window AMP recovery (max8 backoffs, scale floor256). Checkpoint8/16, no automatic resume.

## Single mandatory diagnostic before any training

Full pinned parent manifest, base binary, entire checkpoint750, code/data/environment, corrected
labels and own-prefix scoring must verify. On real full LED weights with unchanged FP16/checkpointing,
require native CE lambda0 loss/gradient bit equality; joint-graph vs streamed gradient allclose
atol1e-5/rtol1e-4 for every456 LoRA tensors, finite losses/gradients and RNG continuity. TempoP02
(exact frozen source4021tokens, padded4096, short target) is the predeclared feasible joint batch;
no truncation or substitute after failure. Accumulate all8 fit businesses with initial scale65536,
compare explicit per-microbatch gradient sum against native backward accumulation exactly, test
native injected-overflow skip/backoff leaving optimizer/adapter unchanged. No successful optimizer
update or generation in diagnosis. At least15% conservative measured VRAM headroom is mandatory
including both joint reference and intended8-example accumulation. Record each stage before guards.
Natural nonfinite diagnostic gradients mean STOP; do not lower scale or silently change precision.
One diagnostic attempt; narrowly scoped correctness fixes must be documented and cannot turn a
numerical/memory failure into an automatic retry. One-hour execution-suite hard stop within a fresh
maximum48h approval window. Earlier reserve confirmation is historical, not a newly fabricated date.

## Conditional execution and quality stop-loss

Only after ALL diagnostic criteria PASS, generate4 untouched CE3 TRAIN probes, run A16+4probes,
then B16+4probes. No extra3-update gates or inference, maximum32updates/12generate calls total.
Native optimizer state counters must match every successful update. Any integrity, memory,
deadline, generation-collapse, RNG or continuity error stops the chain without retry/fallback.
Generation is original pinned beam4 top1, max768; preserve tokens, text, RNG, memory/timing and
all cases. Compare source-supported Core coverage, mixed opinions, factual/visual grounding,
zero-caption behavior, meaningful repetition and regressions in all4 TRAIN probes. Separate pair
ranking/objective changes from actual generated quality; loss changes alone cannot grant GO.
Three-of-four source-supported improvements without serious regression may justify discussing
further study, not proof of efficacy. Weak, mixed or ungrounded improvements mean NO-GO for further
contrastive development. Model-assisted judgments remain provisional, not invented independent
researcher ratings. Four probes were previously examined training businesses, NOT an independent
generalization benchmark. Keep untouched CE3 and primary matched thesis comparisons unchanged.
''',encoding='utf-8')
    print(json.dumps(dict(config_sha256=sha(OUT/'CONFIG.json'),runner_sha256=sha(RUNNER),authorization=str(OUT/'RESEARCHER_AUTHORIZATION.json'))))

if __name__=='__main__':main()
