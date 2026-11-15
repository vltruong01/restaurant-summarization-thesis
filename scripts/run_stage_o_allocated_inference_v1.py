"""Finite Stage O worker: native captions, frozen filtering, 75 LED components.

No training/reference/Teacher/Qwen entry point. Services run once, without autorestart.
"""
from pathlib import Path
import argparse,datetime,gc,hashlib,importlib.metadata,json,os,signal,socket,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/stage_o/tripadvisor_external_v1'
E=O/'RTX3080_EXECUTION_V1'
AUTH=E/'EXECUTION_AUTHORIZATION.json'
TASK=ROOT.parent

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def textsha(x):return hashlib.sha256(x.encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc)

def guard():
    a=read(AUTH)
    assert a['execution_approved'] and not any(a[k] for k in ['training_allowed','Qwen_allowed','Teacher_allowed','human_ratings_allowed','lifecycle_actions_allowed'])
    assert socket.gethostname()==a['container'] and os.environ.get('CONTAINER_ID')==a['instance_id']
    assert now()<datetime.datetime.fromisoformat(a['deadline_utc'])
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert identity==a['GPU_UUID']
    assert sha(O/'COHORT15_FREEZE.json')==a['cohort_freeze_sha256']
    return a

def initial_payload():
    guard()
    skipped=[];checked=0
    for r in read(E/'PAYLOAD_MANIFEST.json'):
        if '/.cache/huggingface/' in r['file']:
            skipped.append(r['file']);continue  # Download timestamps/locks are transport metadata, not scientific assets.
        p=ROOT/r['file'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['file']
        checked+=1
    pins={'torch':'2.1.0+cu118','torchvision':'0.16.0+cu118','transformers':'4.51.3','tokenizers':'0.21.1','huggingface-hub':'0.36.2','accelerate':'1.6.0','numpy':'1.26.4','Pillow':'12.2.0','psutil':'7.0.0','safetensors':'0.8.0','peft':'0.15.2'}
    assert sys.version_info[:2]==(3,11)
    assert {p:importlib.metadata.version(p) for p in pins}==pins
    save(E/'REMOTE_PAYLOAD_AND_RUNTIME_VERIFICATION.json',dict(status='PASS_SCIENTIFIC_ASSETS',files=checked,transport_cache_metadata_excluded=skipped,runtime=pins,host=socket.gethostname(),GPU_UUID=guard()['GPU_UUID'],training=False,UTC=now().isoformat(),worker_code_sha256=sha(Path(__file__))))

def empty_GPU():
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not occupied,'Allocated GPU has a compute process before this phase; preserve and stop.'

def caption():
    guard();empty_GPU()
    import yaml,torch,psutil
    import blip2_caption_pilot as legacy
    from generate_stage_f_validation_captions import generate_locked
    from filelock import FileLock
    job=O/'PREPARED_CAPTION_JOB';cfg=yaml.safe_load((job/'CONFIG.yaml').read_text())
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # The historical 40GiB threshold included downloading the 15.5GB checkpoint.
    # Assets are already restored and checksum-verified here; reserve 8GiB additional workspace.
    preflight_cfg=__import__('copy').deepcopy(cfg)
    preflight_cfg['device']['minimum_free_disk_GiB']=8.0
    assert (TASK/'PINNED_ASSET_DOWNLOAD_RECEIPTS.json').is_file()
    env=legacy.preflight(preflight_cfg,job)
    # Operational offload allocation only: same established 4GiB profile, no precision/model change.
    cgroup_limit=int(Path('/sys/fs/cgroup/memory.max').read_text());cgroup_used=int(Path('/sys/fs/cgroup/memory.current').read_text())
    available=min(psutil.virtual_memory().available,cgroup_limit-cgroup_used)
    assert available>=8*2**30
    env.update(GPU_weight_budget_bytes=min(env['GPU_weight_budget_bytes'],4*2**30),CPU_weight_budget_bytes=int(available-3*2**30),
               actual_cgroup_limit_bytes=cgroup_limit,actual_cgroup_available_bytes=available,
               operational_note='Reuse successful 4GiB FP16 GPU/CPU offload profile. Historical 40GiB pre-download disk check replaced by 8GiB additional reserve after pinned assets restored. No generation/precision/model edits.')
    save(job/'environment_report.json',env)
    print('NATIVE_CAPTION_START',len(read(job/'pilot_photo_plan.json')),flush=True)
    with FileLock(job/'caption_model_execution.lock',timeout=0):result=generate_locked(cfg,job)
    assert result['completed']==5143 and result['full_generation_complete']
    print('NATIVE_CAPTION_COMPLETE',flush=True)

def scoring():
    guard();empty_GPU()
    import torch
    torch.set_num_threads(8)
    from stage_o_frozen_pipeline_v1 import score_construct
    score_construct(AUTH)
    assert len(read(O/'LED_INFERENCE_CALL_PLAN_75.json')['calls'])==75
    print('FROZEN_SCORING_CONSTRUCTION_COMPLETE',flush=True)

def led(condition):
    guard();empty_GPU()
    import torch
    from transformers import LEDConfig,LEDForConditionalGeneration,LEDTokenizerFast,GenerationConfig,set_seed
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    lock=read(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    plan=read(O/'LED_INFERENCE_CALL_PLAN_75.json');selected=[r for r in plan['calls'] if r['condition']==condition]
    assert len(selected)==(15 if condition=='RO' else 30)
    out=O/'LED_OUTPUTS'/condition;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);set_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    backbone=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384'
    assert sha(backbone/'pytorch_model.bin')==lock['backbone']['sha256']
    cp=ROOT/lock['models'][condition]['path'];assert sha(cp/'adapter_model.safetensors')==lock['models'][condition]['adapter_sha256']
    tok=LEDTokenizerFast.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
    base=LEDForConditionalGeneration(LEDConfig(**read(backbone/'config.json')))
    state=torch.load(backbone/'pytorch_model.bin',map_location='cpu',weights_only=True);base.load_state_dict(state,strict=True);del state;base.tie_weights()
    assert sum(p.numel() for p in base.parameters())==459801600
    base.name_or_path=str(backbone);model=get_peft_model(base,LoraConfig.from_pretrained(cp,local_files_only=True))
    weights=load_file(str(cp/'adapter_model.safetensors'),device='cpu')
    assert len(weights)==456 and set(weights)==set(get_peft_model_state_dict(model))
    set_peft_model_state_dict(model,weights)
    assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items())
    assert all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters())
    model.eval();model.to('cuda');g=GenerationConfig(**lock['generation'])
    assert g.num_beams==4 and g.max_length==768 and g.length_penalty==1 and not g.do_sample
    save(out/'RESTORATION.json',dict(status='PASS',checkpoint_sha256=lock['models'][condition]['adapter_sha256'],FP32_parameters=True,FP16_autocast=True,adapter_tensors=456,model_eval=True,training=False,generation=g.to_dict()))
    def finite(module,args,result):assert bool(torch.isfinite(result.logits).all()),'Nonfinite logits; preserve, no selective regeneration.'
    hook=model.base_model.model.register_forward_hook(finite)
    records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for i,c in enumerate(selected,1):
        guard();t=time.perf_counter();ids=c['input_ids']
        assert tok.encode(c['source'])==ids and digest(ids)==c['input_ids_sha256'] and textsha(c['source'])==c['source_sha256']
        N=((len(ids)+1023)//1024)*1024
        batch=dict(input_ids=torch.tensor([ids+[1]*(N-len(ids))],device='cuda'),attention_mask=torch.tensor([[1]*len(ids)+[0]*(N-len(ids))],device='cuda'),global_attention_mask=torch.tensor([[1]+[0]*(N-1)],device='cuda'))
        save(out/'STARTED_CALLS'/f'{i:03}.json',dict(condition=condition,business_id=c['business_id'],task=c['task'],UTC=now().isoformat()))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):tokens=model.generate(**batch,generation_config=g)[0].cpu().tolist()
        text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert 0<len(tokens)<=768 and tokens[0]==2
        r=dict(condition=condition,business_id=c['business_id'],task=c['task'],text=text,text_sha256=textsha(text),token_ids=tokens,
               input_ids=ids,input_ids_sha256=c['input_ids_sha256'],source_sha256=c['source_sha256'],checkpoint_sha256=c['checkpoint_sha256'],
               generation_config_sha256=digest(lock['generation']),generation_settings=lock['generation'],selected_review_ids=c['selected_review_ids'],
               selected_photo_ids=c['selected_photo_ids'],seconds=time.perf_counter()-t,termination=dict(length=len(tokens),EOS_seen=2 in tokens[1:],at_length_limit=len(tokens)>=768),
               format_flags=dict(expected_header=text.startswith('[REVIEW_CONSENSUS]\n' if c['task']=='review' else '[VISUAL_EVIDENCE]\n'),foreign_header=('[VISUAL_EVIDENCE]' if c['task']=='review' else '[REVIEW_CONSENSUS]') in text),
               independent_generation=True,FP32_parameters_FP16_autocast=True)
        save(out/'RAW'/f"{r['business_id']}_{r['task']}.json",r);records.append(r)
        print(json.dumps(dict(condition=condition,completed=i,planned=len(selected),business_id=c['business_id'],task=c['task'],tokens=len(tokens),seconds=round(r['seconds'],2))),flush=True)
    assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in get_peft_model_state_dict(model).items())
    assert all(p.grad is None for p in model.parameters());hook.remove()
    save(out/'raw_predictions.json',records)
    assembled=[]
    for b in sorted({r['business_id'] for r in records}):
        rr={r['task']:r for r in records if r['business_id']==b}
        assembled.append(dict(business_id=b,condition=condition,combined_text='\n\n'.join(rr[t]['text'] for t in ['review','visual'] if t in rr),
                              component_texts={t:r['text'] for t,r in rr.items()},component_hashes={t:r['text_sha256'] for t,r in rr.items()},rewriting=False))
    save(out/'predictions.json',assembled)
    save(out/'COMPLETION.json',dict(status='COMPLETE',condition=condition,businesses=15,native_components=len(records),seconds=time.perf_counter()-start,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_unchanged=True,source_plan_sha256=sha(O/'LED_INFERENCE_CALL_PLAN_75.json'),raw_sha256=sha(out/'raw_predictions.json'),no_training=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['all','caption','scoring','RO','U','F']);a=p.parse_args();guard()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Internal execution deadline reached; preserve evidence.')))
    signal.alarm(max(1,int((datetime.datetime.fromisoformat(guard()['deadline_utc'])-now()).total_seconds())))
    if a.phase!='all':
        return caption() if a.phase=='caption' else scoring() if a.phase=='scoring' else led(a.phase)
    initial_payload();start=now()
    try:
        for phase in ['caption','scoring','RO','U','F']:
            save(E/'WORKER_STATUS.json',dict(state='RUNNING',phase=phase,started_at_utc=start.isoformat(),updated_at_utc=now().isoformat()))
            subprocess.run([sys.executable,'-u','-B',__file__,phase],check=True)
            save(E/'PHASE_RECEIPTS'/(phase+'.json'),dict(status='COMPLETE',UTC=now().isoformat()))
        outputs=[r for c in ['RO','U','F'] for r in read(O/'LED_OUTPUTS'/c/'predictions.json')]
        assert len(outputs)==45
        save(O/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json',outputs)
        save(E/'WORKER_STATUS.json',dict(state='NATIVE_GENERATION_COMPLETE_PENDING_LOCAL_DOWNLOAD_VERIFICATION',outputs=45,components=75,started_at_utc=start.isoformat(),finished_at_utc=now().isoformat(),human_ratings=0,Qwen_started=False))
    except BaseException as exc:
        save(E/'FAILURE_PRESERVE_NO_AUTORETRY.json',dict(state='FAILED',error=str(exc),traceback=traceback.format_exc(),UTC=now().isoformat()))
        raise
if __name__=='__main__':main()
