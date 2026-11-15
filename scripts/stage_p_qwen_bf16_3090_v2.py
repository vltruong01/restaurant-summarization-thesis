"""One Qwen3.5-9B BF16 execution: largest-input restaurant first, then remaining14.

No fallback, quantization, CPU offload, human-app mutation, retry or model API.
Prepare is CPU-only. Execute requires an explicitly supplied current GPU UUID.
"""
import argparse,datetime as dt,hashlib,json,os,platform,shutil,subprocess,sys,threading,time,traceback,uuid as uuidlib
from pathlib import Path

MODEL='Qwen3.5-9B'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def ids_sha(x):return textsha(json.dumps(x,separators=(',',':')))
def normalized_uuid(x):return str(uuidlib.UUID(str(x).removeprefix('GPU-')))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def need(x,s):
    if not x:raise RuntimeError(s)
def inputs(root):
    spec=read(root/'FROZEN_SPEC.json');prompts=read(root/'PROMPTS.json')
    need(spec['preferred_model']==MODEL and spec['precision']=='bfloat16' and spec['quantization'] is None,'Exact9B BF16 required')
    need(spec['enable_thinking'] is False and spec['batch_size']==1,'Non-thinking batch1 required')
    need(sha(root/'INFERENCE_INPUTS.jsonl')==spec['input_sha256'] and sha(root/'PROMPTS.json')==spec['prompts_sha256'],'Frozen source/prompt hash mismatch')
    rows=[json.loads(line) for line in (root/'INFERENCE_INPUTS.jsonl').read_text(encoding='utf8').splitlines() if line]
    need(len(rows)==30 and len({r['call_id'] for r in rows})==30 and len({r['business_id'] for r in rows})==15,'Complete15-restaurant30-call panel required')
    for r in rows:
        need(textsha(r['source'])==r['source_sha256'],'Source hash mismatch')
        need(('[REVIEWS]' in r['source'])==(r['task']=='review') and ('[IMG_CAP_' in r['source'])==(r['task']=='visual'),'Evidence isolation failed')
        need(not any(t in r['source'] for t in ['<|im_start|>','<|im_end|>','<|vision_start|>','<|image_pad|>']),'Source special-token collision')
    for b in {r['business_id'] for r in rows}:need({r['task'] for r in rows if r['business_id']==b}=={'review','visual'},'Each restaurant needs both tasks')
    return spec,prompts,rows
def render(root,spec,prompts,rows):
    from transformers import AutoTokenizer
    folder=root/'MODEL_PROVENANCE/TOKENIZERS'/MODEL
    for file,h in spec['models'][MODEL]['tokenizer_files'].items():need(sha(folder/file)==h,'Frozen tokenizer changed')
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False);calls=[]
    for r in rows:
        messages=[dict(role='system',content=prompts[r['task']]),dict(role='user',content=r['source'])]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        need(r['source'] in text and text.endswith('<think>\n\n</think>\n\n'),'Non-thinking template/source check failed')
        ids=tok.encode(text,add_special_tokens=False);need(len(ids)<=spec['max_input_tokens'],'Input too long: no truncation allowed')
        calls.append(dict(call_id=r['call_id'],business_id=r['business_id'],task=r['task'],input_ids=ids,input_ids_sha256=ids_sha(ids),rendered_prompt_sha256=textsha(text),input_tokens=len(ids),source_sha256=r['source_sha256']))
    return tok,calls
def execution_order(rows,calls):
    # Largest full Review prompt, ID tie-break, chosen without looking at outputs.
    first=sorted([r for r in calls if r['task']=='review'],key=lambda r:(-r['input_tokens'],r['business_id']))[0]['business_id']
    order=[b for b in dict.fromkeys(r['business_id'] for r in rows) if b!=first]
    order.insert(0,first)
    return first,[next(r for r in rows if r['business_id']==b and r['task']==t) for b in order for t in ['review','visual']]
def prepare(previous,dest):
    need(not dest.exists(),'Do not overwrite an existing execution preparation')
    spec,prompts,rows=inputs(previous)
    # Preserve immutable prior spec rather than silently rewriting the old release.
    for file in ['FROZEN_SPEC.json','PROMPTS.json','INFERENCE_INPUTS.jsonl']:
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(previous/file,dest/file)
    shutil.copytree(previous/'MODEL_PROVENANCE/TOKENIZERS'/MODEL,dest/'MODEL_PROVENANCE/TOKENIZERS'/MODEL)
    tok,calls=render(dest,spec,prompts,rows);first,order=execution_order(rows,calls)
    put(dest/'RENDERED_CALLS_30.json',calls)
    plan=dict(format='stage_p_qwen_bf16_3090_execution_v2',created_at_utc=now(),model=MODEL,model_id=spec['models'][MODEL]['model_id'],revision=spec['models'][MODEL]['revision'],
        spec_sha256=sha(dest/'FROZEN_SPEC.json'),inputs_sha256=sha(dest/'INFERENCE_INPUTS.jsonl'),prompts_sha256=sha(dest/'PROMPTS.json'),rendered_calls_sha256=sha(dest/'RENDERED_CALLS_30.json'),
        pilot_restaurant=first,pilot_selection='Largest actual Review prompt; lexicographic ID tie-break; no output inspection',
        execution_call_order=[r['call_id'] for r in order],pilot_calls_reused_in_final=True,full_batch_requires_pilot_pass=True,approved_model_only=MODEL,
        GPU='RTX3090 24GB',precision='BF16',thinking=False,batch_size=1,automatic_fallback=False,automatic_rental=False,quantization=False,CPU_offload=False,human_evaluation=False,
        latest_authorization='Researcher Stage P Qwen3.5-9B Baseline Inference request, current conversation: compatibility/VRAM test first; conditional full15 on PASS; stop on failure',
        preparation_only_until_current_SSH_GPU_identity_available=True)
    put(dest/'EXECUTION_PLAN.json',plan)
    # Verify exact strings and IDs against the native LED F calls again.
    project=Path(__file__).resolve().parents[1]
    native=project/'outputs/stage_o/tripadvisor_external_v1/RTX3080_EXECUTION_V1/DOWNLOADED_NATIVE/STAGE_O'
    led=read(native/'LED_OUTPUTS/F/raw_predictions.json');lk={(r['business_id'],r['task']):r for r in led};cp=read(native/'LED_INFERENCE_CALL_PLAN_75.json')['calls'];cp={(r['business_id'],r['task']):r for r in cp if r['condition']=='F'}
    checks=[]
    for r in rows:
        l=lk[r['business_id'],r['task']];c=cp[r['business_id'],r['task']]
        need(r['source']==c['source'] and r['source_sha256']==l['source_sha256']==c['source_sha256'],'LED F source mismatch')
        need(r['selected_review_ids']==l['selected_review_ids'] and r['selected_photo_ids']==l['selected_photo_ids'],'LED F evidence ID mismatch')
        checks.append(dict(call_id=r['call_id'],exact_source_equal=True,evidence_IDs_equal=True,source_sha256=r['source_sha256']))
    put(dest/'INPUT_ALIGNMENT.json',dict(status='PASS',businesses=15,calls=30,selected_review_presentations=sum(len(r['selected_review_ids']) for r in rows),filtered_caption_presentations=sum(len(r['selected_photo_ids']) for r in rows),checks=checks,source_LED_call_plan_sha256=sha(native/'LED_INFERENCE_CALL_PLAN_75.json'),no_pixels=True,no_LED_outputs_or_Teacher_or_ratings_in_requests=True))
    put(dest/'LED_MODEL_ONLY_RESOURCE_CONTEXT.json',dict(condition='F',native_GPU='RTX3080',components=[dict(business_id=r['business_id'],task=r['task'],seconds=r['seconds'],output_token_ids=len(r['token_ids']),input_token_ids=len(r['input_ids']),checkpoint_sha256=r['checkpoint_sha256']) for r in led],native_component_file_sha256=sha(native/'LED_OUTPUTS/F/raw_predictions.json'),
        interpretation='Original per-call model inference timers only; shared caption generation/filtering excluded. Qwen load/download excluded from warm model-call throughput and reported separately. Different GPUs/tokenizers/decoders/output lengths preclude controlled hardware/model efficiency attribution.',LED_peak_VRAM='Not established by this native per-call file; do not invent an estimate'))
    print(json.dumps(dict(status='PASS_CPU_INPUTS_AND_TEMPLATE_ONLY',pilot=first,max_prompt_tokens=max(r['input_tokens'] for r in calls),GPU_tested=False,root=str(dest)),indent=2))

class Sampler:
    def __init__(self,uuid):self.uuid=uuid;self.samples=[];self.errors=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.work,daemon=True)
    def work(self):
        while not self.stop.is_set():
            try:
                result=subprocess.run(['nvidia-smi','-i',self.uuid,'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
                v=[float(x.strip()) for x in result.stdout.strip().split(',')];self.samples.append(dict(at_utc=now(),device_used_MiB=v[0],device_total_MiB=v[1],device_utilization_pct=v[2]))
            except Exception as e:self.errors.append(dict(at_utc=now(),error=str(e)))
            self.stop.wait(.25)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=4)

def execute(root,dest,uuid):
    spec,prompts,rows=inputs(root);plan=read(root/'EXECUTION_PLAN.json')
    need(uuid,'Current explicitly selected GPU UUID required')
    need(not dest.exists(),'No retry/resume/overwrite; output already exists')
    need(sha(root/'FROZEN_SPEC.json')==plan['spec_sha256'] and sha(root/'RENDERED_CALLS_30.json')==plan['rendered_calls_sha256'],'Plan binding changed')
    tok,calls=render(root,spec,prompts,rows);need(calls==read(root/'RENDERED_CALLS_30.json'),'Rendered prompt changed');first,order=execution_order(rows,calls)
    need([r['call_id'] for r in order]==plan['execution_call_order'] and first==plan['pilot_restaurant'],'Pilot/order changed')
    import torch,transformers,importlib.metadata
    identity=read(root/'REMOTE_IDENTITY.json')
    need(os.environ.get('CONTAINER_ID')==identity['instance_id'] and platform.node()==identity['hostname'],'Instance/container identity changed')
    need(identity['gpu_uuid']==uuid,'GPU differs from the verified current instance')
    need(all(importlib.metadata.version(n).split('+')[0]==v for n,v in spec['software'].items()),'Pinned software required')
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly the selected authorized CUDA GPU')
    prop=torch.cuda.get_device_properties(0);need(normalized_uuid(prop.uuid)==normalized_uuid(uuid),'Current GPU identity mismatch')
    need('RTX 3090' in prop.name and prop.total_memory>=23*1024**3,'Primary trial requires the authorized RTX3090 24GB')
    need(torch.cuda.is_bf16_supported(),'Native BF16 required')
    free,total=torch.cuda.mem_get_info();need(free>=22*1024**3,'Insufficient free VRAM for isolated24GB trial; no alternative GPU action')
    dest.mkdir(parents=True);sampler=Sampler(uuid);sampler.start();started=time.perf_counter();results=[];phase='model_download';model=None;peaks=[]
    def memory():
        f,t=torch.cuda.mem_get_info();return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device_free_bytes=f,device_total_bytes=t)
    hardware=dict(at_utc=now(),name=prop.name,uuid=uuid,total_bytes=total,free_before_load_bytes=free,BF16_supported=True,platform=platform.platform(),python=sys.version,
        versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','safetensors']},execution_plan_sha256=sha(root/'EXECUTION_PLAN.json'),human_evaluation_started=False)
    put(dest/'HARDWARE_AND_EXECUTION_RECEIPT.json',hardware)
    try:
        from huggingface_hub import snapshot_download
        mi=spec['models'][MODEL];t=time.perf_counter();folder=Path(snapshot_download(mi['model_id'],revision=mi['revision'],allow_patterns=['*.json','*.safetensors','*.jinja']));download_seconds=time.perf_counter()-t
        phase='weight_integrity';t=time.perf_counter();need(sha(folder/'config.json')==mi['config_sha256'],'Model configuration hash mismatch')
        for w in mi['weights']:need((folder/w['file']).stat().st_size==w['bytes'] and sha(folder/w['file'])==w['sha256'],'Weight bytes/hash mismatch')
        weight_verification_seconds=time.perf_counter()-t;put(dest/'WEIGHT_VERIFICATION.json',dict(status='PASS',model=mi,download_seconds=download_seconds,weight_hashing_seconds=weight_verification_seconds))
        phase='model_load';torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        from transformers import Qwen3_5ForConditionalGeneration
        model=Qwen3_5ForConditionalGeneration.from_pretrained(folder,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        torch.cuda.synchronize();load_seconds=time.perf_counter()-t;peaks.append(dict(phase='load',seconds=load_seconds,**memory()))
        need(all(p.device.type=='cuda' and p.device.index in [0,None] for p in model.parameters()),'Unexpected CPU/multi-device offload')
        need(all(p.dtype==torch.bfloat16 for p in model.parameters() if p.is_floating_point()),'Unexpected non-BF16 parameters')
        torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);cm={r['call_id']:r for r in calls}
        for index,r in enumerate(order):
            phase='pilot_generation' if index<2 else 'full_generation'
            need(index<2 or (dest/'COMPATIBILITY_PASS.json').exists(),'Full batch before pilot PASS is forbidden')
            rendered=cm[r['call_id']];ids=rendered['input_ids'];torch.cuda.reset_peak_memory_stats();tensor=torch.tensor([ids],device='cuda:0',dtype=torch.long);mask=torch.ones_like(tensor)
            settings={k:v for k,v in spec['generation'].items() if k!='max_new_tokens_by_task'};settings.update(max_new_tokens=spec['generation']['max_new_tokens_by_task'][r['task']],pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=tok.eos_token_id)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():generated=model.generate(input_ids=tensor,attention_mask=mask,**settings)
            torch.cuda.synchronize();seconds=time.perf_counter()-t;tokens=generated[0,len(ids):].tolist();text=tok.decode(tokens,skip_special_tokens=True);raw=tok.decode(tokens,skip_special_tokens=False);m=memory()
            header='[REVIEW_CONSENSUS]' if r['task']=='review' else '[VISUAL_EVIDENCE]'
            eos=tok.eos_token_id if isinstance(tok.eos_token_id,list) else [tok.eos_token_id]
            flags=dict(empty=not text.strip(),heading_missing=not text.startswith(header),unexpected_thinking=('<think>' in text or '</think>' in text),hit_token_cap=len(tokens)>=settings['max_new_tokens'],stop_token_seen=bool(tokens and tokens[-1] in eos))
            result=dict(**{k:r[k] for k in ['call_id','business_id','task','source_sha256','selected_review_ids','selected_photo_ids']},model_id=mi['model_id'],model_revision=mi['revision'],phase=phase,input_ids=ids,input_ids_sha256=rendered['input_ids_sha256'],rendered_prompt_sha256=rendered['rendered_prompt_sha256'],generation_token_ids=tokens,text=text,raw_decoded_with_special_tokens=raw,text_sha256=textsha(text),input_tokens=len(ids),output_tokens=len(tokens),generation_seconds=seconds,tokens_per_second_including_prefill=len(tokens)/seconds,token_count_includes_stop_and_special_tokens=True,VRAM=m,generation_settings=settings,enable_thinking=False,precision='BF16',flags=flags,completed_at_utc=now())
            put(dest/'CALLS'/(r['call_id']+'.json'),result);results.append(result);peaks.append(dict(phase=phase,call_id=r['call_id'],**m));print(json.dumps(dict(call=r['call_id'],phase=phase,seconds=seconds,tokens=len(tokens),peak_allocated_GiB=m['peak_allocated_bytes']/1024**3)),flush=True)
            del generated,tensor,mask
            # Compatibility is numerical/execution only, never a quality-selection gate.
            need(not flags['empty'] and not flags['unexpected_thinking'],'Execution/non-thinking compatibility failed; preserve evidence and stop')
            if index==1:
                put(dest/'COMPATIBILITY_PASS.json',dict(status='PASS_REAL_GPU_TWO_COMPONENT_CALLS',restaurant=first,review_and_visual_complete=True,model_id=mi['model_id'],revision=mi['revision'],precision='BF16',GPU_uuid=uuid,records=[v['call_id'] for v in results],VRAM_phase_peaks=peaks,pilot_outputs_reused=True,semantic_quality_not_assessed_or_used_as_gate=True,created_at_utc=now()))
        need(len(results)==30,'Incomplete execution');assembled=[]
        for b in dict.fromkeys(r['business_id'] for r in rows):
            rr={r['task']:r for r in results if r['business_id']==b};assembled.append(dict(business_id=b,condition='QWEN',review_text=rr['review']['text'],review_text_sha256=rr['review']['text_sha256'],visual_text=rr['visual']['text'],visual_text_sha256=rr['visual']['text_sha256'],combined_text=rr['review']['text']+'\n\n'+rr['visual']['text']))
        put(dest/'QWEN_OUTPUTS_15.json',assembled)
        put(dest/'COMPLETION_RECEIPT.json',dict(status='COMPLETE_30_NATIVE_CALLS',restaurants=15,components=30,outputs_sha256=sha(dest/'QWEN_OUTPUTS_15.json'),native_call_hashes={r['call_id']:sha(dest/'CALLS'/(r['call_id']+'.json')) for r in results},GPU_uuid=uuid,model=mi,download_seconds=download_seconds,weight_verification_seconds=weight_verification_seconds,load_seconds=load_seconds,model_generation_seconds=sum(r['generation_seconds'] for r in results),output_tokens=sum(r['output_tokens'] for r in results),model_only_tokens_per_second=sum(r['output_tokens'] for r in results)/sum(r['generation_seconds'] for r in results),VRAM_phase_peaks=peaks,wall_elapsed_seconds=time.perf_counter()-started,shared_caption_pipeline_excluded=True,full_human_evaluation_started=False,completed_at_utc=now()))
    except Exception as e:
        try:m=memory()
        except Exception:m=None
        put(dest/'FAILURE_RECEIPT.json',dict(status='FAILED_STOP_NO_RETRY_OR_FALLBACK',phase=phase,completed_calls=len(results),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),memory_at_failure=m,previous_phase_peaks=peaks,at_utc=now(),A100_estimate='If true24GB OOM: A10040GB BF16 batch1 provides substantial headroom over~18GiB pinned weights; unbenchmarked estimate, not permission to rent or execute. Diagnose actual trace/peak first.'))
        raise
    finally:
        sampler.close();put(dest/'DEVICE_MEMORY_SAMPLES.json',dict(samples=sampler.samples,errors=sampler.errors,sampling_interval_target_seconds=.25,metric='nvidia-smi whole selected device memory, includes CUDA context/other processes; sampling can miss transient peaks; torch allocator peaks are per-process',peak_device_used_MiB=max((r['device_used_MiB'] for r in sampler.samples),default=None)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','execute']);p.add_argument('--root',required=True,type=Path);p.add_argument('--destination',required=True,type=Path);p.add_argument('--gpu-uuid');a=p.parse_args()
    if a.command=='prepare':prepare(a.root.resolve(),a.destination.resolve())
    else:execute(a.root.resolve(),a.destination.resolve(),a.gpu_uuid)
if __name__=='__main__':main()
