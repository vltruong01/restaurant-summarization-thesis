"""Portable Stage P checks and one frozen text-only Qwen batch. No training/API.

check: CPU tokenizer/config checks only; probe: CUDA identity/memory, no weights;
run: requires a local execution receipt binding GPU, spec, estimate and approval.
No default command launches inference. Never contacts an SSH/provider endpoint.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,os,sys,time,traceback

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def load(root):
    spec=read(root/'FROZEN_SPEC.json');prompts=read(root/'PROMPTS.json')
    require(sha(root/'INFERENCE_INPUTS.jsonl')==spec['input_sha256'],'Frozen inputs changed')
    require(sha(root/'PROMPTS.json')==spec['prompts_sha256'],'Frozen prompts changed')
    rows=[json.loads(l) for l in (root/'INFERENCE_INPUTS.jsonl').read_text(encoding='utf8').splitlines() if l]
    require(len(rows)==30 and len({r['call_id'] for r in rows})==30,'Exactly 30 unique calls required')
    require(len({r['business_id'] for r in rows})==15,'Exactly 15 restaurants required')
    for row in rows:
        require(textsha(row['source'])==row['source_sha256'],'Source changed')
        require(row['task'] in ['review','visual'],'Task isolation')
        require(('[REVIEWS]' in row['source'])==(row['task']=='review'),'Wrong evidence component')
        require(('[IMG_CAP_' in row['source'])==(row['task']=='visual'),'Wrong caption isolation')
    return spec,prompts,rows
def tokenizer(root,name):
    from transformers import AutoConfig,AutoTokenizer
    folder=root/'MODEL_PROVENANCE/TOKENIZERS'/name
    spec=read(root/'FROZEN_SPEC.json')
    for file,expected in spec['models'][name]['tokenizer_files'].items():require(sha(folder/file)==expected,'Tokenizer/config file changed: '+file)
    config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    return tok,config
def messages(prompts,row):return [dict(role='system',content=prompts[row['task']]),dict(role='user',content=row['source'])]
def check(root,destination):
    spec,prompts,rows=load(root);models={}
    for name in spec['models']:
        tok,config=tokenizer(root,name);calls=[]
        for r in rows:
            formatted=tok.apply_chat_template(messages(prompts,r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(formatted,add_special_tokens=False)
            require(len(ids)<=spec['max_input_tokens'],'Input too long: STOP, never silently truncate')
            require(r['source'] in formatted,'Template altered the exact source text')
            # Chat template delimiter strings must not occur inside untrusted evidence.
            require(not any(t in r['source'] for t in ['<|im_start|>','<|im_end|>','<|vision_start|>','<|image_pad|>']),'Unescaped special-token collision requires review')
            require(formatted.endswith('<think>\n\n</think>\n\n'),'Non-thinking template not established')
            calls.append(dict(call_id=r['call_id'],tokens=len(ids),input_ids_sha256=textsha(json.dumps(ids,separators=(',',':'))),
                rendered_prompt_sha256=textsha(formatted),source_sha256=r['source_sha256']))
        models[name]=dict(config_class=type(config).__name__,tokenizer_class=type(tok).__name__,revision=spec['models'][name]['revision'],calls=calls,
            largest_prompt_tokens=max(c['tokens'] for c in calls),all_sources_preserved=True,non_thinking_prefix_verified=True)
    result=dict(status='PASS_CPU_CONFIG_TOKENIZER_CHAT_TEMPLATE_ONLY',models=models,transformers=importlib.metadata.version('transformers'),weights_downloaded=False,
        model_forward_pass_tested=False,GPU_memory_tested=False,created_at_utc=now())
    put(destination,result);print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2));return result
def probe(root,destination):
    import torch,subprocess
    spec,_,_=load(root);require(torch.cuda.is_available(),'No CUDA GPU available')
    require(torch.cuda.device_count()==1,'Expose exactly one selected GPU with CUDA_VISIBLE_DEVICES; do not use unrelated devices')
    p=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info(0)
    uuids=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    result=dict(status='READ_ONLY_HARDWARE_PROBE_ONLY',created_at_utc=now(),gpu_name=p.name,gpu_uuid=str(p.uuid),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),free_bytes=free,total_bytes=total,bfloat16_supported=torch.cuda.is_bf16_supported(),
        nvidia_smi=uuids,torch=torch.__version__,cuda=torch.version.cuda,spec_sha256=sha(root/'FROZEN_SPEC.json'),weights_loaded=False,
        feasible_estimate=bool(total>=23*1024**3 and free>=22*1024**3 and torch.cuda.is_bf16_supported()),
        caveat='Estimate only. Actual model load/peak/generation is not established by this probe.')
    put(destination,result);print(json.dumps(result,indent=2))
def run(root,auth_path,output):
    spec,prompts,rows=load(root)
    require(auth_path and Path(auth_path).is_file(),'No GPU execution approval receipt: STOP before model/weight load')
    auth=read(auth_path);require(auth.get('researcher_approved') is True and auth.get('spec_sha256')==sha(root/'FROZEN_SPEC.json'),'Execution approval not bound to this spec')
    require(auth.get('approved_at_utc') and auth.get('approval_evidence') and auth.get('gpu_uuid'),'Actual approval evidence/GPU identity required')
    require(auth.get('runtime_estimate_minutes') and 'estimated_incremental_USD' in auth,'Present time/cost estimate before full batch')
    expires=datetime.datetime.fromisoformat(auth['expires_at_utc']);require(expires.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc)<expires,'Expired execution authorization')
    name=auth.get('model');require(name in spec['models'],'Only pinned preferred/fallback models allowed')
    if name!=spec['preferred_model']:require(bool(auth.get('preferred_infeasibility_evidence')),'Document hardware/software reason before fallback, not scores')
    require(not output.exists(),'No automatic retry/resume/overwrite: output directory already exists')
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText
    require(transformers.__version__==spec['software']['transformers'],'Use frozen transformers version')
    require(torch.__version__.split('+')[0]==spec['software']['torch'],'Use frozen torch version')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one authorized CUDA GPU only')
    require(str(torch.cuda.get_device_properties(0).uuid)==auth['gpu_uuid'],'GPU identity mismatch')
    require(torch.cuda.is_bf16_supported(),'BF16 support required; do not silently change precision')
    tok,config=tokenizer(root,name)
    rendered=[]
    for r in rows:
        ids=tok.apply_chat_template(messages(prompts,r),tokenize=True,add_generation_prompt=True,enable_thinking=False)
        require(len(ids)<=spec['max_input_tokens'],'Input exceeds frozen limit; no truncation')
        rendered.append(ids)
    output.mkdir(parents=True);started=time.perf_counter();results=[]
    try:
        from huggingface_hub import snapshot_download
        model_info=spec['models'][name]
        folder=Path(snapshot_download(model_info['model_id'],revision=model_info['revision'],allow_patterns=['*.json','*.safetensors','*.jinja']))
        require(sha(folder/'config.json')==model_info['config_sha256'],'Downloaded model config differs from frozen configuration')
        weights=[]
        for w in model_info['weights']:
            path=folder/w['file'];require(path.stat().st_size==w['bytes'] and sha(path)==w['sha256'],'Downloaded weight hash mismatch')
            weights.append(dict(w))
        klass=AutoModelForImageTextToText if name=='Qwen3.5-9B' else AutoModelForCausalLM
        # Official multimodal checkpoint used strictly with input_ids/attention_mask;
        # no processor, pixel_values, image encoder input or external visual API.
        model=klass.from_pretrained(folder,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        model.eval();torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);torch.cuda.reset_peak_memory_stats()
        for r,ids in zip(rows,rendered):
            t=time.perf_counter();tensor=torch.tensor([ids],dtype=torch.long,device='cuda:0')
            settings={k:v for k,v in spec['generation'].items() if k!='max_new_tokens_by_task'}
            settings.update(max_new_tokens=spec['generation']['max_new_tokens_by_task'][r['task']],pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=tok.eos_token_id)
            with torch.inference_mode():generated=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),**settings)
            torch.cuda.synchronize();new=generated[0,len(ids):].tolist();raw=tok.decode(new,skip_special_tokens=False);text=tok.decode(new,skip_special_tokens=True)
            expected='[REVIEW_CONSENSUS]' if r['task']=='review' else '[VISUAL_EVIDENCE]'
            row=dict(call_id=r['call_id'],business_id=r['business_id'],task=r['task'],model_id=model_info['model_id'],model_revision=model_info['revision'],
                source_sha256=r['source_sha256'],spec_sha256=sha(root/'FROZEN_SPEC.json'),prompt_version=spec['prompt_version'],
                input_ids=ids,input_ids_sha256=textsha(json.dumps(ids,separators=(',',':'))),generation_token_ids=new,raw_decoded_with_special_tokens=raw,
                text=text,text_sha256=textsha(text),seconds=time.perf_counter()-t,input_tokens=len(ids),output_tokens=len(new),words=len(text.split()),
                generation_settings=settings,flags=dict(hit_token_cap=len(new)>=settings['max_new_tokens'],heading_missing=not text.startswith(expected),empty=not text.strip(),unexpected_thinking='<think>' in text),
                selected_review_ids=r['selected_review_ids'],selected_photo_ids=r['selected_photo_ids'])
            put(output/'CALLS'/(r['call_id']+'.json'),row);results.append(row)
        require(len(results)==30,'Incomplete batch')
        assembled=[]
        for b in dict.fromkeys(r['business_id'] for r in rows):
            pair={r['task']:r for r in results if r['business_id']==b};assembled.append(dict(business_id=b,condition='QWEN',
                review_text=pair['review']['text'],review_text_sha256=pair['review']['text_sha256'],visual_text=pair['visual']['text'],visual_text_sha256=pair['visual']['text_sha256'],
                combined_text=pair['review']['text']+'\n\n'+pair['visual']['text']))
        put(output/'QWEN_OUTPUTS_15.json',assembled)
        put(output/'COMPLETION_RECEIPT.json',dict(status='COMPLETE_30_NATIVE_CALLS',calls=30,businesses=15,model=model_info,precision=spec['precision'],
            spec_sha256=sha(root/'FROZEN_SPEC.json'),outputs_sha256=sha(output/'QWEN_OUTPUTS_15.json'),
            native_call_hashes={r['call_id']:sha(output/'CALLS'/(r['call_id']+'.json')) for r in results},
            authorization_sha256=sha(auth_path),gpu_uuid=auth['gpu_uuid'],finished_at_utc=now(),elapsed_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            library_versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','safetensors']},weights_verified=weights,
            note='Greedy non-thinking one-pass; cross-device bitwise determinism not claimed. Token/format flags preserved; no selective retries.'))
    except Exception as e:
        put(output/'FAILURE_RECEIPT.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',completed_calls=len(results),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),at_utc=now()))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','probe','run']);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--destination',type=Path);p.add_argument('--authorization',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.command=='check':check(root,a.destination or root/'VALIDATION/CPU_COMPATIBILITY.json')
    elif a.command=='probe':probe(root,a.destination or root/'GPU_HARDWARE_PROBE.json')
    else:run(root,a.authorization,a.destination or root/'QWEN_EXECUTION')
if __name__=='__main__':main()
