"""Exactly48 authorized D054 outputs; read-only checkpoints, no fallback."""
import argparse,copy,gc,hashlib,importlib.metadata,json,os,time,traceback,datetime
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import LEDForConditionalGeneration,LEDTokenizerFast,GenerationConfig,set_seed
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
from stage_i_d054 import ROOT,OUT,ARMS,preflight,write,read,sha,require,semantic_generation,text_sha
from stage_i_paired_pilot import inspect_attached,lora_config
from stage_hi_led_data import LEDCompleteCollator
from run_stage_i_d050_feasibility_v1 import fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
 require(a.execute,'Current explicit authorization and --execute are required')
 authorization=read(OUT/'RESEARCHER_AUTHORIZATION.json');require(authorization['approved'] is True and authorization['allowed_outputs']==48 and authorization['decision']=='D-054','Wrong authorization')
 ctx,cfg,prep,ec,inputs,training,protected,ref=preflight();require(authorization['reference_sha256']==sha(ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json'),'Approval bound to different reference')
 write(OUT/'PREFLIGHT_REFERENCES_AND_FILES.json',{'status':'PASS','reference_validation':ref,'protected':protected,'24_inputs_same_across_arms':True,'runtime':{k:importlib.metadata.version(k) for k in cfg['runtime_pins']},'torch_cuda':torch.version.cuda})
 require(torch.cuda.is_available(),'Local CUDA unavailable; stop, no fallback');set_seed(42);torch.backends.cuda.matmul.allow_tf32=False
 cloud=ROOT/cfg['frozen_training_root'];tokenizer=LEDTokenizerFast.from_pretrained(cloud/'assets/tokenizers/led-base-16384',local_files_only=True)
 for c in inputs:require(tokenizer(c['source'],truncation=False)['input_ids']==c['input_ids'],'Tokenizer mismatch; abort')
 descriptors={x['arm']:x for x in training['architectures']};models={};weights={};runtime=[]
 # Complete actual CPU model/adaptor/config verification for BOTH arms before
 # any generation. GPU feasibility is checked by the first authorized output,
 # not an unreported extra inference call.
 for arm in ARMS:
  set_seed(42);base=LEDForConditionalGeneration.from_pretrained(cfg['model_id'],revision=cfg['revision'],torch_dtype=torch.float32,local_files_only=True)
  hashes={n:fingerprint(v) for n,v in base.named_parameters()};require(hashes==descriptors[arm]['canonical_initial_backbone_hashes'],'Pretrained backbone tensor mismatch')
  model=get_peft_model(base,LoraConfig(**lora_config(cfg,arm)));scope=inspect_attached(model,arm)
  require(scope['modules']==descriptors[arm]['runtime']['scope']['modules'],'Actual LoRA module attachment mismatch')
  weights[arm]=load_file(ROOT/ec['arms'][arm]['adapter_path']);require(set(weights[arm])==set(get_peft_model_state_dict(model)),'Adapter tensor geometry mismatch')
  set_peft_model_state_dict(model,weights[arm]);require(all(torch.equal(v.detach().cpu(),weights[arm][k]) for k,v in get_peft_model_state_dict(model).items()),'Loaded adapter tensor mismatch')
  effective=copy.deepcopy(model.generation_config)
  for k,v in ec['generation_overrides'].items():setattr(effective,k,v)
  effective.use_cache=True;require(semantic_generation(effective.to_dict())==semantic_generation(ec['effective_generation_config']),'Effective generation drift')
  model.generation_config=effective;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False;model.eval();models[arm]=model
  write(OUT/arm/'MODEL_PREFLIGHT.json',{'status':'PASS','base_revision':cfg['revision'],'all295_backbone_tensor_fingerprints_match':True,'scope':scope,'adapter_sha256':sha(ROOT/ec['arms'][arm]['adapter_path']),'generation_config':effective.to_dict()})
 write(OUT/'BOTH_MODELS_PREFLIGHT_PASS.json',{'status':'PASS','models_verified':list(models),'generation_calls_before_preflight_complete':0,'gpu':torch.cuda.get_device_name(0),'physical_batch':1,'weights':'FP32','compute':'FP16 CUDA autocast','tf32':False})
 collator=LEDCompleteCollator(5000,768,1024);predictions=[]
 for arm in ARMS:
  model=models[arm].to('cuda');model.eval();set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  for index,c in enumerate(inputs):
   path=OUT/arm/'cases'/(hashlib.sha256(c['business_id'].encode()).hexdigest()+'.json');require(not path.exists(),'Existing output: refuse duplicate generation')
   batch={k:v.cuda() for k,v in collator([{'input_ids':c['input_ids'],'labels':c['label_ids']}]).items() if k!='labels'}
   require(batch['global_attention_mask'].sum().item()==1 and batch['global_attention_mask'][0,0].item()==1,'Global attention mask differs')
   require(batch['input_ids'][0,:len(c['input_ids'])].tolist()==c['input_ids'],'Source content changed')
   begin=time.perf_counter()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):ids=model.generate(**batch,**ec['generation_overrides'],use_cache=True)[0].cpu().tolist()
   torch.cuda.synchronize();seconds=time.perf_counter()-begin;require(0<len(ids)<=768,'Unexpected generation length')
   text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
   row={'business_id':c['business_id'],'condition':'filtered','subset':c['subset'],'arm':arm,'text':text,'token_ids':ids,'source_sha256':c['source_sha256'],
    'adapter_sha256':ec['arms'][arm]['adapter_sha256'],'checkpoint_sha256':ec['arms'][arm]['adapter_sha256'],'reference_freeze_sha256':sha(ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json'),
    'generation_config':ec['effective_generation_config'],'generation_config_sha256':sha(prep/'expected_generation_config.json'),'text_sha256':text_sha(text),'token_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
    'generation_seconds':seconds,'source_content_tokens':len(c['input_ids']),'padded_source_tokens':int(batch['input_ids'].shape[1]),'source_untruncated':True}
   write(path,row);predictions.append(row)
   print(json.dumps({'arm':arm,'complete':index+1,'total_complete':len(predictions),'business_id':c['business_id'],'tokens':len(ids),'seconds':round(seconds,2),'peak_allocated_MiB':round(torch.cuda.max_memory_allocated()/1024**2,1)}),flush=True)
   del batch
  record={'arm':arm,'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};runtime.append(record)
  model.cpu();torch.cuda.empty_cache()
  require(all(torch.equal(v.detach().cpu(),weights[arm][k]) for k,v in get_peft_model_state_dict(model).items()),'Adapter mutated during inference')
  hashes={n.removeprefix('base_model.model.').replace('.base_layer.','.'):fingerprint(v) for n,v in model.named_parameters() if '.lora_' not in n};require(hashes==descriptors[arm]['canonical_initial_backbone_hashes'],'Backbone changed during inference')
  write(OUT/arm/'COMPLETION.json',{**record,'outputs':24,'all_adapter_and_backbone_tensors_unchanged':True})
  del models[arm];del model;gc.collect()
 require(len(predictions)==48,'Expected exactly48 outputs')
 for rel,digest in protected.items():require(sha(ROOT/rel)==digest,'Protected artifact changed')
 write(OUT/'predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions));write(OUT/'GENERATION_COMPLETION.json',{'status':'ALL48_GENERATED','arms':runtime,'outputs':48,'protected_artifacts_unchanged':True,'new_training':False,'held_out_access':False})
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists() and not (OUT/'GENERATION_FAILURE.json').exists():write(OUT/'GENERATION_FAILURE.json',{'status':'STOP_NO_FALLBACK','exception':str(e),'traceback':traceback.format_exc()})
  raise
