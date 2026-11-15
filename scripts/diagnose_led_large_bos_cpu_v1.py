"""Read-only CPU teacher-forced prefix probe: no generate(), training, or held-out input."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys
import json
import hashlib
import gc
import zipfile
from unittest.mock import patch
from datetime import datetime, timezone
import torch
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizerFast
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from safetensors.torch import load, load_file
from huggingface_hub import try_to_load_from_cache

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'
DEST=ROOT/'outputs/stage_i/led_large_vast_bos_cause_audit_v1'
sys.path.insert(0,str(FROZEN/'scripts'))
from stage_i_paired_pilot import initialize_named_adapters

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(1)
    # A CUDA-capable Windows host may still report availability. Enforce actual
    # tensor placement instead of mistaking hardware availability for GPU work.
    def forbid_cuda(*args,**kwargs):
        raise RuntimeError('CUDA initialization forbidden in this CPU diagnostic')
    torch.cuda._lazy_init=forbid_cuda
    DEST.mkdir(exist_ok=True,parents=True)
    tokenizer=LEDTokenizerFast.from_pretrained(FROZEN/'assets/tokenizers/led-base-16384',local_files_only=True)
    sources=['The restaurant serves pizza and pasta. Customers praise friendly service.',
             'The cafe serves bagels and coffee. Customers report slow service.']
    inputs=tokenizer(sources,return_tensors='pt',padding=True,truncation=False)
    global_mask=torch.zeros_like(inputs['input_ids']);global_mask[:,0]=1
    inputs['global_attention_mask']=global_mask
    first=tokenizer.encode('[REVIEW_CONSENSUS]',add_special_tokens=False)
    decoder=torch.tensor([[2,0,first[0]]]*2)
    assert decoder.device.type=='cpu' and all(t.device.type=='cpu' for t in inputs.values())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU FP32 teacher-forced forward on two synthetic sources; no generate/backward/optimizer or real-case inference',
        sources=sources,source_token_ids=inputs['input_ids'].tolist(),decoder_prefix=decoder.tolist(),
        intended_next_ids=[0,first[0],first[1]],limitations=['Tiny synthetic sources, not long frozen validation cases.',
        'FP32 CPU differs from actual FP16 CUDA generation.','Prefix logits alone do not establish full-summary quality or a causal repair.'],models={})
    if '--base-only' in sys.argv:
        previous=json.loads((DEST/'CPU_PREFIX_LOGITS.json').read_text(encoding='utf-8'))
        assert previous['sources']==sources and previous['decoder_prefix']==decoder.tolist()
        result=previous
    for kind in (('base',) if '--base-only' in sys.argv else ('large','base')):
        if kind=='large':
            config=FROZEN/'assets/models/led-large-16384/config.json'
            weights=FROZEN/'assets/models/led-large-16384/pytorch_model.bin'
            expected='b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081'
        else:
            revision='38335783885b338d93791936c54bb4be46bebed9'
            weights=Path(try_to_load_from_cache('allenai/led-base-16384','pytorch_model.bin',revision=revision))
            config=ROOT/'assets/models/led-base-16384/config.json'
            expected=None
        print('LOAD_CPU',kind,flush=True)
        whash=sha(weights)
        if expected:assert whash==expected
        mc=json.loads(config.read_text(encoding='utf-8'))
        with torch.device('meta'):
            base=LEDForConditionalGeneration(LEDConfig(**mc))
        state=torch.load(str(weights),map_location='cpu',weights_only=True,mmap=True)
        base.load_state_dict(state,strict=True,assign=True);base.tie_weights()
        before_bos=base.led.shared.weight[0].detach().clone()
        with patch('peft.tuners.lora.model.is_bnb_available',return_value=False),patch('peft.tuners.lora.model.is_bnb_4bit_available',return_value=False):
            model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',task_type='SEQ_2_SEQ_LM',target_modules='all-linear'))
        initialize_named_adapters(model,42);model.eval()
        assert all(p.device.type=='cpu' for p in model.parameters())
        def probe(label):
            print('FORWARD_CPU',label,flush=True)
            with torch.no_grad():
                logits=model(**inputs,decoder_input_ids=decoder,use_cache=False).logits.float()
            entries=[]
            for i in range(2):
                positions=[]
                for pos in range(3):
                    values,ids=logits[i,pos].log_softmax(-1).topk(5)
                    positions.append(dict(position=pos,top_ids=ids.tolist(),top_tokens=tokenizer.convert_ids_to_tokens(ids.tolist()),
                        top_log_probabilities=values.tolist(),BOS_logit=logits[i,pos,0].item(),
                        expected_id=result['intended_next_ids'][pos],expected_logit=logits[i,pos,result['intended_next_ids'][pos]].item()))
                entries.append(dict(source=i,positions=positions,
                    logits_position0_vs1_max_abs=(logits[i,0]-logits[i,1]).abs().max().item(),
                    logits_position0_vs1_mean_abs=(logits[i,0]-logits[i,1]).abs().mean().item()))
            result['models'][label]=dict(backbone_sha256=whash,finite=bool(torch.isfinite(logits).all()),
                source0_vs1_position1_max_abs=(logits[0,1]-logits[1,1]).abs().max().item(),rows=entries)
            save(DEST/'CPU_PREFIX_LOGITS.json',result)
            print('RESULT',label,json.dumps(entries[0],ensure_ascii=True),flush=True)
        if kind=='large':
            probe('large_pretrained_zero_LoRA')
            archive=ROOT/'outputs/stage_i/led_large_vast_d059_v1_download_audit/downloaded_archives/F_large_d059_v1.zip'
            with zipfile.ZipFile(archive) as z:
                for step in (750,1250):
                    w=load(z.read(f'F_large_d059_v1/train/checkpoint-{step}/adapter_model.safetensors'))
                    set_peft_model_state_dict(model,w)
                    assert all(torch.equal(v.cpu(),w[k]) for k,v in get_peft_model_state_dict(model).items())
                    probe('large_epoch'+str(step//250))
        else:
            w=load_file(ROOT/'outputs/stage_i/d059_expanded_filtered_v2/train/checkpoint-750/adapter_model.safetensors',device='cpu')
            set_peft_model_state_dict(model,w)
            assert all(torch.equal(v.cpu(),w[k]) for k,v in get_peft_model_state_dict(model).items())
            probe('D059_base_epoch3')
        assert torch.equal(model.base_model.model.led.shared.weight[0],before_bos)
        del model,base,state,w;gc.collect()
    print('COMPLETE_CPU_PROBE',DEST/'CPU_PREFIX_LOGITS.json',flush=True)

if __name__=='__main__':main()
