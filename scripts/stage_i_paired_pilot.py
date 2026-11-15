"""Shared paired-pilot checks. No CUDA/model weights/forward are used by offline checks."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('A_decoder_only', 'B_encoder_decoder')
DECODER = r'led\.decoder\.layers\.[0-5]\.(?:self_attn|encoder_attn)\.(?:q_proj|v_proj)'
ENCODER = r'led\.encoder\.layers\.[0-5]\.self_attn\.longformer_self_attn\.(?:query|value|query_global|value_global)'


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def signature(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expected_modules(arm):
    require(arm in ARMS, 'Unexpected pilot arm')
    dec = {f'led.decoder.layers.{i}.{attn}.{proj}' for i in range(6)
           for attn in ('self_attn','encoder_attn') for proj in ('q_proj','v_proj')}
    enc = {f'led.encoder.layers.{i}.self_attn.longformer_self_attn.{proj}' for i in range(6)
           for proj in ('query','value','query_global','value_global')}
    return dec if arm == ARMS[0] else dec | enc


def canonical(name):
    return name.removeprefix('base_model.model.')


def lora_config(config, arm):
    return {**config['lora'], 'target_modules': DECODER if arm == ARMS[0] else f'(?:{DECODER}|{ENCODER})'}


def inspect_attached(model, arm):
    actual = set(model.targeted_module_names)
    require(actual == expected_modules(arm), 'Actual attached module set differs from exact approved scope')
    trainable = {n: p for n,p in model.named_parameters() if p.requires_grad}
    require(all('.lora_A.' in n or '.lora_B.' in n for n in trainable), 'Unfrozen pretrained parameter')
    expected_count = 294912 if arm == ARMS[0] else 589824
    require(sum(p.numel() for p in trainable.values()) == expected_count, 'Trainable parameter count differs')
    require(len(trainable) == 2*len(actual), 'Unexpected trainable tensor count')
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Weight precision differs')
    for name,m in model.named_modules():
        if canonical(name) in actual:
            require(m.scaling['default'] == 2.0 and not m.disable_adapters, 'Inactive/wrong LoRA scaling')
            require(m.lora_A['default'].weight.requires_grad and m.lora_B['default'].weight.requires_grad, 'Frozen intended adapter')
    return {'arm':arm, 'modules':sorted(actual), 'trainable_parameters':expected_count,
            'trainable_tensor_shapes':{n:list(p.shape) for n,p in trainable.items()},
            'encoder_local':sum(bool(re.fullmatch(ENCODER,n)) and not n.endswith('_global') for n in actual),
            'encoder_global':sum(bool(re.fullmatch(ENCODER,n)) and n.endswith('_global') for n in actual),
            'decoder':sum(bool(re.fullmatch(DECODER,n)) for n in actual)}


def meta_probe(config, model_config, arm):
    from transformers import LEDConfig, LEDForConditionalGeneration
    from peft import get_peft_model, LoraConfig
    with torch.device('meta'):
        model = LEDForConditionalGeneration(LEDConfig(**model_config))
        model = get_peft_model(model, LoraConfig(**lora_config(config,arm)))
        record = inspect_attached(model,arm)
        require(all(p.device.type == 'meta' for p in model.parameters()), 'Real model allocation in meta check')
    return {**record, 'meta_only':True, 'forward_backward':False, 'pretrained_weights_loaded':False}


def module_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}|{canonical(name)}'.encode()).digest()[:8], 'big') % (2**63-1)


def initialized_A(shape, seed, name):
    generator = torch.Generator(device='cpu').manual_seed(module_seed(seed,name))
    result = torch.empty(shape, dtype=torch.float32)
    # Torch2.1 kaiming_uniform_ has no generator kwarg. Its a=sqrt(5)
    # initialization is uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    bound=1/math.sqrt(shape[1])
    result.uniform_(-bound,bound,generator=generator)
    return result


def initialize_named_adapters(model, seed):
    """Independent standard Kaiming A/zero B, keyed by common seed and exact module name."""
    hashes = {}
    with torch.no_grad():
        for name,m in model.named_modules():
            if hasattr(m,'lora_A') and 'default' in m.lora_A:
                a,b = m.lora_A['default'].weight,m.lora_B['default'].weight
                require(a.device.type == b.device.type == 'cpu', 'Initialize real adapters on CPU before CUDA transfer')
                a.copy_(initialized_A(a.shape,seed,name)); b.zero_()
                hashes[canonical(name)] = hashlib.sha256(a.detach().numpy().tobytes()+b.detach().numpy().tobytes()).hexdigest()
    return hashes


def epoch_orders(n, seed, epochs):
    return [torch.randperm(n, generator=torch.Generator(device='cpu').manual_seed(seed+e)).tolist() for e in range(epochs)]


class PairedEpochSampler(Sampler):
    def __init__(self, orders):
        self.orders,self.epoch = orders,0
        require(all(sorted(o)==list(range(len(o))) for o in orders),'Order is not a full permutation')
    def set_epoch(self, epoch):
        require(0 <= epoch < len(self.orders),'Unexpected extra pilot epoch'); self.epoch=epoch
    def __iter__(self): return iter(self.orders[self.epoch])
    def __len__(self): return len(self.orders[0])


class BusinessSubset(Dataset):
    def __init__(self, dataset, ids):
        require(len(ids)==len(set(ids)) and set(ids)<=set(dataset.business_ids),'Missing/duplicate subset ID')
        self.dataset=dataset; self.business_ids=list(ids)
        positions={b:i for i,b in enumerate(dataset.business_ids)}
        self.indices=[positions[b] for b in ids]
    def __len__(self): return len(self.indices)
    def __getitem__(self,i): return self.dataset[self.indices[i]]


def gradient_record(model, expect_later_A=False):
    record={}
    for name,p in model.named_parameters():
        if not p.requires_grad:require(p.grad is None,'Unexpected frozen-base gradient: '+name)
        if p.requires_grad:
            require(p.grad is not None and torch.isfinite(p.grad).all().item(),'Missing/nonfinite adapter gradient: '+name)
            nonzero=bool(torch.count_nonzero(p.grad).item())
            if '.lora_B.' in name or expect_later_A:
                require(nonzero,'Inactive intended adapter gradient: '+name)
            record[name]={'finite':True,'nonzero':nonzero,'norm':float(p.grad.detach().float().norm())}
    require(record,'No adapter gradients')
    return record


def verify_mask_batch(batch, features):
    from transformers.models.led.modeling_led import shift_tokens_right
    shifted=shift_tokens_right(batch['labels'],1,2)
    for i,f in enumerate(features):
        n,m=len(f['input_ids']),len(f['labels'])
        require(batch['input_ids'][i,:n].tolist()==f['input_ids'] and batch['labels'][i,:m].tolist()==f['labels'],'Content altered')
        require(torch.all(batch['labels'][i,m:]==-100).item(),'Label padding is not ignored')
        require(batch['attention_mask'][i].sum().item()==n and not batch['attention_mask'][i,n:].any().item(),'Padding mask differs')
        require(batch['global_attention_mask'][i,0].item()==1 and batch['global_attention_mask'][i].sum().item()==1,'Global BOS differs')
        require(shifted[i,0].item()==2 and shifted[i,1:m].tolist()==f['labels'][:-1],'Decoder shift differs')


def checked_preparation(config_path):
    config=read(config_path); p=ROOT/config['preparation_directory']; manifest=read(p/'PREPARATION_MANIFEST.json')
    for package,version in config['runtime_pins'].items():
        require(importlib.metadata.version(package).split('+')[0]==version,'Runtime version drift: '+package)
    require(sha(config_path)==manifest['config_sha256'],'Prepared config changed')
    for rel,digest in manifest['files'].items(): require(sha(p/rel)==digest,'Prepared artifact changed: '+rel)
    for rel,digest in manifest['code'].items(): require(sha(ROOT/rel)==digest,'Reviewed pilot code changed: '+rel)
    for rel,digest in manifest['upstream'].items(): require(sha(ROOT/rel)==digest,'Frozen input changed: '+rel)
    require(config['condition']=='filtered' and config['training']['epochs']==2 and config['training']['steps']==128,'Scope changed')
    return config,p,manifest


def execution_gate(config_path, mode, arm, approval_path, execute):
    # Must remain before CUDA checks or pretrained weight loading.
    require(execute,'Execution disabled; --execute plus scope-specific researcher approval required')
    require(mode in ('feasibility','train','generate') and arm in ARMS,'Unauthorized mode/arm')
    require(approval_path is not None and Path(approval_path).is_file(),'Approval record missing')
    config,p,manifest=checked_preparation(config_path); a=read(approval_path)
    require(a.get('approved') is True and mode in a.get('scopes',[]) and arm in a.get('arms',[]),'Mode/arm not approved')
    require(a.get('config_sha256')==sha(config_path) and a.get('preparation_manifest_sha256')==sha(p/'PREPARATION_MANIFEST.json'),'Approval bound to different preparation')
    require(a.get('protocol_sha256')==sha(p/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),'Protocol approval mismatch')
    if mode=='train':
        reviewed=a.get('reviewed_feasibility',{})
        for other in ARMS:
            info=reviewed.get(other,{})
            require(info.get('path') and info.get('sha256'),'Both feasibility reports require review')
            report_path=ROOT/info['path']; require(sha(report_path)==info['sha256'],'Feasibility report changed')
            r=read(report_path)
            require(r.get('status')=='TECHNICAL_FEASIBILITY_PASS' and r.get('arm')==other and r.get('config_sha256')==sha(config_path),'Incompatible feasibility evidence')
            rb=r.get('binding',{})
            require(rb.get('preparation_manifest_sha256')==sha(p/'PREPARATION_MANIFEST.json') and
                    rb.get('runner_sha256')==sha(ROOT/'scripts/run_stage_i_paired_lora_pilot_v1.py') and
                    rb.get('helper_sha256')==sha(Path(__file__)),'Stale feasibility binding')
    if mode=='generate':
        require(a.get('verified_reference_units_path') and a.get('verified_reference_units_sha256'),'Freeze source units before new pilot outputs')
        units_path=ROOT/a['verified_reference_units_path']; require(sha(units_path)==a['verified_reference_units_sha256'],'Reference-unit freeze changed')
        verify_reference_freeze(read(units_path),config,p)
    return config,p,manifest,a


def verify_reference_freeze(value, config, prep):
    require(value.get('case_inputs_sha256')==sha(prep/'evaluation_inputs.jsonl'),'Reference units use different sources')
    require(value.get('protocol_sha256')==sha(prep/'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'),'Reference units use different rubric')
    require(value.get('researcher_confirmed_complete') is True,'Source reference sheet not confirmed complete')
    cases=[json.loads(l) for l in (prep/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    byid={c['business_id']:c for c in cases}
    require(set(value.get('business_reviews',{}))==set(byid),'Each evaluation business needs source-unit completeness review')
    for b,r in value['business_reviews'].items():
        require(r.get('human_verified') is True and r.get('positive_negative_and_specificity_checked') is True,'Unreviewed source completeness: '+b)
    require(value.get('units'),'No verified reference units')
    seen=set()
    for unit in value['units']:
        require(unit.get('human_verified') is True and unit.get('support_status')=='supported','Unverified/unsupported reference unit')
        bid=unit['business_id'];require(bid in byid and unit.get('unit_id') and unit.get('unit_id') not in seen,'Unknown business/duplicate reference unit')
        seen.add(unit['unit_id']); case=byid[bid]
        require(unit.get('domain') in ('review','visual') and unit.get('importance') in ('core','supporting'),'Unit domain/importance unresolved')
        require(isinstance(unit.get('aspect'),str) and bool(unit['aspect']) and isinstance(unit.get('restaurant_specific'),bool),'Unit aspect/specificity unresolved')
        require(unit.get('text') and unit.get('evidence'),'Empty unit or evidence')
        allowed=[f'REV_{i+1}' for i in range(len(case['selected_review_ids']))] if unit['domain']=='review' else [x['caption_id'] for x in captions_for(case)]
        for evidence in unit['evidence']:
            marker=evidence.get('marker');require(marker in allowed,'Wrong evidence domain/ID')
            if unit['domain']=='visual':
                cap=next(c for c in captions_for(case) if c['caption_id']==marker)
                require(evidence.get('photo_id')==cap['photo_id'] and evidence.get('quote') and evidence['quote'] in cap['text'],'Caption/photo/quotation mismatch')
            else:
                i=int(marker.split('_')[1])-1
                require(evidence.get('review_id')==case['selected_review_ids'][i],'REV original ID mismatch')
                match=re.search(r'\['+marker+r'\] (.*?)(?=\n\n\[REV_\d+\]|\n\n\[VISUAL_CONTEXT\]|\Z)',case['source'],re.S)
                require(match and evidence.get('quote') and evidence['quote'] in match.group(1),'Review quote not in cited REV')
        require(unit['domain']!='review' or unit.get('polarity') in ('positive','negative','mixed','neutral'),'Review polarity unresolved')
        require(unit.get('human_verified') and unit.get('confirmed_by'),'Human confirmer provenance missing')
    for bid in byid:
        require(any(u['business_id']==bid and u['domain']=='review' for u in value['units']),'Source completeness missing review units: '+bid)


def captions_for(case):
    from stage_i_decoding_diagnostics import captions
    return captions(case)


def paired_bootstrap(a,b,seed=20261016,resamples=5000):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    require(len(a)==len(b),'Unpaired metric lengths')
    if len(pairs)<2:return {'paired_n':len(pairs),'mean_B_minus_A':None,'CI95':None,'missing_pairs':len(a)-len(pairs)}
    delta=np.asarray([y-x for x,y in pairs],dtype=float)
    require(np.isfinite(delta).all(),'Nonfinite paired metric')
    rng=np.random.default_rng(seed);means=delta[rng.integers(0,len(delta),size=(resamples,len(delta)))].mean(1)
    return {'paired_n':len(pairs),'missing_pairs':len(a)-len(pairs),'mean_B_minus_A':float(delta.mean()),
            'CI95':np.quantile(means,[.025,.975]).tolist(),'B_greater_than_A':int((delta>0).sum()),'equal':int((delta==0).sum()),
            'B_less_than_A':int((delta<0).sum()),'sign_is_metric_direction_not_quality':True,'resamples':resamples}
