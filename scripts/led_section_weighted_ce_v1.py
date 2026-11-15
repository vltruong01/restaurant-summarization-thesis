"""Section-aware CE; immutable label masks, no model/optimizer/generation changes."""
import hashlib
import json
from pathlib import Path

REVIEW_HEADER = '[REVIEW_CONSENSUS]\n'
VISUAL_SEPARATOR = '\n\n[VISUAL_EVIDENCE]\n'
HEADER, REVIEW, VISUAL, EOS, IGNORE = 0, 1, 2, 3, -1
OBJECTIVE = dict(policy='SECTION_AWARE_WEIGHTED_CE_V1', review_body=1.5,
                 visual_body=1.0, headers_separators_EOS=1.0, ignore_index=-100,
                 normalization='sum_valid_weights_per_microbatch',
                 accumulation='arithmetic_mean_of_8_microbatch_losses')
_OFFSET_TOKENIZERS={}


def require(ok, message):
    if not ok: raise ValueError(message)


def segment_target(text, tokenizer):
    """Classify offsets only after checking exact markers; reject boundary straddles."""
    require(text.startswith(REVIEW_HEADER) and text.count('[REVIEW_CONSENSUS]') == 1,
            'Missing/duplicated/misplaced review header')
    require(text.count('[VISUAL_EVIDENCE]') == 1 and text.count(VISUAL_SEPARATOR) == 1,
            'Missing/duplicated/ambiguous visual separator')
    boundary = text.index(VISUAL_SEPARATOR)
    spans = dict(review=(len(REVIEW_HEADER), boundary),
                 visual=(boundary + len(VISUAL_SEPARATOR), len(text)))
    require(text[spans['review'][0]:spans['review'][1]].strip(), 'Blank review body')
    # RoBERTa trim_offsets can turn a standalone space token into (a,a).
    # Disable ONLY offset trimming on an in-memory analysis clone; require identical IDs.
    key=id(tokenizer)
    if key not in _OFFSET_TOKENIZERS:
        from tokenizers import Tokenizer
        spec=json.loads(tokenizer.to_str())
        def untrim(value):
            if isinstance(value,dict):
                for k in value:
                    if k=='trim_offsets':value[k]=False
                    else:untrim(value[k])
            elif isinstance(value,list):
                for child in value:untrim(child)
        untrim(spec)
        _OFFSET_TOKENIZERS[key]=(tokenizer,Tokenizer.from_str(json.dumps(spec)))
    enc = _OFFSET_TOKENIZERS[key][1].encode(text, add_special_tokens=True)
    require(enc.ids==tokenizer.encode(text,add_special_tokens=True).ids,'Offset analysis clone changed token IDs')
    require(enc.ids[0] == 0 and enc.ids[-1] == 2 and enc.ids.count(0) == enc.ids.count(2) == 1,
            'Expected one tokenizer BOS/content/EOS')
    require(1 not in enc.ids, 'PAD in raw target')
    roles = []
    for token, (a,b), special in zip(enc.ids[1:], enc.offsets[1:], enc.special_tokens_mask[1:]):
        if special:
            require(token == 2 and (a,b) == (0,0), 'Unexpected special-token offset')
            roles.append(EOS); continue
        require(0 <= a < b <= len(text), 'Invalid ordinary-token offsets')
        role = HEADER
        for name, (start,end) in spans.items():
            overlap = a < end and b > start
            if overlap:
                require(start <= a and b <= end, 'Token straddles section/body boundary')
                role = REVIEW if name == 'review' else VISUAL
        roles.append(role)
    require(roles.count(EOS) == 1 and roles[-1] == EOS and roles.count(REVIEW) > 0,
            'Missing review/EOS supervision')
    return enc.ids[1:], roles, spans


def label_key(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest()


class SectionMaskIndex:
    def __init__(self, path, expected_sha256):
        import numpy as np
        path=Path(path)
        require(hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256,'Section-mask view hash changed')
        self.rows={}
        with np.load(path,allow_pickle=False) as p:
            require(set(p.files)=={'business_ids','labels','roles','offsets','splits'},'Unexpected mask payload')
            require(len(p['offsets'])==len(p['business_ids'])+1 and p['offsets'][0]==0 and
                    p['offsets'][-1]==len(p['labels'])==len(p['roles']) and
                    np.all(np.diff(p['offsets'])>0),'Mask offsets invalid')
            for i in range(len(p['business_ids'])):
                a,b=p['offsets'][i:i+2]; ids=p['labels'][a:b].tolist();roles=p['roles'][a:b].tolist()
                require(ids[-1]==2 and ids.count(2)==1 and 0 not in ids and 1 not in ids,'Mask labels not content/EOS')
                require(roles[-1]==EOS and roles.count(EOS)==1 and set(roles)<={HEADER,REVIEW,VISUAL,EOS},'Mask role invalid')
                k=label_key(ids)
                if k in self.rows: require(self.rows[k]==(ids,roles),'Conflicting duplicate mask')
                self.rows[k]=(ids,roles)

    def for_labels(self, labels):
        import torch
        require(labels.ndim==2 and labels.dtype==torch.long,'Expected int64 batch labels')
        answer=[]
        for row in labels.detach().cpu().tolist():
            n=next((i for i,t in enumerate(row) if t==-100),len(row))
            require(n>0 and all(t==-100 for t in row[n:]),'Labels must have contiguous right padding')
            ids=row[:n];entry=self.rows.get(label_key(ids))
            require(entry is not None and entry[0]==ids,'Unknown/changed target labels; do not infer a mask')
            answer.append(entry[1]+[IGNORE]*(len(row)-n))
        return torch.tensor(answer,device=labels.device,dtype=torch.int8)


def weighted_cross_entropy(logits, labels, roles, review_weight=1.5, visual_weight=1.0, neutral_weight=1.0):
    """Same-time logits/labels (LED already shifts decoder inputs once), FP32 reduction."""
    import torch
    import torch.nn.functional as F
    require(logits.shape[:-1]==labels.shape==roles.shape,'Loss shapes differ')
    valid=labels.ne(-100)
    require(bool(valid.any()),'No valid supervision')
    require(bool(((roles==IGNORE)==(~valid)).all()),'Padding/role mask mismatch')
    require(bool(((roles>=HEADER)&(roles<=EOS)|(~valid)).all()),'Unknown role')
    require(all(w>0 for w in (review_weight,visual_weight,neutral_weight)),'Invalid fixed weight')
    # Float32 prevents half-precision reduction overflow under autocast.
    losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),
                          ignore_index=-100,reduction='none').reshape_as(labels)
    weights=torch.full_like(losses,neutral_weight,dtype=torch.float32)
    weights=torch.where(roles.eq(REVIEW),review_weight,weights)
    weights=torch.where(roles.eq(VISUAL),visual_weight,weights)
    weights=weights*valid
    denominator=weights.sum();loss=(losses*weights).sum()/denominator
    require(bool(torch.isfinite(loss)),'Nonfinite weighted forward CE; STOP')
    with torch.no_grad():
        sections={}
        for name,role in [('review',REVIEW),('visual',VISUAL),('headers_separators',HEADER),('EOS',EOS)]:
            mask=roles.eq(role)&valid;count=int(mask.sum())
            sections[name]=dict(tokens=count,
                ordinary_CE_mean=float(losses[mask].mean()) if count else None,
                weight_sum=float(weights[mask].sum()),
                normalized_loss_contribution=float((losses[mask]*weights[mask]).sum()/denominator))
        evidence=dict(loss=float(loss),ordinary_CE=float(losses.sum()/valid.sum()),
                      valid_tokens=int(valid.sum()),valid_weight_sum=float(denominator),sections=sections)
    return loss,evidence


_INDEX=None


def section_forward(model, inputs, use_cache=False, audit_path=None):
    """Retain baseline labels/forward/shift; replace ONLY output.loss after native CE."""
    global _INDEX
    import torch
    if _INDEX is None:
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/stage_i_led_large_section_wce_v1.json').read_text(encoding='utf-8'))
        require(cfg['section_weighted_ce']['objective']==OBJECTIVE,'Unapproved objective weights/policy')
        _INDEX=SectionMaskIndex(root/cfg['section_weighted_ce']['mask_file'],cfg['section_weighted_ce']['mask_sha256'])
    roles=_INDEX.for_labels(inputs['labels'])
    output=model(**inputs,use_cache=use_cache)
    require(output.loss is not None and bool(torch.isfinite(output.loss)),'Nonfinite native forward CE; STOP')
    # Release unused native CE graph; logits/decoder path and labels remain exact.
    output.loss=None
    loss,evidence=weighted_cross_entropy(output.logits,inputs['labels'],roles)
    output.loss=loss
    if audit_path is not None:
        evidence['phase']='train' if model.training else 'validation'
        path=Path(audit_path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f: f.write(json.dumps(evidence,allow_nan=False)+'\n')
    return output
