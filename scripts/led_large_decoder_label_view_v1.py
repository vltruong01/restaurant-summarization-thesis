"""Versioned decoder-label correction; original Stage H data are read-only."""
from pathlib import Path
import hashlib
import json
import numpy as np

POLICY = 'LED_LARGE_CONTENT_EOS_NO_INITIAL_BOS_V1'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def remove_initial_bos(labels):
    """Remove exactly the tokenizer-added leading BOS; reject ambiguous input."""
    values=list(labels)
    require(2<=len(values)<=768,'Original complete label length invalid')
    require(all(type(x) is int and 0<=x<50265 for x in values),'Invalid unpadded token ID')
    require(values[0]==0 and values[-1]==2,'Expected original BOS/content/EOS labels')
    require(values.count(0)==1 and values.count(2)==1 and 1 not in values,'Duplicate/interior special token or PAD in raw labels')
    corrected=values[1:]
    validate_view_labels(corrected)
    return corrected

def validate_view_labels(labels):
    require(len(labels)>=2 and labels[0] not in (0,1,2) and labels[-1]==2,'Expected content/EOS label view')
    require(all(type(x) is int and 0<=x<50265 for x in labels),'Invalid corrected token ID')
    require(0 not in labels and 1 not in labels and labels.count(2)==1,'Special-token duplicates in corrected labels')

class CorrectedLabelDataset:
    """Materialized label-only view paired with the unmodified original sources."""
    def __init__(self, original, info, root):
        self.original=original;self.business_ids=list(original.business_ids)
        path=Path(root)/info['path']
        require(sha(path)==info['sha256'],'Derived label view hash differs')
        with np.load(path,allow_pickle=False) as pack:
            require(set(pack.files)=={'business_ids','labels','label_offsets'},'Unexpected derived payload')
            self.labels=pack['labels'].copy();self.offsets=pack['label_offsets'].copy()
            require(pack['business_ids'].tolist()==self.business_ids,'Derived view business/order differs')
        require(len(self.offsets)==len(self.business_ids)+1 and self.offsets[0]==0
                and self.offsets[-1]==len(self.labels) and np.all(np.diff(self.offsets)>0),'Invalid view offsets')
        # Verify every row, not only the first minibatch. Sources remain original.
        for i in range(len(self)):self[i]

    def __len__(self):return len(self.business_ids)

    def __getitem__(self,index):
        original=self.original[index]
        a,b=self.offsets[index:index+2]
        labels=self.labels[a:b].astype(np.int64).tolist()
        require(labels==remove_initial_bos(original['labels']),'View changed content/EOS beyond initial BOS')
        return dict(input_ids=list(original['input_ids']),labels=labels)

class CorrectedLabelCollator:
    """Keep original source masks and padded shapes; shift label targets left once."""
    def __init__(self, stress=False):
        from stage_hi_led_data import LEDCompleteCollator
        self.original=LEDCompleteCollator(5000,768,1024,pad_source_to=5120 if stress else None,
            pad_label_to=768 if stress else None)

    def __call__(self,features):
        import torch
        from stage_i_paired_pilot import verify_mask_batch
        for feature in features:validate_view_labels(feature['labels'])
        # Reconstruct raw labels ONLY for the old trusted padding routine.
        # The tensor returned to the model contains content/EOS, never this BOS.
        raw=[dict(input_ids=list(f['input_ids']),labels=[0]+list(f['labels'])) for f in features]
        batch=self.original(raw)
        labels=torch.full_like(batch['labels'],-100)
        labels[:,:-1]=batch['labels'][:,1:]
        batch['labels']=labels
        verify_mask_batch(batch,features)
        require(not (labels==0).any().item(),'BOS leaked into corrected loss targets')
        return batch

def sanity_result(ids,text,max_length=768):
    """Technical anti-collapse screen, not a semantic quality GO."""
    errors=[]
    if not ids or ids[0]!=2:errors.append('Wrong decoder start')
    if 0 in ids[1:]:errors.append('BOS generated after decoder start')
    if not text.strip():errors.append('Empty decoded output')
    if len(ids)>=max_length or len(ids)<2 or ids[-1]!=2:errors.append('No natural EOS before length cap')
    if len([x for x in ids[1:] if x not in (0,1,2,3,50264)])<16:errors.append('Insufficient ordinary content')
    return dict(status='PASS_TECHNICAL_ONLY' if not errors else 'FAIL_STOP',errors=errors,
        semantic_quality_GO=False,token_count=len(ids),BOS_after_start=ids[1:].count(0))
