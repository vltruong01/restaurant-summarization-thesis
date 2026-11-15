"""Independent offline cross-check of every prepared negative against train labels."""
import json
from stage_i_unlikelihood_v1 import *


def audit():
    from tokenizers import Tokenizer
    cfg=read(CONFIG);hm=read(ROOT/cfg['stage_h_manifest'])
    tm=read(ROOT/hm['tokenizer_manifest'])
    tok=Tokenizer.from_file(str(ROOT/tm['relative_directory']/'tokenizer.json'))
    info=hm['datasets']['train_filtered']
    require(sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256'],'Training input changed')
    rows=[json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
    by_id={r['business_id']:r for r in rows}
    sampling=read(PREP/'TRAINING_MANIFEST.json')
    selected=sampling['business_ids']
    require(selected==sorted(by_id,key=lambda b:stable_key(42,b,'UL_train512'))[:512],'Sampling differs')
    require(all(sorted(o)==list(range(512)) for o in sampling['epochs']) and len(sampling['epochs'])==2,'Order differs')
    require(sampling['epochs']==[sorted(range(512),key=lambda i:stable_key(42+e,selected[i],'UL_epoch_order')) for e in range(2)],'Epoch seed/order differs')
    views=[json.loads(s) for s in (PREP/'NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
    require([v['business_id'] for v in views]==selected,'ID/order mismatch')
    counts={'sentence':0,'word':0,'masked_tokens':0,'empty_visual_negatives':0}
    for i,v in enumerate(views):
        row=by_id[v['business_id']];text=row['target'];positive=tok.encode(text).ids
        require(v['kind']==('sentence' if i%2==0 else 'word'),'Treatment mixture differs')
        require(row['source_sha256']==v['source_sha256'],'Source mapping mismatch')
        require(hashlib.sha256(text.encode()).hexdigest()==v['positive_sha256'],'Positive target mismatch')
        require(hashlib.sha256(json.dumps(positive).encode()).hexdigest()==v['positive_ids_sha256'],'Positive tokens mismatch')
        if not v['eligible']:
            require(v['reason']=='NO_SAFE_BOUNDED_NEGATIVE','Unlogged negative omission');continue
        a,b=v['unit_char_span'];negative=text[:b]+' '+text[a:b]
        require(negative==v['negative_text'] and hashlib.sha256(negative.encode()).hexdigest()==v['negative_sha256'],'Synthetic text drift')
        # Independent body-span check without calling build_negative/candidates.
        review=text.index('[REVIEW_CONSENSUS]\n')+len('[REVIEW_CONSENSUS]\n')
        visual=text.index('[VISUAL_EVIDENCE]\n');vbody=visual+len('[VISUAL_EVIDENCE]\n')
        require(review<=a<b<=visual or vbody<=a<b<=len(text),'Marker/section leakage')
        encoded=tok.encode(negative)
        require(encoded.ids==v['negative_ids'] and len(encoded.ids)<=768,'Negative tokenization/overflow')
        require(encoded.ids[0]==0 and encoded.ids[-1]==2 and not v['penalty_mask'][0] and not v['penalty_mask'][-1],'Special tokens penalized')
        require(len(v['penalty_mask'])==len(encoded.ids) and sum(v['penalty_mask'])==v['penalized_tokens']>0,'Mask mismatch')
        branch=v['branch_token_position'];require(encoded.ids[:branch]==positive[:branch],'Incorrect first prediction prefix')
        for t,mask in enumerate(v['penalty_mask']):
            if not mask:continue
            lo,hi=encoded.offsets[t]
            require(lo>=b+1 and hi>lo and hi<=len(negative) and encoded.ids[t] not in (0,1,2),'Wrong penalty token')
            require(not (encoded.ids[:t]==positive[:t] and t<len(positive) and encoded.ids[t]==positive[t]),'Contradictory CE/UL token')
        require(v=={**build_negative(text,tok,row['business_id'],v['kind']),**{k:v[k] for k in ('business_id','original_row_index','source_sha256','positive_ids_sha256')}},'Non-deterministic view')
        counts[v['kind']]+=1;counts['masked_tokens']+=sum(v['penalty_mask'])
        if not row['selected_photo_ids']:
            require(a<visual,'Photo-less view contains visual copy');counts['empty_visual_negatives']+=1
    ev=read(PREP/'EVALUATION_MANIFEST.json')
    require(not set(ev['business_ids'])&set(selected),'Validation used as training negative')
    return {'status':'PASS_ALL_PREPARED_NEGATIVE_BINDINGS','businesses':len(views),**counts,
            'independent_char_token_and_gold_conflict_checks':True,'LED_model_operations':0,'GPU_operations':0}


if __name__=='__main__':
    result=audit()
    write(PREP/'NEGATIVE_BINDING_AUDIT.json',result)
    print(json.dumps(result))
