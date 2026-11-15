"""Experimental CPU-only, deletion/selection-only source-consistency pipeline.

Never reads Teacher targets, approved references, photos or trained LED weights.
No text generator: retained sentences are exact spans of the raw LED prediction.
NLI scores are fallible auxiliary predictions, not verified factual labels.
"""
import json,math,re,time
from pathlib import Path
from stage_i_repetition_guard_v1 import _sentence_ends

CANONICAL=('[REVIEW_CONSENSUS]','[VISUAL_EVIDENCE]')
ALLOWED_INPUT_KEYS={'business_id','subset','source','source_sha256','selected_photo_ids','raw_prediction','raw_prediction_sha256'}


def require(value,message):
    if not value:raise ValueError(message)


def units(text):
    """Exact punctuation-terminated spans plus explicit incomplete remainder."""
    out=[];start=0
    for end in _sentence_ends(text):
        a=start;b=end.end()
        while a<b and text[a].isspace():a+=1
        if a<b:out.append({'text':text[a:b],'start':a,'end':b,'complete':True})
        start=b
    if text[start:].strip():
        a=start
        while a<len(text) and text[a].isspace():a+=1
        out.append({'text':text[a:],'start':a,'end':len(text),'complete':False})
    return out


def prediction_units(raw):
    # Position-based serialization repair is explicit, not a model improvement.
    visual=raw.find(CANONICAL[1]);cut=len(raw) if visual<0 else visual
    result={}
    for domain,a,b in [('review',0,cut),('visual',cut+len(CANONICAL[1]) if visual>=0 else len(raw),len(raw))]:
        segment=raw[a:b]
        masked=re.sub(r'\[[A-Z_]+\]',lambda m:' '*len(m.group()),segment)
        result[domain]=[{**u,'start':u['start']+a,'end':u['end']+a,'domain':domain} for u in units(masked)]
        for u in result[domain]:require(raw[u['start']:u['end']]==u['text'],'Raw sentence span changed')
    return result


def source_records(source,photos):
    require('[REVIEWS]' in source and '[VISUAL_CONTEXT]' in source,'Frozen source template absent')
    a=source.index('[REVIEWS]')+len('[REVIEWS]');b=source.index('[VISUAL_CONTEXT]')
    matches=list(re.finditer(r'\[(REV_\d+)\][^\n]*\n',source[a:b]))
    reviews=[]
    for i,m in enumerate(matches):
        start=a+m.end();end=a+matches[i+1].start() if i+1<len(matches) else b
        while start<end and source[start].isspace():start+=1
        while end>start and source[end-1].isspace():end-=1
        reviews.append({'marker':m.group(1),'text':source[start:end],'source_start':start,'source_end':end,'domain':'review'})
    v=b+len('[VISUAL_CONTEXT]');cm=list(re.finditer(r'\[(IMG_CAP_\d+)\]\s*\n',source[v:]))
    require(len(cm)==len(photos),'Caption/photo alignment mismatch')
    captions=[]
    for i,m in enumerate(cm):
        start=v+m.end();end=v+cm[i+1].start() if i+1<len(cm) else len(source)
        while start<end and source[start].isspace():start+=1
        while end>start and source[end-1].isspace():end-=1
        captions.append({'marker':m.group(1),'photo_id':photos[i],'text':source[start:end],
                         'source_start':start,'source_end':end,'domain':'visual'})
    return reviews,captions


def pack_evidence(records,tokenizer,max_tokens):
    """Whole-sentence windows; never silently truncate an oversized sentence."""
    packed=[];excluded=[]
    for record in records:
        chunks=units(record['text']);group=[]
        def flush():
            if group:
                a,b=group[0]['start'],group[-1]['end'];text=record['text'][a:b]
                packed.append({**record,'text':text,'source_start':record['source_start']+a,'source_end':record['source_start']+b})
                group.clear()
        for u in chunks:
            n=len(tokenizer.encode(u['text'],add_special_tokens=False))
            if n>max_tokens:
                flush();excluded.append({**record,'text':u['text'],'reason':'OVERSIZED_EVIDENCE_SENTENCE_EXPLICITLY_NOT_SCORED','tokens':n});continue
            if group:
                prospective=record['text'][group[0]['start']:u['end']]
                if len(tokenizer.encode(prospective,add_special_tokens=False))>max_tokens:flush()
            group.append(u)
        flush()
    return packed,excluded


def serialize(review,visual):
    return CANONICAL[0]+'\n'+' '.join(u['text'] for u in review)+'\n\n'+CANONICAL[1]+'\n'+' '.join(u['text'] for u in visual)


class CPUScorers:
    def __init__(self,c,output):
        import torch
        from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
        self.torch=torch;self.c=c;self.output=Path(output);self.pairs={};self.emb_cache={}
        self.started=time.perf_counter();self.model_forward_batches=0
        torch.set_num_threads(c['CPU_threads'])
        self.et=AutoTokenizer.from_pretrained(c['embedding_directory'],local_files_only=True)
        self.em=AutoModel.from_pretrained(c['embedding_directory'],local_files_only=True,use_safetensors=True).eval().requires_grad_(False).cpu()
        self.nt=AutoTokenizer.from_pretrained(c['NLI_directory'],use_fast=False,local_files_only=True)
        self.nm=AutoModelForSequenceClassification.from_pretrained(c['NLI_directory'],local_files_only=True,use_safetensors=True).eval().requires_grad_(False).cpu()
        require(self.nm.config.id2label=={0:'contradiction',1:'entailment',2:'neutral'},'Unexpected NLI class order')
        require(all(p.device.type=='cpu' for m in (self.em,self.nm) for p in m.parameters()),'Only CPU scoring is authorized')

    def check_budget(self):
        require(time.perf_counter()-self.started<=self.c['CPU_seconds_max'],'CPU scoring time ceiling exceeded')
        require(len(self.pairs)<=self.c['NLI_pairs_max'],'NLI pair budget exceeded')

    def embeds(self,texts):
        torch=self.torch;fresh=list(dict.fromkeys(t for t in texts if t not in self.emb_cache))
        for start in range(0,len(fresh),self.c['embedding_batch']):
            self.check_budget();batch=fresh[start:start+self.c['embedding_batch']]
            inputs=self.et(batch,padding=True,truncation=False,return_tensors='pt')
            require(inputs['input_ids'].shape[1]<=256,'Embedding input would exceed declared short-text scope')
            with torch.inference_mode():
                hidden=self.em(**inputs).last_hidden_state;mask=inputs['attention_mask'].unsqueeze(-1)
                vec=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vec=torch.nn.functional.normalize(vec,p=2,dim=1)
            require(torch.isfinite(vec).all().item(),'Nonfinite embedding')
            self.model_forward_batches+=1
            for t,v in zip(batch,vec):self.emb_cache[t]=v
        return torch.stack([self.emb_cache[t] for t in texts])

    def nli(self,pairs):
        torch=self.torch;fresh=list(dict.fromkeys(p for p in pairs if p not in self.pairs))
        require(len(self.pairs)+len(fresh)<=self.c['NLI_pairs_max'],'NLI pair budget exceeded before execution')
        for start in range(0,len(fresh),self.c['NLI_batch']):
            self.check_budget();batch=fresh[start:start+self.c['NLI_batch']]
            inputs=self.nt([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=False,return_tensors='pt')
            require(inputs['input_ids'].shape[1]<=512,'NLI pair exceeds512; no truncation/fallback')
            with torch.inference_mode():logits=self.nm(**inputs).logits.float();probs=logits.softmax(-1)
            require(torch.isfinite(logits).all().item() and torch.isfinite(probs).all().item(),'Nonfinite NLI scores')
            self.model_forward_batches+=1
            with (self.output/'NLI_RAW_SCORES.jsonl').open('a',encoding='utf-8') as stream:
                for pair,l,p in zip(batch,logits.tolist(),probs.tolist()):
                    value={'contradiction':p[0],'entailment':p[1],'neutral':p[2],'raw_logits':l}
                    self.pairs[pair]=value
                    stream.write(json.dumps({'premise':pair[0],'hypothesis':pair[1],**value},ensure_ascii=False,allow_nan=False)+'\n')
        return [self.pairs[p] for p in pairs]


def select(record,scorer,c,led_tokenizer):
    require(set(record)==ALLOWED_INPUT_KEYS,'Unexpected input fields; references/Teacher labels prohibited')
    raw=record['raw_prediction'];candidates=prediction_units(raw)
    reviews,captions=source_records(record['source'],record['selected_photo_ids'])
    evidence_reviews,excluded=pack_evidence(reviews,scorer.et,c['evidence_embedding_tokens'])
    decisions=[];retained={};evidence_by_domain={'review':evidence_reviews,'visual':captions}
    for domain in ('review','visual'):
        ev=evidence_by_domain[domain];accepted=[]
        for index,u in enumerate(candidates[domain]):
            row={**u,'candidate_index':index,'decision':None,'evidence':[]}
            if domain=='visual' and not captions:
                row['decision']='DROP_ZERO_CAPTION_VISUAL';decisions.append(row);continue
            if not u['complete']:
                row['decision']='DROP_INCOMPLETE_FRAGMENT';decisions.append(row);continue
            if len(scorer.et.encode(u['text'],add_special_tokens=True))>256:
                row['decision']='DROP_OVERSIZED_CANDIDATE_NO_TRUNCATION';decisions.append(row);continue
            if not ev:
                row['decision']='DROP_NO_SCORABLE_EVIDENCE';decisions.append(row);continue
            emb=scorer.embeds([u['text']]+[e['text'] for e in ev]);sims=(emb[1:]@emb[0]).tolist()
            order=sorted(range(len(ev)),key=lambda k:(-sims[k],k))
            chosen=order[:c['review_retrieval_top_k']] if domain=='review' else order
            premises=[ev[k]['text'] for k in chosen]
            # A multi-review premise can support an opinion contrast. It cannot
            # supply visual relations: visual comparisons always use ONE caption.
            if domain=='review' and len(chosen)>1:
                pieces=[];combined_ids=[]
                for k in chosen:
                    candidate='\n'.join(pieces+[ev[k]['text']])
                    if len(scorer.nt.encode(candidate,u['text'],add_special_tokens=True,truncation=False))<=512:
                        pieces.append(ev[k]['text']);combined_ids.append(k)
                if len(pieces)>1:premises.append('\n'.join(pieces))
            # Explicitly exclude overlong pairs rather than changing the text.
            valid=[(p,u['text']) for p in premises if len(scorer.nt.encode(p,u['text'],add_special_tokens=True,truncation=False))<=512]
            if not valid:
                row['decision']='DROP_NO_LENGTH_COMPATIBLE_EVIDENCE';decisions.append(row);continue
            scores=scorer.nli(valid);best=max(range(len(scores)),key=lambda k:scores[k]['entailment'])
            threshold=c[domain+'_entailment_min'];margin=c[domain+'_entailment_margin']
            row.update({'best_NLI':scores[best],'best_premise_verbatim':valid[best][0],
                        'retrieved_max_cosine':max(sims),'scorer_label_not_verified_fact':True})
            row['evidence']=[{**ev[k],'retrieval_cosine':sims[k]} for k in chosen if ev[k]['text'] in valid[best][0]]
            if scores[best]['entailment']<threshold or scores[best]['entailment']-scores[best]['contradiction']<margin:
                row['decision']='DROP_LOW_MODEL_SOURCE_SUPPORT';decisions.append(row);continue
            row['decision']='SOURCE_SUPPORTED_CANDIDATE_BY_MODEL_NOT_GROUND_TRUTH';accepted.append(row)
        selected=[];remaining=list(accepted)
        while remaining and len(selected)<c[domain+'_sentences_max']:
            vec=scorer.embeds([r['text'] for r in remaining+selected]);n=len(remaining)
            rankings=[]
            for i,u in enumerate(remaining):
                redundancy=float((vec[n:]@vec[i]).max()) if selected else 0.
                score=c['MMR_lambda']*u['retrieved_max_cosine']-(1-c['MMR_lambda'])*redundancy
                rankings.append((score,-u['start'],i))
            _,_,index=max(rankings);row=remaining.pop(index)
            duplicate=None
            if selected:
                values=scorer.embeds([row['text']]+[s['text'] for s in selected]);sims=(values[1:]@values[0]).tolist()
                for other,sim in zip(selected,sims):
                    if sim<c['semantic_cosine_min']:continue
                    pairs=[(row['text'],other['text']),(other['text'],row['text'])]
                    if any(len(scorer.nt.encode(a,b,add_special_tokens=True,truncation=False))>512 for a,b in pairs):continue
                    entail=scorer.nli(pairs)
                    if min(s['entailment'] for s in entail)>=c['mutual_entailment_min']:
                        duplicate={'kept_candidate_index':other['candidate_index'],'cosine':sim,'mutual_NLI':entail};break
            if duplicate:
                row['decision']='DROP_MODEL_MUTUAL_ENTAILMENT_REDUNDANCY';row['redundancy_evidence']=duplicate
            else:
                trial=sorted(selected+[row],key=lambda s:s['start'])
                rv=trial if domain=='review' else retained['review'];vv=[] if domain=='review' else trial
                if 1+len(led_tokenizer.encode(serialize(rv,vv),add_special_tokens=True))>c['output_LED_tokens_max']:
                    row['decision']='DROP_WHOLE_SENTENCE_OUTPUT_BUDGET_NO_TRUNCATION'
                else:row['decision']='KEEP_UNCHANGED_RAW_SENTENCE';selected.append(row)
            decisions.append(row)
        for row in remaining:row['decision']='DROP_MMR_SENTENCE_COUNT_BUDGET';decisions.append(row)
        retained[domain]=sorted(selected,key=lambda r:r['start'])
    text=serialize(retained['review'],retained['visual'])
    require(not (not captions and retained['visual']),'Zero-caption invariant')
    require(all(u['text']==raw[u['start']:u['end']] for group in retained.values() for u in group),'A sentence was rewritten')
    token_ids=[2]+led_tokenizer.encode(text,add_special_tokens=True)
    require(len(token_ids)<=c['output_LED_tokens_max'],'Output silently exceeds budget')
    return {'business_id':record['business_id'],'subset':record['subset'],'text':text,'serialized_token_ids':token_ids,
        'origin':'EXPLICIT_POSTPROCESSING_NOT_NEW_LED_GENERATION','source_sha256':record['source_sha256'],
        'raw_prediction_sha256':record['raw_prediction_sha256'],'selected_sentences':retained,'decisions':decisions,
        'excluded_source_evidence':excluded,'original_markers':re.findall(r'\[[A-Z_]+\]',raw),
        'canonical_serialization_applied':True,'new_content_tokens_generated':0,'human_verified':False,
        'serialized_EOS_and_headers_are_not_new_model_EOS_or_structure_success':True,
        'coverage_warning':'LESS_THAN_THREE_REVIEW_SENTENCES_RETAINED' if len(retained['review'])<3 else 'NO_CORE_COVERAGE_GUARANTEE',
        'photo_verification':'NOT_PERFORMED_BY_THIS_TEXT_SCORER','empty_visual_required':not captions}
