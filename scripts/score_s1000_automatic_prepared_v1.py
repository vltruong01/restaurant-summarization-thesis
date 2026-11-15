"""Prepared component-level scorer. Never invoked on TEST during preparation."""
import argparse,importlib.metadata,collections
from pathlib import Path
from s1000_final_eval_core_v1 import *
from validate_s1000_human_references_v1 import validate_all

def strip_header(text,task):
    marker='[REVIEW_CONSENSUS]\n' if task=='review' else '[VISUAL_EVIDENCE]\n'
    return text[len(marker):] if text.startswith(marker) else text

def descriptive_score(candidate,reference,source):
    from rouge_score.rouge_scorer import RougeScorer
    rouge=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True).score(reference,candidate)
    d=diagnostics(candidate,source)
    for name,s in rouge.items():
        for component in ['precision','recall','fmeasure']:d[name+'_'+component]=float(getattr(s,component))
    rt=set(words(reference));ct=set(words(candidate));d['unique_reference_word_coverage_proxy']=len(rt&ct)/len(rt) if rt else None
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--execute-approved',action='store_true')
    p.add_argument('--researcher-release',type=Path);p.add_argument('--references',type=Path);p.add_argument('--bert-model',type=Path)
    p.add_argument('--results',type=Path,help='Parent with RO/U/F/S result folders');p.add_argument('--output',type=Path)
    a=p.parse_args();lock=read(a.preparation/'METRIC_LOCK.json')
    if not a.execute_approved:print('Preparation only. No TEST metrics computed. Requires approved human references, four completed inference folders and separate release.');return
    require(a.researcher_release and a.researcher_release.is_file() and a.references and a.results and a.output and a.bert_model,'Execution dependencies missing')
    require(all(importlib.metadata.version(k)==v for k,v in lock['packages'].items()),'Metric package pins differ')
    refs,report=validate_all(a.preparation/'TEST100_REFERENCE_ANNOTATION_PACKAGE',a.references,approved=True)
    require(not report['errors'],'Human references not approved')
    require(file_hash(a.bert_model/'pytorch_model.bin')==lock['bertscore']['weights_sha256'],'Scorer model weights differ')
    for f,h in lock['bertscore']['small_file_sha256'].items():require(file_hash(a.bert_model/f)==h,'Metric tokenizer/config differs')
    import torch
    from bert_score import BERTScorer
    # Explicit CPU FP32 prevents accidental unapproved GPU use; batch/order fixed.
    scorer=BERTScorer(model_type=str(a.bert_model),num_layers=17,batch_size=8,nthreads=1,all_layers=False,idf=False,device='cpu',lang='en',rescale_with_baseline=False,use_fast_tokenizer=False)
    scorer._tokenizer.model_max_length=512
    byref={r['business_id']:r for r in refs};plan=read(a.preparation/'TEST100_INFERENCE_PLAN.json');calls={(c['condition'],c['business_id'],c['task']):c for c in plan['calls']}
    raw=[];receipt_bindings={}
    for condition in ['RO','U','F','S']:
        folder=a.results/condition;receipt=read(folder/'COMPLETION.json');pp=folder/'raw_predictions.json'
        require(receipt['condition']==condition and receipt['businesses']==100 and receipt['raw_predictions_sha256']==file_hash(pp),'Inference receipt mismatch')
        rows=read(pp);expected={(c['business_id'],c['task']) for c in plan['calls'] if c['condition']==condition}
        require(len(rows)==len(expected) and {(r['business_id'],r['task']) for r in rows}==expected,'Missing/duplicate components')
        receipt_bindings[condition]=dict(receipt_sha256=file_hash(folder/'COMPLETION.json'),predictions_sha256=file_hash(pp))
        for r in rows:
            c=calls[condition,r['business_id'],r['task']]
            require(all(r[k]==c[k] for k in ['source_sha256','input_ids_sha256','checkpoint_sha256']) and text_hash(r['text'])==r['text_sha256'],'Output identity mismatch')
            raw.append(r)
    require(len(raw)==700,'Exactly700 logical components')
    raw_by={(r['condition'],r['business_id'],r['task']):r for r in raw}
    for c in plan['calls']:
        if c['reuse_condition']:
            r=raw_by[c['condition'],c['business_id'],c['task']];f=raw_by['F',c['business_id'],c['task']]
            require(r['text']==f['text'] and r['token_ids']==f['token_ids'] and r.get('independent_generation') is False,'Reused F output not identical')
    rows=[];candidates=[];references=[]
    for r in raw:
        c=calls[r['condition'],r['business_id'],r['task']];candidate=strip_header(r['text'],r['task']);ref=byref[r['business_id']]['review_consensus' if r['task']=='review' else 'visual_evidence']
        # Explicitly disclose BERTScore's fixed prefix window; ROUGE/diagnostics use full text.
        enc=lambda text:scorer._tokenizer.encode(text.strip(),add_special_tokens=True,add_prefix_space=True,truncation=False)
        cl,rl=len(enc(candidate)),len(enc(ref))
        d=descriptive_score(candidate,ref,c['source']);rows.append(dict(business_id=r['business_id'],condition=r['condition'],task=r['task'],output_sha256=r['text_sha256'],reference_content_sha256=digest(reference_content(byref[r['business_id']])),
            malformed_header=not r['text'].startswith('[REVIEW_CONSENSUS]\n' if r['task']=='review' else '[VISUAL_EVIDENCE]\n'),
            cross_component_marker=('[VISUAL_EVIDENCE]' if r['task']=='review' else '[REVIEW_CONSENSUS]') in r['text'],
            generation_tokens=len(r['token_ids']),at_generation_cap=len(r['token_ids'])>=768,EOS_seen=2 in r['token_ids'][1:],
            bert_candidate_tokens_untruncated=cl,bert_reference_tokens_untruncated=rl,bert_candidate_prefix_truncated=cl>512,bert_reference_prefix_truncated=rl>512,**d))
        candidates.append(candidate);references.append(ref)
    torch.set_num_threads(1)
    with torch.no_grad():P,R,F=scorer.score(candidates,references,batch_size=8,verbose=False)
    for i,r in enumerate(rows):r.update(bertscore_precision=float(P[i]),bertscore_recall=float(R[i]),bertscore_f1=float(F[i]))
    by={(r['condition'],r['business_id'],r['task']):r for r in rows};ids=sorted(byref)
    main=[]
    # Four confirmatory contrasts fixed in advance; F-S Review is structural reuse.
    contrasts=[('review','RO'),('review','U'),('visual','U'),('visual','S')]
    for task,other in contrasts:
        result=paired([by['F',b,task]['bertscore_f1'] for b in ids],[by[other,b,task]['bertscore_f1'] for b in ids])
        main.append(dict(task=task,contrast='F-'+other,metric='bertscore_f1',**result))
    for r,padj in zip(main,holm([r['signflip_two_sided_p'] for r in main])):r['holm_p_four_primary_tests']=padj
    secondary=[]
    for task,other in contrasts:
        for metric in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','words','repeated_3gram_fraction','unique_reference_word_coverage_proxy']:
            secondary.append(dict(task=task,contrast='F-'+other,metric=metric,interpretation='Secondary descriptive; no confirmatory significance claim',**paired([by['F',b,task][metric] for b in ids],[by[other,b,task][metric] for b in ids],B=20000,P=1000)))
    active=[r['recipient_id'] for r in read(a.preparation/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings'] if r['visual_source_changed']]
    intervention=paired([by['F',b,'visual']['bertscore_f1'] for b in active],[by['S',b,'visual']['bertscore_f1'] for b in active])
    aggregate=[]
    numeric=['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1','words','repeated_3gram_fraction','unique_reference_word_coverage_proxy']
    for c,t in sorted({(r['condition'],r['task']) for r in rows}):
        rr=[r for r in rows if r['condition']==c and r['task']==t]
        aggregate.append(dict(condition=c,task=t,businesses=len(rr),
            bert_candidate_truncated_count=sum(r['bert_candidate_prefix_truncated'] for r in rr),
            bert_reference_truncated_count=sum(r['bert_reference_prefix_truncated'] for r in rr),
            bert_candidate_truncated_fraction=sum(r['bert_candidate_prefix_truncated'] for r in rr)/len(rr),
            bert_reference_truncated_fraction=sum(r['bert_reference_prefix_truncated'] for r in rr)/len(rr),
            **{m:sum(r[m] for r in rr)/len(rr) for m in numeric}))
    a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/'PER_BUSINESS_COMPONENT_METRICS.csv',rows)
    write(a.output/'AGGREGATE_METRICS.json',aggregate);write(a.output/'PAIRED_ANALYSIS.json',dict(primary=main,secondary=secondary,F_minus_S_visual_active_only=intervention,
        active_ids=active,full100_primary_reporting=True,RO_visual_score=None,F_S_review_structural_reuse=True,
        automatic_similarity_is_not_verified_factuality=True))
    write(a.output/'AUTOMATIC_EVALUATION_COMPLETION.json',dict(businesses=100,conditions=4,logical_components=700,model_independent_common_reference=True,
        reference_validation=report,input_bindings=receipt_bindings,reference_file_sha256={p.name:file_hash(p) for p in a.references.glob('*.json')},
        metric_lock_sha256=file_hash(a.preparation/'METRIC_LOCK.json'),release_sha256=file_hash(a.researcher_release),metrics_file_sha256=file_hash(a.output/'PER_BUSINESS_COMPONENT_METRICS.csv'),
        human_scoring_performed=False,limitations=['Text metrics and lexical proxies do not verify factuality or original-image grounding.','Prefix BERTScore is limited to512 tokens; full-text ROUGE and lengths retained.','TEST100 has confirmed partial visual-evidence reuse.']))

if __name__=='__main__':main()
