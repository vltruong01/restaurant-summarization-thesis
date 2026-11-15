"""Portable CPU analysis; inputs read only, no live database or model dependencies.
Numerical procedures reproduce Stage N's frozen paired() and kappa_quadratic().
"""
import argparse,collections,csv,hashlib,json,re,statistics,sys,importlib.metadata
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def table(p,rows):
    put(p.with_suffix('.json'),rows)
    if rows:
        with p.with_suffix('.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kappa(x,y):
    a=collections.Counter(x);b=collections.Counter(y);n=len(x)
    observed=sum((u-v)**2 for u,v in zip(x,y))/n
    expected=sum(a[i]*b[j]*(i-j)**2 for i in range(1,6) for j in range(1,6))/(n*n)
    return 1-observed/expected if expected else None
def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--package',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);a=pa.parse_args();p=a.package;o=a.output
    assert not o.exists(),'Use a new output directory'
    cfg=read(p/'ANALYSIS_CONFIG.json');rows=read(p/'COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json')['records'];key=read(p/'COLLECTION/ASSIGNMENT_MANIFEST_90.json')
    assert len(rows)==90 and len({r['alias'] for r in rows})==90
    assert [{k:r[k] for k in key[0]} for r in sorted(rows,key=lambda r:r['alias'])]==sorted(key,key=lambda r:r['alias'])
    ids=cfg['business_order'];cs=cfg['criteria'];conds=cfg['conditions'];rs=cfg['raters'];cells={(r['business_id'],r['condition'],r['annotator']):r for r in rows}
    assert len(ids)==15 and len(cells)==90
    for b in ids:
        for c in conds:
            for r in rs:
                row=cells[b,c,r];assert row['status']=='SUBMITTED' and row['submitted_at_utc']
                for crit in cs:
                    v=row[crit]
                    if c=='RO' and crit==cs[-1]:assert v=='N/A'
                    else:assert type(v) is int and 1<=v<=5
    desc=[];business=[];agreement=[];pairs=[];confusion=[]
    for c in conds:
        for crit in cs:
            for panel in ['COMBINED']+rs:
                rr=[x for x in rows if x['condition']==c and (panel=='COMBINED' or x['annotator']==panel)];v=[r[crit] for r in rr if type(r[crit]) is int]
                d=dict(condition=c,criterion=crit,panel=panel,n_businesses=15,n_valid=len(v),n_NA=len(rr)-len(v),mean=None,sample_sd=None,median=None,q25=None,q75=None,iqr=None,**{f'score_{s}':v.count(s) for s in range(1,6)})
                if v:
                    q1,q3=np.quantile(v,[.25,.75],method='linear');d.update(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),median=float(np.median(v)),q25=float(q1),q75=float(q3),iqr=float(q3-q1))
                    assert np.isclose(d['mean'],statistics.mean(v)) and np.isclose(d['sample_sd'],statistics.stdev(v))
                desc.append(d)
            if c=='RO' and crit==cs[-1]:continue
            x=[cells[b,c,'A01'][crit] for b in ids];y=[cells[b,c,'A02'][crit] for b in ids];d=np.array(x)-y
            agreement.append(dict(condition=c,criterion=crit,n_pairs=15,qwk=kappa(x,y),undefined_reason=None if kappa(x,y) is not None else 'Zero expected disagreement',exact_pct=float(100*np.mean(d==0)),within_one_pct=float(100*np.mean(abs(d)<=1)),mean_A01_minus_A02=float(d.mean()),mean_absolute_difference=float(abs(d).mean()),**{f'diff_{i}':int(sum(d==i)) for i in range(-4,5)}))
            for b,v,w in zip(ids,x,y):
                business.append(dict(business_id=b,condition=c,criterion=crit,A01=v,A02=w,two_rater_mean=(v+w)/2))
                pairs.append(dict(business_id=b,condition=c,criterion=crit,A01=v,A02=w,difference=v-w))
            for u in range(1,6):
                for v in range(1,6):confusion.append(dict(condition=c,criterion=crit,A01=u,A02=v,count=sum(a==u and b==v for a,b in zip(x,y))))
    rng=np.random.default_rng(cfg['seed']);indices=rng.integers(0,15,(cfg['bootstrap_draws'],15));signs=rng.choice([-1,1],(cfg['signflip_draws'],15))
    comparisons=[];deltas=[]
    for crit in cs:
        for other in ['RO','U']:
            if other=='RO' and crit==cs[-1]:continue
            f=np.array([np.mean([cells[b,'F',r][crit] for r in rs]) for b in ids]);comp=np.array([np.mean([cells[b,other,r][crit] for r in rs]) for b in ids]);d=f-comp
            lo,hi=np.quantile(d[indices].mean(axis=1),[.025,.975],method='linear')
            pval=(int(np.sum(abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-15))+1)/(cfg['signflip_draws']+1)
            supplementary=other=='RO' and crit in ['coherence','non_redundancy']
            result=dict(contrast='F-'+other,criterion=crit,scope='SUPPLEMENTARY_COMPONENT_BUDGET_DIFFERENCE' if supplementary else 'FROZEN_EXPLORATORY_PLAN',n_businesses=15,F_mean=float(f.mean()),comparator_mean=float(comp.mean()),mean_difference=float(d.mean()),CI95_low=float(lo),CI95_high=float(hi),raw_signflip_p=pval,difference_sample_sd=float(d.std(ddof=1)),higher=int(sum(d>0)),tie=int(sum(d==0)),lower=int(sum(d<0)),A01_difference=float(np.mean([cells[b,'F','A01'][crit]-cells[b,other,'A01'][crit] for b in ids])),A02_difference=float(np.mean([cells[b,'F','A02'][crit]-cells[b,other,'A02'][crit] for b in ids])))
            comparisons.append(result)
            for b,v,w,z in zip(ids,f,comp,d):deltas.append(dict(business_id=b,contrast='F-'+other,criterion=crit,F_mean=float(v),other_mean=float(w),difference=float(z)))
    checks=[]
    from sklearn.metrics import cohen_kappa_score
    for d in agreement:
        if d['qwk'] is not None:
            subset=[r for r in pairs if r['condition']==d['condition'] and r['criterion']==d['criterion']]
            assert np.isclose(d['qwk'],cohen_kappa_score([r['A01'] for r in subset],[r['A02'] for r in subset],labels=[1,2,3,4,5],weights='quadratic'),atol=1e-12,rtol=0)
    # Independent full two-rater tensor reconstruction of every CI.
    for r in comparisons:
        other=r['contrast'].split('-')[1];crit=r['criterion'];t=np.array([[[cells[b,c,q][crit] for q in rs] for c in ['F',other]] for b in ids])
        v=t[indices].mean(axis=(1,3));ci=np.quantile(v[:,0]-v[:,1],[.025,.975],method='linear');assert np.allclose(ci,[r['CI95_low'],r['CI95_high']],atol=1e-12,rtol=0)
        # Separately seeded, chunked original Stage N procedure.
        rrng=np.random.default_rng(cfg['seed']);d=t[:,0].mean(axis=1)-t[:,1].mean(axis=1);rrng.integers(0,15,(20000,15));n=0
        for start in range(0,50000,1000):n+=int(sum(abs((rrng.choice([-1,1],(1000,15))*d).mean(axis=1))>=abs(d.mean())-1e-15))
        assert (n+1)/50001==r['raw_signflip_p']
    comments=[{k:r[k] for k in ['alias','annotator','position','business_id','condition','evidence_notes','issue_span','evidence_source_ids','issue_certainty','serious_grounding_issue']} for r in sorted(rows,key=lambda r:(r['business_id'],r['annotator'],r['condition']))]
    flags=[]
    for panel in ['COMBINED']+rs:
        for c in conds:
            rr=[r for r in rows if r['condition']==c and (panel=='COMBINED' or r['annotator']==panel)]
            flags.append(dict(panel=panel,condition=c,n_records=len(rr),nonempty_notes=sum(bool(r['evidence_notes'].strip()) for r in rr),serious_flags=sum(r['serious_grounding_issue'] for r in rr),rater_marked_confirmed=sum(r['serious_grounding_issue'] and r['issue_certainty']=='confirmed' for r in rr),rater_marked_uncertain=sum(r['serious_grounding_issue'] and r['issue_certainty']=='uncertain' for r in rr)))
    generation=[];outputs=read(p/'DATA/ORIGINAL_OUTPUTS_45.json');identical=[]
    for c in conds:
        for r in read(p/f'DATA/NATIVE/{c}_raw_predictions.json'):
            txt=re.sub(r'^\[[A-Z_]+\]\s*','',r['text']);words=re.findall(r"[\w'-]+",txt.lower());sent=[s.strip().lower() for s in re.split(r'[.!?]+',txt) if s.strip()];ngrams=list(zip(*(words[i:] for i in range(4))));dup=sum(n-1 for n in collections.Counter(sent).values())
            generation.append(dict(business_id=r['business_id'],condition=c,component=r['task'],words=len(words),generated_token_ids=len(r['token_ids']),empty=not bool(txt.strip()),EOS_seen=r['termination']['EOS_seen'],at_length_limit=r['termination']['at_length_limit'],expected_header=r['format_flags']['expected_header'],foreign_header=r['format_flags']['foreign_header'],sentences=len(sent),duplicate_exact_sentences=dup,repeated_4gram_fraction=(len(ngrams)-len(set(ngrams)))/len(ngrams) if ngrams else 0,text_sha256=r['text_sha256']))
    for b in ids:
        rr={r['condition']:r for r in outputs if r['business_id']==b}
        for c in ['RO','U']:
            identical.append(dict(business_id=b,contrast='F-'+c,review_text_identical=rr['F']['component_texts']['review']==rr[c]['component_texts']['review']))
    comparison_yelp=[];yd=read(p/'YELP_TEST30/TABLES/DESCRIPTIVE_STATISTICS.json');yp=read(p/'YELP_TEST30/TABLES/PAIRED_COMPARISONS.json')
    for r in comparisons:
        y=next(v for v in yp if v['contrast']==r['contrast'] and v['criterion']==r['criterion'])
        comparison_yelp.append(dict(contrast=r['contrast'],criterion=r['criterion'],TripAdvisor_n=15,TripAdvisor_difference=r['mean_difference'],TripAdvisor_low=r['CI95_low'],TripAdvisor_high=r['CI95_high'],Yelp_n=30,Yelp_difference=y['mean_paired_difference'],Yelp_low=y['CI95_low'],Yelp_high=y['CI95_high'],cross_dataset_test='NOT_RUN; different cohorts, no pooled inference'))
    o.mkdir(parents=True)
    for name,v in [('DESCRIPTIVES',desc),('AGREEMENT',agreement),('AGREEMENT_PAIRS',pairs),('AGREEMENT_CONFUSIONS',confusion),('BUSINESS_MEANS',business),('PAIRED_COMPARISONS',comparisons),('BUSINESS_DIFFERENCES',deltas),('RATER_COMMENTS_VERBATIM',comments),('PARTICIPANT_FLAG_COUNTS',flags),('GENERATION_DIAGNOSTICS',generation),('IDENTICAL_REVIEW_TEXTS',identical),('YELP_COMPARISON',comparison_yelp)]:table(o/name,v)
    put(o/'NUMERICAL_VERIFICATION.json',dict(status='PASS',records=90,businesses=15,numeric_scores=510,structural_NA=30,descriptive_rows=len(desc),condition_criterion_QWK_cells=len(agreement),paired_comparisons=len(comparisons),primary_plan_comparisons=9,supplementary_comparisons=2,all_QWK_crosschecked_sklearn=True,all_CI_full_tensor_reconstructed=True,all_pvalues_original_chunked_Stage_N_reconstructed=True,descriptive_means_and_SD_crosschecked_stdlib=True,resample_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),sign_matrix_sha256=hashlib.sha256(signs.astype('<i8').tobytes()).hexdigest(),ratings_sha256=sha(p/'COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json'),excluded_records=[]))
    put(o/'RUNTIME.json',dict(python=sys.version,**{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','matplotlib']},GPU_work=False,API_work=False))
    print(json.dumps(dict(means=[r for r in desc if r['panel']=='COMBINED'],comparisons=comparisons,flags=flags),indent=2))
if __name__=='__main__':main()
