"""Prepared CPU statistics. Requires two complete NEW-session exports; never fills missing ratings."""
import argparse,collections,csv,hashlib,json,statistics,itertools
from pathlib import Path
import numpy as np
from validate_stage_p_human_exports_v1 import validate,CRITERIA,read
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def table(p,rows):
    put(p.with_suffix('.json'),rows)
    with p.with_suffix('.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def qwk(x,y):
    a=collections.Counter(x);b=collections.Counter(y);n=len(x);observed=sum((u-v)**2 for u,v in zip(x,y))/n;expected=sum(a[u]*b[v]*(u-v)**2 for u in range(1,6) for v in range(1,6))/(n*n)
    return 1-observed/expected if expected else None
def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['raw_exact_signflip_p']);running=0
    for j,i in enumerate(order):running=max(running,min(1,(len(rows)-j)*rows[i]['raw_exact_signflip_p']));rows[i]['holm_p_within_family']=running
def analyze(package,exports,out,allow_fixture=False):
    rows,validation=validate(package,exports,allow_fixture);cfg=read(package/'COORDINATOR_ONLY/ANALYSIS_PLAN.json');ids=cfg['restaurant_order'];raters=cfg['raters'];systems=cfg['systems'];n=len(ids)
    if not allow_fixture:assert n==15
    assert not out.exists(),'Choose a fresh output directory';cells={(r['business_id'],r['annotator'],r['system']):r for r in rows};assert len(cells)==n*6
    rng=np.random.default_rng(cfg['seed']);indices=rng.integers(0,n,(cfg['bootstrap_draws'],n));signs=np.array(list(itertools.product([-1,1],repeat=n)),dtype=np.int8)
    desc=[];agreement=[];differences=[];comparisons=[]
    for c in CRITERIA:
        for s in systems:
            for panel in ['COMBINED']+raters:
                scores=[r[c] for r in rows if r['system']==s and (panel=='COMBINED' or r['annotator']==panel)]
                desc.append(dict(system=s,criterion=c,panel=panel,n_restaurants=n,n_ratings=len(scores),mean=statistics.mean(scores),sample_SD=statistics.stdev(scores),median=statistics.median(scores),**{'score_'+str(k):scores.count(k) for k in range(1,6)}))
            x=[cells[b,raters[0],s][c] for b in ids];y=[cells[b,raters[1],s][c] for b in ids]
            agreement.append(dict(system=s,criterion=c,restaurant_pairs=n,quadratic_weighted_kappa=qwk(x,y),exact_agreement_fraction=sum(a==b for a,b in zip(x,y))/n,within_one_fraction=sum(abs(a-b)<=1 for a,b in zip(x,y))/n,mean_absolute_difference=statistics.mean(abs(a-b) for a,b in zip(x,y)),mean_first_minus_second=statistics.mean(a-b for a,b in zip(x,y))))
        for pair in cfg['contrasts']:
            left,right=pair['left'],pair['right'];d=np.array([[(cells[b,r,left][c]-cells[b,r,right][c]) for r in raters] for b in ids],dtype=float);cluster=d.mean(axis=1);lo,hi=np.quantile(cluster[indices].mean(axis=1),[.025,.975],method='linear');p=float(np.mean(abs((signs*cluster).mean(axis=1))>=abs(cluster.mean())-1e-15))
            duplicate=(left==systems[1] and right==systems[2] and c in cfg['Review_only_criteria'])
            comparisons.append(dict(left=left,right=right,criterion=c,family=pair['family'],n_paired_restaurants=n,mean_difference=float(cluster.mean()),pointwise_CI95_low=float(lo),pointwise_CI95_high=float(hi),raw_exact_signflip_p=p,holm_p_within_family=None,first_rater_difference=float(d[:,0].mean()),second_rater_difference=float(d[:,1].mean()),positive_restaurants=int(sum(cluster>0)),tied_restaurants=int(sum(cluster==0)),negative_restaurants=int(sum(cluster<0)),identical_Qwen_Review_interpretation='Same generated Review text; rating differences cannot establish Review generation improvement' if duplicate else 'No special duplicate-content claim'))
            for b,v in zip(ids,d):differences.append(dict(business_id=b,left=left,right=right,criterion=c,first_rater_difference=float(v[0]),second_rater_difference=float(v[1]),two_rater_mean_difference=float(v.mean())))
    for family in ['PRIMARY','ADDITIONAL_EXPLORATORY']:holm([r for r in comparisons if r['family']==family])
    out.mkdir(parents=True);table(out/'DESCRIPTIVES',desc);table(out/'AGREEMENT',agreement);table(out/'PAIRED_COMPARISONS',comparisons);table(out/'BUSINESS_RATER_DIFFERENCES',differences)
    comments=[{k:r[k] for k in ['business_id','system','annotator','evidence_notes','serious_grounding_issue','issue_span','evidence_source_ids','issue_certainty']} for r in rows];put(out/'RATER_COMMENTS_VERBATIM.json',comments)
    put(out/'VALIDATION.json',validation);put(out/'REPRODUCIBILITY.json',dict(seed=cfg['seed'],bootstrap_draws=cfg['bootstrap_draws'],paired_unit='Restaurant cluster; both evaluators and all systems retained together',exact_sign_assignments=len(signs),bootstrap_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in exports},no_old_ratings=True,mean_differences_on_ordinal_scale='Descriptive numeric summaries; no interval-scale guarantee',CI='Pointwise percentile bootstrap, not multiplicity-adjusted',tests='Two-sided exact cluster sign-flip tests are exploratory; exchangeability/symmetry assumptions are not guaranteed by observational model outputs. Holm within12 primary/6 additional contrasts.',inference_limits='n15 restaurants,2 convenience evaluators,prior source/model-output exposure possible. No30-independent-rater or90-independent-item inference. Qwen Review shared; three Review-only criteria do not create two different generated texts. Coherence/non-redundancy can differ across combined outputs.',independently_confirmed_grounding_errors=False))
    return dict(status='COMPLETE_FROM_ACTUAL_SUPPLIED_EXPORTS' if not allow_fixture else 'SYNTHETIC_SOFTWARE_TEST_ONLY',descriptive_rows=len(desc),agreement_cells=len(agreement),comparisons=len(comparisons),restaurants=n)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--exports',nargs=2,type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.package,a.exports,a.output),indent=2))
