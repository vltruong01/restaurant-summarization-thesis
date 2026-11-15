"""Later analysis of complete genuine two-rater records; no scores are generated."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from s1000_final_eval_core_v1 import read, file_hash, paired
from stage_n_human_evaluation_v1 import CRITERIA, RATERS, live_release_valid, direct_rater_access_valid

def kappa_quadratic(x,y):
    """Fixed categories 1..5, weights (i-j)^2/16; Cohen 1-Do/De.

    Primary reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
    Missing pairs are not imputed. Caller uses complete records only.
    """
    if not x or len(x)!=len(y):return {'value':None,'reason':'No complete paired ratings','n':0}
    if any(type(v) is not int or not 1<=v<=5 for v in x+y):raise ValueError('Scores must be integers 1-5')
    n=len(x);a=Counter(x);b=Counter(y)
    observed=sum((u-v)**2 for u,v in zip(x,y))/n
    expected=sum(a[i]*b[j]*(i-j)**2 for i in range(1,6) for j in range(1,6))/(n*n)
    return {'value':1-observed/expected if expected else None,'reason':None if expected else 'Undefined: zero expected disagreement (constant category)', 'n':n,'fixed_categories':[1,2,3,4,5]}

def analyze(payload,key,B=20000,P=50000):
    rows=payload['records'];kk={r['alias']:r for r in key}
    if len(rows)!=240 or len(kk)!=240 or len({r['alias'] for r in rows})!=240 or {r['alias'] for r in rows}!=set(kk):raise ValueError('Require exactly the complete 240 unique assignments')
    g=defaultdict(dict);comment_rows=[];na=0
    for r in rows:
        k=kk[r['alias']]
        if any(r[f]!=k[f] for f in ['annotator','business_id','condition']) or r['status']!='SUBMITTED' or not r.get('submitted_at_utc'):raise ValueError('Assignment or actual submission binding mismatch')
        for criterion in CRITERIA:
            v=r[criterion]
            if k['condition']=='RO' and criterion==CRITERIA[-1]:
                if v!='N/A':raise ValueError('RO Visual must be N/A')
                na+=1;continue
            if type(v) is not int or not 1<=v<=5:raise ValueError('Missing/invalid applicable score; never impute')
            g[k['condition'],criterion,k['business_id']][r['annotator']]=v
        if r.get('evidence_notes') or r.get('issue_span'):
            comment_rows.append({f:r.get(f) for f in ['alias','annotator','business_id','condition','evidence_notes','issue_span','evidence_source_ids','issue_certainty','serious_grounding_issue']})
    descriptions=[];agreements=[];means={};business_rows=[]
    for c in ['RO','U','F','S']:
        for criterion in CRITERIA:
            if c=='RO' and criterion==CRITERIA[-1]:
                descriptions.append({'condition':c,'criterion':criterion,'applicable':False,'valid':0,'missing':0,'na':60,'mean':None,'median':None,'distribution':{}});continue
            units=sorted((b,v) for (condition,crit,b),v in g.items() if condition==c and crit==criterion)
            if len(units)!=30 or any(set(v)!=set(RATERS) for b,v in units):raise ValueError('Incomplete paired two-rater business panel')
            values=[v[r] for b,v in units for r in RATERS]
            per={r:{'mean':statistics.mean(v[r] for b,v in units),'median':statistics.median(v[r] for b,v in units),'valid':30,'missing':0,'distribution':dict(sorted(Counter(v[r] for b,v in units).items()))} for r in RATERS}
            descriptions.append({'condition':c,'criterion':criterion,'applicable':True,'mean':statistics.mean(values),'median':statistics.median(values),'valid':60,'missing':0,'na':0,'distribution':dict(sorted(Counter(values).items())),'per_rater':per})
            agreements.append({'condition':c,'criterion':criterion,**kappa_quadratic([v['A01'] for b,v in units],[v['A02'] for b,v in units])})
            means[c,criterion]={b:sum(v.values())/2 for b,v in units}
            business_rows.extend({'condition':c,'criterion':criterion,'business_id':b,'A01':v['A01'],'A02':v['A02'],'two_rater_average':sum(v.values())/2} for b,v in units)
    pooled=[]
    for criterion in CRITERIA:
        units=sorted((c,b,v) for (c,crit,b),v in g.items() if crit==criterion)
        pooled.append({'criterion':criterion,'interpretation':'Supplementary condition mixture; not independent business samples',**kappa_quadratic([v['A01'] for c,b,v in units],[v['A02'] for c,b,v in units])})
    comparisons=[]
    for criterion in CRITERIA:
        for other in ['RO','U','S']:
            if other=='RO' and criterion==CRITERIA[-1]:continue
            ids=sorted(means['F',criterion]);assert ids==sorted(means[other,criterion])
            comparisons.append({'criterion':criterion,'contrast':'F-'+other,'n_businesses':30,'interpretation':'Exploratory; two-rater mean within business, paired bootstrap across businesses; unadjusted sign-flip diagnostic',
                **paired([means['F',criterion][b] for b in ids],[means[other,criterion][b] for b in ids],seed=20261028,B=B,P=P)})
    return {'descriptive':descriptions,'quadratic_weighted_cohen_kappa':agreements,'supplementary_pooled_kappa':pooled,'paired_comparisons':comparisons,
        'business_averages':business_rows,'participant_comments_verbatim':comment_rows,'qualitative_status':'Comments exported verbatim for grounded human interpretation; no invented conclusions',
        'counts':{'records':240,'numeric_scores':1380,'missing_applicable_scores':0,'structural_na_scores':na},
        'settings':{'seed':20261028,'bootstrap_draws':B,'sign_flip_draws':P,'ci':'95% paired business percentile','agreement':'QWK fixed 1..5 categories','inference':'EXPLORATORY'}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True,type=Path);p.add_argument('--ratings',required=True,type=Path);p.add_argument('--automatic-completion',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    # Do not instantiate Study: analysis must not create or mutate any live DB.
    ws=a.workspace;config=read(ws/'CONFIG.json');release=read(ws/'LIVE_RELEASE.json');data=read(a.ratings)
    if config.get('synthetic') or data.get('synthetic') or not data.get('final'):raise ValueError('CLI requires final genuine export, not technical fixtures')
    for name,h in config['file_bindings'].items():
        if file_hash(ws/name)!=h:raise ValueError('Workspace binding changed')
    if not live_release_valid(ws):raise ValueError('Valid prospective live release required')
    if data.get('config_sha256')!=file_hash(ws/'CONFIG.json') or data.get('amendment_sha256')!=release['amendment_sha256'] or data.get('audit',{}).get('status')!='PASS':raise ValueError('Export binding/audit mismatch')
    for r in RATERS:
        d=data.get('participant_declarations',{}).get(r)
        if not direct_rater_access_valid(ws,r) and (not d or not(d.get('consent') is True and d.get('task_familiarity') is True and d.get('eligibility')=='ELIGIBLE')):raise ValueError('Participant declarations or researcher access authorization missing')
    direct=read(ws/'DIRECT_RATER_ACCESS.json') if (ws/'DIRECT_RATER_ACCESS.json').is_file() else None
    if data.get('direct_access_authorization')!=direct:raise ValueError('Direct access provenance differs from the exported record')
    auto=read(a.automatic_completion)
    if not(auto.get('businesses')==100 and auto.get('condition_records')==400 and auto.get('components')==700):raise ValueError('Existing Stage M completion receipt mismatch')
    if file_hash(a.automatic_completion)!=file_hash(ws/'COORDINATOR_ONLY/STAGE_M_COMPLETION.json'):raise ValueError('Unbound automatic receipt')
    result=analyze(data,read(ws/'COORDINATOR_ONLY/PRIVATE_ITEMS.json'))
    result['release_provenance']=release
    result['direct_access_authorization']=direct
    result['participant_declarations_limitation']='Declaration/coordinator gate waived by researcher; personal declarations and independently verified honesty/independence cannot be inferred' if direct else None
    result['supervisor_approval_limitation']='Supervisor approval unverified; collection opened on direct researcher instruction' if release.get('supervisor_approved') is not True else None
    result['provenance']={str(path.name):file_hash(path) for path in [a.ratings,ws/'CONFIG.json',ws/'LIVE_RELEASE.json',a.automatic_completion]}
    if a.output.exists():raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True);(a.output/'TWO_RATER_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Complete actual-record analysis saved; no winner inferred automatically.')

if __name__=='__main__':main()
