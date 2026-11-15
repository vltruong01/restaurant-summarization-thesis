"""Source-only preparation and objective contracts; never loads a generation model."""
import csv, hashlib, json, re, unicodedata, datetime
from collections import Counter
from pathlib import Path

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def text_hash(x): return hashlib.sha256(x.encode('utf8')).hexdigest()
def file_hash(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f: json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write_csv(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def require(ok,msg):
    if not ok: raise ValueError(msg)
def words(x): return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*",unicodedata.normalize('NFKC',x).casefold())
def isolated(source):
    sep='\n\n[VISUAL_CONTEXT]\n'
    require(source.count(sep)==1,'Invalid frozen visual boundary')
    r,v=source.split(sep)
    require('[REVIEWS]' in r and '[REV_' not in v and '[IMG_CAP_' not in r,'Cross-task source')
    return '[TASK_REVIEW_CONSENSUS]\n'+r,'[TASK_VISUAL_EVIDENCE]\n[VISUAL_CONTEXT]\n'+v
def categories(source,allowed):
    norm=lambda x:unicodedata.normalize('NFKC',x).strip().casefold()
    lines=[s[11:] for s in source.split('[REVIEWS]')[0].splitlines() if s.startswith('categories=')]
    require(len(lines)==1,'Missing category metadata')
    return sorted(set(map(norm,lines[0].split(','))) & set(map(norm,allowed)))

def shuffled_mapping(rows,tok,policy):
    """Globally maximize shared-category edges, then seeded cost, exact-count derangement.

    No target, prediction or reference argument is accepted. All positive recipients
    and donors are the same set. Failure produces no relaxed partial assignment.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    schema={'business_id','split','source','source_sha256','selected_review_ids','selected_photo_ids'}
    require(rows and all(set(r)==schema for r in rows),'Source-only schema required')
    table={r['business_id']:r for r in rows};require(len(table)==len(rows),'Duplicate IDs')
    require(all(r['split']=='test' for r in rows),'TEST source cohort only')
    prepared={}
    for b,r in table.items():
        require(text_hash(r['source'])==r['source_sha256'],'Source hash changed')
        rs,vs=isolated(r['source']);head,body=r['source'].split('\n\n[VISUAL_CONTEXT]\n')
        n=len(r['selected_photo_ids']);require(0<=n<=5,'Caption count')
        require(re.findall(r'\[IMG_CAP_(\d+)\]',body)==[str(i+1) for i in range(n)],'Caption markers')
        require(n or not body.strip(),'Nonempty zero-caption body')
        require(len(tok.encode(rs).ids)<=5000 and len(tok.encode(vs).ids)<=750,'Own task budget')
        require(len(tok.encode(head.split('[REVIEWS]\n',1)[1]).ids)<=4000,'Review block budget')
        prepared[b]=(rs,vs,head,body,n,categories(r['source'],policy['cuisine_subcategory_allowlist']))
    assignment={};edges=[]
    for n in sorted({p[4] for p in prepared.values()}-{0}):
        ids=sorted(b for b,p in prepared.items() if p[4]==n);N=len(ids)
        cost=np.full((N,N),10**13,dtype=np.int64);feasible={}
        for i,a in enumerate(ids):
            for j,b in enumerate(ids):
                if a==b:continue
                ar,av,head,body,_,tags=prepared[a];don=prepared[b]
                source=head+'\n\n[VISUAL_CONTEXT]\n'+don[3]
                inc=len(tok.encode(source).ids)-len(tok.encode(head).ids)
                fit=len(tok.encode(source).ids)<=5000 and 0<inc<=750 and len(tok.encode(don[1]).ids)<=750
                if not fit:continue
                shared=sorted(set(tags)&set(don[5]));rank=int(digest([policy['seed'],policy['version'],a,b])[:12],16)%1000000
                cost[i,j]=(0 if shared else 10**9)+rank
                feasible[a,b]=(source,inc,shared)
        ii,jj=linear_sum_assignment(cost)
        require(len(ii)==N and all(cost[i,j]<10**13 for i,j in zip(ii,jj)),f'NO_FEASIBLE_ONE_TO_ONE_GROUP_{n}; no fallback committed')
        for i,j in zip(ii,jj):assignment[ids[i]]=ids[j]
        edges.append(dict(caption_count=n,recipients=N,feasible_edges=len(feasible),shared_category_edges=sum(bool(v[2]) for v in feasible.values())))
    out=[]
    for a in sorted(table):
        rs,vs,head,body,n,tags=prepared[a];b=assignment.get(a)
        source=table[a]['source'] if b is None else head+'\n\n[VISUAL_CONTEXT]\n'+prepared[b][3]
        sr,sv=isolated(source);require(rs==sr,'Review changed')
        ids=tok.encode(source).ids;ri=tok.encode(sr).ids;vi=tok.encode(sv).ids
        shared=[] if b is None else sorted(set(tags)&set(prepared[b][5]))
        out.append(dict(recipient_id=a,donor_id=b,caption_count=n,split='test',
            matching_tier='ZERO_CAPTION_UNCHANGED' if b is None else 'SHARED_CATEGORY' if shared else 'SAME_COUNT_FALLBACK',
            shared_categories=shared,source=source,source_sha256=text_hash(source),input_ids_sha256=digest(ids),
            full_source_tokens=len(ids),caption_increment_tokens=len(ids)-len(tok.encode(head).ids),
            review_task_source=sr,review_task_source_sha256=text_hash(sr),review_task_input_ids_sha256=digest(ri),
            visual_task_source=sv,visual_task_source_sha256=text_hash(sv),visual_task_input_ids_sha256=digest(vi),
            review_task_tokens=len(ri),visual_task_tokens=len(vi),review_identical_to_F=True,
            selected_review_ids=table[a]['selected_review_ids'],recipient_F_photo_ids=table[a]['selected_photo_ids'],
            donor_photo_ids=[] if b is None else table[b]['selected_photo_ids'],
            donor_changed=b is not None,visual_source_changed=sv!=vs,
            original_F_source_sha256=table[a]['source_sha256']))
    pos=[x for x in out if x['caption_count']]
    require(len({x['donor_id'] for x in pos})==len(pos),'Donor reuse')
    return dict(version=policy['version'],seed=policy['seed'],policy_sha256=digest(policy),source_rows_sha256=digest(sorted(rows,key=lambda r:r['business_id'])),
        one_to_one=True,groups=edges,mappings=out,TEST_inference_performed=False)

def reference_content(r):return {k:r[k] for k in ['business_id','source_bundle_sha256','review_consensus','visual_evidence','review_claims','visual_claims']}
def validate_reference(r,source,approved=False):
    errors=[]
    def check(x,m):
        if not x:errors.append(m)
    check(r.get('business_id')==source['business_id'],'business_id')
    check(r.get('source_bundle_sha256')==digest(source),'source_bundle_sha256')
    for k in ['review_consensus','visual_evidence']:
        check(isinstance(r.get(k),str),'invalid '+k)
        if not isinstance(r.get(k),str):return errors
        if approved:check(bool(r.get(k,'').strip()),'missing '+k)
        check('[REVIEW_CONSENSUS]' not in r.get(k,'') and '[VISUAL_EVIDENCE]' not in r.get(k,''),'headings belong in export only')
    reviews={x['review_id']:x['text'] for x in source['reviews']};photos={x['photo_id'] for x in source['photos']}
    for component,allowed in [('review',reviews),('visual',photos)]:
        claims=r.get(component+'_claims',[]);check(isinstance(claims,list),'claim list')
        if approved:check(bool(claims),'missing '+component+' claim evidence')
        for c in claims:
            check(bool(c.get('output_span')) and c['output_span'] in r.get('review_consensus' if component=='review' else 'visual_evidence',''),'claim output_span')
            ev=c.get('evidence',[]);check(bool(ev),'missing evidence')
            for e in ev:
                key=e.get('review_id' if component=='review' else 'photo_id');check(key in allowed,'foreign or missing evidence ID')
                if component=='review':check(bool(e.get('quote')) and e.get('quote','') in reviews.get(key,''),'review quote not exact')
                elif approved:check(e.get('direct_image_viewed') is True,'direct photo check absent')
    if approved:
        a=r.get('author',{});c=r.get('checker',{});version=digest(reference_content(r))
        check(r.get('status')=='APPROVED','approval missing')
        check(bool(a.get('human_id')) and bool(c.get('human_id')) and a.get('human_id')!=c.get('human_id'),'two distinct humans required')
        check(a.get('human_written_attestation') is True and c.get('independent_check_attestation') is True,'human provenance attestations')
        check(c.get('reviewed_content_sha256')==version,'checker approved another content version')
        check(bool(r.get('history')) and r['history'][-1].get('content_sha256')==version,'version history binding')
        try:
            ta=datetime.datetime.fromisoformat(a['at_utc']);tc=datetime.datetime.fromisoformat(c['at_utc'])
            check(ta.utcoffset()==tc.utcoffset()==datetime.timedelta(0) and tc>=ta,'provenance chronology')
        except (KeyError,ValueError,TypeError):errors.append('missing/invalid UTC provenance')
        check(c.get('predictions_seen') is False and a.get('predictions_seen') is False,'prediction blinding attestation')
    return errors

def diagnostics(s,source=''):
    t=words(s);sent=[x for x in re.split(r'(?<=[.!?])\s+',s.strip()) if x]
    out={'words':len(t),'characters':len(s),'sentences':len(sent),'empty':not bool(s.strip())}
    for n in [1,2,3,4]:
        g=[tuple(t[i:i+n]) for i in range(max(0,len(t)-n+1))]
        out[f'repeated_{n}gram_fraction']=1-len(set(g))/len(g) if g else 0.
    out['duplicate_sentence_fraction']=1-len(set(' '.join(words(x)) for x in sent))/len(sent) if sent else 0.
    sw=set(words(source));out['source_lexical_precision_proxy']=sum(w in sw for w in t)/len(t) if t else 0.
    return out

def paired(x,y,seed=20261028,B=20000,P=50000):
    import numpy as np
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    require(x.shape==y.shape and x.ndim==1 and len(x)>1 and np.isfinite(x).all() and np.isfinite(y).all(),'Complete finite paired data required')
    d=x-y;rng=np.random.default_rng(seed);samples=d[rng.integers(0,len(d),(B,len(d)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975]);extreme=0
    for start in range(0,P,1000):
        n=min(1000,P-start);v=(rng.choice([-1,1],(n,len(d)))*d).mean(axis=1)
        extreme+=int((abs(v)>=abs(d.mean())-1e-15).sum())
    return dict(n=len(d),mean_difference=float(d.mean()),CI95=[float(lo),float(hi)],signflip_two_sided_p=(extreme+1)/(P+1),seed=seed,bootstrap_replicates=B,signflip_replicates=P)

def holm(p):
    order=sorted(range(len(p)),key=lambda i:p[i]);out=[0.]*len(p);last=0.
    for k,i in enumerate(order):last=max(last,min(1.,(len(p)-k)*p[i]));out[i]=last
    return out

def ordinal_alpha(ratings):
    """Krippendorff alpha ordinal using pooled category frequencies, complete 3-rater units."""
    import numpy as np
    a=np.asarray(ratings);require(a.ndim==2 and a.shape[1]==3 and np.isin(a,[1,2,3,4,5]).all(),'Complete ordinal three-rater units required')
    freq=np.bincount(a.astype(int).ravel(),minlength=6)[1:];N=freq.sum();D=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            lo,hi=sorted((i,j));D[i,j]=(freq[lo:hi+1].sum()-(freq[lo]+freq[hi])/2)**2 if i!=j else 0
    observed=sum(D[int(x)-1,int(y)-1] for row in a for i,x in enumerate(row) for j,y in enumerate(row) if i!=j)/(len(a)*6)
    expected=sum(freq[i]*freq[j]*D[i,j] for i in range(5) for j in range(5))/(N*(N-1))
    return None if expected==0 else float(1-observed/expected)
