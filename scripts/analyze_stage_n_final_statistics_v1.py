"""Run the frozen two-rater analysis and requested descriptive extensions.

Only supplied, verified, unchanged records are analyzed. No live DB/server/GPU.
"""
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys

import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile_linear(values,q):
    v=sorted(values);pos=(len(v)-1)*q;lo=int(pos);hi=min(lo+1,len(v)-1)
    return v[lo]+(pos-lo)*(v[hi]-v[lo])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.package;out=args.output
    assert not out.exists(),'Do not overwrite an analysis; use a new output directory.'
    cfg=read(p/'ANALYSIS_CONFIG.json');data=read(p/'DATA/VERIFIED_ANALYSIS_INPUT.json');rows=data['records']
    sys.path.insert(0,str((p/'SCRIPTS_ORIGINAL').resolve()))
    from analyze_stage_n_two_raters_v1 import analyze,kappa_quadratic
    criteria=cfg['criteria'];conditions=cfg['conditions'];raters=cfg['raters'];ids=cfg['business_order']
    key=[{k:r[k] for k in ['alias','annotator','business_id','condition']} for r in rows]
    # Existing analysis core, unchanged. Input binding was established by the
    # original private key in the read-only preparation, not output inspection.
    original=analyze(data,key,B=cfg['bootstrap_draws'],P=cfg['signflip_draws'])
    cells={(r['condition'],r['business_id'],r['annotator']):r for r in rows}
    assert len(cells)==240 and len(ids)==30
    descriptions=[]
    for condition in conditions:
        for criterion in criteria:
            for panel in ['COMBINED']+raters:
                rr=[r for r in rows if r['condition']==condition and (panel=='COMBINED' or r['annotator']==panel)]
                vv=[r[criterion] for r in rr if type(r[criterion]) is int]
                assert len(rr)==(60 if panel=='COMBINED' else 30)
                item={'condition':condition,'criterion':criterion,'component_scope':
                    'Review' if criterion in [criteria[0],criteria[1],criteria[4]] else ('Visual' if criterion==criteria[-1] else 'All available components'),
                    'panel':panel,'n_businesses':30,'n_valid_ratings':len(vv),'n_structural_NA':sum(r[criterion]=='N/A' for r in rr),
                    'n_missing':0,'mean':None,'sample_sd':None,'median':None,'q25':None,'q75':None,'iqr':None,
                    **{'score_'+str(i):vv.count(i) for i in range(1,6)}}
                if vv:
                    q25,q75=np.quantile(vv,[.25,.75],method='linear')
                    item.update(mean=float(np.mean(vv)),sample_sd=float(np.std(vv,ddof=1)),median=float(np.median(vv)),
                        q25=float(q25),q75=float(q75),iqr=float(q75-q25))
                    assert np.isclose(item['mean'],statistics.mean(vv)) and np.isclose(item['sample_sd'],statistics.stdev(vv))
                    assert np.isclose(item['q25'],quantile_linear(vv,.25)) and np.isclose(item['q75'],quantile_linear(vv,.75))
                descriptions.append(item)
    means={(d['condition'],d['criterion']):d['mean'] for d in descriptions if d['panel']=='COMBINED'}
    for d in original['descriptive']:assert d['mean']==means[d['condition'],d['criterion']]
    agreements=[];agreement_pairs=[];confusions=[]
    for condition in conditions+['POOLED_SUPPLEMENTARY']:
        for criterion in criteria:
            if condition=='RO' and criterion==criteria[-1]:continue
            cc=conditions if condition=='POOLED_SUPPLEMENTARY' else [condition]
            pairs=[(c,b,cells[c,b,'A01'][criterion],cells[c,b,'A02'][criterion]) for c in cc for b in ids if not(c=='RO' and criterion==criteria[-1])]
            x=[a for c,b,a,z in pairs];y=[z for c,b,a,z in pairs];d=np.asarray(x)-np.asarray(y);n=len(d)
            result=kappa_quadratic(x,y)
            item={'condition':condition,'criterion':criterion,'n_paired_items':n,'n_unique_businesses':30,
                'qwk':result['value'],'undefined_reason':result['reason'],
                'exact_n':int((d==0).sum()),'exact_agreement_pct':float(100*(d==0).mean()),
                'within_one_n':int((np.abs(d)<=1).sum()),'within_one_pct':float(100*(np.abs(d)<=1).mean()),
                'mean_A01_minus_A02':float(d.mean()),'mean_absolute_difference':float(np.abs(d).mean()),
                **{'difference_'+str(i):int((d==i).sum()) for i in range(-4,5)}}
            agreements.append(item)
            for i in range(1,6):
                for j in range(1,6):confusions.append({'condition':condition,'criterion':criterion,'A01_score':i,'A02_score':j,'n':sum(a==i and b==j for a,b in zip(x,y))})
            if condition!='POOLED_SUPPLEMENTARY':
                for c,b,a,z in pairs:agreement_pairs.append({'condition':c,'business_id':b,'criterion':criterion,'A01':a,'A02':z,'A01_minus_A02':a-z})
    # sklearn is an optional independent verification dependency, not required
    # by the frozen numerical analysis/reproduction itself.
    checks=[]
    try:
        from sklearn.metrics import cohen_kappa_score
        for item in agreements:
            cc=conditions if item['condition']=='POOLED_SUPPLEMENTARY' else [item['condition']];criterion=item['criterion']
            paired_scores=[(cells[c,b,'A01'][criterion],cells[c,b,'A02'][criterion]) for c in cc for b in ids if not(c=='RO' and criterion==criteria[-1])]
            if item['qwk'] is None:continue
            other=cohen_kappa_score([a for a,b in paired_scores],[b for a,b in paired_scores],labels=[1,2,3,4,5],weights='quadratic')
            assert np.isclose(item['qwk'],other,rtol=0,atol=1e-12)
        checks.append({'check':'All defined primary and pooled QWK versus sklearn, fixed1..5 quadratic','result':'PASS'})
    except ImportError:checks.append({'check':'QWK cross-library verification','result':'NOT_AVAILABLE; no dependency installed'})
    business=[];paired_results=[];deltas=[]
    for item in original['business_averages']:
        business.append(item)
    for r in original['paired_comparisons']:
        other=r['contrast'].split('-')[1];criterion=r['criterion']
        f=np.asarray([(cells['F',b,'A01'][criterion]+cells['F',b,'A02'][criterion])/2 for b in ids])
        comp=np.asarray([(cells[other,b,'A01'][criterion]+cells[other,b,'A02'][criterion])/2 for b in ids]);d=f-comp
        assert np.isclose(r['mean_difference'],d.mean())
        item={'criterion':criterion,'contrast':r['contrast'],'n_businesses':30,'ratings_per_business_per_condition':2,
            'F_mean':float(f.mean()),'comparator_mean':float(comp.mean()),'mean_paired_difference':r['mean_difference'],
            'CI95_low':r['CI95'][0],'CI95_high':r['CI95'][1],'paired_difference_sample_sd':float(d.std(ddof=1)),
            'F_higher_businesses':int((d>0).sum()),'tie_businesses':int((d==0).sum()),'F_lower_businesses':int((d<0).sum()),
            'unadjusted_exploratory_signflip_p':r['signflip_two_sided_p'],
            'multiplicity_adjustment':'NONE_PRESPECIFIED','inference':'EXPLORATORY_NOT_CONFIRMATORY',
            'effect_units':'1-5 score points','CI_is_pointwise_not_simultaneous':True,
            'A01_mean_difference':float(np.mean([cells['F',b,'A01'][criterion]-cells[other,b,'A01'][criterion] for b in ids])),
            'A02_mean_difference':float(np.mean([cells['F',b,'A02'][criterion]-cells[other,b,'A02'][criterion] for b in ids]))}
        paired_results.append(item)
        for b,a,z,v in zip(ids,f,comp,d):
            deltas.append({'business_id':b,'criterion':criterion,'contrast':r['contrast'],'F_two_rater_average':float(a),
                'comparator_two_rater_average':float(z),'difference':float(v),
                'A01_difference':cells['F',b,'A01'][criterion]-cells[other,b,'A01'][criterion],
                'A02_difference':cells['F',b,'A02'][criterion]-cells[other,b,'A02'][criterion]})
    # Verify the paired procedures with a separately constructed common-index
    # bootstrap/sign matrix, retaining all four conditions and both raters.
    rng=np.random.default_rng(cfg['seed'])
    indices=rng.integers(0,30,(cfg['bootstrap_draws'],30))
    signs=rng.choice([-1,1],(cfg['signflip_draws'],30))
    for item in paired_results:
        other=item['contrast'].split('-')[1];crit=item['criterion']
        tensor=np.asarray([[[cells[c,b,r][crit] for r in raters] for c in ['F',other]] for b in ids],dtype=float)
        # All raters/conditions of a selected business travel together.
        sample=tensor[indices].mean(axis=(1,3));sample_d=sample[:,0]-sample[:,1]
        lo,hi=np.quantile(sample_d,[.025,.975],method='linear')
        assert np.allclose([lo,hi],[item['CI95_low'],item['CI95_high']],rtol=0,atol=1e-12)
        d=tensor[:,0,:].mean(axis=1)-tensor[:,1,:].mean(axis=1)
        pvalue=(int((np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-15).sum())+1)/(cfg['signflip_draws']+1)
        assert pvalue==item['unadjusted_exploratory_signflip_p']
    checks.append({'check':'All17 CIs and sign-flip p-values independently reconstructed from common business indices and full two-rater tensor','result':'PASS'})
    identity=[]
    for criterion in [criteria[0],criteria[1],criteria[4]]:
        dd=[cells['F',b,r][criterion]-cells['S',b,r][criterion] for b in ids for r in raters]
        identity.append({'criterion':criterion,'identical_output_pairs':60,'equal_scores':dd.count(0),'unequal_scores':sum(v!=0 for v in dd),
            'mean_F_minus_S':float(np.mean(dd)),'max_absolute_difference':max(abs(v) for v in dd),
            'interpretation':'Same frozen Review text, different presentation context/position; any rating difference is not a generated Review improvement.'})
    distribution_long=[]
    for row in descriptions:
        for score in range(1,6):distribution_long.append({k:row[k] for k in ['condition','criterion','panel']}|{'score':score,'count':row['score_'+str(score)],'n_valid':row['n_valid_ratings']})
    valid_by_condition={c:sum(r[crit]!='N/A' for r in rows if r['condition']==c for crit in criteria) for c in conditions}
    versions={'python':sys.version,'platform':platform.platform()}
    for name in ['numpy','scipy','scikit-learn']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    out.mkdir(parents=True)
    write(out/'ORIGINAL_ANALYZER_RESULTS.json',original)
    write(out/'DESCRIPTIVE_STATISTICS.json',descriptions);csvwrite(out/'DESCRIPTIVE_STATISTICS.csv',descriptions)
    write(out/'AGREEMENT.json',agreements);csvwrite(out/'AGREEMENT.csv',agreements)
    csvwrite(out/'AGREEMENT_PAIRED_SCORES.csv',agreement_pairs);csvwrite(out/'AGREEMENT_CONFUSION_MATRICES.csv',confusions)
    write(out/'PAIRED_COMPARISONS.json',paired_results);csvwrite(out/'PAIRED_COMPARISONS.csv',paired_results)
    write(out/'PER_BUSINESS_CRITERION.json',business);csvwrite(out/'PER_BUSINESS_CRITERION.csv',business)
    csvwrite(out/'PER_BUSINESS_PAIRED_DIFFERENCES.csv',deltas);csvwrite(out/'SCORE_DISTRIBUTIONS.csv',distribution_long)
    write(out/'IDENTICAL_F_S_REVIEW_RATINGS.json',identity)
    csvwrite(out/'PER_RATER_DATA.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in rows])
    write(out/'PARTICIPANT_COMMENTS_VERBATIM.json',original['participant_comments_verbatim'])
    write(out/'NUMERICAL_VERIFICATION.json',{'status':'PASS','checks':checks,'descriptive_rows':len(descriptions),
        'primary_agreement_cells':23,'pooled_supplementary_agreement_cells':6,'paired_contrasts':len(paired_results),
        'business_criterion_rows':len(business),'paired_business_differences':len(deltas),'valid_scores_by_condition':valid_by_condition,
        'primary_includes_all30_businesses':True,'excluded_records':[],
        'resample_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
        'signflip_matrix_sha256':hashlib.sha256(signs.astype('<i8').tobytes()).hexdigest(),
        'input_sha256':sha(p/'DATA/VERIFIED_ANALYSIS_INPUT.json'),'config_sha256':sha(p/'ANALYSIS_CONFIG.json')})
    write(out/'RUNTIME.json',{'completed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'versions':versions,
        'GPU_or_API_work':False,'raw_ordinal_alpha':'NOT_RUN_PER_FROZEN_AMENDMENT','multiplicity_adjustment':'NOT_PRESPECIFIED_NO_NEW_ADJUSTMENT'})
    print(json.dumps({'descriptive_rows':len(descriptions),'agreement_cells':len(agreements),'paired_comparisons':len(paired_results),
        'mean_scores':[{ 'condition':c,**{crit:means[c,crit] for crit in criteria}} for c in conditions],
        'F_S_Review_rating_identity':identity,'status':'PASS'},indent=2))


if __name__=='__main__':main()
