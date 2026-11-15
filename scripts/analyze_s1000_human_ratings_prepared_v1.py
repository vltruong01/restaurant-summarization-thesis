"""Later human-rating analysis; preparation never invokes this on real ratings."""
import argparse,csv,collections
from pathlib import Path
from s1000_final_eval_core_v1 import *
CRITERIA=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--ratings',type=Path,required=True)
    p.add_argument('--automatic-completion',type=Path,required=True);p.add_argument('--researcher-release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    auto=read(a.automatic_completion);require(auto['businesses']==100 and auto['conditions']==4 and a.researcher_release.is_file(),'Automatic evaluation must be complete and human stage released')
    key=list(csv.DictReader((a.preparation/'PRIVATE_HUMAN_DESIGN_KEY.csv').open(encoding='utf8')));kk={r['alias']:r for r in key}
    rows=list(csv.DictReader(a.ratings.open(encoding='utf8')));require(len(rows)==360 and len({r['alias'] for r in rows})==360 and {r['alias'] for r in rows}==set(kk),'Incomplete or duplicated ratings')
    grouped=collections.defaultdict(dict)
    for r in rows:
        k=kk[r['alias']];require(r['annotator']==k['annotator'] and bool(r['submitted_at_utc']),'Human identity/submission missing')
        for criterion in CRITERIA:
            value=r[criterion]
            if k['condition']=='RO' and criterion.startswith('visual_'):require(value=='N/A','RO Visual N/A required');continue
            require(value in ['1','2','3','4','5'],'Missing/invalid five-point rating')
            grouped[k['condition'],criterion,k['business_id']][r['annotator']]=int(value)
    agreement=[];means={}
    for c in ['RO','U','F','S']:
        for criterion in CRITERIA:
            if c=='RO' and criterion.startswith('visual_'):continue
            units=sorted((b,v) for (condition,crit,b),v in grouped.items() if condition==c and crit==criterion)
            require(len(units)==30 and all(set(v)=={'A01','A02','A03'} for b,v in units),'Unbalanced three-rater panel')
            matrix=[[v[a] for a in ['A01','A02','A03']] for b,v in units]
            alpha=ordinal_alpha(matrix);means[c,criterion]={b:sum(v.values())/3 for b,v in units}
            agreement.append(dict(condition=c,criterion=criterion,ordinal_krippendorff_alpha=alpha,mean=sum(sum(v) for v in matrix)/90,constant_category_alpha_undefined=alpha is None,
                per_rater={a:dict(mean=sum(v[a] for b,v in units)/30,distribution=dict(collections.Counter(v[a] for b,v in units))) for a in ['A01','A02','A03']}))
    paired_rows=[]
    for criterion in CRITERIA:
        for other in ['RO','U','S']:
            if other=='RO' and criterion.startswith('visual_'):continue
            ids=sorted(means['F',criterion]);paired_rows.append(dict(criterion=criterion,contrast='F-'+other,interpretation='Exploratory human comparison, business-clustered averages over3 raters',**paired([means['F',criterion][b] for b in ids],[means[other,criterion][b] for b in ids])))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'HUMAN_AGREEMENT_AND_PAIRED.json',dict(agreement=agreement,paired=paired_rows,ratings_sha256=file_hash(a.ratings),automatic_receipt_sha256=file_hash(a.automatic_completion),release_sha256=file_hash(a.researcher_release)))
if __name__=='__main__':main()
