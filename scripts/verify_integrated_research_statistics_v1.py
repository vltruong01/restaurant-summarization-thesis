"""Independent CPU arithmetic checks; no metric models, GPU or source edits."""
from pathlib import Path
import argparse,collections,csv,hashlib,itertools,json,math,statistics,subprocess,sys
import numpy as np

CRITERIA=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
METRICS=['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def close(a,b):
 if a is None or b is None:assert a is b,(a,b)
 else:assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=2e-12),(a,b)
def table(p,rows):
 save(p.with_suffix('.json'),rows)
 with p.with_suffix('.csv').open('w',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def qwk(a,b):
 confusion=np.zeros((5,5),dtype=np.int64)
 for x,y in zip(a,b):confusion[int(x)-1,int(y)-1]+=1
 n=len(a);weights=(np.arange(5)[:,None]-np.arange(5)[None,:])**2
 observed=float((confusion*weights).sum())/n
 expected=float((np.outer(confusion.sum(1),confusion.sum(0))*weights).sum())/(n*n)
 return None if expected==0 else 1-observed/expected
def holm(values):
 order=sorted(range(len(values)),key=values.__getitem__);result=[None]*len(values);previous=0
 for rank,index in enumerate(order):previous=max(previous,min(1.,values[index]*(len(values)-rank)));result[index]=previous
 return result
def describe(rows,models,ids,raters):
 result=[];agreement=[];cells={(r['business_id'],r['annotator'],r['model']):r for r in rows};assert len(cells)==len(rows)
 for criterion in CRITERIA:
  for model in models:
   for panel in ['COMBINED']+raters:
    values=[r[criterion] for r in rows if r['model']==model and (panel=='COMBINED' or r['annotator']==panel) and type(r[criterion]) is int]
    result.append(dict(model=model,criterion=criterion,panel=panel,n=len(values),mean=statistics.mean(values) if values else None,sample_SD=statistics.stdev(values) if len(values)>1 else None,median=statistics.median(values) if values else None,**{'score_'+str(k):values.count(k) for k in range(1,6)}))
   x=[cells[b,raters[0],model][criterion] for b in ids];y=[cells[b,raters[1],model][criterion] for b in ids]
   if not all(type(v) is int for v in x+y):continue
   diffs=[a-b for a,b in zip(x,y)]
   agreement.append(dict(model=model,criterion=criterion,n_pairs=len(x),quadratic_weighted_kappa=qwk(x,y),undefined_reason='Both marginal distributions constant at the same score; expected disagreement is zero' if qwk(x,y) is None else None,exact_agreement_fraction=sum(v==0 for v in diffs)/len(x),within_one_fraction=sum(abs(v)<=1 for v in diffs)/len(x),mean_absolute_difference=statistics.mean(abs(v) for v in diffs),mean_first_minus_second=statistics.mean(diffs),**{'difference_'+str(k):diffs.count(k) for k in range(-4,5)}))
 return result,agreement,cells

def analyze_p(root,out):
 p=root/'SOURCES/P';config=read(p/'FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json');mapping=read(p/'FROZEN_STUDY/COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json');key={r['alias']:r for r in mapping}
 exports=[p/'RATINGS/A01_FINAL_45.json',p/'RATINGS/A02_FINAL_45.json'];combined=read(p/'RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')
 assert sha(p/'RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')=='e47b88f1b4d6ccf0bf1649b5b06ead3f538f97af30324993f5e45cf79a58472f'
 sys.path.insert(0,str(p/'STATISTICS'));from validate_stage_p_human_exports_v1 import validate
 validated,validation=validate(p/'FROZEN_STUDY',exports)
 assert {r['alias']:{k:v for k,v in r.items() if k!='system'} for r in validated}=={r['alias']:r for r in combined['records']},'Combined original export must equal both rater exports, including comments and metadata'
 original_verification=subprocess.run([sys.executable,'-B',str(p/'VERIFY_PACKAGE.py'),'--package',str(p)],capture_output=True,text=True,encoding='utf8',check=True)
 original_receipt=json.loads(original_verification.stdout)
 save(out/'P_ORIGINAL_COLLECTION_VERIFICATION.json',original_receipt)
 rows=[{**r,'model':key[r['alias']]['system']} for r in combined['records']]
 assert len(rows)==90 and len({r['business_id'] for r in rows})==15
 assert all(r['status']=='SUBMITTED' and all(type(r[c]) is int and 1<=r[c]<=5 for c in CRITERIA) for r in rows)
 assert {r['alias'] for r in rows}==set(key) and collections.Counter(r['annotator'] for r in rows)=={'A01':45,'A02':45}
 systems=config['systems'];ids=config['restaurant_order'];raters=config['raters']
 descriptions,agreements,cells=describe(rows,systems,ids,raters)
 rng=np.random.default_rng(config['seed']);bootstrap=rng.integers(0,len(ids),(config['bootstrap_draws'],len(ids)))
 signs=np.array(list(itertools.product([-1,1],repeat=15)),dtype=np.int8)
 paired=[];paired_rows=[]
 for criterion in CRITERIA:
  for contrast in config['contrasts']:
   left,right=contrast['left'],contrast['right']
   differences=np.array([[cells[b,r,left][criterion]-cells[b,r,right][criterion] for r in raters] for b in ids],dtype=float)
   cluster=differences.mean(1);ci=np.quantile(cluster[bootstrap].mean(1),[.025,.975],method='linear')
   # Scores averaged over two raters are exact half-integers. Integer totals avoid floating tail ambiguities.
   scaled=(cluster*2).astype(np.int64);assert np.array_equal(scaled,cluster*2)
   pvalue=float(np.mean(abs(signs@scaled)>=abs(scaled.sum())))
   paired.append(dict(left=left,right=right,criterion=criterion,family=contrast['family'],n_paired_restaurants=15,mean_difference=float(cluster.mean()),pointwise_CI95_low=float(ci[0]),pointwise_CI95_high=float(ci[1]),raw_exact_signflip_p=pvalue,holm_p_within_family=None,first_rater_difference=float(differences[:,0].mean()),second_rater_difference=float(differences[:,1].mean()),positive_restaurants=int(sum(cluster>0)),tied_restaurants=int(sum(cluster==0)),negative_restaurants=int(sum(cluster<0)),same_Qwen_Review=(left==systems[1] and right==systems[2] and criterion in config['Review_only_criteria'])))
   for business,pair in zip(ids,differences):paired_rows.append(dict(business_id=business,left=left,right=right,criterion=criterion,A01_difference=float(pair[0]),A02_difference=float(pair[1]),restaurant_mean_difference=float(pair.mean())))
 for family in ['PRIMARY','ADDITIONAL_EXPLORATORY']:
  family_rows=[r for r in paired if r['family']==family]
  for row,adjusted in zip(family_rows,holm([r['raw_exact_signflip_p'] for r in family_rows])):row['holm_p_within_family']=adjusted
 # Independently compare against execution of the unchanged frozen preparation script.
 frozen_out=out/'P_FROZEN_SCRIPT_REPLAY'
 result=subprocess.run([sys.executable,'-B',str(p/'STATISTICS/analyze_stage_p_human_ratings_v1.py'),'--package',str(p/'FROZEN_STUDY'),'--exports',*[str(x) for x in exports],'--output',str(frozen_out)],capture_output=True,text=True,encoding='utf8',check=True)
 old=read(frozen_out/'DESCRIPTIVES.json');assert len(old)==len(descriptions)==54
 for a,b in zip(descriptions,old):
  assert (a['model'],a['criterion'],a['panel'])==(b['system'],b['criterion'],b['panel'])
  for x,y in [('n','n_ratings'),('mean','mean'),('sample_SD','sample_SD'),('median','median')]:close(a[x],b[y])
  for k in range(1,6):assert a['score_'+str(k)]==b['score_'+str(k)]
 for a,b in zip(agreements,read(frozen_out/'AGREEMENT.json')):
  for k in ['quadratic_weighted_kappa','exact_agreement_fraction','within_one_fraction','mean_absolute_difference','mean_first_minus_second']:close(a[k],b[k])
 for a,b in zip(paired,read(frozen_out/'PAIRED_COMPARISONS.json')):
  for k in ['mean_difference','pointwise_CI95_low','pointwise_CI95_high','raw_exact_signflip_p','holm_p_within_family','first_rater_difference','second_rater_difference']:close(a[k],b[k])
 outputs=read(p/'FROZEN_STUDY/COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json');om={(x['business_id'],x['system']):x for x in outputs}
 assert all(om[b,'Qwen-Caption']['review_text']==om[b,'Qwen-Image']['review_text'] for b in ids)
 # Author-reported score table is a prespecified cross-check,not a data replacement.
 expected={'review_faithfulness':[3.27,4.30,4.30],'aspect_coverage':[3.,4.53,4.53],'coherence':[4.07,5.,5.],'non_redundancy':[3.17,4.8,4.8],'usefulness':[2.93,4.4,4.4],'visual_grounding_and_contribution':[3.53,3.53,4.17]}
 for row in descriptions:
  if row['panel']=='COMBINED':assert round(row['mean'],2)==expected[row['criterion']][systems.index(row['model'])]
 assert qwk([5]*15,[5]*15) is None and qwk([2]*15,[3]*15)==0 and qwk([1,2,3,4,5],[1,2,3,4,5])==1
 assert holm([.01,.04,.03])==[.03,.06,.06]
 table(out/'P_DESCRIPTIVES',descriptions);table(out/'P_AGREEMENT',agreements);table(out/'P_PAIRED_COMPARISONS',paired);table(out/'P_BUSINESS_DIFFERENCES',paired_rows)
 length=[]
 for model in systems:
  for component in ['review','visual']:
   texts=[x[component+'_text'] for x in outputs if x['system']==model]
   # Same simple whitespace-token accounting for context;not quality or a new model evaluation.
   lengths=[len(t.split())-int(t.startswith('[REVIEW_CONSENSUS]' if component=='review' else '[VISUAL_EVIDENCE]')) for t in texts]
   length.append(dict(system=model,component=component,n=15,mean_whitespace_words=statistics.mean(lengths),minimum=min(lengths),maximum=max(lengths),definition='Whitespace words after removing one exact leading component marker;descriptive only'))
 table(out/'P_OUTPUT_LENGTH_CONTEXT',length)
 return dict(status='PASS',official_records=90,numeric_scores=540,descriptive_cells=54,agreement_cells=18,paired_tests=18,primary_family_tests=12,additional_family_tests=6,undefined_QWK_cells=[dict(system=r['model'],criterion=r['criterion'],exact_agreement_fraction=r['exact_agreement_fraction']) for r in agreements if r['quadratic_weighted_kappa'] is None],bootstrap_seed=config['seed'],bootstrap_replicates=config['bootstrap_draws'],bootstrap_indices_sha256=hashlib.sha256(bootstrap.astype('<i8').tobytes()).hexdigest(),exact_sign_assignments=len(signs),independent_vs_frozen_script='ALL_DESCRIPTIVES_AGREEMENT_CI_SIGNFLIP_HOLM_MATCH',original_data_hash=sha(p/'RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json'),external_ChatGPT_statistical_ZIP_verified=False,external_ZIP_not_required_by_researcher_clarification=True,unchanged_Qwen_Review_pairs=15)

def verify_no(root,out,stage):
 base=root/'SOURCES'/stage;config=read(base/'ANALYSIS_CONFIG.json')
 raw=read(base/('DATA/VERIFIED_ANALYSIS_INPUT.json' if stage=='N' else 'COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json'))['records']
 rows=[{**r,'model':r['condition']} for r in raw]
 n=30 if stage=='N' else 15;models=config['conditions'];ids=config['business_order'];raters=config['raters']
 assert len(rows)==n*len(models)*2 and all(r['status']=='SUBMITTED' for r in rows)
 assert len(ids)==len(set(ids))==n and set(r['business_id'] for r in rows)==set(ids)
 for row in rows:
  for criterion in CRITERIA:
   if row['model']=='RO' and criterion==CRITERIA[-1]:assert row[criterion] in (None,'N/A')
   else:assert type(row[criterion]) is int and 1<=row[criterion]<=5
 desc,agreement,cells=describe(rows,models,ids,raters)
 prior_desc=read(base/('TABLES/DESCRIPTIVE_STATISTICS.json' if stage=='N' else 'TABLES/DESCRIPTIVES.json'))
 prior_desc_by_key={(b['condition'],b['criterion'],b['panel']):b for b in prior_desc}
 assert len(prior_desc_by_key)==len(prior_desc)==len(desc)
 for a in desc:
  b=prior_desc_by_key[a['model'],a['criterion'],a['panel']]
  close(a['mean'],b['mean']);close(a['sample_SD'],b['sample_sd']);close(a['median'],b['median'])
  for k in range(1,6):assert a['score_'+str(k)]==b['score_'+str(k)]
 prior_agreement=read(base/'TABLES/AGREEMENT.json')
 primary=[r for r in prior_agreement if r['condition'] in models]
 assert len(primary)==len(agreement)
 prior_agreement_by_key={(b['condition'],b['criterion']):b for b in primary}
 assert len(prior_agreement_by_key)==len(primary)
 for a in agreement:
  b=prior_agreement_by_key[a['model'],a['criterion']]
  close(a['quadratic_weighted_kappa'],b['qwk']);close(a['exact_agreement_fraction']*100,b['exact_agreement_pct' if stage=='N' else 'exact_pct'])
  close(a['mean_first_minus_second'],b['mean_A01_minus_A02']);close(a['mean_absolute_difference'],b['mean_absolute_difference'])
 prior_pairs=read(base/'TABLES/PAIRED_COMPARISONS.json')
 rng=np.random.default_rng(config['seed']);idx=rng.integers(0,n,(config['bootstrap_draws'],n));signs=rng.choice([-1,1],(config['signflip_draws'],n))
 for row in prior_pairs:
  criterion=row['criterion'];right=row['contrast'].split('-')[1]
  d=np.array([statistics.mean(cells[b,r,'F'][criterion]-cells[b,r,right][criterion] for r in raters) for b in ids],dtype=float)
  lo,hi=np.quantile(d[idx].mean(1),[.025,.975],method='linear')
  p=(int((abs((signs*d).mean(1))>=abs(d.mean())-1e-15).sum())+1)/(config['signflip_draws']+1)
  close(float(d.mean()),row['mean_paired_difference' if stage=='N' else 'mean_difference']);close(lo,row['CI95_low']);close(hi,row['CI95_high']);close(p,row['unadjusted_exploratory_signflip_p' if stage=='N' else 'raw_signflip_p'])
 table(out/(stage+'_DESCRIPTIVES_VERIFIED'),desc);table(out/(stage+'_AGREEMENT_VERIFIED'),agreement)
 return dict(status='PASS',records=len(rows),businesses=n,descriptive_rows=len(desc),agreement_cells=len(agreement),paired_CI_and_pvalues_checked=len(prior_pairs),no_changed_scores=True,no_new_multiplicity_policy=True)

def verify_m(root,out):
 base=root/'SOURCES/M/RESULTS_ORIGINAL'
 def csvread(name):
  with (base/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
 raw=csvread('PER_BUSINESS_COMPONENT_METRICS.csv');aggregate=csvread('AGGREGATE_METRICS.csv');pairs=csvread('PAIRED_COMPARISONS.csv')
 assert len(raw)==700 and len({r['business_id'] for r in raw})==100
 by={(r['condition'],r['business_id'],r['task']):r for r in raw};assert len(by)==700
 for r in raw:
  assert r['reference_business_id']==r['business_id']
  same=[x['reference_sha256'] for x in raw if (x['business_id'],x['task'])==(r['business_id'],r['task'])]
  assert len(set(same))==1,'All applicable systems must use the same recipient component reference'
 summaries=[]
 for row in aggregate:
  group=[r for r in raw if (r['condition'],r['task'])==(row['condition'],row['task'])];assert len(group)==100
  for metric in METRICS:close(statistics.mean(float(r[metric]) for r in group),float(row[metric]))
  summaries.append({**{k:row[k] for k in ['condition','task','n']},**{k:float(row[k]) for k in METRICS}})
 ids=sorted({r['business_id'] for r in raw});rng=np.random.default_rng(20261028);idx=rng.integers(0,100,(20000,100));signs=rng.choice([-1,1],(50000,100));primary=[]
 for row in pairs:
  task=row['task'];right=row['contrast'].split('-')[1];metric=row['metric'];d=np.array([float(by['F',b,task][metric])-float(by[right,b,task][metric]) for b in ids])
  close(d.mean(),float(row['mean_difference']));ci=np.quantile(d[idx].mean(1),[.025,.975]);saved_ci=json.loads(row['CI95']);close(ci[0],saved_ci[0]);close(ci[1],saved_ci[1])
  if row['signflip_two_sided_p']:
   p=(int((abs((signs*d).mean(1))>=abs(d.mean())-1e-15).sum())+1)/50001;close(p,float(row['signflip_two_sided_p']));primary.append((row,p))
 for (row,_),adjusted in zip(primary,holm([p for _,p in primary])):close(adjusted,float(row['holm_p_four_prespecified_tests']))
 diagnostic=[]
 for row in aggregate:
  group=[r for r in raw if (r['condition'],r['task'])==(row['condition'],row['task'])]
  diagnostic.append(dict(condition=row['condition'],task=row['task'],n=100,mean_words=statistics.mean(float(r['words']) for r in group),mean_repeated_3gram_fraction=statistics.mean(float(r['repeated_3gram_fraction']) for r in group),mean_duplicate_sentence_fraction=statistics.mean(float(r['duplicate_sentence_fraction']) for r in group),empty_candidates=int(row['empty_candidates']),BERT_prefix_truncated=int(row['candidate_prefix_truncated'])))
 table(out/'M_AGGREGATES_VERIFIED',summaries);table(out/'M_OUTPUT_DIAGNOSTICS_VERIFIED',diagnostic)
 return dict(status='PASS',businesses=100,component_rows=700,aggregate_rows_checked=7,metric_means_checked=42,paired_CIs_checked=len(pairs),primary_pvalues_and_Holm_checked=4,new_model_or_metric_computation=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 assert not a.output.exists(),'Write only to a fresh reproduction output directory';a.output.mkdir(parents=True)
 original_hashes={r['package_file']:sha(a.package/r['package_file']) for r in read(a.package/'SOURCE_INVENTORY.json')['member_copies']}
 results={'P':analyze_p(a.package,a.output),'M':verify_m(a.package,a.output),'N':verify_no(a.package,a.output,'N'),'O':verify_no(a.package,a.output,'O')}
 for relative,expected in original_hashes.items():assert sha(a.package/relative)==expected
 results.update(status='PASS',source_copies_unchanged=True,python=sys.version,numpy=np.__version__,no_rerating=True,no_model_calls=True,no_cross_stage_pooling=True)
 save(a.output/'VERIFICATION_RESULTS.json',results);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
