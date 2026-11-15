"""Static scientific figures from frozen derived tables; no model execution."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();root=a.package
 out=root/'FIGURES';out.mkdir(exist_ok=True)
 labels=['Review\nfaithfulness','Aspect\ncoverage','Coherence','Non-\nredundancy','Review\nusefulness','Visual grounding\nand contribution']
 criteria=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
 colors=['#34495e','#bd6b25','#2b7d82','#8d659a']
 fig,axs=plt.subplots(3,1,figsize=(11,11),layout='constrained')
 for ax,stage,models,title in zip(axs,['N','O','P'],[['RO','U','F','S'],['RO','U','F'],['LED F original','Qwen-Caption','Qwen-Image']],['Stage N · Yelp TEST30 · 240 rating records','Stage O · TripAdvisor15 · 90 rating records','Stage P · TripAdvisor15 · original V1 · 90 rating records']):
  data=json.loads((root/'VERIFIED_TABLES'/f'{stage}_DESCRIPTIVES{ "" if stage=="P" else "_VERIFIED"}.json').read_text());lookup={(r['model'],r['criterion']):r['mean'] for r in data if r['panel']=='COMBINED'}
  width=.8/len(models);x=np.arange(6)
  for i,model in enumerate(models):
   values=[lookup[model,c] for c in criteria];positions=x+(i-(len(models)-1)/2)*width
   for pos,value in zip(positions,values):
    if value is None:ax.text(pos,1.08,'N/A',ha='center',va='bottom',fontsize=7,rotation=90,color=colors[i])
   ax.bar(positions,[float('nan') if v is None else v for v in values],width=width*.94,label=model,color=colors[i])
  ax.set(ylim=(1,5.65),yticks=[1,2,3,4,5],xticks=x,xticklabels=labels,ylabel='Mean score (1–5)',title=title)
  ax.legend(loc='upper left',ncol=4,frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Separate-stage descriptive quality scores · higher is better\nNo cross-stage pooling or composite ranking',fontsize=14)
 fig.savefig(out/'HUMAN_MEANS_SEPARATE_STAGES.png',dpi=180);fig.savefig(out/'HUMAN_MEANS_SEPARATE_STAGES.svg');plt.close(fig)
 rows=json.loads((root/'VERIFIED_TABLES/P_PAIRED_COMPARISONS.json').read_text())
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
 for ax,left,right in zip(axs,['LED F original','Qwen-Caption'],['Qwen-Caption','Qwen-Image']):
  selected=[next(r for r in rows if r['left']==left and r['right']==right and r['criterion']==c) for c in criteria]
  for i,r in enumerate(selected):
   value=r['mean_difference'];color='#2b7d82' if r['holm_p_within_family']<.05 else '#697481'
   ax.errorbar(value,i,xerr=[[value-r['pointwise_CI95_low']],[r['pointwise_CI95_high']-value]],fmt='o',capsize=4,color=color)
   ax.text(.98,i,f"Holm p={r['holm_p_within_family']:.4f}",transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
  ax.axvline(0,color='black',lw=.8);ax.set(xlim=(-2.25,.65),xticks=[-2,-1,0],yticks=range(6),yticklabels=[s.replace('\n',' ') for s in labels],xlabel='Mean left − right (score points)',title=left+' − '+right)
  ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
 axs[0].invert_yaxis();fig.suptitle('Stage P primary contrasts · 15 paired restaurants\nPointwise 95% bootstrap intervals; Holm across 12 primary tests',fontsize=13)
 fig.savefig(out/'P_PRIMARY_PAIRED_INTERVALS.png',dpi=180);fig.savefig(out/'P_PRIMARY_PAIRED_INTERVALS.svg');plt.close(fig)
 print('Four static figure files written.')
if __name__=='__main__':main()
