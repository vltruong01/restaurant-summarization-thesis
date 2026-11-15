"""Create readable reports/plots from already computed frozen Stage O statistics."""
import argparse,json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'review_faithfulness':'Review faithfulness','aspect_coverage':'Aspect coverage','coherence':'Coherence','non_redundancy':'Non-redundancy','usefulness':'Review usefulness','visual_grounding_and_contribution':'Visual grounding & contribution'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def md(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s.rstrip()+'\n',encoding='utf8')
def put(p,x):md(p,json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def fmt(v,n=3):return 'N/A' if v is None else f'{v:.{n}f}'
def tab(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def qualitative(p):
    outputs=read(p/'DATA/ORIGINAL_OUTPUTS_45.json');by={(r['business_id'],r['condition']):r for r in outputs};sources={f.stem:read(f) for f in (p/'DATA/SOURCES').glob('*.json')};examples=[]
    def case(id,b,c,component,span,review_ids,interpretation,status='MODEL_ASSISTED_SOURCE_CHECKED_EXAMPLE',photos=None,excerpt=None):
        text=by[b,c]['component_texts'][component];assert span in text
        s=sources[b];reviews=[]
        for rid in review_ids:
            r=next(r for r in s['reviews'] if r['review_id']==rid);q=(excerpt or {}).get(rid,r['text']);assert q in r['text'];reviews.append(dict(review_id=rid,exact_excerpt=q))
        examples.append(dict(example_id=id,business_id=b,business_name=s['business_metadata']['name'],condition=c,component=component,exact_output_span=span,reference_free_evidence=reviews,photo_ids=photos or [],assessment=interpretation,assessment_status=status,not_a_new_rating=True))
    b='TA14622324_2298952'
    case('E01',b,'RO','review','though some selected reviewers report that the owner did not expect them to enjoy the meal.',['TA_REVIEW_256867909'],"Attribution reversal: the visiting group did not expect to enjoy it; the owner accommodated them. The complete 12 selected reviews were read for this bounded example, not all 247 reviews in the study.")
    case('E02',b,'U','review','while others find the prices excessive or the portions not as large as expected.',['TA_REVIEW_860592117','TA_REVIEW_424622783','TA_REVIEW_905248984'],"The source explicitly negates excessive prices and describes large portions. No support for the contrary formulation was found among the 12 selected reviews.")
    case('E03',b,'F','review','while one reviewer found the desserts not too special.',['TA_REVIEW_424622783'],"F preserves a specific qualified dessert opinion omitted from RO/U. This local coverage advantage does not erase F's unsupported not-always-welcoming claim: all selected service accounts here are positive. It is not an overall F win.")
    b='TA14622324_9802783'
    case('E04',b,'RO','review','while others object to paying more for the type of place or cider quality.',['TA_REVIEW_471024507'],"The source discusses its rating of the kind of venue and explicitly says it did not pay/cannot comment on price. All 10 selected reviews were read; this is not a source-supported price complaint. All RO/U/F also insert unsupported waits/inattentive/poor service in this venue.")
    b='TA14622324_23652638';rids=['TA_REVIEW_812571700','TA_REVIEW_813651605','TA_REVIEW_813652082']
    case('E05',b,'F','review',by[b,'F']['component_texts']['review'].split('\n',1)[1],rids,"Coverage omission: none of the three frozen Review outputs mentions the no-alcohol restriction. Three source entries mention it, but the latter two are very similar and must not be presented as three independently verified opinions. This example establishes source content and omission, not a gold-standard coverage score.",excerpt={rids[0]:"it doesn't serve alcohol",rids[1]:"The worst part is there is no alcohol.",rids[2]:"And the worst - no alcohol."})
    b='TA14622324_3512687'
    case('E06',b,'U','visual','A plate of food is on a table with a fork and knife. A plate of food is on a table with a knife and fork. A plate of food is on a table with a fork and knife.',[],"Exact and near-duplicate wording is directly visible in the frozen output. F is also imperfect (relaxing, relaxing in Review) but replaces this table/cutlery repetition with other visual descriptions. This is a text comparison, not a fresh certification of every visual claim.",status='MODEL_ASSISTED_TEXT_CHECKED_EXAMPLE')
    b='TA14622324_2312071'
    case('E07',b,'F','visual','A man is pouring wine into a green bottle.',[],"The inspected source photo shows a raised green bottle angled toward the small glass in the other hand, rather than liquid being poured into the bottle. Wine identity cannot be established from this photograph. This checks one localized relational claim, not the entire 2,040-photo gallery or a claim about all possible unseen scenes.",photos=['TA_PHOTO_19c9f445'])
    b='TA14622324_2687285'
    case('E08',b,'U','visual','The photo shows a receipt for a meal at a Mexican restaurant.',[],"Two inspected receipt images show El Planeta / CIMAVILLA-GIJON. They do not establish a Mexican restaurant. The receipt-location attribution is contradicted by these cited evidence images; this is not an exhaustive search of every photograph.",photos=['TA_PHOTO_07732ad1','TA_PHOTO_1b958219'])
    b='TA14622324_1119478'
    case('E09',b,'F','review','with helpful recommendations for food and drink.',['TA_REVIEW_849637952'],"A source-supported positive example: explicit cheese/tapas recommendations are retained by F. U's first two Review sentences repeat food/drink praise and its two receipt sentences repeat substantially the same information. These are local observations, not a universal condition ranking.")
    b='TA14622324_6600483'
    visual=by[b,'F']['component_texts']['visual'];span=next(s.strip() for s in visual.split('.') if 'onion' in s.lower())+'.'
    case('E10',b,'F','visual',span,[],"Participants questioned onion-ring identity, and A02 explicitly treated the ingredient concern as uncertain. This analysis has not independently inspected that food photograph. Retain as uncertain participant interpretation, not a confirmed semantic error.",status='PARTICIPANT_REPORTED_UNCERTAIN_NOT_INDEPENDENTLY_VERIFIED')
    put(p/'QUALITATIVE/SOURCE_CHECKED_EXAMPLES.json',examples)
    chunks=['# Qualitative findings and limits','All 90 original comments were read; their exact text remains in TABLES/RATER_COMMENTS_VERBATIM.json. The following theme synthesis and ten illustrations are model-assisted analysis, not additional human ratings. Examples were selected after seeing comments/results to illustrate attribution, polarity, omissions, repetition, supported content and uncertainty. They are purposive and cannot estimate error prevalence. Full selected review sets were read only for the two explicitly identified venues; three cited original photos were inspected. No exhaustive independent review of 5,143 images is claimed.','## Participant reports across all restaurants']
    themes={
      '1001674':'Reports of unsupported slow/inattentive service, omitted tasting-menu/wine/dessert details; visual gender inference questioned.',
      '1119478':'Generally positive grounded Review descriptions; English-language assistance omitted. U repeats food praise and receipt content; F visual gender inference uncertain.',
      '2298952':'Owner/visitor attribution reversal in RO, price/portion polarity errors in U; F preserves portions/dessert qualification but adds negative welcome claim.',
      '2312071':'Repeated distinctive decor, omitted mixed food/portion/cider qualifications; F pouring-direction, sign and dog-scene concerns.',
      '23652638':'No-alcohol policy omitted; some food/space complaints placed under service; U clock and glass-type concerns.',
      '2687285':'RO/F omit food-quality criticisms; U captures mixed food but misattributes receipt to Mexico; F dried-fish/glass relation uncertain.',
      '3410297':'Dish/category and on-demand attribution errors; single mentions overgeneralized as repeated. Cold-food/portion qualifications omitted; F receipt-on-wall uncertain.',
      '3512687':'Cachopo/value repetition and missing cider/booking qualifications; U repeats cutlery/table descriptions, F serving-food action uncertain.',
      '3961117':'Service negatives unsupported according to raters; authenticity/ingredient qualifications omitted; food-image identities uncertain.',
      '4004593':'Staff-finding/table incident overgeneralization; move/eat confusion; menu-day/language help omissions; broadly supported visuals reported.',
      '4735772':'Large-portion praise recast as quality complaint in RO/U; price/calamari/portion advice omitted; F map called city picture.',
      '6600483':'Nearby-hotel attribution and price interpretation concerns; tough octopus omission; F onion-ring identity remains uncertain.',
      '7345192':'RO adds negative value claim; U/F retain core coffee/seating issues but omit alfajores/milk-alternative detail; visual sandwich wording ambiguous.',
      '8556009':'Underpriced polarity concern across conditions; hygiene/order/allergy omissions; F liquor/coffee ownership and U pizza identity uncertain.',
      '9802783':'Unsupported negative service/value framing shared by all three outputs; family/cider/portion details incompletely retained.'}
    chunks.append(tab(['Restaurant ID / name','Themes reported by participants (not blanket confirmation)'],[[b+' / '+s['business_metadata']['name'],themes[b.split('_')[-1]]] for b,s in sorted(sources.items())]))
    chunks+=['## Bounded source and text checks']
    for e in examples:
        chunks += [f"### {e['example_id']} — {e['business_name']} / {e['condition']} / {e['component']}",f"Status: `{e['assessment_status']}`",'> '+e['exact_output_span'],e['assessment']]
        for r in e['reference_free_evidence']:chunks.append(f"- {r['review_id']}: {r['exact_excerpt']}")
        for ph in e['photo_ids']:chunks.append(f'- [{ph}](PHOTOS/{ph}.jpg)')
    chunks+=['## What the flags do and do not establish','The exported serious flags are participant-level observations; both raters can flag the same output. Even a stored certainty of confirmed is a rater label, not independent adjudication. Nine flagged F records versus fourteen U records cannot be reported as nine versus fourteen unique confirmed errors. Some concerns appear in notes without a serious flag, so absence of a flag is not evidence of an error-free output. We did not change scores based on these examples.','Review-only comparisons also vary in training checkpoint history; source separation removes caption text from Review input but does not eliminate parametric priors, generic negative templates or attribution errors. Fluent summaries and successful EOS termination do not demonstrate factual correctness.']
    md(p/'QUALITATIVE_FINDINGS.md','\n\n'.join(chunks))

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--package',required=True,type=Path);p=pa.parse_args().package
    cfg=read(p/'ANALYSIS_CONFIG.json');cs=cfg['criteria'];conds=cfg['conditions'];d=read(p/'TABLES/DESCRIPTIVES.json');a=read(p/'TABLES/AGREEMENT.json');pairs=read(p/'TABLES/PAIRED_COMPARISONS.json');g=read(p/'TABLES/GENERATION_DIAGNOSTICS.json');flags=read(p/'TABLES/PARTICIPANT_FLAG_COUNTS.json');cross=read(p/'TABLES/YELP_COMPARISON.json')
    qualitative(p)
    desc=['# Descriptive statistics','Scores are ordinal 1–5, higher is better. COMBINED describes 30 observed scores per applicable condition/criterion from 15 restaurants and two fixed raters; SD is sample SD (ddof=1). These are not 30 independent restaurants. RO Visual is structural N/A. IQR uses linear quantiles. No missing applicable scores, imputations, exclusions or overall composite.']
    for panel in ['COMBINED','A01','A02']:
        desc += ['## '+panel,tab(['Criterion','Condition','Valid / N/A','Mean','SD','Median','Q25–Q75','Scores 1 / 2 / 3 / 4 / 5'],[[LABELS[r['criterion']],r['condition'],f"{r['n_valid']} / {r['n_NA']}",fmt(r['mean']),fmt(r['sample_sd']),fmt(r['median']),fmt(r['q25'])+'–'+fmt(r['q75']),' / '.join(str(r[f'score_{s}']) for s in range(1,6))] for r in d if r['panel']==panel])]
    md(p/'DESCRIPTIVE_STATISTICS.md','\n\n'.join(desc))
    md(p/'INTER_RATER_AGREEMENT.md','# Inter-rater agreement\n\nFixed categories 1–5, quadratic weights (i-j)^2/16; QWK = 1 − observed/expected weighted disagreement. Each cell pairs the same 15 outputs across A01/A02. Exact agreement and signed differences are also reported; no reliability CI or pooled-condition reliability claim was prespecified.\n\n'+tab(['Condition','Criterion','QWK','Exact %','Within 1 %','Mean A01−A02','Mean absolute difference'],[[r['condition'],LABELS[r['criterion']],fmt(r['qwk']),fmt(r['exact_pct'],1),fmt(r['within_one_pct'],1),fmt(r['mean_A01_minus_A02']),fmt(r['mean_absolute_difference'])] for r in a])+'''\nThe agreement range is 0.000–0.904. F Coherence has QWK 0.000 despite 66.7% exact agreement: A02 used only score4, while A01 used4/5. Restricted marginal variation makes kappa prevalence-sensitive; neither the exact rate nor kappa alone certifies rating quality. RO Coverage QWK0.167 coexists with73.3% exact agreement for another restricted-range cell.

Visual agreement is more limited: U QWK0.536/exact40.0%, F QWK0.490/exact53.3%. A01 is higher than A02 by0.733 on U Visual and0.533 on F Visual. Three of30 visual pairs differ by2 points, while26/30 are within1? See exact cell totals below: U13/15 and F14/15, or27/30 within1. The rating differences are retained; no rater was removed or scores harmonized. TABLES/AGREEMENT_PAIRS and AGREEMENT_CONFUSIONS preserve all paired scores and matrices.\n'''.replace('Three of30 visual pairs differ by2 points, while26/30 are within1? See exact cell totals below: U13/15 and F14/15, or27/30 within1.','Three of30 visual pairs differ by2 points; U13/15 and F14/15 (27/30) are within1.'))
    md(p/'PAIRED_COMPARISONS.md','# Restaurant-level paired comparisons\n\nPositive differences favor F. Both raters are averaged within restaurant; all15 restaurants enter every applicable comparison. Intervals are pointwise95% paired percentile bootstrap (20,000 draws); p is raw exploratory two-sided50,000-draw sign flip with+1 correction. Seed20261028, numpy default_rng/PCG64; common indices/signs retained across criteria. No multiplicity adjustment, confirmatory winner or equivalence claim.\n\n'+tab(['Contrast','Criterion','Difference','95% CI','Raw p','F higher/tie/lower','Scope'],[[r['contrast'],LABELS[r['criterion']],fmt(r['mean_difference']),f"[{fmt(r['CI95_low'])}, {fmt(r['CI95_high'])}]",fmt(r['raw_signflip_p'],6),f"{r['higher']}/{r['tie']}/{r['lower']}",'Supplementary' if r['scope'].startswith('SUPPLEMENTARY') else 'Frozen exploratory plan'] for r in pairs])+'''\nF–U Non-Redundancy is the clearest favorable signal (+0.833, CI[0.400,1.333], raw p0.004740; ten higher, four ties, one lower). Both raters show this direction (+0.800/+0.867). This is an exploratory result across multiple criteria with fifteen units, not proof of uniform overall superiority.

F–U Visual is −0.167, CI[−0.633,0.300], raw p0.601748. Review faithfulness is +0.200, CI[−0.200,0.567], raw p0.420472; coverage −0.033, CI[−0.233,0.200], p1.000000. These are inconclusive, not equivalence. F–RO Review-only comparisons likewise provide no clear advantage. CIs touching zero and p-values are reported together, without selecting whichever looks favorable.

F–RO Coherence/Non-Redundancy are supplementary: RO has only Review, whereas F adds Visual. These were not explicitly named as Review-only contrasts in the frozen Stage O amendment and are not clean equal-output-budget comparisons. No Visual comparison with RO is valid.\n''')
    gen=[]
    for c in conds:
        for t in ['review','visual']:
            rr=[r for r in g if r['condition']==c and r['component']==t]
            if rr:gen.append([c,t,len(rr),fmt(np.mean([r['words'] for r in rr]),1),fmt(np.mean([r['generated_token_ids'] for r in rr]),1),sum(r['empty'] for r in rr),sum(r['at_length_limit'] for r in rr),sum(r['duplicate_exact_sentences']>0 for r in rr),fmt(np.mean([r['repeated_4gram_fraction'] for r in rr]))])
    md(p/'GENERATION_AND_FLAGS.md','# Generation diagnostics and participant flags\n\n'+tab(['Condition','Component','n','Mean words','Mean token IDs','Empty','Length cap','Outputs with repeated exact sentences','Mean repeated4-gram fraction'],gen)+'\nExact-sentence matching strips sentence punctuation and lowercases; repeated4-gram fraction=(total−unique)/total within a component. These mechanical, posthoc descriptive diagnostics are not semantic repetition or hallucination labels. Token-ID counts include stored special tokens; word counts use a documented regex and exclude section headers. No outputs were regenerated. All75 native components have EOS, expected header, no foreign header, no empty output or cap termination. Successful generation does not mean factual correctness.\n\n'+tab(['Panel','Condition','Records','Notes','Serious flags','Rater marked confirmed'],[[r['panel'],r['condition'],r['n_records'],r['nonempty_notes'],r['serious_flags'],r['rater_marked_confirmed']] for r in flags])+'''\nAll90 records contain comments. At least one rater flagged7/15 RO outputs,9/15 U and6/15 F; both raters flagged4,5,3 respectively. These are participant flag counts, not independently confirmed error rates. Counts span Review and Visual and differ in component applicability. Confirmed/uncertain language within free-text notes can differ from structured flags; original notes/fields are preserved.\n''')
    md(p/'YELP_COMPARISON.md','# Comparison with Yelp TEST30\n\nYelp numbers below are byte-preserved accepted Stage N result tables, not a rerun. TripAdvisor uses15 external restaurants versus Yelp30 TEST businesses. No pooled cross-dataset test or posthoc checkpoint selection was performed.\n\n'+tab(['Contrast','Criterion','TripAdvisor difference [CI]','Yelp difference [CI]'],[[r['contrast'],LABELS[r['criterion']],f"{fmt(r['TripAdvisor_difference'])} [{fmt(r['TripAdvisor_low'])}, {fmt(r['TripAdvisor_high'])}]",f"{fmt(r['Yelp_difference'])} [{fmt(r['Yelp_low'])}, {fmt(r['Yelp_high'])}]"] for r in cross])+'''\nThe non-redundancy direction favors F over U in both datasets, but Yelp's F–U interval includes zero (+0.250 [−0.117,0.633]); the current external sample has the clearer exploratory signal. F–U Visual intervals include zero on both datasets (Yelp−0.100 [−0.383,0.200]; TripAdvisor−0.167 [−0.633,0.300]). This does not demonstrate consistent F superiority over U or equivalence.

Yelp's coverage/usefulness disadvantages for F versus U are not clearly reproduced externally, but a change from a detectable effect to an inconclusive effect is not itself a statistically established dataset interaction. Yelp's strong F–Shuffled Visual difference (+2.667 [2.300,3.017]) has no Shuffled counterpart here.

This is reference-free evaluation of frozen task-fine-tuned models on externally sourced restaurants against reviews/photos. No TripAdvisor Teacher answers, new reference summaries or reference-agreement metrics are used. Thus it assesses observed external behavior rather than agreement with Teacher wording. It does not prove independence from Teacher-derived training style, absence of model pretraining contamination, or causal filtering benefit. The small, English-review-eligible Gijon-area sample, different photo counts/resolution, review mix, fixed raters and source applicability preclude broad population or causal domain-transfer claims. Yelp's preserved audit/acceptance limitations remain in YELP_TEST30; it is not a perfect ground-truth calibration.\n''')
    md(p/'METHODOLOGY_AND_LIMITATIONS.md','''# Methodology and limitations

## Inputs and preservation
The final unchanged export contains90 submitted records,45 frozen RO/U/F outputs,15 restaurants,two account IDs,510 numeric scores and30 structural RO Visual N/As. COLLECTION contains the original export, assignment key, raw rows,138-event journal, verification script and actual completion/provenance receipts. DATA includes exact native outputs/token IDs/input IDs/call plan and all selected reviews/source ledgers. No live database was opened for writing, no score revised, no restaurant excluded. PROVENANCE records original file hashes and bytewise photo verification. Checkpoint weights were not newly loaded or rehashed; native records retain the existing adapter hashes and generation settings.

## Design and analysis
The frozen two-rater amendment prescribes20,000 paired restaurant bootstrap draws,50,000 two-sided sign flips,+1 correction and seed20261028. Two original raters are arithmetically averaged within each restaurant. Fifteen restaurant differences are the inferential units, not90 rating rows or30 independent rater observations. Both raters/all conditions travel together under common bootstrap indices. Linear2.5/97.5 percentiles give pointwise95%CIs; the sign-flip test counts |mean(sign*d)| >= |mean(d)| with1e-15 numerical tolerance. It assumes exchangeability/sign symmetry under a zero-effect null, not random experimental allocation to condition. P-values are exploratory and unadjusted across nine frozen-plan contrasts plus two separately labeled supplementary contrasts. No composite or reliability threshold is used to delete observations. Fixed-category quadratic kappa is computed by condition; undefined zero-expected-disagreement values would remain null rather than forced to1. Descriptives use sample SD and linear IQR on the observed ordinal scores. No claim of an interval scale's exact psychological spacing is made.

## Raters, provenance and blinding
Two human participants are reported by the researcher. Account IDs, submissions and journal integrity establish recorded participation, not independently authenticated human identity, unaided authorship, independence or complete source inspection. An earlier A01 declaration described an AI-assistant test context. The researcher later identified that declaration as a test and authorized direct access without declaration/coordinator gates. The original declaration and waiver are retained unchanged in COLLECTION, not silently replaced by a new eligibility assertion. Participants' comments claiming all photos were viewed are their assertions; logs/hashes do not independently establish that. Different RO applicability and output style limit perfect condition blinding. This report's thematic synthesis and bounded source checks are model-assisted analysis, not replacement human ratings or external adjudication.

## External evidence and failure interpretation
The sources comprise247 selected reviews and5,143 frozen acquired photographs across15 restaurants; photo count does not equal5,143 independent semantic scenes. Small low-resolution publisher-linked photos and near-duplicate reviews can limit judgments. Raters saw all frozen photos whereas F generation used the bounded filtered caption subset, so Visual contribution/coverage is interpreted with that input budget in mind. All sources were byte-verified, but analysis independently viewed only three illustrative photos. Ten purposive qualitative examples are transparent and do not estimate error prevalence. Generation diagnostics certify recorded termination/format, not absence of hallucination. Serious flags are rater-reported; conflicting interpretations remain visible. Generalization outside this cohort, rater-population inference, Teacher-style independence, pretraining contamination absence and causal architecture/filtering isolation are not established.

## Reproducibility
Run from an extracted package with Python3.11+ and the pinned SCRIPTS/requirements.txt dependencies. `python SCRIPTS/verify_stage_o_final_statistics_v1.py --package .` verifies package hashes and the original journal. `python SCRIPTS/analyze_stage_o_final_statistics_v1.py --package . --output REPRODUCED_TABLES` reconstructs all tables without GPU/model downloads/APIs/full source archives. Numerical table hashes are compared with TABLES; RUNTIME varies by environment. `python SCRIPTS/report_stage_o_final_statistics_v1.py --package .` regenerates reports/figures from those saved TABLES. No original record or input file is edited by the analyzer. Original large photo archives and checkpoint weights are excluded; source/photo hash ledgers and three illustrative original JPEGs are included. Reviewing the full photo galleries requires the preserved original app workspace.
''')
    # Figures: no error bars that could be mistaken for confidence on raw pooled scores.
    figdir=p/'FIGURES';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'RO':'#697586','U':'#d28a23','F':'#187f82'}
    fig,axes=plt.subplots(2,3,figsize=(12,7));
    for ax,crit in zip(axes.flat,cs):
        for i,c in enumerate(conds):
            r=next(v for v in d if v['panel']=='COMBINED' and v['condition']==c and v['criterion']==crit)
            if r['mean'] is None:ax.text(i,1.2,'N/A',ha='center');continue
            ax.bar(i,r['mean'],color=colors[c],width=.65);ax.text(i,r['mean']+.1,fmt(r['mean'],2),ha='center')
        ax.set(xticks=range(3),xticklabels=conds,ylim=(0,5.5),title=LABELS[crit],ylabel='Mean score (1–5)')
    fig.suptitle('TripAdvisor15: descriptive means (two fixed raters; 30 scores per applicable cell)');fig.tight_layout();fig.savefig(figdir/'MEAN_SCORES.png',dpi=170);fig.savefig(figdir/'MEAN_SCORES.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
    for ax,contrast in zip(axes,['F-U','F-RO']):
        rr=[next((r for r in pairs if r['contrast']==contrast and r['criterion']==crit),None) for crit in cs]
        for i,r in enumerate(rr):
            if r is None:ax.text(.05,i,'Not applicable');continue
            col='#697586' if r['scope'].startswith('SUPPLEMENTARY') else colors['F'];ax.plot([r['CI95_low'],r['CI95_high']],[i,i],color=col,lw=2);ax.scatter(r['mean_difference'],i,color=col,zorder=3)
        ax.axvline(0,color='black',ls='--',lw=.8);ax.set(yticks=range(6),yticklabels=[LABELS[c] for c in cs],xlabel='F minus comparator (score points)',title=contrast,xlim=(-1,1.5));ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.suptitle('Restaurant-paired differences and pointwise95% bootstrap CIs (n=15)\nGray: supplementary RO comparison across different component budgets');fig.tight_layout();fig.savefig(figdir/'PAIRED_DIFFERENCES.png',dpi=170);fig.savefig(figdir/'PAIRED_DIFFERENCES.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7));palette=plt.cm.RdYlGn(np.linspace(.1,.9,5))
    for ax,crit in zip(axes.flat,cs):
        for j,c in enumerate(conds):
            rr=next(v for v in d if v['panel']=='COMBINED' and v['condition']==c and v['criterion']==crit);left=0
            for score in range(1,6):
                val=rr[f'score_{score}']/rr['n_valid']*100 if rr['n_valid'] else 0;ax.barh(j,val,left=left,color=palette[score-1],label=str(score) if j==0 else None);left+=val
            if not rr['n_valid']:ax.text(5,j,'N/A')
        ax.set(yticks=range(3),yticklabels=conds,xlim=(0,100),title=LABELS[crit],xlabel='% of observed ratings')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',title='Score');fig.suptitle('All retained scores: 15 restaurants × two raters');fig.tight_layout(rect=(0,.08,1,.97));fig.savefig(figdir/'SCORE_DISTRIBUTIONS.png',dpi=170);fig.savefig(figdir/'SCORE_DISTRIBUTIONS.svg');plt.close(fig)
    md(p/'README.md','''# Stage O final TripAdvisor statistical review

Completed analysis:15 restaurants,45 frozen RO/U/F outputs,two recorded evaluators,90 unchanged submitted ratings. No training, reference generation, regrading, exclusions or Qwen work.

## Findings
- F scores higher than U for Non-Redundancy:3.833 versus3.000; paired difference+0.833, pointwise95%CI[0.400,1.333], raw exploratory p0.004740. Ten restaurants favor F,four tie,one favors U.
- F does not show clear superiority over U for Review faithfulness,coverage,usefulness or Visual Grounding. Visual F3.533 versus U3.700; difference−0.167,CI[−0.633,0.300],p0.601748. This is inconclusive, not equivalence.
- F–RO Review-only contrasts are inconclusive. RO has no Visual component; it is never assigned zero. Fluency and good termination coexist with attribution,polarity and omission concerns.
- QWK varies from0.000 to0.904. U/F Visual agreement is limited and A01 scores Visual higher than A02. Restricted score variation explains some low kappa cells. Original ratings are retained.
- The non-redundancy direction is consistent with Yelp TEST30, but this external small cohort does not establish general F superiority,causal filtering benefit or Teacher-style independence.

## Files
DESCRIPTIVE_STATISTICS.md includes pooled/per-rater mean,SD,median,IQR and complete distributions. PAIRED_COMPARISONS.md and INTER_RATER_AGREEMENT.md report all applicable comparisons and reliability. QUALITATIVE_FINDINGS.md separates original participant flags from ten bounded model-assisted source/text illustrations. GENERATION_AND_FLAGS.md covers all75 native components. YELP_COMPARISON.md retains dataset distinctions. METHODOLOGY_AND_LIMITATIONS.md explains inferential and participation-provenance limits. TABLES stores machine-readable JSON/CSV; FIGURES contains PNG/SVG. COLLECTION is the byte-preserved accepted completion handoff; DATA preserves exact reviews,outputs,native token/input IDs and source manifests. YELP_TEST30 contains existing accepted comparison tables/limitations. PROVENANCE includes checks and file/photo hashes. Three cited photos only are in QUALITATIVE/PHOTOS; full galleries remain in the original workspace.

## Verification and reproduction
See METHODOLOGY_AND_LIMITATIONS.md for commands. The analyzer requires only frozen small files and CPU Python dependencies, no GPU or model download. VERIFY_PACKAGE checks every manifest member and the original90-record/138-event journal. Ratings/assignments are immutable inputs; tables and reports are derived. Preserved declarations and direct-access waiver do not independently authenticate human identities/source inspection. Technical integrity PASS is not semantic certification.

Read SUMMARY.md first. No new research stage is authorized by this package.
''')
    md(p/'SUMMARY.md','# Final statistical summary\n\n'+tab(['Criterion','RO mean±SD','U mean±SD','F mean±SD'],[[LABELS[crit]]+[next((fmt(v['mean'])+' ± '+fmt(v['sample_sd']) if v['mean'] is not None else 'N/A') for v in d if v['panel']=='COMBINED' and v['condition']==c and v['criterion']==crit) for c in conds] for crit in cs])+'''\nF's clearest external advantage is reduced repetition relative to U (+0.833,[0.400,1.333],raw exploratory p0.004740). Review correctness (+0.200,[−0.200,0.567]) and Visual Grounding (−0.167,[−0.633,0.300]) do not establish F superiority over U. Review coverage remains around3.2 for all three conditions; comments identify omissions of important mixed opinions and venue details. F–RO Review-only differences are also inconclusive.

All15 restaurants,both raters,and original90 records are retained. Nine prespecified exploratory contrasts and two supplementary RO component-budget contrasts are fully reported. There is no composite winner score, no claim of equivalence from non-significance, and no independent confirmation of every participant flag. Ten source/text examples include correct retained details and shared failure modes, not just favorable F cases. Numerical calculations passed independent reconstructions and cross-library agreement checks; see PROVENANCE/FINAL_VALIDATION.json for archive/reproduction status.
''')
    print('Reports and three PNG/SVG figure pairs created from unchanged tables.')
if __name__=='__main__':main()
