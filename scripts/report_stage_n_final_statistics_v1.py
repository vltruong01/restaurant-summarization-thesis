"""Report measured Stage N results and bounded source-grounded interpretation."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('outputs/stage_n/final_human_statistical_analysis_20261102_v1')
P=ROOT/'PACKAGE';T=P/'TABLES'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def md(path,text):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n',encoding='utf8')


def main():
    desc=read(T/'DESCRIPTIVE_STATISTICS.json');agreements=read(T/'AGREEMENT.json');paired=read(T/'PAIRED_COMPARISONS.json')
    rows=read(P/'DATA/VERIFIED_ANALYSIS_INPUT.json')['records'];outputs=read(P/'DATA/OUTPUTS_120.json');sources=read(P/'DATA/SOURCES_30.json')
    cfg=read(P/'ANALYSIS_CONFIG.json');criteria=cfg['criteria'];conditions=cfg['conditions']
    names=['Review Faithfulness','Aspect Coverage','Coherence (all available)','Non-Redundancy (all available)','Usefulness (Review)','Visual Grounding/Contribution']
    pretty=dict(zip(criteria,names));d={(x['condition'],x['criterion'],x['panel']):x for x in desc}
    rr={(x['condition'],x['business_id'],x['annotator']):x for x in rows};oo={(x['condition'],x['business_id']):x for x in outputs}
    mm={(x['contrast'],x['criterion']):x for x in paired}
    table=['| Criterion | RO | U | F | S |','|---|---:|---:|---:|---:|']
    for crit in criteria:table.append('| '+pretty[crit]+' | '+' | '.join('N/A' if d[c,crit,'COMBINED']['mean'] is None else f"{d[c,crit,'COMBINED']['mean']:.3f}" for c in conditions)+' |')
    mean_table='\n'.join(table)
    descriptive=['# Descriptive statistics','',
        'All30 businesses and both raters retained. Each applicable pooled cell has60 scores; each rater has30. SD is sample SD, not a standard error or an assumption of60 independent businesses. Quartiles use linear interpolation; IQR=Q75-Q25. Higher is better. No overall composite score.','',mean_table,'',
        '| Condition | Criterion | Panel | Valid | N/A | Mean | SD | Median | Q25 | Q75 | IQR | 1 | 2 | 3 | 4 | 5 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in desc:
        vals=[x['condition'],pretty[x['criterion']],x['panel'],x['n_valid_ratings'],x['n_structural_NA']]
        vals += ['N/A' if x[k] is None else f'{x[k]:.4f}' for k in ['mean','sample_sd','median','q25','q75','iqr']]
        vals += [x['score_'+str(i)] for i in range(1,6)]
        descriptive.append('| '+' | '.join(map(str,vals))+' |')
    md(P/'DESCRIPTIVE_STATISTICS.md','\n'.join(descriptive))
    agr=['# Inter-rater agreement','',
        'Primary: fixed1–5 quadratic-weighted Cohen kappa per condition/criterion,30 paired items each. Exact agreement is a distinct measure. Difference distributions use A01 minus A02. Full5×5 confusion matrices and paired scores are supplied as CSV. Ordinal alpha and agreement CIs were explicitly not part of the frozen two-rater revision and are not added.','',
        '| Condition | Criterion | Paired items | QWK | Exact % | Within1 % | Mean A01−A02 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for x in agreements:
        agr.append(f"| {x['condition']} | {pretty[x['criterion']]} | {x['n_paired_items']} | {x['qwk']:.4f} | {x['exact_agreement_pct']:.2f} | {x['within_one_pct']:.2f} | {x['mean_A01_minus_A02']:.3f} |")
    agr += ['', 'Interpretation:',
        '- Review Faithfulness QWK ranges0.838–0.907 across conditions; this is agreement between these two recorded panels, not proof of factual correctness.',
        '- Coherence shows systematic scale-use differences: for S, exact agreement is16.67%, QWK0.460 and A01−A02=−0.867. A02 tends to give higher coherence scores; this is meaningful disagreement, not missing data to reconcile.',
        '- S Visual exact agreement is93.33% but QWK0.565: most scores cluster at the floor, affecting chance-adjusted agreement. High exact agreement or high average scores alone do not establish reliability.',
        '- Pooled Visual QWK0.911 mixes markedly different conditions and is supplementary; it must not replace condition-specific F Visual QWK0.654 or S Visual QWK0.565.',
        '- Only30 paired units per primary cell and two fixed raters; no rater-population reliability or independently authenticated assessor claim.']
    md(P/'INTER_RATER_AGREEMENT.md','\n'.join(agr))
    pairdoc=['# Paired exploratory comparisons','',
        'F minus comparator, in1–5 score points. First average the two raters within business, then retain all30 matched businesses. Common PCG64 seed20261028;20,000 business bootstrap draws and50,000 two-sided sign-flip draws with+1 correction.95% percentile CIs are pointwise and unadjusted. No multiplicity correction or confirmatory winner selection was prespecified for Stage N; raw p-values are exploratory diagnostics. No RO Visual comparison.','',
        '| Criterion | Contrast | Difference | 95% CI | Raw exploratory p | F higher/tie/lower businesses |',
        '|---|---|---:|---|---:|---|']
    for x in paired:
        pairdoc.append(f"| {pretty[x['criterion']]} | {x['contrast']} | {x['mean_paired_difference']:+.4f} | [{x['CI95_low']:+.4f}, {x['CI95_high']:+.4f}] | {x['unadjusted_exploratory_signflip_p']:.6f} | {x['F_higher_businesses']}/{x['tie_businesses']}/{x['F_lower_businesses']} |")
    pairdoc += ['',
        'F−S Visual is the largest positive contrast: +2.667[2.300,3.017]. This supports an alignment-sensitive difference in recorded source-photo assessment for this fixed negative control. It is not a general population causal effect or an adjusted confirmatory result.',
        '', 'F−U Visual is−0.100[−0.383,0.200], p0.582008: no established advantage for either pipeline in this panel; not an equivalence result. F has lower mean Review Coverage than both RO/U (−0.300 each), and lower Usefulness than U (−0.233), as exploratory findings. Review Faithfulness differences against RO/U remain inconclusive.',
        '', 'F−RO Coherence/Non-Redundancy favor RO, but those criteria judge all available components; RO has only Review, whereas F includes Visual, with different output budgets. This is a configuration comparison, not an isolated effect of visual information.',
        '', 'F−RO Usefulness has a negative percentile interval[−0.500,−0.0167] but raw sign-flip p0.078058. These are different procedures on discrete paired data; neither should be selected because it favors a desired claim. Treat the point estimate/uncertainty as exploratory rather than reporting a confirmed significance result.',
        '', 'All30 F/S Review text/token pairs are byte-identical. The three Review-only criteria nevertheless differ in6 of180 participant×business×criterion pairs, across3 A02 business presentations. Any F/S Review rating difference is a presentation/context/measurement difference, not generated Review improvement. Coherence/Non-Redundancy include differing Visual components, so their F/S differences have a different scope.',
        '', 'The fixed without-replacement S donor permutation induces dependencies; ordinary business resampling/sign flipping retains the frozen procedure but does not remove this design limitation. Two raters remain fixed within each sampled business; inference does not generalize over a random population of raters.']
    md(P/'PAIRED_COMPARISONS.md','\n'.join(pairdoc))
    # Deterministic post-hoc disagreement inventory, not a sensitivity exclusion.
    disagreement=sorted(read_csv(T/'AGREEMENT_PAIRED_SCORES.csv'),key=lambda x:(-abs(int(x['A01_minus_A02'])),x['business_id'],x['condition'],x['criterion']))
    top=[]
    for x in disagreement[:12]:
        rs=[rr[x['condition'],x['business_id'],a] for a in ['A01','A02']]
        top.append({**x,'business_name':oo[x['condition'],x['business_id']]['business_name'],
            'A01_record_id':rs[0]['record_id'],'A02_record_id':rs[1]['record_id'],
            'A01_comment':rs[0]['evidence_notes'],'A02_comment':rs[1]['evidence_notes'],
            'selection_rule':'largest absolute A01-A02 criterion difference, then business ID/condition/criterion; descriptive post-hoc inventory'})
    write(P/'QUALITATIVE/TOP_RECORDED_DISAGREEMENTS.json',top)
    flags=[]
    for condition in conditions:
        for rater in ['A01','A02']:
            selected=[r for r in rows if r['condition']==condition and r['annotator']==rater]
            flags.append({'condition':condition,'rater':rater,'n_records':30,'serious_issue_flagged_records':sum(r['serious_grounding_issue'] for r in selected),
                'interpretation':'Participant flag incidence, not independently verified error prevalence.'})
    write(P/'QUALITATIVE/PARTICIPANT_FLAG_COUNTS.json',flags)
    # Verbatim source excerpts are programmatically checked, while the semantic
    # reading remains explicitly model-assisted and illustrative.
    specs=[
        ('NEGATION_POLARITY','rQKL-bO8Xup4gCPcj_oz6w',['RO','F','S'],'some describe it as greasy, oily',[(7,'Way better than Maoz. No greasy, oily food!!')],
         'PlantPure sources praise no/little added oil and contrast other restaurants. Greasy/oily in recipient Review reverses polarity/attribution. U avoids this assertion but remains generic and omits defining vegan/no-oil and actual negatives.'),
        ('WRONG_PREPARATION_DETAIL','D_CMgRM8nnE94i_newJi9g',['F','S'],'undercooked hot dogs, underseasoned cheese',[(13,'it again was just over cooked hot dogs'),(13,"The cheese wasn't even melted on the sandwich.")],
         'Overcooked becomes undercooked and unmelted becomes underseasoned. This is a grounded detail error, not a stylistic disagreement. Signature slush/ranch information is also largely omitted.'),
        ('COMPETITOR_ATTRIBUTION','4iuNK5svIKV_GrbSwAXM6Q',['F','S'],'limited food or drink options',[(1,'Gas is almost always $0.05/g cheaper at the Gulf across the street, but it has very limited food/services.')],
         'Limited food/services describes Gulf, not Wawa. The output also makes it a reason for access difficulty; selected Wawa reviews discuss traffic/footprint/pumps. F/S Visual is applicable but empty and both recorded scores1 remain primary.'),
        ('MIXED_OPINION_OMISSION','-78Va3llKcPbL6SW4is8lw',['F','U','RO'],None,[(13,'But the pasta sauce is overly sweet and lacks flavor in my opinion.'),(15,'The red sauce is good, and is a bit on the sweet side. I love it, you might not.')],
         'Mixed sweetness reactions are business-specific, supported qualifications. Generic food/service/atmosphere/value accounts omit this contrast. Omission can reduce Coverage/Usefulness despite high claim Faithfulness.'),
        ('REPETITION_COLLAPSE','ymE6ka6y6J-z2y2rX8QH3Q',['F','S'],'Bobbie',[],
         'The Review becomes a long unfinished Bobbie loop; both raters assign Coverage/Coherence/Non-Redundancy/Usefulness1. A supported opening claim can still receive Faithfulness4/5: correctness is not completeness or readability. F Visual catering tray remains grounded; S soup/martini is a wrong recipient scene.')]
    cases=[]
    for topic,bid,conds,span,quotes,interpretation in specs:
        case={'topic':topic,'business_id':bid,'business_name':oo[conds[0],bid]['business_name'],
            'finding_type':'MODEL_ASSISTED_ILLUSTRATIVE_SOURCE_REVIEW','interpretation':interpretation,'conditions':[],
            'verified_source_excerpts':[],'selection':'Purposeful examples of requested error/coverage types; not representative error-rate estimation.'}
        for n,quote in quotes:
            r=sources[bid]['reviews'][n-1];assert quote in r['text']
            case['verified_source_excerpts'].append({'review_id':r['review_id'],'exact_quote':quote})
        for c in conds:
            o=oo[c,bid]
            if span is not None:assert span in o['review_output']
            case['conditions'].append({'condition':c,'exact_output_span':span,'review_output_sha256':o['review_output_sha256'],
                'raters':[{'record_id':rr[c,bid,a]['record_id'],'scores':{crit:rr[c,bid,a][crit] for crit in criteria},
                    'comment':rr[c,bid,a]['evidence_notes']} for a in ['A01','A02']]})
        cases.append(case)
    write(P/'QUALITATIVE/SOURCE_GROUNDED_EXAMPLES.json',cases)
    selection=[]
    for bid in ['oY5EIvb0WZ_4qwioC1eXsw','ymE6ka6y6J-z2y2rX8QH3Q','u5ut1O3T5m22ly6EATA6mQ','4iuNK5svIKV_GrbSwAXM6Q']:
        u=oo['U',bid];f=oo['F',bid]
        selection.append({'business_id':bid,'business_name':f['business_name'],'U_caption_count':u['caption_count'],'F_caption_count':f['caption_count'],
            'U_model_visible_photo_ids':u['model_visible_photo_ids'],'F_model_visible_photo_ids':f['model_visible_photo_ids'],
            'U_only_photo_ids':sorted(set(u['model_visible_photo_ids'])-set(f['model_visible_photo_ids'])),
            'F_only_photo_ids':sorted(set(f['model_visible_photo_ids'])-set(u['model_visible_photo_ids'])),
            'exact_U_visual_output':u['visual_output'],'exact_F_visual_output':f['visual_output'],
            'U_rater_visual_scores':[rr['U',bid,a][criteria[-1]] for a in ['A01','A02']],
            'F_rater_visual_scores':[rr['F',bid,a][criteria[-1]] for a in ['A01','A02']]})
    write(P/'QUALITATIVE/POTENTIAL_SELECTION_COVERAGE_TRADEOFFS.json',selection)
    qual='''# Qualitative findings and audit limitations

These are model-assisted illustrative readings of actual participant comments,
unchanged outputs and selected original recipient sources. They are not new
human labels or an exhaustive independently confirmed error count. Every quoted
review/output span in SOURCE_GROUNDED_EXAMPLES.json is checked against its exact
stored text/ID. All240 participant comments/flags remain verbatim in TABLES/.

## Review claims, omissions and repetition

- PlantPure RO/F/S: greasy/oily reverses no-oil praise or transfers criticism of
  other restaurants to the recipient. Review qPWnBQQymWPaHlt36U2fBQ explicitly says
  “No greasy, oily food!!”. Both raters identify a real problem; differing1/2
  Faithfulness choices remain recorded.
- eegee's F/S: undercooked hot dogs and underseasoned cheese conflict with source
  5DV8jbnSI68JAqdg7Tvq4A's over cooked hot dogs and cheese not melted. Missing
  signature slush/ranch information is a separate coverage issue.
- Wawa F/S: limited food/services belongs to the Gulf competitor in source
  F1tzq5j9Hx5foMJSUpWilQ, not Wawa; access difficulty is not caused by those
  limited options. The empty applicable Visual scores remain1 rather than N/A.
- Mungo's generic Review accounts omit the mixed sweet-red-sauce experience
  despite direct positive/negative source support. High claim Faithfulness can
  coexist with Coverage/Usefulness3; omission is not automatically hallucination.
- Capriotti's F/S Review collapses into a long Bobbie loop. Both raters give
  Coverage/Coherence/Non-Redundancy/Usefulness1. A supported initial statement
  does not rescue completeness/readability. No selective regeneration occurred.

## Visual correctness, contribution and source selection

F's useful examples include the Capriotti sandwich tray, Deli George's three
drinks, Highlander drinks/course/interior and Hong Kong Express noodles/shrimp.
These descriptions are narrower than all recipient photographs but can be
grounded and useful. Visual5 for a narrow correct description is not proof of
exhaustive scene coverage. A single joint Visual score cannot separately identify
factual precision, coverage and contribution.

Shuffled examples introduce donor-like objects/scenes not established in the
recipient set: sushi/chopsticks at Killer B's, steak/blue chairs at Highlander,
and fries/red stools/milkshake at Starbucks. The interpretation is recipient
photo mismatch, not whether a donor photograph could support the sentence.
Earlier image-audit findings are carried forward only after source/photo hash
verification; no new independent human photo adjudication is claimed.

Potential coverage trade-offs are recorded with exact native caption/photo IDs:
Deli U5 versus F1 leaves F describing only drinks; Capriotti U5 versus F1 leaves
only the catering tray; Wawa U5 versus F0 produces empty F/S Visual despite
recipient gas-station photographs. Hong Kong U5 versus F1 changes the selected
photo set, not merely deleting four items from the identical U selection.
U and F may select different top photos and have different checkpoints/budgets;
these cases suggest information-retention questions, not proven causal harm
from a single filtering operation. U also contains errors (Wawa says Walmart;
Deli calls a deli sandwich a hot dog), so more text is not necessarily better.

## Recorded disagreements

TOP_RECORDED_DISAGREEMENTS.json deterministically lists the12 largest absolute
criterion differences, then sorts ties by business/condition/criterion. It is
post-hoc descriptive selection, not an exclusion rule. All690 complete paired
criterion units and confusion matrices are supplied.

Coherence disagreement is systematic, especially S (exact16.67%, A01−A02−0.867),
which may reflect scale interpretation; it is retained rather than reconciled.
F/S Review-only ratings differ in6/180 paired criterion responses, even though
all30 Review outputs/token sequences are identical. Naked Fins A02 differs on
three Review-only criteria, Blue Duck on two and Gallagher's on one. They remain
measurement/presentation/context differences, not improved generated Review.

## Preserved audit concerns

- A02-001, Starbucks/S: recorded Visual4 and comment validate absent recipient
  fries/red-stool/milkshake scenes. This accepted rating remains unchanged.
- A02-003, Gallagher's/F: comment denies speakeasy-like support, but review
  i1Iq_i2JoVSzlAeB04AJpA explicitly supplies it. A comment error does not prove
  the entire Faithfulness4 grade invalid; no substitute score is supplied.
- A02-005, Naked Fins/F: comment documents8/34 photos inspected. Mural/three bao
  are supported by other images, seated man not established; a delayed pickup
  rushed to completion does not establish inattentive service generally.

The researcher accepted all240 as recorded. Acceptance is not evidence that
these concerns were corrected. There is no case removal, imputation, regrading
or sensitivity analysis replacing the30-business primary panel. Optional
calibration notes and taco-count uncertainty remain in the preserved audit
summary; the earlier premature definitive taco-count statement was retracted.

Participant identities, actual source inspection and independence cannot be
independently authenticated from app records. The researcher waived personal
declaration/coordinator gates; supervisor approval remains unverified in the
recorded release. Do not convert acceptance into invented participant consent,
unaided authorship, full blindness or independently verified reliability.
'''
    md(P/'QUALITATIVE_FINDINGS_AND_LIMITATIONS.md',qual)
    methodology='''# Statistical methodology and reporting clarification

The original two-rater amendment and analysis plan are copied byte-for-byte in
PROVENANCE/. ANALYSIS_CONFIG.json was written before the first condition means
were calculated. No inferential method was selected after inspecting favorable
results. The existing analyze_stage_n_two_raters_v1.analyze core and paired
implementation are reused unchanged after independent read-only verification.
The web Study object is never instantiated and no submit/save/export mutation
endpoint is invoked. Package scripts analyze a verified derived export.

## Units and applicability

There are30 fixed businesses,4 conditions and2 recorded raters,240 submissions.
Five criteria apply to RO; all six apply to U/F/S, giving1,380 numeric scores
and60 structural Visual N/A. Review Faithfulness/Coverage/Usefulness judge
Review only. Coherence/Non-Redundancy judge all available components, and cannot
be retrospectively split into uncollected separate Review/Visual scores.
All30 businesses remain, including zero-caption Wawa and applicable empty
Visual components. Original unusual scores/comments/flags are preserved.

## Descriptive and agreement statistics

Report counts, arithmetic mean, sample SD(ddof1), median, Q25/Q75/IQR and1–5
frequencies for each condition/criterion, both combined and per rater. Linear
sample quantiles are a reporting clarification; pooled60-score SD is descriptive,
not60 independent businesses. No heterogeneous overall composite is constructed.

Primary agreement is fixed1–5 quadratic-weighted Cohen kappa:1 minus observed
squared disagreement divided by expected squared disagreement from marginal
frequencies. If expected disagreement is zero, report null with reason; do not
invent perfect agreement. Also supply exact/within-one percentages, signed
A01−A02 difference frequencies and5×5 confusion matrices. Condition-pooled kappa
is supplementary. No ordinal alpha/reliability CI is added, because the frozen
two-rater revision explicitly omits them.

## Paired estimation and exploratory tests

For each business/condition/criterion average the two original scores. Evaluate
17 applicable F−RO/F−U/F−S contrasts on all30 businesses. PCG64 seed20261028,
20,000 common business-resample draws; retain both raters, all conditions and
criteria together.95% intervals are linear2.5/97.5 percentiles. Use the unchanged
50,000 two-sided business sign-flip draws, +1 Monte Carlo correction, and the
same RNG progression as the original code. Effect size is an interpretable
unstandardized difference in1–5 points. No new standardized effect is needed.

All human comparisons are exploratory. No multiplicity adjustment was
prespecified, so report raw p-values without a confirmatory p<.05 winner claim.
Intervals are pointwise, not simultaneous. CI/test discrepancies are shown,
not resolved by selecting a more favorable method. F/S Review identity limits
Review-only difference interpretation. Fixed shuffled donors introduce
dependencies beyond the simple business bootstrap/sign-flip approximation.

## Verification and sources

Numerical verification compares all defined QWK values to scikit-learn,
means/SD to Python statistics and quartiles to independent interpolation. All17
bootstrap intervals/sign-flip p-values are reconstructed using the complete
business×condition×rater tensor with common resample indices. Offline reproduction
from the extracted package is separately checked. No inference/training/API work.

Primary implementation references:
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
- https://numpy.org/doc/stable/reference/generated/numpy.quantile.html

The original protocol is exploratory and provides no population-rater inference.
Sources are the frozen selected reviews and recipient photos, not Teacher targets.
Stage M metrics are copied unchanged for discussion; they are not recomputed or
pooled with Stage N. Reports distinguish participant flags, model-assisted audit
opinions and directly checked text/hash facts. No prespecified inference was
amended; only the requested descriptive/formatting details were clarified.
'''
    md(P/'METHODOLOGY.md',methodology)
    mcompare='''# Stage M versus Stage N: separate evidence

Stage M is TEST100 reference agreement (ROUGE/BERTScore against the approved
Stage L freeze). Stage N is direct original-source assessment by two recorded
raters on a30-business subset. They differ in cohort, construct and score scale.
No scores or sample sizes are pooled. Native Stage M aggregate/paired receipts
are copied unchanged; no metric or reference evaluation was rerun.

| Question | Stage M TEST100 | Stage N TEST30 | Combined reading |
|---|---|---|---|
| F versus S Visual | BERT F1 difference+0.007852,95% CI[0.005865,0.009825], Holm p≈0.000080 | Recorded Visual+2.667 points,95% CI[2.300,3.017], exploratory raw p≈0.000020 | Directionally consistent alignment-sensitive signal; not a pooled causal filtering effect. |
| F versus U Visual | F0.837365,U0.852471; F−U−0.015106, CI[−0.041334,0.002773], Holm p≈0.813384 | F3.90,U4.00; F−U−0.100, CI[−0.383,0.200], raw p0.582008 | Neither establishes F superiority; neither proves equivalence. |
| F versus RO/U Review | F BERT F1 slightly lower; F−RO Holm p≈0.174957,F−U≈0.813384 | Faithfulness F−RO−0.117 and F−U−0.200, intervals cross0; Coverage F−RO/F−U−0.300 | No established F Review factuality gain. Stage N highlights coverage loss not directly measured by reference matching. |
| Review usefulness | Reference agreement is not a direct usefulness construct | F−U−0.233, CI[−0.417,−0.067], raw p0.020240; F−RO−0.233, CI[−0.500,−0.017], raw p0.078058 | Exploratory concern about useful coverage; do not promote an unadjusted finding to confirmation. |
| F/S Review | Identical original text/token IDs; automatic scores identical | Same text, but6/180 Review-only criterion pairs differ | Measurement/context variability, not generated Review improvement. |

The methods therefore agree on a large F/S Visual separation and give no
established F/U Visual advantage. Human Coverage/Usefulness observations add a
different construct, not a contradiction proving automatic metrics invalid.
Coherence/Non-Redundancy against RO compare different component counts/budgets.
All conclusions retain checkpoint ancestry/training-budget/input-selection and
fixed-shuffle limitations. The existing validation-selected F checkpoint remains
fixed; this analysis does not reselect it using TEST or start another experiment.
'''
    md(P/'STAGE_M_VS_STAGE_N.md',mcompare)
    intro='''# Stage N — Final Human Evaluation Statistical Review

All30 frozen TEST businesses, four conditions and two recorded raters are
included:240 submissions,1,380 numeric scores and60 structural N/A values.
Original scores/comments/provenance are unchanged and researcher acceptance is
preserved. No overall composite score or new model selection is produced.

## Main measured results

'''+mean_table+'''

- F Visual exceeds Shuffled by2.667 points,95% business-bootstrap CI[2.300,3.017].
  This is the clearest alignment-sensitive difference in this recorded panel.
- F versus U Visual is−0.100[−0.383,0.200]: no established F advantage.
- F Review Coverage is0.300 points below both RO/U. F Usefulness is0.233 below U.
  These are exploratory findings; Faithfulness contrasts with RO/U cross zero.
- Coherence/Non-Redundancy against RO favor RO but compare a Review-only output
  with two components. F/S Review text is identical; small Review-rating
  differences are measurement/context differences, not generation improvement.
- Primary condition-specific QWK ranges0.460–0.952. Coherence has substantial
  scale-use disagreement (S exact16.67%); pooled Visual agreement must not be
  substituted for condition-specific agreement.

The frozen protocol labels all17 human contrasts EXPLORATORY. Raw p-values and
pointwise95% CIs are reported without a new multiplicity-adjusted winner claim.
Ordinal alpha is not calculated because the two-rater amendment explicitly
omitted it. All30 businesses remain; no unusual/negative/audited cases are removed.

## Reading order

1. DESCRIPTIVE_STATISTICS.md and HUMAN_EVALUATION_RESULTS.xlsx: full tables,
   including SD, median/IQR, distributions, per-rater and combined summaries.
2. INTER_RATER_AGREEMENT.md, PAIRED_COMPARISONS.md and FIGURES/PAIRED_DIFFERENCES.png.
3. QUALITATIVE_FINDINGS_AND_LIMITATIONS.md and STAGE_M_VS_STAGE_N.md.
4. DATA/ and TABLES/: original payloads, exact120 outputs, sources, complete
   machine-readable scores, paired differences, comments and confusion matrices.
5. ANALYSIS_CONFIG.json, METHODOLOGY.md and PROVENANCE/: frozen settings, original
   plan/amendment, researcher acceptance and actual integrity receipts.
6. SCRIPTS/RUN_STATISTICAL_ANALYSIS.py: reproduce with Python+NumPy into a fresh
   output directory; sklearn is optional for cross-library checking.

Run offline structural verification:
`python -B VERIFY_PACKAGE.py`

Recompute supplied statistics without live study access:
`python -B SCRIPTS/RUN_STATISTICAL_ANALYSIS.py --package . --output REPRODUCED_TABLES`

## Audit and provenance

A02-001/003/005 remain included unchanged with their existing audit concerns;
details distinguish participant opinions, model-assisted audit findings and exact
source evidence. Acceptance is not evidence that all concerns were corrected.
Identity/source inspection/independence are not independently authenticated;
the recorded direct-access/declaration and supervisor-approval limitations remain.

This is coordinator-level decoded statistical material. Do not distribute it
or condition-labelled rows to participants during a blinded panel. The original
private alias mapping, password hashes, credentials, sessions and SQLite database
are excluded; necessary mapping provenance is the original key's hash and the
verified decoded records. No donor images are used as recipient grounding.

No Stage L reference, Stage M result, native prediction or checkpoint was changed.
Stage M remains a separate TEST100 automatic track. No training, inference,
additional rating collection, API/GPU spend or next research stage was initiated.
STOP for independent researcher/ChatGPT review.
'''
    md(P/'README.md',intro)
    for name in ['analyze_stage_n_final_statistics_v1.py']:
        dest=P/'SCRIPTS'/('RUN_STATISTICAL_ANALYSIS.py' if name.startswith('analyze_') else name)
        dest.parent.mkdir(exist_ok=True);dest.write_bytes((Path('scripts')/name).read_bytes())
    (P/'PROVENANCE/ANALYSIS_AUTHORIZATION.txt').write_bytes(Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/56ccec04-c2c7-45e4-97fb-1e0fc6a19637/Pasted text.txt').read_bytes())
    md(P/'SCRIPTS/REQUIREMENTS.txt','numpy=='+read(T/'RUNTIME.json')['versions']['numpy']+'\n# Optional numerical cross-check: scikit-learn=='+read(T/'RUNTIME.json')['versions']['scikit-learn'])
    # Plot only already-computed pointwise CIs, with no significance stars.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6.7),layout='constrained')
    colors={'F-RO':'#415A77','F-U':'#B66A25','F-S':'#287365'}
    for ax,crit in zip(axes.flat,criteria):
        rs=[x for x in paired if x['criterion']==crit]
        for y,x in enumerate(rs):
            ax.errorbar(x['mean_paired_difference'],y,xerr=[[x['mean_paired_difference']-x['CI95_low']],[x['CI95_high']-x['mean_paired_difference']]],fmt='o',color=colors[x['contrast']],capsize=4)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=.8)
        ax.set_yticks(range(len(rs)),[r['contrast'] for r in rs]);ax.invert_yaxis();ax.set_title(pretty[crit],fontsize=10)
        ax.set_xlabel('F minus comparator (1–5 points)',fontsize=9);ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('TEST30: paired differences and 95% business-bootstrap intervals',fontsize=13)
    fig.supxlabel('Exploratory, pointwise intervals; axis ranges differ by criterion. Both raters stay within each sampled business.',fontsize=9)
    (P/'FIGURES').mkdir(exist_ok=True)
    fig.savefig(P/'FIGURES/PAIRED_DIFFERENCES.png',dpi=180);fig.savefig(P/'FIGURES/PAIRED_DIFFERENCES.svg');plt.close(fig)
    print(json.dumps({'report_written':True,'descriptive_rows':len(desc),'paired_rows':len(paired),'illustrative_review_cases':len(cases),'no_new_grades':True},indent=2))


def read_csv(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


if __name__=='__main__':main()
