"""Build researcher-review prose and tables from verified, frozen evidence only."""
from pathlib import Path
import csv,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/final_research_analysis_20261106_v1/PACKAGE'
T=P/'VERIFIED_TABLES'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(name,text):(P/name).write_text(text.strip()+'\n',encoding='utf8')
def md(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def f(x,d=3):return 'N/A' if x is None else f'{x:.{d}f}'
def pv(x):return f'{x:.6f}'
def ci(row,a='pointwise_CI95_low',b='pointwise_CI95_high'):return '['+f(row[a])+', '+f(row[b])+']'
CR=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
LABEL=['Review Faithfulness','Aspect Coverage','Coherence','Non-Redundancy','Usefulness','Visual Grounding and Contribution']
def means(stage,models):
 rows=read(T/(stage+'_DESCRIPTIVES'+('' if stage=='P' else '_VERIFIED')+'.json'))
 lookup={(r['model'],r['criterion']):r for r in rows if r['panel']=='COMBINED'}
 return md(['Criterion']+models,[[label]+[f(lookup[m,c]['mean']) for m in models] for c,label in zip(CR,LABEL)])
def main():
 assert read(T/'VERIFICATION_RESULTS.json')['status']=='PASS'
 # Additional, small pre-existing provenance evidence; no source edits.
 inventory=read(P/'SOURCE_INVENTORY.json')
 for source in [ROOT/'outputs/data_integrity/cross_split_images_s1000_v1/EVIDENCE/EXPOSURE_SUMMARY.json',ROOT/'outputs/data_integrity/cross_split_images_s1000_v1/DUPLICATION_CAUSE_AND_IMPACT_SUMMARY.md']:
  target=P/'SOURCES/IMAGE_OVERLAP'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  entry=dict(original_file=source.relative_to(ROOT).as_posix(),package_file=target.relative_to(P).as_posix(),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),bytes=source.stat().st_size)
  if not any(r['package_file']==entry['package_file'] for r in inventory['member_copies']):inventory['member_copies'].append(entry)
 (P/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 p=read(T/'P_PAIRED_COMPARISONS.json')
 ptable=md(['Contrast (left minus right)','Criterion','Difference','Pointwise 95% CI','Exact p','Holm p','Family'],[[r['left']+' − '+r['right'],dict(zip(CR,LABEL))[r['criterion']],f(r['mean_difference']),ci(r),pv(r['raw_exact_signflip_p']),pv(r['holm_p_within_family']),r['family']] for r in p])
 m=read(T/'M_AGGREGATES_VERIFIED.json')
 metrics=['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']
 mtable=md(['Component','Condition','n','ROUGE-1','ROUGE-2','ROUGE-L','BERT P','BERT R','BERT F1'],[[r['task'],r['condition'],r['n']]+[f(r[k],4) for k in metrics] for r in m])
 pp=read(T/'P_DESCRIPTIVES.json')
 pdtable=md(['Criterion','System','Mean ± sample SD','Median','Scores 1/2/3/4/5'],[[dict(zip(CR,LABEL))[r['criterion']],r['model'],f(r['mean'])+' ± '+f(r['sample_SD']),f(r['median'],1),'/'.join(str(r['score_'+str(k)]) for k in range(1,6))] for r in pp if r['panel']=='COMBINED'])
 patable=md(['Criterion','System','QWK','Exact agreement','Mean absolute difference','A01 − A02'],[[dict(zip(CR,LABEL))[r['criterion']],r['model'],('UNDEFINED' if r['quadratic_weighted_kappa'] is None else f(r['quadratic_weighted_kappa'])),f(r['exact_agreement_fraction']*100,1)+'%',f(r['mean_absolute_difference']),f(r['mean_first_minus_second'])] for r in read(T/'P_AGREEMENT.json')])
 raw_p=read(P/'SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json')['records']
 assignment={r['alias']:r['system'] for r in read(P/'SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ASSIGNMENT_KEY_90.json')}
 flags=[]
 for model in ['LED F original','Qwen-Caption','Qwen-Image']:
  for rater in ['A01','A02']:
   selected=[r for r in raw_p if assignment[r['alias']]==model and r['annotator']==rater]
   flags.append(dict(system=model,rater=rater,n_records=len(selected),nonempty_comments=sum(bool(r['evidence_notes'].strip()) for r in selected),serious_flags=sum(r['serious_grounding_issue'] for r in selected),rater_labeled_confirmed=sum(r['serious_grounding_issue'] and r['issue_certainty']=='confirmed' for r in selected),rater_labeled_uncertain=sum(r['serious_grounding_issue'] and r['issue_certainty']=='uncertain' for r in selected)))
 with (T/'P_PARTICIPANT_FLAG_COUNTS.csv').open('w',encoding='utf8',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(flags[0]));w.writeheader();w.writerows(flags)
 flag_table=md(['System','Rater','Serious flags / 15','Rater-labeled confirmed','Rater-labeled uncertain'],[[r['system'],r['rater'],str(r['serious_flags'])+' / 15',r['rater_labeled_confirmed'],r['rater_labeled_uncertain']] for r in flags])
 write('FINAL_RESULTS_ANALYSIS.md',f'''
# Final Research Results Analysis

Researcher-review draft, 6 November 2026. This analysis uses the original, recovered Stage P V1 collection, never the later revised collections. It preserves the frozen models and all recorded ratings. Evidence codes below resolve in [EVIDENCE_TRACEABILITY.md](EVIDENCE_TRACEABILITY.md).

## 1. Evidence scope and verification

| Stage | Cohort | Conditions | Main observations | Interpretation boundary |
| --- | --- | --- | --- | --- |
| M | Yelp TEST100 | RO/U/F/S | 700 component-score rows from 400 outputs | Agreement with common frozen human-approved references |
| N | Yelp TEST30, nested in TEST100 | RO/U/F/S | 240 ratings; 1,380 numeric scores; 60 structural Visual N/A cells | Two-evaluator, source-grounded judgments |
| O | TripAdvisor15 | LED RO/U/F | 90 ratings; 510 numeric scores; 30 structural Visual N/A cells | External-domain, reference-free judgments |
| P | Same TripAdvisor15 | LED F-S1000/Qwen-Caption/Qwen-Image | 90 original V1 ratings; 540 scores | New blinded comparison panel; returning evaluators |

Stages are analyzed separately. Ratings, criterion scores and restaurants are different units: the paired unit is the business/restaurant, with both raters retained together. Ninety Stage P records do not constitute 90 independent restaurants or 540 independent experiments. O and P share restaurants and returning evaluator accounts; N is a subset of M. No cross-stage pooled effect or composite winner is calculated. [E01–E05]

The official P archive SHA-256 is `e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5`; its combined rating export is `e47b88f1b4d6ccf0bf1649b5b06ead3f538f97af30324993f5e45cf79a58472f`. The externally created ChatGPT statistical ZIP was not provided locally. Under the researcher's clarification, verification therefore consists of a new independent arithmetic reconstruction from original V1, cross-checked against execution of the unchanged frozen analysis script. It is not a claim to have inspected that external ZIP. [E01]

All M/N/O/P original archive member manifests and CRC checks passed. Independent calculations matched 42 M metric means, 24 M paired CIs and four primary adjusted tests; N descriptives/agreement and 17 paired tests; O descriptives/agreement and 11 paired tests; and all P descriptives, agreement and 18 contrasts. The included verification receipt identifies exact seeds, source hashes and software. This verifies numerical consistency, not participant identity, every semantic judgment, or fresh execution of ROUGE/BERTScore models. [E02]

## 2. Stage M: final TEST100 reference agreement

All metrics below are macro means on the original **0–1 scale**, including ROUGE. RO has no Visual component and is not scored as zero. These are final TEST100 human-reference results, not VALIDATION200 Teacher-target agreement. [E03]

{mtable}

The four prespecified BERTScore-F1 comparisons use 20,000 paired business bootstrap samples and 50,000 Monte Carlo sign flips, seed 20261028, with Holm correction across four tests. F minus RO Review is −0.002308 (95% CI [−0.005814, −0.000064], raw p=0.058319, Holm p=0.174957). F minus U Review is −0.001546 ([−0.004965, 0.000745], Holm p=0.813384). F minus U Visual is −0.015106 ([−0.041334, 0.002773], Holm p=0.813384). F minus S Visual is +0.007852 ([0.005865, 0.009825], Holm p=0.000080). [E03]

Only the F–S Visual comparison is supported at the adjusted 0.05 threshold. Pointwise bootstrap CIs and multiplicity-adjusted tests answer different inferential questions; the F–RO interval is not a reason to override its nonsignificant adjusted test. U's higher Visual similarity is descriptive evidence against a universal F advantage, not proof of better visual factuality. F and S Review texts are exactly reused, so equal Review metrics are a construction check rather than replicated Review evidence.

ROUGE uses full component text with Porter stemming; BERTScore uses pinned RoBERTa-large, layer 17, IDF off and no baseline rescaling, with a fixed 512-token prefix. One F Review and its S reuse were prefix-truncated; references were not. F/S each have two empty Visual outputs, retained with zero similarity scores. The same recipient-business reference is used for every applicable condition, including S. [E03]

Length/repetition diagnostics show shorter F than U Visual outputs (51.68 versus 67.74 mean words) and lower mean repeated-trigram fraction (0.0605 versus 0.1230). Duplicate-sentence fractions are 0.0032 versus 0.0187. These are descriptive output properties; shorter text can suppress repetition and omit useful content simultaneously. Review repetition does not show the same F advantage. [E04]

## 3. Stage N: Yelp TEST30 human evaluation

Means on the 1–5 rubric; all original 240 records retained. Full SDs, medians, distributions and per-rater results are in the supplied tables. [E05]

{means('N',['RO','U','F','S'])}

F exceeds S on Visual Grounding and Contribution by 2.667 points (95% CI [2.300, 3.017], raw exploratory p=0.000020); 28 businesses favor F, two tie and none favor S. This is substantially larger than the automatic similarity gap and supports the importance of business-aligned visual evidence. It does not isolate which filtering operation causes the effect. [E06]

F does not establish Visual superiority over U: difference −0.100 ([−0.383, 0.200], raw p=0.582008). F–U Aspect Coverage is −0.300 ([−0.500, −0.133], raw p=0.003300) and Usefulness −0.233 ([−0.417, −0.067], raw p=0.020240). F–U Non-Redundancy is +0.250 ([−0.117, 0.633], raw p=0.233235). These exploratory, unadjusted comparisons indicate coverage/usefulness tradeoffs, rather than consistent F superiority or an established Yelp human repetition benefit. No multiplicity correction was prespecified for these 17 N contrasts; no confirmatory interpretation is added retrospectively. [E06]

Condition-specific quadratic-weighted kappa ranges from 0.460 to 0.952. High raw agreement can coexist with modest kappa when scores concentrate near the floor: S Visual exact agreement is 93.3% but kappa is about 0.565. Six of 180 F/S Review-only score cells differ despite identical Review components, illustrating contextual or measurement variation. Existing audit concerns remain disclosed; the researcher accepted the recorded data without changes. Participant flags remain reports rather than an independently adjudicated factual-error census. [E07]

## 4. Stage O: external TripAdvisor generalization

The Yelp-trained LED checkpoints were applied without additional fine-tuning. These 15 restaurants provide external-domain evidence, but are a geographically narrow English-review convenience cohort. They do not support a population-wide generalization estimate. [E08]

{means('O',['RO','U','F'])}

F–U Non-Redundancy is +0.833 ([0.400, 1.333], raw exploratory p=0.004740); ten restaurants favor F, four tie and one favors U. This is the clearest external filtering-associated advantage. F–U Visual is −0.167 ([−0.633, 0.300], raw p=0.601748); Review-only F–U and F–RO contrasts do not establish superiority. These are 11 exploratory comparisons, including two supplementary component-budget comparisons, without a prespecified multiplicity correction. [E09]

The models retain some readable and source-related summarization capability, but coverage is around 3.2 and usefulness around 3.0. Recorded comments describe missing mixed opinions, omitted venue details, unsupported service complaints and repeated phrases. Agreement is heterogeneous (condition-specific QWK 0.000–0.904); U/F Visual kappas are approximately 0.536/0.490. These are qualified external results, not evidence of unchanged in-domain performance or a formal cross-domain non-inferiority result. No TripAdvisor Teacher/reference-similarity evaluation is substituted for source-grounded judgments. [E08–E10]

## 5. Stage P: official V1 comparison with Qwen

### 5.1 Descriptives and distributions

Each combined mean uses 30 scores, two per restaurant. SDs below describe those scores, not 30 independent restaurants. P uses the original LED F outputs and the frozen Qwen outputs. [E11]

{pdtable}

All 15 Qwen Review components are identical between Caption and Image variants. Their three Review-only criteria therefore do not test an image-derived Review benefit. Coherence and Non-Redundancy assess both components. In this collection the two Qwen variants also receive identical scores on those criteria, although their Visual text differs.

### 5.2 Paired inference under the frozen plan

Within each restaurant, the two within-rater differences are averaged. Twenty thousand whole-restaurant bootstrap samples use seed 20261106; the exact sign-flip test enumerates all 32,768 sign assignments. Holm correction is applied separately to 12 primary tests and six additional exploratory tests. CIs are pointwise percentile intervals, not simultaneous family-wise intervals. The tests remain exploratory because an exact enumeration does not make this observational system comparison randomized. [E12]

{ptable}

Qwen-Caption exceeds LED F on all five non-Visual criteria after correction within the primary family. The largest differences concern coverage, non-redundancy and usefulness. Both raters favor Qwen on the mean difference for all five criteria; their repetition-effect magnitudes differ markedly. LED F and Qwen-Caption receive identical Visual scores for every restaurant/rater cell in V1. This exact recorded equality is not an equivalence trial, proof of equal factual accuracy, or equality of overall quality.

Qwen-Image exceeds Qwen-Caption descriptively on Visual by 0.633 points. Expressed as Caption minus Image, the pointwise CI is [−1.033, −0.200], raw exact p=0.021118 and primary-family Holm p=0.147827: the planned primary contrast is inconclusive after correction. The same numerical Image advantage over LED F has additional-family Holm p=0.021118, because that is a different, smaller family of six tests. This ancillary result must not replace the primary Caption–Image conclusion. There is no prespecified composite winner.

### 5.3 Agreement and measurement limitations

{patable}

Both Qwen Coherence cells have perfect exact agreement but undefined kappa: every score is 5 and expected disagreement is zero. They are not assigned kappa=1. LED Non-Redundancy has only 6.7% exact agreement and A01−A02 mean difference +1.400, despite both raters favoring Qwen in the paired contrast. QWK and exact agreement are reported together because score concentration can produce low kappa even when disagreements are small. Two raters and 15 restaurants cannot establish reliability across a wider evaluator population. [E13]

Stage P Review lengths average 81.07 words for LED and 132.87 for either Qwen variant; Visual means are 69.00, 52.13 and 65.53 respectively. Longer Qwen Review scope may help coverage but also changes the task budget. Word counts use one shared whitespace rule; model token counts use different native tokenizers and are not interchangeable. [E14]

### 5.4 Participant comments and flags

All 90 V1 records contain comments; 17 records carry serious-grounding flags (A01: 10; A02: 7). These count flagged rating records, not unique errors, independent adjudications or photo-level accuracy. Shared Qwen Review errors can be flagged in both variants. The original certainty labels remain attributed to the rater. [E22]

{flag_table}

The comments describe omitted mixed opinions and venue details, unsupported service complaints, dish-level polarity misattribution, reversed visual actions, sign-reading errors and cross-scene relations. For example, A02 flags Qwen-Caption's “One man pours wine into a green bottle” for `TA14622324_2312071`, citing `TA_PHOTO_0713755e`, `TA_PHOTO_0f821627` and `TA_PHOTO_1ba1c1f3`; A01 flags Qwen-Image's “typically costing around 15 to 30 euros for two people” for `TA14622324_2298952`, citing `TA_REVIEW_523497914`, `TA_REVIEW_789979868` and `TA_REVIEW_673263312`. These are traceable participant-reported examples, not new independent confirmations. Their presence illustrates that higher mean scores do not imply error-free Qwen outputs.

## 6. Integrated interpretation

The strongest consistent result is the harm caused by recipient-misaligned visual evidence: F exceeds S in TEST100 Visual reference agreement and TEST30 Visual human scores. Filtering is associated with shorter, less repetitive Visual outputs and a TripAdvisor Non-Redundancy improvement, but does not consistently improve Visual human scores over U; some Yelp Review coverage/usefulness results favor U. Stage P places this limitation in context: a large pretrained instruction-following baseline is rated better than LED on five criteria, even with the same selected Review/caption evidence. A smaller measured inference footprint is therefore a resource tradeoff accompanied by observed quality deficits. [E03–E15]

This analysis leaves F-S1000 frozen as the previously selected study system; it does not reselect checkpoints from TEST results. Answers to the original research questions and qualified deployment implications are in the companion documents. No new human scores, model calls, references or experiments were produced.
''')

 write('RESEARCH_QUESTIONS_FINDINGS.md','''
# Research Questions and Findings

The questions below preserve the wording in the approved proposal, `thesis_proposal_v16.docx`. The source hash and paragraph extracts are included in `SOURCES/APPROVED_PROPOSAL/` and `SOURCES/RESEARCH_QUESTIONS.json`. Findings concern the final accepted source-separated LED conditions. No TEST-based checkpoint reselection is performed. [E16]

## RQ1

> To what extent do business-specific visual captions add accurate and useful visual information to long-context business-level review summaries without reducing the faithfulness and quality of the review consensus?

**Answer: partially supported, with an unresolved no-harm requirement.** Business-aligned captions support a separately generated Visual Evidence component: F receives a mean Visual Grounding and Contribution score of 3.900 on Yelp TEST30 and 3.533 on TripAdvisor15. The shuffled negative control is much worse on Yelp (1.233). This supports the relevance of recipient-specific visual evidence. It does not establish that every Visual claim is accurate, or separately quantify its added usefulness: the rubric combines grounding and contribution in one score. [E05–E09]

The condition “without reducing” Review quality is not demonstrated. Yelp F Review coverage and usefulness are lower than U in exploratory paired comparisons (−0.300 and −0.233 points); Review faithfulness is not consistently superior. No non-inferiority margin or adequately powered no-harm test was frozen. Failure to reject a difference cannot establish preservation or equivalence. [E06]

In the final architecture, Review inference receives reviews only and Visual inference receives captions only. Combining components verbatim provides complementary information; it does not show that image captions improve the Review generator through joint input. Review differences across RO/U/F may reflect training supervision and shared-adapter task exposure rather than inference-time visual reasoning. External outputs retain useful details but also exhibit unsupported attributions, omissions and genericness. RQ1 is therefore supported for the feasibility of a visual addendum, not for uniformly accurate visual content or a guaranteed no-loss Review consensus. [E08–E10, E17]

## RQ2

> Does filtering business-specific visual captions improve visual grounding and summary quality compared with unfiltered captions, and how does performance change when the captions are replaced with captions shuffled from other businesses?

**Answer: supported for some repetition outcomes and alignment sensitivity; consistent superiority over U is not established.** TEST100 F Visual outputs are shorter than U (51.68 versus 67.74 words) and have lower mean repeated-trigram fraction (0.0605 versus 0.1230). TripAdvisor Non-Redundancy favors F over U by 0.833 points, pointwise 95% CI [0.400, 1.333], exploratory unadjusted p=0.004740. The analogous Yelp human difference is +0.250, CI [−0.117, 0.633], and is inconclusive. [E04, E06, E09]

Visual human scores do not establish F superiority over U in either domain: Yelp difference −0.100 [−0.383, 0.200], TripAdvisor −0.167 [−0.633, 0.300]. U's TEST100 Visual BERTScore F1 is also descriptively higher, although the primary adjusted F–U test is inconclusive. These results reject a broad claim that the filtering pipeline consistently improves all aspects of summary quality. They do not establish equivalence or that filtering never helps. [E03, E06, E09]

Replacing aligned Visual evidence with frozen donor-business evidence reduces TEST100 Visual BERTScore F1 by 0.007852 relative to F (Holm p=0.000080) and TEST30 Visual human scores by 2.667 points (exploratory unadjusted p=0.000020). F/S Review generation is reused, so the control probes recipient–Visual alignment, not a new Review generation effect. The fixed donor permutation and different U/F supervision prevent attribution of a causal effect to CLIP ranking, redundancy removal, diversity or K selection individually. [E03, E06, E17]

## RQ3

> How computationally efficient is LoRA-based adaptation of LED for long-context business-level review summarization under limited computational resources, in terms of trainable parameters, GPU memory usage, training time, inference time, and summarization performance?

**Answer: feasible parameter-efficient adaptation and a smaller observed prepared-input inference footprint; an end-to-end efficiency advantage is unproven.** Expanded LoRA trains 4,915,200 parameters while 459,801,600 backbone parameters remain frozen: 1.069% of the backbone count, or 1.058% of the combined parameter count. One accepted F continuation segment completed 1,000 successful updates in 7,465.886 seconds on an RTX 3090, with peak PyTorch allocated memory 3.414 GiB and reserved memory 5.189 GiB. It excludes the CE3-750 parent training and preprocessing; there is no matched full-parameter training baseline from which to estimate a causal LoRA speedup. [E17–E18]

On prepared inputs, the later RTX 3090 benchmark reports 97.931 seconds and 2.697 GiB allocated peak for LED, versus 186.844 seconds and 18.536 GiB for Qwen-Caption. Qwen-Image's new Visual generation takes 242.703 seconds; its 379.697-second Review-plus-Visual figure is constructed using the previously generated Qwen Review. Precision, decoding, lengths, warmup and frameworks differ. These are implementation-specific generation measurements, not matched end-to-end performance or quality-equivalent efficiency. [E15]

The actual preparation included 48,642.959 seconds of BLIP-2 captioning for 5,143 photographs and 178.529 seconds of caption scoring; final selection time is not measured. Qwen-Image receives 73 caption-selected photographs and inherits that selection provenance despite receiving no caption text at inference. Full pipeline latency, capacity requirements, energy and monetary efficiency remain unmeasured. Stage P Qwen-Caption is rated higher than LED on five criteria after the frozen correction, while its Visual scores are exactly tied in the recorded sample. The defensible conclusion is a quality–resource tradeoff under prepared or cached sources, not overall LED superiority. [E11–E15]
''')

 write('QUALITY_EFFICIENCY_DISCUSSION.md','''
# Quality and Computational Efficiency

## 1. What was actually measured

The existing read-only efficiency audit is the resource evidence for this discussion. All values refer to the full 15-restaurant cohort unless explicitly marked otherwise. GiB means 2^30 bytes. Allocated and reserved memory are PyTorch allocator measurements; neither is total device usage nor minimum required VRAM. [E15]

| Generation boundary | Time (s) | Calls | Peak allocated / reserved (GiB) | Main limitation |
| --- | ---: | --- | ---: | --- |
| LED F, prepared inputs | 97.931 | 30 per repeat; two measured repeats | 2.697 / 3.045 | Mean after two warmups; 3,140 native output tokens |
| Qwen-Caption | 186.844 | 30; one run | 18.536 / 18.717 | Cold pilot included; 3,651 native tokens |
| Qwen-Image, new Visual only | 242.703 | 15 | 17.757 / 17.820 | 73 selected photos; 1,342 new native tokens |
| Qwen-Image, constructed Review + Visual | 379.697 | 15 earlier Review + 15 new Visual | 18.536 / 18.717 across component stages | Arithmetic across sessions; 3,980 native tokens; not a new combined timed job |

The ratio of observed generation time to LED is 1.908 for Qwen-Caption and 3.877 for the constructed Qwen-Image figure. LED repeat times are 97.435 and 98.428 seconds (sample SD 0.702); the Qwen measurements have no equivalent replication estimate. No inference-time significance test is justified. The actual new Image extension required 242.703 seconds of generation when its Review component already existed.

All three generation measurements use the same RTX 3090, but different work: LED uses FP32 parameters with FP16 autocast and beam search of size four; Qwen uses BF16 and greedy decoding with thinking disabled. Native token counts are tokenizer-specific. Framework versions, input modalities, CPU threading, output lengths and warmup boundaries differ. The LED resource benchmark reproduces 22/30 accepted original output components exactly, while its two repeats agree on 30/30. Frozen original outputs, not benchmark replacements, were evaluated by people. Thus the measured resource/quality association is approximate at the implementation level, not a per-output deterministic replay claim. Model loading was separately recorded (LED 9.279 s, Qwen-Caption 11.533 s, Image Visual 5.848 s) and is excluded from the generation table. [E15]

## 2. Pipeline accounting and missing measurements

| Existing preparation stage | Observed work | Recorded time | Boundary |
| --- | --- | ---: | --- |
| BLIP-2 | 5,143 photos; 5,150 attempts including seven retries | 48,642.959 s (13.51 h) | RTX 3080 with CPU/GPU offload; includes 26.295 s loading |
| CLIP/SBERT caption scoring | Scores/embedding export | 178.529 s | Includes loading, hashing and image I/O; not final selection |
| Final relevance/redundancy/diversity/K selection | Frozen selection exists | NOT MEASURED | Cannot substitute scorer time |
| Review selection, raw source acquisition, complete request overhead | Source records exist | NOT MEASURED as a complete comparable interval | Cannot set to zero |
| Complete source-to-summary wall time | Each whole system | NOT MEASURED | No matched end-to-end conclusion |

The two recorded preparation components sum to 48,821.488 seconds, but adding that number to generation yields historical component accounting, not an observed end-to-end run. Preparation occurred on a different device and software path; missing terms, loading boundaries and possible overlap remain. Dividing BLIP time by 5,143 and multiplying by 73 would fabricate an unobserved comparison and is not performed.

LED F and Qwen-Caption consume frozen Filtered caption text. Qwen-Image consumes the corresponding 73 selected photographs directly, but the actual photograph selection was derived from the BLIP/filter pipeline. It is caption-free at the generation input, not independent of caption-based selection end to end. Removing BLIP/filter costs while retaining those 73 selected photographs assumes precomputed selection. An alternative direct-image selector was not benchmarked. Processing 5,143 source images and 73 selected images is not an equal-photo-workload comparison. [E15]

Sequential recorded stage maxima also differ from inference-only peaks. BLIP's allocated peak of 4.168 GiB exceeds LED generation's 2.697 GiB; the recorded maximum across LED's measured sequential stages is therefore at least 4.168 GiB. This does not certify a 4.168-GiB deployment: device overhead, unmeasured stages and offloaded CPU RAM also matter. Qwen's combined component peak remains 18.536 GiB. Sampled whole-device memory is separately labeled in the original audit and is not mixed with allocator peaks.

## 3. Cached captions and selection

Stage P actually reused Stage O captions and selection. Prepared-input generation figures are useful for that operational boundary, provided prior preparation is disclosed. For hypothetical N repeated full-cohort summarization requests with unchanged sources, an accounting model is `(48,821.488 + unknown_once)/N + G_system + unknown_request`. This is amortization, not measured throughput or total latency. It assumes reusable captions/selection, no source refresh, and no cache of final summaries. No service-load test, energy measurement, cost comparison or break-even point is available. [E15]

## 4. Quality accompanying the resource difference

Qwen-Caption exceeds LED on faithfulness (+1.033), coverage (+1.533), coherence (+0.933), non-redundancy (+1.633) and usefulness (+1.467) in the official P collection. All five differences survive the primary Holm family. Visual scores are identical cell by cell, not evidence of overall quality equivalence. Review outputs are longer for Qwen (132.87 versus 81.07 words), a material scope confounder. The comparison is between a large pretrained instruction-following model and a smaller task-tuned LED, so architecture, parameter count, pretraining, prompts and generation budgets are not isolated. [E11–E14]

The Image variant improves mean Visual scores by 0.633 over Caption, but the primary Holm-adjusted p=0.147827 is inconclusive. The additional Image–LED comparison has adjusted p=0.021118 in its separate six-test family. Both findings must be retained. Direct pixels may recover caption omissions and avoid caption errors, but this sample does not establish a general modality effect or a statistically robust primary Image-over-Caption improvement.

## 5. Defensible thesis and deployment claims

Supported wording is: “For these prepared 15-restaurant inputs, the tested LED implementation had lower measured generation time and allocated-memory peaks than Qwen, while Qwen-Caption received higher human scores on five criteria. Complete end-to-end efficiency was not measured.”

The experiments demonstrate practical LoRA adaptation on a consumer GPU and a potentially useful prepared-input option when memory or latency is constrained. They do not justify claims that LED is faster end to end, has quality comparable to Qwen overall, requires only its inference allocator peak to deploy, or is more energy/cost efficient. A future deployment decision should explicitly choose an acceptable factuality/coverage standard and measure the complete intended service boundary. That prospective work has not been executed here.
''')

 write('LIMITATIONS_AND_THREATS_TO_VALIDITY.md','''
# Limitations and Threats to Validity

## 1. Experimental identification and training history

RO/U/F are complete trained systems, not a controlled ablation differing only in one input at inference. Final RO and U continue their own RO750 and U750 parents; F continues CE3-750. Each receives 1,000 new successful updates, but RO has 4,000 Review and no Visual presentations, whereas U/F each have 4,000 Review and 4,000 Visual presentations. RO's four Review microbatches are normalized by the eight-slot accumulation denominator, unlike U/F's four-plus-four composition. Equal update counts do not imply equal task exposure, gradient scaling, compute or prior supervision. RO's own parent provenance must not be replaced with the inaccurate claim that it starts from caption-trained CE3. U/F target construction and previous training histories are condition-dependent. [E17]

Source-separated inference isolates Review and Visual evidence. This helps attribution but limits claims about multimodal fusion into the Review consensus. The fixed Shuffled donor permutation tests recipient alignment; it does not isolate each filtering component or create independent Review generations. RO coherence/non-redundancy is judged on one component and other conditions on two, introducing an output-budget difference. [E03, E05, E17]

## 2. Source independence and benchmark scope

Business IDs are disjoint across TRAIN2000/VALIDATION200/TEST100, but Yelp image content is partly reused across splits. The prior audit records 17 affected TEST businesses, 123 TEST photo IDs and 490 duplicate cross-split pairs. Actual selected duplicate entries number 66 for U and 44 for F; 59 U and 42 F entries overlap their own-condition TRAIN caption sources and retained Visual-label support. This is a relevant visual exposure limitation, not proof of memorization or a measured inflation estimate. All original 100 businesses are retained; no favorable post-hoc TEST83 subset or new subgroup test is introduced. [E19]

TripAdvisor comprises 15 restaurants from a narrow geographic area and available English reviews, with 247 selected reviews and 5,143 collected photos. Public image renditions are not necessarily original-resolution camera files. This is external-domain evidence, not representative sampling of TripAdvisor or all restaurant types. M/N are nested cohorts; O/P reuse the same restaurants and returning raters. Cross-stage numerical differences cannot be treated as independent replication, causal domain effects or pooled evidence. [E08, E20]

## 3. Reference construction and metric validity

TEST100 references are recorded as author/checker approved and frozen, with documented AI feedback, evidence suggestions and minor editorial assistance. Acceptance and hashes do not independently authenticate unaided human authorship, actual source inspection or participant identity. Human-approved references remain fallible. Teacher-target VALIDATION200 agreement is development evidence and must not be substituted for final TEST100 human-reference scores. [E03, E21]

ROUGE/BERTScore quantify reference similarity, not factual correctness or useful visual grounding. Reference scope can include photos beyond the caption bottleneck. BERTScore uses a 512-token prefix, affecting one F Review and its identical S reuse. Empty Visual outputs remain in the primary cohort with zero similarity scores. Shorter outputs can reduce repetition while reducing coverage; metrics and human criteria do not resolve that tradeoff with a single scalar. [E03–E04]

## 4. Human judgment, provenance and agreement

Two evaluator accounts do not provide broad population reliability or independently authenticated identity. Returning evaluators may remember sources or prior outputs despite randomized blinded assignments. Recorded declarations and technical completion are narrower evidence than verified independent behavior. The historical Stage O record includes an earlier test-context declaration, later identified by the researcher as a test; subsequent human participation is researcher-reported rather than independently authenticated. AI assistance in reference preparation and earlier model-assisted audits is disclosed; those audits are not extra human ratings or gold-standard adjudications. [E07, E10, E20–E21]

The official N data retains accepted-as-recorded audit concerns, including two A02 source/comment findings and a partial-source follow-up. No ratings were corrected by the analysis. P uses the recovered original V1 only; later revisions are excluded and do not retrospectively replace this study's record. This preserves provenance but does not erase measurement uncertainty. No participant is declared dishonest or reliable solely from agreement statistics. [E01, E07]

Criterion scores are ordinal. Means are useful descriptive summaries, not proof of equal interval spacing. Visual Grounding and Contribution combines fidelity and usefulness, so its score does not isolate either construct. Review-only criteria should not be interpreted as effects of pixels when Qwen Review is shared. Coherence ceilings, concentrated marginals and systematic rater offsets limit kappa interpretation. P's two all-5 Coherence cells have undefined kappa; LED repetition ratings show substantial rater disagreement. Exact agreement and score differences accompany kappa. [E13]

Participant “confirmed” grounding flags express the participant's judgment. The integration does not independently verify every statement or all photos, and does not promote flags to a factual-error rate. Sparse comments, differing severity judgments, repeated errors in shared outputs and multi-component judgments limit counting. Missing comments do not prove absence of error. [E07, E10, E22]

## 5. Statistical inference

Whole-business pairing retains both raters and avoids treating 540 P scores as independent. However, 15 restaurants and two fixed raters give limited precision and evaluator generalizability. Bootstrap intervals reflect empirical restaurant resampling conditional on these raters, not uncertainty over a population of raters. The sign-flip tests assume exchangeability/symmetry of paired differences; enumerating all signs does not create random assignment of system identity. Fixed shuffled donors and shared images can weaken business independence. [E02, E12]

M has four adjusted primary BERTScore tests. N/O tests are exploratory and unadjusted under their existing protocols; P has separate Holm families of 12 primary and six additional tests. Pointwise CIs are not simultaneous intervals. An interval excluding zero alongside an adjusted nonsignificant p-value is not a license to select the favorable result. Non-significance is not equivalence, and no non-inferiority margin was specified. No composite winner, post-hoc exclusion or TEST-based checkpoint selection is added. [E03, E06, E09, E12]

## 6. Efficiency and baseline fairness

LED and Qwen differ in parameter count, pretraining, task tuning, prompts, lengths, precision, decoding and framework. Qwen-Image reuses its Review and receives 73 caption-selected images; it is not a completely independent raw-photo pipeline. The historical BLIP run processed 5,143 photos on an offloaded RTX 3080 path, whereas summary benchmarks use an RTX 3090. Final filtering and complete pipeline times are missing. Generation-only ratios, allocator memory and one continuation-training time cannot establish end-to-end latency, minimum hardware, energy or cost superiority. [E15, E18]

## 7. Prospective work, not completed experiments

Reasonable next studies would preregister larger geographically diverse cohorts and additional raters; use participant-authenticated source audits; measure grounding separately from visual usefulness; harmonize summary scopes and training exposures; retain leakage-aware source manifests; test filtering components individually; and measure full pipelines on matched hardware with repeated timing. They could include a prospective non-inferiority question and compute/quality acceptance thresholds. None of these studies has been initiated by this analysis.
''')

 write('FINAL_THESIS_DISCUSSION_DRAFT.md','''
# 22. Integrated Results and Discussion

*Proposed continuation of the numbered stage-based organization in Thesis_Research_Progress_v3.docx. This is an academic English researcher-review draft, not a replacement of the Word report. Evidence codes refer to the accompanying register.*

## 22.1 Evidence base and principal findings

This thesis investigated whether image-caption evidence can complement business-level review summaries while remaining practical under limited computational resources. The final evaluation combines four distinct evidence streams: human-reference automatic evaluation on Yelp TEST100, source-grounded human judgments on Yelp TEST30, reference-free external evaluation on 15 TripAdvisor restaurants, and a three-system comparison with pretrained Qwen. These stages answer related questions but have different units, source domains and measurement settings. Their results are interpreted together without pooling scores or treating nested and reused cohorts as independent replications. [E03–E13]

Three findings define the contribution. First, business-aligned visual evidence is substantially more useful than evidence assigned from other businesses. Second, filtering does not consistently improve quality over minimally validated Unfiltered captions, although it is associated with shorter, less repetitive outputs and an external Non-Redundancy benefit. Third, the tested LED implementation has a smaller prepared-input generation footprint than Qwen, but Qwen receives substantially higher human scores on five criteria. These findings support a qualified quality–resource tradeoff rather than universal superiority of the proposed Filtered system.

## 22.2 Reference agreement and source-grounded quality

Stage M evaluates all 100 frozen TEST businesses against the same approved reference components. F's Visual BERTScore F1 is 0.8374, compared with 0.8525 for U and 0.8295 for Shuffled. F exceeds Shuffled by 0.007852, with a pointwise 95% bootstrap interval of [0.005865, 0.009825] and Holm-adjusted p=0.000080. The F–U Visual comparison is inconclusive under the prespecified adjusted test. Review BERTScore F1 is 0.8572 for F, 0.8587 for U and 0.8595 for RO, giving no evidence of a consistent F advantage. These are similarity results, not direct demonstrations of factual grounding. [E03]

Stage N provides complementary evidence from the sources themselves. F's mean Visual Grounding and Contribution score is 3.900, whereas Shuffled receives 1.233. The paired difference of 2.667 points has a 95% interval of [2.300, 3.017] and exploratory unadjusted p=0.000020. This much larger human-score separation is consistent with the sensitivity of source-grounded assessment to the recipient–image relationship, which lexical or semantic reference similarity alone may not fully capture. It does not establish a calibrated relationship between the two scales. [E05–E06]

The negative control is informative precisely because F/S Review outputs are reused. Its principal intervention concerns the Visual component and recipient alignment. Differences in ratings of identical Review text reflect contextual or measurement variation rather than a new effect on generated Review content. The fixed donor permutation also prevents interpretation as a fully randomized causal experiment on all aspects of multimodal summarization.

## 22.3 Caption filtering and the coverage–repetition tradeoff

Filtering reduces the quantity and redundancy of visual text, but reduction alone is insufficient evidence of better summaries. On TEST100, F Visual outputs average 51.68 words compared with 67.74 for U, with lower mean repeated-trigram fraction, 0.0605 versus 0.1230. In TripAdvisor evaluation, F exceeds U on Non-Redundancy by 0.833 points, with a 95% interval of [0.400, 1.333] and exploratory unadjusted p=0.004740. These results support a repetition-related benefit for the tested complete systems. [E04, E09]

However, Visual Grounding and Contribution does not establish F superiority over U in either human study. The F–U difference is −0.100 on Yelp and −0.167 on TripAdvisor, with both intervals including zero. Yelp Review coverage and usefulness also favor U in exploratory contrasts. It would therefore be inaccurate to conclude that filtering improves quality consistently, or that the absence of a significant Visual difference proves preservation. Short summaries may avoid repeating a scene while omitting a useful aspect. The original hypothesis receives partial support for redundancy control, but limited support for a general grounding or coverage improvement. [E06, E09]

These comparisons concern full pipelines, including different caption conditions and supervision. They do not identify the separate effects of CLIP soft ranking, relevance screening, redundancy suppression, diversity selection or the K limit. A component-level causal claim would require additional matched ablations, which are outside the completed study.

## 22.4 External-domain capability and remaining failures

Without TripAdvisor fine-tuning, the Yelp-trained LED systems produce readable summaries that retain some restaurant-specific information. F receives mean Review Faithfulness 3.700 and Coherence 4.167. Nevertheless, coverage remains 3.233 and usefulness 3.033, while recorded comments identify omissions of mixed opinions, unsupported service attributions and repetitive phrases. External generalization is therefore demonstrated as a capacity to generate partially useful summaries, not as uniformly faithful transfer or no degradation from Yelp. [E08–E10]

The external cohort comprises only 15 restaurants from a restricted region with available English reviews. Domain, source selection, photo availability and evaluator context change together. A direct subtraction of Yelp and TripAdvisor means would not isolate a domain-shift effect. TripAdvisor reference-free human evaluation also differs from Teacher-target agreement: it examines evidence-related judgments rather than reproduction of training-style answers. No new Teacher references are introduced to make those evaluations artificially comparable.

## 22.5 Comparison with pretrained Qwen

The official Stage P analysis uses the original 90 V1 rating records. Independent reconstruction matches the frozen script's descriptive, agreement and paired results. Qwen-Caption exceeds LED F in mean faithfulness, coverage, coherence, non-redundancy and usefulness by 1.033, 1.533, 0.933, 1.633 and 1.467 points respectively. All five comparisons remain supported after Holm correction across the 12 primary tests. LED F and Qwen-Caption receive identical Visual scores in the recorded cells; this does not imply equal overall quality or establish equivalence in visual factual accuracy. [E11–E13]

Qwen Review outputs average 132.87 words against LED's 81.07, so coverage is not assessed under an identical realized output budget. The models also differ in scale, pretraining, instruction following, tuning, prompts and decoding. The result is a practical system comparison rather than an isolated architecture or parameter-count effect. The stronger Qwen results should be acknowledged even though the thesis develops LED.

Qwen-Image receives a mean Visual score of 4.167, compared with 3.533 for Qwen-Caption. The primary Caption-minus-Image difference is −0.633, with a pointwise interval of [−1.033, −0.200], raw p=0.021118 and Holm p=0.147827. The descriptive advantage is promising but inconclusive after primary-family correction. The separate additional Image–LED contrast has adjusted p=0.021118 within its six-test family. Reporting both preserves the frozen analysis rather than elevating the favorable ancillary comparison above the primary one. [E12]

The two Qwen variants share all Review text. Images therefore do not account for an improvement in their Review components. Direct images can in principle expose information lost or distorted by captions, but the actual Image input is restricted to 73 photographs selected by the caption-filter pipeline. The experiment compares text versus pixels at the generation boundary with inherited selection, not a completely caption-independent image-processing system. [E14–E15]

## 22.6 Agreement, provenance and strength of inference

Agreement is criterion- and system-dependent. In Stage P, both evaluators assign every Qwen Coherence score a value of 5, yielding perfect exact agreement but undefined quadratic-weighted kappa because expected disagreement is zero. LED Non-Redundancy has only 6.7% exact agreement and a mean A01–A02 offset of 1.400 points. Both raters nevertheless favor Qwen on its mean repetition contrast. These patterns show why neither kappa alone nor aggregate means can replace examination of per-rater results and score distributions. [E13]

Source-grounded ratings are valuable but are not independently authenticated factual-error labels. Participant comments and “confirmed” flags retain their original certainty categories; the analysis does not adjudicate every output claim or photograph. Human-approved reference construction includes documented AI feedback, evidence suggestions and minor editorial assistance. The recorded approvals and hashes establish workflow and freeze consistency, while participant identity, unaided authorship and actual source inspection have narrower verification. Existing accepted-as-recorded audit concerns remain part of the provenance. [E07, E20–E22]

Statistical pairing uses whole businesses with both raters retained. The resulting intervals are conditional on these raters and small cohorts, rather than a representative evaluator population. Sign-flip tests remain exploratory; their exact enumeration in Stage P does not create randomized system assignment. N/O exploratory tests are unadjusted under their original protocols, while M/P retain their specified correction families. No composite winner, post-hoc exclusion, equivalence claim or TEST-based checkpoint reselection is introduced.

## 22.7 Resource feasibility and deployment implications

Expanded LoRA adapts 4,915,200 parameters while keeping 459,801,600 LED backbone parameters frozen. The trainable count is 1.069% of the backbone count. One accepted 1,000-update F continuation segment completes in 7,465.886 seconds on an RTX 3090, with allocated peak memory 3.414 GiB. This demonstrates feasibility of the segment; it excludes parent training and preprocessing and has no matched full-parameter training control. It does not by itself quantify a LoRA training speedup. [E17–E18]

With prepared inputs, LED generation averages 97.931 seconds and 2.697 GiB allocated peak, compared with 186.844 seconds and 18.536 GiB for Qwen-Caption. Qwen-Image's 379.697-second combined figure reuses the earlier Review time and adds 242.703 seconds of new Visual generation. Different lengths, decoding, precision and timing boundaries limit causal interpretation. The LED benchmark also reproduces only 22 of the 30 original evaluated components exactly, although repeated benchmark outputs agree with each other. Resource figures characterize the implementation rather than an exact replay of every human-rated output. [E15]

Preparation substantially changes the efficiency question. Existing BLIP captioning of 5,143 photos takes 48,642.959 seconds on the historical offloaded RTX 3080 path, and caption scoring takes a further 178.529 seconds. Final selection and complete pipeline intervals are not measured. All three actual Stage P variants inherit caption-derived photograph selection, including Image. Multiplying average caption time by 73 photographs or adding incomplete historical components cannot produce a matched end-to-end benchmark. The thesis can claim lower observed prepared-input generation time and allocator peaks for LED; it cannot claim a demonstrated end-to-end, energy or cost advantage. [E15]

Cached captions and selection may make LED attractive where sources change slowly and response latency or GPU memory is constrained. This is a deployment hypothesis supported by generation measurements, not a measured service-level result. The human-quality deficits relative to Qwen must remain visible when deciding whether that resource tradeoff is acceptable.

## 22.8 Contributions, boundaries and future work

The study contributes a reproducible source-separated caption-to-summary pipeline, a controlled recipient-alignment negative control, frozen common-reference TEST evaluation and complementary human evidence across two domains. It shows that alignment matters, that filtering can reduce repetition without guaranteeing overall quality gains, and that a compact adapted model can operate at a smaller prepared-input inference footprint while trailing a larger pretrained baseline in important quality dimensions.

The main boundaries are condition-dependent training histories and targets, limited raters and cohorts, combined grounding/contribution criteria, source image reuse across Yelp splits, and incomplete full-pipeline resource measurement. The image-overlap audit identifies 17 affected TEST businesses and actual own-condition TRAIN caption overlap; these records remain in the full frozen TEST100 rather than being excluded after outcomes were observed. Performance inflation or memorization is not measured by the overlap counts. [E19]

Future work should preregister broader source cohorts, additional independent evaluators, explicit factuality and incremental-contribution criteria, matched summary budgets and training exposures, filtering-component ablations, and repeated end-to-end measurements on comparable hardware. These are prospective recommendations. The completed evidence supports a qualified answer to the original research questions, with a clear distinction between feasibility, observed benefit and claims that remain unproven.
''')

 # Compact traceability includes exact source/member hashes, not unverifiable prose-only pointers.
 evidence=[
 ('E01','Official P V1 identity and scope','SOURCE_INVENTORY.json; SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json','90 records,540 scores; original archive/export hashes; external ChatGPT ZIP not inspected'),
 ('E02','Independent numerical reconstruction','VERIFIED_TABLES/VERIFICATION_RESULTS.json; SCRIPTS/verify_integrated_research_statistics_v1.py','M42means/24CIs/4tests,N17tests,O11tests,P18tests; no source changes'),
 ('E03','M final TEST100 automatic metrics and inference','SOURCES/M/RESULTS_ORIGINAL/AGGREGATE_METRICS.csv; SOURCES/M/RESULTS_ORIGINAL/PAIRED_COMPARISONS.csv; SOURCES/M/SUMMARY/METHODOLOGY.md','Rows condition/task and contrast/metric; raw0–1 scores;4primary tests;700 component rows'),
 ('E04','M repetition/length','VERIFIED_TABLES/M_OUTPUT_DIAGNOSTICS_VERIFIED.csv; SOURCES/M/RESULTS_ORIGINAL/STRUCTURE_REPETITION_LENGTH.md','U/F visual means; words,repeated_3gram_fraction,duplicate_sentence_fraction'),
 ('E05','N criterion means,SD/distributions/per-rater','VERIFIED_TABLES/N_DESCRIPTIVES_VERIFIED.csv; SOURCES/N/TABLES/DESCRIPTIVE_STATISTICS.json','240records;condition/criterion/panel;Review versus VisualN/A'),
 ('E06','N paired comparisons','SOURCES/N/TABLES/PAIRED_COMPARISONS.json','17rows;contrast/criterion;whole30businesses;unadjusted exploratoryp'),
 ('E07','N agreement and accepted-as-recorded provenance','VERIFIED_TABLES/N_AGREEMENT_VERIFIED.csv; SOURCES/N/METHODOLOGY.md; SOURCES/N/PROVENANCE/MODEL_ASSISTED_AUDIT_SUMMARY.json; SOURCES/N/PROVENANCE/RESEARCHER_ACCEPTANCE_AS_RECORDED.json','23condition-specific agreementcells;knownsource/commentconcerns;notindependentlyauthenticated'),
 ('E08','O source/domain and descriptive results','VERIFIED_TABLES/O_DESCRIPTIVES_VERIFIED.csv; SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md; SOURCES/O/ANALYSIS_CONFIG.json','15restaurants,90ratings;English/geographic/conveniencescope;noTripAdvisorfinetune'),
 ('E09','O paired comparisons','SOURCES/O/TABLES/PAIRED_COMPARISONS.json','11exploratorycontrasts;F−Unonred+.833,Visual−.167'),
 ('E10','O agreement and qualitative scope','VERIFIED_TABLES/O_AGREEMENT_VERIFIED.csv; SOURCES/O/COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json; SOURCES/O/METHODOLOGY_AND_LIMITATIONS.md','17agreementcells;commentsareparticipantjudgments'),
 ('E11','P official descriptive statistics','VERIFIED_TABLES/P_DESCRIPTIVES.csv','model/criterion/panel;54rows,18combinedcells;means/SD/median/all1–5counts'),
 ('E12','P paired analysis and multiplicity','VERIFIED_TABLES/P_PAIRED_COMPARISONS.csv; SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json','18contrasts;15pairedrestaurants;20kbootstrap,seed20261106;32768signs;Holm12/6'),
 ('E13','P condition-specific agreement','VERIFIED_TABLES/P_AGREEMENT.csv','18cells;twoQwenCoherencekappasundefined;LEDNonRedexact1/15,bias1.4'),
 ('E14','P output budget/shared Review','VERIFIED_TABLES/P_OUTPUT_LENGTH_CONTEXT.csv; SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/FROZEN_OUTPUTS_45.json','SharedReview15/15;whitespacewordmeans81.07/132.87;notnativetokens'),
 ('E15','Measured generation and missing pipeline costs','SOURCES/EFFICIENCY/VERIFIED_FACTS.json; SOURCES/EFFICIENCY/EFFICIENCY_MEASUREMENTS.csv; SOURCES/EFFICIENCY/EFFICIENCY_COMPARISON.md; SOURCES/EFFICIENCY/MISSING_MEASUREMENTS.md','Existing audit provenance rows;generation97.931/186.844/constructed379.697;BLIP48642.959;score178.529;E2Eunmeasured'),
 ('E16','Original research questions','SOURCES/RESEARCH_QUESTIONS.json; SOURCES/APPROVED_PROPOSAL/DOCUMENT_EXTRACT.json','Exact3questions;sourceWordhash;noobjective redefinition'),
 ('E17','Final model ancestry and LoRA scope','SOURCES/M/PROTOCOL/MODEL_AND_DECODER_LOCK.json; SOURCES/TRAINING/LORA_TRAINABLE_PREFLIGHT.json; SOURCES/TRAINING/TRAINING_COMPLETION_REPORT.md; SOURCES/TRAINING/INITIALIZATION_HASHES.json','RO/Uown750parents,FCE3parent;4000R+0/4000V;4915200trainable/459801600backbone'),
 ('E18','Accepted F continuation runtime','SOURCES/TRAINING/F_S1000_COMPLETION_NATIVE.json','completion.seconds7465.885813831992;peak_allocated_bytes3665676800;peak_reserved_bytes5572132864;RTX3090'),
 ('E19','Known cross-split image exposure','SOURCES/IMAGE_OVERLAP/EXPOSURE_SUMMARY.json; SOURCES/IMAGE_OVERLAP/DUPLICATION_CAUSE_AND_IMPACT_SUMMARY.md','17businesses,123TESTphotoIDs,490pairs;selectedU66/F44;ownTRAINU59/F42'),
 ('E20','Study measurement/provenance scope','SOURCES/P/FROZEN_STUDY/COORDINATOR_ONLY/ANALYSIS_PLAN.json; SOURCES/N/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md; SOURCES/O/COLLECTION/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md','Returningraters;accountrecordsnotidentityauthentication;noindependentpopulationofraters'),
 ('E21','Reference AI-assistance disclosure','SOURCES/M/REFERENCE_PROVENANCE/AI_ASSISTANCE_AND_LIMITATIONS.md','Author/checkerapprovalandAIassistance;freezehashesnotunaidedauthorshipproof'),
 ('E22','P participant comments/flags','VERIFIED_TABLES/P_FROZEN_SCRIPT_REPLAY/RATER_COMMENTS_VERBATIM.json; SOURCES/P/RATINGS/STAGE_P_TRIPADVISOR15_RATINGS_FINAL_90.json','Verbatimreports;notanindependenterrorcensus'),
 ]
 with (T/'EVIDENCE_REGISTER.csv').open('w',encoding='utf8',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['evidence_id','claim_scope','package_files','row_or_boundary']);writer.writerows(evidence)
 hashes=md(['Stage','Original source archive SHA-256'],[[r['stage'],'`'+r['sha256']+'`'] for r in inventory['archives']])
 register=md(['ID','Claim scope','Source files / row locator'],[[i,scope,'; '.join('['+x+']('+x+')' for x in paths.split('; '))+' — '+note] for i,scope,paths,note in evidence])
 write('EVIDENCE_TRACEABILITY.md',f'''
# Evidence Traceability

All new prose refers to the following internal evidence IDs. Filenames are relative to this package. Exact member byte hashes are in `SOURCE_INVENTORY.json`; final delivered-file hashes are in `PACKAGE_MANIFEST.json`. The review ZIP contains selected small evidence, not every original large experiment archive. Historical scripts/prompts/documents under `SOURCES/` are preserved evidence, not new execution instructions.

## 1. Frozen source archives

{hashes}

Original archive paths, CRC checks and every-member manifest checks are recorded in `SOURCE_INVENTORY.json`. P is explicitly bound to recovered V1, never the live/revised data. The external ChatGPT statistical ZIP is unavailable and was not verified; independent reproduction was authorized as the replacement for comparing that absent artifact. Source hashes establish bytes and provenance, not semantic correctness or authenticated participant identity.

## 2. Claim-level register

{register}

## 3. Calculation and reporting conventions

M means are independently recomputed from stored per-business similarity scores; embedding/ROUGE models are not rerun. N/O/P means, sample SDs, medians, distributions and condition-specific QWK are independently recomputed from the frozen numeric scores. N/O/M paired intervals and tests are reconstructed with their original Monte Carlo order and seed. P independently enumerates integer-scaled half-point differences and all2^15sign assignments, then matches the unchanged frozen script's output. The bootstrap index hash is stored. Numerical comparisons allow rounding-level tolerance, not replacement data.

Pointwise percentile95%CIs and family-wise p-values are labeled separately. No new exclusion, corrected rating, compositewinner,condition pooling or multiplicity policy is introduced. P additional-family significance does not override an inconclusive primary-family contrast. All later P revisions are excluded.

## 4. Style and artifact preservation

`SOURCES/STYLE_REFERENCE/DOCUMENT_EXTRACT.json` records the numbered stage organization and SHA-256 of the unchanged Word style reference. Its original file is not overwritten or bundled. The discussion follows that organization while avoiding the unprespecified six-criterion composite score found in the reference report. No notebook, final thesis rewrite or presentation is produced.

The new statistical script, report builder and archive verifier are supplied under `SCRIPTS/`. `VALIDATION_REPORT.json` and `SOURCE_PRESERVATION.json` document actual final checks. No database, login/password file, model binary, API credential or original photo archive is included. These exclusions do not alter the frozen source evidence retained in the project.
''')
 write('README.md','''
# Final Research Results, Cross-Stage Analysis and Discussion

Researcher-review package. Begin with `FINAL_RESULTS_ANALYSIS.md`, then the original-RQ answers and academic discussion draft. The current Word report is unchanged. This is not a new experiment or a final thesis submission.

## Contents

- Six requested Markdown reports at the root: integrated results, research questions, quality/efficiency, limitations, discussion draft and evidence traceability.
- `VERIFIED_TABLES/`: independent statistics, complete Stage P distributions/per-rater results and all18 contrasts; original frozen-script replay; machine-readable verification receipt.
- `SOURCES/`: byte-preserved selected original M/N/O evidence, the compact recovered original P V1 evidence, efficiency-audit reports, original-RQ/style extracts and accepted training provenance. Original research protocols are historical evidence, not instructions to run new experiments.
- `SCRIPTS/`: CPU-only reproduction and package checks. `requirements.txt` pins the numerical/plotting environment.
- `FIGURES/`: separate-stage descriptive means and Stage P paired intervals. No pooled score or composite ranking.
- `SOURCE_INVENTORY.json`, `SOURCE_PRESERVATION.json`, `PACKAGE_MANIFEST.json` and `VALIDATION_REPORT.json`: exact bindings and actual checks.

## Reproduce using only small included evidence

Python3.11 and NumPy1.26.4 are sufficient for statistical verification. No GPU, paid API, model download or full dataset is required. From this extracted directory:

```text
python -B SCRIPTS/verify_review_package.py --package .
python -B SCRIPTS/verify_integrated_research_statistics_v1.py --package . --output ../fresh_statistics_reproduction
```

The second command requires a new output directory and writes only derived files there. It reruns arithmetic/bootstrap/sign-flip calculations and the included frozen P analyzer, not models, reference creation or similarity scoring. The original archives are not needed for this portable replay; their original hashes/check receipts remain available. Rebuilding prose uses the repository-local report builder and is not required to read or validate the package.

The statistical reproduction also invokes `SOURCES/P/VERIFY_PACKAGE.py --package SOURCES/P` to recheck the original collection, assignment bindings and journal. The root verifier checks the entire integrated package. Do not launch historical application or packaging scripts for review.

## Scope and key conclusions

Official Stage P is recovered original V1, archive SHA-256 `e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5`. The external ChatGPT analysis ZIP is not claimed to have been inspected. Its absence was resolved by the researcher's instruction to reproduce from V1. Later rating revisions are excluded.

Alignment matters: F exceeds Shuffled on Yelp Visual outcomes. Filtering has some repetition benefits but does not show consistent superiority over U. Stage P Qwen-Caption exceeds LED on five criteria after its frozen correction; equal Visual scores do not imply overall equivalence. Qwen-Image's primary Visual advantage over Caption is inconclusive after Holm. LED's smaller recorded prepared-input generation footprint does not establish complete-pipeline superiority.

All stages remain separate. The package preserves original ratings and known provenance limitations. It does not independently authenticate participants or adjudicate every flagged claim. Stop for researcher review; no checkpoint reselection, further model execution or human collection is authorized by this deliverable.
''')
 print('Six requested reports, README and evidence register written from verified inputs.')

if __name__=='__main__':main()
