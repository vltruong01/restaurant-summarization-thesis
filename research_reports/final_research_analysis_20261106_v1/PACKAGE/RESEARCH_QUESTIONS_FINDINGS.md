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
