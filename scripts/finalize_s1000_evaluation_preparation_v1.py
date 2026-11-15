"""Write protocols, preserve documentation history, execute CPU checks and verify review ZIP."""
from pathlib import Path
import shutil,json,csv,sys,subprocess,datetime,zipfile,tempfile,ast,difflib,importlib.metadata,collections
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/final_evaluation/s1000_preparation_v1';E=O/'EVIDENCE';A=O/'TEST100_REFERENCE_ANNOTATION_PACKAGE'

def prose(p,t):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(t.strip()+'\n')
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);require(file_hash(p)==file_hash(q),'Copy failed')
def main():
    require(not (O/'PACKAGE_INTEGRITY_REPORT.json').exists(),'Preserve completed package')
    counts=read(E/'PREPARATION_COUNTS.json');sv=read(E/'SHUFFLED_VALIDATION.json');model=read(O/'MODEL_AND_DECODER_LOCK.json')
    require(counts['TEST_generations']==counts['TEST_scores']==0 and sv['actual_visual_source_changes']==98,'Preparation scope/counts')
    rb=read(O/'METRIC_ASSETS/METRIC_SOURCE_BINDINGS.json');small={r['path']:r['downloaded_sha256'] for r in rb['files'] if r.get('downloaded_sha256')}
    metric=dict(protocol_version='S1000_FINAL_AUTOMATIC_V1_PREOUTCOME',packages={'rouge-score':'0.1.2','bert-score':'0.3.13','transformers':'4.51.3','torch':'2.1.0+cu118','numpy':'1.26.4','nltk':'3.9.1','tokenizers':'0.21.1'},
        rouge=dict(types=['rouge1','rouge2','rougeL'],use_stemmer=True,implementation='rouge_score.rouge_scorer.RougeScorer',input='Full component; remove exactly expected leading header only; retain malformed raw text, record flags',aggregation='macro mean per business; P/R/F1 all retained; no rougeLsum'),
        bertscore=dict(package_version='0.3.13',model_id=rb['repo_id'],revision=rb['revision'],weights_file='pytorch_model.bin',weights_sha256='36a10a8b694fadf9bf4f9049d14e257e88be45313ae02d882af9e60f39b8b2e8',small_file_sha256=small,
            num_layers=17,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False,lang='en',device='cpu',dtype='FP32',batch_size=8,nthreads=1,model_max_length=512,
            prefix_policy='Standard bert_score sent_encode fixed512-token prefix including BOS/EOS; report full untruncated candidate/reference lengths and truncation flags for EVERY row; no generation truncation/change',
            limitation='BERTScore prefix may omit summary suffixes; full-text ROUGE/coverage/length remain. No inference of factuality or image grounding.'),
        lexical_diagnostics=dict(normalization="NFKC/casefold regex [a-z0-9]+ internal apostrophes/hyphens",repetition='1 - unique ngrams / total ngrams for n1..4; duplicate normalized sentence fraction',
            reference_coverage='Unique reference word recall and ROUGE recall; lexical proxies, not verified fact/aspect/scene coverage',source_support='Fraction output tokens also in actual task text; source lexical proxy, not hallucination detection or original-photo grounding'),
        statistics=dict(unit='business',bootstrap='paired row-index resampling with replacement, macro difference F-minus-comparator; numpy default_rng PCG64; percentile95% CI',bootstrap_replicates=20000,signflip_replicates=50000,seed=20261028,
            primary_tests=['Review BERTScore F-RO','Review BERTScore F-U','Visual BERTScore F-U','Visual BERTScore F-S'],multiplicity='Holm over these4 two-sided sign-flip p-values',alpha=.05,
            secondary='Full-text ROUGE, repetition, length and lexical coverage paired CIs, descriptive only; no confirmatory significance claim',
            F_S_Review='Structural reuse; equality check, no independent statistical test',F_S_visual='Report full100 AND predeclared actual-input-change98; active98 is mechanistic intervention estimand, not replacement TEST cohort',
            zero_captions='Both retained in main100; original empty task generated for F, exact result reused in S; no F-S visual intervention; no exclusion of U/F Visual metric rows',
            RO_Visual='N/A; no missing-component penalty',image_overlap='Retain all100; disclose17 affected businesses and condition-specific actual-input exposures. No TEST83 replacement or outcome-based subgroup choice'),
        software_status=dict(bert_score_installed_here=False,bert_weights_downloaded=False,GPU_or_BERT_forward_tested=False),
        sources=['https://github.com/Tiiiger/bert_score','https://pypi.org/project/rouge-score/0.1.2/','https://huggingface.co/FacebookAI/roberta-large/tree/722cf37b1afa9454edce342e7895e588b6ff1d59'])
    write(O/'METRIC_LOCK.json',metric)
    pins=read(R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1/VALIDATION_PHASE/DECODE_CONFIG.json')['runtime_pins']
    write(O/'INFERENCE_RUNTIME_LOCK.json',dict(Python='3.11',packages=pins,precision='FP32 weights, FP16 autocast, no GradScaler/backward',seed=42,
        actual_GPU_runtime_validation=False,source='Previously successful native Source-Separated validation runner',inference_now_authorized=False))
    prose(O/'requirements-metrics.txt','\n'.join(k+'=='+v for k,v in metric['packages'].items())+'\nscipy==1.15.2\n')
    prose(O/'requirements-inference.txt','\n'.join(k+'=='+v for k,v in pins.items()))
    prose(O/'SHUFFLED_MAPPING_VALIDATION_REPORT.md',f'''# TEST100 Shuffled mapping verification

The mapping is frozen before TEST outputs, under the researcher's preparation request. All 100 original TEST businesses remain. The 98 positive-caption recipients form a one-to-one derangement within exact caption-count strata; two zero-F-caption recipients retain their exact original empty Visual input and have no donor. All 98 positive recipients have changed Visual source text. No TEST inference occurred.

Seed 42. Sorted business IDs and count groups feed SciPy 1.15.2 linear_sum_assignment. Feasible edges preserve the unchanged whole donor filtered block, K<=5, 750 incremental-caption tokens, 5000 total source tokens and 750 isolated Visual tokens. The cost first minimizes the GLOBAL number of category fallbacks, then a SHA256-seeded integer tie cost. A fallback penalty of 1,000,000,000 exceeds the sum of all possible secondary costs. Forbidden edges cost 10,000,000,000,000. Self edges and count/budget violations are forbidden. Exact allowlisted NFKC/casefold category intersections give 79 shared-category assignments; 19 use the declared same-count fallback. There is no learned cuisine similarity classifier. With donor competition, this global optimum may differ from individual greedy preference. Cuisine labels and input sizes alone are used; no outcomes/references are used.

There is a feasible perfect assignment in every positive stratum: counts 1/2/3/4/5 have 7/18/21/25/27 businesses. Donor frequencies are exactly one each in these strata, preserving the multiset of filtered caption blocks. The two zero cases are `4iuNK5svIKV_GrbSwAXM6Q` and `PLebbMEYWV-5ggzx0c4lBQ`. No constraints were silently relaxed. Reversing the input-row order reproduces the entire mapping exactly.

Every Review source and input-token hash is identical to F. S uses F-S1000 adapter `{model['models']['F']['adapter_sha256']}` and the original per-task beam4/length-penalty1/max_length768/EOS/BOS attention policy. Evaluation sources and future gold references belong to the recipient, not the donor. S donor photo IDs identify deliberately wrong input, not gold evidence.

Primary F-S Visual intervention reporting must show the 98 changed-input cases alongside full100 results; the two empty cases are intervention N/A, not removed businesses. F-S Review invariance is structural and will be implemented by exact F Review reuse, not described as evidence that a joint decoder ignored distractors. The two identical empty Visual calls also reuse F results explicitly. Thus 400 assembled condition-business records correspond to 700 logical components and 598 new generation calls (RO100/U200/F200/S98), with 102 reused components. No S checkpoint is trained.

Proposal v16 paragraphs 732/898/900/902 require same reviews, same-split different donor, cuisine preference, fixed seed, same constraints, F checkpoint and unchanged recipient reference. The implemented source-separated intervention satisfies these at Visual-task level, whereas the proposal illustrates joint input. The previous with-replacement proposal remains untouched; the latest request authorizes this one-to-one refinement. `SHUFFLED_POLICY.json`, the complete CSV and JSON bind the exact rule and inputs. Final protocols remain for researcher/ChatGPT review; this mapping does not authorize generation.
''')
    guidelines='''# Independent human TEST100 reference guidelines

These are blank preparation materials, not completed references. Exactly one common reference per business will be used for all applicable models. Two DISTINCT human participants are required: an author and an independent checker. Neither receives predictions, automatic scores, model identity keys or TRAIN/VALIDATION Teacher targets. Do not distribute the full administrator review ZIP to annotators; distribute only this annotation package, the guidelines and verified original-photo access.

## Source scope

Review Consensus uses ONLY the frozen selected reviews in each SOURCES business JSON / CASES Markdown. Preserve review IDs, exact text, order, star/date metadata and source SHA. Identify recurring positive and negative opinions, mixed experiences, important minority warnings and business-specific details. Attribute subjective claims to reviewers. Do not turn an individual experience into universal consensus, erase negation, infer reviews from photos, or invent a dish/ingredient. Use exact supporting excerpts and review IDs for important claims. Read all selected reviews before drafting; do not use search-engine reviews or Teacher prose.

Visual Evidence uses ALL original photos listed for that business, in photo-ID order. These are unchanged byte copies whose SHA matches the archived original Yelp image, not thumbnails or generated images. Inspect the source images directly. Photo IDs establish association with the business, not alignment to a review. Describe only observable objects, attributes and relationships. Keep different photos/scenes separate; do not merge people/actions/food across images. Avoid guessed ingredients, exact dish identities and brand/location conclusions unsupported by pixels. Cover useful recurring and distinctive visible information without listing every repetitive image. The common reference is not restricted to any U/F caption subset. No model captions are provided by default, to avoid anchoring; they are optional supporting material only if the researcher later provides a separately logged source, and every visual claim still requires direct photo verification.

The two businesses with no F captions still have their original photo sources; absence of selected captions does NOT require an empty human Visual reference. Do not write the donor's visual scene into a recipient reference. Original images include known cross-split reuse: retain source scope and write observable content honestly; this does not establish model memorization or inflate a reference intentionally.

## Structured workflow

1. Coordinator verifies ANNOTATION_PACKAGE_MANIFEST.json and SOURCE file hashes. Give a human author only the corresponding source files and blank JSONs. CASES files provide reviews and original-photo links. For another machine run materialize_reference_photos_v1.py into a NEW folder, transfer it under the Yelp dataset's terms, and use photo IDs from the JSON to open the files. Do not change the frozen JSON source paths/hashes. No public publication of Yelp data is implied.
2. Copy a blank TEST100_REFERENCE_TEMPLATE/{business_id}.json to an author working directory. Fill plain text `review_consensus` and `visual_evidence`; do not type section markers into the fields. Set `review_claims` / `visual_claims` with exact output spans and evidence. Review evidence entries are `{review_id, quote}` with an exact source excerpt; Visual evidence entries are `{photo_id, direct_image_viewed:true}` with optional observable-detail notes. Do not fill these attestations without actually viewing the source.
3. Fill author.human_id, UTC timestamp and truthful human_written_attestation/predictions_seen fields. The coordinator keeps a separate identity roster and consent record. Pseudonymous IDs are sufficient in the research package; automated checks cannot authenticate people.
4. Preserve each version with snapshot_s1000_reference_version_v1.py. It creates a new immutable version and records content hash, parent-file hash, supplier and time. Never overwrite prior versions. The script writes no prose and grants no approval. A checker receives the source and human draft, independently verifies each component/evidence link, and requests corrections where necessary. Keep rejected/uncertain claims in notes/history, not as supported gold facts.
5. After correction, a DIFFERENT human checker fills checker.human_id, at_utc, independent_check_attestation, predictions_seen:false and reviewed_content_sha256 (the exact reference_content digest printed by the validator). Set status APPROVED only after this actual check. If text or evidence changes, the content hash changes; repeat independent checking. Snapshot the checked version. Record the exact approved-file SHA in the coordinator's REFERENCE_APPROVAL_TRACKER.csv. No script automatically asserts human approval.
6. Validate all100 with validate_s1000_human_references_v1.py. Draft validation checks structure; final validation additionally requires complete text/evidence, two distinct IDs, UTC chronology, exact quotes, same-business valid evidence, approval/history hash bindings and blinding attestations. Export APPROVED_REFERENCES.json and REFERENCE_FREEZE.json only after all100 pass. Freeze references before final inference/automatic evaluation. No automatic validator proves semantic factuality or reference quality; that is the humans' responsibility.

Command examples from the thesis workspace (use copied scripts on another machine):
```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/validate_s1000_human_references_v1.py --package outputs/final_evaluation/s1000_preparation_v1/TEST100_REFERENCE_ANNOTATION_PACKAGE --annotations HUMAN_DRAFTS --draft
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/validate_s1000_human_references_v1.py --package outputs/final_evaluation/s1000_preparation_v1/TEST100_REFERENCE_ANNOTATION_PACKAGE --annotations HUMAN_CHECKED --export HUMAN_REFERENCE_FREEZE
```

Reference authorship and later human output scoring are separate activities. Human scoring is not opened until automatic evaluation is complete. Prefer different raters; if personnel overlap, disclose that exposure and do not claim complete independence. All reference fields in this delivered preparation are EMPTY; no GPT-authored reference is labeled human-written.
'''
    prose(O/'HUMAN_REFERENCE_GUIDELINES.md',guidelines);prose(A/'HUMAN_REFERENCE_GUIDELINES.md',guidelines)
    prose(O/'AUTOMATIC_EVALUATION_PROTOCOL.md','''# Frozen S1000 automatic evaluation protocol

Version S1000_FINAL_AUTOMATIC_V1_PREOUTCOME. Prepared and hash-frozen before TEST predictions; subject to researcher/ChatGPT protocol review. Order: preparation approval -> independent human references and second-human checking -> approved final inference and automatic evaluation -> separately released human scoring -> results/discussion. No training recovery or model selection from TEST.

## Models and data

The complete frozen TEST100 is the primary cohort for RO/U/F/S. No TEST83 replacement or data removal. RO/U/F use their exact S1000 adapters; S uses F, with unchanged recipient reviews and frozen donor Visual source. RO has100 Review components; U/F/S each100 Review and100 Visual logical components. F/S share100 Review outputs; two unchanged empty Visual sources also reuse F. Raw tokens/text and reuse bindings are retained; do not count these as independent draws. Source-separated Review inputs contain no captions, so Review differences among RO/U/F reflect trained adapter histories, not direct Visual conditioning. U/F differ in input selection and supervision; this is a pipeline comparison, not an isolated filter-only causal effect. Whole-photo reference scope also measures caption bottleneck/selection omissions. Caption-free RO receives no Visual score or penalty.

## Text metrics and preprocessing

Score each raw TASK output against the corresponding component of the SAME common human reference. Remove only the exact expected leading component marker. Do not repair wrong headings, split out a prettier answer, rerank or selectively regenerate. Record missing/foreign headings, empty output, EOS/cap status and raw sequence length. The score input and its raw output hash stay traceable. Genuine missing files are execution failures to resolve transparently, never silently dropped; a completed empty/malformed output is retained as generated.

ROUGE-score0.1.2 computes full-text ROUGE-1/2/L with Porter stemming, P/R/F1. No rougeLsum or sentence-layout tuning. Report per-business values and unweighted macro means. Lengths use explicit NFKC/casefold regex words, raw characters, raw generation tokens and sentence-boundary regex. Repetition uses duplicate normalized sentence fraction and repeated1..4gram fractions. Lexical reference-word recall and ROUGE recall are automatic information-coverage proxies. Input word overlap is only a lexical support diagnostic; for S it measures fidelity to the deliberately wrong donor input, NOT grounding in recipient photos. No automatic claim validator or image-grounding certification is implied.

BERTScore0.3.13 uses FacebookAI/roberta-large at immutable revision722cf37b1afa9454edce342e7895e588b6ff1d59, layer17, slow tokenizer, IDF false, no baseline rescaling, CPU FP32, batch8, one thread. Binary/config/tokenizer hashes are in METRIC_LOCK.json. P/R/F1 are retained. The library uses a fixed512-token prefix including special tokens. This policy is declared BEFORE outcomes: record original candidate/reference token counts and truncation flags per row; report numbers/fractions affected beside all BERTScore results. Full generation remains max_length768 per task, unchanged. Full-text ROUGE, length and coverage are retained even when BERTScore omits suffixes. Do not silently describe prefix BERTScore as full-document semantic fidelity. This explicit limitation is a protocol-review item, not an outcome-triggered adjustment.

Official implementations: [BERTScore](https://github.com/Tiiiger/bert_score), [ROUGE-score0.1.2](https://pypi.org/project/rouge-score/0.1.2/). All package versions, scorer checkpoint and preprocessing rules are in METRIC_LOCK.json. Missing BERTScore dependency/large weights is currently a runtime blocker; this stage downloaded only the small package wheel/config/tokenizer files and metadata. No BERT forward pass or TEST metric was run. The scorer runs offline against the explicitly verified local model directory; no automatic download/rental fallback.

## Paired statistical plan

Unit of analysis is business, paired on identical IDs. Report F-minus-comparator mean differences with20,000 paired business bootstrap replicates and95% percentile intervals, numpy default_rng PCG64 seed20261028. Two-sided paired sign-flip tests use50,000 sign draws and (extreme+1)/(P+1). These rely on business sampling/exchangeability assumptions; chain/photo reuse and repeated scene content may weaken independence, so p-values are conditional descriptive evidence, not population causal proof.

Four predeclared primary tests: Review BERTScore-F1 F-RO and F-U; Visual BERTScore-F1 F-U and F-S. Apply Holm across all four at family-wise alpha0.05. Full-text ROUGE, length, repetition and coverage differences are secondary descriptive paired analyses, with intervals and no new confirmatory significance claims. F-S Review is structurally identical and receives an equality check, not an independent hypothesis test. No training/validation loss ranking is used.

Report all100 for every applicable component. Also report F-S Visual over the98 prospectively frozen changed-input businesses to interpret the actual intervention; the two empty-F cases are intervention N/A. This is a nested mechanistic analysis, not an alternative83-business benchmark. Do not exclude difficult cases based on outputs. Known original-photo overlap:17 businesses/123 TEST photo IDs/490 cross-split pairs; selected U66/F44, including own-TRAIN caption/support overlap U59/F42. Disclose these condition-specific counts; no memorization, Teacher mimicry or performance inflation has been proven. No unapproved image-disjoint secondary cohort is created.

## Execution and reporting

Validate/freeze the100 two-human references first. Review the protocols, authorize available compute, then run the finite inference entry point once per condition. Default script behavior performs CPU preflight only. GPU generation requires an explicit later release, valid human-reference freeze and explicit --execute. No new resource is rented. On a substantive failure preserve partial raw evidence and stop; no selective retry. Final scoring requires all four inference receipts, source/checkpoint bindings, approved references and pinned metric assets. Keep training/runtime/cost provenance separate from quality; S has no training cost, and its reused components have no new-generation runtime. Report actual measured runtime when available; no future cost is fabricated.

Package authoring, code tests and source checks are complete preparation, not final Automatic Evaluation. The human-output study is executed only after automatic completion. External model comparisons, TEST reference drafting by AI and further fine-tuning remain outside this task.
''')
    prose(O/'HUMAN_EVALUATION_PROTOCOL.md','''# Human output evaluation design only

Do not distribute scoring forms or predictions yet. Actual scoring begins only AFTER automatic evaluation completes and the researcher releases the human study. Reference construction happens earlier and is a separate task. Three human annotators A01/A02/A03 independently rate the same30 TEST businesses and four conditions (120 outputs each;360 rating sets). The cohort is frozen by source/outcome-independent seeded hash ranking in HUMAN_EVALUATION_COHORT30.json. It is a reproducible simple random sample, not a handpicked quality panel.

PRIVATE_HUMAN_DESIGN_KEY.csv maps per-rater aliases and independently seeded randomized positions to businesses/conditions. Keep it coordinator-only. The delivered blank rating template is DESIGN ONLY; no output text, ratings or automatic scores have been filled. Later rendering must read only the frozen predictions and the recipient's selected reviews/original photos; never donor photos as recipient evidence. Remove model names, filenames and condition labels from displays, and do not expose reference texts or automatic scores. Randomized position differs per annotator. RO lacks Visual text, so structural absence may reveal its condition; identity masking cannot guarantee complete blinding. F/S identical Review outputs are known structural dependence, not four independent Review samples. Raters' prior reference-author exposure must be recorded; prefer separate personnel and disclose overlap.

## Five-point rubric aligned with proposal v16 Table3.11

Higher is better for all six criteria; `non_redundancy` stores the proposal's Redundancy criterion (avoidance of repetition). Scores2 and4 are intermediate anchors. Require exact output spans and review/photo IDs in notes for serious grounding defects; keep uncertainty distinct from confirmed unsupported claims.

| Criterion | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Review Faithfulness | Major unsupported/contradictory review claims or polarity/attribution errors | Mostly supported but some material imprecision | Claims consistently supported with correct attribution/polarity |
| Aspect Coverage | Misses central recurring positives/negatives and mixed feedback | Captures some central aspects, misses important qualifications | Representative positive/negative/mixed coverage with business-specific detail |
| Coherence | Hard to follow, fragmented or illogical | Generally readable with noticeable organization issues | Fluent, logical, clearly structured components |
| Redundancy avoidance | Severe repeated ideas/sentences | Noticeable local repetition | Concise without unnecessary repetition |
| Usefulness | Misleading/generic, poor account of recurring customer experience | Partly useful but important context missing | Useful, representative, specific account grounded in selected reviews |
| Visual Grounding and Contribution | Major unsupported entities/relations, wrong scenes, or no useful observable content | Mostly grounded but omissions or uncertain details | Faithful and useful observable content verified in the original recipient photos |

The first, second and fifth criteria judge Review Consensus. Coherence and redundancy concern available generated components, as v16 specifies; the different output budgets/component counts are a comparability limitation. Visual criterion is N/A for RO and excluded from its means, never scored1 for absence. U/F/S are assessed against actual original recipient photos. For zero-F inputs keep the case and mark intervention N/A; its generated Visual component is still assessed for support/usefulness against available original photos. Do not fabricate evidence to justify an empty-caption output.

## Agreement and uncertainty

Require complete three-rater records before analysis; do not impute missing scores. Report each rater's score distribution, per-condition/criterion mean and business-level averages. Ordinal Krippendorff alpha uses pooled category frequencies for ordinal distance, separately per condition/criterion (30 units,3 raters); constant-category expected disagreement yields undefined alpha, not fabricated perfect agreement. Script implementation and synthetic tests are included. Preserve rater-level data and disagreement notes; do not replace independent ratings with consensus for agreement estimates. Descriptive paired F-RO/F-U/F-S differences average the three raters within each business, then use the same seeded paired business bootstrap for confidence intervals. Human tests are exploratory; no new winner selection or unplanned multiplicity claims. Agreement is not semantic accuracy.

No human has been scored or approved by this preparation. analyze_s1000_human_ratings_prepared_v1.py requires the automatic-completion receipt and a subsequent researcher release. Do not run it on the blank template. The coordinator identity/consent roster and actual annotator recruitment remain human dependencies.
''')
    prose(A/'SOURCE_ACCESS_INSTRUCTIONS.md','''# Access the original photo sources

The annotation package includes100 source JSONs,100 readable review cases,100 blank reference JSONs, an aggregate blank template and approval tracker. It contains all2817 exact selected reviews and IDs. It does not contain model outputs, Teacher targets, Shuffled donors, alias keys, automatic scores or any filled reference.

All1136 original photos have verified local paths and byte SHA in the source JSONs. They match original raw photo members listed in the immutable private HF backup manifest. This review ZIP omits JPEG binaries to avoid unnecessary large datasets. The existing workspace copies are already the same original JPEG bytes; no image processing is needed. Open their local paths from CASES, or run the supplied materializer into a new folder and transfer it privately to annotators. The materializer verifies every input and output SHA and refuses overwrite. A copied-photo folder plus the source JSONs is the portable equivalent of the workspace paths. The original private backup is vltruong01/multimodal_summarization_thesis at revision7e38650c9d93effab7c33d1ec3dd4c4ececf0965; restoring from it requires the researcher's existing authorized HF access, not credentials embedded here. No network restore or upload is performed by the materializer.

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/materialize_reference_photos_v1.py --workspace C:/thesis_proposal/yelp_dataset --annotation-package outputs/final_evaluation/s1000_preparation_v1/TEST100_REFERENCE_ANNOTATION_PACKAGE --destination C:/ANNOTATION_ORIGINAL_PHOTOS_TEST100
```

Read only original recipient photos, not donor photos. The common reference may include directly visible details missed by captions. Do not use restaurant-name knowledge or external reviews to guess visual facts. Do not start writing until the preparation protocols are approved and actual humans are assigned.
''')
    prose(O/'INFERENCE_PREPARATION.md','''# Prepared final inference

TEST100_INFERENCE_PLAN.json freezes exact task text, input token IDs, source/token/checkpoint hashes, review/photo IDs and explicit F reuse for all400 records. Raw calls preserve selected reviews, captions, original EOS/forced EOS and BOS-only global attention. The implementation recreates the successful native LED-large architecture from the hash-verified public binary, restores the exact456-tensor adapter, uses FP32 weights and FP16 autocast with no optimizer or backward pass, pads on the right to1024, and calls the original complete generation config (beam4, length penalty1, max_length768 per task). Components concatenate verbatim, with malformed headers flagged rather than repaired. Each logical component preserves raw generation tokens and source/adapter/decoder hashes.

RO/U/F each load only their own frozen checkpoint. S loads F;100 Review and2 unchanged empty Visual components reuse exact F tokens/text after identity checks. The98 actual changed S Visual calls are generated once. Maximum new calls598; all400 assembled records remain. Failed/missing executions are not silently replaced or selectively regenerated.

Default execution runs file/source/token preflight only and constructs no model/CUDA context. Later --execute requires a new researcher release file, approved-reference freeze, pinned packages, allocated visible single GPU, condition/output directory, and for S completed F results. No provider rental or lifecycle API exists here. The earlier instance was destroyed; currently no compute allocation is asserted. A future actual GPU restore/forward is untested in this preparation and must not be represented as passed. No discarded gate cycle is introduced.

CPU preflight command (does not generate):
```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/run_s1000_test_inference_prepared_v1.py --workspace C:/thesis_proposal/yelp_dataset --preparation outputs/final_evaluation/s1000_preparation_v1
```

Actual inference and scoring commands are intentionally not executed or released by this package. Hashes/pins and CPU tests establish code/source preparation, not a completed GPU run. Final results must retain real runtime/memory/provider-cost evidence when compute is later approved; no estimates are represented as measured costs.
''')
    # Objective tests and real blank-reference rejection. No GPU imports or metric model forwards.
    testcmd=[sys.executable,'-X','utf8','-B',str(R/'scripts/test_s1000_final_evaluation_preparation_v1.py')]
    result=subprocess.run(testcmd,capture_output=True,text=True,encoding='utf8',cwd=R)
    prose(E/'CPU_REGRESSION_TESTS.txt',result.stdout+result.stderr);require(result.returncode==0,'CPU tests failed')
    from validate_s1000_human_references_v1 import validate_all
    refs,draft=validate_all(A,A/'TEST100_REFERENCE_TEMPLATE',False);_,final=validate_all(A,A/'TEST100_REFERENCE_TEMPLATE',True)
    require(not draft['errors'] and len(final['errors'])==100 and all(not r['review_consensus'] and not r['visual_evidence'] and not r['history'] for r in refs),'Blank human references were altered or incorrectly approved')
    write(E/'REFERENCE_VALIDATION_PREPARATION_RECEIPT.json',dict(draft_structure=draft,final_approval_check=dict(status=final['status'],expected_unapproved_businesses=len(final['errors'])),human_written=0,human_checked=0,blank_reference_fields_preserved=True))
    prep=subprocess.run([sys.executable,'-X','utf8','-B',str(R/'scripts/run_s1000_test_inference_prepared_v1.py'),'--workspace',str(R),'--preparation',str(O)],capture_output=True,text=True,encoding='utf8',cwd=R)
    prose(E/'INFERENCE_CPU_PREFLIGHT.txt',prep.stdout+prep.stderr);require(prep.returncode==0,'Inference CPU preflight failed')
    write(E/'CPU_CHECK_RECEIPT.json',dict(status='PASS_PREPARATION_ONLY',focused_tests=14,synthetic_only=True,actual_manifest_file_source_token_preflight=True,actual_GPU_restore_or_generation=False,BERT_forward=False,TEST_scores=False))
    prose(O/'READINESS_AND_BLOCKERS.md','''# S1000 final evaluation preparation readiness

Preparation is complete and frozen for researcher/ChatGPT review. It is NOT authorization to execute final evaluation.

| Item | Actual status |
| --- | --- |
| Frozen RO/U/F adapters, backbone, tokenizer | File hashes reverified on actual local files; no checkpoint changed |
| TEST100 | Complete100 accepted; no removal or TEST83 replacement |
| Shuffled | One-to-one98 changed Visual inputs;2 unchanged empty inputs;79 category matches/19 same-count fallbacks; source/token checks and full reversed-order repeat pass |
| Human references |100 blank templates,2817 selected reviews,1136 original-photo hashes/access;0 human-authored,0 independently checked |
| Automatic protocol | Metric/checkpoint/version/seed/preprocessing/statistics locked before outcomes; implemented scripts and14 focused CPU tests pass |
| Inference |400 records/700 logical components/598 prospective new calls; actual local CPU source/token/model-file preflight passes; no model or GPU forward |
| Human output study |30-business cohort/3 raters/4 conditions/360 blank rating sets; private key separated; no scoring |
| TEST predictions / TEST metrics / GPU work |0 /0 /0 |

Required next dependencies, in order:

1. Researcher and ChatGPT review/approve these exact protocols. In particular review the fixed512-token-prefix BERTScore limitation, common all-photo reference scope and structural F/S Review reuse. Do not alter methods after seeing TEST outcomes.
2. Recruit a real human author and a different independent human checker. Provide only the reference source package plus original-photo access. Preserve drafts/identity roster/evidence/versions. Complete and freeze all100 references; blank templates are expected to FAIL final approval checks today. No AI reference generation or fabricated approvals.
3. Separately release final inference/automatic evaluation and identify authorized compute. The previously used instance is destroyed; this preparation has not rented another. Install the pinned BERTScore dependency and obtain/verify its locked scorer weights (~1.426GB) in a local offline model directory. Small metric assets/wheel are already included; no scorer model was loaded or benchmarked.
4. Generate once with frozen DecoderPolicyV2, preserve raw artifacts, score automatically with paired analysis. Only afterward recruit/release the human-output scoring stage. External baseline execution and thesis results/defense writing remain separate tasks.

Material limitations remain: inherited original750 checkpoint histories and different U/F supervision/input pipelines; source-separated Review has no direct captions; two calls versus RO's one output budget; bounded prior model-assisted qualitative analysis is not independent human evaluation; preliminary mimicry screening is lexical; TEST visual content reuse is confirmed (17businesses; U66/F44 selected duplicate photos; U59/F42 own-TRAIN caption/support overlaps). No measured TEST inflation/memorization or unseen TEST superiority is claimed. Common full-photo gold may include information unavailable to every caption pipeline, so automatic Visual scores reflect both upstream bottlenecks and downstream summaries.

No final inference, final metric, human reference drafting, human scoring, training, Shuffled training, external-model run, caption regeneration or rental occurred. STOP after this preparation package.
''')
    # Replace only researcher-facing current narratives, with exact before snapshots and diffs.
    docs={
    'THESIS_PROGRESS.md':'''# Thesis progress for the frozen S1000 method

Updated2026-10-28. Training, validation selection, readiness checks and preliminary Teacher-mimicry audit are complete. Source-Separated LED-large S1000 is selected and frozen. The complete TEST100 is accepted for final evaluation. Final predictions, independent human references, automatic scores and human output scoring have NOT been executed by this preparation.

## Accepted method and completed evidence

The Yelp cohort has2000 TRAIN/200 VALIDATION/100 TEST businesses with disjoint business IDs. Frozen v4.2 review selection keeps at most30 whole reviews with star/time stratification, original ordering and4000 review-token allocation. RO/U/F/S share selected reviews. BLIP-2 COCO captions, U Basic selection and F relevance/redundancy/diversity filtering are complete; K<=5, total source5000 and incremental captions750 remain fixed. CLIP is soft-only and pHash diagnostic-only. No TEST tuning or caption regeneration.

RO/U/F use original LED-large and a shared Expanded-LoRA adapter per model (rank8, alpha16, dropout0,456 tensors). Review and Visual tasks are isolated; BOS-only attention and native DecoderPolicyV2 beam4/length1/max_length768 per call remain. RO has no Visual task. Each S1000 new phase adds1000 successful updates after its own RO750/U750/CE3-750 parent, with4000 Review presentations; U/F add4000 Visual presentations. RO uses four actual Review microbatches divided by8; U/F use4Review+4Visual divided by8. Caption exposure, optimizer/recovery histories and different auxiliary work are material provenance, not equal-compute guarantees. Exact hashes/full states/logs/AMP and archive receipts remain in the original stage_i roots and MODEL_AND_DECODER_LOCK.json. The old instance was destroyed with provider confirmation; no new GPU was rented.

VALIDATION200 has600 assembled outputs/1000 raw generations. Model-assisted bounded source reviews and lexical mimicry screening support development conclusions only; they are not independent human evaluation or exhaustive semantic adjudication. U/F differ in caption sets and Visual supervision. Review inputs are caption-free, so Review differences reflect trained adapters/history rather than direct visual conditioning.

## Final evaluation preparation

Current package: `outputs/final_evaluation/s1000_preparation_v1/`. Shuffled mapping is frozen at seed42 with one-to-one donors in exact caption-count strata:98 changed Visual inputs,2 unchanged empty cases,79 shared-category and19 declared same-count fallbacks. S uses F, preserves all Review inputs, and reuses exact F Review outputs; no S training. Two identical empty Visual calls also reuse F. The plan has400 assembled records,700 logical components and598 new calls when later released.

The human-reference package covers100 businesses/2817 selected reviews/all1136 original photos with source hashes. All reference fields and provenance remain blank: a human author and a different checker must supply/approve them. The common Visual reference is photo-grounded and not restricted to U/F captions. Metrics, paired statistics, seeds and scorer versions are prepared before TEST outputs. BERTScore uses an explicitly disclosed512-token prefix; full-text ROUGE and diagnostics are retained. Human output evaluation has a fixed30-business sample,3 raters, blinded/randomized presentation design and a five-point rubric; it runs AFTER automatic evaluation, not before.

## Scientific integrity and remaining order

Retain complete TEST100. Source image duplication is confirmed in the original HF snapshot:17businesses/123 TESTphoto IDs/490 pairs. U selects66 duplicate photos, F44;59U/42F have own-TRAIN caption/support overlap. No current mount/copy assignment error was found. Business independence remains; visual evidence is partly reused. This requires disclosure, not claims of proven memorization or inflated scores. No TEST83 replacement is adopted. Proposal/old logs/policies and earlier narrative versions remain preserved.

Next: approve preparation protocols -> independently construct/check/freeze100 human references -> authorize existing compute and finite final inference/Automatic Evaluation -> separately perform human output evaluation -> results/discussion. External MultimodalSum/direct multimodal baselines and generalization checks remain separate approved tasks; thesis writing, limitations, computational analysis and defense materials remain. No further model development is reopened. STOP for researcher/ChatGPT review.
''',
    'LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md':'''# Selected Source-Separated S1000 methodology and final evaluation plan

This filename remains the accepted-method entry point. Earlier text is preserved in exact before snapshots. Training and validation selection are complete; no new experiment is proposed. Model SHA identities and full source/decoder locks are in `outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json`.

RO-S1000 starts from RO750; U-S1000 from U750; F-S1000 from CE3-750. Each has1000 new successful updates. Each retains4000 new Review presentations; U/F have4000 new Visual presentations. Original caption exposure, different auxiliary gradients/clipping/work and retained recovery provenance remain disclosed. Source-separated Review receives original business metadata and selected reviews only; Visual receives only own captions. Native CE/FP16/AdamW, LR2e-5, warmup25/linear1000, rank8/alpha16/dropout0,456 tensors are frozen. RO's4Review microbatches divide by8 versus U/F4Review+4Visual divided by8; total compute differs.

Inference preserves original LED-large/backbone/tokenizer, BOS-only global attention, beam4, length penalty1, max_length768 per task, deterministic original EOS/forced-EOS/cache behavior and raw tokens. Source/review/incremental-caption limits5000/4000/750, K<=5; no silent truncation or reselection. Components concatenate verbatim; RO has one task and no Visual metric. Adapters/configs retain obsolete container paths only as provenance; restore with the explicit verified backbone.

The latest preparation request freezes a one-to-one Shuffled derangement of98 positive-caption TEST businesses within exact counts, prioritizing exact shared allowlisted cuisine/category globally. Seed42;79 category matches/19 declared same-count fallbacks. Two zero-F cases remain unchanged and intervention N/A. S reuses F checkpoint,100 identical F Review outputs and2 identical empty Visual outputs. Only98 S Visual calls are new. All100 recipients remain. The previous with-replacement proposal is historical and unchanged. No inference was run.

Proposal v16 illustrates joint-input inference; the accepted method isolates tasks, so F/S Review invariance is structural rather than learned resistance to distractor captions. Its provisional output/LR table is superseded by actual768-per-task/2e-5 settings; proposal itself is unchanged. U/F comparisons are pipeline comparisons because inputs, inherited adapter histories and Visual supervision differ.

Common independent human TEST references must be written from selected reviews and original photos, not Teacher targets or predictions, and checked by a second distinct human. Preparation provides blank fields and source access for all100; none is claimed human-authored yet. Automatic evaluation follows reference freeze; actual human output scoring follows automatic completion. Complete TEST100 is retained with the disclosed original-source visual reuse (U66/F44 selected duplicates; own-TRAIN U59/F42). No83 replacement or unseen-image-independence claim. New protocols and CPU tests are in `outputs/final_evaluation/s1000_preparation_v1/`; stop for researcher review before execution.
''',
    'CODEX_THESIS_IMPLEMENTATION_PLAN.md':'''# Implementation roadmap after S1000 model freeze

Training, validation selection, model/input readiness, preliminary mimicry screening and cross-split-image investigation are complete. Accepted checkpoints are RO/U/F-S1000; S reuses F without training. Preserve original logs/full states, recovery/initialization histories, frozen sources and numerical evidence. No new development experiment is authorized.

Preparation package: `outputs/final_evaluation/s1000_preparation_v1/`. Exact mapping, source/token/checkpoint/decoder locks,100 blank human-reference cases, objective validators, metric/statistical code,30-case human design and review ZIP are prepared. Shuffled seed42 gives98 actual Visual changes with one-to-one exact-count donors and2 unchanged empty cases. Review is byte/token-identical to F;100 Review and2 empty Visual outputs are explicitly reused when inference is later released.400 records/700 logical components/598 new calls. Full TEST100 stays intact.

The milestone order is fixed:

1. Researcher/ChatGPT review this preparation, including512-token-prefix BERTScore limitations and the common all-original-photo gold scope.
2. Real human author plus independent human checker write/check all100 common references. Preserve exact evidence IDs, versions, identities, blinding and approval. Automated validators check integrity only. No AI-written human gold.
3. Separately authorize available compute, verify runtime/assets and perform finite final RO/U/F/S inference and Automatic Evaluation with frozen metrics, business-paired bootstrap/sign-flip tests and Holm family. No automatic rental, selective generation or TEST tuning. RO Visual N/A.
4. After automatic completion, separately release three-rater human scoring on frozen30 businesses, randomized blinded aliases, proposal-v16 five-point criteria and ordinal agreement. Reference construction and human scoring are distinct activities.
5. Results/discussion, limitations, runtime/cost analysis and defense writing. External MultimodalSum/direct multimodal baselines and generalization work remain separate scope/compute decisions.

Disclose source-separated Review conditioning, different adapter ancestry/auxiliary training budgets, U/F supervision differences, bounded model-assisted development assessment and preliminary lexical mimicry limits. Original-photo reuse affects17TEST businesses; selected U66/F44 and own-TRAIN caption/support U59/F42. Keep all100 and disclose partial visual-evidence reuse; no TEST83 replacement and no proven performance inflation. Historical policy files/experiments remain preserved. Final references, TEST outputs/metrics and human ratings are still unexecuted. STOP for researcher review.
''',
    'CHATGPT_NEW_CONVERSATION_HANDOFF.md':'''# S1000 final evaluation preparation handoff

Current preparation: `outputs/final_evaluation/s1000_preparation_v1/`. Send `S1000_FINAL_EVALUATION_PREPARATION_REVIEW.zip` for protocol review. Training/validation selection and audits are complete. RO/U/F-S1000 are frozen; S uses F. Do not reopen training, generate TEST outputs/scores, write AI human references or start human scoring from this handoff.

Models: RO `2a2fbc2cd76176abb09a43615f12a8c43bfec83dad526798d0c5708333cdda2d`; U `41ecc5691dfda45dbfcac1508bb0dc2eedd0a6ccec76aa650a4b92bcbb6ef535`; F `6dcb08275e96f3972b15058f6abf370d83fa8ce05553214af94b5a0bd9c1c699`. Explicit backbone/tokenizer locks and source/task IDs are included; old base paths are not portable resolvers. Original full states/training/runtime/recovery evidence is retained off-instance. Instance58067102 was provider-confirmed destroyed; no new GPU is claimed or rented.

TEST100 is accepted unchanged. New mapping: one-to-one seed42 within exact caption counts,79 shared-category/19 same-count fallback donors,98 actual Visual changes,2 unchanged empty cases. All Review inputs identical to F; later S reuses100 F Review and2 empty F Visual outputs. Raw generation budget598 for400 assembled records,700 logical components. No model outputs exist from this preparation.

Reference materials contain100 blank templates,2817 exact selected reviews and1136 original photo IDs/paths/hashes, plus materialization instructions. No Teacher targets/predictions/model identity key enters the annotator source package. Real humans must write and independently check all100, preserving evidence/version/authorship/approval. Current authored/checked count0.

Approve protocols -> human reference construction/check/freeze -> finite inference and Automatic Evaluation -> separate30-business/three-rater human output evaluation -> results/discussion. Metric/statistical settings are pre-outcome locked; BERTScore uses the declared512-token prefix and full-text ROUGE remains. Scorer weights/dependency installation, actual human work and subsequent compute release are outstanding. CPU tests/file/source/token preflight passed; no GPU restore/forward or BERT scoring is claimed.

Disclose adapter ancestry, source-separated Review invariance, output-budget differences, U/F input-supervision pipeline differences and model-assisted prior evaluation limits. Cross-split image audit confirms original-source reuse for17businesses/123photo IDs/490pairs, U66/F44 selected duplicates and U59/F42 own-TRAIN caption/support overlap. This is partial visual-evidence reuse, not proven memorization or score inflation. Retain100; no83 replacement. History snapshots and original evidence are preserved. STOP for researcher/ChatGPT review.
'''}
    changes=[]
    for name,t in docs.items():
        path=R/'thesis_sdd_docs'/name;old=path.read_bytes();backup=O/'DOCUMENTATION_BEFORE'/name;backup.parent.mkdir(exist_ok=True);backup.write_bytes(old)
        path.write_text(t.strip()+'\n',encoding='utf8')
        diff=''.join(difflib.unified_diff(old.decode('utf8').splitlines(True),path.read_text(encoding='utf8').splitlines(True),fromfile=name+' before',tofile=name+' current'))
        prose(O/'DOCUMENTATION_DIFFS'/f'{name}.patch',diff)
        changes.append(dict(path='thesis_sdd_docs/'+name,before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=file_hash(path),snapshot=backup.relative_to(O).as_posix()))
    decision=R/'DECISIONS.md';before=decision.read_bytes();backup=O/'DOCUMENTATION_BEFORE/DECISIONS.md';backup.write_bytes(before)
    note='\n\n## S1000 final evaluation preparation — 2026-10-28\n\nLatest researcher request freezes complete TEST100 and one-to-one Shuffled preparation: seed42,98 actual changed Visual inputs,2 unchanged empty cases,79 shared-category and19 declared same-count fallbacks; old with-replacement policy remains historical. S reuses F and its identical Review outputs.100 human-reference templates remain blank; real author plus different checker are required. Automatic Evaluation follows reference freeze; human scoring follows automatic completion. No TEST inference/metric/human writing/scoring/training/rental occurred. Package: outputs/final_evaluation/s1000_preparation_v1/. All previous records preserved below/in original snapshots; retain the original-source visual-overlap limitation and no TEST83 replacement.\n'
    with decision.open('ab') as f:f.write(note.encode('utf8'))
    changes.append(dict(path='DECISIONS.md',before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=file_hash(decision),snapshot=backup.relative_to(O).as_posix(),old_bytes_retained_as_prefix=True))
    write(E/'DOCUMENTATION_CHANGE_MANIFEST.json',changes)
    prose(O/'THESIS_SDD_DOCS_UPDATE_SUMMARY.md','''# Documentation update summary

Four existing thesis_sdd_docs entry points now focus on the accepted S1000 method and current preparation: THESIS_PROGRESS, LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1, CODEX_THESIS_IMPLEMENTATION_PLAN and CHATGPT_NEW_CONVERSATION_HANDOFF. Root DECISIONS.md received an appended decision, retaining its previous bytes as an exact prefix. Exact before snapshots, SHA bindings and current diffs are retained locally in this preparation directory. No historical experiment log, frozen checkpoint/source/policy or proposal v16 file was modified.

Updates distinguish completed training/development/audits from blank independent references and pending final evaluation. They record accepted TEST100, actual one-to-one Shuffled mapping and F reuse, automatic-before-human-scoring order, human provenance requirements, metric/runtime dependencies and external-baseline/thesis-writing work. The obsolete main100/secondary83 execution wording and pending-with-replacement mapping wording were removed from the CURRENT narrative, with originals preserved. Source image reuse, condition histories and scientific limitations remain explicit. No discarded training technique history was reintroduced.
''')
    prose(O/'README.md','''# S1000 final evaluation preparation review

Start with READINESS_AND_BLOCKERS.md, then SHUFFLED_MAPPING_VALIDATION_REPORT.md, HUMAN_REFERENCE_GUIDELINES.md, AUTOMATIC_EVALUATION_PROTOCOL.md, HUMAN_EVALUATION_PROTOCOL.md and INFERENCE_PREPARATION.md. Mapping/cohort/model/decoder/metric locks bind the prospective four-condition evaluation. SCRIPTS are reproducible CPU preparation/validators and unexecuted future inference/scoring entries. EVIDENCE contains source/hash/proposal/CPU checks. ADMIN_ONLY contains the private30-case human presentation key and design-only blank rating form; do not distribute these before automatic completion or expose identities to raters.

TEST100_REFERENCE_ANNOTATION_PACKAGE is the ONLY source package intended for future reference annotators, after researcher approval. It has100 source cases, exact selected reviews, original-photo paths/hashes and blank templates/tracker. JPEG binaries and model weights are excluded; source-access instructions and the materializer provide a verified private portable-photo copy from existing local originals. It has no model predictions, Teacher targets, scoring judgments or model identities. All reference text fields are empty. Original human authorship/checking remains a dependency.

Complete TEST100 is retained with disclosed partial visual-evidence reuse. No TEST predictions/metrics, GPU work, human references/ratings, training, external baseline or paid resource was executed. The code/default modes and negative tests preserve this boundary. Current documentation is included with history manifests; full before snapshots remain local. PACKAGE_MANIFEST.csv binds every payload file. The outer ZIP checksum and reopen/extraction/every-member verification receipt are delivered alongside the ZIP. STOP for researcher/ChatGPT review.
''')
    # Copy scripts/protocol sources and create final hash manifest after all current-doc edits.
    scripts=['s1000_final_eval_core_v1.py','prepare_s1000_final_evaluation_v1.py','validate_s1000_human_references_v1.py','materialize_reference_photos_v1.py','snapshot_s1000_reference_version_v1.py','run_s1000_test_inference_prepared_v1.py','score_s1000_automatic_prepared_v1.py','analyze_s1000_human_ratings_prepared_v1.py','test_s1000_final_evaluation_preparation_v1.py','finalize_s1000_evaluation_preparation_v1.py']
    for name in scripts:ast.parse((R/'scripts'/name).read_text(encoding='utf8'));cp(R/'scripts'/name,O/'SCRIPTS'/name)
    # The annotator-only source bundle receives only its independent validators/materializer, never inference or model keys.
    for name in ['s1000_final_eval_core_v1.py','validate_s1000_human_references_v1.py','materialize_reference_photos_v1.py','snapshot_s1000_reference_version_v1.py']:cp(R/'scripts'/name,A/'TOOLS'/name)
    P=O/'REVIEW_PACKAGE';P.mkdir()
    for p in O.iterdir():
        if p.is_file():
            target=P/('ADMIN_ONLY' if p.name.startswith(('PRIVATE_','HUMAN_RATING_TEMPLATE')) else '')/p.name;cp(p,target)
    for sub in ['EVIDENCE','TEST100_REFERENCE_ANNOTATION_PACKAGE','METRIC_ASSETS','SCRIPTS']:
        for p in (O/sub).rglob('*'):
            if p.is_file():cp(p,P/sub/p.relative_to(O/sub))
    for c in changes:
        if c['path'].startswith('thesis_sdd_docs/'):cp(R/c['path'],P/'UPDATED_DOCUMENTATION'/c['path'])
    manifest=[dict(filename=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(P.rglob('*')) if p.is_file()]
    write_csv(P/'PACKAGE_MANIFEST.csv',manifest)
    Z=O/'S1000_FINAL_EVALUATION_PREPARATION_REVIEW.zip'
    with zipfile.ZipFile(Z,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P).as_posix())
    with zipfile.ZipFile(Z) as z:
        require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)+1,'ZIP member/CRC')
        with tempfile.TemporaryDirectory(prefix='s1000_prep_zipcheck_',dir=O) as tmp:
            t=Path(tmp);z.extractall(t)
            for f in manifest:require((t/f['filename']).stat().st_size==f['bytes'] and file_hash(t/f['filename'])==f['sha256'],'ZIP member SHA')
            for p in t.rglob('*.json'):read(p)
            rr=read(t/'TEST100_REFERENCE_ANNOTATION_PACKAGE/TEST100_REFERENCE_TEMPLATE.json');require(len(rr)==100 and all(not r['review_consensus'] and not r['visual_evidence'] for r in rr),'Reference fields not blank')
            require(len(list(csv.DictReader((t/'TEST100_SHUFFLED_MAPPING.csv').open(encoding='utf8'))))==100,'Mapping CSV frame')
    receipt=dict(status='PASS_PREPARATION_AND_PACKAGE_INTEGRITY',archive=Z.name,bytes=Z.stat().st_size,sha256=file_hash(Z),payload_files=len(manifest),ZIP_members=len(manifest)+1,
        reopened_extracted_all_member_SHA_and_JSON_checked=True,focused_CPU_tests=14,real_input_preflight='PASS_CPU_ONLY',
        prepared_counts=counts,Shuffled_validation=sv,reference_fields_empty=True,independent_human_approvals_created=0,
        actual_GPU_or_BERT_forward=False,TEST_predictions=0,TEST_quality_metrics=0,execution_authorized_now=False,
        current_documentation_updated=5,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(O/'PACKAGE_INTEGRITY_REPORT.json',receipt)
    prose(O/(Z.name+'.sha256'),receipt['sha256']+'  '+Z.name)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
