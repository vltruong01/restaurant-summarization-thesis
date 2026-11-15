"""Read-only export of completed thesis evidence into an executed portable notebook.

No model calls, dataset mutations, metric reruns or access to live annotation stores.
"""
from pathlib import Path
import ast
import csv
import hashlib
import html
import json
import re
import shutil
import sys
import textwrap
import time
import zipfile

import nbformat as nb
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/professor_notebook/20261103_v1'
PKG = OUT / 'THESIS_JUPYTER_NOTEBOOK_FOR_PROFESSOR'
M = ROOT / 'outputs/stage_m/final_test100_evaluation_review_20261101_v1/PACKAGE'
N = ROOT / 'outputs/stage_n/final_human_statistical_analysis_20261102_v1/PACKAGE'
F = ROOT / 'outputs/stage_f/production_caption_construction_d033_colab_v1'
V = ROOT / 'outputs/stage_m1/existing_teacher_validation200_v1'
INPUTS = {}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def track(p):
    p = Path(p)
    INPUTS[p.relative_to(ROOT).as_posix()] = sha(p)
    return p


def read(p):
    return json.loads(track(p).read_text(encoding='utf-8-sig'))


def dump(rel, value):
    p = PKG / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(p, rel):
    dest = PKG / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(track(p), dest)


def frame(p):
    return pd.read_csv(track(p))


def excerpt(name, path, functions=None, span=None):
    p = track(ROOT / path)
    s = p.read_text(encoding='utf-8-sig')
    lines = s.splitlines(keepends=True)
    spans = [span] if span else [(x.lineno, x.end_lineno) for x in ast.parse(s).body
                               if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in functions]
    assert spans
    body = '\n\n'.join(''.join(lines[a-1:b]).rstrip() for a, b in spans)
    (PKG / 'code' / (name + '.txt')).write_text(body + '\n', encoding='utf-8')
    return dict(file=path, sha256=sha(p), line_spans=spans, excerpt='code/' + name + '.txt')


def export_data():
    (PKG / 'code').mkdir(parents=True, exist_ok=True)
    (PKG / 'data').mkdir(exist_ok=True)
    copy(ROOT / 'data/processed/splits/business_split.csv', 'data/business_split.csv')
    copy(M / 'RESULTS_ORIGINAL/AGGREGATE_METRICS.csv', 'data/test100_aggregate.csv')
    copy(M / 'RESULTS_ORIGINAL/PER_BUSINESS_COMPONENT_METRICS.csv', 'data/test100_per_business.csv')
    dump('data/test100_paired.json', read(M / 'RESULTS_ORIGINAL/PAIRED_COMPARISONS.json'))
    copy(N / 'TABLES/PER_BUSINESS_CRITERION.csv', 'data/test30_per_business_criterion.csv')
    copy(N / 'TABLES/DESCRIPTIVE_STATISTICS.csv', 'data/test30_descriptive.csv')
    copy(N / 'TABLES/AGREEMENT.csv', 'data/test30_agreement.csv')
    dump('data/test30_paired.json', read(N / 'TABLES/PAIRED_COMPARISONS.json'))
    raw = read(N / 'DATA/VERIFIED_ANALYSIS_INPUT.json')['records']
    criteria = read(N / 'ANALYSIS_CONFIG.json')['criteria']
    clean = [{k: r[k] for k in ['business_id', 'condition', 'annotator'] + criteria} for r in raw]
    dump('data/test30_ratings_240.json', clean)
    val = frame(V / 'AGGREGATE_VALIDATION200_METRICS.csv')
    val[val.model.isin(['RO', 'U', 'F'])].to_csv(PKG / 'data/validation200_teacher_aggregate.csv', index=False)
    vp = frame(V / 'ALL_VALIDATION200_PER_BUSINESS_METRICS.csv')
    vp[vp.condition.isin(['RO', 'U', 'F'])].rename(columns={'condition':'model'}).to_csv(PKG / 'data/validation200_teacher_per_business.csv', index=False)
    target = read(V / 'TARGET_AND_SPLIT_AUDIT.json')
    for row in target['dataset_inventory']:
        assert sha(track(ROOT / row['file'])) == row['sha256']
    dump('data/teacher_provenance.json', {k: target[k] for k in ['records', 'unique_businesses', 'dataset_inventory', 'equivalence', 'completion_sha256', 'business_split_sha256', 'TEST_target_records_in_canonical_Stage_G']})
    copy(F / 'completion_report.json', 'data/caption_completion.json')
    ledger = [json.loads(x) for x in track(F / 'manifests/token_ledger.jsonl').read_text().splitlines()]
    cols = ['business_id', 'split', 'condition', 'review_block_tokens_with_special', 'caption_block_increment_tokens', 'full_source_tokens', 'review_budget_overflow', 'caption_budget_overflow', 'source_budget_overflow', 'source_sha256']
    budget = [{**{k: r[k] for k in cols}, 'n_reviews': len(r['selected_review_ids']), 'n_captions': len(r['selected_photo_ids'])} for r in ledger]
    pd.DataFrame(budget).to_csv(PKG / 'data/source_budget_ledger.csv', index=False)
    lock = read(ROOT / 'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    models = {c: {k: r[k] for k in ['adapter_sha256', 'full_state_sha256', 'initialization_adapter_sha256', 'successful_new_phase_updates', 'original_parent_updates', 'review_presentations_new_phase', 'visual_presentations_new_phase']} for c, r in lock['models'].items()}
    models['S'] = {**models['F'], 'note': 'Same adapter as F; fixed caption-source reassignment, no separate training.'}
    cfg = read(ROOT / 'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1/TRAINING_CONFIG_1000_V1.json')
    dump('data/model_and_method.json', dict(models=models, backbone={k: lock['backbone'][k] for k in ['sha256', 'bytes']}, generation=lock['generation'], adapter={k: lock['models']['F']['adapter_configuration'][k] for k in ['r', 'lora_alpha', 'lora_dropout', 'target_modules']}, F_optimization=cfg['optimization'], F_budgets=cfg['budgets']))
    copy(ROOT / 'configs/led_source_template_d032.json', 'data/source_template.json')
    method = read(ROOT / 'outputs/stage_f/methodology_amendment_d033/frozen_production_method.json')
    dump('data/caption_method.json', {k: method[k] for k in ['K', 'common_budget', 'image_duplicate_policy', 'pHash', 'CLIP', 'caption_thresholds', 'diversity', 'unfiltered', 'model_id', 'model_revision', 'generation', 'all_eligible_photos_before_K_selection']})
    selection = read(ROOT / 'outputs/stage_i/led_large_filtered_validation200_selection_v1/SELECTION_DECISION.json')
    dump('data/selection_decision.json', selection)
    track(ROOT / 'outputs/stage_i/led_large_filtered_validation200_selection_v1/RO_U_F_FAIRNESS_AUDIT.md')
    copy(M / 'FINAL_REFERENCES/EXPORT_MANIFEST.json', 'data/test100_reference_export_manifest.json')
    completion = read(M / 'RESULTS_ORIGINAL/COMPLETION.json')
    assert sha(M / 'FINAL_REFERENCES/EXPORT_MANIFEST.json') == completion['reference_files']['EXPORT_MANIFEST.json']
    freeze = read(M / 'FINAL_REFERENCES/REFERENCE_FREEZE.json')
    reference_checks=[]
    for p in sorted((M / 'FINAL_REFERENCES/ANNOTATIONS').glob('*.json')):
        ref=read(p)
        content={k:ref[k] for k in ['business_id','source_bundle_sha256','review_consensus','visual_evidence','review_claims','visual_claims']}
        h=hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
        assert ref['status']=='APPROVED' and h == ref['checker']['reviewed_content_sha256'] == freeze['content_hashes'][ref['business_id']]
        assert sha(p) == completion['reference_files']['ANNOTATIONS/'+p.name]
        reference_checks.append(dict(business_id=ref['business_id'],content_sha256=h,annotation_file_sha256=sha(p),status='APPROVED',both_components_present=bool(ref['review_consensus'] and ref['visual_evidence'])))
    assert len(reference_checks)==100
    dump('data/test100_reference_binding_check.json',reference_checks)
    dump('data/automatic_metric_receipt.json',{k:completion[k] for k in ['status','businesses','condition_records','components','runtime','score_cache','reference_validation']})
    track(ROOT / 'outputs/stage_k/s1000_test100_phase1_v1/TEST100_FINAL_INFERENCE_REPORT.md')
    copy(N / 'PROVENANCE/RUBRIC.json','data/human_rubric.json')
    copy(ROOT / 'configs/stage_f_validation_protocol.yaml','data/caption_scoring_protocol.yaml')
    sources = read(N / 'DATA/SOURCES_30.json')
    examples = []
    for bid, reason in [('-78Va3llKcPbL6SW4is8lw', 'First TEST business in frozen order; illustrates generic Review and caption-limited Visual output.'),
                        ('zGmK4tkN9X_wBi_9i1jIuw', 'Illustrative pub case; compare U/F/S source alignment without treating one example as aggregate proof.'),
                        ('ymE6ka6y6J-z2y2rX8QH3Q', 'Known generation-cap repetition case, retained as a failure example rather than silently repaired.')]:
        case = read(M / 'BUSINESS_CASES' / (bid + '.json'))
        caps = {}
        for cond in ['unfiltered', 'filtered']:
            rows = [json.loads(x) for x in track(F / f'manifests/test/{cond}_captions.jsonl').read_text().splitlines()]
            row = next(x for x in rows if x['business_id'] == bid)
            caps[cond] = [{k: c[k] for k in ['photo_id', 'normalized_caption']} for c in row['captions']]
        for condition, components in case['models'].items():
            for task, component in components.items():
                if component:
                    assert hashlib.sha256(component['raw_output_text'].encode()).hexdigest() == component['output_sha256']
        examples.append(dict(selection_reason=reason, case=case, exact_selected_reviews=sources[bid]['reviews'], captions=caps))
    dump('data/illustrative_cases.json', examples)
    # Input receipts for the material split limitation; kept as reporting evidence, never a new exclusion rule.
    copy(M / 'RESULTS_ORIGINAL/SENSITIVITY_ANALYSIS.json', 'data/test100_sensitivity_original.json')
    code = {}
    specs = [
        ('review_selection', 'scripts/budget_aware_review_selection_v4_2.py', ['signature', 'refine'], None),
        ('filter_ranking', 'scripts/stage_f_math.py', ['quality_order', 'redundancy_reduce', 'farthest_first'], None),
        ('filter_exact_duplicates', 'scripts/stage_f_d031_protocol.py', ['exact_duplicate_reduce'], None),
        ('filter_relevance', 'scripts/stage_f_d030_protocol.py', ['relevance_eligible', 'unfiltered_fixed_seed_order'], None),
        ('label_collation', 'scripts/stage_hi_led_data.py', ['LEDCompleteCollator'], None),
        ('native_training_step', 'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1/downloaded/RUN_TRAIN.py', None, (139, 160)),
        ('metric_scoring', 'scripts/stage_m_evaluation_v1.py', ['score_components'], None),
        ('paired_bootstrap', 'scripts/s1000_final_eval_core_v1.py', ['paired', 'holm'], None),
        ('quadratic_kappa', 'scripts/analyze_stage_n_two_raters_v1.py', ['kappa_quadratic'], None),
    ]
    for name, path, functions, span in specs:
        code[name] = excerpt(name, path, functions, span)
    dump('data/code_provenance.json', code)
    return criteria


def build_notebook():
    cells = []

    def md(s):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def py(s):
        cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))

    md('''
    # Thesis research overview
    ### Source-separated, image-grounded restaurant summarization on Yelp
    **Supervisor review edition · frozen research through 3 November 2026**

    This notebook connects the implemented pipeline to its frozen evidence. All tables, figures and
    examples below are saved in the notebook. **Run All performs only local, lightweight analysis of
    bundled exports**: it never trains, generates text, scores with a downloaded model, calls an API,
    or changes research records.

    **Main finding.** Filtered captions (F) reduce Visual surface repetition relative to Unfiltered (U)
    and substantially improve Visual Grounding over deliberately mismatched captions (S). The completed
    results **do not establish consistent superiority of F over U**. Review coverage remains a limitation.
    Teacher-target agreement on VALIDATION200 and human-reference agreement on TEST100 are distinct results.

    **Reading map:** 1 Design → 2 Data → 3 Reviews/budgets → 4 Captions/filtering → 5 Targets/training
    → 6 Frozen models/inference → 7 Validation → 8 TEST100 automatic → 9 TEST30 human → 10 Examples
    → 11 Findings/limitations → 12 Reproducibility.
    ''')
    py('''
    from pathlib import Path
    import hashlib, html, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, HTML, Code

    # Launch Jupyter from the extracted package directory; no original repository path is needed.
    BASE = Path.cwd()
    assert (BASE / 'data/test100_per_business.csv').is_file(), 'Open this notebook in the extracted package folder.'
    def load(name):
        return json.loads((BASE / 'data' / name).read_text(encoding='utf-8'))
    def table(name):
        return pd.read_csv(BASE / 'data' / name)
    def implementation(name):
        binding = load('code_provenance.json')[name]
        display(Markdown('**Original implementation:** `' + binding['file'] + '`; lines ' + str(binding['line_spans'])
                         + '. Source-file SHA-256: `' + binding['sha256'] + '`. Display only; not executed here.'))
        display(Code((BASE / binding['excerpt']).read_text(encoding='utf-8'), language='python'))
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    COLORS = {'RO':'#697586', 'U':'#d68b29', 'F':'#167a80', 'S':'#a95c89'}
    ORDER = ['RO', 'U', 'F', 'S']
    pd.set_option('display.max_columns', 16)
    pd.set_option('display.max_colwidth', 90)
    manifest = load('package_data_manifest.json')
    for item in manifest:
        p = BASE / item['file']
        assert p.stat().st_size == item['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['file']
    print(f'PASS: {len(manifest)} bundled evidence/code files verified by byte size and SHA-256.')
    ''')
    md('''
    ## 1. Research objectives and controlled comparisons
    The system produces two explicitly separated components: **Review Consensus**, supported only by
    selected reviews, and **Visual Evidence**, supported by image captions. The research asks whether
    captions add useful information, whether filtering improves that information, and whether correct
    business-to-caption alignment matters.

    | Condition | Review input | Visual input | Interpretation |
    |---|---|---|---|
    | RO | Frozen selected reviews + metadata | None | Review-only pipeline |
    | U | Same selected reviews + metadata | Basic-quality captions, deterministic selection | Unfiltered caption pipeline |
    | F | Same selected reviews + metadata | Relevance/redundancy/diversity-filtered captions | Selected Source-Separated V1-S1000 |
    | S | Exactly F Review output | F captions reassigned from another business | Alignment negative control; F adapter |

    Source separation is shared across final conditions. The historical Original CE3 used joint inputs;
    it is a parent and validation comparator, **not the final F checkpoint**. Pipeline differences include
    training ancestry and supervision, so this is not a perfectly isolated causal test of filtering alone.
    ''')
    md('''
    ## 2. Dataset preparation and frozen splits
    Restaurant eligibility requires at least ten reviews and five local photos, with accommodation
    businesses excluded. English-review eligibility is checked before final selection. Sorted eligible
    business IDs are sampled deterministically with seed 42; the frozen order allocates 2,000/200/100.
    Training, validation and TEST have disjoint **business IDs**. TEST is reserved for final evaluation;
    it is not used for checkpoint or filter selection.

    Business-disjoint splitting does not guarantee image-content independence. The later image audit
    found cross-split reuse affecting 17 TEST businesses; the primary evaluation retains all 100.
    ''')
    py('''
    splits = table('business_split.csv')
    assert len(splits) == splits.business_id.nunique() == 2300
    assert splits.groupby('split').size().to_dict() == {'test':100, 'train':2000, 'validation':200}
    split_counts = splits.groupby('split').agg(businesses=('business_id','size'),
                                             raw_reviews=('raw_review_records','sum'),
                                             photos=('local_photo_records','sum')).reindex(['train','validation','test'])
    display(split_counts)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(split_counts.index[::-1], split_counts.businesses[::-1], color=['#9a6eae','#d68b29','#167a80'])
    for i, value in enumerate(split_counts.businesses[::-1]):
        ax.text(value+15, i, str(value), va='center')
    ax.set(xlabel='Unique businesses', xlim=(0,2250), title='Frozen business-level split')
    plt.tight_layout(); plt.show()
    ''')
    md('''
    ## 3. Representative review selection and source budgets
    The accepted v4.2 selector keeps **whole English reviews**, with a maximum of 30 per business.
    Star-rating quotas and time strata preserve positive, mixed and negative experiences. A secondary
    source-length-distribution refinement swaps within star/time cells: it preserves review count and
    both histograms while respecting the budget. It does not simply choose the shortest reviews.

    Frozen allocations are **4,000 review tokens**, **750 incremental caption tokens** and **5,000
    complete-source tokens**. The caption increment includes formatting overhead, not just caption words.
    Selection/packing never silently cuts review or caption text. Token counts below are recorded with
    the pinned tokenizer; rerunning this notebook does not download or tokenize with a different model.
    ''')
    py('''
    budgets = table('source_budget_ledger.csv')
    assert len(budgets) == 6900 and not budgets.duplicated(['business_id','condition']).any()
    assert budgets.business_id.nunique() == 2300
    assert budgets.n_reviews.between(1,30).all()
    assert budgets.review_block_tokens_with_special.le(4000).all()
    assert budgets.caption_block_increment_tokens.le(750).all()
    assert budgets.full_source_tokens.le(5000).all()
    assert not budgets[['review_budget_overflow','caption_budget_overflow','source_budget_overflow']].any().any()
    display(budgets.groupby('condition').agg(rows=('business_id','size'),
        mean_reviews=('n_reviews','mean'), max_reviews=('n_reviews','max'),
        mean_caption_count=('n_captions','mean'), max_review_tokens=('review_block_tokens_with_special','max'),
        max_caption_increment=('caption_block_increment_tokens','max'), max_source_tokens=('full_source_tokens','max')).round(2))
    print('PASS: all 6,900 frozen condition/source records satisfy the original budgets.')
    implementation('review_selection')
    ''')
    md('''
    ## 4. Image captions and Filtered versus Unfiltered construction
    BLIP-2 (`Salesforce/blip2-opt-2.7b-coco`, pinned revision in the next cell) captions **all eligible
    photos before K selection**. Generation uses beam 3, no sampling, maximum 64 new tokens; a failed
    Basic-quality check permits one fixed-prompt retry. Basic checks are not factuality certification.

    **U:** Basic-passing pool → fixed-seed photo order → at most K captions.

    **F:** Basic-passing pool → exact normalized-pixel duplicate collapse → relevance margin ≥0.08
    → semantic redundancy reduction at 0.59 → diversity ordering → at most **K=5**.
    Broader scene grouping uses complete-link threshold 0.46. CLIP supplies a **soft quality ranking**,
    never a hard rejection threshold. pHash is diagnostic only. Captions themselves are not rewritten.
    K=5 is the smallest feasible choice meeting the validation-only 95%-of-best coverage rule.

    CLIP ViT-B/32 ranks normalized image–caption cosine agreement. SBERT `all-mpnet-base-v2`
    provides normalized, attention-mask-mean caption embeddings. Relevance is the maximum cosine to
    six dining-domain prototypes minus the maximum cosine to six unrelated-domain prototypes; it
    checks dining-domain relevance, not exact business identity or factual truth. Redundancy uses
    greedy representative cosine similarity; farthest-first diversity maximizes minimum distance
    (1 − cosine), with quality/photo-ID tie-breaking. The exact pinned scoring protocol is bundled.

    K is a ceiling, not a quota. Empty Filtered pools are valid and retained. F's selected photos need
    not be a subset of U's selected five, since both select from the complete pool differently.
    ''')
    py('''
    caption_method = load('caption_method.json')
    display(pd.Series({k: caption_method[k] for k in ['model_id','model_revision','K','CLIP','pHash','caption_thresholds','generation']}, name='Frozen setting').to_frame())
    caption_report = load('caption_completion.json')
    caption_rows = []
    for split, r in caption_report['split_reports'].items():
        caption_rows.append(dict(split=split, businesses=r['businesses'], generated=r['caption_pool_generated'],
            basic_pass=r['basic_pass'], U_selected=r['conditions']['unfiltered']['retained_captions']['total'],
            F_selected=r['conditions']['filtered']['retained_captions']['total'],
            empty_F=len(r['conditions']['filtered']['zero_caption_businesses'])))
    caption_counts = pd.DataFrame(caption_rows).set_index('split')
    assert caption_counts.generated.sum() == 30027 and caption_counts.F_selected.sum() == 7919
    assert caption_counts.U_selected.sum() == 11487 and caption_counts.empty_F.sum() == 12
    display(caption_counts)
    fig, axes = plt.subplots(1,2,figsize=(10,3.2))
    visual_budget = budgets[budgets.condition.isin(['unfiltered','filtered'])]
    for c, label, color in [('unfiltered','U',COLORS['U']),('filtered','F',COLORS['F'])]:
        counts = visual_budget[visual_budget.condition.eq(c)].n_captions.value_counts().reindex(range(6),fill_value=0)
        axes[0].plot(counts.index,counts.values,'o-',label=label,color=color)
        part = visual_budget[visual_budget.condition.eq(c)].groupby('split').n_captions.mean().reindex(['train','validation','test'])
        axes[1].plot(part.index,part.values,'o-',label=label,color=color)
    axes[0].set(xlabel='Retained captions per business',ylabel='Businesses',title='K≤5; no forced filling')
    axes[1].set(ylabel='Mean captions',ylim=(0,5.4),title='Caption counts by split')
    for ax in axes: ax.legend()
    plt.tight_layout(); plt.show()
    ''')
    py('''
    # Source excerpts are exact, including deterministic tie-breaking.
    implementation('filter_relevance')
    implementation('filter_exact_duplicates')
    implementation('filter_ranking')
    ''')
    md('''
    ## 5. Teacher supervision and LED-large Expanded LoRA
    The accepted Teacher corpus contains **6,600 condition records** for 2,000 TRAIN and 200 VALIDATION
    businesses (RO/U/F). Review targets are byte-identical across conditions. U and F Visual targets
    are condition-specific and differ for all 2,200 businesses. No canonical Stage G TEST Teacher
    targets exist. The provider model alias is recorded; an immutable backend snapshot is not attested.
    Automated target acceptance does not prove every Teacher claim factual.

    The final summarizer uses `allenai/led-large-16384`, one **shared Expanded LoRA** adapter per model
    (rank 8, alpha 16, dropout 0; 456 trainable tensors / 4,915,200 parameters). Review-only and
    caption-only task views independently tokenize their original component labels; padding is masked
    with −100 and target EOS is preserved. BOS is the only globally attended source position.

    F continues the original CE3 step-750 adapter for **1,000 successful updates**: two passes through
    TRAIN2000, 4 Review + 4 Visual microbatches per update, physical batch 1 / accumulation 8,
    native cross-entropy, AdamW, FP16 autocast, LR 2e−5, warmup 25 and linear horizon 1,000.
    Review and Visual each receive 4,000 successful presentations. Final inference uses the original
    V1 targets, not a later relabeling experiment.
    ''')
    py('''
    teacher = load('teacher_provenance.json')
    display(pd.DataFrame(teacher['equivalence']))
    assert teacher['records'] == 6600 and teacher['TEST_target_records_in_canonical_Stage_G'] == 0
    method = load('model_and_method.json')
    display(pd.Series(method['adapter'], name='Shared Expanded LoRA').to_frame())
    display(pd.Series({k: method['F_optimization'][k] for k in ['physical_batch','accumulation','successful_updates','per_window','learning_rate','warmup_steps','scheduler','optimizer','precision']}, name='F training contract').to_frame())
    implementation('label_collation')
    implementation('native_training_step')
    ''')
    md('''
    ## 6. Final frozen selection and TEST100 inference
    F is **Source-Separated V1-S1000**, selected on VALIDATION before TEST comparison, using full-cohort
    descriptive diagnostics and a bounded source-grounded qualitative assessment. It was not selected
    by training loss or retrospectively chosen from TEST scores. The final RO and U models also have
    1,000 new successful updates, but start from their own step-750 parents; F starts from CE3.

    RO receives 4 Review microbatches per update divided by 8; U/F receive 4 Review + 4 Visual divided
    by 8. Thus matching update counts does not equalize compute or effective loss composition. These
    ancestries and supervision histories limit a pure input-only causal interpretation. RO follows its
    Review-only RO750 parent with no experiment-specific caption fine-tuning; it is not initialized
    from caption-trained CE3. Public pretraining exposure is not exhaustively audited.

    Review pass: metadata + selected reviews, no captions → `[REVIEW_CONSENSUS]`.
    Visual pass: captions only, no reviews or business metadata → `[VISUAL_EVIDENCE]`.
    The sections are combined verbatim. Frozen decoding: beam 4, length penalty 1, no sampling,
    max sequence length **768 per call**, no reranking or output repair. This is a per-component budget,
    not the same total output budget as historical joint-input CE3.

    S uses F weights and a frozen one-to-one caption reassignment. F/S Review is identical; S is not
    a separately trained model. The two zero-Filtered-caption businesses have no Visual intervention.
    All 400 condition records remain in TEST; there are 700 applicable component records (RO has no Visual).
    ''')
    py('''
    model_rows = []
    for c, r in method['models'].items():
        model_rows.append(dict(condition=c, new_updates=r['successful_new_phase_updates'], parent_updates=r['original_parent_updates'],
            Review_presentations=r['review_presentations_new_phase'], Visual_presentations=r['visual_presentations_new_phase'],
            adapter_sha256=r['adapter_sha256'], parent_adapter_sha256=r['initialization_adapter_sha256']))
    display(pd.DataFrame(model_rows).set_index('condition'))
    display(Markdown('**Frozen backbone SHA-256:** `' + method['backbone']['sha256'] + '`. These are existing freeze receipts; this small package does not contain or rehash the multi-GB weights.'))
    ''')
    md('''
    ## 7. VALIDATION200: agreement with existing Teacher targets
    This track measures model–Teacher agreement, not independent human quality. Review uses the common
    identical target for RO/U/F. Visual uses U-specific targets for U and F-specific targets for F;
    **their Visual scores cannot establish which pipeline is better grounded**. Each row includes all
    200 businesses. The compact bundle contains the corresponding per-business frozen metrics.
    ''')
    py('''
    metric_columns = ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']
    validation = table('validation200_teacher_aggregate.csv')
    validation_per = table('validation200_teacher_per_business.csv')
    assert validation_per.business_id.nunique() == 200 and len(validation_per) == 1000
    assert set(validation_per.business_id) == set(splits.loc[splits.split.eq('validation'),'business_id'])
    for _, r in validation.iterrows():
        x = validation_per[validation_per.model.eq(r.model) & validation_per.task.eq(r.task)]
        assert len(x) == 200
        for metric in metric_columns:
            assert np.isclose(x[metric].mean(),r[metric],atol=1e-12,rtol=0)
    display(validation[['model','task','n','reference_scope']+metric_columns].round(4))
    print('PASS: all VALIDATION200 aggregate metrics reproduced from 1,000 applicable component rows.')
    ''')
    md('''
    ## 8. TEST100: automatic evaluation against frozen human references
    All applicable models use the **same approved reference version for each TEST business**. RO is
    evaluated only on Review. References were authored/checked in the recorded workflow with disclosed
    AI-assisted feedback, evidence suggestions and minor editorial help; unaided authorship is not
    independently authenticated. Reference agreement is useful but is not direct factuality proof.

    **Metric contract:** ROUGE-1/2/L F1 (`rouge-score` 0.1.2, Porter stemming); BERTScore 0.3.13,
    `roberta-large` revision `722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17,
    slow tokenizer, no IDF or baseline rescaling, CPU FP32, batch 8. Only the exact leading component
    header is removed. BERTScore scores the 512-token prefix including boundary tokens; one F Review
    output (also reused by S) is truncated for that metric, while ROUGE uses full text. An empty
    candidate or reference receives zero; all rows remain. No summary rewriting occurs.
    ''')
    py('''
    test = table('test100_per_business.csv')
    test_aggregate = table('test100_aggregate.csv')
    assert len(test) == 700 and test.business_id.nunique() == 100
    assert not test.duplicated(['business_id','condition','task']).any()
    assert set(test.business_id) == set(splits.loc[splits.split.eq('test'),'business_id'])
    assert test.business_id.eq(test.reference_business_id).all()
    assert test.groupby('business_id').reference_sha256.nunique().eq(1).all()
    reference_checks = pd.DataFrame(load('test100_reference_binding_check.json')).set_index('business_id')
    assert len(reference_checks) == 100 and reference_checks.both_components_present.all()
    assert test.apply(lambda r:r.reference_sha256 == reference_checks.loc[r.business_id,'content_sha256'],axis=1).all()
    assert len(test[test.condition.eq('RO') & test.task.eq('visual')]) == 0
    for _, r in test_aggregate.iterrows():
        part = test[test.condition.eq(r.condition) & test.task.eq(r.task)]
        assert len(part) == 100
        for metric in metric_columns:
            assert np.isclose(part[metric].mean(),r[metric],atol=1e-12,rtol=0)
    for task in ['review','visual']:
        display(Markdown('### ' + task.title() + ' · TEST100 human-reference scores (0–1)'))
        display(test_aggregate[test_aggregate.task.eq(task)][['condition','n']+metric_columns+['empty_candidates','candidate_prefix_truncated']].round(4))
    print('PASS: all 42 displayed aggregate metric values reproduced; same reference hash per business.')
    ''')
    md('''
    ### Paired statistical reading
    Automatic primary inference comprises four prespecified BERTScore-F1 tests. The original plan uses
    20,000 paired-business bootstrap draws for pointwise percentile 95% CIs, 50,000 two-sided sign flips
    with +1 correction, seed 20261028, and Holm correction across those four tests. Other metrics have
    descriptive paired CIs. No business is excluded. Pointwise CIs and Holm-adjusted tests are different
    procedures; do not select whichever gives a preferred conclusion.
    ''')
    py('''
    auto_paired = pd.DataFrame(load('test100_paired.json'))
    primary = auto_paired[auto_paired.metric.eq('bertscore_f1')].copy()
    primary['CI_low'] = primary.CI95.map(lambda x:x[0]); primary['CI_high'] = primary.CI95.map(lambda x:x[1])
    for _, r in primary.iterrows():
        pivot = test[test.task.eq(r.task)].pivot(index='business_id',columns='condition',values=r.metric)
        comp = r.contrast.split('-')[1]
        assert np.isclose((pivot.F-pivot[comp]).mean(),r.mean_difference,atol=1e-12,rtol=0)
    display(primary[['task','contrast','mean_difference','CI_low','CI_high','signflip_two_sided_p','holm_p_four_prespecified_tests']].round(6))
    fig, axes = plt.subplots(1,2,figsize=(10,3.2))
    for ax, task in zip(axes,['review','visual']):
        part = test_aggregate[test_aggregate.task.eq(task)]
        ax.bar(part.condition,part.bertscore_f1,color=[COLORS[c] for c in part.condition])
        ax.set(title=task.title()+' · reference agreement',ylabel='BERTScore F1',ylim=(0,1))
    plt.tight_layout(); plt.show()
    ''')
    py('''
    surface = test.groupby(['condition','task'])[['words','repeated_3gram_fraction','duplicate_sentence_fraction']].mean()
    display(surface.round(6))
    fig, ax = plt.subplots(figsize=(6.5,3))
    v = surface.xs('visual',level='task').reindex(['U','F','S'])
    ax.bar(v.index,v.repeated_3gram_fraction,color=[COLORS[c] for c in v.index])
    ax.set(ylabel='Mean repeated-trigram fraction',title='Visual surface repetition · all 100 TEST businesses',ylim=(0,.16))
    plt.tight_layout(); plt.show()
    display(Markdown('F has less **Visual** repetition than U. This is a surface diagnostic, not proof of factuality. '
        'F/S have the same Visual output multiset under permutation, hence the same global repetition; recipient grounding differs. '
        'The Review repetition table does not support a blanket claim that F reduces repetition in every component.'))
    implementation('metric_scoring')
    ''')
    md('''
    ## 9. TEST30 human evaluation: two evaluators, 240 records
    The frozen subset has 30 businesses × 4 outputs × 2 evaluators. Ratings are 1–5, higher better.
    Review Faithfulness, Aspect Coverage and Usefulness concern Review; Coherence and Non-Redundancy
    concern all available components; Visual Grounding and Contribution concerns Visual and is
    structurally N/A for RO. Do not turn N/A into zero or aggregate unrelated criteria into a new score.

    For inference, both raters are averaged **within business** before comparing conditions: n=30
    business pairs, not 240 independent observations. The accepted human analysis is exploratory:
    20,000 bootstrap draws, 50,000 sign flips, common business resampling, seed 20261028, pointwise
    95% CIs, **no prespecified multiplicity adjustment**. Raw p-values are not a confirmatory winner test.
    ''')
    py('''
    ratings = pd.DataFrame(load('test30_ratings_240.json'))
    human_per = table('test30_per_business_criterion.csv')
    human_desc = table('test30_descriptive.csv')
    criteria = ['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
    # Preserve structural 'N/A' in the raw export; convert only in this in-memory numeric view.
    assert ratings.loc[ratings.condition.eq('RO'),'visual_grounding_and_contribution'].eq('N/A').all()
    ratings[criteria] = ratings[criteria].where(ratings[criteria].ne('N/A')).astype(float)
    assert len(ratings) == 240 and ratings.business_id.nunique() == 30
    assert not ratings.duplicated(['business_id','condition','annotator']).any()
    assert ratings.groupby(['annotator','condition']).size().eq(30).all()
    assert set(ratings.business_id) <= set(test.business_id)
    assert ratings.loc[ratings.condition.eq('RO'),'visual_grounding_and_contribution'].isna().all()
    observed = ratings.groupby('condition')[criteria].mean().reindex(ORDER)
    for _, r in human_desc[human_desc.panel.eq('COMBINED')].iterrows():
        actual = observed.loc[r.condition,r.criterion]
        assert (pd.isna(actual) and pd.isna(r['mean'])) or np.isclose(actual,r['mean'],atol=1e-12,rtol=0)
    for _, r in human_per.iterrows():
        pair = ratings[ratings.business_id.eq(r.business_id) & ratings.condition.eq(r.condition)][r.criterion]
        assert len(pair) == 2 and np.isclose(pair.mean(),r.two_rater_average,atol=1e-12,rtol=0)
    display(observed.T.round(3).rename_axis('Criterion'))
    fig, ax = plt.subplots(figsize=(10,4.5))
    im = ax.imshow(observed.T, vmin=1,vmax=5,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(4),ORDER)
    ax.set_yticks(range(6),['Review faithfulness','Aspect coverage','Coherence','Non-redundancy','Review usefulness','Visual grounding / contribution'])
    for i in range(6):
        for j in range(4):
            x=observed.T.iloc[i,j]; ax.text(j,i,'N/A' if pd.isna(x) else f'{x:.2f}',ha='center',va='center',color='white' if x>3.6 else '#17212b')
    fig.colorbar(im,ax=ax,label='Mean human rating (1–5)'); ax.set_title('TEST30 · unchanged ratings from two evaluators')
    plt.tight_layout(); plt.show()
    print('PASS: 240 records, 690 applicable paired-business criterion rows and all combined means agree.')
    ''')
    py('''
    human_paired = pd.DataFrame(load('test30_paired.json'))
    for _, r in human_paired.iterrows():
        p = human_per[human_per.criterion.eq(r.criterion)].pivot(index='business_id',columns='condition',values='two_rater_average')
        assert np.isclose((p.F-p[r.contrast.split('-')[1]]).mean(),r.mean_paired_difference,atol=1e-12,rtol=0)
    display(human_paired[['criterion','contrast','n_businesses','mean_paired_difference','CI95_low','CI95_high','unadjusted_exploratory_signflip_p']].round(6))
    part = human_paired[human_paired.contrast.isin(['F-U','F-S'])].copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9,6))
    for y, (_,r) in enumerate(part.iterrows()):
        ax.plot([r.CI95_low,r.CI95_high],[y,y],color=COLORS[r.contrast.split('-')[1]],lw=2)
        ax.scatter(r.mean_paired_difference,y,color=COLORS['F'],s=30,zorder=3)
    ax.axvline(0,color='#555',ls='--',lw=1)
    ax.set_yticks(range(len(part)),[r.criterion.replace('_',' ')+' · '+r.contrast for _,r in part.iterrows()])
    ax.set(xlabel='F minus comparator · 1–5 rating points (pointwise 95% CI)',title='Exploratory paired human comparisons · n=30 businesses')
    plt.tight_layout(); plt.show()
    ''')
    md('''
    ### Agreement and measurement limitations
    Fixed 1–5 quadratic-weighted Cohen kappa is reported per condition and criterion (30 pairs).
    Pooled agreement is supplementary because mixing easy and hard conditions can inflate it. There
    are only two evaluators; these intervals do not generalize over a sampled population of raters.
    F and S reuse exactly the same Review generation. Six of 180 Review-only repeated-rating pairs
    differ, illustrating rating/context variability rather than a Review model improvement.
    Recorded roles and participant declarations do not independently authenticate identity, source
    inspection or independence. Accepted ratings are retained without manufacturing corrections.
    ''')
    py('''
    implementation('quadratic_kappa')
    # Execute only this self-contained, original statistics helper, not its live-application imports.
    from collections import Counter
    scope = {'Counter': Counter}
    exec((BASE / 'code/quadratic_kappa.txt').read_text(encoding='utf-8'),scope)
    agreement = table('test30_agreement.csv')
    display(agreement.head(10))
    for condition in ORDER:
        for criterion in criteria:
            if condition == 'RO' and criterion == 'visual_grounding_and_contribution': continue
            pairs = ratings[ratings.condition.eq(condition)].pivot(index='business_id',columns='annotator',values=criterion)
            k = scope['kappa_quadratic'](pairs.A01.astype(int).tolist(), pairs.A02.astype(int).tolist())['value']
            original = agreement[agreement.condition.eq(condition) & agreement.criterion.eq(criterion)].iloc[0].qwk
            assert (k is None and pd.isna(original)) or np.isclose(k,original,atol=1e-12,rtol=0)
    print('PASS: all 23 applicable kappa panels reproduce the original agreement table.')
    ''')
    md('''
    ## 10. Actual generated summaries: deliberately illustrative, not a second evaluation sample
    Three examples are selected explicitly: the first frozen TEST case, a pub case, and the known
    generation-cap failure. These are not a random or exhaustive qualitative assessment. Open each
    disclosure to inspect **unmodified complete outputs**, reference text, and exact selected review
    records/captions. Long failures remain long; they are not shortened or repaired. The original
    source/checkpoint/reference/output hashes are bundled beside the examples. Caption grounding and
    photographic grounding can differ; this compact package does not include original photo archives.
    ''')
    py('''
    cases = load('illustrative_cases.json')
    for example in cases:
        case=example['case']
        display(Markdown('### '+case['business_name']+' · `'+case['business_id']+'`' + chr(10) + example['selection_reason']))
        def detail(label, text, opened=False):
            return ('<details open>' if opened else '<details>')+'<summary>'+html.escape(label)+'</summary><pre style="white-space:pre-wrap;max-height:400px;overflow:auto">'+html.escape(text)+'</pre></details>'
        sections=[]
        for c in ORDER:
            for task in ['review','visual']:
                item=case['models'][c][task]
                if item:
                    assert hashlib.sha256(item['raw_output_text'].encode()).hexdigest() == item['output_sha256']
                    sections.append(detail(c+' · '+task+' · original output',item['raw_output_text'],c=='F' and case['business_id']!='ymE6ka6y6J-z2y2rX8QH3Q'))
        for condition,captions in example['captions'].items():
            sections.append(detail(condition+' · exact BLIP-2 captions',json.dumps(captions,ensure_ascii=False,indent=2)))
        sections.append(detail('Exact selected review records',json.dumps(example['exact_selected_reviews'],ensure_ascii=False,indent=2)))
        sections.append(detail('Frozen final reference (comparison aid, not unquestionable truth)',json.dumps(case['reference'],ensure_ascii=False,indent=2)))
        display(HTML(''.join(sections)))
    ''')
    md('''
    ## 11. Findings, research questions and limits

    **Caption alignment matters.** F–S Visual BERTScore-F1 is +0.00785, pointwise 95% CI
    [0.00587, 0.00983], Holm-adjusted p≈0.000080. Human Visual Grounding/Contribution is 3.900 versus
    1.233: paired difference +2.667 points, 95% CI [2.300, 3.017], exploratory raw p≈0.000020.
    This supports useful business-aligned Visual information compared with mismatched captions. It is
    not evidence that every Visual claim is correct, or a test of raw-image understanding by LED.

    **Filtering reduces surface repetition, but does not consistently beat U.** Visual repeated-trigram
    fraction drops from approximately 0.1230 (U) to 0.0605 (F). F–U Visual BERTScore-F1 is −0.01511,
    CI [−0.04133, 0.00277], Holm p≈0.8134; human Visual difference is −0.100,
    CI [−0.383, 0.200]. Human F–U Non-Redundancy is +0.250, CI [−0.117, 0.633].
    Less repetition alone does not prove improved grounding or useful coverage.

    **Review remains a weakness.** Human F Review faithfulness averages 4.033 versus U 4.233;
    F–U Aspect Coverage is −0.300, CI [−0.500, −0.133], and Usefulness −0.233,
    CI [−0.417, −0.067]. These are exploratory paired findings. The F–RO Usefulness bootstrap CI
    excludes zero while its raw sign-flip p≈0.078; both must be reported rather than selecting one.
    Final F was frozen on validation and is not reselected using TEST.

    **Limits governing interpretation:**
    - Business splits are disjoint, but image reuse affects 17 TEST businesses. No primary exclusions
      were made. This is a contamination risk, not demonstrated memorization or measured score inflation.
    - Model ancestry, target supervision, task exposure/loss composition and historical joint versus
      separated output budgets are confounders. Caption filtering alone is not isolated causally.
    - Validation Teacher similarity is not independent factuality evidence; U/F Visual references differ.
      TEST references have disclosed assistance and recorded approval, not independently certified authorship.
    - TEST30 is small, model-assisted auditing does not replace the two recorded human evaluators,
      and known annotation uncertainty remains. Human inference is exploratory and unadjusted.
    - Fixed reassignment creates dependency in S; shared F/S Review is not independent generation.
      Empty Visual outputs and a repeated generation-cap failure remain in all primary results.
    - ROUGE/BERTScore reward reference overlap, not source truth; BERTScore uses a prefix. Review and
      Visual scores should not be collapsed into an invented overall quality score.
    ''')
    md('''
    ## 12. Reproducibility and evidence boundaries
    The notebook recomputes split/budget summaries, all displayed mean metrics and human means from
    bundled frozen records, verifies paired point estimates and runs the original agreement helper.
    Original confidence intervals and tests are exported verbatim; the lightweight numerical checker
    validates the tables without rerunning model-based scoring. Every bundled input and source-code
    excerpt has a SHA-256 binding. `data/original_artifact_bindings.json` maps each export to its original
    repository artifact. Excerpts preserve original code; surrounding imports/helpers and heavy weights
    remain in the repository.

    Reproducing BLIP-2, Teacher production, LED training/inference or BERTScore itself requires the
    original data, pinned runtimes and model assets (and, where applicable, separate compute/API
    authorization). None is required or initiated by this notebook. The summary verification is not
    a fresh claim to have rehashed all model weights or independently authenticated human activity.
    ''')
    py('''
    summary = {'split_businesses':len(splits), 'source_records':len(budgets),
               'TEST100_metric_rows':len(test), 'TEST30_rating_records':len(ratings),
               'TEST30_businesses':ratings.business_id.nunique(),
               'automatic_primary_contrasts':len(primary), 'human_exploratory_contrasts':len(human_paired),
               'new_model_calls':0, 'frozen_research_records_modified':0}
    display(pd.Series(summary,name='Validated summary').to_frame())
    print('Notebook analysis complete. No model training, inference, API calls or new research experiments were run.')
    ''')
    book = nb.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3 (portable CPU analysis)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    nb.validate(book)
    return book


def render_html(book, path):
    # Offline review companion; images and outputs already embedded, no external JS/fonts.
    def inline(s):
        s = html.escape(s)
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    def markdown(s):
        result=[]
        for block in s.split('\n\n'):
            lines=block.splitlines()
            if not lines: continue
            if lines[0].startswith('|'):
                rows=[line.strip('|').split('|') for line in lines]
                result.append('<table><thead><tr>'+''.join('<th>'+inline(x.strip())+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in row)+'</tr>' for row in rows[2:])+'</tbody></table>')
            elif lines[0].startswith('#'):
                for line in lines:
                    match=re.match(r'^(#{1,6}) (.*)',line)
                    result.append(f'<h{len(match[1])}>'+inline(match[2])+f'</h{len(match[1])}>' if match else '<p>'+inline(line)+'</p>')
            elif lines[0].startswith('- '):
                items=[]
                for line in lines:
                    if line.startswith('- '): items.append(line[2:])
                    elif items: items[-1]+=' '+line.strip()
                result.append('<ul>'+''.join('<li>'+inline(x)+'</li>' for x in items)+'</ul>')
            else: result.append('<p>'+inline(' '.join(lines))+'</p>')
        return ''.join(result)
    import mistune
    markdown = mistune.create_markdown(escape=False, plugins=['table'])
    body=[]
    for cell in book.cells:
        if cell.cell_type == 'markdown':
            body.append(markdown(cell.source))
        else:
            body.append('<details class="code"><summary>Python implementation / analysis cell</summary><pre>'+html.escape(cell.source)+'</pre></details>')
            for output in cell.get('outputs',[]):
                if output.output_type == 'stream': body.append('<pre>'+html.escape(output.text)+'</pre>')
                data=output.get('data',{})
                if 'image/png' in data: body.append('<img class="figure" src="data:image/png;base64,'+data['image/png']+'">')
                elif 'text/html' in data: body.append(data['text/html'])
                elif 'text/markdown' in data: body.append(markdown(data['text/markdown']))
                elif 'text/plain' in data: body.append('<pre>'+html.escape(data['text/plain'])+'</pre>')
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thesis research overview</title><style>body{max-width:1120px;margin:48px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17212b;background:#fff}h1,h2,h3{line-height:1.25;color:#155d66}h2{margin-top:60px;border-top:2px solid #dce6e8;padding-top:24px}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:20px 0;max-width:100%}td,th{padding:7px 10px;border:1px solid #dce6e8;text-align:left}th{background:#edf5f5}pre{font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:14px}details{padding:9px;margin:10px 0;border:1px solid #dce6e8;border-radius:5px}summary{cursor:pointer;color:#155d66}.figure{max-width:100%;height:auto}.code{background:#f8fafb}code{font-size:.9em}a{color:#167a80}</style><body>'+''.join(body)+'</body></html>',encoding='utf-8')


def main():
    start = time.monotonic()
    PKG.mkdir(parents=True, exist_ok=True)
    export_data()
    dump('data/original_artifact_bindings.json', [{'repository_file':p,'sha256':h} for p,h in sorted(INPUTS.items())])
    data_manifest = [{'file':p.relative_to(PKG).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
                     for folder in ['data','code'] for p in sorted((PKG/folder).rglob('*')) if p.is_file() and p.name!='package_data_manifest.json']
    dump('data/package_data_manifest.json', data_manifest)
    notebook = build_notebook()
    path = PKG / 'THESIS_RESEARCH_OVERVIEW.ipynb'
    nb.write(notebook,path)
    client = NotebookClient(notebook,timeout=180,kernel_name='thesis-caption-pilot',resources={'metadata':{'path':str(PKG)}})
    client.execute()
    # Portable metadata, without a machine-specific kernel name.
    notebook.metadata.kernelspec={'display_name':'Python 3 (portable CPU analysis)','language':'python','name':'python3'}
    nb.validate(notebook)
    nb.write(notebook,path)
    render_html(notebook,PKG/'THESIS_RESEARCH_OVERVIEW.html')
    original_preserved=all(sha(ROOT/p)==h for p,h in INPUTS.items())
    assert original_preserved
    report={'status':'PASS','original_artifacts_rehashed_and_unchanged':len(INPUTS),
            'evidence_and_code_files':len(data_manifest),'code_cells_executed':sum(c.cell_type=='code' for c in notebook.cells),
            'code_cells_with_errors':sum(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs),
            'figures_embedded':sum('image/png' in o.get('data',{}) for c in notebook.cells if c.cell_type=='code' for o in c.outputs),
            'execution_seconds':round(time.monotonic()-start,2),'new_model_calls':0,
            'verification_scope':'Recomputed frozen descriptive analyses; original paired inferential receipts retained; no new model-based scoring or independent authorship authentication.'}
    assert report['code_cells_with_errors']==0
    dump('VALIDATION_REPORT.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
