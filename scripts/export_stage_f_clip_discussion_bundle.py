"""Export current corrected Stage F evidence for user/ChatGPT discussion; no scientific mutations."""
from pathlib import Path, PurePosixPath
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from stage_f_verification_workflow import read, write, sha
from analyze_stage_f_consistency_stability import validate_declared_prefix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/consistency_conservative_gate_diagnostic_v1'
ASSISTED=ROOT/'outputs/stage_f/ai_assisted_calibration_v1'
BASE=ROOT/'outputs/stage_f/validation_calibration_v1'


def export():
    target=OUT/'chatgpt_stage_f_clip_discussion_bundle.zip'
    assert not target.exists(),'Preserve previous exports; use a new version for a new state'
    source_paths=[OUT/'diagnostic_report.json',OUT/'diagnostic_report.md',OUT/'threshold_table.json',
                  OUT/'operating_points.json',OUT/'leave_one_negative_out.json',OUT/'diagnostic_settings.json',
                  OUT/'conservative_gate_diagnostics.png',ASSISTED/'latest_human_export.json',
                  ASSISTED/'ai_suggestions.json',ASSISTED/'verification_order.json',ASSISTED/'active_selection.json',
                  BASE/'human_check/annotation_dataset.json',BASE/'human_check/private_calibration_scores.json',
                  BASE/'business_roles.json',ROOT/'configs/stage_f_validation_protocol.yaml',
                  ROOT/'configs/stage_f_validation_caption_pool.yaml',ROOT/'configs/stage_f_consistency_stability.yaml',
                  ROOT/'specs/006b_stage_f_validation_execution.md',ROOT/'DECISIONS.md',
                  ROOT/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',ROOT/'docs/STAGE_F_ANNOTATION_PROVENANCE.md']
    protected={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_paths}
    report=read(OUT/'diagnostic_report.json');labels=read(ASSISTED/'latest_human_export.json')
    dataset=read(BASE/'human_check/annotation_dataset.json');order=read(ASSISTED/'verification_order.json')
    ai=read(ASSISTED/'ai_suggestions.json');selection=read(ASSISTED/'active_selection.json')
    actual,prefix=validate_declared_prefix(dataset,order,selection,ai,labels)
    roles={r['business_id']:r['role'] for r in read(BASE/'business_roles.json')}
    scores=read(BASE/'human_check/private_calibration_scores.json')
    fit={r['case_id']:r for r in report['fit_cases']}
    uncertain=set(report['uncertain_excluded_case_ids'])
    source_cases={r['case_id']:r for r in dataset['cases']}
    ai_cases={r['case_id']:r for r in ai['cases']}
    rows=[];images={}
    # All currently verified image cases in the frozen order; no outcome-based subsampling.
    for cid in order['orders']['image_caption']:
        h=actual[cid]
        if h['human_verified'] is not True:continue
        c=source_cases[cid];assert c['kind']=='image_caption' and roles[c['business_id']]=='calibration'
        pid=c['photo_ids'][0];image_file=f'images/{pid}.jpg';source=ROOT/c['image_paths'][0]
        images[image_file]=(source,sha(source))
        in_fit=cid in fit
        assert in_fit==(h['human_final_label']['consistency']!='uncertain')
        assert in_fit or cid in uncertain
        assert h['ai_suggested_label']==ai_cases[cid]['ai_suggested_label']
        clip=scores[cid]['clip_score']
        if in_fit:
            assert fit[cid]['clip_score']==clip and fit[cid]['caption']==c['normalized_captions'][0]
        rows.append({'case_id':cid,'photo_id':pid,'business_id':c['business_id'],'business_name':c['name'],
                     'split':'validation','internal_role':'calibration','image_file':image_file,
                     'caption':c['normalized_captions'][0],'raw_caption':c['raw_captions'][0],
                     'ai_suggested_label':h['ai_suggested_label'],
                     'human_final_label':h['human_final_label'],'human_consistency':h['human_final_label']['consistency'],
                     'human_relevance':h['human_final_label']['relevance'],'human_verified':True,
                     'verification_action':h['verification_action'],'verified_at':h['verified_at'],
                     'verification_source':h.get('verification_source'),
                     'human_label_revision_history':h.get('human_label_revision_history',[]),'notes':h.get('notes',''),
                     'clip_score':clip,'clip_decision_score_6decimals':round(float(clip),6),
                     'relevance_score':scores[cid]['relevance_margin'],
                     'used_in_consistency_diagnostic_fit':in_fit,
                     'exclusion_reason':None if in_fit else 'human_verified_uncertain_excluded_from_binary_fit',
                     'operating_point_decisions':{rule:{'threshold':p['threshold'],'accepted':round(float(clip),6)>=p['threshold']} for rule,p in report['operating_points'].items()} if in_fit else None})
    assert len(rows)==130 and len(images)==130 and len(fit)==117 and len(uncertain)==13
    assert Counter(r['human_consistency'] for r in rows)==Counter(consistent=112,inconsistent=5,uncertain=13)
    assert {r['case_id'] for r in rows if r['used_in_consistency_diagnostic_fit']}==set(fit)
    corrected=next(r for r in rows if r['case_id']=='IMAGE_NvOxiwPwBMO6Gkv5zb4-Fw')
    assert corrected['human_consistency']=='uncertain' and corrected['used_in_consistency_diagnostic_fit'] is False
    previous_bundle=ASSISTED/'chatgpt_pending20_review_bundle_round05.zip'
    with ZipFile(previous_bundle) as z:
        old=z.read(next(n for n in z.namelist() if n.endswith('/rubric.md'))).decode('utf-8')
    definitions='## Consistency: exact active UI question'+old.split('## Consistency: exact active UI question',1)[1].split('## Keep the criteria separate',1)[0]
    separation=old.split('## Keep the criteria separate',1)[1].split('Do not manufacture',1)[0]
    rubric='''# Active Consistency / Relevance rubric — definitions unchanged

This is an export for discussion, not a new annotation batch. All130image cases are already human-verified. The binary diagnostic uses117non-uncertain cases; all13uncertain cases are excluded. The user-confirmed correction of IMAGE_NvOxiwPwBMO6Gkv5zb4-Fw is already reflected. Initial AI suggestions remain separate and unchanged. ChatGPT is an optional second opinion, not the final human-label author.

'''+definitions+'## Keep the criteria separate'+separation+'''
Do not manufacture negatives, reinterpret uncertain as negative, or decide visual labels from CLIP scores/cutoffs. Generic factually compatible captions are not automatically inconsistent. Incorrect visual claims can be inconsistent while remaining dining-domain relevant. Relevance is the broad adopted restaurant/bar/cafe domain, not exact named-business identity.

Scores are exposed here for retrospective analysis, so any new comparison is not blinded independent annotation. Human-confirmed labels remain fixed for this discussion. Put suggested corrections in separate review fields with image evidence; they require explicit user confirmation before adoption. No threshold freeze, protocol amendment, K-study or annotation expansion is authorized by this export.
'''
    points=report['operating_points'];boot=report['bootstrap_summary']
    lines=['# Discussion brief for ChatGPT — current Stage F CLIP results','',
           'Prepared for the researcher on 2026-10-09 (Asia/Saigon). Please discuss the evidence with me in Vietnamese. '
           'This package requests methodological discussion, not implementation or automatic threshold selection.','',
           '## Thesis context and controls','',
           '- Restaurant multimodal review summarization with BLIP-2 captions and LoRA-tuned LED.',
           '- Cohort: 2,300 businesses = 2,000 train / 200 validation / 100 held-out test.',
           '- Full validation caption/scoring pool: 2,504 photos; reused, not regenerated.',
           '- Frozen internal validation roles: 40 calibration businesses and 160 K-study businesses.',
           '- Calibration uses only human-verified cases from the 40-business role. No train/test or K-study outcomes fit thresholds.',
           '- Reviews-only, Unfiltered-caption and Filtered-caption experiments remain the plan. Unfiltered is minimally validated; Filtered adds substantive filtering. Both caption conditions must have the same eventual maximum K and caption-token budget.',
           '- The master plan explicitly includes CLIP consistency filtering for Filtered. Removing its gate or changing its objective/rubric is a research-design change requiring explicit researcher approval, not an already approved decision.',
           '- K candidates remain {3,5,8,10}; no K-study run or final K selected. Thresholds must be settled before inspecting K-study outcomes. Final 30-business human evaluation is separate and unchanged.','',
           '## Corrected labels and provenance','',
           'The user is the primary human annotator: initial AI pre-label -> user inspects image/caption and chooses/edits -> optional ChatGPT comparison -> user confirms final decision. ChatGPT suggestions never automatically become human labels.','',
           'The user manually re-inspected IMAGE_NvOxiwPwBMO6Gkv5zb4-Fw and confirmed inconsistent -> uncertain. Relevance remains relevant. The case remains human-verified but is excluded from binary fitting; its initial AI suggestion stays inconsistent. All other labels remain unchanged.','',
           'Current verified images: 112 consistent / 5 inconsistent / 13 uncertain = 130. The binary diagnostic uses only117cases spanning39of the40calibration businesses; uncertain-only evidence does not enter the fit. All130images are supplied for transparency, not a request to relabel them. There are270human-confirmed cases including140pairs. The previously opened20image cases remain pending and are NOT in this package. No extra annotation batch has been opened.','',
           'D-029 removed the old pragmatic minimum10-negative gate. It does not convert uncertainty or authorize automatic expansion. Relevance and pairwise criteria already satisfy their calibration balance requirements; do not expand them.','',
           '## Frozen CLIP definition and exploratory grid','',
           'Model: openai/clip-vit-base-patch32; revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. Score is unscaled cosine between L2-normalized image/caption features, computed with the existing FP32 scorer. Decisions round scores to6decimals and accept if score >= cutoff. Cosine is not a calibrated probability of factual correctness.','',
           'The inherited grid remains -1..1 by0.01; the existing maximum-BA tie order is precision, recall, then higher cutoff. The user-requested0.20..0.35 table and >=95% / >=90% recall points are exploratory comparisons, NOT a protocol amendment or replacement objective. The inherited grid results are retained separately and give the same full-data BA optimum0.27.','',
           '## Three diagnostic operating points (not final choices)','',
           '| Point | Cutoff | TP | TN | FP | FN | Consistent recall | Inconsistent specificity | Precision | BA | Consistent rejected | Inconsistent accepted |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rule,p in points.items():
        lines.append(f"| {rule} | {p['threshold']:.2f} | {p['TP']} | {p['TN']} | {p['FP']} | {p['FN']} | {p['consistent_recall']:.2%} | {p['inconsistent_recall_specificity']:.2%} | {p['precision']:.2%} | {p['balanced_accuracy']:.2%} | {p['consistent_rejected_count']}/{p['consistent_total']} ({p['consistent_rejected_pct']:.2f}%) | {p['inconsistent_accepted_count']}/{p['inconsistent_total']} ({p['inconsistent_accepted_pct']:.2f}%) |")
    lines+=['','TP=consistent accepted; TN=inconsistent rejected; FP=inconsistent accepted; FN=consistent rejected. '
            'Accepting everything would give precision112/117=95.73%and BA50%; high precision alone therefore does not establish utility.','',
            'The full16-row table and exact accepted/rejected ID partitions are included. Both high-recall points accept all five observed inconsistents. The BA-optimal point rejects33.93%of the verified consistent captions.','',
            '## Stability','',
            '2,000case-stratified bootstrap draws, seed42, independently resampling the112positive and5negative cases with replacement. Both fixed-original-cutoff metrics and refitted comparison rules are reported; they must not be confused.','',
            '| Point | Refit empirical2.5–97.5%cutoff range | Original cutoff fraction of all draws | Feasible/infeasible draws | LONO refit cutoffs |',
            '|---|---|---:|---|---|']
    for rule,p in points.items():
        b=boot[rule]['refit_selection_rule'];t=b['refitted_threshold'];lo=[f['rules'][rule]['candidate']['threshold'] for f in report['leave_one_negative_out']]
        lines.append(f"| {rule} | {t['percentile_2_5']:.2f}–{t['percentile_97_5']:.2f} | {b['same_as_full_fit_fraction_all_draws']:.2%} | {b['feasible_draws']}/{b['infeasible_draws']} | {', '.join(f'{v:.2f}' for v in lo)} |")
    lines+=['','36of2,000draws had no feasible >=95%recall cutoff on0.20..0.35; these remain null, with no below-grid fallback. '
            'At the fixed0.22cutoff, bootstrap recall meets95%in98.2%of draws; fixed0.24meets90%in87.85%. '
            'Refitting a higher cutoff in a resample can violate the desired recall when evaluated back on the original cases; this is sensitivity evidence, not a population recall guarantee.','',
            'LONO leaves112positives and4negatives. The BA rule drops from0.27to0.26when IMAGE_oGc0UjVHqK1GB9K1p6dYQQ is removed, changing15original-set predictions. Recall-constrained cutoffs stay0.22/0.24by construction because negative removal cannot change the positive recall constraint. Their unchanged LONO thresholds are not independent proof of robustness.','',
            'Only five observed negatives, deliberate score-quantile sampling and shared-business dependence limit inference. Bootstrap repeats the same cases. The0..0fixed-specificity interval at low cutoffs is conditional on these five negatives, not certainty of population specificity. Neither percentile summaries nor apparent calibration BA are held-out generalization estimates.','',
            '## Open research question','',
            'Can the frozen CLIP score serve as a conservative coarse image-text alignment gate in this thesis, even though these data do not show useful factual-error rejection at high consistent recall? Distinguish coarse visual alignment from the current human Consistency rubric (significant incorrect visual claims). Do not redefine existing labels to suit CLIP.','',
            'Codex preliminary interpretation: the observed high-recall cutoffs remove only human-consistent cases and retain all five inconsistents; this subset does not demonstrate useful rejection. This is NOT a conclusion that CLIP universally cannot help, and NO alternative has been selected.','',
            '## Please help us discuss','',
            '1. Assess whether the evidence supports any of the three operating points as a conservative gate. Explain what the current sample can and cannot establish; do not recommend0.22merely because it meets recall.',
            '2. Use the actual supplied images where needed, especially the five confirmed inconsistent cases and low-score consistent cases, to distinguish coarse mismatches from subtle object/ingredient/count/attribute errors. This is not a new130-case annotation assignment. Keep any disagreement as a separate second-opinion field requiring user confirmation.',
            '3. Compare possible next steps without silently approving any: retain a conservative CLIP gate with explicit empirical limitations; retain CLIP for diagnostics/quality ranking while removing its hard gate (requires amendment to the master plan); or request a separately defined alignment diagnostic if the intended construct differs from factual consistency (requires a reviewed protocol, cannot retroactively rewrite labels).',
            '4. State exactly which proposal/Stage F requirements each alternative preserves or changes. Identify any evidence still necessary and the smallest justified next step. Do not default to more labeling solely to reach10negatives.',
            '5. Suggest honest thesis wording for the calibration tradeoff, natural imbalance and AI-assisted human verification. Keep final30-business evaluation and Unfiltered/Filtered K/budget controls unchanged.',
            '6. Return your recommendation and rationale for discussion in Vietnamese. Do not freeze thresholds, pick K, open cases, regenerate captions/scores, or change model/rubric/config/protocol. Researcher review comes first.','',
            '## Package navigation','',
            '- cases.json + images/: all130already verified calibration images;117flagged as used in binary fit,13flagged excluded uncertain.',
            '- diagnostic_report.md / .json: complete corrected analysis, including all16thresholds and exact case partitions.',
            '- threshold_table.json, operating_points.json, bootstrap_summary.json, leave_one_negative_out.json: machine-readable evidence.',
            '- scorer_configuration.yaml, generation_configuration.yaml, active_stage_f_spec.md, master_plan_filtering_excerpt.md, annotation_provenance.md: unchanged source snapshots. Older historical statements in these documents are superseded by their dated active amendments; they are not authorization to execute anything.',
            '- conservative_gate_diagnostics.png: curves and bootstrap cutoff frequencies.','']
    brief='\n'.join(lines)
    # Improve narrative spacing while protecting all exact scientific identifiers.
    import re
    brief=brief.replace('revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`')
    identifiers=list(source_cases)+list(points)+['openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','L2','FP32']
    tokens={f'@@ID_{i}@@':value for i,value in enumerate(identifiers)}
    for token,value in tokens.items():brief=brief.replace(value,token)
    brief=re.sub(r'(?<=[A-Za-z%])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',brief)
    brief=re.sub(r'(?<=%)(?=[A-Za-z])|(?<=;)(?=\d)',' ',brief)
    for token,value in tokens.items():brief=brief.replace(token,value)
    readme='''# Start here

Please read GPT_DISCUSSION_BRIEF.md and discuss the current Stage F CLIP decision with me in Vietnamese. Use the corrected diagnostic table and stability results, not pre-correction44-case ZIP results. Do not choose/freeze a threshold or change the protocol automatically.

The researcher remains the primary human annotator. These are already confirmed labels; ChatGPT is only an optional second opinion. No new labeling batch is requested. Any suggested correction must be separate and explicitly confirmed by the user before adoption.

This portable package contains130verified validation-calibration image cases. Only117non-uncertain cases are used for the binary audit;13uncertain cases are included for transparency and explicitly excluded. All relative image_file paths resolve within this folder. JPEGs are exact source copies. Images outside calibration, unverified pending cases and train/test/K-study data are not included.

Quantitative discussion may rely on the reports. Inspect actual images where a proposed interpretation needs visual evidence, especially observed negative cases and rejected positives. If an image cannot be accessed, state the limitation; do not invent observations or require a new annotation round.

Questions and boundaries are in the brief. All substantive alternatives remain proposals for researcher review, including any change from a CLIP hard gate to ranking/diagnostics only.
'''
    plan=(ROOT/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md').read_text(encoding='utf-8')
    excerpt='## 7.1 Unfiltered condition'+plan.split('## 7.1 Unfiltered condition',1)[1].split('## 8.',1)[0]
    # Stop before subsequent unrelated plan stages if their heading differs.
    excerpt=excerpt.split('\n---\n',1)[0]
    files={'README.md':readme.encode('utf-8'),'GPT_DISCUSSION_BRIEF.md':brief.encode('utf-8'),
           'rubric.md':rubric.encode('utf-8'),'cases.json':(json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode('utf-8'),
           'bootstrap_summary.json':(json.dumps(report['bootstrap_summary'],ensure_ascii=False,indent=2)+'\n').encode('utf-8'),
           'master_plan_filtering_excerpt.md':('# Unchanged master-plan excerpt\n\nSource: thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md, section7.\n\n'+excerpt).encode('utf-8')}
    copies={'diagnostic_report.md':OUT/'diagnostic_report.md','diagnostic_report.json':OUT/'diagnostic_report.json',
            'threshold_table.json':OUT/'threshold_table.json','operating_points.json':OUT/'operating_points.json',
            'leave_one_negative_out.json':OUT/'leave_one_negative_out.json','diagnostic_settings.json':OUT/'diagnostic_settings.json',
            'conservative_gate_diagnostics.png':OUT/'conservative_gate_diagnostics.png',
            'scorer_configuration.yaml':ROOT/'configs/stage_f_validation_protocol.yaml',
            'generation_configuration.yaml':ROOT/'configs/stage_f_validation_caption_pool.yaml',
            'active_stage_f_spec.md':ROOT/'specs/006b_stage_f_validation_execution.md',
            'annotation_provenance.md':ROOT/'docs/STAGE_F_ANNOTATION_PROVENANCE.md'}
    for name,path in copies.items():files[name]=path.read_bytes()
    (OUT/'GPT_DISCUSSION_BRIEF.md').write_bytes(files['GPT_DISCUSSION_BRIEF.md'])
    prefix='chatgpt_stage_f_clip_discussion/'
    with ZipFile(target,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in files.items():z.writestr(prefix+name,data)
        for name,(source,_) in images.items():z.write(source,prefix+name)
    with ZipFile(target) as z:
        assert z.testzip() is None
        names=set(z.namelist());exported=json.loads(z.read(prefix+'cases.json'))
        assert exported==rows
        for row in exported:
            cid=row['case_id'];c=source_cases[cid];h=actual[cid]
            assert row['human_final_label']==h['human_final_label'] and row['human_verified'] is True
            assert row['caption']==c['normalized_captions'][0] and row['raw_caption']==c['raw_captions'][0]
            assert row['photo_id']==c['photo_ids'][0] and row['business_id']==c['business_id']
            p=PurePosixPath(row['image_file']);assert not p.is_absolute() and '..' not in p.parts and ':' not in str(p)
            assert prefix+row['image_file'] in names
            assert hashlib.sha256(z.read(prefix+row['image_file'])).hexdigest()==images[row['image_file']][1]
        assert '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268' in z.read(prefix+'GPT_DISCUSSION_BRIEF.md').decode('utf-8')
        for name,data in files.items():assert z.read(prefix+name)==data
    assert protected=={name:sha(ROOT/name) for name in protected}
    assert all(sha(p)==s for p,s in images.values())
    assert not (BASE/'frozen_thresholds.json').exists()
    receipt={'status':'PASS_EXPORT_ONLY','date_local':'2026-10-09','timezone':'Asia/Saigon',
             'zip_path':str(target),'zip_bytes':target.stat().st_size,'zip_MB':target.stat().st_size/1e6,'zip_sha256':sha(target),
             'images':130,'verified_image_cases':130,'used_in_binary_fit':117,'uncertain_excluded_from_fit':13,
             'Consistency_counts':dict(Counter(r['human_consistency'] for r in rows)),
             'Relevance_counts':dict(Counter(r['human_relevance'] for r in rows)),
             'pending_cases':0,'case_ids':[r['case_id'] for r in rows],
             'selection':'all_current_human_verified_images_in_frozen_calibration_order_not_new_sampling',
             'captions_IDs_human_labels_exact':True,'all_relative_image_paths_valid':True,'all_JPEG_bytes_match_source':True,
             'all_source_files_byte_unchanged':True,'source_sha256':protected,
             'original_rubric_source_zip':str(previous_bundle.relative_to(ROOT)).replace('\\','/'),
             'threshold_calibration_run':False,'threshold_frozen':False,'protocol_amended':False,'K_study_run':False,
             'case_expansion':False,'ChatGPT_role':'optional_comparison_and_methodology_discussion_no_automatic_adoption',
             'brief_path':str(OUT/'GPT_DISCUSSION_BRIEF.md'),'brief_sha256':sha(OUT/'GPT_DISCUSSION_BRIEF.md')}
    write(OUT/'chatgpt_stage_f_clip_discussion_bundle_report.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['case_ids','source_sha256']}))


if __name__=='__main__':export()
