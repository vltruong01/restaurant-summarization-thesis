"""Portable second-opinion export only; source labels and scientific outputs stay fixed."""
from pathlib import Path, PurePosixPath
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from stage_f_verification_workflow import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_f/consistency_stability_v1'
ASSISTED = ROOT/'outputs/stage_f/ai_assisted_calibration_v1'


def export():
    human_path = ASSISTED/'latest_human_export.json'
    queue_path = ROOT/'outputs/stage_f/validation_calibration_v1/human_check/annotation_dataset.json'
    report_path = OUT/'stability_report.json'
    audit_path = OUT/'consistency_audit_export_report.json'
    previous_bundle = ASSISTED/'chatgpt_pending20_review_bundle_round05.zip'
    protected = [human_path, queue_path, report_path, audit_path, previous_bundle,
                 ASSISTED/'active_selection.json', ASSISTED/'verification_order.json',
                 ASSISTED/'ai_suggestions.json', ROOT/'configs/stage_f_validation_protocol.yaml',
                 ROOT/'configs/stage_f_consistency_stability.yaml']
    before = {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in protected}
    human = read(human_path); actual = {r['case_id']:r for r in human['cases']}
    queue = {r['case_id']:r for r in read(queue_path)['cases']}
    ai = {r['case_id']:r for r in read(ASSISTED/'ai_suggestions.json')['cases']}
    fit = read(report_path); fit_cases = {r['case_id']:r for r in fit['fit_cases']}
    ids = read(audit_path)['case_ids']; assert len(ids) == len(set(ids)) == 44
    threshold = fit['full_fit']['threshold']; rows = []; images = {}
    for cid in ids:
        c = queue[cid]; h = actual[cid]; r = fit_cases[cid]
        assert c['kind'] == 'image_caption' and h['human_verified'] is True
        assert h['human_final_label']['consistency'] == r['human_final_label']
        assert h['ai_suggested_label'] == ai[cid]['ai_suggested_label']
        assert c['normalized_captions'][0] == r['caption'] and c['raw_captions'][0] == r['raw_caption']
        assert c['photo_ids'][0] == r['photo_id'] and c['business_id'] == r['business_id']
        group = 'verified_negative' if not r['positive'] else 'verified_positive_rejected_at_provisional_0.27'
        assert not r['positive'] or r['decision_score'] < threshold
        image_file = f"images/{r['photo_id']}.jpg"
        image_path = ROOT/c['image_paths'][0]
        images[image_file] = (image_path, sha(image_path))
        rows.append({'case_id':cid,'photo_id':r['photo_id'],'business_id':r['business_id'],
                     'image_file':image_file,'caption':r['caption'],'raw_caption':r['raw_caption'],
                     'ai_suggested_consistency':ai[cid]['ai_suggested_label']['consistency'],
                     'ai_suggested_relevance':ai[cid]['ai_suggested_label']['relevance'],
                     'ai_suggested_label':ai[cid]['ai_suggested_label'],
                     'human_consistency':h['human_final_label']['consistency'],
                     'human_relevance':h['human_final_label']['relevance'],
                     'human_final_label':h['human_final_label'],'human_verified':h['human_verified'],
                     'verification_action':h.get('verification_action'),'verified_at':h.get('verified_at'),
                     'notes':h.get('notes',''),'clip_score':r['clip_score'],
                     'clip_decision_score_6decimals':r['decision_score'],
                     'relevance_score':None,'audit_group':group,
                     'provisional_CLIP_threshold':threshold,'threshold_frozen':False,
                     'passes_provisional_CLIP_cutoff':r['decision_score'] >= threshold})
    with ZipFile(previous_bundle) as z:
        rubric_name = next(n for n in z.namelist() if n.endswith('/rubric.md'))
        previous_rubric = z.read(rubric_name).decode('utf-8')
    # Copy the active definitions verbatim; replace only the obsolete pending20 scope.
    definitions = previous_rubric.split('## Consistency: exact active UI question',1)[1].split('## Keep the criteria separate',1)[0]
    separate = previous_rubric.split('## Keep the criteria separate',1)[1].split('Do not manufacture',1)[0]
    rubric = '''# Stage F — Consistency audit second-opinion rubric

## Scope and provenance

This is a read-only, score-visible retrospective audit of 44 ALREADY human-verified image-caption cases from the frozen40-business calibration role: all6verified non-uncertain Consistency negatives and all38verified positives rejected by the provisional CLIP0.27 cutoff. This is a deliberately selected diagnostic subset, NOT a representative accuracy sample and NOT a new calibration queue. No pending cases or train/test/K-study cases are included. The full-fit and stability summaries were computed on118verified non-uncertain cases, not just these44.

The caption is the exact normalized BLIP-2 UI text; raw_caption preserves decoded text. Source JPEGs are copied byte-for-byte. All image_file paths are relative to this folder. Human labels, original AI suggestions and case/photo/business IDs are unchanged.

The user is the primary annotator. ChatGPT is only an optional comparison/adjudication aid after the user's own inspection; the final label remains a user-confirmed human decision. A ChatGPT recommendation must remain separate and does not automatically change a human label. Since existing labels and scores are exposed, this is not blinded independent annotation. Judge the image/caption first rather than rationalizing scores or supplied labels.

## Consistency: exact active UI question''' + definitions + '''## Keep the criteria separate''' + separate + '''Do not manufacture negative examples to satisfy class balance. Do not convert uncertain into negative. Generic but compatible captions are not automatically inconsistent; incorrect visual claims can be inconsistent while still domain-relevant.

## Score interpretation and D-029 boundary

CLIP score is the frozen unscaled L2-normalized image/caption cosine; decisions round to6decimals and pass if score>=the provisional0.27 cutoff. The cutoff is NOT frozen and is not a label criterion. A high CLIP score does not prove every caption detail is correct; a low score does not prove inconsistency. Relevance score is not part of this audit and is null. Do not choose new cutoffs, modify scoring definitions, fit thresholds, select K or request negatives solely to reach10 examples.

D-029 paused further20-case expansion solely for the old minimum10-negative heuristic. Verified uncertain remains excluded from fitting and unchanged. The stability analysis is awaiting researcher review; no automatic threshold freeze or expansion is authorized by this export.

## Suggested comparison output

Review all44actual images. For each case retain unchanged case_id/photo_id and return separate review_consistency, review_relevance, visual_evidence, agreement_with_human_consistency and agreement_with_human_relevance. Allowed Consistency labels: consistent/inconsistent/uncertain; allowed Relevance labels: relevant/irrelevant/uncertain. Give specific evidence for a disagreement, and retain uncertainty where the image cannot resolve a claim. If an image cannot be accessed, say not_reviewed rather than inventing a judgment.

Leave every supplied human_final_label and human_verified unchanged. Review suggestions are not human ground truth and must not be automatically imported. Any eventual change requires explicit user confirmation. Summarize observable CLIP failure patterns without treating this selected44-case subset as full-pool error rates.
'''
    prompt = '''# Instructions for optional ChatGPT comparison

Open cases.json and inspect the actual image_file for EVERY one of the44 cases. Read rubric.md first. Provide independent visual reasoning before comparing existing human labels or CLIP scores. Scores are diagnostic information only; do not use them to define consistency/relevance.

Return a separate JSON containing reviews with case_id, photo_id, review_consistency, review_relevance, visual_evidence, agreement_with_human_consistency, agreement_with_human_relevance, and review_status. Mark inaccessible images not_reviewed. Do not overwrite the supplied human labels. ChatGPT is a second-opinion tool; the human user must explicitly confirm any later decision.

Summarize why CLIP might pass some inconsistent captions or reject compatible ones. This44-case diagnostic subset was selected by current human label and cutoff effect; do not infer representative accuracy from it. No threshold/K selection, calibration, source-label modification or further queue expansion is requested.

Files:44original JPEGs, cases.json, active rubric.md, prior stability_report.md and its chart. Full stability fitting used112positive+6negative cases, with12uncertain excluded. No model/scorer/caption rerun occurred for this export.
'''
    target = OUT/'chatgpt_consistency_audit44_review_bundle.zip'
    assert not target.exists(), 'Do not overwrite an existing export'
    prefix = 'chatgpt_consistency_audit44_review_bundle/'
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr(prefix+'cases.json',json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        z.writestr(prefix+'rubric.md',rubric)
        z.writestr(prefix+'README.md',prompt)
        for name in ['stability_report.md','stability_distributions.png']:
            z.write(OUT/name,prefix+name)
        for image_file,(source,_) in images.items():z.write(source,prefix+image_file)
    with ZipFile(target) as z:
        assert z.testzip() is None
        names=set(z.namelist()); exported=json.loads(z.read(prefix+'cases.json'))
        assert exported == rows and [r['case_id'] for r in exported] == ids
        for row in exported:
            path=PurePosixPath(row['image_file'])
            assert not path.is_absolute() and '..' not in path.parts and ':' not in row['image_file']
            assert prefix+row['image_file'] in names
            assert row['human_final_label'] == actual[row['case_id']]['human_final_label']
            assert row['caption'] == queue[row['case_id']]['normalized_captions'][0]
            assert row['raw_caption'] == queue[row['case_id']]['raw_captions'][0]
            assert hashlib.sha256(z.read(prefix+row['image_file'])).hexdigest() == images[row['image_file']][1]
    assert before == {name:sha(ROOT/name) for name in before}
    assert all(sha(p)==s for p,s in images.values())
    receipt={'status':'PASS_EXPORT_ONLY','zip_path':str(target),'zip_bytes':target.stat().st_size,
             'zip_MB':target.stat().st_size/1e6,'zip_sha256':sha(target),'cases':44,
             'verified':44,'pending':0,'negative_cases':6,'rejected_positive_cases':38,
             'human_consistency_counts':dict(Counter(r['human_consistency'] for r in rows)),
             'human_relevance_counts':dict(Counter(r['human_relevance'] for r in rows)),
             'case_ids':ids,'protected_inputs_byte_unchanged':True,'source_sha256':before,
             'source_JPEGs_byte_identical':True,'all_relative_image_paths_valid':True,
             'captions_IDs_and_human_labels_exact':True,'active_rubric_definitions_verbatim':True,
             'previous_rubric_source_zip':str(previous_bundle.relative_to(ROOT)).replace('\\','/'),
             'threshold_calibration_run':False,'threshold_frozen':False,'K_selection_run':False,
             'queue_expansion_performed':False,'ChatGPT_role':'optional_second_opinion_not_human_ground_truth',
             'selection':'exact_same44_existing_read_only_audit_cases_not_new_calibration_sample'}
    write(OUT/'chatgpt_consistency_audit44_review_bundle_report.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','case_ids']}))


if __name__ == '__main__': export()
