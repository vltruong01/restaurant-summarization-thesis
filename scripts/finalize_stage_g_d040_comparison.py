"""Evidence-linked AI second-opinion report; never edits model output or labels."""
import json
import re
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, immutable

NEW=ROOT/'outputs/stage_g/teacher_pilot_v6_focused_d040_v1'


def run():
    machine=read(NEW/'machine_comparison.json');audit=read(NEW/'independent_integrity_report.json')
    completion=read(NEW/'execution_v1/completion.json');versions=machine['versions'];findings=[]

    def visual(fid,bid,pid,category,reason,pattern=None):
        evidence={v:[r for r in d['visual_claims'] if r['business_id']==bid and r['photo_id']==pid
            and (pattern is None or re.search(pattern,r['text'],re.I))] for v,d in versions.items()}
        current=evidence['D040'];previous=evidence['D039']
        retained=[r for r in current if r['verdict']=='verified']
        if category=='unsupported_detail':
            outcome=('remaining_unsupported_or_ambiguous_verified_claim' if any(r['verdict']=='verified' for r in previous)
                else 'new_verified_detail_regression_or_ambiguity') if retained else (
                'improvement_detail_no_longer_verified' if any(r['verdict']=='verified' for r in previous) else 'exclusion_preserved')
        elif category=='ordinary_control':
            outcome='ordinary_description_preserved' if retained else 'possible_overstrictness_or_omission_review_needed'
        else:outcome=category
        findings.append({'finding_id':fid,'business_id':bid,'photo_id':pid,'classification':outcome,
            'assessment':reason,'evidence':evidence,'new_verified_occurrences':len(retained),
            'AI_second_opinion':True,'human_verified':False})

    visual('ASPARAGUS','bLJP0fVgscuUQy0hE_CRXA','pbVvz17fpqf6m72XIelpZw','unsupported_detail',
        'Actual photo has leafy greens with thick stalks, not clearly identifiable asparagus. Specific identity should not enter the target; ordinary white plate/knife remains useful. New rejected versus uncertain wording is still teacher judgment, not a human-final annotation.',r'asparagus')
    visual('TWO_DRINKS','bLJP0fVgscuUQy0hE_CRXA','ajfOEJJQ6D8AN1aIxY53vA','unsupported_detail',
        'One recognizable drink; an edge object cannot establish a second. Excluding the exact count is an improvement if generic drink/table/restaurant remains. Ambiguity normally warrants uncertain; rejected wording may overstate evidence rather than proving no hidden second object.',r'\btwo\b|\b2\b')
    visual('TWO_DRINKS_GENERIC','bLJP0fVgscuUQy0hE_CRXA','ajfOEJJQ6D8AN1aIxY53vA','ordinary_control',
        'A drink on a table in a dining setting is independently supported and caption-recoverable; retain this simple content rather than discard the entire caption.',r'drink|table|restaurant')
    visual('GREEN_BEANS','FFV8_b-ec5Z0E3xgvkVxKw','H94oFnARbFwDwWAluQmjNA','unsupported_detail',
        'Green stalk pieces among pasta do not securely establish green-bean identity. Report both conditions and actual claims; do not force equal verdicts or call changed paraphrases exact matches.',r'green beans')
    visual('CHICKEN','FFV8_b-ec5Z0E3xgvkVxKw','H94oFnARbFwDwWAluQmjNA','unsupported_detail',
        'Coated protein pieces are visible but exact chicken identity is not secure. Keep pasta/plate distinct from this ambiguous ingredient.',r'chicken')
    visual('GLASS_WATER','PRrWtQ3rdINngwESZNefVQ','Mry2XDZYS_rKCGy5b9ZNJw','ambiguity_and_consistency_review',
        'Partial transparent vessel is visible at the upper-left crop. Ordinary glass/water interpretation can be plausible, but precise claim confidence differs; examine current verdicts without declaring the interpretation conclusively false.',r'glass|water')
    visual('TWO_PLATES','PRrWtQ3rdINngwESZNefVQ','Mry2XDZYS_rKCGy5b9ZNJw','ordinary_control',
        'Two food-bearing plates, one carrying a bowl, support an ordinary two-plates description. Do not require both plates to hold food directly.',r'plate')
    visual('FORK_HOLDER','PRrWtQ3rdINngwESZNefVQ','qcXO3blRqZmre1xVVq1soQ','unsupported_detail',
        'Fork/food/plate are visible but the hand/person holding it is not. A holder claim must stay excluded while ordinary visible fork relation survives.',r'person|holding|holds')
    visual('FORK_GENERIC','PRrWtQ3rdINngwESZNefVQ','qcXO3blRqZmre1xVVq1soQ','ordinary_control',
        'Clearly visible fork over food/plate supports a simple relation; no need for actor inference.',r'fork|close')
    visual('HOTDOG','8xTHtLoNIwdpf0FEvIpQIw','YGPmXXkzu6x_OifETfXQfw','unsupported_detail',
        'Shredded-meat roll lacks recognizable sausage hot-dog structure. Do not restore wrong identity or add a replacement photo-only dish name.',r'hot dog')
    visual('BOAT','UCzFaBbPDKtHTWsBE5LGtQ','gpeR5DxyJN-eIduZEoMZIg','unsupported_detail',
        'White basin-like fixture lacks recognizable boat hull. Generic placement near road/tree remains caption-recoverable; alternative fixture noun must not enter target.',r'boat')
    visual('PIZZA','nv59KN19fEQax9RL0lUdkw','6rheOeS8NRwjxVU-iG92kw','ordinary_control',
        'Round sliced cheese-covered flatbread supports ordinary large-pizza wording under the researcher-approved interpretation. D039 accepted it; D040 rejects pizza while keeping generic size/table. This is a new regression against the approved ordinary-description guidance, not a model verdict to rewrite manually.',r'pizza')
    visual('LARGE_HAMBURGER','jMStvE-tQzSpRCAO0nAE6g','HTP0pKcn4dbJoDVQAp9TFQ','ordinary_control',
        'Substantial burger visibly supports qualitative large, with no ruler requirement.',r'large.*hamburger|large.*burger')
    visual('SIMPLE_PIE','UCzFaBbPDKtHTWsBE5LGtQ','-uGxVqv-Dh4OFaCqFMHMgQ','ordinary_control',
        'Pie/plate/checkered cloth are simple but factually supported. Low information density is not an error.')
    visual('BACKGROUND_PIZZAS','8xTHtLoNIwdpf0FEvIpQIw','tc0o8BwiiVoATMpSsUs5Hw','background_identity_review',
        'Foreground pizza supported; background looks like stacked rolls. Inspect exact clauses, especially other-pizzas versus generic background phrasing; decomposition changes do not imply equivalent claims.',r'pizza|background')
    visual('PIZZA_PLATE_TRAY','8xTHtLoNIwdpf0FEvIpQIw','tc0o8BwiiVoATMpSsUs5Hw','possible_new_overstrictness_and_semantic_conflict',
        'Foreground pizza rests on a shallow round metal serving dish. Ordinary plate wording can be reasonable; F rejecting it solely as tray/pan while U accepts plate is a possible overstrictness regression. F explanation calls foreground pizza rectangular, contrary to the visible round crust; that explanation-only shape is not emitted target content.',r'plate')
    visual('OCCLUDED_FORK','bLJP0fVgscuUQy0hE_CRXA','gBwXl-M8GcXDH_vx6dWhEw','possible_overstrictness_or_semantic_verdict_difference',
        'Knife clear, small partly occluded fork-like tines near upper plate edge. U uncertainty versus F verification can indicate excessive specificity/context dependence; assess the exact clauses and original crop, not automatic reconciliation.',r'fork')
    visual('SEATED_MEN','FFV8_b-ec5Z0E3xgvkVxKw','4-M3iicg2AFOIBlsC_dyzw','ordinary_posture_and_food_presence_review',
        'Three men at table/counter with drinks clearly visible. Normal sitting description can be reasonable without legs visible. No food clearly visible; report food-presence clauses separately.',r'sitt|seated|food')
    visual('MEN_DRINK_PLURALITY','FFV8_b-ec5Z0E3xgvkVxKw','4-M3iicg2AFOIBlsC_dyzw','semantic_count_confidence_difference_review',
        'Foreground drink glass and red can/container are visible. Ordinary multiple-drinks interpretation can be reasonable; U/F difference is a semantic confidence disagreement if wording differs, not an exact literal conflict.',r'drink')
    visual('WINE','FFV8_b-ec5Z0E3xgvkVxKw','Gs2aQAsJWgh7GIlHbRox9g','ordinary_liquid_identity_review',
        'Pizza and pale liquid in stemmed glass clear. Wine is plausible ordinary interpretation; any exclusion is a precision tradeoff, not conclusively a factual error.',r'wine|glass')
    visual('BACON','FFV8_b-ec5Z0E3xgvkVxKw','O8c5X6dENYj94NAhnig6LQ','ordinary_food_identity_review',
        'Cheese pizza with reddish cured-meat pieces visible. Bacon versus another cured meat needs interpretation; this targeted report does not require laboratory certainty.',r'bacon')
    visual('WOODEN_TABLE','nv59KN19fEQax9RL0lUdkw','RmeIk6J22rLDWAkxJT_gtg','spatial_material_precision_and_consistency_review',
        'Food dish sits on a visible wooden base/support; whether ordinary wooden-table phrasing is adequate is a precision question. Keep exact/paraphrased condition judgments separate.',r'wood|table')
    visual('WRAPPED_CUTLERY','jMStvE-tQzSpRCAO0nAE6g','FNtGtXamR9Uduv85WIcf7g','unsupported_detail',
        'Takeout bag/trays clear, wrapped utensil-like objects present. Bunch-of-silverware identity/count not securely established. D039 uncertainty versus any new confident verification is a concerning wrapped-content regression; generic bag description remains useful.',r'silverware|cutlery')
    visual('OPEN_PIANO','EzJI5x25U5GgbYBWhu5ZQQ','PvPlFCrN8HpiIt1hxY5Phg','ordinary_open_state_vs_specific_panel_review',
        'Piano has exposed front mechanism; ordinary open appearance is acceptable. Top-panel-specific assertion may be more exact than the evidence. Report wording/verdict, not a fixed expected label.',r'open|top')
    visual('STRAW','EzJI5x25U5GgbYBWhu5ZQQ','HVkUbl7niy2hIADyPh17MA','ordinary_control',
        'Visible straw/cup/counter supported. Existing original-only user confirmation is preserved and not copied automatically to new AI judgments.',r'straw')
    visual('NO_WINE_BOTTLE','EzJI5x25U5GgbYBWhu5ZQQ','Q6Ez6RPvOVnm_tOywa1GxA','unsupported_detail',
        'Mineral-water bottle label contradicts wine identity; napkin/table is independently supported.',r'wine')
    visual('CHAIR_VASE_RELATION','EzJI5x25U5GgbYBWhu5ZQQ','_-SDy-iDn-KetI7Ch6NQrg','chair_vase_relation_review',
        'White chairs and tables clear; vases are on tables, not atop chairs. Read decomposed claims for accidental relation propagation.',r'vase|top')
    bid='UCzFaBbPDKtHTWsBE5LGtQ'
    review_evidence={v:[r for r in d['review_claims'] if r['business_id']==bid] for v,d in versions.items()}
    findings.append({'finding_id':'UC_REVIEW_CATEGORY_AND_CONTRASTS','business_id':bid,
        'classification':'fresh_review_category_scope_and_contrast_assessment',
        'assessment':'All11 selected UC reviews read in full. D039 inherited-v5 seafood umbrella wrongly included gator bites/grits/pies/onion rings; not caused by reviewv6. Freshv7 now separates fish, onion rings, grits/pies and gator-bite praise without the false seafood umbrella. It preserves qualified greasiness/value/counter-ordering/drink-setup/menu-limit/gator-taste contrasts and positive staff/atmosphere evidence. No new category error or lost material contrasting opinion identified in this full11-review check. Full exact claims and original REV quotes follow; valid IDs alone do not prove semantics.',
        'evidence':review_evidence,'AI_second_opinion':True,'human_verified':False})
    reused=[b for b in machine['cohort_business_ids'] if b!=bid]
    findings.append({'finding_id':'UNCHANGED_REVIEW_CATEGORY_SCREEN','business_ids':reused,
        'classification':'unchanged_review_screen_not_new_prompt_evaluation',
        'assessment':'All7 inherited consensus texts screened for category/list scope. No additional clear false umbrella found in these texts; ordinary noodle/curry/coffee/Italian/Guatemalan/burger groups do not replicate the UC seafood error. This is not an exhaustive factual certificate over every original review. PR food/BYOB/value and Ez minority contrasts remain exactly unchanged; prior PR handmade-tortilla omission persists, not a new regression.',
        'evidence':{v:[r for r in d['review_claims'] if r['business_id'] in reused] for v,d in versions.items()},
        'AI_second_opinion':True,'human_verified':False})
    zero=[x for x in audit['zero_caption_cases'] if x['business_id']=='EzJI5x25U5GgbYBWhu5ZQQ' and x['condition']=='filtered']
    prioritized=[f for f in findings if f['finding_id'] in ['ASPARAGUS','TWO_DRINKS','GREEN_BEANS','GLASS_WATER','HOTDOG','BOAT','PIZZA','LARGE_HAMBURGER','FORK_HOLDER','WRAPPED_CUTLERY','PIZZA_PLATE_TRAY']]
    priority_table=[{'finding_id':f['finding_id'],'business_id':f['business_id'],'photo_id':f['photo_id'],
        'assessment_classification':f['classification'],
        'old_claims':[{'condition':r['condition'],'claim':r['text'],'verdict':r['verdict'],'caption_id':r['caption_id'],'photo_marker':r['photo_marker']} for r in f['evidence']['D039']],
        'new_claims':[{'condition':r['condition'],'claim':r['text'],'verdict':r['verdict'],'caption_id':r['caption_id'],'photo_marker':r['photo_marker']} for r in f['evidence']['D040']]} for f in prioritized]
    summary={
        'AI_assessed_improvements':['Asparagus removed from visual target; white plate/knife retained.',
            'Exact two-drinks count removed; generic drink/table/restaurant retained.',
            'Fresh UC review removes false seafood umbrella, preserves praise and meaningful contrasts.'],
        'AI_remaining_concerns':['Green-bean identity verified in both conditions; chicken identity remains ambiguous.',
            'Water U uncertain/F verified remains a semantic confidence contradiction with changed compound wording.'],
        'AI_new_regressions_or_concerns':['Previously accepted pizza control rejected under excessive taxonomy precision.',
            'Wrapped bunch-of-silverware returns from uncertainty to unsupported confident verification.',
            'Foreground pizza plate/tray rejection in F and drink-plurality differences need ordinary-description review.'],
        'AI_preserved_controls':['Large hamburger accepted in both conditions.','Hot-dog and boat identities remain rejected.',
            'Unseen fork-holder excluded in both conditions.','Zero-Filtered header preserved without fabricated visual evidence.',
            'Correct simple pie/plate/checkered-cloth and ordinary dining descriptions retained.'],
        'zero_literal_conflicts_does_not_prove_semantic_consistency':True,
        'rejected_vs_uncertain_caution':'Two-drink exact count exclusion is correct for target safety, but rejected can overstate a genuinely cropped/ambiguous case; report without relabeling.'}
    report={'status':'D040_BOUNDED_COMPARISON_FOR_RESEARCHER_REVIEW_NOT_PRODUCTION_APPROVAL',
        'role':'AI actual-photo/source-grounded second opinion, not independent human ground truth',
        'completion':completion,'automated_integrity_summary':{k:v for k,v in audit.items() if k not in ['task_slots','artifact_sha256']},
        'assessment_summary':summary,'priority_comparison_table':priority_table,'findings':findings,'exact_cross_condition':machine['exact_cross_condition'],
        'zero_filtered_behavior_preserved':bool(zero),'fresh_review_businesses':[bid],
        'reused_review_businesses':reused,'actual_original_photos_inspected':45,
        'review_semantic_scope':'all11 fresh UC selected reviews; all7 inherited consensus texts category-screened; prior source audits preserved; not every quote of all7 re-audited semantically',
        'automatic_retries':0,'manual_model_target_or_verdict_edits':False,
        'human_labels_changed':False,'production_protocol_frozen':False,
        'no_test_or_full_generation_or_Stage_F_change_or_LED_training':True,
        'warning':'Changed claim granularity, stochastic gateway generation and diagnostic selection prevent causal proof/population error-rate claims.'}
    immutable(NEW/'comparison_report.json',report)
    lines=['# D-040 focused comparison after D-039 — 2026-10-11','',
        '**Researcher review required; production is not frozen.** Automated validation checks schema, source binding and serialization. Factual assessments below are AI second opinions from actual photos/reviews, not human ground truth.','',
        f"Completed {completion['businesses_completed']}/8 businesses, {completion['target_texts']} target texts; {completion['new_API_request_intents']} new request intents. Reused reviews keep their actual old prompts. No manual target/verdict editing, no automatic retry, no Stage F or test/training execution.",'',
        '## Assessment summary','']
    for category in ['AI_assessed_improvements','AI_remaining_concerns','AI_new_regressions_or_concerns','AI_preserved_controls']:
        lines+=['**'+category+'**','']+[ '- '+x for x in summary[category]]+['']
    lines+=[summary['rejected_vs_uncertain_caution'],'',
        '## Priority old/new comparison','',
        '| Case | Business / photo | D039 | D040 |','|---|---|---|---|']
    for row in priority_table:
        old='; '.join(x['condition']+': '+x['verdict']+' — '+x['claim'] for x in row['old_claims']) or 'not emitted'
        new='; '.join(x['condition']+': '+x['verdict']+' — '+x['claim'] for x in row['new_claims']) or 'not emitted'
        lines.append('| '+row['finding_id']+' | `'+row['business_id']+'` / `'+row['photo_id']+'` | '+old.replace('|','\\|')+' | '+new.replace('|','\\|')+' |')
    lines+=['','Different claim wording is shown rather than assigned an artificial one-to-one match. Exact REV/IMG_CAP/PHOTO IDs and verbatim evidence appear below.','',
        '## Automated integrity','', '```json',json.dumps(report['automated_integrity_summary'],indent=2),'```','',
        '## Matching photo/caption/claim consistency','']
    for name,check in machine['exact_cross_condition'].items():
        lines.append(f"- {name}: {check['conflict_groups']} literal conflicts / {check['matched_groups']} matched groups. Different decomposition changes matching opportunities; these are not error rates.")
    lines+=['','No verdicts are reconciled. Semantic paraphrases/compound versus atomic clauses are reviewed separately.','',
        '## Evidence-linked factual assessment','']
    for f in findings:
        lines += ['### '+f['finding_id'],'',f"Business: `{f.get('business_id', ', '.join(f.get('business_ids',[])))}`. Classification: `{f['classification']}`.",'',f['assessment'],'']
        if f.get('photo_id'):lines += [f"Original photo: `{f['photo_id']}`.",'']
        for name,rows in f['evidence'].items():
            lines+=['**'+name+'**','']
            if not rows:lines+=['No matching emitted claim (not an automatically assigned verdict).','']
            for row in rows:
                if 'verdict' in row:
                    lines += [f"- `{row['caption_id']}` / `{row['photo_marker']}` / photo `{row['photo_id']}`: **{row['verdict']}** — {row['text']}",
                        '  Caption: '+json.dumps(row['caption'],ensure_ascii=False),
                        '  Exact caption quote: '+json.dumps(row['caption_quote'],ensure_ascii=False),
                        '  Teacher explanation: '+row['explanation']]
                else:
                    lines += [f"- `{row['business_id']}` [{row['polarity']}] {row['text']}"]
                    for support in row['support']:
                        lines.append(f"  `{support['review_marker']}` / original review `{support['review_id']}`: "+json.dumps(support['quote'],ensure_ascii=False))
            lines+=['']
    lines+=['## Limits and stop','',
        'All45 mapped original images were actually opened. All11 selected UC reviews were read in full; all7 inherited consensus texts were screened for category scope. Imported review outputs are unchanged and do not evaluate reviewv7. Detailed image notes and machine comparison preserve the broader evidence beyond selected findings. Correct simple content is separate from errors. No semantic diversity/Stage F change, K/threshold retuning, new human labels, held-out teacher references,2200production or LED training. Return this report for researcher review; do not infer production approval from structural PASS.','']
    text='\n'.join(lines);path=NEW/'comparison_report.md'
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:path.write_text(text,encoding='utf-8')
    print(json.dumps({'comparison_report':'CREATED','AI_findings':len(findings),'production_frozen':False},indent=2))


if __name__=='__main__':run()
