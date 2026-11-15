"""Evidence-linked AI second-opinion report for the completed D-039 bounded pilot; no API."""
import json
import re
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, immutable

NEW=ROOT/'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'


def run():
    machine=read(NEW/'machine_comparison.json');audit=read(NEW/'independent_integrity_report.json')
    completion=read(NEW/'execution_v1/completion.json');versions=machine['versions'];findings=[]

    def visual(fid,bid,pid,classification,reason,pattern=None):
        evidence={name:[r for r in data['visual_claims'] if r['business_id']==bid and r['photo_id']==pid
                        and (pattern is None or re.search(pattern,r['text'],re.I))] for name,data in versions.items()}
        assert evidence['D039'], 'Missing current claim evidence for '+fid
        findings.append({'finding_id':fid,'business_id':bid,'photo_id':pid,'classification':classification,
            'reason':reason,'evidence':evidence,'AI_second_opinion':True,'human_verified':False})

    def review(fid,bid,classification,reason,pattern):
        evidence={name:[r for r in data['review_claims'] if r['business_id']==bid and re.search(pattern,r['text'],re.I)] for name,data in versions.items()}
        assert evidence['D039']
        findings.append({'finding_id':fid,'business_id':bid,'classification':classification,'reason':reason,
            'evidence':evidence,'AI_second_opinion':True,'human_verified':False})

    visual('HOTDOG_REMOVED','8xTHtLoNIwdpf0FEvIpQIw','YGPmXXkzu6x_OifETfXQfw','acceptable_improvement',
        'Actual photo shows a shredded-meat filled roll, not an established sausage hot dog. D038 verified three hot-dog clauses; D039 rejects identity and retains caption-recoverable generic food/wrapping/table claims without substituting a photo-only dish name.')
    visual('BACKGROUND_PIZZA','8xTHtLoNIwdpf0FEvIpQIw','tc0o8BwiiVoATMpSsUs5Hw','unsupported_claim_excluded_semantic_verdict_difference',
        'Foreground pizza is ordinary and supported. Background looks like stacked rolls/sandwich bread, not clearly pizzas. D039 excludes other-pizzas assertion in both conditions, but U rejected versus F uncertain remains a semantic verdict difference; distinct wording is not a literal match.')
    visual('ASPARAGUS_REMAINS','bLJP0fVgscuUQy0hE_CRXA','pbVvz17fpqf6m72XIelpZw','unsupported_specific_ingredient_remaining',
        'White plate and knife are supported. Leafy greens with thick pale-green stems do not show clear asparagus identity; D039 still verifies asparagus and describes visible spears. The D038 regression remains. No AI or human label was changed.',r'asparagus')
    visual('BOAT_REMOVED','UCzFaBbPDKtHTWsBE5LGtQ','gpeR5DxyJN-eIduZEoMZIg','acceptable_improvement',
        'Original photo has a basin/toilet-like fixture beside road and tree, not a recognizable boat. D039 rejects the boat identity and retains generic object placement recoverable from the caption; no replacement toilet fact enters target.')
    visual('GREEN_BEAN_CONFLICT','FFV8_b-ec5Z0E3xgvkVxKw','H94oFnARbFwDwWAluQmjNA','unsupported_or_uncertain_specific_ingredient_with_exact_verdict_conflict',
        'Same photo/hash/caption and exact claim The pasta dish contains green beans. D038 U rejected/F verified; D039 U uncertain/F verified. Visible green stalk pieces do not establish green beans. Filtered verification remains concerning and the literal contradiction remains; not reconciled.',r'green beans')
    visual('CHICKEN_AMBIGUITY','FFV8_b-ec5Z0E3xgvkVxKw','H94oFnARbFwDwWAluQmjNA','uncertain_identity_review_needed',
        'Coated protein pieces among pasta are visible, but chicken identity is not secure from this image. Both D039 conditions verify it. Separate this ambiguity from the unmistakably supported pasta/white plate; not independently human confirmed.',r'chicken')
    visual('FORK_HOLDER_EXCLUDED','PRrWtQ3rdINngwESZNefVQ','qcXO3blRqZmre1xVVq1soQ','acceptable_preserved_and_verdict_consistency_improved',
        'Fork above food is visible but no hand/person holding it is shown. D039 both conditions mark exact holder claim uncertain and exclude it; ordinary fork-over-plate/close-up claims remain verified. D038 rejected/uncertain difference becomes agreement.',r'person|fork|close')
    visual('PIZZA_ACCEPTABLE','nv59KN19fEQax9RL0lUdkw','6rheOeS8NRwjxVU-iG92kw','acceptable_preserved',
        'Round sliced cheese-topped flatbread supports ordinary large-pizza interpretation. D039 keeps it verified; do not call this a new improvement because original and D038 already accepted it.')
    visual('LARGE_HAMBURGER_ACCEPTABLE','jMStvE-tQzSpRCAO0nAE6g','HTP0pKcn4dbJoDVQAp9TFQ','acceptable_preserved',
        'Stacked substantial burger visibly supports qualitative large without a ruler. Both D039 conditions verify it and visible caption-listed lettuce/tomato/cheese/red background. No extra photo-only topping inserted.',r'large hamburger')
    visual('QUOTATION_FAILURE_RESOLVED','pWT3WAUvww4iqF-v2htBXA','-_seY1mjA62XKXwi5FZGBQ','schema_source_quote_failure_resolved_without_manual_edit',
        'D038 failed exact substring evidence, not JSON type/field schema: punctuation inside the rejected OCR quote was lost. D039 copies that says "bocconor." exactly, passes the unchanged guard, rejects unsupported OCR and retains the visible sign/brick-wall claim. The missing business now has a complete target set. No failed response promoted or quote edited.')
    visual('TWO_DRINKS_CONCERN','bLJP0fVgscuUQy0hE_CRXA','ajfOEJJQ6D8AN1aIxY53vA','unsupported_count_review_needed',
        'One tall drink is clearly visible. Teacher D039 claims a second partially visible drink at left; edge form is not reliably identifiable in actual original photo. Specific count verification needs human review; generic drink/table/restaurant description is ordinary.')
    visual('WRAPPED_SILVERWARE','jMStvE-tQzSpRCAO0nAE6g','FNtGtXamR9Uduv85WIcf7g','uncertain_handling_improved',
        'Opaque/wrapped takeout contents do not clearly establish multiple cutlery pieces. D039 marks bunch-of-silverware uncertain while preserving visible close-up/plastic bag, avoiding D038 unsupported verified plurality.')
    visual('MEN_POSTURE_AND_FOOD','FFV8_b-ec5Z0E3xgvkVxKw','4-M3iicg2AFOIBlsC_dyzw','semantic_verdict_differences_review_needed',
        'Three men and drinks are visible. Full seated posture is partly hidden; no food is clearly discernible. D039 still gives U uncertain/F verified for all-seated interpretation and U rejected/F uncertain for food presence. Different text forms are reported as semantic differences, not silently grouped/forced.',r'sitt|food')
    visual('WINE_INTERPRETATION','FFV8_b-ec5Z0E3xgvkVxKw','Gs2aQAsJWgh7GIlHbRox9g','uncertain_versus_ordinary_description_discussion',
        'Small pizza and stemmed glass are supported. D039 wine is uncertain; this may be a reasonable caution about liquid identity or excessive strictness for ordinary wine appearance. Researcher interpretation remains open, not counted as a proven error.',r'glass|wine')
    visual('TORTILLA_TAXONOMY','nv59KN19fEQax9RL0lUdkw','3ZtMuD2FC6mgo45gADLvWQ','uncertain_ordinary_food_category_discussion',
        'Bowl/plate clear; accompanying triangular flatbread has naan-like appearance. D039 tortilla label uncertain. Ordinary-food taxonomy needs researcher interpretation; no photo-only replacement label used.',r'tortilla')
    visual('TWO_FOOD_PLATES','PRrWtQ3rdINngwESZNefVQ','Mry2XDZYS_rKCGy5b9ZNJw','acceptable_preserved',
        'One plate carries food directly and the other carries a food bowl. Ordinary two-food-bearing-plates interpretation remains supported, not rejected merely because one has a bowl.')
    visual('CORRECT_SIMPLE_CONTENT','UCzFaBbPDKtHTWsBE5LGtQ','-uGxVqv-Dh4OFaCqFMHMgQ','correct_but_simple',
        'Pie/plate/checkered cloth are supported ordinary observations. Low information density is not factual failure; generic scene subclaims throughout pilot are kept separately from error findings.')
    visual('OPEN_PIANO','EzJI5x25U5GgbYBWhu5ZQQ','PvPlFCrN8HpiIt1hxY5Phg','ordinary_description_precision_review_needed',
        'Black upright piano exposes upper-front mechanism. Normal open-piano interpretation can be acceptable; distinguish generic open state from an overly precise top-panel assertion. See exact D039 verdict without treating every material/panel ambiguity as error.')
    visual('STRAW_EXISTING_SCOPE','EzJI5x25U5GgbYBWhu5ZQQ','HVkUbl7niy2hIADyPh17MA','ordinary_supported_claim_existing_original_human_confirmation',
        'Plastic cup/straw/stone-counter description is reasonable. User already confirmed straw only in the original pilot. Original annotation file unchanged; D039 judgments are still AI, no automatic human-final propagation.',r'straw|cup|counter')
    review('FOOD_CATEGORY_AND_FLAVOR','PRrWtQ3rdINngwESZNefVQ','review_category_improved_contrast_restored',
        'D038 incorrectly grouped plantain chips under corn-based items. D039 guacamole/plantain claim keeps separate category and restores minority bland-guacamole opinion (REV16); separate flavor/elote/steak mixed claim cites mild/non-spicy praise and bland/ketchup criticism. Only selected reviews used.',r'guacamole|corn|plantain|flavor|bland|heat')
    review('BYOB_VALUE_RESTORED','PRrWtQ3rdINngwESZNefVQ','review_contrast_restored',
        'D039 reinstates corkage objection REV2 versus mixer-waiver REV8/10; value mixed claim keeps positive value versus pricey/lunch concerns REV13/27. These are scoped review reports, not asserted current policy.',r'BYOB|corkage|value|price')
    review('TORTILLA_PRAISE_OMISSION','PRrWtQ3rdINngwESZNefVQ','review_coverage_difference_not_false_fact',
        'D039 restores negatives/policy contrasts and many distinct topics but omits recurring handmade-tortilla praise in REV6/28, which D038 included. This is a positive-topic coverage tradeoff, not invented fact or proof that every opinion must fit. Target-length increase is diagnostic, not grounds for truncation.',r'guacamole|corn|plantain|tortilla')
    review('IMPORTED_SEAFOOD_CATEGORY','UCzFaBbPDKtHTWsBE5LGtQ','persistent_imported_review_category_concern_not_new_v6_test',
        'The reused v5 claim groups gator bites/grits/pies/onion rings under seafood. Those category memberships are not supported as seafood; same wording remains in D038 and D039 because this review component was imported unchanged. Report limitation, not a new v6 regression or silently regenerate it.',r'seafood')
    review('ZERO_CONTROL_REVIEW_CONTRAST','EzJI5x25U5GgbYBWhu5ZQQ','review_control_semantic_check',
        'All30selected reviews inspected: mostly food/coffee/staff praise, with bland tacos REV3, average tacos REV4, tepid chocolate REV9, self-service preferences REV15, unfinished decor REV20 and chicken decline REV30. Check exact new claims/quotes for qualified minority retention; do not infer all customers or invent criticism.',r'.')
    report={'status':'D039_BOUNDED_COMPARISON_RESEARCHER_REVIEW_PENDING_NOT_PRODUCTION',
        'role':'AI_second_opinion_not_independent_human_ground_truth','completion':completion,
        'integrity_summary':{k:v for k,v in audit.items() if k!='artifact_sha256'},'findings':findings,
        'exact_cross_condition':machine['exact_cross_condition'],
        'review_semantic_scope':{'complete_selected_reviews_read':['PRrWtQ3rdINngwESZNefVQ','EzJI5x25U5GgbYBWhu5ZQQ'],
            'review_v6_tested_businesses':2,'other_review_components_imported_unchanged':7,
            'not_exhaustive_entailment_for_all_imported_business_reviews':True},
        'actual_original_photos_inspected':50,'whole_businesses_human_verified':0,
        'production_protocol_frozen':False,'additional_calls_after_bounded_pilot':0,
        'no_manual_target_or_label_edits':True,'no_test_or_Stage_F_or_LED_changes':True,
        'interpretation':'Two strong ingredient grounding concerns and remaining semantic/count ambiguities prevent automatic production approval. Targeted, single-output comparison is not a population error estimate or causal prompt study.'}
    immutable(NEW/'comparison_report.json',report)
    lines=['# Stage G D-039: focused post-D038 pilot comparison',
        '', '**Return for researcher review; do not freeze production.** Hot-dog/boat identities and PR review-category/contrast issues improve; unsupported asparagus and green-bean verification remain. Existing captions/reviews/labels and all old pilots are unchanged.',
        '',f"- Scope: {completion['businesses_attempted']} businesses; {completion['businesses_completed']} complete sets / {completion['target_texts']} provisional target texts.",
        f"- Calls: {completion['new_API_calls']} new calls, seven unchanged v5 review imports, one deterministic empty Filtered task. No full20 rerun, no automatic retries.",
        '- Only two businesses test reviewv6. The seven imported review components must not be described as new-v6 results.',
        '- Same requested/reported model alias and null immutable-snapshot status; settings/schema/validators/source inputs unchanged. No independent gateway backend attestation.',
        '', '## Automated integrity (separate from factual assessment)', '',
        f"Status: {audit['status']}; {audit['request_SHA_checks']} request hash checks, {audit['exact_source_span_checks']} exact source spans, {audit['slow_native_LED_target_ID_checks']} slow/native LED target-ID matches. {audit['original_files_preserved']} historical/source files preserved. Source IDs/text/order/images unchanged; no truncation, quote repair or manual target edit.",
        '', 'D038 blocked pWT quotation was an exact-substring evidence failure, not malformed JSON/schema. Revised response passes the unchanged validator, including the period inside the rejected OCR quote. No old failed result was promoted.',
        '', '| Condition | Min | Mean | Max | >512 (diagnostic only) |','|---|---:|---:|---:|---:|']
    for c,s in audit['target_tokens'].items():
        lines.append(f"| {c} | {s['min']} | {s['mean']:.2f} | {s['max']} | {audit['over512_diagnostic_only'][c]} |")
    lines+=['','512 is NOT a frozen target cap. No target is shortened or discarded. Source5000/review4000/caption750 unchanged. Review completeness/target length remains a production decision, separate from source budgets.',
        '', '## Exact cross-condition verdicts', '', '| Version (same diagnostic9) | Literal shared groups | Conflicts |','|---|---:|---:|']
    for name,c in machine['exact_cross_condition'].items():lines.append(f"| {name} | {c['matched_groups']} | {c['conflict_groups']} |")
    lines+=['','Different sentence decompositions change match counts. Do not report these fractions as correctness rates. Green-bean contradiction remains; background pizza/posture/food-presence semantic differences are separately listed. No verdict is silently reconciled.',
        '', '## Evidence-linked AI assessment', '',
        'All50mapped original JPEGs were actually opened. Findings are AI second opinions, not independent human ground truth. The two fresh-review businesses were checked against all30selected reviews each; other imported review components are not exhaustively re-annotated. Correct-but-simple content is separate from errors.']
    for f in findings:
        lines+=['',f"### {f['finding_id']} — {f['business_id']}",'',f"**{f['classification']}** — {f['reason']}"]
        if f.get('photo_id'):lines.append(f"Photo: `{f['photo_id']}` ([original JPEG](../../../data/processed/yelp_2300_v1/photos/{f['photo_id']}.jpg)).")
        for version,rows in f['evidence'].items():
            lines+=['',f'**{version}**']
            if not rows:lines.append('- No accepted matching component/claim in this version; inspect machine inventory and historical failed responses.');continue
            for r in rows:
                if 'condition' in r:
                    lines.append(f"- {r['condition']} / [{r['caption_id']}] / [{r['photo_marker']}] / `{r['photo_id']}` / claim{r['claim_index']+1} / **{r['verdict']}**: {r['text']}")
                    lines.append(f"  - Exact caption quote: `{r['caption_quote']}`")
                    lines.append(f"  - Teacher explanation (AI): {r['explanation']}")
                else:
                    lines.append(f"- Review claim{r['claim_index']+1}, {r['polarity']}: {r['text']}")
                    for s in r['support']:lines.append(f"  - [{s['review_marker']}] / `{s['review_id']}`: {s['quote'].strip()}")
    lines+=['','## Zero-caption behavior, preservation and next decision', '',
        '- EzJI5x25U5GgbYBWhu5ZQQ Filtered remains zero captions: shared review consensus and empty visual header; no image call or fabricated evidence. Old user-confirmed straw annotation is preserved in its original scope; no new human verification is inferred.',
        '- Strong unresolved concerns: asparagus U and green beans F remain verified despite weak/contradictory specific-identity evidence; two-drinks count and chicken/posture/ordinary-category interpretations need researcher inspection.',
        '- Original and D038 targets/claims/verdicts/prompts/audits/human labels remain unchanged. No new annotation import, Stage F change, held-out teacher references, full2200 generation or LED training.',
        '- Stop here. Researcher should review remaining ingredient/count ambiguity, generic-versus-specific naming, contrasting-opinion coverage and target-length tradeoffs before approving any production protocol. No repeated regeneration to obtain desired verdicts.',
        '',f"New-call reported usage: {audit['native_usage']['input_tokens']:,} input / {audit['native_usage']['output_tokens']:,} output tokens ({audit['native_usage']['reasoning_tokens']:,} reasoning included). Imported-call usage excluded; this is not a gateway invoice.",
        '', 'API design reference only: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses) distinguishes schema conformance from model mistakes. This official guidance does not attest the third-party gateway backend or verify this pilot\'s facts.']
    md=NEW/'comparison_report.md'
    if md.exists():raise FileExistsError('Preserve completed report; new version required')
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Freeze the exact pilot artifacts needed to reproduce/check results without credentials.
    snapshot=NEW/'code_and_protocol_snapshot'
    for relative in ['configs/stage_g_teacher_pilot_v5_focused_d039.json','prompts/stage_g_visual_evidence_v3.txt',
                     'prompts/stage_g_review_consensus_v6.txt','prompts/stage_g_review_consensus_v5.txt',
                     'schemas/stage_g_teacher_pilot_v3.json','scripts/run_stage_g_d039_focused_pilot.py',
                     'scripts/run_stage_g_teacher_pilot_v4_1_recovery.py','scripts/audit_stage_g_d039_focused_pilot.py',
                     'scripts/finalize_stage_g_d039_comparison.py','scripts/stage_g_canonical_review_ids.py',
                     'scripts/stage_g_visual_target_serialization.py']:
        dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert file_sha(dest)==file_sha(ROOT/relative)
        else:dest.write_bytes((ROOT/relative).read_bytes())
    immutable(NEW/'code_snapshot_manifest.json',{'files':{p.relative_to(snapshot).as_posix():file_sha(p) for p in snapshot.rglob('*') if p.is_file()},'credentials_included':False})
    print(json.dumps({'findings':len(findings),'status':report['status'],'report':md.relative_to(ROOT).as_posix()},indent=2))


if __name__=='__main__':run()
