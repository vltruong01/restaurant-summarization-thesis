"""Evidence-linked B assessment; analysis only, no changes to source or generated text."""
import hashlib
import json
from pathlib import Path
import re

from stage_i_decoding_diagnostics import read_jsonl, sections, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1'
PREP = ROOT / 'outputs/stage_i/decoding_diagnostic_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    comparisons = read(OUT / 'baseline_A_B_cases.json')
    bykey = {(c['business_id'], c['condition']): c for c in comparisons}
    originals = {(c['business_id'], c['condition']): c for c in read_jsonl(PREP / 'cases.jsonl')}
    notes = []

    def add(bid, condition, classification, quote, explanation, cap_ids=(), rev_ids=()):
        c = bykey[bid, condition]
        assert quote in c['beam_ngram_B'], (bid, condition, quote)
        reviews = []
        for rid in rev_ids:
            match = re.search(r'\[' + re.escape(rid) + r'\] (.*?)(?=\n\n\[REV_\d+\]|\Z)', c['source'], re.S)
            assert match, rid
            reviews.append({'REV': rid, 'original_review_id': originals[bid, condition]['selected_review_ids'][int(rid.split('_')[1])-1],
                            'full_selected_review_with_metadata': match.group(1)})
        is_ro = condition == 'reviews_only'
        notes.append({'business_id': bid, 'condition': condition, 'subset': c['subset'],
                      'classification': classification, 'B_exact_excerpt': quote, 'explanation': explanation,
                      'baseline_relevant_section': sections(c['baseline'])[0 if is_ro else 1],
                      'A_relevant_section': sections(c['greedy_A'])[0 if is_ro else 1],
                      'B_relevant_section': sections(c['beam_ngram_B'])[0 if is_ro else 1],
                      'B_full_output_including_literal_headers': c['beam_ngram_B'],
                      'teacher_relevant_section': sections(c['teacher'])[0 if is_ro else 1],
                      'caption_photo_evidence': [cap for cap in c['captions'] if not cap_ids or cap['caption_id'] in cap_ids],
                      'review_evidence': reviews,
                      'human_verified': False, 'original_photo_independently_verified': False,
                      'provenance': 'Codex AI assessment of original unchanged sources/outputs; not human-final annotation'})

    add('HhD8DLES5ZeULKDXmfkOOg', 'reviews_only', 'SPECIFIC_REVIEW_INFORMATION_STILL_OMITTED',
        'Food opinions are mixed:',
        'B still falls back to generic food/service/value statements and adjacent-word lists. REV_4 and REV_23 explicitly describe sports/TVs and beer selection; those distinguishing facts and the specific food/wait-time contrast are not recovered. Lower exact repetition is not evidence of recovered review coverage.', rev_ids=('REV_4','REV_23'))
    add('gpSv4OvISIeyBAqcNN5bqw', 'reviews_only', 'A_CATEGORY_ERROR_DISAPPEARS_BUT_SPECIFICITY_NOT_RECOVERED',
        'some reviewers praise the pizza',
        'A groups buffalo chicken pasta under pizza despite REV_10 distinguishing them. B no longer states that category error, but retreats to generic pizza/service wording and continues bland/rude/attentive lists. This is local error removal, not a complete informative repair.', rev_ids=('REV_10',))
    add('ZhA88ILcgEP9DNwZDx7f0A', 'reviews_only', 'INCORRECT_REVIEW_CATEGORY_AND_SENTIMENT_BINDING',
        'Selected reviewers frequently praise the music, including live music, beer, and food,',
        'Beer and food are not subtypes of music. REV_6 separates drinks from music, REV_10 criticizes music/service, and REV_22 contrasts music with mediocre/unwarmed food. B conflates domains and later applies overpriced/underpriced to music; facts and contrasts are not faithfully organized.', rev_ids=('REV_6','REV_10','REV_22'))
    add('HhD8DLES5ZeULKDXmfkOOg', 'filtered', 'LOCAL_INFORMATION_GAIN_WITH_CAPTION_SUPPORT',
        'A white plate is topped with shrimp and grits.',
        'B restores the shrimp/grits identity lost by A and adds people-at-bar and the beer dispenser. Captions5/2/4 support these useful details. The sports-bar interpretation is reasonable for people watching TVs in a bar; it is not rejected for being ordinary. Baseline repeats only the plate identity; B is a local gain, not proof that all businesses improved.', cap_ids=('IMG_CAP_2','IMG_CAP_4','IMG_CAP_5'))
    add('aZN5R7Xfea0GkDBwBZOQrQ', 'filtered', 'CORRECT_SIMPLE_LOCAL_IMPROVEMENT',
        'A glass of wine is on the table.',
        'Baseline and A repeat the same plate-on-table sentence. B states the supported fish/chips and table facts once and adds the wine detail from caption2. Correct simple content counts as valid and locally more useful, even though it is not elaborate.', cap_ids=('IMG_CAP_2','IMG_CAP_3'))
    add('noUma64stIOSYAQ1GIIFjw', 'filtered', 'CONCISE_SUPPORTED_BUT_PARTIAL_VISUAL_COVERAGE',
        'A person is holding a plate of food.',
        'B keeps two ordinary caption4-supported statements and removes the Baseline/A plate/table/drink repetition. No unsupported detail appears in this short visual section under caption-level review. Other available bar/drink/sandwich information is omitted; factual precision and pool coverage are different questions.', cap_ids=('IMG_CAP_4',))
    add('thlyktt2c7v1DyOpSVSibw', 'unfiltered', 'NEW_UNSUPPORTED_OBJECT_RELATION_AND_PERSISTENT_IDENTITY_ERROR',
        'A dog is with a spoon in it.',
        'None of the five captions mention a dog; caption4 describes a spoon in a bowl. Baseline focuses on a sandwich and A has a spoon/bowl without a dog. B introduces this new unsupported relation. Its blueberry muffin also exceeds caption5, which says cereal/yogurt with a blueberry. That muffin error already existed in A; distinguish new from persistent errors.')
    add('b1pdnbYvBdtTxJOOom5WmA', 'unfiltered', 'CROSS_PHOTO_OBJECT_BINDING_ERROR',
        'A bowl of sweet potato fries sits on a table. The bowl contains rice and meat.',
        'Caption1 describes bowls of sweet potato fries; captions2/4 separately describe rice/meat bowls. B binds these to the same bowl with no supplied-caption support for that combination. Multiple supported object names do not establish their asserted relationship.', cap_ids=('IMG_CAP_1','IMG_CAP_2','IMG_CAP_4'))
    add('-Wd5KqFszREg0XdXfSv5tg', 'unfiltered', 'SUPPORTED_POURING_PLUS_UNSUPPORTED_ACTION_AND_COUNT',
        'There are three people in line.',
        'No caption supports three people in a line; three refers to drinks in caption5. The man pouring coffee is supported by caption2, but the woman holding a cup is stronger than caption1 sitting with a cup. B corrects neither every action nor count merely by adding varied wording.')
    add('3v0tSwHXR9LhpsoGXLgsTA', 'unfiltered', 'UNSUPPORTED_RELATIONS_AND_UNRESOLVED_CONTRADICTION',
        'The wooden bench is sitting on a table.',
        'Caption1 places a bench in front of a wall, not on a table. The captions show restaurant decor/empty seating, not people filling a bar. B says the bar area is filled with people and then the bar is empty without distinguishing images/referents. This is unsupported/coherence failure; different real photos would not automatically make all occupancy claims contradictory.')
    add('p7-1xG891NdGrcYBMOvxpw', 'unfiltered', 'WRONG_VISUAL_OBJECT_LOCATION_PERSISTS',
        'A fish is sitting on a wooden bar.',
        'Caption3 places the fish on the wall behind a woman; caption2 describes bottles on a wooden bar. B avoids A\'s woman-holding-fish action but creates a fish-on-bar relationship. Changing one unsupported action to a different unsupported location is not reliable grounding.', cap_ids=('IMG_CAP_2','IMG_CAP_3'))
    add('gpSv4OvISIeyBAqcNN5bqw', 'unfiltered', 'UNSUPPORTED_REVIEW_TO_VISUAL_FOOD_IDENTITY',
        'The photo is a close-up of a pizza.',
        'None of the five supplied captions identifies a pizza: they describe people/counter, a menu, beer and bowls of food. Pizza is in the reviews, which cannot supply an IMG_CAP visual identity. This is a wrong evidence basis; the original photos have not independently been verified.')
    add('noUma64stIOSYAQ1GIIFjw', 'unfiltered', 'SUPPORTED_FORK_WITH_NEW_UNSUPPORTED_SPOON',
        'The plate of food has a spoon sticking in it.',
        'Caption5 supports a fork sticking into food, and caption4 supports chopsticks. No caption supports a spoon. B retains A\'s valid fork gain but adds an unsupported spoon and several paraphrases of the same plate/utensil relation. BPE4 suppression does not remove semantic redundancy.')
    add('JcdOQvvx3BqZ5Ga2_9n_Bg', 'filtered', 'PERSISTENT_UNSUPPORTED_CROSS_OBJECT_WINE_RELATION',
        'The wine is on a plate.',
        'Caption3 describes red wine poured into a glass of white wine; captions1/5 describe food on a plate/in a bowl. The plate/red-liquid and wine-on-plate assertions are not supported by those mappings. The word wine is supported; the relationship is not.')
    add('j9fzpALNCgvDwCXmA5Vcjw', 'filtered', 'A_INGREDIENT_ERROR_REMOVED_WITH_DIFFERENT_UNSUPPORTED_IDENTITY',
        'The photo shows a close-up of a biscuit.',
        'B drops A\'s bowl-topped-with-bergamot claim, but none of these Filtered captions identifies a biscuit. They describe a man eating, people, a coffee machine, a man at a bar, and bowls with coffee. Avoiding bergamot is a local removal, not resolution of visual evidence binding.')
    add('Zz1utCut0psbC4brBjvSEg', 'filtered', 'NEW_CROSS_OBJECT_LOCATION_BINDING',
        'The plate is on a wooden rack.',
        'Caption3 locates wine bottles in a wooden rack, while caption2 locates the meat/vegetable platter on a table. B moves the plate to the bottle rack. The separate woman-with-wine description remains supported by caption4.', cap_ids=('IMG_CAP_2','IMG_CAP_3','IMG_CAP_4'))

    # Required targeted cases are assessed on actual B output below, not forced expected answers.
    zero = bykey['j3kGicU3mPG_ciIKG1Vuyg', 'filtered']
    assert not zero['captions']
    zvisual = sections(zero['beam_ngram_B'])[1]
    add('j3kGicU3mPG_ciIKG1Vuyg', 'filtered', 'ZERO_CAPTION_FAILURE_PERSISTS',
        zvisual.split('. ')[0],
        'The Filtered input has zero captions/photos and the teacher visual section is empty. B still emits visual claims. Bagels or cream cheese in selected reviews cannot be used as visual evidence. All three settings fail the approved zero-caption behavior.')
    for condition in ['unfiltered', 'filtered']:
        c = bykey['OpnIXsrqaK-e8Wz5gco7Vw', condition]
        visual = sections(c['beam_ngram_B'])[1]
        add('OpnIXsrqaK-e8Wz5gco7Vw', condition, 'CAP_LOOP_TERMINATION_IMPROVES_WITH_UNSUPPORTED_VISUALS',
            visual.split('. ')[0],
            'Baseline loops to768 and omits its visual section. A and B terminate earlier with visual content, but the full actual supplied caption set does not support the resulting table/drink/utensil relationships. Inspect the exact three sections and mappings; format recovery is not factual repair.')
    for condition in ['unfiltered', 'filtered']:
        c = bykey['7u40cyMxw4iGosMdkesoEQ', condition]
        markers = re.findall(r'\[[A-Z_]+\]', c['beam_ngram_B'])
        add('7u40cyMxw4iGosMdkesoEQ', condition, 'HEADER_FAILURE_REMAINS_CHECK_LITERAL_MARKERS',
            markers[0],
            'B retains the incorrect first header and introduces VISUAL__EVIDENCE (double underscore) in both conditions. Actual B markers are ' + repr(markers) + '. Do not normalize or repair them; a missed parsed visual block is not proof of safe absence of visual claims. The F tail also changes a woman holding a sandwich into a group holding it. Full source/teacher/Baseline/A/B remain in the viewer.')
    c = bykey['gcQSAQIx-KQOdf5E8HF6Lw', 'filtered']
    add('gcQSAQIx-KQOdf5E8HF6Lw', 'filtered', 'TARGETED_HEADER_AND_VISUAL_BINDING_RECHECK',
        re.findall(r'\[[A-Z_]+\]', c['beam_ngram_B'])[0],
        'The wrong VISUAL_CONSENSUS review header persists and B introduces VISUAL__EVIDENCE with a double underscore. The lemon wedge is now placed on a plate although caption2 only puts it in iced tea, while a single donut becomes donuts next to each other. Do not count the unparsed visual section as empty compliance; full raw text remains preserved.')
    # Exact token arithmetic explains a constraint conflict, not a training root cause.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / 'assets/tokenizers/led-base-16384/tokenizer.json'))
    canonical = tokenizer.encode('[VISUAL_EVIDENCE]', add_special_tokens=False).ids
    header_rows = []
    for bid, condition in [('7u40cyMxw4iGosMdkesoEQ','unfiltered'),('7u40cyMxw4iGosMdkesoEQ','filtered'),('gcQSAQIx-KQOdf5E8HF6Lw','filtered')]:
        p = next(x for x in read_jsonl(ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1' / condition / 'predictions.jsonl') if x['business_id'] == bid)
        ids = p['token_ids']
        matches = [i for i in range(len(ids)-3) if ids[i:i+4] == canonical[:4]]
        assert matches and '[VISUAL__EVIDENCE]' in p['text']
        header_rows.append({'business_id':bid, 'condition':condition,
            'earlier_emitted_canonical_visual_prefix_4gram_positions':matches,
            'canonical_VISUAL_EVIDENCE_first_4_ids':canonical[:4],
            'actual_second_header':'[VISUAL__EVIDENCE]',
            'canonical_second_header_tokenization_would_repeat_prior_4gram':True})
    write_json(OUT / 'HEADER_CONSTRAINT_AUDIT.json', {'scope':'Offline token arithmetic on saved B outputs, no new inference',
        'canonical_required_visual_marker_ids':canonical,
        'explanation':'The wrong first VISUAL_CONTEXT or VISUAL_CONSENSUS header emits [ / VIS / UAL / _. A later canonical VISUAL_EVIDENCE header reuses that exact four-token prefix and is forbidden by no_repeat_ngram_size=4. Actual B uses __ as one token instead. This demonstrates a canonical-token-path conflict after an already wrong header, not that all alternative tokenizations are impossible or a proven cause of the original wrong header/learning failures.',
        'cases':header_rows})
    write_json(OUT / 'AI_QUALITATIVE_REVIEW.json', {'status': 'EVIDENCE_LINKED_ILLUSTRATIVE_AI_REVIEW_NOT_HUMAN_GROUND_TRUTH',
        'assessment_denominators': 'Automated metrics cover all72. These qualitative notes are purposive examples, not a prevalence or independently verified factual-accuracy estimate.',
        'original_photo_verification': False, 'human_final_labels_created': False, 'notes': notes})
    paired = read(OUT / 'paired_summary.json')
    ng = read(OUT / 'ngram_lexical_summary.json')
    integrity = read(OUT / 'execution_integrity_and_runtime.json')
    lines = ['# Diagnostic B - Baseline / A / beam4 no-repeat4 comparison', '',
        '**Assessment:** B suppresses exact BPE4 repetition by construction and improves some local outputs. It does not establish reliable, informative, evidence-grounded generation across this sample. Do not adopt B as the production decoder on repetition metrics alone. Review training/learning and evidence-binding limitations before proposing another experiment or retraining; no root cause is proven.', '',
        '## Execution integrity', '',
        'Exactly72 B outputs, same24 validation businesses (20 random +4 targeted; no overlap), same three existing best adapters and complete source/teacher inputs as Baseline/A. Checkpoint/dataset/source/tokenizer/code bindings and token/text roundtrips PASS. Actual full GenerationConfigs match the frozen expected settings, including beam4/do_samplefalse/no_repeat_ngram4/use_cachetrue. **Only no_repeat_ngram_size changes0->4 from Baseline.** Researcher explicitly resolved the initial conflicting wording to **max_length=768 including decoder-start**, with max_new_tokens null; no769-token cap was adopted. FP32 weights/FP16 autocast, batch1, globalBOS, seed42 and pinned backbone revision are unchanged. No fallback, truncation, manual target/output rewrite, retraining, Teacher calls, held-out data or extra decoding experiment.', '',
        '## Paired automatic results', '',
        'Triples below are **Baseline / A / B**. Random and targeted denominators remain separate. Sentence repetition uses the existing exact normalized-sentence rule; adjacent words ignore punctuation. No significance or population factual-error estimate is implied.', '',
        '| Group | Condition | Repeated sentence | Adjacent word repeat | Correct headers | EOS before cap | Cap hits | Mean tokens |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    rr = {(r['subset'],r['condition'],r['experiment']):r for r in paired['rows']}
    for role in ['random','targeted']:
        for condition in ['reviews_only','unfiltered','filtered']:
            rows = [rr[role,condition,e] for e in ['baseline','greedy_A','beam_ngram_B']]
            vals = []
            for k in ['exact_repeated_sentence_count','consecutive_repeated_word_count','section_headers_compliant_count','eos_before_length_cap_count','cap_reached_count']:
                vals.append(' / '.join(str(x[k]) for x in rows))
            vals.append(' / '.join(f'{x["length_mean"]:.2f}' for x in rows))
            lines.append(f'| {role}{rows[0]["outputs"]} | {condition} | ' + ' | '.join(vals) + ' |')
    lines += ['', 'Full min/mean/median/p90/p95/max lengths and all EOS/cap flags are in summary/per-case files. The initial decoder-start2 is not mistaken for terminating EOS. EOS at the cap may be forced and is not semantic completeness. Wrong or malformed headers are not normalized. Three previously wrong-header U/F cases now emit VISUAL__EVIDENCE with two underscores. HEADER_CONSTRAINT_AUDIT.json shows that the canonical required header shares the four-token prefix [ / VIS / UAL / _ with an already emitted wrong first visual header, blocking that canonical token path under B. This is a concrete constraint conflict after an existing header error, not proof of a training root cause or impossibility of alternative tokenizations.', '',
        '## Ngrams and lexical diversity', '',
        '| Group | Condition | Setting | BPE4 duplicate fraction | Lexical4 duplicate fraction | Distinct1 | Distinct2 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in ng['rows']:
        if row['experiment'] == 'teacher':
            continue
        lines.append('| ' + ' | '.join([row['subset']+str(row['outputs']),row['condition'],row['experiment']] +
            [f'{row[k]:.4f}' for k in ['full_BPE_including_specials_4gram_duplicate_fraction_macro_mean','lexical_4gram_duplicate_fraction_macro_mean','lexical_distinct_1_macro_mean','lexical_distinct_2_macro_mean']]) + ' |')
    lines += ['', 'BPE4 uses the exact generated IDs including specials; lexical words lowercase and ignore punctuation/headers, so the two measures differ. Distinct1 is length-dependent type-token ratio; neither high diversity nor absent exact sentences proves new supported facts. Teacher and lexical2/3/4 references are in ngram_lexical_summary.json.', '',
        '## Teacher-target agreement', '',
        '| Group | Condition | Setting | ROUGE1 F1 | ROUGE2 F1 | ROUGEL F1 | Review ROUGEL recall |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in paired['rows']:
        lines.append('| ' + ' | '.join([r['subset']+str(r['outputs']),r['condition'],r['experiment']] +
            [f'{r[k]:.4f}' for k in ['teacher_agreement_full_rouge1_fmeasure','teacher_agreement_full_rouge2_fmeasure','teacher_agreement_full_rougeL_fmeasure','teacher_agreement_review_rougeL_recall']]) + ' |')
    lines += ['', 'ROUGE uses the same CPU rouge-score/no-stemming definitions and full exact teacher references. It measures teacher-target agreement, not factual accuracy. **BERTScore not computed:** no accepted pinned scorer/backbone is installed in the reviewed evaluation runtime. No unreported metric-model choice/download or fabricated score. This is a reported evaluation limitation.', '',
        '## Information, grounding and representative evidence', '',
        'Correct simple descriptions remain acceptable. The notes below are AI source/caption assessments with human-final fields unfilled, not independent original-photo verification or a factual-error prevalence estimate. Complete sources and all three outputs are in COMPARISON.html and baseline_A_B_cases.json. Full exact section comparisons and complete selected REV / IMG_CAP / PHOTO mappings accompany each JSON note.', '']
    for n in notes:
        capref = ', '.join(f'{x["caption_id"]}/PHOTO:{x["photo_id"]}' for x in n['caption_photo_evidence'])
        revref = ', '.join(x['REV'] + '/' + x['original_review_id'] for x in n['review_evidence'])
        lines += [f'### {n["business_id"]} / {n["condition"]} ({n["subset"]})', '',
                  '> ' + n['B_exact_excerpt'], '', n['explanation'], '',
                  'Evidence: ' + (revref or capref or 'No Filtered caption/photo evidence exists.') + '', '']
    runtime = paired['runtime']
    lines += ['## Local execution measurements and next boundary', '',
        f'Local GPU: {runtime["device"]}. Pipeline {runtime["pipeline_seconds_including_model_load_monitoring_tail"]:.2f}s including load/monitor tail; summed generation {runtime["B_generation_seconds"]:.2f}s. Torch allocated/reserved peaks {runtime["torch_peak_allocated_GiB"]:.4f}/{runtime["torch_peak_reserved_GiB"]:.4f}GiB. Whole-GPU sampled peak {runtime["sampled_whole_GPU_peak_MiB"]:.0f}MiB ({runtime["samples"]} approximately5-second samples; instantaneous short peaks may be missed). No OOM, failure or retry. A/B use local RTX3050 while the saved baseline used RTX3090; no isolated cross-hardware speed or bitwise identity claim.', '',
        f'All{integrity["existing_imported_archive_files_unchanged"]} imported archive-manifest files still match their original SHA256 values. Earlier A/baseline/preparation artifacts are preserved. No production decoding method is frozen.', '',
        '**Recommendation:** B is a useful bounded diagnostic and yields some clean, short visual summaries, but semantic redundancy, generic review imitation, wrong evidence binding and zero-caption hallucination remain. New unsupported objects/relationships appear even where literal string repetition is suppressed. Do not treat lower repetition or better ROUGE as sufficient quality acceptance. Next discuss an audit of learning/training/teacher imitation and section/evidence binding; causes such as decoder-only LoRA or inadequate learning remain hypotheses. No further generation or retraining is authorized. Stop after this B report and portable review export.']
    with (OUT / 'DIAGNOSTIC_REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    import html
    viewer = (OUT / 'COMPARISON.html').read_text(encoding='utf-8')
    checks = 0
    for c in comparisons:
        for field in ['source','teacher','baseline','greedy_A','beam_ngram_B']:
            assert html.escape(c[field]) in viewer
            checks += 1
    assert viewer.count('<article>') == 288 and '<script' not in viewer.lower()
    for name in ['review_coverage_worksheet.jsonl','visual_support_worksheet.jsonl']:
        for row in read_jsonl(OUT / name):
            assert row['human_verified'] is False
    write_json(OUT / 'FINAL_INTEGRITY_CHECK.json', {
        'status':'PASS','new_B_outputs':72,'unique_case_condition_pairs':len(bykey),
        'HTML_source_teacher_Baseline_A_B_exact_text_checks':checks,'HTML_cards':288,
        'BPE4_repeated_B_outputs':integrity['full_BPE_4gram_repeated_B_outputs'],
        'all_existing_imported_archive_files_unchanged':integrity['existing_imported_archive_files_unchanged'],
        'human_labels_unfilled':True,'generation_failures_or_retries':0,
        'additional_model_experiments_training_teacher_calls_or_test_use':False,
        'grounding_notes_are_AI_not_human_final':True})
    files = [{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(OUT.rglob('*')) if p.is_file()]
    write_json(OUT / 'EVALUATION_MANIFEST.json', {'files':files,'manifest_excludes_itself':True})
    print(json.dumps({'status':'REPORT_AND_EVIDENCE_NOTES_WRITTEN','notes':len(notes),'report':str(OUT/'DIAGNOSTIC_REPORT.md')}))


if __name__ == '__main__':
    main()
