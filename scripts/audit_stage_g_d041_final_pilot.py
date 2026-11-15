"""Offline final D041 audit/export. No credentials, paid calls, or target edits."""
import base64
import csv
import hashlib
import html
import io
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, immutable
from prepare_stage_g_review_evidence_ids import evidence_spans
from stage_g_visual_target_serialization import visual_claims
from audit_stage_g_d041_recovery_v4 import FOLDER, results, tokenizer_pair

OUT = ROOT / 'outputs/stage_g/d041_final_pilot_audit_v1'
BASE = FOLDER.parent
EXE = FOLDER / 'execution_v1'
CASES = [
    ('A', 'gns-g7jjjG9LfKggL8JEQA', 'u0lWxor7lA1EGyhTHhq3dg', 'Chopsticks versus skewers'),
    ('B', 'QAlWB_JXJNQyg-kqAK_Ciw', '1_fnQjHMDEUqSsOHOPc1Rw', 'Doughnuts versus dessert cups'),
    ('C', 'UziY1NWbqeJUUEkCzlNxWw', '4sRPjfug8zyhRG9CI30-sA', 'Pulled-pork verification consistency'),
]
OPINIONS = {
    'A': ('potential_factual_identity_error',
          'The original image shows fried pieces and several short wooden implements inserted into them. '
          'This is compatible with skewers/picks; their appearance and use do not clearly establish chopsticks. '
          'The verified chopsticks identification is concerning. The ordinary fried-food description remains acceptable.'),
    'B': ('potential_unsupported_food_identity',
          'The original image shows hands holding cups with scooped dessert-like contents and spoons. '
          'No clearly recognizable doughnut shape is visible. The sign alone does not establish doughnuts. '
          'The verified doughnuts identification is concerning; cups, hands and the background sign remain ordinary supported descriptions.'),
    'C': ('conceptually_overlapping_cross_condition_verdict_difference',
          'The image shows a sandwich with shredded meat and chips. Pulled pork is a reasonable ordinary interpretation, '
          'but precise meat identity is not independently established. Unfiltered separates a generic verified sandwich '
          'from uncertain pork identity; Filtered verifies the compound pulled-pork-sandwich claim. '
          'These are different exact claims under identical caption/photo evidence, so flag conceptual inconsistency '
          'without automatically forcing agreement or rejecting this ordinary description.'),
}
GROUNDING_NOTES = [
    'The source span supports the fun-atmosphere part of the aggregate claim.',
    'The source span supports the positive staff-description part.',
    'The source span supports praise of the salsa bar.',
    'The source span supports the unfriendly-phone-interaction part.',
    'The source span supports positive sushi quality.',
    'The source span supports positive ambiance.',
    'The source spans support attentive service and improvement over previous visits.',
    'The source spans support friendly, helpful and accommodating staff.',
    'The source span supports interior design and prohibition-themed decor.',
    'The source span supports live jazz and 1920s-style swing music.',
    'The source span supports positive steak and mac-and-cheese opinions.',
    'The source span supports positive music feedback.',
    'The source spans support a takeout window and no tables, preserving a seating contrast.',
    'The first source span supports the salty-soup criticism; the salad span is supplementary and does not establish bitterness of guacamole.',
    'The source spans support negative atmosphere feedback about televisions.',
]


def snapshot():
    return {p.relative_to(ROOT).as_posix(): file_sha(p)
            for p in BASE.rglob('*') if p.is_file()}


def freeze_text(path, text):
    data = text.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'Existing audit artifact differs; use a new version'
    else:
        path.write_bytes(data)


def inventory_training():
    pattern = re.compile(r'max_target_length|DataCollatorForSeq2Seq|Seq2SeqTrainer|'
                         r'label_pad_token_id|generation_max_length|as_target_tokenizer|text_target')
    searched, matches = [], []
    for folder in ['configs', 'scripts', 'notebooks', 'cloud', '_archive_old']:
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path == Path(__file__):
                continue
            if path.suffix not in ['.py', '.yaml', '.yml', '.json', '.ipynb']:
                continue
            # Packaging manifests and derived JSON outputs are not executable training config.
            if path.name == 'package_manifest.json' or path.stat().st_size > 20_000_000:
                continue
            searched.append(path.relative_to(ROOT).as_posix())
            if path.suffix == '.ipynb':
                cells = read(path)['cells']
                blocks = [(f'cell {i}', ''.join(c.get('source', []))) for i, c in enumerate(cells)]
            else:
                blocks = [('', path.read_text(encoding='utf-8-sig'))]
            for location, source in blocks:
                for line, text in enumerate(source.splitlines(), 1):
                    if pattern.search(text):
                        matches.append({'file': path.relative_to(ROOT).as_posix(), 'location': location,
                                        'line': line, 'text': text})
    return {'scanned_files': searched, 'training_term_matches': matches,
            'active_LED_training_config_found': False,
            'max_target_length': None, 'decoder_label_preprocessing': 'not_implemented',
            'student_generation_limit': None, 'training_started': False,
            '512_role': 'diagnostic_only_and_master_plan_planned_range_not_executable_training_cap',
            'teacher_max_output_tokens': 8192,
            'teacher_cap_note': 'API output cap includes schema/evidence/reasoning and uses provider tokens; it is not the LED label limit.',
            'current_silent_training_truncation': 'none_training_not_implemented',
            'future_risk': 'Using truncation=True,max_length=512 for labels would silently cut nine current targets.',
            'smallest_recommended_correction_not_applied':
                'Keep complete labels; explicitly select a common decoder label limit covering all training/validation targets '
                'before training and fail on overflow. Current pilot needs at least575 encoded tokens including BOS/EOS. '
                '640 would cover this pilot, but is only a reviewed candidate, not frozen or guaranteed for unseen production targets. '
                'Freeze student generation limits separately and detect cap-without-EOS outputs; do not confuse max_length '
                '(total decoder sequence) with max_new_tokens. No target truncation, regeneration or teacher change is needed.'}


def audit_lengths(native, slow, jobs, flags):
    rows, all_lengths = [], []
    expected = {(x['business_id'], x['condition']): x['tokens'] for x in flags
                if x['flag'] == 'target_over512_diagnostic_not_rejection'}
    for path in sorted((EXE / 'provisional_targets').glob('*.json')):
        target = read(path)
        for condition, text in target['targets'].items():
            ids = native.encode(text, add_special_tokens=True).ids
            assert ids == slow.encode(text, add_special_tokens=True, truncation=False)
            all_lengths.append(len(ids))
            if len(ids) <= 512:
                continue
            assert len(ids) == expected[(target['business_id'], condition)]
            rows.append({'business_id': target['business_id'], 'split': target['split'], 'condition': condition,
                         'tokens_including_BOS_EOS': len(ids), 'tokens_without_specials': len(native.encode(text, add_special_tokens=False).ids),
                         'tokens_above_512': len(ids)-512, 'hypothetical_512_label_tokens_lost': len(ids)-512,
                         'current_target_truncated': False, 'target_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'review_only_tokens': len(native.encode(target['targets']['reviews_only']).ids),
                         'target_file': path.relative_to(ROOT).as_posix()})
    assert len(rows) == 9 and len(all_lengths) == 300
    return rows, {'targets': 300, 'above512': 9, 'unique_businesses_above512': len({r['business_id'] for r in rows}),
                  'max_tokens': max(all_lengths), 'tokens_include_specials': True, 'native_slow_checks': 300,
                  'no_targets_truncated': True, 'teacher_target_length_policy': read(ROOT/'configs/stage_g_teacher_extended_pilot_d041.json')['target_length_policy']}


def audit_ids(jobs, flags):
    by_business = {j['business_id']: j for j in jobs if j['condition'] == 'shared'}
    warnings = [x for x in flags if x['flag'] == 'redundant_reported_review_ID_mismatch']
    rows = []
    assert len(warnings) == 15
    for i, warning in enumerate(warnings):
        job = by_business[warning['business_id']]
        cp = read(EXE / 'checkpoints' / (job['job_id'] + '.json'))
        claim = cp['output']['claims'][warning['claim_index']]
        support = claim['support'][warning['support_index']]
        source = next(r for r in job['evidence']['reviews'] if r['review_marker'] == support['review_marker'])
        spans = {s['evidence_span_id']: s for s in evidence_spans(source)}
        quotes = []
        for span_id in support['evidence_span_ids']:
            span = spans[span_id]
            quote = source['text'][span['start_character']:span['end_character']]
            assert quote == span['exact_source_text']
            assert hashlib.sha256(quote.encode()).hexdigest() == span['text_sha256']
            assert any(p['claim_index'] == warning['claim_index'] and p['support_index'] == warning['support_index']
                       and p['evidence_span_id'] == span_id and p['review_id'] == source['review_id']
                       and p['teacher_reported_review_id'] == support['review_id'] for p in cp['evidence_resolution'])
            quotes.append(span)
        copied_from = [r['review_marker'] for r in job['evidence']['reviews'] if r['review_id'] == support['review_id']]
        assert support['review_id'] != source['review_id']
        rows.append({**warning, 'warning_number': i+1, 'claim': claim['text'], 'polarity': claim['polarity'],
                     'source_review': source, 'exact_source_spans': quotes,
                     'all_resolved_claim_support': cp['resolved_output']['claims'][warning['claim_index']]['support'],
                     'teacher_ID_matches_other_source_markers': copied_from,
                     'classification': 'auxiliary_ID_copied_from_other_review' if copied_from else 'auxiliary_ID_typographic_error',
                     'marker_and_span_source_binding': 'PASS', 'student_target_text_affected': False,
                     'evidence_grounding_error_due_to_ID_warning': False,
                     'AI_second_opinion_grounding_note': GROUNDING_NOTES[i],
                     'semantic_scope': 'Selected spans support a relevant component; an aggregate claim uses its other support too. Source-ID validation alone is not semantic certification.',
                     'human_verified': False, 'raw_output_or_resolved_evidence_modified': False})
    assert Counter(r['classification'] for r in rows) == {'auxiliary_ID_copied_from_other_review': 13, 'auxiliary_ID_typographic_error': 2}
    return rows


def visual_evidence(jobs):
    output = []
    for label, bid, pid, concern in CASES:
        cases = []
        for job in jobs:
            if job['business_id'] != bid or job['condition'] == 'shared':
                continue
            cp = read(EXE / 'checkpoints' / (job['job_id'] + '.json'))
            target = read(EXE / 'provisional_targets' / (bid + '.json'))
            for cap, photo, assessment in zip(job['evidence']['captions'], job['images'], cp['output']['assessments']):
                if cap['photo_id'] != pid:
                    continue
                original = ROOT / photo['relative_path']
                assert file_sha(original) == photo['raw_sha256']
                destination = OUT / 'images' / (pid + '.jpg')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(original, destination)
                assert file_sha(destination) == photo['raw_sha256']
                claims = []
                for ci, claim in enumerate(assessment['claims']):
                    included = any(s['photo_id'] == pid and s['caption_id'] == cap['caption_id'] and s['raw_claim_index'] == ci
                                   for c in target['visual_target_claims_with_all_evidence'][job['condition']] for s in c['support'])
                    assert included == (claim['verdict'] == 'verified')
                    claims.append({**claim, 'raw_claim_index': ci, 'included_in_training_target': included})
                cases.append({'condition': job['condition'], 'job_id': job['job_id'], 'caption': cap,
                              'source_image': photo, 'image_file': 'images/' + pid + '.jpg',
                              'raw_teacher_assessment': assessment, 'claims': claims,
                              'complete_condition_target': target['targets'][job['condition']],
                              'raw_response_file': (EXE / 'raw_responses' / (job['job_id'] + '.json')).relative_to(ROOT).as_posix()})
        assert len(cases) == (1 if label == 'B' else 2)
        output.append({'case': label, 'business_id': bid, 'photo_id': pid, 'concern': concern,
                       'condition_evidence': cases, 'AI_second_opinion_classification': OPINIONS[label][0],
                       'AI_second_opinion': OPINIONS[label][1], 'human_final_label': None, 'human_verified': False,
                       'assessment_provenance': 'Codex inspected original photo; diagnostic second opinion only, not human ground truth.'})
    return output


def check_serialization(jobs, validation):
    outputs, excluded, serialized, raw_calls = {}, 0, 0, 0
    for job in jobs:
        cp = read(EXE / 'checkpoints' / (job['job_id'] + '.json'))
        assert cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'
        outputs.setdefault(job['business_id'], {})[job['condition']] = cp.get('resolved_output', cp['output'])
        raw_calls += bool(cp.get('api_called'))
    for path in (EXE / 'provisional_targets').glob('*.json'):
        target = read(path); content = outputs[target['business_id']]
        shared = '[REVIEW_CONSENSUS]\n' + ' '.join(c['text'] for c in content['shared']['claims'])
        assert target['targets']['reviews_only'] == shared
        for condition in ['unfiltered', 'filtered']:
            selected = visual_claims(content[condition]['assessments'])
            assert selected == target['visual_target_claims_with_all_evidence'][condition]
            assert target['targets'][condition] == shared + '\n\n[VISUAL_EVIDENCE]\n' + ' '.join(c['text'] for c in selected)
            serialized += len(selected)
            for assessment in content[condition]['assessments']:
                for index, claim in enumerate(assessment['claims']):
                    if claim['verdict'] != 'verified':
                        excluded += 1
                        assert not any(s['caption_id'] == assessment['caption_id'] and s['raw_claim_index'] == index
                                       for row in selected for s in row['support'])
        assert target['human_verified'] is False
    assert len(outputs) == 100 and raw_calls == 292
    original_manual = BASE / 'manual_review_25_manifest.json'
    assert file_sha(original_manual) == file_sha(FOLDER / 'manual_review_25_manifest.json')
    completion = read(EXE / 'completion.json')
    assert len(list((FOLDER/'previous_failed_attempts/checkpoints').glob('*.json'))) == 3
    return {'businesses_complete': 100, 'condition_aligned_targets': 300, 'completed_teacher_API_responses': 292,
            'deterministic_empty_filtered_components': 8,
            'content_origin': 'All substantive target text is verbatim teacher API content; frozen headers/separators and eight empty visual components are deterministic code serialization, not API-generated content.',
            'same_review_consensus_all_conditions': True,
            'rejected_or_uncertain_visual_assessments_not_serialized': excluded,
            'deduplicated_verified_visual_statements_serialized': serialized,
            'source_span_checks': validation['exact_review_source_span_checks'],
            'request_body_checks': validation['exact_API_request_checks'],
            'manual25_sha256': file_sha(original_manual), 'manual25_unchanged': True,
            'completed_human_annotations': 0, 'human_accuracy_estimate': None,
            'zero_filtered_businesses': validation['zero_caption_businesses_complete'],
            'usage': completion['usage'], 'historical_failed_attempts_preserved': 3,
            'current_unresolved_requests': [], 'scorer_method_and_source_mappings_unchanged': True,
            'exact_cross_condition_matching_groups': validation['exact_cross_condition']['matched_groups'],
            'exact_cross_condition_conflict_groups': validation['exact_cross_condition']['conflict_groups'],
            'exact_conflicts_reconciled': False}


def render_markdown(report, lengths, warnings, concerns):
    lines = ['# D-041 final pilot audit before production', '',
             '**Recommendation: READY WITH DOCUMENTED LIMITATIONS — researcher approval pending.**', '',
             'The existing D-040 review-v7/visual-v4 protocol may be frozen for Stage G production after researcher approval. '
             'No systematic source/schema/serialization defect was found. Two visual-identity concerns and cross-condition '
             'verification variability remain; this focused audit does not establish a population factual error rate. '
             'The fixed25 sample has not acquired completed human-final annotations.', '',
             '## Target lengths', '',
             'LED target tokens include BOS/EOS, with pinned tokenizer revision38335783885b338d93791936c54bb4be46bebed9. '
             '512 is diagnostic only. There is no active student training/preprocessing/generation implementation or frozen decoder cap. '
             'No target is currently truncated. A future512 label cap would cut these nine, all in the visual-conditioned targets.', '',
             '| Business | Split | Condition | Tokens | Above512 |', '|---|---|---|---:|---:|']
    lines += [f"| {r['business_id']} | {r['split']} | {r['condition']} | {r['tokens_including_BOS_EOS']} | {r['tokens_above_512']} |" for r in lengths]
    lines += ['', report['training_status']['smallest_recommended_correction_not_applied'], '',
              'This is a blocker to starting LED training until a decoder/preprocessing policy is explicitly resolved, '
              'not an existing defect requiring teacher regeneration. No cap is changed/frozen by this audit.', '',
              '## All15 review-ID warnings', '',
              'All15 are auxiliary ID metadata discrepancies:13 copied another review ID and2 typographic mismatches. '
              'For every warning, the authoritative REV marker and selected span IDs agree, exact original quotes and offsets/hash match, '
              'and the already-approved D-037 source projection preserves model claim text/polarity. '
              'No student sentence changes and no grounding error attributable to these ID warnings was found. '
              'Semantic entailment remains separate from identifier validation.', '']
    for row in warnings:
        lines += [f"### Warning{row['warning_number']} — {row['business_id']}", '',
                  f"Claim (raw index{row['claim_index']}): {row['claim']}", '',
                  f"{row['review_marker']} / {', '.join(row['evidence_span_ids'])}; reportedID `{row['teacher_reported_review_id']}`; "
                  f"canonicalID `{row['canonical_source_review_id']}`.", '',
                  f"Classification: {row['classification']}. Source bindingPASS; student target unaffected.", '',
                  'Exact supporting source spans:', '']
        for span in row['exact_source_spans']:
            lines += [f"```text\n{span['exact_source_text']}\n```", '']
        lines += ['AI second-opinion scope: ' + row['AI_second_opinion_grounding_note'], '']
    lines += ['## Original evidence for three visual concerns', '']
    for case in concerns:
        lines += [f"### Case{case['case']} — {case['business_id']}", '', f"Photo `{case['photo_id']}`", '',
                  f"![Original photo](images/{case['photo_id']}.jpg)", '',
                  '**AI second opinion, not a human-final label:** ' + case['AI_second_opinion'], '']
        for cond in case['condition_evidence']:
            cap = cond['caption']
            lines += [f"**{cond['condition']} — {cap['caption_id']} / {cap['photo_marker']}**", '',
                      'Exact BLIP-2 caption: ' + cap['caption'], '',
                      '| Exact teacher claim | Teacher verdict | Included in target |', '|---|---|---|']
            lines += [f"| {r['text']} | {r['verdict']} | {r['included_in_training_target']} |" for r in cond['claims']]
            lines += ['']
    c = report['completed_pilot_checks']
    lines += ['## Completed pilot checks', '',
              f"100completebusinesses/300targets;292completeAPIresponses and8deterministic empty-Filtered components. "
              f"{c['source_span_checks']}exactreviewspanchecks;{c['request_body_checks']}exactrequestchecks;300native/slowLEDtargetIDchecks. "
              f"Shared review consensus matches across allthreeconditions. {c['rejected_or_uncertain_visual_assessments_not_serialized']} "
              'rejected/uncertain claim assessments are excluded by provenance from final visual targets; no leakage found.', '',
              'Original mappings/source bytes and prior outputs unchanged. Eight zero-Filtered businesses retain the agreed empty visual header. '
              'Raw responses,usage,292successful paid task records and3historical failed operator-recovered attempts preserved. '
              'All100 and fixed25 sampling unchanged; human-final completedbusinesses remain0.', '',
              f"Exact matching photo/caption/claim audit: {c['exact_cross_condition_conflict_groups']}conflicts in "
              f"{c['exact_cross_condition_matching_groups']}matchinggroups. All conflicts are exported separately; "
              '27lexical-similarity flags are reporting heuristics, not27proven equivalent contradictions. '
              'No verdict was reconciled or rewritten.', '',
              '## Readiness and stopping conditions', '',
              'No confirmed Stage G schema/source/target-serialization blocker remains. Document visual factual limitations, '
              'verification variability and provider-alias provenance (gpt-6.1-sol; immutable backend snapshot unavailable). '
              'Freeze the existing protocol only after researcher approval; carry completed100 pilot targets forward under '
              'compatible frozen bindings, preserving their provisional/unverified provenance. Production retry/version policies '
              'still need explicit operational recording; none is silently inferred here.', '',
              'The student decoder label/generation-length policy is unresolved and must be settled before training. '
              'A640pilot-fitting suggestion is not a production cap. No prompt/settings/schema/labels/targets changed, '
              'no APIcalls or remaining2100generation, no LEDtraining. Independent review reported by the user is context, '
              'not an imported set of human-final annotations.', '']
    return '\n'.join(lines)


def render_html(markdown, warnings, concerns):
    cards = []
    for case in concerns:
        src = 'data:image/jpeg;base64,' + base64.b64encode((OUT/'images'/f"{case['photo_id']}.jpg").read_bytes()).decode()
        parts = [f"<section><h2>Case{case['case']}: {html.escape(case['concern'])}</h2>",
                 f"<p>{case['business_id']} / {case['photo_id']}</p><img src='{src}'>",
                 '<p><b>AI second opinion; human confirmation pending:</b> ' + html.escape(case['AI_second_opinion']) + '</p>']
        for cond in case['condition_evidence']:
            cap = cond['caption']
            parts += [f"<h3>{cond['condition']} — {cap['caption_id']} / {cap['photo_marker']}</h3>",
                      '<p><b>Exact caption:</b> ' + html.escape(cap['caption']) + '</p><table>',
                      '<tr><th>Exact claim</th><th>Teacher verdict</th><th>In target</th><th>Teacher explanation</th></tr>']
            parts += ['<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in
                      [r['text'], r['verdict'], r['included_in_training_target'], r['explanation']]) + '</tr>' for r in cond['claims']]
            parts += ['</table><details><summary>Complete unchanged condition target</summary><pre>'
                      + html.escape(cond['complete_condition_target']) + '</pre></details>']
        parts += ['</section>']; cards.append(''.join(parts))
    rows = ['<details><summary>All15 review-ID warnings with exact original reviews</summary>']
    for row in warnings:
        rows += [f"<h3>{row['warning_number']}: {row['business_id']} / {row['review_marker']}</h3>",
                 '<p>' + html.escape(row['claim']) + '</p>',
                 '<pre>' + html.escape(json.dumps({k:row[k] for k in ['classification','teacher_reported_review_id','canonical_source_review_id','evidence_span_ids']},ensure_ascii=False,indent=2)) + '</pre>',
                 '<b>Exact selected spans</b><pre>' + html.escape(''.join(s['exact_source_text'] for s in row['exact_source_spans'])) + '</pre>',
                 '<details><summary>Full original selected review</summary><pre>' + html.escape(row['source_review']['text']) + '</pre></details>']
    rows += ['</details>']
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>D041 final audit</title><style>' \
           'body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1150px;margin:32px auto;padding:20px}' \
           'section{border:1px solid #475569;border-radius:12px;padding:20px;margin:24px 0}img{max-width:100%;max-height:650px}' \
           'table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid #475569;text-align:left}' \
           'pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;padding:12px}</style><h1>D041 final pilot audit</h1>' \
           '<p><b>READY WITH DOCUMENTED LIMITATIONS — researcher approval pending.</b> No targets or labels edited. No APIcalls.</p>' \
           + ''.join(cards) + ''.join(rows) + '<details><summary>Full audit report</summary><pre>' + html.escape(markdown) + '</pre></details></html>'


def main():
    # An executable network guard makes the offline audit unable to call the teacher.
    def deny_network(*args, **kwargs):
        raise RuntimeError('Offline final audit: network/API calls forbidden')
    urllib.request.urlopen = deny_network
    before = snapshot()
    validation = results()  # Revalidate immutable saved results; compatible artifacts only.
    jobs = read(FOLDER/'pilot_jobs.json'); flags = read(FOLDER/'technical_flags.json')['flags']
    native, slow = tokenizer_pair()
    lengths, length_summary = audit_lengths(native, slow, jobs, flags)
    training = inventory_training()
    assert not training['training_term_matches'], 'Unexpected training implementation: inspect before readiness recommendation'
    warnings = audit_ids(jobs, flags)
    concerns = visual_evidence(jobs)
    checks = check_serialization(jobs, validation)
    report = {'status': 'OFFLINE_FINAL_AUDIT_COMPLETE_RESEARCHER_APPROVAL_PENDING',
              'recommendation': 'READY WITH DOCUMENTED LIMITATIONS', 'API_calls': 0,
              'source_namespace': FOLDER.relative_to(ROOT).as_posix(),
              'automated_validation': validation['status'], 'target_length_summary': length_summary,
              'training_status': training, 'review_ID_warning_count': len(warnings),
              'review_ID_warning_classes': dict(Counter(r['classification'] for r in warnings)),
              'visual_concerns_count': 3, 'factual_assessment_is_human_ground_truth': False,
              'human_final_annotations_imported_or_created': 0, 'completed_pilot_checks': checks,
              'confirmed_Stage_G_technical_blockers': [],
              'LED_training_prerequisite': 'Explicit complete-label decoder/generation length policy;512cannot be used for silent label truncation.',
              'Stage_G_production_protocol_freeze_recommended_not_applied': True,
              'operational_production_decisions_still_to_record': ['model-alias/backend-snapshot provenance policy','production retry/unknown-request policy','target-length monitoring policy'],
              'limitations': ['Potential chopsticks and doughnuts misidentifications; no manual relabeling.',
                              'Cross-condition verification variability; exact10conflicts retained plus heuristic lexical flags.',
                              'Focused audit is not exhaustive factual verification or a measured population error rate.',
                              'Independent ChatGPT review mentioned by researcher has not been imported as human-final annotations.'],
              'unchanged': ['prompts','model_settings','schemas','targets','human_labels','Stage_F','sampling_100_and_25','Codex_configuration'],
              'remaining_2100_generation_started': False, 'LED_training_started': False}
    immutable(OUT/'audit_report.json', report)
    immutable(OUT/'target_length_cases.json', lengths)
    immutable(OUT/'review_ID_warnings.json', {'role':'identifier/source audit; semantic notes are AI second opinions','cases':warnings})
    immutable(OUT/'visual_concerns.json', {'role':'original evidence plus AI second opinion, not human-final labels','cases':concerns})
    immutable(OUT/'exact_cross_condition_conflicts.json', validation['exact_cross_condition']['conflicts'])
    csv_buffer = io.StringIO(); writer = csv.DictWriter(csv_buffer,fieldnames=list(lengths[0])); writer.writeheader(); writer.writerows(lengths)
    freeze_text(OUT/'target_lengths.csv',csv_buffer.getvalue())
    markdown = render_markdown(report, lengths, warnings, concerns)
    freeze_text(OUT/'AUDIT_REPORT.md', markdown)
    freeze_text(OUT/'index.html', render_html(markdown, warnings, concerns))
    immutable(OUT/'preserved_pilot_sha256.json', before)
    assert snapshot() == before, 'A completed D041 pilot artifact changed during offline audit'
    members = [p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name != 'export_integrity_report.json']
    archive = OUT/'chatgpt_d041_final_pilot_audit.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(members):
            data = path.read_bytes()
            assert not re.search(rb'sk-[A-Za-z0-9_-]{20,}',data), 'Secret-like value in audit export'
            z.writestr('chatgpt_d041_final_pilot_audit/'+path.relative_to(OUT).as_posix(), data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for case in concerns:
            for evidence in case['condition_evidence']:
                member = 'chatgpt_d041_final_pilot_audit/' + evidence['image_file']
                assert hashlib.sha256(z.read(member)).hexdigest() == evidence['source_image']['raw_sha256']
    immutable(OUT/'export_integrity_report.json', {'status':'PASS','all_original_pilot_files_unchanged':True,
              'protected_pilot_file_count':len(before),'all3original_export_photos_SHA_match':True,
              'ZIP_CRC_PASS':True,'zip_path':archive.relative_to(ROOT).as_posix(),'zip_bytes':archive.stat().st_size,
              'zip_sha256':file_sha(archive),'API_calls':0,'human_labels_or_teacher_outputs_modified':False})
    print(json.dumps({'status':report['status'],'recommendation':report['recommendation'],
                      'max_target_tokens':length_summary['max_tokens'],'ID_warning_classes':report['review_ID_warning_classes'],
                      'zip':str(archive),'bytes':archive.stat().st_size},indent=2))


if __name__ == '__main__':
    main()
