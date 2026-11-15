"""Update documentation/read-only notebook after the bounded pilot passes its offline audit."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/stage_g_teacher_pilot_v3_canonical_ids.json'


def main():
    cfg = json.loads((ROOT / CONFIG).read_text(encoding='utf-8'))
    execution = ROOT / cfg['output_directory'] / 'execution_v1'
    audit = json.loads((execution / 'pilot_integrity_and_tokens_report.json').read_text(encoding='utf-8'))
    completion = json.loads((execution / 'completion_20_businesses.json').read_text(encoding='utf-8'))
    assert audit['businesses_complete'] == 20 and audit['condition_target_texts'] == 60
    assert audit['status'] == 'PILOT_INTEGRITY_PASS_HUMAN_REVIEW_PENDING'
    assert audit['slow_LED_target_token_ID_checks'] == 60
    assert not completion['production_method_frozen']
    folder = execution.relative_to(ROOT).as_posix()
    current = ('Stage G bounded pilot COMPLETE:20 businesses (10train/10validation),60 provisional '
        'condition target texts; requested/returned model gpt-6.1-sol. Integrity PASS; human grounding '
        'review pending. No full2200 production generation or LED training.')
    detail = (f'Active config `{CONFIG}`; outputs `{folder}`. '
        f"{completion['api_tasks_completed']} completed API tasks, including "
        f"{completion['compatible_API_responses_imported']} exact-request compatible responses imported "
        f"and {completion['new_API_calls_for_initial_completion']} new calls in this execution; "
        f"{completion['empty_visual_tasks_without_API']} deterministic empty visual tasks. "
        f"{audit['exact_source_spans_resolved']} exact source spans resolved; "
        f"{audit['reconstructed_actual_request_SHA_checks']} request-SHA checks and60native/slowLED "
        'target-token-ID checks PASS. Shared review consensus is identical across all three conditions; '
        'original selected review IDs/text/order, captions/photos and StageFcontrols unchanged. '
        'All raw model responses and prior failed versions remain preserved.')
    correction = ('D-037: two supports in validation business FFV8_b-ec5Z0E3xgvkVxKw had an incorrect '
        'redundant model-reported opaque review ID, while exact REV_18 marker/span IDs selected the '
        'correct immutable source text. Canonical IDs/quotes/offsets are derived from strict agreeing '
        'marker/span mapping; original reported IDs are preserved as separate diagnostics. No fuzzy '
        'repair, span substitution, claim/polarity changes or relaxed distinct-review support rule. '
        'Prompts/model/settings/schema and outgoing API request bytes did not change. '
        'D-036 identical cross-photo verified statements remain serialized once with all evidence retained.')
    mismatches = audit['redundant_reported_review_ID_mismatches']
    correction += (f" Final pilot: {len(mismatches)} redundant reported-ID disagreements across "
                   f"{len({d['business_id'] for d in mismatches})} businesses; "
                   'all are listed individually in the immutable integrity report.')
    human = ('Human verification: the user confirmed one straw claim after inspecting the enlarged image; '
        'the previous AI visibility flag is resolved. One claim verified, zero whole businesses fully '
        'human-verified. These are teacher pseudo-targets, not human references. Structural/source checks '
        'do not certify semantic entailment or factual correctness. The fixed pilot deliberately includes '
        'zero-Filtered cases and is separate from the later random manual-inspection subset and the '
        'official30-business human evaluation.')
    token_lines = ['| Condition | Min LED target tokens | Mean | Median | p95 | Max |',
                   '|---|---:|---:|---:|---:|---:|']
    for condition, v in audit['LED_target_tokens_with_special_tokens'].items():
        token_lines.append(f"| {condition} | {v['min']} | {v['mean']:.2f} | {v['median']:.2f} | {v['p95']:.2f} | {v['max']} |")
    token_table = '\n'.join(token_lines)
    evidence = (f"Reported model usage for the58completedtasks (including imported tasks): "
        f"{completion['input_tokens_reported']} input / {completion['output_tokens_reported']} output "
        f"tokens, including {completion['reasoning_tokens_reported']} reasoning tokens. This excludes "
        'historical failed/rejected attempts and is not an account billing total. No target exceeds512 '
        'in this pilot;512is a diagnostic planning value, NOT a frozen target cap. No target text was '
        'truncated or edited to achieve these lengths.')
    unresolved = ('Before full generation, review pilot grounding and finalize the production prompt/settings, '
        'model alias/version provenance policy, target-length/overflow policy, rejection/retry policy and '
        'random manual-inspection sample size. None is silently frozen by this pilot. Teacher production '
        'scope remains2000train+200validation only;100held-outtest businesses receive no teacher '
        'pseudo-targets. StageF/D026/D030/D031/D032/D033 remain frozen; no captions, thresholds, K '
        'or source allocation are retuned. Shuffled assignment, human test references, external '
        'generalization evaluation and LED training remain later work.')
    history_heading = '## 2026-10-10 - Complete20-business StageG pilot; D-037 canonical citation provenance'
    documents = ['thesis_sdd_docs/THESIS_PROGRESS.md',
                 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                 'THESIS_PROGRESS.md', 'configs/research_pending.yaml',
                 'specs/007_teacher_target_construction.md',
                 'notebooks/00_thesis_dataset_filtering_review_selection.ipynb']
    snapshots = execution / 'before_completion_status_update'
    for relative in documents:
        copy = snapshots / relative
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, copy)

    path = ROOT / documents[0]
    text = path.read_text(encoding='utf-8')
    first_end = text.index('\n\n', text.index('Current state'))
    first_start = text.index('Current state')
    text = text[:first_start] + ('Current state (2026-10-10): StageFproduction COMPLETE independently '
        'validated;2300businesses,30027captionrecords, finalUnfiltered11487/Filtered7919, '
        '0sourceoverflows. ' + current) + text[first_end:]
    begin, end = text.index('### 5.'), text.index('### 6.')
    text = text[:begin] + '### 5. Current Stage G pilot and remaining work\n\n' + current + '\n\n' + detail + \
        '\n\n' + correction + '\n\n' + token_table + '\n\n' + evidence + '\n\n' + human + \
        '\n\n' + unresolved + '\n\n' + text[end:]
    paragraph_start = text.index('\n\n', text.index('### 6.')) + 2
    paragraph_end = text.index('\n\n', paragraph_start)
    text = text[:paragraph_start] + ('The completed20-business pilot is now available in Notebook00 '
        'sections12.1/12.2. Review actual targets and remaining production decisions, keeping the '
        'master-plan invariants below. Older prototype failures are historical, not the current state.') + text[paragraph_end:]
    if history_heading not in text:
        text += '\n\n' + history_heading + '\n\n' + detail + '\n\n' + correction + '\n\n' + token_table + \
            '\n\n' + evidence + '\n\n' + human + '\n\n' + unresolved + \
            '\n\n32StageGoffline validation tests PASS. Full report: `' + folder + \
            '/pilot_integrity_and_tokens_report.json`; completion: `completion_20_businesses.json`.\n'
    path.write_text(text, encoding='utf-8')

    path = ROOT / documents[1]
    text = path.read_text(encoding='utf-8')
    stage_header = '# 8. Stage G \u2014 Teacher target construction'
    start = text.index('\n\n', text.index(stage_header)) + 2
    if '**Current completed pilot' not in text[start:start+100]:
        text = text[:start] + '**Current completed pilot (2026-10-10):** ' + current + '\n\n' + detail + \
            '\n\n' + correction + '\n\n' + human + '\n\n' + unresolved + \
            '\n\n**Historical execution notes below are superseded by this completed-pilot status.**\n\n' + text[start:]
    path.write_text(text, encoding='utf-8')

    path = ROOT / documents[2]
    text = path.read_text(encoding='utf-8')
    start = text.index('\n\n') + 2
    end = text.index('\n\n', start)
    text = text[:start] + 'The maintained progress log is thesis_sdd_docs/THESIS_PROGRESS.md. ' + current + \
        ' Older entries below are historical and may describe superseded states.' + text[end:]
    path.write_text(text, encoding='utf-8')

    path = ROOT / documents[3]
    text = path.read_text(encoding='utf-8')
    old_status = 'preselected20_business_pilot_running_D037_canonical_source_IDs_human_grounding_pending'
    text = text.replace(old_status, 'twenty_business_pilot_integrity_PASS_D037_human_grounding_pending_production_unfrozen')
    path.write_text(text, encoding='utf-8')

    path = ROOT / documents[4]
    text = path.read_text(encoding='utf-8')
    if history_heading not in text:
        text += '\n\n' + history_heading + '\n\n' + detail + '\n\n' + human + '\n\n' + unresolved + '\n'
    path.write_text(text, encoding='utf-8')

    path = ROOT / documents[5]
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for i in [41,43,45,47]:
        source = ''.join(notebook['cells'][i]['source']).replace(
            'stage_g_teacher_pilot_v3_visual_merge.json', 'stage_g_teacher_pilot_v3_canonical_ids.json')
        notebook['cells'][i]['source'] = source.splitlines(keepends=True)
    notebook['cells'][38]['source'] = ('## 11. Next steps - current status\n\nStageFcomplete; '
        'K5/budgets5000/4000/750 frozen. ' + current + '\n\n' + unresolved + '\n').splitlines(keepends=True)
    notebook['cells'][40]['source'] = ('## 12. Stage G - completed20-business teacher pilot\n\n' + current +
        '\n\nD035sourceIDs/D036visualstatementserialization/D037canonicalcitationmetadata. '
        'Section12.1shows targets;12.2shows audit/lengths and expandable outputs for all20businesses. '
        'RunAll only reads existing results; no teacher calls.\n').splitlines(keepends=True)
    notebook['cells'][42]['source'] = ('### 12.1. Actual pilot targets and source inspection\n\n'
        'Change BUSINESS_ID_TO_VIEW to any of the20IDs listed in12.2. '
        'Human grounding remains pending; model verdicts are not human labels.\n').splitlines(keepends=True)
    notebook['cells'][44]['source'] = ('### 12.2. Completed pilot, integrity and all20target previews\n\n' +
        current + '\nHistorical failures remain preserved; one strawclaim user-confirmed, '
        'zero wholebusinesses fullyhuman-verified.\n').splitlines(keepends=True)
    source = ''.join(notebook['cells'][45]['source'])
    if 'view_stage_g_completion()' not in source:
        source += '\nfrom stage_g_notebook_view import view_stage_g_completion\nview_stage_g_completion()\n'
    notebook['cells'][45]['source'] = source.splitlines(keepends=True)
    notebook['cells'][46]['source'] = ('### 12.3. Historical V3 review-only/network-stop evidence\n\n'
        'This earlier failed version is retained as provenance and is superseded by the '
        'complete20-business pilot above. Its outputs are not overwritten.\n').splitlines(keepends=True)
    notebook['cells'][39]['outputs'] = []
    notebook['cells'][39]['execution_count'] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETED_PILOT_DOCS_AND_READ_ONLY_VIEWS_UPDATED',
                      'files': documents, 'scientific_labels_or_targets_modified': False,
                      'production_settings_frozen': False}, indent=2))


if __name__ == '__main__':
    main()
