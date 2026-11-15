"""Replay a proposed minimal rule on stored E0 attempts, without inference."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import yaml
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
CLOSERS = '\"\'\u201d\u2019)]}'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(raw, cfg):
    text = ' '.join(raw.split())
    if not WORD.search(text):
        return text
    end = len(text)
    while end and text[end-1] in CLOSERS:
        end -= 1
    if text[:end].endswith(tuple(cfg['terminal_punctuation'])):
        return text
    return text[:end]+'.'+text[end:]


def repeated_blocks(words, minimum, maximum):
    for size in range(minimum, maximum+1):
        for start in range(len(words)-2*size+1):
            if words[start:start+size] == words[start+size:start+2*size]:
                return [' '.join(words[start:start+size])]
    return []


def evaluate(attempt, cfg):
    raw = attempt['raw_caption']
    words = WORD.findall(raw)
    normalized = normalize(raw, cfg)
    assert WORD.findall(normalized) == words, 'Normalization must preserve lexical content.'
    flags = []
    if not raw.strip():
        flags.append('empty')
    elif not words:
        flags.append('nonlexical')
    if len(words) < cfg['minimum_words']:
        flags.append('under5_words')
    repetition = cfg['repetition']
    repeats = repeated_blocks([w.lower() for w in words],
                              repetition['adjacent_repeated_block_min_words'],
                              repetition['adjacent_repeated_block_max_words'])
    if repeats:
        flags.append('adjacent_repeated_4_to_8_word_block')
    generic_key = ' '.join(raw.lower().split()).strip(' .!?\"\'\u201d\u2019')
    if generic_key in cfg['generic']['phrases']:
        flags.append('exact_generic_phrase')
    cap, eos = attempt.get('token_cap_reached'), attempt.get('eos_seen')
    termination = 'unknown_missing_stop_metadata'
    if isinstance(cap, bool) and isinstance(eos, bool):
        termination = 'token_cap_without_EOS' if cap and not eos else 'no_token_cap_without_EOS'
        if cap and not eos:
            flags.append('evidenced_truncation')
    return {'raw_caption': raw, 'normalized_caption': normalized,
            'word_count': len(words), 'basic_quality_fail': bool(flags),
            'failure_reasons': flags, 'repeated_blocks': repeats,
            'short_repetition_diagnostic': repeated_blocks([w.lower() for w in words], 3, 3),
            'punctuation_added': normalized != ' '.join(raw.split()),
            'termination_diagnostic': termination,
            'semantic_grammar_grounding_verified': False}


def edge_checks(cfg):
    assert normalize('  a bowl\n of rice  ', cfg) == 'a bowl of rice.'
    assert normalize('"a bowl of rice."', cfg) == '"a bowl of rice."'
    assert normalize('"a bowl of rice"', cfg) == '"a bowl of rice."'
    assert normalize('', cfg) == '' and normalize('!!!', cfg) == '!!!'
    assert evaluate({'raw_caption': 'a close up of a close up of a burger'}, cfg)['basic_quality_fail']
    assert not evaluate({'raw_caption': 'a picture of a picture of a building'}, cfg)['basic_quality_fail']
    assert evaluate({'raw_caption': 'a plate of food on a table'}, cfg)['basic_quality_fail']
    assert not evaluate({'raw_caption': 'a plate of dumplings on a table'}, cfg)['basic_quality_fail']
    full = 'a plate of dumplings on a table'
    assert not evaluate({'raw_caption': full}, cfg)['basic_quality_fail']
    assert evaluate({'raw_caption': full, 'token_cap_reached': True, 'eos_seen': False}, cfg)['basic_quality_fail']
    assert not evaluate({'raw_caption': full, 'token_cap_reached': True, 'eos_seen': True}, cfg)['basic_quality_fail']
    return 'normalization_quotes_empty_specificity_repetition_and_truncation_checks_passed'


def main():
    config_path = ROOT/'configs/basic_caption_quality_draft.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert cfg['status'] == 'draft_not_adopted'
    source, out = ROOT/cfg['source_directory'], ROOT/cfg['output_directory']
    out.mkdir(parents=True, exist_ok=True)
    checks = edge_checks(cfg)
    plan = read(source/'pilot_photo_plan.json')
    assert len(plan) == 24 and {p['split'] for p in plan} == {'train', 'validation'}
    protected = list((source/'caption_checkpoints').glob('*.json'))
    protected += [source/name for name in ['pilot_photo_plan.json', 'generation_report.json',
                 'model_load_report.json', 'pilot_report.json', 'validation_report.json',
                 'raw_caption_statistics.csv', 'pilot_source_token_ledger.csv',
                 'manual_caption_review_notes.json', 'complete_provisional_source_examples.json']]
    protected += [ROOT/'configs/blip2_caption_feasibility_pilot.yaml', ROOT/'configs/research_pending.yaml',
                  ROOT/'scripts/blip2_caption_pilot.py']
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}
    records = []
    for photo in plan:
        existing = read(source/'caption_checkpoints'/f"{photo['photo_id']}.json")
        attempts = [dict(attempt_number=i, **evaluate(a, cfg))
                    for i, a in enumerate(existing['attempts'], 1)]
        assert len(attempts) == 2
        needs_retry = attempts[0]['basic_quality_fail']
        chosen = (1 if not needs_retry else 2 if not attempts[1]['basic_quality_fail'] else None)
        current = attempts[(chosen or 1)-1]
        records.append({k: photo[k] for k in ['photo_id', 'business_id', 'name', 'split', 'raw_sha256']} |
                       {'attempts': attempts, 'initial_basic_fail': needs_retry,
                        'retry_needed_under_draft': needs_retry,
                        'historical_retry_considered': needs_retry,
                        'draft_chosen_attempt': chosen,
                        'raw_caption': current['raw_caption'],
                        'normalized_caption': current['normalized_caption'],
                        'draft_basic_eligible': chosen is not None,
                        'legacy_raw_caption': existing['raw_caption'],
                        'legacy_surface_eligible': existing['surface_eligible'],
                        'production_caption_set': False})
    sources = [s for s in read(source/'complete_provisional_source_examples.json')
               if s['condition'] == 'all_raw_pilot_captions_capacity_only']
    tok = Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    tok.no_truncation(); tok.no_padding()
    ledger = []
    for original in sources:
        base, old_block = original['complete_provisional_source'].rsplit('\n\n[CAPTIONS]\n', 1)
        old_records = [read(source/'caption_checkpoints'/f"{pid}.json") for pid in original['caption_photo_ids']]
        assert old_block == '\n'.join(f"[IMG_CAP_{i}] {r['raw_caption']}" for i, r in enumerate(old_records, 1))
        selected = [r for r in records if r['business_id'] == original['business_id'] and r['draft_basic_eligible']]
        block = '\n'.join(f"[IMG_CAP_{i}] {r['normalized_caption']}" for i, r in enumerate(selected, 1))
        full = base+('\n\n[CAPTIONS]\n'+block if selected else '')
        old_tokens = len(tok.encode(original['complete_provisional_source']).ids)
        assert old_tokens == original['actual_full_source_tokens']
        tokens = len(tok.encode(full).ids)
        ledger.append({'business_id': original['business_id'], 'name': original['name'],
                       'old_raw_full_source_tokens': old_tokens, 'draft_normalized_full_source_tokens': tokens,
                       'token_delta': tokens-old_tokens, 'draft_captions': len(selected),
                       'within_diagnostic5000': tokens <= 5000, 'review_IDs_and_base_unchanged': True,
                       'production_source': False})
    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}
    assert before == after
    initial_reasons = Counter(f for r in records for f in r['attempts'][0]['failure_reasons'])
    summary = {'status': 'DRAFT_OFFLINE_REPLAY_ONLY_NOT_ADOPTED', 'policy_version': cfg['version'],
               'photos': 24, 'new_model_calls': 0, 'legacy_surface_eligible': 0,
               'draft_initial_eligible': sum(not r['initial_basic_fail'] for r in records),
               'draft_initial_failure_reasons': dict(initial_reasons),
               'draft_retry_triggers': sum(r['retry_needed_under_draft'] for r in records),
               'draft_final_eligible_with_historical_retries': sum(r['draft_basic_eligible'] for r in records),
               'punctuation_alone_causes_retry': False,
               'initial_stop_metadata_unknown': sum(r['attempts'][0]['termination_diagnostic'].startswith('unknown') for r in records),
               'normalized_source_min': min(r['draft_normalized_full_source_tokens'] for r in ledger),
               'normalized_source_max': max(r['draft_normalized_full_source_tokens'] for r in ledger),
               'source5000_overflows': sum(not r['within_diagnostic5000'] for r in ledger),
               'historical_outputs_and_canonical_policy_unchanged': True,
               'development_data_used_to_design_rules': True,
               'production_quality_accepted': False, 'retry_protocol_frozen': False,
               'final_caption_policy_manifest_source_allocation': False, 'edge_checks': checks,
               'config_sha256': digest(config_path), 'protected_artifact_sha256': before}
    for name, data in [('candidate_records.json', records), ('normalized_source_token_ledger.json', ledger),
                       ('comparison_report.json', summary)]:
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'protected_artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
