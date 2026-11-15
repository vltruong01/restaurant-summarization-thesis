"""Approved minimal Basic Quality; pure text/termination rules, no image filtering."""
import re

WORD = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
CLOSERS = '\"\'\u201d\u2019)]}'


def normalize(raw, policy):
    text = ' '.join(raw.split())
    if not WORD.search(text):
        return text
    end = len(text)
    while end and text[end-1] in CLOSERS:
        end -= 1
    if text[:end].endswith(tuple(policy['terminal_punctuation'])):
        return text
    return text[:end]+'.'+text[end:]


def repeated_blocks(words, minimum, maximum):
    for size in range(minimum, maximum+1):
        for start in range(len(words)-2*size+1):
            if words[start:start+size] == words[start+size:start+2*size]:
                return [' '.join(words[start:start+size])]
    return []


def evaluate(attempt, policy):
    raw = attempt['raw_caption']
    words = WORD.findall(raw)
    normalized = normalize(raw, policy)
    assert WORD.findall(normalized) == words
    flags = []
    if not raw.strip():
        flags.append('empty')
    elif not words:
        flags.append('nonlexical')
    if len(words) < policy['minimum_words']:
        flags.append('under5_words')
    repetition = policy['repetition']
    repeats = repeated_blocks([w.lower() for w in words],
                              repetition['adjacent_repeated_block_min_words'],
                              repetition['adjacent_repeated_block_max_words'])
    if repeats:
        flags.append('adjacent_repeated_4_to_8_word_block')
    key = ' '.join(raw.lower().split()).strip(' .!?\"\'\u201d\u2019')
    if key in policy['generic']['phrases']:
        flags.append('exact_generic_phrase')
    cap, eos = attempt.get('token_cap_reached'), attempt.get('eos_seen')
    termination = 'unknown_missing_stop_metadata'
    if isinstance(cap, bool) and isinstance(eos, bool):
        termination = 'token_cap_without_EOS' if cap and not eos else 'no_token_cap_without_EOS'
        if cap and not eos:
            flags.append('evidenced_truncation')
    return {'normalized_caption': normalized, 'word_count': len(words),
            'basic_quality_fail': bool(flags), 'basic_quality_flags': flags,
            'surface_ok': not flags, 'repeated_blocks': repeats,
            'short_repetition_diagnostic': repeated_blocks([w.lower() for w in words], 3, 3),
            'punctuation_added': normalized != ' '.join(raw.split()),
            'termination_diagnostic': termination,
            'semantic_grammar_grounding_verified': False}


def termination_metadata(sequence, prefix, eos_ids, pad_id, token_cap):
    """Exclude prompt/BOS before EOS checks; ignore padding after first generated EOS."""
    suffix = sequence[prefix:]
    eos_ids = sorted(set(eos_ids))
    first_eos = next((i for i, token in enumerate(suffix) if token in eos_ids), None)
    effective = len(suffix) if first_eos is None else first_eos+1
    return {'generated_sequence_ids': sequence, 'generated_suffix_ids': suffix,
            'prefix_tokens': prefix, 'eos_token_ids': eos_ids, 'pad_token_id': pad_id,
            'eos_seen': first_eos is not None, 'first_generated_eos_index': first_eos,
            'generated_suffix_tokens': len(suffix), 'effective_generated_tokens': effective,
            'configured_max_new_tokens': token_cap, 'token_cap_reached': effective >= token_cap,
            'termination_reason': ('eos_at_cap' if effective >= token_cap else 'eos')
                                  if first_eos is not None else 'token_cap_without_EOS'
                                  if effective >= token_cap else 'unknown_non_EOS_stop'}
