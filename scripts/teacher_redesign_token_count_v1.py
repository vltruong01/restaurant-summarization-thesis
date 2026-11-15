"""Offline LED token accounting / structural checks; no model or API imports."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

CONDITIONS = ('reviews_only', 'unfiltered', 'filtered')
REVIEW_HEADER = '[REVIEW_CONSENSUS]\n'
VISUAL_SEPARATOR = '\n\n[VISUAL_EVIDENCE]\n'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_tokenizer(folder):
    from tokenizers import Tokenizer
    folder = Path(folder)
    spec = json.loads((folder / 'TOKENIZER_BINDING.json').read_text(encoding='utf-8'))
    if importlib.metadata.version('tokenizers') != spec['tokenizers_version']:
        raise ValueError('Use the exact pinned tokenizers version; do not invent comparable counts.')
    for name, expected in spec['files'].items():
        if sha(folder / name) != expected:
            raise ValueError('Tokenizer hash differs: ' + name)
    tok = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    tok.no_padding()
    tok.no_truncation()
    if [tok.token_to_id(t) for t in ('<s>', '<pad>', '</s>')] != [0, 1, 2]:
        raise ValueError('Special token mapping differs')
    return tok


def split_target(text, condition):
    if condition not in CONDITIONS or not text.startswith(REVIEW_HEADER):
        raise ValueError('Unknown condition or missing review header')
    if text.count('[REVIEW_CONSENSUS]') != 1:
        raise ValueError('Repeated review header')
    if condition == 'reviews_only':
        if '[VISUAL_EVIDENCE]' in text:
            raise ValueError('RO must not contain visual section')
        review, visual = text[len(REVIEW_HEADER):], None
    else:
        if text.count('[VISUAL_EVIDENCE]') != 1 or VISUAL_SEPARATOR not in text:
            raise ValueError('U/F must preserve the exact visual separator')
        review, visual = text[len(REVIEW_HEADER):].split(VISUAL_SEPARATOR, 1)
    if not review.strip():
        raise ValueError('Empty review consensus')
    return review, visual


def target_counts(tok, text, condition):
    review, visual = split_target(text, condition)
    raw = tok.encode(text, add_special_tokens=True).ids
    if raw[0] != 0 or raw[-1] != 2 or raw.count(0) != 1 or raw.count(2) != 1 or 1 in raw:
        raise ValueError('Ambiguous initial BOS/content/EOS sequence; inspect without editing')
    return {
        'target_utf8_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'original_tokenizer_BOS_content_EOS_tokens': len(raw),
        'corrected_training_content_EOS_label_tokens': len(raw) - 1,
        'target_content_without_special_tokens': len(tok.encode(text, add_special_tokens=False).ids),
        'review_body_standalone_no_special': len(tok.encode(review, add_special_tokens=False).ids),
        'visual_body_standalone_no_special': None if visual is None else len(tok.encode(visual, add_special_tokens=False).ids),
        'review_section_with_header_no_special': len(tok.encode(REVIEW_HEADER + review, add_special_tokens=False).ids),
        'visual_section_with_header_no_special': None if visual is None else len(tok.encode('[VISUAL_EVIDENCE]\n' + visual, add_special_tokens=False).ids),
        'within_current_768_label_limit': len(raw) - 1 <= 768,
        'truncation_performed': False,
    }


def validate_proposal(record, source):
    """Mechanical evidence/serialization checks; cannot verify semantic entailment."""
    if record['business_id'] != source['business_id'] or record['split'] != 'train':
        raise ValueError('Wrong business or split')
    review_map = {r['evidence_id']: r['text'] for r in source['reviews']}
    consensus = record['canonical_review_consensus']
    if consensus != ' '.join(c['text'] for c in record['review_claims']):
        raise ValueError('Canonical consensus must serialize exactly from review claims')
    for claim in record['review_claims']:
        if not claim['evidence']:
            raise ValueError('Uncited review claim')
        for e in claim['evidence']:
            if not e['quote'] or e['quote'] not in review_map.get(e['evidence_id'], ''):
                raise ValueError('Wrong review evidence ID or nonliteral quote')
    for condition in CONDITIONS:
        target = record['targets'][condition]
        review, visual = split_target(target['text'], condition)
        if review != consensus:
            raise ValueError('RO/U/F canonical review consensus differs')
        claims = target['visual_claims']
        captions = [] if condition == 'reviews_only' else source['captions'][condition]
        cmap = {c['evidence_id']: c for c in captions}
        if not captions and (claims or (visual is not None and visual.strip())):
            raise ValueError('Visual assertion with zero condition-visible captions')
        if visual is not None and visual != ' '.join(c['text'] for c in claims):
            raise ValueError('Visual claims and visual text differ')
        for claim in claims:
            photos = set()
            if not claim['evidence']:
                raise ValueError('Uncited visual claim')
            for e in claim['evidence']:
                cap = cmap.get(e['evidence_id'])
                if cap is None or e['photo_id'] != cap['photo_id'] or not e['quote'] or e['quote'] not in cap['model_visible_caption']:
                    raise ValueError('Wrong-condition caption, photo mapping or nonliteral quote')
                photos.add(e['photo_id'])
            if len(photos) != 1:
                raise ValueError('Each visual claim must stay within one photo; no cross-photo relationships')
    return {'status': 'PASS_MECHANICAL_ONLY', 'semantic_support_requires_review': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, default=Path(__file__).resolve().parents[1] / 'tokenizer')
    parser.add_argument('--targets', type=Path, required=True, help='Original or proposed target JSONL; never modified')
    parser.add_argument('--sources', type=Path, help='Optional SOURCE_EVIDENCE.jsonl for mechanical proposal checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tok = load_tokenizer(args.tokenizer)
    source = {} if args.sources is None else {
        r['business_id']: r for r in (json.loads(line) for line in args.sources.read_text(encoding='utf-8').splitlines())}
    results = []
    for line in args.targets.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        result = {'business_id': record['business_id'], 'conditions': {}}
        if args.sources is not None:
            result['structural_validation'] = validate_proposal(record, source[record['business_id']])
        for condition in CONDITIONS:
            value = record['targets'][condition]
            text = value if isinstance(value, str) else value['text']
            result['conditions'][condition] = target_counts(tok, text, condition)
        results.append(result)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'Counted {len(results)} businesses offline; input files unchanged; no truncation.')


if __name__ == '__main__':
    main()
