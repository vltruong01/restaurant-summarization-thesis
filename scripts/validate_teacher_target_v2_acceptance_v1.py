"""CPU-only validator for the fixed32 TRAIN acceptance check; no generation path."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

try:
    from count_led_tokens import load_tokenizer, target_counts, validate_proposal
except ModuleNotFoundError:
    from teacher_redesign_token_count_v1 import load_tokenizer, target_counts, validate_proposal


def check_record(row, source, schema, tok):
    import jsonschema
    if importlib.metadata.version('jsonschema') != '4.23.0':
        raise ValueError('Pinned validator dependency jsonschema==4.23.0 required')
    jsonschema.Draft202012Validator(schema).validate(row)
    if source['split'] != 'train':
        raise ValueError('This acceptance utility is TRAIN32-only; no validation/test records')
    expected = {c: v['utf8_sha256'] for c, v in source['exact_frozen_source_inputs'].items()}
    if row['source_inputs_sha256'] != expected:
        raise ValueError('V2 draft source bindings differ')
    technical = validate_proposal(row, source)
    measurements = {c: target_counts(tok, row['targets'][c]['text'], c)
                    for c in ('reviews_only', 'unfiltered', 'filtered')}
    if not all(m['within_current_768_label_limit'] for m in measurements.values()):
        raise ValueError('Target exceeds unchanged768label ceiling; no truncation/repair')
    warnings = []
    review_count = measurements['reviews_only']['review_body_standalone_no_special']
    if not 170 <= review_count <= 260:
        warnings.append('Review body outside flexible170–260guidance; substantive review, not automatic rejection')
    for c in ('unfiltered', 'filtered'):
        if source['caption_counts'][c] and not 20 <= measurements[c]['visual_body_standalone_no_special'] <= 70:
            warnings.append(c + ': visual body outside flexible20–70guidance')
        if not 200 <= measurements[c]['corrected_training_content_EOS_label_tokens'] <= 330:
            warnings.append(c + ': complete target outside flexible200–330guidance')
    return dict(business_id=row['business_id'], mechanical=technical,
        exact_LED_counts=measurements, guidance_warnings=warnings,
        source_entailment_or_Core_quality_proved=False, researcher_quality_confirmation=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--drafts', type=Path, required=True)
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--schema', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    parse = lambda p: [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    sources, drafts = parse(args.sources), parse(args.drafts)
    by_id = {s['business_id']: s for s in sources}
    if len(sources) != 32 or len(by_id) != 32 or len(drafts) != 32:
        raise ValueError('Exactly32source and32draft business records required')
    if Counter(r['business_id'] for r in drafts) != Counter(by_id.keys()):
        raise ValueError('Missing/duplicate/unexpected draft ID')
    schema = json.loads(args.schema.read_text(encoding='utf-8'))
    tok = load_tokenizer(args.tokenizer)
    results = [check_record(r, by_id[r['business_id']], schema, tok) for r in drafts]
    hashes = {}
    for p in (args.sources, args.drafts):
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(dict(status='PASS_MECHANICAL_ONLY_QUALITY_GATE_PENDING',
            input_SHA256=hashes, businesses=32, target_conditions=96, results=results,
            production_authorized=False, GPU_authorized=False), f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('Mechanical checks/counts complete. No semantic-quality GO or execution authorization.')


if __name__ == '__main__':
    main()
