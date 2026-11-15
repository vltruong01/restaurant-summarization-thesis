"""Read-only upstream audit and lossless packed LED IDs. No API/model execution."""
import argparse
import csv
import hashlib
import json
import re
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage_h_led_dataset_v1.json'
CONDITIONS = ('reviews_only', 'unfiltered', 'filtered')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def stats(values):
    a = np.asarray(values, dtype=np.int64)
    return dict(n=len(a), min=int(a.min()), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                p99=float(np.quantile(a, .99)), max=int(a.max()),
                histogram={str(k): v for k, v in sorted(Counter(values).items())})


def check(condition, message):
    if not condition:
        raise ValueError(message)


def selected_review_texts(block, ids):
    markers = list(re.finditer(r'(?:^|\n\n)\[REV_(\d+)\] stars=[1-5] date=[^\n]+\n', block))
    check([int(m.group(1)) for m in markers] == list(range(1, len(ids) + 1)), 'Review marker/order mismatch')
    return {f'REV_{i+1}': block[m.end():markers[i+1].start() if i+1 < len(markers) else len(block)]
            for i, m in enumerate(markers)}


def load_bound_pointer(path):
    p = read(path)
    check(sha(ROOT / p['completion_report']) == p['completion_report_sha256'], 'Completion pointer hash mismatch')
    return p, read(ROOT / p['completion_report'])


def array_signature(pack):
    h = hashlib.sha256()
    for name in sorted(pack):
        a = np.ascontiguousarray(pack[name])
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main(validate_only=False):
    cfg = read(CONFIG)
    gp, complete = load_bound_pointer(ROOT / cfg['stage_g_pointer'])
    fp, _ = load_bound_pointer(ROOT / cfg['stage_f_pointer'])
    g, f = ROOT / gp['production_directory'], ROOT / fp['production_directory']
    check(complete['businesses'] == 2200 and complete['targets'] == 6600, 'Stage G incomplete')
    validation_path = ROOT / complete['validation_report']
    old_validation = read(validation_path)
    read(g / 'training_readiness_checklist.json')
    length_path = g / 'final_validation/target_lengths.csv'
    with length_path.open(encoding='utf-8', newline='') as stream:
        old_lengths = {(r['business_id'], r['condition']): (int(r['source_tokens']), int(r['target_tokens']))
                       for r in csv.DictReader(stream)}
    check(len(old_lengths) == 6600, 'Incomplete prior target-length report')
    upstream = dict(complete['dataset_sha256'])
    for p in [CONFIG, ROOT / cfg['stage_g_pointer'], ROOT / cfg['stage_f_pointer'],
              ROOT / cfg['source_template'], validation_path, length_path, g / 'training_readiness_checklist.json',
              ROOT / cfg['split_csv'], ROOT / cfg['tokenizer_manifest'], ROOT / cfg['model_config_manifest']]:
        upstream[p.relative_to(ROOT).as_posix()] = sha(p)
    for rel, digest in upstream.items():
        check(sha(ROOT / rel) == digest, f'Upstream hash changed: {rel}')
    tm = read(ROOT / cfg['tokenizer_manifest']); model_manifest = read(ROOT / cfg['model_config_manifest'])
    check(tm['revision'] == model_manifest['revision'] == '38335783885b338d93791936c54bb4be46bebed9', 'Revision mismatch')
    for name, digest in tm['sha256'].items():
        path = ROOT / tm['relative_directory'] / name
        check(sha(path) == digest, 'Tokenizer asset mismatch')
        upstream[path.relative_to(ROOT).as_posix()] = digest
    check(sha(ROOT / model_manifest['config_file']) == model_manifest['config_sha256'], 'Model config hash mismatch')
    upstream[model_manifest['config_file']] = model_manifest['config_sha256']
    mc = read(ROOT / model_manifest['config_file'])
    tok = Tokenizer.from_file(str(ROOT / tm['relative_directory'] / 'tokenizer.json'))
    tok.no_padding(); tok.no_truncation()
    check((tok.token_to_id('<s>'), tok.token_to_id('</s>'), tok.token_to_id('<pad>')) == (0, 2, 1), 'Special-token mismatch')
    with (ROOT / cfg['split_csv']).open(encoding='utf-8', newline='') as stream:
        split_rows = list(csv.DictReader(stream))
    assignment = {r['business_id']: r['split'] for r in split_rows}
    check(len(assignment) == len(split_rows) == 2300, 'Split ID duplicates')
    check(Counter(assignment.values()) == {'train': 2000, 'validation': 200, 'test': 100}, 'Split counts changed')
    output = ROOT / cfg['output_directory']
    if validate_only:
        manifest = read(output / 'manifest.json')
        check(manifest['status'] == 'STAGE_H_DATA_VALIDATED_DECODER_POLICY_PENDING', 'Incomplete Stage H build')
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {'status': 'BUILDING', 'datasets': {}}
    report_rows, summaries, empty_cases = [], {}, []
    review_ids_by_split, photo_ids_by_split = {}, {}
    review_support_checks = visual_support_checks = artifact_checks = 0
    artifacts = {}; dataset_signatures = {}
    for split, expected in cfg['counts'].items():
        # No held-out source/caption/review file is opened.
        sources, captions = {}, {}
        for c in CONDITIONS:
            source_path = f / f'manifests/{split}/{c}_sources.jsonl'
            upstream[source_path.relative_to(ROOT).as_posix()] = sha(source_path)
            sources[c] = {r['business_id']: r for r in rows(source_path)}
            check(len(sources[c]) == expected, 'Stage F source count mismatch')
            if c != 'reviews_only':
                caption_path = f / f'manifests/{split}/{c}_captions.jsonl'
                upstream[caption_path.relative_to(ROOT).as_posix()] = sha(caption_path)
                captions[c] = {r['business_id']: [(v['photo_id'], v['normalized_caption']) for v in r['captions']]
                               for r in rows(caption_path)}
        ro = sources['reviews_only']; consensus = {}; shared_evidence = {}; expected_order = None
        split_review_ids, split_photo_ids = set(), set()
        for c in CONDITIONS:
            rel = f"{gp['production_directory']}/dataset/{split}_{c}.jsonl"
            check(rel in complete['dataset_sha256'], 'Unbound Stage G dataset')
            inputs, labels = array('i'), array('i')
            input_offsets, label_offsets = [0], [0]
            business_ids, source_hashes, target_hashes, caption_counts, review_counts = [], [], [], [], []
            source_lengths, target_lengths, review_lengths, caption_lengths = [], [], [], []
            for row in rows(ROOT / rel):
                bid = row['business_id']
                check(bid not in business_ids and assignment.get(bid) == split, 'Duplicate/wrong-split business')
                check(row['split'] == split and row['condition'] == c, 'Condition alignment failure')
                s, base = sources[c][bid], ro[bid]
                check(row['source'] == s['source'] and text_sha(row['source']) == row['source_sha256'] == s['source_sha256'], 'Source changed')
                check(row['selected_review_ids'] == s['selected_review_ids'] == base['selected_review_ids'], 'Selected reviews changed')
                check(row['selected_photo_ids'] == s['selected_photo_ids'], 'Photo order changed')
                block = base['source'].split('[REVIEWS]\n', 1)[1]
                check(text_sha(block) == base['review_block_sha256'] == s['review_block_sha256'], 'Review text/order changed')
                check(0 < len(row['selected_review_ids']) <= 30, 'Review count invalid')
                if c == 'reviews_only':
                    check(not row['selected_photo_ids'], 'RO has photos')
                    check(not split_review_ids.intersection(row['selected_review_ids']), 'Review ID assigned to multiple businesses')
                    split_review_ids.update(row['selected_review_ids'])
                    review_texts = selected_review_texts(block, row['selected_review_ids'])
                    claims = row['shared_review_evidence']['claims']
                    check(claims and all(v['text'].strip() for v in claims), 'Empty review consensus')
                    for claim in claims:
                        check(len({v['review_id'] for v in claim['support']}) >= 2, 'Insufficient canonical review support')
                        for support in claim['support']:
                            marker = support['review_marker']; i = int(marker.split('_')[1]) - 1
                            check(row['selected_review_ids'][i] == support['review_id'], 'Canonical review-ID mismatch')
                            check(support['quote'] and support['quote'] in review_texts[marker], 'Non-exact/wrong-review quotation')
                            review_support_checks += 1
                    consensus[bid] = '[REVIEW_CONSENSUS]\n' + ' '.join(v['text'] for v in claims)
                    shared_evidence[bid] = row['shared_review_evidence']
                    check(row['target'] == consensus[bid], 'RO target serialization changed')
                    check(row['evidence_records'] == shared_evidence[bid], 'RO evidence mismatch')
                else:
                    cap = captions[c][bid]
                    check([v[0] for v in cap] == row['selected_photo_ids'] and len(cap) <= 5, 'Caption mapping/K mismatch')
                    visual = '[VISUAL_CONTEXT]\n' + '\n\n'.join(f'[IMG_CAP_{i}]\n{v[1]}' for i, v in enumerate(cap, 1))
                    check(row['source'] == base['source'] + '\n\n' + visual, 'Frozen source template/caption text changed')
                    check(row['shared_review_evidence'] == shared_evidence[bid], 'Shared review evidence changed')
                    assessments = row['evidence_records']['assessments']
                    check([(v['caption_id'], v['photo_id']) for v in assessments] == [(f'IMG_CAP_{i}', v[0]) for i, v in enumerate(cap, 1)], 'Assessment mapping mismatch')
                    cap_map = {(f'IMG_CAP_{i}', v[0]): v[1] for i, v in enumerate(cap, 1)}
                    assessment_map = {(v['caption_id'], v['photo_id']): v for v in assessments}
                    visual_claims = row['visual_claims_with_evidence']
                    for claim in visual_claims:
                        check(claim['text'].strip() and claim['support'], 'Empty visual claim/support')
                        for support in claim['support']:
                            key = support['caption_id'], support['photo_id']
                            raw = assessment_map[key]['claims'][support['raw_claim_index']]
                            check(raw['verdict'] == support['teacher_verdict'] == 'verified', 'Unverified visual serialized')
                            check(raw['text'] == claim['text'] and raw['caption_quote'] == support['caption_quote'], 'Visual claim was rewritten')
                            check(support['caption_quote'] and support['caption_quote'] in cap_map[key], 'Non-exact caption quote')
                            visual_support_checks += 1
                    check(row['target'] == consensus[bid] + '\n\n[VISUAL_EVIDENCE]\n' + ' '.join(v['text'] for v in visual_claims), 'Visual target serialization mismatch')
                    if not cap:
                        check(c == 'filtered' and not visual_claims, 'Unexpected empty visual case')
                        empty_cases.append({'business_id': bid, 'split': split, 'condition': c})
                    split_photo_ids.update(row['selected_photo_ids'])
                check(row['target_content_origin'] == 'teacher_API_only_frozen_serialization', 'Non-teacher target origin')
                path = row['target_artifact']
                if path not in artifacts:
                    check(sha(ROOT / path) == row['target_artifact_sha256'], 'Teacher target artifact changed')
                    artifacts[path] = read(ROOT / path)
                    upstream[path] = row['target_artifact_sha256']
                    artifact_checks += 1
                check(artifacts[path]['targets'][c] == row['target'], 'Dataset target not equal to teacher artifact')
                src = tok.encode(row['source'], add_special_tokens=True)
                tgt = tok.encode(row['target'], add_special_tokens=True)
                check(sum(src.special_tokens_mask) == sum(tgt.special_tokens_mask) == 2, 'Unexpected special tokens')
                check(src.ids[0] == tgt.ids[0] == 0 and src.ids[-1] == tgt.ids[-1] == 2, 'BOS/EOS missing')
                rs = len(tok.encode(block, add_special_tokens=True).ids)
                increment = len(src.ids) - base['tokens']
                check(len(src.ids) == s['tokens'] == row['source_tokens'] <= 5000, 'Source overflow/count mismatch')
                check(rs <= 4000 and 0 <= increment <= 750, 'Review/caption allowance exceeded')
                check(len(tgt.ids) == row['target_tokens'] and (len(src.ids), len(tgt.ids)) == old_lengths[bid, c], 'Label length changed')
                check(len(tgt.ids) <= mc['max_decoder_position_embeddings'], 'Architectural decoder overflow')
                business_ids.append(bid); source_hashes.append(text_sha(row['source'])); target_hashes.append(text_sha(row['target']))
                caption_counts.append(len(row['selected_photo_ids'])); review_counts.append(len(row['selected_review_ids']))
                inputs.extend(src.ids); labels.extend(tgt.ids)
                input_offsets.append(len(inputs)); label_offsets.append(len(labels))
                source_lengths.append(len(src.ids)); target_lengths.append(len(tgt.ids)); review_lengths.append(rs); caption_lengths.append(increment)
                report_rows.append(dict(business_id=bid, split=split, condition=c, source_tokens=len(src.ids), target_tokens=len(tgt.ids),
                                        review_tokens=rs, caption_increment=increment, captions=len(row['selected_photo_ids']),
                                        padded_encoder_tokens=((len(src.ids)+1023)//1024)*1024))
            check(len(business_ids) == expected and set(business_ids) == {b for b, s in assignment.items() if s == split}, 'Incomplete cohort')
            if expected_order is None: expected_order = business_ids
            else: check(business_ids == expected_order, 'Business row order differs across conditions')
            pack = dict(input_ids=np.asarray(inputs, dtype=np.int32), labels=np.asarray(labels, dtype=np.int32),
                        input_offsets=np.asarray(input_offsets, dtype=np.int64), label_offsets=np.asarray(label_offsets, dtype=np.int64),
                        business_ids=np.asarray(business_ids, dtype='<U22'), source_sha256=np.asarray(source_hashes, dtype='<U64'),
                        target_sha256=np.asarray(target_hashes, dtype='<U64'), caption_counts=np.asarray(caption_counts, dtype=np.int32),
                        review_counts=np.asarray(review_counts, dtype=np.int32))
            name = f'{split}_{c}'; path = output / (name + '.npz')
            if not validate_only:
                with path.open('xb') as stream: np.savez_compressed(stream, **pack)
            with np.load(path, allow_pickle=False) as loaded:
                check(set(loaded.files) == set(pack), 'NPZ columns mismatch')
                for k in pack: check(np.array_equal(pack[k], loaded[k]), f'Packed IDs/metadata differ: {name}/{k}')
            signature = array_signature(pack); dataset_signatures[name] = signature
            info = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size,
                        content_signature=signature, stage_g_jsonl=rel, stage_g_jsonl_sha256=complete['dataset_sha256'][rel], rows=expected)
            if validate_only: check(info == manifest['datasets'][name], 'Non-deterministic/modified packed dataset')
            else: manifest['datasets'][name] = info
            summaries[name] = dict(rows=expected, source=stats(source_lengths), target=stats(target_lengths), review=stats(review_lengths),
                                   caption_increment=stats(caption_lengths), captions=stats(caption_counts),
                                   total_selected_captions=sum(caption_counts), review_count=stats(review_counts),
                                   zero_caption_businesses=sum(v == 0 for v in caption_counts) if c != 'reviews_only' else None)
            print(json.dumps({'dataset': name, 'rows': expected, 'max_source': max(source_lengths), 'max_target': max(target_lengths), 'roundtrip': 'PASS'}), flush=True)
        review_ids_by_split[split], photo_ids_by_split[split] = split_review_ids, split_photo_ids
        artifacts.clear()
    check(not review_ids_by_split['train'].intersection(review_ids_by_split['validation']), 'Selected review-ID leakage')
    check(not photo_ids_by_split['train'].intersection(photo_ids_by_split['validation']), 'Selected photo-ID leakage')
    check(len(empty_cases) == 10 and {r['business_id'] for r in empty_cases} == set(old_validation['zero_filtered_business_ids']), 'Zero-caption cohort changed')
    for rel, digest in upstream.items(): check(sha(ROOT / rel) == digest, f'Upstream changed during construction: {rel}')
    check(len(report_rows) == 6600, 'Target count mismatch')
    lengths = [r['target_tokens'] for r in report_rows]
    candidates = {'status': 'RECOMMENDATION_ONLY_NOT_FROZEN', 'recommended_label_limit': 768,
                  'recommended_generation_max_length': 768, 'generation_prefix_tokens': 1,
                  'recommended_max_new_tokens_equivalent': 767, 'max_labels_with_BOS_EOS': max(lengths),
                  'max_label_plus_decoder_start': max(lengths)+1, 'decoder_position_capacity': mc['max_decoder_position_embeddings'],
                  'candidates': [{'limit': v, 'labels_exceeding': sum(n > v for n in lengths),
                                  'conservative_generation_sequences_exceeding': sum(n+1 > v for n in lengths)} for v in cfg['candidate_label_limits']],
                  'training_cap_approved': False, 'GPU_generation_behavior_tested': False,
                  'note': 'No label is truncated. Generation max_length includes decoder-start; forced EOS or an early EOS is not a completeness certificate. 768 retains 112 label-token / 111 conservative generation-token headroom; 1024 has no observed label-preservation benefit.'}
    if validate_only:
        check(upstream == manifest['upstream_sha256'], 'Upstream provenance set changed')
        repeat = dict(status='STAGE_H_INDEPENDENT_RETOKENIZATION_PASS', rows=6600, signatures=dataset_signatures,
                      upstream_unchanged=True, packed_arrays_unchanged=True, API_calls=0, model_execution=False)
        if not (output / 'repeat_validation.json').exists(): save(output / 'repeat_validation.json', repeat)
        else: check(read(output / 'repeat_validation.json') == repeat, 'Repeat validation differs')
        print(json.dumps(repeat)); return repeat
    manifest.update(status='STAGE_H_DATA_VALIDATED_DECODER_POLICY_PENDING', spec='specs/008_led_dataset_construction.md',
                    tokenizer_manifest=cfg['tokenizer_manifest'], tokenizer_revision=tm['revision'], source_template=cfg['source_template'],
                    special_tokens=dict(bos=0, eos=2, pad=1, decoder_start=mc['decoder_start_token_id']),
                    budgets=cfg['budgets'], conditions=list(CONDITIONS), counts=cfg['counts'],
                    target_limit=None, upstream_sha256=upstream, validation_report=(output/'validation_report.json').relative_to(ROOT).as_posix(),
                    script_sha256=sha(Path(__file__)), teacher_API_calls=0, training_started=False)
    report = dict(status=manifest['status'], audited_utc=datetime.now(timezone.utc).isoformat(), businesses=2200, targets=6600,
                  summaries=summaries, empty_filtered_cases=empty_cases, review_support_checks=review_support_checks,
                  verified_visual_support_checks=visual_support_checks, teacher_target_artifact_checks=artifact_checks,
                  no_missing_or_duplicate_records=True, shared_consensus_and_reviews_exact=True, source_equals_frozen_stage_f=True,
                  split_business_leakage=0, selected_review_ID_train_validation_overlap=0, selected_photo_ID_train_validation_overlap=0,
                  whole_empty_targets=0, label_truncations=0, source_truncations=0, over_source5000=0, over_review4000=0, over_caption750=0,
                  zero_caption_behavior_preserved=True, all_packed_arrays_roundtrip_equal=True, upstream_sha256_unchanged=True,
                  candidates=candidates, diagnostic_warnings_preserved=dict(verdict_conflict_groups=184, auxiliary_review_ID_entries=288),
                  structural_PASS_is_not_factual_certification=True, held_out_test_content_read=False, held_out_teacher_targets=0,
                  teacher_API_calls=0, GPU_execution=False, training_started=False, packed_total_bytes=sum(v['bytes'] for v in manifest['datasets'].values()))
    save(output/'target_limit_candidate_report.json', candidates)
    save(output/'validation_report.json', report)
    manifest['validation_report_sha256'] = sha(output/'validation_report.json')
    save(output/'manifest.json', manifest)
    with (output/'token_lengths.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    lines = ['# Stage H acceptance report', '', '**Data construction PASS; decoder policy pending researcher approval.**', '',
             '2,200 businesses / 6,600 targets; original Stage G JSONL reused with exact text/evidence/ID checks. Packed arrays contain complete token IDs, not duplicated text or photos.', '',
             '| Dataset | Rows | Source min/mean/max | Label min/mean/max | Captions | Empty visual |', '| --- | ---: | --- | --- | ---: | ---: |']
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['rows']} | {s['source']['min']}/{s['source']['mean']:.2f}/{s['source']['max']} | {s['target']['min']}/{s['target']['mean']:.2f}/{s['target']['max']} | {s['total_selected_captions']} | {s['zero_caption_businesses'] if s['zero_caption_businesses'] is not None else 'N/A'} |")
    lines += ['', 'All counts, exact source/target mappings, canonical review quotes, caption quotes/teacher-verdict support, zero-caption handling, special tokens, budgets, train/validation split IDs and array round trips PASS. No source/target truncation, teacher API or GPU run. Test IDs were used solely for exclusion checks; no test content/results inspected.', '',
              'Recommend 768 label tokens including BOS/EOS and generation max_length768 including decoder-start (max_new_tokens767 equivalent). All labels fit; maximum656; conservative generated sequence657. Proposal only: no training limit adopted. Actual generation/VRAM must be tested after approval.', '',
              'Preserve 184 verdict-conflict groups and 288 auxiliary review-ID diagnostics. Automated PASS is not independent human factual certification. See Stage G reports and qualitative audit.', '',
              f"Compressed packed data: {report['packed_total_bytes']/1024**2:.2f} MiB. Original JSONL text/evidence is referenced rather than copied. Re-tokenization command: `.venv-caption-pilot/Scripts/python.exe scripts/build_stage_h_led_dataset.py --validate-only`.", '',
              'Stop before GPU execution/training. See spec009/readiness report for pending approvals and actual A2 feasibility.']
    (output/'ACCEPTANCE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status','businesses','targets','label_truncations','source_truncations','packed_total_bytes']}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    main(args.validate_only)
