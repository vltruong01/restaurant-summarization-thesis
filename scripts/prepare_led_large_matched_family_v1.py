"""Single offline materialization; never imports LED model code or opens test data."""
from pathlib import Path
import sys
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from tokenizers import Tokenizer
import led_large_matched_family_v1 as m
from led_large_decoder_label_view_v1 import remove_initial_bos, CorrectedLabelDataset
from stage_hi_led_data import PackedLEDDataset
from led_large_shuffled_fixture_v1 import build_mapping
from led_large_final_evaluation_spec_v1 import sample_fixture, annotation_fixture

R = m.ROOT
P = R / m.PREP
PARENT = R / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'


def token_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def main():
    m.require(not (P / 'LABEL_VIEWS.json').exists(), 'Materialized package already exists; do not overwrite')
    P.mkdir(parents=True, exist_ok=True)
    old = m.read(PARENT / 'configs/stage_i_led_large_decoder_policy_v2.json')
    manifest = m.read(PARENT / old['stage_h_manifest'])
    tok = Tokenizer.from_file(str(PARENT / old['source_tokenizer'] / 'tokenizer.json'))
    views = dict(policy=old['decoder_label_policy'], views={}, originals_overwritten=False)
    source_files, rows_by_key, audits, datasets = {}, {}, [], {}
    old_cases_path = PARENT / old['early_sanity']['cases_file']
    old_cases = [json.loads(s) for s in old_cases_path.read_text(encoding='utf-8').splitlines()]
    for key, info in manifest['datasets'].items():
        m.require(key.split('_', 1)[0] in ('train', 'validation'), 'Refuse held-out packs')
        for field, h in [('path', 'sha256'), ('stage_g_jsonl', 'stage_g_jsonl_sha256')]:
            rel = info[field]
            m.require(m.sha(R / rel) == info[h], 'Original data changed: ' + rel)
            source_files[rel] = info[h]
        data = PackedLEDDataset(info, 5000, 768, root=R)
        datasets[key] = data
        rows = [json.loads(s) for s in (R / info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
        rows_by_key[key] = rows
        split, condition = key.split('_', 1)
        m.require(len(rows) == len(data) == (2000 if split == 'train' else 200), 'Row count mismatch')
        labels, offsets = [], [0]
        for i, row in enumerate(rows):
            feature = data[i]
            m.require(row['business_id'] == data.business_ids[i] and row['split'] == split
                      and row['condition'] == condition, 'Identity/split/condition mismatch')
            m.require(hashlib.sha256(row['source'].encode()).hexdigest() == row['source_sha256'], 'Source text hash mismatch')
            m.require(tok.encode(row['source']).ids == feature['input_ids'], 'Source text/token mismatch')
            m.require(tok.encode(row['target']).ids == feature['labels'], 'Original Teacher text/label mismatch')
            fixed = remove_initial_bos(feature['labels'])
            labels.extend(fixed); offsets.append(len(labels))
            audits.append(dict(key=key, business_id=row['business_id'], split=split, condition=condition,
                               source_sha256=row['source_sha256'], target_text_sha256=hashlib.sha256(row['target'].encode()).hexdigest(),
                               original_target_artifact_sha256=row['target_artifact_sha256'],
                               input_token_sha256=token_hash(feature['input_ids']),
                               original_label_token_sha256=token_hash(feature['labels']), corrected_label_token_sha256=token_hash(fixed),
                               review_ids=row['selected_review_ids'], photo_ids=row['selected_photo_ids'],
                               original_label_length=len(feature['labels']), corrected_label_length=len(fixed),
                               source_length=len(feature['input_ids']), sole_change='remove_exactly_initial_BOS'))
        if condition in m.CONDITIONS:
            path = P / f'{key}_content_eos_labels.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, business_ids=np.asarray(data.business_ids), labels=np.asarray(labels, dtype=np.int32),
                                    label_offsets=np.asarray(offsets, dtype=np.int64))
            views['views'][key] = dict(path=path.relative_to(R).as_posix(), sha256=m.sha(path), rows=len(data),
                                      original_npz=info['path'], original_npz_sha256=info['sha256'])
            CorrectedLabelDataset(data, views['views'][key], R)
    for split in ('train', 'validation'):
        ids = datasets[split + '_filtered'].business_ids
        for condition in m.CONDITIONS:
            m.require(datasets[split + '_' + condition].business_ids == ids, 'Cross-condition order mismatch')
            a, b = rows_by_key[split + '_filtered'], rows_by_key[split + '_' + condition]
            m.require(all(x['selected_review_ids'] == y['selected_review_ids'] for x, y in zip(a, b)), 'Selected reviews changed')
    m.write(P / 'LABEL_VIEWS.json', views)
    with (P / 'IMMUTABLE_ROW_AUDIT.jsonl').open('x', encoding='utf-8') as f:
        for row in audits:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    for condition in m.CONDITIONS:
        c = m.expected_config(old, condition)
        m.write(m.paths(condition), c)
        key = 'validation_' + condition
        ds, rows = datasets[key], rows_by_key[key]
        with (P / f'{condition}_validation_cases.jsonl').open('x', encoding='utf-8') as f:
            for original_case in old_cases:
                i = ds.business_ids.index(original_case['business_id'])
                row = dict(business_id=ds.business_ids[i], condition=condition, split='validation',
                           subset=original_case['subset'], source_sha256=rows[i]['source_sha256'],
                           input_ids=ds[i]['input_ids'], label_ids=ds[i]['labels'])
                f.write(json.dumps(row, sort_keys=True) + '\n')
        ds, rows = datasets['train_' + condition], rows_by_key['train_' + condition]
        longest_source = max(range(len(ds)), key=lambda i: len(ds[i]['input_ids']))
        longest_target = max(range(len(ds)), key=lambda i: len(ds[i]['labels']))
        indices = list(dict.fromkeys([longest_source, longest_target]))
        if len(indices) < 2:
            indices.append(next(i for i in range(len(ds)) if i not in indices))
        m.write(P / f'{condition}_FEASIBILITY_CASES.json', dict(business_ids=[ds.business_ids[i] for i in indices],
                selection='TRAIN-only source and target maxima, deterministic original-order ties',
                cases=[dict(business_id=ds.business_ids[i], source=len(ds[i]['input_ids']),
                            original_target=len(ds[i]['labels']), corrected_target=len(ds[i]['labels'])-1) for i in indices],
                padded_source=5120, padded_label=768, padding_is_not_semantic_full_length=True))
    # Only TRAIN/validation source fixtures, explicitly artificial eligibility edges.
    fixtures, edges = [], {}
    for split in ('train', 'validation'):
        rows = rows_by_key[split + '_filtered']
        zero = [r for r in rows if not r['selected_photo_ids']]
        if zero:
            fixtures.append(zero[0])
        for count in (1, 2, 3, 4, 5):
            group = [r for r in rows if len(r['selected_photo_ids']) == count][:2]
            if len(group) == 2:
                fixtures.extend(group)
                for i, r in enumerate(group):
                    edges[r['business_id']] = [group[1-i]['business_id']]
    safe = [{k: r[k] for k in ('business_id', 'split', 'source', 'source_sha256', 'selected_photo_ids', 'selected_review_ids')}
            for r in fixtures]
    m.write(P / 'SHUFFLED_SOURCE_FIXTURES.json', dict(fixture_only=True, rows=safe, explicit_test_edges=edges,
            warning='Edges exercise constraints only; they are NOT cuisine eligibility judgments'))
    result = build_mapping(safe, edges, tok)
    m.require(result == build_mapping(safe, edges, tok), 'Fixture nondeterminism')
    m.write(P / 'SHUFFLED_FIXTURE_RESULTS.json', result)
    # Synthetic identities avoid generating or exposing actual held-out sample IDs.
    frame = [dict(business_id=f'FIXTURE_{i:03}', review_tokens=1000+i*25, usable_photo_count=i%6, split='train') for i in range(100)]
    sampling = sample_fixture(frame)
    m.write(P / 'SAMPLING_FIXTURE.json', sampling)
    m.write(P / 'ANNOTATION_BLANK_FIXTURE.json', annotation_fixture(sampling['ids']))
    rels = ['VAST_SHA256_MANIFEST.json', 'configs/stage_i_led_large_decoder_policy_v2.json',
            'scripts/run_stage_i_led_large_decoder_policy_v2.py', 'scripts/stage_i_led_large_decoder_policy_v2.py',
            'scripts/led_large_decoder_label_view_v1.py', 'scripts/led_large_decoder_early_sanity_v2.py',
            'scripts/led_large_dynamic_amp_v2.py', old['stage_h_manifest'], old['effective_generation_config_file'],
            'outputs/stage_i/led_large_preparation_v1/EPOCH_ORDERS.json',
            'outputs/stage_i/led_large_decoder_policy_v2/PREFLIGHT_MANIFEST.json']
    rels += [p.relative_to(PARENT).as_posix() for p in (PARENT / old['source_tokenizer']).iterdir() if p.is_file()]
    # RO/U originals travel in the new overlay; the accepted F-only parent stays unchanged.
    m.write(P / 'ORIGINAL_SOURCE_FILES.json', dict(files={p: h for p, h in source_files.items()
            if '_filtered' not in p}, F_originals_audited_not_copied=True))
    m.write(P / 'PARENT_REQUIRED_FILES.json', dict(files={p: m.sha(PARENT / p) for p in rels}, full_parent_preflight_required=True,
            model_sha256=old['weights_sha256'], model_revision=old['revision']))
    m.write(P / 'MATERIALIZATION_REPORT.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            status='OFFLINE_RO_U_VIEWS_VALIDATED', audited_original_examples=len(audits),
            new_RO_U_views=4, corrected_RO_U_examples=4400, original_F_audited_examples=2200,
            same_business_and_review_order=True, full_text_to_token_identity=True,
            original_source_and_target_bytes_unchanged=True, original_labels_unchanged=True,
            GPU_operations=0, held_out_payloads_opened=0, Teacher_calls=0, optimizer_updates=0,
            original_stage_H_status='DECODER_POLICY_PENDING is historical; corrected views are separately versioned',
            shuffled_fixture_rows=len(safe), production_Shuffled_mapping='BLOCKED_CUISINE_FALLBACK_ADJUDICATION'))
    print('Materialization PASS:', len(audits), 'audited; 4400 RO/U labels derived; fixtures', len(safe))


if __name__ == '__main__':
    main()
