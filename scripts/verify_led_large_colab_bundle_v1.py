"""Portable offline data preflight. No torch import, model load, or GPU operation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(root, rel):
    p = PurePosixPath(rel)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in rel
            and ':' not in rel, 'Unsafe manifest path')
    dest = (root / rel).resolve()
    require(dest.is_relative_to(root.resolve()), 'Path outside extracted root')
    return dest


def verify(root, tokenize=False):
    import numpy as np
    root = Path(root).resolve()
    contract = read(root / 'TRANSFER_CONTRACT.json')
    require(contract['GPU_execution_approved'] is False, 'Transfer is not GPU approval')
    manifest = read(root / 'SHA256_MANIFEST.json')['files']
    for rel, digest in manifest.items():
        require(sha(safe_path(root, rel)) == digest, 'Hash mismatch: ' + rel)
    cfg = read(root / 'configs/stage_i_led_large_proposal_v1.json')
    require(cfg['model_id'] == 'allenai/led-large-16384'
            and cfg['revision'] == 'cd59d11c3528415b7dda4dfc95cc8f138aceda2e', 'Initialization changed')
    require(cfg['expected_modules'] == 228 and cfg['expected_trainable_parameters'] == 4915200,
            'LoRA scope changed')
    require(cfg['source_limit'] == 5000 and cfg['label_limit'] == 768, 'Budgets changed')
    require(not cfg['GPU_execution_approved'] and not cfg['training_authorized'], 'Offline flags changed')
    hm = read(root / cfg['stage_h_manifest'])
    transfer = read(root / 'TRANSFER_DATASETS.json')
    require(set(transfer['datasets']) == {'train_filtered', 'validation_filtered'}, 'F-only payload required')
    require(all(info == hm['datasets'][name] for name, info in transfer['datasets'].items()),
            'Transfer subset changed frozen dataset metadata')
    tok = None
    if tokenize:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(root / cfg['source_tokenizer'] / 'tokenizer.json'))
        tok.no_truncation()
        tok.no_padding()
    groups, stats, photos, artifacts, targets, zero_ids = {}, {}, set(), set(), {}, {}
    for split, count in [('train', 2000), ('validation', 200)]:
        groups[split] = {}
        for cond in ('filtered',):
            name = split + '_' + cond
            info = hm['datasets'][name]
            require(sha(root / info['path']) == info['sha256'], 'Frozen NPZ differs')
            require(sha(root / info['stage_g_jsonl']) == info['stage_g_jsonl_sha256'], 'Frozen JSONL differs')
            rows = [json.loads(s) for s in (root / info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
            with np.load(root / info['path'], allow_pickle=False) as z:
                pack = {k: z[k] for k in z.files}
            ids = pack['business_ids'].tolist()
            require(ids == [x['business_id'] for x in rows] and len(ids) == count
                    and len(set(ids)) == count, 'Business alignment differs: ' + name)
            groups[split][cond] = ids
            zero_ids[name] = []
            for i, row in enumerate(rows):
                require(row['split'] == split and row['condition'] == cond, 'Split/condition differs')
                require(hashlib.sha256(row['source'].encode()).hexdigest() == row['source_sha256'], 'Source digest differs')
                src = pack['input_ids'][pack['input_offsets'][i]:pack['input_offsets'][i+1]].tolist()
                lab = pack['labels'][pack['label_offsets'][i]:pack['label_offsets'][i+1]].tolist()
                require(len(src) == row['source_tokens'] <= 5000 and len(lab) == row['target_tokens'] <= 768,
                        'Length mismatch; silent truncation forbidden')
                require(src[0] == lab[0] == 0 and src[-1] == lab[-1] == 2, 'BOS/EOS differs')
                if tok:
                    require(tok.encode(row['source']).ids == src and tok.encode(row['target']).ids == lab,
                            'Token content differs: ' + row['business_id'])
                art = row['target_artifact']
                if art not in artifacts:
                    require(sha(safe_path(root, art)) == row['target_artifact_sha256'], 'Teacher artifact differs')
                    artifacts.add(art)
                original = read(root / art)
                require(original['business_id'] == row['business_id'] and original['targets'][cond] == row['target'],
                        'Teacher content differs from packed target')
                rc = [original['targets'][c].split('\n\n[VISUAL_EVIDENCE]\n')[0]
                      for c in ('reviews_only', 'unfiltered', 'filtered')]
                require(rc[0] == rc[1] == rc[2], 'Shared review consensus differs in preserved Teacher artifact')
                targets[(row['business_id'], cond)] = row['target']
                require(row['target'].startswith('[REVIEW_CONSENSUS]\n'), 'Review section absent')
                if cond != 'reviews_only':
                    require(row['target'].count('[VISUAL_EVIDENCE]') == 1, 'Visual section count differs')
                if cond == 'filtered' and not row['selected_photo_ids']:
                    zero_ids[name].append(row['business_id'])
                    require(row['target'].split('[VISUAL_EVIDENCE]\n', 1)[1].strip() == '',
                            'Approved zero-caption representation changed')
                for pid in row['selected_photo_ids']:
                    require((root / 'data/processed/yelp_2300_v1/photos' / (pid + '.jpg')).is_file(), 'Missing original photo')
                    photos.add(pid)
            stats[name] = dict(rows=count, max_source=int(np.diff(pack['input_offsets']).max()),
                               max_target=int(np.diff(pack['label_offsets']).max()),
                               zero_filtered=len(zero_ids[name]))
    require(not set(groups['train']['filtered']) & set(groups['validation']['filtered']), 'Train/validation leakage')
    require(len(zero_ids['train_filtered']) == 9 and len(zero_ids['validation_filtered']) == 1,
            'Zero-caption population differs')
    cases = [json.loads(s) for s in (root / 'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation = read(root / 'outputs/stage_i/led_large_preparation_v1/EVALUATION_MANIFEST.json')
    require([c['business_id'] for c in cases] == evaluation['business_ids'] and len(cases) == 24,
            'Fixed evaluation sample differs')
    require(set(evaluation['random20']) | set(evaluation['targeted4']) == set(evaluation['business_ids']), 'Subset mismatch')
    refs = read(root / cfg['final_references'])
    require({u['business_id'] for u in refs['units']} == set(evaluation['business_ids']), 'Reference cases differ')
    require(len(photos) == contract['selected_original_photos'] and len(artifacts) == 2200, 'Incomplete payload')
    result = dict(status='PASS_OFFLINE_TRANSFER_INTEGRITY_ONLY', files_verified=len(manifest),
                  businesses=2200, training_validation_pairs=2200, condition='filtered',
                  teacher_artifacts=len(artifacts), original_photos=len(photos),
                  datasets=stats, exact_tokenization_checked=bool(tok), evaluation_cases=24,
                  model_forward_backward_generation=0, GPU_operations=0, held_out_content_access=False,
                  training_runner_status='HISTORICAL_BASE_REFERENCE_ONLY_LARGE_RUNNER_REVIEW_REQUIRED')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--tokenize', action='store_true', help='Check every source and target against original token IDs')
    ap.add_argument('--runtime', action='store_true', help='Check pinned package versions without importing torch')
    ap.add_argument('--report', type=Path)
    a = ap.parse_args()
    if a.runtime:
        import importlib.metadata
        cfg = read(a.root / 'configs/stage_i_led_large_proposal_v1.json')
        require(sys.version_info[:2] == (3, 11), 'Python 3.11 required; no implicit runtime fallback')
        actual = {k: importlib.metadata.version(k) for k in cfg['runtime_pins']}
        require(actual == cfg['runtime_pins'], 'Pinned ML runtime mismatch')
    result = verify(a.root, a.tokenize)
    result['runtime_pins_checked'] = a.runtime
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        with a.report.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
