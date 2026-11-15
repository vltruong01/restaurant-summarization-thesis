"""Offline post-training preservation check; hashes existing artifacts only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d052_paired_pilot_v1'
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    before = read(OUT / 'PRE_EXECUTION_INTEGRITY.json')
    config_path = ROOT / 'configs/stage_i_paired_lora_pilot_v1.json'
    config = read(config_path)
    cache_base = Path.home() / '.cache/huggingface/hub'
    cached = cache_base / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['revision']
    groups = {
        'prepared_files': PREP,
        'code': ROOT,
        'upstream': ROOT,
        'original_archive_files': ROOT,
        'cached_backbone_files': cached,
        'D050_D051_prior_feasibility_files': ROOT,
    }
    checks = {}
    failures = []
    for group, base in groups.items():
        records = before[group]
        for relative, record in records.items():
            path = base / relative
            expected = record['sha256'] if isinstance(record, dict) else record
            actual = sha(path) if path.is_file() else None
            if actual != expected:
                failures.append({'group': group, 'file': relative, 'expected': expected, 'actual': actual})
            if isinstance(record, dict) and 'bytes' in record:
                actual_bytes = path.stat().st_size if path.is_file() else None
                if actual_bytes != record['bytes']:
                    failures.append({'group': group, 'file': relative, 'expected_bytes': record['bytes'],
                                     'actual_bytes': actual_bytes})
        checks[group] = {'checked_files': len(records), 'unchanged': not any(f['group'] == group for f in failures)}
    singles = {
        'config_sha256': config_path,
        'preparation_manifest_sha256': PREP / 'PREPARATION_MANIFEST.json',
        'source_authorization_sha256': OUT / 'USER_AUTHORIZATION.md',
    }
    for key, path in singles.items():
        if sha(path) != before[key]:
            failures.append({'group': 'single', 'file': str(path), 'expected': before[key], 'actual': sha(path)})
    approval_path = OUT / 'approval_D052_train_only.json'
    approval = read(approval_path)
    for key, path in {
        'wrapper_sha256': ROOT / 'scripts/run_stage_i_d052_paired_pilot_v2.py',
        'authorization_module_sha256': ROOT / 'scripts/stage_i_d052_authorization.py',
        'config_sha256': config_path,
        'preparation_manifest_sha256': PREP / 'PREPARATION_MANIFEST.json',
        'protocol_sha256': PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md',
    }.items():
        if sha(path) != approval[key]:
            failures.append({'group': 'D052_authorization_binding', 'file': str(path)})
    for record in [approval['user_request'], approval['decision_record'], approval['operational_correction']]:
        path = Path(record['path'])
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != record['sha256']:
            failures.append({'group': 'D052_authorization_binding', 'file': str(path)})
    for arm in ['A_decoder_only', 'B_encoder_decoder']:
        observer = read(OUT / arm / 'observation_completion.json')
        train = ROOT / config['run_root'] / arm / 'train'
        completion = read(train / 'completion.json')
        if (observer['status'] != 'TRAINING_AND_AUDIT_PASS' or observer['generation_calls'] != 0
                or completion['steps'] != 128 or completion['epochs'] != 2 or completion['generation_calls'] != 0):
            failures.append({'group': 'completion', 'arm': arm})
        if any((train.parent / name).exists() for name in ['generation', 'generate']):
            failures.append({'group': 'unauthorized_generation_directory', 'arm': arm})
    result = {
        'status': 'PASS' if not failures else 'FAIL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'pre_execution_integrity_sha256': sha(OUT / 'PRE_EXECUTION_INTEGRITY.json'),
        'groups': checks, 'single_hash_bindings_checked': len(singles),
        'D052_approval_sha256': sha(approval_path),
        'failures': failures,
        'generation_calls': 0,
        'held_out_access': False,
        'scope_note': 'Training code uses the frozen 512 train and 200 validation IDs only; no test dataset is opened by this preservation check. No new summary generation was executed.',
        'D051_strict_replay_status': 'FAIL_UNCHANGED',
    }
    target = OUT / 'POST_EXECUTION_INTEGRITY.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'groups': checks, 'failures': failures}, indent=2))
    if failures:
        raise ValueError('Preservation mismatch: see POST_EXECUTION_INTEGRITY.json')


if __name__ == '__main__':
    main()
