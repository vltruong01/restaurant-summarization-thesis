"""CPU-only archive/receipt verification before retiring the remote instance."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path('C:/Users/VUONGLOCTRUONG/Downloads')
DEST = ROOT / 'outputs/stage_i/led_large_vast_d059_v1_download_audit'
BINDING = json.loads((ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1/VAST_EXECUTION_BINDINGS.json').read_text(encoding='utf-8'))

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def audit_archive(path):
    hashes = {}
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate ZIP paths')
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename
                    and ':' not in info.filename, 'Unsafe ZIP member')
            if info.is_dir():
                continue
            with z.open(info) as f:
                hashes[info.filename] = hashlib.file_digest(f, 'sha256').hexdigest()
    return hashes

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='Offline download preservation audit; no GPU or pickle deserialization', archives={}, checks=[])
    names = ['F_large_d059_v1.zip', 'launcher_logs.zip', 'LED_LARGE_F_QUALITY_REVIEW.zip']
    all_hashes = {}
    for name in names:
        p = DOWNLOADS/name
        all_hashes[name] = audit_archive(p)
        report['archives'][name] = dict(bytes=p.stat().st_size, sha256=sha(p), members_crc_and_sha256_checked=len(all_hashes[name]))
    print('All three ZIPs fully decompressed and CRC/SHA256 scanned', flush=True)
    run_hashes = all_hashes[names[0]]
    prefix = 'F_large_d059_v1/'
    with zipfile.ZipFile(DOWNLOADS/names[0]) as z:
        def read(rel):
            return json.loads(z.read(prefix+rel))
        gate = read('feasibility/REPORT.json')
        train = read('train/COMPLETION.json')
        evaluation = read('evaluate/COMPLETION.json')
        # Current execution bindings file also carries provenance metadata.
        expected = {k: BINDING[k] for k in ('config_sha256','runner_sha256','preflight_manifest_sha256')}
        for doc in (gate, train, evaluation):
            require(doc['binding'] == expected, 'Execution binding mismatch')
        require(gate['status'] == 'PASS' and gate['optimizer_updates'] == 3 and gate['headroom_fraction'] >= .15, 'Gate evidence incomplete')
        require(train['steps'] == 1250 and train['epochs'] == 5, 'Training budget incomplete')
        require(evaluation['generation_calls'] == 48, 'Evaluation incomplete')
        require(not any(n.endswith('FAILURE_STOP_NO_RETRY.json') for n in run_hashes), 'Failure marker in run')
        required = {'adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json','training_args.bin'}
        checked = []
        for step in (250,500,750,1000,1250):
            rel = f'train/checkpoint-{step}/'
            receipt = read(rel+'checkpoint_integrity.json')
            require(receipt['binding'] == expected and receipt['global_step'] == step, 'Checkpoint binding differs')
            require(required.issubset(receipt['files']), 'Missing full-state checkpoint files')
            for name,digest in receipt['files'].items():
                require(run_hashes.get(prefix+rel+name) == digest, 'Checkpoint checksum differs: '+rel+name)
            state = read(rel+'trainer_state.json')
            require(state['global_step'] == step and state['max_steps'] == 1250, 'Trainer step differs')
            checked.append(dict(step=step, files_verified=len(receipt['files'])))
        report['checkpoints'] = checked
        receipts = [json.loads(x) for x in z.read(prefix+'train/CHECKPOINT_RECEIPTS.jsonl').decode().splitlines()]
        require([x['step'] for x in receipts] == [250,500,750,1000,1250], 'Checkpoint receipt sequence differs')
        for row in receipts:
            require(row['checkpoint_integrity_sha256'] == run_hashes[prefix+f"train/checkpoint-{row['step']}/checkpoint_integrity.json"], 'Receipt chain differs')
        for log in ('progress.jsonl','gradient_checks.jsonl'):
            rows = [json.loads(x) for x in z.read(prefix+'train/'+log).decode().splitlines()]
            require([r['step'] for r in rows] == list(range(1,1251)), 'Missing/duplicate training steps: '+log)
        for epoch in (3,5):
            rows = read(f'evaluate/epoch{epoch}/predictions.json')
            require(len(rows) == 24 and len({x['business_id'] for x in rows}) == 24, 'Prediction count differs')
            require(sum(x['subset']=='random' for x in rows) == 20 and sum(x['subset']=='targeted' for x in rows) == 4, 'Subset differs')
        bundle = read('QUALITY_BUNDLE_RECEIPT.json')
        actual = report['archives'][names[2]]
        require(bundle['sha256'] == actual['sha256'] and bundle['bytes'] == actual['bytes'], 'Downloaded review ZIP differs from remote receipt')
        require(run_hashes[prefix+names[2]] == actual['sha256'], 'Nested review ZIP differs')
        manifest = read('quality_review/SHA256_MANIFEST.json')['files']
        for rel,digest in manifest.items():
            require(run_hashes.get(prefix+'quality_review/'+rel) == digest, 'Run quality manifest differs: '+rel)
            require(all_hashes[names[2]].get(rel) == digest, 'Review ZIP manifest differs: '+rel)
        integrity = read('feasibility/checkpoint_step3/integrity.json')
        report['gate_checkpoint_integrity'] = integrity
        for name,digest in integrity['files'].items():
            require(run_hashes.get(prefix+'feasibility/checkpoint_step3/'+name) == digest, 'Gate checkpoint differs')
        report['quality_manifest_members_verified'] = len(manifest)
        report['training_summary'] = train
        report['evaluation_summary'] = evaluation
    with zipfile.ZipFile(DOWNLOADS/names[1]) as z:
        log = z.read('launcher_logs/large_d059_v1.log').decode(errors='replace')
        require('Finished 48 outputs. Stop for researcher review' in log, 'Missing successful launcher closeout')
        require('launcher_logs/HOST_PREFLIGHT.json' in all_hashes[names[1]], 'Missing GPU host evidence')
    original = ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1.zip'
    require(sha(original) == 'e04a9cf4977bc9be743fddf2fbe4846466585280de3e306657a550be7dd827c1', 'Original upload archive missing/changed')
    backup = DEST/'downloaded_archives'
    backup.mkdir(exist_ok=True)
    for name in names:
        target = backup/name
        if not target.exists():
            shutil.copyfile(DOWNLOADS/name, target)
        require(sha(target) == report['archives'][name]['sha256'], 'Workspace backup differs')
    report['status'] = 'PASS_REQUIRED_RUN_ARTIFACTS_PRESERVED_LOCALLY'
    report['limits'] = 'File integrity verified, not runtime checkpoint reload or substantive quality GO. Remote unreported edits/files cannot be inventoried from local downloads.'
    (DEST/'DOWNLOAD_PRESERVATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checkpoints=checked, predictions=48, archives=report['archives'], backup=str(backup)),indent=2),flush=True)

if __name__ == '__main__':
    main()
