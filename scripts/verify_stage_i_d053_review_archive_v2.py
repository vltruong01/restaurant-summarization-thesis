"""Extract and verify the actual D053 ZIP outside the original workspace.

No model libraries, researcher labels, references or predictions are generated.
Only synthetic tests run; archive payload hashes must remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = args.zip.resolve(strict=True)
    original_archive_sha256 = sha(archive_path)
    extraction = Path(tempfile.mkdtemp(prefix='d053_extracted_zip_v2_')).resolve()
    require(not extraction.is_relative_to(args.original_root.resolve()), 'Extraction inside original project')
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        prefixes = set()
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename,
                    'Unsafe ZIP member')
            prefixes.add(p.parts[0])
            target = (extraction / Path(*p.parts)).resolve()
            require(target.is_relative_to(extraction), 'ZIP path escapes extraction')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                require(not target.exists(), 'Duplicate ZIP member')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
        require(len(prefixes) == 1, 'Expected a single ZIP root')
    root = extraction / prefixes.pop()
    manifest = json.loads((root / 'SHA256_MANIFEST.json').read_text(encoding='utf-8-sig'))
    for row in manifest['files']:
        p = root / row['path']
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Extracted payload hash mismatch')
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'NODE_OPTIONS', 'NODE_PATH',
                 'D053_SYNTHETIC_FIXTURE_OUT', 'D053_UI_EXPORTED_FIXTURE_OUT', 'D053_UI_TRANSITION_FIXTURE_OUT'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    output = args.output_dir.resolve()
    commands = {
        'original83': [sys.executable, '-s', '-X', 'utf8', 'code/run_stage_i_d053_extracted_tests_v2.py',
                       '--root', str(root), '--output', str(output / 'ORIGINAL_83_TESTS.json'),
                       '--forbid-original-root', str(args.original_root.resolve()),
                       '--portable-readiness-script', 'code/check_stage_i_d053_bundle_readiness_v2.py'],
        'portable_readiness': [sys.executable, '-s', '-X', 'utf8',
                               'code/check_stage_i_d053_bundle_readiness_v2.py', '--readiness',
                               '--output', str(output / 'PORTABLE_READINESS.json')],
        'v2_ui_supplemental': ['node', 'code/test_stage_i_d053_ui_v2.js', 'SOURCE_REFERENCE_ANNOTATION.html'],
        'v2_readiness_supplemental': [sys.executable, '-s', '-X', 'utf8', '-m', 'unittest', 'discover',
                                      '-s', 'code', '-p', 'test_stage_i_d053_portable_readiness_v2.py', '-v'],
    }
    results = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                   encoding='utf-8', errors='strict', timeout=180)
        logs = {}
        for kind, content in [('stdout', completed.stdout), ('stderr', completed.stderr)]:
            path = output / (name + '.' + kind + '.log')
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
            logs[kind] = {'file': path.name, 'sha256': sha(path)}
        results[name] = {'returncode': completed.returncode, 'command': command, 'logs': logs}
        require(completed.returncode == 0, name + ' failed; preserved logs: ' + str(output))
    original = json.loads((output / 'ORIGINAL_83_TESTS.json').read_text(encoding='utf-8'))
    require(original['status'] == 'PASS' and original['total_tests'] == 83 and original['skipped'] == 0,
            'Original83 count/pass differs')
    ui = [json.loads(line) for line in (output / 'v2_ui_supplemental.stdout.log').read_text(encoding='utf-8').splitlines() if line.startswith('{')][-1]
    require(ui['status'] == 'PASS' and ui['tests'] == 30 and ui['synthetic_only'], 'V2 UI checks differ')
    require('Ran 8 tests' in (output / 'v2_readiness_supplemental.stderr.log').read_text(encoding='utf-8'), 'Supplemental test count differs')
    for row in manifest['files']:
        require(sha(root / row['path']) == row['sha256'], 'Extracted payload modified during tests')
    require(sha(archive_path) == original_archive_sha256, 'ZIP changed during verification')
    report = {'status': 'PASS', 'archive': str(archive_path), 'archive_sha256': sha(archive_path),
              'archive_bytes': archive_path.stat().st_size, 'extracted_root': str(root),
              'all_payload_sha256_before_after': 'PASS', 'payload_file_count': len(manifest['files']),
              'original83': {'pass': 83, 'skipped': 0, 'annotations': 22, 'evaluation': 34, 'ui': 27},
              'supplemental_v2_ui_tests': 30, 'supplemental_portable_readiness_tests': 8,
              'practical_browser_smoke': 'NOT_EXECUTED_POLICY_BLOCKED_CURRENT_URL_UNDETERMINED',
              'human_labels_created': 0, 'actual_reference_units_created': 0, 'predictions_created': 0,
              'GPU_model_API_execution': False, 'commands': results}
    with (output / 'EXTRACTED_ZIP_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'archive_sha256', 'extracted_root', 'original83',
                                           'supplemental_v2_ui_tests', 'supplemental_portable_readiness_tests']}, indent=2))


if __name__ == '__main__':
    main()
