"""Correct portable D053 packaging only. No labels, model execution or API calls.

Usage: --prepare, then --pack --zip NEW.zip. Never overwrites published ZIPs.
Frozen v1 material remains preserved. UI v2 keeps the original JSON payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'stage_i_d053_reference_verification_chatgpt_bundle_v2'
DEST = ROOT / 'outputs/stage_i' / NAME
OUT = ROOT / 'outputs/stage_i/d053_standalone_bundle_correction_v2'
V1 = ROOT / 'outputs/stage_i/stage_i_d053_reference_verification_chatgpt_bundle_v1.zip'
V1_SHA = '115a208b37b7b01e57c78262ef436acbd739b5e2d0607356046de45ddc6d06a1'
CONFIG_SHA = 'a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def copy(source, relative):
    target = DEST / relative
    require(not target.exists(), 'Refuse overwrite: ' + relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(target) == sha(source), 'Copy differs: ' + relative)


def archive_old(relative):
    source = DEST / relative
    target = DEST / 'provenance/v1' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    require(source.is_file() and not target.exists(), 'Cannot archive copy: ' + relative)
    source.rename(target)


README = '''# D-053 corrected standalone review bundle v2

Offline preparation only. Open SOURCE_REFERENCE_ANNOTATION.html after extracting
the complete ZIP. Keep images/ beside it. No server, model, API key or network is
needed for annotation. Your existing verified annotations are not replaced.
The delivered initial state has 0 human decisions and 0 reference units.

## Researcher workflow (required)

For each business, open “Đọc toàn bộ nguồn của business”. Inspect EVERY complete
selected review and EVERY caption with its corresponding original PHOTO, including
sources not quoted by the Teacher candidates. There are 652 reviews and 90 photos
across the fixed 20 random and four targeted validation businesses.

The 226 unverified Teacher candidates are suggestions, NOT a complete reference.
Use “+ Thêm ý quan trọng từ nguồn” to add important omitted information as separately
verified atomic units with exact REV / IMG_CAP / PHOTO evidence and quotations.
Missing-information notes alone do not add a reference unit. Preserve meaningful
positive, negative and contrasting opinions. Visual content must be recoverable
from the supplied caption AND corroborated by the matching original photo; do not
add photo-only facts. Correct but simple descriptions are acceptable.

Label and explicitly confirm candidates yourself. Candidate importance and
informativeness are optional. Final supported units require all specified metadata
and separate researcher confirmation. Save progress frequently. Incomplete final
export must stay blocked. Verify one or two businesses before proceeding to all24.
ChatGPT is an optional comparison aid; it does not assign human labels.

## Portable offline commands

Use Python 3.11+ with NumPy1.26.4 and jsonschema4.23.0, and Node.js (tested22.14.0).
These are CPU-only third-party dependencies, not original-project dependencies.
If needed: `python -m pip install -r requirements-offline.txt`.
Run these commands from this extracted folder:

```text
python -s -X utf8 code/check_stage_i_d053_bundle_readiness_v2.py --readiness
python -s -X utf8 code/run_stage_i_d053_extracted_tests_v2.py --root . --output MY_OFFLINE_TEST_RESULTS.json --portable-readiness-script code/check_stage_i_d053_bundle_readiness_v2.py
node code/test_stage_i_d053_ui_v2.js
python -s -X utf8 -m unittest discover -s code -p test_stage_i_d053_portable_readiness_v2.py -v
```

The original83 tests are unchanged:22 annotation +34 evaluation +27 UI. Supplemental
v2 checks are reported separately. Use a NEW report filename on each run.

Readiness verifies14 included frozen bindings, both adapter byte hashes (without
loading), source evidence, blank annotations and the final-export guard. The full
historical case-input file is withheld because it contains previous outputs/Teacher
targets; the blind display key stays separate. Their unchanged hashes remain in
PORTABLE_BINDING_MAP.json and the frozen config, explicitly NOT_REVERIFIED here.
Thus preparation readiness is not the full future scientific evaluation gate.
Confirmed references, full-project binding validation and separate approval are
still required before generating48 Pilot A/B summaries. See FUTURE_EVALUATION_USAGE.md.

## Corrections and evidence

schema/ is corrected to schemas/; the unchanged stage_i_paired_pilot.py is now
included for AST-only bootstrap compatibility testing. The frozen evaluator and
config are unchanged. A new portable checker resolves explicit package paths.
UIv2 fixes optional candidate priority controls to match the accepted validator;
all final-unit confirmation safeguards remain. Its embedded source payload is
identical to v1. Old interface, instructions and manifests remain in provenance/v1/.

Read PORTABILITY_CORRECTION_REPORT.md and validation_v2/. Practical Chrome/Edge
smoke testing was blocked by computer-use policy because the current browser URL
could not be established reliably. It is NOT reported PASS. Automated synthetic
UI tests are not a substitute for that practical check. MANUAL_BROWSER_SMOKE_TEST.md
lists the remaining check; use a throwaway copy/profile and synthetic practice data.

SHA256_MANIFEST.json and SHA256SUMS.txt cover the corrected payload. The original
EVALUATION_FREEZE_MANIFEST.json remains byte-identical historical provenance; use
the NEW SHA manifest for v2 packaging. No student predictions, human annotations,
reference units, training or Teacher calls were created by this correction.
'''

FUTURE = '''# D-053 portable preparation readiness and future evaluation

From the extracted bundle root:

```text
python -s -X utf8 code/check_stage_i_d053_bundle_readiness_v2.py --readiness
```

This command is fully standalone for the provided review material. It validates
14 unchanged config bindings and two adapter SHA-256 hashes without loading a
model. It also checks all24 sources,652 full reviews,90 photos,226 candidates,
1,407 exact evidence quotations and blank initial human state. No predictions
are generated. The initial state MUST fail final-reference acceptance.

Frozen config SHA-256:
`a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461`.
Original config paths remain logical project provenance, not paths the portable
command tries to open. PORTABLE_BINDING_MAP.json explicitly binds those entries
to supplied relative files with the SAME hashes.

Two entries are deliberately NOT_REVERIFIED_IN_STANDALONE: original case_inputs
(historical targets/student outputs withheld from source annotation) and
blinding_key (kept separate). The full original scientific readiness gate remains
PENDING and unchanged; this wrapper does not impersonate its PASS or grant future
generation permission. Both adapters are provided as inert hash-verification
evidence, not permission to load or run them.

The original full-project evaluator and usage instructions remain byte-identical
under code/stage_i_d053_evaluation.py and provenance/v1/FUTURE_EVALUATION_USAGE.md.
Its full-project --readiness / --compare-existing commands remain for later use in
the complete project after researcher-confirmed references and separate approval.
They are NOT the standalone preparation command. No new generation is authorized.

All original evaluation definitions, seeds, uncertainty handling, case IDs,
generation settings and scientific hashes remain frozen. Teacher agreement is
not independent factual truth. Missing human judgments remain N/A. The20 random
and four targeted businesses must always be reported separately.
'''

SMOKE = '''# Practical Chrome/Edge smoke test: pending

Status: NOT_EXECUTED_POLICY_BLOCKED. Windows computer-use could not reliably
determine the current Chrome URL and stopped. No alternate automation was used.
Synthetic Node/DOM checks are separate and do not establish browser rendering,
file-dialog, downloaded-file or localStorage behavior in Chrome/Edge.

Before real annotation, use a throwaway copy and browser profile. For an entirely
synthetic rehearsal, the included Node v2 regression uses synthetic evidence only;
never import its synthetic state into the real worksheet. Do not create practice
labels for real cases and then treat them as human annotation.

Check: open the worksheet and several original-photo zooms; verify images display
and complete source panes are readable. On a throwaway synthetic worksheet,
choose/confirm a candidate with optional priority blank, split a compound candidate
into two independently supported atomic units, export progress, reload/import
that file, and compare restored state/confirmation provenance. Attempt final export
while incomplete: it must stay blocked and not download a final reference file.
Confirm full-source reading and missing-important-information instructions are
visible. Record browser/version and actual observed results, not just test plans.

The researcher will verify one or two real businesses manually only after reviewing
the corrected interface. All real labels and reference units must come from that
explicit researcher process.
'''


def prepare():
    require(not DEST.exists() and not OUT.exists(), 'Use a new namespace; v2 already exists')
    require(sha(V1) == V1_SHA, 'Original published ZIP changed')
    original = ROOT / 'outputs/stage_i/d053_reference_verification_v1'
    freeze = read(original / 'EVALUATION_FREEZE_MANIFEST.json')
    for relative, digest in freeze['workspace_files'].items():
        require(sha(ROOT / relative) == digest, 'Frozen workspace changed before correction: ' + relative)
    for relative, digest in freeze['package_files'].items():
        require(sha(original / relative) == digest, 'Frozen v1 package changed: ' + relative)
    preservation = read(original / 'IMMUTABLE_INPUT_PRESERVATION.json')['files']
    for relative, digest in preservation.items():
        require(sha(ROOT / relative) == digest, 'Accepted artifact changed: ' + relative)
    DEST.mkdir()
    OUT.mkdir()
    with zipfile.ZipFile(V1) as archive:
        require(archive.testzip() is None, 'v1 ZIP CRC error')
        prefix = V1.stem + '/'
        for member in archive.infolist():
            require(member.filename.startswith(prefix), 'Unexpected v1 ZIP root')
            relative = member.filename[len(prefix):]
            require(relative and '..' not in PurePosixPath(relative).parts
                    and not PurePosixPath(relative).is_absolute() and '\\' not in relative,
                    'Unsafe ZIP member')
            if member.is_dir():
                continue
            path = DEST / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
    for relative in ('SHA256_MANIFEST.json', 'SHA256SUMS.txt', 'EXPORT_SOURCE_INDEX.json',
                     'README_FIRST.md', 'CHATGPT_REVIEW_REQUEST.md', 'FUTURE_EVALUATION_USAGE.md',
                     'SOURCE_REFERENCE_ANNOTATION.html'):
        archive_old(relative)
    schema = DEST / 'schema/stage_i_d053_annotation_v1.schema.json'
    target = DEST / 'schemas' / schema.name
    target.parent.mkdir()
    schema.rename(target)
    schema.parent.rmdir()  # Known new copy, empty directory only.
    for name in ('stage_i_paired_pilot.py', 'check_stage_i_d053_bundle_readiness_v2.py',
                 'test_stage_i_d053_portable_readiness_v2.py', 'run_stage_i_d053_extracted_tests_v2.py',
                 'stage_i_d053_annotation_ui_v2.html', 'test_stage_i_d053_ui_v2.js',
                 'build_stage_i_d053_review_bundle_v2.py'):
        copy(ROOT / 'scripts' / name, 'code/' + name)
    old_html = (DEST / 'provenance/v1/SOURCE_REFERENCE_ANNOTATION.html').read_text(encoding='utf-8')
    payload = re.search(r'<script type="application/json" id="payload">([\s\S]*?)</script>', old_html)[1]
    template = (DEST / 'code/stage_i_d053_annotation_ui_v2.html').read_text(encoding='utf-8')
    require(template.count('__PAYLOAD__') == 1, 'UI v2 placeholder differs')
    write(DEST / 'SOURCE_REFERENCE_ANNOTATION.html', template.replace('__PAYLOAD__', payload))
    require(json.loads(payload)['initial_working_state']['units'] == [], 'Initial reference units not blank')
    config_path = ROOT / 'configs/stage_i_d053_evaluation_v1.json'
    require(sha(config_path) == CONFIG_SHA, 'Frozen config changed')
    copy(config_path, 'configs/stage_i_d053_evaluation_v1.json')
    config = read(config_path)
    mapping = {'frozen_config_file': 'FROZEN_EVALUATION_CONFIG.json', 'frozen_config_sha256': CONFIG_SHA,
               'bindings': {}, 'adapters': {}, 'frozen_code': {}}
    shared = {'d049_protocol': 'D049_FROZEN_EVALUATION_PROTOCOL.md',
              'candidates': 'candidates_UNVERIFIED.jsonl', 'source_cases': 'source_cases.json',
              'pilot_manifest': 'D049_FROZEN_PILOT_MANIFEST.json',
              'generation_config': 'FROZEN_GENERATION_CONFIG.json',
              'accepted_text_diagnostics': 'code/stage_i_decoding_diagnostics.py'}
    for name, item in config['bindings'].items():
        require(sha(ROOT / item['path']) == item['sha256'], 'Frozen binding changed: ' + name)
        entry = {'logical_project_path': item['path'], 'expected_sha256': item['sha256']}
        if name in {'case_inputs', 'blinding_key'}:
            entry.update(disposition='withheld_source_independence', bundle_file=None,
                         reason='Historical outputs/Teacher targets excluded from source-only review' if name == 'case_inputs'
                         else 'Blinded architecture display key must stay separate from researcher worksheet')
        else:
            relative = shared.get(name, 'readiness_payload/' + name + '/' + Path(item['path']).name)
            if name not in shared:
                copy(ROOT / item['path'], relative)
            require(sha(DEST / relative) == item['sha256'], 'Portable copy differs: ' + name)
            entry.update(disposition='included', bundle_file=relative)
        mapping['bindings'][name] = entry
    for arm, item in config['arms'].items():
        require(sha(ROOT / item['adapter_path']) == item['adapter_sha256'], 'Adapter changed: ' + arm)
        relative = 'readiness_payload/adapters/' + arm + '/adapter_model.safetensors'
        copy(ROOT / item['adapter_path'], relative)
        mapping['adapters'][arm] = {'disposition': 'included', 'bundle_file': relative,
                                   'logical_project_path': item['adapter_path'], 'expected_sha256': item['adapter_sha256']}
    for name in ('stage_i_d053_evaluation.py', 'stage_i_d053_annotations.py',
                 'stage_i_paired_pilot.py', 'stage_i_decoding_diagnostics.py'):
        mapping['frozen_code']['code/' + name] = sha(ROOT / 'scripts' / name)
    write(DEST / 'PORTABLE_BINDING_MAP.json', mapping)
    write(DEST / 'README_FIRST.md', README)
    write(DEST / 'FUTURE_EVALUATION_USAGE.md', FUTURE)
    write(DEST / 'MANUAL_BROWSER_SMOKE_TEST.md', SMOKE)
    write(DEST / 'requirements-offline.txt', 'numpy==1.26.4\njsonschema==4.23.0\n')
    write(DEST / 'CHATGPT_REVIEW_REQUEST.md', '''# Independent D-053 v2 review

Review corrected standalone packaging and annotation usability before real labels.
Read README_FIRST.md, PORTABILITY_CORRECTION_REPORT.md and validation_v2/. Check
schema path, bootstrap helper, extracted83-test results, standalone readiness scope,
original scientific hashes and explicit pending browser smoke test.
Check ALL-source review and important omitted-unit additions, candidate versus
reference-unit separation and final-export safeguards. Do not supply automatic
human labels, infer readiness to generate predictions or treat synthetic tests as
practical Chrome/Edge QA. The researcher will first annotate one or two businesses.
''')
    write(OUT / 'PRE_CORRECTION_INTEGRITY.json', {
        'status': 'PASS', 'v1_zip_sha256': V1_SHA, 'v1_workspace_files': freeze['workspace_files'],
        'v1_package_files': freeze['package_files'], 'accepted_artifacts': preservation,
        'original_frozen_config_sha256': CONFIG_SHA, 'human_labels_created': 0,
        'reference_units_created': 0, 'model_API_training_execution': False})
    correction = {'status': 'CORRECTED_PENDING_EXTRACTED_TESTS_AND_BROWSER_REVIEW',
                  'original_zip_sha256': V1_SHA, 'frozen_config_sha256': CONFIG_SHA,
                  'original_payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                  'new_payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                  'payload_identical': True, 'candidate_priority_fix': 'Optional nullable candidate fields; final-unit criteria unchanged',
                  'original_freeze_manifest_unchanged': True, 'original_files_preserved': True,
                  'practical_browser_smoke': 'NOT_EXECUTED_POLICY_BLOCKED_CURRENT_URL_UNDETERMINED',
                  'human_labels_created': 0, 'reference_units_created': 0, 'predictions_created': 0}
    write(DEST / 'PORTABILITY_CORRECTION_MANIFEST.json', correction)
    print(json.dumps({'status': 'PREPARED', 'bundle': str(DEST), 'report': str(OUT)}, indent=2))


def pack(path):
    require(DEST.is_dir() and not path.exists(), 'Missing preparation or ZIP already exists')
    # Regenerate only v2 packaging manifests; old v1 manifests are archived untouched.
    files = sorted(p for p in DEST.rglob('*') if p.is_file() and p.name not in {'SHA256_MANIFEST.json', 'SHA256SUMS.txt'})
    payload = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    for row in payload:
        p = DEST / row['path']
        if p.suffix.lower() in {'.json', '.jsonl', '.py', '.md', '.txt', '.html', '.js'}:
            require(not re.search(rb'\bsk-[A-Za-z0-9_-]{20,}', p.read_bytes()), 'Potential credential: ' + row['path'])
    # Explicitly exclude only root manifest paths, not archived v1 manifests.
    payload = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(DEST.rglob('*')) if p.is_file()
               and p.relative_to(DEST).as_posix() not in {'SHA256_MANIFEST.json', 'SHA256SUMS.txt'}]
    (DEST / 'SHA256_MANIFEST.json').write_text(json.dumps({'files': payload, 'excludes': ['SHA256_MANIFEST.json', 'SHA256SUMS.txt']}, indent=2) + '\n', encoding='utf-8')
    (DEST / 'SHA256SUMS.txt').write_text(''.join(r['sha256'] + '  ' + r['path'] + '\n' for r in payload), encoding='utf-8')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                archive.write(p, NAME + '/' + p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failure')
        require(len(archive.namelist()) == len(payload) + 2, 'ZIP member count differs')
        for row in payload:
            data = archive.read(NAME + '/' + row['path'])
            require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'ZIP payload hash differs')
    path.with_suffix(path.suffix + '.sha256').write_text(sha(path) + '  ' + path.name + '\n', encoding='ascii')
    print(json.dumps({'status': 'ZIP_CRC_SHA256_PASS', 'zip': str(path), 'bytes': path.stat().st_size,
                      'sha256': sha(path), 'payload_files': len(payload)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--pack', action='store_true')
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        require(args.zip is not None, '--zip required')
        pack(args.zip.resolve())
