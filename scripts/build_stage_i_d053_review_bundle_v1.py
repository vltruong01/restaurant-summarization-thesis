"""Export an offline, source-only D053 annotation review bundle.

No models, APIs, output generation, factual labeling or dataset mutation.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/d053_reference_verification_v1'
NAME = 'stage_i_d053_reference_verification_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


REVIEW_REQUEST = '''# Independent review request — D-053 source-based references

Please review this annotation workflow and its evidence before any new student predictions.

1. Check fixed 24 validation Filtered businesses: 20 random and 4 targeted. Confirm 226 original candidate IDs/claims (136 review / 90 visual), complete selected reviews, original photos and exact REV/IMG_CAP/PHOTO bindings are present. Technical evidence/quote checks are not human factual endorsement.
2. Inspect SOURCE_REFERENCE_ANNOTATION.html and its guide for usability and method completeness: five explicit candidate labels, optional candidate priority/informativeness, required final-unit metadata, notes/corrected text, manual splitting, missing-important-information addition, original context and photo zoom, progress export/import and final confirmation gates.
3. Confirm original candidate suggestions, candidate decisions, corrected drafts and supported atomic reference units remain separate. No AI candidate is automatically human-verified. Only the researcher explicitly decides labels. Negative/uncertain candidates need no artificial priority. Uncertain is retained separately; Needs revision must be resolved with separately verified units or explicitly excluded with a reason.
4. For visual references, require supplied-caption recoverability AND corroboration by the corresponding original photograph. Do not add photo-only details. Ordinary correct but simple descriptions are acceptable; lack of detail is not a factual error. Preserve the zero-Filtered-caption case.
5. Verify D-049 protocol and datasets/checkpoints remain unchanged. D-053 explicitly adds original-photo corroboration during source-reference verification while preserving caption-only content boundaries. No new image scorer, threshold, factual judge, decoding setting or training parameter was introduced.
6. Review annotation schema, confirmation snapshots/invalidation, exact-quote/source checks, synthetic tests and immutable/freeze manifests. A software provenance flag cannot prove the person independently inspected the evidence; identify that limitation honestly. DOM/behavioral tests are not browser visual QA.
7. Inspect prepared future A/B evaluation procedures: repetition, human semantic redundancy, source-unit coverage/omissions, positive/negative opinions, factual/cross-caption errors, zero-caption behavior, structure/EOS/cap handling and paired uncertainty. Missing/unverified judgments must stay N/A. Keep random and targeted denominators separate. No unvalidated composite quality score or automatic winner.
8. The same D052 best checkpoints, 24 inputs and complete frozen generation config will be used only after source references are confirmed and separately approved. Historical full F is not the matched baseline. There are no new Pilot A/B predictions in this package; do not invent quality results or infer that lower loss proves better summaries.

Please return concrete workflow/interface issues, missing fields, scientific risks and recommended corrections before human labeling begins. Do not replace the researcher's labels or silently populate final references. Do not run inference, training, Teacher requests or held-out evaluation.
'''


def main():
    require(not DEST.exists() and not ZIP.exists(), 'Preserve previous exports; use a new version')
    readiness = read(OUT / 'ANNOTATION_READINESS_REPORT.json')
    require(readiness['human_verified_candidates'] == readiness['confirmed_reference_units'] == 0,
            'Initial preparation export must not contain fabricated human labels')
    require(readiness['new_model_inference_or_generation'] == 0, 'Unexpected model execution')
    checks = read(OUT / 'OFFLINE_VALIDATION_SUMMARY.json')
    require(checks['status'] == 'PASS', 'Complete offline validation first')
    freeze = read(OUT / 'EVALUATION_FREEZE_MANIFEST.json')
    for relative, digest in freeze['workspace_files'].items():
        require(sha(ROOT / relative) == digest, 'Frozen implementation/config changed: ' + relative)
    for relative, digest in freeze['package_files'].items():
        require(sha(OUT / relative) == digest, 'Frozen package payload changed: ' + relative)
    preserved = read(OUT / 'IMMUTABLE_INPUT_PRESERVATION.json')
    for relative, digest in preserved['files'].items():
        require(sha(ROOT / relative) == digest, 'Previous accepted evidence changed: ' + relative)
    sources = {}

    def include(relative, source):
        path = Path(source)
        require(path.is_file(), 'Missing source: ' + str(path))
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and relative not in sources,
                'Unsafe/duplicate bundle path: ' + relative)
        require(path.stat().st_size < 20 * 2**20, 'Unexpected large source; inspect: ' + str(path))
        sources[relative] = path

    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name not in ['BUNDLE_EXPORT_RECEIPT.json']:
            include(path.relative_to(OUT).as_posix(), path)
    for name in ['prepare_stage_i_d053_reference_verification_v1.py', 'stage_i_d053_annotations.py',
                 'test_stage_i_d053_annotations.py', 'stage_i_d053_annotation_ui.html', 'test_stage_i_d053_ui.js',
                 'stage_i_d053_evaluation.py', 'test_stage_i_d053_evaluation.py',
                 'stage_i_decoding_diagnostics.py', 'build_stage_i_d053_review_bundle_v1.py',
                 'STAGE_I_D053_FUTURE_EVALUATION_USAGE.md']:
        include('code/' + name, ROOT / 'scripts' / name)
    include('schema/stage_i_d053_annotation_v1.schema.json', ROOT / 'schemas/stage_i_d053_annotation_v1.schema.json')
    for relative in ['thesis_sdd_docs/THESIS_PROGRESS.md', 'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                     'DECISIONS.md', 'specs/009h_d053_reference_verification.md']:
        include('methodology/' + Path(relative).name, ROOT / relative)
    for name in ['STAGE_I_D052_PAIRED_TRAINING_REPORT.md', 'SOURCE_REFERENCE_READINESS.json',
                 'D052_CONDITIONAL_REPRODUCIBILITY_DECISION.md']:
        include('D052_context_no_predictions/' + name, ROOT / 'outputs/stage_i/d052_paired_pilot_v1' / name)
    for relative, source in sources.items():
        if source.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raw = source.read_bytes()
            encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            require(not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', raw.decode(encoding)), 'Potential credential in: ' + relative)
    DEST.mkdir()
    origins = []
    for relative, source in sources.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, target)
        require(sha(target) == digest, 'Copied payload differs')
        origins.append({'path': relative, 'workspace_source': source.relative_to(ROOT).as_posix(), 'sha256': digest})
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text(REVIEW_REQUEST, encoding='utf-8')
    # Append review instructions to the copied README only; original package is intact.
    with (DEST / 'README_FIRST.md').open('a', encoding='utf-8') as stream:
        stream.write('''
## ChatGPT review package

Start with CHATGPT_REVIEW_REQUEST.md, the reference-verification guide, annotation HTML and readiness reports. All source photos are in images/. Candidate suggestions and initial annotations are explicitly UNVERIFIED.

The HTML works after extraction with no dependencies or network access. Python annotation validation uses only the standard library. Future evaluation uses the existing accepted NumPy bootstrap; no model, API, embedding download or inference is part of this package. See FUTURE_EVALUATION_USAGE.md.

Code, schema, frozen configurations, synthetic tests/logs, source/evidence index and SHA-256 manifests are included. No checkpoint binary, full training dataset, Teacher final target or historical/new student output is needed for this source-only review. D052_context_no_predictions contains training/provenance reports only.

The original full D049 evaluation-input hash is retained as provenance; that historical file also contains auxiliary targets/outputs and is intentionally not distributed here. The actual supplied 24 source-only cases have their own independently verifiable hash. No new reference labels or architecture quality results have been produced.

Browser automation was unavailable, so DOM/state/import/export tests are reported separately from visual layout review. Please inspect the interface visually in your own browser. No user annotation progress outside the supplied initial files is assumed.

Stop after reviewing preparation. Researcher confirmation and an offline-validated immutable reference freeze must precede separately approved generation of 48 summaries.
''')
    write_new(DEST / 'EXPORT_SOURCE_INDEX.json', {'files': origins, 'mode': 'OFFLINE_SOURCE_ONLY_EXPORT',
                                               'new_model_or_API_calls': 0, 'human_labels_created': 0})
    payload = [{'path': path.relative_to(DEST).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(DEST.rglob('*')) if path.is_file()]
    write_new(DEST / 'SHA256_MANIFEST.json', {'files': payload, 'excludes': ['SHA256_MANIFEST.json', 'SHA256SUMS.txt']})
    (DEST / 'SHA256SUMS.txt').write_text(''.join(row['sha256'] + '  ' + row['path'] + '\n' for row in payload), encoding='utf-8')
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        require(archive.testzip() is None, 'ZIP CRC error')
        require(len(archive.namelist()) == len(payload) + 2, 'ZIP members mismatch')
        for row in payload:
            data = archive.read(NAME + '/' + row['path'])
            require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'ZIP member mismatch')
    ZIP.with_suffix('.zip.sha256').write_text(sha(ZIP) + '  ' + ZIP.name + '\n', encoding='ascii')
    receipt = {'status': 'ZIP_CRC_SHA256_PASS', 'zip': ZIP.relative_to(ROOT).as_posix(),
               'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size / 2**20, 'sha256': sha(ZIP),
               'payload_files': len(payload), 'human_labels_created': 0, 'new_predictions': 0,
               'no_original_scientific_outputs_modified': True}
    write_new(OUT / 'BUNDLE_EXPORT_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
