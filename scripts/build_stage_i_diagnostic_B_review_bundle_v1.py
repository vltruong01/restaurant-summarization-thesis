"""Export completed B evidence for review; no model, metrics or API execution."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

from stage_i_decoding_diagnostics import CONDITIONS, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
NAME = 'stage_i_diagnostic_B_chatgpt_bundle_v1'
DEST = ROOT / 'outputs/stage_i' / NAME
ZIP = DEST.with_suffix('.zip')
PREP = ROOT / 'outputs/stage_i/decoding_diagnostic_v1'
RESULT = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1'
A = ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2'
B = ROOT / 'outputs/stage_i/decoding_beam_ngram_B_local_v1'
CONTROL = ROOT / 'outputs/stage_i/decoding_B_execution_control_v1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not DEST.exists() and not ZIP.exists(), 'Preserve old exports'
    mapping = {}

    def include(relative, source):
        assert source.is_file() and relative not in mapping
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        mapping[relative] = source

    for entry in read(RESULT / 'EVALUATION_MANIFEST.json')['files']:
        source = RESULT / entry['path']
        assert sha(source) == entry['sha256'] and source.stat().st_size == entry['bytes']
        include('results/' + entry['path'], source)
    include('results/EVALUATION_MANIFEST.json', RESULT / 'EVALUATION_MANIFEST.json')
    for name in ['cases.jsonl', 'subset_manifest.json', 'decoding_configurations.json']:
        include('inputs/' + name, PREP / name)
    include('inputs/frozen_source_template.json', ROOT / 'configs/led_source_template_d032.json')
    population_path = ROOT / 'outputs/stage_i/stage_i_decoding_diagnostic_chatgpt_bundle_v1/validation_population_ids.json'
    include('inputs/validation_population_ids.json', population_path)
    for name, folder in [('A', A), ('B', B)]:
        include(f'execution/{name}/run_binding.json', folder / 'run_binding.json')
        for path in sorted((folder / 'attempts').rglob('*.json')):
            include(f'execution/{name}/attempts/' + path.relative_to(folder / 'attempts').as_posix(), path)
        for condition in CONDITIONS:
            include(f'predictions/{name}/{condition}/predictions.jsonl', folder / condition / 'predictions.jsonl')
            casefiles = sorted((folder / condition / 'cases').glob('*.json'))
            assert len(casefiles) == 24
            for path in casefiles:
                include(f'predictions/{name}/{condition}/cases/{path.name}', path)
    for path in sorted(CONTROL.glob('*')):
        if path.is_file():
            include('execution/B/control/' + path.name, path)
    include('execution/B/approval.json', ROOT / 'outputs/stage_i/decoding_approval_B_local_v1.json')
    for path in (ROOT / 'outputs/stage_i/decoding_B_preflight_v1').iterdir():
        if path.is_file():
            include('execution/B/preflight/' + path.name, path)
    for name in ['run_stage_i_decoding_diagnostic_B_local_v1.py', 'run_stage_i_decoding_diagnostic_local_v2.py',
                 'orchestrate_stage_i_diagnostic_B_local_v1.py', 'preflight_stage_i_diagnostic_B_v1.py',
                 'evaluate_stage_i_diagnostic_B_local_v1.py', 'report_stage_i_diagnostic_B_local_v1.py',
                 'build_stage_i_diagnostic_B_review_bundle_v1.py', 'stage_i_decoding_diagnostics.py',
                 'stage_hi_led_data.py', 'test_stage_i_decoding_diagnostics.py']:
        include('code/' + name, ROOT / 'scripts' / name)
    for name in ['summary.json', 'per_case_metrics.csv']:
        include('prior_metrics/A/' + name, ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1' / name)
        include('prior_metrics/baseline/' + name, PREP / 'baseline_evaluation' / name)
    cases = read_jsonl(PREP / 'cases.jsonl')
    subset = read(PREP / 'subset_manifest.json')
    assert len(cases) == 72 and len(subset['execution_business_ids']) == 24
    assert random.Random(subset['seed']).sample(read(population_path)['sorted_business_ids'], 20) == subset['random_business_ids']
    comparisons = read(RESULT / 'baseline_A_B_cases.json')
    assert len(comparisons) == 72
    for case, comp in zip(cases, comparisons):
        assert case['business_id'] == comp['business_id'] and case['condition'] == comp['condition']
        assert case['source'] == comp['source'] and case['target'] == comp['teacher']
        assert case['baseline_prediction']['text'] == comp['baseline']
        for name, folder, field in [('A', A, 'greedy_A'), ('B', B, 'beam_ngram_B')]:
            pp = read_jsonl(folder / case['condition'] / 'predictions.jsonl')
            assert [p['business_id'] for p in pp] == subset['execution_business_ids']
            assert next(p for p in pp if p['business_id'] == case['business_id'])['text'] == comp[field]
    DEST.mkdir()
    copies = []
    for relative, source in mapping.items():
        raw = source.read_bytes()
        # PowerShell 5 redirection writes UTF-16 logs; retain the original bytes.
        text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', text), 'Potential credential in export source: ' + source.name
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, target)
        assert sha(target) == digest
        copies.append({'path': relative, 'sha256': digest, 'bytes': target.stat().st_size})
    index = {'all_paths_relative_to_bundle_root': True, 'businesses': []}
    for i, bid in enumerate(subset['execution_business_ids']):
        index['businesses'].append({'business_id': bid, 'split': 'validation',
            'subset': 'random' if bid in subset['random_business_ids'] else 'targeted',
            'conditions': {c: {'source_teacher_baseline_file': 'inputs/cases.jsonl',
                               'source_teacher_baseline_row_zero_based': next(j for j,x in enumerate(cases) if x['business_id'] == bid and x['condition'] == c),
                               'A_file': f'predictions/A/{c}/predictions.jsonl', 'B_file': f'predictions/B/{c}/predictions.jsonl',
                               'prediction_row_zero_based': i} for c in CONDITIONS}})
    for b in index['businesses']:
        for c in b['conditions'].values():
            for k in ['source_teacher_baseline_file', 'A_file', 'B_file']:
                assert (DEST / c[k]).is_file()
    write_json(DEST / 'INDEX.json', index)
    (DEST / 'README_FIRST.md').write_text('''# Completed Stage I Diagnostic B: independent review bundle

Read `CHATGPT_REVIEW_REQUEST.md`, `results/DIAGNOSTIC_REPORT.md`, `results/baseline_A_B_comparison.csv`, `results/ngram_lexical_summary.json` and `results/AI_QUALITATIVE_REVIEW.json`.

Open `results/COMPARISON.html` locally for all 72 source/teacher/Baseline/A/B comparisons. The static page is self-contained. JSON is available in `results/baseline_A_B_cases.json`; original complete reviews, captions, teacher evidence and token IDs are in `inputs/cases.jsonl`. `INDEX.json` joins every business/condition to portable relative paths and zero-based JSONL record numbers. Photo IDs are supplied; original photos and model weights are not included. Caption support is not independent photo verification.

Exactly the same 20 fixed-seed random validation businesses plus 4 predeclared targeted cases were used, 24 unique businesses / 72 outputs per setting. Report these subsets separately. All 200 validation IDs permit sampling verification; there is no held-out data. Saved Baseline and A outputs were reused without generation. B changes only no_repeat_ngram_size from 0 to 4 relative to beam4 Baseline. Following explicit researcher clarification, all settings keep **max_length=768 including decoder-start**, not max_new_tokens=768. B and A use the local RTX3050 6GB; Baseline used RTX3090. No bitwise cross-device or isolated speed claim is made.

The original preflight mismatch report is preserved together with the later explicit length resolution. Earlier unapproved/offline wording in frozen preparation or A metadata is historical. New B approval, actual full GenerationConfigs, raw cases, logs, monitoring, checksums and code are included. This is an evidence-review package, not a standalone model-runtime download.

Four-BPE-gram suppression is expected by construction. Lexical word grams ignore punctuation/case and may still repeat even if BPE grams do not. Higher distinct-n or lower repetition alone is not proof of informativeness or factual grounding. ROUGE is agreement with teacher targets; BERTScore is explicitly not computed because the reviewed runtime has no installed/pinned scorer/backbone. No score is fabricated. Human support/coverage worksheet fields remain blank. The evidence-linked qualitative review is Codex AI assessment, not user-confirmed human ground truth or a population factual-accuracy estimate.

All targets and predictions are unchanged. No retraining, teacher requests, test tuning or additional decoding experiment occurred. B is a bounded diagnostic, not automatic approval of a production decoder. The bundle has a SHA256 sidecar and a complete payload manifest.
''', encoding='utf-8')
    (DEST / 'CHATGPT_REVIEW_REQUEST.md').write_text('''# Review completed Diagnostic B before any further experiment

Please independently inspect this self-contained package, including full sources and outputs. Do not rely only on Codex's recommendation.

1. Check exact same checkpoints/inputs/24 validation IDs across Baseline/A/B. Inspect the B approval and actual GenerationConfigs: beam4, do_samplefalse, no_repeat_ngram_size4, max_length768. Researcher explicitly clarified the conflicting max_new_tokens wording to preserve the original total cap. Only the ngram constraint changes from Baseline.
2. Compare random20 and targeted4 separately for RO/U/F: sentence and adjacent-word repetition, BPE versus lexical ngrams, distinct-n, headers, EOS/cap, lengths and zero-caption compliance. Suppressing a BPE4 string does not establish semantic improvement; lexical diversity depends on length.
3. Compare informativeness and review-specific facts/contrasts using actual REV evidence and complete reviews. Look for filler, contradictory sentiment, incorrect food categories, unsupported generalization and reduced specificity.
4. Inspect all supplied captions/photo IDs before judging visual claims. Separate simple supported descriptions, semantic repetition, unsupported identities/actions/ingredients/relations and true metadata placed in VISUAL_EVIDENCE. Original photos are absent; do not claim photo verification. Check the zero-Filtered-caption and severe-repetition targeted cases explicitly.
5. Check AI qualitative notes against exact excerpts and evidence. They are not human-final labels. ROUGE is teacher-target agreement, not factual accuracy. BERTScore is unavailable, not zero. Do not infer causality about decoder-only LoRA, teacher imitation or insufficient training from this decoding comparison alone.
6. Recommend whether B meaningfully improves repetition, useful information and grounding together; identify remaining regressions/uncertainty and the smallest proposed next decision. No further experiment, retraining, teacher regeneration, dataset change or held-out tuning is authorized by this review.

Return a concise evidence-grounded assessment citing business_id / condition / REV / IMG_CAP / PHOTO IDs, correct any errors in Codex's interpretation and discuss next steps with the researcher. Do not rewrite model/teacher outputs or manufacture human labels.
''', encoding='utf-8')
    write_json(DEST / 'EXPORT_REPORT.json', {'status': 'EXPORT_ONLY_PASS', 'random_businesses': 20, 'targeted_businesses': 4,
               'B_outputs': 72, 'A_reused_outputs': 72, 'baseline_reused_outputs': 72,
               'source_target_prediction_copy_integrity': True, 'relative_index_integrity': True,
               'teacher_requests': 0, 'generation_during_export': False, 'heldout_data': False,
               'original_photos_or_weights_included': False, 'copies': copies})
    payload = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(DEST.rglob('*')) if p.is_file()]
    write_json(DEST / 'MANIFEST.json', {'files': payload, 'manifest_excludes_itself': True})
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                z.write(p, NAME + '/' + p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for e in payload:
            data = z.read(NAME + '/' + e['path'])
            assert len(data) == e['bytes'] and hashlib.sha256(data).hexdigest() == e['sha256']
    for relative, source in mapping.items():
        assert sha(source) == sha(DEST / relative)
    digest = sha(ZIP)
    ZIP.with_suffix('.zip.sha256').write_text(f'{digest}  {ZIP.name}\n', encoding='ascii')
    print(json.dumps({'status': 'PASS', 'path': str(ZIP), 'bytes': ZIP.stat().st_size, 'MiB': ZIP.stat().st_size / 1024**2,
                      'sha256': digest, 'files': len(payload) + 1}, indent=2))


if __name__ == '__main__':
    main()
