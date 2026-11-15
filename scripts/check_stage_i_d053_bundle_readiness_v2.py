"""Portable D-053 preparation readiness, not the frozen full-project gate.

Reads only files beneath an extracted review bundle. It never imports model
libraries, generates labels, loads adapters, or grants generation permission.
The two explicitly withheld source-independence bindings remain NOT REVERIFIED.
The original evaluator and its complete scientific gate remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re

CONFIG_SHA256 = 'a139ee88824b6b5489b871cdbd5b9c13317139444f7a95033d5d9c5019a48461'
ORIGINAL_FREEZE_SHA256 = 'a8e1f346f0129084446a11c6a10e2a9a4ec37808a109c2adb5eed25729a55043'
FROZEN_CODE_SHA256 = {
    'code/stage_i_d053_evaluation.py': 'a086141cecb925c2d5d45d30a714a3d7ee9c69bec7a7abe7486d13564bea045a',
    'code/stage_i_d053_annotations.py': 'b5762faf33e66591094ca384a56ad3a8b7c967d96d313f10cc6020c133f2b752',
    'code/stage_i_paired_pilot.py': 'cde7192bde7e4ec2612fb1e96038e3455798809c16e64c050e6229163feb94b7',
    'code/stage_i_decoding_diagnostics.py': '9c5ac087c31c90a4eaa54554ab55fefcaad47bf58192f52c2abcc166e2fd85ce',
}
WITHHELD = {'case_inputs', 'blinding_key'}
ARMS = {'A_decoder_only', 'B_encoder_decoder'}


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_file(root, relative):
    require(isinstance(relative, str) and relative, 'Missing relative bundle path')
    candidate = Path(relative)
    require(not candidate.is_absolute() and not PureWindowsPath(relative).is_absolute()
            and '..' not in candidate.parts and '..' not in PureWindowsPath(relative).parts,
            'Unsafe bundle path: ' + relative)
    resolved = (root / candidate).resolve()
    require(resolved.is_relative_to(root.resolve()), 'Bundle path escapes extracted root')
    require(resolved.is_file(), 'Missing supplied file: ' + relative)
    return resolved


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_file(root, relative, digest):
    path = safe_file(root, relative)
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid expected SHA-256')
    require(sha(path) == digest, 'SHA-256 differs: ' + relative)
    return path


def check(root):
    root = Path(root).resolve()
    mapping = read(safe_file(root, 'PORTABLE_BINDING_MAP.json'))
    require(mapping['frozen_config_sha256'] == CONFIG_SHA256, 'Frozen configuration hash changed')
    config_path = verified_file(root, mapping['frozen_config_file'], CONFIG_SHA256)
    config = read(config_path)
    require(set(mapping['bindings']) == set(config['bindings']), 'Portable binding set differs from frozen configuration')
    supplied, withheld, locations = [], [], {}
    for name, original in config['bindings'].items():
        item = mapping['bindings'][name]
        require(item['logical_project_path'] == original['path'] and item['expected_sha256'] == original['sha256'],
                'Frozen binding metadata changed: ' + name)
        if name in WITHHELD:
            require(item['disposition'] == 'withheld_source_independence' and item.get('bundle_file') is None
                    and isinstance(item.get('reason'), str) and bool(item['reason'].strip()),
                    'Source-independence withholding must be explicit: ' + name)
            withheld.append({'name': name, 'logical_project_path': original['path'],
                             'expected_sha256': original['sha256'], 'status': 'NOT_REVERIFIED_IN_STANDALONE',
                             'reason': item['reason']})
        else:
            require(item['disposition'] == 'included', 'Unexpected withheld or missing binding: ' + name)
            locations[name] = verified_file(root, item['bundle_file'], original['sha256'])
            supplied.append(name)
    require(set(mapping['adapters']) == ARMS == set(config['arms']), 'Adapter arm set changed')
    adapters = []
    for arm, original in config['arms'].items():
        item = mapping['adapters'][arm]
        require(item['disposition'] == 'included' and item['logical_project_path'] == original['adapter_path']
                and item['expected_sha256'] == original['adapter_sha256'], 'Adapter binding changed: ' + arm)
        verified_file(root, item['bundle_file'], original['adapter_sha256'])
        adapters.append(arm)
    require(isinstance(mapping.get('frozen_code'), dict) and mapping['frozen_code'], 'Frozen code hashes missing')
    for relative, digest in mapping['frozen_code'].items():
        verified_file(root, relative, digest)
    for relative, digest in FROZEN_CODE_SHA256.items():
        require(mapping['frozen_code'].get(relative) == digest, 'Required frozen implementation missing or changed: ' + relative)
    require(mapping['frozen_code']['code/stage_i_decoding_diagnostics.py'] == config['bindings']['accepted_text_diagnostics']['sha256'],
            'Accepted diagnostic code binding changed')
    paired, generation, manifest = (read(locations[k]) for k in ('paired_training_config', 'generation_config', 'pilot_manifest'))
    require(paired['generation'] == config['generation_overrides'] and generation == config['effective_generation_config'],
            'Frozen generation settings differ')
    random_ids, targeted_ids = config['random_business_ids'], config['targeted_business_ids']
    require(len(random_ids) == 20 and len(targeted_ids) == 4 and len(set(random_ids + targeted_ids)) == 24,
            'Expected fixed20 random and4 targeted businesses')
    require(random_ids == manifest['generation_evaluation_random20'] and targeted_ids == manifest['generation_evaluation_targeted4'],
            'Frozen evaluation IDs/order changed')
    cases = read(locations['source_cases'])
    candidates = [json.loads(line) for line in locations['candidates'].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require([c['business_id'] for c in cases] == random_ids + targeted_ids, 'Source case IDs/order changed')
    require(len(candidates) == 226 and len({c['unit_id'] for c in candidates}) == 226, 'Candidate count/IDs invalid')
    require(sum(c['domain'] == 'review' for c in candidates) == 136 and sum(c['domain'] == 'visual' for c in candidates) == 90,
            'Candidate domain counts changed')
    case_map = {c['business_id']: c for c in cases}
    original_manifest = read(verified_file(root, 'EVALUATION_FREEZE_MANIFEST.json', ORIGINAL_FREEZE_SHA256))
    photos, evidence_count, review_count, zero_caption = set(), 0, 0, 0
    for case in cases:
        bid, source = case['business_id'], case['source']
        require(case['condition'] == 'filtered' and case['split'] == 'validation'
                and case['subset'] == ('random' if bid in random_ids else 'targeted'), 'Source split/condition/role mismatch')
        require(not any(k in case for k in ('baseline_prediction', 'target', 'label_ids')), 'Historical output/target in source-only cases')
        require(hashlib.sha256(source.encode('utf-8')).hexdigest() == case['source_sha256'], 'Source text/hash differs')
        require(case['selected_review_ids'] == [r['review_id'] for r in case['reviews']], 'Review IDs/order differ')
        require(case['selected_photo_ids'] == [c['photo_id'] for c in case['captions']], 'Photo IDs/order differ')
        parsed = re.findall(r'\[(REV_\d+)\] (.*?)(?=\n\n\[REV_\d+\]|\n\n\[VISUAL_CONTEXT\]|\Z)', source, re.S)
        require(parsed == [(r['marker'], r['text']) for r in case['reviews']], 'Full review text/order differs')
        parsed_caps = re.findall(r'\[(IMG_CAP_\d+)\]\n(.*?)(?=\n\n\[IMG_CAP_\d+\]|\Z)', source.split('[VISUAL_CONTEXT]', 1)[-1], re.S)
        require([(m, t.strip()) for m, t in parsed_caps] == [(c['caption_id'], c['text']) for c in case['captions']], 'Caption text/order differs')
        review_count += len(case['reviews'])
        zero_caption += not case['captions']
        for caption in case['captions']:
            relative = 'images/' + caption['photo_id'] + '.jpg'
            require(caption['image_file'] == relative, 'Photo relative path/mapping differs')
            verified_file(root, relative, original_manifest['package_files'][relative])
            photos.add(relative)
    for candidate in candidates:
        case = case_map[candidate['business_id']]
        require(candidate['human_verified'] is False and candidate.get('confirmed_by') is None and candidate.get('support_status') is None,
                'Candidate promoted to human label')
        domain = candidate['domain']
        sources = {x['marker' if domain == 'review' else 'caption_id']: x for x in case['reviews' if domain == 'review' else 'captions']}
        id_key = 'review_id' if domain == 'review' else 'photo_id'
        require(candidate['evidence'], 'Candidate evidence missing')
        for evidence in candidate['evidence']:
            item = sources.get(evidence.get('marker'))
            require(item and evidence.get(id_key) == item[id_key] and evidence.get('quote') and evidence['quote'] in item['text'],
                    'Candidate evidence ID/quote/domain mismatch')
            evidence_count += 1
    binding_doc = read(safe_file(root, 'bindings.json'))
    expected = {'case_inputs_sha256': config['bindings']['case_inputs']['sha256'],
                'protocol_sha256': config['bindings']['d049_protocol']['sha256'],
                'candidate_sha256': config['bindings']['candidates']['sha256'],
                'source_cases_sha256': config['bindings']['source_cases']['sha256'], 'evaluation_config_sha256': CONFIG_SHA256}
    require(binding_doc['bindings'] == expected, 'Annotation package bindings differ')
    require(binding_doc['bundle_id'] == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(), 'Annotation bundle ID differs')
    initial = read(safe_file(root, 'initial_annotations_UNVERIFIED.json'))
    require(initial['bundle_id'] == binding_doc['bundle_id'], 'Initial annotation bundle ID differs')
    require(not initial['units'] and not initial['researcher_confirmed_complete'] and not initial['reviewer_id'], 'Initial annotation state is not empty/unverified')
    require(all(a.get('support_label') is None and a.get('human_verified') is False for a in initial['candidate_annotations'].values()), 'Initial human labels must remain empty')
    require(all(b.get('human_verified') is False for b in initial['business_reviews'].values()), 'Initial business confirmation changed')
    validator_path = safe_file(root, 'code/stage_i_d053_annotations.py')
    spec = importlib.util.spec_from_file_location('d053_bundle_annotation_validator', validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    annotation = validator.validate_annotations(initial, cases, candidates, expected, final=False)
    final = validator.validate_annotations(initial, cases, candidates, expected, final=True)
    require(annotation['valid'] and not annotation['ready_for_freeze'] and not final['valid'], 'Initial state/final-export safeguard differs')
    return {'status': 'PORTABLE_ANNOTATION_AND_OFFLINE_EVALUATION_PREPARATION_READY',
            'scope': 'Provided review-package integrity only; not the frozen full-project readiness gate',
            'provided_bindings_verified': supplied, 'provided_binding_count': len(supplied),
            'adapter_hashes_verified_without_loading': adapters, 'frozen_code_files_verified': len(mapping['frozen_code']),
            'withheld_bindings': withheld, 'withheld_binding_status': 'NOT_REVERIFIED_IN_STANDALONE',
            'full_project_future_evaluation_gate': 'PENDING_FULL_PROJECT_BINDINGS_AND_HUMAN_REFERENCES',
            'counts': {'businesses': 24, 'random': 20, 'targeted': 4, 'candidates': 226, 'complete_reviews': review_count,
                       'original_photos': len(photos), 'zero_caption_businesses': zero_caption, 'exact_evidence_checks': evidence_count},
            'initial_annotations_valid': True, 'initial_final_export_blocked': True,
            'human_labels_created': 0, 'reference_units_created': 0, 'generation_authorized': False,
            'model_or_GPU_execution': False, 'full_project_gate_modified': False,
            'factual_judgments_independently_verified_by_software': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--readiness', required=True, action='store_true')
    parser.add_argument('--output', type=Path, help='Optional new report file; never overwrites')
    args = parser.parse_args()
    try:
        report, exit_code = check(args.root), 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        report, exit_code = {'status': 'FAIL', 'error': str(exc), 'generation_authorized': False,
                             'full_project_gate_modified': False}, 1
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
