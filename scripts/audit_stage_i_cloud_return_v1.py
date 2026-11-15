"""Read-only CPU audit of the imported Stage I cloud archive; no model execution."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from stage_hi_led_data import generation_format

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('reviews_only', 'unfiltered', 'filtered')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def signature(config):
    value = {k: v for k, v in config.items()
             if k not in ('approval', 'status', 'output_root')
             and not k.startswith('approved_feasibility_report')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def split_sections(text):
    parts = text.split('[VISUAL_EVIDENCE]', 1)
    return parts[0].replace('[REVIEW_CONSENSUS]', '').strip(), parts[1].strip() if len(parts) == 2 else ''


def repetition(text):
    # Diagnostic definition only: split on sentence-final .!? plus whitespace;
    # remove section markers, lowercase and collapse whitespace. No semantic judgment.
    clean = re.sub(r'\[[A-Z_]+\]', '', text)
    sentences = [' '.join(x.lower().split()) for x in re.split(r'(?<=[.!?])\s+', clean) if x.strip()]
    counts = Counter(sentences)
    duplicates = sum(n - 1 for n in counts.values())
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean.lower())
    four = Counter(tuple(words[i:i+4]) for i in range(max(0, len(words)-3)))
    return dict(sentence_instances=len(sentences), duplicate_sentence_instances=duplicates,
                has_exact_repeated_sentence=duplicates > 0,
                duplicate_sentence_fraction=duplicates / len(sentences) if sentences else 0,
                max_fourgram_frequency=max(four.values(), default=0))


def tensors(path):
    with safe_open(path, framework='numpy') as handle:
        return {k: handle.get_tensor(k) for k in handle.keys()}


def audit(cloud, output):
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), 'Refuse to overwrite an existing audit'
    config = read(cloud/'configs/stage_i_lora_led_training_approved_v2.json')
    local = read(ROOT/'configs/stage_i_lora_led_v1.json')
    assert signature(config) == signature(local), 'Scientific config drift'
    assert all(config['approval'].values())
    assert sha(cloud/config['approved_feasibility_report']) == config['approved_feasibility_report_sha256']
    assert read(cloud/config['approved_feasibility_report'])['status'] == 'GPU_FEASIBILITY_PASS_PENDING_RESEARCHER_REVIEW'
    manifest = read(cloud/config['stage_h_manifest'])
    assert sha(cloud/config['stage_h_manifest']) == config['stage_h_manifest_sha256']
    tokenizer = Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    teacher_dir = ROOT/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset'
    predictions, references, stats, warnings, diagnostics = {}, {}, {}, [], []
    data_checks = {}
    for split, expected in [('train', 2000), ('validation', 200)]:
        first = None
        for condition in CONDITIONS:
            key = f'{split}_{condition}'; info = manifest['datasets'][key]
            assert sha(cloud/info['path']) == info['sha256'] == sha(ROOT/info['path'])
            with np.load(cloud/info['path'], allow_pickle=False) as pack:
                ids = pack['business_ids'].tolist()
                assert len(ids) == len(set(ids)) == expected
                assert first is None or ids == first
                first = ids
                source_lengths = np.diff(pack['input_offsets'])
                label_lengths = np.diff(pack['label_offsets'])
                assert source_lengths.max() <= 5000 and label_lengths.max() <= 768
                data_checks[key] = dict(rows=expected, source_max=int(source_lengths.max()),
                                        label_max=int(label_lengths.max()), hash_match=True)
                if split == 'validation':
                    ref = rows(teacher_dir/f'{key}.jsonl')
                    pred = rows(cloud/f'outputs/stage_i/runs_v3/{condition}/validation_predictions.jsonl')
                    assert [x['business_id'] for x in ref] == ids == [x['business_id'] for x in pred]
                    assert all(x['condition'] == condition for x in pred)
                    for i, (r, p) in enumerate(zip(ref, pred)):
                        a, b = pack['input_offsets'][i:i+2]; c, d = pack['label_offsets'][i:i+2]
                        assert tokenizer.encode(r['source']).ids == pack['input_ids'][a:b].tolist()
                        assert tokenizer.encode(r['target']).ids == pack['labels'][c:d].tolist()
                        assert tokenizer.decode(p['token_ids'], skip_special_tokens=True) == p['text']
                        assert p['tokens_including_decoder_start'] == len(p['token_ids']) <= 768
                        assert p['cap_reached'] == (len(p['token_ids']) == 768)
                        pr, tr = repetition(p['text']), repetition(r['target'])
                        review, visual = split_sections(p['text'])
                        cap_count = int(pack['caption_counts'][i])
                        no_visual = condition == 'filtered' and cap_count == 0
                        assert p['section_format_ok'] == generation_format(
                            p['text'], condition, allow_empty_visual=no_visual)
                        flags = []
                        if not p['section_format_ok']: flags.append('reported_format_warning')
                        if p['cap_reached']: flags.append('generation_cap_reached')
                        if not p['text'].strip(): flags.append('empty_prediction')
                        if no_visual and visual: flags.append('visual_claims_generated_without_caption_input')
                        if flags:
                            warnings.append(dict(condition=condition, business_id=r['business_id'], flags=flags,
                                                 caption_count=cap_count, prediction=p['text']))
                        diagnostics.append(dict(condition=condition, business_id=r['business_id'],
                                                caption_count=cap_count, generated_tokens=len(p['token_ids']),
                                                reference_tokens=len(tokenizer.encode(r['target']).ids),
                                                **pr, reference_has_exact_repeated_sentence=tr['has_exact_repeated_sentence'],
                                                visual_has_exact_repeated_sentence=repetition(visual)['has_exact_repeated_sentence'],
                                                flags=';'.join(flags)))
                    predictions[condition] = {p['business_id']: p for p in pred}
                    references[condition] = {r['business_id']: r for r in ref}
    train_ids = set(np.load(cloud/manifest['datasets']['train_reviews_only']['path'], allow_pickle=False)['business_ids'].tolist())
    validation_ids = set(predictions['reviews_only'])
    assert not train_ids.intersection(validation_ids)
    for bid in validation_ids:
        shared = [split_sections(references[c][bid]['target'])[0] for c in CONDITIONS]
        assert len(set(shared)) == 1
    for condition in CONDITIONS:
        folder = cloud/f'outputs/stage_i/runs_v3/{condition}'
        complete, state, runtime = (read(folder/f) for f in ('completion_report.json', 'trainer_state.json', 'runtime_initial.json'))
        binding = read(folder/'run_binding.json')
        assert binding['settings_signature'] == signature(config) == complete['settings_signature']
        assert binding['runner_sha256'] == sha(cloud/'scripts/run_stage_i_lora_led_cloud_v2.py')
        assert binding['data_code_sha256'] == sha(cloud/'scripts/stage_hi_led_data.py')
        assert signature(runtime['config']) == signature(config)
        assert complete['status'] == 'TRAINING_COMPLETED_STOP_BEFORE_TEST' and complete['test_used'] is False
        assert state['epoch'] == 5 and state['global_step'] == state['max_steps'] == 1250
        history = [x for x in state['log_history'] if 'eval_loss' in x]
        assert len(history) == 5 and all(math.isfinite(x['eval_loss']) for x in history)
        assert min(x['eval_loss'] for x in history) == complete['best_eval_loss'] == state['best_metric']
        best_step = Path(complete['best_checkpoint']).name
        assert best_step == 'checkpoint-1250'
        adapter = read(folder/'best_adapter/adapter_config.json')
        assert adapter['r'] == 8 and adapter['lora_alpha'] == 16 and set(adapter['target_modules']) == {'q_proj', 'v_proj'}
        best, last, prior = (tensors(folder/f'{sub}/adapter_model.safetensors')
                             for sub in ('best_adapter', 'checkpoint-1250', 'checkpoint-1000'))
        assert best.keys() == last.keys() == prior.keys() and len(best) == 48
        assert sum(v.size for v in best.values()) == 294912 == complete['trainable_parameters']
        assert all('led.decoder.' in k and np.isfinite(v).all() for k, v in best.items())
        assert all(np.array_equal(v, last[k]) for k, v in best.items())
        changed = sum(not np.array_equal(last[k], prior[k]) for k in last)
        assert changed > 0
        assert all(np.count_nonzero(v) > 0 for k, v in best.items() if 'lora_B' in k)
        for step in ('checkpoint-1000', 'checkpoint-1250'):
            assert all((folder/step/f).is_file() for f in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'))
        selected = [x for x in diagnostics if x['condition'] == condition]
        stats[condition] = dict(completion=complete, eval_history=history, device=runtime['device'],
            adapter_tensor_count=48, adapter_tensors_changed_1000_to_1250=changed,
            best_adapter_equals_best_checkpoint=True, adapter_values_finite=True,
            predictions=len(selected), empty_predictions=sum(not x['text'].strip() for x in predictions[condition].values()),
            repeated_sentence_cases=sum(x['has_exact_repeated_sentence'] for x in selected),
            reference_repeated_sentence_cases=sum(x['reference_has_exact_repeated_sentence'] for x in selected),
            visual_repeated_sentence_cases=sum(x['visual_has_exact_repeated_sentence'] for x in selected),
            median_duplicate_sentence_fraction=statistics.median(x['duplicate_sentence_fraction'] for x in selected),
            generated_tokens_mean=statistics.mean(x['generated_tokens'] for x in selected),
            generated_tokens_max=max(x['generated_tokens'] for x in selected),
            reference_tokens_mean=statistics.mean(x['reference_tokens'] for x in selected))
    with (cloud/'outputs/stage_i/cloud_training_v2/gpu_metrics.csv').open() as stream:
        gpu_rows = list(csv.DictReader(stream, skipinitialspace=True))
    gpu = {}
    for key in ('memory.used [MiB]', 'utilization.gpu [%]', 'power.draw [W]'):
        v = [float(x[key].split()[0]) for x in gpu_rows]
        gpu[key] = dict(samples=len(v), maximum=max(v), mean=statistics.mean(v), median=statistics.median(v))
    random_ids = random.Random(20261015).sample(sorted(validation_ids), 10)
    flagged_ids = sorted({x['business_id'] for x in warnings} - set(random_ids))
    sample = []
    for role, ids in [('random_seed_20261015', random_ids), ('additional_automated_warning_cases', flagged_ids)]:
        for bid in ids:
            sample.append(dict(business_id=bid, role=role,
                review_source=references['reviews_only'][bid]['source'],
                conditions={c: dict(prediction=predictions[c][bid]['text'],
                    target=references[c][bid]['target'],
                    source_caption_block=references[c][bid]['source'].split('[VISUAL_CONTEXT]', 1)[1]
                        if '[VISUAL_CONTEXT]' in references[c][bid]['source'] else '',
                    selected_photo_ids=references[c][bid]['selected_photo_ids']) for c in CONDITIONS}))
    report = dict(status='TECHNICAL_TRAINING_PASS_OUTPUT_QUALITY_CONCERNS_REQUIRE_REVIEW',
        source_cloud_root=str(cloud.relative_to(ROOT)).replace('\\', '/'),
        data_checks=data_checks, validation_source_target_ID_roundtrips=600,
        prediction_ID_text_roundtrips=600, common_scientific_signature=signature(config),
        conditions=stats, gpu_10second_sample_statistics=gpu,
        warnings=warnings, random_sample_seed=20261015, random_sample_businesses=random_ids,
        additional_warning_businesses=flagged_ids,
        diagnostic_definition='Exact sentences split on .!? plus whitespace, section markers removed, lowercase/whitespace normalized; any sentence repeated twice flags a diagnostic, not a human factual error.',
        limits=['No held-out test evaluation', 'No independent human quality ground truth',
                'Cross-condition full-target validation losses use different target texts and are not a final RO/U/F quality ranking',
                '10-second GPU samples may miss shorter peaks; Torch peaks also retained',
                'Existing outputs preserved; no generation, training, threshold or decoding change'],
        GPU_or_API_execution=False)
    for name, value in [('report.json', report), ('review_sample.json', sample)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output/'prediction_diagnostics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)
    print(json.dumps(dict(status=report['status'], warnings=len(warnings),
        conditions={c: {k: stats[c][k] for k in ('predictions', 'repeated_sentence_cases', 'reference_repeated_sentence_cases',
                       'visual_repeated_sentence_cases', 'adapter_tensors_changed_1000_to_1250')} for c in CONDITIONS}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-root', type=Path, default=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/stage_i/cloud_return_20261015_v1/audit_v1')
    args = parser.parse_args()
    audit(args.cloud_root, args.output)
